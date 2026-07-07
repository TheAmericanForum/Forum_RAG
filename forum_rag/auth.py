"""Google SSO login (OAuth2/OIDC) gated by an email whitelist.

Built directly on google-auth / google-auth-oauthlib (no Authlib): we drive the
authorization-code flow with `Flow.from_client_config`, then verify the returned
ID token with `google.oauth2.id_token.verify_oauth2_token`.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

from fastapi import HTTPException, Request
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import id_token as google_id_token
from google_auth_oauthlib.flow import Flow

from .config import get_settings

log = logging.getLogger(__name__)

_EMAILS_TTL = 300  # re-fetch from Drive at most once every 5 minutes
# Module-global cache, not lock-protected: concurrent requests may race to refresh
# it, but the worst case is a redundant Drive fetch (or briefly serving the stale
# list), not corruption — acceptable at this app's traffic level.
_emails_cache: list[str] = []
_emails_fetched_at: float = 0.0


def _get_allowed_emails() -> list[str]:
    """Return the allowed-email list, refreshing from Drive at most every 5 minutes.

    Fallback chain: if ALLOWED_EMAILS_FILE_ID isn't configured, use the env var list
    directly. Otherwise, serve the in-memory cache until it's older than
    _EMAILS_TTL, then refresh from Drive; if that refresh fails, keep serving the
    (possibly stale) cache — and only fall back to the env var list if the cache is
    itself still empty (i.e. the very first fetch failed on cold start).
    """
    global _emails_cache, _emails_fetched_at
    settings = get_settings()
    if not settings.allowed_emails_file_id:
        return settings.allowed_emails
    now = time.monotonic()
    if now - _emails_fetched_at < _EMAILS_TTL:
        return _emails_cache
    try:
        from .drive import read_allowed_emails
        _emails_cache = read_allowed_emails(settings.allowed_emails_file_id, settings.tenant)
        _emails_fetched_at = now
        log.info(
            "Loaded %d allowed emails from Drive file %s (tenant=%s)",
            len(_emails_cache), settings.allowed_emails_file_id, settings.tenant,
        )
    except Exception as e:
        log.warning("Could not read allowed emails from Drive: %s; using cached/env list", e)
        if not _emails_cache:
            _emails_cache = settings.allowed_emails
    return _emails_cache

# oauthlib refuses to complete a token exchange over plain HTTP. QDRANT_URL is only
# set in production (Heroku, behind HTTPS); its absence means we're running local dev
# over http://localhost, where this check needs to be disabled.
# NOTE: this runs at import time (mutating process-wide os.environ as a side effect
# of importing this module), so it must happen before google-auth-oauthlib's first
# token exchange — import order matters here.
if not get_settings().qdrant_url:
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]


def build_flow(redirect_uri: str) -> Flow:
    """Build the Google OAuth2 authorization-code Flow for one login attempt.

    Uses a synthetic in-memory `client_config` (built from settings) rather than a
    downloaded client_secret.json file, matching this app's env-var-only-secrets
    approach (no credential files on disk).
    """
    settings = get_settings()
    client_id, client_secret = settings.require_google_oauth()
    client_config = {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": AUTH_URI,
            "token_uri": TOKEN_URI,
        }
    }
    return Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=redirect_uri)


def is_allowed_email(email: str) -> bool:
    """Check whether `email` is on the allowlist (env list and/or Drive-hosted file)."""
    return email.lower() in _get_allowed_emails()


def verify_id_token(flow: Flow) -> dict:
    """Verify the ID token returned by Google (signature + audience) and return its claims.

    Raises ValueError (from the underlying google-auth library) if the token is
    invalid, expired, or issued for a different OAuth client — the expected failure
    mode for a stale/replayed/misconfigured login attempt. Callers should catch it
    and redirect back to login rather than let it surface as a raw 500.
    """
    try:
        return google_id_token.verify_oauth2_token(
            flow.credentials.id_token,
            GoogleAuthRequest(),
            audience=get_settings().google_oauth_client_id,
        )
    except ValueError as e:
        log.warning("ID token verification failed: %s", e)
        raise


def get_current_user(request: Request) -> Optional[dict]:
    """Return the logged-in user's session dict, or None if not logged in."""
    return request.session.get("user")


def require_user(request: Request) -> dict:
    """Return the logged-in user's session dict, or raise 401 if not logged in."""
    user = get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user
