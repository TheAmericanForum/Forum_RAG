"""Google SSO login (OAuth2/OIDC) gated by an email whitelist.

Built directly on google-auth / google-auth-oauthlib (no Authlib): we drive the
authorization-code flow with `Flow.from_client_config`, then verify the returned
ID token with `google.oauth2.id_token.verify_oauth2_token`.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import HTTPException, Request
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import id_token as google_id_token
from google_auth_oauthlib.flow import Flow

from .config import get_settings

# oauthlib refuses to complete a token exchange over plain HTTP. QDRANT_URL is only
# set in production (Heroku, behind HTTPS); its absence means we're running local dev
# over http://localhost, where this check needs to be disabled.
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
    s = get_settings()
    client_id, client_secret = s.require_google_oauth()
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
    s = get_settings()
    return email.lower() in s.allowed_emails


def verify_id_token(flow: Flow) -> dict:
    return google_id_token.verify_oauth2_token(
        flow.credentials.id_token,
        GoogleAuthRequest(),
        audience=get_settings().google_oauth_client_id,
    )


def get_current_user(request: Request) -> Optional[dict]:
    return request.session.get("user")


def require_user(request: Request) -> dict:
    user = get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user
