"""FastAPI web service: GET / (UI), POST /query (streamed cited answer), GET /health.

POST /query streams Server-Sent Events. An early 'progress' event is emitted during
retrieval so the first byte reaches Heroku's router well under its 30s limit.
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from forum_rag import store
from forum_rag.agent import answer
from forum_rag.auth import build_flow, get_current_user, is_allowed_email, require_user, verify_id_token
from forum_rag.config import BRANDING_DIR, TENANT, get_settings
from forum_rag.errors import ConfigError, ExternalServiceError

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup/shutdown. Ensures the Qdrant collection exists before serving
    traffic; a Qdrant outage at boot shouldn't crash the whole dyno (Heroku would
    just keep restarting it), so failures here are logged, not raised."""
    try:
        store.ensure_collection()
    except Exception:  # don't crash boot if Qdrant is briefly unreachable
        log.exception("ensure_collection failed at startup")
    yield


app = FastAPI(title=get_settings().brand.app_name, lifespan=lifespan)
app.add_middleware(
    SessionMiddleware,
    secret_key=get_settings().require_session_secret(),
    same_site="lax",
    https_only=bool(get_settings().qdrant_url),
)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
# Per-tenant branding assets (logo-mark.svg, favicon.svg, theme.css) served at /brand.
app.mount("/brand", StaticFiles(directory=str(BRANDING_DIR / TENANT)), name="brand")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
# Expose brand text to every template without threading it through each handler.
templates.env.globals["brand"] = get_settings().brand


class Query(BaseModel):
    question: str
    policy_area: str
    session: Optional[str] = None
    speaker: Optional[str] = None


def _callback_url(request: Request) -> str:
    """Absolute URL for the OAuth callback route, resolved from the live request
    (rather than hardcoded) so it's correct across local dev and any deployed host."""
    return str(request.url_for("auth_callback"))


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Chat UI for logged-in users; the sign-in page otherwise."""
    user = get_current_user(request)
    if user is None:
        return templates.TemplateResponse(request, "login.html", {})
    settings = get_settings()
    return templates.TemplateResponse(
        request,
        "index.html",
        {"policy_areas": [area.name for area in settings.policy_areas], "user": user},
    )


@app.get("/auth/login")
def auth_login(request: Request):
    """Start the Google OAuth2 login flow: redirect to Google's consent screen."""
    flow = build_flow(_callback_url(request))
    auth_url, state = flow.authorization_url(
        access_type="offline", include_granted_scopes="true", prompt="select_account"
    )
    request.session["oauth_state"] = state
    request.session["oauth_code_verifier"] = flow.code_verifier
    return RedirectResponse(auth_url)


@app.get("/auth/callback", name="auth_callback")
def auth_callback(request: Request):
    """Handle Google's redirect back after login: verify state, exchange the auth
    code for tokens, verify the ID token, then check the result email against the
    allowlist before establishing a session."""
    expected_state = request.session.pop("oauth_state", None)
    code_verifier = request.session.pop("oauth_code_verifier", None)
    if not expected_state or request.query_params.get("state") != expected_state:
        raise HTTPException(status_code=400, detail="Invalid OAuth state")

    flow = build_flow(_callback_url(request))
    flow.code_verifier = code_verifier
    try:
        flow.fetch_token(authorization_response=str(request.url))
        claims = verify_id_token(flow)
    except Exception as e:
        # Google's token endpoint can fail transiently, and a stale/replayed/
        # expired auth code or ID token is a normal (if rare) occurrence — send the
        # user back to try logging in again instead of surfacing a raw 500.
        log.warning("OAuth callback failed, redirecting to login: %s", e)
        return RedirectResponse("/auth/login")

    email = claims.get("email", "")
    if not is_allowed_email(email):
        log.warning("Login denied for %r: not on the allowed-emails list", email)
        return templates.TemplateResponse(
            request, "not_authorized.html", {"contact_email": get_settings().contact_email}
        )

    log.info("Login allowed for %r", email)
    request.session["user"] = {
        "email": email,
        "name": claims.get("name"),
        "picture": claims.get("picture"),
    }
    return RedirectResponse("/")


@app.get("/auth/logout")
def auth_logout(request: Request):
    """Clear the session cookie, logging the user out."""
    request.session.clear()
    return RedirectResponse("/")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/source/{citation_id}", response_class=HTMLResponse)
def source(request: Request, citation_id: str):
    """Render the full passage a citation permalink (/source/<citation_id>) points to."""
    user = get_current_user(request)
    if user is None:
        return templates.TemplateResponse(request, "login.html", {})
    try:
        chunk = store.get_by_citation_id(citation_id)
    except ExternalServiceError as e:
        log.error("Could not look up citation_id=%r: %s", citation_id, e)
        raise HTTPException(status_code=503, detail="Source lookup temporarily unavailable") from e
    if chunk is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return templates.TemplateResponse(request, "source.html", {"source": chunk, "user": user})


@app.post("/query")
def query(payload: Query, user: dict = Depends(require_user)):
    def event_stream():
        try:
            for ev in answer(
                payload.question,
                policy_area=payload.policy_area,
                session=payload.session or None,
                speaker=payload.speaker or None,
            ):
                yield f"data: {json.dumps(ev)}\n\n"
        except (ConfigError, ExternalServiceError) as e:
            # Known failure modes (bad config, upstream API/Qdrant down) — message is
            # safe to show as-is, it doesn't leak internals.
            log.error("Query failed: %r: %s", payload.question, e)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        except Exception:
            # Anything else is an unexpected bug — log the real detail server-side but
            # don't echo exception internals to the client.
            log.exception("Unexpected error handling query: %r", payload.question)
            yield f"data: {json.dumps({'type': 'error', 'message': 'Something went wrong. Please try again.'})}\n\n"

    # Errors are sent as a final SSE event (type: "error") rather than an HTTP error
    # status, since the response has already started streaming by the time most
    # failures happen — the client's EventSource can render this event inline
    # instead of the connection just dying with no explanation.
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
