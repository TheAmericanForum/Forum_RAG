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

from sc_rag import store
from sc_rag.agent import answer
from sc_rag.auth import build_flow, get_current_user, is_allowed_email, require_user, verify_id_token
from sc_rag.config import get_settings

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        store.ensure_collection()
    except Exception:  # don't crash boot if Qdrant is briefly unreachable
        log.exception("ensure_collection failed at startup")
    yield


app = FastAPI(title="The South Carolina Forum", lifespan=lifespan)
app.add_middleware(
    SessionMiddleware,
    secret_key=get_settings().require_session_secret(),
    same_site="lax",
    https_only=bool(get_settings().qdrant_url),
)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))


class Query(BaseModel):
    question: str
    policy_area: str
    session: Optional[str] = None
    speaker: Optional[str] = None


def _callback_url(request: Request) -> str:
    return str(request.url_for("auth_callback"))


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    user = get_current_user(request)
    if user is None:
        return templates.TemplateResponse(request, "login.html", {})
    s = get_settings()
    return templates.TemplateResponse(
        request,
        "index.html",
        {"policy_areas": [a.name for a in s.policy_areas], "user": user},
    )


@app.get("/auth/login")
def auth_login(request: Request):
    flow = build_flow(_callback_url(request))
    auth_url, state = flow.authorization_url(
        access_type="offline", include_granted_scopes="true", prompt="select_account"
    )
    request.session["oauth_state"] = state
    request.session["oauth_code_verifier"] = flow.code_verifier
    return RedirectResponse(auth_url)


@app.get("/auth/callback", name="auth_callback")
def auth_callback(request: Request):
    expected_state = request.session.pop("oauth_state", None)
    code_verifier = request.session.pop("oauth_code_verifier", None)
    if not expected_state or request.query_params.get("state") != expected_state:
        raise HTTPException(status_code=400, detail="Invalid OAuth state")

    flow = build_flow(_callback_url(request))
    flow.code_verifier = code_verifier
    flow.fetch_token(authorization_response=str(request.url))
    claims = verify_id_token(flow)

    email = claims.get("email", "")
    if not is_allowed_email(email):
        return templates.TemplateResponse(
            request, "not_authorized.html", {"contact_email": get_settings().contact_email}
        )

    request.session["user"] = {
        "email": email,
        "name": claims.get("name"),
        "picture": claims.get("picture"),
    }
    return RedirectResponse("/")


@app.get("/auth/logout")
def auth_logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/query")
def query(q: Query, user: dict = Depends(require_user)):
    def gen():
        try:
            for ev in answer(
                q.question,
                policy_area=q.policy_area,
                session=q.session or None,
                speaker=q.speaker or None,
            ):
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:  # surface errors to the client as a final event
            log.exception("Query failed: %r", q.question)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
