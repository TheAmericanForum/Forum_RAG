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

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from sc_rag import store
from sc_rag.agent import answer
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


app = FastAPI(title="ScForum Chat", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))


class Query(BaseModel):
    question: str
    policy_area: str
    session: Optional[str] = None
    speaker: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    s = get_settings()
    return templates.TemplateResponse(
        request,
        "index.html",
        {"policy_areas": [a.name for a in s.policy_areas]},
    )


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/query")
def query(q: Query):
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
