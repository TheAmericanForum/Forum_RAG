"""OpenAI embeddings. This is the ONLY module that talks to OpenAI."""
from __future__ import annotations

import logging
from typing import Optional

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .config import get_settings
from .errors import ExternalServiceError, is_retryable_api_error

log = logging.getLogger(__name__)

_client = None


def _client_():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=get_settings().require_openai_key())
    return _client


@retry(
    retry=retry_if_exception(is_retryable_api_error),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, max=30),
    reraise=True,
)
def _embed_batch(texts: list[str], model: str) -> list[list[float]]:
    try:
        resp = _client_().embeddings.create(model=model, input=texts)
    except Exception as e:
        if is_retryable_api_error(e):
            log.warning("Embedding call failed, will retry: %s", e)
            raise
        log.error("Embedding call failed (non-retryable): %s", e)
        raise ExternalServiceError(f"OpenAI embedding request failed: {e}") from e
    return [d.embedding for d in resp.data]


def embed_texts(texts: list[str], *, batch_size: int = 100) -> list[list[float]]:
    if not texts:
        return []
    model = get_settings().models.embed
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        out.extend(_embed_batch(texts[i : i + batch_size], model))
    return out


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
