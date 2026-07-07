"""OpenAI embeddings. This is the ONLY module that talks to OpenAI."""
from __future__ import annotations

import logging
from typing import Optional

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .config import get_settings
from .errors import ExternalServiceError, is_retryable_api_error

log = logging.getLogger(__name__)

_client = None


def _get_client():
    """Return the process-wide OpenAI client, creating it on first use."""
    global _client
    if _client is None:
        import httpx
        from openai import OpenAI

        # Force IPv4: some Heroku dynos have broken outbound IPv6 routing, which
        # makes every connection to api.openai.com fail deterministically even
        # though DNS resolves fine and retries don't help.
        http_client = httpx.Client(transport=httpx.HTTPTransport(local_address="0.0.0.0"))
        # max_retries=0: tenacity below is the sole retry authority. Letting the SDK
        # retry internally too compounds backoff delay without buying extra resilience
        # against the multi-second connectivity bursts seen from this dyno.
        _client = OpenAI(
            api_key=get_settings().require_openai_key(), http_client=http_client, max_retries=0
        )
    return _client


@retry(
    # 7 attempts with exponential backoff (capped at 15s/attempt) is enough to ride
    # out a typical OpenAI rate-limit or transient-outage window without the caller
    # waiting an excessive amount of time overall.
    retry=retry_if_exception(is_retryable_api_error),
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=1, max=15),
    reraise=True,
)
def _embed_batch(texts: list[str], model: str) -> list[list[float]]:
    """Embed one batch of texts, retrying transient failures and wrapping the rest."""
    try:
        resp = _get_client().embeddings.create(model=model, input=texts)
    except Exception as e:
        if is_retryable_api_error(e):
            log.warning("Embedding call failed, will retry: %s", e)
            raise
        log.error("Embedding call failed (non-retryable): %s", e)
        raise ExternalServiceError(f"OpenAI embedding request failed: {e}") from e
    return [embedding_data.embedding for embedding_data in resp.data]


def embed_texts(texts: list[str], *, batch_size: int = 100) -> list[list[float]]:
    """Embed a list of texts, batching requests to stay under OpenAI's per-call limits."""
    if not texts:
        return []
    model = get_settings().models.embed
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        out.extend(_embed_batch(texts[i : i + batch_size], model))
    return out


def embed_query(text: str) -> list[float]:
    """Embed a single query string (e.g. for a similarity search)."""
    return embed_texts([text])[0]
