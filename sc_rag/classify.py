"""Classify a whole transcript into a single configured policy area with Claude Haiku 4.5.

Single-label, structured output (json_schema). Every transcript addresses exactly
one issue area, so classification happens once per transcript, not per chunk.
"""
from __future__ import annotations

import json
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
        from anthropic import Anthropic

        _client = Anthropic(api_key=get_settings().require_anthropic_key())
    return _client


def _schema(area_names: list[str]) -> dict:
    enum = area_names + ["other"]
    return {
        "type": "object",
        "properties": {
            "area": {"type": "string", "enum": enum},
        },
        "required": ["area"],
        "additionalProperties": False,
    }


@retry(
    retry=retry_if_exception(is_retryable_api_error),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, max=20),
    reraise=True,
)
def _classify_one(text: str, area_names: list[str], descriptions: list[str]) -> str:
    s = get_settings()
    areas_desc = "\n".join(f"- {n}: {d}" for n, d in zip(area_names, descriptions))
    prompt = (
        "You label a community policy-discussion transcript by the single policy area "
        "it substantively discusses. Every transcript addresses exactly one issue.\n\n"
        f"Policy areas:\n{areas_desc}\n- other: none of the above.\n\n"
        "Return the one area that best matches the whole transcript. Use 'other' only "
        "if none apply.\n\n"
        f"Transcript:\n{text}"
    )
    try:
        resp = _client_().messages.create(
            model=s.models.classify,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": _schema(area_names)}},
        )
    except Exception as e:
        if is_retryable_api_error(e):
            log.warning("Classify call failed, will retry: %s", e)
            raise
        log.error("Classify call failed (non-retryable): %s", e)
        raise ExternalServiceError(f"Anthropic classify request failed: {e}") from e

    text_out = next((b.text for b in resp.content if b.type == "text"), "{}")
    try:
        data = json.loads(text_out)
    except json.JSONDecodeError as e:
        log.error("Classify response was not valid JSON: %r", text_out[:500])
        raise ExternalServiceError(f"Anthropic classify returned malformed JSON: {e}") from e
    return data.get("area") or "other"


def classify_transcript(text: str) -> str:
    """Return the single policy area for a whole transcript's text."""
    s = get_settings()
    if not text:
        return "other"
    if not s.has_policy_areas:
        # Areas not configured yet — label "other" so ingestion still works.
        return "other"

    area_names = [a.name for a in s.policy_areas]
    descriptions = [a.description for a in s.policy_areas]
    return _classify_one(text, area_names, descriptions)
