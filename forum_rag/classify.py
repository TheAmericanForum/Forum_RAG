"""Classify a whole transcript into a single configured policy area with Claude Haiku 4.5.

Single-label, structured output (json_schema). Every transcript addresses exactly
one issue area, so classification happens once per transcript, not per chunk.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .config import get_settings
from .errors import ExternalServiceError, is_retryable_api_error

log = logging.getLogger(__name__)

_client = None


def _get_client():
    """Return the process-wide Anthropic client, creating it on first use."""
    global _client
    if _client is None:
        from anthropic import Anthropic

        _client = Anthropic(api_key=get_settings().require_anthropic_key())
    return _client


def _schema(area_names: list[str]) -> dict:
    """JSON schema for structured classifier output: one area name, or 'other'."""
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
    # Fewer attempts than embed.py's 7: a classify failure isn't fatal to ingestion —
    # resolve_policy_area() falls back to filename keywords/manual choice/"other", so
    # it's fine to give up on the model sooner and let those fallbacks take over.
    retry=retry_if_exception(is_retryable_api_error),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, max=20),
    reraise=True,
)
def _classify_one(text: str, area_names: list[str], descriptions: list[str]) -> str:
    """Ask the classifier model for the single best-matching policy area."""
    settings = get_settings()
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
        resp = _get_client().messages.create(
            model=settings.models.classify,
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
    settings = get_settings()
    if not text:
        return "other"
    if not settings.has_policy_areas:
        # Areas not configured yet — label "other" so ingestion still works.
        return "other"

    area_names = [area.name for area in settings.policy_areas]
    descriptions = [area.description for area in settings.policy_areas]
    return _classify_one(text, area_names, descriptions)


# Words too generic to identify an area from a filename.
_STOP = {"and", "or", "the", "of", "a", "an", "in", "on", "for", "to", "sc", "table",
         "recording", "recordings", "transcript", "transcripts"}


def _keywords(name: str) -> set[str]:
    """Lowercase, stopword-filtered tokens from a name, for keyword-overlap matching."""
    return {w for w in re.split(r"[^a-z0-9]+", name.lower()) if len(w) > 2 and w not in _STOP}


def infer_area_from_filename(filename: str) -> Optional[str]:
    """Best-effort policy area from the filename, by keyword overlap with the area names.

    Returns the single best-matching area, or None if the filename carries no signal.
    """
    settings = get_settings()
    if not settings.has_policy_areas or not filename:
        return None
    filename_words = _keywords(Path(filename).stem)
    if not filename_words:
        return None
    best, best_score = None, 0
    for area in settings.policy_areas:
        score = len(_keywords(area.name) & filename_words)
        if score > best_score:
            best, best_score = area.name, score
    return best if best_score > 0 else None


def prompt_manual_area(filename: str) -> str:
    """Interactively ask which policy area a transcript belongs to. Returns 'other' if skipped."""
    area_names = [area.name for area in get_settings().policy_areas]
    print(f"\nCould not classify transcript: {filename}", file=sys.stderr)
    print("Choose a policy area:", file=sys.stderr)
    for i, name in enumerate(area_names, 1):
        print(f"  {i}) {name}", file=sys.stderr)
    print(f"  {len(area_names) + 1}) other", file=sys.stderr)
    while True:
        try:
            choice = input("Enter number (blank = other): ").strip()
        except (EOFError, KeyboardInterrupt):
            # stdin closed/interrupted mid-prompt (e.g. an automated run that
            # mis-detected a TTY) — fall back to "other" instead of crashing.
            log.warning("Manual classification prompt interrupted for %r; labeling 'other'.", filename)
            print("\nNo input available; labeling 'other'.", file=sys.stderr)
            return "other"
        if not choice:
            return "other"
        if choice.isdigit():
            choice_num = int(choice)
            if 1 <= choice_num <= len(area_names):
                return area_names[choice_num - 1]
            if choice_num == len(area_names) + 1:
                return "other"
        print("Invalid choice, try again.", file=sys.stderr)


def resolve_policy_area(text: str, filename: str, *, interactive: Optional[bool] = None) -> str:
    """Classify a transcript, with two fallbacks before giving up and labeling it 'other'.

    1. Ask the model (``classify_transcript``).
    2. If it can't decide, infer the area from the filename (``infer_area_from_filename``).
    3. If the filename is uninformative, ask for manual classification — but only when
       running interactively. ``interactive`` defaults to whether stdin is a TTY, so
       automated ingests (e.g. Heroku Scheduler) fall back to 'other' instead of blocking.
    """
    area = classify_transcript(text)
    if area != "other" or not get_settings().has_policy_areas:
        return area

    guessed = infer_area_from_filename(filename)
    if guessed:
        log.info("Classifier returned 'other' for %r; inferred %r from filename.", filename, guessed)
        return guessed

    if interactive is None:
        interactive = sys.stdin.isatty()
    if interactive:
        return prompt_manual_area(filename)

    log.warning("Could not classify %r (no filename signal, non-interactive); labeling 'other'.", filename)
    return "other"
