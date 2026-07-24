"""Generate the test-question fixture used by tests/test_retrieval_vector.py and
tests/test_retrieval_agent.py.

For every transcript currently indexed in Qdrant, ask Claude Haiku (the same model
classify.py uses for structured output) to write one narrow, specific question whose
answer requires that transcript's content in particular — grounded in one real chunk,
not a generic topic question many transcripts could equally answer.

Incremental by default: transcript_ids already in the fixture are left untouched, so
re-running after a fresh ingest only costs a call per *new* transcript. Pass
--regenerate to rebuild everything from scratch.

Usage:
  python generate_test_questions.py                # add questions for new transcripts only
  python generate_test_questions.py --regenerate    # rebuild the whole fixture
  python generate_test_questions.py --limit 20      # cap how many transcripts to process this run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from typing import Any

from forum_rag import store
from forum_rag.config import get_settings
from forum_rag.errors import ExternalServiceError
from forum_rag.logging import setup_logging
from tests.conftest import fixture_path

log = logging.getLogger(__name__)

_QUESTION_TOOL = {
    "name": "emit_question",
    "description": "Emit one specific, answerable question grounded in the given transcript excerpt.",
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "A specific question that can only be answered using this excerpt's "
                    "content — reference a concrete proposal, concern, number, or claim "
                    "actually raised here, not a generic topic question."
                ),
            }
        },
        "required": ["question"],
    },
}

_SYSTEM = (
    "You write test questions for a RAG system's retrieval eval. Given one excerpt from a "
    "community policy-discussion transcript, write ONE question a real user might ask that "
    "can only be answered using THIS excerpt — grounded in a specific proposal, concern, "
    "trade-off, or number actually mentioned in the text. Avoid generic questions "
    "('what did people think about X area?') that many unrelated transcripts could also "
    "answer. Call emit_question with your result."
)


def _pick_representative_chunk(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the longest chunk (most content to ground a specific question in)."""
    return max(chunks, key=lambda c: len(c.get("text") or ""))


def _group_by_transcript() -> dict[str, list[dict[str, Any]]]:
    fields = ["transcript_id", "chunk_id", "text", "session", "table", "date", "drive_file_id", "policy_areas"]
    by_transcript: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in store.iter_all_points(with_payload=fields):
        payload = point.payload or {}
        transcript_id = payload.get("transcript_id")
        if transcript_id:
            by_transcript[transcript_id].append(payload)
    return by_transcript


def _generate_question(client, model: str, chunk: dict[str, Any]) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=500,
        system=_SYSTEM,
        tools=[_QUESTION_TOOL],
        tool_choice={"type": "tool", "name": "emit_question"},
        messages=[{"role": "user", "content": f"Excerpt:\n\n{chunk.get('text', '')}"}],
    )
    for block in resp.content:
        if block.type == "tool_use" and block.name == "emit_question":
            return block.input["question"]
    raise ExternalServiceError("Haiku did not return an emit_question tool call")


def _load_existing(path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regenerate", action="store_true", help="Rebuild the whole fixture from scratch.")
    parser.add_argument("--limit", type=int, default=None, help="Cap how many transcripts to (re)generate this run.")
    args = parser.parse_args()

    from anthropic import Anthropic

    settings = get_settings()
    client = Anthropic(api_key=settings.require_anthropic_key())
    model = settings.models.classify
    path = fixture_path(settings.qdrant.collection)

    existing = {} if args.regenerate else {row["transcript_id"]: row for row in _load_existing(path)}

    try:
        by_transcript = _group_by_transcript()
    except ExternalServiceError as e:
        log.error("Failed to read Qdrant: %s", e)
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    todo = [tid for tid in by_transcript if tid not in existing]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"{len(existing)} question(s) already in fixture, generating {len(todo)} new one(s)...")

    for i, transcript_id in enumerate(todo, 1):
        chunks = by_transcript[transcript_id]
        chunk = _pick_representative_chunk(chunks)
        policy_areas = chunk.get("policy_areas") or []
        if not policy_areas:
            log.warning("Skipping transcript_id=%r: no policy_areas set on its chunks", transcript_id)
            continue
        try:
            question = _generate_question(client, model, chunk)
        except Exception as e:
            log.error("Failed to generate question for transcript_id=%r: %s", transcript_id, e)
            continue
        existing[transcript_id] = {
            "transcript_id": transcript_id,
            "drive_file_id": chunk.get("drive_file_id"),
            "session": chunk.get("session"),
            "table": chunk.get("table"),
            "date": chunk.get("date"),
            "policy_area": policy_areas[0],
            "question": question,
            "expected_chunk_id": chunk.get("chunk_id"),
        }
        print(f"  [{i}/{len(todo)}] {transcript_id}: {question}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(existing.values(), key=lambda r: r["transcript_id"]), indent=2))
    print(f"\nWrote {len(existing)} question(s) to {path}")


if __name__ == "__main__":
    main()
