"""Turn-level completeness audit: do a transcript's indexed chunks in Qdrant actually
cover every turn in the source file, or did chunking/upsert silently drop some?

Complements sources.py's reconcile_sources(), which only checks file-level
presence/staleness (a whole transcript missing or out of date) — this checks inside
an already-"synced" file for gaps a file-level check can't see.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from . import drive, sources, store
from .parse import Transcript, parse_transcript

_FIELDS = ["transcript_id", "drive_file_id", "turn_start", "turn_end"]


def turn_coverage_gaps(transcript: Transcript, chunks_for_transcript: list[dict[str, Any]]) -> list[int]:
    """Return turn_index values that have text in `transcript` but aren't covered by
    any chunk's turn_start..turn_end range."""
    covered: set[int] = set()
    for chunk in chunks_for_transcript:
        turn_start = chunk.get("turn_start")
        turn_end = chunk.get("turn_end")
        if turn_start is None or turn_end is None:
            continue
        covered.update(range(turn_start, turn_end + 1))
    expected = {turn.turn_index for turn in transcript.turns if turn.text}
    return sorted(expected - covered)


def _qdrant_chunks_by_transcript() -> dict[str, list[dict[str, Any]]]:
    by_transcript: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in store.iter_all_points(with_payload=_FIELDS):
        payload = point.payload or {}
        transcript_id = payload.get("transcript_id")
        if transcript_id:
            by_transcript[transcript_id].append(payload)
    return by_transcript


def audit_chunk_coverage() -> dict[str, Any]:
    """For every Drive file that's indexed (synced or stale) per reconcile_sources(),
    re-download and re-parse it, then check its indexed chunks cover every turn.

    Skips files reconcile_sources() already flagged "missing" (nothing indexed yet,
    not a turn-coverage question) and "local:" dev ingests (no Drive file to
    re-download). Returns {"rows": [{"transcript_id", "drive_file_id", "name",
    "gaps": [...]}]} — only rows with a non-empty "gaps" list indicate a problem.
    """
    reconciled = sources.reconcile_sources()
    drive_by_id = {f.id: f for f in drive.list_transcript_files()}
    chunks_by_transcript = _qdrant_chunks_by_transcript()

    rows: list[dict[str, Any]] = []
    for row in reconciled["rows"]:
        if row["status"] == "missing":
            continue
        drive_file_id = row["drive_file_id"]
        if drive_file_id.startswith("local:"):
            continue
        drive_file = drive_by_id.get(drive_file_id)
        if drive_file is None:
            continue

        raw = drive.download_file(drive_file_id)
        transcript = parse_transcript(json.loads(raw), filename=drive_file.name)
        chunks = chunks_by_transcript.get(transcript.transcript_id, [])
        gaps = turn_coverage_gaps(transcript, chunks)
        rows.append(
            {
                "transcript_id": transcript.transcript_id,
                "drive_file_id": drive_file_id,
                "name": drive_file.name,
                "gaps": gaps,
            }
        )

    return {"rows": rows}
