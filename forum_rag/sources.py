"""Reconcile what's ingested into Qdrant against the live Google Drive listing, so the
Sources tab can show whether every transcript is actually indexed and up to date.

Mirrors the grouping approach in report_classifications.py's collect(), but keyed by
drive_file_id (Drive's identity for a file) instead of transcript_id, since that's the
join key against drive.list_transcript_files().
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from . import drive, store

# Sort/priority order for rows: problems first, so they're what you see without scrolling.
_STATUS_ORDER = ["missing", "stale", "synced"]

_DRIVE_FILE_URL = "https://drive.google.com/file/d/{}/view"

_FIELDS = ["drive_file_id", "transcript_id", "source_file", "session", "table", "date", "policy_areas", "source_md5"]


def _qdrant_sources() -> dict[str, dict[str, Any]]:
    """Group every chunk in the collection by drive_file_id.

    session/table/date/source_md5 are taken from the first chunk seen for a file,
    since ingest_data.py stamps the same file-level metadata onto every chunk of it.
    """
    by_file: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "transcript_id": None,
            "source_file": None,
            "session": None,
            "table": None,
            "date": None,
            "source_md5": None,
            "policy_areas": set(),
            "chunks": 0,
        }
    )
    for point in store.iter_all_points(with_payload=_FIELDS):
        payload = point.payload or {}
        drive_file_id = payload.get("drive_file_id") or "unknown"
        record = by_file[drive_file_id]
        record["chunks"] += 1
        record["policy_areas"].update(payload.get("policy_areas") or [])
        if record["transcript_id"] is None:
            record["transcript_id"] = payload.get("transcript_id")
            record["source_file"] = payload.get("source_file")
            record["session"] = payload.get("session")
            record["table"] = payload.get("table")
            record["date"] = payload.get("date")
            record["source_md5"] = payload.get("source_md5")
    return by_file


def reconcile_sources() -> dict[str, Any]:
    """Join Qdrant's indexed files against the live Drive listing.

    Returns {"summary": {status: count}, "rows": [...]}, rows sorted problem-statuses
    first. Raises ConfigError/ExternalServiceError (from drive/store) on failure —
    callers handle those the same way as any other Qdrant/Drive-backed endpoint.
    """
    qdrant_by_file = _qdrant_sources()
    drive_files = drive.list_transcript_files()
    drive_by_id = {f.id: f for f in drive_files}

    rows: list[dict[str, Any]] = []

    for drive_file_id, record in qdrant_by_file.items():
        is_local = drive_file_id.startswith("local:")
        drive_file = None if is_local else drive_by_id.get(drive_file_id)
        # A missing Drive match (moved/renamed/deleted upstream since ingest) and a
        # dev-only local: ingest both just mean "no live Drive file to compare
        # against" — neither is a problem worth a distinct status, so both count as
        # synced (it's indexed, and there's nothing further to check it against).
        if drive_file is not None and drive_file.md5 and drive_file.md5 != record["source_md5"]:
            status = "stale"
        else:
            status = "synced"
        rows.append(
            {
                "drive_file_id": drive_file_id,
                "name": record["source_file"] or (drive_file.name if drive_file else record["transcript_id"]),
                "status": status,
                "session": record["session"],
                "table": record["table"],
                "date": record["date"],
                "chunks": record["chunks"],
                "policy_areas": sorted(record["policy_areas"]),
                "modified_time": drive_file.modified_time if drive_file else None,
                "drive_url": None if is_local else _DRIVE_FILE_URL.format(drive_file_id),
            }
        )

    for drive_file in drive_files:
        if drive_file.id in qdrant_by_file:
            continue
        rows.append(
            {
                "drive_file_id": drive_file.id,
                "name": drive_file.name,
                "status": "missing",
                "session": None,
                "table": None,
                "date": None,
                "chunks": 0,
                "policy_areas": [],
                "modified_time": drive_file.modified_time,
                "drive_url": _DRIVE_FILE_URL.format(drive_file.id),
            }
        )

    rows.sort(key=lambda row: (_STATUS_ORDER.index(row["status"]), row["name"] or ""))

    summary = {status: 0 for status in _STATUS_ORDER}
    for row in rows:
        summary[row["status"]] += 1

    return {"summary": summary, "rows": rows}
