"""Ingestion entry point.

Usage:
  python ingest_data.py                         # ingest from Google Drive (default)
  python ingest_data.py --source drive
  python ingest_data.py --source local:./data   # ingest local JSON files (dev/testing)

Incremental: a file is skipped when its md5 already matches what is stored in Qdrant,
so re-runs (e.g. from Heroku Scheduler) only process new/changed files. No local state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from forum_rag import store
from forum_rag.chunk import chunk_transcript
from forum_rag.classify import resolve_policy_area
from forum_rag.config import Settings, get_settings
from forum_rag.embed import embed_texts
from forum_rag.errors import ConfigError, ExternalServiceError
from forum_rag.parse import parse_transcript

log = logging.getLogger(__name__)


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def ingest_one(
    name: str,
    raw: bytes,
    drive_file_id: str,
    source_md5: str,
    s: Settings,
    *,
    interactive: Optional[bool] = None,
) -> int:
    if store.stored_md5_for_file(drive_file_id) == source_md5:
        print(f"  skip (unchanged): {name}")
        return 0

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{name} is not valid JSON: {e}") from e

    transcript = parse_transcript(data, filename=name)
    chunks = chunk_transcript(
        transcript,
        target_tokens=s.chunk.target_tokens,
        overlap_turns=s.chunk.overlap_turns,
        max_turns_per_chunk=s.chunk.max_turns_per_chunk,
    )
    if not chunks:
        print(f"  no chunks: {name}")
        return 0

    texts = [c.text for c in chunks]
    full_text = "\n".join(f"{t.speaker}: {t.text}" for t in transcript.turns if t.text)
    area = resolve_policy_area(full_text, name, interactive=interactive)
    areas = [[area]] * len(chunks)
    vectors = embed_texts(texts)

    store.delete_file(drive_file_id)  # replace any prior version of this file
    store.upsert_chunks(
        chunks,
        vectors,
        drive_file_id=drive_file_id,
        source_md5=source_md5,
        policy_areas_by_chunk=areas,
    )
    print(f"  ingested {len(chunks)} chunks: {name}")
    return len(chunks)


def run_local(path: str, s: Settings, *, interactive: Optional[bool] = None) -> tuple[int, list[str]]:
    p = Path(path)
    files = [p] if p.is_file() else sorted(p.glob("*.json"))
    if not files:
        print(f"No JSON files found at {path}")
        return 0, []
    total = 0
    failed: list[str] = []
    for f in files:
        try:
            raw = f.read_bytes()
            total += ingest_one(f.name, raw, f"local:{f.name}", _md5(raw), s, interactive=interactive)
        except Exception as e:
            log.exception("Failed to ingest %s", f.name)
            print(f"  ERROR ingesting {f.name}: {e}")
            failed.append(f.name)
    return total, failed


def run_drive(s: Settings, *, interactive: Optional[bool] = None) -> tuple[int, list[str]]:
    from forum_rag import drive

    files = drive.list_transcript_files()
    print(f"Found {len(files)} JSON file(s) in Drive")
    total = 0
    failed: list[str] = []
    for df in files:
        try:
            raw = drive.download_file(df.id)
            md5 = df.md5 or _md5(raw)
            total += ingest_one(df.name, raw, df.id, md5, s, interactive=interactive)
        except Exception as e:
            log.exception("Failed to ingest %s (file_id=%s)", df.name, df.id)
            print(f"  ERROR ingesting {df.name}: {e}")
            failed.append(df.name)
    return total, failed


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest transcripts into the vector store.")
    ap.add_argument("--source", default="drive", help="'drive' (default) or 'local:PATH'")
    ap.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never prompt for manual classification; label 'other' when undecidable "
        "(use for automated/scheduled runs). Default auto-detects a TTY.",
    )
    args = ap.parse_args()
    interactive = False if args.non_interactive else None  # None = auto-detect a TTY

    try:
        s = get_settings()
        store.ensure_collection()
    except (ConfigError, ExternalServiceError) as e:
        log.error("Ingestion cannot start: %s", e)
        print(f"ERROR: {e}")
        sys.exit(1)

    if not s.has_policy_areas:
        print(
            "WARNING: policy_areas in config.yaml are placeholders — chunks will be "
            "labeled 'other'. Fill them in and re-ingest for area filtering."
        )

    try:
        if args.source.startswith("local"):
            _, _, path = args.source.partition(":")
            n, failed = run_local(path or "./data", s, interactive=interactive)
        else:
            n, failed = run_drive(s, interactive=interactive)
    except (ConfigError, ExternalServiceError) as e:
        log.error("Ingestion aborted: %s", e)
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Done. {n} chunk(s) ingested/updated.")
    if failed:
        print(f"{len(failed)} file(s) failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
