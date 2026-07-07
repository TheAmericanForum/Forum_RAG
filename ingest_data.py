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
from typing import Callable, Iterable, Optional

from forum_rag import store
from forum_rag.chunk import chunk_transcript
from forum_rag.classify import resolve_policy_area
from forum_rag.config import Settings, get_settings
from forum_rag.embed import embed_texts
from forum_rag.errors import ConfigError, ExternalServiceError
from forum_rag.logging import setup_logging
from forum_rag.parse import parse_transcript

log = logging.getLogger(__name__)


def _md5(raw: bytes) -> str:
    return hashlib.md5(raw).hexdigest()


def ingest_one(
    name: str,
    raw: bytes,
    drive_file_id: str,
    source_md5: str,
    settings: Settings,
    *,
    interactive: Optional[bool] = None,
) -> int:
    """Parse, chunk, classify, embed, and upsert one transcript file.

    Skipped entirely if `source_md5` already matches what's stored for this
    `drive_file_id` — that's the incremental-ingestion mechanism: unchanged files
    cost nothing on repeat runs (e.g. hourly Heroku Scheduler invocations).
    Returns the number of chunks ingested (0 if skipped or the file had no chunks).
    """
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
        target_tokens=settings.chunk.target_tokens,
        overlap_turns=settings.chunk.overlap_turns,
        max_turns_per_chunk=settings.chunk.max_turns_per_chunk,
    )
    if not chunks:
        print(f"  no chunks: {name}")
        return 0

    texts = [chunk.text for chunk in chunks]
    full_text = "\n".join(f"{turn.speaker}: {turn.text}" for turn in transcript.turns if turn.text)
    area = resolve_policy_area(full_text, name, interactive=interactive)
    areas = [[area]] * len(chunks)
    vectors = embed_texts(texts)

    existing_citation_ids = store.existing_citation_ids_for_file(drive_file_id)
    store.delete_file(drive_file_id)  # replace any prior version of this file
    store.upsert_chunks(
        chunks,
        vectors,
        drive_file_id=drive_file_id,
        source_md5=source_md5,
        policy_areas_by_chunk=areas,
        existing_citation_ids=existing_citation_ids,
    )
    log.info("Ingested %d chunk(s) for %s (drive_file_id=%s)", len(chunks), name, drive_file_id)
    print(f"  ingested {len(chunks)} chunks: {name}")
    return len(chunks)


def _run_ingest(
    items: Iterable[tuple[str, Callable[[], tuple[bytes, str, str]]]],
    settings: Settings,
    *,
    interactive: Optional[bool] = None,
) -> tuple[int, list[str]]:
    """Run ingest_one() over a series of files, isolating failures per file.

    `items` yields (display_name, fetch) pairs. `fetch()` resolves the file's
    (raw_bytes, drive_file_id, source_md5) lazily — reading from disk or downloading
    from Drive happens inside the same try/except as ingest_one(), so a read/download
    failure is caught and logged per file exactly like a parse/embed/upsert failure,
    instead of aborting the whole run. Shared by run_local() and run_drive().
    """
    total_chunks = 0
    failed: list[str] = []
    for name, fetch in items:
        try:
            raw, drive_file_id, source_md5 = fetch()
            total_chunks += ingest_one(name, raw, drive_file_id, source_md5, settings, interactive=interactive)
        except (ConfigError, ExternalServiceError) as e:
            log.error("Failed to ingest %s: %s", name, e)
            print(f"  ERROR ingesting {name}: {e}")
            failed.append(name)
        except Exception:
            log.exception("Failed to ingest %s", name)
            print(f"  ERROR ingesting {name}: unexpected failure (see log)")
            failed.append(name)
    return total_chunks, failed


def run_local(path: str, settings: Settings, *, interactive: Optional[bool] = None) -> tuple[int, list[str]]:
    base_path = Path(path)
    files = [base_path] if base_path.is_file() else sorted(base_path.glob("*.json"))
    if not files:
        print(f"No JSON files found at {path}")
        return 0, []

    def _fetch_local(file_path: Path) -> tuple[bytes, str, str]:
        raw = file_path.read_bytes()
        return raw, f"local:{file_path.name}", _md5(raw)

    items = [(file_path.name, lambda file_path=file_path: _fetch_local(file_path)) for file_path in files]
    return _run_ingest(items, settings, interactive=interactive)


def run_drive(settings: Settings, *, interactive: Optional[bool] = None) -> tuple[int, list[str]]:
    from forum_rag import drive

    files = drive.list_transcript_files()
    print(f"Found {len(files)} JSON file(s) in Drive")

    def _fetch_drive(drive_file) -> tuple[bytes, str, str]:
        raw = drive.download_file(drive_file.id)
        return raw, drive_file.id, drive_file.md5 or _md5(raw)

    items = [(drive_file.name, lambda drive_file=drive_file: _fetch_drive(drive_file)) for drive_file in files]
    return _run_ingest(items, settings, interactive=interactive)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Ingest transcripts into the vector store.")
    parser.add_argument("--source", default="drive", help="'drive' (default) or 'local:PATH'")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never prompt for manual classification; label 'other' when undecidable "
        "(use for automated/scheduled runs). Default auto-detects a TTY.",
    )
    args = parser.parse_args()
    interactive = False if args.non_interactive else None  # None = auto-detect a TTY

    try:
        settings = get_settings()
        store.ensure_collection()
    except (ConfigError, ExternalServiceError) as e:
        log.error("Ingestion cannot start: %s", e)
        print(f"ERROR: {e}")
        sys.exit(1)

    if not settings.has_policy_areas:
        print(
            "WARNING: policy_areas in config.yaml are placeholders — chunks will be "
            "labeled 'other'. Fill them in and re-ingest for area filtering."
        )

    try:
        if args.source.startswith("local"):
            _, _, path = args.source.partition(":")
            chunk_count, failed = run_local(path or "./data", settings, interactive=interactive)
        else:
            chunk_count, failed = run_drive(settings, interactive=interactive)
    except (ConfigError, ExternalServiceError) as e:
        log.error("Ingestion aborted: %s", e)
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Done. {chunk_count} chunk(s) ingested/updated.")
    if failed:
        print(f"{len(failed)} file(s) failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
