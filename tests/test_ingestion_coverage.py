"""(a) Every transcript in Drive parses cleanly.
(b) Every transcript in Drive is indexed in Qdrant and up to date (file-level).

Mirrors what the Sources tab already shows (forum_rag.sources.reconcile_sources),
turned into an assertion so a regression fails a build instead of only being visible
if someone happens to open that tab.
"""
from __future__ import annotations

import json

from forum_rag import drive, sources
from forum_rag.parse import parse_transcript


def test_no_missing_or_stale_sources():
    result = sources.reconcile_sources()
    bad_rows = [row for row in result["rows"] if row["status"] in ("missing", "stale")]
    assert not bad_rows, (
        f"{len(bad_rows)} transcript(s) not fully indexed (summary={result['summary']}):\n"
        + "\n".join(f"  [{row['status']}] {row['name']} (drive_file_id={row['drive_file_id']})" for row in bad_rows)
    )


def test_all_drive_transcripts_parse_cleanly():
    files = drive.list_transcript_files()
    assert files, "No transcript files found in the configured Drive folders (DRIVE_FOLDER_IDS)."

    failures: list[str] = []
    empty: list[str] = []
    for f in files:
        try:
            raw = drive.download_file(f.id)
            transcript = parse_transcript(json.loads(raw), filename=f.name)
        except Exception as e:
            failures.append(f"{f.name} (id={f.id}): {e}")
            continue
        if not transcript.turns:
            empty.append(f"{f.name} (id={f.id})")

    assert not failures, f"{len(failures)} transcript(s) failed to parse:\n" + "\n".join(f"  {x}" for x in failures)
    assert not empty, f"{len(empty)} transcript(s) parsed but have zero turns:\n" + "\n".join(f"  {x}" for x in empty)
