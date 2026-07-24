"""(b), turn-level: for every indexed transcript, does the union of its chunks'
turn ranges actually cover every turn in the source file — or did chunking/upsert
silently drop part of the conversation (e.g. the tail after a re-ingest that only
partially completed)?
"""
from __future__ import annotations

from forum_rag import integrity


def test_no_turn_coverage_gaps():
    result = integrity.audit_chunk_coverage()
    bad_rows = [row for row in result["rows"] if row["gaps"]]
    assert not bad_rows, (
        f"{len(bad_rows)} transcript(s) have turns missing from every indexed chunk:\n"
        + "\n".join(f"  {row['name']} (transcript_id={row['transcript_id']}): turns {row['gaps']}" for row in bad_rows)
    )
