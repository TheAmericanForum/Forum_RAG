"""Report how each ingested transcript was classified into a policy area.

Classification is per-transcript (one label stamped on every chunk), so this collapses
all chunks by transcript and prints the assigned area, grouped by area.

Usage:
  python report_classifications.py            # read the configured store (QDRANT_URL or local ./.qdrant)
  python report_classifications.py --json      # machine-readable output

Note: in local embedded mode (QDRANT_URL unset) the store is single-writer — stop the
running app/ingest first, or this will report that the folder is already in use.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict

from forum_rag import store
from forum_rag.errors import ExternalServiceError
from forum_rag.logging import setup_logging

log = logging.getLogger(__name__)


def collect() -> list[dict]:
    """Group every chunk in the collection by transcript, returning one summary row
    per transcript: its assigned policy area(s), session/table/date, and chunk count.

    A transcript is identified by transcript_id, falling back to source_file and then
    the literal "unknown" if neither is present. session/table/date are taken from
    the first chunk seen for a transcript, since chunk.py stamps the same
    transcript-level metadata onto every chunk of that transcript.
    """
    by_transcript: dict[str, dict] = defaultdict(
        lambda: {"areas": set(), "chunks": 0, "session": None, "table": None, "date": None}
    )
    fields = ["transcript_id", "session", "table", "date", "source_file", "policy_areas"]
    for point in store.iter_all_points(with_payload=fields):
        payload = point.payload or {}
        transcript_id = payload.get("transcript_id") or payload.get("source_file") or "unknown"
        record = by_transcript[transcript_id]
        record["chunks"] += 1
        record["areas"].update(payload.get("policy_areas") or [])
        if record["session"] is None:
            record["session"] = payload.get("session")
            record["table"] = payload.get("table")
            record["date"] = payload.get("date")

    out = []
    for transcript_id, record in by_transcript.items():
        out.append(
            {
                "transcript_id": transcript_id,
                "area": ", ".join(sorted(record["areas"])) or "other",
                "session": record["session"],
                "table": record["table"],
                "date": record["date"],
                "chunks": record["chunks"],
            }
        )
    return out


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = parser.parse_args()

    try:
        rows = collect()
    except ExternalServiceError as e:
        log.error("Report failed: %s", e)
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    rows.sort(key=lambda row: (row["area"], row["transcript_id"]))
    totals: dict[str, int] = defaultdict(int)

    print(f"\n{len(rows)} transcript(s), {sum(row['chunks'] for row in rows)} chunks\n")
    print("Transcripts by policy area")
    print("=" * 72)
    for row in rows:
        totals[row["area"]] += 1
        table = f"table {row['table']}" if row["table"] else ""
        location = " · ".join(x for x in [row["session"], table, row["date"]] if x)
        flag = "  ⚠️ unclassified" if row["area"] == "other" else ""
        print(f"[{row['area']}]{flag}")
        print(f"    {row['transcript_id']}")
        print(f"    {location}  ({row['chunks']} chunks)")
    print("=" * 72)
    print("Totals by area (transcripts):")
    for area, count in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"  {count:3d}  {area}")


if __name__ == "__main__":
    main()
