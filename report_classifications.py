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
import sys
from collections import defaultdict

from forum_rag import store
from forum_rag.config import get_settings
from forum_rag.errors import ExternalServiceError


def collect() -> list[dict]:
    s = get_settings()
    client = store.get_client()
    coll = s.qdrant.collection

    by_tx: dict[str, dict] = defaultdict(
        lambda: {"areas": set(), "chunks": 0, "session": None, "table": None, "date": None}
    )
    next_page = None
    while True:
        points, next_page = client.scroll(
            collection_name=coll,
            limit=256,
            offset=next_page,
            with_payload=["transcript_id", "session", "table", "date", "source_file", "policy_areas"],
            with_vectors=False,
        )
        for p in points:
            pl = p.payload or {}
            tx = pl.get("transcript_id") or pl.get("source_file") or "unknown"
            rec = by_tx[tx]
            rec["chunks"] += 1
            rec["areas"].update(pl.get("policy_areas") or [])
            if rec["session"] is None:
                rec["session"], rec["table"], rec["date"] = pl.get("session"), pl.get("table"), pl.get("date")
        if not next_page:
            break

    out = []
    for tx, rec in by_tx.items():
        out.append(
            {
                "transcript_id": tx,
                "area": ", ".join(sorted(rec["areas"])) or "other",
                "session": rec["session"],
                "table": rec["table"],
                "date": rec["date"],
                "chunks": rec["chunks"],
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = ap.parse_args()

    try:
        rows = collect()
    except ExternalServiceError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    rows.sort(key=lambda r: (r["area"], r["transcript_id"]))
    totals: dict[str, int] = defaultdict(int)

    print(f"\n{len(rows)} transcript(s), {sum(r['chunks'] for r in rows)} chunks\n")
    print("Transcripts by policy area")
    print("=" * 72)
    for r in rows:
        totals[r["area"]] += 1
        table = f"table {r['table']}" if r["table"] else ""
        loc = " · ".join(x for x in [r["session"], table, r["date"]] if x)
        flag = "  ⚠️ unclassified" if r["area"] == "other" else ""
        print(f"[{r['area']}]{flag}")
        print(f"    {r['transcript_id']}")
        print(f"    {loc}  ({r['chunks']} chunks)")
    print("=" * 72)
    print("Totals by area (transcripts):")
    for area, n in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"  {n:3d}  {area}")


if __name__ == "__main__":
    main()
