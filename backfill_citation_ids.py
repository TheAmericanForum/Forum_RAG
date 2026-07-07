"""One-off: assign a citation_id to any already-indexed chunk that lacks one.

Needed once for data ingested before citation_id existed. Going forward, ingest_data.py
assigns citation_id at upsert time and carries it forward across re-ingestion, so this
script should never need to run again after a single pass over the live collection.

Usage:
  python backfill_citation_ids.py
"""
from __future__ import annotations

import sys
import uuid

from forum_rag import store
from forum_rag.config import get_settings
from forum_rag.errors import ExternalServiceError


def main() -> None:
    s = get_settings()
    client = store.get_client()
    coll = s.qdrant.collection

    updated = 0
    scanned = 0
    next_page = None
    try:
        while True:
            points, next_page = client.scroll(
                collection_name=coll,
                limit=256,
                offset=next_page,
                with_payload=["citation_id"],
                with_vectors=False,
            )
            for p in points:
                scanned += 1
                if (p.payload or {}).get("citation_id"):
                    continue
                client.set_payload(
                    collection_name=coll,
                    payload={"citation_id": str(uuid.uuid4())},
                    points=[p.id],
                )
                updated += 1
            if not next_page:
                break
    except ExternalServiceError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanned {scanned} point(s); assigned citation_id to {updated} that were missing one.")


if __name__ == "__main__":
    main()
