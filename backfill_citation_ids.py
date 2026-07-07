"""One-off: assign a citation_id to any already-indexed chunk that lacks one.

Needed once for data ingested before citation_id existed. Going forward, ingest_data.py
assigns citation_id at upsert time and carries it forward across re-ingestion, so this
script should never need to run again after a single pass over the live collection.

Usage:
  python backfill_citation_ids.py
"""
from __future__ import annotations

import logging
import sys
import uuid

from forum_rag import store
from forum_rag.config import get_settings
from forum_rag.errors import ExternalServiceError
from forum_rag.logging import setup_logging

log = logging.getLogger(__name__)


def main() -> None:
    """Scan every point in the collection and backfill a fresh citation_id onto
    any that don't already have one (pre-existing data from before permalinks)."""
    setup_logging()
    settings = get_settings()
    client = store.get_client()
    collection_name = settings.qdrant.collection

    updated = 0
    scanned = 0
    try:
        for point in store.iter_all_points(with_payload=["citation_id"]):
            scanned += 1
            if (point.payload or {}).get("citation_id"):
                continue
            try:
                client.set_payload(
                    collection_name=collection_name,
                    payload={"citation_id": str(uuid.uuid4())},
                    points=[point.id],
                )
            except Exception as e:
                log.error("Failed to set citation_id on point %r: %s", point.id, e)
                raise ExternalServiceError(f"Could not set citation_id on point {point.id!r}: {e}") from e
            updated += 1
    except ExternalServiceError as e:
        log.error("Backfill failed: %s", e)
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    log.info("Scanned %d point(s); assigned citation_id to %d that were missing one.", scanned, updated)
    print(f"Scanned {scanned} point(s); assigned citation_id to {updated} that were missing one.")


if __name__ == "__main__":
    main()
