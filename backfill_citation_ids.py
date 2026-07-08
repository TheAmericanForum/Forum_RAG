"""One-off: assign a citation_id to any already-indexed chunk that lacks one.

Needed once for data ingested before citation_id existed. Going forward, ingest_data.py
assigns citation_id at upsert time and carries it forward across re-ingestion, so this
script should never need to run again after a single pass over the live collections.

Usage:
  python backfill_citation_ids.py                       # sweeps the default tenant collections
  python backfill_citation_ids.py nh_chunks nv_chunks   # or name specific collections
"""
from __future__ import annotations

import logging
import sys
import uuid

from forum_rag import store
from forum_rag.errors import ExternalServiceError
from forum_rag.logging import setup_logging

log = logging.getLogger(__name__)

# The per-tenant collections (NH / NV / SC) swept when no collections are named on
# the command line.
DEFAULT_COLLECTIONS = ["nh_chunks", "nv_chunks", "sc_chunks"]


def _backfill_collection(client, collection_name: str) -> tuple[int, int]:
    """Backfill one collection; returns (scanned, updated)."""
    updated = 0
    scanned = 0
    for point in store.iter_all_points(with_payload=["citation_id"], collection_name=collection_name):
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
            log.error("Failed to set citation_id on point %r in %r: %s", point.id, collection_name, e)
            raise ExternalServiceError(
                f"Could not set citation_id on point {point.id!r} in {collection_name!r}: {e}"
            ) from e
        updated += 1
    return scanned, updated


def main() -> None:
    """Scan every point in each target collection and backfill a fresh citation_id
    onto any that don't already have one (pre-existing data from before permalinks)."""
    setup_logging()
    client = store.get_client()
    collections = sys.argv[1:] or DEFAULT_COLLECTIONS

    total_scanned = 0
    total_updated = 0
    try:
        for collection_name in collections:
            scanned, updated = _backfill_collection(client, collection_name)
            total_scanned += scanned
            total_updated += updated
            log.info(
                "[%s] scanned %d point(s); assigned citation_id to %d that were missing one.",
                collection_name, scanned, updated,
            )
            print(f"[{collection_name}] scanned {scanned} point(s); assigned citation_id to {updated} missing one.")
    except ExternalServiceError as e:
        log.error("Backfill failed: %s", e)
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Done. Across {len(collections)} collection(s): scanned {total_scanned} point(s), "
        f"assigned citation_id to {total_updated} that were missing one."
    )


if __name__ == "__main__":
    main()
