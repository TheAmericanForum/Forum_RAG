"""One-off cleanup: remove duplicate recordings from the vector store.

The same recording can be ingested from two Drive paths (e.g. a copy filed under
'.../Concord/' and a flat copy in the root folder). The copies are not byte-identical
files — each embeds its own source_file path, so their md5s differ — but their
transcript content is identical, so they are found by content fingerprint (the same
one ingest_data.py now stamps on new points as `content_sha256`).

For every file this also backfills the `content_sha256` payload field, so the
ingest-time duplicate guard can see files indexed before that field existed —
without it, a deleted duplicate would be silently re-added on the next ingest run.

Duplicate groups keep the copy with the deeper source path (more specific session
label — see ingest_data.dup_rank) and delete the rest.

Runs against the collection resolved from the environment (TENANT / QDRANT_COLLECTION),
so run it once per tenant.

Usage:
  python dedupe_qdrant.py --dry-run   # report keep/delete decisions, write nothing
  python dedupe_qdrant.py             # backfill content_sha256 and delete duplicates
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict

from forum_rag import store
from forum_rag.config import get_settings
from forum_rag.errors import ExternalServiceError
from forum_rag.logging import setup_logging
from ingest_data import NEAR_DUP_JACCARD, content_fingerprint, dup_rank, ms_jaccard

log = logging.getLogger(__name__)


def dedupe(dry_run: bool) -> None:
    settings = get_settings()
    fields = ["drive_file_id", "source_file", "text", "turn_start", "start_ms", "end_ms", "date", "content_sha256"]
    files: dict[str, dict] = defaultdict(
        lambda: {"source_file": "", "date": None, "chunks": [], "ms": set(), "point_ids": [], "stored_hash": None}
    )
    for point in store.iter_all_points(with_payload=fields):
        payload = point.payload or {}
        record = files[payload.get("drive_file_id") or "unknown"]
        record["source_file"] = payload.get("source_file") or record["source_file"]
        record["date"] = payload.get("date") or record["date"]
        record["chunks"].append((payload.get("turn_start") or 0, payload.get("text") or ""))
        record["ms"].add((payload.get("start_ms") or 0, payload.get("end_ms") or 0))
        record["point_ids"].append(point.id)
        record["stored_hash"] = payload.get("content_sha256") or record["stored_hash"]

    # Chunks come out of chunk_transcript() ordered by turn_start, so sorting by it
    # reproduces ingest order and thus the exact fingerprint ingest_data.py computes.
    for record in files.values():
        record["hash"] = content_fingerprint(text for _, text in sorted(record["chunks"], key=lambda c: c[0]))

    # Cluster duplicates (union-find): exact content-hash matches, plus same-date
    # files whose chunk timestamps overlap — the same audio re-transcribed has
    # slight text differences but near-identical timestamps.
    parent = {fid: fid for fid in files}

    def find(fid: str) -> str:
        while parent[fid] != fid:
            parent[fid] = parent[parent[fid]]
            fid = parent[fid]
        return fid

    file_ids = list(files)
    for i, a in enumerate(file_ids):
        for b in file_ids[i + 1:]:
            same_hash = files[a]["hash"] == files[b]["hash"]
            same_recording = (
                files[a]["date"] and files[a]["date"] == files[b]["date"]
                and ms_jaccard(files[a]["ms"], files[b]["ms"]) >= NEAR_DUP_JACCARD
            )
            if same_hash or same_recording:
                parent[find(a)] = find(b)

    groups: dict[str, list[str]] = defaultdict(list)
    for fid in files:
        groups[find(fid)].append(fid)
    dup_groups = [ids for ids in groups.values() if len(ids) > 1]

    to_delete: list[str] = []
    print(f"Collection {settings.qdrant.collection!r}: {len(files)} file(s), {len(dup_groups)} duplicate group(s).")
    for group in sorted(dup_groups, key=lambda ids: files[ids[0]]["source_file"]):
        # Chunk count breaks exact-rank ties (e.g. two copies under the same path,
        # one a partial transcription): keep the fuller one.
        group.sort(key=lambda fid: (dup_rank(files[fid]["source_file"]), -len(files[fid]["point_ids"])))
        keep, losers = group[0], group[1:]
        print(f"  keep   {files[keep]['source_file']} ({len(files[keep]['point_ids'])} chunks)")
        for file_id in losers:
            print(f"  delete {files[file_id]['source_file']} ({len(files[file_id]['point_ids'])} chunks)")
            to_delete.append(file_id)

    backfill = [fid for fid in files if fid not in to_delete and files[fid]["stored_hash"] != files[fid]["hash"]]
    print(f"{len(to_delete)} file(s) to delete, {len(backfill)} file(s) need content_sha256 backfilled.")
    if dry_run:
        print("Dry run: no changes written.")
        return

    client = store.get_client()
    for file_id in backfill:
        client.set_payload(
            collection_name=settings.qdrant.collection,
            payload={"content_sha256": files[file_id]["hash"]},
            points=files[file_id]["point_ids"],
        )
    for file_id in to_delete:
        store.delete_file(file_id)
    deleted_chunks = sum(len(files[fid]["point_ids"]) for fid in to_delete)
    print(f"Deleted {len(to_delete)} duplicate file(s) ({deleted_chunks} chunks); backfilled {len(backfill)} file(s).")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Report decisions without writing.")
    args = parser.parse_args()

    try:
        dedupe(dry_run=args.dry_run)
    except ExternalServiceError as e:
        log.error("Dedupe failed: %s", e)
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
