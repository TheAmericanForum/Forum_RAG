"""Normalize case-drifted policy_areas payloads to the exact configured spellings.

Search hard-filters on byte-equal policy_areas labels, so a chunk stored with e.g.
"Housing Affordability And Ownership" (title-cased "And") never matches the canonical
"Housing Affordability and Ownership" filter and is unreachable. This sweeps the
configured collection and rewrites any label that case-insensitively matches a
configured area but isn't byte-equal. Payload-only (set_payload) — no re-embedding.

Runs against the collection resolved from the environment (TENANT / QDRANT_COLLECTION),
so run it once per tenant. Labels matching no configured area (even case-insensitively)
are left untouched and reported.

Usage:
  python normalize_policy_areas.py --dry-run   # report what would change
  python normalize_policy_areas.py             # apply
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict

from forum_rag import store
from forum_rag.config import get_settings
from forum_rag.errors import ExternalServiceError
from forum_rag.logging import setup_logging

log = logging.getLogger(__name__)


def normalize(dry_run: bool) -> None:
    settings = get_settings()
    if not settings.has_policy_areas:
        print("ERROR: no policy areas configured for this tenant; nothing to normalize.", file=sys.stderr)
        sys.exit(1)

    canonical_by_fold = {area.name.casefold(): area.name for area in settings.policy_areas}
    canonical_by_fold["other"] = "other"

    # {fixed_labels_tuple: [point_ids]} so each distinct rewrite is one batched set_payload
    to_fix: dict[tuple[str, ...], list] = defaultdict(list)
    fixed_labels: Counter[str] = Counter()
    unknown_labels: Counter[str] = Counter()
    total = 0

    for point in store.iter_all_points(with_payload=["policy_areas"]):
        total += 1
        labels = (point.payload or {}).get("policy_areas") or []
        fixed = []
        changed = False
        for label in labels:
            canonical = canonical_by_fold.get(label.casefold())
            if canonical is None:
                unknown_labels[label] += 1
                fixed.append(label)
            else:
                if canonical != label:
                    changed = True
                    fixed_labels[label] += 1
                fixed.append(canonical)
        if changed:
            to_fix[tuple(fixed)].append(point.id)

    fix_count = sum(len(ids) for ids in to_fix.values())
    print(f"Collection {settings.qdrant.collection!r}: {total} point(s) scanned, {fix_count} need normalizing.")
    for label, count in fixed_labels.most_common():
        print(f"  {count:4d}  {label!r} -> {canonical_by_fold[label.casefold()]!r}")
    for label, count in unknown_labels.most_common():
        print(f"  {count:4d}  ⚠️ {label!r} matches no configured area — left untouched")

    if dry_run or not to_fix:
        if dry_run and to_fix:
            print("Dry run: no changes written.")
        return

    client = store.get_client()
    for labels, point_ids in to_fix.items():
        client.set_payload(
            collection_name=settings.qdrant.collection,
            payload={"policy_areas": list(labels)},
            points=point_ids,
        )
    print(f"Normalized {fix_count} point(s).")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing.")
    args = parser.parse_args()

    try:
        normalize(dry_run=args.dry_run)
    except ExternalServiceError as e:
        log.error("Normalization failed: %s", e)
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
