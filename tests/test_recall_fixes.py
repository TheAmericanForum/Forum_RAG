"""Unit tests for the retrieval-recall fixes: canonical policy-area labels,
duplicate-recording fingerprinting/preference, per-source search diversity, and
the synthesis context cap. No live search calls — the Qdrant client is faked."""
from __future__ import annotations

from types import SimpleNamespace

from forum_rag import store
from forum_rag.agent import _cap_for_synthesis
from forum_rag.classify import canonical_area
from forum_rag.config import get_settings

from ingest_data import content_fingerprint, dup_rank


def test_canonical_area_fixes_case_drift():
    canonical = get_settings().policy_areas[0].name
    assert canonical_area(canonical) == canonical
    assert canonical_area(canonical.title()) == canonical
    assert canonical_area(canonical.upper()) == canonical


def test_canonical_area_unknown_label_is_other():
    assert canonical_area("Not A Configured Area") == "other"
    assert canonical_area(None) == "other"
    assert canonical_area("OTHER") == "other"


def test_content_fingerprint_identity():
    texts = ["S1: hello", "S2: world"]
    assert content_fingerprint(texts) == content_fingerprint(list(texts))
    assert content_fingerprint(texts) != content_fingerprint(["S1: hello", "S2: world!"])
    # Boundary-sensitive: shifting text across chunks must change the fingerprint.
    assert content_fingerprint(["ab", "c"]) != content_fingerprint(["a", "bc"])


def test_dup_rank_prefers_deeper_path():
    nested = "NH1 - In-Person Recordings/Concord/Concord - table 8 - 2026-06-18.m4a"
    flat = "NH1 - In-Person Recordings/NH1 - In-Person Recordings - table 8 - 2026-06-18.m4a"
    assert dup_rank(nested) < dup_rank(flat)
    assert min([flat, nested], key=dup_rank) == nested


def _fake_points(spec: list[tuple[str, float]]):
    """Build fake Qdrant points from (transcript_id, score) pairs (descending order)."""
    return SimpleNamespace(
        points=[
            SimpleNamespace(score=score, payload={"transcript_id": tid, "chunk_id": f"{tid}:{i}"})
            for i, (tid, score) in enumerate(spec)
        ]
    )


def test_search_caps_results_per_transcript(monkeypatch):
    max_per_source = get_settings().retrieval.max_per_source
    spec = [("big", 1.0 - i * 0.01) for i in range(20)] + [("small", 0.5), ("tiny", 0.4)]
    captured = {}

    def fake_query_points(collection_name, query, limit, query_filter, with_payload):
        captured["limit"] = limit
        return _fake_points(spec)

    monkeypatch.setattr(store, "get_client", lambda: SimpleNamespace(query_points=fake_query_points))
    results = store.search([0.0], top_k=8)

    assert captured["limit"] > 8  # oversampled so the cap has candidates to backfill from
    by_source = {}
    for r in results:
        by_source[r["transcript_id"]] = by_source.get(r["transcript_id"], 0) + 1
    assert by_source["big"] == max_per_source
    assert "small" in by_source and "tiny" in by_source


def test_search_session_filter_skips_diversity_cap(monkeypatch):
    spec = [("one-session", 1.0 - i * 0.01) for i in range(10)]
    monkeypatch.setattr(
        store,
        "get_client",
        lambda: SimpleNamespace(query_points=lambda **kwargs: _fake_points(spec)),
    )
    results = store.search([0.0], top_k=8, session="Concord")
    assert len(results) == 8  # a session-scoped search must not be truncated per source


def test_cap_for_synthesis_bounds_and_diversifies():
    retrieval = get_settings().retrieval
    chunks = [
        {"transcript_id": "big", "chunk_id": f"big:{i}", "score": 0.9 - i * 0.001}
        for i in range(retrieval.max_synthesis_chunks + 10)
    ] + [{"transcript_id": "small", "chunk_id": f"small:{i}", "score": 0.5} for i in range(4)]

    capped = _cap_for_synthesis(chunks)
    assert len(capped) == retrieval.max_synthesis_chunks
    # The low-scoring minority transcript survives the cap instead of being crowded out
    # (it gets its per-source share, not zero).
    assert sum(1 for c in capped if c["transcript_id"] == "small") == min(4, retrieval.max_per_source)


def test_cap_for_synthesis_no_op_under_limit():
    chunks = [{"transcript_id": "t", "chunk_id": f"t:{i}", "score": 0.5} for i in range(5)]
    assert _cap_for_synthesis(chunks) == chunks
