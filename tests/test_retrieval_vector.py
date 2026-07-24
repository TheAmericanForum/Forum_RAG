"""(c), cheap layer: for a realistic question grounded in each transcript, does the
vector-search layer alone (embed_query + store.search, no LLM planner/synthesis)
surface that transcript's content?

Cheap enough to run nightly (only OpenAI embedding + Qdrant calls, no Anthropic
tokens) — this is the layer that catches indexing/embedding/filter bugs. It does NOT
tell you whether the agent's retrieval *planner* would actually issue a query good
enough to find it; that's tests/test_retrieval_agent.py (run on demand only).

Parametrized one case per transcript so a failure names exactly which transcript
isn't retrievable, instead of one aggregate pass/fail for the whole corpus.
"""
from __future__ import annotations

import json

import pytest

from forum_rag.embed import embed_query
from forum_rag import store

from .conftest import fixture_path

TOP_K = 10


def _load_cases() -> list[dict]:
    path = fixture_path()
    if not path.exists():
        return []
    return json.loads(path.read_text())


_CASES = _load_cases()


def test_fixture_has_questions(question_cases):
    """Fails loudly (via the question_cases fixture) if tests/fixtures/test_questions.json
    is missing or empty, instead of the parametrized test below silently collecting zero
    cases and reporting nothing ran."""
    assert question_cases


@pytest.mark.parametrize("case", _CASES, ids=[c["transcript_id"] for c in _CASES])
def test_vector_search_recall(case):
    query_vector = embed_query(case["question"])
    results = store.search(query_vector, top_k=TOP_K, policy_area=case["policy_area"])
    found = any(
        r.get("transcript_id") == case["transcript_id"] or r.get("chunk_id") == case["expected_chunk_id"]
        for r in results
    )
    got = [(r.get("transcript_id"), r.get("chunk_id")) for r in results]
    assert found, (
        f"transcript_id={case['transcript_id']!r} not found via vector search.\n"
        f"  question={case['question']!r}\n"
        f"  policy_area={case['policy_area']!r}\n"
        f"  expected_chunk_id={case['expected_chunk_id']!r}\n"
        f"  top-{TOP_K} results: {got}"
    )
