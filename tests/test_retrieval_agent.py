"""(c), expensive layer: does the REAL agent (Sonnet retrieval planner + Opus
synthesis, forum_rag.agent.answer()) actually cite each transcript for a realistic
question grounded in it?

This is the layer most likely to reproduce manually-observed failures where the
agent doesn't pull from a relevant transcript — a passage can pass
test_retrieval_vector.py (proving it's indexed and embeddable) yet still never get
cited here if the retrieval planner never issues a search that surfaces it, or issues
too few/too narrow searches across its `max_rounds`.

Marked @pytest.mark.e2e (see pytest.ini: excluded from a bare `pytest` run) since it
costs real Anthropic tokens per case. Run explicitly:
  pytest -m e2e tests/test_retrieval_agent.py -v
"""
from __future__ import annotations

import json

import pytest

from forum_rag import agent

from .conftest import fixture_path

pytestmark = pytest.mark.e2e


def _load_cases() -> list[dict]:
    path = fixture_path()
    if not path.exists():
        return []
    return json.loads(path.read_text())


_CASES = _load_cases()


def test_fixture_has_questions(question_cases):
    assert question_cases


@pytest.mark.parametrize("case", _CASES, ids=[c["transcript_id"] for c in _CASES])
def test_agent_cites_expected_transcript(case):
    expected_prefix = f"{case['transcript_id']}:"
    sources = []
    for event in agent.answer(case["question"], policy_area=case["policy_area"]):
        if event["type"] == "done":
            sources = event["sources"]

    found = any((s["source"] or {}).get("chunk_id", "").startswith(expected_prefix) for s in sources)
    got = [(s["source"] or {}).get("chunk_id") for s in sources]
    assert found, (
        f"transcript_id={case['transcript_id']!r} was not cited by the agent.\n"
        f"  question={case['question']!r}\n"
        f"  policy_area={case['policy_area']!r}\n"
        f"  expected_chunk_id={case['expected_chunk_id']!r}\n"
        f"  chunk_ids actually cited: {got}"
    )
