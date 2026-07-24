"""Shared fixtures for the integration test suite.

Every test here talks to real, live services (Drive, Qdrant, OpenAI, Anthropic) using
the same credentials ingest_data.py and the app itself need — there's no mocking, this
is a smoke/audit suite, not a unit-test suite. If the required secrets aren't
configured in this environment, skip the whole suite with a clear message rather than
failing with a confusing lower-level error.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from forum_rag.config import get_settings

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def fixture_path(collection: str | None = None) -> Path:
    """Path to the question fixture for one Qdrant collection.

    Scoped per collection (not a single shared file) because this repo is
    multi-tenant — each tenant has its own collection (sc_chunks, nh_chunks,
    nv_chunks) with entirely different transcripts. A shared file would mix
    transcript_ids across tenants, so a question grounded in an SC transcript
    would get searched against NH's collection and fail spuriously.
    """
    collection = collection or get_settings().qdrant.collection or "default"
    return FIXTURE_DIR / f"test_questions.{collection}.json"


_REQUIRED_ENV_SETTINGS = [
    ("anthropic_api_key", "ANTHROPIC_API_KEY"),
    ("openai_api_key", "OPENAI_API_KEY"),
    ("google_service_account_json", "GOOGLE_SERVICE_ACCOUNT_JSON"),
]


@pytest.fixture(scope="session", autouse=True)
def _require_live_credentials():
    settings = get_settings()
    missing = [env_name for attr, env_name in _REQUIRED_ENV_SETTINGS if not getattr(settings, attr)]
    if not settings.drive_folder_ids:
        missing.append("DRIVE_FOLDER_IDS")
    if missing:
        pytest.skip(
            "Skipping integration test suite: missing required env var(s) "
            f"{', '.join(missing)}. These tests hit live Drive/Qdrant/OpenAI/Anthropic "
            "services and need the same credentials as ingest_data.py (see .env.example)."
        )


@pytest.fixture(scope="session")
def question_cases() -> list[dict]:
    path = fixture_path()
    if not path.exists():
        pytest.fail(f"{path} not found. Generate it first: python generate_test_questions.py")
    cases = json.loads(path.read_text())
    if not cases:
        pytest.fail(f"{path} is empty. Generate it first: python generate_test_questions.py")
    return cases
