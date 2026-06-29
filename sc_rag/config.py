"""Typed configuration: non-secret settings from config.yaml + secrets from env."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from .errors import ConfigError
from .logging import setup_logging

load_dotenv()
setup_logging()

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


class PolicyArea(BaseModel):
    name: str
    description: str = ""


class ChunkCfg(BaseModel):
    target_tokens: int = 350
    overlap_turns: int = 1
    max_turns_per_chunk: int = 40


class ModelsCfg(BaseModel):
    embed: str = "text-embedding-3-large"
    classify: str = "claude-haiku-4-5"
    agent: str = "claude-opus-4-8"


class RetrievalCfg(BaseModel):
    top_k: int = 8


class QdrantCfg(BaseModel):
    collection: str = "chunks"
    vector_size: int = 3072
    local_path: str = "./.qdrant"


class Settings(BaseModel):
    policy_areas: list[PolicyArea] = []
    chunk: ChunkCfg = ChunkCfg()
    models: ModelsCfg = ModelsCfg()
    retrieval: RetrievalCfg = RetrievalCfg()
    qdrant: QdrantCfg = QdrantCfg()

    # secrets / env (populated in get_settings)
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    google_service_account_json: Optional[str] = None
    drive_folder_ids: list[str] = []

    # auth (Google SSO)
    session_secret_key: Optional[str] = None
    google_oauth_client_id: Optional[str] = None
    google_oauth_client_secret: Optional[str] = None
    allowed_emails: list[str] = []
    contact_email: Optional[str] = None

    @property
    def policy_area_names(self) -> list[str]:
        return [a.name for a in self.policy_areas]

    @property
    def has_policy_areas(self) -> bool:
        # Treat the placeholder template as "not configured".
        return bool(self.policy_areas) and not all(
            a.name.startswith("Policy Area ") for a in self.policy_areas
        )

    def require_anthropic_key(self) -> str:
        if not self.anthropic_api_key:
            raise ConfigError(
                "ANTHROPIC_API_KEY is not set. Add it to .env (local) or Heroku config vars."
            )
        return self.anthropic_api_key

    def require_openai_key(self) -> str:
        if not self.openai_api_key:
            raise ConfigError(
                "OPENAI_API_KEY is not set. Add it to .env (local) or Heroku config vars."
            )
        return self.openai_api_key

    def require_session_secret(self) -> str:
        if not self.session_secret_key:
            raise ConfigError(
                "SESSION_SECRET_KEY is not set. Add it to .env (local) or Heroku config vars."
            )
        return self.session_secret_key

    def require_google_oauth(self) -> tuple[str, str]:
        if not self.google_oauth_client_id or not self.google_oauth_client_secret:
            raise ConfigError(
                "GOOGLE_OAUTH_CLIENT_ID / GOOGLE_OAUTH_CLIENT_SECRET are not set. "
                "Add them to .env (local) or Heroku config vars."
            )
        return self.google_oauth_client_id, self.google_oauth_client_secret


@lru_cache
def get_settings() -> Settings:
    data: dict = {}
    if CONFIG_PATH.exists():
        try:
            data = yaml.safe_load(CONFIG_PATH.read_text()) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"config.yaml is not valid YAML: {e}") from e
    else:
        log.warning("config.yaml not found at %s; using defaults.", CONFIG_PATH)
    s = Settings(**data)
    s.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    s.openai_api_key = os.getenv("OPENAI_API_KEY")
    s.qdrant_url = os.getenv("QDRANT_URL")
    s.qdrant_api_key = os.getenv("QDRANT_API_KEY")
    s.google_service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    ids = os.getenv("DRIVE_FOLDER_IDS", "")
    s.drive_folder_ids = [x.strip() for x in ids.split(",") if x.strip()]

    s.session_secret_key = os.getenv("SESSION_SECRET_KEY")
    s.google_oauth_client_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
    s.google_oauth_client_secret = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")
    emails = os.getenv("ALLOWED_EMAILS", "")
    s.allowed_emails = [x.strip().lower() for x in emails.split(",") if x.strip()]
    s.contact_email = os.getenv("CONTACT_EMAIL")
    return s
