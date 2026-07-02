"""Shared error types and retry policy for external API calls.

Anthropic/OpenAI SDK errors that are 4xx (bad request, auth, insufficient
credit, not found) are not transient — retrying them just burns time and
hides the real problem. Only rate limits (429) and 5xx/connection errors
are worth retrying.
"""
from __future__ import annotations


class ConfigError(RuntimeError):
    """Missing or invalid configuration (env vars, config.yaml)."""


class ExternalServiceError(RuntimeError):
    """Wraps a non-retryable failure from Anthropic/OpenAI/Qdrant/Drive with context."""


def is_retryable_api_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        # Connection errors etc. have no status_code — worth retrying.
        return not isinstance(exc, (ValueError, TypeError, KeyError))
    if status == 429:
        return True
    return status >= 500
