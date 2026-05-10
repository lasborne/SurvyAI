"""
Configuration helpers for desktop and automated tests.

Phase 1 keeps `get_settings()` as the default source of truth (.env + env vars).
Callers that need injected keys (e.g. merged from a cloud login) build a fresh
`Settings` instance with `merge_settings()` and pass it to `SurvyAIAgentService`.
"""

from __future__ import annotations

from typing import Any

from config import Settings, get_settings


def merge_settings(**overrides: Any) -> Settings:
    """
    Return a new Settings instance from the current env-backed values with
    explicit field overrides. Does not mutate the global settings singleton.

    Example:
        from agent.agent import SurvyAIAgent
        s = merge_settings(openai_api_key="sk-...", survyai_access_token="...")
        agent = SurvyAIAgent(settings=s)
    """
    base = get_settings()
    return base.model_copy(update=overrides)


__all__ = ["merge_settings"]
