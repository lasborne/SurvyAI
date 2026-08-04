"""
Primary LLM selection routing for the SurvyAI desktop agent.

UI can persist symbolic selections such as ``auto``. Those must be resolved to a
concrete provider id (openai/gemini/claude/deepseek/ollama) before Settings /
agent initialization — the agent never speaks a symbolic provider name.

``auto`` means: pick the best paid hosted model for the task. Today that is
OpenAI (the only paid provider configured in production). This mapping is the
single place to expand when more paid providers become available.

The desktop shell resets the persisted primary selection to ``auto`` on every
application cold start; users may switch providers for the current session.
"""

from __future__ import annotations

from typing import Final

AUTO_PRIMARY_LLM: Final[str] = "auto"
# Best paid hosted provider while only OpenAI is configured for SurvyAI Pro.
DEFAULT_AUTO_PAID_PROVIDER: Final[str] = "openai"

_CONCRETE_PROVIDERS: Final[frozenset[str]] = frozenset(
    {"openai", "gemini", "claude", "deepseek", "ollama"}
)


def normalize_primary_llm_selection(selection: str | None) -> str:
    """Normalize a UI/state primary selection. Empty → ``auto`` (product default)."""
    s = str(selection or "").strip().lower()
    return s or AUTO_PRIMARY_LLM


def resolve_primary_llm_selection(selection: str | None) -> str:
    """
    Map a persisted primary selection to a concrete provider for Settings/agent.

    - ``auto`` (or empty) → best paid provider (currently OpenAI)
    - concrete provider ids pass through unchanged
    """
    s = normalize_primary_llm_selection(selection)
    if s == AUTO_PRIMARY_LLM:
        return DEFAULT_AUTO_PAID_PROVIDER
    if s in _CONCRETE_PROVIDERS:
        return s
    # Unknown values: keep as-is so validation / startup can surface a clear error.
    return s


__all__ = [
    "AUTO_PRIMARY_LLM",
    "DEFAULT_AUTO_PAID_PROVIDER",
    "normalize_primary_llm_selection",
    "resolve_primary_llm_selection",
]
