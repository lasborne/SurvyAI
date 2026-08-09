"""
Service façade over SurvyAIAgent for GUI, CLI, and future HTTP API.

Phase 1 responsibilities:
- Lazy agent construction (optional) to allow capability checks without LLM init
- Session helpers
- Typed AgentRunResult instead of raw dicts
- Optional injected Settings for desktop / cloud token flows
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from config import Settings, get_settings

from survyai.feature_flags import FeatureFlags
from survyai.types import AgentRunResult


class SurvyAIAgentService:
    """
    Thin orchestration layer around `SurvyAIAgent`.

    Use `run_task()` for normal queries. Use `feature_flags` + `scan_machine_capabilities`
    from `survyai.capabilities` for UI gating.
    """

    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        feature_flags: Optional[FeatureFlags] = None,
        eager_init: bool = False,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        self.feature_flags = feature_flags if feature_flags is not None else FeatureFlags.from_env()
        self._agent: Optional["SurvyAIAgent"] = None
        if eager_init:
            self.ensure_agent()

    @property
    def settings(self) -> Settings:
        return self._settings

    def _get_agent(self) -> "SurvyAIAgent":
        if self._agent is None:
            from agent.agent import SurvyAIAgent

            self._agent = SurvyAIAgent(
                settings=self._settings,
                feature_flags=self.feature_flags,
            )
        return self._agent

    def ensure_agent(self) -> "SurvyAIAgent":
        """Force agent initialization (loads LLMs, tools, graph)."""
        return self._get_agent()

    def apply_runtime_auth(self, settings: Settings) -> None:
        """
        Hot-swap rotating cloud auth onto the live agent without rebuilding.

        Used by the warm worker when only ``survyai_access_token`` (etc.) changed.
        """
        self._settings = settings
        agent = self._agent
        if agent is None:
            return
        apply = getattr(agent, "apply_runtime_auth", None)
        if callable(apply):
            apply(settings)
        else:
            try:
                agent.settings = settings
            except Exception:
                pass

    def start_session(self, session_id: Optional[str] = None) -> str:
        """Set or create a conversation session id."""
        sid = session_id or str(uuid.uuid4())
        self._get_agent().set_session_id(sid)
        return sid

    def run_task(
        self,
        query: str,
        *,
        use_fallback_llm: bool = False,
        session_id: Optional[str] = None,
        interactive: bool = False,
    ) -> AgentRunResult:
        """
        Run one user task through the agent.

        `interactive` should be True when the UI can show permission prompts
        (e.g. internet search approval), matching CLI `--interactive`.
        """
        agent = self._get_agent()
        raw = agent.process_query(
            query,
            use_fallback=use_fallback_llm,
            session_id=session_id,
            interactive_mode=interactive,
        )
        raw = agent.finalize_query_result_dict(raw)
        return AgentRunResult.from_process_query_dict(raw)

    def run_task_raw(
        self,
        query: str,
        *,
        use_fallback_llm: bool = False,
        session_id: Optional[str] = None,
        interactive: bool = False,
    ) -> Dict[str, Any]:
        """Same as run_task but returns the original dict (escape hatch)."""
        agent = self._get_agent()
        raw = agent.process_query(
            query,
            use_fallback=use_fallback_llm,
            session_id=session_id,
            interactive_mode=interactive,
        )
        return agent.finalize_query_result_dict(raw)


__all__ = ["SurvyAIAgentService"]
