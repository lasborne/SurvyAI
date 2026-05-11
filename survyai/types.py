"""
Structured types for the service layer (GUI, CLI, and future API gateway).

These wrap the dict returned by SurvyAIAgent.process_query so callers do not
depend on raw key names everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentRunResult:
    """Normalized result of a single agent task."""

    success: bool
    response: str
    query: str
    session_id: str
    llm_used: Optional[str] = None
    model_name: Optional[str] = None
    error: Optional[str] = None
    context_retrieved: Optional[bool] = None
<<<<<<< HEAD
    llm_cost_usd: float = 0.0
=======
>>>>>>> a7b8ca66d633fcc18cfb695d86c8b7d288367d37
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_process_query_dict(cls, data: Dict[str, Any]) -> AgentRunResult:
        return cls(
            success=bool(data.get("success", True)),
            response=str(data.get("response") or ""),
            query=str(data.get("query") or ""),
            session_id=str(data.get("session_id") or ""),
            llm_used=data.get("llm_used"),
            model_name=data.get("model_name"),
            error=data.get("error"),
            context_retrieved=data.get("context_retrieved"),
<<<<<<< HEAD
            llm_cost_usd=float(data.get("llm_cost_usd") or 0.0),
=======
>>>>>>> a7b8ca66d633fcc18cfb695d86c8b7d288367d37
            raw=dict(data),
        )

    @property
    def permission_error_code(self) -> Optional[str]:
        """e.g. 'internet_permission_required' when interactive approval is needed."""
        err = self.raw.get("error")
        return str(err) if err else None


__all__ = ["AgentRunResult"]
