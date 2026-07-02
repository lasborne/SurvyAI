"""
Background execution of `SurvyAIAgentService.run_task`.

The GUI needs true cancellation, so the blocking agent run happens inside a
**persistent, warm** child process (see `survyai.gui.agent_process`). That
process builds the heavy agent exactly once and reuses it for every prompt,
which removes the multi-second cold-start that previously occurred on each run.
The Qt thread stays responsive, emits progress text, and can terminate the
child process immediately when the user clicks Cancel.
"""

from __future__ import annotations

import traceback
from typing import Any, Optional

from PySide6.QtCore import QThread, Signal

from survyai.agent_service import SurvyAIAgentService
from survyai.gui.agent_process import get_shared_agent_process
from survyai.types import AgentRunResult


class AgentRunThread(QThread):
    """Runs one killable `run_task` call against the warm agent process."""

    result_ready = Signal(object)
    failed = Signal(str)
    progress_text = Signal(str)
    cancelled = Signal(str)

    def __init__(
        self,
        service: SurvyAIAgentService,
        query: str,
        *,
        use_fallback_llm: bool = False,
        session_id: Optional[str] = None,
        interactive: bool = True,
        working_directory: Optional[str] = None,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._query = query
        self._use_fallback_llm = use_fallback_llm
        self._session_id = session_id
        self._interactive = interactive
        self._working_directory = working_directory
        self._cancel_requested = False
        self._settings_payload = service.settings.model_dump()
        self._feature_flags_payload = {
            "license_mode": service.feature_flags.license_mode,
            "allow_autocad": service.feature_flags.allow_autocad,
            "allow_arcgis": service.feature_flags.allow_arcgis,
            "allow_blue_marble": service.feature_flags.allow_blue_marble,
            "allow_internet_tools": service.feature_flags.allow_internet_tools,
            "allow_vector_store": service.feature_flags.allow_vector_store,
        }

    def request_cancel(self) -> None:
        self._cancel_requested = True
        self.progress_text.emit(
            "Cancellation requested. SurvyAI is terminating the active agent run now."
        )

    def run(self) -> None:
        proc = get_shared_agent_process()
        try:
            if self._working_directory:
                self.progress_text.emit(f"Workspace active: {self._working_directory}")
            self.progress_text.emit("Starting agent run…")

            proc.ensure_started()

            req_id = proc.submit(
                {
                    "kind": "run",
                    "settings_payload": self._settings_payload,
                    "feature_flags_payload": self._feature_flags_payload,
                    "query": self._query,
                    "use_fallback_llm": self._use_fallback_llm,
                    "session_id": self._session_id,
                    "interactive": self._interactive,
                    "working_directory": self._working_directory,
                }
            )
            self.progress_text.emit(f"Agent process ready (PID {proc.pid}).")

            while True:
                if self._cancel_requested:
                    proc.kill()
                    self.cancelled.emit("Task cancelled. The active agent run was terminated.")
                    return

                if not proc.is_alive():
                    # Drain any final message the worker managed to emit.
                    message = proc.poll(timeout=0.1)
                    if message is None:
                        self.failed.emit("Agent process exited unexpectedly before returning a result.")
                        return
                else:
                    message = proc.poll(timeout=0.35)
                    if message is None:
                        continue

                # Ignore stale messages (e.g. a leftover warmup acknowledgement)
                # that don't belong to this run.
                if message.get("kind") in ("warmed",) or message.get("req_id") not in (req_id, None):
                    continue

                kind = message.get("kind")
                payload = message.get("payload")
                if kind == "result":
                    result = AgentRunResult.from_process_query_dict(payload or {})
                    self.result_ready.emit(result)
                    return
                if kind == "error":
                    self.failed.emit(str(payload or "Unknown agent subprocess error"))
                    return
                if kind == "fatal":
                    proc.kill()
                    self.failed.emit(str(payload or "Agent worker failed to initialise"))
                    return
        except Exception:
            self.failed.emit(traceback.format_exc())
