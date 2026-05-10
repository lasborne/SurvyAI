"""
Background execution of `SurvyAIAgentService.run_task`.

The GUI needs true cancellation, so the blocking agent run happens inside a child
process. The Qt thread stays responsive, can emit progress text, and can terminate
the child process immediately when the user clicks Cancel.
"""

from __future__ import annotations

import os
import multiprocessing
import queue
import traceback
from typing import Any, Optional

from PySide6.QtCore import QThread, Signal

from config import Settings
from survyai.feature_flags import FeatureFlags
from survyai.agent_service import SurvyAIAgentService
from survyai.types import AgentRunResult


def _run_agent_task_in_subprocess(
    result_queue: multiprocessing.Queue,
    *,
    settings_payload: dict,
    feature_flags_payload: dict,
    query: str,
    use_fallback_llm: bool,
    session_id: Optional[str],
    interactive: bool,
    working_directory: Optional[str],
) -> None:
    try:
        if working_directory:
            try:
                os.chdir(working_directory)
            except Exception:
                pass
        settings = Settings(**settings_payload)
        feature_flags = FeatureFlags(**feature_flags_payload)
        service = SurvyAIAgentService(
            settings=settings,
            feature_flags=feature_flags,
            eager_init=False,
        )
        result = service.run_task(
            query,
            use_fallback_llm=use_fallback_llm,
            session_id=session_id,
            interactive=interactive,
        )
        result_queue.put({"kind": "result", "payload": result.raw})
    except Exception:
        result_queue.put({"kind": "error", "payload": traceback.format_exc()})


class AgentRunThread(QThread):
    """Runs one killable `run_task` call off the GUI thread."""

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
        self._process: Optional[multiprocessing.Process] = None
        self._result_queue: Optional[multiprocessing.Queue] = None

    def request_cancel(self) -> None:
        self._cancel_requested = True
        self.progress_text.emit(
            "Cancellation requested. SurvyAI is terminating the active agent run now."
        )

    def _terminate_active_process(self) -> None:
        proc = self._process
        if proc is None:
            return
        try:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2)
        except Exception:
            pass

    def run(self) -> None:
        try:
            if self._working_directory:
                self.progress_text.emit(f"Workspace active: {self._working_directory}")
            self.progress_text.emit("Starting agent run…")

            ctx = multiprocessing.get_context("spawn")
            self._result_queue = ctx.Queue()
            self._process = ctx.Process(
                target=_run_agent_task_in_subprocess,
                kwargs={
                    "result_queue": self._result_queue,
                    "settings_payload": self._settings_payload,
                    "feature_flags_payload": self._feature_flags_payload,
                    "query": self._query,
                    "use_fallback_llm": self._use_fallback_llm,
                    "session_id": self._session_id,
                    "interactive": self._interactive,
                    "working_directory": self._working_directory,
                },
            )
            self._process.start()
            self.progress_text.emit(f"Agent process started (PID {self._process.pid}).")

            while True:
                if self._cancel_requested:
                    self._terminate_active_process()
                    self.cancelled.emit("Task cancelled. The active agent run was terminated.")
                    return

                try:
                    assert self._result_queue is not None
                    message = self._result_queue.get(timeout=0.35)
                except queue.Empty:
                    if self._process is not None and not self._process.is_alive():
                        break
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

            if self._cancel_requested:
                self.cancelled.emit("Task cancelled. The active agent run was terminated.")
            elif self._process is not None and self._process.exitcode not in (0, None):
                self.failed.emit(
                    f"Agent process exited unexpectedly with code {self._process.exitcode}."
                )
            else:
                self.failed.emit("Agent process ended without returning a result.")
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            try:
                self._terminate_active_process()
            except Exception:
                pass
            try:
                if self._result_queue is not None:
                    self._result_queue.close()
            except Exception:
                pass
            self._process = None
            self._result_queue = None
