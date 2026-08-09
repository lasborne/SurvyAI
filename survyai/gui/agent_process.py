"""
Persistent, warm agent worker process for the SurvyAI desktop GUI.

Why this exists
---------------
The agent is *extremely* expensive to construct: building one ``SurvyAIAgent``
imports the full dependency tree (LangChain, LangGraph, GeoPandas, ezdxf,
sentence-transformers, …), loads a local embedding model, initialises the
primary + fallback LLMs, creates every tool, and builds/compiles the LangGraph.
On Windows this easily costs ~30-45 seconds.

The previous design spawned a *brand new* Python process — and therefore paid
that entire cold-start cost — on **every single prompt**. That is the dominant
source of the "it takes ~45s before anything happens" latency.

This module keeps **one** worker process alive for the lifetime of the app. The
heavy agent is built **once** and then reused for every prompt, so steady-state
latency collapses to just the real LLM round-trip. Cancellation terminates the
process (a rare event); the next prompt transparently respawns + re-warms it.

The agent logic, routing, tools, accuracy and output quality are completely
unchanged — only the process lifecycle is optimised.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
import threading
import traceback
import uuid
from typing import Any, Dict, Optional


# Windows SetThreadExecutionState flags (keep system awake during runs).
# Do NOT include ES_DISPLAY_REQUIRED: forcing the display on during long/hung
# local LLM calls contributes to thermal/power stress and poor lock-screen UX.
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def _windows_acquire_run_awake() -> bool:
    """Ask Windows not to sleep/hibernate while a SurvyAI task is active."""
    if os.name != "nt":
        return False
    try:
        import ctypes

        flags = _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))
    except Exception:
        return False


def _windows_release_run_awake() -> None:
    """Clear the keep-awake request so normal power policy resumes."""
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
    except Exception:
        pass


# Auth/session fields that rotate often (token refresh / cloud poll) but must NOT
# force a full SurvyAIAgent rebuild. Hot-swapped onto the live service instead.
_EPHEMERAL_SETTINGS_KEYS = frozenset(
    {
        "survyai_access_token",
        "survyai_refresh_token",  # if ever present in settings dumps
    }
)

# Bump when agent routing/pipelines change so a running app picks up new logic.
_WORKER_CODE_REV = "20260809-warm-auth-hotswap-v1"


def _structural_settings_payload(settings_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Settings used for rebuild fingerprint (excludes rotating auth tokens)."""
    if not isinstance(settings_payload, dict):
        return {}
    return {k: v for k, v in settings_payload.items() if k not in _EPHEMERAL_SETTINGS_KEYS}


def _payload_signature(settings_payload: Dict[str, Any], ff_payload: Dict[str, Any]) -> str:
    """Stable fingerprint of the inputs the agent is built from.

    When this changes (e.g. user switches primary LLM or toggles a feature flag
    in Settings) the warm worker rebuilds the agent; otherwise it reuses it.

    Access tokens are excluded: they rotate on cloud refresh and are hot-swapped
    onto the live agent so Pro sessions stay warm.
    """
    structural = _structural_settings_payload(settings_payload)
    try:
        return json.dumps([structural, ff_payload, _WORKER_CODE_REV], sort_keys=True, default=str)
    except Exception:
        return repr((structural, ff_payload, _WORKER_CODE_REV))


def _agent_worker_loop(in_queue: "multiprocessing.Queue", out_queue: "multiprocessing.Queue") -> None:
    """Long-lived child-process loop.

    Heavy imports are done lazily *inside* this function so that importing this
    module in the parent (and re-importing it during ``spawn`` bootstrap) stays
    cheap. The agent/service is constructed once and reused across requests.
    """
    try:
        from config import Settings
        from survyai.agent_service import SurvyAIAgentService
        from survyai.feature_flags import FeatureFlags
    except Exception:
        try:
            out_queue.put({"kind": "fatal", "payload": traceback.format_exc()})
        except Exception:
            pass
        return

    service: Optional[Any] = None
    current_sig: Optional[str] = None
    base_cwd = os.getcwd()

    def _ensure_service(settings_payload: Dict[str, Any], ff_payload: Dict[str, Any]):
        nonlocal service, current_sig
        sig = _payload_signature(settings_payload, ff_payload)
        settings = Settings(**(settings_payload or {}))
        feature_flags = FeatureFlags(**(ff_payload or {}))
        if service is None or sig != current_sig:
            service = SurvyAIAgentService(
                settings=settings,
                feature_flags=feature_flags,
                eager_init=True,  # build the heavy agent now, once
            )
            current_sig = sig
        else:
            # Same structural config: hot-swap rotating cloud auth onto the warm agent.
            try:
                service.apply_runtime_auth(settings)
            except Exception:
                pass
            try:
                service.feature_flags = feature_flags
            except Exception:
                pass
        return service

    while True:
        try:
            req = in_queue.get()
        except (EOFError, OSError, KeyboardInterrupt):
            break
        if req is None:
            break

        kind = req.get("kind")
        if kind == "shutdown":
            break

        req_id = req.get("req_id")
        keep_awake = False
        try:
            # Apply the workspace directory for this request. Reset to the
            # original cwd when none is supplied so requests stay isolated.
            wd = req.get("working_directory")
            target_wd = wd if wd else base_cwd
            try:
                if target_wd:
                    from pathlib import Path as _Path

                    _Path(str(target_wd)).mkdir(parents=True, exist_ok=True)
                    os.chdir(str(target_wd))
            except Exception as chdir_err:
                try:
                    out_queue.put(
                        {
                            "kind": "progress",
                            "req_id": req_id,
                            "payload": {
                                "message": (
                                    f"Warning: could not activate workspace "
                                    f"'{target_wd}': {chdir_err}. "
                                    f"Outputs may fall back to {base_cwd}."
                                )
                            },
                        }
                    )
                except Exception:
                    pass

            svc = _ensure_service(
                req.get("settings_payload") or {},
                req.get("feature_flags_payload") or {},
            )

            if kind == "warmup":
                out_queue.put({"kind": "warmed", "req_id": req_id})
                continue

            # Skip keep-awake for local Ollama: it does not prevent RAM thrash/hibernate
            # and can interact poorly with laptop power policy during long CPU inference.
            settings_payload = req.get("settings_payload") or {}
            use_fb = bool(req.get("use_fallback_llm", False))
            active_llm = str(
                settings_payload.get("fallback_llm" if use_fb else "primary_llm", "") or ""
            ).strip().lower()
            keep_awake = active_llm != "ollama"
            if keep_awake:
                _windows_acquire_run_awake()
            try:
                # Route CAD file-conflict prompts to the GUI via the out/in queues
                # so the styled dialog appears on the SurvyAI window (not a hidden MessageBox).
                try:
                    from agent.agent import set_cad_file_conflict_handler

                    def _ask_cad_conflict(path: str, mode: str = "overwrite") -> bool:
                        out_queue.put(
                            {
                                "kind": "confirm_overwrite",
                                "req_id": req_id,
                                "payload": {
                                    "path": str(path or ""),
                                    "mode": str(mode or "overwrite"),
                                },
                            }
                        )
                        # Timed get so we can emit heartbeats and never sit forever
                        # if the GUI reply is lost (previous hang after Overwrite click).
                        waited_s = 0.0
                        while True:
                            try:
                                reply = in_queue.get(timeout=1.0)
                            except queue.Empty:
                                waited_s += 1.0
                                if waited_s >= 15.0 and int(waited_s) % 15 == 0:
                                    try:
                                        out_queue.put(
                                            {
                                                "kind": "confirm_overwrite_waiting",
                                                "req_id": req_id,
                                                "payload": {
                                                    "message": (
                                                        "Still waiting for overwrite confirmation "
                                                        "in SurvyAI…"
                                                    ),
                                                    "waited_s": waited_s,
                                                },
                                            }
                                        )
                                    except Exception:
                                        pass
                                continue
                            except (EOFError, OSError, KeyboardInterrupt):
                                return False
                            if reply is None:
                                return False
                            if reply.get("kind") == "shutdown":
                                return False
                            if (
                                reply.get("kind") == "confirm_overwrite_reply"
                                and reply.get("req_id") == req_id
                            ):
                                payload = reply.get("payload") or {}
                                return bool(payload.get("accepted"))
                            # Ignore unrelated mid-run messages.

                    set_cad_file_conflict_handler(_ask_cad_conflict)
                except Exception:
                    pass

                try:
                    result = svc.run_task(
                        req.get("query") or "",
                        use_fallback_llm=bool(req.get("use_fallback_llm", False)),
                        session_id=req.get("session_id"),
                        interactive=bool(req.get("interactive", False)),
                    )
                    out_queue.put({"kind": "result", "req_id": req_id, "payload": result.raw})
                finally:
                    try:
                        from agent.agent import set_cad_file_conflict_handler

                        set_cad_file_conflict_handler(None)
                    except Exception:
                        pass
            finally:
                if keep_awake:
                    _windows_release_run_awake()
        except Exception:
            try:
                if keep_awake:
                    _windows_release_run_awake()
            except Exception:
                pass
            try:
                from agent.agent import set_cad_file_conflict_handler

                set_cad_file_conflict_handler(None)
            except Exception:
                pass
            out_queue.put({"kind": "error", "req_id": req_id, "payload": traceback.format_exc()})


class PersistentAgentProcess:
    """Manages a single warm worker process and a request/response channel.

    Only one request is ever in flight at a time (the GUI disables submit while
    a run is active), which keeps the queue protocol simple: callers match
    responses by ``req_id`` and ignore anything that doesn't belong to them
    (e.g. a leftover ``warmed`` acknowledgement).
    """

    def __init__(self) -> None:
        self._ctx = multiprocessing.get_context("spawn")
        self._proc: Optional[multiprocessing.Process] = None
        self._in: Optional[multiprocessing.Queue] = None
        self._out: Optional[multiprocessing.Queue] = None
        self._lock = threading.Lock()

    def ensure_started(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                return
            # Clean up any dead handles before respawning.
            self._close_queues_locked()
            self._in = self._ctx.Queue()
            self._out = self._ctx.Queue()
            self._proc = self._ctx.Process(
                target=_agent_worker_loop,
                args=(self._in, self._out),
                daemon=True,
            )
            self._proc.start()

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc is not None else None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def submit(self, request: Dict[str, Any]) -> str:
        """Queue a request for the worker. Returns the request id used to match
        the eventual response."""
        self.ensure_started()
        req_id = request.get("req_id") or uuid.uuid4().hex
        request["req_id"] = req_id
        assert self._in is not None
        self._in.put(request)
        return req_id

    def warmup(self, settings_payload: Dict[str, Any], feature_flags_payload: Dict[str, Any]) -> str:
        """Fire-and-forget: build the agent ahead of the first real prompt."""
        return self.submit(
            {
                "kind": "warmup",
                "settings_payload": settings_payload,
                "feature_flags_payload": feature_flags_payload,
            }
        )

    def poll(self, timeout: float = 0.3) -> Optional[Dict[str, Any]]:
        if self._out is None:
            return None
        try:
            return self._out.get(timeout=timeout)
        except (queue.Empty, OSError, ValueError):
            return None

    def kill(self) -> None:
        """Terminate the worker (used on cancel). The next request respawns it."""
        with self._lock:
            proc = self._proc
            self._proc = None
            try:
                if proc is not None:
                    if proc.is_alive():
                        proc.terminate()
                        proc.join(timeout=5)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=2)
            except Exception:
                pass
            self._close_queues_locked()

    def _close_queues_locked(self) -> None:
        for q_ in (self._in, self._out):
            try:
                if q_ is not None:
                    q_.close()
            except Exception:
                pass
        self._in = None
        self._out = None


_shared_process: Optional[PersistentAgentProcess] = None
_shared_lock = threading.Lock()


def get_shared_agent_process() -> PersistentAgentProcess:
    """Return the process-wide warm agent worker, creating it on first use."""
    global _shared_process
    with _shared_lock:
        if _shared_process is None:
            _shared_process = PersistentAgentProcess()
        return _shared_process


def prewarm_shared_agent_process(
    settings_payload: Dict[str, Any],
    feature_flags_payload: Dict[str, Any],
) -> None:
    """Start + warm the shared worker so the first prompt is fast too.

    Safe to call from a background thread; it only enqueues work and returns
    immediately (the heavy build happens inside the child process).
    """
    try:
        proc = get_shared_agent_process()
        proc.warmup(settings_payload, feature_flags_payload)
    except Exception:
        pass


def shutdown_shared_agent_process() -> None:
    """Best-effort teardown of the warm worker (call on app exit)."""
    global _shared_process
    with _shared_lock:
        proc = _shared_process
        _shared_process = None
    if proc is not None:
        proc.kill()


__all__ = [
    "PersistentAgentProcess",
    "get_shared_agent_process",
    "prewarm_shared_agent_process",
    "shutdown_shared_agent_process",
    "_payload_signature",
    "_structural_settings_payload",
    "_EPHEMERAL_SETTINGS_KEYS",
]
