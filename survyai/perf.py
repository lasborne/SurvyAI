"""Lightweight stage timings and operation counters for SurvyAI pipelines.

Enabled by default for structured logging; attach the resulting dict to
internal pipeline results under ``perf`` without changing user-facing text.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

_TLS = threading.local()


def _enabled() -> bool:
    raw = str(os.environ.get("SURVYAI_PERF", "1") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


class PerfTracker:
    """Collect monotonic stage spans and integer counters for one request."""

    def __init__(self, *, pipeline: str = "") -> None:
        self.pipeline = pipeline or ""
        self.spans: List[Dict[str, Any]] = []
        self.counters: Dict[str, int] = {}
        self.meta: Dict[str, Any] = {}
        self._t0 = time.perf_counter()

    def incr(self, name: str, amount: int = 1) -> None:
        self.counters[name] = int(self.counters.get(name, 0) or 0) + int(amount)

    def set_meta(self, key: str, value: Any) -> None:
        self.meta[key] = value

    @contextmanager
    def span(self, name: str, **extra: Any) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            entry: Dict[str, Any] = {"name": name, "ms": round(ms, 1)}
            if extra:
                entry.update(extra)
            self.spans.append(entry)

    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000.0, 1)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "total_ms": self.elapsed_ms(),
            "spans": list(self.spans),
            "counters": dict(self.counters),
            "meta": dict(self.meta),
        }

    def summary_line(self) -> str:
        parts = [f"pipeline={self.pipeline or '?'}", f"total_ms={self.elapsed_ms():.0f}"]
        for s in self.spans[-8:]:
            parts.append(f"{s.get('name')}={s.get('ms')}ms")
        if self.counters:
            top = sorted(self.counters.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
            parts.append("counts=" + ",".join(f"{k}:{v}" for k, v in top))
        return " | ".join(parts)


def get_tracker() -> Optional[PerfTracker]:
    return getattr(_TLS, "tracker", None)


def set_tracker(tracker: Optional[PerfTracker]) -> None:
    _TLS.tracker = tracker


@contextmanager
def track_pipeline(pipeline: str) -> Iterator[PerfTracker]:
    """Bind a tracker to the current thread for the duration of a pipeline."""
    tracker = PerfTracker(pipeline=pipeline)
    prev = get_tracker()
    set_tracker(tracker)
    try:
        yield tracker
    finally:
        set_tracker(prev)


@contextmanager
def span(name: str, **extra: Any) -> Iterator[None]:
    tracker = get_tracker()
    if tracker is None or not _enabled():
        yield
        return
    with tracker.span(name, **extra):
        yield


def incr(name: str, amount: int = 1) -> None:
    tracker = get_tracker()
    if tracker is None or not _enabled():
        return
    tracker.incr(name, amount)


def set_meta(key: str, value: Any) -> None:
    tracker = get_tracker()
    if tracker is None or not _enabled():
        return
    tracker.set_meta(key, value)


__all__ = [
    "PerfTracker",
    "get_tracker",
    "set_tracker",
    "track_pipeline",
    "span",
    "incr",
    "set_meta",
]
