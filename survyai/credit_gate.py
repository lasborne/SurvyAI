from __future__ import annotations

import os

from survyai.device_identity import compute_machine_fingerprint


def credit_limit_enforcement_enabled() -> bool:
    """Return False when credit exhaustion should not block hosted LLM runs (builder PCs)."""
    raw = (os.environ.get("SURVYAI_BYPASS_CREDIT_LIMIT") or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return False
    want = (os.environ.get("SURVYAI_CREDIT_BYPASS_FINGERPRINT") or "").strip().lower()
    if want and compute_machine_fingerprint().lower() == want:
        return False
    return True


__all__ = ["credit_limit_enforcement_enabled"]

