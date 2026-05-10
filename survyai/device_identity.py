"""
Stable, privacy-conscious machine fingerprint for cloud device registration.

Used to bind a SurvyAI Pro subscription to a limited number of PCs via the cloud API
(POST /v1/devices + X-SurvyAI-Device-Id on bootstrap).
"""

from __future__ import annotations

import hashlib
import os
import uuid


def compute_machine_fingerprint() -> str:
    """
    Return a hex string (64 chars) derived from host identity signals.

    Not a secret; only used to recognize this machine for the same user account.
    """
    parts = [
        os.environ.get("COMPUTERNAME", "") or os.environ.get("HOSTNAME", ""),
        os.environ.get("USERDOMAIN", ""),
        os.environ.get("USERNAME", "") or os.environ.get("USER", ""),
        format(uuid.getnode(), "x"),
    ]
    raw = "|".join(p.strip() for p in parts).encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()
