"""Hugging Face network guard for SurvyAI.

``huggingface_hub`` performs network HEAD validation for every cached file unless
offline mode is set *before* ``sentence_transformers`` / ``transformers`` import.
On a machine with no DNS to ``huggingface.co`` that produces repeated 1/2/4/8/8s
retry storms (~23s per file) that dominate startup and first-embed latency.

This module decides — once, as early as possible — whether to force offline mode.
"""

from __future__ import annotations

import os
import socket
from typing import Optional

_CONFIGURED = False
_ETAG_TIMEOUT = "3"
_DOWNLOAD_TIMEOUT = "10"


def _model_is_cached(model_name: str) -> bool:
    name = (model_name or "").strip() or "all-MiniLM-L6-v2"
    candidates = [name]
    if "/" not in name:
        candidates.append(f"sentence-transformers/{name}")
    try:
        from huggingface_hub import try_to_load_from_cache

        for repo in candidates:
            for fname in ("config.json", "modules.json", "tokenizer_config.json"):
                try:
                    path = try_to_load_from_cache(repo_id=repo, filename=fname)
                except Exception:
                    path = None
                if path and path != "___not_found___":
                    return True
    except Exception:
        pass
    try:
        from pathlib import Path

        roots = [Path.home() / ".cache" / "huggingface" / "hub"]
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            roots.append(Path(hf_home) / "hub")
        for repo in candidates:
            short = repo.replace("/", "--")
            for root in roots:
                if root.exists() and (root / f"models--{short}").exists():
                    return True
    except Exception:
        pass
    return False


def _has_hf_connectivity(timeout_s: float = 1.5) -> bool:
    try:
        sock = socket.create_connection(("huggingface.co", 443), timeout=timeout_s)
        sock.close()
        return True
    except Exception:
        return False


def configure_hf_offline_if_appropriate(model_name: Optional[str] = None) -> bool:
    """Set HF/Transformers env vars once. Returns True if offline mode was forced."""
    global _CONFIGURED
    if _CONFIGURED:
        return str(os.environ.get("HF_HUB_OFFLINE", "")).strip() in {"1", "true", "yes"}
    _CONFIGURED = True

    try:
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        os.environ.setdefault("HF_HUB_DISABLE_EXPERIMENTAL_WARNING", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", _ETAG_TIMEOUT)
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", _DOWNLOAD_TIMEOUT)

        if str(os.environ.get("HF_HUB_OFFLINE", "")).strip() in {"1", "true", "yes"}:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            return True

        model = model_name or os.environ.get("SURVYAI_LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
        force_offline = False
        if _model_is_cached(model):
            force_offline = True
        elif not _has_hf_connectivity():
            force_offline = True

        if force_offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            return True
    except Exception:
        pass
    return False
