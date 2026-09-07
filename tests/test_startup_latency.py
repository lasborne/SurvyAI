"""Startup stall regressions: HF fail-fast and Ollama reject cache."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_ollama_reject_cached_for_process_lifetime() -> None:
    from survyai.ollama_support import (
        cached_ollama_reject_reason,
        clear_ollama_reject_cache,
        remember_ollama_reject,
        ollama_ram_policy,
    )

    clear_ollama_reject_cache()
    remember_ollama_reject("llama3", "Not enough free memory")
    assert "memory" in (cached_ollama_reject_reason("llama3") or "").lower()

    allowed, reason, _ctx = ollama_ram_policy(model_name="llama3")
    assert allowed is False
    assert "memory" in (reason or "").lower()
    clear_ollama_reject_cache()


def test_local_embedding_fails_fast_offline_without_retry_storm(tmp_path, monkeypatch) -> None:
    from tools.vector_store import LocalEmbeddingProvider

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    # Point cache at empty dir so model is missing locally.
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path / "hf"))

    provider = LocalEmbeddingProvider("sentence-transformers/all-MiniLM-L6-v2", lazy=True)

    call_count = {"n": 0}

    def boom(*_a, **_k):
        call_count["n"] += 1
        raise OSError("Network unreachable / Repository not found")

    with patch("tools.vector_store.SentenceTransformer", side_effect=boom):
        # Some implementations import inside _ensure_loaded — also patch that path.
        with patch.dict("sys.modules", {}):
            with pytest.raises(Exception):
                provider.embed(["hello"])

    # Must not perform the historical 1/2/4/8/8 retry storm (5+ loads).
    assert call_count["n"] <= 2


def test_hf_env_forces_offline_when_model_cached(monkeypatch) -> None:
    import importlib
    import os

    import survyai.hf_env as hf_env

    hf_env = importlib.reload(hf_env)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr(hf_env, "_model_is_cached", lambda *_a, **_k: True)
    probe = {"called": False}

    def _no_probe(*_a, **_k):
        probe["called"] = True
        return True

    monkeypatch.setattr(hf_env, "_has_hf_connectivity", _no_probe)

    forced = hf_env.configure_hf_offline_if_appropriate("all-MiniLM-L6-v2")
    assert forced is True
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert probe["called"] is False


def test_hf_env_offline_when_uncached_and_unreachable(monkeypatch) -> None:
    import importlib
    import os

    import survyai.hf_env as hf_env

    hf_env = importlib.reload(hf_env)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(hf_env, "_model_is_cached", lambda *_a, **_k: False)
    monkeypatch.setattr(hf_env, "_has_hf_connectivity", lambda *_a, **_k: False)

    forced = hf_env.configure_hf_offline_if_appropriate("some/model")
    assert forced is True
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def test_importing_vector_store_does_not_import_torch() -> None:
    import subprocess
    import sys

    code = (
        "import sys;"
        "import tools.vector_store as v;"
        "print('torch' in sys.modules, 'sentence_transformers' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("False False"), out.stdout


def test_store_conversation_runs_off_response_path() -> None:
    import threading
    import time

    from agent.agent import SurvyAIAgent

    agent = SurvyAIAgent.__new__(SurvyAIAgent)
    agent.settings = MagicMock()
    agent.settings.auto_store_conversations = True

    started = threading.Event()
    release = threading.Event()

    vs = MagicMock()

    def _slow_add(*_a, **_k):
        started.set()
        release.wait(timeout=5)

    vs.add_conversation.side_effect = _slow_add
    agent.vector_store = vs

    t0 = time.perf_counter()
    agent._store_conversation("q", "r", "session-1234567890", llm_used="primary")
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0
    assert started.wait(timeout=5)
    release.set()
