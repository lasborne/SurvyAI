from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


OLLAMA_DOWNLOAD_PAGE = "https://ollama.com/download"


def host_ram_mb() -> Tuple[int, int]:
    """Return ``(total_mb, available_mb)``. ``(0, 0)`` if unavailable."""
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)) == 0:
                return 0, 0
            return int(stat.ullTotalPhys // (1024 * 1024)), int(stat.ullAvailPhys // (1024 * 1024))
        except Exception:
            return 0, 0
    try:
        total = avail = 0
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) // 1024
        return total, avail
    except Exception:
        return 0, 0


def estimate_ollama_working_mb(model_name: str = "") -> int:
    """
    Rough resident working-set for a local Ollama model (weights + runtime overhead).
    Intentionally conservative — underestimating is what freezes PCs.
    """
    name = (model_name or "").strip().lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", name) or re.search(r":(\d+(?:\.\d+)?)b", name)
    if not m:
        return 2800
    billions = float(m.group(1))
    if billions <= 1.5:
        return 1800
    if billions <= 3.5:
        return 3200
    if billions <= 8.0:
        return 6500
    if billions <= 14.0:
        return 11000
    return 20000


def ollama_ram_policy(model_name: str = "") -> Tuple[bool, str, int]:
    """
    Host-aware Ollama safety policy.

    Returns ``(ok_to_run, error_message, num_ctx)``.
    Refuses when free RAM cannot cover OS reserve + estimated model working set.
    Caps context window by installed RAM / remaining headroom.
    """
    total_mb, avail_mb = host_ram_mb()
    if total_mb <= 0:
        return True, "", 1024

    # Keep a hard OS reserve free: max(1.5 GiB, 22% of total RAM).
    reserve_mb = max(1536, int(total_mb * 0.22))
    working_mb = estimate_ollama_working_mb(model_name)
    # Free RAM must cover reserve *and* loading/running the model.
    need_mb = reserve_mb + working_mb
    if avail_mb < need_mb:
        return (
            False,
            (
                f"Not enough free memory for a local Ollama run on this PC "
                f"({avail_mb} MB free; need ~{need_mb} MB: {reserve_mb} MB system reserve "
                f"+ ~{working_mb} MB for '{(model_name or 'model').strip() or 'model'}' "
                f"on {total_mb} MB total). Close other heavy apps, then try again. "
                f"This hard-cap stops local models from overloading RAM and locking the system."
            ),
            1024,
        )

    if total_mb < 8192:
        num_ctx = 1024
    elif total_mb < 16384:
        num_ctx = 1024  # 8–16 GiB hosts: keep KV-cache small
    elif total_mb < 32768:
        num_ctx = 2048
    else:
        num_ctx = 3072

    headroom = avail_mb - need_mb
    if headroom < 1024:
        num_ctx = min(num_ctx, 1024)
    elif headroom < 2048:
        num_ctx = min(num_ctx, 1024 if total_mb < 16384 else 2048)
    return True, "", int(num_ctx)


def release_ollama_model(
    base_url: str = "http://localhost:11434",
    model: str = "",
    timeout_seconds: float = 2.0,
) -> None:
    """Best-effort: unload a model from Ollama RAM/VRAM (stops leftover thrash)."""
    model = (model or "").strip()
    if not model:
        return
    url = (base_url or "http://localhost:11434").rstrip("/") + "/api/generate"
    payload = json.dumps({"model": model, "keep_alive": 0, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds):
            pass
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        pass
    except Exception:
        pass


@dataclass(frozen=True)
class OllamaInstallStatus:
    installed: bool
    exe_path: str = ""
    reason: str = ""


def _candidate_exe_paths() -> list[Path]:
    candidates: list[Path] = []
    # Most common Windows default path
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(Path(local) / "Programs" / "Ollama" / "ollama.exe")
    # Fallback locations some installers use
    progfiles = os.environ.get("ProgramFiles")
    if progfiles:
        candidates.append(Path(progfiles) / "Ollama" / "ollama.exe")
    progfiles_x86 = os.environ.get("ProgramFiles(x86)")
    if progfiles_x86:
        candidates.append(Path(progfiles_x86) / "Ollama" / "ollama.exe")
    return candidates


def detect_ollama_executable() -> Optional[str]:
    found = shutil.which("ollama")
    if found:
        return found
    for p in _candidate_exe_paths():
        try:
            if p.is_file():
                return str(p)
        except Exception:
            continue
    return None


def is_ollama_installed() -> OllamaInstallStatus:
    exe = detect_ollama_executable()
    if not exe:
        return OllamaInstallStatus(installed=False, reason="ollama.exe not found in PATH or common install paths")
    return OllamaInstallStatus(installed=True, exe_path=exe)


def try_connect_ollama(base_url: str = "http://localhost:11434", timeout_seconds: float = 1.2) -> bool:
    # Lightweight health probe: /api/tags is present on modern Ollama.
    # We intentionally avoid pulling models here.
    url = (base_url.rstrip("/") + "/api/tags").strip()
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as resp:
            return 200 <= int(getattr(resp, "status", 200)) < 300
    except Exception:
        return False


def install_ollama_with_winget() -> subprocess.Popen | None:
    """
    Best-effort installer path for Windows.
    Returns the spawned process if winget exists, otherwise None.
    """
    winget = shutil.which("winget")
    if not winget:
        return None
    # `--accept-package-agreements` avoids extra prompts where supported.
    # `--accept-source-agreements` is needed on fresh winget installs.
    cmd = [
        winget,
        "install",
        "-e",
        "--id",
        "Ollama.Ollama",
        "--accept-package-agreements",
        "--accept-source-agreements",
    ]
    try:
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return None


def list_local_models() -> list[str]:
    """
    Return locally available Ollama models (best-effort).
    Uses `ollama list` (works even if the HTTP server isn't reachable).
    """
    exe = detect_ollama_executable()
    if not exe:
        return []
    try:
        out = subprocess.check_output([exe, "list"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return []

    lines = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
    if not lines:
        return []

    # Typical output:
    # NAME            ID              SIZE    MODIFIED
    # llama3.2:3b     ...             ...     ...
    # We take the first whitespace-separated column, skipping header if present.
    models: list[str] = []
    for i, ln in enumerate(lines):
        if i == 0 and ln.lower().startswith("name"):
            continue
        name = ln.split()[0].strip()
        if name and name.lower() != "name":
            models.append(name)
    return sorted(set(models))


def start_pull_model(model: str) -> subprocess.Popen | None:
    """
    Start `ollama pull <model>` and stream stdout for progress in the GUI.
    Returns the process handle or None if Ollama isn't installed.
    """
    exe = detect_ollama_executable()
    if not exe:
        return None
    model = (model or "").strip()
    if not model:
        return None
    try:
        return subprocess.Popen(
            [exe, "pull", model],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception:
        return None

