from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


OLLAMA_DOWNLOAD_PAGE = "https://ollama.com/download"


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

