"""One-shot build script for the SurvyAI Windows desktop installer.

Steps:
  1. Run PyInstaller against survyai.spec  -> dist/SurvyAI/SurvyAI.exe
  2. (Optional) Run Inno Setup (ISCC.exe)  -> installer/Output/SurvyAI-Setup-<ver>.exe

Usage:
  python build_desktop.py                 # build exe + installer (if ISCC found)
  python build_desktop.py --no-installer  # build exe only
  python build_desktop.py --clean         # clean build/ and dist/ first

Prereqs:
  pip install -r requirements.txt -r requirements-build.txt
  Inno Setup 6 installed (for the installer step): https://jrsoftware.org/isdl.php
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "survyai.spec"
ISS = ROOT / "installer" / "survyai.iss"
DIST_APP = ROOT / "dist" / "SurvyAI"


def _app_version() -> str:
    # Read the single source of truth without importing the whole package.
    version_file = ROOT / "survyai" / "version.py"
    ns: dict = {}
    exec(version_file.read_text(encoding="utf-8"), ns)  # noqa: S102 - trusted local file
    return str(ns.get("__version__", "0.0.0"))


def _run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}\n", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        sys.exit(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _find_iscc() -> str | None:
    found = shutil.which("ISCC") or shutil.which("iscc")
    if found:
        return found
    candidates = [
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Inno Setup 6" / "ISCC.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Inno Setup 6" / "ISCC.exe",
        # Per-user install (common when winget installs without elevation).
        Path(os.environ.get("LocalAppData", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the SurvyAI desktop installer.")
    parser.add_argument("--no-installer", action="store_true", help="Build the exe only; skip Inno Setup.")
    parser.add_argument("--clean", action="store_true", help="Remove build/ and dist/ before building.")
    args = parser.parse_args()

    version = _app_version()
    print(f"SurvyAI build — version {version}")

    if args.clean:
        for d in (ROOT / "build", ROOT / "dist"):
            if d.exists():
                print(f"Removing {d}")
                shutil.rmtree(d, ignore_errors=True)

    # 1) PyInstaller
    _run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", str(SPEC)])

    exe = DIST_APP / "SurvyAI.exe"
    if not exe.is_file():
        sys.exit(f"PyInstaller did not produce {exe}")
    print(f"\nOK: {exe}")

    # 2) Inno Setup installer
    if args.no_installer:
        print("Skipping installer step (--no-installer).")
        return 0

    iscc = _find_iscc()
    if not iscc:
        print(
            "\nInno Setup (ISCC.exe) not found — skipping installer step.\n"
            "Install from https://jrsoftware.org/isdl.php, then run:\n"
            f'   ISCC.exe "{ISS}" /DAppVersion={version}'
        )
        return 0

    _run([iscc, str(ISS), f"/DAppVersion={version}"])
    out = ROOT / "installer" / "Output" / f"SurvyAI-Setup-{version}.exe"
    print(f"\nDONE. Installer: {out}" if out.is_file() else "\nInno Setup finished (check installer/Output).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
