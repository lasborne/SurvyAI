from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "SurvyAI"


def is_frozen_app() -> bool:
    return bool(getattr(sys, "frozen", False))


def project_root() -> Path:
    return Path(__file__).resolve().parent


def bundled_root() -> Path:
    if is_frozen_app():
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            return Path(meipass).resolve()
        return Path(sys.executable).resolve().parent
    return project_root()


def user_data_root() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return base / APP_NAME


def resource_path(*parts: str) -> Path:
    return bundled_root().joinpath(*parts)


def user_data_path(*parts: str) -> Path:
    return user_data_root().joinpath(*parts)


def prefer_user_data_path(*parts: str) -> Path:
    candidate = user_data_path(*parts)
    if candidate.exists():
        return candidate
    return resource_path(*parts)


def default_documents_folder() -> Path:
    """Return the user's primary Documents folder (Windows shell folder when available)."""
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders",
            ) as key:
                personal, _ = winreg.QueryValueEx(key, "Personal")
            docs = Path(os.path.expandvars(str(personal))).resolve()
            if docs.is_dir():
                return docs
        except Exception:
            pass
    docs = (Path.home() / "Documents").resolve()
    docs.mkdir(parents=True, exist_ok=True)
    return docs


__all__ = [
    "APP_NAME",
    "bundled_root",
    "default_documents_folder",
    "is_frozen_app",
    "prefer_user_data_path",
    "project_root",
    "resource_path",
    "user_data_path",
    "user_data_root",
]
