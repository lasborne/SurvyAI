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


__all__ = [
    "APP_NAME",
    "bundled_root",
    "is_frozen_app",
    "prefer_user_data_path",
    "project_root",
    "resource_path",
    "user_data_path",
    "user_data_root",
]
