"""
Windows-oriented secure secret storage for desktop session material.

Uses DPAPI on Windows so tokens/bootstrap secrets are protected per-user.
Falls back to a local JSON file on non-Windows platforms for development only.
"""

from __future__ import annotations

import base64
import ctypes
import json
import os
from ctypes import wintypes
from pathlib import Path
from typing import Any, Dict

from utils.logger import get_logger

logger = get_logger(__name__)


class _DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_byte)),
    ]


def _blob_from_bytes(data: bytes) -> _DATA_BLOB:
    size = len(data)
    if size == 0:
        return _DATA_BLOB(0, ctypes.POINTER(ctypes.c_byte)())
    buf = ctypes.create_string_buffer(data, size)
    return _DATA_BLOB(size, ctypes.cast(buf, ctypes.POINTER(ctypes.c_byte)))


def _bytes_from_blob(blob: _DATA_BLOB) -> bytes:
    if not blob.cbData or not blob.pbData:
        return b""
    return ctypes.string_at(blob.pbData, blob.cbData)


class DesktopSecretStore:
    def __init__(self, app_dir: Path) -> None:
        self.app_dir = Path(app_dir)
        self.app_dir.mkdir(parents=True, exist_ok=True)
        self.secret_path = self.app_dir / "desktop_secrets.bin"

    def load(self) -> Dict[str, Any]:
        raw = self._read_bytes()
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning("Desktop secret store load failed: %s", exc)
            return {}

    def save(self, payload: Dict[str, Any]) -> None:
        clean = {k: v for k, v in (payload or {}).items() if v not in (None, "", {}, [])}
        if not clean:
            self.clear()
            return
        raw = json.dumps(clean, ensure_ascii=True).encode("utf-8")
        self._write_bytes(raw)

    def clear(self) -> None:
        try:
            self.secret_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Could not clear desktop secret store: %s", exc)

    def _read_bytes(self) -> bytes:
        if not self.secret_path.is_file():
            return b""
        try:
            raw = self.secret_path.read_bytes()
        except Exception as exc:
            logger.warning("Could not read desktop secret store: %s", exc)
            return b""
        if not raw:
            return b""
        if os.name == "nt":
            return _dpapi_unprotect(raw)
        try:
            return base64.b64decode(raw)
        except Exception as exc:
            logger.warning("Could not decode desktop secret store fallback: %s", exc)
            return b""

    def _write_bytes(self, raw: bytes) -> None:
        try:
            if os.name == "nt":
                enc = _dpapi_protect(raw)
                self.secret_path.write_bytes(enc)
            else:
                self.secret_path.write_bytes(base64.b64encode(raw))
                try:
                    os.chmod(self.secret_path, 0o600)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("Could not write desktop secret store: %s", exc)


def _dpapi_protect(data: bytes) -> bytes:
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    in_blob = _blob_from_bytes(data)
    out_blob = _DATA_BLOB()
    if not crypt32.CryptProtectData(
        ctypes.byref(in_blob),
        "SurvyAI Desktop Secrets",
        None,
        None,
        None,
        0,
        ctypes.byref(out_blob),
    ):
        raise ctypes.WinError()
    try:
        return _bytes_from_blob(out_blob)
    finally:
        if out_blob.pbData:
            kernel32.LocalFree(out_blob.pbData)


def _dpapi_unprotect(data: bytes) -> bytes:
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    in_blob = _blob_from_bytes(data)
    out_blob = _DATA_BLOB()
    desc = wintypes.LPWSTR()
    if not crypt32.CryptUnprotectData(
        ctypes.byref(in_blob),
        ctypes.byref(desc),
        None,
        None,
        None,
        0,
        ctypes.byref(out_blob),
    ):
        raise ctypes.WinError()
    try:
        return _bytes_from_blob(out_blob)
    finally:
        if out_blob.pbData:
            kernel32.LocalFree(out_blob.pbData)
        if desc:
            kernel32.LocalFree(desc)


__all__ = ["DesktopSecretStore"]
