"""Background threads for cloud API work (keeps the Qt event loop responsive)."""

from __future__ import annotations

import traceback
from typing import Any, Optional

from PySide6.QtCore import QThread, Signal

from survyai.cloud_api import CloudApiError, get_update_manifest
from survyai.gui.cloud_sync import (
    CloudAccountSyncPayload,
    CloudCreditsSyncPayload,
    sync_cloud_account,
    sync_cloud_credits,
)
from survyai.updater import UpdateManifest
from survyai.version import __version__


class CloudAccountSyncThread(QThread):
    """Runs `sync_cloud_account` off the GUI thread."""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        payload: CloudAccountSyncPayload,
        *,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self._payload = payload

    def run(self) -> None:
        try:
            result = sync_cloud_account(self._payload)
            if result.ok:
                self.succeeded.emit(result)
            elif result.error_message:
                self.failed.emit(result.error_message)
            else:
                self.failed.emit("Cloud account sync failed.")
        except Exception:
            self.failed.emit(traceback.format_exc())


class CloudCreditsSyncThread(QThread):
    """Runs `sync_cloud_credits` off the GUI thread."""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        payload: CloudCreditsSyncPayload,
        *,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self._payload = payload

    def run(self) -> None:
        try:
            result = sync_cloud_credits(self._payload)
            if result.ok:
                self.succeeded.emit(result)
            elif result.error_message:
                self.failed.emit(result.error_message)
            else:
                self.failed.emit("Credits sync failed.")
        except Exception:
            self.failed.emit(traceback.format_exc())


class UpdateCheckThread(QThread):
    """Fetches the update manifest off the GUI thread."""

    update_available = Signal(object)  # UpdateManifest
    up_to_date = Signal(object)  # UpdateManifest
    failed = Signal(str)

    def __init__(
        self,
        *,
        base_url: str,
        channel: str = "stable",
        platform: str = "windows-x64",
        current_version: str = "",
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self._base_url = base_url
        self._channel = channel
        self._platform = platform
        self._current_version = (current_version or __version__).strip() or __version__

    def run(self) -> None:
        try:
            manifest_dict = get_update_manifest(
                base_url=self._base_url,
                current_version=self._current_version,
                channel=self._channel,
                platform=self._platform,
                timeout_s=10,
            )
            manifest = UpdateManifest.from_dict(manifest_dict)
            if manifest.is_newer_than(self._current_version):
                self.update_available.emit(manifest)
            else:
                self.up_to_date.emit(manifest)
        except CloudApiError as exc:
            self.failed.emit(str(exc))
        except Exception:
            self.failed.emit(traceback.format_exc())
