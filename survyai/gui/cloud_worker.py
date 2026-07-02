"""Background threads for cloud API work (keeps the Qt event loop responsive)."""

from __future__ import annotations

import traceback
from typing import Any, Optional

from PySide6.QtCore import QThread, Signal

from survyai.gui.cloud_sync import (
    CloudAccountSyncPayload,
    CloudAccountSyncResult,
    CloudCreditsSyncPayload,
    CloudCreditsSyncResult,
    sync_cloud_account,
    sync_cloud_credits,
)


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
