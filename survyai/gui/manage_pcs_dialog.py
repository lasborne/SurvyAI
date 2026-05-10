"""
Manage registered PCs for SurvyAI Cloud (device slots for hosted Pro keys).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from survyai.cloud_api import CloudApiError, delete_cloud_device, list_devices
from survyai.cloud_user_message import user_facing_cloud_message


def _format_last_seen(raw: Any) -> str:
    if raw is None:
        return "—"
    s = str(raw).strip()
    if not s:
        return "—"
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return s[:19] if len(s) >= 19 else s


class ManagePcsDialog(QDialog):
    """
    List devices from GET /v1/devices and remove via DELETE /v1/devices/{id}.

    ``removed_current_pc`` is True when the row matching ``current_device_id`` was deleted.
    """

    def __init__(
        self,
        parent,
        *,
        base_url: str,
        access_token: str,
        current_device_id: str,
        max_devices: Optional[int] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manage PCs")
        self.setMinimumWidth(520)
        self._base = (base_url or "").strip().rstrip("/")
        self._token = (access_token or "").strip()
        cur = (current_device_id or "").strip().lower()
        self._current_id = cur
        self._max_devices = max_devices
        self.removed_any = False
        self.removed_current_pc = False

        layout = QVBoxLayout(self)
        hint = QLabel(
            "Registered PCs are used to deliver hosted Pro model keys only to machines you authorize. "
            "Remove computers you no longer use so you can sign in on a new PC."
        )
        hint.setWordWrap(True)
        hint.setObjectName("hintLabel")
        layout.addWidget(hint)
        if max_devices is not None and max_devices > 0:
            cap = QLabel(f"Maximum PCs for your current plan: {max_devices}")
            cap.setStyleSheet("font-weight: 600;")
            layout.addWidget(cap)

        self._list = QListWidget()
        self._list.setMinimumHeight(220)
        self._list.setAlternatingRowColors(True)
        layout.addWidget(self._list, 1)

        actions = QHBoxLayout()
        self._remove_btn = QPushButton("Remove selected PC…")
        self._remove_btn.setObjectName("secondaryButton")
        self._remove_btn.setToolTip("Permanently remove this registration from your account.")
        self._remove_btn.clicked.connect(self._on_remove_clicked)
        actions.addWidget(self._remove_btn)
        self._reload_btn = QPushButton("Reload list")
        self._reload_btn.setObjectName("secondaryButton")
        self._reload_btn.clicked.connect(self._load_devices)
        actions.addWidget(self._reload_btn)
        actions.addStretch()
        layout.addLayout(actions)

        bbox = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bbox.rejected.connect(self.reject)
        layout.addWidget(bbox)

        self._load_devices()

    def _load_devices(self) -> None:
        self._list.clear()
        self._remove_btn.setEnabled(False)
        try:
            rows = list_devices(base_url=self._base, access_token=self._token)
        except CloudApiError as exc:
            QMessageBox.warning(self, "Could not load PCs", user_facing_cloud_message(exc))
            return

        if not rows:
            item = QListWidgetItem("No PCs registered yet. Use Refresh cloud account after sign-in.")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self._list.addItem(item)
            self._remove_btn.setEnabled(False)
            return

        self._remove_btn.setEnabled(True)
        for dev in rows:
            did = str(dev.get("id") or "").strip()
            if not did:
                continue
            label = (dev.get("label") or "").strip() or "Unnamed PC"
            fp = str(dev.get("fingerprint") or "")
            fp_tail = fp[-10:] if len(fp) >= 10 else fp or "—"
            last = _format_last_seen(dev.get("last_seen_at"))
            is_here = did.lower() == self._current_id
            title = f"{label}"
            if is_here:
                title += "  (this PC)"
            lines = f"{title}\nLast seen: {last}  ·  Fingerprint …{fp_tail}"
            item = QListWidgetItem(lines)
            item.setData(Qt.ItemDataRole.UserRole, did)
            item.setToolTip(f"Device id: {did}")
            self._list.addItem(item)

    def _selected_device_id(self) -> Optional[str]:
        item = self._list.currentItem()
        if item is None:
            return None
        did = item.data(Qt.ItemDataRole.UserRole)
        if not did:
            return None
        s = str(did).strip()
        return s or None

    def _on_remove_clicked(self) -> None:
        did = self._selected_device_id()
        if not did:
            QMessageBox.information(self, "Manage PCs", "Select a PC in the list first.")
            return
        is_current = did.lower() == self._current_id
        if is_current:
            msg = (
                "Remove this PC from your account?\n\n"
                "Hosted Pro keys will stop working here until you use "
                "“Refresh cloud account” to register this machine again."
            )
        else:
            msg = (
                "Remove the selected PC from your account?\n\n"
                "That computer will no longer receive hosted keys until it is registered again."
            )
        confirm = QMessageBox.question(
            self,
            "Remove PC",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            delete_cloud_device(base_url=self._base, access_token=self._token, device_id=did)
        except CloudApiError as exc:
            QMessageBox.warning(self, "Could not remove PC", user_facing_cloud_message(exc))
            return

        self.removed_any = True
        if is_current:
            self.removed_current_pc = True
        self._load_devices()
