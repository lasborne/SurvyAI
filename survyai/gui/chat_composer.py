"""ChatGPT-style composer: + attach button, chips, drag-drop, and image paste."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage, QKeyEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from survyai.attachments import (
    DEFAULT_ATTACHMENTS_ONLY_PROMPT,
    format_attachments_block,
    format_user_transcript,
    is_attachable_path,
)


class _AttachmentChip(QWidget):
    """Compact removable filename chip."""

    removed = Signal(str)

    def __init__(self, path: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.path = path
        self.setObjectName("attachmentChip")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 4, 2)
        layout.setSpacing(4)
        name = Path(path).name or path
        label = QLabel(name)
        label.setObjectName("attachmentChipLabel")
        label.setToolTip(path)
        label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        layout.addWidget(label)
        btn = QToolButton()
        btn.setObjectName("attachmentChipRemove")
        btn.setText("\u00d7")
        btn.setToolTip("Remove attachment")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setAutoRaise(True)
        btn.clicked.connect(lambda: self.removed.emit(self.path))
        layout.addWidget(btn)


class ChatComposer(QWidget):
    """
    Composer strip: optional chips + (+ button | chat input).

    The embedded ``input_widget`` keeps objectName ``chatInput`` for existing QSS.
    """

    sendRequested = Signal()
    attachmentsChanged = Signal()
    layoutChanged = Signal()

    DEFAULT_MAX_FILES = 4
    DEFAULT_MAX_FILE_MB = 10

    def __init__(
        self,
        input_widget: QWidget,
        *,
        parent: Optional[QWidget] = None,
        max_files: int = DEFAULT_MAX_FILES,
        max_file_mb: int = DEFAULT_MAX_FILE_MB,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("chatComposer")
        self._input = input_widget
        self._max_files = max(1, int(max_files or self.DEFAULT_MAX_FILES))
        self._max_file_mb = max(1, int(max_file_mb or self.DEFAULT_MAX_FILE_MB))
        self._paths: List[str] = []
        self._workspace_path: str = ""

        self.setAcceptDrops(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        self._chips_wrap = QWidget()
        self._chips_wrap.setObjectName("attachmentChipsRow")
        self._chips_layout = QHBoxLayout(self._chips_wrap)
        self._chips_layout.setContentsMargins(0, 0, 0, 0)
        self._chips_layout.setSpacing(6)
        self._chips_layout.addStretch(1)
        self._chips_wrap.setVisible(False)
        root.addWidget(self._chips_wrap, 0)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self._attach_btn = QToolButton()
        self._attach_btn.setObjectName("attachButton")
        self._attach_btn.setText("+")
        self._attach_btn.setToolTip(
            "Attach images or documents (.png, .jpg, .pdf, .docx, …).\n"
            "You can also type a file path in the prompt."
        )
        self._attach_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._attach_btn.setAutoRaise(True)
        self._attach_btn.clicked.connect(self._on_attach_clicked)
        row.addWidget(self._attach_btn, 0, Qt.AlignmentFlag.AlignBottom)

        self._input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row.addWidget(self._input, 1)
        root.addLayout(row)

        # Paste images into the chat input (Ctrl+V).
        if hasattr(self._input, "keyPressEvent"):
            self._orig_input_key_press = self._input.keyPressEvent
            self._input.keyPressEvent = self._input_key_press  # type: ignore[method-assign]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def input_widget(self) -> QWidget:
        return self._input

    def set_workspace_path(self, path: str) -> None:
        self._workspace_path = (path or "").strip()

    def set_limits(self, *, max_files: Optional[int] = None, max_file_mb: Optional[int] = None) -> None:
        if max_files is not None:
            self._max_files = max(1, int(max_files))
        if max_file_mb is not None:
            self._max_file_mb = max(1, int(max_file_mb))

    def attachment_paths(self) -> List[str]:
        return list(self._paths)

    def clear_attachments(self) -> None:
        if not self._paths:
            return
        self._paths.clear()
        self._rebuild_chips()
        self.attachmentsChanged.emit()
        self.layoutChanged.emit()

    def set_attachments(self, paths: Sequence[str]) -> None:
        """Replace chips (used by Retry). Invalid / missing files are skipped."""
        self._paths.clear()
        for raw in paths:
            self._try_add_path(str(raw), warn=False)
        self._rebuild_chips()
        self.attachmentsChanged.emit()
        self.layoutChanged.emit()

    def plain_text(self) -> str:
        if hasattr(self._input, "toPlainText"):
            return str(self._input.toPlainText() or "")
        return ""

    def set_plain_text(self, text: str) -> None:
        if hasattr(self._input, "setPlainText"):
            self._input.setPlainText(text)

    def clear_input(self) -> None:
        if hasattr(self._input, "clear"):
            self._input.clear()

    def build_agent_query(self) -> str:
        """Snapshot attachments + text into the agent-facing query string."""
        return format_attachments_block(self._paths, self.plain_text())

    def build_transcript_text(self) -> str:
        return format_user_transcript(self.plain_text(), self._paths)

    def can_send(self) -> bool:
        return bool(self.plain_text().strip() or self._paths)

    def chips_height(self) -> int:
        if not self._chips_wrap.isVisible():
            return 0
        return max(0, int(self._chips_wrap.sizeHint().height()))

    # ------------------------------------------------------------------
    # Attach / validate
    # ------------------------------------------------------------------

    def _on_attach_clicked(self) -> None:
        filters = (
            "Images and documents ("
            "*.png *.jpg *.jpeg *.webp *.tif *.tiff *.bmp *.gif *.pdf *.docx *.doc);;"
            "Images (*.png *.jpg *.jpeg *.webp *.tif *.tiff *.bmp *.gif);;"
            "Documents (*.pdf *.docx *.doc);;"
            "All files (*.*)"
        )
        start = self._workspace_path or str(Path.home())
        paths, _ = QFileDialog.getOpenFileNames(self, "Attach files", start, filters)
        for p in paths:
            self._try_add_path(p, warn=True)
        if paths:
            self._rebuild_chips()
            self.attachmentsChanged.emit()
            self.layoutChanged.emit()

    def _try_add_path(self, path: str, *, warn: bool) -> bool:
        p = Path(str(path or "").strip().strip('"').strip("'"))
        if not str(p):
            return False
        if not p.is_file():
            if warn:
                QMessageBox.warning(self, "Attachment", f"File not found:\n{p}")
            return False
        if not is_attachable_path(p):
            if warn:
                QMessageBox.warning(
                    self,
                    "Attachment",
                    f"Unsupported type: {p.suffix or '(no extension)'}\n"
                    "Attach images (.png, .jpg, …) or documents (.pdf, .docx).",
                )
            return False
        try:
            size_mb = p.stat().st_size / (1024 * 1024)
        except OSError as exc:
            if warn:
                QMessageBox.warning(self, "Attachment", f"Cannot read file:\n{exc}")
            return False
        if size_mb > self._max_file_mb:
            if warn:
                QMessageBox.warning(
                    self,
                    "Attachment too large",
                    f"{p.name} is {size_mb:.1f} MB.\n"
                    f"Maximum allowed size is {self._max_file_mb} MB.",
                )
            return False
        resolved = str(p.resolve())
        if any(existing.lower() == resolved.lower() for existing in self._paths):
            return False
        if len(self._paths) >= self._max_files:
            if warn:
                QMessageBox.warning(
                    self,
                    "Too many attachments",
                    f"You can attach at most {self._max_files} files per message.",
                )
            return False
        self._paths.append(resolved)
        return True

    def _remove_path(self, path: str) -> None:
        before = list(self._paths)
        self._paths = [p for p in self._paths if p != path]
        if self._paths == before:
            return
        self._rebuild_chips()
        self.attachmentsChanged.emit()
        self.layoutChanged.emit()

    def _rebuild_chips(self) -> None:
        while self._chips_layout.count():
            item = self._chips_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for path in self._paths:
            chip = _AttachmentChip(path, self._chips_wrap)
            chip.removed.connect(self._remove_path)
            self._chips_layout.addWidget(chip, 0)
        self._chips_layout.addStretch(1)
        self._chips_wrap.setVisible(bool(self._paths))

    # ------------------------------------------------------------------
    # Drag / drop / paste
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData() and event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            super().dropEvent(event)
            return
        added = False
        for url in mime.urls():
            local = url.toLocalFile()
            if local and self._try_add_path(local, warn=True):
                added = True
        if added:
            self._rebuild_chips()
            self.attachmentsChanged.emit()
            self.layoutChanged.emit()
        event.acceptProposedAction()

    def _input_key_press(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.sendRequested.emit()
            return
        if (
            event.key() == Qt.Key.Key_V
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            if self._try_paste_clipboard_image():
                return
        self._orig_input_key_press(event)

    def _try_paste_clipboard_image(self) -> bool:
        from PySide6.QtWidgets import QApplication

        clipboard = QApplication.clipboard()
        if clipboard is None:
            return False
        mime = clipboard.mimeData()
        image: Optional[QImage] = None
        if mime is not None and mime.hasImage():
            raw = mime.imageData()
            if isinstance(raw, QImage) and not raw.isNull():
                image = raw
        if image is None:
            img = clipboard.image()
            if isinstance(img, QImage) and not img.isNull():
                image = img
        if image is None:
            return False

        dest_dir = self._paste_dir()
        dest_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dest = dest_dir / f"pasted_{stamp}_{uuid.uuid4().hex[:8]}.png"
        if not image.save(str(dest), "PNG"):
            QMessageBox.warning(self, "Paste image", "Could not save the clipboard image.")
            return True
        if self._try_add_path(str(dest), warn=True):
            self._rebuild_chips()
            self.attachmentsChanged.emit()
            self.layoutChanged.emit()
        return True

    def _paste_dir(self) -> Path:
        base = Path(self._workspace_path) if self._workspace_path else Path.home() / "SurvyAI"
        return base / ".survyai" / "attachments"


__all__ = ["ChatComposer", "DEFAULT_ATTACHMENTS_ONLY_PROMPT"]
