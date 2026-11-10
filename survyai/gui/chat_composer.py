"""ChatGPT-style composer: + attach button, chips, drag-drop, and image paste."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QImage,
    QKeyEvent,
    QPainter,
    QPen,
    QPixmap,
)
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
    attach_unsupported_message,
    file_dialog_filters,
    format_attachments_block,
    format_user_transcript,
    is_attachable_path,
    max_file_mb_for_path,
)


def _stroke_icon(kind: str, color: str, *, logical: int = 16) -> QIcon:
    """One optical size for the prompt marks, so +, retry, x, and C sit evenly."""
    pm = QPixmap(logical * 2, logical * 2)
    pm.setDevicePixelRatio(2.0)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(QColor(color))
    pen.setWidthF(1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    s = float(logical)
    inset = 3.2
    if kind == "plus":
        mid = s / 2
        painter.drawLine(QPointF(inset, mid), QPointF(s - inset, mid))
        painter.drawLine(QPointF(mid, inset), QPointF(mid, s - inset))
    elif kind == "retry":
        import math

        radius = (s - 5.4) / 2.0
        origin = (s - 2.0 * radius) / 2.0
        painter.drawArc(
            QRectF(origin, origin, radius * 2.0, radius * 2.0),
            125 * 16,
            290 * 16,
        )
        end = math.radians(125 + 290)
        cx = origin + radius
        cy = origin + radius
        ex = cx + radius * math.cos(end)
        ey = cy - radius * math.sin(end)
        tx, ty = -math.sin(end), -math.cos(end)
        px, py = -ty, tx
        tip = QPointF(ex + tx * 3.4, ey + ty * 3.4)
        painter.drawLine(QPointF(ex + px * 2.15, ey + py * 2.15), tip)
        painter.drawLine(QPointF(ex - px * 2.15, ey - py * 2.15), tip)
    elif kind == "clear":
        painter.drawArc(QRectF(4.2, 3.2, s - 7.2, s - 6.4), 50 * 16, 260 * 16)
    else:
        painter.drawLine(QPointF(inset + 0.4, inset + 0.4), QPointF(s - inset, s - inset))
        painter.drawLine(QPointF(s - inset, inset + 0.4), QPointF(inset + 0.4, s - inset))
    painter.end()
    icon = QIcon()
    icon.addPixmap(pm)
    return icon


def _send_arrow_icon() -> QIcon:
    """A slightly larger right arrow, white, for the blue send control."""
    logical = 18
    pm = QPixmap(logical * 2, logical * 2)
    pm.setDevicePixelRatio(2.0)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(QColor("#ffffff"))
    pen.setWidthF(2.0)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.drawLine(QPointF(4.2, 9.0), QPointF(13.2, 9.0))
    painter.drawLine(QPointF(9.2, 4.6), QPointF(13.8, 9.0))
    painter.drawLine(QPointF(9.2, 13.4), QPointF(13.8, 9.0))
    painter.end()
    icon = QIcon()
    icon.addPixmap(pm)
    return icon


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

    DEFAULT_MAX_FILES = 8
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

        shell = QWidget()
        shell.setObjectName("consolePromptShell")
        shell.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        shell_row = QHBoxLayout(shell)
        shell_row.setContentsMargins(10, 8, 10, 8)
        shell_row.setSpacing(8)

        self._retry_sign = self._console_sign(
            _stroke_icon("retry", "#475569"),
            "Re-run the last prompt in this conversation.",
        )
        self._attach_btn = self._console_sign(
            _stroke_icon("plus", "#475569"),
            "Attach files for this prompt: images, Excel/CSV, CAD (.dwg/.dxf/.dwf),\n"
            "GIS (.shp/.gpkg/.kml/.las), PDF/Word, PowerPoint, text, survey data, …\n"
            "You can also drag-and-drop or type a file path in the prompt.",
        )
        self._attach_btn.clicked.connect(self._on_attach_clicked)
        self._retry_sign.setEnabled(False)
        left_col = QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(4)
        left_col.addStretch(1)
        left_col.addWidget(self._retry_sign, 0, Qt.AlignmentFlag.AlignLeft)
        left_col.addWidget(self._attach_btn, 0, Qt.AlignmentFlag.AlignLeft)

        self._cancel_sign = self._console_sign(
            _stroke_icon("x", "#475569"),
            "Stop the current run. Partial files already written stay on disk.",
        )
        self._cancel_sign.setEnabled(False)
        self._clear_sign = self._console_sign(
            _stroke_icon("clear", "#475569"),
            "Clear the Console prompt and any attached files.",
        )
        self._send_sign = self._console_sign(
            _send_arrow_icon(),
            "Send the Console prompt. Enter sends; Shift+Enter starts a new line.",
            object_name="consoleSendSign",
            side=34,
            icon_side=18,
        )
        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(4)
        right_col.addStretch(1)
        right_col.addWidget(self._cancel_sign, 0, Qt.AlignmentFlag.AlignRight)
        right_col.addWidget(self._clear_sign, 0, Qt.AlignmentFlag.AlignRight)
        right_col.addWidget(self._send_sign, 0, Qt.AlignmentFlag.AlignRight)

        self._input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        shell_row.addLayout(left_col, 0)
        shell_row.addWidget(self._input, 1)
        shell_row.addLayout(right_col, 0)
        root.addWidget(shell)
        self.set_prompt_icon_tone(False)

        # Paste images into the chat input (Ctrl+V).
        if hasattr(self._input, "keyPressEvent"):
            self._orig_input_key_press = self._input.keyPressEvent
            self._input.keyPressEvent = self._input_key_press  # type: ignore[method-assign]

    @staticmethod
    def _console_sign(
        icon: QIcon,
        tip: str,
        object_name: str = "consoleSign",
        *,
        side: int = 28,
        icon_side: int = 16,
    ) -> QToolButton:
        btn = QToolButton()
        btn.setObjectName(object_name)
        btn.setIcon(icon)
        btn.setIconSize(QSize(icon_side, icon_side))
        btn.setToolTip(tip)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setAutoRaise(True)
        btn.setFixedSize(side, side)
        return btn

    def set_prompt_icon_tone(self, dark: bool) -> None:
        """Match the prompt marks to light or dark without changing their size."""
        color = "#e4e4e7" if dark else "#334155"
        for button, kind in (
            (self._retry_sign, "retry"),
            (self._attach_btn, "plus"),
            (self._cancel_sign, "x"),
            (self._clear_sign, "clear"),
        ):
            button.setIcon(_stroke_icon(kind, color))
        self._send_sign.setIcon(_send_arrow_icon())

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
        start = self._workspace_path or str(Path.home())
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Attach files", start, file_dialog_filters()
        )
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
                    attach_unsupported_message(p.suffix),
                )
            return False
        try:
            size_mb = p.stat().st_size / (1024 * 1024)
        except OSError as exc:
            if warn:
                QMessageBox.warning(self, "Attachment", f"Cannot read file:\n{exc}")
            return False
        size_cap = max_file_mb_for_path(p, image_max_mb=self._max_file_mb)
        if size_mb > size_cap:
            if warn:
                QMessageBox.warning(
                    self,
                    "Attachment too large",
                    f"{p.name} is {size_mb:.1f} MB.\n"
                    f"Maximum allowed size is {size_cap} MB.",
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
