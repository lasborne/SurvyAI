"""
Polished in-app markdown help dialogs (Getting Started, README, etc.).

Keeps help inside SurvyAI so users are not bounced to external editors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


class MarkdownHelpDialog(QDialog):
    """Readable markdown viewer with consistent footer actions."""

    def __init__(
        self,
        parent: Optional[QWidget],
        *,
        title: str,
        subtitle: str,
        markdown_path: Path,
        primary_label: str = "Close",
        show_dont_show_again: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("helpDialog")
        self.setWindowTitle(title)
        self.resize(920, 720)
        self.setMinimumSize(640, 480)

        self._markdown_path = Path(markdown_path)
        self._dont_show: Optional[QCheckBox] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(12)

        header = QLabel(title)
        header.setObjectName("helpDialogTitle")
        header.setWordWrap(True)
        root.addWidget(header)

        if subtitle.strip():
            sub = QLabel(subtitle.strip())
            sub.setObjectName("helpDialogSubtitle")
            sub.setWordWrap(True)
            root.addWidget(sub)

        viewer = QTextBrowser()
        viewer.setObjectName("helpBrowser")
        viewer.setOpenExternalLinks(True)
        try:
            content = self._markdown_path.read_text(encoding="utf-8")
        except OSError as exc:
            content = f"Could not open documentation:\n\n{exc}"
        try:
            viewer.setMarkdown(content)
        except Exception:
            viewer.setPlainText(content)
        root.addWidget(viewer, 1)

        if show_dont_show_again:
            self._dont_show = QCheckBox("Don't show this guide automatically again")
            self._dont_show.setChecked(True)
            self._dont_show.setObjectName("helpDontShowAgain")
            root.addWidget(self._dont_show)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        open_folder = QPushButton("Open docs folder")
        open_folder.setObjectName("secondaryButton")
        open_folder.clicked.connect(self._open_docs_folder)
        actions.addWidget(open_folder)
        actions.addStretch()
        primary = QPushButton(primary_label)
        primary.setObjectName("primaryButton")
        primary.setDefault(True)
        primary.clicked.connect(self.accept)
        actions.addWidget(primary)
        root.addLayout(actions)

    def _open_docs_folder(self) -> None:
        folder = self._markdown_path.parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def mark_seen_preferred(self) -> bool:
        """True when first-run checkbox is checked (or when checkbox is absent)."""
        if self._dont_show is None:
            return True
        return bool(self._dont_show.isChecked())
