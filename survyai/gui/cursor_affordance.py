from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QApplication, QAbstractButton, QWidget


class ClickableCursorFilter(QObject):
    """
    Qt does not support setting cursor via QSS (it prints: 'Unknown property cursor').

    We apply a pointing-hand cursor to clickable controls at runtime instead,
    including buttons created inside dialogs (QMessageBox/QInputDialog).
    """

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        et = event.type()

        if isinstance(obj, QAbstractButton):
            if et in (
                QEvent.Type.Polish,
                QEvent.Type.Show,
                QEvent.Type.EnabledChange,
            ):
                obj.setCursor(Qt.CursorShape.PointingHandCursor if obj.isEnabled() else Qt.CursorShape.ArrowCursor)
            return False

        # When a widget gets children later (dialogs), ensure buttons get cursors once created.
        if isinstance(obj, QWidget) and et == QEvent.Type.ChildAdded:
            ch = event.child()
            if isinstance(ch, QAbstractButton):
                ch.setCursor(Qt.CursorShape.PointingHandCursor if ch.isEnabled() else Qt.CursorShape.ArrowCursor)
            return False

        return False


_installed = False


def install_clickable_cursor_affordance(app: QApplication) -> None:
    global _installed
    if _installed:
        return
    app.installEventFilter(ClickableCursorFilter(app))
    _installed = True

