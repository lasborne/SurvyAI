"""
Pill-style light/dark theme switch for the title bar.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Property, QEasingCurve, QPointF, QPropertyAnimation, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


class ThemeToggle(QWidget):
    """Compact sun/moon pill toggle — checked means dark mode."""

    toggled = Signal(bool)

    _W = 58
    _H = 30

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("themeToggle")
        self.setFixedSize(self._W, self._H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Switch light / dark appearance")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._checked = False
        self._thumb_pos = 0.0
        self._anim = QPropertyAnimation(self, b"thumbPos", self)
        self._anim.setDuration(160)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool, *, animate: bool = True) -> None:
        checked = bool(checked)
        target = 1.0 if checked else 0.0
        if self._checked == checked and abs(self._thumb_pos - target) < 0.02:
            return
        self._checked = checked
        self._anim.stop()
        if animate:
            self._anim.setStartValue(self._thumb_pos)
            self._anim.setEndValue(target)
            self._anim.start()
        else:
            self._thumb_pos = target
            self.update()
        if not animate:
            pass

    def _get_thumb_pos(self) -> float:
        return self._thumb_pos

    def _set_thumb_pos(self, value: float) -> None:
        self._thumb_pos = max(0.0, min(1.0, float(value)))
        self.update()

    thumbPos = Property(float, _get_thumb_pos, _set_thumb_pos)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.setChecked(not self._checked)
            self.toggled.emit(self._checked)
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w, h = float(self.width()), float(self.height())
        pad = 3.0
        track = QRectF(pad, pad, w - 2 * pad, h - 2 * pad)
        radius = track.height() / 2.0

        if self._checked:
            track_bg = QColor("#2a2a32")
            track_border = QColor("#3f3f48")
            thumb_bg = QColor("#f4f4f5")
            icon_on_track = QColor("#71717a")
            icon_on_thumb = QColor("#18181b")
        else:
            track_bg = QColor("#e4e4e7")
            track_border = QColor("#d4d4d8")
            thumb_bg = QColor("#ffffff")
            icon_on_track = QColor("#a1a1aa")
            icon_on_thumb = QColor("#f59e0b")

        p.setPen(QPen(track_border, 1.0))
        p.setBrush(track_bg)
        p.drawRoundedRect(track, radius, radius)

        # Sun (left) and moon (right) hints on track
        self._paint_sun(p, QPointF(track.left() + 11, track.center().y()), 5.5, icon_on_track)
        self._paint_moon(p, QPointF(track.right() - 11, track.center().y()), 5.0, icon_on_track)

        thumb_d = track.height() - 4.0
        travel = track.width() - thumb_d - 4.0
        thumb_x = track.left() + 2.0 + travel * self._thumb_pos
        thumb_y = track.top() + 2.0
        thumb_rect = QRectF(thumb_x, thumb_y, thumb_d, thumb_d)

        p.setPen(QPen(QColor(0, 0, 0, 18), 1.0))
        p.setBrush(thumb_bg)
        p.drawEllipse(thumb_rect)

        cx, cy = thumb_rect.center().x(), thumb_rect.center().y()
        if self._checked:
            self._paint_moon(p, QPointF(cx, cy), 4.5, icon_on_thumb)
        else:
            self._paint_sun(p, QPointF(cx, cy), 4.5, icon_on_thumb)

        p.end()

    @staticmethod
    def _paint_sun(p: QPainter, center: QPointF, r: float, color: QColor) -> None:
        p.setPen(QPen(color, 1.35, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(center, r * 0.42, r * 0.42)
        for i in range(8):
            angle = i * 45.0
            rad = angle * 3.14159265 / 180.0
            inner = r * 0.62
            outer = r * 0.95
            p.drawLine(
                QPointF(center.x() + inner * math.cos(rad), center.y() - inner * math.sin(rad)),
                QPointF(center.x() + outer * math.cos(rad), center.y() - outer * math.sin(rad)),
            )

    @staticmethod
    def _paint_moon(p: QPainter, center: QPointF, r: float, color: QColor) -> None:
        path = QPainterPath()
        path.addEllipse(center, r, r)
        cut = QPainterPath()
        cut.addEllipse(QPointF(center.x() + r * 0.38, center.y() - r * 0.12), r * 0.82, r * 0.82)
        crescent = path.subtracted(cut)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawPath(crescent)


__all__ = ["ThemeToggle"]
