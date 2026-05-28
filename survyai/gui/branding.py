"""
SurvyAI mark: refined geometric monogram and application icon.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, QSize
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget


def _s_glyph_path(box: QRectF) -> QPainterPath:
    """Centred monogram glyph — DemiBold display cut for crisp scaling."""
    path = QPainterPath()
    pt_size = max(8, int(box.height() * 0.82))
    font = QFont("Segoe UI Variable Display", pt_size)
    font.setStyleName("Display")
    font.setWeight(QFont.Weight.DemiBold)
    font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 94)
    metrics = QFontMetricsF(font)
    text = "S"
    tw = metrics.horizontalAdvance(text)
    x = box.x() + (box.width() - tw) / 2.0
    baseline = box.y() + (box.height() + metrics.ascent() - metrics.descent()) / 2.0
    path.addText(QPointF(x, baseline), font, text)
    return path


def _draw_logo(painter: QPainter, size: int, *, dark_ui: bool = False) -> None:
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

    s = float(size)
    margin = s * 0.06
    outer = QRectF(margin, margin, s - 2 * margin, s - 2 * margin)
    radius = outer.width() * 0.28

    # Soft drop shadow
    shadow = QPainterPath()
    shadow.addRoundedRect(outer.translated(0, s * 0.02), radius, radius)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(0, 0, 0, 42 if not dark_ui else 80))
    painter.drawPath(shadow)

    # Base plate — deep blue with subtle conical sheen
    plate = QPainterPath()
    plate.addRoundedRect(outer, radius, radius)

    if dark_ui:
        base = QLinearGradient(outer.topLeft(), outer.bottomRight())
        base.setColorAt(0.0, QColor("#4f8fff"))
        base.setColorAt(0.45, QColor("#2563eb"))
        base.setColorAt(1.0, QColor("#1e3a8a"))
        rim = QColor("#93c5fd")
    else:
        base = QLinearGradient(outer.topLeft(), outer.bottomRight())
        base.setColorAt(0.0, QColor("#5b9aff"))
        base.setColorAt(0.35, QColor("#2563eb"))
        base.setColorAt(1.0, QColor("#1d4ed8"))
        rim = QColor("#1e40af")

    painter.setBrush(base)
    painter.setPen(QPen(rim, max(0.8, s * 0.035)))
    painter.drawPath(plate)

    # Inner gloss (top-left highlight)
    gloss_rect = QRectF(
        outer.left() + outer.width() * 0.08,
        outer.top() + outer.height() * 0.06,
        outer.width() * 0.72,
        outer.height() * 0.48,
    )
    gloss = QRadialGradient(gloss_rect.center(), gloss_rect.width() * 0.85)
    gloss.setColorAt(0.0, QColor(255, 255, 255, 72))
    gloss.setColorAt(0.55, QColor(255, 255, 255, 18))
    gloss.setColorAt(1.0, QColor(255, 255, 255, 0))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(gloss)
    painter.drawRoundedRect(gloss_rect, radius * 0.7, radius * 0.7)

    # Subtle survey crosshair ring (brand cue — very light)
    ring_c = outer.center()
    ring_r = outer.width() * 0.46
    painter.setPen(QPen(QColor(255, 255, 255, 28), max(0.5, s * 0.018)))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(ring_c, ring_r, ring_r)
    tick = max(1.0, s * 0.02)
    painter.drawLine(
        QPointF(ring_c.x() - ring_r, ring_c.y()),
        QPointF(ring_c.x() + ring_r, ring_c.y()),
    )
    painter.drawLine(
        QPointF(ring_c.x(), ring_c.y() - ring_r),
        QPointF(ring_c.x(), ring_c.y() + ring_r),
    )

    # Monogram
    glyph_box = outer.adjusted(
        outer.width() * 0.22,
        outer.height() * 0.20,
        -outer.width() * 0.22,
        -outer.height() * 0.18,
    )
    glyph = _s_glyph_path(glyph_box)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#ffffff"))
    painter.drawPath(glyph)

    # Specular edge on S (upper-left)
    spec = QLinearGradient(glyph_box.topLeft(), glyph_box.bottomRight())
    spec.setColorAt(0.0, QColor(255, 255, 255, 90))
    spec.setColorAt(0.35, QColor(255, 255, 255, 0))
    painter.setBrush(spec)
    painter.drawPath(glyph)


def make_logo_pixmap(size: int = 32, *, dark_ui: bool = False) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    try:
        _draw_logo(painter, size, dark_ui=dark_ui)
    finally:
        painter.end()
    return pm


def make_app_icon() -> QIcon:
    icon = QIcon()
    for sz in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(make_logo_pixmap(sz, dark_ui=False), QIcon.Mode.Normal, QIcon.State.Off)
    return icon


class SurvyLogoWidget(QWidget):
    """Header mark with optional dark-chrome tuning."""

    def __init__(self, size: int = 32, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._size = size
        self._dark_ui = False
        self.setObjectName("survyLogo")
        self.setFixedSize(size, size)
        self.setToolTip("SurvyAI")

    def set_dark_ui(self, dark: bool) -> None:
        if self._dark_ui != dark:
            self._dark_ui = dark
            self.update()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self._size, self._size)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        try:
            _draw_logo(painter, min(self.width(), self.height()), dark_ui=self._dark_ui)
        finally:
            painter.end()


__all__ = ["SurvyLogoWidget", "make_app_icon", "make_logo_pixmap"]
