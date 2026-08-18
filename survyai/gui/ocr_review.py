"""Click-to-verify OCR review dialog (image + editable extracted fields)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


_META_LABELS = (
    ("organization", "Organization"),
    ("phone", "Phone"),
    ("telephone", "Phone"),
    ("surveyed_by", "Surveyed by"),
    ("computed_by", "Computed by"),
    ("instrument", "Instrument"),
    ("instrument_name", "Instrument"),
    ("serial", "Serial number"),
    ("serial_number", "Serial number"),
    ("date", "Date"),
    ("page", "Page"),
    ("page_number", "Page"),
    ("sheet", "Sheet"),
)

_ROW_FIELDS = (
    ("instrument_station", "Inst. Stn (from)"),
    ("reference_station", "Ref. Stn (to)"),
    ("hz_fl", "HA Face Left"),
    ("hz_fr", "HA Face Right"),
    ("va_fl", "VA Face Left"),
    ("va_fr", "VA Face Right"),
    ("slope_distance", "Slope dist"),
    ("horizontal_distance", "Hor. dist"),
    ("backsight", "BS"),
    ("intermediate_sight", "IS"),
    ("foresight", "FS"),
    ("reduced_level", "RL"),
    ("height_of_collimation", "HI"),
    ("distance", "Distance"),
    ("station", "Station"),
)


def _cell_value(cell: Any) -> str:
    if cell is None:
        return ""
    if isinstance(cell, dict):
        v = cell.get("value")
        if v is None:
            v = cell.get("raw")
        return "" if v is None else str(v)
    return str(cell)


def _cell_bbox(cell: Any) -> Optional[List[float]]:
    if not isinstance(cell, dict):
        return None
    bbox = cell.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        except (TypeError, ValueError):
            return None
    return None


def _cell_uncertain(cell: Any, field_key: str, uncertain: List[str]) -> bool:
    if field_key and any(field_key in str(u) for u in uncertain):
        return True
    if isinstance(cell, dict):
        conf = cell.get("confidence")
        try:
            if conf is not None and float(conf) < 0.55:
                return True
        except (TypeError, ValueError):
            pass
    return False


class OcrReviewDialog(QDialog):
    """Side-by-side original photo and editable OCR fields."""

    def __init__(
        self,
        review: Dict[str, Any],
        *,
        parent: Optional[QWidget] = None,
        workspace: Optional[Path] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("OCR image review")
        self.resize(980, 640)
        self._review = dict(review or {})
        self._workspace = Path(workspace) if workspace else Path.cwd()
        self._rows_data: List[Tuple[str, str, Any, Optional[List[float]]]] = []
        # (kind, key, original_cell, bbox) where kind is "meta" or "row:N:field"
        self._entries: List[Dict[str, Any]] = []
        self._highlight: Optional[QGraphicsRectItem] = None
        self._pix_item: Optional[QGraphicsPixmapItem] = None
        self._img_w = 1
        self._img_h = 1
        self.applied_summary: Optional[str] = None

        root = QVBoxLayout(self)
        title = str(self._review.get("title") or self._review.get("document_type") or "Document")
        paths = self._review.get("image_paths") or []
        src = Path(paths[0]).name if paths else "(image)"
        header = QLabel(f"<b>{title}</b> — {src}")
        header.setWordWrap(True)
        root.addWidget(header)

        quality = self._review.get("quality") or {}
        overall = str(quality.get("overall") or "")
        if overall and overall != "good":
            qlab = QLabel(str(quality.get("reason") or f"Image quality: {overall}"))
            qlab.setWordWrap(True)
            qlab.setStyleSheet("color: #b8860b;")
            root.addWidget(qlab)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        splitter.addWidget(self._view)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("Edit values if needed. Select a row to highlight its location on the photo.")
        hint.setWordWrap(True)
        right_layout.addWidget(hint)
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Field", "Value", ""])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        right_layout.addWidget(self._table, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Close
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).setText("Apply corrections")
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._apply)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        close_btn = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.clicked.connect(self.accept)
        root.addWidget(buttons)

        self._load_image()
        self._populate_table()

    def _load_image(self) -> None:
        paths = self._review.get("image_paths") or []
        if not paths:
            return
        path = Path(str(paths[0]))
        if not path.is_file():
            return
        image = QImage(str(path))
        if image.isNull():
            return
        self._img_w = max(1, image.width())
        self._img_h = max(1, image.height())
        pix = QPixmap.fromImage(image)
        self._scene.clear()
        self._pix_item = self._scene.addPixmap(pix)
        self._scene.setSceneRect(QRectF(pix.rect()))
        self._view.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if self._pix_item is not None:
            self._view.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _populate_table(self) -> None:
        self._entries.clear()
        uncertain = [str(u) for u in (self._review.get("uncertain") or [])]
        metadata = self._review.get("metadata") if isinstance(self._review.get("metadata"), dict) else {}
        seen_meta: set = set()
        for key, label in _META_LABELS:
            if key in seen_meta or key not in metadata:
                continue
            # Prefer short aliases; skip duplicate instrument_name if instrument present
            if key == "instrument_name" and "instrument" in metadata:
                continue
            if key == "serial_number" and "serial" in metadata:
                continue
            if key == "page_number" and "page" in metadata:
                continue
            if key == "telephone" and ("phone" in metadata or "telephone" in seen_meta):
                continue
            if key == "phone" and "telephone" in seen_meta:
                continue
            seen_meta.add(key)
            cell = metadata.get(key)
            self._entries.append(
                {
                    "kind": "meta",
                    "key": key,
                    "label": label,
                    "cell": cell,
                    "bbox": _cell_bbox(cell),
                    "uncertain": _cell_uncertain(cell, key, uncertain),
                }
            )

        rows = self._review.get("rows") if isinstance(self._review.get("rows"), list) else []
        known_row = {f for f, _ in _ROW_FIELDS}
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_no = row.get("row", "?")
            matched = False
            for field, label in _ROW_FIELDS:
                if field not in row:
                    continue
                matched = True
                cell = row.get(field)
                self._entries.append(
                    {
                        "kind": "row",
                        "row": row_no,
                        "key": field,
                        "label": f"R{row_no} {label}",
                        "cell": cell,
                        "bbox": _cell_bbox(cell),
                        "uncertain": _cell_uncertain(cell, field, uncertain),
                    }
                )
            if not matched:
                # Spreadsheet / generic table rows: show whatever keys were extracted
                for field, cell in row.items():
                    if field == "row" or field in known_row:
                        continue
                    if isinstance(cell, dict) and cell.get("value") in (None, "") and cell.get("raw") in (None, ""):
                        continue
                    if cell in (None, ""):
                        continue
                    self._entries.append(
                        {
                            "kind": "row",
                            "row": row_no,
                            "key": str(field),
                            "label": f"R{row_no} {field}",
                            "cell": cell,
                            "bbox": _cell_bbox(cell),
                            "uncertain": _cell_uncertain(cell, str(field), uncertain),
                        }
                    )

        self._table.setRowCount(len(self._entries))
        for i, entry in enumerate(self._entries):
            label_item = QTableWidgetItem(str(entry["label"]))
            label_item.setFlags(label_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(i, 0, label_item)
            value_item = QTableWidgetItem(_cell_value(entry.get("cell")))
            self._table.setItem(i, 1, value_item)
            flag = QTableWidgetItem("⚠" if entry.get("uncertain") else "")
            flag.setFlags(flag.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if entry.get("uncertain"):
                flag.setToolTip("Low confidence or failed a silent survey-consistency check.")
                flag.setForeground(QColor("#b8860b"))
            self._table.setItem(i, 2, flag)

    def _on_selection_changed(self) -> None:
        rows = self._table.selectionModel().selectedRows() if self._table.selectionModel() else []
        if not rows:
            self._clear_highlight()
            return
        idx = rows[0].row()
        if idx < 0 or idx >= len(self._entries):
            self._clear_highlight()
            return
        bbox = self._entries[idx].get("bbox")
        self._set_highlight(bbox)

    def _clear_highlight(self) -> None:
        if self._highlight is not None:
            self._scene.removeItem(self._highlight)
            self._highlight = None

    def _set_highlight(self, bbox: Optional[List[float]]) -> None:
        self._clear_highlight()
        if not bbox or len(bbox) < 4:
            return
        x0, y0, x1, y1 = bbox[:4]
        # Normalized 0–1 → pixel coords
        if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
            px0, py0 = x0 * self._img_w, y0 * self._img_h
            px1, py1 = x1 * self._img_w, y1 * self._img_h
        else:
            px0, py0, px1, py1 = x0, y0, x1, y1
        rect = QRectF(
            min(px0, px1),
            min(py0, py1),
            max(2.0, abs(px1 - px0)),
            max(2.0, abs(py1 - py0)),
        )
        self._highlight = QGraphicsRectItem(rect)
        pen = QPen(QColor("#f5c518"))
        pen.setWidth(3)
        self._highlight.setPen(pen)
        self._highlight.setBrush(QColor(245, 197, 24, 40))
        self._scene.addItem(self._highlight)
        self._view.ensureVisible(rect)

    def _apply(self) -> None:
        updated_meta: Dict[str, Any] = {}
        updated_rows: Dict[int, Dict[str, Any]] = {}
        previous_metadata: Dict[str, str] = {}
        previous_rows: Dict[str, Dict[str, str]] = {}
        for i, entry in enumerate(self._entries):
            item = self._table.item(i, 1)
            new_val = (item.text() if item else "").strip()
            old_val = _cell_value(entry.get("cell")).strip()
            if new_val == old_val:
                continue
            if entry["kind"] == "meta":
                updated_meta[entry["key"]] = new_val
                previous_metadata[entry["key"]] = old_val
            else:
                r = int(entry.get("row") or 0)
                updated_rows.setdefault(r, {})[entry["key"]] = new_val
                previous_rows.setdefault(str(r), {})[entry["key"]] = old_val

        if not updated_meta and not updated_rows:
            QMessageBox.information(self, "OCR review", "No corrections to apply.")
            return

        # Persist sidecar for later reprocess + wrong→right learning
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = self._workspace / ".survyai" / "ocr"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "image_paths": self._review.get("image_paths"),
                "document_type": self._review.get("document_type"),
                "metadata_corrections": updated_meta,
                "row_corrections": {str(k): v for k, v in updated_rows.items()},
                "previous_metadata": previous_metadata,
                "previous_rows": previous_rows,
                "review": self._review,
            }
            path = out_dir / f"{stamp}.json"
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "OCR review", f"Could not save corrections: {exc}")
            return

        # Feed handwriting memory with corrected glyph habits only lightly (date/serial notes)
        try:
            from agent.vision_ocr import save_handwriting_style

            style = dict(self._review.get("style_card") or {})
            writer = ""
            meta = self._review.get("metadata") or {}
            if isinstance(meta.get("surveyed_by"), dict):
                writer = str(meta["surveyed_by"].get("value") or "")
            elif meta.get("surveyed_by"):
                writer = str(meta.get("surveyed_by"))
            if updated_meta.get("date") and "7" in str(updated_meta.get("date")):
                style["7"] = style.get("7") or "cross-barred angular 7"
            save_handwriting_style(style, workspace=self._workspace, writer=writer)
        except Exception:
            pass

        n = len(updated_meta) + sum(len(v) for v in updated_rows.values())
        # Merge corrections into review for caller / full re-display
        for k, v in updated_meta.items():
            md = self._review.setdefault("metadata", {})
            if isinstance(md.get(k), dict):
                md[k] = {**md[k], "value": v, "raw": v}
            else:
                md[k] = {"value": v, "raw": v}
        for r, fields in updated_rows.items():
            for row in self._review.get("rows") or []:
                if isinstance(row, dict) and int(row.get("row") or 0) == r:
                    for fk, fv in fields.items():
                        cur = row.get(fk)
                        if isinstance(cur, dict):
                            row[fk] = {**cur, "value": fv, "raw": fv}
                        else:
                            row[fk] = {"value": fv, "raw": fv}
        try:
            from agent.vision_ocr import (
                format_ocr_review_for_user,
                save_last_ocr_extraction,
                structured_from_ocr_review,
            )

            self.applied_summary = format_ocr_review_for_user(
                self._review,
                note=f"Updated {n} field(s) from image review.",
            )
            save_last_ocr_extraction(
                structured_from_ocr_review(self._review),
                workspace=self._workspace,
                image_paths=self._review.get("image_paths") or [],
                document_type=str(self._review.get("document_type") or ""),
                source="ocr_review_apply",
            )
        except Exception:
            self.applied_summary = f"Updated {n} field(s) from image review."
        QMessageBox.information(self, "OCR review", f"Applied {n} correction(s). Full updated extraction is shown in chat.")
        self.accept()


def open_ocr_review_dialog(
    review: Dict[str, Any],
    *,
    parent: Optional[QWidget] = None,
    workspace: Optional[Path] = None,
) -> Optional[str]:
    """Show the review dialog. Returns full updated extraction text after Apply."""
    if not isinstance(review, dict):
        return None
    paths = review.get("image_paths") or []
    if not paths or not any(Path(str(p)).is_file() for p in paths):
        return None
    if not (review.get("rows") or review.get("metadata") or review.get("plain_text")):
        # Still allow open when quality-only failure left empty rows but images exist
        if not review.get("quality"):
            return None
    dlg = OcrReviewDialog(review, parent=parent, workspace=workspace)
    dlg.exec()
    return dlg.applied_summary


__all__ = ["OcrReviewDialog", "open_ocr_review_dialog"]
