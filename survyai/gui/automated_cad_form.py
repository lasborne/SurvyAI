"""Structured Automated CAD section form (values only — no free-form prompt)."""

from __future__ import annotations

from typing import Callable, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from survyai.gui.automated_cad_prompt import (
    ADJUST_BEARING,
    ADJUST_BOWDITCH,
    FENCE_CONCRETE,
    FENCE_DWARF,
    FENCE_NONE,
    MODE_BEARINGS,
    MODE_COORDINATES,
    AccessRoad,
    CadFormState,
    TraverseLeg,
    WallFence,
    compose_cad_prompt,
    ordinal_label,
    parse_cad_prompt,
)


def _plus_button(tooltip: str) -> QToolButton:
    btn = QToolButton()
    btn.setObjectName("cadAddButton")
    btn.setText("+")
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setToolTip(tooltip)
    btn.setAutoRaise(False)
    return btn


def _remove_button(tooltip: str) -> QToolButton:
    btn = QToolButton()
    btn.setObjectName("cadRemoveButton")
    btn.setText("\u00d7")
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setToolTip(tooltip)
    btn.setAutoRaise(False)
    return btn


def _info_button(tooltip: str) -> QToolButton:
    info = QToolButton()
    info.setObjectName("cadInfoButton")
    info.setText("i")
    info.setCursor(Qt.CursorShape.PointingHandCursor)
    info.setToolTip(tooltip)
    info.setAutoRaise(False)
    return info


class AutomatedCadForm(QWidget):
    """Vertical field/value strip for automated cadastral plan plotting."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("cadFormRoot")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 14, 16)
        root.setSpacing(12)

        hint = QLabel(
            "Fill the survey values below. SurvyAI will plot the plan from these fields — "
            "you do not need to write a prompt. Switch to Console for follow-ups "
            "(roads, title, subdivision, save as)."
        )
        hint.setToolTip(
            "Send plots from this form. Use Console in the same conversation for later edits."
        )
        hint.setObjectName("cadHintLabel")
        hint.setWordWrap(True)
        root.addWidget(hint)

        root.addWidget(self._build_mode_switch())
        root.addWidget(self._build_identity_block())
        root.addWidget(self._build_site_block())
        root.addWidget(self._build_pillars_block())
        self._coords_section = self._build_coordinates_block()
        root.addWidget(self._coords_section)
        self._bearings_section = self._build_bearings_block()
        root.addWidget(self._bearings_section)
        root.addWidget(self._build_roads_block())
        root.addWidget(self._build_fences_block())
        root.addWidget(self._build_certification_block())
        root.addStretch(1)

        self._set_mode(MODE_COORDINATES)
        self._mode_coords_btn.toggled.connect(self._on_mode_toggled)
        self._mode_bearings_btn.toggled.connect(self._on_mode_toggled)

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------

    def _build_mode_switch(self) -> QWidget:
        wrap = QWidget()
        wrap.setObjectName("cadModeSwitch")
        layout = QHBoxLayout(wrap)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._mode_coords_btn = QPushButton("Coordinates")
        self._mode_coords_btn.setObjectName("cadModeButton")
        self._mode_coords_btn.setCheckable(True)
        self._mode_coords_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._mode_coords_btn.setToolTip(
            "Plot from a closed ring of Easting, Northing coordinates. "
            "Bearing and distance boxes stay hidden."
        )
        self._mode_bearings_btn = QPushButton("Bearings and distances")
        self._mode_bearings_btn.setObjectName("cadModeButton")
        self._mode_bearings_btn.setCheckable(True)
        self._mode_bearings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._mode_bearings_btn.setToolTip(
            "Plot from one start coordinate (E, N) plus traverse legs of bearing and distance."
        )
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_group.addButton(self._mode_coords_btn)
        self._mode_group.addButton(self._mode_bearings_btn)
        layout.addWidget(self._mode_coords_btn, 1)
        layout.addWidget(self._mode_bearings_btn, 1)
        return wrap

    def _on_mode_toggled(self, _checked: bool) -> None:
        if self._mode_bearings_btn.isChecked():
            self._set_mode(MODE_BEARINGS)
        elif self._mode_coords_btn.isChecked():
            self._set_mode(MODE_COORDINATES)

    def _set_mode(self, mode: str) -> None:
        is_coords = mode != MODE_BEARINGS
        self._mode_coords_btn.blockSignals(True)
        self._mode_bearings_btn.blockSignals(True)
        self._mode_coords_btn.setChecked(is_coords)
        self._mode_bearings_btn.setChecked(not is_coords)
        self._mode_coords_btn.blockSignals(False)
        self._mode_bearings_btn.blockSignals(False)
        self._coords_section.setVisible(is_coords)
        self._bearings_section.setVisible(not is_coords)

    def current_mode(self) -> str:
        if self._mode_bearings_btn.isChecked():
            return MODE_BEARINGS
        return MODE_COORDINATES

    # ------------------------------------------------------------------
    # Identity / site
    # ------------------------------------------------------------------

    def _build_identity_block(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        header = QLabel("Plot identity")
        header.setObjectName("cadSectionTitle")
        layout.addWidget(header)
        self._save_as_edit = self._add_single_row(
            layout,
            "Save File As:",
            "Optional filename (without or with .dwg). Blank → first owner name.",
            tooltip=(
                "Optional .dwg name. Blank uses the first owner name. "
                "Existing files ask before overwrite."
            ),
        )
        self._owners_host = QVBoxLayout()
        self._owners_host.setSpacing(6)
        layout.addLayout(self._owners_host)
        self._owner_rows: List[_SimpleValueRow] = []
        self._add_owner_row(removable=False)
        return box

    def _add_owner_row(self, *, removable: bool) -> None:
        row = _SimpleValueRow(
            "Owner/Buyer name:",
            placeholder="e.g. Mr. Richyblue James Doe",
            plus_tooltip="Add another real owner or buyer. Names appear together on the title block.",
            remove_tooltip="Remove this owner / buyer name",
            field_tooltip="Use only real owners. Several names are joined on the title block.",
            removable=removable,
            on_plus=lambda: self._add_owner_row(removable=True),
            on_remove=lambda r: self._remove_simple_row(self._owner_rows, r, min_count=1),
        )
        self._owner_rows.append(row)
        self._owners_host.addWidget(row)
        _refresh_plus_on_last(self._owner_rows)

    def _build_site_block(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        header = QLabel("Site")
        header.setObjectName("cadSectionTitle")
        layout.addWidget(header)

        self._location_edit = self._add_single_row(
            layout,
            "Location:",
            "Parcel location (one place)",
            tooltip="Site or locality as it should appear on the plan.",
        )
        self._lga_edit = self._add_single_row(
            layout,
            "Local Govt. Area:",
            "e.g. Etche Local Government Area",
            tooltip="Local government area for the title block.",
        )
        self._state_edit = self._add_single_row(
            layout,
            "State:",
            "e.g. Rivers",
            tooltip="State or region for the title block.",
        )

        scale_row = QWidget()
        scale_layout = QHBoxLayout(scale_row)
        scale_layout.setContentsMargins(0, 2, 0, 0)
        scale_layout.setSpacing(8)
        self._auto_scale_cb = QCheckBox("Use Auto-scale")
        self._auto_scale_cb.setChecked(True)
        self._auto_scale_cb.setObjectName("cadAutoScaleCheck")
        self._auto_scale_cb.setToolTip(
            "On (default): SurvyAI picks a standard survey scale. "
            "Off: type a scale such as 1:500."
        )
        info = _info_button(
            "Do you want to specify the survey scale manually, or do you want to let "
            "the App automatically choose a suitable scale?\n\n"
            "Auto-scale is on by default. If you turn it off but leave Scale blank "
            "(or type an invalid scale), SurvyAI still plots with Auto-scale."
        )
        scale_layout.addWidget(self._auto_scale_cb, 0)
        scale_layout.addWidget(info, 0, Qt.AlignmentFlag.AlignVCenter)
        scale_layout.addStretch(1)
        layout.addWidget(scale_row)

        self._scale_row = QWidget()
        scale_field = QHBoxLayout(self._scale_row)
        scale_field.setContentsMargins(0, 0, 0, 0)
        scale_field.setSpacing(8)
        scale_label = QLabel("Scale:")
        scale_label.setObjectName("cadFieldLabel")
        scale_label.setMinimumWidth(168)
        self._scale_edit = QLineEdit()
        self._scale_edit.setPlaceholderText("e.g. 1:500")
        self._scale_edit.setToolTip(
            "Optional. Allowed survey scales: 1:250, 1:500, 1:1000, 1:2000, "
            "1:2500, 1:5000, 1:10000, 1:20000, 1:25000. Blank or invalid → Auto-scale."
        )
        scale_field.addWidget(scale_label, 0)
        scale_field.addWidget(self._scale_edit, 1)
        self._scale_row.setVisible(False)
        layout.addWidget(self._scale_row)
        self._auto_scale_cb.toggled.connect(self._on_auto_scale_toggled)

        self._origin_edit = self._add_single_row(
            layout,
            "Origin:",
            "e.g. UTM Zone 32N",
            tooltip="Coordinate reference system, for example UTM Zone 32N.",
        )
        return box

    def _on_auto_scale_toggled(self, checked: bool) -> None:
        self._scale_row.setVisible(not checked)
        if checked:
            return
        self._scale_edit.setFocus(Qt.FocusReason.OtherFocusReason)

    def _add_single_row(
        self,
        parent: QVBoxLayout,
        label: str,
        placeholder: str,
        tooltip: str = "",
    ) -> QLineEdit:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        lab = QLabel(label)
        lab.setObjectName("cadFieldLabel")
        lab.setMinimumWidth(168)
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        if tooltip:
            lab.setToolTip(tooltip)
            edit.setToolTip(tooltip)
        layout.addWidget(lab, 0)
        layout.addWidget(edit, 1)
        parent.addWidget(row)
        return edit

    # ------------------------------------------------------------------
    # Pillars / coordinates / bearings
    # ------------------------------------------------------------------

    def _build_pillars_block(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        header = QLabel("Traverse stations")
        header.setObjectName("cadSectionTitle")
        layout.addWidget(header)
        note = QLabel("A traverse needs at least 3 pillars (1st, 2nd, 3rd…).")
        note.setObjectName("cadHintLabel")
        note.setWordWrap(True)
        layout.addWidget(note)
        self._pillars_host = QVBoxLayout()
        self._pillars_host.setSpacing(6)
        layout.addLayout(self._pillars_host)
        self._pillar_rows: List[_SimpleValueRow] = []
        for _ in range(3):
            self._add_pillar_row(removable=False)
        return box

    def _add_pillar_row(self, *, removable: bool) -> None:
        row = _SimpleValueRow(
            "1st pillar:",
            placeholder="e.g. SP/RV 1000",
            plus_tooltip="Add another station. A closed traverse needs at least three.",
            remove_tooltip="Remove this pillar",
            field_tooltip="Station mark as it should appear on the plan, for example SP/RV 1000.",
            removable=removable,
            on_plus=lambda: self._add_pillar_row(removable=True),
            on_remove=lambda r: self._remove_simple_row(self._pillar_rows, r, min_count=3),
            index_noun="pillar",
        )
        self._pillar_rows.append(row)
        self._pillars_host.addWidget(row)
        _refresh_plus_on_last(self._pillar_rows)
        _relabel_indexed_rows(self._pillar_rows)

    def _build_coordinates_block(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        header = QLabel("Coordinates")
        header.setObjectName("cadSectionTitle")
        layout.addWidget(header)
        note = QLabel(
            "Enter Easting, Northing for each station in the same order as the pillars "
            "(1st coordinate with 1st pillar). Example: 200200.400, 576000.100"
        )
        note.setObjectName("cadHintLabel")
        note.setWordWrap(True)
        layout.addWidget(note)
        self._coords_host = QVBoxLayout()
        self._coords_host.setSpacing(6)
        layout.addLayout(self._coords_host)
        self._coord_rows: List[_SimpleValueRow] = []
        for _ in range(3):
            self._add_coord_row(removable=False)
        return box

    def _add_coord_row(self, *, removable: bool) -> None:
        row = _SimpleValueRow(
            "1st coordinate:",
            placeholder="Easting, Northing",
            plus_tooltip="Add another station coordinate (Easting, Northing).",
            remove_tooltip="Remove this coordinate",
            field_tooltip="Easting, Northing for this station, matching the same-numbered pillar.",
            removable=removable,
            on_plus=lambda: self._add_coord_row(removable=True),
            on_remove=lambda r: self._remove_simple_row(self._coord_rows, r, min_count=3),
            index_noun="coordinate",
        )
        self._coord_rows.append(row)
        self._coords_host.addWidget(row)
        _refresh_plus_on_last(self._coord_rows)
        _relabel_indexed_rows(self._coord_rows)

    def _build_bearings_block(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        header = QLabel("Bearings and distances")
        header.setObjectName("cadSectionTitle")
        layout.addWidget(header)
        note = QLabel(
            "1st coordinate is the start station (E, N), then at least 3 traverse legs. "
            "The last leg closes the traverse back to the first station. "
            "Blank minutes or seconds are treated as 0; degrees must be filled."
        )
        note.setObjectName("cadHintLabel")
        note.setWordWrap(True)
        layout.addWidget(note)

        start_row = QWidget()
        start_layout = QHBoxLayout(start_row)
        start_layout.setContentsMargins(0, 0, 0, 0)
        start_layout.setSpacing(8)
        start_lab = QLabel("1st coordinate:")
        start_lab.setObjectName("cadFieldLabel")
        start_lab.setMinimumWidth(168)
        self._start_coord_lab = start_lab
        self._start_coord_edit = QLineEdit()
        self._start_coord_edit.setPlaceholderText("Easting, Northing  (E, N)")
        self._start_coord_edit.setToolTip(
            "1st coordinate — start station (E, N) for the bearing traverse. "
            "Later legs close back to this point."
        )
        start_lab.setToolTip(self._start_coord_edit.toolTip())
        start_layout.addWidget(start_lab, 0)
        start_layout.addWidget(self._start_coord_edit, 1)
        layout.addWidget(start_row)

        self._legs_host = QVBoxLayout()
        self._legs_host.setSpacing(10)
        layout.addLayout(self._legs_host)
        self._leg_rows: List[_TraverseLegRow] = []
        for _ in range(3):
            self._add_leg_row(removable=False)
        layout.addWidget(self._build_adjustment_row())
        return box

    def _add_leg_row(self, *, removable: bool) -> None:
        row = _TraverseLegRow(
            index=len(self._leg_rows),
            removable=removable,
            on_plus=lambda: self._add_leg_row(removable=True),
            on_remove=self._remove_leg_row,
        )
        self._leg_rows.append(row)
        self._legs_host.addWidget(row)
        self._relabel_legs()

    def _remove_leg_row(self, row: "_TraverseLegRow") -> None:
        if len(self._leg_rows) <= 3:
            return
        self._leg_rows.remove(row)
        self._legs_host.removeWidget(row)
        row.deleteLater()
        self._relabel_legs()

    def _relabel_legs(self) -> None:
        for i, row in enumerate(self._leg_rows):
            row.set_index(i)
        _refresh_plus_on_last(self._leg_rows)

    def _build_adjustment_row(self) -> QWidget:
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(6)
        note = QLabel("Traverse adjustment (one method only):")
        note.setObjectName("cadHintLabel")
        layout.addWidget(note)

        bearing_row = QHBoxLayout()
        bearing_row.setContentsMargins(0, 0, 0, 0)
        bearing_row.setSpacing(8)
        self._bearing_adj_cb = QCheckBox("Bearing adjustment (hold distances constant)")
        self._bearing_adj_cb.setObjectName("cadAutoScaleCheck")
        self._bearing_adj_cb.setChecked(True)
        self._bearing_adj_cb.setToolTip(
            "Standard field-traverse closure: measured distances are held fixed and "
            "only bearings are adjusted so the traverse closes. This is the default."
        )
        bearing_info = _info_button(
            "Bearing adjustment (default)\n\n"
            "Distances are held constant. Only the bearings are adjusted by a small "
            "uniform amount so the last station meets the first.\n\n"
            "Use this when tape / EDM distances are more reliable than compass or "
            "theodolite bearings — the usual cadastral field case."
        )
        bearing_row.addWidget(self._bearing_adj_cb, 0)
        bearing_row.addWidget(bearing_info, 0, Qt.AlignmentFlag.AlignVCenter)
        bearing_row.addStretch(1)
        layout.addLayout(bearing_row)

        bow_row = QHBoxLayout()
        bow_row.setContentsMargins(0, 0, 0, 0)
        bow_row.setSpacing(8)
        self._bowditch_cb = QCheckBox("Bowditch adjustment (compass rule)")
        self._bowditch_cb.setObjectName("cadAutoScaleCheck")
        self._bowditch_cb.setChecked(False)
        self._bowditch_cb.setToolTip(
            "Compass-rule closure: both bearings and distances are corrected in "
            "proportion to each leg length so the traverse closes."
        )
        bow_info = _info_button(
            "Bowditch / compass-rule adjustment\n\n"
            "Misclosure is distributed over every leg in proportion to its length. "
            "Both bearings and distances change; coordinates are then taken from the "
            "adjusted legs.\n\n"
            "Use this only when you want equal trust in angles and distances, or "
            "when a specification calls for Bowditch / compass rule."
        )
        bow_row.addWidget(self._bowditch_cb, 0)
        bow_row.addWidget(bow_info, 0, Qt.AlignmentFlag.AlignVCenter)
        bow_row.addStretch(1)
        layout.addLayout(bow_row)

        self._adj_group = QButtonGroup(self)
        self._adj_group.setExclusive(True)
        self._adj_group.addButton(self._bearing_adj_cb)
        self._adj_group.addButton(self._bowditch_cb)
        self._bearing_adj_cb.toggled.connect(self._on_adjustment_toggled)
        self._bowditch_cb.toggled.connect(self._on_adjustment_toggled)
        return wrap

    def _on_adjustment_toggled(self, checked: bool) -> None:
        if checked:
            return
        if not self._bearing_adj_cb.isChecked() and not self._bowditch_cb.isChecked():
            sender = self.sender()
            if sender is self._bowditch_cb:
                self._bearing_adj_cb.setChecked(True)
            else:
                self._bearing_adj_cb.setChecked(True)

    # ------------------------------------------------------------------
    # Roads / fences / certification
    # ------------------------------------------------------------------

    def _build_roads_block(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        head = QHBoxLayout()
        header = QLabel("Access roads")
        header.setObjectName("cadSectionTitle")
        head.addWidget(header, 1)
        layout.addLayout(head)
        note = QLabel("Leave blank to omit roads from the plan.")
        note.setObjectName("cadHintLabel")
        layout.addWidget(note)
        self._roads_host = QVBoxLayout()
        self._roads_host.setSpacing(10)
        layout.addLayout(self._roads_host)
        self._road_groups: List[_AccessRoadGroup] = []
        self._add_road_group(removable=False)
        return box

    def _add_road_group(self, *, removable: bool) -> None:
        group = _AccessRoadGroup(
            index=len(self._road_groups),
            removable=removable,
            on_plus=lambda: self._add_road_group(removable=True),
            on_remove=self._remove_road_group,
        )
        self._road_groups.append(group)
        self._roads_host.addWidget(group)
        self._relabel_roads()

    def _remove_road_group(self, group: "_AccessRoadGroup") -> None:
        if len(self._road_groups) <= 1:
            return
        self._road_groups.remove(group)
        self._roads_host.removeWidget(group)
        group.deleteLater()
        self._relabel_roads()

    def _relabel_roads(self) -> None:
        for i, group in enumerate(self._road_groups):
            group.set_index(i)
        _refresh_plus_on_last(self._road_groups)

    def _build_fences_block(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        head = QHBoxLayout()
        header = QLabel("Wall fences")
        header.setObjectName("cadSectionTitle")
        head.addWidget(header, 1)
        layout.addLayout(head)
        self._fences_host = QVBoxLayout()
        self._fences_host.setSpacing(10)
        layout.addLayout(self._fences_host)
        self._fence_groups: List[_WallFenceGroup] = []
        self._add_fence_group(removable=False)
        return box

    def _add_fence_group(self, *, removable: bool) -> None:
        group = _WallFenceGroup(
            index=len(self._fence_groups),
            removable=removable,
            on_plus=lambda: self._add_fence_group(removable=True),
            on_remove=self._remove_fence_group,
        )
        self._fence_groups.append(group)
        self._fences_host.addWidget(group)
        self._relabel_fences()

    def _remove_fence_group(self, group: "_WallFenceGroup") -> None:
        if len(self._fence_groups) <= 1:
            return
        self._fence_groups.remove(group)
        self._fences_host.removeWidget(group)
        group.deleteLater()
        self._relabel_fences()

    def _relabel_fences(self) -> None:
        for i, group in enumerate(self._fence_groups):
            group.set_index(i)
        _refresh_plus_on_last(self._fence_groups)

    def _build_certification_block(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        header = QLabel("Certification")
        header.setObjectName("cadSectionTitle")
        layout.addWidget(header)
        self._plan_number_edit = self._add_single_row(
            layout,
            "Plan number:",
            "e.g. RV/0000/2026/001",
            tooltip="Plan or job number printed on the title block.",
        )
        self._surveyor_name_edit = self._add_single_row(
            layout,
            "Surveyor's name:",
            "e.g. Surv. Robotics John Doe (mnis)",
            tooltip="Certifying surveyor as it should appear on the plan.",
        )
        self._surveyor_company_edit = self._add_single_row(
            layout,
            "Surveyor's company:",
            "Optional company name",
            tooltip="Firm name on the title block. Leave blank if none.",
        )
        self._surveyor_address_edit = self._add_single_row(
            layout,
            "Surveyor's address:",
            "Surveyor’s office address",
            tooltip="Office address printed with the surveyor details.",
        )
        return box

    def _remove_simple_row(
        self,
        rows: List["_SimpleValueRow"],
        row: "_SimpleValueRow",
        *,
        min_count: int,
    ) -> None:
        if len(rows) <= min_count:
            return
        rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        _refresh_plus_on_last(rows)
        _relabel_indexed_rows(rows)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def collect_state(self) -> CadFormState:
        adj = ADJUST_BOWDITCH if self._bowditch_cb.isChecked() else ADJUST_BEARING
        return CadFormState(
            mode=self.current_mode(),
            save_as=self._save_as_edit.text(),
            owners=[r.text() for r in self._owner_rows],
            location=self._location_edit.text(),
            lga=self._lga_edit.text(),
            state=self._state_edit.text(),
            auto_scale=self._auto_scale_cb.isChecked(),
            scale=self._scale_edit.text(),
            origin=self._origin_edit.text(),
            pillars=[r.text() for r in self._pillar_rows],
            coordinates=[r.text() for r in self._coord_rows],
            start_coordinate=self._start_coord_edit.text(),
            legs=[r.leg() for r in self._leg_rows],
            roads=[g.road() for g in self._road_groups],
            fences=[g.fence() for g in self._fence_groups],
            plan_number=self._plan_number_edit.text(),
            surveyor_name=self._surveyor_name_edit.text(),
            surveyor_company=self._surveyor_company_edit.text(),
            surveyor_address=self._surveyor_address_edit.text(),
            traverse_adjustment=adj,
        )

    def apply_state(self, state: CadFormState) -> None:
        self._set_mode(state.mode or MODE_COORDINATES)
        self._save_as_edit.setText(state.save_as or "")
        self._sync_simple_rows(
            self._owner_rows,
            self._owners_host,
            _nonempty_or_one(state.owners),
            min_count=1,
            factory=lambda removable: self._add_owner_row(removable=removable),
        )
        self._location_edit.setText(state.location)
        self._lga_edit.setText(state.lga)
        self._state_edit.setText(state.state)
        self._auto_scale_cb.setChecked(bool(state.auto_scale))
        self._scale_edit.setText(state.scale)
        self._on_auto_scale_toggled(self._auto_scale_cb.isChecked())
        self._origin_edit.setText(state.origin)
        self._sync_simple_rows(
            self._pillar_rows,
            self._pillars_host,
            _pad_min(state.pillars, 3),
            min_count=3,
            factory=lambda removable: self._add_pillar_row(removable=removable),
        )
        self._sync_simple_rows(
            self._coord_rows,
            self._coords_host,
            _pad_min(state.coordinates, 3),
            min_count=3,
            factory=lambda removable: self._add_coord_row(removable=removable),
        )
        self._start_coord_edit.setText(state.start_coordinate)
        self._sync_leg_rows(_pad_min_objs(state.legs, 3, TraverseLeg))
        self._sync_road_groups(state.roads or [AccessRoad()])
        self._sync_fence_groups(state.fences or [WallFence()])
        self._plan_number_edit.setText(state.plan_number)
        self._surveyor_name_edit.setText(state.surveyor_name)
        self._surveyor_company_edit.setText(state.surveyor_company)
        self._surveyor_address_edit.setText(state.surveyor_address)
        bow = (state.traverse_adjustment or ADJUST_BEARING).strip().lower() == ADJUST_BOWDITCH
        self._bearing_adj_cb.blockSignals(True)
        self._bowditch_cb.blockSignals(True)
        self._bowditch_cb.setChecked(bow)
        self._bearing_adj_cb.setChecked(not bow)
        self._bearing_adj_cb.blockSignals(False)
        self._bowditch_cb.blockSignals(False)

    def compose_prompt(self) -> tuple[str, str]:
        return compose_cad_prompt(self.collect_state())

    def apply_prompt_template(self, text: str) -> str:
        state = parse_cad_prompt(text)
        self.apply_state(state)
        return "CAD form filled from the default survey-plan prompt."

    def _sync_simple_rows(
        self,
        rows: List["_SimpleValueRow"],
        _host: QVBoxLayout,
        values: List[str],
        *,
        min_count: int,
        factory: Callable[[bool], None],
    ) -> None:
        while len(rows) > max(min_count, len(values)):
            row = rows[-1]
            if not row.is_removable():
                break
            self._remove_simple_row(rows, row, min_count=min_count)
        while len(rows) < len(values):
            factory(True)
        for row, value in zip(rows, values):
            row.set_text(value)
        _refresh_plus_on_last(rows)
        _relabel_indexed_rows(rows)

    def _sync_leg_rows(self, legs: List[TraverseLeg]) -> None:
        while len(self._leg_rows) > max(3, len(legs)):
            self._remove_leg_row(self._leg_rows[-1])
        while len(self._leg_rows) < len(legs):
            self._add_leg_row(removable=True)
        for row, leg in zip(self._leg_rows, legs):
            row.set_leg(leg)

    def _sync_road_groups(self, roads: List[AccessRoad]) -> None:
        while len(self._road_groups) > max(1, len(roads)):
            self._remove_road_group(self._road_groups[-1])
        while len(self._road_groups) < len(roads):
            self._add_road_group(removable=True)
        for group, road in zip(self._road_groups, roads):
            group.set_road(road)

    def _sync_fence_groups(self, fences: List[WallFence]) -> None:
        while len(self._fence_groups) > max(1, len(fences)):
            self._remove_fence_group(self._fence_groups[-1])
        while len(self._fence_groups) < len(fences):
            self._add_fence_group(removable=True)
        for group, fence in zip(self._fence_groups, fences):
            group.set_fence(fence)


def _nonempty_or_one(values: List[str]) -> List[str]:
    filled = [v for v in values if str(v or "").strip()]
    return filled or [""]


def _pad_min(values: List[str], minimum: int) -> List[str]:
    out = list(values or [])
    while len(out) < minimum:
        out.append("")
    return out


def _pad_min_objs(values: List, minimum: int, factory) -> List:
    out = list(values or [])
    while len(out) < minimum:
        out.append(factory())
    return out


def _refresh_plus_on_last(rows: List) -> None:
    """Keep the + control on the last instance of a repeatable field."""
    last = len(rows) - 1
    for i, row in enumerate(rows):
        setter = getattr(row, "set_plus_visible", None)
        if callable(setter):
            setter(i == last)


def _relabel_indexed_rows(rows: List) -> None:
    """Keep 1st / 2nd / 3rd… labels in list order after add or remove."""
    for i, row in enumerate(rows):
        setter = getattr(row, "set_index", None)
        if callable(setter) and getattr(row, "_index_noun", ""):
            setter(i)


class _SimpleValueRow(QWidget):
    def __init__(
        self,
        label: str,
        *,
        placeholder: str,
        plus_tooltip: str,
        remove_tooltip: str,
        removable: bool,
        on_plus: Callable[[], None],
        on_remove: Callable[["_SimpleValueRow"], None],
        field_tooltip: str = "",
        index_noun: str = "",
    ) -> None:
        super().__init__()
        self._removable = removable
        self._index_noun = (index_noun or "").strip()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        lab = QLabel(label)
        lab.setObjectName("cadFieldLabel")
        lab.setMinimumWidth(178 if self._index_noun else 168)
        self._label = lab
        self._edit = QLineEdit()
        self._edit.setPlaceholderText(placeholder)
        if field_tooltip:
            lab.setToolTip(field_tooltip)
            self._edit.setToolTip(field_tooltip)
        layout.addWidget(lab, 0)
        layout.addWidget(self._edit, 1)
        self._plus = _plus_button(plus_tooltip)
        self._plus.clicked.connect(on_plus)
        layout.addWidget(self._plus, 0)
        if removable:
            remove = _remove_button(remove_tooltip)
            remove.clicked.connect(lambda: on_remove(self))
            layout.addWidget(remove, 0)

    def set_index(self, index: int) -> None:
        if not self._index_noun:
            return
        self._label.setText(f"{ordinal_label(index + 1)} {self._index_noun}:")

    def set_plus_visible(self, visible: bool) -> None:
        self._plus.setVisible(bool(visible))

    def text(self) -> str:
        return self._edit.text()

    def set_text(self, value: str) -> None:
        self._edit.setText(value or "")

    def is_removable(self) -> bool:
        return self._removable


class _TraverseLegRow(QWidget):
    def __init__(
        self,
        *,
        index: int,
        removable: bool,
        on_plus: Callable[[], None],
        on_remove: Callable[["_TraverseLegRow"], None],
    ) -> None:
        super().__init__()
        self._index = index
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        title_row = QHBoxLayout()
        self._title = QLabel()
        self._title.setObjectName("cadSubSectionTitle")
        title_row.addWidget(self._title, 1)
        self._plus = _plus_button("Add another traverse leg")
        self._plus.clicked.connect(on_plus)
        title_row.addWidget(self._plus, 0)
        if removable:
            remove = _remove_button("Remove this traverse leg")
            remove.clicked.connect(lambda: on_remove(self))
            title_row.addWidget(remove, 0)
        outer.addLayout(title_row)

        bearing_row = QWidget()
        b = QHBoxLayout(bearing_row)
        b.setContentsMargins(0, 0, 0, 0)
        b.setSpacing(6)
        b_lab = QLabel("Bearing:")
        b_lab.setObjectName("cadFieldLabel")
        b_lab.setMinimumWidth(168)
        self._deg = QLineEdit()
        self._deg.setObjectName("cadDmsEdit")
        self._deg.setPlaceholderText("deg")
        self._deg.setToolTip("Whole degrees. Required for this leg.")
        self._min = QLineEdit()
        self._min.setObjectName("cadDmsEdit")
        self._min.setPlaceholderText("min")
        self._min.setToolTip("Minutes. Leave blank for 0.")
        self._sec = QLineEdit()
        self._sec.setObjectName("cadDmsEdit")
        self._sec.setPlaceholderText("sec")
        self._sec.setToolTip("Seconds. Leave blank for 0.")
        b.addWidget(b_lab, 0)
        b.addWidget(self._deg, 0)
        b.addWidget(self._unit("°"), 0)
        b.addWidget(self._min, 0)
        b.addWidget(self._unit("'"), 0)
        b.addWidget(self._sec, 0)
        b.addWidget(self._unit('"'), 0)
        b.addStretch(1)
        outer.addWidget(bearing_row)

        dist_row = QWidget()
        d = QHBoxLayout(dist_row)
        d.setContentsMargins(0, 0, 0, 0)
        d.setSpacing(8)
        d_lab = QLabel("Distance:")
        d_lab.setObjectName("cadFieldLabel")
        d_lab.setMinimumWidth(168)
        self._dist = QLineEdit()
        self._dist.setPlaceholderText("metres")
        self._dist.setToolTip("Ground distance for this leg, in metres.")
        d.addWidget(d_lab, 0)
        d.addWidget(self._dist, 1)
        outer.addWidget(dist_row)
        self.set_index(index)

    @staticmethod
    def _unit(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setObjectName("cadUnitLabel")
        return lab

    def set_index(self, index: int) -> None:
        self._index = index
        self._title.setText(f"{ordinal_label(index + 1)} traverse leg")

    def set_plus_visible(self, visible: bool) -> None:
        self._plus.setVisible(bool(visible))

    def leg(self) -> TraverseLeg:
        return TraverseLeg(
            degrees=self._deg.text(),
            minutes=self._min.text(),
            seconds=self._sec.text(),
            distance=self._dist.text(),
        )

    def set_leg(self, leg: TraverseLeg) -> None:
        self._deg.setText(leg.degrees or "")
        self._min.setText(leg.minutes or "")
        self._sec.setText(leg.seconds or "")
        self._dist.setText(leg.distance or "")


class _AccessRoadGroup(QWidget):
    def __init__(
        self,
        *,
        index: int,
        removable: bool,
        on_plus: Callable[[], None],
        on_remove: Callable[["_AccessRoadGroup"], None],
    ) -> None:
        super().__init__()
        self.setObjectName("cadNestedGroup")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(6)
        title_row = QHBoxLayout()
        self._title = QLabel()
        self._title.setObjectName("cadSubSectionTitle")
        title_row.addWidget(self._title, 1)
        self._plus = _plus_button("Add another access road along the traverse")
        self._plus.clicked.connect(on_plus)
        title_row.addWidget(self._plus, 0)
        if removable:
            remove = _remove_button("Remove this access road")
            remove.clicked.connect(lambda: on_remove(self))
            title_row.addWidget(remove, 0)
        outer.addLayout(title_row)
        self._width = self._row(
            outer,
            "Access road width:",
            "e.g. 6",
            tooltip="Width in metres. Drawn as a dashed strip on the named edge.",
        )
        self._start = self._row(
            outer,
            "Starting pillar for Access road:",
            "e.g. SP/RV 1000",
            tooltip="Must match a pillar number already entered above.",
        )
        self._end = self._row(
            outer,
            "Ending pillar for Access road:",
            "e.g. SP/RV 1001",
            tooltip="Must be the adjacent station that shares this road edge.",
        )
        self.set_index(index)

    def _row(
        self,
        parent: QVBoxLayout,
        label: str,
        placeholder: str,
        tooltip: str = "",
    ) -> QLineEdit:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        lab = QLabel(label)
        lab.setObjectName("cadFieldLabel")
        lab.setMinimumWidth(210)
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        if tooltip:
            lab.setToolTip(tooltip)
            edit.setToolTip(tooltip)
        layout.addWidget(lab, 0)
        layout.addWidget(edit, 1)
        parent.addWidget(row)
        return edit

    def set_index(self, index: int) -> None:
        self._title.setText(f"{ordinal_label(index + 1)} Access road")

    def set_plus_visible(self, visible: bool) -> None:
        self._plus.setVisible(bool(visible))

    def road(self) -> AccessRoad:
        return AccessRoad(
            width=self._width.text(),
            start_pillar=self._start.text(),
            end_pillar=self._end.text(),
        )

    def set_road(self, road: AccessRoad) -> None:
        self._width.setText(road.width or "")
        self._start.setText(road.start_pillar or "")
        self._end.setText(road.end_pillar or "")


class _WallFenceGroup(QWidget):
    def __init__(
        self,
        *,
        index: int,
        removable: bool,
        on_plus: Callable[[], None],
        on_remove: Callable[["_WallFenceGroup"], None],
    ) -> None:
        super().__init__()
        self.setObjectName("cadNestedGroup")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(6)
        title_row = QHBoxLayout()
        self._title = QLabel()
        self._title.setObjectName("cadSubSectionTitle")
        title_row.addWidget(self._title, 1)
        self._plus = _plus_button("Add another wall fence along the traverse")
        self._plus.clicked.connect(on_plus)
        title_row.addWidget(self._plus, 0)
        if removable:
            remove = _remove_button("Remove this wall fence")
            remove.clicked.connect(lambda: on_remove(self))
            title_row.addWidget(remove, 0)
        outer.addLayout(title_row)

        type_row = QWidget()
        t = QHBoxLayout(type_row)
        t.setContentsMargins(0, 0, 0, 0)
        t.setSpacing(8)
        t_lab = QLabel("Type:")
        t_lab.setObjectName("cadFieldLabel")
        t_lab.setMinimumWidth(210)
        self._type = QComboBox()
        self._type.addItems([FENCE_NONE, FENCE_DWARF, FENCE_CONCRETE])
        self._type.setCurrentText(FENCE_NONE)
        self._type.setToolTip(
            "None omits a fence. Dwarf or concrete wall is drawn between the named pillars."
        )
        t.addWidget(t_lab, 0)
        t.addWidget(self._type, 1)
        outer.addWidget(type_row)

        self._pillars_wrap = QWidget()
        p = QVBoxLayout(self._pillars_wrap)
        p.setContentsMargins(0, 0, 0, 0)
        p.setSpacing(6)
        self._start = self._row(
            p,
            "Starting pillar for Wall fence:",
            "e.g. SP/RV 1001",
            tooltip="Must match a pillar number already entered above.",
        )
        self._end = self._row(
            p,
            "Ending pillar for Wall fence:",
            "e.g. SP/RV 1002",
            tooltip="Adjacent station that ends this fence run.",
        )
        outer.addWidget(self._pillars_wrap)
        self._pillars_wrap.setVisible(False)
        self._type.currentTextChanged.connect(self._on_type_changed)
        self.set_index(index)

    def _row(
        self,
        parent: QVBoxLayout,
        label: str,
        placeholder: str,
        tooltip: str = "",
    ) -> QLineEdit:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        lab = QLabel(label)
        lab.setObjectName("cadFieldLabel")
        lab.setMinimumWidth(210)
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        if tooltip:
            lab.setToolTip(tooltip)
            edit.setToolTip(tooltip)
        layout.addWidget(lab, 0)
        layout.addWidget(edit, 1)
        parent.addWidget(row)
        return edit

    def _on_type_changed(self, text: str) -> None:
        self._pillars_wrap.setVisible(text != FENCE_NONE)

    def set_index(self, index: int) -> None:
        self._title.setText(f"{ordinal_label(index + 1)} Wall fence")

    def set_plus_visible(self, visible: bool) -> None:
        self._plus.setVisible(bool(visible))

    def fence(self) -> WallFence:
        return WallFence(
            fence_type=self._type.currentText() or FENCE_NONE,
            start_pillar=self._start.text(),
            end_pillar=self._end.text(),
        )

    def set_fence(self, fence: WallFence) -> None:
        kind = fence.fence_type or FENCE_NONE
        if kind not in {FENCE_NONE, FENCE_DWARF, FENCE_CONCRETE}:
            kind = FENCE_NONE
        self._type.setCurrentText(kind)
        self._start.setText(fence.start_pillar or "")
        self._end.setText(fence.end_pillar or "")
        self._on_type_changed(kind)
