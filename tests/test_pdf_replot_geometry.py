"""Geometry validation regressions for PDF→DWG replot safety."""

from __future__ import annotations

import re
from unittest.mock import patch

from agent.pdf_survey_plan import (
    SurveyPlanExtraction,
    SurveyTraverseLeg,
    _assign_unique_edge_labels,
    _drop_stale_anchor_pillars,
    _extraction_geometry_is_consistent,
    _is_plausible_plan_number,
    _legs_agreeing_with_drawn_edges,
    _polygon_is_simple,
    _replacement_traverse_is_better,
    build_cadastral_subprompt,
    extract_plan_number_from_plan_text,
    extract_heuristics_from_layout_text,
    extract_user_requested_pillar_ids,
    is_incomplete_printed_pillar_label,
    repair_survey_extraction_from_pdf,
    restore_incomplete_printed_pillar_labels,
    retain_attested_pillar_ids,
    split_cadastral_pillar_label,
    traverse_misclosure_metrics,
    validate_extraction_for_replot,
    validate_extraction_topology,
    validate_subprompt_geometry,
)


def _legs(specs: list[tuple[float, float, float]], pillars: list[str]) -> list[SurveyTraverseLeg]:
    out = []
    for i, (bd, bm, dist) in enumerate(specs):
        out.append(
            SurveyTraverseLeg(
                from_pillar=pillars[i % len(pillars)],
                to_pillar=pillars[(i + 1) % len(pillars)],
                bearing_deg=bd,
                bearing_min=bm,
                distance_m=dist,
            )
        )
    return out


def _crossed_but_closing_legs() -> list[SurveyTraverseLeg]:
    """A bowtie: closes to zero yet the outline crosses itself.

    (0,0) -> (10,0) -> (0,10) -> (10,10) -> back. The two diagonals intersect, so
    this is not a parcel — but every side is a valid reading and the vector sum is
    exactly zero, so closure alone cannot tell it apart from a real quadrilateral.
    """
    pillars = ["SC/BE 7001", "SC/BE 7002", "SC/BE 7003", "SC/BE 7004"]
    return _legs(
        [(90, 0, 10.0), (315, 0, 14.1421), (90, 0, 10.0), (225, 0, 14.1421)], pillars
    )


def _concave_l_shape_legs() -> list[SurveyTraverseLeg]:
    """A legitimate concave parcel — must never be rejected as crossed."""
    pillars = [f"SC/BE 71{i:02d}" for i in range(6)]
    return _legs(
        [(90, 0, 20.0), (0, 0, 10.0), (270, 0, 10.0), (0, 0, 10.0), (270, 0, 10.0), (180, 0, 20.0)],
        pillars,
    )


def _closed_square_legs() -> list[SurveyTraverseLeg]:
    # 10m square: N, E, S, W
    pillars = ["A", "B", "C", "D"]
    specs = [(0, 0, 10.0), (90, 0, 10.0), (180, 0, 10.0), (270, 0, 10.0)]
    legs = []
    for i, (bd, bm, dist) in enumerate(specs):
        legs.append(
            SurveyTraverseLeg(
                from_pillar=pillars[i],
                to_pillar=pillars[(i + 1) % 4],
                bearing_deg=bd,
                bearing_min=bm,
                distance_m=dist,
            )
        )
    return legs


def _open_traverse_legs() -> list[SurveyTraverseLeg]:
    """Same tidy counts as a square, but the last side is 1.5 m short."""
    legs = _closed_square_legs()
    legs[-1].distance_m = 8.5
    return legs


def _extraction_with(legs: list[SurveyTraverseLeg]) -> SurveyPlanExtraction:
    pillars = ["SC/BE 6060", "SC/BE 6061", "SC/BE 6062", "SC/BE 6063"]
    for i, leg in enumerate(legs):
        leg.from_pillar = pillars[i % len(pillars)]
        leg.to_pillar = pillars[(i + 1) % len(pillars)]
    return SurveyPlanExtraction(
        pillar_numbers=pillars,
        traverse_legs=legs,
        source="layout_text",
    )


def test_consistency_requires_closure_not_just_matching_counts() -> None:
    assert _extraction_geometry_is_consistent(_extraction_with(_closed_square_legs())) is True
    assert _extraction_geometry_is_consistent(_extraction_with(_open_traverse_legs())) is False


def test_repair_runs_on_a_mis_paired_traverse_with_tidy_counts() -> None:
    """Matching counts used to skip repair entirely, stranding a bad traverse."""
    bad = _extraction_with(_open_traverse_legs())
    good = _closed_square_legs()

    with patch(
        "agent.pdf_survey_plan.extract_boundary_legs_from_pdf", return_value=good
    ) as from_pdf:
        out = repair_survey_extraction_from_pdf(bad, "plan.pdf", "")

    from_pdf.assert_called_once()
    assert _extraction_geometry_is_consistent(out) is True


def test_repair_leaves_a_well_closed_traverse_alone() -> None:
    closed = _extraction_with(_closed_square_legs())
    with patch("agent.pdf_survey_plan.extract_boundary_legs_from_pdf") as from_pdf:
        out = repair_survey_extraction_from_pdf(closed, "plan.pdf", "")
    from_pdf.assert_not_called()
    assert out.traverse_legs[-1].distance_m == 10.0


def test_replacement_must_improve_closure_to_be_accepted() -> None:
    closed = _closed_square_legs()
    open_legs = _open_traverse_legs()
    worse = _closed_square_legs()
    worse[-1].distance_m = 3.0

    # An unusable current traverse is always replaced.
    assert _replacement_traverse_is_better([], closed) is True
    # A tighter candidate wins over an open traverse.
    assert _replacement_traverse_is_better(open_legs, closed) is True
    # A looser candidate never wins.
    assert _replacement_traverse_is_better(open_legs, worse) is False
    # An already-acceptable traverse is never gambled away.
    assert _replacement_traverse_is_better(closed, worse) is False


def test_misclosure_is_blind_to_the_order_sides_are_joined_in() -> None:
    """Documents *why* a shape check is required, not just a tighter closure gate."""
    legs = _closed_square_legs()
    reordered = [legs[2], legs[0], legs[3], legs[1]]
    a = traverse_misclosure_metrics(legs)
    b = traverse_misclosure_metrics(reordered)
    assert abs(a["misclosure_m"] - b["misclosure_m"]) < 1e-9
    assert abs(a["perimeter_m"] - b["perimeter_m"]) < 1e-9


def test_crossed_outline_is_rejected_even_though_it_closes_perfectly() -> None:
    crossed = _crossed_but_closing_legs()
    metrics = traverse_misclosure_metrics(crossed)
    assert metrics["misclosure_m"] < 0.01  # closes to millimetres

    ext = SurveyPlanExtraction(
        pillar_numbers=["SC/BE 7001", "SC/BE 7002", "SC/BE 7003", "SC/BE 7004"],
        traverse_legs=crossed,
        source="pdf_vector",
    )
    issues = validate_extraction_topology(ext)
    assert any("crosses itself" in i for i in issues)


def test_concave_parcel_is_accepted() -> None:
    ext = SurveyPlanExtraction(
        pillar_numbers=[f"SC/BE 71{i:02d}" for i in range(6)],
        traverse_legs=_concave_l_shape_legs(),
        source="pdf_vector",
    )
    assert not any("crosses itself" in i for i in validate_extraction_topology(ext))


def test_polygon_simplicity_helper() -> None:
    square = [{"e": 0.0, "n": 0.0}, {"e": 10.0, "n": 0.0}, {"e": 10.0, "n": 10.0}, {"e": 0.0, "n": 10.0}]
    bowtie = [{"e": 0.0, "n": 0.0}, {"e": 10.0, "n": 0.0}, {"e": 0.0, "n": 10.0}, {"e": 10.0, "n": 10.0}]
    assert _polygon_is_simple(square) is True
    assert _polygon_is_simple(bowtie) is False
    # A triangle can never self-intersect.
    assert _polygon_is_simple(square[:3]) is True


def _edge(bearing_hint: float, len_pt: float) -> dict:
    return {"bearing_hint": bearing_hint, "len_pt": len_pt}


def test_label_must_point_along_the_side_it_was_matched_to() -> None:
    pillars = ["SC/BE 7001", "SC/BE 7002", "SC/BE 7003", "SC/BE 7004"]
    good = _legs([(90, 0, 25.0)], pillars)[0]
    wrong_way = _legs([(200, 0, 25.0)], pillars)[0]

    kept = _legs_agreeing_with_drawn_edges([(_edge(92.0, 100.0), good)])
    assert len(kept) == 1

    # 110 deg from the drawn side: impossible for a correct reading.
    kept = _legs_agreeing_with_drawn_edges([(_edge(92.0, 100.0), wrong_way)])
    assert kept == []


def test_distance_implying_a_different_scale_is_rejected() -> None:
    pillars = ["SC/BE 7001", "SC/BE 7002", "SC/BE 7003", "SC/BE 7004"]
    specs = [(90, 0, 25.0), (0, 0, 25.0), (270, 0, 25.0), (180, 0, 70.0)]
    legs = _legs(specs, pillars)
    # Three sides imply 0.25 m/pt; the fourth would need 0.70 m/pt.
    candidates = [
        (_edge(90.0, 100.0), legs[0]),
        (_edge(0.0, 100.0), legs[1]),
        (_edge(270.0, 100.0), legs[2]),
        (_edge(180.0, 100.0), legs[3]),
    ]
    kept = _legs_agreeing_with_drawn_edges(candidates)
    assert len(kept) == 3
    assert all(leg.distance_m == 25.0 for leg in kept)


def test_split_road_name_counts_as_one_road() -> None:
    """'ACCESS' / 'ROAD' on one rotated baseline is a single corridor, not two."""
    from agent.pdf_survey_plan import _merge_collinear_label_fragments

    # Two roads, each name broken across two lines: one reading nearly vertically,
    # one nearly horizontally (the shape of a real corner plan).
    fragments = [
        ("ACCESS    ROAD", (404.0, 469.5), (0.224, -0.975)),
        ("ACCESS    ROAD", (419.5, 402.3), (0.222, -0.975)),
        ("ACCESS    ROAD", (238.0, 352.8), (0.997, 0.077)),
        ("ACCESS    ROAD", (388.1, 364.3), (0.997, 0.078)),
    ]
    merged = _merge_collinear_label_fragments(fragments)
    assert len(merged) == 2


def test_two_roads_on_one_baseline_direction_are_not_merged_when_far_apart() -> None:
    from agent.pdf_survey_plan import _merge_collinear_label_fragments

    # Same reading direction but on opposite sides of the parcel: the across-baseline
    # offset is large, so these stay two roads.
    fragments = [
        ("ACCESS    ROAD", (100.0, 100.0), (1.0, 0.0)),
        ("ACCESS    ROAD", (140.0, 300.0), (1.0, 0.0)),
    ]
    assert len(_merge_collinear_label_fragments(fragments)) == 2


def test_different_titles_never_merge() -> None:
    from agent.pdf_survey_plan import _merge_collinear_label_fragments

    fragments = [
        ("ACCESS    ROAD", (100.0, 100.0), (1.0, 0.0)),
        ("ACCESS CLOSE", (140.0, 100.0), (1.0, 0.0)),
    ]
    assert len(_merge_collinear_label_fragments(fragments)) == 2


def test_road_is_never_offset_into_a_concave_parcel() -> None:
    """The re-entrant side of a U-shaped parcel: the centroid heuristic points inward."""
    from agent.agent import _outward_normal_for_edge, _point_in_parcel

    pts = [
        {"x": 0.0, "y": 0.0}, {"x": 30.0, "y": 0.0}, {"x": 30.0, "y": 30.0},
        {"x": 20.0, "y": 30.0}, {"x": 20.0, "y": 10.0}, {"x": 10.0, "y": 10.0},
        {"x": 10.0, "y": 30.0}, {"x": 0.0, "y": 30.0},
    ]
    p1, p2 = pts[4], pts[5]  # notch floor, (20,10) -> (10,10)

    # The parcel centroid falls in the notch, i.e. outside the parcel itself.
    cx = sum(p["x"] for p in pts) / len(pts)
    cy = sum(p["y"] for p in pts) / len(pts)
    assert not _point_in_parcel(pts, cx, cy)

    outx, outy = _outward_normal_for_edge(pts, p1, p2)
    midx, midy = (p1["x"] + p2["x"]) / 2.0, (p1["y"] + p2["y"]) / 2.0
    # Stepping along the returned normal must leave the parcel, not enter it.
    assert not _point_in_parcel(pts, midx + 0.5 * outx, midy + 0.5 * outy)
    assert _point_in_parcel(pts, midx - 0.5 * outx, midy - 0.5 * outy)


def test_outward_normal_on_a_convex_parcel_points_away() -> None:
    from agent.agent import _outward_normal_for_edge, _point_in_parcel

    pts = [
        {"x": 0.0, "y": 0.0}, {"x": 10.0, "y": 0.0},
        {"x": 10.0, "y": 10.0}, {"x": 0.0, "y": 10.0},
    ]
    for i in range(4):
        p1, p2 = pts[i], pts[(i + 1) % 4]
        outx, outy = _outward_normal_for_edge(pts, p1, p2)
        midx, midy = (p1["x"] + p2["x"]) / 2.0, (p1["y"] + p2["y"]) / 2.0
        assert not _point_in_parcel(pts, midx + 0.5 * outx, midy + 0.5 * outy)


def test_labels_are_assigned_one_to_one() -> None:
    edges = [
        {"mx": 0.0, "my": 0.0, "bearing_hint": 0.0, "len_pt": 20.0},
        {"mx": 10.0, "my": 0.0, "bearing_hint": 90.0, "len_pt": 20.0},
    ]
    # One shared nearby label + one far label — first edge gets the near one; second cannot reuse it.
    labels = [
        (0.5, 0.5, 9.6),
        (50.0, 50.0, 12.0),
    ]

    def score(edge, lab, d):
        return d

    assigned = _assign_unique_edge_labels(edges, labels, max_dist=80.0, score_fn=score)
    assert len(assigned) == 2
    assert assigned[0][2] == 9.6
    assert assigned[1][2] == 12.0
    # Re-run with only one label within range → only one assignment
    assigned2 = _assign_unique_edge_labels(
        edges, [(0.2, 0.2, 9.6)], max_dist=80.0, score_fn=score
    )
    assert len(assigned2) == 1
    assert 0 in assigned2


def test_gross_misclosure_rejected() -> None:
    legs = [
        SurveyTraverseLeg(from_pillar="A", to_pillar="B", bearing_deg=0, bearing_min=0, distance_m=10.0),
        SurveyTraverseLeg(from_pillar="B", to_pillar="C", bearing_deg=90, bearing_min=0, distance_m=10.0),
        SurveyTraverseLeg(from_pillar="C", to_pillar="D", bearing_deg=180, bearing_min=0, distance_m=10.0),
        # Intentionally wrong: should be 270°/10m; reuse-like bad side
        SurveyTraverseLeg(from_pillar="D", to_pillar="A", bearing_deg=90, bearing_min=0, distance_m=10.0),
    ]
    metrics = traverse_misclosure_metrics(legs)
    assert metrics["misclosure_m"] > 1.0
    ext = SurveyPlanExtraction(
        pillar_numbers=["A", "B", "C", "D"],
        traverse_legs=legs,
        anchor_easting=500000.0,
        anchor_northing=1000000.0,
        area_sq_m=100.0,
        source="test",
    )
    issues = validate_extraction_for_replot(ext)
    assert any("misclosure" in i.lower() for i in issues)


def test_valid_small_closure_supported() -> None:
    legs = _closed_square_legs()
    metrics = traverse_misclosure_metrics(legs)
    assert metrics["misclosure_m"] < 0.01
    ext = SurveyPlanExtraction(
        pillar_numbers=["A", "B", "C", "D"],
        traverse_legs=legs,
        anchor_easting=500000.0,
        anchor_northing=1000000.0,
        area_sq_m=100.0,
        source="test",
    )
    issues = validate_extraction_for_replot(ext)
    assert not any("misclosure" in i.lower() for i in issues)


def test_printed_area_mismatch_blocks_plot() -> None:
    legs = _closed_square_legs()  # ~100 sq m
    ext = SurveyPlanExtraction(
        pillar_numbers=["A", "B", "C", "D"],
        traverse_legs=legs,
        anchor_easting=500000.0,
        anchor_northing=1000000.0,
        area_sq_m=500.0,  # printed disagrees badly
        source="test",
    )
    issues = validate_extraction_for_replot(ext)
    assert any("area" in i.lower() for i in issues)


def test_pdf_subprompt_forbids_auto_adjust() -> None:
    legs = _closed_square_legs()
    ext = SurveyPlanExtraction(
        pillar_numbers=["A", "B", "C", "D"],
        traverse_legs=legs,
        anchor_easting=500000.0,
        anchor_northing=1000000.0,
        source="layout_text",
    )
    prompt = build_cadastral_subprompt(ext, output_dwg_path=r"C:\tmp\det3.dwg")
    assert "do not auto-adjust traverse" in prompt.lower()
    assert validate_subprompt_geometry(prompt) == []


def test_subprompt_geometry_accepts_bearing_only_and_singular_point() -> None:
    legs = _closed_square_legs()
    ext = SurveyPlanExtraction(
        pillar_numbers=["SC/BE 6060", "SC/BE 6061", "SC/BE 6062", "SC/BE 6063"],
        traverse_legs=legs,
        source="layout_text",
    )
    prompt = build_cadastral_subprompt(ext, output_dwg_path=r"C:\tmp\det3.dwg")
    # No UTM pair on the sheet — traverse legs must still count as coordinates.
    issues = validate_subprompt_geometry(prompt)
    assert issues == []
    swapped = re.sub(
        r"coordinates for the points\s*=",
        "coordinates for the point =",
        prompt,
        count=1,
        flags=re.IGNORECASE,
    )
    assert validate_subprompt_geometry(swapped) == []


def test_plan_number_rejects_neighbouring_surv_token() -> None:
    assert _is_plausible_plan_number("SURV") is False
    assert _is_plausible_plan_number("MNIS") is False
    assert extract_plan_number_from_plan_text("PLAN NO.\nSURV. O.R. EDE\nRV/1124/2026/020") == (
        "RV/1124/2026/020"
    )
    assert extract_plan_number_from_plan_text("PLAN NO.\nRV11242026020") == "RV/1124/2026/020"


def test_printed_title_block_fills_gaps_the_layout_regex_misses() -> None:
    text = """
PLAN SHEWING LANDED PROPERTY
OF
MR. CHIBUIKE EGBOLUCHE
AND
MR. CHUKWUDEREA NWA-DAVID
AT
SUNSHINE ESTATE PHASE 3
ETCHE LOCAL GOVERNMENT AREA
RIVERS STATE
AREA:- 437.608 SQ. MTRS.
PLAN NO.
RV/1124/2026/020
SURV. O.R. EDE (MNIS)
NO. 7B WOJI ESTATE ROAD WOJI, PORT HARCOURT,
RIVERS STATE
MADE BY ME ON 20-08-2026
"""
    ext = extract_heuristics_from_layout_text(text)
    assert "CHUKWUDEREA" in (ext.buyer_name or "").upper()
    assert (ext.lga or "").upper() == "ETCHE"
    assert (ext.state or "").upper() == "RIVERS"
    assert abs(float(ext.area_sq_m) - 437.608) < 1e-6
    assert ext.plan_number == "RV/1124/2026/020"
    assert ext.certification_date == "20-08-2026"
    assert "WOJI" in (ext.surveyor_address or "").upper()
    prompt = build_cadastral_subprompt(ext, output_dwg_path=r"C:\tmp\det6.dwg")
    assert "437.608" in prompt
    assert "RV/1124/2026/020" in prompt


def test_heuristic_crs_requires_explicit_utm_zone() -> None:
    invented = extract_heuristics_from_layout_text(
        "GRID IN UTM\nZONE 32 ticks on the border\nRIVERS STATE"
    )
    assert "UTM ZONE 32" not in (invented.origin_crs or "").upper()
    z31 = extract_heuristics_from_layout_text("TITLE BLOCK\nUTM ZONE 31N\nRIVERS STATE")
    assert z31.origin_crs.upper().replace(" ", "") == "UTMZONE31N"


def test_stale_anchor_pillar_is_dropped_after_geometry_repair() -> None:
    ext = SurveyPlanExtraction(
        pillar_numbers=["SC/CJ 2436", "SC/CJ 2437", "SC/CJ 2438", "SC/CJ 2439"],
        anchor_pillar="SC/CJ 100",
        grid_easting_pillar="SC/CJ 542305",
        source="layout_heuristic",
    )
    _drop_stale_anchor_pillars(ext)
    assert ext.anchor_pillar == ""
    assert ext.grid_easting_pillar == ""


def test_grid_origin_is_axis_intersection_not_lowest_pillar_number() -> None:
    """Easting on a north–south tick, northing on an east–west tick: they meet at SW."""
    from agent.pdf_survey_plan import _grid_origin_from_en_labels, _match_grid_coordinate_block

    positions = {
        "SC/CJ 2436": (232.0, 324.0),
        "SC/CJ 2437": (370.0, 322.0),
        "SC/CJ 2438": (338.0, 504.0),
        "SC/CJ 2439": (195.0, 489.0),
    }
    pillars = list(positions.keys())
    e_labels = [(290747.931, 236.4, 621.6)]
    n_labels = [(542305.462, 504.3, 492.0)]
    ev, nv, pillar = _grid_origin_from_en_labels(e_labels, n_labels, positions, pillars)
    assert pillar == "SC/CJ 2439"
    assert ev == 290747.931
    assert nv == 542305.462
    # A leftover "first pillar / lowest number" hint must not steal the station.
    _be, _bn, hinted = _match_grid_coordinate_block(
        e_labels, n_labels, positions, pillars, anchor_hint="SC/CJ 2436"
    )
    assert hinted == "SC/CJ 2439"


def test_printed_en_stay_on_the_grid_station_not_the_first_listed_pillar() -> None:
    from agent.pdf_survey_plan import _compute_absolute_parcel_coordinates

    pillars = ["SC/CJ 2436", "SC/CJ 2437", "SC/CJ 2438", "SC/CJ 2439"]
    legs = [
        SurveyTraverseLeg(from_pillar=pillars[0], to_pillar=pillars[1], bearing_deg=100, bearing_min=44, distance_m=14.00),
        SurveyTraverseLeg(from_pillar=pillars[1], to_pillar=pillars[2], bearing_deg=189, bearing_min=35, distance_m=30.50),
        SurveyTraverseLeg(from_pillar=pillars[2], to_pillar=pillars[3], bearing_deg=280, bearing_min=44, distance_m=14.70),
        SurveyTraverseLeg(from_pillar=pillars[3], to_pillar=pillars[0], bearing_deg=10, bearing_min=54, distance_m=30.50),
    ]
    ext = SurveyPlanExtraction(
        pillar_numbers=pillars,
        traverse_legs=legs,
        # Poison: list order / lowest number, which is NOT the origin on the sheet.
        anchor_pillar="SC/CJ 2436",
        anchor_easting=290747.931,
        anchor_northing=542305.462,
        source="layout_heuristic",
    )
    abs_coords = _compute_absolute_parcel_coordinates(
        ext,
        grid_e=290747.931,
        grid_e_pillar="SC/CJ 2439",
        grid_n=542305.462,
        grid_n_pillar="SC/CJ 2439",
    )
    assert abs_coords is not None
    origin = abs_coords[3]
    assert abs(origin["e"] - 290747.931) < 1e-6
    assert abs(origin["n"] - 542305.462) < 1e-6
    # 2436 is north of 2439 along 010°54' — it must not keep the printed pair.
    assert abs(abs_coords[0]["e"] - 290747.931) > 1.0


def test_retain_attested_drops_invented_sequence_and_keeps_slots() -> None:
    candidates = ["SC/CJ 3934", "SC/CJ 3935", "SC/CJ 3936", "SC/CJ 3937"]
    attested = ["SC/CJ 3934", "SC/CJ 3937"]
    kept = retain_attested_pillar_ids(candidates, attested)
    assert kept == ["SC/CJ 3934", "", "", "SC/CJ 3937"]


def test_retain_attested_keeps_user_requested_ids() -> None:
    candidates = ["SC/CJ 3934", "SC/CJ 3935", "SC/CJ 3936", "SC/CJ 3937"]
    kept = retain_attested_pillar_ids(
        candidates,
        attested=["SC/CJ 3934"],
        user_requested=["SC/CJ 3935", "SC/CJ 3936"],
    )
    assert kept[0] == "SC/CJ 3934"
    assert kept[1] == "SC/CJ 3935"
    assert kept[2] == "SC/CJ 3936"
    assert kept[3] == ""


def test_retain_attested_passthrough_when_plan_has_no_printed_ids() -> None:
    candidates = ["SC/CJ 3934", "SC/CJ 3935", "SC/CJ 3936", "SC/CJ 3937"]
    kept = retain_attested_pillar_ids(candidates, attested=[], user_requested=[])
    assert kept == candidates


def test_retain_attested_keeps_a_fully_printed_ring() -> None:
    ring = ["SC/CJ 3934", "SC/CJ 3935", "SC/CJ 3936", "SC/CJ 3937"]
    assert retain_attested_pillar_ids(ring, ring) == ring


def test_user_prompt_can_name_pillar_ids_absent_from_the_plan() -> None:
    ids = extract_user_requested_pillar_ids(
        "replot this plan and set pillar numbers: SC/CJ 3935, SC/CJ 3936"
    )
    assert "SC/CJ 3935" in ids
    assert "SC/CJ 3936" in ids


def test_validate_allows_fewer_printed_pillars_than_sides() -> None:
    pillars = ["SC/CJ 3934", "", "", "SC/CJ 3937"]
    specs = [(0, 0, 10.0), (90, 0, 10.0), (180, 0, 10.0), (270, 0, 10.0)]
    legs = []
    for i, (bd, bm, dist) in enumerate(specs):
        legs.append(
            SurveyTraverseLeg(
                from_pillar=pillars[i] or f"P{i + 1}",
                to_pillar=pillars[(i + 1) % 4] or f"P{(i + 1) % 4 + 1}",
                bearing_deg=bd,
                bearing_min=bm,
                distance_m=dist,
            )
        )
    ext = SurveyPlanExtraction(
        pillar_numbers=pillars,
        traverse_legs=legs,
        anchor_easting=500000.0,
        anchor_northing=1000000.0,
        area_sq_m=100.0,
        source="test",
    )
    issues = validate_extraction_for_replot(ext)
    assert not any("exceeds" in i.lower() for i in issues)
    assert not any("fewer than three pillar" in i.lower() for i in issues)
    assert not any("does not match traverse legs" in i.lower() for i in issues)


def test_validate_still_rejects_more_pillars_than_sides() -> None:
    ring = ["SC/BE 7001", "SC/BE 7002", "SC/BE 7003", "SC/BE 7004"]
    legs = _legs(
        [(0, 0, 10.0), (90, 0, 10.0), (180, 0, 10.0), (270, 0, 10.0)],
        ring,
    )
    ext = SurveyPlanExtraction(
        pillar_numbers=ring + ["SC/BE 7005"],
        traverse_legs=legs,
        anchor_easting=500000.0,
        anchor_northing=1000000.0,
        area_sq_m=100.0,
        source="test",
    )
    issues = validate_extraction_for_replot(ext)
    assert any("exceeds" in i.lower() for i in issues)


def test_incomplete_printed_pillar_is_split_as_seen() -> None:
    assert is_incomplete_printed_pillar_label("SC/.. ......") is True
    assert is_incomplete_printed_pillar_label("SC/CJ 3934") is False
    split = split_cadastral_pillar_label("SC/.. ......")
    assert split == {"prefix": "SC/..", "number": "......"}
    assert split_cadastral_pillar_label("SC/CJ 3934") == {
        "prefix": "SC/CJ",
        "number": "3934",
    }


def test_retain_keeps_incomplete_printed_labels() -> None:
    candidates = ["SC/CJ 3934", "SC/CJ 3935", "SC/.. ......", "SC/.. ......"]
    kept = retain_attested_pillar_ids(candidates, attested=["SC/CJ 3934", "SC/CJ 3935"])
    assert kept[0] == "SC/CJ 3934"
    assert kept[1] == "SC/CJ 3935"
    assert kept[2] == "SC/.. ......"
    assert kept[3] == "SC/.. ......"


def test_restore_incomplete_from_notes_when_omitted() -> None:
    legs = _legs(
        [(0, 0, 10.0), (90, 0, 10.0), (180, 0, 10.0), (270, 0, 10.0)],
        ["SC/CJ 3934", "SC/CJ 3935", "P3", "P4"],
    )
    ext = SurveyPlanExtraction(
        pillar_numbers=["SC/CJ 3934", "SC/CJ 3935"],
        traverse_legs=legs,
        anchor_easting=500000.0,
        anchor_northing=1000000.0,
        area_sq_m=100.0,
        notes=(
            "Two lower-corner pillar labels are printed only as incomplete "
            "'SC/.. .....' text and are therefore omitted."
        ),
        source="test",
    )
    out = restore_incomplete_printed_pillar_labels(ext)
    assert out.pillar_numbers[0] == "SC/CJ 3934"
    assert out.pillar_numbers[1] == "SC/CJ 3935"
    assert len(out.pillar_numbers) == 4
    assert all(is_incomplete_printed_pillar_label(p) for p in out.pillar_numbers[2:])
    assert "transcribed as printed" in (out.notes or "").lower()
    assert "therefore omitted" not in (out.notes or "").lower()


def test_restore_fills_empty_slots_left_by_attested_filter() -> None:
    legs = _legs(
        [(0, 0, 10.0), (90, 0, 10.0), (180, 0, 10.0), (270, 0, 10.0)],
        ["SC/CJ 3934", "SC/CJ 3935", "P3", "P4"],
    )
    ext = SurveyPlanExtraction(
        pillar_numbers=["SC/CJ 3934", "SC/CJ 3935", "", ""],
        traverse_legs=legs,
        anchor_easting=500000.0,
        anchor_northing=1000000.0,
        area_sq_m=100.0,
        notes="Two lower-corner pillar labels are printed only as incomplete 'SC/.. ......'.",
        source="test",
    )
    out = restore_incomplete_printed_pillar_labels(ext)
    assert out.pillar_numbers == [
        "SC/CJ 3934",
        "SC/CJ 3935",
        "SC/.. ......",
        "SC/.. ......",
    ]


def test_clustered_multi_parcel_text_scales_caps() -> None:
    from agent.excel_cadastral import clustered_multi_parcel_text_scales

    assert clustered_multi_parcel_text_scales(
        parcel_count=1, chosen_denom=5000, multi_parcel=False
    ) == (1.0, 1.0)
    assert clustered_multi_parcel_text_scales(
        parcel_count=2, chosen_denom=500, multi_parcel=True
    ) == (1.0, 1.0)
    assert clustered_multi_parcel_text_scales(
        parcel_count=17, chosen_denom=5000, layout_span_m=700.0, multi_parcel=True
    ) == (0.75, 0.80)
    # 25% of conventional 12 at 1:5000 is 9.
    assert abs(12.0 * 0.75 - 9.0) < 1e-9


def test_shared_reverse_boundary_is_one_edge() -> None:
    import math

    from agent.excel_cadastral import (
        build_multi_parcel_layout_draw_ops,
        is_shared_or_reverse_traverse_edge,
    )

    # 186°54' / 53.88 m and its reverse 6°54' / 53.88 m with centimetre drift.
    az = 186.0 + 54.0 / 60.0
    dist = 53.88
    dx = dist * math.sin(math.radians(az))
    dy = dist * math.cos(math.radians(az))
    assert is_shared_or_reverse_traverse_edge(
        0.0,
        0.0,
        dx,
        dy,
        dx + 0.12,
        dy - 0.09,
        0.08,
        -0.07,
    )
    # Parallel sides of a 5 m corridor must not collapse.
    assert not is_shared_or_reverse_traverse_edge(
        0.0, 0.0, 50.0, 0.0,
        0.0, 5.0, 50.0, 5.0,
    )
    ops = build_multi_parcel_layout_draw_ops(
        [
            {
                "label": "AMADI FAMILY (A)",
                "points": [
                    {"x": 0.0, "y": 0.0},
                    {"x": 53.88, "y": 0.0},
                    {"x": 53.88, "y": 40.0},
                    {"x": 0.0, "y": 40.0},
                ],
            },
            {
                "label": "OKACHI FAMILY (B)",
                "points": [
                    {"x": 53.97, "y": 0.08},
                    {"x": 53.97, "y": 40.06},
                    {"x": 90.0, "y": 40.0},
                    {"x": 90.0, "y": 0.0},
                ],
            },
        ]
    )
    assert ops["edge_count"] == 7


def test_source_owner_title_rows_are_kept_as_written() -> None:
    from agent.excel_cadastral import parse_family_parcels_from_rows

    parsed = parse_family_parcels_from_rows(
        [
            ["AMADI FAMILY", None, None],
            [500000.0, 400000.0, "JI/OD 001"],
            [500020.0, 400000.0, "JI/OD 002"],
            [500020.0, 400020.0, "JI/OD 003"],
            [None, None, None],
            ["NO FAMILY NAME", None, None],
            [500020.0, 400000.0, "JI/OD 002"],
            [500040.0, 400000.0, "JI/OD 004"],
            [500040.0, 400020.0, "JI/OD 005"],
        ]
    )
    assert parsed["success"] is True
    parcels = parsed["parcels"]
    assert len(parcels) == 2
    assert parcels[0].labeled_name == "AMADI FAMILY (A)"
    assert parcels[1].labeled_name == "NO FAMILY NAME (B)"
    assert len(parcels[1].points) == 3
