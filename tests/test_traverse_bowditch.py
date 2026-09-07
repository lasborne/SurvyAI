"""Bowditch / compass-rule intent and adjustment — no AutoCAD required."""

from __future__ import annotations

from agent.cadastral_compose import build_subprompt_from_coordinates_blob
from agent.cadastral_intent import (
    bowditch_instruction_for_subprompt,
    format_traverse_adjustment_chat_lines,
    user_requests_bowditch_adjustment,
)
from agent.pdf_survey_plan import extract_coordinates_blob_from_cadastral_query
from tools.traverse_bowditch import apply_bowditch_rule

# The field job that previously plotted bearing-only adjustment while claiming Bowditch.
_USER_PROMPT = (
    "Generate delete2.dwg in the same folder as this project with buyer name: Abigail Ufere Anyaoba, "
    "location: No. 7 Goke street, Akpajo, local government area: Eleme Local Government Area, "
    "state: Rivers state, origin_crs: UTM Zone 32N, plan number: OUM/6215/2026/008SP, "
    "date on the certification: 31/08/2026, Surveyor name: Surv. Okeke Uzochukwu Michael (mnis), "
    "Surveyor company and address: CARDINAL GEO-NET SERVICES LTD. "
    "Rumuokwurusi Civic centre, Rumuokwurusi road, Port Harcourt, Rivers state, "
    "pillar numbers: SP/RV 4810, SP/RV 4811, SP/RV 4812, SP/RV 4813, SP/RV 4814, SP/RV 4815, "
    "coordinates for the point: (288300.050mE, 533272.002mN), with bearing: 285degrees 57 min, "
    "distance = 8.60m (first traverse leg); bearing: 45 deg 00min, dist = 5.00m (second traverse leg); "
    "bearing: 90d 10minutes, dist. = 4.00m (3rd traverse leg); bearing: 10deg 39', "
    "measured distance = 1.50m (fourth traverse leg); bearing: 95deg 43', measured distance = 4.00m "
    "(fifth traverse leg); bearing: 226deg 04', measured distance = 6.50m (for the final traverse leg). "
    "Add an access of width 4m on the side of SP/RV 4810 and SP/RV 4811. "
    "Use Bowditch adjustment method to perform closure of traverse"
)

_FIELD_LEGS = [
    (285.0 + 57.0 / 60.0, 8.60),
    (45.0, 5.00),
    (90.0 + 10.0 / 60.0, 4.00),
    (10.0 + 39.0 / 60.0, 1.50),
    (95.0 + 43.0 / 60.0, 4.00),
    (226.0 + 4.0 / 60.0, 6.50),
]
_START_E, _START_N = 288300.050, 533272.002

# Distances-held bearing-adjustment labels from the failed delete2.dwg plot.
_BEARING_ONLY_LABELS_DEG = [
    285.0 + 57.0 / 60.0,
    52.0 + 58.0 / 60.0,
    93.0 + 49.0 / 60.0,
    17.0 + 50.0 / 60.0,
    98.0 + 50.0 / 60.0,
    213.0 + 32.0 / 60.0,
]


def test_bowditch_request_is_detected_in_varied_wording() -> None:
    positives = [
        _USER_PROMPT,
        "please close using the compass rule",
        "Apply Bowditch's rule if there is a misclose",
        "compass method for traverse closure",
        "can you run a compass adjustment on this traverse?",
        "use the bowdich method",  # common misspelling
    ]
    for text in positives:
        assert user_requests_bowditch_adjustment(text), text


def test_generic_adjust_or_close_is_not_bowditch() -> None:
    negatives = [
        "Generate plan.dwg ... coordinates for the point: (100mE, 200mN), bearing 10deg 00', distance=5m",
        "adjust the traverse and close it",
        "perform closure of traverse",
        "use bearing adjustment",
        "close the polygon",
        "do not use Bowditch; keep distances fixed",
        "don't apply the compass rule",
        "without Bowditch adjustment",
        "no compass method please",
        "this is not the compass bearing of the first line",
    ]
    for text in negatives:
        assert not user_requests_bowditch_adjustment(text), text


def test_pdf_forbid_clause_does_not_cancel_a_later_bowditch_ask() -> None:
    text = (
        "coordinates for the points = (100mE, 200mN); bearing 10deg 00 min, distance = 5m\n"
        "do not auto-adjust traverse\n"
        "Use Bowditch adjustment method to close the traverse."
    )
    assert user_requests_bowditch_adjustment(text)


def test_trimmed_coordinates_blob_drops_the_bowditch_sentence() -> None:
    blob = extract_coordinates_blob_from_cadastral_query(_USER_PROMPT)
    assert "288300.050" in blob
    assert "8.60" in blob
    assert "bowditch" not in blob.lower()
    # The plotter must therefore search the full prompt, not only this blob.
    assert user_requests_bowditch_adjustment(_USER_PROMPT)
    assert not user_requests_bowditch_adjustment(blob)


def test_composed_subprompt_keeps_an_explicit_bowditch_line() -> None:
    sub = build_subprompt_from_coordinates_blob(
        output_dwg="delete2.dwg",
        coordinates_blob="(288300.050mE, 533272.002mN), bearing 285deg 57', distance = 8.60m",
        buyer_name="Abigail Ufere Anyaoba",
        source_query=_USER_PROMPT,
    )
    assert "Bowditch" in sub
    assert bowditch_instruction_for_subprompt(_USER_PROMPT)
    assert not bowditch_instruction_for_subprompt("just plot the parcel")


def test_bowditch_closes_and_changes_both_bearings_and_distances() -> None:
    result = apply_bowditch_rule(
        _START_E,
        _START_N,
        [b for b, _d in _FIELD_LEGS],
        [d for _b, d in _FIELD_LEGS],
    )
    assert result["misclosure_m"] > 2.5
    verts = result["adjusted_vertices"]
    assert len(verts) == 6
    assert abs(verts[0]["e"] - _START_E) < 1e-9
    assert abs(verts[0]["n"] - _START_N) < 1e-9

    closed = result["adjusted_points_with_closure"]
    assert abs(closed[-1]["e"] - closed[0]["e"]) < 1e-6
    assert abs(closed[-1]["n"] - closed[0]["n"]) < 1e-6

    adj = result["adjusted_legs"]
    assert len(adj) == 6
    observed_d = [d for _b, d in _FIELD_LEGS]
    adjusted_d = [float(L["distance"]) for L in adj]
    # Compass rule must not hold field distances (that was the old bearing-only path).
    assert any(abs(a - o) > 0.01 for a, o in zip(adjusted_d, observed_d))
    assert sum(abs(x) for x in result["delta_distance_m"]) > 0.05
    assert any(abs(x) > 0.1 for x in result["delta_bearing_deg"])

    # Must not reproduce the distances-held AutoCAD labels from the failed plot.
    for i, (plotted, bow) in enumerate(zip(_BEARING_ONLY_LABELS_DEG, adj)):
        if i == 0:
            continue
        assert abs((bow["bearing_deg"] - plotted + 180.0) % 360.0 - 180.0) > 0.4, i


def test_already_closed_square_is_unchanged() -> None:
    result = apply_bowditch_rule(0.0, 0.0, [0.0, 90.0, 180.0, 270.0], [10.0, 10.0, 10.0, 10.0])
    assert result["misclosure_m"] < 1e-9
    for d in result["delta_distance_m"]:
        assert abs(d) < 1e-9
    for b in result["delta_bearing_deg"]:
        assert abs(b) < 1e-6


def test_chat_lines_name_the_method_that_actually_ran() -> None:
    bow_lines = format_traverse_adjustment_chat_lines(
        {
            "mode": "bearing_distance",
            "method": "bowditch",
            "applied": True,
            "misclosure_m": 2.712,
            "misclosure_e_m": -1.157,
            "misclosure_n_m": 2.453,
            "max_point_shift_m": 1.1,
        }
    )
    assert any("Bowditch (compass-rule)" in x for x in bow_lines)
    assert any("bearings and distances" in x for x in bow_lines)

    ba_lines = format_traverse_adjustment_chat_lines(
        {
            "mode": "bearing_distance",
            "method": "bearing_adjustment",
            "applied": True,
            "misclosure_m": 2.712,
            "misclosure_e_m": -1.157,
            "misclosure_n_m": 2.453,
            "max_point_shift_m": 2.7,
        }
    )
    assert any("Bearing adjustment" in x for x in ba_lines)
    assert not any("Bowditch (compass-rule)" in x for x in ba_lines)
