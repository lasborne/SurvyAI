"""Tests for access-road label/side detection on survey plan PDFs."""

from agent.pdf_survey_plan import (
    SurveyPlanExtraction,
    SurveyTraverseLeg,
    _format_access_road_spec,
    _merge_access_road_specs,
    _pair_from_access_road_spec,
    extract_access_road_title_from_text,
    infer_access_roads_from_text,
    normalize_access_road_title,
)


def test_normalize_access_road_title():
    assert normalize_access_road_title("ACCESS ROAD") == "ACCESS    ROAD"
    assert normalize_access_road_title("ACCESS CLOSE") == "ACCESS CLOSE"
    assert normalize_access_road_title("ACCESS/CLOSE") == "ACCESS CLOSE"
    assert normalize_access_road_title("") == "ACCESS    ROAD"


def test_extract_access_road_title_from_text():
    assert extract_access_road_title_from_text("label ACCESS ROAD beside plot") == "ACCESS    ROAD"
    assert extract_access_road_title_from_text("ACCESS CLOSE on shortest side") == "ACCESS CLOSE"
    assert extract_access_road_title_from_text("no road label here") == ""


def test_infer_access_road_without_pdf_returns_empty_not_guessed_side():
    """Without PDF geometry/labels, do not guess a road side."""
    extraction = SurveyPlanExtraction(
        pillar_numbers=[
            "SP/RV 4796",
            "SP/RV 4797",
            "SP/RV 4798",
            "SP/RV 4799",
            "SP/RV 4795",
        ],
        traverse_legs=[
            SurveyTraverseLeg(distance_m=30.70),
            SurveyTraverseLeg(distance_m=11.30),
            SurveyTraverseLeg(distance_m=9.50),
            SurveyTraverseLeg(distance_m=33.00),
            SurveyTraverseLeg(distance_m=21.10),
        ],
        scale_denom=500,
    )
    roads, title = infer_access_roads_from_text("ACCESS ROAD along boundary", extraction, pdf_path=None)
    assert roads == []
    assert title == ""


def test_format_and_merge_multiple_road_specs():
    pair_a = ("SC/BV 6015", "SC/BV 6012")
    pair_b = ("SC/BV 6013", "SC/BV 6014")
    spec_a = _format_access_road_spec(18.0, pair_a, title="ACCESS    ROAD")
    spec_b = _format_access_road_spec(7.0, pair_b, title="ACCESS CLOSE")
    assert "6015" in spec_a and "6012" in spec_a
    assert "ACCESS CLOSE" in spec_b
    merged = _merge_access_road_specs([spec_a], [spec_b])
    assert len(merged) == 2
    pairs = [_pair_from_access_road_spec(s) for s in merged]
    assert pairs[0] == pair_a
    assert pairs[1] == pair_b


def test_merge_deduplicates_same_boundary_side():
    pair = ("SC/Q 573", "SC/CK 2285")
    a = _format_access_road_spec(6.0, pair)
    b = _format_access_road_spec(8.0, pair, title="ACCESS CLOSE")
    merged = _merge_access_road_specs([a], [b])
    assert len(merged) == 1
    assert merged[0] == a


def test_parse_uchechukwu_three_access_roads_from_prompt():
    from agent.agent import _parse_access_road_specs_from_query

    q = (
        "Add an access of width 7m on the side of SC/CL 2453 and SC/BM 7161, "
        "and another access road of width 12m along side SC/BM 7161 and SC/BM 7160, "
        "and yet another road of width 5m on side SC/CL 2454 to SC/CL 2453"
    )
    roads = _parse_access_road_specs_from_query(q)
    assert len(roads) == 3
    assert "7m width on the side of SC/CL 2453 and SC/BM 7161" in roads
    assert "12m width on the side of SC/BM 7161 and SC/BM 7160" in roads
    assert "5m width on the side of SC/CL 2454 and SC/CL 2453" in roads
