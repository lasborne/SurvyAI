"""Tests for SC/BV pillar extraction and proactive PDF repair (JOB_300 regression)."""

from agent.pdf_survey_plan import (
    SurveyPlanExtraction,
    SurveyTraverseLeg,
    _normalize_pillar_id,
    _parse_pillar_token,
    finalize_survey_extraction,
    infer_fences_from_pdf,
    repair_survey_extraction_from_pdf,
    validate_extraction_for_replot,
    validate_subprompt_geometry,
    build_cadastral_subprompt,
)


def test_normalize_sc_bv_pillar():
    assert _normalize_pillar_id("SC/BV 6015") == "SC/BV 6015"
    assert _normalize_pillar_id("SCBV6015") == "SC/BV 6015"
    assert _normalize_pillar_id("sc/bv 6012") == "SC/BV 6012"


def test_parse_split_pillar_tokens():
    assert _parse_pillar_token("SC/BV", "6015") == "SC/BV 6015"
    assert _parse_pillar_token("SC/BV6015", "") == "SC/BV 6015"


def test_repair_fills_pillars_from_text():
    ext = SurveyPlanExtraction(
        pillar_numbers=[],
        traverse_legs=[
            SurveyTraverseLeg(bearing_deg=10, bearing_min=20, distance_m=30.0),
        ],
    )
    text = (
        "SC/BV 6015 SC/BV 6012 SC/BV 6013 SC/BV 6014 "
        "86° 59' 13.20m 140° 20' 26.50m"
    )
    repaired = repair_survey_extraction_from_pdf(ext, None, text)
    assert len(repaired.pillar_numbers) >= 3
    assert any("6015" in p for p in repaired.pillar_numbers)


def test_job300_style_finalize_and_validate():
    pillars = ["SC/BV 6015", "SC/BV 6012", "SC/BV 6013", "SC/BV 6014"]
    legs = [
        SurveyTraverseLeg(bearing_deg=86, bearing_min=59, distance_m=13.20),
        SurveyTraverseLeg(bearing_deg=140, bearing_min=20, distance_m=26.50),
        SurveyTraverseLeg(bearing_deg=249, bearing_min=28, distance_m=20.80),
        SurveyTraverseLeg(bearing_deg=338, bearing_min=38, distance_m=29.00),
    ]
    ext = SurveyPlanExtraction(
        pillar_numbers=pillars,
        traverse_legs=legs,
        anchor_easting=294231.196,
        anchor_northing=537609.561,
        anchor_pillar="SC/BV 6014",
        scale_denom=500,
    )
    text = (
        "294231.196m.E 537609.561m.N C.W.F. C.W.F. C.W.F. C.W.F. ACCESS ROAD"
    )
    finalized = finalize_survey_extraction(ext, text, plain_text=text, pdf_path=None)
    assert validate_extraction_for_replot(finalized) == []
    fences = infer_fences_from_pdf(finalized, combined_text=text)
    assert len(fences) == 4
    sub = build_cadastral_subprompt(finalized, output_dwg_path=r"C:\out\JOB_300.dwg")
    assert validate_subprompt_geometry(sub) == []
    assert "SC/BV 6015" in sub
    assert "294231.196" in sub
