"""JOB_271-style PDF replot: C.W.F. fences and metadata override phrasing."""

from agent.pdf_survey_plan import (
    SurveyPlanExtraction,
    SurveyTraverseLeg,
    apply_plan_overrides_to_extraction,
    build_cadastral_subprompt,
    enrich_extraction_coordinates,
    filter_user_facing_extraction_notes,
    infer_fences_from_boundary_text,
    parse_fence_specs_from_text,
    prepare_extraction_for_cadastral,
    resolve_plan_overrides_from_query,
    validate_extraction_for_replot,
    validate_subprompt_geometry,
)

JOB_271_SCOPE = (
    'Go to this pdf file - "C:\\Users\\USER\\Documents\\JOB_271.pdf", extract all the important '
    "survey plan details, and replot the plan and save strictly as 'JOB_271.dwg'. "
    "Buyer's name changes to 'DR. MADUELECHI CORNELIUS AKO', the date should be changed to "
    "29/06/2026. The plan number now is RV/1124/2026/042"
)

JOB_271_NOTES = (
    "C.W.F. explicitly labeled along boundaries SC/CJ 2140-SC/CJ 2141 and "
    "SC/CJ 2142-SC/CJ 2143. User-requested name/date/plan-number changes were ignored; "
    "extracted original visible plan details only."
)


def test_job_271_metadata_overrides():
    overrides = resolve_plan_overrides_from_query(JOB_271_SCOPE, scope_text=JOB_271_SCOPE)
    assert overrides.buyer_name == "DR. MADUELECHI CORNELIUS AKO"
    assert overrides.certification_date == "29-06-2026"
    assert overrides.plan_number == "RV/1124/2026/042"
    assert "buyer_name" in overrides.override_fields
    assert "certification_date" in overrides.override_fields
    assert "plan_number" in overrides.override_fields


def test_infer_fences_from_boundary_notes():
    specs = infer_fences_from_boundary_text(JOB_271_NOTES)
    assert len(specs) == 2
    assert all("Concrete wall fence" in s for s in specs)
    assert any("2140" in s and "2141" in s for s in specs)
    assert any("2142" in s and "2143" in s for s in specs)


def test_prepare_extraction_merges_note_fences():
    extraction = SurveyPlanExtraction(
        pillar_numbers=["SC/CJ 2140", "SC/CJ 2141", "SC/CJ 2142", "SC/CJ 2143"],
        notes=JOB_271_NOTES,
    )
    prepared = prepare_extraction_for_cadastral(extraction, JOB_271_NOTES)
    assert len(prepared.fences) == 2


def test_subprompt_includes_fences_and_overrides():
    base = SurveyPlanExtraction(
        buyer_name="MR. CHIMEZIE EMMANUEL KOSISOCHUKWU",
        plan_number="RV/OLD/001",
        pillar_numbers=["SC/CJ 2140", "SC/CJ 2141", "SC/CJ 2142", "SC/CJ 2143"],
        notes=JOB_271_NOTES,
        scale_denom=500,
    )
    prepared = prepare_extraction_for_cadastral(base, JOB_271_NOTES)
    overrides = resolve_plan_overrides_from_query(JOB_271_SCOPE, scope_text=JOB_271_SCOPE)
    merged = apply_plan_overrides_to_extraction(prepared, overrides)
    subprompt = build_cadastral_subprompt(
        merged,
        output_dwg_path=r"C:\Users\USER\Documents\JOB_271.dwg",
        certification_date=overrides.certification_date,
    )
    fences = parse_fence_specs_from_text(subprompt)
    assert len(fences) == 2
    assert "DR. MADUELECHI CORNELIUS AKO" in subprompt
    assert "RV/1124/2026/042" in subprompt


def test_filter_contradictory_override_notes():
    filtered = filter_user_facing_extraction_notes(
        JOB_271_NOTES,
        ["buyer_name", "plan_number", "certification_date"],
    )
    assert "ignored" not in filtered.lower()
    assert "user-requested" not in filtered.lower()
    assert "C.W.F." in filtered


def test_enrich_coordinates_from_spaced_notes():
    """Scanned plans often put UTM E/N only in vision notes with spaced thousands."""
    legs = [
        SurveyTraverseLeg(bearing_deg=52, bearing_min=10, distance_m=27.50),
        SurveyTraverseLeg(bearing_deg=136, bearing_min=5, distance_m=23.50),
        SurveyTraverseLeg(bearing_deg=248, bearing_min=7, distance_m=30.50),
        SurveyTraverseLeg(bearing_deg=319, bearing_min=38, distance_m=15.00),
    ]
    extraction = SurveyPlanExtraction(
        pillar_numbers=["SC/CJ 2140", "SC/CJ 2141", "SC/CJ 2142", "SC/CJ 2143"],
        traverse_legs=legs,
        anchor_northing=537935.100,
        notes=(
            "Easting 293 147.570mE and northing 537 935.100m.N are shown on separate grid lines "
            "intersecting at/through SC/CJ 2140."
        ),
        absolute_parcel_coords=[
            {"e": 0.0, "n": 537935.100},
            {"e": 27.5, "n": 537960.0},
            {"e": 50.0, "n": 537940.0},
            {"e": 15.0, "n": 537920.0},
        ],
    )
    enriched = enrich_extraction_coordinates(extraction, "")
    assert validate_extraction_for_replot(enriched) == []
    subprompt = build_cadastral_subprompt(
        enriched,
        output_dwg_path=r"C:\temp\plot.dwg",
    )
    assert validate_subprompt_geometry(subprompt) == []
    assert "293147.570mE" in subprompt.replace(" ", "")
