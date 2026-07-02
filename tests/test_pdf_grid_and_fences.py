"""Tests for PDF grid coordinate anchoring and CWF/DCWF fence detection."""

from agent.pdf_survey_plan import (
    SurveyPlanExtraction,
    SurveyTraverseLeg,
    _compute_absolute_parcel_coordinates,
    _compute_relative_traverse_vertices,
    _pick_primary_pillar_index,
    apply_pdf_grid_coordinates,
    build_cadastral_subprompt,
    extract_grid_coordinates_from_text,
    infer_fences_from_pdf,
    is_explicit_fence_label,
    _fence_kind_from_token,
)


def _job302_extraction() -> SurveyPlanExtraction:
    """JOB_302-style quadrilateral (bearings/distances from the reference PDF)."""
    pillars = ["SC/CM 2493", "SC/CM 2494", "SC/CW 2495", "SC/CW 3743"]
    legs = [
        SurveyTraverseLeg(bearing_deg=134, bearing_min=39, distance_m=9.50),
        SurveyTraverseLeg(bearing_deg=217, bearing_min=25, distance_m=25.93),
        SurveyTraverseLeg(bearing_deg=311, bearing_min=2, distance_m=8.90),
        SurveyTraverseLeg(bearing_deg=36, bearing_min=13, distance_m=26.50),
    ]
    return SurveyPlanExtraction(
        pillar_numbers=pillars,
        traverse_legs=legs,
        scale_denom=500,
    )


def test_extract_labelled_grid_coordinates_from_text():
    text = (
        "291807.909m.E\n536651.002m.N\n"
        "291792.267m.E\n536629.611m.N"
    )
    e, n, all_e, all_n = extract_grid_coordinates_from_text(text)
    assert e == 291807.909
    assert n == 536651.002
    assert 291792.267 in all_e


def test_anchor_traverse_to_pdf_grid_easting():
    ext = _job302_extraction()
    grid_e = 291807.909
    grid_e_pillar = "SC/CW 3743"

    abs_coords = _compute_absolute_parcel_coordinates(
        ext,
        grid_e=grid_e,
        grid_e_pillar=grid_e_pillar,
        grid_n=None,
        grid_n_pillar="",
    )
    assert abs_coords is not None
    idx_3743 = ext.pillar_numbers.index("SC/CW 3743")
    assert abs(abs_coords[idx_3743]["e"] - grid_e) < 0.001

    primary_idx = _pick_primary_pillar_index(abs_coords)
    assert ext.pillar_numbers[primary_idx] == "SC/CW 3743"
    assert abs(abs_coords[primary_idx]["e"] - grid_e) < 0.001


def test_apply_grid_without_pdf_uses_text_labels():
    ext = _job302_extraction()
    text = "291807.909m.E 536651.002m.N"
    apply_pdf_grid_coordinates(ext, None, text)
    assert ext.absolute_parcel_coords
    idx_3743 = ext.pillar_numbers.index("SC/CW 3743")
    assert abs(ext.absolute_parcel_coords[idx_3743]["e"] - 291807.909) < 0.001


def test_subprompt_emits_absolute_coordinate_pairs():
    ext = _job302_extraction()
    apply_pdf_grid_coordinates(ext, None, "291807.909m.E 536651.002m.N")
    prompt = build_cadastral_subprompt(ext, output_dwg_path=r"C:\out\JOB_302.dwg")
    assert "(291807.909mE," in prompt
    assert "bearing" not in prompt.split("coordinates for the points =")[1].split("\n")[0]


def test_fence_kind_detection():
    assert _fence_kind_from_token("C.W.F.")[0] == "CWF"
    assert _fence_kind_from_token("D.C.W.F.")[0] == "DCWF"
    assert _fence_kind_from_token("CWF") is not None


def test_infer_fences_from_text_context():
    ext = _job302_extraction()
    text = "C.W.F. on side of SC/CW 3743 and SC/CM 2493"
    fences = infer_fences_from_pdf(ext, combined_text=text)
    assert len(fences) == 1
    assert "Concrete wall fence" in fences[0]
    assert "3743" in fences[0]
    assert "2493" in fences[0]


def test_job298_style_full_anchor_at_pillar():
    """Both E and N at SC/Q 572 must anchor the full traverse (JOB_298 regression)."""
    pillars = ["SC/Q 573", "SC/CK 2285", "SC/CK 2286", "SC/Q 572"]
    legs = [
        SurveyTraverseLeg(bearing_deg=78, bearing_min=18, distance_m=18.10),
        SurveyTraverseLeg(bearing_deg=199, bearing_min=59, distance_m=30.10),
        SurveyTraverseLeg(bearing_deg=254, bearing_min=44, distance_m=18.20),
        SurveyTraverseLeg(bearing_deg=19, bearing_min=8, distance_m=31.30),
    ]
    ext = SurveyPlanExtraction(
        pillar_numbers=pillars,
        traverse_legs=legs,
        anchor_easting=292276.500,
        anchor_northing=536648.932,
        anchor_pillar="SC/Q 572",
    )
    coords = _compute_absolute_parcel_coordinates(
        ext,
        grid_e=292276.500,
        grid_e_pillar="SC/Q 572",
        grid_n=536648.932,
        grid_n_pillar="SC/Q 572",
    )
    assert coords is not None
    idx = pillars.index("SC/Q 572")
    assert abs(coords[idx]["e"] - 292276.500) < 0.001
    assert abs(coords[idx]["n"] - 536648.932) < 0.001


def test_explicit_fence_labels():
    assert is_explicit_fence_label("ACCESS ROAD") is None
    assert is_explicit_fence_label("C.W.F.")[0] == "CWF"
    assert is_explicit_fence_label("DCWF")[0] == "DCWF"
    assert is_explicit_fence_label("wall fence")[0] == "CWF"
    assert is_explicit_fence_label("Wall Fence")[0] == "CWF"
    assert is_explicit_fence_label("fence")[0] == "CWF"
    assert is_explicit_fence_label("FENCE")[0] == "CWF"


def test_infer_fences_from_fence_text():
    ext = _job302_extraction()
    text = "Fence on side of SC/CW 3743 and SC/CM 2493"
    fences = infer_fences_from_pdf(ext, combined_text=text)
    assert len(fences) == 1
    assert "Concrete wall fence" in fences[0]

    ext = _job302_extraction()
    text = "Wall Fence on side of SC/CW 3743 and SC/CM 2493"
    fences = infer_fences_from_pdf(ext, combined_text=text)
    assert len(fences) == 1
    assert "Concrete wall fence" in fences[0]


def test_infer_fences_empty_without_cwf_label():
    ext = _job302_extraction()
    text = "ACCESS ROAD on side of SC/CW 3743 and SC/CM 2493. Railway boundary."
    fences = infer_fences_from_pdf(ext, combined_text=text)
    assert fences == []
