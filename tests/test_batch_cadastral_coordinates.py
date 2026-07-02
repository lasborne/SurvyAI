"""Batch cadastral coordinate extraction — varied prompt phrasing from real user prompts."""

from agent.pdf_survey_plan import (
    extract_coordinates_blob_from_cadastral_query,
    parse_fence_specs_from_text,
)

CHECK20_COORD = (
    "coordinates for the point: (292080.160mE, 537258.052mN), with bearing: 60degrees 59 min, "
    "distance = 24.50m (first traverse leg); bearing = 155 deg 34min, dist = 12.80m"
)

CHECK17_COORD = (
    "coordinates for the point: 291200.165mE, 537230.450mN, with bearing: 60degrees 30 min, "
    "distance = 152.50m (first traverse leg); bearing: 162 deg 34min, dist = 69.9m"
)

CHECK18_COORD = (
    "coordinates for the points = (291254.239mE, 537189.450mN), (291309.06mE, 537207.811mN), "
    "(291305.551mE, 537172.998mN), and (291249.528mE, 537172.04mN)"
)

CHECK20_FENCE = (
    "Add 3 Concrete wall fences on the sides joining SC/BE 6060 to SC/BG 1665 to SC/BE 3041 to SC/BE 6059"
)


def test_coordinates_for_the_point_colon_with_parens():
    blob = extract_coordinates_blob_from_cadastral_query(CHECK20_COORD)
    assert blob.startswith("(292080.160mE")
    assert "bearing" in blob.lower()
    assert "24.50m" in blob


def test_coordinates_for_the_point_colon_without_parens():
    blob = extract_coordinates_blob_from_cadastral_query(CHECK17_COORD)
    assert blob.startswith("291200.165mE")
    assert "537230.450mN" in blob
    assert "bearing" in blob.lower()


def test_coordinates_for_the_points_equals_still_works():
    blob = extract_coordinates_blob_from_cadastral_query(CHECK18_COORD)
    assert "(291254.239mE" in blob
    assert "(291249.528mE" in blob


def test_fence_chain_expands_to_per_leg_specs():
    fences = parse_fence_specs_from_text(CHECK20_FENCE)
    assert len(fences) == 3
    specs = " ".join(f["spec"] for f in fences)
    assert "6060" in specs and "6059" in specs
