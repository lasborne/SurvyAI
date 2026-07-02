"""Cadastral prompt parsing: metadata boundaries, coordinates, named pillars."""

import re

from agent.pdf_survey_plan import (
    CADASTRAL_FIELD_BOUNDARY,
    extract_coordinates_blob_from_cadastral_query,
)

UCHECHUKWU_BLOCK = (
    "Generate 'Uchechukwu_Njoku.dwg' in the same folder as this project with buyer name: Chief Uchechukwu Njoku, "
    "location: Egbelu Umuokolobe (camp V), Oyigbo, local government area: Oyigbo Local Government Area, "
    "state: Rivers state, origin_crs: UTM Zone 32N, plan number: RV/1124/2026/029, "
    "date on the certification: 04/05/2026, Surveyor name: Surv. O.R. Ede (mnis), "
    "Surveyor company and address: No. 7B Woji Estate Road Woji, Port Harcourt, Rivers state, "
    "pillar numbers: SC/CL 2453, SC/BM 7161, SC/BM 7160, SC/CL 2454, "
    "coordinates for the point SC/CL 2453: (292001.563mE, 537081.323mN), with bearing: 159degrees 6 min, "
    "distance = 9.90m (first traverse leg); bearing: 249 deg 26min, dist = 23.65m (second traverse leg); "
    "bearing: 339d 23minutes, dist. = 10.30m (3rd traverse leg); bearing: 69deg 26', measured distance = 23.40m "
    "(for the final traverse leg). Add an access of width 7m on the side of SC/CL 2453 and SC/BM 7161"
)

_COORDINATES_FOR_STOP = r"coordinates\s+for\s+the\s+point(?:s)?\s+(?:[^\n:=]+?\s*)?[:=]"


def _capture(pattern: str, text: str) -> str:
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    assert m, f"pattern did not match: {pattern[:60]}"
    return (m.group(1) or "").strip()


def test_uchechukwu_named_pillar_coordinates_and_pillars():
    blob = extract_coordinates_blob_from_cadastral_query(UCHECHUKWU_BLOCK)
    assert blob.startswith("(292001.563mE, 537081.323mN)")
    assert "bearing" in blob.lower()
    assert "9.90m" in blob

    m = re.search(
        rf"pillar\s+numbers\s*[:=]\s*(.*?)(?={_COORDINATES_FOR_STOP}|$)",
        UCHECHUKWU_BLOCK,
        re.I | re.S,
    )
    assert m
    raw = m.group(1).strip().rstrip(",").strip()
    assert raw == "SC/CL 2453, SC/BM 7161, SC/BM 7160, SC/CL 2454"

    # Traverse legs parse from blob
    legs = re.findall(
        r"\bbearing\b\s*(?:(?:=|:|-)|\bis\b)?\s*"
        r"(\d{1,3})\s*(?:deg|degree|degrees|°|d)\s*"
        r"([0-5]?\d)\s*(?:min|mins|minute|minutes|['’])"
        r"(?:[^0-9]{0,80}?)"
        r"(?:distance|dist\.?|measured\s+distance)\s*(?:=|is|:)?\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:m)?\b",
        blob,
        re.I | re.S,
    )
    assert len(legs) == 4
    assert float(legs[0][2]) == 9.90


def test_comma_separated_metadata_fields():
    text = (
        "location: Umuakuru Farmland, Igbo Etche, local government area: Etche Local Government Area, "
        "state: Rivers state, origin_crs: UTM Zone 32N, plan number: RV/1124/2026/027"
    )
    boundary = CADASTRAL_FIELD_BOUNDARY
    assert _capture(rf"location\s*[:=]\s*(.+?){boundary}", text) == "Umuakuru Farmland, Igbo Etche"
    assert _capture(rf"state\s*[:=]\s*(.+?){boundary}", text) == "Rivers state"
