"""Packaged default CAD survey-plan prompt and resolve helpers."""

from __future__ import annotations

SYSTEM_DEFAULT_CAD_PROMPT = (
    "Generate Buyer_Name.dwg in the same folder as this project with buyer name: "
    "Mr. Richyblue James Doe, location: Livingstone Chokogba Farmland, Chokota Etche, "
    "local government area: Etche Local Government Area, state: Rivers state, "
    "origin_crs: UTM Zone 32N, plan number: RV/0000/2026/001, "
    "date on the certification: 01/01/2026, Surveyor name: Surv. Robotics John Doe (mnis), "
    "Surveyor company and address: SURVYAI GEO-NET SERVICES LTD.\n"
    "00A Rumuokwurusi-Oil Mill Tower, Rumuokwurusi road, Port Harcourt, Rivers state, "
    "pillar numbers: SP/RV 1000, SP/RV 1001, SP/RV 1002, SP/RV 1003, "
    "coordinates for the point: (200200.400mE, 576000.100mN), "
    "with bearing: 59degrees 58 min, distance = 30.50m (first traverse leg); "
    "bearing: 154 deg 34min, dist = 15.25m (second traverse leg); "
    "bearing: 239d 50minutes, dist. = 30.50m (3rd traverse leg); "
    "bearing: 334deg 39', measured distance = 15.25m (for the final traverse leg). "
    "Add an access of width 6m on the side of SP/RV 1000 and SP/RV 1001; "
    "and another road 10m wide on side SP/RV 1002 to SP/RV 1003. "
    "Add 2 Concrete wall fences on the sides joining SP/RV 1001 to SP/RV 1002 to SP/RV 1003."
)


def resolve_active_cad_prompt(stored: str) -> str:
    """Return the active CAD prompt: user-defined if set, otherwise the system default."""
    text = (stored or "").strip()
    return text if text else SYSTEM_DEFAULT_CAD_PROMPT


def is_system_default_text(text: str) -> bool:
    """True when *text* matches the packaged system default (ignoring outer whitespace)."""
    return (text or "").strip() == SYSTEM_DEFAULT_CAD_PROMPT.strip()
