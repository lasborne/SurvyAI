"""Tests for natural-language plan field overrides on PDF replot prompts."""

from agent.pdf_survey_plan import (
    SurveyPlanExtraction,
    apply_plan_overrides_to_extraction,
    resolve_buyer_name_from_query,
    resolve_plan_metadata_overrides_from_query,
    resolve_plan_overrides_from_query,
)

JOB_289_SCOPE = (
    'Given this Survey plan (pdf version) - "C:\\Users\\USER\\Documents\\JOB_289.pdf", '
    "extract all the important plan details, and replot the plan and save strictly as "
    "'JOB_289.dwg'. Buyer's name should now be 'CAPT. ADOLPHUS NWAGBARA' and "
    "the date should now be 24/06/2026."
)


def test_job_289_buyer_name_natural_language():
    assert (
        resolve_buyer_name_from_query(JOB_289_SCOPE, scope_text=JOB_289_SCOPE)
        == "CAPT. ADOLPHUS NWAGBARA"
    )


def test_job_289_combined_metadata_overrides():
    overrides = resolve_plan_metadata_overrides_from_query(
        JOB_289_SCOPE,
        scope_text=JOB_289_SCOPE,
    )
    assert overrides.buyer_name == "CAPT. ADOLPHUS NWAGBARA"
    assert overrides.certification_date == "24-06-2026"


def test_buyer_name_colon_form_still_works():
    scope = "Generate 'Plot_A.dwg' with buyer name: Chief Ada Obi, location: Port Harcourt"
    assert resolve_buyer_name_from_query(scope, scope_text=scope) == "Chief Ada Obi"


def test_surveyor_and_plan_number_natural_language():
    scope = (
        "Replot JOB_300.pdf as JOB_300.dwg. Plan number should now be RV/1124/2026/099. "
        "Surveyor name should now be 'Surv. A. B. Okoro (mnis)'."
    )
    overrides = resolve_plan_overrides_from_query(scope, scope_text=scope)
    assert overrides.plan_number == "RV/1124/2026/099"
    assert overrides.surveyor_name == "Surv. A. B. Okoro (mnis)"
    assert "plan_number" in overrides.override_fields
    assert "surveyor_name" in overrides.override_fields


def test_apply_overrides_on_extraction():
    base = SurveyPlanExtraction(
        buyer_name="OLD OWNER",
        plan_number="RV/OLD/001",
        pillar_numbers=["SC/KK 7323", "SC/KK 7326", "SC/KK 7324", "SC/KK 7327"],
    )
    overrides = resolve_plan_overrides_from_query(
        "Change buyer name to 'NEW OWNER' and plan number to RV/NEW/002",
        scope_text="Change buyer name to 'NEW OWNER' and plan number to RV/NEW/002",
    )
    merged = apply_plan_overrides_to_extraction(base, overrides)
    assert merged.buyer_name == "NEW OWNER"
    assert merged.plan_number == "RV/NEW/002"
    assert merged.pillar_numbers == base.pillar_numbers
