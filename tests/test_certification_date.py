"""Tests for certification date resolution from user prompts."""

from datetime import datetime, timedelta, timezone

from agent.pdf_survey_plan import (
    _add_months,
    _parse_relative_offset_phrase,
    resolve_certification_date_from_query,
    today_certification_date_str,
)


def test_tomorrow_not_treated_as_today():
    scope = (
        "Given JOB_303.pdf, replot and save as JOB_304.dwg. "
        "Change the date on the plan to tomorrow's date."
    )
    result = resolve_certification_date_from_query(scope, scope_text=scope)
    tomorrow = (
        datetime.now(timezone.utc).astimezone() + timedelta(days=1)
    ).strftime("%d-%m-%Y")
    assert result == tomorrow, f"expected {tomorrow}, got {result}"


def test_today_still_works():
    scope = "Change the date on the plan to today's date"
    assert resolve_certification_date_from_query(scope, scope_text=scope) == today_certification_date_str()


def test_explicit_date():
    scope = "change the date on the plan to 25-12-2026"
    assert resolve_certification_date_from_query(scope, scope_text=scope) == "25-12-2026"


def test_three_months_and_one_week_before_today():
    ref = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc).astimezone()
    phrase = "exactly 3 months and 1 week before today"
    parsed = _parse_relative_offset_phrase(phrase, ref)
    assert parsed is not None
    assert parsed.strftime("%d-%m-%Y") == "10-03-2026"

    scope = (
        'Given JOB_303.pdf, replot and save as JOB_304.dwg. '
        "Change the date on the plan to the date of exactly 3 months and 1 week before today."
    )
    expected = (_add_months(ref, -3) - timedelta(weeks=1)).strftime("%d-%m-%Y")
    result = resolve_certification_date_from_query(scope, scope_text=scope)
    assert result == expected


def test_two_months_before_today_wins_over_history_tomorrow():
    scope = (
        'Given this Survey plan (pdf version) - "C:\\Users\\USER\\Documents\\JOB_303.pdf", '
        "extract all the important plan details, and replot the plan and save strictly as "
        "'JOB_304.dwg'. Change the date on the plan to the date of exactly 2 months before today"
    )
    history = (
        "=== CONVERSATION CONTEXT ===\n"
        "User: Change the date to tomorrow's date.\n"
        "Assistant: Done.\n\n"
        f"NOW, the user wants you to continue with this new request:\n{scope}"
    )
    result = resolve_certification_date_from_query(history, scope_text=scope)
    ref = datetime.now(timezone.utc).astimezone()
    expected = _add_months(ref, -2).strftime("%d-%m-%Y")
    assert result == expected, f"expected {expected}, got {result} (history must not force tomorrow)"


def test_date_should_now_be_natural_language():
    scope = (
        'Given this Survey plan (pdf version) - "C:\\Users\\USER\\Documents\\JOB_289.pdf", '
        "extract all the important plan details, and replot the plan and save strictly as "
        "'JOB_289.dwg'. Buyer's name should now be 'CAPT. ADOLPHUS NWAGBARA' and "
        "the date should now be 24/06/2026."
    )
    assert resolve_certification_date_from_query(scope, scope_text=scope) == "24-06-2026"
