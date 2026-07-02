"""Regression tests for strict PDF/DWG path resolution in survey replot."""

from pathlib import Path

from agent.pdf_survey_plan import (
    extract_pdf_paths_from_text,
    resolve_output_dwg_path,
    resolve_pdf_path_for_replot,
)


def test_scope_pdf_wins_over_history():
    scope = (
        'Given this Survey plan (pdf version) - "C:\\Users\\USER\\Documents\\JOB_303.pdf", '
        "replot and save strictly as 'JOB_303.dwg'."
    )
    history = (
        "--- prior ---\n"
        "Processed C:\\Users\\USER\\Documents\\JOB_301.pdf -> JOB_302.dwg\n\n"
        f"NOW, the user wants you to continue with this new request:\n{scope}"
    )
    pdfs = extract_pdf_paths_from_text(scope)
    assert any(p.endswith("JOB_303.pdf") for p in pdfs)

    res = resolve_pdf_path_for_replot(scope, history)
    assert res.get("requested", "").endswith("JOB_303.pdf")
    if res.get("success"):
        assert res["path"].endswith("JOB_303.pdf")
    else:
        assert res.get("needs_user_approval") in (True, False)

    out = resolve_output_dwg_path(history, r"C:\Users\USER\Documents\JOB_303.pdf", scope_text=scope)
    assert out.endswith("JOB_303.dwg")


def test_bare_filename_in_scope_not_replaced_by_history():
    scope = "Given JOB_303.pdf save as JOB_303.dwg"
    history = "Previous: C:\\Users\\USER\\Documents\\JOB_301.pdf was used"
    res = resolve_pdf_path_for_replot(scope, history)
    requested = res.get("requested", "")
    assert requested.endswith("JOB_303.pdf"), res
    assert "JOB_301.pdf" not in requested


def test_save_strictly_as_pattern():
    scope = "save strictly as 'JOB_303.dwg'"
    out = resolve_output_dwg_path(
        scope,
        r"C:\folder\plan.pdf",
        scope_text=scope,
    )
    assert Path(out) == Path(r"C:\folder\JOB_303.dwg")
