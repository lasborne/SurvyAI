"""Regression tests for PDF→DWG path normalization and DWG output resolution."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pytest

from agent.pdf_survey_plan import (
    extract_dwg_paths_from_text,
    extract_pdf_paths_from_text,
    resolve_output_dwg_path,
    resolve_pdf_path_for_replot,
)
from survyai.attachments import normalize_user_path


def test_normalize_file_uri_windows_drive(tmp_path: Path) -> None:
    pdf = tmp_path / "Abigail_Ufere.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    uri = "file:///" + str(pdf).replace("\\", "/")
    got = normalize_user_path(uri)
    assert Path(got).resolve() == pdf.resolve()
    # Must not collapse to drive-relative C:Users\...
    assert "C:Users" not in got.replace("/", "\\")
    assert ":\\" in got.replace("/", "\\") or got[1:3] == ":\\"


def test_normalize_forward_slash_windows_path(tmp_path: Path) -> None:
    pdf = tmp_path / "plan.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    forward = str(pdf.resolve()).replace("\\", "/")
    got = normalize_user_path(forward)
    assert Path(got).resolve() == pdf.resolve()


def test_normalize_percent_encoded_spaces(tmp_path: Path) -> None:
    pdf = tmp_path / "My Plan.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    encoded = "file:///" + quote(str(pdf).replace("\\", "/"))
    got = normalize_user_path(encoded)
    assert Path(got).resolve() == pdf.resolve()


def test_extract_pdf_paths_file_uri(tmp_path: Path) -> None:
    pdf = tmp_path / "Abigail_Ufere.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    uri = "file:///" + str(pdf).replace("\\", "/")
    query = f"Replot {uri} save strictly as det3.dwg"
    paths = extract_pdf_paths_from_text(query)
    assert paths
    assert Path(paths[-1]).resolve() == pdf.resolve()


def test_resolve_pdf_path_current_turn_precedence(tmp_path: Path) -> None:
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    a.write_bytes(b"%PDF-1.4")
    b.write_bytes(b"%PDF-1.4")
    scope = f"Replot {b} as out.dwg"
    full = f"Earlier used {a}\n\nUser: {scope}"
    res = resolve_pdf_path_for_replot(scope, full)
    assert res["success"] is True
    assert Path(res["path"]).resolve() == b.resolve()


def test_resolve_pdf_missing_file_does_not_substitute(tmp_path: Path) -> None:
    missing = tmp_path / "missing_plan.pdf"
    nearby = tmp_path / "missing_plan_v2.pdf"
    nearby.write_bytes(b"%PDF-1.4")
    res = resolve_pdf_path_for_replot(f"Replot {missing} as x.dwg", "")
    assert res["success"] is False
    assert res.get("needs_user_approval") is True
    assert any(Path(p).name == nearby.name for p in (res.get("similar") or []))


def test_bare_dwg_defaults_to_workspace_not_pdf_folder(tmp_path: Path) -> None:
    pdf = tmp_path / "Abigail_Ufere.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    q = f"Replot {pdf} save strictly as det3.dwg"
    out = resolve_output_dwg_path(q, str(pdf), scope_text=q)
    assert out is not None
    assert Path(out).resolve() == (Path.cwd() / "det3.dwg").resolve()
    assert Path(out).parent.resolve() != tmp_path.resolve()


def test_current_folder_means_workspace_not_source(tmp_path: Path) -> None:
    pdf = tmp_path / "JOB_299.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    q = f'plot this plan "{pdf}" and save as \'det6.dwg\' in the current folder'
    out = resolve_output_dwg_path(q, str(pdf), scope_text=q)
    assert Path(out).resolve() == (Path.cwd() / "det6.dwg").resolve()
    assert Path(out).parent.resolve() != tmp_path.resolve()


def test_bare_dwg_beside_source_when_user_asks(tmp_path: Path) -> None:
    pdf = tmp_path / "Abigail_Ufere.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    q = f"Replot {pdf} and save det3.dwg in the same folder as the input file"
    out = resolve_output_dwg_path(q, str(pdf), scope_text=q)
    assert out is not None
    assert Path(out).resolve() == (tmp_path / "det3.dwg").resolve()


def test_dwg_extractor_does_not_merge_pdf_into_dwg_name(tmp_path: Path) -> None:
    pdf = tmp_path / "Abigail_Ufere.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    q = f"Replot {pdf} save strictly as det3.dwg"
    dwgs = extract_dwg_paths_from_text(q)
    assert dwgs
    assert all(".pdf" not in d.lower() for d in dwgs)
    assert any(Path(d).name.lower() == "det3.dwg" for d in dwgs)


def test_affirmation_uses_history_for_path(tmp_path: Path) -> None:
    pdf = tmp_path / "hist.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    full = f"Please replot {pdf} save as out.dwg\n\nUser: Proceed"
    res = resolve_pdf_path_for_replot("Proceed", full)
    assert res["success"] is True
    assert Path(res["path"]).resolve() == pdf.resolve()


def test_join_workspace_path_keeps_relative_folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.output_paths import join_workspace_path

    monkeypatch.chdir(tmp_path)
    got = join_workspace_path("exports/out.csv")
    assert got == (tmp_path / "exports" / "out.csv").resolve()
    abs_p = tmp_path / "abs.csv"
    assert join_workspace_path(str(abs_p)) == abs_p.resolve()
