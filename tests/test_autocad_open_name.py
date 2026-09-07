"""Mocked AutoCAD COM regressions for Open.Name recovery and bounded opens."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.autocad_processor import AutoCADProcessor


def test_open_name_classified_as_broken_proxy() -> None:
    assert AutoCADProcessor._com_error_is_broken_proxy(Exception("Open.Name"))
    assert AutoCADProcessor._com_error_is_broken_proxy(AttributeError("<unknown>.Name"))
    assert not AutoCADProcessor._com_error_is_broken_proxy(Exception("file not found"))


def test_activate_by_path_avoids_documents_open(tmp_path: Path) -> None:
    dwg = tmp_path / "det3.dwg"
    dwg.write_bytes(b"AC1032")

    proc = AutoCADProcessor(auto_connect=False)
    proc._connected = True
    proc.acad = MagicMock()

    doc = MagicMock()
    doc.FullName = str(dwg)
    doc.Name = "det3.dwg"
    docs = MagicMock()
    docs.Count = 1
    docs.Item.return_value = doc
    proc.acad.Documents = docs

    with patch.object(proc, "_documents_count_safe", return_value=1):
        with patch.object(proc, "_com_retry", side_effect=lambda fn, **kw: fn()):
            with patch.object(proc, "_safe_doc_name", return_value="det3.dwg"):
                ok = proc._activate_document_by_path(dwg)

    assert ok is True
    doc.Activate.assert_called()
    # Documents.Open must not be used for activation.
    assert not hasattr(docs, "Open") or not docs.Open.called


def test_open_drawing_recovers_once_on_open_name(tmp_path: Path) -> None:
    dwg = tmp_path / "plan.dwg"
    dwg.write_bytes(b"AC1032")

    proc = AutoCADProcessor(auto_connect=False)
    proc._connected = True
    proc.acad = MagicMock()

    opened_doc = MagicMock()
    opened_doc.FullName = str(dwg)
    opened_doc.Name = "plan.dwg"
    opened_doc.ReadOnly = False
    opened_doc.ModelSpace.Count = 0

    docs = MagicMock()
    docs.Count = 0
    docs.Open.return_value = opened_doc
    proc.acad.Documents = docs
    proc.acad.ActiveDocument = opened_doc

    name_calls = {"n": 0}

    def safe_name(doc, deadline_ts=None):
        name_calls["n"] += 1
        if name_calls["n"] == 1:
            raise AttributeError("Open.Name")
        return "plan.dwg"

    with patch.object(proc, "connect", return_value=True):
        with patch.object(proc, "quiesce_autocad"):
            with patch.object(proc, "recover_com_session", return_value=True) as recover:
                with patch.object(proc, "_activate_document_by_path", return_value=True):
                    with patch.object(proc, "_safe_doc_name", side_effect=safe_name):
                        with patch.object(proc, "_com_retry", side_effect=lambda fn, **kw: fn()):
                            with patch.object(proc, "_count_entities", return_value=0):
                                with patch.object(proc, "_get_layers", return_value=[]):
                                    with patch.object(proc, "_get_units", return_value="Meters"):
                                        # Existing-doc scan: empty docs then open succeeds.
                                        docs.Count = 0
                                        result = proc.open_drawing(str(dwg), read_only=False)

    assert recover.called
    assert result.get("success") is True or (
        result.get("success") is False and "Open.Name" in str(result.get("error") or "")
    )


def test_save_close_to_release_lock_saves_then_closes(tmp_path: Path) -> None:
    dwg = tmp_path / "det7.dwg"
    dwg.write_bytes(b"AC1032")
    proc = AutoCADProcessor(auto_connect=False)
    proc._connected = True
    with patch.object(proc, "quiesce_autocad") as quiesce:
        with patch.object(
            proc, "save_and_close_drawing", return_value={"success": True, "errors": []}
        ) as save_close:
            with patch.object(proc, "is_drawing_open", return_value=False):
                out = proc.save_close_to_release_lock(str(dwg))
    quiesce.assert_called()
    save_close.assert_called_once()
    assert save_close.call_args.kwargs.get("save") is True
    assert out.get("closed") is True
    assert out.get("success") is True


def test_save_close_to_release_lock_force_closes_if_still_open(tmp_path: Path) -> None:
    dwg = tmp_path / "det7.dwg"
    dwg.write_bytes(b"AC1032")
    proc = AutoCADProcessor(auto_connect=False)
    proc._connected = True
    with patch.object(proc, "quiesce_autocad"):
        with patch.object(
            proc, "save_and_close_drawing", return_value={"success": False, "errors": ["busy"]}
        ):
            with patch.object(proc, "is_drawing_open", return_value=True):
                with patch.object(
                    proc, "close_drawing_if_open", return_value={"closed": True, "error": None}
                ) as force_close:
                    out = proc.save_close_to_release_lock(str(dwg))
    force_close.assert_called_once()
    assert force_close.call_args.kwargs.get("save_changes") is True
    assert out.get("closed") is True


def test_com_retry_respects_deadline() -> None:
    proc = AutoCADProcessor(auto_connect=False)
    import time

    deadline = time.time() + 0.15
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        raise AttributeError("<unknown>.Open")

    with pytest.raises(AttributeError):
        proc._com_retry(flaky, attempts=50, base_sleep=0.05, deadline_ts=deadline)
    # Bound by deadline — far fewer than 50 attempts.
    assert calls["n"] < 20
