"""Mocked COM latency regressions: one open, handle cache, batched delete."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.autocad_processor import AutoCADProcessor


def test_delete_entities_on_layers_single_modelspace_pass() -> None:
    proc = AutoCADProcessor(auto_connect=False)
    proc._connected = True
    proc.doc = MagicMock()
    proc.ensure_workflow_document = MagicMock(return_value=True)

    e1 = MagicMock()
    e1.Layer = "CADA_BOUNDARY"
    e1.ObjectName = "AcDbPolyline"
    e2 = MagicMock()
    e2.Layer = "CADA_ROAD"
    e2.ObjectName = "AcDbPolyline"
    e3 = MagicMock()
    e3.Layer = "KEEP"
    e3.ObjectName = "AcDbText"

    ms = MagicMock()
    ms.Count = 3
    ms.Item.side_effect = lambda i: [e1, e2, e3][i]
    proc.doc.ModelSpace = ms

    result = proc.delete_entities_on_layers(["CADA_BOUNDARY", "CADA_ROAD", "CADA_TEXT"])
    assert result["success"] is True
    assert result["deleted"] == 2
    e1.Delete.assert_called_once()
    e2.Delete.assert_called_once()
    e3.Delete.assert_not_called()
    # One reverse pass: Item called once per entity index.
    assert ms.Item.call_count == 3
    assert proc._modelspace_scan_count == 1


def test_busy_com_is_not_treated_as_a_dead_session() -> None:
    """'Call was rejected by callee' means busy, not broken.

    Reconnecting cannot make AutoCAD less busy, so classifying it as a dead proxy
    turned every busy moment into a full (and futile) session teardown.
    """
    busy = Exception("(-2147418111, 'Call was rejected by callee.', None, None)")
    assert AutoCADProcessor._com_error_is_busy(busy) is True
    assert AutoCADProcessor._com_error_is_broken_proxy(busy) is False

    dead = Exception("<unknown>.Count failed")
    assert AutoCADProcessor._com_error_is_broken_proxy(dead) is True
    assert AutoCADProcessor._com_error_is_busy(dead) is False

    for msg in ("RPC server is unavailable", "Invalid class string", "Open.Name"):
        assert AutoCADProcessor._com_error_is_broken_proxy(Exception(msg)) is True


def test_busy_com_settles_instead_of_recovering_session() -> None:
    proc = AutoCADProcessor(auto_connect=False)
    proc._connected = True
    proc.acad = MagicMock()
    proc.doc = MagicMock()
    proc.recover_com_session = MagicMock(return_value=True)

    assert proc._settle_busy_com(Exception("Call was rejected by callee."), wait_s=0.0) is True
    proc.recover_com_session.assert_not_called()

    assert proc._settle_busy_com(Exception("<unknown>.Count"), wait_s=0.0) is False


def test_busy_com_is_retried_by_com_retry() -> None:
    proc = AutoCADProcessor(auto_connect=False)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("Call was rejected by callee.")
        return "ok"

    assert proc._com_retry(flaky, attempts=5, base_sleep=0.0) == "ok"
    assert calls["n"] == 3


def _snapshot_proc(entities: list) -> AutoCADProcessor:
    proc = AutoCADProcessor(auto_connect=False)
    proc._connected = True
    proc.doc = MagicMock()
    proc.doc.FullName = r"C:\tmp\out.dwg"
    proc._ensure_active_document = MagicMock(return_value=True)
    proc.ensure_workflow_document = MagicMock(return_value=True)
    ms = MagicMock()
    ms.Count = len(entities)
    ms.Item.side_effect = lambda i: entities[i]
    proc.doc.ModelSpace = ms
    return proc


def _bbox_entity(layer: str, box: tuple) -> MagicMock:
    e = MagicMock()
    e.Layer = layer
    e.ObjectName = "AcDbPolyline"
    e.Handle = layer
    e.GetBoundingBox.return_value = ((box[0], box[1], 0.0), (box[2], box[3], 0.0))
    return e


def test_repeated_queries_share_one_modelspace_classification_pass() -> None:
    """Bbox fitting runs many times per plot; each must not re-walk ModelSpace."""
    ents = [_bbox_entity("CADA_BOUNDARY", (0, 0, 10, 10)), _bbox_entity("KEEP", (0, 0, 99, 99))]
    proc = _snapshot_proc(ents)
    ms = proc.doc.ModelSpace

    for _ in range(5):
        r = proc.get_modelspace_bbox(layers=["CADA_BOUNDARY"])
        assert r["success"] is True
        assert r["maxx"] == 10.0
    proc.get_sample_text_height(layers=["CADA_BOUNDARY"])
    proc.list_tables()

    assert ms.Item.call_count == len(ents)
    assert proc._modelspace_scan_count == 1


def test_snapshot_reads_geometry_live_after_move() -> None:
    """Only classification is cached: a move must be reflected immediately."""
    ent = _bbox_entity("CADA_BOUNDARY", (0, 0, 10, 10))
    proc = _snapshot_proc([ent])

    first = proc.get_modelspace_bbox(layers=["CADA_BOUNDARY"])
    assert first["minx"] == 0.0

    ent.GetBoundingBox.return_value = ((100.0, 200.0, 0.0), (110.0, 210.0, 0.0))
    second = proc.get_modelspace_bbox(layers=["CADA_BOUNDARY"])
    assert second["minx"] == 100.0
    assert second["maxy"] == 210.0


def test_snapshot_rebuilds_when_entity_count_changes() -> None:
    ents = [_bbox_entity("CADA_BOUNDARY", (0, 0, 10, 10))]
    proc = _snapshot_proc(ents)
    assert proc.get_modelspace_bbox(layers=["CADA_BOUNDARY"])["maxx"] == 10.0

    ents.append(_bbox_entity("CADA_BOUNDARY", (0, 0, 40, 40)))
    proc.doc.ModelSpace.Count = 2
    assert proc.get_modelspace_bbox(layers=["CADA_BOUNDARY"])["maxx"] == 40.0
    assert proc._modelspace_scan_count == 2


def test_quiesce_throttled_between_edits_but_rearmed_by_com_error() -> None:
    """ESC is a blocking SendCommand; it must not fire before every entity write."""
    proc = AutoCADProcessor(auto_connect=False)
    proc._connected = True
    proc.acad = MagicMock()
    proc.doc = MagicMock()

    proc.quiesce_autocad()
    first = proc.doc.SendCommand.call_count
    assert first > 0

    for _ in range(20):
        proc.quiesce_autocad()
    assert proc.doc.SendCommand.call_count == first

    # A busy/rejected COM call must restore the cancel immediately.
    try:
        proc._com_retry(lambda: (_ for _ in ()).throw(RuntimeError("boom")), attempts=1)
    except Exception:
        pass
    proc.quiesce_autocad()
    assert proc.doc.SendCommand.call_count > first


def test_quiesce_rearms_after_window_elapses() -> None:
    proc = AutoCADProcessor(auto_connect=False)
    proc._connected = True
    proc.acad = MagicMock()
    proc.doc = MagicMock()

    proc.quiesce_autocad()
    baseline = proc.doc.SendCommand.call_count
    proc.quiesce_autocad()
    assert proc.doc.SendCommand.call_count == baseline

    # Operator interaction is still cancelled once the trust window expires.
    proc._last_quiesce_ts -= proc._QUIESCE_REARM_S + 1.0
    proc.quiesce_autocad()
    assert proc.doc.SendCommand.call_count > baseline


def test_handle_cache_reuses_and_invalidates_stale_proxy() -> None:
    proc = AutoCADProcessor(auto_connect=False)
    proc.doc = MagicMock()
    proc.doc.FullName = r"C:\tmp\out.dwg"
    proc._ensure_active_document = MagicMock(return_value=True)

    good = MagicMock()
    good.Handle = "ABC"
    good.ObjectName = "AcDbTable"
    good.GetText.return_value = "hello"

    ms = MagicMock()
    ms.Count = 1
    ms.Item.return_value = good
    proc.doc.ModelSpace = ms

    r1 = proc.get_table_cell_text("ABC", 0, 0)
    assert r1["success"] is True
    assert r1["text"] == "hello"
    assert ms.Item.call_count == 1

    r2 = proc.get_table_cell_text("ABC", 1, 0)
    assert r2["success"] is True
    # Cache hit — no second ModelSpace scan.
    assert ms.Item.call_count == 1

    good2 = MagicMock()
    good2.Handle = "ABC"
    good2.ObjectName = "AcDbTable"
    good2.GetText.return_value = "again"
    ms.Item.return_value = good2

    class Broken:
        ObjectName = "AcDbTable"

        @property
        def Handle(self):
            raise RuntimeError("stale")

        def GetText(self, *_a, **_k):
            return "no"

    proc._entity_handle_cache["ABC|AcDbTable"] = Broken()
    proc._handle_cache_doc_key = str(Path(r"C:\tmp\out.dwg").resolve()).lower()
    r3 = proc.get_table_cell_text("ABC", 0, 0)
    assert r3["success"] is True
    assert r3["text"] == "again"
    assert ms.Item.call_count >= 2


def test_open_drawing_counts_documents_open_once(tmp_path: Path) -> None:
    dwg = tmp_path / "out.dwg"
    dwg.write_bytes(b"AC1032")

    proc = AutoCADProcessor(auto_connect=False)
    proc._connected = True
    proc.acad = MagicMock()

    opened = MagicMock()
    opened.FullName = str(dwg)
    opened.Name = "out.dwg"
    opened.ReadOnly = False
    opened.ModelSpace.Count = 0

    docs = MagicMock()
    docs.Count = 0
    docs.Open.return_value = opened
    proc.acad.Documents = docs
    proc.acad.ActiveDocument = opened

    with patch.object(proc, "connect", return_value=True):
        with patch.object(proc, "quiesce_autocad"):
            with patch.object(proc, "_activate_document_by_path", return_value=True):
                with patch.object(proc, "_safe_doc_name", return_value="out.dwg"):
                    with patch.object(proc, "_com_retry", side_effect=lambda fn, **kw: fn()):
                        with patch.object(proc, "_count_entities", return_value=0):
                            with patch.object(proc, "_get_layers", return_value=[]):
                                with patch.object(proc, "_get_units", return_value="Meters"):
                                    result = proc.open_drawing(str(dwg))

    assert result["success"] is True
    assert docs.Open.call_count == 1
    assert proc._documents_open_count == 1


def test_ensure_output_saved_activates_without_reopen(tmp_path: Path) -> None:
    from agent.agent import SurvyAIAgent

    dwg = tmp_path / "det3.dwg"
    dwg.write_bytes(b"AC1032")

    agent = SurvyAIAgent.__new__(SurvyAIAgent)
    agent._is_protected_template_path = MagicMock(return_value=False)
    agent.autocad = MagicMock()
    agent.autocad.is_drawing_open.return_value = True
    agent.autocad._activate_document_by_path.return_value = True
    agent.autocad.get_active_document_path.return_value = str(dwg)

    agent._ensure_output_saved(str(dwg))

    agent.autocad.open_drawing.assert_not_called()
    agent.autocad._activate_document_by_path.assert_called()
    agent.autocad.save_active_drawing.assert_called_once()


def test_bulk_cad_primitives_can_skip_repeated_document_probes() -> None:
    """Clustered loops validate the document once, not once per entity."""
    proc = AutoCADProcessor(auto_connect=False)
    proc.doc = MagicMock()
    proc._ensure_active_document = MagicMock(
        side_effect=AssertionError("per-entity active-document probe")
    )
    proc.ensure_workflow_document = MagicMock(
        side_effect=AssertionError("per-entity workflow-document probe")
    )

    mt = MagicMock()
    mt.InsertionPoint = (1.0, 2.0, 0.0)
    mt.Height = 1.2
    proc.doc.ModelSpace.AddMText.return_value = mt

    block = MagicMock()
    block.Handle = "B1"
    block.Layer = "CADA_PILLARS"
    proc.doc.ModelSpace.InsertBlock.return_value = block

    poly = MagicMock()
    poly.Handle = "P1"
    poly.Layer = "CADA_BOUNDARY"
    proc.doc.ModelSpace.AddLightWeightPolyline.return_value = poly

    assert proc.add_mtext(
        "JI/OD\\P001",
        1.0,
        2.0,
        layer="CADA_PILLARNUMBERS",
        height=1.2,
        assume_active=True,
    )["success"]
    assert proc.insert_block(
        "PEG_SYMBOL",
        1.0,
        2.0,
        layer="CADA_PILLARS",
        assume_active=True,
    )["success"]
    assert proc.create_lwpolyline(
        [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
        "CADA_BOUNDARY",
        assume_active=True,
    )["success"]
    proc._ensure_active_document.assert_not_called()
    proc.ensure_workflow_document.assert_not_called()
