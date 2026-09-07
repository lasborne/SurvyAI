"""Latency-focused regressions for PDF→DWG orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from agent.pdf_survey_plan import (
    SurveyPlanExtraction,
    SurveyTraverseLeg,
    _plan_override_change_requested,
    analyze_pdf_access_roads,
    clear_pdf_source_cache,
    resolve_plan_overrides_from_query,
    store_cached_pdf_sources,
)


def _closed_extraction() -> SurveyPlanExtraction:
    pillars = ["SC/CR 1", "SC/CR 2", "SC/CR 3", "SC/CR 4"]
    specs = [(0, 0, 10.0), (90, 0, 10.0), (180, 0, 10.0), (270, 0, 10.0)]
    legs = []
    for i, (bd, bm, dist) in enumerate(specs):
        legs.append(
            SurveyTraverseLeg(
                from_pillar=pillars[i],
                to_pillar=pillars[(i + 1) % 4],
                bearing_deg=bd,
                bearing_min=bm,
                distance_m=dist,
            )
        )
    return SurveyPlanExtraction(
        buyer_name="Abigail Ufere",
        location="Test Site",
        lga="Obio/Akpor",
        state="Rivers",
        plan_number="RS/ABJ/123",
        surveyor_name="SURV. TEST",
        certification_date="01-01-2026",
        pillar_numbers=pillars,
        traverse_legs=legs,
        coordinates=[
            {"easting": 100.0, "northing": 200.0, "pillar": pillars[0]},
        ],
        area_sq_m=100.0,
        scale_denom=500,
        source="pdf_vector",
        confidence=0.95,
        origin_crs="UTM Zone 32N",
    )


def test_plain_replot_does_not_request_override_llm() -> None:
    assert not _plan_override_change_requested(
        "Replot this plan Abigail_Ufere.pdf and save as det3.dwg"
    )
    assert _plan_override_change_requested(
        "Replot Abigail_Ufere.pdf, change buyer name to Jane Doe, save as det3.dwg"
    )


def test_resolve_overrides_without_llm_on_plain_replot() -> None:
    base = _closed_extraction()
    overrides = resolve_plan_overrides_from_query(
        "Replot this plan and save as det3.dwg",
        scope_text="Replot this plan and save as det3.dwg",
        base_extraction=base,
        llm=MagicMock(),
        run_with_timeout=MagicMock(side_effect=AssertionError("override LLM must not run")),
    )
    assert overrides.override_fields == []


def test_access_road_skips_llm_when_labels_found_without_width(tmp_path: Path) -> None:
    pdf = tmp_path / "road.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    extraction = _closed_extraction()
    llm = MagicMock()
    run = MagicMock(side_effect=AssertionError("road LLM must not run"))

    page = MagicMock()
    doc = MagicMock()
    doc.__getitem__.return_value = page
    fake_fitz = MagicMock()
    fake_fitz.open.return_value = doc

    with patch.dict("sys.modules", {"fitz": fake_fitz}):
        with patch(
            "agent.pdf_survey_plan._extract_all_access_road_labels_from_pdf",
            return_value=[("ACCESS ROAD", (10.0, 20.0))],
        ):
            with patch("agent.pdf_survey_plan.calibrate_pdf_meters_per_point", return_value=0.0):
                with patch("agent.pdf_survey_plan._pdf_page_line_segments", return_value=[]):
                    with patch("agent.pdf_survey_plan._road_spec_for_label", return_value=None):
                        with patch(
                            "agent.pdf_survey_plan.resolve_access_roads_with_llm"
                        ) as resolve_llm:
                            result = analyze_pdf_access_roads(
                                str(pdf),
                                extraction,
                                "ACCESS ROAD",
                                llm=llm,
                                run_with_timeout=run,
                            )

    resolve_llm.assert_not_called()
    assert result is not None
    specs, title = result
    assert specs == []
    assert "ACCESS" in title.upper()
    assert "width not printed" in (extraction.notes or "").lower()


def test_pdf_source_cache_avoids_duplicate_text_loads(tmp_path: Path) -> None:
    from agent.pdf_survey_plan import _load_pdf_extraction_sources

    clear_pdf_source_cache()
    pdf = tmp_path / "plan.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    store_cached_pdf_sources(
        str(pdf),
        {
            "layout_text": "LAYOUT",
            "plain_text": "PLAIN",
            "images": [],
            "page_count": 1,
        },
    )

    with patch(
        "agent.pdf_survey_plan.extract_layout_text_from_pdf",
        side_effect=AssertionError("layout must not reload"),
    ):
        with patch(
            "agent.pdf_survey_plan.extract_plain_text_from_pdf",
            side_effect=AssertionError("plain must not reload"),
        ):
            layout, plain, images = _load_pdf_extraction_sources(
                str(pdf), vision_max_pages=1, skip_vision=True
            )
    assert layout == "LAYOUT"
    assert plain == "PLAIN"
    assert images == []


def test_tier_fallback_returns_cached_sources_and_skips_complex_for_metadata() -> None:
    """Vector/heuristic extraction with only soft metadata issues should not escalate."""
    from agent.agent import SurvyAIAgent

    agent = SurvyAIAgent.__new__(SurvyAIAgent)
    agent.settings = MagicMock()
    agent.settings.openai_model = "gpt-test"
    agent.llm_primary = MagicMock()
    agent._try_openai_tier_llm = MagicMock(return_value=(MagicMock(), "avg-model"))
    agent._llm_run_with_timeout = MagicMock(return_value=lambda *a, **k: None)
    agent._current_openai_model = None

    extraction = _closed_extraction()
    extraction.source = "heuristic"
    sources = ("LAYOUT", "PLAIN", [])

    with patch(
        "agent.pdf_survey_plan._load_pdf_extraction_sources", return_value=sources
    ):
        with patch("agent.pdf_survey_plan.get_cached_pdf_sources", return_value={"page_count": 1}):
            with patch(
                "agent.pdf_survey_plan.extract_survey_plan_from_pdf", return_value=extraction
            ) as extract:
                with patch(
                    "agent.pdf_survey_plan.validate_extraction_for_replot",
                    return_value=["buyer name missing"],
                ):
                    out, model, cached = agent._extract_pdf_survey_plan_with_tier_fallback(
                        "dummy.pdf",
                        user_notes="replot",
                        timeout_s=30,
                        total_deadline_s=60,
                    )

    assert out is extraction
    assert model == "avg-model"
    assert cached == sources
    assert extract.call_count == 1


def test_replot_pipeline_reuses_cached_sources_and_skips_simple_warm(tmp_path: Path) -> None:
    from agent.agent import SurvyAIAgent

    pdf = tmp_path / "Abigail_Ufere.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    extraction = _closed_extraction()
    sources = ("L", "P", [])

    agent = SurvyAIAgent.__new__(SurvyAIAgent)
    agent.settings = MagicMock()
    agent._cadastral_user_message_body = lambda q: q
    agent._resolve_pdf_path_for_replot = MagicMock(
        return_value={"success": True, "path": str(pdf)}
    )
    agent._resolve_cadastral_template_from_memory = MagicMock(
        return_value={"template_path": "T.dwg", "profile_path": "T.json"}
    )
    agent._ensure_autocad_connected = MagicMock()
    agent._try_openai_tier_llm = MagicMock(
        side_effect=AssertionError("simple tier must not warm on plain replot")
    )
    agent._llm_run_with_timeout = MagicMock(return_value=None)
    agent._extract_pdf_survey_plan_with_tier_fallback = MagicMock(
        return_value=(extraction, "avg", sources)
    )
    agent._run_cadastral_cad_prompt_pipeline = MagicMock(
        return_value={"success": True, "output_dwg": str(tmp_path / "det3.dwg")}
    )

    with patch(
        "agent.pdf_survey_plan.extract_layout_text_from_pdf",
        side_effect=AssertionError("must reuse cached sources"),
    ):
        with patch(
            "agent.pdf_survey_plan.extract_plain_text_from_pdf",
            side_effect=AssertionError("must reuse cached sources"),
        ):
            with patch(
                "agent.pdf_survey_plan.validate_extraction_for_replot", return_value=[]
            ):
                with patch(
                    "agent.pdf_survey_plan.validate_subprompt_geometry", return_value=[]
                ):
                    with patch(
                        "agent.pdf_survey_plan.enrich_extraction_coordinates",
                        side_effect=lambda e, *_a, **_k: e,
                    ):
                        result = agent._run_pdf_survey_replot_pipeline(
                            f"Replot this plan {pdf} and save as 'det3.dwg'"
                        )

    assert result.get("success") is True
    agent._try_openai_tier_llm.assert_not_called()
    cad_kwargs = agent._run_cadastral_cad_prompt_pipeline.call_args.kwargs
    assert cad_kwargs.get("skip_session_prep") is True
    assert cad_kwargs.get("template_override_path") == "T.dwg"
    assert "perf" in result


def _tier_fallback_agent() -> Any:
    from agent.agent import SurvyAIAgent

    agent = SurvyAIAgent.__new__(SurvyAIAgent)
    agent.settings = MagicMock()
    agent.settings.openai_model = "gpt-test"
    agent.llm_primary = MagicMock()
    agent._try_openai_tier_llm = MagicMock(return_value=(MagicMock(), "avg-model"))
    agent._llm_run_with_timeout = MagicMock(return_value=lambda *a, **k: None)
    agent._current_openai_model = None
    return agent


def _run_tier_fallback(validation_issues: List[str]) -> Any:
    agent = _tier_fallback_agent()
    extraction = _closed_extraction()
    extraction.source = "layout_text"
    sources = ("LAYOUT", "PLAIN", [])

    with patch("agent.pdf_survey_plan._load_pdf_extraction_sources", return_value=sources):
        with patch("agent.pdf_survey_plan.get_cached_pdf_sources", return_value={"page_count": 1}):
            with patch(
                "agent.pdf_survey_plan.extract_survey_plan_from_pdf", return_value=extraction
            ) as extract:
                with patch(
                    "agent.pdf_survey_plan.validate_extraction_for_replot",
                    return_value=validation_issues,
                ):
                    agent._extract_pdf_survey_plan_with_tier_fallback(
                        "dummy.pdf",
                        user_notes="replot",
                        timeout_s=30,
                        total_deadline_s=60,
                    )
    return extract


def test_tier_fallback_escalates_when_closure_is_unusable() -> None:
    """Latency work must never trade away the stronger pass a bad closure needs.

    Having the right *number* of pillars and legs says nothing about whether the
    bearing/distance labels were paired correctly, so an out-of-tolerance
    misclosure has to escalate rather than be accepted as "usable geometry".
    """
    extract = _run_tier_fallback(
        ["traverse misclosure 1.500 m over 28.1 m perimeter (1:18.7)"]
    )
    assert extract.call_count == 2


def test_tier_fallback_stops_at_first_tier_when_validation_passes() -> None:
    extract = _run_tier_fallback([])
    assert extract.call_count == 1


def test_heuristic_not_ready_when_closure_out_of_tolerance() -> None:
    """The no-vision fast path must clear the same gate the replot enforces."""
    from agent.pdf_survey_plan import _heuristic_pdf_extraction_is_replot_ready

    ext = _closed_extraction()
    ext.anchor_easting = 352100.0
    ext.anchor_northing = 713450.0
    # Complete pillar/leg counts and plausible UTM anchors, but the traverse no
    # longer closes: the last leg is short by 1.5 m.
    ext.traverse_legs[-1].distance_m = 8.5
    assert _heuristic_pdf_extraction_is_replot_ready(ext, "", pdf_path=None) is False


def test_heuristic_ready_when_extraction_passes_full_validation() -> None:
    from agent.pdf_survey_plan import _heuristic_pdf_extraction_is_replot_ready

    ext = _closed_extraction()
    ext.anchor_easting = 352100.0
    ext.anchor_northing = 713450.0
    with patch("agent.pdf_survey_plan.validate_extraction_for_replot", return_value=[]):
        assert _heuristic_pdf_extraction_is_replot_ready(ext, "", pdf_path=None) is True


def test_heuristic_not_ready_when_title_block_is_missing() -> None:
    """A closed traverse is not a finished cadastral plan without the title block."""
    from agent.pdf_survey_plan import _heuristic_pdf_extraction_is_replot_ready

    ext = _closed_extraction()
    ext.anchor_easting = 352100.0
    ext.anchor_northing = 713450.0
    ext.lga = ""
    ext.plan_number = "SURV"
    ext.certification_date = ""
    with patch("agent.pdf_survey_plan.validate_extraction_for_replot", return_value=[]):
        assert _heuristic_pdf_extraction_is_replot_ready(ext, "", pdf_path=None) is False


def _overwrite_agent() -> Any:
    from agent.agent import SurvyAIAgent

    agent = SurvyAIAgent.__new__(SurvyAIAgent)
    agent._is_protected_template_path = MagicMock(return_value=False)
    return agent


def test_declined_overwrite_is_decided_before_any_extraction(tmp_path: Path) -> None:
    """A decline must cost nothing: no PDF parse, no vision, no AutoCAD."""
    existing = tmp_path / "det.dwg"
    existing.write_bytes(b"dwg")
    agent = _overwrite_agent()

    with patch("agent.agent._confirm_overwrite_existing_dwg", return_value=False) as ask:
        blocked = agent._confirm_output_overwrite_upfront(str(existing))

    ask.assert_called_once()
    assert blocked is not None
    assert blocked["cancelled"] is True


def test_approved_overwrite_is_not_asked_again_by_the_plot_step(tmp_path: Path) -> None:
    existing = tmp_path / "det.dwg"
    existing.write_bytes(b"dwg")
    agent = _overwrite_agent()

    with patch("agent.agent._confirm_overwrite_existing_dwg", return_value=True) as ask:
        assert agent._confirm_output_overwrite_upfront(str(existing)) is None
    assert ask.call_count == 1

    resolved = str(Path(existing).resolve())
    assert agent._consume_overwrite_confirmation(resolved) is True
    # One-shot: a later output of the same name must ask again.
    assert agent._consume_overwrite_confirmation(resolved) is False


def test_missing_output_is_never_prompted(tmp_path: Path) -> None:
    agent = _overwrite_agent()
    with patch("agent.agent._confirm_overwrite_existing_dwg", return_value=False) as ask:
        assert agent._confirm_output_overwrite_upfront(str(tmp_path / "new.dwg")) is None
    ask.assert_not_called()


def test_stale_approval_does_not_suppress_a_later_prompt(tmp_path: Path) -> None:
    existing = tmp_path / "det.dwg"
    existing.write_bytes(b"dwg")
    agent = _overwrite_agent()

    with patch("agent.agent._confirm_overwrite_existing_dwg", return_value=True):
        agent._confirm_output_overwrite_upfront(str(existing))
    # A new request that never reached the plot step must not inherit the token.
    with patch("agent.agent._confirm_overwrite_existing_dwg", return_value=True):
        agent._confirm_output_overwrite_upfront(str(tmp_path / "other.dwg"))
    assert agent._consume_overwrite_confirmation(str(Path(existing).resolve())) is False
