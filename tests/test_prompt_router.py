"""Unit tests for cheap paid-model prompt routing (no live LLM)."""

from types import SimpleNamespace

from survyai.prompt_router import (
    apply_route_floors,
    execution_candidates_for_provider,
    match_candidate_model,
    parse_router_response,
    router_model_for_provider,
    router_models_to_try,
    should_use_llm_prompt_router,
    PromptRouteDecision,
)


def test_openai_router_is_cheapest_luna():
    assert router_model_for_provider("openai") == "gpt-5.6-luna"
    tried = router_models_to_try("openai")
    assert tried[0] == "gpt-5.6-luna"
    assert "gpt-5.4-nano" in tried
    assert "gpt-5.6-sol" not in tried


def test_other_providers_use_cheapest_paid_router():
    assert router_model_for_provider("claude") == "claude-3-5-haiku-20241022"
    assert router_model_for_provider("gemini") == "gemini-1.5-flash"
    assert router_model_for_provider("deepseek") == "deepseek-chat"


def test_openai_candidates_include_mid_and_flagship():
    models = {c.model for c in execution_candidates_for_provider("openai")}
    assert "gpt-5.6-luna" in models
    assert "gpt-5.6-terra" in models
    assert "gpt-5.4-mini" in models
    assert "gpt-5.5" in models
    assert "gpt-5.6-sol" in models
    assert "gpt-5.5-pro" not in models


def test_parse_router_json_picks_allowlisted_model():
    cands = execution_candidates_for_provider("openai")
    decision = parse_router_response(
        '{"complexity":"average","model":"gpt-5.6-terra","elevated_average":false,'
        '"reason":"typical GIS orchestration"}',
        cands,
        heuristic_complexity="complex",
    )
    assert decision is not None
    assert decision.model == "gpt-5.6-terra"
    assert decision.complexity == "average"
    assert decision.elevated_average is False


def test_parse_router_markdown_and_gpt55():
    cands = execution_candidates_for_provider("openai")
    raw = """```json
    {"complexity": "average", "model": "gpt-5.5", "elevated_average": true, "reason": "compare"}
    ```"""
    decision = parse_router_response(raw, cands)
    assert decision is not None
    assert decision.model == "gpt-5.5"
    assert decision.elevated_average is True
    assert decision.complexity == "average"


def test_parse_alias_gpt56_maps_to_sol():
    cands = execution_candidates_for_provider("openai")
    decision = parse_router_response(
        '{"complexity":"complex","model":"gpt-5.6","elevated_average":false,"reason":"hard"}',
        cands,
    )
    assert decision is not None
    assert decision.model == "gpt-5.6-sol"
    assert decision.complexity == "complex"


def test_unknown_model_falls_back_to_complexity_default():
    cands = execution_candidates_for_provider("openai")
    decision = parse_router_response(
        '{"complexity":"simple","model":"totally-unknown","reason":"x"}',
        cands,
    )
    assert decision is not None
    assert decision.model == "gpt-5.6-luna"
    assert decision.complexity == "simple"


def test_unparseable_returns_none_for_heuristic_fallback():
    cands = execution_candidates_for_provider("openai")
    assert parse_router_response("I think terra is fine", cands) is None


def test_file_driven_floor_rejects_cheapest_model():
    cands = execution_candidates_for_provider("openai")
    raw = PromptRouteDecision(
        complexity="simple",
        model="gpt-5.6-luna",
        elevated_average=False,
        reason="lookup",
        source="llm",
    )
    out = apply_route_floors(raw, cands, file_driven=True)
    assert out.complexity == "average"
    assert out.model == "gpt-5.6-terra"
    # Non-file knowledge may stay on luna.
    kept = apply_route_floors(raw, cands, file_driven=False)
    assert kept.model == "gpt-5.6-luna"


def test_file_driven_may_still_choose_sol_or_gpt55():
    cands = execution_candidates_for_provider("openai")
    sol = apply_route_floors(
        PromptRouteDecision(complexity="complex", model="gpt-5.6-sol", source="llm"),
        cands,
        file_driven=True,
    )
    assert sol.model == "gpt-5.6-sol"
    mid = apply_route_floors(
        PromptRouteDecision(
            complexity="average", model="gpt-5.5", elevated_average=True, source="llm"
        ),
        cands,
        file_driven=True,
    )
    assert mid.model == "gpt-5.5"
    assert mid.elevated_average is True


def test_should_skip_router_for_overrides_and_local():
    assert should_use_llm_prompt_router(provider="openai") is True
    assert should_use_llm_prompt_router(provider="ollama") is False
    assert should_use_llm_prompt_router(provider="openai", user_tier_override="complex") is False
    assert should_use_llm_prompt_router(provider="openai", fast_mode_forced_simple=True) is False
    assert should_use_llm_prompt_router(provider="openai", enable_tiered=False) is False
    assert should_use_llm_prompt_router(provider="openai", enable_llm_prompt_router=False) is False
    assert should_use_llm_prompt_router(provider="openai", heuristic_confidence="simple") is False
    assert should_use_llm_prompt_router(provider="openai", heuristic_confidence="complex") is False
    assert should_use_llm_prompt_router(provider="openai", heuristic_confidence="ambiguous") is True


def test_router_failover_capped_at_two_attempts():
    tried = router_models_to_try("openai")
    assert len(tried) <= 2
    assert tried[0] == "gpt-5.6-luna"


def test_heavy_gis_skips_classifier_as_complex():
    from survyai.prompt_router import heuristic_route_confidence, is_heavy_geospatial_task

    q = "Run ArcGIS IDW raster then cut/fill volume between pre and post surfaces"
    assert is_heavy_geospatial_task(q) is True
    assert heuristic_route_confidence(q, complexity="complex", file_driven=True) == "complex"


def test_short_knowledge_skips_classifier_as_simple():
    from survyai.prompt_router import heuristic_route_confidence

    q = "What is a theodolite in surveying history?"
    assert (
        heuristic_route_confidence(
            q, complexity="simple", file_driven=False, intent="knowledge", kind="general_knowledge"
        )
        == "simple"
    )


def test_typical_gis_file_job_stays_ambiguous_for_llm_router():
    from survyai.prompt_router import heuristic_route_confidence

    q = "Convert this Excel of beacons to UTM, make polygons and 5m buffers in ArcGIS"
    assert (
        heuristic_route_confidence(
            q, complexity="average", file_driven=True, intent="task", kind="file_task"
        )
        == "ambiguous"
    )


def test_knowledge_only_ceiling_demotes_sol():
    cands = execution_candidates_for_provider("openai")
    out = apply_route_floors(
        PromptRouteDecision(complexity="complex", model="gpt-5.6-sol", source="llm"),
        cands,
        file_driven=False,
        knowledge_only=True,
    )
    assert out.complexity == "average"
    assert out.model == "gpt-5.6-terra"


def test_settings_override_nano_slot():
    settings = SimpleNamespace(openai_model_nano="gpt-5.4-nano", enable_tiered_models=True)
    assert router_model_for_provider("openai", settings) == "gpt-5.4-nano"
    assert match_candidate_model("GPT-5.4-MINI", execution_candidates_for_provider("openai")) is not None


def test_parse_alignment_response_true_false_and_unusable():
    from survyai.prompt_router import parse_alignment_response

    assert parse_alignment_response('{"aligned": true, "reason": "same job"}') is True
    assert parse_alignment_response(
        '```json\n{"aligned": false, "reason": "OCR dump vs CAD plot"}\n```'
    ) is False
    assert parse_alignment_response("looks fine to me") is None


def test_openai_reasoning_models_use_responses_api_with_effort_on():
    from survyai.openai_models import (
        chat_openai_official_kwargs,
        openai_model_needs_responses_api,
        openai_reasoning_effort_for_model,
    )

    for mid in (
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.4-mini",
        "gpt-5",
        "o3-mini",
    ):
        kw = chat_openai_official_kwargs(mid)
        assert openai_model_needs_responses_api(mid)
        assert kw["use_responses_api"] is True
        assert kw["reasoning"]["effort"] != "none"
        assert kw["reasoning"]["effort"] == openai_reasoning_effort_for_model(mid)

    assert openai_reasoning_effort_for_model("gpt-5.6-sol") == "high"
    assert openai_reasoning_effort_for_model("gpt-5.6-luna") == "medium"
    assert chat_openai_official_kwargs("gpt-4o") == {}
    assert chat_openai_official_kwargs("gpt-5-chat-latest") == {}


def test_paid_llm_constructor_kwargs_keep_reasoning_on():
    from survyai.provider_models import paid_llm_constructor_kwargs

    openai_kw = paid_llm_constructor_kwargs("openai", "gpt-5.6-sol")
    assert openai_kw["use_responses_api"] is True
    assert openai_kw["reasoning"]["effort"] == "high"

    claude_kw = paid_llm_constructor_kwargs(
        "claude", "claude-sonnet-4-5", max_tokens=8192
    )
    assert claude_kw["thinking"]["type"] == "enabled"
    assert claude_kw["thinking"]["budget_tokens"] > 0
    assert paid_llm_constructor_kwargs("claude", "claude-3-5-sonnet-20241022") == {}

    gemini_kw = paid_llm_constructor_kwargs("gemini", "gemini-2.5-pro")
    assert gemini_kw["thinking_level"] == "high"
    assert paid_llm_constructor_kwargs("gemini", "gemini-2.0-flash") == {}

    assert paid_llm_constructor_kwargs("deepseek", "deepseek-chat") == {}
    assert paid_llm_constructor_kwargs("ollama", "llama3.2:1b") == {}

