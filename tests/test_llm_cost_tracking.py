"""Tests for per-query LLM cost tracking (Credits & Usage dashboard)."""

from agent.agent import SurvyAIAgent


def test_finalize_query_result_prefers_graph_cost_over_pipeline():
    agent = SurvyAIAgent.__new__(SurvyAIAgent)
    agent._pipeline_llm_cost_usd = 0.05
    out = agent.finalize_query_result_dict({"success": True, "llm_cost_usd": 0.12})
    assert out["llm_cost_usd"] == 0.12


def test_finalize_query_result_uses_pipeline_when_graph_missing():
    agent = SurvyAIAgent.__new__(SurvyAIAgent)
    agent._pipeline_llm_cost_usd = 0.034567
    out = agent.finalize_query_result_dict({"success": True, "model_name": "gpt-5.4"})
    assert out["llm_cost_usd"] == 0.034567


def test_track_llm_invoke_result_accumulates():
    try:
        from langchain_core.messages import AIMessage
    except ImportError:
        import unittest

        raise unittest.SkipTest("langchain_core not installed")

    class _Settings:
        openai_model = "gpt-5-mini"
        openai_model_mini = "gpt-5-mini"

    agent = SurvyAIAgent.__new__(SurvyAIAgent)
    agent._pipeline_llm_cost_usd = 0.0
    agent.settings = _Settings()
    agent._current_openai_model = "gpt-5-mini"

    msg = AIMessage(
        content="ok",
        response_metadata={
            "token_usage": {"prompt_tokens": 1000, "completion_tokens": 200}
        },
    )
    agent._track_llm_invoke_result(msg, "gpt-5-mini")
    assert agent._pipeline_llm_cost_usd > 0
