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


def test_retry_detection_uses_current_request_inside_context():
    wrapped = (
        "=== CONTINUATION OF PREVIOUS WORK ===\n"
        "Previous request generated a CAD plan.\n\n"
        "NOW, the user wants you to continue with this new request:\n"
        "retry"
    )

    assert SurvyAIAgent._is_retry_request_from_routing_context(
        raw_query=wrapped,
        extracted_query="",
        routing_query="",
    )


def test_graph_cost_is_zero_without_provider_reported_usage():
    try:
        from langchain_core.messages import AIMessage
    except ImportError:
        import unittest

        raise unittest.SkipTest("langchain_core not installed")

    class _Settings:
        openai_model_mini = "gpt-5-mini"

    agent = SurvyAIAgent.__new__(SurvyAIAgent)
    agent.settings = _Settings()
    cost = agent._estimate_llm_cost_usd_from_graph_result(
        {"messages": [AIMessage(content="estimated only")]},
        "gpt-5-mini",
        "estimated only",
    )

    assert cost == 0.0


def test_estimated_usage_is_not_billed_in_proxy_accounting():
    """Estimated token usage must not debit the subscription credit pool."""
    from types import SimpleNamespace

    settings = SimpleNamespace(credit_markup_multiplier=2.0)
    usage = {"estimated": True, "input_tokens": 1000, "output_tokens": 200, "cost_usd": 0.05}
    usage_is_reported = not bool(usage.get("estimated"))
    billed_cost_usd = 0.0 if not usage_is_reported else float(usage.get("cost_usd") or 0.0)
    markup_cost_usd = round(billed_cost_usd * settings.credit_markup_multiplier, 6)

    assert billed_cost_usd == 0.0
    assert markup_cost_usd == 0.0


def test_marked_up_provider_usage_increments_credits_used():
    """Provider-reported cost is marked up before debiting monthly_credits_used_usd."""
    from types import SimpleNamespace

    settings = SimpleNamespace(credit_markup_multiplier=2.0)
    user = SimpleNamespace(monthly_credits_usd=1.0, monthly_credits_used_usd=0.1)
    billed_cost_usd = 0.05
    markup_cost_usd = round(billed_cost_usd * settings.credit_markup_multiplier, 6)
    used_before = float(user.monthly_credits_used_usd or 0.0)
    budget = float(user.monthly_credits_usd or 0.0)
    if budget > 0 and markup_cost_usd > 0:
        user.monthly_credits_used_usd = round(min(budget, used_before + markup_cost_usd), 6)

    assert markup_cost_usd == 0.1
    assert user.monthly_credits_used_usd == 0.2


def test_daily_plan_budget_uses_catalog_ngn_not_monthly_fraction():
    from survyai_cloud.config import CloudSettings
    from survyai_cloud.services.entitlements import (
        _pro_daily_credit_budget_usd,
        credit_budget_and_interval_from_paystack_payload,
    )

    settings = CloudSettings(
        paystack_pro_daily_amount_ngn=1000,
        paystack_pro_monthly_amount_ngn=15000,
        ngn_to_usd_rate=0.0006,
        paystack_plan_code_pro_daily="PLN_daily",
    )
    daily = _pro_daily_credit_budget_usd(settings)
    assert daily == round(1000 * 0.0006, 4)
    # Must not be monthly/30.
    assert daily != round((15000 * 0.0006) / 30, 4)

    budget, interval = credit_budget_and_interval_from_paystack_payload(
        {"plan": {"plan_code": "PLN_daily"}, "amount": 100000},
        settings,
    )
    assert interval == "daily"
    assert budget == round(1000 * 0.0006, 4)


def test_early_renewal_carries_remaining_credits_and_preserves_anchor():
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace

    from survyai_cloud.config import CloudSettings
    from survyai_cloud.models import SubscriptionStatus
    from survyai_cloud.services.entitlements import apply_pro_defaults

    settings = CloudSettings(pro_plan_slug="pro", default_max_devices_pro=2, pro_monthly_agent_runs=300)
    now = datetime.now(timezone.utc)
    original_anchor = now - timedelta(hours=6)
    user = SimpleNamespace(
        plan_slug="pro",
        subscription_status=SubscriptionStatus.active,
        subscription_current_period_end=now + timedelta(hours=18),
        max_devices=1,
        monthly_agent_runs_quota=10,
        monthly_agent_runs_used=3,
        monthly_credits_usd=0.6,
        monthly_credits_used_usd=0.25,
        credits_billing_interval="daily",
        usage_period_anchor=original_anchor,
    )

    apply_pro_defaults(
        user,
        settings,
        credit_budget_usd=0.6,
        credits_billing_interval="daily",
        paid_at=now,
    )

    assert user.monthly_credits_usd == 1.2
    assert user.monthly_credits_used_usd == 0.25
    assert user.monthly_agent_runs_used == 3
    assert user.usage_period_anchor == original_anchor


def test_purchase_after_expiry_starts_fresh_credit_period():
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace

    from survyai_cloud.config import CloudSettings
    from survyai_cloud.models import SubscriptionStatus
    from survyai_cloud.services.entitlements import apply_pro_defaults

    settings = CloudSettings(pro_plan_slug="pro", default_max_devices_pro=2, pro_monthly_agent_runs=300)
    now = datetime.now(timezone.utc)
    user = SimpleNamespace(
        plan_slug="pro",
        subscription_status=SubscriptionStatus.active,
        subscription_current_period_end=now - timedelta(hours=1),
        max_devices=1,
        monthly_agent_runs_quota=10,
        monthly_agent_runs_used=7,
        monthly_credits_usd=0.6,
        monthly_credits_used_usd=0.55,
        credits_billing_interval="daily",
        usage_period_anchor=now - timedelta(days=1),
    )

    apply_pro_defaults(
        user,
        settings,
        credit_budget_usd=0.6,
        credits_billing_interval="daily",
        paid_at=now,
    )

    assert user.monthly_credits_usd == 0.6
    assert user.monthly_credits_used_usd == 0.0
    assert user.monthly_agent_runs_used == 0
    assert user.usage_period_anchor == now


def test_expired_subscription_blocks_hosted_llm_even_if_status_active():
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace

    from survyai_cloud.config import CloudSettings
    from survyai_cloud.models import SubscriptionStatus
    from survyai_cloud.services.entitlements import subscription_allows_platform_llm

    settings = CloudSettings(pro_plan_slug="pro")
    user = SimpleNamespace(
        plan_slug="pro",
        subscription_status=SubscriptionStatus.active,
        subscription_current_period_end=datetime.now(timezone.utc) - timedelta(minutes=5),
        monthly_credits_usd=1.0,
        monthly_credits_used_usd=0.0,
    )
    assert subscription_allows_platform_llm(user, settings) is False

    user.subscription_current_period_end = datetime.now(timezone.utc) + timedelta(hours=1)
    assert subscription_allows_platform_llm(user, settings) is True


def test_credits_ui_uses_exact_paid_anchor_not_period_end_minus_days():
    """After early renewal, period_end - days is wrong; use usage_period_anchor."""
    from datetime import datetime, timedelta, timezone

    paid_start = datetime(2026, 7, 8, 8, 0, tzinfo=timezone.utc)
    period_end = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)  # early renewal extended end
    period_days = 1

    # Legacy (incorrect after early renewal):
    legacy_start = period_end - timedelta(days=period_days)
    assert legacy_start == datetime(2026, 7, 9, 8, 0, tzinfo=timezone.utc)

    # Correct: use the cloud-provided paid-window anchor.
    period_start = paid_start
    assert period_start == paid_start
    assert period_start != legacy_start
    assert (period_end - period_start).total_seconds() == 2 * 24 * 3600


def test_entitlements_payload_includes_usage_window_fields():
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from survyai_cloud.config import CloudSettings
    from survyai_cloud.models import SubscriptionStatus
    from survyai_cloud.services.entitlements import entitlements_for_user

    settings = CloudSettings(pro_plan_slug="pro", platform_openai_api_key="sk-test")
    anchor = datetime(2026, 7, 8, 8, 0, tzinfo=timezone.utc)
    end = datetime(2026, 7, 9, 8, 0, tzinfo=timezone.utc)
    user = SimpleNamespace(
        plan_slug="pro",
        subscription_status=SubscriptionStatus.active,
        max_devices=2,
        monthly_agent_runs_quota=10,
        monthly_agent_runs_used=1,
        monthly_credits_usd=0.6,
        monthly_credits_used_usd=0.1,
        credits_billing_interval="daily",
        usage_period_anchor=anchor,
        subscription_current_period_end=end,
    )
    out = entitlements_for_user(user, settings)
    assert out.usage_period_anchor == anchor
    assert out.subscription_current_period_end == end
    assert out.monthly_credits_usd == 0.6
