from __future__ import annotations

import asyncio
from typing import Any

from fastapi import HTTPException, status
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from survyai_cloud.config import CloudSettings, get_cloud_settings
from survyai_cloud.models import Device, UsageEvent, User
from survyai_cloud.schemas import LlmMessageIn, LlmProxyChatIn, LlmProxyChatOut, LlmToolCallOut
from survyai_cloud.services.entitlements import (
    ensure_usage_month_rolled,
    has_platform_credit_remaining,
    resolve_platform_llm_provider,
    subscription_allows_platform_llm,
)
from utils.cost_estimator import estimate_token_cost_usd, estimate_tokens, extract_message_token_usage


async def run_proxy_chat(
    *,
    body: LlmProxyChatIn,
    user: User,
    device: Device | None,
    db: AsyncSession,
    settings: CloudSettings | None = None,
) -> LlmProxyChatOut:
    settings = settings or get_cloud_settings()
    await ensure_usage_month_rolled(user, db)

    if not subscription_allows_platform_llm(user, settings):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Active Pro subscription required for hosted SurvyAI LLM access.",
        )
    if device is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Hosted SurvyAI LLM calls require a registered PC: "
                "set header X-SurvyAI-Device-Id to the id returned by POST /v1/devices."
            ),
        )
    if not has_platform_credit_remaining(user, settings):
        raise HTTPException(
            status_code=402,
            detail="Subscription API credit balance exhausted for this period",
        )

    llm, resolved_model, resolved_provider = _build_server_chat_model(body, settings)
    llm_to_invoke = llm.bind_tools(list(body.tools or [])) if body.tools else llm
    messages = [_to_langchain_message(msg) for msg in body.messages]
    response = await asyncio.to_thread(llm_to_invoke.invoke, messages)
    ai_message = response if isinstance(response, AIMessage) else AIMessage(content=str(response))

    usage = _usage_dict_from_message(ai_message, body)
    billed_cost_usd = round(
        estimate_token_cost_usd(
            str(
                resolved_model
                or body.model
                or ai_message.response_metadata.get("model_name")
                or ""
            ),
            int(usage.get("input_tokens") or 0),
            int(usage.get("output_tokens") or 0),
            cached_input_tokens=int(usage.get("cached_input_tokens") or 0),
        ),
        6,
    )
    markup_cost_usd = round(billed_cost_usd * settings.credit_markup_multiplier, 6)
    budget = float(user.monthly_credits_usd or 0.0)
    used_before = float(user.monthly_credits_used_usd or 0.0)
    used_after = used_before
    if budget > 0 and markup_cost_usd > 0:
        used_after = round(min(budget, used_before + markup_cost_usd), 6)
        user.monthly_credits_used_usd = used_after
        db.add(user)

    event_meta = {
        "provider": resolved_provider,
        "model": str(resolved_model or body.model or ""),
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "cached_input_tokens": int(usage.get("cached_input_tokens") or 0),
        "billing_basis": "llm_proxy_server_usage",
        "markup_cost_usd": markup_cost_usd,
    }
    db.add(
        UsageEvent(
            user_id=user.id,
            kind="llm_proxy_turn",
            quantity=1,
            cost_usd=billed_cost_usd,
            meta=event_meta,
            device_id=device.id,
        )
    )

    tool_calls: list[LlmToolCallOut] = []
    for tc in ai_message.tool_calls or []:
        if not isinstance(tc, dict):
            continue
        tool_calls.append(
            LlmToolCallOut(
                id=tc.get("id"),
                name=str(tc.get("name") or ""),
                args=tc.get("args") if isinstance(tc.get("args"), dict) else {},
            )
        )

    billing = {
        "cost_usd": billed_cost_usd,
        "markup_cost_usd": markup_cost_usd,
        "monthly_credits_used_usd": round(float(user.monthly_credits_used_usd or 0.0), 6),
        "monthly_credits_usd": budget,
        "credit_exhausted": bool(budget > 0 and user.monthly_credits_used_usd >= budget - 1e-6),
    }
    usage["cost_usd"] = billed_cost_usd
    return LlmProxyChatOut(
        provider=resolved_provider,
        model=str(resolved_model or body.model or ""),
        content=ai_message.content,
        tool_calls=tool_calls,
        usage=usage,
        billing=billing,
    )


def _build_server_chat_model(body: LlmProxyChatIn, settings: CloudSettings) -> tuple[Any, str, str]:
    provider = str(body.provider or "").strip().lower()
    requested_provider = provider
    if provider and not _provider_has_platform_key(provider, settings):
        fallback = resolve_platform_llm_provider(settings)
        if fallback != provider and _provider_has_platform_key(fallback, settings):
            provider = fallback
    model = str(body.model or "").strip()
    if provider != requested_provider:
        model = ""
    if provider == "openai":
        if not settings.platform_openai_api_key.strip():
            raise HTTPException(status_code=503, detail="Server missing platform OpenAI configuration")
        resolved_model = model or settings.platform_openai_model
        return ChatOpenAI(
            model=resolved_model,
            api_key=settings.platform_openai_api_key,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        ), resolved_model, provider
    if provider == "deepseek":
        if not settings.platform_deepseek_api_key.strip():
            raise HTTPException(status_code=503, detail="Server missing platform DeepSeek configuration")
        resolved_model = model or "deepseek-chat"
        return ChatOpenAI(
            model=resolved_model,
            api_key=settings.platform_deepseek_api_key,
            base_url=settings.platform_deepseek_base_url,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        ), resolved_model, provider
    if provider == "claude":
        if not settings.platform_anthropic_api_key.strip():
            raise HTTPException(status_code=503, detail="Server missing platform Anthropic configuration")
        resolved_model = model or settings.platform_claude_model
        return ChatAnthropic(
            model=resolved_model,
            anthropic_api_key=settings.platform_anthropic_api_key,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        ), resolved_model, provider
    if provider == "gemini":
        if not settings.platform_google_api_key.strip():
            raise HTTPException(status_code=503, detail="Server missing platform Google configuration")
        resolved_model = model or settings.platform_gemini_model
        return ChatGoogleGenerativeAI(
            model=resolved_model,
            google_api_key=settings.platform_google_api_key,
            temperature=body.temperature,
            max_output_tokens=body.max_tokens,
        ), resolved_model, provider
    raise HTTPException(status_code=400, detail=f"Unsupported provider: {body.provider}")


def _provider_has_platform_key(provider: str, settings: CloudSettings) -> bool:
    provider = str(provider or "").strip().lower()
    if provider == "openai":
        return bool(settings.platform_openai_api_key.strip())
    if provider == "claude":
        return bool(settings.platform_anthropic_api_key.strip())
    if provider == "gemini":
        return bool(settings.platform_google_api_key.strip())
    if provider == "deepseek":
        return bool(settings.platform_deepseek_api_key.strip())
    return False


def _to_langchain_message(msg: LlmMessageIn) -> Any:
    role = str(msg.role or "").strip().lower()
    if role == "system":
        return SystemMessage(content=msg.content)
    if role == "assistant":
        return AIMessage(
            content=msg.content,
            tool_calls=[
                {
                    "id": tc.id,
                    "name": tc.name,
                    "args": dict(tc.args or {}),
                    "type": "tool_call",
                }
                for tc in (msg.tool_calls or [])
            ],
        )
    if role == "tool":
        return ToolMessage(
            content=msg.content,
            tool_call_id=str(msg.tool_call_id or ""),
        )
    return HumanMessage(content=msg.content)


def _usage_dict_from_message(message: AIMessage, body: LlmProxyChatIn) -> dict[str, Any]:
    usage = extract_message_token_usage(message)
    if usage:
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": int(usage.get("total_tokens") or (input_tokens + output_tokens)),
            "cached_input_tokens": int(usage.get("cached_input_tokens") or 0),
        }
    input_tokens = 0
    for msg in body.messages:
        input_tokens += _estimate_payload_tokens(msg.content)
    output_tokens = _estimate_payload_tokens(message.content)
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(input_tokens + output_tokens),
        "cached_input_tokens": 0,
        "estimated": True,
    }


def _estimate_payload_tokens(payload: Any) -> int:
    if isinstance(payload, str):
        return estimate_tokens(payload, method="characters")
    if isinstance(payload, list):
        total = 0
        for item in payload:
            total += _estimate_payload_tokens(item)
        return total
    if isinstance(payload, dict):
        return estimate_tokens(str(payload), method="characters")
    return estimate_tokens(str(payload or ""), method="characters")


__all__ = ["run_proxy_chat"]
