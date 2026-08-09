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
    reconcile_pro_access,
    resolve_platform_llm_provider,
    subscription_allows_platform_llm,
)
from utils.cost_estimator import estimate_token_cost_usd, estimate_tokens, extract_message_token_usage

# Process-level Chat* client cache (provider, model, temp, max_tokens) → client.
# Safe: platform keys are stable for the cloud process lifetime; tools are bound per request.
_SERVER_CHAT_MODEL_CACHE: dict[tuple, Any] = {}


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

    try:
        llm, resolved_model, resolved_provider = _build_server_chat_model(body, settings)
        llm_to_invoke = llm.bind_tools(list(body.tools or [])) if body.tools else llm
        messages = [_to_langchain_message(msg) for msg in body.messages]
        try:
            response = await asyncio.to_thread(llm_to_invoke.invoke, messages)
        except HTTPException:
            raise
        except Exception as exc:
            raise _http_error_from_upstream_llm(exc) from exc

        ai_message = response if isinstance(response, AIMessage) else AIMessage(content=str(response))

        usage = _usage_dict_from_message(ai_message, body)
        usage_is_reported = not bool(usage.get("estimated"))
        billed_cost_usd = 0.0
        if usage_is_reported:
            billed_cost_usd = round(
                estimate_token_cost_usd(
                    str(
                        resolved_model
                        or body.model
                        or (ai_message.response_metadata or {}).get("model_name")
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
        credit_exhausted = bool(budget > 0 and used_before >= budget - 1e-6)
        if budget > 0 and markup_cost_usd > 0:
            used_after = round(min(budget, used_before + markup_cost_usd), 6)
            user.monthly_credits_used_usd = used_after
            credit_exhausted = used_after >= budget - 1e-6
            db.add(user)
            # Flip Pro → Free as soon as the pool is spent so /me and desktop agree.
            reconcile_pro_access(user, settings, db=db)

        event_meta = {
            "provider": resolved_provider,
            "model": str(resolved_model or body.model or ""),
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "cached_input_tokens": int(usage.get("cached_input_tokens") or 0),
            "billing_basis": "llm_proxy_provider_reported_usage"
            if usage_is_reported
            else "llm_proxy_estimated_usage_not_billed",
            "markup_cost_usd": markup_cost_usd,
            "usage_estimated": bool(usage.get("estimated")),
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

        tool_calls = _normalize_tool_calls(ai_message)
        safe_content = _json_safe_content(ai_message.content)

        billing = {
            "cost_usd": billed_cost_usd,
            "markup_cost_usd": markup_cost_usd,
            "monthly_credits_used_usd": round(float(used_after), 6),
            "monthly_credits_usd": budget,
            "credit_exhausted": credit_exhausted,
        }
        usage["cost_usd"] = billed_cost_usd
        return LlmProxyChatOut(
            provider=resolved_provider,
            model=str(resolved_model or body.model or ""),
            content=safe_content,
            tool_calls=tool_calls,
            usage=usage,
            billing=billing,
        )
    except HTTPException:
        raise
    except Exception as exc:
        # Never leak a plain-text/HTML 500 to the desktop — always JSON detail.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Hosted LLM proxy failed: {str(exc)[:800]}",
        ) from exc


def _http_error_from_upstream_llm(exc: Exception) -> HTTPException:
    err = str(exc or "").strip() or exc.__class__.__name__
    low = err.lower()
    if any(k in low for k in ("rate limit", "rate_limit", "429", "too many requests")):
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Upstream LLM rate limited: {err[:700]}",
        )
    if any(k in low for k in ("unauthorized", "invalid api key", "incorrect api key", "401", "auth")):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Upstream LLM authentication failed on the SurvyAI server: {err[:700]}",
        )
    if any(
        k in low
        for k in (
            "context length",
            "maximum context",
            "token limit",
            "too many tokens",
            "max_tokens",
            "context_window",
            "request too large",
        )
    ):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Upstream LLM rejected the request size/context: {err[:700]}",
        )
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=f"Upstream LLM call failed: {err[:700]}",
    )


def _json_safe_content(content: Any) -> Any:
    """Coerce LangChain content (incl. content blocks) into JSON-safe values."""
    if content is None:
        return ""
    if isinstance(content, (str, int, float, bool)):
        return content
    if isinstance(content, list):
        out: list[Any] = []
        for item in content:
            if isinstance(item, (str, int, float, bool)) or item is None:
                out.append(item)
            elif isinstance(item, dict):
                out.append(item)
            elif hasattr(item, "model_dump"):
                try:
                    out.append(item.model_dump())
                    continue
                except Exception:
                    pass
            out.append(str(item))
        return out
    if isinstance(content, dict):
        return content
    if hasattr(content, "model_dump"):
        try:
            return content.model_dump()
        except Exception:
            pass
    return str(content)


def _normalize_tool_calls(ai_message: AIMessage) -> list[LlmToolCallOut]:
    """Accept dict tool_calls and LangChain tool-call objects."""
    tool_calls: list[LlmToolCallOut] = []
    for tc in getattr(ai_message, "tool_calls", None) or []:
        if isinstance(tc, dict):
            name = str(tc.get("name") or "").strip()
            args = tc.get("args") if isinstance(tc.get("args"), dict) else {}
            tid = tc.get("id")
        else:
            name = str(getattr(tc, "name", "") or "").strip()
            raw_args = getattr(tc, "args", None)
            args = raw_args if isinstance(raw_args, dict) else {}
            tid = getattr(tc, "id", None)
        if not name:
            continue
        tool_calls.append(LlmToolCallOut(id=tid, name=name, args=args))
    return tool_calls


def _server_chat_cache_key(
    provider: str,
    resolved_model: str,
    *,
    temperature: float | None,
    max_tokens: int | None,
) -> tuple:
    return (
        str(provider or "").strip().lower(),
        str(resolved_model or "").strip(),
        float(temperature if temperature is not None else 0.0),
        int(max_tokens or 0),
    )


def _get_or_create_server_chat_model(
    *,
    provider: str,
    resolved_model: str,
    body: LlmProxyChatIn,
    settings: CloudSettings,
) -> Any:
    """Reuse Chat* clients across proxy requests with identical provider/model/temp/max_tokens."""
    cache_key = _server_chat_cache_key(
        provider,
        resolved_model,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
    )
    cached = _SERVER_CHAT_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if provider == "openai":
        llm = ChatOpenAI(
            model=resolved_model,
            api_key=settings.platform_openai_api_key,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
    elif provider == "deepseek":
        llm = ChatOpenAI(
            model=resolved_model,
            api_key=settings.platform_deepseek_api_key,
            base_url=settings.platform_deepseek_base_url,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
    elif provider == "claude":
        llm = ChatAnthropic(
            model=resolved_model,
            anthropic_api_key=settings.platform_anthropic_api_key,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
    elif provider == "gemini":
        llm = ChatGoogleGenerativeAI(
            model=resolved_model,
            google_api_key=settings.platform_google_api_key,
            temperature=body.temperature,
            max_output_tokens=body.max_tokens,
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")

    _SERVER_CHAT_MODEL_CACHE[cache_key] = llm
    return llm


def clear_server_chat_model_cache() -> None:
    """Test helper: drop process-level Chat* cache."""
    _SERVER_CHAT_MODEL_CACHE.clear()


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
        if model:
            resolved_model = model
        else:
            try:
                from survyai.openai_models import migrate_legacy_platform_model

                resolved_model = migrate_legacy_platform_model(
                    "openai_model", settings.platform_openai_model
                ) or settings.platform_openai_model
            except Exception:
                resolved_model = settings.platform_openai_model
        return (
            _get_or_create_server_chat_model(
                provider=provider,
                resolved_model=resolved_model,
                body=body,
                settings=settings,
            ),
            resolved_model,
            provider,
        )
    if provider == "deepseek":
        if not settings.platform_deepseek_api_key.strip():
            raise HTTPException(status_code=503, detail="Server missing platform DeepSeek configuration")
        resolved_model = model or "deepseek-chat"
        return (
            _get_or_create_server_chat_model(
                provider=provider,
                resolved_model=resolved_model,
                body=body,
                settings=settings,
            ),
            resolved_model,
            provider,
        )
    if provider == "claude":
        if not settings.platform_anthropic_api_key.strip():
            raise HTTPException(status_code=503, detail="Server missing platform Anthropic configuration")
        resolved_model = model or settings.platform_claude_model
        return (
            _get_or_create_server_chat_model(
                provider=provider,
                resolved_model=resolved_model,
                body=body,
                settings=settings,
            ),
            resolved_model,
            provider,
        )
    if provider == "gemini":
        if not settings.platform_google_api_key.strip():
            raise HTTPException(status_code=503, detail="Server missing platform Google configuration")
        resolved_model = model or settings.platform_gemini_model
        return (
            _get_or_create_server_chat_model(
                provider=provider,
                resolved_model=resolved_model,
                body=body,
                settings=settings,
            ),
            resolved_model,
            provider,
        )
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
    content = msg.content if msg.content is not None else ""
    if role == "system":
        return SystemMessage(content=content)
    if role == "assistant":
        tool_calls = []
        for tc in msg.tool_calls or []:
            name = str(getattr(tc, "name", "") or "").strip()
            if not name:
                continue
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": name,
                    "args": dict(tc.args or {}),
                    "type": "tool_call",
                }
            )
        return AIMessage(content=content, tool_calls=tool_calls)
    if role == "tool":
        return ToolMessage(
            content=content if isinstance(content, str) else str(content),
            tool_call_id=str(msg.tool_call_id or ""),
        )
    return HumanMessage(content=content)


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
