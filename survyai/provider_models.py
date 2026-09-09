"""
Multi-provider complexity → model routing for SurvyAI.

Paid providers (OpenAI, Claude, Gemini, DeepSeek) share the same three
complexity buckets used by the agent:

- simple  → cheapest accurate model (lookups, short historic Q&A)
- average → balanced model (typical GIS/CAD orchestration)
- complex → strongest model (hard multi-step / unordered geospatial work)

OpenAI also supports an *elevated average* pick (gpt-5.5) when the task is
medium–high reasoning without full complex-tier signals.

``enable_tiered_models=False`` keeps legacy single-model settings per provider.

When ``enable_llm_prompt_router`` is True (default), the cheapest paid model
for the active provider classifies the prompt first; heuristic buckets remain
the fallback if that classifier fails.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional, Sequence

from survyai.openai_models import (
    DEFAULT_MODEL_FOR_COMPLEXITY,
    chat_openai_official_kwargs,
    next_fallback_model as openai_next_fallback_model,
    resolve_model_for_complexity as resolve_openai_model_for_complexity,
)

Complexity = Literal["simple", "average", "complex"]
PaidProvider = Literal["openai", "claude", "gemini", "deepseek"]

PAID_PROVIDERS = frozenset({"openai", "claude", "gemini", "deepseek"})

# Default Low / Medium / Advanced models per provider (exact API ids).
PROVIDER_TIER_DEFAULTS: Dict[str, Dict[Complexity, str]] = {
    "openai": {
        "simple": DEFAULT_MODEL_FOR_COMPLEXITY["simple"],  # gpt-5.6-luna
        "average": DEFAULT_MODEL_FOR_COMPLEXITY["average"],  # gpt-5.6-terra
        "complex": DEFAULT_MODEL_FOR_COMPLEXITY["complex"],  # gpt-5.6-sol
    },
    "claude": {
        "simple": "claude-3-5-haiku-20241022",
        "average": "claude-3-5-sonnet-20241022",
        "complex": "claude-3-opus-20240229",
    },
    "gemini": {
        "simple": "gemini-1.5-flash",
        "average": "gemini-2.0-flash",
        "complex": "gemini-pro-latest",
    },
    "deepseek": {
        "simple": "deepseek-chat",
        "average": "deepseek-chat",
        "complex": "deepseek-reasoner",
    },
}

# Medium–high OpenAI reasoning (compare / multi-hop) without full complex tier.
OPENAI_ELEVATED_AVERAGE_MODEL = "gpt-5.5"

# Ordered failover within a provider after quota / upstream failure.
PROVIDER_FALLBACK_CHAINS: Dict[str, Dict[str, tuple[str, ...]]] = {
    "claude": {
        "claude-3-opus-20240229": (
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
        ),
        "claude-3-5-sonnet-20241022": (
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ),
        "claude-3-5-haiku-20241022": (
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
        ),
    },
    "gemini": {
        "gemini-pro-latest": ("gemini-2.0-flash", "gemini-1.5-flash"),
        "gemini-2.0-flash": ("gemini-1.5-flash", "gemini-pro-latest"),
        "gemini-1.5-flash": ("gemini-2.0-flash", "gemini-pro-latest"),
    },
    "deepseek": {
        "deepseek-reasoner": ("deepseek-chat",),
        "deepseek-chat": ("deepseek-reasoner",),
    },
}


def _settings_get(settings: Any, key: str, default: Optional[str] = None) -> Optional[str]:
    if settings is None:
        return default
    val = getattr(settings, key, default)
    if val is None:
        return default
    text = str(val).strip()
    return text or default


def legacy_single_model(provider: str, settings: Any = None) -> str:
    """Single-model legacy id when tiered selection is disabled."""
    p = str(provider or "").strip().lower()
    defaults = PROVIDER_TIER_DEFAULTS.get(p, {})
    if p == "openai":
        return _settings_get(settings, "openai_model", defaults.get("average", "gpt-5.6-terra")) or "gpt-5.6-terra"
    if p == "claude":
        return (
            _settings_get(settings, "claude_model", defaults.get("average", "claude-3-5-sonnet-20241022"))
            or "claude-3-5-sonnet-20241022"
        )
    if p == "gemini":
        return _settings_get(settings, "gemini_model", defaults.get("average", "gemini-2.0-flash")) or "gemini-2.0-flash"
    if p == "deepseek":
        return _settings_get(settings, "deepseek_model", defaults.get("average", "deepseek-chat")) or "deepseek-chat"
    return p or "unknown"


def resolve_provider_model_for_complexity(
    provider: str,
    complexity: Complexity,
    *,
    settings: Any = None,
    enable_tiered: Optional[bool] = None,
    elevated_average: bool = False,
) -> str:
    """
    Pick the exact model id for ``provider`` + ``complexity``.

    OpenAI uses the existing catalog/settings nano/mini/complex slots.
    Other paid providers use :data:`PROVIDER_TIER_DEFAULTS`, with the
    provider's legacy single setting overriding the **average** slot when set
    (so existing CLAUDE_MODEL / GEMINI_MODEL env values keep working).
    """
    p = str(provider or "").strip().lower()
    if enable_tiered is None:
        enable_tiered = bool(getattr(settings, "enable_tiered_models", True)) if settings is not None else True

    if p not in PAID_PROVIDERS:
        return legacy_single_model(p, settings)

    if not enable_tiered:
        return legacy_single_model(p, settings)

    c: Complexity = complexity if complexity in ("simple", "average", "complex") else "average"

    if p == "openai":
        if c == "average" and elevated_average:
            return OPENAI_ELEVATED_AVERAGE_MODEL
        return resolve_openai_model_for_complexity(
            c,
            nano=_settings_get(settings, "openai_model_nano"),
            mini=_settings_get(settings, "openai_model_mini"),
            complex_model=_settings_get(settings, "openai_model_complex"),
            legacy=_settings_get(settings, "openai_model"),
        )

    defaults = PROVIDER_TIER_DEFAULTS[p]
    # Optional future overrides: claude_model_nano / _mini / _complex, etc.
    nano = _settings_get(settings, f"{p}_model_nano")
    mini = _settings_get(settings, f"{p}_model_mini")
    complex_m = _settings_get(settings, f"{p}_model_complex")
    # Existing single-model env wins for the average (balanced) slot.
    legacy_avg = legacy_single_model(p, settings)

    mapping = {
        "simple": nano or defaults["simple"],
        "average": mini or legacy_avg or defaults["average"],
        "complex": complex_m or defaults["complex"],
    }
    return mapping[c]


def next_provider_failover_model(
    provider: str,
    current_model: Optional[str],
    *,
    complexity: Complexity = "average",
    tried: Optional[Sequence[str]] = None,
    settings: Any = None,
    elevated_average: bool = False,
) -> Optional[str]:
    """Next unused model for the same provider after quota / upstream failure."""
    p = str(provider or "").strip().lower()
    if p == "openai":
        return openai_next_fallback_model(
            current_model,
            complexity=complexity,
            tried=tried,
        )

    excluded = {str(x).strip().lower() for x in (tried or []) if x}
    cur = str(current_model or "").strip()
    if cur:
        excluded.add(cur.lower())

    chain: List[str] = []
    prov_chains = PROVIDER_FALLBACK_CHAINS.get(p) or {}
    if cur and cur in prov_chains:
        chain.extend(prov_chains[cur])
    # Prefer tier neighbors for this complexity, then all catalog defaults.
    for tier in ("simple", "average", "complex"):
        chain.append(
            resolve_provider_model_for_complexity(
                p,
                tier,  # type: ignore[arg-type]
                settings=settings,
                elevated_average=elevated_average and tier == "average",
            )
        )
    for mid in PROVIDER_TIER_DEFAULTS.get(p, {}).values():
        chain.append(mid)

    seen: set[str] = set()
    for mid in chain:
        key = str(mid or "").strip().lower()
        if not key or key in excluded or key in seen:
            continue
        seen.add(key)
        return mid
    return None


def escalate_provider_from_model(
    provider: str,
    current_model: Optional[str],
    *,
    settings: Any = None,
) -> Optional[str]:
    """Escalate one tier above whatever ``current_model`` maps to for ``provider``."""
    p = str(provider or "").strip().lower()
    cur = str(current_model or "").strip().lower()
    defaults = PROVIDER_TIER_DEFAULTS.get(p) or {}
    if p == "openai":
        from survyai.openai_models import escalate_tier_model, infer_tier

        return escalate_tier_model(
            infer_tier(current_model),
            mini=_settings_get(settings, "openai_model_mini"),
            complex_model=_settings_get(settings, "openai_model_complex"),
        )

    current_c: Complexity = "average"
    for c, mid in defaults.items():
        if str(mid).lower() == cur:
            current_c = c  # type: ignore[assignment]
            break
    else:
        # Heuristic when the live model id is not exactly a catalog default.
        if any(k in cur for k in ("haiku", "flash-lite", "nano")):
            current_c = "simple"
        elif any(k in cur for k in ("opus", "reasoner", "pro", "ultra")):
            current_c = "complex"
        else:
            current_c = "average"
    order: List[Complexity] = ["simple", "average", "complex"]
    idx = order.index(current_c)
    if idx >= len(order) - 1:
        return None
    return resolve_provider_model_for_complexity(
        p, order[idx + 1], settings=settings, enable_tiered=True
    )


def provider_tier_summary(provider: str, settings: Any = None) -> Dict[str, str]:
    """Low/medium/advanced ids for UI (Settings / Credits)."""
    p = str(provider or "").strip().lower()
    if p not in PAID_PROVIDERS:
        single = legacy_single_model(p, settings)
        return {"low": single, "medium": single, "advanced": single}
    return {
        "low": resolve_provider_model_for_complexity(p, "simple", settings=settings),
        "medium": resolve_provider_model_for_complexity(p, "average", settings=settings),
        "advanced": resolve_provider_model_for_complexity(p, "complex", settings=settings),
    }


def is_elevated_average_task(query: str) -> bool:
    """
    True when the prompt looks medium–high reasoning (prefer gpt-5.5 on OpenAI)
    without full complex-tier GIS/raster signals.
    """
    ql = (query or "").lower()
    if not ql.strip():
        return False
    elevated_signals = (
        "compare",
        "which is more correct",
        "which is correct",
        "why do they differ",
        "trade-off",
        "tradeoff",
        "pros and cons",
        "multi-hop",
        "step by step reasoning",
        "reconcile",
        "cross-check",
        "cross check",
        "medium-high",
        "medium to high",
    )
    return any(s in ql for s in elevated_signals)


# Providers whose hosted APIs accept OpenAI-style image_url multimodal content
# in the SurvyAI desktop → proxy → provider path.
VISION_CAPABLE_PROVIDERS = frozenset({"openai", "claude", "gemini"})


def provider_supports_vision(provider: str) -> bool:
    """True when the provider can run LLM vision / OCR for image inputs."""
    return str(provider or "").strip().lower() in VISION_CAPABLE_PROVIDERS


def vision_unsupported_user_message(provider: str) -> str:
    """Clear, non-silent message when the active provider cannot do vision OCR."""
    p = str(provider or "").strip().lower() or "the selected provider"
    return (
        f"Vision OCR is not available with {p}. "
        "Switch your primary LLM to OpenAI, Claude, or Gemini to scan images "
        "(.png, .jpg, …) and extract text or geospatial plan components. "
        "Typed file paths and document tools for PDF/Word are unchanged."
    )


def claude_extended_thinking_kwargs(model: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
    """Enable extended thinking on Claude models that support it with tools.

    Claude 3 / 3.5 do not support this; leaving them unchanged avoids new errors.
    Temperature is forced to 1 (Anthropic requirement). Effort is never disabled.
    """
    m = str(model or "").strip().lower()
    if not m or any(
        tag in m
        for tag in (
            "claude-3-5",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku",
        )
    ):
        return {}
    if not any(
        tag in m
        for tag in ("3-7", "3.7", "sonnet-4", "opus-4", "haiku-4", "claude-4")
    ):
        return {}
    cap = int(max_tokens or 0) or 8192
    budget = min(16000, max(2048, cap // 2))
    if budget >= cap:
        cap = budget + 2048
    return {
        "thinking": {"type": "enabled", "budget_tokens": budget},
        "temperature": 1,
        "max_tokens": cap,
    }


def gemini_thinking_kwargs(model: str) -> Dict[str, Any]:
    """Highest supported thinking for Gemini 2.5 / 3.x. Never sets budget to 0."""
    m = str(model or "").strip().lower()
    if any(tag in m for tag in ("2.5", "gemini-3", "flash-thinking")):
        return {"thinking_level": "high"}
    return {}


def paid_llm_constructor_kwargs(
    provider: str,
    model: str,
    *,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Extra Chat* kwargs so tools and reasoning work together.

    DeepSeek / Ollama stay empty: they speak OpenAI chat completions and have
    no /v1/responses. Official OpenAI reasoning models use the Responses API
    with effort left on.
    """
    p = str(provider or "").strip().lower()
    if p == "openai":
        return dict(chat_openai_official_kwargs(model))
    if p == "claude":
        return claude_extended_thinking_kwargs(model, max_tokens)
    if p == "gemini":
        return gemini_thinking_kwargs(model)
    return {}


_SKIP_CONTENT_BLOCK_TYPES = frozenset(
    {
        "reasoning",
        "thinking",
        "redacted_thinking",
        "function_call",
        "tool_use",
        "tool_call",
        "web_search_call",
        "file_search_call",
        "computer_call",
        "code_interpreter_call",
        "mcp_call",
        "mcp_list_tools",
        "mcp_approval_request",
        "image_generation_call",
    }
)


def _is_reasoning_payload(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    kind = str(obj.get("type") or "").strip().lower()
    if kind in {"reasoning", "thinking", "redacted_thinking"}:
        return True
    if obj.get("encrypted_content") and str(obj.get("id") or "").startswith("rs_"):
        return True
    return False


def _text_from_content_block(block: Any) -> str:
    if isinstance(block, str):
        return block.strip()
    if not isinstance(block, dict) or _is_reasoning_payload(block):
        if isinstance(block, dict):
            summary = block.get("summary")
            if isinstance(summary, list):
                parts = [
                    str(item.get("text") or "").strip()
                    if isinstance(item, dict)
                    else str(item).strip()
                    for item in summary
                ]
                return "\n".join(p for p in parts if p)
        return ""
    kind = str(block.get("type") or "").strip().lower()
    if kind in _SKIP_CONTENT_BLOCK_TYPES:
        return ""
    for key in ("text", "output_text", "refusal"):
        val = block.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    inner = block.get("content")
    if isinstance(inner, str) and inner.strip():
        return inner.strip()
    if isinstance(inner, list):
        return llm_visible_text_from_content(inner)
    return ""


def llm_visible_text_from_content(content: Any) -> str:
    """User-visible model text; skips Responses-API reasoning / encrypted blobs."""
    if content is None:
        return ""
    if isinstance(content, str):
        raw = content.strip()
        if not raw:
            return ""
        if ("encrypted_content" in raw[:1200] and "reasoning" in raw[:400]) and not any(
            key in raw for key in ("traverse_legs", "pillar_numbers", "buyer_name", "plan_number")
        ):
            return ""
        return raw
    if isinstance(content, dict):
        text = _text_from_content_block(content)
        if text:
            return text
        if _is_reasoning_payload(content):
            return ""
        kind = str(content.get("type") or "").strip().lower()
        if kind in _SKIP_CONTENT_BLOCK_TYPES or kind in {"text", "output_text", "refusal"}:
            return ""
        try:
            return json.dumps(content)
        except Exception:
            return ""
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(llm_visible_text_from_content(item))
            else:
                parts.append(_text_from_content_block(item))
        return "\n".join(p for p in parts if p).strip()
    if hasattr(content, "model_dump"):
        try:
            return llm_visible_text_from_content(content.model_dump())
        except Exception:
            pass
    return str(content).strip()


def _balance_truncated_json(s: str) -> str:
    """Close an unfinished JSON string and any open [ / { so json.loads can run."""
    in_str = False
    escape = False
    stack: List[str] = []
    for ch in s:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif stack and ch == stack[-1]:
            stack.pop()
    out = s
    if in_str:
        if out.endswith("\\") and not out.endswith("\\\\"):
            out = out[:-1]
        out += '"'
    out = out.rstrip().rstrip(",")
    for closer in reversed(stack):
        out += closer
    return out


def _usable_llm_json_object(parsed: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(parsed, dict) or _is_reasoning_payload(parsed):
        return None
    kind = str(parsed.get("type") or "").strip().lower()
    if kind in _SKIP_CONTENT_BLOCK_TYPES:
        return None
    return parsed


def extract_llm_json_object(text: Any) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from LLM output, including truncated Responses-API text."""
    if isinstance(text, dict):
        return _usable_llm_json_object(text)
    raw = str(text or "").strip()
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        usable = _usable_llm_json_object(json.loads(raw))
        if usable is not None:
            return usable
    except Exception:
        pass
    decoder = json.JSONDecoder()
    tries = 0
    idx = 0
    n = len(raw)
    while tries < 32 and idx < n:
        nxt = raw.find("{", idx)
        if nxt < 0:
            break
        tries += 1
        try:
            parsed, end = decoder.raw_decode(raw, nxt)
        except Exception:
            idx = nxt + 1
            continue
        usable = _usable_llm_json_object(parsed)
        if usable is not None:
            return usable
        idx = max(int(end), nxt + 1)
    start = raw.find("{")
    if start < 0:
        return None
    blob = raw[start:]
    for _ in range(12):
        try:
            usable = _usable_llm_json_object(json.loads(_balance_truncated_json(blob)))
            if usable is not None:
                return usable
        except Exception:
            pass
        cut = max(blob.rfind(","), blob.rfind("\n"))
        if cut < 1:
            break
        blob = blob[:cut].rstrip().rstrip(",").rstrip()
        if blob.endswith(":"):
            prev_q = blob.rfind('"')
            prev_q2 = blob.rfind('"', 0, prev_q)
            if prev_q2 >= 0:
                blob = blob[:prev_q2].rstrip().rstrip(",")
    return None


__all__ = [
    "Complexity",
    "PaidProvider",
    "PAID_PROVIDERS",
    "PROVIDER_TIER_DEFAULTS",
    "OPENAI_ELEVATED_AVERAGE_MODEL",
    "VISION_CAPABLE_PROVIDERS",
    "legacy_single_model",
    "resolve_provider_model_for_complexity",
    "next_provider_failover_model",
    "escalate_provider_from_model",
    "provider_tier_summary",
    "is_elevated_average_task",
    "provider_supports_vision",
    "vision_unsupported_user_message",
    "claude_extended_thinking_kwargs",
    "gemini_thinking_kwargs",
    "paid_llm_constructor_kwargs",
    "llm_visible_text_from_content",
    "extract_llm_json_object",
]
