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
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence

from survyai.openai_models import (
    DEFAULT_MODEL_FOR_COMPLEXITY,
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


__all__ = [
    "Complexity",
    "PaidProvider",
    "PAID_PROVIDERS",
    "PROVIDER_TIER_DEFAULTS",
    "OPENAI_ELEVATED_AVERAGE_MODEL",
    "legacy_single_model",
    "resolve_provider_model_for_complexity",
    "next_provider_failover_model",
    "escalate_provider_from_model",
    "provider_tier_summary",
    "is_elevated_average_task",
]
