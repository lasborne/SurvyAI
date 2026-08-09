"""
OpenAI chat-model catalog and intelligent tier / failover routing for SurvyAI.

Model IDs are taken from OpenAI's public Models docs
(https://developers.openai.com/api/docs/models) as of 2026-08:

Frontier GPT-5.6: gpt-5.6-sol (alias gpt-5.6), gpt-5.6-terra, gpt-5.6-luna
GPT-5.5: gpt-5.5, gpt-5.5-pro
GPT-5.4: gpt-5.4, gpt-5.4-pro, gpt-5.4-mini, gpt-5.4-nano
Prior: gpt-5.2, gpt-5.1, gpt-5, gpt-5-mini, gpt-5-nano, gpt-4.1, gpt-4.1-mini, gpt-4o, gpt-4o-mini

Selection priority (product policy): accuracy → speed → cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

Complexity = Literal["simple", "average", "complex"]
TierName = Literal["nano", "mini", "complex"]


@dataclass(frozen=True)
class OpenAIChatModel:
    """One Chat Completions / Responses-capable GPT model."""

    model_id: str
    tier: TierName
    # Higher = stronger reasoning / tool orchestration (0–100).
    capability: int
    # Higher = more expensive (0–100); used after capability when ranking fallbacks.
    cost_rank: int
    aliases: Tuple[str, ...] = ()
    notes: str = ""


# Curated catalog (≥12 chat models). Keep IDs exact API strings.
OPENAI_CHAT_MODELS: Tuple[OpenAIChatModel, ...] = (
    OpenAIChatModel("gpt-5.6-sol", "complex", 100, 95, aliases=("gpt-5.6",), notes="Flagship GPT-5.6"),
    OpenAIChatModel("gpt-5.6-terra", "mini", 88, 55, notes="Balanced GPT-5.6 (intelligence vs cost)"),
    OpenAIChatModel("gpt-5.6-luna", "nano", 62, 12, notes="Cost-sensitive GPT-5.6"),
    OpenAIChatModel("gpt-5.5", "complex", 96, 90, notes="Frontier GPT-5.5"),
    OpenAIChatModel("gpt-5.5-pro", "complex", 98, 98, notes="Higher-precision GPT-5.5 Pro"),
    OpenAIChatModel("gpt-5.4", "complex", 92, 80, notes="Prior frontier GPT-5.4"),
    OpenAIChatModel("gpt-5.4-pro", "complex", 94, 92, notes="GPT-5.4 Pro"),
    OpenAIChatModel("gpt-5.4-mini", "mini", 78, 35, notes="Strong mini for agents/tools"),
    OpenAIChatModel("gpt-5.4-nano", "nano", 55, 10, notes="Cheapest GPT-5.4-class"),
    OpenAIChatModel("gpt-5.2", "complex", 86, 75, notes="Prior frontier"),
    OpenAIChatModel("gpt-5.1", "complex", 84, 72, notes="Prior frontier"),
    OpenAIChatModel("gpt-5", "complex", 82, 70, notes="GPT-5 base"),
    OpenAIChatModel("gpt-5-mini", "mini", 70, 28, notes="GPT-5 mini"),
    OpenAIChatModel("gpt-5-nano", "nano", 50, 8, notes="GPT-5 nano"),
    OpenAIChatModel("gpt-4.1", "complex", 76, 60, notes="GPT-4.1"),
    OpenAIChatModel("gpt-4.1-mini", "mini", 65, 25, notes="GPT-4.1 mini"),
    OpenAIChatModel("gpt-4o", "complex", 74, 58, notes="GPT-4o"),
    OpenAIChatModel("gpt-4o-mini", "mini", 58, 15, notes="GPT-4o mini (legacy fallback)"),
)

# Default preferred model per complexity bucket (settings may override).
DEFAULT_MODEL_FOR_COMPLEXITY: Dict[Complexity, str] = {
    "simple": "gpt-5.6-luna",
    "average": "gpt-5.6-terra",
    # Balanced flagship routing: Sol for hardest work; failover chain still has Sol peers.
    "complex": "gpt-5.6-sol",
}

# Explicit ordered failover chains (preferred first). Used when a model is
# quota-exhausted / unavailable. Prefer same-or-adjacent capability, then cheaper.
FALLBACK_CHAINS: Dict[str, Tuple[str, ...]] = {
    "gpt-5.6-sol": (
        "gpt-5.6",
        "gpt-5.5",
        "gpt-5.6-terra",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5-mini",
        "gpt-4o-mini",
    ),
    "gpt-5.6": (
        "gpt-5.6-sol",
        "gpt-5.5",
        "gpt-5.6-terra",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-4o-mini",
    ),
    "gpt-5.6-terra": (
        "gpt-5.4-mini",
        "gpt-5.6-luna",
        "gpt-5-mini",
        "gpt-5.4",
        "gpt-5.5",
        "gpt-4o-mini",
    ),
    "gpt-5.6-luna": (
        "gpt-5.4-nano",
        "gpt-5-nano",
        "gpt-5.4-mini",
        "gpt-4o-mini",
    ),
    "gpt-5.5": (
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-4o-mini",
    ),
    "gpt-5.5-pro": (
        "gpt-5.5",
        "gpt-5.6-sol",
        "gpt-5.4-pro",
        "gpt-5.4",
        "gpt-5.4-mini",
    ),
    "gpt-5.4": (
        "gpt-5.5",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.4-mini",
        "gpt-5.2",
        "gpt-5",
        "gpt-4o-mini",
    ),
    "gpt-5.4-pro": (
        "gpt-5.5-pro",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.6-sol",
        "gpt-5.4-mini",
    ),
    "gpt-5.4-mini": (
        "gpt-5.6-terra",
        "gpt-5-mini",
        "gpt-5.4-nano",
        "gpt-4.1-mini",
        "gpt-4o-mini",
    ),
    "gpt-5.4-nano": (
        "gpt-5.6-luna",
        "gpt-5-nano",
        "gpt-4o-mini",
        "gpt-5.4-mini",
    ),
}


def all_chat_model_ids() -> List[str]:
    """Return canonical + alias IDs for UI / validation."""
    out: List[str] = []
    seen: set[str] = set()
    for m in OPENAI_CHAT_MODELS:
        for mid in (m.model_id, *m.aliases):
            key = mid.lower()
            if key not in seen:
                seen.add(key)
                out.append(mid)
    return out


def normalize_model_id(model_name: Optional[str]) -> str:
    """Resolve aliases to the canonical catalog id when known."""
    raw = str(model_name or "").strip()
    if not raw:
        return ""
    low = raw.lower()
    for m in OPENAI_CHAT_MODELS:
        if m.model_id.lower() == low:
            return m.model_id
        if any(a.lower() == low for a in m.aliases):
            return m.model_id
    return raw


def get_model_meta(model_name: Optional[str]) -> Optional[OpenAIChatModel]:
    mid = normalize_model_id(model_name)
    if not mid:
        return None
    low = mid.lower()
    for m in OPENAI_CHAT_MODELS:
        if m.model_id.lower() == low or any(a.lower() == low for a in m.aliases):
            return m
    return None


def infer_tier(model_name: Optional[str]) -> TierName:
    """Map a model id to nano/mini/complex (legacy SurvyAI tier names)."""
    meta = get_model_meta(model_name)
    if meta:
        return meta.tier
    low = str(model_name or "").lower()
    if any(k in low for k in ("nano", "luna")):
        return "nano"
    if any(k in low for k in ("mini", "terra")):
        return "mini"
    if any(k in low for k in ("pro", "sol", "gpt-5.5", "gpt-5.4", "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-4o", "gpt-4.1")):
        # bare gpt-5.4-mini already caught; remaining are complex-ish
        if "mini" in low or "nano" in low:
            return "mini" if "mini" in low else "nano"
        return "complex"
    return "mini"


def default_model_for_complexity(complexity: Complexity) -> str:
    return DEFAULT_MODEL_FOR_COMPLEXITY.get(complexity, DEFAULT_MODEL_FOR_COMPLEXITY["average"])


def resolve_model_for_complexity(
    complexity: Complexity,
    *,
    nano: Optional[str] = None,
    mini: Optional[str] = None,
    complex_model: Optional[str] = None,
    legacy: Optional[str] = None,
) -> str:
    """Pick the configured model for a complexity bucket, with catalog defaults."""
    mapping = {
        "simple": (nano or "").strip() or default_model_for_complexity("simple"),
        "average": (mini or "").strip() or default_model_for_complexity("average"),
        "complex": (complex_model or "").strip() or default_model_for_complexity("complex"),
    }
    chosen = mapping.get(complexity) or mapping["average"]
    if not chosen:
        chosen = (legacy or "").strip() or "gpt-4o-mini"
    return normalize_model_id(chosen) or chosen


def _dedupe_preserve(ids: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for mid in ids:
        n = normalize_model_id(mid) or str(mid or "").strip()
        if not n:
            continue
        key = n.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


def fallback_models_for(
    model_name: Optional[str],
    *,
    complexity: Complexity = "average",
    exclude: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Ordered alternate models when ``model_name`` fails (quota / 5xx / unavailable).

    Prefer explicit chain for the model; otherwise rank catalog by capability then cost.
    """
    preferred = normalize_model_id(model_name) or str(model_name or "").strip()
    excluded = {normalize_model_id(x).lower() for x in (exclude or []) if x}
    if preferred:
        excluded.add(preferred.lower())

    chain: List[str] = []
    if preferred in FALLBACK_CHAINS:
        chain.extend(FALLBACK_CHAINS[preferred])
    elif preferred:
        # Try alias key
        for key, vals in FALLBACK_CHAINS.items():
            if preferred.lower() == key.lower() or preferred.lower() in {
                a.lower() for a in (get_model_meta(key).aliases if get_model_meta(key) else ())
            }:
                chain.extend(vals)
                break

    # Complexity-aware extras: ensure cheaper siblings appear for average/simple.
    if complexity == "complex":
        chain.extend(
            (
                "gpt-5.5",
                "gpt-5.6-terra",
                "gpt-5.4",
                "gpt-5.4-mini",
                "gpt-5-mini",
                "gpt-4o-mini",
            )
        )
    elif complexity == "average":
        chain.extend(
            (
                "gpt-5.6-terra",
                "gpt-5.4-mini",
                "gpt-5.6-luna",
                "gpt-5-mini",
                "gpt-4o-mini",
            )
        )
    else:
        chain.extend(
            (
                "gpt-5.6-luna",
                "gpt-5.4-nano",
                "gpt-5-nano",
                "gpt-4o-mini",
            )
        )

    # Capability-sorted remainder from catalog (strong first, then cheaper).
    ranked = sorted(
        OPENAI_CHAT_MODELS,
        key=lambda m: (-m.capability, m.cost_rank),
    )
    chain.extend(m.model_id for m in ranked)

    return [m for m in _dedupe_preserve(chain) if m.lower() not in excluded]


def next_fallback_model(
    model_name: Optional[str],
    *,
    complexity: Complexity = "average",
    tried: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Return the next unused fallback model, or None if exhausted."""
    exclude = list(tried or [])
    if model_name:
        exclude.append(model_name)
    chain = fallback_models_for(model_name, complexity=complexity, exclude=exclude)
    return chain[0] if chain else None


def escalate_tier_model(
    current_tier: TierName,
    *,
    mini: Optional[str] = None,
    complex_model: Optional[str] = None,
) -> Optional[str]:
    """Legacy nano→mini→complex escalation (capability upgrade, not quota failover)."""
    if current_tier == "nano":
        return resolve_model_for_complexity(
            "average", mini=mini, complex_model=complex_model
        )
    if current_tier == "mini":
        return resolve_model_for_complexity(
            "complex", mini=mini, complex_model=complex_model
        )
    return None


# Prior SurvyAI platform defaults (pre GPT-5.6 luna/terra/sol). Remap only when
# a bootstrap/settings *slot* still carries the exact retired product default for
# that slot — never rewrite an intentional cross-tier choice (e.g. complex=gpt-5.4-mini).
_LEGACY_SLOT_DEFAULTS: Dict[str, Dict[str, str]] = {
    "openai_model_nano": {
        "gpt-5.4-nano": DEFAULT_MODEL_FOR_COMPLEXITY["simple"],
        "gpt-5-nano": DEFAULT_MODEL_FOR_COMPLEXITY["simple"],
    },
    "openai_model_mini": {
        "gpt-5.4-mini": DEFAULT_MODEL_FOR_COMPLEXITY["average"],
        "gpt-5-mini": DEFAULT_MODEL_FOR_COMPLEXITY["average"],
    },
    "openai_model_complex": {
        "gpt-5.4": DEFAULT_MODEL_FOR_COMPLEXITY["complex"],
        "gpt-5.4-pro": DEFAULT_MODEL_FOR_COMPLEXITY["complex"],
        "gpt-5": DEFAULT_MODEL_FOR_COMPLEXITY["complex"],
    },
    "openai_model": {
        "gpt-5.4": DEFAULT_MODEL_FOR_COMPLEXITY["average"],
        "gpt-5.4-mini": DEFAULT_MODEL_FOR_COMPLEXITY["average"],
        "gpt-4o-mini": DEFAULT_MODEL_FOR_COMPLEXITY["average"],
    },
}


def migrate_legacy_platform_model(slot: str, model_id: Optional[str]) -> str:
    """
    Rewrite a retired SurvyAI platform default in ``slot`` to the current catalog default.

    Safe for cloud bootstrap that still ships gpt-5.4* / gpt-4o-mini from older env files.
    Leaves any non-default value untouched.
    """
    raw = str(model_id or "").strip()
    if not raw:
        return ""
    key = normalize_model_id(raw) or raw
    slot_map = _LEGACY_SLOT_DEFAULTS.get(str(slot or "").strip())
    if not slot_map:
        return key
    for legacy, replacement in slot_map.items():
        if key.lower() == legacy.lower():
            return replacement
    return key


def migrate_legacy_platform_models(models: Dict[str, Optional[str]]) -> Dict[str, str]:
    """Apply :func:`migrate_legacy_platform_model` to a settings/bootstrap model dict."""
    out: Dict[str, str] = {}
    for slot, value in (models or {}).items():
        if value is None:
            continue
        migrated = migrate_legacy_platform_model(str(slot), value)
        if migrated:
            out[str(slot)] = migrated
    return out


def openai_tier_models_for_display(
    *,
    nano: Optional[str] = None,
    mini: Optional[str] = None,
    complex_model: Optional[str] = None,
    legacy: Optional[str] = None,
    enable_tiered: bool = True,
) -> Dict[str, str]:
    """
    Exact model IDs shown for Low / Medium / Advanced task tiers.

    Keys: ``low``, ``medium``, ``advanced`` (product labels) plus ``legacy``.
    """
    if enable_tiered:
        low = resolve_model_for_complexity("simple", nano=nano, mini=mini, complex_model=complex_model, legacy=legacy)
        medium = resolve_model_for_complexity("average", nano=nano, mini=mini, complex_model=complex_model, legacy=legacy)
        advanced = resolve_model_for_complexity("complex", nano=nano, mini=mini, complex_model=complex_model, legacy=legacy)
    else:
        single = normalize_model_id(legacy) or (legacy or "").strip() or default_model_for_complexity("average")
        low = medium = advanced = single
    return {
        "low": low,
        "medium": medium,
        "advanced": advanced,
        "legacy": normalize_model_id(legacy) or (legacy or "").strip() or medium,
    }


def format_openai_tier_summary(
    *,
    nano: Optional[str] = None,
    mini: Optional[str] = None,
    complex_model: Optional[str] = None,
    legacy: Optional[str] = None,
    enable_tiered: bool = True,
) -> str:
    """One-line Low/Medium/Advanced summary for Settings / Credits."""
    tiers = openai_tier_models_for_display(
        nano=nano,
        mini=mini,
        complex_model=complex_model,
        legacy=legacy,
        enable_tiered=enable_tiered,
    )
    if not enable_tiered:
        return f"openai ({tiers['legacy']})"
    return (
        f"openai — Low: {tiers['low']} · Medium: {tiers['medium']} · "
        f"Advanced: {tiers['advanced']}"
    )


__all__ = [
    "Complexity",
    "TierName",
    "OpenAIChatModel",
    "OPENAI_CHAT_MODELS",
    "DEFAULT_MODEL_FOR_COMPLEXITY",
    "FALLBACK_CHAINS",
    "all_chat_model_ids",
    "normalize_model_id",
    "get_model_meta",
    "infer_tier",
    "default_model_for_complexity",
    "resolve_model_for_complexity",
    "fallback_models_for",
    "next_fallback_model",
    "escalate_tier_model",
    "migrate_legacy_platform_model",
    "migrate_legacy_platform_models",
    "openai_tier_models_for_display",
    "format_openai_tier_summary",
]
