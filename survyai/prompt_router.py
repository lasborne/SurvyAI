"""
Cheap paid-model prompt router for SurvyAI.

Before a paid provider executes a task, the cheapest hosted model for that
provider (OpenAI gpt-5.6-luna, Claude Haiku, Gemini Flash, DeepSeek Chat)
classifies the current user turn and picks the execution model.

Design constraints (no breaking changes):
- Heuristic complexity routing remains the fallback when the classifier
  fails, times out, or is disabled.
- Explicit user tier overrides and fast-mode still skip this classifier.
- File-driven GIS/CAD is never executed on the cheapest model (floor: average).
- The classifier cannot pick a model outside the provider allow-list.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from survyai.openai_models import normalize_model_id
from survyai.provider_models import (
    OPENAI_ELEVATED_AVERAGE_MODEL,
    PAID_PROVIDERS,
    Complexity,
    resolve_provider_model_for_complexity,
)

MAX_ROUTER_QUERY_CHARS = 1600
ROUTER_TIMEOUT_SECONDS = 5
ROUTER_MAX_OUTPUT_TOKENS = 160
# Never stack more than one cheap failover (avoids 15–20s stalls).
ROUTER_MAX_ATTEMPTS = 2
ROUTER_CACHE_MAX = 48

# Ordered cheap failover if the primary router model is unavailable.
# Never escalate the classifier itself to flagship / opus / reasoner / pro.
ROUTER_FAILOVER_MODELS: Dict[str, tuple[str, ...]] = {
    "openai": ("gpt-5.4-nano", "gpt-5-nano", "gpt-4o-mini"),
    "claude": ("claude-3-haiku-20240307",),
    "gemini": ("gemini-2.0-flash", "gemini-1.5-flash"),
    "deepseek": ("deepseek-chat",),
}

_COMPLEXITY_ALIASES: Dict[str, Complexity] = {
    "simple": "simple",
    "low": "simple",
    "nano": "simple",
    "cheap": "simple",
    "lite": "simple",
    "luna": "simple",
    "haiku": "simple",
    "flash": "simple",
    "average": "average",
    "medium": "average",
    "mini": "average",
    "standard": "average",
    "balanced": "average",
    "typical": "average",
    "terra": "average",
    "sonnet": "average",
    "complex": "complex",
    "advanced": "complex",
    "high": "complex",
    "hard": "complex",
    "sol": "complex",
    "opus": "complex",
    "reasoner": "complex",
    "pro": "complex",
}

_TIER_RANK: Dict[str, int] = {"simple": 0, "average": 1, "complex": 2}

_HEAVY_GIS = (
    "cutfill",
    "cut fill",
    "cut/fill",
    "idw",
    "inverse distance",
    "geoprocessing",
    "spatial analyst",
    "raster",
    "volume between",
    "pre and post",
    "living atlas",
    "deep learning model",
    "land cover",
    "landcover",
)
_UNORDERED_GIS = (
    "in any order",
    "no particular order",
    "not sure which first",
    "whichever first",
    "multiple difficult",
    "several difficult",
    "no order",
    "unordered",
)
_GIS_CONTEXT = (
    "gis",
    "geospatial",
    "arcgis",
    "survey",
    "cadastral",
    "coordinate",
    "polygon",
)
_TOOLISH = (
    "autocad",
    "arcgis",
    "excel",
    "convert",
    "buffer",
    "polygon",
    "dwg",
    "dxf",
    "gdb",
    "shapefile",
)
_HISTORIC_SIMPLE = (
    "historic",
    "historical",
    "in history",
    "what year",
    "who was",
    "old standard",
    "define ",
    "definition of",
    "what does",
    "meaning of",
)


@dataclass(frozen=True)
class RouteCandidate:
    """One allowed execution model for the active paid provider."""

    model: str
    complexity: Complexity
    elevated: bool = False
    role: str = ""


@dataclass(frozen=True)
class PromptRouteDecision:
    """Resolved execution route after parse + safety floors."""

    complexity: Complexity
    model: str
    elevated_average: bool = False
    reason: str = ""
    router_model: str = ""
    source: str = "llm"  # llm | fallback


def router_model_for_provider(provider: str, settings: Any = None) -> str:
    """Cheapest paid (Low / simple) model — this is the classifier."""
    p = str(provider or "").strip().lower()
    return resolve_provider_model_for_complexity(
        p, "simple", settings=settings, enable_tiered=True
    )


def router_models_to_try(provider: str, settings: Any = None) -> List[str]:
    """Primary cheap router plus cheaper/equal failovers (deduped)."""
    p = str(provider or "").strip().lower()
    primary = router_model_for_provider(p, settings)
    out: List[str] = []
    seen: set[str] = set()
    for mid in (primary, *ROUTER_FAILOVER_MODELS.get(p, ())):
        key = str(mid or "").strip()
        if not key:
            continue
        low = key.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(key)
        if len(out) >= ROUTER_MAX_ATTEMPTS:
            break
    return out


def execution_candidates_for_provider(
    provider: str, settings: Any = None
) -> List[RouteCandidate]:
    """Allow-list of execution models the cheap router may pick."""
    p = str(provider or "").strip().lower()
    simple = resolve_provider_model_for_complexity(p, "simple", settings=settings, enable_tiered=True)
    average = resolve_provider_model_for_complexity(p, "average", settings=settings, enable_tiered=True)
    complex_m = resolve_provider_model_for_complexity(p, "complex", settings=settings, enable_tiered=True)

    rows: List[RouteCandidate] = [
        RouteCandidate(simple, "simple", role="cheapest accurate (lookups / short Q&A)"),
        RouteCandidate(average, "average", role="balanced GIS/CAD orchestration"),
    ]
    if p == "openai":
        mini_extra = "gpt-5.4-mini"
        if mini_extra.lower() not in {simple.lower(), average.lower(), complex_m.lower()}:
            rows.append(
                RouteCandidate(
                    mini_extra,
                    "average",
                    role="strong mini for agents/tools (cost-efficient mid tier)",
                )
            )
        elevated = OPENAI_ELEVATED_AVERAGE_MODEL
        if elevated.lower() not in {c.model.lower() for c in rows}:
            rows.append(
                RouteCandidate(
                    elevated,
                    "average",
                    elevated=True,
                    role="elevated reasoning (compare / reconcile / multi-hop) without full flagship",
                )
            )
    rows.append(RouteCandidate(complex_m, "complex", role="hardest multi-step / raster / volume / unordered GIS"))

    out: List[RouteCandidate] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.model or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def default_model_for_complexity(
    candidates: Sequence[RouteCandidate],
    complexity: Complexity,
    *,
    elevated_average: bool = False,
) -> str:
    """Pick the catalog default for a complexity bucket from the allow-list."""
    c: Complexity = complexity if complexity in ("simple", "average", "complex") else "average"
    if elevated_average and c == "average":
        for row in candidates:
            if row.elevated:
                return row.model
    for row in candidates:
        if row.complexity == c and not row.elevated:
            return row.model
    for row in candidates:
        if row.complexity == c:
            return row.model
    if candidates:
        return candidates[0].model
    return ""


def _normalize_complexity(raw: Any) -> Optional[Complexity]:
    text = str(raw or "").strip().lower().replace("_", " ").replace("-", " ")
    if not text:
        return None
    if text in _COMPLEXITY_ALIASES:
        return _COMPLEXITY_ALIASES[text]
    for token in text.split():
        if token in _COMPLEXITY_ALIASES:
            return _COMPLEXITY_ALIASES[token]
    return None


def _candidate_lookup(candidates: Sequence[RouteCandidate]) -> Dict[str, RouteCandidate]:
    lookup: Dict[str, RouteCandidate] = {}
    for row in candidates:
        lookup[row.model.strip().lower()] = row
        aliased = normalize_model_id(row.model)
        if aliased:
            lookup[aliased.strip().lower()] = row
    return lookup


def match_candidate_model(
    raw_model: Any, candidates: Sequence[RouteCandidate]
) -> Optional[RouteCandidate]:
    """Resolve a router-returned model id onto the allow-list (aliases OK)."""
    raw = str(raw_model or "").strip()
    if not raw:
        return None
    lookup = _candidate_lookup(candidates)
    keys = [raw.lower(), (normalize_model_id(raw) or raw).lower()]
    for key in keys:
        if key in lookup:
            return lookup[key]
    return None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.S | re.I)
    blob = fence.group(1) if fence else None
    if blob is None:
        brace = re.search(r"\{.*\}", raw, flags=re.S)
        blob = brace.group(0) if brace else raw
    try:
        data = json.loads(blob)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def parse_router_response(
    raw_text: str,
    candidates: Sequence[RouteCandidate],
    *,
    heuristic_complexity: Complexity = "average",
) -> Optional[PromptRouteDecision]:
    """
    Parse the cheap model's JSON. Returns None if unusable (caller falls back).
    """
    data = _extract_json_object(raw_text)
    if not data:
        return None

    complexity = _normalize_complexity(data.get("complexity") or data.get("tier") or data.get("level"))
    matched = match_candidate_model(
        data.get("model") or data.get("model_name") or data.get("execution_model"),
        candidates,
    )
    elevated_flag = data.get("elevated_average")
    if isinstance(elevated_flag, str):
        elevated_flag = elevated_flag.strip().lower() in {"1", "true", "yes", "y"}
    else:
        elevated_flag = bool(elevated_flag)

    reason = str(data.get("reason") or data.get("why") or "").strip()[:240]

    if matched is None and complexity is None:
        return None

    if matched is not None:
        complexity = matched.complexity
        elevated_average = bool(matched.elevated or (elevated_flag and matched.complexity == "average"))
        model = matched.model
    else:
        c: Complexity = complexity or (
            heuristic_complexity if heuristic_complexity in ("simple", "average", "complex") else "average"
        )
        elevated_average = bool(elevated_flag and c == "average")
        model = default_model_for_complexity(candidates, c, elevated_average=elevated_average)
        complexity = c

    if not model:
        return None
    return PromptRouteDecision(
        complexity=complexity,  # type: ignore[arg-type]
        model=model,
        elevated_average=elevated_average,
        reason=reason,
        source="llm",
    )


def is_heavy_geospatial_task(query: str) -> bool:
    """True for raster / volume / unordered multi-step GIS that needs the flagship tier."""
    raw = query or ""
    ql = raw.lower()
    if not ql.strip():
        return False
    if len(re.findall(r"[a-zA-Z]:\\", raw)) >= 3:
        return True
    if any(s in ql for s in _UNORDERED_GIS) and any(s in ql for s in _GIS_CONTEXT):
        return True
    if any(s in ql for s in _HEAVY_GIS) and (
        "arcgis" in ql or "arcpy" in ql or "raster" in ql
    ):
        return True
    if "arcpy" in ql and any(s in ql for s in ("raster", "idw", "cut", "fill", "gdb")):
        return True
    return False


def heuristic_route_confidence(
    query: str,
    *,
    complexity: Complexity = "average",
    file_driven: bool = False,
    intent: str = "",
    kind: str = "",
) -> str:
    """
    ``simple`` / ``complex`` → skip the paid classifier (already obvious).
    ``ambiguous`` → run the cheap LLM router (cost-saving decision).
    """
    ql = (query or "").lower()
    kind_l = str(kind or "").strip().lower()
    intent_l = str(intent or "").strip().lower()
    if is_heavy_geospatial_task(query):
        return "complex"
    if file_driven or kind_l in ("file_task", "current_fact_lookup", "permission_affirm"):
        return "ambiguous"
    if complexity == "simple" and not any(s in ql for s in _TOOLISH):
        return "simple"
    words = len(ql.split())
    if (
        intent_l == "knowledge"
        and words <= 40
        and not any(s in ql for s in _TOOLISH)
        and (
            any(s in ql for s in _HISTORIC_SIMPLE)
            or ql.startswith(("what is", "what are", "define ", "explain "))
        )
    ):
        return "simple"
    return "ambiguous"


def apply_route_floors(
    decision: PromptRouteDecision,
    candidates: Sequence[RouteCandidate],
    *,
    file_driven: bool = False,
    knowledge_only: bool = False,
) -> PromptRouteDecision:
    """
    Safety clamps that do not undo cost routing:

    - File/CAD/GIS jobs must not run on the cheapest model (floor: average).
    - Knowledge-only Q&A must not run on the flagship tier (ceiling: average).
    - Unknown/empty model maps back onto the allow-list.
    """
    complexity: Complexity = (
        decision.complexity if decision.complexity in ("simple", "average", "complex") else "average"
    )
    elevated = bool(decision.elevated_average and complexity == "average")
    if file_driven and _TIER_RANK.get(complexity, 0) < _TIER_RANK["average"]:
        complexity = "average"
        elevated = False
    if knowledge_only and not file_driven and _TIER_RANK.get(complexity, 0) > _TIER_RANK["average"]:
        complexity = "average"
        elevated = False
    model = decision.model
    matched = match_candidate_model(model, candidates)
    if matched is None:
        model = default_model_for_complexity(candidates, complexity, elevated_average=elevated)
    else:
        model = matched.model
        complexity = matched.complexity
        elevated = bool(matched.elevated)
        if file_driven and _TIER_RANK.get(complexity, 0) < _TIER_RANK["average"]:
            complexity = "average"
            elevated = False
            model = default_model_for_complexity(candidates, "average", elevated_average=False)
        elif knowledge_only and not file_driven and _TIER_RANK.get(complexity, 0) > _TIER_RANK["average"]:
            complexity = "average"
            elevated = False
            model = default_model_for_complexity(candidates, "average", elevated_average=False)
    return PromptRouteDecision(
        complexity=complexity,
        model=model,
        elevated_average=elevated,
        reason=decision.reason,
        router_model=decision.router_model,
        source=decision.source,
    )


def truncate_router_query(query: str) -> str:
    text = " ".join((query or "").split()).strip()
    if len(text) <= MAX_ROUTER_QUERY_CHARS:
        return text
    return text[: MAX_ROUTER_QUERY_CHARS - 3] + "..."


def build_router_messages(
    *,
    provider: str,
    query: str,
    candidates: Sequence[RouteCandidate],
    heuristic_complexity: Complexity = "average",
    kind: str = "",
    file_driven: bool = False,
    needs_internet: bool = False,
) -> tuple[str, str]:
    """System + user prompts for the cheap classifier (JSON only, no tools)."""
    lines = []
    for row in candidates:
        extra = " [elevated_average=true]" if row.elevated else ""
        lines.append(f"- {row.model} → complexity={row.complexity}{extra} — {row.role}".rstrip(" —"))
    catalog = "\n".join(lines) if lines else "- (no catalog)"

    system = (
        "SurvyAI cost router. Do not answer the user. Do not call tools. "
        "Pick the cheapest accurate execution model.\n"
        "simple=lookups/short Q&A, no files. "
        "average=typical GIS/CAD/Excel/CRS/drafting. "
        "complex=raster/volume/IDW/cut-fill/unordered multi-file GIS. "
        "OpenAI: gpt-5.5 only for compare/reconcile/multi-hop that is not full GIS. "
        "Prefer cheaper when accuracy holds. "
        "JSON only: complexity, model, elevated_average, reason. Copy model exactly."
    )
    user = (
        f"Provider: {provider}\n"
        f"Heuristic estimate (software, advisory only): {heuristic_complexity}\n"
        f"Prompt kind: {kind or 'unknown'}\n"
        f"File-driven: {bool(file_driven)}\n"
        f"Needs internet: {bool(needs_internet)}\n"
        f"Candidates:\n{catalog}\n\n"
        f"Current user request:\n{truncate_router_query(query)}\n"
    )
    return system, user


def message_content_to_text(msg: Any) -> str:
    raw = getattr(msg, "content", msg)
    if isinstance(raw, list):
        parts: List[str] = []
        for part in raw:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "") or ""))
            else:
                parts.append(str(part))
        return "\n".join(parts).strip()
    return str(raw or "").strip()


def should_use_llm_prompt_router(
    *,
    provider: str,
    enable_tiered: bool = True,
    enable_llm_prompt_router: bool = True,
    user_tier_override: Optional[str] = None,
    fast_mode_forced_simple: bool = False,
    heuristic_confidence: Optional[str] = None,
) -> bool:
    """True when the cheap LLM classifier should run for this turn."""
    p = str(provider or "").strip().lower()
    if p not in PAID_PROVIDERS:
        return False
    if not enable_tiered or not enable_llm_prompt_router:
        return False
    if user_tier_override:
        return False
    if fast_mode_forced_simple:
        return False
    # Obvious routes skip the extra round-trip (speed + cost); keep LLM for ambiguous.
    if str(heuristic_confidence or "").strip().lower() in {"simple", "complex"}:
        return False
    return True


__all__ = [
    "MAX_ROUTER_QUERY_CHARS",
    "ROUTER_TIMEOUT_SECONDS",
    "ROUTER_MAX_OUTPUT_TOKENS",
    "ROUTER_MAX_ATTEMPTS",
    "ROUTER_CACHE_MAX",
    "RouteCandidate",
    "PromptRouteDecision",
    "router_model_for_provider",
    "router_models_to_try",
    "execution_candidates_for_provider",
    "default_model_for_complexity",
    "match_candidate_model",
    "parse_router_response",
    "is_heavy_geospatial_task",
    "heuristic_route_confidence",
    "apply_route_floors",
    "truncate_router_query",
    "build_router_messages",
    "message_content_to_text",
    "should_use_llm_prompt_router",
]
