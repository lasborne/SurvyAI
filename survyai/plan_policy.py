"""SurvyAI plan policy definitions (dormant scaffolding).

Enforcement stays OFF by default. When you are ready to activate constraints in production,
set `ENFORCE_PLAN_POLICIES=true` and wire the quota checks + warnings.

Rules (current intended commercial behavior)
- Free: Ollama-only models; 10 successful CAD jobs per rolling 30 days (hard cap).
- Pro monthly: full model switching; 100 successful CAD jobs per rolling 30 days (hard cap).
- Pro annual: full model switching; 100 successful CAD jobs per rolling 30 days (SOFT warning only);
  1300 successful CAD jobs per rolling 365 days (hard cap).

Definitions here are safe to ship while building, because nothing enforces them unless enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

PlanSlug = Literal["free", "pro"]
PlanInterval = Literal["daily", "weekly", "monthly", "annual"]


@dataclass(frozen=True)
class ModelPolicy:
    allowed_primary_llms: Sequence[str]
    allowed_fallback_llms: Sequence[str]
    allow_switching: bool


@dataclass(frozen=True)
class CadQuotaPolicy:
    # Rolling windows (not calendar months/years).
    rolling_30d_success_cap: int
    rolling_30d_cap_is_soft_warning: bool = False
    rolling_365d_success_cap: int | None = None


@dataclass(frozen=True)
class PlanPolicy:
    slug: PlanSlug
    models: ModelPolicy

    # CAD quotas depend on interval (monthly vs annual) for Pro.
    cad_monthly: CadQuotaPolicy
    cad_annual: CadQuotaPolicy | None = None


FREE_POLICY = PlanPolicy(
    slug="free",
    models=ModelPolicy(
        allowed_primary_llms=("ollama",),
        allowed_fallback_llms=("ollama",),
        allow_switching=False,
    ),
    cad_monthly=CadQuotaPolicy(
        rolling_30d_success_cap=10,
        rolling_30d_cap_is_soft_warning=False,
        rolling_365d_success_cap=None,
    ),
    cad_annual=None,
)


PRO_POLICY = PlanPolicy(
    slug="pro",
    models=ModelPolicy(
        allowed_primary_llms=("openai", "gemini", "claude", "deepseek", "ollama"),
        allowed_fallback_llms=("openai", "gemini", "claude", "deepseek", "ollama"),
        allow_switching=True,
    ),
    # Pro monthly: hard 50/30d. No annual cap relevant.
    cad_monthly=CadQuotaPolicy(
        rolling_30d_success_cap=100,
        rolling_30d_cap_is_soft_warning=False,
        rolling_365d_success_cap=None,
    ),
    # Pro annual: warn at 50/30d but do not block; hard cap at 650/365d.
    cad_annual=CadQuotaPolicy(
        rolling_30d_success_cap=100,
        rolling_30d_cap_is_soft_warning=True,
        rolling_365d_success_cap=1300,
    ),
)


def policy_for_plan(plan_slug: str) -> PlanPolicy:
    return PRO_POLICY if (plan_slug or "").strip().lower() == "pro" else FREE_POLICY


def cad_policy_for(plan_slug: str, *, interval: PlanInterval) -> CadQuotaPolicy:
    p = policy_for_plan(plan_slug)
    if p.slug == "pro" and interval == "annual" and p.cad_annual is not None:
        return p.cad_annual
    base = p.cad_monthly
    if p.slug == "pro" and interval == "weekly":
        cap = max(1, base.rolling_30d_success_cap // 4)
        return CadQuotaPolicy(
            rolling_30d_success_cap=cap,
            rolling_30d_cap_is_soft_warning=base.rolling_30d_cap_is_soft_warning,
            rolling_365d_success_cap=None,
        )
    if p.slug == "pro" and interval == "daily":
        cap = max(1, base.rolling_30d_success_cap // 30)
        return CadQuotaPolicy(
            rolling_30d_success_cap=cap,
            rolling_30d_cap_is_soft_warning=base.rolling_30d_cap_is_soft_warning,
            rolling_365d_success_cap=None,
        )
    return base


__all__ = [
    "PlanSlug",
    "PlanInterval",
    "ModelPolicy",
    "CadQuotaPolicy",
    "PlanPolicy",
    "FREE_POLICY",
    "PRO_POLICY",
    "policy_for_plan",
    "cad_policy_for",
]