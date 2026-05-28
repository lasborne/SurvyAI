"""
License mode and feature flags (Phase 2: enforced in LangGraph tool registration).

Payment model (current):
- **pro** — what you ship to paying customers (single product; Paystack plans on the cloud API).
- **builder** — free unlimited use for development/testing; all integrations stay enabled
  regardless of SURVYAI_FEATURE_* so you never lock yourself out while building.

`SURVYAI_LICENSE_MODE=pro` + optional `SURVYAI_FEATURE_*=0` removes tools for support/abuse cases.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


ToolCategory = Literal["autocad", "internet", "arcgis", "blue_marble", "vector_store"]


def categorize_tool_for_license(tool_name: str) -> Optional[ToolCategory]:
    """
    Map LangChain tool name → integration group for gating.

    None = core tools (Excel, documents, coordinates, filesystem, etc.) — always registered.
    """
    if tool_name.startswith("autocad_"):
        return "autocad"
    if tool_name == "internet_search":
        return "internet"
    if tool_name.startswith("arcgis_"):
        return "arcgis"
    if tool_name.startswith("geographic_calculator_"):
        return "blue_marble"
    if tool_name in ("semantic_search", "store_document", "vector_store_stats"):
        return "vector_store"
    return None


@dataclass(frozen=True)
class FeatureFlags:
    """
    license_mode:
        - builder: dev/test — all integration tools enabled (ignores SURVYAI_FEATURE_* off).
        - pro: commercial — respects allow_* flags (defaults all True).
    """

    license_mode: Literal["builder", "pro"] = "builder"
    allow_autocad: bool = True
    allow_arcgis: bool = True
    allow_blue_marble: bool = True
    allow_internet_tools: bool = True
    allow_vector_store: bool = True

    @classmethod
    def from_env(cls) -> FeatureFlags:
        raw = (os.environ.get("SURVYAI_LICENSE_MODE") or "builder").strip().lower()
        license_mode: Literal["builder", "pro"] = "builder"
        if raw == "pro":
            license_mode = "pro"
        elif raw == "builder":
            license_mode = "builder"
        else:
            license_mode = "builder"

        return cls(
            license_mode=license_mode,
            allow_autocad=_env_bool("SURVYAI_FEATURE_AUTOCAD", True),
            allow_arcgis=_env_bool("SURVYAI_FEATURE_ARCGIS", True),
            allow_blue_marble=_env_bool("SURVYAI_FEATURE_BLUE_MARBLE", True),
            allow_internet_tools=_env_bool("SURVYAI_FEATURE_INTERNET", True),
            allow_vector_store=_env_bool("SURVYAI_FEATURE_VECTOR_STORE", True),
        )

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Whether this tool should be registered for the current license."""
        cat = categorize_tool_for_license(tool_name)
        if cat is None:
            return True
        if self.license_mode == "builder":
            return True
        if cat == "autocad":
            return self.allow_autocad
        if cat == "internet":
            return self.allow_internet_tools
        if cat == "arcgis":
            return self.allow_arcgis
        if cat == "blue_marble":
            return self.allow_blue_marble
        if cat == "vector_store":
            return self.allow_vector_store
        return True

    # --- For UI / capability checks (respect builder = all on) ---

    @property
    def effective_allow_autocad(self) -> bool:
        return True if self.license_mode == "builder" else self.allow_autocad

    @property
    def effective_allow_arcgis(self) -> bool:
        return True if self.license_mode == "builder" else self.allow_arcgis

    @property
    def effective_allow_blue_marble(self) -> bool:
        return True if self.license_mode == "builder" else self.allow_blue_marble

    @property
    def effective_allow_internet_tools(self) -> bool:
        return True if self.license_mode == "builder" else self.allow_internet_tools

    @property
    def effective_allow_vector_store(self) -> bool:
        return True if self.license_mode == "builder" else self.allow_vector_store


def can_use_integration(
    name: Literal["autocad", "arcgis", "blue_marble", "internet", "dxf"],
    *,
    flags: FeatureFlags,
    machine_has_pywin32: bool,
    machine_arcgis_installed: bool,
    machine_blue_marble_cli: bool,
    machine_ezdxf: bool,
) -> tuple[bool, Optional[str]]:
    """
    Return (allowed, reason_if_blocked).

    Uses *effective* allows so builder mode stays fully enabled in UI checks.
    """
    if name == "autocad":
        if not flags.effective_allow_autocad:
            return False, "AutoCAD integration disabled for this license."
        if not machine_has_pywin32:
            return False, "pywin32 is not available (COM automation requires Windows + pywin32)."
        return True, None

    if name == "dxf":
        if not flags.effective_allow_autocad:
            return False, "CAD file features disabled for this license."
        if not machine_ezdxf:
            return False, "ezdxf is not installed; DXF fallback unavailable."
        return True, None

    if name == "arcgis":
        if not flags.effective_allow_arcgis:
            return False, "ArcGIS integration disabled for this license."
        if not machine_arcgis_installed:
            return False, "ArcGIS Pro not detected."
        return True, None

    if name == "blue_marble":
        if not flags.effective_allow_blue_marble:
            return False, "Geographic Calculator integration disabled for this license."
        if not machine_blue_marble_cli:
            return False, "Geographic Calculator CLI not found."
        return True, None

    if name == "internet":
        if not flags.effective_allow_internet_tools:
            return False, "Internet tools disabled for this license."
        return True, None

    return False, "Unknown integration."


__all__ = [
    "FeatureFlags",
    "categorize_tool_for_license",
    "can_use_integration",
]
