"""
Task-scoped tool / prompt packs for SurvyAI (efficiency).

``lite`` — simple knowledge / historic Q&A: no CAD/GIS tool schemas, short system prompt.
``full`` — default for file-driven, GIS/CAD, multi-step, or internet workflows.

Behavior is intentionally conservative: when unsure, return ``full``.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

ToolScope = Literal["lite", "full"]

# Short system prompt for lite knowledge turns (avoids ~46KB full pack + tool schemas).
KNOWLEDGE_LITE_SYSTEM_PROMPT = (
    "You are SurvyAI, a professional surveying, geospatial, and CAD assistant.\n"
    "Answer only the user's current question accurately and concisely.\n"
    "Do not invent coordinates, survey results, or file paths.\n"
    "Do not continue prior CAD/GIS jobs unless the user explicitly asks.\n"
    "If the question needs live files, ArcGIS, AutoCAD, or Excel tools, say what "
    "you need and stop — do not pretend to run tools you do not have in this turn.\n"
)

_CAD_GIS_MARKERS = (
    "autocad",
    "arcgis",
    "arcpy",
    "dwg",
    "dxf",
    "gdb",
    "aprx",
    "shapefile",
    ".shp",
    "geodatabase",
    "buffer",
    "polygon",
    "cutfill",
    "cut fill",
    "idw",
    "cadastral",
    "survey plan",
    "excel",
    "xlsx",
    "geopandas",
    "coordinate convert",
    "utm",
    "mid-belt",
    "midbelt",
)


def select_tool_scope(
    query: str,
    *,
    complexity: str = "average",
    intent: str = "other",
    prompt_action: Any = None,
    file_driven: bool = False,
) -> ToolScope:
    """
    Choose ``lite`` vs ``full`` tool/prompt pack for this turn.

    Lite is reserved for simple non-file knowledge / historic lookups.
    """
    q = (query or "").strip()
    ql = q.lower()
    if not q:
        return "full"
    if file_driven:
        return "full"
    if complexity in ("average", "complex"):
        # Average GIS orchestration and hard jobs need the full tool surface.
        if complexity == "complex":
            return "full"
        if any(m in ql for m in _CAD_GIS_MARKERS):
            return "full"
    if any(m in ql for m in _CAD_GIS_MARKERS):
        return "full"

    kind = str(getattr(prompt_action, "kind", "") or "")
    if kind in ("current_fact_lookup", "permission_affirm", "file_task"):
        return "full"
    if bool(getattr(prompt_action, "needs_internet", False)):
        return "full"
    if bool(getattr(prompt_action, "needs_tools", False)):
        return "full"

    if complexity == "simple" and intent == "knowledge":
        # Historic / definitional / short explanatory Q&A only.
        return "lite"
    return "full"


def lite_system_prompt(*, base_full_prompt: Optional[str] = None) -> str:
    """Return the lite knowledge system prompt (ignores full pack on purpose)."""
    _ = base_full_prompt
    return KNOWLEDGE_LITE_SYSTEM_PROMPT


__all__ = [
    "ToolScope",
    "KNOWLEDGE_LITE_SYSTEM_PROMPT",
    "select_tool_scope",
    "lite_system_prompt",
]
