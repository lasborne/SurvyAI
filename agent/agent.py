"""
================================================================================
SurvyAI Agent - LangGraph Implementation
================================================================================

This module implements the core AI agent for SurvyAI using LangGraph, a framework
for building stateful, multi-step AI applications.

ARCHITECTURE OVERVIEW:
----------------------
LangGraph uses a graph-based architecture where:
- NODES: Functions that process and transform state
- EDGES: Define the flow between nodes
- STATE: A shared data structure passed through the graph

For SurvyAI, the graph flow is:
    
    [START] 
       │
       ▼
    ┌─────────────────┐
    │   Agent Node    │  ◄── LLM reasons about the query
    │  (LLM Reasoning)│      and decides what to do
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐     ┌─────────────────┐
    │  Should Use     │────►│   Tool Node     │
    │    Tools?       │ Yes │ (Execute Tools) │
    └────────┬────────┘     └────────┬────────┘
             │ No                    │
             ▼                       │
    ┌─────────────────┐              │
    │      END        │◄─────────────┘
    │  (Return Result)│
    └─────────────────┘

WHY LANGGRAPH FOR SURVYAI:
--------------------------
1. Complex workflows: Surveying tasks often require multiple steps
2. Tool orchestration: Need to coordinate AutoCAD, Excel, ArcGIS, etc.
3. State management: Track context across tool calls
4. Error recovery: Handle failures gracefully
5. Extensibility: Easy to add new capabilities

MODULES AND DEPENDENCIES:
-------------------------
- langgraph: Graph-based agent framework
- langchain_core: Base classes for messages, tools
- langchain_google_genai: Google Gemini LLM integration
- langchain_openai: OpenAI models (GPT-4/4o/5) and OpenAI-compatible API (for DeepSeek)
- langchain_anthropic: Anthropic Claude models (Opus/Sonnet/Haiku)

REFACTORED LAYOUT:
-----------------
- agent.prompts: SYSTEM_PROMPT and other prompt strings (editable without touching agent logic).
- agent.state: AgentState (LangGraph state), RAGRouteDecision, and looks_like_file_driven_task().

Author: SurvyAI Team
License: MIT
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

# Ensure Any is available globally (for Pydantic model evaluation)
# This prevents "name 'Any' is not defined" errors
# Import it explicitly and make it available in globals
import typing
# Make Any available in multiple ways for different evaluation contexts
Any = typing.Any
globals()['Any'] = typing.Any
# For eval/exec contexts, ensure it's available
if isinstance(__builtins__, dict):
    __builtins__['Any'] = typing.Any
elif hasattr(__builtins__, '__dict__'):
    __builtins__.__dict__['Any'] = typing.Any

# LangGraph imports for building the agent graph
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports for LLM and tool integration
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

# Pydantic for input validation
from pydantic import BaseModel, Field

# Local imports
from config import get_settings
from config.settings import Settings
from survyai.proxy_chat_model import SurvyAIProxyChatModel
from utils.logger import get_logger
from utils.token_limiter import (
    estimate_message_tokens,
    check_tpm_limit,
    chunk_messages,
    wait_for_rate_limit,
    format_token_warning,
)
from tools import (
    ExcelProcessor,
    DocumentProcessor,
    AutoCADProcessor,
    BlueMarbleConverter,
    GeographicCalculatorCLI,
    ArcGISProcessor,
    VectorStore,
    COLLECTION_DOCUMENTS,
    COLLECTION_DRAWINGS,
    COLLECTION_COORDINATES,
    COLLECTION_CONVERSATIONS,
)
from tools.autocad_processor import DXFProcessor
from tools.geopandas_tools import GeoPandasExecutor
from datetime import datetime
from utils.coordinate_parsing import extract_points, infer_crs_from_text
from utils.area import best_area
from utils.internet import (
    internet_search as _internet_search,
    research as _web_research,
    rule_based_query_variants as _rule_based_query_variants,
)

# Prompts and state live in separate modules for smaller, maintainable agent.py
from agent.prompts import SYSTEM_PROMPT
from agent.runtime_config import resolve_agent_runtime_config
from agent.state import (
    AgentState,
    PromptActionAssessment,
    RAGRouteDecision,
    looks_like_file_driven_task,
)
from runtime_paths import is_frozen_app, project_root, resource_path, user_data_path
from survyai.feature_flags import FeatureFlags

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

# Get a logger instance for this module
# All log messages will be prefixed with 'agent' for easy filtering
logger = get_logger(__name__)


# Cadastral DWG fast-path must not capture ArcGIS / volumetric workflows that only
# reference a .dwg as a boundary (those prompts lack generate 'Out.dwg' cadastral output).
_CADASTRAL_FASTPATH_EXCLUDE_MARKERS: Tuple[str, ...] = (
    "arcgis",
    "arcgis pro",
    "arcpy",
    "cutfill",
    "cut fill",
    "cut/fill",
    "idw",
    "inverse distance",
    "geoprocessing",
    "spatial analyst",
    "feature class",
    "shapefile",
    "file geodatabase",
    ".gdb",
    "geodatabase",
    "volume between",
    "borrow pit",
    "cut fill tool",
    "inverse distance weighting",
)


# Stop cadastral metadata captures at the next labelled field (newline or comma-separated).
from agent.pdf_survey_plan import (
    CADASTRAL_FIELD_BOUNDARY as _CADASTRAL_NEXT_FIELD,
    CADASTRAL_COORDINATES_FOR_STOP as _COORDINATES_FOR_STOP,
    extract_coordinates_blob_from_cadastral_query,
    resolve_cadastral_coordinates_blob,
)
from agent.excel_cadastral import (
    coordinates_deferred_to_external_source,
    default_excel_plot_output_name,
    explicit_cadastral_plot_intent,
    is_tabular_inspect_only_request,
)


_ACCESS_ROAD_SPEC_RE = re.compile(
    r"(?:add\s+)?(?:another\s+)?(?:an?\s+)?access(?:\s+road)?\s+(?:of\s+)?(?:width\s+)?(\d+(?:\.\d+)?)\s*m\s+.*?"
    r"(?:on\s+the\s+side\s+of|(?:on|along)\s+(?:the\s+)?side(?:\s+of)?|joining\s+pillars|(?:on|along)\s+(?:the\s+)?(?:boundary\s+line\s+)?connecting)\s+(.+?)"
    r"(?=\s*;|,\s*and\s+(?:yet\s+)?(?:another\s+)?(?:road|access)|\s+and\s+(?:yet\s+)?(?:another\s+)?(?:road|access)|"
    r"\s*Add\s+(?:another\s+)?(?:an?\s+)?access|\s*and\s+add\s+(?:another\s+)?(?:an?\s+)?access|"
    r"\n\s*(?:Plot\s+using|date\s+on|title\s+as|pillar\s+numbers?|coordinates\s+for)|\.\s*Add|\.\s*$|$)",
    re.IGNORECASE | re.DOTALL,
)

_EXTRA_ROAD_SPEC_RE = re.compile(
    r"(?:,\s*|\s*)(?:and\s+)?(?:yet\s+another\s+|another\s+)road\s+(?:of\s+)?(?:width\s+)?(\d+(?:\.\d+)?)\s*m\s+.*?"
    r"(?:on\s+the\s+side\s+of|(?:on|along)\s+(?:the\s+)?side(?:\s+of)?)\s+(.+?)"
    r"(?=,\s*and\s+(?:yet\s+)?(?:another\s+)?(?:road|access)|\s+and\s+(?:yet\s+)?(?:another\s+)?(?:road|access)|"
    r"\s*Add\s|\.\s*Add|\.\s*$|$)",
    re.IGNORECASE | re.DOTALL,
)


# Optional GUI/IPC handler: (path, mode) -> bool. Set by the desktop agent worker.
_cad_file_conflict_handler: Optional[Callable[[str, str], bool]] = None


def set_cad_file_conflict_handler(
    handler: Optional[Callable[[str, str], bool]],
) -> None:
    """Install a confirm callback used by CAD overwrite/modify prompts (GUI IPC)."""
    global _cad_file_conflict_handler
    _cad_file_conflict_handler = handler


def _confirm_overwrite_existing_dwg(path: str, *, mode: str = "overwrite") -> bool:
    """
    Ask whether to overwrite or modify an existing output DWG.

    Prefers the GUI handler (styled SurvyAI dialog). Falls back to a native
    MessageBox only when no handler is installed (e.g. CLI).
    """
    handler = _cad_file_conflict_handler
    if callable(handler):
        try:
            return bool(handler(str(path or ""), str(mode or "overwrite")))
        except Exception:
            return False
    try:
        import ctypes

        mode_l = (mode or "overwrite").strip().lower()
        if mode_l == "modify":
            text = (
                f"This drawing already exists:\n\n{path}\n\n"
                "Do you want to apply modifications to this existing drawing?\n\n"
                "Yes — continue and modify the drawing\n"
                "No — cancel and leave the drawing unchanged"
            )
            title = "SurvyAI — Modify existing drawing"
        else:
            text = (
                f"This drawing already exists:\n\n{path}\n\n"
                "Do you want to overwrite it with a new plan from the template?\n\n"
                "Yes — overwrite the existing drawing\n"
                "No — keep the existing drawing unchanged"
            )
            title = "SurvyAI — Drawing already exists"
        # MB_YESNO | MB_ICONWARNING | MB_SETFOREGROUND | MB_TOPMOST | MB_SYSTEMMODAL
        flags = 0x00000004 | 0x00000030 | 0x00010000 | 0x00040000 | 0x00001000
        try:
            ctypes.windll.user32.AllowSetForegroundWindow(0xFFFFFFFF)
        except Exception:
            pass
        result = ctypes.windll.user32.MessageBoxW(0, text, title, flags)
        return int(result) == 6  # IDYES
    except Exception:
        return False


def _ollama_ram_policy(model_name: str = "") -> Tuple[bool, str, int]:
    """Host-aware Ollama safety policy (see ``survyai.ollama_support.ollama_ram_policy``)."""
    from survyai.ollama_support import ollama_ram_policy

    return ollama_ram_policy(model_name)


def _parse_access_road_specs_from_query(query: str) -> List[str]:
    """Extract one or more access-road specs from natural-language cadastral prompts."""
    q = query or ""
    specs: List[str] = []
    seen: set[str] = set()

    def _add(width: str, ref: str, seg: str, *, boundary: bool = False) -> None:
        ref = re.sub(r"\s+", " ", (ref or "").strip()).strip(" ,.;")
        ref = re.sub(r"\s+to\s+", " and ", ref, flags=re.IGNORECASE)
        if not width or not ref:
            return
        if boundary:
            spec = f"{width}m width on the boundary line connecting {ref}"
        else:
            spec = f"{width}m width on the side of {ref}"
        m_o = re.search(
            r"offset\s+of\s+(\d+(?:\.\d+)?)\s*m|offset\s+(\d+(?:\.\d+)?)\s*m",
            seg,
            re.IGNORECASE,
        )
        if m_o:
            spec += f" offset {(m_o.group(1) or m_o.group(2))}m"
        key = spec.lower()
        if key not in seen:
            seen.add(key)
            specs.append(spec)

    quoted = re.search(r"access\s+road\s*=\s*'([^']+)'", q, re.IGNORECASE)
    if not quoted:
        quoted = re.search(r'access\s+road\s*=\s*"([^"]+)"', q, re.IGNORECASE)
    if quoted:
        for part in re.split(
            r"\s*;\s*|\s+and\s+an?\s+access\s+|\s+,\s*and\s+an?\s+access\s+",
            quoted.group(1),
            flags=re.IGNORECASE,
        ):
            part = (part or "").strip()
            if part and (re.search(r"\d+(?:\.\d+)?\s*m", part) or "width" in part.lower()):
                key = part.lower()
                if key not in seen:
                    seen.add(key)
                    specs.append(part)
        if specs:
            return specs

    for m in _ACCESS_ROAD_SPEC_RE.finditer(q):
        _add(m.group(1), m.group(2), m.group(0))

    for m in _EXTRA_ROAD_SPEC_RE.finditer(q):
        _add(m.group(1), m.group(2), m.group(0))

    # Legacy single-match patterns (boundary / joining pillars) when finditer misses phrasing
    if not specs:
        segments = re.split(
            r"(?<=[.;])\s+|\s*;\s*(?:Add\s+)?(?:another\s+)?(?:an?\s+)?access\s+|\s+and\s+add\s+(?:another\s+)?(?:an?\s+)?access\s+",
            q,
            flags=re.IGNORECASE,
        )
        for seg in segments:
            seg = (seg or "").strip()
            if not seg or not re.search(r"access|road|width|side\s+of|connecting|joining", seg, re.IGNORECASE):
                continue
            m_ar = re.search(
                r"access\s+road\s+of\s+(\d+(?:\.\d+)?)\s*m\s+width\s+(?:should\s+be\s+)?(?:on|along)\s+(?:the\s+)?(?:boundary\s+line\s+)?connecting\s+(.+?)(?:\.|$)",
                seg,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if m_ar:
                _add(m_ar.group(1), m_ar.group(2), seg, boundary=True)
                continue
            m_ar2 = re.search(
                r"(?:add\s+)?(?:another\s+)?(?:an?\s+)?access(?:\s+road)?\s+(?:of\s+)?(?:width\s+)?(\d+(?:\.\d+)?)\s*m\s+.*?joining\s+pillars\s+(.+?)(?:\.|$)",
                seg,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if m_ar2:
                _add(m_ar2.group(1), f"pillars {m_ar2.group(2).strip()}", seg)
                continue
            m_ar3 = re.search(
                r"(?:add\s+)?(?:another\s+)?(?:an?\s+)?access(?:\s+road)?\s+(?:of\s+)?(?:width\s+)?(\d+(?:\.\d+)?)\s*m\s+.*?(?:on\s+the\s+side\s+of|(?:on|along)\s+(?:the\s+)?side(?:\s+of)?)\s+(.+?)(?=\s*;|,\s*and\s+(?:another\s+)?(?:an?\s+)?access|\s+and\s+(?:another\s+)?(?:an?\s+)?access|\s*Add\s|$|\n\s*(?:Plot\s+using|date\s+on|title\s+as)|\.)",
                seg,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if m_ar3:
                _add(m_ar3.group(1), m_ar3.group(2), seg)

    return specs


def _split_buyer_owner_names(name: str) -> List[str]:
    """Split a buyer/owner string into ordered display names (preserves sequence)."""
    raw = (name or "").strip()
    if not raw:
        return []
    # Normalize "A, B and C" / "A & B" into a flat ordered list.
    chunks = [p.strip() for p in re.split(r",\s*", raw) if p.strip()]
    names: List[str] = []
    for chunk in chunks:
        and_parts = [
            p.strip()
            for p in re.split(r"\s+(?:and|&)\s+", chunk, flags=re.IGNORECASE)
            if p.strip()
        ]
        if len(and_parts) <= 1:
            names.append(and_parts[0] if and_parts else chunk)
        else:
            names.extend(and_parts)
    return names


def _pack_owner_names_into_title_lines(
    names: Sequence[str],
    *,
    max_chars_per_line: int = 38,
) -> List[str]:
    """
    Pack consecutive owner names onto title-block rows (ordered only).

    Names stay in input order: 1st+2nd may share a row when they fit; never
    jump ahead to pair 1st with 4th. No per-name text shrinking — only line breaks.
    """
    max_chars = max(12, int(max_chars_per_line))
    clean = [re.sub(r"\s+", " ", str(n or "").strip()) for n in names if str(n or "").strip()]
    if not clean:
        return []
    if len(clean) == 1:
        return [clean[0]]

    lines: List[str] = []
    current_parts: List[str] = []

    def _flush(final: bool = False) -> None:
        nonlocal current_parts
        if not current_parts:
            return
        # Trailing comma on every row except the last completed block when finalizing last line.
        body = ", ".join(current_parts)
        if not final:
            body = f"{body},"
        lines.append(body)
        current_parts = []

    for i, nm in enumerate(clean):
        is_last = i == len(clean) - 1
        if not current_parts:
            current_parts = [nm]
            if is_last:
                _flush(final=True)
            continue
        candidate = ", ".join(current_parts + [nm])
        # Leave room for a trailing comma on non-final rows.
        limit = max_chars if is_last and len(current_parts) >= 1 else max_chars - 1
        if len(candidate) <= limit:
            current_parts.append(nm)
            if is_last:
                _flush(final=True)
        else:
            _flush(final=False)
            current_parts = [nm]
            if is_last:
                _flush(final=True)
    if current_parts:
        _flush(final=True)
    return lines


def _pack_location_clauses_into_lines(
    parts: Sequence[str],
    *,
    max_chars_per_line: int = 36,
    preferred_max_lines: int = 3,
) -> List[str]:
    """
    Pack location address clauses into as few title-block lines as practical.

    Cartographic default: prefer ≤ ``preferred_max_lines`` (3). Consecutive clauses
    share a line when they fit; only exceed 3 lines when the address cannot
    reasonably compress further.
    """
    clean = [re.sub(r"\s+", " ", str(p or "").strip()) for p in parts if str(p or "").strip()]
    if not clean:
        return []

    def _wrap_long_clause(clause: str, limit: int) -> List[str]:
        if len(clause) <= limit:
            return [clause]
        words = clause.split()
        out: List[str] = []
        buf: List[str] = []
        for w in words:
            cand = (" ".join(buf + [w])).strip()
            if buf and len(cand) > limit:
                out.append(" ".join(buf))
                buf = [w]
            else:
                buf.append(w)
        if buf:
            out.append(" ".join(buf))
        return out or [clause]

    def _pack(limit: int) -> List[str]:
        lines: List[str] = []
        current = ""
        for clause in clean:
            chunks = _wrap_long_clause(clause, limit)
            for chunk in chunks:
                if not current:
                    current = chunk
                    continue
                candidate = f"{current}, {chunk}"
                if len(candidate) <= limit:
                    current = candidate
                else:
                    lines.append(current)
                    current = chunk
        if current:
            lines.append(current)
        return lines

    limit = max(16, int(max_chars_per_line))
    preferred = max(1, int(preferred_max_lines))
    best = _pack(limit)
    if len(best) <= preferred:
        return best
    for widen in (1.15, 1.30, 1.50, 1.75):
        wider = max(limit, int(round(limit * widen)))
        packed = _pack(wider)
        if len(packed) < len(best):
            best = packed
        if len(best) <= preferred:
            return best
    return best


def _format_location_for_titleblock(
    location: str,
    *,
    max_chars_per_line: Optional[int] = None,
    preferred_max_lines: int = 3,
) -> str:
    """
    Format location clauses for the title-block AT row as MTEXT lines.

    Comma- or newline-separated clauses are packed into preferably ≤3 lines
    (cartographic compression). LGA / state stay on their own title-block rows.
    """
    raw = (location or "").strip()
    if not raw:
        return ""
    parts = [p.strip(" ,;:-") for p in re.split(r"\s*,\s*|\n+", raw) if p.strip(" ,;:-")]
    if not parts:
        return raw.upper()
    packed = _pack_location_clauses_into_lines(
        parts,
        max_chars_per_line=int(max_chars_per_line) if max_chars_per_line else 36,
        preferred_max_lines=preferred_max_lines,
    )
    return "\\P".join(p.upper() for p in packed)


def _format_buyer_name_for_titleblock(
    name: str,
    *,
    max_chars_per_line: Optional[int] = None,
) -> str:
    """
    Format buyer/owner names for CADA_TITLEBLOCK row 2 (MTEXT with \\P line breaks).

    Multi-owner lists pack consecutive names onto the same row when they fit
    (cartographic compression of vertical title space). Single / dual names keep
    a simple readable layout. \"AND\" still appears on its own line when the source
    used \"and\" between the last two groups and packing is not used (<3 names).
    """
    raw = (name or "").strip()
    if not raw:
        return ""
    names = _split_buyer_owner_names(raw)
    if not names:
        return raw.upper()

    # Multi-owner / multi-parcel title: pack consecutive names to reduce title height.
    if len(names) >= 3:
        max_chars = int(max_chars_per_line) if max_chars_per_line else 38
        packed = _pack_owner_names_into_title_lines(names, max_chars_per_line=max_chars)
        return "\\P".join(line.upper() for line in packed)

    # 1–2 names (or legacy \"A and B\"): keep explicit AND line when present in source.
    parts = [p.strip() for p in re.split(r",\s*", raw) if p.strip()]
    lines: List[str] = []
    if len(parts) > 1:
        for part in parts[:-1]:
            lines.append(f"{part},")
        tail = parts[-1]
    else:
        tail = parts[0] if parts else raw
    and_parts = [p.strip() for p in re.split(r"\s+and\s+", tail, flags=re.IGNORECASE) if p.strip()]
    if len(and_parts) <= 1:
        lines.append(and_parts[0] if and_parts else tail)
    else:
        for idx, segment in enumerate(and_parts):
            if idx > 0:
                lines.append("AND")
            lines.append(segment)
    return "\\P".join(line.upper() for line in lines)


def _titleblock_owner_line_count(formatted_buyer: str) -> int:
    text = (formatted_buyer or "").strip()
    if not text:
        return 1
    return max(1, text.count("\\P") + 1)


def _mtext_content_line_count(raw: str) -> int:
    """Count MTEXT content lines in a table cell, ignoring format wrapper codes."""
    raw = raw or ""
    content = raw
    if raw.startswith("{") and raw.endswith("}") and ";" in raw:
        idx = raw.rfind(";")
        content = raw[idx + 1 : -1]
    content = content.strip()
    if not content:
        return 1
    return max(1, content.count("\\P") + 1)


# Approximate MTEXT width in drawing units (condensed title-block fonts).
_CARTO_CHAR_WIDTH_FACTOR = 0.50


def _mtext_plain_len(text: str) -> int:
    """Visible character count for layout (ignores AutoCAD format codes)."""
    t = str(text or "")
    t = t.replace("\\P", "")
    t = re.sub(r"\\[A-Za-z][^;\\]*;", "", t)
    t = re.sub(r"\{[^}]*\}", "", t)
    return len(t.strip())


def _mtext_height_prefix(height_scale: float) -> str:
    """Flat MTEXT height prefix (must not be nested inside another {…} group)."""
    scale = float(height_scale)
    if scale >= 0.999:
        return ""
    scale_s = f"{scale:.3f}".rstrip("0").rstrip(".")
    return f"\\H{scale_s}x;"


def _mtext_apply_height_scale(content: str, height_scale: float) -> str:
    """Inline MTEXT height override for a plain-text fragment."""
    scale = float(height_scale)
    if scale >= 0.999 or not (content or "").strip():
        return content or ""
    scale_s = f"{scale:.3f}".rstrip("0").rstrip(".")
    return f"{{\\H{scale_s}x;{content}}}"


def _mtext_with_uniform_height(content: str, height_scale: float) -> str:
    """Apply one \\H override to an entire cell body (single or \\P-separated lines)."""
    if height_scale >= 0.999 or not (content or "").strip():
        return content or ""
    body = content or ""
    if body.startswith("{") and body.endswith("}"):
        return body
    return _mtext_height_prefix(height_scale) + body


def _mtext_preserve_style_set_content(
    existing_cell: str,
    new_content: str,
    *,
    height_scale: float = 1.0,
) -> str:
    """Keep colour/font wrapper; replace textual body (prevents template leakage)."""
    body = _mtext_with_uniform_height(new_content or "", height_scale)
    raw = existing_cell or ""
    if raw.startswith("{") and raw.endswith("}"):
        color = ""
        font = ""
        try:
            m = re.search(r"(\\C\d+;)", raw)
            if m:
                color = m.group(1)
        except Exception:
            color = ""
        try:
            m = re.search(r"(\\f[^;]+;)", raw)
            if m:
                font = m.group(1)
        except Exception:
            font = ""
        if color or font:
            # Flat {\\f;\\C;\\H;text} — nested {\\H{…}} groups break table-cell height.
            return "{" + (color or "") + (font or "") + body + "}"
    if raw.startswith("{") and raw.endswith("}") and ";" in raw:
        idx = raw.rfind(";")
        return raw[: idx + 1] + body + "}"
    return body


def _mtext_strip_inline_size_codes(text: str) -> str:
    """Remove inline \\H / \\W overrides that fight SetTextHeight in TABLE cells."""
    t = str(text or "")
    t = re.sub(r"\\H[\d.]+x;", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\\W[\d.]+x;", "", t, flags=re.IGNORECASE)
    return t


def _ideal_plan_number_text_height(
    *,
    template_nominal_h: Optional[float],
    template_denom: int,
    chosen_denom: int,
    profile_ref: Optional[float] = None,
) -> float:
    """
    Nominal CADA_PLANNUMBER height at the output plan scale.

    ``template_denom`` must be the template authoring scale (1:500), not the
    output scale. Example: 1.2 at 1:500 → 0.6 at 1:250.
    """
    ref = float(profile_ref or 0.0)
    if ref <= 0:
        ref = float(template_nominal_h or 0.0)
    if ref <= 0:
        ref = _CADASTRAL_TEMPLATE_PLANNUMBER_REF_H
    td = max(1, int(template_denom or _CADASTRAL_TEMPLATE_REF_DENOM))
    cd = max(1, int(chosen_denom or td))
    return ref * (float(cd) / float(td))


_CADASTRAL_ALLOWED_SCALES = [250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 25000]
# Ground span (m) at/below which cartography prefers 1:250 when the sheet fit allows.
_CADASTRAL_AUTO_FINE_SPAN_M = 45.0
# survey_plan_template3.dwg CADA_PLANNUMBER nominal height at template authoring scale 1:500.
_CADASTRAL_TEMPLATE_REF_DENOM = 500
_CADASTRAL_TEMPLATE_PLANNUMBER_REF_H = 1.2
# CADA_PILLARNUMBERS single-column TABLE width at template authoring scale 1:500.
# Widened from 8.0 → 9.2 so longer Nigerian peg tokens fit on one line.
_CADASTRAL_TEMPLATE_PILLARNUMBER_REF_W = 9.2


def _ideal_pillar_number_table_width(
    *,
    template_denom: int,
    chosen_denom: int,
    template_nominal_w: Optional[float] = None,
) -> float:
    """
    Drawing-unit width for CADA_PILLARNUMBERS tables at the output plan scale.

    ``template_nominal_w`` defaults to the survey_plan_template3 authoring width (9.2 @ 1:500).
    """
    ref = float(template_nominal_w) if template_nominal_w and float(template_nominal_w) > 0 else (
        _CADASTRAL_TEMPLATE_PILLARNUMBER_REF_W
    )
    td = max(1, int(template_denom or _CADASTRAL_TEMPLATE_REF_DENOM))
    cd = max(1, int(chosen_denom or td))
    return float(ref) * (float(cd) / float(td))


def _cartographic_denom_for_layout_extent(
    width_m: float,
    height_m: float,
    allowed_denoms: Optional[List[int]] = None,
    *,
    parcel_count: int = 1,
    dense_title: bool = False,
) -> int:
    """
    Suggested cadastral plan scale from ground extent (meters).

    Span bands follow common survey/cartography practice for ownership layouts
    (versatile across sites — not tuned to one workbook). Dense multi-owner title
    blocks may take one coarser step within the same band for sheet room.

    Very small parcels (span ≤ ~45 m) prefer 1:250 so bearings/pillars remain
    readable — the inverse of auto-coarsening large land to 1:1000+.
    """
    allowed = allowed_denoms or _CADASTRAL_ALLOWED_SCALES
    span = max(float(width_m or 0.0), float(height_m or 0.0))
    # Unknown extent → template convention (1:500), never auto-fine.
    if span <= 1e-6:
        target = 500
    elif span <= _CADASTRAL_AUTO_FINE_SPAN_M:
        target = 250
    elif span <= 100:
        target = 500
    elif span <= 220:
        target = 1000
    elif span <= 400:
        target = 2000
    elif span <= 750:
        target = 5000
    elif span <= 1400:
        target = 5000
    elif span <= 2500:
        target = 10000
    else:
        target = 20000
    # Many owners / dense legend: prefer a touch more sheet inside the same span class.
    # Never coarsen away from 1:250 for tiny single parcels.
    if (
        (dense_title or int(parcel_count) >= 8)
        and span > _CADASTRAL_AUTO_FINE_SPAN_M
        and span <= 400
        and target <= 2000
    ):
        target = 5000
    cands = [d for d in allowed if d >= int(target)]
    return int(min(cands) if cands else max(allowed))


def _allowed_denom_steps_above(
    start_denom: int,
    steps: int,
    allowed_denoms: Optional[List[int]] = None,
) -> int:
    """Return the denom `steps` coarser than start (or the coarsest allowed)."""
    allowed = sorted(allowed_denoms or _CADASTRAL_ALLOWED_SCALES)
    at_or_above = [d for d in allowed if d >= int(start_denom)]
    if not at_or_above:
        return int(max(allowed))
    idx = min(max(0, int(steps)), len(at_or_above) - 1)
    return int(at_or_above[idx])


def _choose_layout_plan_scale_after_plot(
    *,
    plotted_w: float,
    plotted_h: float,
    usable_w: float,
    usable_h: float,
    interior_w: float,
    interior_h: float,
    template_denom: int,
    user_scale_denom: Optional[int],
    parcel_count: int = 1,
    allowed_denoms: Optional[List[int]] = None,
) -> tuple[int, float, str]:
    """
    Intelligent post-plot scale for multi-parcel layouts.

    1) Cartographic scale from ground span (primary — versatile across prompts/sites)
    2) Fit-required scale from measured plot window (secondary, only if trustworthy)
    3) Cap so pathological usable-window math cannot explode to 1:20000 for mid-size land
    """
    allowed = allowed_denoms or _CADASTRAL_ALLOWED_SCALES
    td = max(1, int(template_denom or 500))
    pw = max(float(plotted_w or 0.0), 1e-6)
    ph = max(float(plotted_h or 0.0), 1e-6)
    uw = max(float(usable_w or 0.0), 1e-6)
    uh = max(float(usable_h or 0.0), 1e-6)
    iw = max(float(interior_w or 0.0), 1e-6)
    ih = max(float(interior_h or 0.0), 1e-6)

    carto = _cartographic_denom_for_layout_extent(
        pw,
        ph,
        allowed,
        parcel_count=int(parcel_count or 1),
        dense_title=int(parcel_count or 1) >= 6,
    )
    # Soft enclosure headroom: at most one allowed step coarser than cartographic
    # (e.g. 1:5000 → 1:10000). Two steps would still allow 1:20000 for mid-size land.
    carto_cap = _allowed_denom_steps_above(carto, 1, allowed)

    pad_w = max(pw * 1.12, pw + 10.0)
    pad_h = max(ph * 1.12, ph + 10.0)
    req_k = max(pad_w / uw, pad_h / uh)
    fit_denom, _fit_k, _fit_reason = _resolve_cadastral_output_scale(
        template_denom=td,
        user_scale_denom=None,
        required_k=float(req_k),
        allowed_denoms=allowed,
        multi_parcel_layout=False,
        boundary_w=pw,
        boundary_h=ph,
        enforce_enclosure=True,
    )

    usable_frac = (uw * uh) / max(iw * ih, 1e-6)
    pathological_window = (
        usable_frac < 0.18
        or uw < 0.22 * iw
        or uh < 0.28 * ih
        or req_k > (float(carto) / float(td)) * 1.15
    )

    if user_scale_denom and int(user_scale_denom) in allowed:
        usd = int(user_scale_denom)
        # Honour user scale when it can enclose under a trustworthy window, else fall through.
        if (not pathological_window) and req_k <= (float(usd) / float(td)) + 0.02:
            return usd, float(usd) / float(td), "user_scale"
        if pathological_window and usd >= carto:
            return usd, float(usd) / float(td), "user_scale"

    if pathological_window:
        chosen = int(carto)
        reason = "layout_cartographic"
    elif int(carto) < int(fit_denom) and req_k <= (float(carto) / float(td)) + 0.02:
        # Tiny layouts: cartographic 1:250 must not be blocked by template fit_denom.
        chosen = int(carto)
        reason = "layout_cartographic_fine"
    else:
        chosen = max(int(carto), int(fit_denom))
        if chosen > int(carto_cap):
            chosen = int(carto_cap)
            reason = "layout_cartographic_capped_fit"
        elif chosen > int(carto):
            reason = "layout_cartographic_and_fit"
        else:
            reason = "layout_cartographic"

    return int(chosen), float(chosen) / float(td), reason


def _multi_parcel_owner_label_height(
    parcel_rings: Sequence[Sequence[Dict[str, float]]],
    *,
    layout_w: float,
    layout_h: float,
    native_bearing_height: float,
) -> float:
    """
    Cartographic owner-label height (drawing meters) for multi-parcel layouts.

    Sized from typical parcel span so labels stay inside parcels; independent of
    an over-large sheet scale factor.
    """
    spans: List[float] = []
    for ring in parcel_rings or []:
        xs = [float(p.get("x", 0.0)) for p in ring if isinstance(p, dict)]
        ys = [float(p.get("y", 0.0)) for p in ring if isinstance(p, dict)]
        if len(xs) < 2:
            continue
        spans.append(max(max(xs) - min(xs), max(ys) - min(ys)))
    spans_sorted = sorted(spans)
    median_span = spans_sorted[len(spans_sorted) // 2] if spans_sorted else 0.0
    layout_span = max(float(layout_w or 0.0), float(layout_h or 0.0), 1.0)
    native = max(0.5, float(native_bearing_height or 1.2))
    # ~7% of typical parcel, but never larger than ~1.5% of whole layout span.
    h = median_span * 0.07 if median_span > 1e-6 else native
    h = min(h, layout_span * 0.015)
    h = max(1.0, min(h, native * 1.35, 12.0))
    if median_span > 1e-6:
        h = min(h, max(1.0, median_span * 0.22))
    return float(h)


def _resolve_cadastral_output_scale(
    *,
    template_denom: int,
    user_scale_denom: Optional[int],
    required_k: float,
    allowed_denoms: Optional[List[int]] = None,
    multi_parcel_layout: bool = False,
    boundary_w: float = 0.0,
    boundary_h: float = 0.0,
    enforce_enclosure: bool = False,
) -> tuple[int, float, str]:
    """
    Choose output plan scale (denominator) and sheet scale factor.

    Fit criterion (symmetric):
    - ``required_k`` is the minimum sheet scale factor needed for the padded
      parcel to sit inside the usable interior window (``scale_k = denom / template``).
    - If ``required_k > 1``, auto-**coarsen** (e.g. 1:500 → 1:1000+) so the land fits.
    - If ``required_k`` is small enough that a finer scale still fits, auto-**refine**
      (typically template 1:500 → 1:250) for very small parcels — same gate, opposite
      direction. Cartographic span bands decide *whether* refining is desirable;
      fit decides *whether it is allowed*.

    When the user/PDF states a scale (e.g. 1:250), that scale is used unless the
    parcel (+ road padding) cannot fit — then the next coarser allowed scale is chosen.

    ``enforce_enclosure=True`` (plot-then-fit): never choose a scale finer than the
    fit-required factor — the border must enclose the plotted land.
    """
    import math

    allowed = allowed_denoms or _CADASTRAL_ALLOWED_SCALES
    td = max(1, int(template_denom or 500))
    rk = max(0.0, float(required_k))
    span = max(float(boundary_w or 0.0), float(boundary_h or 0.0))

    if user_scale_denom and int(user_scale_denom) in allowed:
        usd = int(user_scale_denom)
        scale_k_pref = float(usd) / float(td)
        if rk <= scale_k_pref + 0.015:
            return usd, scale_k_pref, "user_scale"
        min_denom = max(usd, int(math.ceil(rk * float(td) - 1e-9)))
        candidates = [d for d in allowed if d >= min_denom]
        chosen = min(candidates) if candidates else max(allowed)
        return chosen, float(chosen) / float(td), "user_scale_overflow"

    chosen = td
    scale_k = 1.0
    reason = "template"
    if rk > 1.0 + 1e-6:
        target_denom = float(td) * rk * 1.02
        candidates = [s for s in allowed if s >= target_denom and s >= td]
        chosen = min(candidates) if candidates else max(allowed)
        scale_k = float(chosen) / float(td)
        reason = "auto_upscale"
    else:
        # Auto-downscale (refine): mirror of auto_upscale — only when fit allows.
        # Typical path: template 1:500 → 1:250 when the parcel is cartographically small.
        finer = sorted(d for d in allowed if int(d) < int(td))
        if finer and not (multi_parcel_layout and span > _CADASTRAL_AUTO_FINE_SPAN_M):
            carto = _cartographic_denom_for_layout_extent(
                boundary_w,
                boundary_h,
                allowed,
                parcel_count=1,
                dense_title=False,
            )
            # Prefer cartographic fine scale if it still encloses; else any finer
            # denom that fits when span is in the tiny-parcel band.
            candidates_fit = [
                d
                for d in finer
                if rk <= (float(d) / float(td)) + 0.015
            ]
            if candidates_fit:
                prefer = int(carto) if int(carto) in candidates_fit else None
                if prefer is None and span > 1e-6 and span <= _CADASTRAL_AUTO_FINE_SPAN_M:
                    # Small ground span: pick the finest that still fits (usually 250).
                    prefer = int(min(candidates_fit))
                if prefer is not None and int(prefer) < int(td):
                    # Enclosure mode: never go finer than fit requires (already gated).
                    if enforce_enclosure and rk > (float(prefer) / float(td)) + 0.015:
                        prefer = None
                    if prefer is not None:
                        chosen = int(prefer)
                        scale_k = float(chosen) / float(td)
                        reason = "auto_downscale"

    # Pre-plot estimate only: may trim pathological oversize. Never when enclosing plotted land.
    if (
        multi_parcel_layout
        and not enforce_enclosure
        and (boundary_w > 1e-6 or boundary_h > 1e-6)
    ):
        carto = _cartographic_denom_for_layout_extent(boundary_w, boundary_h, allowed)
        if int(chosen) > int(carto) * 2:
            chosen = int(carto)
            scale_k = float(chosen) / float(td)
            reason = "multi_parcel_cartographic"
        elif int(chosen) > int(carto) and rk <= (float(carto) / float(td)) * 1.08:
            chosen = int(carto)
            scale_k = float(chosen) / float(td)
            reason = "multi_parcel_cartographic"

    return int(chosen), float(scale_k), reason


def _cadastral_sheet_layers(profile: Dict[str, Any]) -> List[str]:
    """Sheet/presentation layers that may move/scale — never survey geometry."""
    layers = list(profile.get("sheet_layers") or []) or [
        "CADA_BORDER",
        "CADA_INTERIORBORDER",
        "CADA_SCALEBAR",
        "CADA_NORTHARROW",
        "CADA_EASTARROW",
        "CADA_TITLEBLOCK",
        "CADA_PLANNUMBER",
        "CADA_CERTIFICATION",
        "CADA_SURVEYOR",
        "CADA_COORDINATES",
        "CADA_NORTHCOORDINATES",
        "CADA_EASTCOORDINATES",
        "CADA_PRIMARYPILLAR_ARROWS",
        "TITLE",
        "text",
    ]
    for req in [
        "CADA_BORDER",
        "CADA_INTERIORBORDER",
        "CADA_NORTHARROW",
        "CADA_EASTARROW",
        "CADA_NORTHCOORDINATES",
        "CADA_EASTCOORDINATES",
        "CADA_COORDINATES",
        "CADA_PRIMARYPILLAR_ARROWS",
        "CADA_SCALEBAR",
        "CADA_TITLEBLOCK",
        "CADA_PLANNUMBER",
        "CADA_CERTIFICATION",
        "CADA_SURVEYOR",
    ]:
        if req not in layers:
            layers.append(req)
    ban = {
        "CADA_BOUNDARY",
        "CADA_PILLARS",
        "CADA_TEXT",
        "CADA_BEARING_DIST",
        "CADA_ROAD",
        "CADA_CWF",
        "CADA_PILLARNUMBERS",
    }
    return [L for L in layers if str(L).upper() not in ban]


def _cadastral_sheet_translate_layers(profile: Dict[str, Any]) -> List[str]:
    """
    Sheet layers safe to translate/recenter about the plot.

    Excludes ``CADA_PRIMARYPILLAR_ARROWS`` so the primary crosshair stays locked
    to the primary peg after ``move_all_modelspace``. Those arrows still scale
    about the template primary via ``_cadastral_sheet_layers`` / scale lists.
    """
    ban = {
        "CADA_PRIMARYPILLAR_ARROWS",
        "CADA_NORTHARROW",
        "CADA_EASTARROW",
        "CADA_COORDINATES",
        "CADA_NORTHCOORDINATES",
        "CADA_EASTCOORDINATES",
        "CADA_BOUNDARY",
        "CADA_BEARING_DIST",
        "CADA_PILLARS",
        "CADA_PILLARNUMBERS",
        "CADA_ROAD",
        "CADA_CWF",
        "CADA_TEXT",
    }
    return [L for L in _cadastral_sheet_layers(profile) if str(L).upper() not in ban]


def _usable_plot_window_from_bboxes(
    interior_bb: Dict[str, Any],
    title_bb: Optional[Dict[str, Any]],
    *,
    margin: float = 0.08,
) -> Optional[Dict[str, float]]:
    """
    Usable land-plot rectangle inside the interior border, excluding the title block.
    Returns minx/miny/maxx/maxy/width/height or None.
    """
    if not interior_bb or not interior_bb.get("success"):
        return None
    ix0 = float(interior_bb.get("minx", 0.0))
    iy0 = float(interior_bb.get("miny", 0.0))
    ix1 = float(interior_bb.get("maxx", 0.0))
    iy1 = float(interior_bb.get("maxy", 0.0))
    iw = ix1 - ix0
    ih = iy1 - iy0
    if iw <= 1e-6 or ih <= 1e-6:
        return None
    minx = ix0 + margin * iw
    maxx = ix1 - margin * iw
    miny = iy0 + margin * ih
    maxy = iy1 - margin * ih
    if title_bb and title_bb.get("success"):
        tminx = float(title_bb.get("minx", 0.0))
        tminy = float(title_bb.get("miny", 0.0))
        tmaxx = float(title_bb.get("maxx", 0.0))
        tmaxy = float(title_bb.get("maxy", 0.0))
        tcx = 0.5 * (tminx + tmaxx)
        tcy = 0.5 * (tminy + tmaxy)
        gap = max(3.0, 0.02 * max(iw, ih))
        # Title on the right → plot band is to the left of title.
        if tcx > ix0 + 0.55 * iw and tminx < ix1 and tmaxx > ix0:
            maxx = min(maxx, tminx - gap)
        # Title in the upper band → plot band is below title.
        elif tcy > iy0 + 0.55 * ih and tminy < iy1 and tmaxy > iy0:
            maxy = min(maxy, tminy - gap)
    if maxx <= minx + 1.0 or maxy <= miny + 1.0:
        # Fallback: majority of interior.
        minx = ix0 + margin * iw
        maxx = ix1 - margin * iw
        miny = iy0 + margin * ih
        maxy = iy1 - margin * ih
    return {
        "minx": float(minx),
        "miny": float(miny),
        "maxx": float(maxx),
        "maxy": float(maxy),
        "width": float(maxx - minx),
        "height": float(maxy - miny),
        "center_x": float(0.5 * (minx + maxx)),
        "center_y": float(0.5 * (miny + maxy)),
    }


def _split_address_segments(raw: str) -> List[str]:
    addr = (raw or "").strip().upper()
    if not addr:
        return []
    if "\\P" in addr:
        return [ln.strip() for ln in addr.split("\\P") if ln.strip()]
    return [p.strip() for p in re.split(r",\s*", addr) if p.strip()]


def _merge_city_state_tail(parts: List[str]) -> List[str]:
    """Prefer 'PORT HARCOURT, RIVERS STATE' on one line (cartographic convention)."""
    if len(parts) < 2:
        return parts
    city, state = parts[-2], parts[-1]
    state_u = state.upper()
    if "STATE" in state_u or state_u in {"FCT", "ABUJA"}:
        merged = parts[:-2] + [f"{city}, {state}"]
        return merged
    return parts


def _pack_segments_into_lines(
    parts: List[str],
    *,
    max_chars_per_line: int,
    max_lines: int,
) -> List[str]:
    """Greedy horizontal packing: join segments with ', ' until the line would overflow."""
    max_chars = max(8, int(max_chars_per_line))
    max_lines = max(1, int(max_lines))
    lines: List[str] = []
    current = ""
    for part in parts:
        segment = part.strip()
        if not segment:
            continue
        candidate = f"{current}, {segment}" if current else segment
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            lines.append(current)
            current = segment
    if current:
        lines.append(current)

    while len(lines) > max_lines and len(lines) >= 2:
        tail = lines.pop()
        lines[-1] = f"{lines[-1]}, {tail}"

    return lines[:max_lines]


# Plan numbers use BOLD_SURVEY — wider than body text in the same cell width.
_PLAN_NUMBER_CHAR_WIDTH_FACTOR = 0.64
# Cartographic rule: shrink to fit one line, but never below 85% of template height.
_PLAN_NUMBER_MIN_HEIGHT_SCALE = 0.85


def _plan_number_line_width(
    line: str, text_height: float, *, height_scale: float = 1.0
) -> float:
    h = max(0.01, float(text_height) * float(height_scale))
    return _mtext_plain_len(line) * h * _PLAN_NUMBER_CHAR_WIDTH_FACTOR


def _plan_number_line_fits(
    line: str,
    *,
    usable_width: float,
    text_height: float,
    height_scale: float,
) -> bool:
    if not (line or "").strip():
        return True
    return _plan_number_line_width(line, text_height, height_scale=height_scale) <= usable_width


def _plan_number_two_line_candidates(plain: str) -> List[str]:
    """
    Deliberate two-line breaks at slash/year boundaries (never mid-token like 202|6).
    """
    plain = (plain or "").strip()
    if not plain:
        return []
    out: List[str] = []
    seen: set[str] = set()

    def _add(left: str, right: str, *, keep_slash_on_left: bool) -> None:
        left = left.strip()
        right = right.strip()
        if not left or not right:
            return
        text = f"{left}/\\P{right}" if keep_slash_on_left else f"{left}\\P{right}"
        if text not in seen:
            seen.add(text)
            out.append(text)

    parts = [p for p in plain.split("/") if p != ""]
    if len(parts) >= 2:
        for i, part in enumerate(parts):
            if re.fullmatch(r"\d{4}", part):
                left = "/".join(parts[: i + 1])
                right = "/".join(parts[i + 1 :])
                if right:
                    _add(left, right, keep_slash_on_left=False)
        for i in range(len(parts) - 1, 0, -1):
            left = "/".join(parts[:i])
            right = "/".join(parts[i:])
            _add(left, right, keep_slash_on_left=False)
            _add(left, right, keep_slash_on_left=True)

    if "-" in plain:
        idx = plain.find("-", plain.find("/") if "/" in plain else 0)
        if 0 < idx < len(plain) - 1:
            _add(plain[:idx], plain[idx + 1 :], keep_slash_on_left=False)
            _add(plain[:idx], plain[idx:], keep_slash_on_left=False)

    if not out:
        mid = max(1, len(plain) // 2)
        out.append(f"{plain[:mid]}\\P{plain[mid:]}")
    return out


def _fit_plan_number_mtext(
    text: str,
    *,
    cell_width: float,
    cell_height: float,
    base_text_height: float,
    line_step: float,
    min_height_scale: float = _PLAN_NUMBER_MIN_HEIGHT_SCALE,
) -> tuple[str, float]:
    """
    Fit CADA_PLANNUMBER text without ever enlarging the template height.

    1. Try single line, reducing height in small steps down to ``min_height_scale`` (0.85×).
    2. Only if still too wide at 0.85×, use a cartographic two-line break (e.g. after year).
    """
    plain = (text or "").strip()
    if not plain or base_text_height <= 0:
        return plain, 1.0

    min_scale = max(0.5, min(1.0, float(min_height_scale)))
    pad_x = max(0.1, float(base_text_height) * 0.28)
    pad_y = max(0.08, float(base_text_height) * 0.22)
    usable_w = max(float(cell_width) - 2.0 * pad_x, float(base_text_height) * 3.0)
    step = float(line_step) if line_step > 0 else float(base_text_height) * (5.0 / 3.0)
    usable_h = (
        max(float(cell_height) - 2.0 * pad_y, step * 2.0)
        if cell_height > 0
        else step * 2.5
    )

    scales: List[float] = []
    s = 1.0
    while True:
        scales.append(round(s, 4))
        if s <= min_scale + 1e-6:
            break
        s = max(min_scale, round(s - 0.01, 4))

    for scale in scales:
        if _plan_number_line_fits(
            plain, usable_width=usable_w, text_height=base_text_height, height_scale=scale
        ):
            # Ladder runs 1.0 → min_scale: first match is the largest size that fits one line.
            return plain, scale

    two_line_candidates = _plan_number_two_line_candidates(plain)

    def _plan_break_rank(candidate: str) -> int:
        left = candidate.split("\\P")[0]
        if re.search(r"/\d{4}$", left):
            return 0
        return 1

    two_line_candidates.sort(key=_plan_break_rank)

    # Prefer two lines at full ideal height before shrinking below 0.85×.
    for candidate in two_line_candidates:
        lines = [ln for ln in candidate.split("\\P") if ln.strip()]
        if len(lines) > 2:
            continue
        if len(lines) == 2 and 2.0 * step > usable_h + 1e-6:
            continue
        if all(
            _plan_number_line_fits(
                ln, usable_width=usable_w, text_height=base_text_height, height_scale=1.0
            )
            for ln in lines
        ):
            return candidate, 1.0

    for candidate in two_line_candidates:
        lines = [ln for ln in candidate.split("\\P") if ln.strip()]
        if len(lines) > 2:
            continue
        if len(lines) == 2 and 2.0 * step * min_scale > usable_h + 1e-6:
            continue
        if all(
            _plan_number_line_fits(
                ln, usable_width=usable_w, text_height=base_text_height, height_scale=min_scale
            )
            for ln in lines
        ):
            return candidate, min_scale

    fallback = _plan_number_two_line_candidates(plain)
    return (fallback[0] if fallback else plain), min_scale


def _apply_plan_number_table_cell(
    autocad: Any,
    *,
    plan_h: str,
    plan_number: str,
    get_cell: Callable[..., str],
    set_cell: Callable[..., Any],
    ideal_text_height: Optional[float] = None,
    cell_text_style: str = "",
    sheet_scale_k: float = 1.0,
) -> Dict[str, Any]:
    """
    Fit CADA_PLANNUMBER after all sheet scaling/regen (final cell size and text height).

    Uses the computed ideal height for the output plan scale (never enlarges above it).
    Shrinks down to 0.85× ideal when needed for a single line.
    """
    debug: Dict[str, Any] = {}
    if not plan_h or not (plan_number or "").strip():
        return debug

    import time

    plan_plain = plan_number.strip().upper()
    plan_row, plan_col = 1, 0
    plan_template = get_cell(plan_h, plan_row, plan_col)

    try:
        autocad.recompute_table(plan_h)
    except Exception:
        pass

    measured_th = 0.0
    plan_cell_w = 0.0
    plan_cell_h = 0.0
    plan_step = 0.0
    try:
        step_res = autocad.get_table_cell_mtext_line_step(
            plan_h, plan_row, plan_col, plan_template
        )
        if step_res.get("success"):
            if step_res.get("text_height"):
                measured_th = float(step_res["text_height"])
            plan_step = float(step_res.get("line_step") or 0.0)
    except Exception:
        pass

    ideal_base = float(ideal_text_height) if ideal_text_height and ideal_text_height > 0 else 0.0
    if ideal_base <= 0 and measured_th > 0:
        ideal_base = measured_th
    if ideal_base <= 0:
        ideal_base = 0.6
    sk = float(sheet_scale_k) if sheet_scale_k else 1.0
    sheet_already_scaled = abs(sk - 1.0) > 1e-6
    # After sheet scaling, COM may still report pre-scale TABLE text height (e.g. 1.2 not 0.6).
    if measured_th > ideal_base + 1e-6:
        debug["measured_inflated"] = float(measured_th)
        if sheet_already_scaled and measured_th * sk <= ideal_base + 0.08:
            debug["measured_scaled_to"] = float(measured_th * sk)
    # Never derive nominal height from inflated COM readback when caller supplied ideal.
    plan_th = ideal_base

    style_name = str(cell_text_style or "").strip()
    if not style_name:
        try:
            for r in (plan_row, 0):
                st = autocad.get_table_cell_text_style(plan_h, r, plan_col)
                if st.get("success") and str(st.get("style") or "").strip():
                    style_name = str(st["style"]).strip()
                    break
        except Exception:
            pass

    try:
        ext_inner = autocad.get_table_cell_extents(
            plan_h, plan_row, plan_col, outer=False
        )
        ext_outer = autocad.get_table_cell_extents(
            plan_h, plan_row, plan_col, outer=True
        )
        widths: List[float] = []
        heights: List[float] = []
        for ext in (ext_inner, ext_outer):
            if ext.get("success"):
                w = float(ext["maxx"]) - float(ext["minx"])
                h = float(ext["maxy"]) - float(ext["miny"])
                if w > 0:
                    widths.append(w)
                if h > 0:
                    heights.append(h)
        if widths:
            plan_cell_w = min(widths)
        if heights:
            plan_cell_h = min(heights)
    except Exception:
        pass
    if plan_cell_w <= 0:
        plan_cell_w = max(10.0, plan_th * 14.0)
    if plan_cell_h <= 0:
        plan_cell_h = max(plan_th * 2.5, plan_step * 2.0 if plan_step > 0 else plan_th * 2.5)

    # After sheet scaling, cell geometry is final — always shrink-to-fit from ideal height.
    if sheet_already_scaled:
        plan_body, plan_scale = _fit_plan_number_mtext(
            plan_plain,
            cell_width=plan_cell_w,
            cell_height=plan_cell_h,
            base_text_height=plan_th,
            line_step=plan_step,
        )
        plan_scale = min(1.0, float(plan_scale))
        debug["sheet_scaled_nominal"] = True
    else:
        plan_body, plan_scale = _fit_plan_number_mtext(
            plan_plain,
            cell_width=plan_cell_w,
            cell_height=plan_cell_h,
            base_text_height=plan_th,
            line_step=plan_step,
        )
        plan_scale = min(1.0, float(plan_scale))

    def _plan_mtext(body: str, *, inline_height_scale: float = 1.0) -> str:
        tpl = _mtext_strip_inline_size_codes(plan_template)
        raw = _mtext_preserve_style_set_content(
            tpl,
            body,
            height_scale=inline_height_scale,
        )
        if inline_height_scale >= 0.999:
            return _mtext_strip_inline_size_codes(raw)
        return raw

    def _apply_style_and_height(target_h: float) -> None:
        if style_name:
            try:
                autocad.set_table_cell_text_style(
                    plan_h, plan_row, plan_col, style_name
                )
            except Exception:
                pass
        try:
            autocad.set_table_cell_text_height(plan_h, plan_row, plan_col, target_h)
        except Exception:
            pass

    def _write_plan_cell(body: str, scale: float, *, inline_scale: float = 1.0) -> float:
        target_h = plan_th * scale
        # SetText resets style/height — always apply text first, then style, then height.
        set_cell(plan_h, plan_row, plan_col, _plan_mtext(body, inline_height_scale=inline_scale))
        _apply_style_and_height(target_h)
        try:
            autocad.recompute_table(plan_h)
        except Exception:
            pass
        readback_h = 0.0
        try:
            hr = autocad.get_table_cell_text_height(plan_h, plan_row, plan_col)
            if hr.get("success"):
                readback_h = float(hr.get("height") or 0.0)
        except Exception:
            pass
        return readback_h

    def _enforce_plan_cell_height(body: str, scale: float, target_h: float) -> float:
        """Retry style+height until TABLE cell readback matches the cartographic target."""
        style_candidates: List[str] = []
        for candidate in (style_name, "BOLD_SURVEY", "Standard"):
            c = str(candidate or "").strip()
            if c and c not in style_candidates:
                style_candidates.append(c)
        readback_h = 0.0
        for attempt in range(6):
            readback_h = _write_plan_cell(body, scale)
            if abs(readback_h - target_h) <= 0.03:
                return readback_h
            for st in style_candidates:
                try:
                    autocad.set_table_cell_text_style(plan_h, plan_row, plan_col, st)
                except Exception:
                    pass
            try:
                autocad.set_table_cell_text_height(plan_h, plan_row, plan_col, target_h)
                autocad.recompute_table(plan_h)
            except Exception:
                pass
            try:
                hr = autocad.get_table_cell_text_height(plan_h, plan_row, plan_col)
                if hr.get("success"):
                    readback_h = float(hr.get("height") or 0.0)
                    if abs(readback_h - target_h) <= 0.03:
                        return readback_h
            except Exception:
                pass
            if readback_h > target_h + 0.03 and readback_h > 1e-6:
                inline = max(0.5, target_h / readback_h)
                readback_h = _write_plan_cell(body, scale, inline_scale=inline)
                if abs(readback_h - target_h) <= 0.05:
                    return readback_h
            time.sleep(0.08 * (attempt + 1))
        return readback_h

    readback_h = _write_plan_cell(plan_body, plan_scale)
    target_h = float(plan_th * plan_scale)
    # Inline \\H shrink only when the sheet was NOT scaled but COM still reports template height.
    if measured_th > plan_th + 1e-6 and not sheet_already_scaled:
        pre_inline = max(_PLAN_NUMBER_MIN_HEIGHT_SCALE, plan_th / max(measured_th, 1e-6))
        readback_h = _write_plan_cell(plan_body, plan_scale, inline_scale=pre_inline)
        debug["pre_inline_height_scale"] = float(pre_inline)
        debug["readback_height_pre_inline"] = float(readback_h)
        target_h = float(plan_th * plan_scale)
    readback_h = _enforce_plan_cell_height(plan_body, plan_scale, target_h)
    debug.update(
        {
            "ideal_height": float(plan_th),
            "shrink_scale": float(plan_scale),
            "target_height": float(target_h),
            "readback_height": float(readback_h),
            "text_style": style_name,
        }
    )

    # SetText often resets TABLE cells to Standard (~0.9); force height if still too large.
    if readback_h > target_h + 0.03:
        ratio = max(0.5, target_h / max(readback_h, 1e-6))
        readback_h = _enforce_plan_cell_height(plan_body, plan_scale, target_h)
        debug["inline_height_scale"] = float(ratio)
        debug["readback_height_after_inline"] = float(readback_h)

    # If AutoCAD still wrapped a single-line fit, step down until one line or 0.85× floor.
    if (
        not sheet_already_scaled
        and "\\P" not in plan_body
        and plan_scale > _PLAN_NUMBER_MIN_HEIGHT_SCALE + 1e-6
    ):
        try:
            readback = get_cell(plan_h, plan_row, plan_col) or ""
            if "\\P" in readback:
                retry_scale = plan_scale
                while retry_scale > _PLAN_NUMBER_MIN_HEIGHT_SCALE + 1e-6:
                    retry_scale = max(
                        _PLAN_NUMBER_MIN_HEIGHT_SCALE,
                        round(retry_scale - 0.02, 4),
                    )
                    if _plan_number_line_fits(
                        plan_plain,
                        usable_width=max(
                            0.01,
                            plan_cell_w
                            - 2.0 * max(0.1, plan_th * 0.28),
                        ),
                        text_height=plan_th,
                        height_scale=retry_scale,
                    ):
                        readback_h = _write_plan_cell(plan_plain, retry_scale)
                        debug["retry_shrink_scale"] = float(retry_scale)
                        debug["readback_height_retry"] = float(readback_h)
                        break
        except Exception:
            pass

    return debug


def _layout_surveyor_address_mtext(
    raw_address: str,
    *,
    cell_width: float,
    cell_height: float,
    text_height: float,
    line_step: float,
) -> str:
    """
    Pack surveyor company/address inside the table cell without vertical overflow.
    Uses horizontal joins (city + state on one line) before adding extra lines.
    """
    parts = _merge_city_state_tail(_split_address_segments(raw_address))
    if not parts:
        return ""

    pad_x = max(0.15, float(text_height) * 0.35)
    pad_y = max(0.10, float(text_height) * 0.25)
    usable_w = max(float(cell_width) - 2.0 * pad_x, float(text_height) * 4.0)
    usable_h = max(float(cell_height) - 2.0 * pad_y, float(line_step) or float(text_height))
    step = float(line_step) if line_step > 0 else float(text_height) * (5.0 / 3.0)
    max_lines = max(1, int(usable_h / step))

    height_scale = 1.0
    for scale in (1.0, 0.97, 0.94, 0.92, 0.9, 0.88, 0.85):
        chars_per_line = int(usable_w / (max(0.01, text_height * scale) * _CARTO_CHAR_WIDTH_FACTOR))
        lines = _pack_segments_into_lines(
            parts, max_chars_per_line=chars_per_line, max_lines=max_lines
        )
        if len(lines) <= max_lines and len(lines) * step * scale <= usable_h + 1e-6:
            height_scale = scale
            packed = "\\P".join(lines)
            if scale < 0.999:
                packed = _mtext_apply_height_scale(packed, scale)
            return packed
        height_scale = scale

    chars_per_line = int(usable_w / (max(0.01, text_height * 0.85) * _CARTO_CHAR_WIDTH_FACTOR))
    lines = _pack_segments_into_lines(
        parts, max_chars_per_line=max(6, chars_per_line), max_lines=max_lines
    )
    while len(lines) > max_lines and len(lines) >= 2:
        tail = lines.pop()
        lines[-1] = f"{lines[-1]}, {tail}"
    packed = "\\P".join(lines[:max_lines])
    return _mtext_apply_height_scale(packed, 0.85)


def _find_title_scale_label_row(
    get_cell,
    title_handle: str,
    tables_meta: Optional[Dict[str, Any]] = None,
    default_row: int = 8,
) -> int:
    """Row index of the main \"SCALE:- 1:xxx\" cell in the title-block table."""
    if not title_handle:
        return default_row
    scale_pattern = re.compile(r"1\s*:\s*\d+", re.IGNORECASE)
    secondary_re = re.compile(r"\bSCALE\b\s*:.*\bto\b", re.IGNORECASE)
    main_hint_re = re.compile(r"\bSCALE\b\s*[:-]", re.IGNORECASE)
    tbl = (tables_meta or {}).get(title_handle, {}) if isinstance(tables_meta, dict) else {}
    rows = int(tbl.get("rows", 25))
    cols = int(tbl.get("cols", 2))
    main_scale_cell = None
    for r in range(min(rows, 60)):
        for c in range(min(cols, 10)):
            cell = get_cell(title_handle, r, c) or ""
            if not cell.strip():
                continue
            if secondary_re.search(cell) and scale_pattern.search(cell):
                continue
            if main_scale_cell is None and main_hint_re.search(cell) and scale_pattern.search(cell):
                main_scale_cell = (r, c)
    if main_scale_cell is None:
        return default_row
    return int(main_scale_cell[0])


# ==============================================================================
# MAIN AGENT CLASS
# ==============================================================================

class SurvyAIAgent:
    """
    The main AI agent for SurvyAI, built with LangGraph.
    
    This class orchestrates:
    1. LLM initialization (Gemini, DeepSeek, Claude, or OpenAI)
    2. Tool creation (AutoCAD, Excel, Document processors)
    3. Graph construction (nodes, edges, routing logic)
    4. Query processing (invoking the graph with user input)
    
    Architecture:
    ------------
    The agent uses a ReAct (Reasoning + Acting) pattern where:
    1. The LLM receives a query and decides what to do
    2. If tools are needed, they are executed
    3. Tool results are fed back to the LLM
    4. The LLM formulates a final response
    
    Usage:
    ------
    ```python
    agent = SurvyAIAgent()
    result = agent.process_query("Calculate area from survey.dwg")
    print(result["response"])
    ```
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        feature_flags: Optional[FeatureFlags] = None,
    ):
        """
        Initialize the SurvyAI agent.
        
        Args:
            settings: Optional explicit `Settings` instance (e.g. from `merge_settings()`
                in desktop builds). If omitted, uses `get_settings()` (.env / environment).
            feature_flags: Desktop/service integration flags (`SurvyAIAgentService` passes this).
                If omitted, loads from environment via `FeatureFlags.from_env()`.
        
        Initialization sequence:
        1. Load configuration settings
        2. Create application processors (AutoCAD, Excel, etc.)
        3. Initialize LLMs (primary and fallback)
        4. Create tool definitions
        5. Build the LangGraph
        
        Raises:
            Exception: If LLM initialization fails
        """
        # ------------------------------------------------------------------
        # Step 1: Load configuration
        # ------------------------------------------------------------------
        # Settings come from environment variables and .env file, or are injected
        # (e.g. desktop app with merged cloud tokens) via `settings=`.
        self.settings = settings if settings is not None else get_settings()
        runtime_cfg = resolve_agent_runtime_config(
            local_config_path=str(getattr(self.settings, "agent_config_path", "") or ""),
            cloud_config_json=str(getattr(self.settings, "agent_cloud_config_json", "") or ""),
        )
        runtime_overrides = runtime_cfg.to_settings_overrides()
        if runtime_overrides:
            self.settings = self.settings.model_copy(update=runtime_overrides)
        self._system_prompt = str(runtime_cfg.system_prompt or SYSTEM_PROMPT)
        self._agent_runtime_config = runtime_cfg
        self.feature_flags = (
            feature_flags if feature_flags is not None else FeatureFlags.from_env()
        )
        logger.info(
            "Agent runtime config loaded (source=%s, version=%s)",
            runtime_cfg.source,
            runtime_cfg.version,
        )

        # Validate that primary LLM is set correctly
        logger.info(f"Configuration loaded - Primary LLM: {self.settings.primary_llm}, Fallback LLM: {self.settings.fallback_llm}")
        
        # Track which Gemini model we're using (for logging/debugging)
        self._current_gemini_model: Optional[str] = getattr(
            self.settings, "gemini_model", None
        )
        
        # Track which OpenAI model we're currently using (for tiered model selection)
        self._current_openai_model: Optional[str] = None

        # Lightweight caches to avoid expensive re-initialization/re-compilation
        # (No functional impact; improves latency and reduces mid-flight churn.)
        # Key: (provider, transport, model, temp, max_tokens, auth_fingerprint)
        self._llm_client_cache: Dict[tuple, BaseChatModel] = {}
        # Back-compat alias used by older call sites / tests.
        self._openai_llm_cache = self._llm_client_cache
        self._gemini_models_cache: Optional[List[str]] = None
        self._prompt_router_cache: Dict[tuple, Any] = {}
        self._app_signature: Optional[tuple] = None
        self._pipeline_llm_cost_usd: float = 0.0
        # Serialize LLM invokes so a timed-out Ollama/HTTP call cannot stack
        # concurrent generations and thrash the host into lock/hibernate.
        self._llm_invoke_gate = threading.Lock()
        self._cloud_proxy_enabled = bool(
            getattr(self.settings, "survyai_llm_proxy_enabled", False)
            and str(getattr(self.settings, "survyai_api_base_url", "") or "").strip()
            and str(getattr(self.settings, "survyai_access_token", "") or "").strip()
        )
        
        # Provider keys are validated lazily inside _initialize_llm().  Do not
        # fail construction here: packaged desktop installs do not ship provider
        # API keys, and should still start on Ollama or the SurvyAI cloud proxy.
        #
        # We keep these log messages as diagnostics only.  The actual startup
        # selection below tries the configured provider, then fallback, then
        # Ollama, and records the provider that really became active.
        if self.settings.primary_llm == "openai":
            if not self._cloud_proxy_enabled and (
                not self.settings.openai_api_key or not self.settings.openai_api_key.strip()
            ):
                logger.warning(
                    "Primary LLM is openai but OPENAI_API_KEY is not configured; "
                    "startup will try fallback/Ollama instead."
                )
            model_name = getattr(self.settings, "openai_model", "gpt-4o-mini")
            logger.info(f"OpenAI configured as primary LLM (model: {model_name})")
        
        # If primary is Claude, ensure API key is configured
        elif self.settings.primary_llm == "claude":
            if not self._cloud_proxy_enabled and (
                not self.settings.anthropic_api_key or not self.settings.anthropic_api_key.strip()
            ):
                logger.warning(
                    "Primary LLM is claude but ANTHROPIC_API_KEY is not configured; "
                    "startup will try fallback/Ollama instead."
                )
            logger.info("Claude configured as primary LLM")
        
        # ------------------------------------------------------------------
        # Step 2: Initialize application processors
        # ------------------------------------------------------------------
        # These are the "backends" that actually do the work.
        # They connect to AutoCAD, read Excel files, etc.
        
        # AutoCAD processor - connects via COM API
        # auto_connect=False means we connect on first use, not at startup
        self.autocad = AutoCADProcessor(auto_connect=False)
        
        # DXF fallback processor - works without AutoCAD installed
        # Uses ezdxf library to read DXF files directly
        self.dxf_fallback = DXFProcessor()
        if self.dxf_fallback.is_available:
            logger.info("✓ DXF fallback processor available (ezdxf)")
        
        # Excel processor - reads .xlsx and .xls files
        self.excel_processor = ExcelProcessor()
        
        # Document processor - extracts text from PDF and Word
        self.document_processor = DocumentProcessor()
        
        # Coordinate conversions:
        # IMPORTANT: pyproj is the default main method; Geographic Calculator COM is optional.
        # We lazy-connect COM only when explicitly requested to avoid noisy/irrelevant startup scans.
        self.blue_marble = BlueMarbleConverter(auto_connect=False)
        
        # Geographic Calculator CLI - lazy scan only when tool is used
        self.geocalc_cli = GeographicCalculatorCLI(auto_detect=False)
        
        # ArcGIS processor - advanced geospatial analysis
        self.arcgis_processor = ArcGISProcessor()

        # GeoPandas executor - dynamic GIS analysis without ArcGIS licence
        # Handles: spatial join, point-in-polygon, buffer, clip, export to Excel/CSV/shapefile
        self.geopandas_executor = GeoPandasExecutor(
            timeout_seconds=getattr(self.settings, "geopandas_execution_timeout", 300)
        )
        
        # Vector store – PostgreSQL/pgvector/PostGIS backend
        # Reads VECTOR_DB_URL (or DATABASE_URL) from settings/environment.
        self.vector_store = None
        self._vs_search_mode: str = getattr(self.settings, "vector_search_mode", "hybrid")
        self._vs_hybrid_alpha: float = float(getattr(self.settings, "vector_hybrid_alpha", 0.6))
        if getattr(self.settings, "vector_store_enabled", True):
            try:
                self.vector_store = VectorStore(
                    db_url=(
                        getattr(self.settings, "vector_db_url", None)
                        or os.environ.get("VECTOR_DB_URL", "")
                        or os.environ.get("DATABASE_URL", "")
                    ) or None,
                    embedding_provider=getattr(self.settings, "embedding_provider", "local"),
                    openai_api_key=getattr(self.settings, "openai_api_key", None),
                    local_model_name=getattr(self.settings, "local_embedding_model", "all-MiniLM-L6-v2"),
                    openai_model_name=getattr(self.settings, "openai_embedding_model", "text-embedding-3-small"),
                )
                logger.info("✓ Vector store (PostgreSQL/pgvector) initialized successfully")
            except Exception as e:
                logger.warning(f"⚠ Vector store initialization failed: {e}")
                logger.warning("Semantic search will be unavailable until PostgreSQL is configured.")
                self.vector_store = None
        
        # ------------------------------------------------------------------
        # Step 3: Initialize LLMs
        # ------------------------------------------------------------------
        # We have a primary LLM (default: OpenAI) and a fallback (default: Gemini)
        # If the primary fails, we automatically try the fallback
        
        requested_primary = str(self.settings.primary_llm or "ollama").strip().lower()
        # Symbolic "auto" (best paid hosted model) must never reach provider init.
        if requested_primary == "auto":
            try:
                from survyai.llm_routing import resolve_primary_llm_selection

                requested_primary = resolve_primary_llm_selection("auto")
            except Exception:
                requested_primary = "openai"
        requested_fallback = str(self.settings.fallback_llm or "ollama").strip().lower()
        logger.info(f"Initializing primary LLM: {requested_primary}")
        logger.info(f"Initializing fallback LLM: {requested_fallback}")

        self.llm_primary: Optional[BaseChatModel] = None
        self.llm_fallback: Optional[BaseChatModel] = None

        def _startup_candidates(preferred: str) -> List[str]:
            """Provider order for installed desktop startup.

            A fresh installed app has no provider .env.  It should still start
            with Ollama (or cloud proxy after sign-in) instead of crashing.
            """
            out: List[str] = []
            for item in (preferred, requested_fallback, "ollama"):
                item = str(item or "").strip().lower()
                if item and item not in out:
                    out.append(item)
            return out

        selected_primary = ""
        last_primary_error: Optional[Exception] = None
        for candidate in _startup_candidates(requested_primary):
            try:
                self.llm_primary = self._initialize_llm(candidate)
                selected_primary = candidate
                logger.info(f"✓ Primary LLM active: {candidate}")
                break
            except Exception as e:
                last_primary_error = e
                logger.warning(f"⚠ Could not initialize startup LLM '{candidate}': {e}")

        if self.llm_primary is None or not selected_primary:
            raise ValueError(
                "Could not initialize any LLM backend. Install/start Ollama, sign in "
                "for the SurvyAI cloud proxy, or configure a provider API key. "
                f"Last error: {last_primary_error}"
            )

        if selected_primary and str(self.settings.primary_llm or "").strip().lower() != selected_primary:
            self.settings = self.settings.model_copy(update={"primary_llm": selected_primary})

        selected_fallback = ""
        for candidate in _startup_candidates(requested_fallback):
            if candidate == selected_primary:
                self.llm_fallback = self.llm_primary
                selected_fallback = candidate
                break
            try:
                self.llm_fallback = self._initialize_llm(candidate)
                selected_fallback = candidate
                logger.info(f"✓ Fallback LLM active: {candidate}")
                break
            except Exception as e:
                logger.warning(f"⚠ Could not initialize fallback LLM '{candidate}': {e}")

        if self.llm_fallback is None:
            self.llm_fallback = self.llm_primary
            selected_fallback = selected_primary

        if selected_fallback and selected_fallback != requested_fallback:
            self.settings = self.settings.model_copy(update={"fallback_llm": selected_fallback})
        
        # ------------------------------------------------------------------
        # Step 4: Create tools
        # ------------------------------------------------------------------
        # Tools are functions the LLM can call to perform actions
        self.tools = self._create_tools()
        
        # Bind tools to the LLM so it knows what's available
        # This creates a new LLM instance that includes tool definitions
        logger.info(f"Binding tools to primary LLM: {self.settings.primary_llm}")
        self.llm_with_tools = self.llm_primary.bind_tools(self.tools)
        logger.info(f"✓ Tools bound to {self.settings.primary_llm}")
        
        # ------------------------------------------------------------------
        # Step 5: Build the LangGraph
        # ------------------------------------------------------------------
        # The graph defines how the agent processes queries
        self.graph = self._build_graph()
        
        # Compile the graph with a memory checkpointer
        # This enables conversation history and state persistence
        self.memory = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory)

        # Pre-warm paid-provider tier models so the first query avoids cold init.
        try:
            from survyai.provider_models import PAID_PROVIDERS as _PAID_WARM

            _warm_prov = str(getattr(self.settings, "primary_llm", "") or "").strip().lower()
            if _warm_prov in _PAID_WARM and getattr(self.settings, "enable_tiered_models", True):
                for _tier in ("simple", "average", "complex"):
                    _tier_model = self._get_model_for_complexity(_warm_prov, _tier)
                    try:
                        self._initialize_llm(_warm_prov, model_name=_tier_model)
                        logger.debug(
                            "Pre-warmed %s LLM cache: %s (%s)",
                            _warm_prov,
                            _tier_model,
                            _tier,
                        )
                    except Exception as _e:
                        logger.debug(
                            "Could not pre-warm %s tier '%s' model '%s': %s",
                            _warm_prov,
                            _tier,
                            _tier_model,
                            _e,
                        )
        except Exception:
            pass

        # Bind the average (typical) tier at startup so common GIS/CAD queries
        # reuse the compiled graph; complex tier binds lazily on first hard query.
        try:
            from survyai.provider_models import PAID_PROVIDERS as _PAID_BIND

            _bind_prov = str(getattr(self.settings, "primary_llm", "") or "").strip().lower()
            if _bind_prov in _PAID_BIND and getattr(self.settings, "enable_tiered_models", True):
                try:
                    _avg_model = self._get_model_for_complexity(_bind_prov, "average")
                    _avg_llm = self._initialize_llm(_bind_prov, model_name=_avg_model)
                    self.llm_with_tools = _avg_llm.bind_tools(self.tools)
                    self.graph = self._build_graph()
                    self.app = self.graph.compile(checkpointer=self.memory)
                    model_sig = _avg_model
                    logger.debug(
                        "App bound to average-tier %s model at startup: %s",
                        _bind_prov,
                        _avg_model,
                    )
                except Exception as _e:
                    logger.debug("Could not bind average-tier model at startup: %s", _e)
                    model_sig = (
                        getattr(self.llm_primary, "model", None)
                        or getattr(self.settings, "openai_model", None)
                        or self.settings.primary_llm
                    )
            else:
                try:
                    model_sig = (
                        getattr(self.llm_primary, "model", None)
                        or getattr(self.settings, "openai_model", None)
                        or self.settings.primary_llm
                    )
                except Exception:
                    model_sig = self.settings.primary_llm
        except Exception:
            try:
                model_sig = getattr(self.llm_primary, "model", None) or self.settings.primary_llm
            except Exception:
                model_sig = self.settings.primary_llm

        tool_sig = tuple(sorted([t.name for t in self.tools]))
        self._app_signature = (model_sig, tool_sig)
        
        logger.info("SurvyAI agent initialized successfully with LangGraph")
        
        # Session tracking for conversation continuity
        self._current_session_id: Optional[str] = None

        # Last cadastral plan output (for in-session modifications without re-prompting)
        # Template file remains read-only; modifications apply only to this output file.
        self._last_cadastral_output_dwg: Optional[str] = None
        self._last_cadastral_profile_path: Optional[str] = None

        # STRICT: Survey plan template paths must never be written (read-only to avoid corruption).
        # Populated from template_profiles/*.json and when learning a template.
        self._protected_template_paths: set = set()

        # Persistent cadastral CAD template memory (multi-template registry).
        # This lets users omit the template path after successful prior runs.
        self._cad_template_memory_file: Optional[str] = None

        # Internet permission (interactive, user-controlled)
        # Default: False (must ask user before searching the internet)
        self._internet_permission_granted: bool = False
        self._pending_permission_requests: Dict[str, Dict[str, Any]] = {}
        # Set per-query when the user affirmatively answered an internet-permission
        # request (deterministic or conversational); forces the search to actually run.
        self._force_internet_search_this_query: bool = False
    
    # ==========================================================================
    # CONTEXT RETRIEVAL AND STORAGE
    # ==========================================================================
    
    def _extract_document_paths(self, query: str) -> List[str]:
        """
        Extract document file paths from a query string.
        
        Looks for common document file patterns:
        - .docx, .doc (Word documents)
        - .pdf (PDF documents)
        - Paths in quotes or as-is
        
        Args:
            query: The user's query string
            
        Returns:
            List of detected document file paths
        """
        import re
        from pathlib import Path

        from survyai.attachments import parse_attachments_block
        
        document_paths = []
        seen = set()

        # Attachment marker block from the chat composer (may include images + docs).
        marker_paths, _ = parse_attachments_block(query or "")
        for path_str in marker_paths:
            path = Path(path_str)
            if path.exists() and path.is_file() and path.suffix.lower() in [".docx", ".doc", ".pdf"]:
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    document_paths.append(key)
        
        # Pattern to match file paths with document extensions.
        #
        # IMPORTANT: Do NOT rely on quoted-string extraction, because real Windows paths
        # may contain apostrophes (e.g. "MICHAEL's") which breaks naive single-quote parsing.
        #
        # Instead, match Windows/Unix-like paths up to a known extension.
        patterns = [
            # Windows absolute paths: allow spaces and apostrophes, but stop before illegal filename chars / quotes
            r'([A-Za-z]:\\[^\r\n"<>|]+?\.(?:docx?|pdf))',
            # Unix/relative paths (also allow backslashes for relative Windows-ish paths)
            r'((?:/|\\)[^\r\n"<>|]+?\.(?:docx?|pdf))',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                path_str = (match or "").strip()
                # Trim common trailing punctuation / wrappers
                path_str = path_str.strip().strip('"').strip("'").rstrip(").,;")
                path = Path(path_str)
                # Verify the file exists and is a document
                if path.exists() and path.is_file():
                    if path.suffix.lower() in ['.docx', '.doc', '.pdf']:
                        key = str(path.resolve())
                        if key not in seen:
                            seen.add(key)
                            document_paths.append(key)
        
        return document_paths

    def _extract_image_paths(self, query: str) -> List[str]:
        """
        Extract image file paths from a query (typed paths + attachment markers).

        Kept separate from ``_extract_document_paths`` so PDF/Word preflight
        never tries to pdfplumber a PNG.
        """
        from agent.vision_ocr import extract_image_paths_from_query

        return extract_image_paths_from_query(query or "")

    def _infer_output_path_from_input(
        self, 
        input_path: str, 
        output_filename: Optional[str] = None,
        output_type: str = "file"
    ) -> Optional[str]:
        """
        Infer output path from input file path when user doesn't specify location.
        
        CRITICAL RULE: If user doesn't explicitly specify where to create/locate a file or operation,
        default to the SAME FOLDER as the input file/folder/document.
        
        Args:
            input_path: Path to input file/folder
            output_filename: Optional output filename (if None, returns just the parent directory)
            output_type: "file" (returns file path) or "folder" (returns directory path)
            
        Returns:
            Inferred output path, or None if input_path is invalid
        """
        from pathlib import Path
        
        try:
            input_p = Path(input_path)
            # If input_path is a file, use its parent; if it's a directory, use it directly
            if input_p.is_file():
                parent_dir = input_p.parent
            elif input_p.is_dir():
                parent_dir = input_p
            else:
                # Path doesn't exist yet, but we can still extract parent from the path string
                parent_dir = input_p.parent if input_p.suffix else input_p
            
            if output_type == "folder":
                return str(parent_dir.resolve())
            elif output_filename:
                # If output_filename is already absolute, return as-is
                output_p = Path(output_filename)
                if output_p.is_absolute():
                    return str(output_p.resolve())
                # Otherwise, resolve relative to input's parent folder
                return str((parent_dir / output_filename).resolve())
            else:
                return str(parent_dir.resolve())
        except Exception as e:
            logger.debug(f"Failed to infer output path from {input_path}: {e}")
            return None

    def _extract_explicit_output_folder(self, query: str) -> Optional[Path]:
        """Return a user-named output directory from the prompt, if present."""
        q = query or ""
        patterns = (
            r"(?:in|into|to)\s+(?:the\s+)?(?:folder|directory)\s+['\"]([^'\"]+)['\"]",
            r"(?:save|saved|export|create|write|store|generate)\s+(?:\w+\s+){0,10}(?:in|into|to)\s+(?:the\s+)?(?:folder|directory)\s+['\"]([^'\"]+)['\"]",
            r"(?:save|saved|export|create|write)\s+(?:to|in|into)\s+(?:the\s+)?(?:folder|directory)\s+['\"]([^'\"]+)['\"]",
        )
        for pat in patterns:
            m = re.search(pat, q, flags=re.IGNORECASE)
            if not m:
                continue
            raw = (m.group(1) or "").strip().strip("\"'").rstrip(").,;")
            if not raw:
                continue
            folder = Path(raw)
            if folder.suffix.lower() in {
                ".dwg", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".pdf", ".aprx", ".gdb", ".shp",
            }:
                folder = folder.parent
            try:
                return folder.resolve()
            except Exception:
                return folder
        return None

    def _resolve_user_output_path(
        self,
        query: str,
        output_ref: str,
        *,
        fallback_dir: Optional[Path] = None,
    ) -> Path:
        """
        Resolve an output file path from the user's prompt.

        Priority: absolute path in output_ref > explicit folder in query + filename >
        active workspace (fallback_dir or Path.cwd()).
        """
        ws = (fallback_dir or Path.cwd()).resolve()
        ref = (output_ref or "").strip().strip("\"'").rstrip(").,;")
        if not ref:
            return ws

        p = Path(ref)
        if p.is_absolute():
            return self._coerce_write_path_away_from_install_root(p, ws)

        if len(p.parts) > 1 or ("/" in ref) or ("\\" in ref):
            return (ws / p).resolve()

        explicit_folder = self._extract_explicit_output_folder(query or "")
        if explicit_folder is not None:
            try:
                explicit_folder.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            return (explicit_folder / p.name).resolve()

        return (ws / p.name).resolve()

    @staticmethod
    def _path_looks_like_survyai_install_root(folder: Path) -> bool:
        """True when *folder* looks like the SurvyAI source/install tree (not a user workspace)."""
        try:
            root = folder.resolve()
        except Exception:
            root = folder
        return (root / "agent").is_dir() and (root / "survyai").is_dir()

    def _coerce_write_path_away_from_install_root(self, path: Path, workspace: Path) -> Path:
        """
        If a model invents an absolute path under the SurvyAI install/repo root while a
        different active workspace is set, redirect the basename into the workspace.
        Explicit user absolute paths outside the install root are preserved.
        """
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        try:
            ws = workspace.resolve()
        except Exception:
            ws = workspace
        if resolved.parent == ws:
            return resolved
        if self._path_looks_like_survyai_install_root(resolved.parent) and ws != resolved.parent:
            return (ws / resolved.name).resolve()
        return resolved

    def _resolve_output_path_for_write(self, file_path: str) -> Path:
        """Resolve tool save paths: bare/relative names → active workspace (cwd)."""
        ref = (file_path or "").strip().strip("\"'")
        if not ref:
            raise ValueError("file_path is required")
        return self._resolve_user_output_path("", ref, fallback_dir=Path.cwd())
    
    def _extract_requested_output_docx(self, query: str, input_doc_path: str) -> Optional[str]:
        """
        Try to infer an output .docx filename/path from a query.
        
        Preference order:
        0) Explicit "Save ... as 'X.docx'" or "save the Summary file as 'X.docx'" (user intent)
        1) A quoted .docx path that does NOT exist (assumed intended output path)
        2) A quoted .docx filename that is NOT a substring of the input filename,
           resolved into the same folder as the input doc
        3) None (unknown)
        """
        import re
        from pathlib import Path

        input_path = Path(input_doc_path)
        q = query
        input_name_lower = input_path.name.lower()

        # 0) PRIORITY: Explicit "Save ... as 'X.docx'" - respect user intent
        explicit_patterns = [
            r"(?:save|saved|export)\s+(?:the\s+)?(?:summary\s+)?(?:file\s+)?as\s+['\"]([^'\"]+\.docx)['\"]",
            r"(?:save|saved|export)\s+(?:the\s+)?(?:summary\s+)?(?:file\s+)?as\s+([^\s,\.]+\.docx)",
            r"as\s+['\"]([^'\"]+\.docx)['\"]\s+in\s+(?:the\s+)?same\s+folder",
        ]
        for pat in explicit_patterns:
            m = re.search(pat, q, flags=re.IGNORECASE)
            if m:
                out = (m.group(1) or "").strip().strip("'").strip('"').rstrip(").,;")
                if out and out.lower().endswith(".docx"):
                    p = Path(out)
                    if p.is_absolute() and not p.exists():
                        return str(p)
                    if not p.is_absolute():
                        return str((input_path.parent / p.name).resolve())
                    return str(p)

        # Find any .docx-like candidates in the query.
        candidates: list[str] = []
        candidates.extend(re.findall(r'([A-Za-z]:\\[^\r\n"<>|]+?\.docx)', q, flags=re.IGNORECASE))
        candidates.extend(re.findall(r'([^\s"<>|]+?\.docx)', q, flags=re.IGNORECASE))

        # Normalize: keep order, strip whitespace, trim wrappers
        seen = set()
        normed: list[str] = []
        for c in candidates:
            if not c:
                continue
            s = c.strip().strip('"').strip("'").rstrip(").,;")
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            normed.append(s)
        candidates = normed
        if not candidates:
            return None

        # Filter: remove input path and any candidate that is a trailing substring of input filename
        # (e.g. "PROJECTS.docx" from "REPORT ON FIVE PROJECTS.docx" must be excluded)
        input_abs = str(input_path.resolve()).lower()

        def _is_input_or_substring(c: str) -> bool:
            if str(Path(c).resolve()).lower() == input_abs:
                return True
            c_name = Path(c).name.lower()
            if c_name == input_name_lower:
                return True
            if input_name_lower.endswith(c_name):
                return True
            return False

        candidates = [c for c in candidates if not _is_input_or_substring(c)]
        if not candidates:
            return None

        # 1) Full path candidates that don't exist -> likely output
        for c in candidates:
            p = Path(c)
            if p.is_absolute() and (p.suffix.lower() == ".docx") and (not p.exists()):
                return str(p)

        # 2) Relative candidates -> resolve relative to input folder
        for c in candidates:
            p = Path(c)
            if not p.is_absolute() and p.suffix.lower() == ".docx":
                return str((input_path.parent / p.name).resolve())

        return None

    def _extract_any_output_docx(self, query: str) -> Optional[str]:
        """
        Extract an intended output .docx path/filename from a query (no input doc context).

        - Supports quoted/unquoted strings
        - Supports Windows absolute paths and filename-only outputs
        """
        import re
        from pathlib import Path

        q = self._cadastral_user_message_body(query) or (query or "")
        candidates: list[str] = []
        candidates.extend(re.findall(r'([A-Za-z]:\\[^\r\n"<>|]+?\.docx)', q, flags=re.IGNORECASE))
        candidates.extend(re.findall(r'([^\s"<>|]+?\.docx)', q, flags=re.IGNORECASE))

        # Normalize
        normed: list[str] = []
        seen = set()
        for c in candidates:
            if not c:
                continue
            s = c.strip().strip('"').strip("'").rstrip(").,;")
            if not s:
                continue
            k = s.lower()
            if k in seen:
                continue
            seen.add(k)
            normed.append(s)

        if not normed:
            return None

        # Prefer absolute; else first filename
        for s in normed:
            p = Path(s)
            if p.is_absolute():
                return str(p)
        return normed[0]

    def _should_fastpath_large_doc_summary(self, query: str, doc_info: Dict[str, Any]) -> bool:
        """
        Decide if we should bypass LangGraph and run a deterministic pipeline.
        
        We do this for large documents + explicit summarize/save requests to avoid
        multi-iteration tool loops, TPM overflow, and timeouts. Uses multiple signals:
        - Metadata-based: pages, words, estimated_tokens (from get_resource_estimation)
        - File-size fallback: when metadata is missing/unreliable, use file_size_mb
        """
        q = (query or "").lower()
        if not doc_info:
            return False

        # Accept full doc_info (from preflight) or estimation sub-dict
        est = doc_info.get("estimation") or doc_info
        pages = int(doc_info.get("page_count") or est.get("page_count") or 0)
        words = int(doc_info.get("word_count") or est.get("word_count") or 0)
        tokens = int(doc_info.get("estimated_tokens") or est.get("estimated_tokens") or 0)
        file_size_mb = float(doc_info.get("file_size_mb") or est.get("file_size_mb") or 0)

        # Relaxed thresholds: trigger earlier to avoid TPM overflow (500K limit)
        is_large = (
            pages > 50 or words > 25000 or tokens > 50000 or
            # Fallback: large file + summarize request (metadata may be missing for table-heavy docs)
            (file_size_mb > 3 and (pages > 0 or words > 0 or tokens > 0))
        )
        # File-size-only fallback when metadata estimation failed or returned zeros
        if not is_large and file_size_mb > 5:
            is_large = True

        wants_summary = (
            ("summar" in q or "summary" in q) and
            (".docx" in q or "save" in q or "same folder" in q or "document" in q or
             "projects" in q or "key survey" in q or "key details" in q or "professionally" in q)
        )
        return bool(is_large and wants_summary)

    def _should_fastpath_docx_report(self, query: str) -> bool:
        """
        Fast-path: user asks for a generated narrative report and to save it to a .docx,
        without providing an input document to summarize.

        Must NEVER hijack operational GIS/CAD/Excel workflows that merely contain the
        word "save" (e.g. save converted coordinates as an .xlsx) or re-execute
        instructions that mention conversation "history".
        """
        from survyai.gis_session_intent import looks_like_full_gis_workflow_request

        q = (query or "").strip()
        ql = q.lower()
        if not ql:
            return False

        # Operational GIS/CAD/Excel jobs are tool workflows — never Report.docx.
        if looks_like_full_gis_workflow_request(q):
            return False
        if self._looks_like_operational_workflow_request(q):
            return False
        if ql.startswith("re-execute full gis workflow") or "re-execute full gis workflow" in ql:
            return False
        if ql.startswith("continuation (retain session gis context"):
            return False

        # Require an actual Word deliverable intent — bare "save" is not enough
        # (users often "save as … excel file" inside GIS jobs).
        has_output_docx = (
            ".docx" in ql
            or "history.docx" in ql
            or "word document" in ql
            or "word doc" in ql
            or bool(re.search(r"\bas\s+a\s+word\b", ql))
            or bool(re.search(r"\b(?:save|export|write)\s+(?:it\s+)?(?:as\s+)?(?:a\s+)?(?:word\s+)?report\b", ql))
        )
        wants_save = any(
            k in ql
            for k in (
                "save",
                "saved",
                "export",
                "write a report",
                "write the report",
                "create a report",
                "generate a report",
            )
        )
        # Word-boundary report markers — avoid matching "history" inside GIS re-run
        # preambles or "process" inside unrelated words.
        is_report_like = bool(
            re.search(
                r"\b(?:report|essay|write-?up|overview|licensing|well-structured)\b",
                ql,
            )
            or any(
                k in ql
                for k in (
                    "trace the history",
                    "brief history",
                    "explain the process",
                    "turn this into",
                    "turn the previous",
                )
            )
        )
        has_input_doc = bool(self._extract_document_paths(query))
        return bool(has_output_docx and wants_save and is_report_like and not has_input_doc)

    def _is_pdf_replot_affirmation(self, routing_query: str, full_query: str) -> bool:
        """True when the user affirms a prior PDF→DWG replot request in injected history."""
        body = (self._cadastral_user_message_body(full_query) or routing_query or "").lower().strip()
        if not self._is_affirmative_reply(body) and not body.startswith("proceed"):
            return False
        hist = self._extract_history_block(full_query).lower()
        if ".pdf" not in hist or ".dwg" not in hist:
            return False
        return any(
            k in hist
            for k in ("replot", "generate", "plot using", "survey plan", "cadastral", "save strictly as")
        )

    def _is_explicit_session_docx_save_request(self, routing_query: str) -> bool:
        """True only when the user clearly asks to save a prior essay/report to .docx."""
        q = (routing_query or "").lower().strip()
        if not q:
            return False
        # Never treat GIS Excel "save as …xlsx" as a Word essay save.
        from survyai.gis_session_intent import looks_like_full_gis_workflow_request

        if looks_like_full_gis_workflow_request(routing_query):
            return False
        if self._looks_like_operational_workflow_request(routing_query) and ".docx" not in q and "word" not in q:
            return False
        has_docx = ".docx" in q or bool(re.search(r"\bessay[\w\-]*\.docx\b", q, flags=re.IGNORECASE))
        wants_word = has_docx or "word document" in q or "word doc" in q
        wants_save = any(
            k in q for k in ("save", "saved", "write it", "write this", "into '", 'into "')
        )
        explicit_essay = any(
            k in q
            for k in (
                "essay", "well-structured", "turn this", "turn the previous",
                "previous topic", "previous answer", "last answer",
            )
        )
        if has_docx and wants_save and explicit_essay:
            return True
        if wants_save and explicit_essay and ("turn this" in q or "turn the" in q):
            return True
        if wants_save and wants_word:
            return True
        return False

    def _looks_like_operational_workflow_request(self, routing_query: str) -> bool:
        """True for GIS/CAD/file automation jobs that must never route to essay-save."""
        from survyai.gis_session_intent import looks_like_full_gis_workflow_request

        q = (routing_query or "").lower()
        if looks_like_full_gis_workflow_request(routing_query):
            return True
        if not looks_like_file_driven_task(routing_query):
            # Still catch ArcGIS/buffer/overlap jobs that omit an explicit file extension.
            soft_markers = (
                "arcgis", "arcpy", "buffer", "overlap", "polygon", "utm",
                "mid-belt", "midbelt", "geodatabase", ".aprx", ".gdb",
                "convert", "easting", "northing", "owner",
            )
            if sum(1 for m in soft_markers if m in q) >= 3:
                return True
            return False
        operational_markers = (
            "arcgis", "arcpy", "cutfill", "cut fill", "cut/fill", "tin", "idw",
            "volume", "point feature", "feature class", "geodatabase", "create a copy",
            "copy each", "import", "compute", "calculate", "generate point",
            "borrow pit", "surface", "exported result", "buffer", "overlap",
            "polygon", "utm", "mid-belt", "midbelt", "convert", "crs",
            "fill color", "symbology", ".aprx", ".gdb", "excel",
        )
        return any(m in q for m in operational_markers)

    def _should_fastpath_save_session_docx(self, routing_query: str, full_query: str) -> bool:
        """
        Save a prior assistant answer (essay/report) from session history to .docx.

        Handles explicit save requests and short affirmations ('go ahead') after the
        assistant offered to create the document.
        """
        q = (routing_query or "").lower().strip()
        if not q:
            return False

        # Prefer OCR extraction → Word when a prior vision OCR result exists.
        try:
            from agent.vision_ocr import is_ocr_word_export_request, load_last_ocr_extraction

            if is_ocr_word_export_request(routing_query) and load_last_ocr_extraction(Path.cwd()):
                return False
        except Exception:
            pass

        # Never hijack operational file/GIS/CAD workflows (e.g. PRE/POST CSV + DWG volume).
        if self._classify_query_intent(routing_query) == "task":
            if not self._is_explicit_session_docx_save_request(routing_query):
                return False
        if self._looks_like_operational_workflow_request(routing_query):
            if not self._is_explicit_session_docx_save_request(routing_query):
                return False

        has_docx = ".docx" in q or bool(re.search(r"\bessay[\w\-]*\.docx\b", q, flags=re.IGNORECASE))
        wants_word = (
            has_docx
            or "word document" in q
            or "word doc" in q
            or bool(re.search(r"\bas\s+a\s+word\b", q))
        )
        wants_save = any(
            k in q
            for k in ("save", "saved", "write it", "write this", "into '", 'into "')
        )
        wants_essay = any(
            k in q
            for k in (
                "essay", "well-structured", "turn this", "previous topic", "previous answer",
                "last answer", "above", "report",
            )
        )
        has_history = "Assistant:" in (full_query or "")
        if has_docx and wants_save and (wants_essay or "turn this" in q):
            return has_history
        # "Save it as a word document 'S_G_G'" after a prior assistant answer.
        if wants_save and wants_word and has_history:
            return True
        if self._is_affirmative_reply(q) and has_history:
            if self._last_assistant_offered_session_docx_save(full_query):
                return True
        return False

    def _last_assistant_offered_session_docx_save(self, query: str) -> bool:
        block = self._extract_history_block(query)
        idx = block.rfind("Assistant:")
        if idx == -1:
            return False
        last_assistant = block[idx + len("Assistant:"):].lower()
        markers = (
            "essay", "essay1.docx", "save it as", "well-structured essay",
            "turn the previous topic", "turn this into", "save as **essay",
            "save as essay", "i can turn", "word document", "save it into",
        )
        return any(m in last_assistant for m in markers)

    _SESSION_TEXT_TRUNCATION_MARKERS = ("…[truncated]", "[truncated]")
    _MAX_DOCX_ESSAY_SOURCE_CHARS = 1_000_000

    @classmethod
    def _session_text_looks_truncated(cls, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        return any(m in t for m in cls._SESSION_TEXT_TRUNCATION_MARKERS)

    def _get_full_assistant_response_from_session(
        self, session_id: Optional[str] = None
    ) -> str:
        """Return the longest stored assistant reply for this session (untruncated)."""
        if self.vector_store is None:
            return ""
        sid = session_id or self.get_session_id()
        if not sid:
            return ""
        try:
            recents = self.vector_store.get_recent_conversations(
                session_id=sid, limit=30, role="assistant"
            )
            if not recents:
                recents = self.vector_store.get_recent_conversations(
                    session_id=sid, limit=30
                )
            best = ""
            for conv in recents:
                meta = conv.get("metadata") or {}
                role = meta.get("role") or conv.get("role")
                if role and role != "assistant":
                    continue
                content = (conv.get("content") or "").strip()
                if len(content) > len(best):
                    best = content
            return best
        except Exception as exc:
            logger.debug("Session assistant lookup failed: %s", exc)
            return ""

    def _extract_user_question_for_docx_essay(self, query: str) -> str:
        """Pick the substantive user question that produced the essay source material."""
        block = self._extract_history_block(query)
        users: List[str] = []
        for line in block.splitlines():
            if line.startswith("User:"):
                users.append(line[len("User:"):].strip())
        skip = (
            "turn this", "well-structured", "essay", "save it", "go ahead",
            "same folder", "workspace",
        )
        for text in reversed(users):
            tl = text.lower()
            if len(text) < 20:
                continue
            if any(k in tl for k in skip):
                continue
            return text
        return users[0] if users else ""

    def _resolve_docx_save_source_text(self, query: str) -> str:
        """Prefer full vector-store text over truncated injected history."""
        history_text = self._extract_assistant_content_for_docx_save(query)
        session_text = self._get_full_assistant_response_from_session()
        candidates = [t for t in (history_text, session_text) if (t or "").strip()]
        if not candidates:
            return ""

        def _score(text: str) -> tuple:
            return (0 if self._session_text_looks_truncated(text) else 1, len(text))

        return max(candidates, key=_score).strip()

    def _extract_assistant_content_for_docx_save(self, query: str) -> str:
        """Pick the substantive assistant answer to persist (not a short follow-up)."""
        block = self._extract_history_block(query)
        if not block.strip():
            return ""
        assistants: List[str] = []
        current: List[str] = []
        for line in block.splitlines():
            if line.startswith("Assistant:"):
                if current:
                    assistants.append("\n".join(current).strip())
                current = [line[len("Assistant:"):].strip()]
            elif line.startswith("User:") or line.startswith("--- Exchange"):
                if current:
                    assistants.append("\n".join(current).strip())
                    current = []
            elif line.startswith("--- End of History"):
                if current:
                    assistants.append("\n".join(current).strip())
                    current = []
                break
            elif current:
                current.append(line)
        if current:
            assistants.append("\n".join(current).strip())
        if not assistants:
            return ""
        skip_markers = (
            "i'm ready to proceed",
            "i'm missing",
            "missing the source",
            "please send either",
            "understood — i'll use",
            "understood - i'll use",
            "permission required",
            "may i search",
        )
        for text in reversed(assistants):
            tl = text.lower()
            if len(text) < 120:
                continue
            if any(m in tl for m in skip_markers):
                continue
            return text.strip()
        return assistants[-1].strip()

    def _extract_requested_save_basename(self, routing_query: str) -> Optional[str]:
        """Extract a save-as basename from prompts like: save as a word document 'S_G_G'."""
        q = routing_query or ""
        patterns = (
            r"(?:save|saved|export)\s+(?:it\s+)?as\s+(?:a\s+)?(?:word\s+)?(?:document|doc|docx)\s+['\"]([^'\"]+)['\"]",
            r"(?:word\s+document|word\s+doc)\s+['\"]([^'\"]+)['\"]",
            r"(?:save|saved|export)\s+(?:it\s+)?as\s+['\"]([^'\"]+)['\"]",
        )
        for pat in patterns:
            m = re.search(pat, q, flags=re.IGNORECASE)
            if not m:
                continue
            name = (m.group(1) or "").strip().strip("\"'").rstrip(").,;")
            if not name:
                continue
            # Ignore obvious non-document tokens / paths with separators when bare name expected.
            if any(sep in name for sep in ("/", "\\")) and not name.lower().endswith((".docx", ".doc")):
                # Absolute/relative path — still usable as output ref.
                return name
            if name.lower() in {"it", "this", "that", "word", "document", "docx"}:
                continue
            return name
        return None

    def _resolve_session_docx_output_path(self, routing_query: str, workspace: Optional[Path] = None) -> Path:
        ws = (workspace or Path.cwd()).resolve()
        out = self._extract_any_output_docx(routing_query)
        if out:
            resolved = self._resolve_user_output_path(routing_query, out, fallback_dir=ws)
            return resolved if resolved.suffix.lower() == ".docx" else resolved.with_suffix(".docx")
        m = re.search(r"['\"]([^'\"]+\.docx)['\"]", routing_query or "", flags=re.IGNORECASE)
        if m:
            return self._resolve_user_output_path(routing_query, m.group(1), fallback_dir=ws)
        bare = self._extract_requested_save_basename(routing_query)
        if bare:
            name = bare if bare.lower().endswith((".docx", ".doc")) else f"{bare}.docx"
            resolved = self._resolve_user_output_path(routing_query, name, fallback_dir=ws)
            return resolved if resolved.suffix.lower() == ".docx" else resolved.with_suffix(".docx")
        m2 = re.search(r"\b(essay\d*)\b", routing_query or "", flags=re.IGNORECASE)
        if m2:
            name = m2.group(1)
            return self._resolve_user_output_path(
                routing_query,
                name if name.lower().endswith(".docx") else f"{name}.docx",
                fallback_dir=ws,
            )
        return ws / "essay1.docx"

    def _run_save_session_docx_pipeline(
        self,
        *,
        query: str,
        routing_query: str,
        llm: Optional[BaseChatModel] = None,
        model_name_used: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Format the prior assistant answer as an essay/report and save to .docx."""
        from pathlib import Path

        if (
            self._classify_query_intent(routing_query) == "task"
            or self._looks_like_operational_workflow_request(routing_query)
        ) and not self._is_explicit_session_docx_save_request(routing_query):
            return {
                "success": False,
                "error": "misroute_operational_task",
                "response": (
                    "This request looks like a file/GIS/CAD operation, not saving a prior essay. "
                    "Use the standard agent workflow with the appropriate tools."
                ),
            }

        workspace = Path.cwd().resolve()
        output_path = self._resolve_session_docx_output_path(routing_query, workspace)
        source_text = self._resolve_docx_save_source_text(query)
        rq_lower = (routing_query or "").lower()
        wants_essay_format = any(
            k in rq_lower
            for k in ("essay", "well-structured", "turn this", "report")
        )
        needs_llm_essay = llm is not None and (
            len(source_text) < 200
            or self._session_text_looks_truncated(source_text)
            or wants_essay_format
        )

        if needs_llm_essay:
            user_question = self._extract_user_question_for_docx_essay(query)
            hist = self._extract_history_block(query)
            truncated_note = ""
            if self._session_text_looks_truncated(source_text):
                truncated_note = (
                    "\nIMPORTANT: The SOURCE MATERIAL below may end abruptly or contain "
                    "'[truncated]'. Complete every section logically from the topic — "
                    "do not stop mid-sentence.\n"
                )
            prompt = (
                "Write a complete, well-structured professional essay for "
                "surveyors/geospatial analysts.\n"
                "REQUIREMENTS:\n"
                "- Use the SOURCE MATERIAL as the factual basis; preserve technical detail.\n"
                "- Include an introduction, numbered sections, and a short conclusion.\n"
                "- Output the final essay text only — no preambles or follow-up questions.\n"
                f"{truncated_note}\n"
                f"ORIGINAL USER QUESTION:\n{user_question or routing_query}\n\n"
                f"SOURCE MATERIAL (prior assistant answer):\n"
                f"{source_text[: self._MAX_DOCX_ESSAY_SOURCE_CHARS]}\n\n"
                f"CONVERSATION (reference):\n{hist[-8000:]}\n\n"
                f"CURRENT SAVE REQUEST:\n{routing_query}\n"
            )
            from langchain_core.messages import HumanMessage

            msg, err, timed_out = self._run_with_timeout(
                120, lambda: llm.invoke([HumanMessage(content=prompt)]), llm_model_name=model_name_used
            )
            if not timed_out and not err and msg is not None:
                raw = msg.content if hasattr(msg, "content") else str(msg)
                if isinstance(raw, list):
                    raw = "\n".join(
                        str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in raw
                    )
                if str(raw).strip():
                    source_text = str(raw).strip()

        if not source_text.strip():
            return {
                "success": False,
                "error": "No prior assistant content found to save.",
                "response": (
                    "I could not find the essay text in this session. "
                    "Ask your question again, then request saving to essay1.docx."
                ),
            }

        title = output_path.stem.replace("_", " ").title()
        create_result = self.document_processor.create_word_document(
            str(output_path), source_text, title=title
        )
        if not create_result.get("success"):
            return {
                "success": False,
                "error": create_result.get("error", "Failed to create Word document"),
                "response": str(create_result),
                "output_path": str(output_path),
            }

        return {
            "success": True,
            "response": (
                f"Saved essay to Word document.\n"
                f"- Output: {output_path}\n"
                f"- Title: {title}\n"
            ),
            "output_path": str(output_path),
            "model_name": model_name_used,
        }

    def _should_fastpath_dwg_plan_extract_to_docx(self, query: str) -> bool:
        """Fast-path: extract cadastral plan details from one or more DWGs into Word."""
        from agent.pdf_survey_plan import should_fastpath_dwg_plan_extract_to_docx

        return should_fastpath_dwg_plan_extract_to_docx(query)

    def _run_dwg_plan_extract_to_docx_pipeline(self, query: str) -> Dict[str, Any]:
        """Structured cadastral extraction per DWG → Word (heuristics + LLM + vector context)."""
        from pathlib import Path

        from agent.pdf_survey_plan import run_dwg_plan_extract_to_docx

        workspace = Path.cwd().resolve()
        scope = self._cadastral_user_message_body(query)

        llm, model_name = self._try_openai_tier_llm("average")
        run_with_timeout = self._llm_run_with_timeout(model_name) if llm else None

        field_context = ""
        try:
            snippets: List[str] = []
            for collection in (COLLECTION_DOCUMENTS, COLLECTION_DRAWINGS):
                hits = self._vs_search(
                    "Nigerian cadastral survey plan buyer location LGA surveyor plan number certification",
                    collection,
                    top_k=2,
                )
                for hit in hits or []:
                    text = (hit.get("text") or hit.get("content") or "").strip()
                    if text:
                        snippets.append(text[:800])
            if snippets:
                field_context = (
                    "RELEVANT SURVEY KNOWLEDGE (from vector store):\n"
                    + "\n---\n".join(snippets[:4])
                )
        except Exception:
            pass

        return run_dwg_plan_extract_to_docx(
            query=scope,
            autocad=self.autocad,
            dxf_fallback=self.dxf_fallback,
            document_processor=self.document_processor,
            workspace=workspace,
            llm=llm,
            run_with_timeout=run_with_timeout,
            field_context=field_context,
        )

    def _should_fastpath_pdf_plan_key_details(self, query: str) -> bool:
        """Fast-path: extract key survey-plan details from a PDF via layout + vision."""
        from agent.pdf_survey_plan import should_fastpath_pdf_plan_key_details

        return should_fastpath_pdf_plan_key_details(query)

    def _run_pdf_plan_key_details_pipeline(self, query: str) -> Dict[str, Any]:
        """Scanned/image survey PDFs: vision+layout key details (not plain text only)."""
        from agent.pdf_survey_plan import run_pdf_plan_key_details_extract

        scope = self._cadastral_user_message_body(query)
        return run_pdf_plan_key_details_extract(
            query=scope,
            full_query=query,
            extract_fn=self._extract_pdf_survey_plan_with_tier_fallback,
        )

    def _vision_ocr_user_text(self, query: str) -> str:
        """User-facing text without the attachment marker block (for mode selection)."""
        from survyai.attachments import parse_attachments_block

        _, remaining = parse_attachments_block(query or "")
        return (remaining or "").strip()

    def _extract_images_survey_plan_with_tier_fallback(
        self,
        image_paths: List[str],
        *,
        user_notes: str,
        timeout_s: int = 120,
    ) -> tuple[Any, Optional[str]]:
        """Survey-plan extraction from standalone images (average → complex)."""
        from agent.pdf_survey_plan import SurveyPlanExtraction, validate_extraction_for_replot
        from agent.vision_ocr import extract_survey_plan_from_images

        tiers: List[tuple[str, int]] = [("average", 1), ("complex", 2)]
        last: Optional[Any] = None
        last_model: Optional[str] = None
        for tier, _pages in tiers:
            llm, model_name = self._try_openai_tier_llm(tier)  # type: ignore[arg-type]
            if llm is None:
                llm = getattr(self, "llm_primary", None)
            if llm is None:
                break
            if not model_name:
                model_name = getattr(self, "_current_openai_model", None) or getattr(
                    self.settings, "openai_model", "gpt-5.4-mini"
                )
            self._current_openai_model = model_name
            extraction = extract_survey_plan_from_images(
                image_paths,
                llm=llm,
                run_with_timeout=self._llm_run_with_timeout(model_name),
                user_notes=user_notes,
                timeout_s=timeout_s,
            )
            last = extraction
            last_model = model_name
            if extraction.source in ("error", "llm_parse_failed"):
                continue
            if not validate_extraction_for_replot(extraction):
                return extraction, model_name
            logger.info(
                "Image survey extraction incomplete on tier %s — retrying stronger model if available",
                tier,
            )
        if last is None:
            return (
                SurveyPlanExtraction(source="error", notes="No LLM available for image extraction"),
                None,
            )
        return last, last_model

    def _run_image_survey_replot_pipeline(self, query: str) -> Dict[str, Any]:
        """Extract survey plan from image(s) and replot via cadastral CAD pipeline."""
        from pathlib import Path

        from agent.pdf_survey_plan import (
            apply_plan_overrides_to_extraction,
            build_cadastral_subprompt,
            filter_user_facing_extraction_notes,
            resolve_output_dwg_path,
            resolve_plan_overrides_from_query,
            validate_extraction_for_replot,
            validate_subprompt_geometry,
        )
        from agent.vision_ocr import extract_image_paths_from_query

        scope = self._cadastral_user_message_body(query)
        image_paths = extract_image_paths_from_query(query)
        if not image_paths:
            return {"success": False, "error": "No image path found in the request."}

        # Prefer an explicit .dwg in the query; else derive beside the first image.
        output_dwg = resolve_output_dwg_path(query, image_paths[0], scope_text=scope)
        if not output_dwg:
            stem = Path(image_paths[0]).stem or "survey_plan"
            output_dwg = str(Path(image_paths[0]).with_name(f"{stem}_replot.dwg"))

        mem = self._resolve_cadastral_template_from_memory(query)
        self._ensure_autocad_connected()
        override_llm, override_model = self._try_openai_tier_llm("simple")

        extraction, model_name = self._extract_images_survey_plan_with_tier_fallback(
            image_paths,
            user_notes=scope,
            timeout_s=120,
        )
        if override_llm is None:
            override_llm = getattr(self, "llm_primary", None)
        if not override_model:
            override_model = model_name
        if model_name:
            self._current_openai_model = model_name
        if extraction.source in ("error", "llm_parse_failed"):
            return {
                "success": False,
                "error": extraction.notes or "Image survey plan extraction failed.",
                "extraction": extraction.model_dump(),
            }

        validation_issues = validate_extraction_for_replot(extraction)
        if validation_issues:
            return {
                "success": False,
                "error": (
                    "Image extraction is incomplete and cannot be replotted safely: "
                    + "; ".join(validation_issues)
                ),
                "validation_issues": validation_issues,
                "extraction": extraction.model_dump(),
            }

        template_hint = mem.get("template_path") if mem else None
        plan_overrides = resolve_plan_overrides_from_query(
            query,
            scope_text=scope,
            base_extraction=extraction,
            llm=override_llm,
            run_with_timeout=self._llm_run_with_timeout(override_model),
        )
        extraction = apply_plan_overrides_to_extraction(extraction, plan_overrides)
        from agent.pdf_survey_plan import enrich_extraction_coordinates

        extraction = enrich_extraction_coordinates(extraction, "")
        cert_date = plan_overrides.certification_date or extraction.certification_date or None
        subprompt = build_cadastral_subprompt(
            extraction,
            output_dwg_path=output_dwg,
            certification_date=cert_date,
            template_path=template_hint,
        )
        subprompt_issues = validate_subprompt_geometry(subprompt)
        if subprompt_issues:
            return {
                "success": False,
                "error": (
                    "Generated CAD sub-prompt is missing required geometry: "
                    + "; ".join(subprompt_issues)
                ),
                "validation_issues": subprompt_issues,
                "extraction": extraction.model_dump(),
                "subprompt": subprompt,
            }

        plot_result = self._run_cadastral_cad_prompt_pipeline(
            subprompt,
            source_scale_denom=extraction.scale_denom,
            skip_intent_assessment=True,
            user_scope_for_extras=scope,
        )
        if not plot_result.get("success"):
            return {
                "success": False,
                "error": plot_result.get("error", "Cadastral replot failed."),
                "extraction": extraction.model_dump(),
                "subprompt": subprompt,
            }

        out_lines = [
            "✅ Survey plan image(s) replotted to CAD.",
            f"- Source image(s): {', '.join(image_paths)}",
            f"- Output DWG: {plot_result.get('output_dwg') or output_dwg}",
            f"- Extraction: {extraction.source} (confidence {extraction.confidence:.0%})",
        ]
        filtered_notes = filter_user_facing_extraction_notes(
            extraction.notes or "",
            plan_overrides.override_fields,
        )
        if filtered_notes:
            out_lines.append(f"- Notes: {filtered_notes}")
        return {
            "success": True,
            "response": "\n".join(out_lines) + "\n",
            "output_dwg": plot_result.get("output_dwg") or output_dwg,
            "output_path": plot_result.get("output_dwg") or output_dwg,
            "extraction": extraction.model_dump(),
            "model_name": model_name,
        }

    def _run_vision_ocr_for_query(
        self,
        query: str,
        *,
        mode: Optional[str] = None,
    ) -> Any:
        """Run LLM vision OCR on image paths found in the query."""
        from agent.vision_ocr import (
            OCR_BUDGET_S,
            VisionOcrResult,
            extract_image_paths_from_query,
            looks_like_survey_sheet,
            run_vision_ocr,
            select_vision_ocr_mode,
        )

        image_paths = extract_image_paths_from_query(query)
        if not image_paths:
            return VisionOcrResult(success=False, error="No image paths found.")

        user_text = self._vision_ocr_user_text(query)
        resolved_mode = mode or select_vision_ocr_mode(user_text)
        if resolved_mode == "plain_text" and looks_like_survey_sheet(user_text, image_paths):
            resolved_mode = "structured"

        # One average-tier vision call (style lock + extract). Escalate to complex
        # only when the first pass is empty — keeps typical OCR ≤ ~60s.
        ocr_timeout = int(getattr(self.settings, "vision_ocr_timeout_s", OCR_BUDGET_S) or OCR_BUDGET_S)
        last: Optional[VisionOcrResult] = None
        for tier in ("average", "complex"):
            llm, model_name = self._try_openai_tier_llm(tier)  # type: ignore[arg-type]
            if llm is None:
                llm = getattr(self, "llm_primary", None)
            if llm is None:
                break
            if not model_name:
                model_name = getattr(self, "_current_openai_model", None) or getattr(
                    self.settings, "openai_model", "gpt-5.4-mini"
                )
            result = run_vision_ocr(
                image_paths,
                user_text=user_text,
                llm=llm,
                run_with_timeout=self._llm_run_with_timeout(model_name),
                mode=resolved_mode,  # type: ignore[arg-type]
                timeout_s=ocr_timeout,
                max_files=int(getattr(self.settings, "vision_ocr_max_files", 4) or 4),
                max_file_mb=int(getattr(self.settings, "vision_ocr_max_file_mb", 10) or 10),
                model_name=model_name,
                high_accuracy=True,
                verify_uncertain=False,
                workspace=Path.cwd(),
            )
            last = result
            if not result.success:
                err = (result.error or "").lower()
                if "timed out" in err or tier == "complex":
                    return result
                continue
            if resolved_mode == "plain_text":
                return result
            structured = result.structured or {}
            has_meat = bool(
                structured.get("coordinates")
                or structured.get("bearings_distances")
                or structured.get("pillars")
                or structured.get("tables")
                or structured.get("fields")
                or structured.get("rows")
                or (result.text and len(result.text) > 40)
            )
            if has_meat or tier == "complex":
                return result
        return last or VisionOcrResult(success=False, error="No vision LLM available.")

    # ==========================================================================
    # FAST-PATH: CAD CADASTRAL PLAN (Template DWG -> Output DWG)
    # ==========================================================================

    def _cad_template_profiles_dir(self):
        """Writable directory for learned CAD template profiles + memory.

        Frozen-safe: in a packaged (PyInstaller) build the source tree lives under
        a read-only/ephemeral _MEIPASS temp dir, so learned profiles MUST be
        written to a stable, user-writable location (%APPDATA%\\SurvyAI on
        Windows).  In dev (non-frozen) we also use the user-data dir for
        consistency; existing project-root profiles are still read for seeding via
        _cad_template_profiles_seed_dirs().
        """
        d = user_data_path("template_profiles").resolve()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _cad_template_profiles_seed_dirs(self):
        """Read-only directories scanned to seed/bootstrap template memory.

        Order: shipped default templates (bundled_templates/), other bundled
        template_profiles/, and in dev only the project-root template_profiles/
        folder.  These are never written to.
        """
        from pathlib import Path

        seeds = []
        for rel in ("bundled_templates", "template_profiles"):
            try:
                bundled = resource_path(rel).resolve()
                if bundled.exists():
                    seeds.append(bundled)
            except Exception:
                pass
        if not is_frozen_app():
            for rel in ("bundled_templates", "template_profiles"):
                try:
                    dev_dir = (project_root() / rel).resolve()
                    if dev_dir.exists():
                        seeds.append(dev_dir)
                except Exception:
                    pass
        # De-duplicate while preserving order, and never include the writable dir here.
        out = []
        seen = set()
        writable = str(self._cad_template_profiles_dir())
        for s in seeds:
            key = str(s)
            if key != writable and key not in seen:
                seen.add(key)
                out.append(s)
        return out

    def _prefer_shipped_bundled_template(self, tp: Path) -> Path:
        """
        Prefer ``bundled_templates/<name>`` when the remembered path is a stale
        same-named copy outside the bundle (common for survey_plan_template3.dwg).

        User-edited bundled templates then take effect without requiring a manual
        template-memory reset. Custom uniquely-named templates are left alone.
        """
        from pathlib import Path

        try:
            cur = Path(tp).resolve()
        except Exception:
            return tp
        name = cur.name
        if not name.lower().startswith("survey_plan_template") or not name.lower().endswith(".dwg"):
            return cur
        # Already the bundled file.
        try:
            if "bundled_templates" in str(cur).replace("\\", "/").lower():
                return cur
        except Exception:
            pass
        candidates = []
        try:
            candidates.append(Path(resource_path("bundled_templates", name)))
        except Exception:
            pass
        if not is_frozen_app():
            try:
                candidates.append((project_root() / "bundled_templates" / name).resolve())
            except Exception:
                pass
        best = None
        for c in candidates:
            try:
                cr = Path(c).resolve()
                if cr.is_file():
                    best = cr
                    break
            except Exception:
                continue
        if best is None:
            return cur
        try:
            # Prefer bundled when it is same age or newer than the remembered copy.
            if best.stat().st_mtime + 1.0 >= cur.stat().st_mtime:
                return best
        except Exception:
            return best
        return cur

    def _resolve_cad_template_file_path(
        self,
        path_raw: str,
        *,
        profile_file: Optional[Any] = None,
    ):
        """Resolve a template DWG path from a profile (absolute, relative, or bundled)."""
        from pathlib import Path

        raw = str(path_raw or "").strip()
        if not raw:
            return None
        p = Path(raw)
        name = p.name if p.suffix else f"{p.name}.dwg"
        if not name.lower().endswith((".dwg", ".dxf")):
            name = f"{Path(name).stem}.dwg"

        candidates = []
        # Prefer shipped bundled defaults first for the canonical template names so
        # edits under bundled_templates/ win over stale project-root copies.
        if name.lower().startswith("survey_plan_template"):
            candidates.append(resource_path("bundled_templates", name))
            if not is_frozen_app():
                candidates.append(project_root() / "bundled_templates" / name)
        if p.is_absolute():
            candidates.append(p)
        else:
            if profile_file is not None:
                try:
                    prof_parent = Path(profile_file).resolve().parent
                    candidates.append(prof_parent / p)
                    candidates.append(prof_parent / name)
                except Exception:
                    pass
            candidates.append(resource_path("bundled_templates", name))
            candidates.append(resource_path(name))
            if not is_frozen_app():
                candidates.append(project_root() / "bundled_templates" / name)
                candidates.append(project_root() / name)
        # Absolute paths that vanished on another machine: fall back by filename.
        if p.is_absolute():
            candidates.append(resource_path("bundled_templates", name))
            if not is_frozen_app():
                candidates.append(project_root() / "bundled_templates" / name)
                candidates.append(project_root() / name)

        for c in candidates:
            try:
                cr = Path(c).resolve()
                if cr.is_file():
                    return self._prefer_shipped_bundled_template(cr)
            except Exception:
                continue
        return None

    def _cad_template_memory_path(self):
        from pathlib import Path
        if self._cad_template_memory_file:
            return Path(self._cad_template_memory_file).resolve()
        p = (self._cad_template_profiles_dir() / "template_memory.json").resolve()
        self._cad_template_memory_file = str(p)
        return p

    def _load_cad_template_memory(self) -> Dict[str, Any]:
        import json as _json
        from pathlib import Path

        mem_path = self._cad_template_memory_path()
        data: Dict[str, Any] = {"templates": []}
        if not mem_path.exists():
            return self._bootstrap_cad_template_memory_from_profiles(data)
        try:
            raw = _json.loads(mem_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("templates"), list):
                data = raw
        except Exception:
            data = {"templates": []}
        if not (data.get("templates") or []):
            data = self._bootstrap_cad_template_memory_from_profiles(data)
            return data
        # Fresh installs / moved machines may keep stale absolute paths. If nothing
        # on disk is usable, re-seed from bundled defaults (merge, don't wipe).
        any_available = False
        for ent in data.get("templates") or []:
            try:
                if Path(str(ent.get("path") or "")).exists():
                    any_available = True
                    break
            except Exception:
                continue
        if not any_available:
            data = self._bootstrap_cad_template_memory_from_profiles(data)
        return data

    def _save_cad_template_memory(self, data: Dict[str, Any]) -> None:
        import json as _json
        mem_path = self._cad_template_memory_path()
        mem_path.parent.mkdir(parents=True, exist_ok=True)
        mem_path.write_text(_json.dumps(data, indent=2), encoding="utf-8")

    def _bootstrap_cad_template_memory_from_profiles(self, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Seed template memory from existing learned template profiles.
        This supports older installations that already have template_profiles/*.json
        but were created before template_memory.json existed.
        """
        import json as _json
        from pathlib import Path

        data = dict(data or {"templates": []})
        entries = list(data.get("templates") or [])
        by_path: Dict[str, Dict[str, Any]] = {}

        for ent in entries:
            try:
                p = str(Path(str(ent.get("path") or "")).resolve())
                if p:
                    by_path[p] = ent
            except Exception:
                continue

        # Scan the writable profiles dir first, then read-only seed dirs (bundled
        # resources + dev project root) so a fresh install can bootstrap from
        # shipped profiles while still preferring user-learned ones.
        scan_dirs = [self._cad_template_profiles_dir()] + list(self._cad_template_profiles_seed_dirs())
        seen_profile_names: set = set()
        dirty = False
        for profile_dir in scan_dirs:
            for prof_path in profile_dir.glob("*.json"):
                try:
                    if prof_path.name.lower() == "template_memory.json":
                        continue
                    # First occurrence wins (writable dir takes precedence over seeds).
                    if prof_path.name.lower() in seen_profile_names:
                        continue
                    seen_profile_names.add(prof_path.name.lower())
                    raw = _json.loads(prof_path.read_text(encoding="utf-8"))
                    template_meta = raw.get("template") or {}
                    tp_raw = str(template_meta.get("path") or "").strip()
                    if not tp_raw:
                        continue
                    tp = self._resolve_cad_template_file_path(tp_raw, profile_file=prof_path)
                    if tp is None:
                        continue
                    tp_res = str(tp.resolve())
                    learned_at = str(template_meta.get("learned_at") or "")
                    sig = template_meta.get("signature") or {}
                    stat = tp.stat() if tp.exists() else None
                    # Prefer a writable profile copy so learning/apply never writes
                    # into the read-only install bundle.
                    writable_prof = (self._cad_template_profiles_dir() / prof_path.name).resolve()
                    if not writable_prof.exists():
                        try:
                            seeded = dict(raw)
                            tmpl = dict(seeded.get("template") or {})
                            tmpl["path"] = tp_res
                            tmpl["name"] = tp.name
                            seeded["template"] = tmpl
                            drawing = seeded.get("drawing_info")
                            if isinstance(drawing, dict):
                                drawing = dict(drawing)
                                drawing["path"] = tp_res
                                drawing["name"] = tp.name
                                seeded["drawing_info"] = drawing
                            writable_prof.write_text(
                                _json.dumps(seeded, indent=2),
                                encoding="utf-8",
                            )
                        except Exception:
                            writable_prof = prof_path.resolve()
                    profile_for_entry = writable_prof if writable_prof.exists() else prof_path.resolve()
                    entry = by_path.get(tp_res) or {
                        "id": tp.stem,
                        "path": tp_res,
                        "name": tp.name,
                        "aliases": self._candidate_template_aliases(tp_res),
                        "use_count": 0,
                    }
                    entry["profile_path"] = str(profile_for_entry)
                    entry["last_used_at"] = str(entry.get("last_used_at") or learned_at or "")
                    entry["is_available"] = bool(tp.exists())
                    entry["signature"] = {
                        "size_bytes": int(sig.get("size_bytes") or (stat.st_size if stat else -1)),
                        "mtime_ns": int(sig.get("mtime_ns") or (getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)) if stat else -1)),
                    }
                    by_path[tp_res] = entry
                    dirty = True
                except Exception:
                    continue

        if dirty:
            data["templates"] = list(by_path.values())
            try:
                self._save_cad_template_memory(data)
            except Exception:
                pass
        else:
            data["templates"] = list(by_path.values()) if by_path else list(entries)
        return data

    def _candidate_template_aliases(self, template_path: str) -> List[str]:
        import re
        from pathlib import Path
        tp = Path(template_path)
        stem = tp.stem.strip()
        name = tp.name.strip()
        aliases = [stem, name]
        # Split common separators so prompts like "template 3" or "1000 template" can still match.
        parts = re.split(r"[_\-\s]+", stem)
        aliases.extend([p for p in parts if p])
        out: List[str] = []
        seen: set[str] = set()
        for a in aliases:
            key = str(a).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(str(a))
        return out

    def _register_cad_template_memory(self, template_path: str, profile_path: str) -> None:
        import time
        from pathlib import Path

        try:
            tp = Path(template_path).resolve()
            prof = Path(profile_path).resolve()
            stat = tp.stat()
            data = self._load_cad_template_memory()
            entries = list(data.get("templates") or [])
            tp_res = str(tp)
            prof_res = str(prof)
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            idx_hit = -1
            for idx, ent in enumerate(entries):
                try:
                    if str(Path(str(ent.get("path") or "")).resolve()) == tp_res:
                        idx_hit = idx
                        break
                except Exception:
                    continue
            new_entry = {
                "id": tp.stem,
                "path": tp_res,
                "name": tp.name,
                "aliases": self._candidate_template_aliases(tp_res),
                "last_used_at": now,
                "use_count": 1,
                "profile_path": prof_res,
                "signature": {
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
                },
                "is_available": True,
            }
            if idx_hit >= 0:
                cur = dict(entries[idx_hit] or {})
                new_entry["use_count"] = int(cur.get("use_count") or 0) + 1
                aliases = list(cur.get("aliases") or []) + new_entry["aliases"]
                seen: set[str] = set()
                dedup_aliases: List[str] = []
                for a in aliases:
                    key = str(a).strip().lower()
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    dedup_aliases.append(str(a))
                new_entry["aliases"] = dedup_aliases
                entries[idx_hit] = new_entry
            else:
                entries.append(new_entry)
            data["templates"] = entries
            self._save_cad_template_memory(data)
        except Exception:
            pass

    def _resolve_cadastral_template_from_memory(self, query: str) -> Optional[Dict[str, str]]:
        import time
        from pathlib import Path

        q = (query or "").lower()
        data = self._load_cad_template_memory()
        entries = list(data.get("templates") or [])
        valid_entries: List[Dict[str, Any]] = []
        dirty = False

        for ent in entries:
            try:
                tp_raw = str(ent.get("path") or "")
                tp = Path(tp_raw).resolve() if tp_raw else None
                if tp is None or not tp.exists():
                    # Recover relocated / bundled default templates by filename.
                    recovered = self._resolve_cad_template_file_path(
                        tp_raw or str(ent.get("name") or ent.get("id") or ""),
                        profile_file=ent.get("profile_path"),
                    )
                    if recovered is not None and recovered.exists():
                        ent["path"] = str(recovered.resolve())
                        ent["name"] = recovered.name
                        tp = recovered
                        dirty = True
                elif tp is not None and tp.exists():
                    # Stale project-root copies of shipped templates → bundled_templates.
                    preferred = self._prefer_shipped_bundled_template(tp)
                    try:
                        if preferred.resolve() != tp.resolve():
                            ent["path"] = str(preferred.resolve())
                            ent["name"] = preferred.name
                            tp = preferred
                            dirty = True
                    except Exception:
                        pass
                exists = bool(tp is not None and tp.exists())
                if bool(ent.get("is_available")) != bool(exists):
                    ent["is_available"] = bool(exists)
                    dirty = True
                if exists:
                    valid_entries.append(ent)
            except Exception:
                ent["is_available"] = False
                dirty = True

        if dirty:
            data["templates"] = entries
            try:
                self._save_cad_template_memory(data)
            except Exception:
                pass

        if not valid_entries:
            # Last resort: seed bundled defaults, then retry once.
            data = self._bootstrap_cad_template_memory_from_profiles(data)
            entries = list(data.get("templates") or [])
            for ent in entries:
                try:
                    tp = Path(str(ent.get("path") or "")).resolve()
                    if tp.exists():
                        valid_entries.append(ent)
                except Exception:
                    continue
            if not valid_entries:
                return None

        def _score(ent: Dict[str, Any]) -> Tuple[int, str]:
            score = 0
            aliases = [str(a).lower() for a in (ent.get("aliases") or []) if str(a).strip()]
            name = str(ent.get("name") or "").lower()
            stem = str(ent.get("id") or "").lower()
            for token in aliases + [name, stem]:
                if token and token in q:
                    score += max(5, len(token))
            # Most recent valid template wins when there is no clear semantic match.
            last_used = str(ent.get("last_used_at") or "")
            return (score, last_used)

        best = sorted(valid_entries, key=_score, reverse=True)[0]
        try:
            best["last_used_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            data["templates"] = entries
            self._save_cad_template_memory(data)
        except Exception:
            pass
        return {
            "template_path": str(Path(str(best.get("path") or "")).resolve()),
            "profile_path": str(Path(str(best.get("profile_path") or "")).resolve()) if best.get("profile_path") else "",
            "template_name": str(best.get("name") or ""),
        }

    @staticmethod
    def _cadastral_user_message_body(query: str) -> str:
        """
        Strip GUI continuation wrappers so cadastral parsing only sees the *current* user request.

        Otherwise injected history can repeat 'Generate \\'Check16.dwg\\' ...' many times and the
        batch splitter will run the same plan repeatedly (wipes credits / terrible UX).
        """
        q = (query or "").strip()
        marker = "NOW, the user wants you to continue with this new request:"
        if marker in q:
            return q.split(marker, 1)[-1].strip()
        return q

    def _should_fastpath_pdf_survey_replot(
        self, query: str, routing_query: Optional[str] = None
    ) -> bool:
        """
        True when the user wants to replot a survey/cadastral plan PDF to a DWG.

        Uses routing_query (current turn only) for intent/keywords so injected GUI
        history cannot hijack unrelated knowledge questions.
        """
        if not getattr(self.settings, "pdf_survey_replot_enabled", True):
            return False

        body = self._cadastral_user_message_body(query).lower().strip()
        scope_q = (routing_query or body or "").lower().strip()

        if self._classify_query_intent(scope_q) == "knowledge":
            return False
        if any(m in scope_q for m in _CADASTRAL_FASTPATH_EXCLUDE_MARKERS):
            return False

        affirmation = self._is_pdf_replot_affirmation(scope_q, query)
        if affirmation:
            return True

        if ".pdf" not in scope_q:
            return False
        if ".dwg" not in scope_q:
            return False
        if not any(
            k in scope_q
            for k in ("replot", "plot using", "generate", "create", "draw", "cad", "dwg")
        ):
            return False
        if not any(
            k in scope_q
            for k in (
                "cadastral", "survey plan", "survey/cadastral", "replot",
                "pillar", "bearing", "parcel", ".pdf",
            )
        ):
            return False
        return True

    def _resolve_pdf_path_for_replot(self, query: str) -> Dict[str, Any]:
        """
        Resolve the survey-plan PDF from the *current* user request only.

        Conversation history must not override an explicit path in the latest turn.
        Missing files return similar candidates for user approval — never auto-substitute.
        """
        from agent.pdf_survey_plan import resolve_pdf_path_for_replot

        scope = self._cadastral_user_message_body(query)
        return resolve_pdf_path_for_replot(scope, query)

    def _ensure_autocad_connected(self) -> bool:
        """Connect to AutoCAD early so COM is warm before template plotting."""
        try:
            if self.autocad.is_connected:
                return True
            return bool(self.autocad.connect())
        except Exception as exc:
            logger.debug("AutoCAD pre-connect failed: %s", exc)
            return False

    @staticmethod
    def _scope_requests_cadastral_extras(text: str) -> bool:
        """True when the user turn asks to add/change roads, fences, or road titles."""
        ql = (text or "").lower()
        markers = (
            "add an access", "add access", "add another", "another access",
            "concrete wall", "dwarf concrete", "c.w.f", "d.c.w.f",
            "add fence", "add a fence", "add concrete", "wall fence",
            "title as", "give it the title", "road title",
        )
        return any(m in ql for m in markers)

    def _extract_pdf_survey_plan_with_tier_fallback(
        self,
        pdf_path: str,
        *,
        user_notes: str,
        timeout_s: int = 120,
    ) -> tuple[Any, Optional[str]]:
        """Extract from PDF using average-tier vision first; escalate to complex if needed."""
        from agent.pdf_survey_plan import (
            SurveyPlanExtraction,
            extract_survey_plan_from_pdf,
            validate_extraction_for_replot,
        )

        tiers: List[tuple[str, int]] = [("average", 1), ("complex", 2)]
        last: Optional[SurveyPlanExtraction] = None
        last_model: Optional[str] = None

        for tier, vision_pages in tiers:
            llm, model_name = self._try_openai_tier_llm(tier)  # type: ignore[arg-type]
            if llm is None:
                llm = getattr(self, "llm_primary", None)
            if llm is None:
                break
            if not model_name:
                model_name = getattr(self, "_current_openai_model", None) or getattr(
                    self.settings, "openai_model", "gpt-5.4-mini"
                )
            self._current_openai_model = model_name
            extraction = extract_survey_plan_from_pdf(
                pdf_path,
                llm=llm,
                run_with_timeout=self._llm_run_with_timeout(model_name),
                user_notes=user_notes,
                timeout_s=timeout_s,
                vision_max_pages=vision_pages,
            )
            last = extraction
            last_model = model_name
            if extraction.source in ("error", "llm_parse_failed"):
                continue
            if not validate_extraction_for_replot(extraction):
                return extraction, model_name
            logger.info(
                "PDF extraction incomplete on tier %s — retrying with stronger model if available",
                tier,
            )

        if last is None:
            return SurveyPlanExtraction(source="error", notes="No LLM available for PDF extraction"), None
        return last, last_model

    def _run_pdf_survey_replot_pipeline(self, query: str) -> Dict[str, Any]:
        """Extract survey plan from PDF (layout + vision) and replot via cadastral CAD pipeline."""
        from pathlib import Path

        from agent.pdf_survey_plan import (
            apply_plan_overrides_to_extraction,
            build_cadastral_subprompt,
            filter_user_facing_extraction_notes,
            resolve_output_dwg_path,
            resolve_plan_overrides_from_query,
            validate_extraction_for_replot,
            validate_subprompt_geometry,
        )

        scope = self._cadastral_user_message_body(query)
        pdf_resolution = self._resolve_pdf_path_for_replot(query)
        if not pdf_resolution.get("success"):
            return {
                "success": False,
                "error": pdf_resolution.get("error") or "No PDF path found in the request.",
                "requested_pdf": pdf_resolution.get("requested"),
                "similar_pdfs": pdf_resolution.get("similar") or [],
                "needs_user_approval": bool(pdf_resolution.get("needs_user_approval")),
            }

        pdf_path = str(pdf_resolution.get("path") or "")
        if not pdf_path or not Path(pdf_path).exists():
            return {
                "success": False,
                "error": f"PDF not found: {pdf_path or pdf_resolution.get('requested')}",
            }

        output_dwg = resolve_output_dwg_path(query, pdf_path, scope_text=scope)
        if not output_dwg:
            return {"success": False, "error": "Could not resolve output DWG path."}

        logger.info(
            "PDF survey replot paths (strict): pdf=%s output_dwg=%s scope=%r",
            pdf_path,
            output_dwg,
            scope[:160],
        )

        template_hint: Optional[str] = None
        mem: Optional[Dict[str, str]] = None

        def _resolve_template() -> Optional[Dict[str, str]]:
            return self._resolve_cadastral_template_from_memory(query)

        with ThreadPoolExecutor(max_workers=3) as pool:
            fut_template = pool.submit(_resolve_template)
            fut_simple = pool.submit(self._try_openai_tier_llm, "simple")
            fut_cad = pool.submit(self._ensure_autocad_connected)
            mem = fut_template.result()
            override_llm, override_model = fut_simple.result()
            fut_cad.result()

        extraction, model_name = self._extract_pdf_survey_plan_with_tier_fallback(
            pdf_path,
            user_notes=self._cadastral_user_message_body(query),
            timeout_s=120,
        )

        if override_llm is None:
            override_llm = getattr(self, "llm_primary", None)
        if not override_model:
            override_model = model_name
        if model_name:
            self._current_openai_model = model_name
        if extraction.source in ("error", "llm_parse_failed"):
            return {
                "success": False,
                "error": extraction.notes or "PDF survey plan extraction failed.",
                "extraction": extraction.model_dump(),
            }

        validation_issues = validate_extraction_for_replot(extraction)
        if validation_issues:
            return {
                "success": False,
                "error": (
                    "PDF extraction is incomplete and cannot be replotted safely: "
                    + "; ".join(validation_issues)
                ),
                "validation_issues": validation_issues,
                "extraction": extraction.model_dump(),
            }

        if mem and mem.get("template_path"):
            template_hint = mem["template_path"]

        plan_overrides = resolve_plan_overrides_from_query(
            query,
            scope_text=scope,
            base_extraction=extraction,
            llm=override_llm,
            run_with_timeout=self._llm_run_with_timeout(override_model),
        )
        extraction = apply_plan_overrides_to_extraction(extraction, plan_overrides)
        from agent.pdf_survey_plan import enrich_extraction_coordinates

        extraction = enrich_extraction_coordinates(extraction, "")
        cert_date = plan_overrides.certification_date or extraction.certification_date or None

        subprompt = build_cadastral_subprompt(
            extraction,
            output_dwg_path=output_dwg,
            certification_date=cert_date,
            template_path=template_hint,
        )
        subprompt_issues = validate_subprompt_geometry(subprompt)
        if subprompt_issues:
            return {
                "success": False,
                "error": (
                    "Generated CAD sub-prompt is missing required geometry: "
                    + "; ".join(subprompt_issues)
                ),
                "validation_issues": subprompt_issues,
                "extraction": extraction.model_dump(),
                "subprompt": subprompt,
            }
        logger.info("PDF survey replot sub-prompt:\n%s", subprompt)

        plot_result = self._run_cadastral_cad_prompt_pipeline(
            subprompt,
            source_scale_denom=extraction.scale_denom,
            skip_intent_assessment=True,
            user_scope_for_extras=scope,
        )
        if not plot_result.get("success"):
            return {
                "success": False,
                "error": plot_result.get("error", "Cadastral replot failed."),
                "extraction": extraction.model_dump(),
                "subprompt": subprompt,
            }

        out_lines = [
            "✅ Survey plan PDF replotted to CAD.",
            f"- Source PDF: {pdf_path}",
            f"- Output DWG: {plot_result.get('output_dwg') or output_dwg}",
            f"- Extraction: {extraction.source} (confidence {extraction.confidence:.0%})",
        ]
        if plan_overrides.override_fields:
            out_lines.append(
                f"- User overrides applied: {', '.join(plan_overrides.override_fields)}"
            )
            if extraction.buyer_name and "buyer_name" in plan_overrides.override_fields:
                out_lines.append(f"- Buyer name: {extraction.buyer_name}")
            if extraction.plan_number and "plan_number" in plan_overrides.override_fields:
                out_lines.append(f"- Plan number: {extraction.plan_number}")
            if cert_date and "certification_date" in plan_overrides.override_fields:
                out_lines.append(f"- Certification date: {cert_date}")
        elif cert_date:
            out_lines.append(f"- Certification date updated to: {cert_date}")
        if extraction.pillar_numbers:
            out_lines.append(f"- Pillars: {', '.join(extraction.pillar_numbers)}")
        if extraction.traverse_legs:
            out_lines.append(f"- Traverse legs: {len(extraction.traverse_legs)}")
        if extraction.fences:
            out_lines.append(f"- Concrete wall fences: {len(extraction.fences)} boundary side(s)")
        filtered_notes = filter_user_facing_extraction_notes(
            extraction.notes or "",
            plan_overrides.override_fields,
        )
        if filtered_notes:
            out_lines.append(f"- Notes: {filtered_notes}")

        return {
            "success": True,
            "response": "\n".join(out_lines) + "\n",
            "output_dwg": plot_result.get("output_dwg") or output_dwg,
            "output_path": plot_result.get("output_dwg") or output_dwg,
            "extraction": extraction.model_dump(),
            "model_name": model_name,
        }

    def _should_fastpath_excel_cadastral(self, query: str) -> bool:
        """True when CAD plan coords must be composed from an Excel ownership workbook."""
        body = self._cadastral_user_message_body(query)
        ql = body.lower()
        # Inspect/read/deeper-extract turns must never enter the Excel→DWG pipeline.
        if is_tabular_inspect_only_request(body) or not explicit_cadastral_plot_intent(body):
            return False
        if ".dwg" not in ql:
            return False
        if any(m in ql for m in _CADASTRAL_FASTPATH_EXCLUDE_MARKERS):
            return False
        if not any(k in ql for k in ("generate", "create", "produce", "plot")):
            return False
        if not any(k in ql for k in ("excel", ".xlsx", ".xls", "spreadsheet", "workbook")):
            return False
        # Prefer dedicated Excel path only when Excel is the clear tabular source
        # (not when the user primarily points at CSV/TXT/DOCX).
        if any(k in ql for k in (".csv", "csv file", ".txt", "text file", ".docx", "word document")):
            if not any(k in ql for k in ("excel", ".xlsx", ".xls", "spreadsheet", "workbook")):
                return False
        return coordinates_deferred_to_external_source(body)

    def _should_fastpath_intelligent_cadastral(self, query: str) -> bool:
        """
        Complex / multi-format deferred cadastral composition.

        Simple conventional prompts with inline EmE/NmN or bearings stay on the
        hard-coded CAD fastpath (this returns False when coords are not deferred).
        """
        from agent.cadastral_compose import should_intelligent_cadastral_compose

        body = self._cadastral_user_message_body(query)
        if any(m in body.lower() for m in _CADASTRAL_FASTPATH_EXCLUDE_MARKERS):
            return False
        # Excel-only clear cases keep the cheaper dedicated Excel fastpath.
        if self._should_fastpath_excel_cadastral(query):
            return False
        return should_intelligent_cadastral_compose(body)

    def _run_intelligent_cadastral_pipeline(self, query: str) -> Dict[str, Any]:
        """
        File-deferred / complex cadastral goals:
        discover CSV|TXT|DOCX|Excel → deterministic extract → optional RAG+LLM compose → plot.
        """
        import re
        from pathlib import Path

        from agent.cadastral_compose import (
            build_subprompt_from_coordinates_blob,
            compose_cadastral_from_files,
            store_compose_example,
        )
        from agent.excel_cadastral import build_excel_cadastral_subprompt
        from agent.pdf_survey_plan import extract_plan_details_for_dwg

        scope = self._cadastral_user_message_body(query)
        workspace = Path.cwd().resolve()

        llm, _model = self._try_openai_tier_llm("simple")
        if llm is None:
            llm = getattr(self, "llm_primary", None)

        composed = compose_cadastral_from_files(
            scope,
            workspace=workspace,
            document_processor=getattr(self, "document_processor", None),
            llm=llm,
            run_with_timeout=self._run_with_timeout,
            vector_store=getattr(self, "vector_store", None),
            search_fn=self._vs_search,
            prefer_llm=False,
        )
        if not composed.get("success"):
            return {
                "success": False,
                "error": composed.get("error") or "Intelligent cadastral composition failed.",
                "notes": composed.get("notes") or [],
                "source_files": composed.get("source_files") or [],
            }

        notes = list(composed.get("notes") or [])
        meta = dict(composed.get("metadata") or {})
        ref_dwg = composed.get("reference_dwg")
        if ref_dwg:
            try:
                self._ensure_autocad_connected()
                details = extract_plan_details_for_dwg(
                    self.autocad,
                    getattr(self, "dxf_fallback", None),
                    str(ref_dwg),
                    llm=llm,
                    run_with_timeout=self._run_with_timeout,
                )
                plan = details.get("survey_plan")
                if plan is not None:
                    try:
                        from agent.pdf_survey_plan import (
                            ensure_surveyor_professional_title,
                            normalize_lga_name,
                        )
                    except Exception:
                        ensure_surveyor_professional_title = None  # type: ignore
                        normalize_lga_name = None  # type: ignore
                    for key, attr in (
                        ("location", "location"),
                        ("lga", "lga"),
                        ("state", "state"),
                        ("plan_number", "plan_number"),
                        ("surveyor_name", "surveyor_name"),
                        ("surveyor_address", "surveyor_address"),
                        ("origin_crs", "origin_crs"),
                    ):
                        val = (getattr(plan, attr, None) or "").strip()
                        if key == "lga" and val and normalize_lga_name is not None:
                            val = normalize_lga_name(val) or val
                        if key == "surveyor_name" and val and ensure_surveyor_professional_title is not None:
                            val = ensure_surveyor_professional_title(val) or val
                        if val and not meta.get(key):
                            meta[key] = val
                errs = details.get("errors") or []
                if errs:
                    notes.append("Reference plan warnings: " + "; ".join(str(e) for e in errs[:3]))
            except Exception as exc:
                notes.append(f"Could not fully extract metadata from reference DWG: {exc}")

        # Prompt CRS/date overrides
        m_crs = re.search(
            r"(?:origin_crs|crs_origin)\s*[:=]\s*([^,\n]+)",
            scope,
            flags=re.IGNORECASE,
        )
        if m_crs:
            meta["origin_crs"] = m_crs.group(1).strip().strip("'\"")
        try:
            from agent.pdf_survey_plan import (
                extract_user_requested_certification_date,
                resolve_certification_date_from_query,
            )

            user_cert = (
                extract_user_requested_certification_date(scope)
                or resolve_certification_date_from_query(scope, scope_text=scope)
            )
        except Exception:
            user_cert = None
        if user_cert:
            meta["certification_date"] = user_cert

        # Explicit starting / field plan number in the prompt wins over reference DWG.
        try:
            from agent.pdf_survey_plan import extract_user_requested_plan_number

            user_plan_no = extract_user_requested_plan_number(scope)
        except Exception:
            user_plan_no = None
        if user_plan_no:
            prev = (meta.get("plan_number") or "").strip()
            if prev and prev != user_plan_no:
                notes.append(
                    f"Using user-requested starting plan number {user_plan_no} "
                    f"(overrides reference/composed plan no. {prev})."
                )
            meta["plan_number"] = user_plan_no

        parcels = composed.get("parcels") or []
        output_dwg = composed.get("output_dwg")

        # Separate owner plans (Excel/CSV ownership blocks → one DWG each).
        from agent.excel_cadastral import (
            build_separate_owner_plan_jobs,
            wants_separate_owner_plans,
        )

        if wants_separate_owner_plans(scope) and len(parcels) >= 1:
            try:
                from agent.pdf_survey_plan import (
                    extract_user_requested_scale_denom,
                    scrub_surveyor_metadata_value,
                )

                scope_scale = extract_user_requested_scale_denom(scope)
            except Exception:
                scrub_surveyor_metadata_value = None  # type: ignore
                scope_scale = None
            if not scope_scale:
                try:
                    scope_scale = int(meta.get("scale_denom") or 0) or None
                except Exception:
                    scope_scale = None
            surv_nm = meta.get("surveyor_name", "")
            surv_addr = meta.get("surveyor_address", "")
            if scrub_surveyor_metadata_value is not None:
                try:
                    surv_nm = scrub_surveyor_metadata_value(surv_nm, max_len=100)
                    surv_addr = scrub_surveyor_metadata_value(surv_addr, max_len=200)
                except Exception:
                    pass
            jobs_built = build_separate_owner_plan_jobs(
                parcels=parcels,
                workspace=workspace,
                location=meta.get("location", ""),
                lga=meta.get("lga", ""),
                state=meta.get("state", ""),
                origin_crs=meta.get("origin_crs", "") or "UTM Zone 32N",
                plan_number=meta.get("plan_number", ""),
                surveyor_name=surv_nm,
                surveyor_address=surv_addr,
                certification_date=meta.get("certification_date", ""),
                scale_denom=scope_scale,
            )
            if not jobs_built.get("success"):
                return {
                    "success": False,
                    "error": jobs_built.get("error")
                    or "Failed to compose per-owner cadastral prompts.",
                    "notes": notes,
                    "source_files": composed.get("source_files") or [],
                    "mode": "separate_owner_plans",
                }
            plan_results: List[Dict[str, Any]] = []
            ok = 0
            batch_template: Optional[str] = None
            batch_profile: Optional[str] = None
            try:
                mem = self._resolve_cadastral_template_from_memory(scope)
                if mem:
                    batch_template = mem.get("template_path") or None
                    batch_profile = mem.get("profile_path") or None
            except Exception:
                pass
            jobs_list = list(jobs_built.get("jobs") or [])
            n_jobs = len(jobs_list)
            for i, job in enumerate(jobs_list):
                # True single-parcel route per owner; reuse CAD session/template after the first.
                is_first = i == 0
                is_last = i >= n_jobs - 1
                plot = self._run_cadastral_cad_prompt_pipeline(
                    job["subprompt"],
                    skip_intent_assessment=True,
                    # Parent scope keeps access-road / fence extras from the user prompt.
                    user_scope_for_extras=scope,
                    extra_parcels=[],
                    extent_points=[],
                    main_parcel_label="",
                    skip_session_prep=not is_first,
                    batch_mode=not is_first,
                    template_override_path=batch_template,
                    profile_override_path=batch_profile,
                    # Close finished owner DWGs between plans so Documents does not wedge.
                    close_output_after_save=not is_last,
                    source_scale_denom=job.get("scale_denom") or scope_scale,
                )
                entry = {
                    "owner_name": job.get("owner_name"),
                    "plan_number": job.get("plan_number"),
                    "output_dwg": plot.get("output_dwg") or job.get("output_dwg"),
                    "success": bool(plot.get("success")),
                    "error": plot.get("error"),
                }
                plan_results.append(entry)
                if plot.get("success"):
                    ok += 1
                    self._last_cadastral_output_dwg = plot.get("output_dwg")
                    self._last_cadastral_profile_path = plot.get("profile_path")
                    if plot.get("template_path"):
                        batch_template = str(plot.get("template_path"))
                    if plot.get("profile_path"):
                        batch_profile = str(plot.get("profile_path"))
                    try:
                        store_compose_example(
                            getattr(self, "vector_store", None),
                            query=scope,
                            composed_subprompt=job["subprompt"],
                            source_files=composed.get("source_files") or [],
                        )
                    except Exception:
                        pass
                elif not is_last:
                    # Failed mid-batch: drop any half-open tab and recover COM before next owner.
                    try:
                        out_try = plot.get("output_dwg") or job.get("output_dwg")
                        if out_try and hasattr(self.autocad, "save_and_close_drawing"):
                            self.autocad.save_and_close_drawing(str(out_try), save=False)
                        elif hasattr(self.autocad, "recover_com_session"):
                            self.autocad.recover_com_session(reason="separate-owner plan failed")
                    except Exception:
                        pass

            outputs = [r.get("output_dwg") for r in plan_results if r.get("success")]
            return {
                "success": ok > 0,
                "output_dwg": outputs[-1] if outputs else None,
                "output_dwgs": outputs,
                "plans": plan_results,
                "plans_total": n_jobs,
                "plans_success": ok,
                "plans_failed": n_jobs - ok,
                "dup_xlsx": composed.get("dup_xlsx"),
                "parcel_count": len(parcels),
                "buyer_name": ", ".join(
                    str(j.get("owner_name") or "") for j in jobs_list
                ),
                "reference_dwg": ref_dwg,
                "source_files": composed.get("source_files") or [],
                "compose_source": composed.get("compose_source"),
                "notes": notes,
                "mode": "separate_owner_plans",
                "requested_scale": scope_scale,
                "error": None
                if ok > 0
                else (plan_results[-1].get("error") if plan_results else "No owner plans plotted."),
            }

        if parcels:
            try:
                from agent.pdf_survey_plan import extract_user_requested_scale_denom

                scope_scale = extract_user_requested_scale_denom(scope)
            except Exception:
                scope_scale = None
            rebuilt = build_excel_cadastral_subprompt(
                output_dwg=str(output_dwg),
                parcels=parcels,
                location=meta.get("location", ""),
                lga=meta.get("lga", ""),
                state=meta.get("state", ""),
                origin_crs=meta.get("origin_crs", "") or "UTM Zone 32N",
                plan_number=meta.get("plan_number", ""),
                surveyor_name=meta.get("surveyor_name", ""),
                surveyor_address=meta.get("surveyor_address", ""),
                certification_date=meta.get("certification_date", ""),
                scale_denom=scope_scale,
            )
            if not rebuilt.get("success"):
                return {
                    "success": False,
                    "error": rebuilt.get("error") or "Failed to rebuild cadastral subprompt.",
                    "notes": notes,
                }
            subprompt = rebuilt["subprompt"]
            extra_parcels = rebuilt.get("extra_parcels") or []
            extent_points = rebuilt.get("extent_points") or []
            main_label = rebuilt.get("main_parcel_label") or ""
            buyer_name = rebuilt.get("buyer_name") or composed.get("buyer_name")
        else:
            subprompt = build_subprompt_from_coordinates_blob(
                output_dwg=str(output_dwg),
                coordinates_blob=str(composed.get("coordinates_blob") or ""),
                buyer_name=meta.get("buyer_name") or composed.get("buyer_name") or "Buyer",
                location=meta.get("location", ""),
                lga=meta.get("lga", ""),
                state=meta.get("state", ""),
                origin_crs=meta.get("origin_crs", "") or "UTM Zone 32N",
                plan_number=meta.get("plan_number", ""),
                surveyor_name=meta.get("surveyor_name", ""),
                surveyor_address=meta.get("surveyor_address", ""),
                certification_date=meta.get("certification_date", ""),
                pillar_numbers=meta.get("pillar_numbers") or [],
            )
            extra_parcels = []
            extent_points = []
            main_label = ""
            buyer_name = meta.get("buyer_name") or composed.get("buyer_name")

        plot = self._run_cadastral_cad_prompt_pipeline(
            subprompt,
            skip_intent_assessment=True,
            user_scope_for_extras=scope,
            extra_parcels=extra_parcels,
            extent_points=extent_points,
            main_parcel_label=main_label,
        )
        if not plot.get("success"):
            return {
                "success": False,
                "error": plot.get("error") or "Cadastral plot after intelligent composition failed.",
                "composed_prompt": subprompt,
                "dup_xlsx": composed.get("dup_xlsx"),
                "notes": notes,
                "source_files": composed.get("source_files") or [],
            }

        try:
            store_compose_example(
                getattr(self, "vector_store", None),
                query=scope,
                composed_subprompt=subprompt,
                source_files=composed.get("source_files") or [],
            )
        except Exception:
            pass

        self._last_cadastral_output_dwg = plot.get("output_dwg")
        self._last_cadastral_profile_path = plot.get("profile_path")
        return {
            "success": True,
            "output_dwg": plot.get("output_dwg"),
            "geometry": plot.get("geometry"),
            "profile_path": plot.get("profile_path"),
            "dup_xlsx": composed.get("dup_xlsx"),
            "parcel_count": len(parcels) if parcels else 1,
            "buyer_name": buyer_name,
            "reference_dwg": ref_dwg,
            "source_files": composed.get("source_files") or [],
            "compose_source": composed.get("compose_source"),
            "notes": notes,
            "access_road_title": plot.get("access_road_title"),
            "mode": "intelligent_compose",
        }

    def _run_excel_cadastral_pipeline(self, query: str) -> Dict[str, Any]:
        """
        Excel family/ownership parcels → optional normalized copy (any name the user asks) →
        metadata from reference DWG → cadastral plot(s).

        Semantic layout:
        - Separate owner plans: one DWG per owner (buyer name = filename; plan nos. increment).
        - Multi-parcel (default for multi-owner Excel when not separate): one DWG, all owners.

        Plotting may use the input workbook directly; workbook choice is semantic
        (real owners / family-block layout), not filename.
        """
        import re
        from pathlib import Path

        from agent.excel_cadastral import (
            build_excel_cadastral_subprompt,
            build_separate_owner_plan_jobs,
            find_reference_dwg_from_query,
            resolve_ownership_excel_for_plot,
            wants_separate_owner_plans,
            write_dup_xlsx_with_headers,
        )
        from agent.pdf_survey_plan import extract_plan_details_for_dwg

        scope = self._cadastral_user_message_body(query)
        workspace = Path.cwd().resolve()

        # Hard gate: never invent a CAD drawing from a read/extract/deeper-search turn.
        if not explicit_cadastral_plot_intent(scope):
            return {
                "success": False,
                "error": (
                    "Excel cadastral plotting was refused: the current request does not "
                    "explicitly ask to generate/create/plot a CAD/.dwg plan. "
                    "Continue with spreadsheet read/extract tools (or a deeper extract) instead. "
                    "Only call excel_cadastral_plot after the user clearly requests a DWG/CAD plan."
                ),
                "refused_reason": "missing_explicit_cadastral_plot_intent",
            }

        resolved = resolve_ownership_excel_for_plot(scope, workspace)
        if not resolved.get("success"):
            return {
                "success": False,
                "error": resolved.get("error")
                or (
                    "Could not find a usable Excel ownership workbook in the workspace. "
                    "Place the coordinates file in the active workspace (or quote its path) and retry."
                ),
            }
        excel_path = Path(resolved["path"])
        parcels = list(resolved.get("parcels") or [])

        # Optional normalized copy — only when the user asked (any filename they choose).
        dup_path = None
        if resolved.get("write_copy"):
            dup = write_dup_xlsx_with_headers(
                excel_path,
                dest_name=resolved.get("copy_name"),
                parcels=parcels,
                query=scope,
            )
            if not dup.get("success"):
                return {
                    "success": False,
                    "error": dup.get("error") or "Failed to write the requested Excel copy.",
                }
            dup_path = dup.get("output_path")
            # If the user asked for a copy, prefer plotting from the freshly written names.
            try:
                from agent.excel_cadastral import parse_family_parcels_from_excel

                reparsed = parse_family_parcels_from_excel(dup_path)
                if reparsed.get("success") and reparsed.get("parcels"):
                    parcels = reparsed["parcels"]
            except Exception:
                pass

        # Output DWG: honor Generate … .dwg; otherwise derive from first owner (never invent
        # a fixed brand name such as Excel_Families.dwg).
        out_name = default_excel_plot_output_name(scope, parcels)
        output_dwg = str((workspace / out_name).resolve())

        # Metadata from an existing plan named in the prompt (location/LGA/surveyor/plan no.).
        location = lga = state = plan_number = surveyor_name = surveyor_address = ""
        origin_crs = ""
        m_crs = re.search(
            r"(?:origin_crs|crs_origin)\s*[:=]\s*([^,\n]+)",
            scope,
            flags=re.IGNORECASE,
        )
        if m_crs:
            origin_crs = m_crs.group(1).strip().strip("'\"")
        cert_date = ""
        try:
            from agent.pdf_survey_plan import (
                extract_user_requested_certification_date,
                resolve_certification_date_from_query,
            )

            cert_date = (
                extract_user_requested_certification_date(scope)
                or resolve_certification_date_from_query(scope, scope_text=scope)
                or ""
            )
        except Exception:
            m_cert = re.search(
                r"(?:date\s+on\s+the\s+certification|certification\s+date|\bdate)\s*[:=]\s*"
                r"([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
                scope,
                flags=re.IGNORECASE,
            )
            if m_cert:
                cert_date = m_cert.group(1).strip()

        ref_dwg = find_reference_dwg_from_query(scope, workspace)
        meta_notes: List[str] = []
        if ref_dwg:
            try:
                # Warm AutoCAD COM on this worker thread before reference metadata open.
                # Prevents RPC_E_CALL_REJECTED when the first COM call is Documents.Open.
                self._ensure_autocad_connected()
                llm, _model = self._try_openai_tier_llm("simple")
                details = extract_plan_details_for_dwg(
                    self.autocad,
                    getattr(self, "dxf_fallback", None),
                    str(ref_dwg),
                    llm=llm or getattr(self, "llm_primary", None),
                    run_with_timeout=self._run_with_timeout,
                )
                plan = details.get("survey_plan")
                if plan is not None:
                    location = (getattr(plan, "location", None) or "").strip()
                    try:
                        from agent.pdf_survey_plan import (
                            ensure_surveyor_professional_title,
                            normalize_lga_name,
                        )

                        lga = normalize_lga_name(getattr(plan, "lga", None) or "") or (
                            getattr(plan, "lga", None) or ""
                        ).strip()
                    except Exception:
                        ensure_surveyor_professional_title = None  # type: ignore
                        lga = (getattr(plan, "lga", None) or "").strip()
                    state = (getattr(plan, "state", None) or "").strip()
                    plan_number = (getattr(plan, "plan_number", None) or "").strip()
                    surveyor_name = (getattr(plan, "surveyor_name", None) or "").strip()
                    if surveyor_name and ensure_surveyor_professional_title is not None:
                        try:
                            surveyor_name = ensure_surveyor_professional_title(surveyor_name) or surveyor_name
                        except Exception:
                            pass
                    surveyor_address = (getattr(plan, "surveyor_address", None) or "").strip()
                    try:
                        from agent.pdf_survey_plan import scrub_surveyor_metadata_value

                        surveyor_name = scrub_surveyor_metadata_value(surveyor_name, max_len=100)
                        surveyor_address = scrub_surveyor_metadata_value(
                            surveyor_address, max_len=200
                        )
                    except Exception:
                        pass
                    if not origin_crs:
                        origin_crs = (getattr(plan, "origin_crs", None) or "").strip()
                errs = details.get("errors") or []
                if errs:
                    meta_notes.append("Reference plan warnings: " + "; ".join(str(e) for e in errs[:3]))
            except Exception as exc:
                meta_notes.append(f"Could not fully extract metadata from {ref_dwg.name}: {exc}")
        else:
            meta_notes.append(
                "No reference .dwg found in the workspace for location/surveyor/plan number; "
                "title-block fields from the prompt (if any) will be used."
            )

        # Explicit starting / field plan number in the prompt ALWAYS wins over the
        # reference DWG plan no. (e.g. "start from plan number 'RV/018/2026/SP'").
        # "Take location/surveyor from existing plan" does NOT imply take plan number
        # when the user also stated a start-from / plan-number base.
        try:
            from agent.pdf_survey_plan import extract_user_requested_plan_number

            user_plan_no = extract_user_requested_plan_number(scope)
        except Exception:
            user_plan_no = None
        if user_plan_no:
            if plan_number and plan_number != user_plan_no:
                meta_notes.append(
                    f"Using user-requested starting plan number {user_plan_no} "
                    f"(overrides reference plan no. {plan_number})."
                )
            elif not plan_number:
                meta_notes.append(
                    f"Using user-requested starting plan number {user_plan_no}."
                )
            plan_number = user_plan_no

        # --- Separate owner plans: one DWG per Excel ownership block ---
        if wants_separate_owner_plans(scope) and len(parcels) >= 1:
            try:
                from agent.pdf_survey_plan import extract_user_requested_scale_denom

                scope_scale = extract_user_requested_scale_denom(scope)
            except Exception:
                scope_scale = None
            jobs_built = build_separate_owner_plan_jobs(
                parcels=parcels,
                workspace=workspace,
                location=location,
                lga=lga,
                state=state,
                origin_crs=origin_crs or "UTM Zone 32N",
                plan_number=plan_number,
                surveyor_name=surveyor_name,
                surveyor_address=surveyor_address,
                certification_date=cert_date,
                template_path=None,
                scale_denom=scope_scale,
            )
            if not jobs_built.get("success"):
                return {
                    "success": False,
                    "error": jobs_built.get("error")
                    or "Failed to compose per-owner cadastral prompts from Excel.",
                    "normalized_xlsx": dup_path,
                    "dup_xlsx": dup_path,
                    "notes": meta_notes,
                    "mode": "separate_owner_plans",
                }
            if jobs_built.get("truncated"):
                meta_notes.append(
                    f"Plotted first {jobs_built.get('plan_count')} of "
                    f"{jobs_built.get('usable_parcel_count')} owner parcels "
                    "(separate-plan batch cap)."
                )
            plan_results: List[Dict[str, Any]] = []
            ok = 0
            batch_template: Optional[str] = None
            batch_profile: Optional[str] = None
            try:
                mem = self._resolve_cadastral_template_from_memory(scope)
                if mem:
                    batch_template = mem.get("template_path") or None
                    batch_profile = mem.get("profile_path") or None
            except Exception:
                pass
            jobs_list = list(jobs_built.get("jobs") or [])
            n_jobs = len(jobs_list)
            for i, job in enumerate(jobs_list):
                # True single-parcel route per owner; reuse CAD session/template after the first.
                is_first = i == 0
                is_last = i >= n_jobs - 1
                plot = self._run_cadastral_cad_prompt_pipeline(
                    job["subprompt"],
                    skip_intent_assessment=True,
                    # Parent scope keeps access-road / fence extras from the user prompt.
                    user_scope_for_extras=scope,
                    extra_parcels=[],
                    extent_points=[],
                    main_parcel_label="",
                    skip_session_prep=not is_first,
                    batch_mode=not is_first,
                    template_override_path=batch_template,
                    profile_override_path=batch_profile,
                    # Close finished owner DWGs between plans so Documents does not wedge.
                    close_output_after_save=not is_last,
                    source_scale_denom=job.get("scale_denom") or scope_scale,
                )
                entry = {
                    "owner_name": job.get("owner_name"),
                    "plan_number": job.get("plan_number"),
                    "output_dwg": plot.get("output_dwg") or job.get("output_dwg"),
                    "success": bool(plot.get("success")),
                    "error": plot.get("error"),
                    "cancelled": bool(plot.get("cancelled")),
                    "locked": bool(plot.get("locked")),
                    "geometry": plot.get("geometry"),
                    "profile_path": plot.get("profile_path"),
                    "access_road_title": plot.get("access_road_title"),
                }
                plan_results.append(entry)
                if plot.get("success"):
                    ok += 1
                    self._last_cadastral_output_dwg = plot.get("output_dwg")
                    self._last_cadastral_profile_path = plot.get("profile_path")
                    if plot.get("template_path"):
                        batch_template = str(plot.get("template_path"))
                    if plot.get("profile_path"):
                        batch_profile = str(plot.get("profile_path"))
                elif plot.get("cancelled") or plot.get("locked"):
                    return {
                        "success": False,
                        "cancelled": bool(plot.get("cancelled")),
                        "locked": bool(plot.get("locked")),
                        "error": plot.get("error") or "CAD generation stopped.",
                        "normalized_xlsx": dup_path,
                        "dup_xlsx": dup_path,
                        "plans": plan_results,
                        "plans_success": ok,
                        "plans_total": n_jobs,
                        "parcel_count": len(parcels),
                        "reference_dwg": str(ref_dwg) if ref_dwg else None,
                        "excel_path": str(excel_path),
                        "notes": meta_notes,
                        "mode": "separate_owner_plans",
                    }
                elif not is_last:
                    # Failed mid-batch: drop any half-open tab and recover COM before next owner.
                    try:
                        out_try = plot.get("output_dwg") or job.get("output_dwg")
                        if out_try and hasattr(self.autocad, "save_and_close_drawing"):
                            self.autocad.save_and_close_drawing(str(out_try), save=False)
                        elif hasattr(self.autocad, "recover_com_session"):
                            self.autocad.recover_com_session(reason="separate-owner plan failed")
                    except Exception:
                        pass

            outputs = [r.get("output_dwg") for r in plan_results if r.get("success")]
            return {
                "success": ok > 0,
                "output_dwg": outputs[-1] if outputs else None,
                "output_dwgs": outputs,
                "plans": plan_results,
                "plans_total": n_jobs,
                "plans_success": ok,
                "plans_failed": n_jobs - ok,
                "normalized_xlsx": dup_path,
                "dup_xlsx": dup_path,
                "parcel_count": len(parcels),
                "buyer_name": ", ".join(
                    str(j.get("owner_name") or "") for j in jobs_list
                ),
                "reference_dwg": str(ref_dwg) if ref_dwg else None,
                "excel_path": str(excel_path),
                "notes": meta_notes,
                "mode": "separate_owner_plans",
                "requested_scale": scope_scale,
                "error": None
                if ok > 0
                else (plan_results[-1].get("error") if plan_results else "No owner plans plotted."),
            }

        try:
            from agent.pdf_survey_plan import extract_user_requested_scale_denom

            scope_scale = extract_user_requested_scale_denom(scope)
        except Exception:
            scope_scale = None
        composed = build_excel_cadastral_subprompt(
            output_dwg=output_dwg,
            parcels=parcels,
            location=location,
            lga=lga,
            state=state,
            origin_crs=origin_crs or "UTM Zone 32N",
            plan_number=plan_number,
            surveyor_name=surveyor_name,
            surveyor_address=surveyor_address,
            certification_date=cert_date,
            template_path=None,
            scale_denom=scope_scale,
        )
        if not composed.get("success"):
            return {
                "success": False,
                "error": composed.get("error") or "Failed to compose cadastral prompt from Excel.",
            }

        plot = self._run_cadastral_cad_prompt_pipeline(
            composed["subprompt"],
            skip_intent_assessment=True,
            user_scope_for_extras=scope,
            extra_parcels=composed.get("extra_parcels") or [],
            extent_points=composed.get("extent_points") or [],
            main_parcel_label=composed.get("main_parcel_label") or "",
        )
        if not plot.get("success"):
            return {
                "success": False,
                "cancelled": bool(plot.get("cancelled")),
                "locked": bool(plot.get("locked")),
                "error": plot.get("error") or "Cadastral plot from Excel failed.",
                "normalized_xlsx": dup_path,
                "dup_xlsx": dup_path,
                "composed_prompt": composed.get("subprompt"),
                "notes": meta_notes,
                "mode": "multi_parcel" if len(parcels) > 1 else "single",
            }

        self._last_cadastral_output_dwg = plot.get("output_dwg")
        self._last_cadastral_profile_path = plot.get("profile_path")
        return {
            "success": True,
            "output_dwg": plot.get("output_dwg"),
            "geometry": plot.get("geometry"),
            "profile_path": plot.get("profile_path"),
            "normalized_xlsx": dup_path,
            "dup_xlsx": dup_path,
            "parcel_count": len(parcels),
            "buyer_name": composed.get("buyer_name"),
            "reference_dwg": str(ref_dwg) if ref_dwg else None,
            "excel_path": str(excel_path),
            "notes": meta_notes,
            "access_road_title": plot.get("access_road_title"),
            "mode": "multi_parcel" if len(parcels) > 1 else "single",
        }

    def _should_fastpath_cadastral_cad(self, query: str) -> bool:
        import re

        raw = self._cadastral_user_message_body(query or "")
        q = raw.lower()
        if ".dwg" not in q:
            return False
        if any(m in q for m in _CADASTRAL_FASTPATH_EXCLUDE_MARKERS):
            return False
        # Coordinates live in Excel/files — handled by excel-cadastral fastpath / LLM tools.
        if coordinates_deferred_to_external_source(raw):
            return False

        has_generate = any(k in q for k in ["generate", "create", "produce", "save"])
        if not has_generate:
            return False

        # Cadastral parser cues — avoid matching unrelated prompts where "create" + "(" appears
        # (e.g. UI text) or substrings like "mn" inside common words.
        has_coords = (
            "coordinates for the point" in q
            or "coordinates for the points" in q
            or re.search(r"\bpillar\s+numbers\b", q) is not None
            or (
                "coordinates" in q
                and re.search(r"\(\s*[\d.]+\s*[,;]\s*[\d.]+\s*\)", raw) is not None
            )
            or (
                "coordinates" in q
                and re.search(r"template\s+['\"][^'\"]+?\.dwg['\"]", raw, flags=re.IGNORECASE) is not None
            )
        )
        return bool(has_coords)

    def _should_fastpath_cadastral_template_registration(self, query: str) -> bool:
        """True when the user is only asking SurvyAI to remember a CAD template."""
        q = (query or "").lower()
        if ".dwg" not in q or "template" not in q:
            return False
        if self._should_fastpath_cadastral_cad(query):
            return False
        return any(
            k in q
            for k in (
                "use this",
                "valid cad template",
                "cad template",
                "remember",
                "learn",
                "use as",
            )
        )

    def _run_cadastral_template_registration_pipeline(self, query: str) -> Dict[str, Any]:
        """Learn and remember a cadastral CAD template without invoking the LLM."""
        from pathlib import Path

        template_path = self._extract_dwg_path_from_query(query)
        if not template_path:
            return {"success": False, "error": "No .dwg template path was found in the request."}
        template_p = Path(template_path).resolve()
        if not template_p.exists():
            return {"success": False, "error": f"Template DWG not found: {template_p}"}

        profile_path = (self._cad_template_profiles_dir() / f"{template_p.stem}.json").resolve()
        learned = self._learn_cadastral_template_profile(
            str(template_p),
            profile_output=str(profile_path),
        )
        if not learned.get("success"):
            return learned

        self._register_cad_template_memory(str(template_p), str(profile_path))
        return {
            "success": True,
            "response": (
                "✅ CAD template learned and remembered.\n"
                f"- Template: {template_p}\n"
                f"- Profile: {profile_path}\n\n"
                "You can now generate cadastral plans without repeating the template path."
            ),
            "template_path": str(template_p),
            "profile_path": str(profile_path),
        }

    def _should_fastpath_cadastral_cad_batch(self, query: str) -> bool:
        """
        True when the user asks to plot multiple cadastral plans in one request.

        Supported patterns:
        - "Plot up to 10 plans..." with per-plan blocks like:
            Plan 1: generate 'a.dwg' ... coordinates for the points = ...
            Plan 2: generate 'b.dwg' ... coordinates for the points = ...
        - Or multiple "generate '...dwg'" directives in one prompt.
        """
        import re
        from pathlib import Path

        body = self._cadastral_user_message_body(query)
        ql = body.lower()
        if ".dwg" not in ql:
            return False
        if any(m in ql for m in _CADASTRAL_FASTPATH_EXCLUDE_MARKERS):
            return False
        if coordinates_deferred_to_external_source(body):
            return False
        if "coordinates" not in ql:
            return False

        # Either explicit "plan 1/2/..." blocks, or multiple *distinct* output .dwg basenames.
        has_blocks = bool(re.search(r"(?:^|\n)\s*(?:plan|plot)\s*#?\s*\d+\s*[:\-]", ql))
        gen_re = re.compile(
            r"(?is)\b(?:generate|create|produce)\s*[-]?\s*"
            r"(?:cad\s+drawing\s+|cad\s+|drawing\s+|file\s+)?['\"]?"
            r"([^'\"\s]+?\.dwg)"
            r"['\"]?"
        )
        distinct_out: List[str] = []
        seen: set[str] = set()
        for m in gen_re.finditer(body):
            bn = Path(m.group(1)).name.lower()
            if bn not in seen:
                seen.add(bn)
                distinct_out.append(bn)
        return bool(has_blocks or len(distinct_out) >= 2)

    def _split_cadastral_batch_requests(self, query: str) -> List[str]:
        """
        Split a batch cadastral plotting request into per-plan sub-prompts.

        Rules:
        - Max 10 plans.
        - A "global" template at the top is inherited by any plan block that omits a template.
        - Each plan must contain a generate/output .dwg and coordinates.
        - **One run per distinct output .dwg name** (avoids duplicate runs from continuation history).
        """
        import re
        from pathlib import Path

        q = self._cadastral_user_message_body(query)

        # Extract global template (if provided once).
        m_tpl = re.search(r"template\s+['\"]([^'\"]+?\.dwg)['\"]", q, flags=re.IGNORECASE | re.DOTALL)
        global_template = (m_tpl.group(1).strip() if m_tpl else "")

        gen_re = re.compile(
            r"(?is)\b(?:generate|create|produce)\s*[-]?\s*"
            r"(?:cad\s+drawing\s+|cad\s+|drawing\s+|file\s+)?['\"]?"
            r"([^'\"\s]+?\.dwg)"
            r"['\"]?"
        )

        raw_blocks: List[str] = []

        # Prefer explicit blocks: "Plan 1: ..." / "Plot 2 - ..."
        parts = re.split(r"(?:^|\n)\s*(?:plan|plot)\s*#?\s*\d+\s*[:\-]\s*", q, flags=re.IGNORECASE)
        plan_blocks = [p.strip() for p in parts[1:]] if len(parts) > 1 else []

        if plan_blocks:
            raw_blocks = [b for b in plan_blocks if b.strip()]
        else:
            matches = list(gen_re.finditer(q))
            for i, m in enumerate(matches):
                end = matches[i + 1].start() if i + 1 < len(matches) else len(q)
                chunk = q[m.start() : end].strip()
                if chunk:
                    raw_blocks.append(chunk)

        # Deduplicate by output basename (keep longest chunk per file — robust to repeated lines).
        by_name: Dict[str, List[str]] = {}
        order: List[str] = []
        for blk in raw_blocks:
            m = gen_re.search(blk)
            if not m:
                continue
            bn = Path(m.group(1)).name.lower()
            if bn not in by_name:
                order.append(bn)
            by_name.setdefault(bn, []).append(blk)

        sub_prompts: List[str] = []
        for bn in order:
            if len(sub_prompts) >= 10:
                break
            candidates = by_name.get(bn) or []
            if not candidates:
                continue
            blk = max(candidates, key=len)
            blk_l = blk.lower()
            if ".dwg" not in blk_l or "coordinates" not in blk_l:
                continue

            # If the block doesn't specify template, inherit global template.
            if global_template and not re.search(
                r"template\s+['\"]([^'\"]+?\.dwg)['\"]", blk, flags=re.IGNORECASE | re.DOTALL
            ):
                sub = f"template '{global_template}'\n{blk}"
            else:
                sub = blk

            if not gen_re.search(sub):
                continue
            sub_prompts.append(sub.strip())

        return sub_prompts

    def _run_cadastral_cad_batch_pipeline(self, query: str) -> Dict[str, Any]:
        """
        Run up to 10 cadastral plan plots in one request by reusing the existing
        single-plan deterministic pipeline.
        """
        subs = self._split_cadastral_batch_requests(query)
        if not subs:
            return {
                "success": False,
                "error": (
                    "Could not parse multiple plan requests. Provide per-plan blocks like:\n"
                    "Plan 1: template '...dwg' generate 'out1.dwg' ... coordinates for the points = ...\n"
                    "Plan 2: generate 'out2.dwg' ... coordinates for the points = ...\n"
                    "(You may specify the template once at the top and omit it in later plans.)"
                ),
            }

        results: List[Dict[str, Any]] = []
        ok = 0
        for i, sub in enumerate(subs, start=1):
            try:
                r = self._run_cadastral_cad_prompt_pipeline(sub)
            except Exception as e:
                r = {"success": False, "error": f"Unhandled exception: {type(e).__name__}: {e}"}
            r["_plan_index"] = i
            results.append(r)
            if r.get("success"):
                ok += 1
                # Keep last successful plan for in-session modifications.
                self._last_cadastral_output_dwg = r.get("output_dwg")
                self._last_cadastral_profile_path = r.get("profile_path")

        return {
            "success": ok > 0,
            "plans_total": len(subs),
            "plans_success": ok,
            "plans_failed": len(subs) - ok,
            "results": results,
        }

    def _enrich_cadastral_extras_with_intent_assessment(
        self,
        query: str,
        *,
        access_roads: List[str],
        fences: List[Dict[str, str]],
        pillar_list: List[str],
    ) -> tuple[List[str], List[Dict[str, str]], Optional[str]]:
        """
        Use vector-store recall + a cheap LLM pass to interpret plan extras
        (access roads, fences) from natural language. Falls back to regex-only
        results when assessment is disabled or unavailable.
        """
        if not getattr(self.settings, "cadastral_intent_assessment_enabled", True):
            return access_roads, fences, None

        try:
            from agent.cadastral_intent import (
                assess_cadastral_plan_extras,
                merge_access_roads,
                merge_fences,
            )
        except Exception as exc:
            logger.info("Cadastral intent module unavailable: %s", exc)
            return access_roads, fences, None

        llm = None
        try:
            if self.settings.primary_llm == "openai" and getattr(
                self.settings, "enable_tiered_models", True
            ):
                model = self._get_openai_model_for_complexity("simple")
                llm = self._initialize_llm("openai", model_name=model)
        except Exception:
            llm = None
        if llm is None:
            llm = getattr(self, "llm_primary", None)

        threshold = float(getattr(self.settings, "context_score_threshold", 0.3))
        assessment = assess_cadastral_plan_extras(
            query,
            pillar_numbers=pillar_list,
            vector_store=self.vector_store,
            search_fn=self._vs_search,
            llm=llm,
            run_with_timeout=self._run_with_timeout,
            score_threshold=threshold,
        )

        if assessment.source in ("unavailable", "error", "llm_parse_failed", "none"):
            logger.info(
                "Cadastral extras assessment skipped (%s): %s",
                assessment.source,
                assessment.notes or "no notes",
            )
            return access_roads, fences, None

        merged_roads = merge_access_roads(
            access_roads,
            assessment.access_roads,
            confidence=assessment.confidence,
        )
        merged_fences = merge_fences(
            fences,
            assessment.fences,
            confidence=assessment.confidence,
            query=query,
        )

        if len(merged_roads) != len(access_roads) or len(merged_fences) != len(fences):
            logger.info(
                "Cadastral intent assessment (%s, confidence=%.2f) adjusted extras: "
                "roads %s→%s, fences %s→%s",
                assessment.source,
                assessment.confidence,
                len(access_roads),
                len(merged_roads),
                len(fences),
                len(merged_fences),
            )

        return merged_roads, merged_fences, assessment.access_road_title

    def _run_cadastral_cad_prompt_pipeline(
        self,
        query: str,
        *,
        source_scale_denom: Optional[int] = None,
        skip_intent_assessment: bool = False,
        user_scope_for_extras: Optional[str] = None,
        extra_parcels: Optional[List[Dict[str, Any]]] = None,
        extent_points: Optional[List[Dict[str, Any]]] = None,
        main_parcel_label: Optional[str] = None,
        skip_session_prep: bool = False,
        batch_mode: bool = False,
        template_override_path: Optional[str] = None,
        profile_override_path: Optional[str] = None,
        close_output_after_save: bool = False,
    ) -> Dict[str, Any]:
        """
        Deterministic parser for your cadastral prompts.
        - Learns a template profile if missing
        - Applies the template and replots parcel + bearings/dist + tables
        - ABSOLUTE: The survey template DWG is never modified; all edits are to a copy (output drawing).
        - Access road / fence extras: regex baseline plus optional vector-store + LLM assessment
          (see cadastral_intent_assessment_enabled) so varied prompt styles still plot correctly.
        - Optional multi-parcel layouts: ``extra_parcels`` + ``main_parcel_label`` drawn in local
          space (unique pegs, deduped traverse edges, owner labels) before the UTM move;
          ``extent_points`` enlarge scale fit. Legacy post-move overlay is a fallback only.
        """
        import re
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path

        cad_pool = ThreadPoolExecutor(max_workers=1)
        cad_future = cad_pool.submit(self._ensure_autocad_connected)

        q = self._cadastral_user_message_body(query or "")
        extra_parcels = list(extra_parcels or [])
        extent_points = list(extent_points or [])
        main_parcel_label = (main_parcel_label or "").strip()
        close_output_after_save = bool(close_output_after_save)

        simple_override_llm: Any = None
        simple_override_model: Optional[str] = None

        def _simple_override_llm() -> Any:
            nonlocal simple_override_llm, simple_override_model
            if simple_override_llm is not None:
                return simple_override_llm
            llm_inst, model_inst = self._try_openai_tier_llm("simple")
            simple_override_llm = llm_inst or getattr(self, "llm_primary", None)
            simple_override_model = model_inst
            return simple_override_llm

        def _pick(pats: list[str]) -> Optional[str]:
            for pat in pats:
                m = re.search(pat, q, flags=re.IGNORECASE | re.DOTALL)
                if m:
                    return (m.group(1) or "").strip()
            return None

        def _quoted_list(after_key_pat: str) -> list[str]:
            m = re.search(after_key_pat, q, flags=re.IGNORECASE | re.DOTALL)
            if not m:
                return []
            tail = m.group(1)
            toks = re.findall(r"'([^']+)'", tail)
            return [t.strip() for t in toks if t.strip()]

        template = _pick([r"template\s+'([^']+?\.dwg)'", r"template\s+\"([^\"]+?\.dwg)\""])
        # Support both quoted and unquoted output patterns:
        # - "Generate 'Check20.dwg' ..."
        # - "Generate Check20.dwg ..."
        # - "Generate- Check18.dwg ..."
        output = _pick([
            r"(?:generate|create|produce)\s*[-]?\s+(?:cad\s+drawing\s+|cad\s+|drawing\s+|file\s+)?'([^']+?\.dwg)'",
            r'(?:generate|create|produce)\s*[-]?\s+(?:cad\s+drawing\s+|cad\s+|drawing\s+|file\s+)?"([^"]+?\.dwg)"',
            r"(?:generate|create|produce)\s*[-]?\s+(?:cad\s+drawing\s+|cad\s+|drawing\s+|file\s+)?([^\s'\"\,]+?\.dwg)",
            r"(?:generate|create|produce)\s+'([^']+?\.dwg)'",
            r'(?:generate|create|produce)\s+"([^"]+?\.dwg)"',
        ])

        # Quoted first, then flexible ':' / '=' and unquoted values (comma-delimited fields as in real prompts).
        buyer = _pick(
            [
                r"buyer\s*'?s?\s*name\s*[:=]\s*'([^']+)'",
                r'buyer\s*\'?s?\s*name\s*[:=]\s*"([^"]+)"',
                rf"buyer\s*'?s?\s*name\s*[:=]\s*(.+?){_CADASTRAL_NEXT_FIELD}",
                r"buyer\s*'?s?\s*name\s+should\s+(?:now\s+)?be\s+['\"]([^'\"]+)['\"]",
                r"owner\s*'?s?\s*name\s+should\s+(?:now\s+)?be\s+['\"]([^'\"]+)['\"]",
                r"(?:change|update|set)\s+(?:the\s+)?(?:buyer|owner)(?:'s)?\s*name\s+to\s+['\"]([^'\"]+)['\"]",
            ]
        )
        if not buyer:
            try:
                from agent.pdf_survey_plan import resolve_buyer_name_from_query

                buyer = resolve_buyer_name_from_query(
                    q,
                    scope_text=q,
                    llm=_simple_override_llm(),
                    run_with_timeout=self._run_with_timeout,
                )
            except Exception:
                buyer = None
        location = _pick(
            [
                r"location\s*[:=]\s*'([^']+)'",
                r'location\s*[:=]\s*"([^"]+)"',
                rf"location\s*[:=]\s*(.+?){_CADASTRAL_NEXT_FIELD}",
            ]
        )
        lga = _pick(
            [
                r"local\s+(?:govt\.?|government)\s+area\s*[:=]\s*'([^']+)'",
                r'local\s+(?:govt\.?|government)\s+area\s*[:=]\s*"([^"]+)"',
                rf"local\s+(?:govt\.?|government)\s+area\s*[:=]\s*(.+?){_CADASTRAL_NEXT_FIELD}",
            ]
        )
        state = _pick(
            [
                r"state\s*[:=]\s*'([^']+)'",
                r'state\s*[:=]\s*"([^"]+)"',
                rf"state\s*[:=]\s*(.+?){_CADASTRAL_NEXT_FIELD}",
            ]
        )
        origin = _pick(
            [
                r"(?:crs_?origin|origin_?crs|origin(?:_crs|/crs)?)\s*[:=]\s*'([^']+)'",
                r'(?:crs_?origin|origin_?crs|origin(?:_crs|/crs)?)\s*[:=]\s*"([^"]+)"',
                rf"(?:crs_?origin|origin_?crs)\s*[:=]\s*(.+?){_CADASTRAL_NEXT_FIELD}",
            ]
        )
        plan_no = _pick(
            [
                r"plan\s*(?:no\.?|number)\s*[:=]\s*'([^']+)'",
                r'plan\s*(?:no\.?|number)\s*[:=]\s*"([^"]+)"',
                r"plan\s*(?:no\.?|number)\s*[:=]\s*([A-Z0-9][A-Z0-9/\-]+)\s*(?=\n|\r|$|\s+Surveyor|\s*,\s*date\s+on)",
            ]
        )
        if plan_no:
            try:
                from agent.pdf_survey_plan import normalize_plan_number

                plan_no = normalize_plan_number(plan_no.split("\n")[0].strip())
            except Exception:
                plan_no = plan_no.split("\n")[0].strip()
        cert_date = _pick(
            [
                r"date\s+on\s+the\s+certification\s*[:=]\s*'([^']+)'",
                r'date\s+on\s+the\s+certification\s*[:=]\s*"([^"]+)"',
                r"date\s+on\s+the\s+certification\s*[:=]\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
                r"(?:^|[\n,;])\s*date\s*[:=]\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
                r"\bdate\s*[:=]\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
                r"certification\s+date\s*[:=]\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
            ]
        )
        if not cert_date:
            try:
                from agent.pdf_survey_plan import (
                    extract_user_requested_certification_date,
                    resolve_plan_overrides_from_query,
                )

                cert_date = extract_user_requested_certification_date(q)
                if not cert_date:
                    cert_date = resolve_plan_overrides_from_query(
                        q,
                        scope_text=q,
                        llm=_simple_override_llm(),
                        run_with_timeout=self._run_with_timeout,
                    ).certification_date
            except Exception:
                cert_date = None
        surveyor = _pick(
            [
                r"surveyor\s+name\s*[:=]\s*'([^']+)'",
                r'surveyor\s+name\s*[:=]\s*"([^"]+)"',
                rf"surveyor\s+name\s*[:=]\s*(.+?){_CADASTRAL_NEXT_FIELD}",
            ]
        )
        surveyor_addr = _pick(
            [
                r"surveyor\s+company\s+and\s+address\s*[:=]\s*'([^']+)'",
                r'surveyor\s+company\s+and\s+address\s*[:=]\s*"([^"]+)"',
                rf"surveyor\s+company\s+and\s+address\s*[:=]\s*(.+?){_CADASTRAL_NEXT_FIELD}",
            ]
        )

        # Stop before "coordinates for the point" or "points" (users vary wording).
        # Support both quoted and unquoted pillar lists:
        # - pillar numbers = 'SC/BE 6060, SC/BG 1665, ...'
        # - pillar numbers: SC/BE 6060, SC/BG 1665, ...
        pillar_list = _quoted_list(
            rf"pillar\s+numbers\s*[:=]\s*(.*?)(?={_COORDINATES_FOR_STOP}|$)"
        )
        if not pillar_list:
            m_p = re.search(
                rf"pillar\s+numbers\s*[:=]\s*(.*?)(?={_COORDINATES_FOR_STOP}|$)",
                q,
                flags=re.IGNORECASE | re.DOTALL,
            )
            raw = (m_p.group(1).strip() if m_p else "").strip().rstrip(",").strip()
            pillar_list = [p.strip().strip("'\"") for p in re.split(r"[,\n]+", raw) if p.strip()]
        pillars = ", ".join(pillar_list)

        coords_blob = extract_coordinates_blob_from_cadastral_query(q)
        if not coords_blob:
            coords_blob = resolve_cadastral_coordinates_blob(
                q,
                pillar_list=pillar_list,
                llm=_simple_override_llm(),
                run_with_timeout=self._run_with_timeout,
                vector_store=self.vector_store,
                search_fn=self._vs_search,
            )
        if not coords_blob and re.search(
            r"coordinates\s+for\s+the\s+points?\b|(?:plot|draw|use)\s+(?:using\s+)?coordinate[s]?\b",
            q,
            re.IGNORECASE,
        ):
            cad_pool.shutdown(wait=False)
            return {
                "success": False,
                "error": (
                    "Could not parse coordinates/traverse from this plan block. "
                    "Use 'coordinates for the point: (EmE, NmN), with bearing ...' or "
                    "'coordinates for the point SC/XX 1234: (EmE, NmN), with bearing ...'."
                ),
            }

        # Parse access road(s):
        access_roads: List[str] = _parse_access_road_specs_from_query(q)

        # Parse Concrete Wall Fence / Dwarf Concrete Wall Fence requests (C.W.F / D.C.W.F)
        # Supports multiple fences across different traverse legs, but max 1 fence per leg.
        from agent.pdf_survey_plan import parse_fence_specs_from_text

        fences: List[Dict[str, str]] = parse_fence_specs_from_text(q)

        extras_scope = (user_scope_for_extras or q).strip()
        extras_requested = self._scope_requests_cadastral_extras(extras_scope)
        # Keep normal cadastral plotting deterministic and fast when regex already
        # extracted roads/fences. Run semantic assessment when skipping is disabled
        # (direct cadastral prompts) OR when the user scope still asks for extras
        # (e.g. PDF replot sub-prompts that omit natural-language road/fence text).
        run_intent_assessment = (
            getattr(self.settings, "cadastral_intent_assessment_enabled", True)
            and not (access_roads or fences)
            and (not skip_intent_assessment or extras_requested)
        )
        intent_title: Optional[str] = None
        if run_intent_assessment:
            access_roads, fences, intent_title = self._enrich_cadastral_extras_with_intent_assessment(
                extras_scope or q,
                access_roads=access_roads,
                fences=fences,
                pillar_list=pillar_list,
            )
        else:
            logger.info(
                "Cadastral intent assessment skipped "
                "(regex extras sufficient, assessment disabled, or skipped with no extras in user scope)"
            )

        cad_future.result()
        cad_pool.shutdown(wait=False)

        # Parse user-requested plot scale (honour explicit request; fit may coarsen later).
        # Prefer the composed cadastral prompt, then the parent user scope (Excel/batch),
        # then a PDF/source extraction — never let source overwrite an explicit prompt scale.
        user_scale_denom = None
        try:
            from agent.pdf_survey_plan import extract_user_requested_scale_denom

            user_scale_denom = extract_user_requested_scale_denom(q)
            if not user_scale_denom and user_scope_for_extras:
                user_scale_denom = extract_user_requested_scale_denom(user_scope_for_extras)
        except Exception:
            user_scale_denom = None
            scale_m = re.search(r"plot\s+using\s+scale\s+1\s*:\s*(\d+)", q, re.IGNORECASE)
            if scale_m:
                user_scale_denom = int(scale_m.group(1))
            if not user_scale_denom:
                scale_m = re.search(r"\bscale\s*[:=]?\s*1\s*:\s*(\d+)", q, re.IGNORECASE)
                if scale_m:
                    user_scale_denom = int(scale_m.group(1))
        if not user_scale_denom and source_scale_denom and int(source_scale_denom) > 0:
            user_scale_denom = int(source_scale_denom)

        # Parse optional road title override for first road (e.g. "title as 'UMUAKURU-UMUALILI ROAD'")
        access_road_title = None
        if access_roads:
            access_road_title = intent_title or _pick([
                r"(?:road\s+)?title\s+as\s+['\"]([^'\"]+)['\"]",
                r"(?:give\s+it\s+the\s+)?title\s+as\s+['\"]([^'\"]+)['\"]",
                r"title\s+['\"]([^'\"]+)['\"]",
                r"(?:labeled|named)\s+['\"]([^'\"]+)['\"]",
                r"road\s+title\s+['\"]([^'\"]+)['\"]",
            ])
            if access_road_title:
                access_road_title = access_road_title.strip()

        if buyer:
            try:
                from agent.pdf_survey_plan import trim_metadata_field

                # Multi-owner Excel layouts: "AMADI (A), ODUOHA (B), ..." need a longer budget.
                buyer_max = 900 if (buyer.count(",") >= 2 or re.search(r"\([A-Z]{1,3}\)", buyer)) else 140
                buyer = trim_metadata_field(buyer, max_len=buyer_max)
            except Exception:
                buyer = buyer.split("\n")[0].strip()[:900]
        if location:
            try:
                from agent.pdf_survey_plan import trim_metadata_field

                location = trim_metadata_field(location, max_len=200)
            except Exception:
                location = location.split("\n")[0].strip()[:200]
        if lga:
            try:
                from agent.pdf_survey_plan import normalize_lga_name, trim_metadata_field

                lga = normalize_lga_name(lga) or trim_metadata_field(lga, max_len=80)
                lga = trim_metadata_field(lga, max_len=80)
            except Exception:
                lga = lga.split("\n")[0].strip()[:80]
        if state:
            try:
                from agent.pdf_survey_plan import trim_metadata_field

                state = trim_metadata_field(state, max_len=40)
            except Exception:
                state = state.split("\n")[0].strip()[:40]
        if origin:
            try:
                from agent.pdf_survey_plan import trim_metadata_field

                origin = trim_metadata_field(origin, max_len=60)
            except Exception:
                origin = origin.split("\n")[0].strip()[:60]
        if surveyor:
            try:
                from agent.pdf_survey_plan import (
                    ensure_surveyor_professional_title,
                    scrub_surveyor_metadata_value,
                )

                surveyor = ensure_surveyor_professional_title(
                    scrub_surveyor_metadata_value(surveyor, max_len=100)
                )
            except Exception:
                surveyor = (surveyor or "").strip()[:100]
        if surveyor_addr:
            try:
                from agent.pdf_survey_plan import scrub_surveyor_metadata_value

                surveyor_addr = scrub_surveyor_metadata_value(surveyor_addr, max_len=200)
            except Exception:
                surveyor_addr = (surveyor_addr or "").strip().split("\n")[0][:200]

        resolved_from_memory = None
        if template_override_path:
            template = str(template_override_path)
        if not template:
            resolved_from_memory = self._resolve_cadastral_template_from_memory(q)
            if resolved_from_memory:
                template = resolved_from_memory.get("template_path")

        if not output:
            return {"success": False, "error": "Could not parse output DWG from prompt."}
        if not template:
            return {
                "success": False,
                "error": (
                    "No CAD template was provided and no valid remembered cadastral template is available on this system. "
                    "Please provide a template DWG path once so SurvyAI can learn and remember it."
                ),
            }

        template_p = Path(template)
        if not template_p.is_absolute():
            template_p = (Path.cwd() / template_p).resolve()
        if not template_p.exists():
            return {"success": False, "error": f"Template DWG not found: {str(template_p)}"}
        out_p = self._resolve_user_output_path(q, output or "")

        profile_dir = self._cad_template_profiles_dir()
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_path = None
        if profile_override_path:
            try:
                p_ov = Path(str(profile_override_path)).resolve()
                if p_ov.exists():
                    profile_path = p_ov
            except Exception:
                profile_path = None
        if profile_path is None and resolved_from_memory:
            try:
                p_mem = Path(str(resolved_from_memory.get("profile_path") or "")).resolve()
                if p_mem.exists():
                    profile_path = p_mem
            except Exception:
                profile_path = None
        if profile_path is None:
            profile_path = (profile_dir / f"{template_p.stem}.json").resolve()
        # (Re)learn profile when missing OR when template file changed (mtime/size) OR when profile points to a different template path.
        need_learn = not profile_path.exists()
        if not need_learn:
            try:
                import json as _json

                data = _json.loads(profile_path.read_text(encoding="utf-8"))
                prof_tp = str((data.get("template") or {}).get("path") or "")
                try:
                    resolved_prof = self._resolve_cad_template_file_path(
                        prof_tp,
                        profile_file=profile_path,
                    )
                    prof_tp_res = str(resolved_prof.resolve()) if resolved_prof is not None else ""
                except Exception:
                    prof_tp_res = ""
                cur_tp_res = str(template_p.resolve())
                sig = (data.get("template") or {}).get("signature") or {}
                cur_stat = template_p.stat()
                if prof_tp_res and prof_tp_res != cur_tp_res:
                    need_learn = True
                elif int(sig.get("size_bytes") or -1) != int(cur_stat.st_size):
                    need_learn = True
                elif int(sig.get("mtime_ns") or -1) != int(getattr(cur_stat, "st_mtime_ns", int(cur_stat.st_mtime * 1e9))):
                    need_learn = True
            except Exception:
                # If profile is corrupt/unreadable, re-learn safely.
                need_learn = True

        if need_learn:
            learned = self._learn_cadastral_template_profile(str(template_p), profile_output=str(profile_path))
            if not learned.get("success"):
                return learned

        result = self._apply_cadastral_template(
            profile_path=str(profile_path),
            template_override_path=str(template_p),
            output_dwg_path=str(out_p),
            buyer_name=buyer or "",
            location=location or "",
            lga=lga or "",
            state=state or "",
            origin_crs=origin or "",
            plan_number=plan_no or "",
            surveyor_name=surveyor or "",
            surveyor_company_address=surveyor_addr or "",
            pillar_numbers=pillars or "",
            coordinates=coords_blob,
            certification_date=cert_date,
            access_roads=access_roads,
            fences=fences,
            access_road_title=access_road_title,
            user_scale_denom=user_scale_denom,
            extra_parcels=extra_parcels,
            extent_points=extent_points,
            main_parcel_label=main_parcel_label,
            skip_session_prep=bool(skip_session_prep),
            batch_mode=bool(batch_mode),
            close_output_after_save=bool(close_output_after_save),
        )
        if isinstance(result, dict) and result.get("success"):
            self._register_cad_template_memory(str(template_p), str(profile_path))
            try:
                from agent.cadastral_intent import store_cadastral_plan_extras

                store_cadastral_plan_extras(
                    self.vector_store,
                    query=q,
                    output_dwg=str(out_p),
                    access_roads=access_roads,
                    fences=fences,
                    pillar_numbers=pillars or "",
                )
            except Exception:
                pass
        return result

    def _learn_cadastral_template_profile(
        self,
        template_path: str,
        profile_output: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Template learner: store key table handles + basic layer/layout metadata from the template.
        """
        from pathlib import Path
        import json as _json
        import time

        tp = Path(template_path).resolve()
        if not tp.exists():
            return {"success": False, "error": f"Template not found: {str(tp)}"}
        # Default to the frozen-safe, user-writable profiles dir (%APPDATA%\SurvyAI)
        # instead of a CWD-relative path, so learned profiles persist in packaged builds.
        outp = (
            Path(profile_output).resolve()
            if profile_output
            else (self._cad_template_profiles_dir() / f"{tp.stem}.json").resolve()
        )
        outp.parent.mkdir(parents=True, exist_ok=True)

        if not self.autocad.is_connected and not self.autocad.connect():
            return {"success": False, "error": "Could not connect to AutoCAD via COM"}
        # Template is always opened read-only; the survey template must never be tampered with.
        opened = self.autocad.open_drawing(str(tp), read_only=True)
        if not opened.get("success"):
            return {"success": False, "error": opened.get("error", "Failed to open template")}

        drawing_info = {}
        try:
            drawing_info = self.autocad.get_drawing_info() or {}
        except Exception:
            drawing_info = {}

        tables = self.autocad.list_tables().get("tables", [])
        by_layer = {}
        for t in tables:
            by_layer.setdefault(str(t.get("layer")), []).append(t)

        # Identify coordinate tables by cell suffix
        def _cell(handle: str) -> str:
            r = self.autocad.get_table_cell_text(handle, 0, 0)
            return str(r.get("text") or "") if r.get("success") else ""

        coord_handles = []
        for lyr in ["CADA_COORDINATES", "CADA_NORTHCOORDINATES", "CADA_EASTCOORDINATES"]:
            coord_handles.extend([str(t.get("handle")) for t in by_layer.get(lyr, []) if t.get("handle")])
        
        e_h = n_h = None
        for h in coord_handles:
            txt = _cell(h).upper()
            if ".E" in txt:
                e_h = h
            if ".N" in txt:
                n_h = h

        inserts = self.autocad.list_inserts(layer="CADA_PILLARS").get("inserts", [])
        block_name = None
        if inserts:
            block_name = inserts[0].get("block_name")

        # Sample text height from bearing/distance and road layers for template-matching
        bearing_road_height = 1.2
        try:
            hr = self.autocad.get_sample_text_height(layers=["CADA_BEARING_DIST", "CADA_ROAD"])
            if hr.get("success") and "height" in hr:
                bearing_road_height = float(hr["height"])
        except Exception:
            pass

        # "Sheet" layers that should move together as one unit when recentring the plan.
        # This must include the border/boxes and any sheet text so tables stay inside their boxes.
        # Geometry layers (boundary/bearing/pegs) are intentionally excluded.
        # CADA_SCALEBAR: scale bar (and any hatching/hashing in it, e.g. survey_plan_template2.dwg) is taken from the template and scaled with the sheet.
        layers_in_template = [str(x) for x in (drawing_info.get("layers") or []) if str(x)]
        sheet_layers_default = [
            "CADA_BORDER",
            "CADA_INTERIORBORDER",
            "CADA_SCALEBAR",
            "CADA_NORTHARROW",
            "CADA_EASTARROW",
            "CADA_TITLEBLOCK",
            "CADA_PLANNUMBER",
            "CADA_CERTIFICATION",
            "CADA_SURVEYOR",
            "CADA_COORDINATES",
            "CADA_NORTHCOORDINATES",
            "CADA_EASTCOORDINATES",
            # Some templates use generic text layers for headings/labels.
            "TITLE",
            "text",
        ]
        # Keep only ones that exist in the template (plus the core ones even if layer list was unavailable).
        layers_upper = {l.upper() for l in layers_in_template}
        sheet_layers = []
        for l in sheet_layers_default:
            if not layers_upper or l.upper() in layers_upper:
                sheet_layers.append(l)

        # Layers we expect a cadastral template to contain; used as a sanity check / metadata.
        # Include any CADA_* layers present plus key generic title/text layers.
        layers_expected = []
        for l in layers_in_template:
            lu = l.upper()
            if lu.startswith("CADA_") or lu in ("TITLE", "TEXT"):
                layers_expected.append(l)

        profile = {
            "success": True,
            "template": {
                "path": str(tp),
                "name": tp.name,
                "learned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "signature": {
                    "size_bytes": int(tp.stat().st_size),
                    "mtime_ns": int(getattr(tp.stat(), "st_mtime_ns", int(tp.stat().st_mtime * 1e9))),
                },
            },
            "drawing_info": drawing_info,
            "layers_expected": layers_expected,
            "sheet_layers": sheet_layers,
            "text_heights": {"bearing_dist_road": bearing_road_height},
            "tables": {
                "title_block": {"handle": (by_layer.get("CADA_TITLEBLOCK", [{}])[0] or {}).get("handle")},
                "plan_number": {"handle": (by_layer.get("CADA_PLANNUMBER", [{}])[0] or {}).get("handle")},
                "surveyor": {"handle": (by_layer.get("CADA_SURVEYOR", [{}])[0] or {}).get("handle")},
                "certification": {"handle": (by_layer.get("CADA_CERTIFICATION", [{}])[0] or {}).get("handle")},
                "coordinates": {"easting_table_handle": e_h, "northing_table_handle": n_h},
                "pillar_numbers": {"tables": by_layer.get("CADA_PILLARNUMBERS", [])},
            },
            "blocks": {"pillars": {"block_name": block_name or "PEG_SYMBOL"}},
        }
        outp.write_text(_json.dumps(profile, indent=2), encoding="utf-8")
        # STRICT: Register template so it is never written (read-only to avoid corruption).
        self._protected_template_paths.add(str(tp.resolve()))
        return {"success": True, "profile_path": str(outp), "profile": profile}

    def _apply_cadastral_template(
        self,
        profile_path: str,
        output_dwg_path: str,
        buyer_name: str,
        location: str,
        lga: str,
        state: str,
        origin_crs: str,
        plan_number: str,
        surveyor_name: str,
        surveyor_company_address: str,
        pillar_numbers: str,
        template_override_path: Optional[str] = None,
        coordinates: Optional[str] = None,
        certification_date: Optional[str] = None,
        access_road: Optional[str] = None,
        access_roads: Optional[List[str]] = None,
        fences: Optional[List[Dict[str, str]]] = None,
        access_road_title: Optional[str] = None,
        user_scale_denom: Optional[int] = None,
        extra_parcels: Optional[List[Dict[str, Any]]] = None,
        extent_points: Optional[List[Dict[str, Any]]] = None,
        main_parcel_label: Optional[str] = None,
        skip_session_prep: bool = False,
        batch_mode: bool = False,
        close_output_after_save: bool = False,
    ) -> Dict[str, Any]:
        # Normalize to list: support legacy single access_road
        roads_to_draw = access_roads if access_roads is not None else ([access_road] if access_road else [])
        fences_to_draw = fences or []
        # Multi-parcel Excel/CSV overlays (must be locals here — not in the prompt parser above).
        extra_parcels = list(extra_parcels or [])
        extent_points = list(extent_points or [])
        main_parcel_label = (main_parcel_label or "").strip()
        skip_session_prep = bool(skip_session_prep or batch_mode)
        close_output_after_save = bool(close_output_after_save)
        import json as _json
        import math
        import re
        import shutil
        import time
        from pathlib import Path

        prof = Path(profile_path).resolve()
        if not prof.exists():
            return {"success": False, "error": f"Profile not found: {str(prof)}"}
        profile = _json.loads(prof.read_text(encoding="utf-8"))
        template_src = template_override_path or profile.get("template", {}).get("path", "")
        template = None
        if template_override_path:
            try:
                candidate = Path(str(template_override_path)).resolve()
                if candidate.is_file():
                    template = candidate
            except Exception:
                template = None
        if template is None:
            template = self._resolve_cad_template_file_path(
                str(template_src or ""),
                profile_file=prof,
            )
        if template is None or not template.exists():
            return {"success": False, "error": f"Template not found: {str(template_src)}"}

        # ABSOLUTE RULE: The survey template DWG is never modified, regardless of user prompt.
        # We only read from it (or copy from it). All edits are made to the output drawing copy.

        outp = Path(output_dwg_path)
        if not outp.is_absolute():
            outp = self._resolve_user_output_path("", str(outp)).resolve()

        # Never treat the survey-plan template itself as a regeneratable output.
        try:
            if outp.resolve() == Path(template).resolve() or self._is_protected_template_path(str(outp)):
                return {
                    "success": False,
                    "error": (
                        "Refusing to overwrite the survey plan template DWG. "
                        "Choose a different output file name for the new plan."
                    ),
                }
        except Exception:
            if self._is_protected_template_path(str(outp)):
                return {
                    "success": False,
                    "error": "Refusing to overwrite the survey plan template DWG.",
                }

        if not self.autocad.is_connected and not self.autocad.connect():
            return {"success": False, "error": "Could not connect to AutoCAD via COM"}

        # Save (and close) other open/unsaved drawings so they do not block the new run.
        # Protected templates stay open and are never written.
        # Separate-owner batch skips the heavy session thrash after the first plan.
        if not skip_session_prep:
            try:
                self._ensure_protected_templates_loaded()
                exclude = list(self._protected_template_paths) + [str(Path(template).resolve())]
                prep = self.autocad.save_and_close_other_drawings(
                    exclude_paths=exclude,
                    close_after_save=True,
                )
                if prep.get("saved") or prep.get("closed"):
                    logger.info(
                        "Prepared AutoCAD session for new output (saved=%s closed=%s)",
                        prep.get("saved"),
                        prep.get("closed"),
                    )
            except Exception as e:
                logger.warning("save_and_close_other_drawings failed (continuing): %s", e)
        else:
            try:
                # Soft prep: only close a prior output with the same basename if open.
                self.autocad.close_drawing_if_open(str(outp), save_changes=False)
            except Exception:
                pass

        try:
            outp.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return {"success": False, "error": f"Cannot create output folder: {e}"}

        # If the output already exists (on disk or still open), ask before overwriting.
        output_exists = False
        try:
            output_exists = outp.exists()
        except Exception:
            output_exists = False
        if not output_exists:
            try:
                output_exists = bool(self.autocad.is_drawing_open(str(outp)))
            except Exception:
                output_exists = False

        if output_exists:
            if not _confirm_overwrite_existing_dwg(str(outp), mode="overwrite"):
                return {
                    "success": False,
                    "cancelled": True,
                    "error": (
                        f"Kept existing drawing unchanged: {str(outp)}\n"
                        "Overwrite was declined. Change the output name or choose Overwrite to replace it."
                    ),
                }
            # Release any remaining AutoCAD lock on the target before disk overwrite.
            try:
                if hasattr(self.autocad, "quiesce_autocad"):
                    self.autocad.quiesce_autocad(esc_count=3)
                self.autocad.close_drawing_if_open(str(outp), save_changes=False)
                time.sleep(0.35)
            except Exception:
                pass

        def _copy_template_to_output() -> Optional[Dict[str, Any]]:
            try:
                shutil.copy2(str(template), str(outp))
                return None
            except PermissionError:
                # One retry after forcing the target closed (common when still locked).
                try:
                    if hasattr(self.autocad, "quiesce_autocad"):
                        self.autocad.quiesce_autocad(esc_count=3)
                    self.autocad.close_drawing_if_open(str(outp), save_changes=False)
                    time.sleep(0.5)
                    shutil.copy2(str(template), str(outp))
                    return None
                except PermissionError:
                    return {
                        "success": False,
                        "cancelled": False,
                        "locked": True,
                        "error": (
                            f"Cannot write output DWG (still locked): {str(outp)}\n"
                            "Close the drawing in AutoCAD (or any other program using it), then retry."
                        ),
                    }
                except Exception as e2:
                    return {"success": False, "error": f"Failed to copy template to output: {e2}"}
            except Exception as e:
                return {"success": False, "error": f"Failed to copy template to output: {e}"}

        copy_err = _copy_template_to_output()
        if copy_err is not None:
            return copy_err

        opened = self.autocad.open_drawing(str(outp))
        if not opened.get("success"):
            # Documents may be wedged from a prior owner plan — recover once and retry open.
            err = opened.get("error", "Failed to open output drawing")
            try:
                if hasattr(self.autocad, "recover_com_session") and hasattr(
                    self.autocad, "_com_error_is_broken_proxy"
                ):
                    if self.autocad._com_error_is_broken_proxy(Exception(str(err))):
                        self.autocad.recover_com_session(force=True, reason=str(err))
                        time.sleep(0.5)
                        opened = self.autocad.open_drawing(str(outp))
            except Exception:
                pass
            if not opened.get("success"):
                return {
                    "success": False,
                    "error": opened.get("error", "Failed to open output drawing"),
                }
        self.autocad.set_workflow_document(str(outp))
        if batch_mode or skip_session_prep:
            # One open is enough; light re-activate (no second Documents.Open).
            time.sleep(0.05)
            self.autocad.ensure_workflow_document(light=True)
        else:
            time.sleep(0.3)
            # With several DWGs open (template + prior outputs), the active tab can be wrong.
            # Force the output we just copied to be active before any table/geometry edits.
            act2 = self.autocad.open_drawing(str(outp), read_only=False)
            if not act2.get("success"):
                logger.warning("Could not re-activate output drawing before edits: %s", act2.get("error"))
            self.autocad.ensure_workflow_document()
            time.sleep(0.2)

        def _cad_checkpoint() -> None:
            try:
                # Batch / soft-prep: never call full open_drawing from checkpoints —
                # that is what spam-logged <unknown>.Count/.Open when COM was wedged.
                if batch_mode or skip_session_prep:
                    self.autocad.ensure_workflow_document(light=True)
                else:
                    self.autocad.ensure_workflow_document()
            except Exception:
                pass

        def _release_output_tab(*, already_saved: bool = False) -> None:
            """Save+close this plan's DWG so the next owner plot does not pile up tabs."""
            if not close_output_after_save:
                return
            try:
                if hasattr(self.autocad, "save_and_close_drawing"):
                    self.autocad.save_and_close_drawing(
                        str(outp),
                        save=not already_saved,
                    )
                else:
                    if not already_saved:
                        self._ensure_output_saved(str(outp))
                    self.autocad.close_drawing_if_open(str(outp), save_changes=False)
                    self.autocad.set_workflow_document(None)
                    self.autocad.doc = None
                    time.sleep(0.35)
            except Exception as release_exc:
                logger.warning("Batch release of output DWG failed: %s", release_exc)
                try:
                    if hasattr(self.autocad, "recover_com_session"):
                        self.autocad.recover_com_session(reason="batch release failed")
                except Exception:
                    pass

        # --- helpers for table formatting preservation ---
        def _get_cell(h: str, r: int, c: int = 0) -> str:
            res = self.autocad.get_table_cell_text(h, r, c)
            return str(res.get("text") or "") if res.get("success") else ""

        def _set_cell(h: str, r: int, c: int, val: str):
            return self.autocad.set_table_cell_text(h, r, c, val)

        def _mtxt_replace(raw: str, new_content: str) -> str:
            raw = raw or ""
            if raw.startswith("{") and raw.endswith("}") and ";" in raw:
                idx = raw.rfind(";")
                return raw[: idx + 1] + new_content + "}"
            return new_content

        def _replace_after_label(raw: str, label: str, new_tail: str) -> str:
            raw = raw or ""
            if raw.startswith("{") and raw.endswith("}") and ";" in raw:
                idx = raw.rfind(";")
                wrapper = raw[: idx + 1]
                content = raw[idx + 1 : -1]
                pos = content.upper().find(label.upper())
                if pos == -1:
                    new_content = f"{label} {new_tail}".strip()
                else:
                    keep = content[: pos + len(label)]
                    new_content = (keep + " " + new_tail).strip()
                return wrapper + new_content + "}"
            pos = raw.upper().find(label.upper())
            if pos == -1:
                return f"{label} {new_tail}".strip()
            return (raw[: pos + len(label)] + " " + new_tail).strip()

        tables = profile.get("tables", {})
        text_heights = profile.get("text_heights") or {}
        bearing_road_height = float(text_heights.get("bearing_dist_road", 1.2))
        if bearing_road_height == 1.2:
            try:
                hr = self.autocad.get_sample_text_height(layers=["CADA_BEARING_DIST", "CADA_ROAD"])
                if hr.get("success") and hr.get("height"):
                    bearing_road_height = float(hr["height"])
            except Exception:
                pass
        # Clamp only to a minimum to avoid corrupted or COM-default tiny height; no max so scaled height (e.g. 24 for 1:10000) is allowed
        bearing_road_height = max(0.5, float(bearing_road_height))
        title_h = str((tables.get("title_block") or {}).get("handle") or "")
        plan_h = str((tables.get("plan_number") or {}).get("handle") or "")
        surv_h = str((tables.get("surveyor") or {}).get("handle") or "")
        cert_h = str((tables.get("certification") or {}).get("handle") or "")
        east_h = str(((tables.get("coordinates") or {}).get("easting_table_handle")) or "")
        north_h = str(((tables.get("coordinates") or {}).get("northing_table_handle")) or "")

        # Multi-parcel layout jobs are flagged early (extras / multi-owner title).
        multi_parcel_layout = bool(extra_parcels) or bool(main_parcel_label) or (
            (buyer_name or "").count(",") >= 2
        )
        # Provisional; refined with title-cell width before write (multi-owner packing).
        formatted_buyer_name = _format_buyer_name_for_titleblock(buyer_name)
        template_owner_cell_raw = ""
        template_owner_lines = 1
        template_scale_label_bottom = None
        title_scale_row = 8
        # Snapshot title-band BEFORE writing many owners — expanded owner lists must not
        # collapse usable plot height and force an absurdly small plan scale (e.g. 1:20000).
        template_title_bb_for_scale: Optional[Dict[str, Any]] = None
        try:
            _tbb0 = self.autocad.get_modelspace_bbox(layers=["CADA_TITLEBLOCK"])
            if _tbb0.get("success"):
                template_title_bb_for_scale = _tbb0
        except Exception:
            template_title_bb_for_scale = None
        bearing_road_height_native = float(bearing_road_height)

        tables_now = {}
        try:
            for t in (self.autocad.list_tables().get("tables") or []):
                h = str(t.get("handle") or "")
                if h:
                    tables_now[h] = t
        except Exception:
            tables_now = {}

        # Title block
        if title_h:
            template_owner_cell_raw = _get_cell(title_h, 2, 0)
            template_owner_lines = _mtext_content_line_count(template_owner_cell_raw)
            title_scale_row = _find_title_scale_label_row(_get_cell, title_h, tables_now)
            try:
                ext0 = self.autocad.get_table_cell_extents(title_h, title_scale_row, 0)
                if ext0.get("success"):
                    template_scale_label_bottom = float(ext0["miny"])
            except Exception:
                pass
            # Pack consecutive owners into as many names per row as the cell width allows
            # (no uneven per-name shrinking — vertical title height stays cartographic).
            owner_max_chars = 38
            try:
                ext_owner = self.autocad.get_table_cell_extents(title_h, 2, 0, outer=True)
                step_owner = self.autocad.get_table_cell_mtext_line_step(
                    title_h, 2, 0, template_owner_cell_raw
                )
                if ext_owner.get("success"):
                    ow = float(ext_owner.get("maxx", 0.0)) - float(ext_owner.get("minx", 0.0))
                    oth = float((step_owner or {}).get("text_height") or 0.0)
                    if multi_parcel_layout and oth > 0.05:
                        oth = oth * 0.9  # matches uniform multi-parcel title shrink below
                    if ow > 1.0 and oth > 0.05:
                        owner_max_chars = max(
                            16,
                            int(ow / (oth * _CARTO_CHAR_WIDTH_FACTOR)),
                        )
            except Exception:
                pass
            formatted_buyer_name = _format_buyer_name_for_titleblock(
                buyer_name, max_chars_per_line=owner_max_chars
            )
            _set_cell(title_h, 2, 0, _mtxt_replace(template_owner_cell_raw, formatted_buyer_name))
            loc_cell = _get_cell(title_h, 4)
            # Pack AT location into preferably ≤3 lines using measured cell width.
            loc_max_chars = 36
            try:
                ext_loc = self.autocad.get_table_cell_extents(title_h, 4, 0, outer=True)
                step_loc = self.autocad.get_table_cell_mtext_line_step(
                    title_h, 4, 0, loc_cell or ""
                )
                if ext_loc.get("success"):
                    lw = float(ext_loc.get("maxx", 0.0)) - float(ext_loc.get("minx", 0.0))
                    lth = float((step_loc or {}).get("text_height") or 0.0)
                    if lw > 1.0 and lth > 0.05:
                        loc_max_chars = max(
                            16,
                            int(lw / (lth * _CARTO_CHAR_WIDTH_FACTOR)),
                        )
            except Exception:
                pass
            loc_val = _format_location_for_titleblock(
                location,
                max_chars_per_line=loc_max_chars,
                preferred_max_lines=3,
            )
            if loc_val:
                if re.search(r"\bAT\b", loc_cell or "", re.IGNORECASE):
                    _set_cell(title_h, 4, 0, _replace_after_label(loc_cell, "AT", loc_val))
                else:
                    _set_cell(title_h, 4, 0, _mtxt_replace(loc_cell, loc_val))
            # Bare LGA name only — template already carries / receives "LOCAL GOVERNMENT AREA".
            try:
                from agent.pdf_survey_plan import normalize_lga_name

                lga_bare = normalize_lga_name(lga) or (lga or "").strip()
            except Exception:
                lga_bare = (lga or "").strip()
            lga_u = lga_bare.upper()
            lga_cell = _get_cell(title_h, 5)
            if lga_u and re.search(r"\\P\s*LOCAL\s+GOVERNMENT\s+AREA", lga_cell or "", re.I):
                # Preserve two-line template: NAME\PLOCAL GOVERNMENT AREA
                lga_content = f"{lga_u}\\PLOCAL GOVERNMENT AREA"
            elif lga_u:
                lga_content = f"{lga_u} LOCAL GOVERNMENT AREA"
            else:
                lga_content = ""
            if lga_content:
                _set_cell(title_h, 5, 0, _mtxt_replace(lga_cell, lga_content))
            _set_cell(title_h, 6, 0, _mtxt_replace(_get_cell(title_h, 6), state.strip().upper()))
            _set_cell(title_h, 11, 0, _replace_after_label(_get_cell(title_h, 11), "ORIGIN:-", origin_crs.strip().upper()))
            # Multi-parcel only: shrink title-block text up to 90% of template to fit many owners.
            if multi_parcel_layout:
                for _r in (2, 4, 5, 6):
                    try:
                        _hr = self.autocad.get_table_cell_text_height(title_h, _r, 0)
                        _th = float((_hr or {}).get("text_height") or 0.0)
                        if (_hr or {}).get("success") and _th > 0.05:
                            self.autocad.set_table_cell_text_height(title_h, _r, 0, _th * 0.9)
                    except Exception:
                        pass

        # CADA_PLANNUMBER is fitted after sheet scaling (see _apply_plan_number_table_cell).
        template_plan_nominal_h: Optional[float] = None
        template_plan_text_style = ""
        if plan_h:
            _pn_row, _pn_col = 1, 0
            _pn_tpl = _get_cell(plan_h, _pn_row, _pn_col)
            try:
                _pn_step = self.autocad.get_table_cell_mtext_line_step(
                    plan_h, _pn_row, _pn_col, _pn_tpl
                )
                if _pn_step.get("success") and _pn_step.get("text_height"):
                    template_plan_nominal_h = float(_pn_step["text_height"])
            except Exception:
                pass
            try:
                _pn_st = self.autocad.get_table_cell_text_style(plan_h, _pn_row, _pn_col)
                if _pn_st.get("success") and str(_pn_st.get("style") or "").strip():
                    template_plan_text_style = str(_pn_st["style"]).strip()
            except Exception:
                pass

        if surv_h:
            # Surveyor name: keep professional title (SURV.) and optional bracket credentials.
            def _format_surveyor_name(raw: str) -> str:
                try:
                    from agent.pdf_survey_plan import (
                        ensure_surveyor_professional_title,
                        scrub_surveyor_metadata_value,
                    )

                    s = ensure_surveyor_professional_title(
                        scrub_surveyor_metadata_value(raw or "", max_len=100)
                    )
                except Exception:
                    s = (raw or "").strip()
                    if s and not re.match(r"^(?:surv\.?|surveyor)\b", s, flags=re.IGNORECASE):
                        s = f"SURV. {s}"
                s = s.upper()
                if not s:
                    return s
                # Prefer bracket at end: "SURV. ... (MNIS)"
                m = re.match(r"^(.*?)(\s*\([^)]*\))\s*$", s)
                if m:
                    main = m.group(1).strip()
                    br = m.group(2).strip()
                    return f"{main} {{\\H0.67x;{br}}}".strip()
                return s

            surv_cell = _get_cell(surv_h, 0, 0)
            surv_fmt = _format_surveyor_name(surveyor_name)
            if re.search(r"\bSURV\.?\b", surv_cell or "", flags=re.IGNORECASE) and surv_fmt:
                # Preserve template "SURV." label when present; replace the name tail.
                name_tail = re.sub(
                    r"^(?:SURV\.?|SURVEYOR)\s*",
                    "",
                    surv_fmt,
                    count=1,
                    flags=re.IGNORECASE,
                ).strip()
                _set_cell(
                    surv_h,
                    0,
                    0,
                    _replace_after_label(surv_cell, "SURV.", name_tail)
                    if re.search(r"SURV\.", surv_cell or "", flags=re.IGNORECASE)
                    else _mtext_preserve_style_set_content(surv_cell, surv_fmt),
                )
            else:
                _set_cell(
                    surv_h,
                    0,
                    0,
                    _mtext_preserve_style_set_content(surv_cell, surv_fmt),
                )

            # Surveyor address: pack horizontally (city + state on one line) and
            # vertically inside the cell — never one comma segment per line by default.
            try:
                from agent.pdf_survey_plan import scrub_surveyor_metadata_value

                surveyor_company_address = scrub_surveyor_metadata_value(
                    surveyor_company_address or "", max_len=200
                )
            except Exception:
                surveyor_company_address = (surveyor_company_address or "").strip()
            addr_template = _get_cell(surv_h, 1, 0)
            addr_th = 0.6
            addr_step = 0.0
            addr_cell_w = 0.0
            addr_cell_h = 0.0
            try:
                step_res = self.autocad.get_table_cell_mtext_line_step(surv_h, 1, 0, addr_template)
                if step_res.get("success"):
                    addr_th = float(step_res.get("text_height") or addr_th)
                    addr_step = float(step_res.get("line_step") or 0.0)
            except Exception:
                pass
            try:
                ext = self.autocad.get_table_cell_extents(surv_h, 1, 0, outer=True)
                if ext.get("success"):
                    addr_cell_w = float(ext["maxx"]) - float(ext["minx"])
                    addr_cell_h = float(ext["maxy"]) - float(ext["miny"])
            except Exception:
                pass
            if addr_cell_w <= 0:
                addr_cell_w = max(20.0, addr_th * 30.0)
            if addr_cell_h <= 0:
                addr_cell_h = max(6.0, addr_th * 8.0)
            addr_u = _layout_surveyor_address_mtext(
                surveyor_company_address,
                cell_width=addr_cell_w,
                cell_height=addr_cell_h,
                text_height=addr_th,
                line_step=addr_step,
            )

            # Apply to all columns in the address row (some templates use multiple columns)
            cols = int((tables_now.get(surv_h) or {}).get("cols") or 1)
            for c in range(max(1, cols)):
                cur = _get_cell(surv_h, 1, c)
                _set_cell(surv_h, 1, c, _mtext_preserve_style_set_content(cur, addr_u))
            try:
                self.autocad.recompute_table(surv_h)
            except Exception:
                pass

        # Certification date
        if cert_h and certification_date:
            date_in = certification_date.strip()
            m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$", date_in)
            if m:
                dd, mm, yy = m.group(1).zfill(2), m.group(2).zfill(2), m.group(3)
                if len(yy) == 2:
                    yy = "20" + yy
                date_norm = f"{dd}-{mm}-{yy}"
            else:
                date_norm = date_in.replace("/", "-")
            date_pat = re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b")

            def _replace_any_date(raw: str) -> str:
                raw = raw or ""
                # Preserve template MTEXT wrapper if present
                if raw.startswith("{") and raw.endswith("}") and ";" in raw:
                    idx = raw.rfind(";")
                    wrapper = raw[: idx + 1]
                    content = raw[idx + 1 : -1]
                    if date_pat.search(content):
                        content2 = date_pat.sub(date_norm, content, count=1)
                        return wrapper + content2 + "}"
                    # Fallback: replace after "ON" if present, else append
                    if re.search(r"\bON\b", content, flags=re.IGNORECASE):
                        content2 = re.sub(r"(\bON\b\s*)(.*)$", lambda mm: (mm.group(1) + date_norm).strip(), content, flags=re.IGNORECASE)
                        return wrapper + content2 + "}"
                    return wrapper + (content.rstrip() + f" ON {date_norm}").strip() + "}"
                if date_pat.search(raw):
                    return date_pat.sub(date_norm, raw, count=1)
                if re.search(r"\bON\b", raw, flags=re.IGNORECASE):
                    return re.sub(r"(\bON\b\s*)(.*)$", lambda mm: (mm.group(1) + date_norm).strip(), raw, flags=re.IGNORECASE)
                return (raw.rstrip() + f" ON {date_norm}").strip()

            cols = int((tables_now.get(cert_h) or {}).get("cols") or 1)
            rows = int((tables_now.get(cert_h) or {}).get("rows") or 1)
            for r in range(max(1, rows)):
                for c in range(max(1, cols)):
                    cur = _get_cell(cert_h, r, c)
                    upd = _replace_any_date(cur)
                    if upd != cur:
                        _set_cell(cert_h, r, c, upd)

        # Parse coordinate pairs from prompt
        coord_pairs = []
        bowditch_info = None
        bearing_distance_legs: Optional[List[Dict[str, float]]] = None
        if coordinates:
            pairs = re.findall(
                r"\(\s*([0-9]+(?:\.[0-9]+)?)\s*(?:m)?\s*[eE]\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*(?:m)?\s*[nN]\s*\)",
                coordinates,
            )
            for e_s, n_s in pairs:
                coord_pairs.append({"e": float(e_s), "n": float(n_s)})

            # Fallback: single coordinate + bearings/distances traverse legs
            # Example inputs:
            # - "plot using coordinate 286638.060mE, 544692.450mN then bearing 17deg 49' and distance 16.14m ..."
            # - "use coordinate 286638.060mE, 544692.450mN; bearing = 17 degrees 49 min, distance = 16.14m ..."
            if len(coord_pairs) < 3:
                try:
                    # 1) Start coordinate (E,N)
                    m0 = re.search(
                        r"([0-9]+(?:\.[0-9]+)?)\s*(?:m)?\s*[eE]\s*[,; ]+\s*([0-9]+(?:\.[0-9]+)?)\s*(?:m)?\s*[nN]\b",
                        coordinates,
                        flags=re.IGNORECASE,
                    )
                    if m0:
                        e0 = float(m0.group(1))
                        n0 = float(m0.group(2))

                        # 2) Traverse legs: bearing + distance (flexible phrasing)
                        leg_re = re.compile(
                            # Accept "bearing-", "bearing:", "bearing=", and "bearing is"
                            r"\bbearing\b\s*(?:(?:=|:|-)|\bis\b)?\s*"
                            r"(\d{1,3})\s*(?:deg|degree|degrees|°|d)\s*"
                            r"([0-5]?\d)\s*(?:min|mins|minute|minutes|['’])"
                            r"(?:[^0-9]{0,80}?)"
                            r"(?:distance|dist\.?|measured\s+distance)\s*(?:=|is|:)?\s*"
                            r"([0-9]+(?:\.[0-9]+)?)\s*(?:m)?\b",
                            flags=re.IGNORECASE | re.DOTALL,
                        )
                        legs = []
                        for mm in leg_re.finditer(coordinates):
                            d = int(mm.group(1))
                            m = int(mm.group(2))
                            dist = float(mm.group(3))
                            # Normalize bearing to decimal degrees (from North, clockwise)
                            bdeg = (float(d) % 360.0) + (float(m) / 60.0)
                            legs.append((bdeg, dist))

                        if legs:
                            # Build unadjusted deltas + vertices
                            deltas = []
                            tmp_unadj = [{"e": float(e0), "n": float(n0)}]
                            ce, cn = float(e0), float(n0)
                            total_len = 0.0
                            for (bdeg, dist) in legs:
                                br = math.radians(float(bdeg))
                                de = float(dist) * math.sin(br)
                                dn = float(dist) * math.cos(br)
                                deltas.append((de, dn, float(dist)))
                                total_len += float(dist)
                                ce += de
                                cn += dn
                                tmp_unadj.append({"e": float(ce), "n": float(cn)})

                            mis_e = float(tmp_unadj[-1]["e"] - tmp_unadj[0]["e"])
                            mis_n = float(tmp_unadj[-1]["n"] - tmp_unadj[0]["n"])
                            mis = float(math.hypot(mis_e, mis_n))
                            threshold = 0.01  # 1 cm

                            tmp = tmp_unadj
                            applied = False
                            max_shift = 0.0
                            method_used = None
                            db_deg = None
                            bearing_distance_legs = [{"bearing_deg": float(bd), "distance": float(di)} for (bd, di) in legs]

                            # Default: bearing-adjustment (keep distances fixed).
                            # Bowditch is ONLY used when the user explicitly mentions it.
                            wants_bowditch = bool(re.search(r"\bbowditch\b", coordinates or "", flags=re.IGNORECASE))

                            if mis > threshold and total_len > 1e-9:
                                if wants_bowditch:
                                    method_used = "bowditch"
                                    applied = True
                                    tmp_adj = [{"e": float(e0), "n": float(n0)}]
                                    ce2, cn2 = float(e0), float(n0)
                                    for (de, dn, Li) in deltas:
                                        # distribute misclosure proportional to line length
                                        cde = (-mis_e) * (float(Li) / float(total_len))
                                        cdn = (-mis_n) * (float(Li) / float(total_len))
                                        ce2 += float(de + cde)
                                        cn2 += float(dn + cdn)
                                        tmp_adj.append({"e": float(ce2), "n": float(cn2)})
                                    tmp = tmp_adj

                                    # compute max point shift (excluding start)
                                    for k in range(1, min(len(tmp_unadj), len(tmp_adj))):
                                        sh = math.hypot(tmp_adj[k]["e"] - tmp_unadj[k]["e"], tmp_adj[k]["n"] - tmp_unadj[k]["n"])
                                        if sh > max_shift:
                                            max_shift = float(sh)
                                else:
                                    method_used = "bearing_adjustment"
                                    try:
                                        from tools import traverse_bearing_adjustment as _tba
                                        import numpy as _np

                                        b_list = [float(bd) for (bd, _di) in legs]
                                        d_list = [float(_di) for (_bd, _di) in legs]
                                        trv = _tba.traverse_from_agent(e0, n0, b_list, d_list, bearings_in_degrees=True)
                                        ba = _tba.form_matrices(trv)  # adjusted bearings in radians
                                        db = ba - _np.deg2rad(_np.asarray(b_list, dtype=float))
                                        db_deg = (_np.rad2deg(db)).tolist()
                                        bearing_distance_legs = [
                                            {"bearing_deg": float(_np.rad2deg(ba[i])), "distance": float(d_list[i])}
                                            for i in range(len(d_list))
                                        ]

                                        D_adj = _np.asarray(d_list, dtype=float) * _np.sin(ba)
                                        L_adj = _np.asarray(d_list, dtype=float) * _np.cos(ba)

                                        tmp_adj = [{"e": float(e0), "n": float(n0)}]
                                        ce2, cn2 = float(e0), float(n0)
                                        for i in range(len(d_list)):
                                            ce2 += float(D_adj[i])
                                            cn2 += float(L_adj[i])
                                            tmp_adj.append({"e": float(ce2), "n": float(cn2)})
                                        tmp = tmp_adj

                                        for k in range(1, min(len(tmp_unadj), len(tmp_adj))):
                                            sh = math.hypot(tmp_adj[k]["e"] - tmp_unadj[k]["e"], tmp_adj[k]["n"] - tmp_unadj[k]["n"])
                                            if sh > max_shift:
                                                max_shift = float(sh)
                                        applied = True
                                    except Exception:
                                        # If bearing-adjustment fails for any reason, keep unadjusted points
                                        method_used = "bearing_adjustment_failed"
                                        applied = False

                            # For start-coordinate + bearing/distance parcel traverses, the final computed point
                            # is the closure back to the start and must NOT be treated as an extra pillar.
                            # Keep only the unique parcel vertices; boundary polyline closure is handled separately.
                            if len(legs) >= 3 and len(tmp) == len(legs) + 1:
                                tmp = tmp[:-1]
                            # Safety fallback for any other near-duplicate closure point
                            elif len(tmp) >= 4:
                                dx0 = tmp[-1]["e"] - tmp[0]["e"]
                                dy0 = tmp[-1]["n"] - tmp[0]["n"]
                                if math.hypot(dx0, dy0) <= 0.25:
                                    tmp = tmp[:-1]

                            if len(tmp) >= 3:
                                # Default: stated (E,N) applies to the **first user-listed pillar** (traverse start).
                                # Optional override: user names the pillar (e.g. "coordinates for SC/Q 572") → translate
                                # so that vertex receives the stated values.
                                def _explicit_coord_pillar_index(c_text, p_raw, n_v):
                                    parts = [p.strip() for p in re.split(r"[,\n]+", (p_raw or "").strip()) if p.strip()]
                                    ct = c_text or ""
                                    for i, tok in enumerate(parts):
                                        if i >= n_v:
                                            break
                                        t_esc = re.escape(tok).replace("\\ ", "\\s+")
                                        if re.search(
                                            rf"(?:coordinate|coordinates)\s+for\s+(?:the\s+)?(?:pillar|peg|point(?:s)?)?\s*{t_esc}\b",
                                            ct,
                                            re.IGNORECASE,
                                        ):
                                            return i
                                        if re.search(
                                            rf"(?:pillar|peg)\s+{t_esc}\s*(?:has|holds|=)\s*(?:the\s+)?(?:coordinate|coordinates)",
                                            ct,
                                            re.IGNORECASE,
                                        ):
                                            return i
                                    return None

                                n_v = len(tmp)
                                stated_e = float(tmp[0]["e"])
                                stated_n = float(tmp[0]["n"])
                                exp_i = _explicit_coord_pillar_index(coordinates, pillar_numbers, n_v)
                                if exp_i is not None and 0 <= exp_i < n_v:
                                    de_a = stated_e - float(tmp[exp_i]["e"])
                                    dn_a = stated_n - float(tmp[exp_i]["n"])
                                    if abs(de_a) + abs(dn_a) > 1e-9:
                                        for _p in tmp:
                                            _p["e"] = float(_p["e"]) + de_a
                                            _p["n"] = float(_p["n"]) + dn_a

                                # Stated (E,N) + legs: first-listed pillar unless overridden above. Primary pillar
                                # (min E, tie min N) is taken later from this geometry for sheet rules; coordinate tables
                                # show the primary pillar's **computed** E/N (see e0/n0 after vertex rotation).
                                coord_pairs = tmp
                                bowditch_info = {
                                    "mode": "bearing_distance",
                                    "method": method_used or ("bowditch" if wants_bowditch else "bearing_adjustment"),
                                    "applied": bool(applied),
                                    "threshold_m": float(threshold),
                                    "misclosure_m": float(mis),
                                    "misclosure_e_m": float(mis_e),
                                    "misclosure_n_m": float(mis_n),
                                    "perimeter_m": float(total_len),
                                    "max_point_shift_m": float(max_shift),
                                    "db_deg": db_deg,
                                    # Keep preview small (most jobs are 4-10 legs)
                                    "adjusted_points_preview": [
                                        {"idx": i, "e": float(p["e"]), "n": float(p["n"])}
                                        for i, p in enumerate(coord_pairs[: min(6, len(coord_pairs))])
                                    ],
                                }
                except Exception:
                    pass

        if coordinates and len(coord_pairs) < 3:
            _release_output_tab(already_saved=False)
            return {
                "success": False,
                "error": (
                    "Could not build a closed parcel from the supplied coordinates/traverse "
                    f"(parsed {len(coord_pairs)} point(s); need at least 3). "
                    "Check bearing/distance legs or provide explicit (E,N) pairs for each pillar."
                ),
            }

        geometry = {"pillars_moved": 0, "boundary_redrawn": False, "bearing_mtext": 0, "access_road_title": None}
        if bowditch_info:
            geometry["bowditch"] = bowditch_info

        # Plan-scale state (updated inside the coordinate plotting block when present).
        scale_k = 1.0
        chosen_denom = 500
        template_denom = 500
        template_native_denom = 500
        output_plan_denom = 500
        output_scale_k = 1.0

        if coord_pairs and len(coord_pairs) >= 3:
            # Pillar ↔ vertex order follows the user's pillar list and traverse-leg order.
            # We may rotate vertices so the template primary peg matches the survey primary (min E, tie min N);
            # pillar **names** must rotate with the same offset (see pn_list).
            pts = list(coord_pairs)
            try:
                # West-most = minimum Easting. Near-identical Easting (survey noise / adjustment):
                # tie-break to lower Northing (south-west-most).
                es = [float(pts[i].get("e", 0.0)) for i in range(len(pts))]
                min_e = min(es)
                spread = float(max(es) - min_e) if len(es) > 1 else 0.0
                eps_e = max(0.01, spread * 1e-9)  # ≥1 cm, or ~0.1 ppm of parcel width
                near_west = [
                    i for i in range(len(pts)) if abs(float(pts[i].get("e", 0.0)) - min_e) <= eps_e
                ]
                primary_idx = int(
                    min(near_west, key=lambda i: float(pts[i].get("n", 0.0)))
                )
                geometry["primary_easting_eps_m"] = float(eps_e)
            except Exception:
                primary_idx = 0
            if primary_idx:
                pts = pts[primary_idx:] + pts[:primary_idx]
                if bearing_distance_legs and len(bearing_distance_legs) == len(pts):
                    bearing_distance_legs = bearing_distance_legs[primary_idx:] + bearing_distance_legs[:primary_idx]
            primary = pts[0]
            e0, n0 = primary["e"], primary["n"]
            geometry["primary_vertex_index_original"] = int(primary_idx)
            geometry["primary_easting"] = float(e0)
            geometry["primary_northing"] = float(n0)

            # Surveyor convention: bearings labelled clockwise around the parcel.
            # Excel / CSV rings may be entered CCW — reverse (keeping primary at index 0)
            # so bearing texts are never backbearings of the clockwise traverse.
            def _signed_area_en(ring: List[Dict[str, Any]]) -> float:
                a = 0.0
                n = len(ring)
                for i in range(n):
                    j = (i + 1) % n
                    a += float(ring[i]["e"]) * float(ring[j]["n"])
                    a -= float(ring[j]["e"]) * float(ring[i]["n"])
                return 0.5 * a

            if len(pts) >= 3 and _signed_area_en(pts) > 0.0:
                pts = [pts[0]] + list(reversed(pts[1:]))
                if bearing_distance_legs and len(bearing_distance_legs) == len(pts):
                    # Edges reverse with the ring; each leg becomes its backbearing.
                    bearing_distance_legs = [
                        {
                            "bearing_deg": (float(L.get("bearing_deg", 0.0)) + 180.0) % 360.0,
                            "distance": float(L.get("distance", 0.0)),
                        }
                        for L in reversed(bearing_distance_legs)
                    ]
                geometry["ring_reversed_to_clockwise"] = True
            else:
                geometry["ring_reversed_to_clockwise"] = False

            def _primary_survey_index(vertex_pts: List[Dict[str, Any]]) -> int:
                best = 0
                for i in range(1, len(vertex_pts)):
                    e_i = float(vertex_pts[i].get("e", 0.0))
                    n_i = float(vertex_pts[i].get("n", 0.0))
                    e_b = float(vertex_pts[best].get("e", 0.0))
                    n_b = float(vertex_pts[best].get("n", 0.0))
                    if e_i < e_b - 1e-9:
                        best = i
                    elif abs(e_i - e_b) <= 1e-9 and n_i < n_b - 1e-9:
                        best = i
                return best

            primary_idx = _primary_survey_index(pts)
            primary = pts[primary_idx]
            e0, n0 = float(primary["e"]), float(primary["n"])
            geometry["primary_pillar_index"] = int(primary_idx)

            # Existing pillar inserts in output (copied from template)
            ins = self.autocad.list_inserts(layer="CADA_PILLARS")
            pillar_ins = (ins.get("inserts") or []) if isinstance(ins, dict) else []
            # Template primary position in drawing coords (min X)
            def _ix(i):
                pt = i.get("insertion_point") or {}
                return float(pt.get("x", 0.0))
            primary_ins = sorted(pillar_ins, key=_ix)[0] if pillar_ins else None
            base_x = float((primary_ins.get("insertion_point") or {}).get("x", 0.0)) if primary_ins else 0.0
            base_y = float((primary_ins.get("insertion_point") or {}).get("y", 0.0)) if primary_ins else 0.0

            # Decide whether the template must be upscaled to the next allowed smaller plan scale
            # (e.g. template labeled 1:500 → treat output as 1:1000) BEFORE plotting/aligning,
            # so the result stays neat like the template.
            scale_k = 1.0
            chosen_denom = 500  # plan scale 1:chosen_denom; used for road min-length check and title
            template_denom = 500
            try:
                allowed_denoms = [250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 25000]

                # Parse template denom from title block (best-effort)
                if title_h:
                    try:
                        tbl = tables_now.get(title_h, {}) if isinstance(tables_now, dict) else {}
                        rows = int(tbl.get("rows", 25))
                        cols = int(tbl.get("cols", 2))
                        # Parse template scale from title block. We must use the MAIN scale (e.g. 1:500), never "SCALE: to 1:1000".
                        scale_pat = re.compile(r"1\s*:\s*(\d+)", re.IGNORECASE)
                        secondary_re = re.compile(r"\bSCALE\b\s*:.*\bto\b", re.IGNORECASE)
                        main_scale_re = re.compile(r"\bSCALE\b\s*:-?\s*", re.IGNORECASE)  # "SCALE:-" or "SCALE : -"
                        candidates = []
                        for r in range(min(rows, 60)):
                            for c in range(min(cols, 10)):
                                cell = _get_cell(title_h, r, c) or ""
                                if secondary_re.search(cell):
                                    continue
                                m = scale_pat.search(cell)
                                if not m:
                                    continue
                                d = int(m.group(1))
                                if d not in allowed_denoms:
                                    continue
                                is_main = "SCALE" in cell.upper() and main_scale_re.search(cell) and not secondary_re.search(cell)
                                candidates.append((is_main, r, d))
                        if candidates:
                            main_candidates = [c for c in candidates if c[0]]
                            if main_candidates:
                                template_denom = max(c[2] for c in main_candidates)
                            else:
                                template_denom = min(c[2] for c in candidates)
                            found = True
                        else:
                            found = False
                    except Exception:
                        template_denom = 500
                if template_denom not in allowed_denoms:
                    template_denom = 500
                template_native_denom = int(template_denom)
                chosen_denom = template_denom  # plan scale denominator (1:chosen_denom); may be updated below

                # Interior border bbox (template coords)
                interior_bb = self.autocad.get_modelspace_bbox(layers=["CADA_INTERIORBORDER"])
                if not interior_bb.get("success"):
                    interior_bb = self.autocad.get_modelspace_bbox(layers=["CADA_INTERIORBOUNDARY"])
                if not interior_bb.get("success"):
                    interior_bb = self.autocad.get_modelspace_bbox(block_name_contains="INTERIOR", prefer_largest=True)

                # Boundary extents from user coordinates (meters).
                # Multi-parcel Excel layouts may pass extent_points covering every ownership ring.
                extent_src = extent_points if len(extent_points) >= 3 else pts
                es = [float(p.get("e", 0.0)) for p in extent_src]
                ns = [float(p.get("n", 0.0)) for p in extent_src]
                boundary_w = (max(es) - min(es)) if es else 0.0
                boundary_h = (max(ns) - min(ns)) if ns else 0.0

                # Debug always returned so we can see what happened
                geometry["scale_debug"] = {
                    "template_denom": int(template_denom),
                    "boundary_w": float(boundary_w),
                    "boundary_h": float(boundary_h),
                    "interior_found": bool(interior_bb.get("success")),
                    "extent_point_count": int(len(extent_src)),
                }

                if interior_bb.get("success") and boundary_w > 1e-6 and boundary_h > 1e-6:
                    interior_w = float(interior_bb.get("maxx", 0.0)) - float(interior_bb.get("minx", 0.0))
                    interior_h = float(interior_bb.get("maxy", 0.0)) - float(interior_bb.get("miny", 0.0))
                    iy0 = float(interior_bb.get("miny", 0.0))
                    iy1 = float(interior_bb.get("maxy", 0.0))
                    margin = 0.08
                    interior_usable_w = interior_w * (1.0 - 2.0 * margin)
                    interior_usable_h = interior_h * (1.0 - 2.0 * margin)
                    # Access roads / offsets extend outside the boundary; pad so scale-up still clears title/border.
                    road_pad_m = 0.0
                    try:
                        for _spec in roads_to_draw or []:
                            if not _spec:
                                continue
                            _mx = re.search(r"(\d+(?:\.\d+)?)\s*m", str(_spec), re.IGNORECASE)
                            if _mx:
                                road_pad_m = max(road_pad_m, float(_mx.group(1)) * 0.55)
                    except Exception:
                        pass
                    boundary_w_pad = float(boundary_w) + 2.0 * road_pad_m
                    boundary_h_pad = float(boundary_h) + 2.0 * road_pad_m
                    # PDF/user-stated finer scale already implies the parcel fits at that scale;
                    # do not inflate extents with road padding (avoids false overflow to 1:500).
                    if (
                        user_scale_denom
                        and int(user_scale_denom) in allowed_denoms
                        and int(user_scale_denom) < int(template_denom)
                    ):
                        boundary_w_pad = float(boundary_w)
                        boundary_h_pad = float(boundary_h)
                    # Vertical band where parcel + road may be drawn: inside inner border, but not in CADA_TITLEBLOCK.
                    y_lo = iy0 + margin * interior_h
                    y_hi_interior = iy1 - margin * interior_h
                    y_hi = y_hi_interior
                    title_bb = self.autocad.get_modelspace_bbox(layers=["CADA_TITLEBLOCK"])
                    # Multi-parcel: ignore the post-write expanded owner list when reserving the title band.
                    title_bb_for_fit = title_bb
                    if multi_parcel_layout and template_title_bb_for_scale and template_title_bb_for_scale.get("success"):
                        title_bb_for_fit = template_title_bb_for_scale
                    title_gap = max(3.0, 0.02 * interior_h)
                    title_excluded = False
                    if title_bb_for_fit.get("success"):
                        tminy = float(title_bb_for_fit.get("miny", 0.0))
                        tmaxy = float(title_bb_for_fit.get("maxy", 0.0))
                        t_cy = 0.5 * (tminy + tmaxy)
                        # If the title block sits in the upper part of the interior, reserve space below its bottom edge.
                        if tmaxy > iy0 and tminy < iy1 and t_cy > iy0 + 0.5 * interior_h:
                            y_cand = tminy - title_gap
                            if y_cand > y_lo + 1.0:
                                y_hi = min(y_hi_interior, y_cand)
                                title_excluded = True
                    usable_h_plot = max(0.0, y_hi - y_lo)
                    usable_w_plot = float(interior_usable_w)
                    if usable_h_plot < 1e-3:
                        usable_h_plot = float(interior_usable_h)
                    # Multi-parcel layouts need a stable plot band; never let title overflow
                    # shrink usable height below ~55% of the interior usable band.
                    if multi_parcel_layout:
                        min_h = float(interior_usable_h) * 0.55
                        min_w = float(interior_usable_w) * 0.85
                        if usable_h_plot < min_h:
                            usable_h_plot = min_h
                        if usable_w_plot < min_w:
                            usable_w_plot = min_w
                    geometry["scale_debug"].update({
                        "interior_w": float(interior_w),
                        "interior_h": float(interior_h),
                        "interior_usable_w": float(interior_usable_w),
                        "interior_usable_h": float(interior_usable_h),
                        "usable_w_plot": float(usable_w_plot),
                        "usable_h_plot": float(usable_h_plot),
                        "road_pad_m": float(road_pad_m),
                        "boundary_w_padded": float(boundary_w_pad),
                        "boundary_h_padded": float(boundary_h_pad),
                        "title_bb_used": bool(title_bb_for_fit.get("success")),
                        "title_excluded_upper": bool(title_excluded),
                        "title_bb_pre_owner": bool(
                            multi_parcel_layout and template_title_bb_for_scale is not None
                        ),
                        "margin": float(margin),
                        "multi_parcel_layout": bool(multi_parcel_layout),
                    })
                    if usable_w_plot > 1e-6 and usable_h_plot > 1e-6:
                        required_k = max(
                            boundary_w_pad / usable_w_plot,
                            boundary_h_pad / usable_h_plot,
                        )
                        geometry["scale_debug"]["required_k"] = float(required_k)

                chosen_denom, scale_k, scale_reason = _resolve_cadastral_output_scale(
                    template_denom=int(template_denom),
                    user_scale_denom=user_scale_denom,
                    required_k=float(geometry["scale_debug"].get("required_k") or 0.0),
                    allowed_denoms=allowed_denoms,
                    multi_parcel_layout=bool(multi_parcel_layout),
                    boundary_w=float(boundary_w),
                    boundary_h=float(boundary_h),
                )
                geometry["scale_debug"].update({
                    "chosen_denom": int(chosen_denom),
                    "k": float(scale_k),
                    "scale_reason": scale_reason,
                    "user_scale_denom": user_scale_denom,
                })
                # Multi-parcel cartography: plot parcels first at template scale, then fit the
                # sheet from the real drawn extent (avoids pre-plot 1:20000 / title overlap).
                defer_multi_parcel_sheet_scale = bool(multi_parcel_layout)
                if defer_multi_parcel_sheet_scale:
                    geometry["scale_debug"]["deferred_plot_then_fit"] = True
                    geometry["scale_debug"]["prefit_chosen_denom"] = int(chosen_denom)
                    geometry["scale_debug"]["prefit_k"] = float(scale_k)
                    geometry["scale_debug"]["prefit_reason"] = str(scale_reason)
                    chosen_denom = int(template_denom)
                    scale_k = 1.0
                    scale_reason = "multi_parcel_defer_until_plot"
                    geometry["scale_debug"]["chosen_denom"] = int(chosen_denom)
                    geometry["scale_debug"]["k"] = float(scale_k)
                    geometry["scale_debug"]["scale_reason"] = scale_reason
                output_plan_denom = int(chosen_denom)
                output_scale_k = float(scale_k)
                geometry["output_plan_denom"] = int(output_plan_denom)
                geometry["output_scale_k"] = float(output_scale_k)
                geometry["template_native_denom"] = int(template_native_denom)

                _cad_checkpoint()
                if scale_k != 1.0:
                            # Scale the template/sheet about the TEMPLATE PRIMARY PILLAR (base_x/base_y).
                            # This preserves arrow/coordinate geometry emanating from the pillar.
                            layers_to_scale = list(profile.get("sheet_layers") or []) or [
                                "CADA_BORDER",
                                "CADA_INTERIORBORDER",
                                "CADA_SCALEBAR",
                                "CADA_NORTHARROW",
                                "CADA_EASTARROW",
                                "CADA_TITLEBLOCK",
                                "CADA_PLANNUMBER",
                                "CADA_CERTIFICATION",
                                "CADA_SURVEYOR",
                                "CADA_COORDINATES",
                                "CADA_NORTHCOORDINATES",
                                "CADA_EASTCOORDINATES",
                                "CADA_PRIMARYPILLAR_ARROWS",
                                "TITLE",
                                "text",
                            ]
                            # Ensure critical template layers are always included even if the learned profile omitted them
                            for req in [
                                "CADA_BORDER",
                                "CADA_INTERIORBORDER",
                                "CADA_NORTHARROW",
                                "CADA_EASTARROW",
                                "CADA_NORTHCOORDINATES",
                                "CADA_EASTCOORDINATES",
                                "CADA_COORDINATES",
                                "CADA_PRIMARYPILLAR_ARROWS",
                                "CADA_SCALEBAR",
                                "CADA_TITLEBLOCK",
                            ]:
                                if req not in layers_to_scale:
                                    layers_to_scale.append(req)
                            # Scale pillar-number TABLES too (they are part of the template look)
                            layers_to_scale += ["CADA_PILLARNUMBERS"]
                            sc = self.autocad.scale_modelspace_by_layers(base_x, base_y, scale_k, layers_to_scale)
                            geometry["scale_debug"]["scaled_entities"] = int(sc.get("scaled_entities", 0) or 0) if sc.get("success") else 0

                            # Keep scalebar labels consistent with the new plan scale.
                            # The label factor is the same ratio used for the plan scale change (e.g., 1:500 -> 1:250 => 0.5).
                            try:
                                sb_factor = float(chosen_denom) / float(template_denom)
                                if abs(sb_factor - 1.0) > 1e-9:
                                    self.autocad.scale_scalebar_text_values(sb_factor, layers=["scalebar", "CADA_SCALEBAR"])
                            except Exception:
                                pass

                            # Scale bar hashing: scaling is applied once. The scale bar is a block insert on
                            # CADA_SCALEBAR; scale_modelspace_by_layers above already scales that insert by scale_k,
                            # so the block (and the hatch inside it) is scaled once. We do NOT also call
                            # scale_hatch_pattern_scale_by_layers(scale_k), or the hatch would be scaled twice
                            # (insert scale_k × PatternScale scale_k = scale_k²).

                            # Force hatch regen so the scalebar hashing updates immediately
                            try:
                                self.autocad.execute_command("REGEN")
                            except Exception:
                                pass

                            # Ensure new bearing/road text is created at the correct size
                            try:
                                bearing_road_height = float(bearing_road_height) * float(scale_k)
                            except Exception:
                                pass

                            # Edit only the existing main scale text in CADA_TITLEBLOCK.
                            # IMPORTANT: Clear the extra "SCALE: to 1:xxxx" line (row 9 should be empty).
                            if title_h:
                                try:
                                    scale_pattern = re.compile(r"1\s*:\s*\d+", re.IGNORECASE)
                                    replacement = f"1:{chosen_denom}"
                                    # Locate the main "SCALE:- 1:xxx" cell and update only that one.
                                    # Also remove any secondary "SCALE: to 1:xxx" cell so it stays blank.
                                    tbl = tables_now.get(title_h, {}) if isinstance(tables_now, dict) else {}
                                    rows = int(tbl.get("rows", 25))
                                    cols = int(tbl.get("cols", 2))

                                    main_scale_cell = None  # (r, c)
                                    secondary_scale_cells = []  # [(r, c), ...]
                                    secondary_re = re.compile(r"\bSCALE\b\s*:.*\bto\b", re.IGNORECASE)
                                    main_hint_re = re.compile(r"\bSCALE\b\s*[:-]", re.IGNORECASE)  # SCALE:- / SCALE:

                                    for r in range(min(rows, 60)):
                                        for c in range(min(cols, 10)):
                                            cell = _get_cell(title_h, r, c) or ""
                                            if not cell.strip():
                                                continue
                                            if secondary_re.search(cell) and scale_pattern.search(cell):
                                                secondary_scale_cells.append((r, c))
                                                continue
                                            if main_scale_cell is None and main_hint_re.search(cell) and scale_pattern.search(cell):
                                                # Prefer the explicit "SCALE:- 1:xxx" style cell.
                                                main_scale_cell = (r, c)

                                    # Fallback to the previously-known position if we didn't locate it.
                                    if main_scale_cell is None:
                                        main_scale_cell = (8, 0)

                                    mr, mc = main_scale_cell
                                    title_scale_row = int(mr)
                                    cell_main = _get_cell(title_h, mr, mc) or ""
                                    new_cell_main = scale_pattern.sub(replacement, cell_main)
                                    if new_cell_main != cell_main:
                                        _set_cell(title_h, mr, mc, new_cell_main)
                                    elif not (cell_main or "").strip():
                                        _set_cell(title_h, mr, mc, f"SCALE:- {replacement}")

                                    # Clear secondary scale cells (row 9 should be empty / no duplicate scale line).
                                    for (sr, sc_) in secondary_scale_cells:
                                        try:
                                            _set_cell(title_h, sr, sc_, "")
                                        except Exception:
                                            pass

                                    # Extra safety: if row 9 contains a secondary "SCALE: to ..." line, blank it.
                                    # (We only clear it when it matches the secondary pattern to avoid wiping other content.)
                                    for rr in (9, 8):  # handle possible off-by-one table indexing variations
                                        for cc in range(min(cols, 10)):
                                            v = _get_cell(title_h, rr, cc) or ""
                                            if v.strip() and secondary_re.search(v) and scale_pattern.search(v):
                                                _set_cell(title_h, rr, cc, "")
                                except Exception:
                                    pass
            except Exception as scale_exc:
                logger.warning("Cadastral scale/title-block step failed (keeping resolved scale): %s", scale_exc)

            # Always keep CADA_SCALEBAR below the "SCALE:- 1:xxx" cell. Owner and/or
            # location growth can push the SCALE label down; the bar must follow.
            if title_h:
                try:
                    scale_row = _find_title_scale_label_row(
                        _get_cell, title_h, tables_now, default_row=title_scale_row
                    )
                    adj = self.autocad.adjust_scalebar_below_scale_label(
                        title_h,
                        scale_label_row=scale_row,
                        scale_label_col=0,
                        scalebar_layers=["CADA_SCALEBAR"],
                        template_scale_label_bottom=template_scale_label_bottom,
                        scale_base_y=base_y,
                        scale_k=float(scale_k),
                    )
                    if not adj.get("success"):
                        owner_lines = _titleblock_owner_line_count(formatted_buyer_name)
                        extra_owner_lines = max(0, owner_lines - template_owner_lines)
                        step_res = self.autocad.get_table_cell_mtext_line_step(
                            title_h, 2, 0, template_owner_cell_raw
                        )
                        line_step = float(step_res.get("line_step") or 0.0)
                        if line_step <= 0:
                            th = float(step_res.get("text_height") or 0.0)
                            if th > 0:
                                line_step = th * (5.0 / 3.0)
                        if line_step > 0 and extra_owner_lines > 0:
                            self.autocad.move_modelspace_by_layers(
                                0.0,
                                -extra_owner_lines * line_step,
                                ["CADA_SCALEBAR"],
                            )
                except Exception:
                    pass

            local_pts = [{"x": base_x + (p["e"] - e0), "y": base_y + (p["n"] - n0)} for p in pts]

            # Multi-parcel Excel/CSV layouts: skip slow pillar-number TABLE cloning and
            # bearing/distance annotation (COM-heavy). Owner labels + rings are enough.
            # Geometry flag is extras/labels only (not multi-comma buyer titles alone).
            multi_parcel_layout = bool(extra_parcels) or bool(main_parcel_label)
            if multi_parcel_layout:
                geometry["multi_parcel_layout"] = True

            def _en_to_local(ring: List[Dict[str, Any]]) -> List[Dict[str, float]]:
                out_l: List[Dict[str, float]] = []
                for p in ring or []:
                    if not isinstance(p, dict):
                        continue
                    try:
                        ee = float(p.get("e", p.get("x")))
                        nn = float(p.get("n", p.get("y")))
                    except Exception:
                        continue
                    out_l.append({"x": float(base_x + (ee - e0)), "y": float(base_y + (nn - n0))})
                return out_l

            def _insert_pegs_at(points_xy: List[Dict[str, float]]) -> None:
                blk_name = profile.get("blocks", {}).get("pillars", {}).get("block_name") or "PEG_SYMBOL"
                inserted_ok = True
                for p in points_xy:
                    r = self.autocad.insert_block(
                        str(blk_name),
                        float(p["x"]),
                        float(p["y"]),
                        layer="CADA_PILLARS",
                        xscale=float(scale_k),
                        yscale=float(scale_k),
                        zscale=float(scale_k),
                    )
                    if not r.get("success"):
                        inserted_ok = False
                        break
                if inserted_ok:
                    return
                # Rebuild via clipboard seed (copied before CADA_PILLARS clear) — InsertBlock COM quirk.
                try:
                    import pythoncom
                    pythoncom.CoInitialize()
                except Exception:
                    pass
                try:
                    out_doc = self.autocad.doc
                    ms_out = out_doc.ModelSpace
                    # Prefer pre-copied seed; else try any remaining peg.
                    has_seed = _peg_seed_ent is not None
                    if not has_seed:
                        for ii in range(ms_out.Count):
                            e = ms_out.Item(ii)
                            if str(getattr(e, "Layer", "")).upper() == "CADA_PILLARS" and "BlockReference" in str(
                                getattr(e, "ObjectName", "")
                            ):
                                e.Copy()
                                has_seed = True
                                break
                    if not has_seed:
                        return
                    try:
                        self.autocad.delete_entities("CADA_PILLARS")
                    except Exception:
                        pass
                    for p in points_xy:
                        new_ent = ms_out.Paste()
                        try:
                            new_ent.Layer = "CADA_PILLARS"
                        except Exception:
                            pass
                        try:
                            for attr in ("XScaleFactor", "YScaleFactor", "ZScaleFactor"):
                                setattr(new_ent, attr, float(scale_k))
                        except Exception:
                            pass
                        ip = getattr(new_ent, "InsertionPoint", None)
                        if ip is not None:
                            dxm = float(p["x"]) - float(ip[0])
                            dym = float(p["y"]) - float(ip[1])
                            try:
                                new_ent.Move((0.0, 0.0, 0.0), (dxm, dym, 0.0))
                            except Exception:
                                pass
                    try:
                        out_doc.Activate()
                    except Exception:
                        pass
                except Exception:
                    pass

            _cad_checkpoint()
            # Preserve one template peg seed before clearing (InsertBlock often fails; clone needs a seed).
            _peg_seed_ent = None
            try:
                _ms_seed = self.autocad.doc.ModelSpace
                for _ii in range(_ms_seed.Count):
                    _e = _ms_seed.Item(_ii)
                    if str(getattr(_e, "Layer", "")).upper() == "CADA_PILLARS" and "BlockReference" in str(
                        getattr(_e, "ObjectName", "")
                    ):
                        _peg_seed_ent = _e
                        break
                if _peg_seed_ent is not None:
                    _peg_seed_ent.Copy()
            except Exception:
                _peg_seed_ent = None
            # Clear old parcel graphics (not tables/border)
            self.autocad.delete_entities("CADA_BEARING_DIST")
            self.autocad.delete_entities("CADA_BOUNDARY")
            self.autocad.delete_entities("CADA_PILLARS")
            self.autocad.delete_entities("CADA_ROAD")
            self.autocad.delete_entities("CADA_CWF")
            self.autocad.delete_entities("CADA_TEXT")
            if multi_parcel_layout:
                # Avoid leftover template pillar-number TABLEs cluttering layout plans.
                try:
                    self.autocad.delete_entities("CADA_PILLARNUMBERS")
                except Exception:
                    pass
            time.sleep(0.2)
            # IMPORTANT: Do NOT delete generic sheet/title layers; they are part of the template
            # border/title presentation and must remain aligned with the border boxes.

            self.autocad.set_layer_color("CADA_BOUNDARY", 1)
            geometry["pillars_moved"] = 0

            if multi_parcel_layout:
                # Cartographic multi-parcel: unique pegs → deduped traverse edges → owner labels.
                # All in local/template space (same as single-parcel) so move_all keeps sheet alignment.
                try:
                    from agent.excel_cadastral import build_multi_parcel_layout_draw_ops

                    layout_parcels: List[Dict[str, Any]] = []
                    if local_pts and len(local_pts) >= 2:
                        layout_parcels.append(
                            {"label": main_parcel_label or "", "points": list(local_pts)}
                        )
                    for ep in extra_parcels:
                        if not isinstance(ep, dict):
                            continue
                        loc_ring = _en_to_local(list(ep.get("points") or []))
                        if len(loc_ring) < 2:
                            continue
                        layout_parcels.append(
                            {
                                "label": str(ep.get("label") or "").strip(),
                                "points": loc_ring,
                            }
                        )
                    draw_ops = build_multi_parcel_layout_draw_ops(layout_parcels)
                    peg_xy = list(draw_ops.get("pegs") or [])
                    if peg_xy:
                        _insert_pegs_at(peg_xy)
                    time.sleep(0.1)
                    edges_drawn = 0
                    for ed in draw_ops.get("edges") or []:
                        a = ed.get("a") or {}
                        b = ed.get("b") or {}
                        pl_e = self.autocad.create_lwpolyline(
                            [
                                {"x": float(a.get("x", 0.0)), "y": float(a.get("y", 0.0))},
                                {"x": float(b.get("x", 0.0)), "y": float(b.get("y", 0.0))},
                            ],
                            layer="CADA_BOUNDARY",
                            closed=False,
                        )
                        if pl_e.get("success"):
                            edges_drawn += 1
                    # Owner labels sized from parcel spans (meters), not sheet scale_k —
                    # avoids 1:20000-scale text drowning a few-hundred-metre layout.
                    try:
                        _lw = float((geometry.get("scale_debug") or {}).get("boundary_w") or 0.0)
                        _lh = float((geometry.get("scale_debug") or {}).get("boundary_h") or 0.0)
                    except Exception:
                        _lw, _lh = 0.0, 0.0
                    label_h_mp = _multi_parcel_owner_label_height(
                        [list(p.get("points") or []) for p in layout_parcels],
                        layout_w=_lw,
                        layout_h=_lh,
                        native_bearing_height=float(bearing_road_height_native or bearing_road_height or 1.2),
                    )
                    # Mild sheet-aware bump only (never * full scale_k).
                    try:
                        label_h_mp = float(label_h_mp) * max(1.0, min(float(scale_k), 2.5) ** 0.35)
                    except Exception:
                        pass
                    labels_drawn = 0
                    for lab in draw_ops.get("labels") or []:
                        try:
                            self.autocad.add_mtext(
                                str(lab.get("label") or ""),
                                float(lab.get("x", 0.0)),
                                float(lab.get("y", 0.0)),
                                layer="CADA_TEXT",
                                height=float(label_h_mp),
                                width=max(12.0, float(label_h_mp) * 12.0),
                                attachment_point=5,
                            )
                            labels_drawn += 1
                        except Exception:
                            pass
                    geometry["boundary_redrawn"] = edges_drawn > 0
                    geometry["multi_parcel_pegs"] = int(len(peg_xy))
                    geometry["multi_parcel_edges"] = int(edges_drawn)
                    geometry["multi_parcel_labels"] = int(labels_drawn)
                    geometry["multi_parcel_label_height"] = float(label_h_mp)
                    geometry["multi_parcel_drawn_local"] = True
                    # Title AREA from main ring (shoelace) — no single closed entity in edge mode.
                    if title_h and local_pts and len(local_pts) >= 3:
                        try:
                            area_acc = 0.0
                            nlp = len(local_pts)
                            for i in range(nlp):
                                x1 = float(local_pts[i]["x"])
                                y1 = float(local_pts[i]["y"])
                                x2 = float(local_pts[(i + 1) % nlp]["x"])
                                y2 = float(local_pts[(i + 1) % nlp]["y"])
                                area_acc += x1 * y2 - x2 * y1
                            sq_m = abs(area_acc) * 0.5
                            _set_cell(
                                title_h,
                                12,
                                0,
                                _replace_after_label(
                                    _get_cell(title_h, 12), "AREA:-", f"{sq_m:.3f} SQ. MTRS."
                                ),
                            )
                        except Exception:
                            pass

                    # ---- Plot-then-fit: enlarge/position sheet until it ENCLOSES plotted land ----
                    try:
                        geom_bb = self.autocad.get_modelspace_bbox(
                            layers=["CADA_BOUNDARY", "CADA_PILLARS", "CADA_TEXT"]
                        )
                        if not geom_bb.get("success"):
                            geom_bb = self.autocad.get_modelspace_bbox(layers=["CADA_BOUNDARY"])
                        interior_bb2 = self.autocad.get_modelspace_bbox(layers=["CADA_INTERIORBORDER"])
                        if not interior_bb2.get("success"):
                            interior_bb2 = self.autocad.get_modelspace_bbox(
                                layers=["CADA_INTERIORBOUNDARY"]
                            )
                        title_bb2 = template_title_bb_for_scale
                        if not (title_bb2 and title_bb2.get("success")):
                            title_bb2 = self.autocad.get_modelspace_bbox(layers=["CADA_TITLEBLOCK"])
                        win0 = _usable_plot_window_from_bboxes(interior_bb2, title_bb2)
                        if geom_bb.get("success") and win0:
                            g_minx = float(geom_bb.get("minx", 0.0))
                            g_miny = float(geom_bb.get("miny", 0.0))
                            g_maxx = float(geom_bb.get("maxx", 0.0))
                            g_maxy = float(geom_bb.get("maxy", 0.0))
                            g_w = max(1e-6, g_maxx - g_minx)
                            g_h = max(1e-6, g_maxy - g_miny)
                            g_cx = 0.5 * (g_minx + g_maxx)
                            g_cy = 0.5 * (g_miny + g_maxy)
                            # Cartographic margin: enclose all features with breathing room.
                            g_w_pad = max(g_w * 1.12, g_w + 10.0)
                            g_h_pad = max(g_h * 1.12, g_h + 10.0)
                            usable_w2 = max(1e-6, float(win0["width"]))
                            usable_h2 = max(1e-6, float(win0["height"]))
                            iw0 = float(interior_bb2.get("maxx", 0.0)) - float(interior_bb2.get("minx", 0.0))
                            ih0 = float(interior_bb2.get("maxy", 0.0)) - float(interior_bb2.get("miny", 0.0))
                            td_fit = int(template_native_denom or template_denom or 500)
                            n_parcels_fit = max(
                                1,
                                int(len(layout_parcels) or geometry.get("multi_parcel_labels") or 1),
                            )
                            # Primary: cartographic scale from ground span; fit is secondary/capped.
                            chosen2, k2, reason2 = _choose_layout_plan_scale_after_plot(
                                plotted_w=float(g_w),
                                plotted_h=float(g_h),
                                usable_w=float(usable_w2),
                                usable_h=float(usable_h2),
                                interior_w=float(iw0),
                                interior_h=float(ih0),
                                template_denom=td_fit,
                                user_scale_denom=user_scale_denom,
                                parcel_count=n_parcels_fit,
                                allowed_denoms=_CADASTRAL_ALLOWED_SCALES,
                            )
                            carto_cap2 = _allowed_denom_steps_above(
                                _cartographic_denom_for_layout_extent(
                                    g_w,
                                    g_h,
                                    _CADASTRAL_ALLOWED_SCALES,
                                    parcel_count=n_parcels_fit,
                                    dense_title=n_parcels_fit >= 6,
                                ),
                                1,
                                _CADASTRAL_ALLOWED_SCALES,
                            )
                            layers_fit = _cadastral_sheet_translate_layers(profile)
                            # Scale sheet about the land centroid so the frame grows around the layout.
                            # Primary crosshair is excluded so it stays locked to the primary peg.
                            if float(k2) > 1.0 + 1e-9:
                                sc2 = self.autocad.scale_modelspace_by_layers(
                                    float(g_cx), float(g_cy), float(k2), layers_fit
                                )
                                geometry["scale_debug"]["scaled_entities"] = int(
                                    sc2.get("scaled_entities", 0) or 0
                                ) if sc2.get("success") else 0
                                try:
                                    sb_factor2 = float(chosen2) / float(td_fit)
                                    if abs(sb_factor2 - 1.0) > 1e-9:
                                        self.autocad.scale_scalebar_text_values(
                                            sb_factor2, layers=["scalebar", "CADA_SCALEBAR"]
                                        )
                                except Exception:
                                    pass
                            # Position sheet so the usable plot window centers on the land.
                            # Iterate: measure → translate → optionally grow one more scale step.
                            cumulative_k = float(k2)
                            for _fit_i in range(3):
                                interior_now = self.autocad.get_modelspace_bbox(
                                    layers=["CADA_INTERIORBORDER"]
                                )
                                if not interior_now.get("success"):
                                    interior_now = self.autocad.get_modelspace_bbox(
                                        layers=["CADA_INTERIORBOUNDARY"]
                                    )
                                title_now = self.autocad.get_modelspace_bbox(
                                    layers=["CADA_TITLEBLOCK"]
                                )
                                win_now = _usable_plot_window_from_bboxes(interior_now, title_now)
                                geom_now = self.autocad.get_modelspace_bbox(
                                    layers=["CADA_BOUNDARY", "CADA_PILLARS", "CADA_TEXT"]
                                )
                                if not geom_now.get("success"):
                                    geom_now = self.autocad.get_modelspace_bbox(
                                        layers=["CADA_BOUNDARY"]
                                    )
                                if not (win_now and geom_now.get("success")):
                                    break
                                gx0 = float(geom_now.get("minx", 0.0))
                                gy0 = float(geom_now.get("miny", 0.0))
                                gx1 = float(geom_now.get("maxx", 0.0))
                                gy1 = float(geom_now.get("maxy", 0.0))
                                gcx = 0.5 * (gx0 + gx1)
                                gcy = 0.5 * (gy0 + gy1)
                                dx_fit = float(gcx) - float(win_now["center_x"])
                                dy_fit = float(gcy) - float(win_now["center_y"])
                                if abs(dx_fit) > 0.05 or abs(dy_fit) > 0.05:
                                    self.autocad.move_modelspace_by_layers(
                                        dx_fit, dy_fit, layers_fit
                                    )
                                    # Re-measure window after move
                                    interior_now = self.autocad.get_modelspace_bbox(
                                        layers=["CADA_INTERIORBORDER"]
                                    )
                                    if not interior_now.get("success"):
                                        interior_now = self.autocad.get_modelspace_bbox(
                                            layers=["CADA_INTERIORBOUNDARY"]
                                        )
                                    title_now = self.autocad.get_modelspace_bbox(
                                        layers=["CADA_TITLEBLOCK"]
                                    )
                                    win_now = _usable_plot_window_from_bboxes(
                                        interior_now, title_now
                                    )
                                if not win_now:
                                    break
                                # Enclosure test with a small tolerance.
                                pad = 0.5
                                enclosed = (
                                    gx0 >= float(win_now["minx"]) - pad
                                    and gy0 >= float(win_now["miny"]) - pad
                                    and gx1 <= float(win_now["maxx"]) + pad
                                    and gy1 <= float(win_now["maxy"]) + pad
                                )
                                if enclosed:
                                    geometry["scale_debug"]["enclosed"] = True
                                    break
                                # Grow at most toward cartographic cap (never explode to 1:20000).
                                next_denoms = [
                                    d
                                    for d in _CADASTRAL_ALLOWED_SCALES
                                    if int(chosen2) < d <= int(carto_cap2)
                                ]
                                if not next_denoms:
                                    geometry["scale_debug"]["enclosed"] = False
                                    break
                                next_d = int(next_denoms[0])
                                step_k = float(next_d) / float(chosen2)
                                self.autocad.scale_modelspace_by_layers(
                                    gcx, gcy, step_k, layers_fit
                                )
                                try:
                                    self.autocad.scale_scalebar_text_values(
                                        step_k, layers=["scalebar", "CADA_SCALEBAR"]
                                    )
                                except Exception:
                                    pass
                                chosen2 = next_d
                                cumulative_k *= step_k
                                k2 = float(cumulative_k)
                                reason2 = "layout_enclosure_grow"
                            try:
                                self.autocad.execute_command("REGEN")
                            except Exception:
                                pass
                            # Peg symbols follow final plan scale (positions unchanged).
                            try:
                                ms_peg = self.autocad.doc.ModelSpace
                                for ii in range(ms_peg.Count):
                                    ent = ms_peg.Item(ii)
                                    if str(getattr(ent, "Layer", "")).upper() != "CADA_PILLARS":
                                        continue
                                    if "BlockReference" not in str(getattr(ent, "ObjectName", "")):
                                        continue
                                    for attr in ("XScaleFactor", "YScaleFactor", "ZScaleFactor"):
                                        try:
                                            setattr(ent, attr, float(k2))
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                            chosen_denom = int(chosen2)
                            scale_k = float(k2)
                            scale_reason = str(reason2)
                            output_plan_denom = int(chosen2)
                            output_scale_k = float(k2)
                            geometry["output_plan_denom"] = int(output_plan_denom)
                            geometry["output_scale_k"] = float(output_scale_k)
                            try:
                                bearing_road_height = float(bearing_road_height_native) * float(k2)
                            except Exception:
                                pass
                            geometry["scale_debug"].update({
                                "plot_then_fit": True,
                                "plotted_w": float(g_w),
                                "plotted_h": float(g_h),
                                "plotted_w_padded": float(g_w_pad),
                                "plotted_h_padded": float(g_h_pad),
                                "fit_usable_w": float(usable_w2),
                                "fit_usable_h": float(usable_h2),
                                "fit_required_k": float(
                                    max(g_w_pad / usable_w2, g_h_pad / usable_h2)
                                ),
                                "carto_cap": int(carto_cap2),
                                "chosen_denom": int(chosen2),
                                "k": float(k2),
                                "scale_reason": str(reason2),
                            })
                            if title_h:
                                try:
                                    scale_pattern = re.compile(r"1\s*:\s*\d+", re.IGNORECASE)
                                    replacement = f"1:{chosen2}"
                                    tbl = tables_now.get(title_h, {}) if isinstance(tables_now, dict) else {}
                                    rows = int(tbl.get("rows", 25))
                                    cols = int(tbl.get("cols", 2))
                                    main_scale_cell = None
                                    secondary_re = re.compile(r"\bSCALE\b\s*:.*\bto\b", re.IGNORECASE)
                                    main_hint_re = re.compile(r"\bSCALE\b\s*[:-]", re.IGNORECASE)
                                    for r in range(min(rows, 60)):
                                        for c in range(min(cols, 10)):
                                            cell = _get_cell(title_h, r, c) or ""
                                            if not cell.strip():
                                                continue
                                            if secondary_re.search(cell) and scale_pattern.search(cell):
                                                try:
                                                    _set_cell(title_h, r, c, "")
                                                except Exception:
                                                    pass
                                                continue
                                            if (
                                                main_scale_cell is None
                                                and main_hint_re.search(cell)
                                                and scale_pattern.search(cell)
                                            ):
                                                main_scale_cell = (r, c)
                                    if main_scale_cell is None:
                                        main_scale_cell = (8, 0)
                                    mr, mc = main_scale_cell
                                    title_scale_row = int(mr)
                                    cell_main = _get_cell(title_h, mr, mc) or ""
                                    new_cell_main = scale_pattern.sub(replacement, cell_main)
                                    if new_cell_main != cell_main:
                                        _set_cell(title_h, mr, mc, new_cell_main)
                                    elif not (cell_main or "").strip():
                                        _set_cell(title_h, mr, mc, f"SCALE:- {replacement}")
                                except Exception:
                                    pass
                            # Mark so final recenter uses plot-window alignment, not raw frame center.
                            geometry["multi_parcel_sheet_fitted"] = True
                    except Exception as fit_exc:
                        logger.warning("Multi-parcel plot-then-fit scale failed: %s", fit_exc)
                except Exception as mp_exc:
                    logger.warning("Multi-parcel local draw failed, falling back to main ring: %s", mp_exc)
                    _insert_pegs_at(local_pts)
                    pl = self.autocad.create_lwpolyline(local_pts, layer="CADA_BOUNDARY", closed=True)
                    if pl.get("success"):
                        geometry["boundary_redrawn"] = True
            else:
                # Single-parcel: insert peg at every vertex, then one closed boundary polyline.
                _insert_pegs_at(local_pts)
                time.sleep(0.15)
                pl = self.autocad.create_lwpolyline(local_pts, layer="CADA_BOUNDARY", closed=True)
                time.sleep(0.1)
                if pl.get("success"):
                    geometry["boundary_redrawn"] = True
                    a = self.autocad.calculate_entity_area(str(pl.get("handle")))
                    if a.get("success") and title_h:
                        sq_m = float(a.get("area_conversions", {}).get("sq_meters"))
                        _set_cell(
                            title_h,
                            12,
                            0,
                            _replace_after_label(
                                _get_cell(title_h, 12), "AREA:-", f"{sq_m:.3f} SQ. MTRS."
                            ),
                        )

            # Update coordinate tables to primary
            def _replace_first_num(raw: str, val: float) -> str:
                if raw.startswith("{") and raw.endswith("}") and ";" in raw:
                    idx = raw.rfind(";")
                    wrapper = raw[: idx + 1]
                    content = raw[idx + 1 : -1]
                    return wrapper + re.sub(r"[-+]?\d+(?:\.\d+)?", f"{val:.3f}", content, count=1) + "}"
                return re.sub(r"[-+]?\d+(?:\.\d+)?", f"{val:.3f}", raw, count=1)
            if east_h:
                _set_cell(east_h, 0, 0, _replace_first_num(_get_cell(east_h, 0), e0))
            if north_h:
                _set_cell(north_h, 0, 0, _replace_first_num(_get_cell(north_h, 0), n0))

            # Update pillar number tables (CADA_PILLARNUMBERS) to match user prompt and
            # position them near each corresponding boundary vertex (clockwise from primary).
            # Pillar-number entities in the template are TABLE objects on this layer.
            def _parse_pillar_numbers(raw: str) -> List[Dict[str, str]]:
                raw = (raw or "").strip()
                if not raw:
                    return []
                try:
                    from agent.pdf_survey_plan import split_cadastral_pillar_label
                except Exception:
                    split_cadastral_pillar_label = None  # type: ignore
                parts = [p.strip() for p in re.split(r"[,\n]+", raw) if p.strip()]
                out = []
                for p in parts:
                    split = None
                    if split_cadastral_pillar_label is not None:
                        try:
                            split = split_cadastral_pillar_label(p)
                        except Exception:
                            split = None
                    if split:
                        out.append(
                            {
                                "prefix": str(split.get("prefix") or ""),
                                "number": str(split.get("number") or ""),
                            }
                        )
                        continue
                    # Legacy fallback: Letter/Letter + digits only
                    m = re.search(
                        r"([A-Za-z]+\s*/\s*[A-Za-z]+)\s*([A-Za-z0-9]{3,12})\b",
                        p,
                    )
                    if not m:
                        continue
                    prefix = re.sub(r"\s+", "", m.group(1)).upper()
                    num = m.group(2).upper()
                    if not re.search(r"\d", num):
                        continue
                    out.append({"prefix": prefix, "number": num})
                return out

            pn_list = _parse_pillar_numbers(pillar_numbers)
            # Rotate pillar labels with the same offset used for boundary vertices (primary at index 0).
            try:
                _prot = int(geometry.get("primary_vertex_index_original", 0) or 0)
                if _prot and pn_list and len(pn_list) == len(local_pts):
                    pn_list = pn_list[_prot:] + pn_list[:_prot]
                # Match clockwise ring reversal (primary stays at index 0).
                if (
                    geometry.get("ring_reversed_to_clockwise")
                    and pn_list
                    and len(pn_list) == len(local_pts)
                    and len(pn_list) >= 2
                ):
                    pn_list = [pn_list[0]] + list(reversed(pn_list[1:]))
            except Exception:
                pass
            # If the agent extracted fewer pillar numbers than vertices (often due to punctuation/quoting),
            # auto-extend sequentially using the last known prefix/number so we never drop pillar labels.
            # Only when the peg token is pure digits (alphanumeric pegs like AS3459RP are not sequenced).
            try:
                need_n = len(local_pts) if isinstance(local_pts, list) else 0
                if need_n and pn_list and len(pn_list) < need_n:
                    last = pn_list[-1]
                    prefix = str(last.get("prefix") or "").strip() or "SP"
                    num_s = str(last.get("number") or "").strip()
                    if num_s.isdigit():
                        start_num = int(num_s)
                        width = max(len(num_s), 4)
                        k = start_num
                        while len(pn_list) < need_n and k < start_num + 2000:
                            k += 1
                            pn_list.append({"prefix": prefix, "number": f"{k:0{width}d}"})
            except Exception:
                pass
            pn_meta: List[Dict[str, Any]] = []
            if pn_list and not multi_parcel_layout:
                # Compute a "template-typical" offset distance between a peg and its pillar-number table.
                # This makes the placement look like the template: close to the peg, but not on it.
                off = 4.0
                try:
                    # Use the already-copied output drawing as the reference so we never
                    # open the template DWG in AutoCAD.
                    out_doc = self.autocad.doc
                    ms_t = out_doc.ModelSpace
                    t_pegs = []
                    t_tabs = []
                    for ii in range(ms_t.Count):
                        e = ms_t.Item(ii)
                        lyr = str(getattr(e, "Layer", "")).upper()
                        on = str(getattr(e, "ObjectName", ""))
                        if lyr == "CADA_PILLARS" and "BlockReference" in on:
                            ip = getattr(e, "InsertionPoint", None)
                            if ip is not None:
                                t_pegs.append((float(ip[0]), float(ip[1])))
                        if lyr == "CADA_PILLARNUMBERS" and on == "AcDbTable":
                            ip = None
                            for attr in ("InsertionPoint", "Position"):
                                try:
                                    ip = getattr(e, attr, None)
                                    if ip is not None:
                                        break
                                except Exception:
                                    pass
                            if ip is not None:
                                t_tabs.append((float(ip[0]), float(ip[1])))
                    dists = []
                    if t_pegs and t_tabs:
                        for px, py in t_pegs:
                            tx, ty = min(t_tabs, key=lambda t: (t[0] - px) ** 2 + (t[1] - py) ** 2)
                            dists.append(math.hypot(tx - px, ty - py))
                    if dists:
                        dists.sort()
                        off = float(dists[len(dists) // 2])
                except Exception:
                    off = 4.0
                off = max(2.5, min(8.0, off))

                # Reuse pillar-number tables already in the copied output (from template),
                # move them close (but not on) each new pillar, and delete any extras from the template.
                t_res = self.autocad.list_tables(layer="CADA_PILLARNUMBERS")
                pn_tables = (t_res.get("tables") or []) if isinstance(t_res, dict) else []
                # Enforce modern CADA_PILLARNUMBERS column width (9.2 @ 1:500, scaled with plan).
                # Older templates still at 8.0 are widened so long alphanumeric pegs fit on one line.
                try:
                    target_pn_w = _ideal_pillar_number_table_width(
                        template_denom=int(template_denom or _CADASTRAL_TEMPLATE_REF_DENOM),
                        chosen_denom=int(
                            output_plan_denom
                            or chosen_denom
                            or template_denom
                            or _CADASTRAL_TEMPLATE_REF_DENOM
                        ),
                    )
                    widened = 0
                    for t in pn_tables:
                        h0 = str((t or {}).get("handle") or "")
                        if not h0:
                            continue
                        cur = self.autocad.get_table_column_width(h0, 0)
                        cur_w = float((cur or {}).get("width") or 0.0)
                        if cur_w <= 0.0 or cur_w + 1e-6 < float(target_pn_w):
                            sw = self.autocad.set_table_column_width(h0, 0, float(target_pn_w))
                            if (sw or {}).get("success"):
                                widened += 1
                    if widened:
                        geometry.setdefault("scale_debug", {})["pillar_number_width"] = {
                            "target": float(target_pn_w),
                            "widened_tables": int(widened),
                            "ref_at_1_500": float(_CADASTRAL_TEMPLATE_PILLARNUMBER_REF_W),
                        }
                except Exception:
                    pass
                used_handles: set[str] = set()
                pn_meta: List[Dict[str, Any]] = []
                cxp = sum(p["x"] for p in local_pts) / len(local_pts)
                cyp = sum(p["y"] for p in local_pts) / len(local_pts)
                pillar_box_bbs: List[Dict[str, Any]] = []
                try:
                    pbb = self.autocad.list_entity_bboxes(
                        layers=["CADA_PILLARS"],
                        object_names=["AcDbBlockReference"],
                    )
                    if (pbb or {}).get("success"):
                        pillar_box_bbs = list(pbb.get("bboxes") or [])
                except Exception:
                    pillar_box_bbs = []

                def _dist2(t, vx, vy):
                    ip = (t.get("insertion_point") or {})
                    tx, ty = float(ip.get("x", 0.0)), float(ip.get("y", 0.0))
                    return (tx - vx) ** 2 + (ty - vy) ** 2

                def _nearest_pillar_bb(vx: float, vy: float) -> Optional[Dict[str, Any]]:
                    if not pillar_box_bbs:
                        return None
                    def _bb_d2(b: Dict[str, Any]) -> float:
                        mn = b.get("min") or {}
                        mx = b.get("max") or {}
                        cx = (float(mn.get("x", 0.0)) + float(mx.get("x", 0.0))) / 2.0
                        cy = (float(mn.get("y", 0.0)) + float(mx.get("y", 0.0))) / 2.0
                        return (cx - vx) ** 2 + (cy - vy) ** 2
                    return min(pillar_box_bbs, key=_bb_d2)

                for i_v, v in enumerate(local_pts[: len(pn_list)]):
                    vx, vy = float(v["x"]), float(v["y"])
                    cand = [t for t in pn_tables if t.get("handle") and str(t.get("handle")) not in used_handles]
                    if not cand:
                        # Template may contain fewer pillar-number tables than parcel vertices.
                        # Clone an existing table so all pillar numbers are preserved.
                        try:
                            seed = None
                            if pn_tables:
                                # Prefer the last used handle as it matches styling/scale in this drawing.
                                for hh in reversed(list(used_handles)):
                                    if hh:
                                        seed = hh
                                        break
                                if seed is None:
                                    seed = str((pn_tables[0] or {}).get("handle") or "")
                            if seed:
                                c = self.autocad.copy_entity_by_handle(seed, dx=0.0, dy=0.0, layer="CADA_PILLARNUMBERS")
                                if (c or {}).get("success") and c.get("handle"):
                                    new_h = str(c.get("handle"))
                                    try:
                                        tw = _ideal_pillar_number_table_width(
                                            template_denom=int(
                                                template_denom or _CADASTRAL_TEMPLATE_REF_DENOM
                                            ),
                                            chosen_denom=int(
                                                output_plan_denom
                                                or chosen_denom
                                                or template_denom
                                                or _CADASTRAL_TEMPLATE_REF_DENOM
                                            ),
                                        )
                                        self.autocad.set_table_column_width(new_h, 0, float(tw))
                                    except Exception:
                                        pass
                                    pn_tables.append({"handle": new_h, "layer": "CADA_PILLARNUMBERS", "insertion_point": c.get("insertion_point") or {}})
                                    cand = [t for t in pn_tables if t.get("handle") and str(t.get("handle")) not in used_handles]
                        except Exception:
                            cand = cand
                        if not cand:
                            # Hard fallback: create an MTEXT-based pillar-number label so we never drop a pillar number.
                            try:
                                lbl = f"{pn_list[i_v]['prefix']}\\P{pn_list[i_v]['number']}"
                                peg_tok = str(pn_list[i_v].get("number") or "")
                                # Wider box for longer Nigerian peg tokens (up to ~9+ chars).
                                mtext_w = 12.0 if len(peg_tok) <= 5 else min(28.0, 8.0 + 1.6 * len(peg_tok))
                                self.autocad.add_mtext(
                                    f"{{\\fVerdana|b0|i0|c0|p34;{lbl}}}",
                                    vx + 2.0,
                                    vy + 2.0,
                                    layer="CADA_PILLARNUMBERS",
                                    rotation_rad=0.0,
                                    height=max(0.5, float(bearing_road_height) * 0.9),
                                    width=mtext_w,
                                    attachment_point=1,  # top-left
                                )
                            except Exception:
                                pass
                            continue
                    best = min(cand, key=lambda t: _dist2(t, vx, vy))
                    h = str(best.get("handle"))
                    used_handles.add(h)
                    try:
                        _set_cell(h, 0, 0, _mtxt_replace(_get_cell(h, 0), pn_list[i_v]["prefix"]))
                        _set_cell(h, 1, 0, _mtxt_replace(_get_cell(h, 1), pn_list[i_v]["number"]))
                    except Exception:
                        pass
                    # Place the TABLE close to the pillar but not on it.
                    # TABLE insertion points are often at a corner, so use its bounding box
                    # to position the *nearest table edge* a small gap away from the pillar.
                    dxv, dyv = vx - cxp, vy - cyp
                    Lvv = math.hypot(dxv, dyv) or 1.0
                    ux, uy = dxv / Lvv, dyv / Lvv  # outward from centroid
                    try:
                        pn_meta.append({"handle": h, "vx": vx, "vy": vy, "ux": ux, "uy": uy, "off": float(off)})
                    except Exception:
                        pass
                    # Gap target (survey drafting): ~1.0 unit from peg, except primary peg ~1.5 units.
                    gap = 1.5 if i_v == primary_idx else 1.0
                    try:
                        ms = self.autocad.doc.ModelSpace
                        ent = None
                        for ii in range(ms.Count):
                            e = ms.Item(ii)
                            if getattr(e, "Handle", None) == h:
                                ent = e
                                break
                        if ent is not None:
                            # insertion point
                            ip = None
                            for attr in ("InsertionPoint", "Position"):
                                try:
                                    ip = getattr(ent, attr, None)
                                    if ip is not None:
                                        break
                                except Exception:
                                    pass
                            ix, iy = (float(ip[0]), float(ip[1])) if ip is not None else (0.0, 0.0)
                            # bbox
                            bb = ent.GetBoundingBox()
                            pmin, pmax = bb[0], bb[1]
                            minx, miny = float(pmin[0]), float(pmin[1])
                            maxx, maxy = float(pmax[0]), float(pmax[1])
                            cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
                            # center offset relative to insertion
                            dcx, dcy = cx - ix, cy - iy

                            def _aabb(minx0: float, miny0: float, maxx0: float, maxy0: float) -> Dict[str, Dict[str, float]]:
                                return {"min": {"x": minx0, "y": miny0}, "max": {"x": maxx0, "y": maxy0}}

                            def _shift_bbox(bb0: Dict[str, Dict[str, float]], dx0: float, dy0: float) -> Dict[str, Dict[str, float]]:
                                mn = bb0.get("min") or {}
                                mx = bb0.get("max") or {}
                                return _aabb(
                                    float(mn.get("x", 0.0)) + dx0,
                                    float(mn.get("y", 0.0)) + dy0,
                                    float(mx.get("x", 0.0)) + dx0,
                                    float(mx.get("y", 0.0)) + dy0,
                                )

                            def _aabb_overlaps_local(a: Dict[str, Dict[str, float]], b: Dict[str, Dict[str, float]], pad0: float = 0.0) -> bool:
                                amin, amax = a.get("min") or {}, a.get("max") or {}
                                bmin, bmax = b.get("min") or {}, b.get("max") or {}
                                return not (
                                    float(amax.get("x", 0.0)) + pad0 < float(bmin.get("x", 0.0)) - pad0
                                    or float(amin.get("x", 0.0)) - pad0 > float(bmax.get("x", 0.0)) + pad0
                                    or float(amax.get("y", 0.0)) + pad0 < float(bmin.get("y", 0.0)) - pad0
                                    or float(amin.get("y", 0.0)) - pad0 > float(bmax.get("y", 0.0)) + pad0
                                )

                            def _point_in_poly(px0: float, py0: float) -> bool:
                                inside0 = False
                                n0 = len(local_pts)
                                j0 = n0 - 1
                                for i0 in range(n0):
                                    xi0, yi0 = float(local_pts[i0]["x"]), float(local_pts[i0]["y"])
                                    xj0, yj0 = float(local_pts[j0]["x"]), float(local_pts[j0]["y"])
                                    hit0 = ((yi0 > py0) != (yj0 > py0)) and (
                                        px0 < (xj0 - xi0) * (py0 - yi0) / ((yj0 - yi0) or 1e-12) + xi0
                                    )
                                    if hit0:
                                        inside0 = not inside0
                                    j0 = i0
                                return inside0

                            def _seg_int(ax: float, ay: float, bx: float, by: float, cx0: float, cy0: float, dx0: float, dy0: float) -> bool:
                                def _orient(px1: float, py1: float, px2: float, py2: float, px3: float, py3: float) -> float:
                                    return (px2 - px1) * (py3 - py1) - (py2 - py1) * (px3 - px1)
                                o1 = _orient(ax, ay, bx, by, cx0, cy0)
                                o2 = _orient(ax, ay, bx, by, dx0, dy0)
                                o3 = _orient(cx0, cy0, dx0, dy0, ax, ay)
                                o4 = _orient(cx0, cy0, dx0, dy0, bx, by)
                                return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

                            def _crosses_traverse(bb0: Dict[str, Dict[str, float]]) -> bool:
                                mn = bb0.get("min") or {}
                                mx = bb0.get("max") or {}
                                minx0, miny0 = float(mn.get("x", 0.0)), float(mn.get("y", 0.0))
                                maxx0, maxy0 = float(mx.get("x", 0.0)), float(mx.get("y", 0.0))
                                corners0 = [(minx0, miny0), (minx0, maxy0), (maxx0, miny0), (maxx0, maxy0)]
                                if any(_point_in_poly(px0, py0) for (px0, py0) in corners0):
                                    return True
                                redges = [
                                    (minx0, miny0, minx0, maxy0),
                                    (minx0, maxy0, maxx0, maxy0),
                                    (maxx0, maxy0, maxx0, miny0),
                                    (maxx0, miny0, minx0, miny0),
                                ]
                                for kk in range(len(local_pts)):
                                    p_a = local_pts[kk]
                                    p_b = local_pts[(kk + 1) % len(local_pts)]
                                    ax0, ay0 = float(p_a["x"]), float(p_a["y"])
                                    bx0, by0 = float(p_b["x"]), float(p_b["y"])
                                    for (rx1, ry1, rx2, ry2) in redges:
                                        if _seg_int(ax0, ay0, bx0, by0, rx1, ry1, rx2, ry2):
                                            return True
                                return False

                            cur_bb = _aabb(minx, miny, maxx, maxy)
                            # extent along preferred direction for the current table box
                            corners = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
                            ext = 0.0
                            for (qx, qy) in corners:
                                proj = abs((qx - cx) * ux + (qy - cy) * uy)
                                if proj > ext:
                                    ext = proj

                            # desired center should clear the pillar box edge, not the pillar point.
                            pillar_bb = _nearest_pillar_bb(vx, vy)
                            p_cx, p_cy = vx, vy
                            p_ext = 0.0
                            own_pillar_bb = pillar_bb
                            if pillar_bb is not None:
                                pb_min = pillar_bb.get("min") or {}
                                pb_max = pillar_bb.get("max") or {}
                                pminx, pminy = float(pb_min.get("x", vx)), float(pb_min.get("y", vy))
                                pmaxx, pmaxy = float(pb_max.get("x", vx)), float(pb_max.get("y", vy))
                                p_cx = (pminx + pmaxx) / 2.0
                                p_cy = (pminy + pmaxy) / 2.0
                                for (qx, qy) in [(pminx, pminy), (pminx, pmaxy), (pmaxx, pminy), (pmaxx, pmaxy)]:
                                    proj = abs((qx - p_cx) * ux + (qy - p_cy) * uy)
                                    if proj > p_ext:
                                        p_ext = proj

                            # Candidate search: stay near the peg, prefer outside the parcel, and never cross the peg box.
                            placed_bbs: List[Dict[str, Any]] = []
                            for uu in used_handles:
                                if str(uu).upper() == str(h).upper():
                                    continue
                                for jj in range(ms.Count):
                                    ee = ms.Item(jj)
                                    if str(getattr(ee, "Handle", "")).upper() != str(uu).upper():
                                        continue
                                    try:
                                        bb_u = ee.GetBoundingBox()
                                        pmn, pmx = bb_u[0], bb_u[1]
                                        placed_bbs.append(_aabb(float(pmn[0]), float(pmn[1]), float(pmx[0]), float(pmx[1])))
                                    except Exception:
                                        pass
                                    break

                            base_r = p_ext + gap + ext
                            perp_x, perp_y = -uy, ux
                            best_pos = None
                            best_cost = 1e18
                            for r_mult in (1.0, 1.15, 1.35, 1.6):
                                rr = base_r * r_mult
                                for ang_deg in (0, 20, -20, 40, -40, 60, -60, 85, -85, 110, -110, 145, -145, 180):
                                    th = math.radians(float(ang_deg))
                                    dux = ux * math.cos(th) + perp_x * math.sin(th)
                                    duy = uy * math.cos(th) + perp_y * math.sin(th)
                                    tcx = p_cx + dux * rr
                                    tcy = p_cy + duy * rr
                                    cand_bb = _shift_bbox(cur_bb, tcx - cx, tcy - cy)
                                    cost = 0.0
                                    if own_pillar_bb is not None and _aabb_overlaps_local(cand_bb, own_pillar_bb, pad0=0.05):
                                        cost += 10000.0
                                    for other_pb in pillar_box_bbs:
                                        if own_pillar_bb is not None and other_pb is own_pillar_bb:
                                            continue
                                        if _aabb_overlaps_local(cand_bb, other_pb, pad0=0.05):
                                            cost += 5000.0
                                    if _crosses_traverse(cand_bb):
                                        cost += 10000.0
                                    for pb in placed_bbs:
                                        if _aabb_overlaps_local(cand_bb, pb, pad0=0.1):
                                            cost += 2500.0
                                    # prefer outward / shortest move
                                    cost += abs(float(ang_deg)) * 2.0
                                    cost += (r_mult - 1.0) * 200.0
                                    if cost < best_cost:
                                        best_cost = cost
                                        best_pos = (tcx, tcy)
                                    if cost <= 1e-6:
                                        break
                                if best_cost <= 1e-6:
                                    break

                            if best_pos is None:
                                best_pos = (p_cx + ux * base_r, p_cy + uy * base_r)

                            tcx, tcy = best_pos
                            tx = tcx - dcx
                            ty = tcy - dcy
                            self.autocad.move_entity_to_xy(h, tx, ty)
                        else:
                            # fallback: simple outward offset
                            self.autocad.move_entity_to_xy(h, vx + ux * 4.0, vy + uy * 4.0)
                    except Exception:
                        # fallback: simple outward offset
                        self.autocad.move_entity_to_xy(h, vx + ux * 4.0, vy + uy * 4.0)

                for t in pn_tables:
                    h = str(t.get("handle") or "")
                    if h and h not in used_handles:
                        self.autocad.delete_entity_by_handle(h)

            # Bearings/distances (DD° MM' only) aligned to each edge, template-like MTEXT wrapper and height 1.2
            # Re-ensure active document before drawing text (avoids one bearing/distance drawn wrong if COM glitched)
            try:
                self.autocad._ensure_active_document()
                time.sleep(0.1)
            except Exception:
                pass
            # Use scaled bearing/road height (no max cap so e.g. 1:500→1:10000 gives height 24)
            _bd_height = max(0.5, float(bearing_road_height))
            def _bearing_ddmm(az_deg: float, hard_spaces: int = 1) -> str:
                az_deg = az_deg % 360.0
                d = int(az_deg)
                m = int(round((az_deg - d) * 60.0))
                if m == 60:
                    d = (d + 1) % 360
                    m = 0
                # Use AutoCAD MTEXT "hard space" (\\~) so the bearing never wraps mid-token.
                # Example: 143°~05'
                hs = max(1, int(hard_spaces or 1))
                return f"{d:03d}°" + ("\\~" * hs) + f"{m:02d}'"
            # Fetch interior border bbox once for clamping projecting-arrow text inside the drawing area.
            _interior_bb = None
            try:
                _ibb = self.autocad.get_modelspace_bbox(layers=["CADA_INTERIORBORDER"])
                if not (_ibb or {}).get("success"):
                    _ibb = self.autocad.get_modelspace_bbox(layers=["CADA_BORDER"], prefer_largest=True)
                if (_ibb or {}).get("success"):
                    # get_modelspace_bbox returns flat minx/miny/maxx/maxy; clamp helpers expect nested min/max.
                    _interior_bb = {
                        **dict(_ibb),
                        "min": {
                            "x": float(_ibb.get("minx", 0.0)),
                            "y": float(_ibb.get("miny", 0.0)),
                        },
                        "max": {
                            "x": float(_ibb.get("maxx", 0.0)),
                            "y": float(_ibb.get("maxy", 0.0)),
                        },
                    }
            except Exception:
                pass

            # Minimum leg length (in drawing units) below which a projecting arrow is used.
            # Threshold: the bearing string "DDD° MM'" at 1 hard-space is roughly 8 chars wide;
            # stacked with distance, estimate ~9 char-widths at 0.6*height per char.
            _min_text_span = 9.0 * 0.6 * _bd_height

            # Smarter placement: avoid mixing with other plan contents by checking AABB collisions
            # against existing annotation (pillar-number tables, coordinate tables) and already-placed labels.
            _occupied_bboxes: List[Dict[str, Any]] = []
            try:
                bb = self.autocad.list_entity_bboxes(
                    layers=[
                        "CADA_PILLARNUMBERS",
                        "CADA_COORDINATES",
                        "CADA_NORTHCOORDINATES",
                        "CADA_EASTCOORDINATES",
                        "CADA_ROAD",
                        "CADA_TITLEBLOCK",
                    ],
                    object_names=["AcDbTable", "AcDbText", "AcDbMText"],
                )
                if (bb or {}).get("success"):
                    _occupied_bboxes = list(bb.get("bboxes") or [])
            except Exception:
                _occupied_bboxes = []
            _placed_label_bboxes: List[Dict[str, Dict[str, float]]] = []

            def _aabb_overlaps(a: Dict[str, Dict[str, float]], b: Dict[str, Dict[str, float]], pad: float = 0.0) -> bool:
                amin, amax = a.get("min") or {}, a.get("max") or {}
                bmin, bmax = b.get("min") or {}, b.get("max") or {}
                return not (
                    float(amax.get("x", 0.0)) + pad < float(bmin.get("x", 0.0)) - pad
                    or float(amin.get("x", 0.0)) - pad > float(bmax.get("x", 0.0)) + pad
                    or float(amax.get("y", 0.0)) + pad < float(bmin.get("y", 0.0)) - pad
                    or float(amin.get("y", 0.0)) - pad > float(bmax.get("y", 0.0)) + pad
                )

            def _aabb_for_centered_rect(cx: float, cy: float, w: float, h: float, rot_rad: float) -> Dict[str, Dict[str, float]]:
                hw, hh = 0.5 * float(w), 0.5 * float(h)
                c, s = math.cos(float(rot_rad)), math.sin(float(rot_rad))
                # Four corners around origin
                pts = [(-hw, -hh), (-hw, hh), (hw, -hh), (hw, hh)]
                xs = []
                ys = []
                for (px, py) in pts:
                    rx = cx + px * c - py * s
                    ry = cy + px * s + py * c
                    xs.append(rx)
                    ys.append(ry)
                return {"min": {"x": min(xs), "y": min(ys)}, "max": {"x": max(xs), "y": max(ys)}}

            def _aabb_inside_interior(a: Dict[str, Dict[str, float]], margin: float) -> bool:
                if _interior_bb is None:
                    return True
                ib_min = _interior_bb.get("min") or {}
                ib_max = _interior_bb.get("max") or {}
                xmin = float(ib_min.get("x", -1e18)) + margin
                ymin = float(ib_min.get("y", -1e18)) + margin
                xmax = float(ib_max.get("x", 1e18)) - margin
                ymax = float(ib_max.get("y", 1e18)) - margin
                amin, amax = a.get("min") or {}, a.get("max") or {}
                return (
                    float(amin.get("x", 0.0)) >= xmin
                    and float(amax.get("x", 0.0)) <= xmax
                    and float(amin.get("y", 0.0)) >= ymin
                    and float(amax.get("y", 0.0)) <= ymax
                )

            def _point_in_polygon(px: float, py: float, poly: List[Dict[str, float]]) -> bool:
                inside = False
                n = len(poly)
                if n < 3:
                    return False
                j = n - 1
                for i in range(n):
                    xi, yi = float(poly[i]["x"]), float(poly[i]["y"])
                    xj, yj = float(poly[j]["x"]), float(poly[j]["y"])
                    try:
                        hit = ((yi > py) != (yj > py)) and (
                            px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-12) + xi
                        )
                    except Exception:
                        hit = False
                    if hit:
                        inside = not inside
                    j = i
                return inside

            def _segments_intersect(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, dx: float, dy: float) -> bool:
                def _orient(px1: float, py1: float, px2: float, py2: float, px3: float, py3: float) -> float:
                    return (px2 - px1) * (py3 - py1) - (py2 - py1) * (px3 - px1)

                def _on_segment(px1: float, py1: float, px2: float, py2: float, qx: float, qy: float) -> bool:
                    return (
                        min(px1, px2) - 1e-9 <= qx <= max(px1, px2) + 1e-9
                        and min(py1, py2) - 1e-9 <= qy <= max(py1, py2) + 1e-9
                    )

                o1 = _orient(ax, ay, bx, by, cx, cy)
                o2 = _orient(ax, ay, bx, by, dx, dy)
                o3 = _orient(cx, cy, dx, dy, ax, ay)
                o4 = _orient(cx, cy, dx, dy, bx, by)

                if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
                    return True
                if abs(o1) <= 1e-9 and _on_segment(ax, ay, bx, by, cx, cy):
                    return True
                if abs(o2) <= 1e-9 and _on_segment(ax, ay, bx, by, dx, dy):
                    return True
                if abs(o3) <= 1e-9 and _on_segment(cx, cy, dx, dy, ax, ay):
                    return True
                if abs(o4) <= 1e-9 and _on_segment(cx, cy, dx, dy, bx, by):
                    return True
                return False

            def _aabb_crosses_traverse(a: Dict[str, Dict[str, float]], poly: List[Dict[str, float]]) -> bool:
                amin, amax = a.get("min") or {}, a.get("max") or {}
                minx, miny = float(amin.get("x", 0.0)), float(amin.get("y", 0.0))
                maxx, maxy = float(amax.get("x", 0.0)), float(amax.get("y", 0.0))
                corners = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
                if any(_point_in_polygon(cx, cy, poly) for (cx, cy) in corners):
                    return True
                rect_edges = [
                    (minx, miny, minx, maxy),
                    (minx, maxy, maxx, maxy),
                    (maxx, maxy, maxx, miny),
                    (maxx, miny, minx, miny),
                ]
                n = len(poly)
                for i in range(n):
                    p1 = poly[i]
                    p2 = poly[(i + 1) % n]
                    x1, y1 = float(p1["x"]), float(p1["y"])
                    x2, y2 = float(p2["x"]), float(p2["y"])
                    for (rx1, ry1, rx2, ry2) in rect_edges:
                        if _segments_intersect(x1, y1, x2, y2, rx1, ry1, rx2, ry2):
                            return True
                return False

            # Multi-parcel layouts already flagged above — skip bearing/distance COM work.
            if multi_parcel_layout:
                geometry["bearing_mtext"] = 0

            for i in range(0 if multi_parcel_layout else len(local_pts)):
                p1 = local_pts[i]
                p2 = local_pts[(i + 1) % len(local_pts)]
                dx = p2["x"] - p1["x"]
                dy = p2["y"] - p1["y"]
                L_geom = math.hypot(dx, dy)
                if L_geom <= 1e-6:
                    continue
                # For start-coordinate + bearing/distance traverses, preserve the adjusted leg
                # bearings/distances for annotation even after the duplicate closure vertex is removed.
                if bearing_distance_legs and len(bearing_distance_legs) == len(local_pts):
                    leg_meta = bearing_distance_legs[i]
                    az = float(leg_meta.get("bearing_deg", (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0)) % 360.0
                    L_disp = float(leg_meta.get("distance", L_geom))
                else:
                    az = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
                    L_disp = L_geom
                az_orient = az if az <= 180.0 else (az - 180.0)
                rot_deg = (90.0 - az_orient) % 360.0
                rot = math.radians(rot_deg)
                midx = (p1["x"] + p2["x"]) / 2.0
                midy = (p1["y"] + p2["y"]) / 2.0

                poly_cx = sum(p["x"] for p in local_pts) / len(local_pts)
                poly_cy = sum(p["y"] for p in local_pts) / len(local_pts)
                vx = poly_cx - midx
                vy = poly_cy - midy
                n1x, n1y = dy / L_geom, -dx / L_geom
                n2x, n2y = -dy / L_geom, dx / L_geom
                if (n1x * vx + n1y * vy) >= (n2x * vx + n2y * vy):
                    inx, iny = n1x, n1y
                else:
                    inx, iny = n2x, n2y
                outx, outy = -inx, -iny

                use_projecting_arrow = L_geom < _min_text_span

                if not use_projecting_arrow:
                    # --- Normal on-leg placement (unchanged logic) ---
                    target_span = 0.6 * L_geom
                    base_span_est = 4.6 * _bd_height
                    space_span_est = 0.6 * _bd_height
                    hs = 1
                    if target_span > base_span_est and space_span_est > 1e-9:
                        hs = 1 + int(math.ceil((target_span - base_span_est) / space_span_est))
                        hs = max(1, min(hs, 60))
                    bearing_str = _bearing_ddmm(az, hard_spaces=hs)
                    dist_str = f"{L_disp:.2f}m"
                    first_line_out = -math.sin(rot) * outx + math.cos(rot) * outy
                    if first_line_out > 0:
                        stacked_text = f"{{\\fVerdana|b0|i0|c0|p34;{bearing_str}\\P{dist_str}}}"
                    else:
                        stacked_text = f"{{\\fVerdana|b0|i0|c0|p34;{dist_str}\\P{bearing_str}}}"
                    w = min(max(2.0, 0.75 * L_geom), 0.95 * L_geom)
                    h_est = 2.2 * _bd_height
                    margin_bd = 2.0 * _bd_height
                    tcx, tcy = midx, midy
                    tw = float(w)
                    try:
                        if _interior_bb is not None and not _aabb_inside_interior(
                            _aabb_for_centered_rect(tcx, tcy, tw, h_est, rot), margin_bd
                        ):
                            for _nz in range(36):
                                tcx += 0.18 * (poly_cx - tcx)
                                tcy += 0.18 * (poly_cy - tcy)
                                if _aabb_inside_interior(_aabb_for_centered_rect(tcx, tcy, tw, h_est, rot), margin_bd):
                                    break
                            else:
                                tw2 = tw
                                for _sw in range(22):
                                    tw2 *= 0.9
                                    tw2 = max(2.0, tw2)
                                    if _aabb_inside_interior(_aabb_for_centered_rect(tcx, tcy, tw2, h_est, rot), margin_bd):
                                        tw = tw2
                                        break
                    except Exception:
                        tcx, tcy, tw = midx, midy, float(w)
                    self.autocad.add_mtext(
                        stacked_text,
                        tcx,
                        tcy,
                        layer="CADA_BEARING_DIST",
                        rotation_rad=rot,
                        height=_bd_height,
                        width=tw,
                        attachment_point=5,
                    )
                    try:
                        _placed_label_bboxes.append(_aabb_for_centered_rect(tcx, tcy, tw, h_est, rot))
                    except Exception:
                        pass
                else:
                    # --- Smart leader for short legs ---
                    # Prefer an L-shaped leader with a horizontal (90/270) branch; if that collides or
                    # approaches the border, progressively extend and try alternative orientations.

                    bearing_str = _bearing_ddmm(az, hard_spaces=1)
                    dist_str = f"{L_disp:.2f}m"
                    stacked_text = f"{{\\fVerdana|b0|i0|c0|p34;{bearing_str}\\P{dist_str}}}"

                    def _clamp_xy(x: float, y: float, margin: float) -> Tuple[float, float]:
                        if _interior_bb is None:
                            return x, y
                        ib_min = _interior_bb.get("min") or {}
                        ib_max = _interior_bb.get("max") or {}
                        xmin = float(ib_min.get("x", -1e18)) + margin
                        ymin = float(ib_min.get("y", -1e18)) + margin
                        xmax = float(ib_max.get("x", 1e18)) - margin
                        ymax = float(ib_max.get("y", 1e18)) - margin
                        return max(xmin, min(xmax, x)), max(ymin, min(ymax, y))

                    stem_base = max(_min_text_span * 0.9, 6.0 * _bd_height)
                    branch_base = max(_min_text_span * 1.35, 10.0 * _bd_height)
                    pad = 1.25 * _bd_height
                    margin = 2.0 * _bd_height

                    # Unit directions along the leg vector.
                    # NOTE: Use L_geom (the computed leg length) rather than an undefined `L`.
                    t_ux, t_uy = dx / L_geom, dy / L_geom
                    stem_dirs = [
                        (outx, outy),            # outward normal
                        (-outx, -outy),          # inward normal
                        (t_ux, t_uy),            # along the leg
                        (-t_ux, -t_uy),          # reverse along the leg
                    ]
                    # Branch options in preference order: horizontal (E/W), vertical (N/S), then along-leg.
                    branch_opts = [
                        (1.0, 0.0, 0.0),     # East, rotation 0
                        (-1.0, 0.0, 0.0),    # West, rotation 0
                        (0.0, 1.0, math.pi / 2.0),   # North, rotation 90°
                        (0.0, -1.0, math.pi / 2.0),  # South, rotation 90°
                        (t_ux, t_uy, rot),   # along leg
                        (-t_ux, -t_uy, rot), # reverse along leg (rotation kept readable via rot)
                    ]

                    chosen = None
                    for (sdx, sdy) in stem_dirs:
                        for sm in (1.0, 1.6, 2.4, 3.4):
                            stem_len = stem_base * sm
                            ex = midx + sdx * stem_len
                            ey = midy + sdy * stem_len
                            ex, ey = _clamp_xy(ex, ey, margin)
                            for (bdx, bdy, text_rot) in branch_opts:
                                for bm in (1.0, 1.35, 1.8):
                                    bl = branch_base * bm
                                    bx = ex + bdx * bl
                                    by = ey + bdy * bl
                                    bx2, by2 = _clamp_xy(bx, by, margin)
                                    actual_bl = math.hypot(bx2 - ex, by2 - ey)
                                    if actual_bl < 0.55 * bl:
                                        continue

                                    tcx = (ex + bx2) / 2.0
                                    tcy = (ey + by2) / 2.0
                                    tw = max(2.0, actual_bl * 0.95)
                                    th = 2.2 * _bd_height
                                    aabb = _aabb_for_centered_rect(tcx, tcy, tw, th, text_rot)
                                    if not _aabb_inside_interior(aabb, margin):
                                        continue
                                    bad = False
                                    for ob in _occupied_bboxes:
                                        if _aabb_overlaps(aabb, ob, pad=pad):
                                            bad = True
                                            break
                                    if not bad:
                                        for pb in _placed_label_bboxes:
                                            if _aabb_overlaps(aabb, pb, pad=pad):
                                                bad = True
                                                break
                                    if bad:
                                        continue

                                    chosen = {
                                        "elbow": (ex, ey),
                                        "end": (bx2, by2),
                                        "text": (tcx, tcy),
                                        "text_rot": float(text_rot),
                                        "text_w": float(tw),
                                        "aabb": aabb,
                                    }
                                    break
                                if chosen:
                                    break
                            if chosen:
                                break
                        if chosen:
                            break
                    if not chosen:
                        # Fallback that should always succeed: long outward stem + horizontal branch with clamping.
                        ex = midx + outx * (stem_base * 3.4)
                        ey = midy + outy * (stem_base * 3.4)
                        ex, ey = _clamp_xy(ex, ey, margin)
                        bx = ex + (1.0 if outx >= 0 else -1.0) * (branch_base * 1.8)
                        by = ey
                        bx, by = _clamp_xy(bx, by, margin)
                        tcx = (ex + bx) / 2.0
                        tcy = (ey + by) / 2.0
                        tw = max(2.0, math.hypot(bx - ex, by - ey) * 0.95)
                        th = 2.2 * _bd_height
                        aabb = _aabb_for_centered_rect(tcx, tcy, tw, th, 0.0)
                        chosen = {"elbow": (ex, ey), "end": (bx, by), "text": (tcx, tcy), "text_rot": 0.0, "text_w": tw, "aabb": aabb}

                    elbow_x, elbow_y = chosen["elbow"]
                    branch_end_x, branch_end_y = chosen["end"]
                    text_cx, text_cy = chosen["text"]
                    text_rot = chosen["text_rot"]
                    text_w = chosen["text_w"]

                    # Draw leader polyline: midpoint → elbow → end.
                    try:
                        self.autocad.create_lwpolyline(
                            [
                                {"x": midx, "y": midy},
                                {"x": elbow_x, "y": elbow_y},
                                {"x": branch_end_x, "y": branch_end_y},
                            ],
                            layer="CADA_BEARING_DIST",
                            closed=False,
                        )
                    except Exception:
                        pass

                    # Arrowhead at the boundary midpoint, pointing along the first leader segment.
                    try:
                        arrow_size = _bd_height * 0.6
                        stem_dx = elbow_x - midx
                        stem_dy = elbow_y - midy
                        stem_L = math.hypot(stem_dx, stem_dy) or 1.0
                        su, sv = stem_dx / stem_L, stem_dy / stem_L
                        sp, sq = -sv, su
                        arrow_pts = [
                            {"x": midx, "y": midy},
                            {"x": midx + su * arrow_size + sp * arrow_size * 0.35, "y": midy + sv * arrow_size + sq * arrow_size * 0.35},
                            {"x": midx + su * arrow_size - sp * arrow_size * 0.35, "y": midy + sv * arrow_size - sq * arrow_size * 0.35},
                        ]
                        self.autocad.create_lwpolyline(arrow_pts, layer="CADA_BEARING_DIST", closed=True)
                    except Exception:
                        pass

                    # Place MTEXT on the second segment, oriented to that segment's bearing.
                    try:
                        self.autocad.add_mtext(
                            stacked_text,
                            text_cx,
                            text_cy,
                            layer="CADA_BEARING_DIST",
                            rotation_rad=float(text_rot),
                            height=_bd_height,
                            width=float(text_w),
                            attachment_point=5,
                        )
                    except Exception:
                        # Last-resort: try horizontal at elbow
                        try:
                            self.autocad.add_mtext(
                                stacked_text,
                                elbow_x,
                                elbow_y,
                                layer="CADA_BEARING_DIST",
                                rotation_rad=0.0,
                                height=_bd_height,
                                width=max(2.0, branch_base),
                                attachment_point=5,
                            )
                        except Exception:
                            pass
                    try:
                        _placed_label_bboxes.append(chosen["aabb"])
                    except Exception:
                        pass

                geometry["bearing_mtext"] += 1
                time.sleep(0.05)

            # Second pass (cartographic cleanup): nudge pillar-number labels away from overlaps.
            # This addresses very short legs where pegs and annotations cluster tightly.
            try:
                if pn_meta:
                    t_res2 = self.autocad.list_tables(layer="CADA_PILLARNUMBERS")
                    tbs = (t_res2.get("tables") or []) if isinstance(t_res2, dict) else []
                    ins_map = {}
                    for t in tbs:
                        hh = str(t.get("handle") or "").upper()
                        ip = t.get("insertion_point") or {}
                        if hh and ("x" in ip) and ("y" in ip):
                            ins_map[hh] = (float(ip.get("x", 0.0)), float(ip.get("y", 0.0)))

                    bb2 = self.autocad.list_entity_bboxes(
                        layers=["CADA_BEARING_DIST", "CADA_PILLARNUMBERS", "CADA_PILLARS"],
                        object_names=["AcDbTable", "AcDbText", "AcDbMText", "AcDbBlockReference"],
                    )
                    bbs = list((bb2 or {}).get("bboxes") or []) if (bb2 or {}).get("success") else []
                    bb_by_handle = {str((b.get("handle") or "")).upper(): b for b in bbs if b.get("handle")}

                    bearing_bbs = [b for b in bbs if str(b.get("layer") or "").upper() == "CADA_BEARING_DIST"]
                    pillar_bbs = {str((b.get("handle") or "")).upper(): b for b in bbs if str(b.get("layer") or "").upper() == "CADA_PILLARNUMBERS" and b.get("handle")}
                    pillar_obj_bbs = [b for b in bbs if str(b.get("layer") or "").upper() == "CADA_PILLARS"]

                    def _shift_aabb(a: Dict[str, Dict[str, float]], dx0: float, dy0: float) -> Dict[str, Dict[str, float]]:
                        amin, amax = a.get("min") or {}, a.get("max") or {}
                        return {
                            "min": {"x": float(amin.get("x", 0.0)) + dx0, "y": float(amin.get("y", 0.0)) + dy0},
                            "max": {"x": float(amax.get("x", 0.0)) + dx0, "y": float(amax.get("y", 0.0)) + dy0},
                        }

                    # Keep post-placement nudges local so pillar-number labels remain close to their pegs.
                    step = max(0.35, min(0.7, 0.18 * _bd_height))
                    pad = 1.15 * _bd_height
                    margin = 2.0 * _bd_height

                    # Iterate a few passes so clustered labels can settle without leaving residual overlaps.
                    for _pass in range(3):
                        moved = 0
                        for meta in pn_meta:
                            h_raw = str(meta.get("handle") or "")
                            hh = h_raw.upper()
                            if not hh or hh not in ins_map or hh not in pillar_bbs:
                                continue
                            cur_ip = ins_map[hh]
                            cur_bb = pillar_bbs[hh]

                            def _overlap_count(test_bb: Dict[str, Dict[str, float]]) -> int:
                                c = 0
                                # against bearing/distance texts
                                for ob in bearing_bbs:
                                    if _aabb_overlaps(test_bb, ob, pad=pad):
                                        c += 1
                                # against already-placed bearing labels (local estimates)
                                for ob in (_placed_label_bboxes or []):
                                    if _aabb_overlaps(test_bb, ob, pad=pad):
                                        c += 1
                                # against other sheet annotations we sampled earlier (exclude self layer to avoid double-counting)
                                for ob in (_occupied_bboxes or []):
                                    lyr = str(ob.get("layer") or "").upper()
                                    if lyr == "CADA_PILLARNUMBERS":
                                        continue
                                    if _aabb_overlaps(test_bb, ob, pad=pad):
                                        c += 1
                                # against other pillar numbers
                                for oh, ob in pillar_bbs.items():
                                    if oh == hh:
                                        continue
                                    if _aabb_overlaps(test_bb, ob, pad=pad):
                                        c += 1
                                # hard drafting penalties: don't cross peg symbols or intrude into traverse unless unavoidable
                                for ob in pillar_obj_bbs:
                                    if _aabb_overlaps(test_bb, ob, pad=0.05):
                                        c += 50
                                if _aabb_crosses_traverse(test_bb, local_pts):
                                    c += 50
                                return c

                            base_overlaps = _overlap_count(cur_bb)
                            if base_overlaps <= 0:
                                continue

                            ux0 = float(meta.get("ux", 0.0))
                            uy0 = float(meta.get("uy", 0.0))
                            L0 = math.hypot(ux0, uy0) or 1.0
                            ux0, uy0 = ux0 / L0, uy0 / L0
                            vx0, vy0 = -uy0, ux0

                            # Hard cap the search radius to preserve survey drafting style:
                            # labels should stay close to their corresponding pillar.
                            off0 = float(meta.get("off", 4.0) or 4.0)
                            max_r = max(1.0, min(2.2 * off0, 4.5))
                            angles_deg = [0, 20, -20, 40, -40, 60, -60, 80, -80, 100, -100, 120, -120, 150, -150, 180]
                            best = {"score": base_overlaps, "cost": float(base_overlaps) * 1000.0, "r": 0.0, "dx": 0.0, "dy": 0.0, "bb": cur_bb}

                            r = step
                            while r <= max_r + 1e-9:
                                for ad in angles_deg:
                                    th = math.radians(float(ad))
                                    dx0 = (ux0 * math.cos(th) + vx0 * math.sin(th)) * r
                                    dy0 = (uy0 * math.cos(th) + vy0 * math.sin(th)) * r
                                    test_bb = _shift_aabb(cur_bb, dx0, dy0)
                                    if not _aabb_inside_interior(test_bb, margin):
                                        continue
                                    score = _overlap_count(test_bb)
                                    # Prefer fewer overlaps, then shorter moves (cost function).
                                    cost = float(score) * 1000.0 + float(r)
                                    if (score < best["score"]) or (score == best["score"] and cost < best["cost"]):
                                        best = {"score": score, "cost": cost, "r": r, "dx": dx0, "dy": dy0, "bb": test_bb}
                                        if score == 0:
                                            break
                                if best["score"] == 0:
                                    break
                                r += step

                            # Only move if we strictly improve overlap score (prevents oscillation).
                            if best["r"] > 0.0 and best["score"] < base_overlaps and (best["dx"] != 0.0 or best["dy"] != 0.0):
                                try:
                                    self.autocad.move_entity_to_xy(h_raw, cur_ip[0] + float(best["dx"]), cur_ip[1] + float(best["dy"]))
                                    ins_map[hh] = (cur_ip[0] + float(best["dx"]), cur_ip[1] + float(best["dy"]))
                                    pillar_bbs[hh] = best["bb"]
                                    moved += 1
                                except Exception:
                                    pass
                        if moved == 0:
                            break

                    # Final proximity normalization: keep each pillar-number table close to its own pillar.
                    # This prevents any residual "too far" appearance after overlap reduction.
                    bb3 = self.autocad.list_entity_bboxes(
                        layers=["CADA_PILLARNUMBERS"],
                        object_names=["AcDbTable"],
                    )
                    bbs3 = list((bb3 or {}).get("bboxes") or []) if (bb3 or {}).get("success") else []
                    bb3_by_h = {str((b.get("handle") or "")).upper(): b for b in bbs3 if b.get("handle")}

                    def _dist_point_to_bbox(px: float, py: float, b: Dict[str, Any]) -> float:
                        mn = b.get("min") or {}
                        mx = b.get("max") or {}
                        minx, miny = float(mn.get("x", 0.0)), float(mn.get("y", 0.0))
                        maxx, maxy = float(mx.get("x", 0.0)), float(mx.get("y", 0.0))
                        dx0 = 0.0 if (minx <= px <= maxx) else (minx - px if px < minx else px - maxx)
                        dy0 = 0.0 if (miny <= py <= maxy) else (miny - py if py < miny else py - maxy)
                        return math.hypot(dx0, dy0)

                    target_min_gap = 0.8
                    target_max_gap = 2.0
                    for meta in pn_meta:
                        h_raw = str(meta.get("handle") or "")
                        hh = h_raw.upper()
                        if not hh or hh not in ins_map or hh not in bb3_by_h:
                            continue
                        vx = float(meta.get("vx", 0.0))
                        vy = float(meta.get("vy", 0.0))
                        cur_ip = ins_map[hh]
                        cur_bb = bb3_by_h[hh]
                        g = _dist_point_to_bbox(vx, vy, cur_bb)
                        # Move only when clearly outside desired proximity band.
                        if (g >= target_min_gap - 0.05) and (g <= target_max_gap + 0.05):
                            continue

                        vec_x = float(cur_ip[0]) - vx
                        vec_y = float(cur_ip[1]) - vy
                        vec_L = math.hypot(vec_x, vec_y)
                        if vec_L <= 1e-9:
                            vec_x = float(meta.get("ux", 1.0))
                            vec_y = float(meta.get("uy", 0.0))
                            vec_L = math.hypot(vec_x, vec_y) or 1.0
                        ux1, uy1 = vec_x / vec_L, vec_y / vec_L

                        if g < target_min_gap:
                            delta = target_min_gap - g
                        else:
                            delta = -(g - target_max_gap)

                        nx = float(cur_ip[0]) + ux1 * delta
                        ny = float(cur_ip[1]) + uy1 * delta
                        # Guardrail: do not worsen annotation collisions while enforcing proximity.
                        cur_score = 0
                        for ob in bearing_bbs:
                            if _aabb_overlaps(cur_bb, ob, pad=pad):
                                cur_score += 1
                        for oh, ob in bb3_by_h.items():
                            if oh == hh:
                                continue
                            if _aabb_overlaps(cur_bb, ob, pad=pad):
                                cur_score += 1
                        for ob in pillar_obj_bbs:
                            if _aabb_overlaps(cur_bb, ob, pad=0.05):
                                cur_score += 50
                        if _aabb_crosses_traverse(cur_bb, local_pts):
                            cur_score += 50
                        test_bb = _shift_aabb(cur_bb, nx - float(cur_ip[0]), ny - float(cur_ip[1]))
                        if not _aabb_inside_interior(test_bb, margin):
                            continue
                        new_score = 0
                        for ob in bearing_bbs:
                            if _aabb_overlaps(test_bb, ob, pad=pad):
                                new_score += 1
                        for oh, ob in bb3_by_h.items():
                            if oh == hh:
                                continue
                            if _aabb_overlaps(test_bb, ob, pad=pad):
                                new_score += 1
                        for ob in pillar_obj_bbs:
                            if _aabb_overlaps(test_bb, ob, pad=0.05):
                                new_score += 50
                        if _aabb_crosses_traverse(test_bb, local_pts):
                            new_score += 50
                        if new_score > cur_score:
                            continue
                        try:
                            self.autocad.move_entity_to_xy(h_raw, nx, ny)
                            ins_map[hh] = (nx, ny)
                            bb3_by_h[hh] = test_bb
                        except Exception:
                            pass
            except Exception:
                pass

            try:
                self.autocad.execute_command("REGEN")
            except Exception:
                pass
            time.sleep(0.2)

            # Draw Concrete Wall Fence(s) if requested (C.W.F / D.C.W.F) on CADA_CWF
            # - Single line parallel to traverse leg(s)
            # - Sits outside the traverse (outward normal)
            # - Offset scales with plan scale: 0.3 @ 1:500, 0.15 @ 1:250, 0.6 @ 1:1000, etc.
            used_fence_edges: set[int] = set()
            try:
                fence_offset = 0.3 * (float(chosen_denom) / 500.0)
            except Exception:
                fence_offset = 0.3

            for f in fences_to_draw:
                try:
                    kind = str((f or {}).get("kind") or "").upper().strip()
                    spec = str((f or {}).get("spec") or "")
                    if kind not in ("CWF", "DCWF") or not spec:
                        continue

                    spec_lower = spec.lower()
                    ref_match = re.search(
                        r"(?:linking|between|connecting|on|along|joining)\s+(?:the\s+)?(?:side\s+)?(?:of\s+)?(?:boundary\s+)?(?:line\s+)?(?:pillars\s+)?(.*)$",
                        spec_lower,
                    )
                    target_indices: List[int] = []
                    if ref_match and pn_list:
                        ref_str = ref_match.group(1).strip()
                        ref_str_norm = re.sub(r"\s+", " ", ref_str)

                        def _pillar_idx_in_text(chunk: str) -> Optional[int]:
                            """Resolve a single pillar mention (substring) to one pn_list index."""
                            ch = re.sub(r"[.,;]+$", "", (chunk or "").strip()).lower()
                            if not ch:
                                return None
                            best_i: Optional[int] = None
                            best_key = -1
                            for idx, p_info in enumerate(pn_list):
                                num = str(p_info.get("number", "")).strip()
                                prefix = str(p_info.get("prefix", "")).strip()
                                if not num:
                                    continue
                                fl = f"{prefix} {num}".lower().replace("  ", " ")
                                if fl in ch:
                                    if len(fl) > best_key:
                                        best_key = len(fl)
                                        best_i = idx
                                else:
                                    nlu = num.lower()
                                    if re.search(r"\b" + re.escape(nlu) + r"\b", ch):
                                        sc = 50 + len(nlu)
                                        if sc > best_key:
                                            best_key = sc
                                            best_i = idx
                            return best_i

                        n_pts = len(local_pts)
                        explicit_pairs_done = False
                        # "A to B and C to D" = TWO legs only (do not chain B–C). "A to B to C" = chain AB, BC.
                        if re.search(r"\s+and\s+", ref_str_norm, re.IGNORECASE) and re.search(
                            r"\s+to\s+", ref_str_norm, re.IGNORECASE
                        ):
                            explicit_pairs_done = True
                            for chunk in re.split(r"\s+and\s+", ref_str_norm):
                                ck = (chunk or "").strip()
                                if not ck:
                                    continue
                                if not re.search(r"\s+to\s+", ck, re.IGNORECASE):
                                    continue
                                parts = re.split(r"\s+to\s+", ck, maxsplit=1, flags=re.IGNORECASE)
                                if len(parts) != 2:
                                    continue
                                a_idx = _pillar_idx_in_text(parts[0])
                                b_idx = _pillar_idx_in_text(parts[1])
                                if a_idx is None or b_idx is None or a_idx == b_idx:
                                    continue
                                for i in range(n_pts):
                                    j = (i + 1) % n_pts
                                    if (i == a_idx and j == b_idx) or (i == b_idx and j == a_idx):
                                        target_indices.append(i)
                                        break

                        if not explicit_pairs_done:
                            matched_ordered: List[Tuple[int, int]] = []
                            for idx, p_info in enumerate(pn_list):
                                num = str(p_info.get("number", "")).strip()
                                prefix = str(p_info.get("prefix", "")).strip()
                                if not num:
                                    continue
                                full_label = (prefix + " " + num).lower()
                                num_lower = num.lower()
                                ref_lower = ref_str_norm.lower()
                                pos = ref_lower.find(full_label)
                                if pos >= 0:
                                    matched_ordered.append((pos, idx))
                                    continue
                                m_num = re.search(r"\b" + re.escape(num_lower) + r"\b", ref_lower)
                                if m_num:
                                    matched_ordered.append((m_num.start(), idx))
                            if len(matched_ordered) >= 2:
                                matched_ordered.sort(key=lambda x: x[0])
                                ordered_indices: List[int] = []
                                seen_idx: set[int] = set()
                                for _, idx in matched_ordered:
                                    if idx not in seen_idx:
                                        ordered_indices.append(idx)
                                        seen_idx.add(idx)
                                # Chain: "A to B to C to D" (appearance order) → fences on AB, BC, CD.
                                for kk in range(len(ordered_indices) - 1):
                                    a_idx = ordered_indices[kk]
                                    b_idx = ordered_indices[kk + 1]
                                    for i in range(n_pts):
                                        j = (i + 1) % n_pts
                                        if (i == a_idx and j == b_idx) or (i == b_idx and j == a_idx):
                                            target_indices.append(i)
                                            break
                    if not target_indices:
                        continue
                    for target_idx in target_indices:
                        if target_idx in used_fence_edges:
                            continue
                        used_fence_edges.add(target_idx)

                        p1 = local_pts[target_idx]
                        p2 = local_pts[(target_idx + 1) % len(local_pts)]
                        dx = p2["x"] - p1["x"]
                        dy = p2["y"] - p1["y"]
                        L_bound = math.hypot(dx, dy)
                        if L_bound <= 1e-6:
                            continue

                        ux, uy = dx / L_bound, dy / L_bound

                        # Outward normal using centroid
                        midx = (p1["x"] + p2["x"]) / 2.0
                        midy = (p1["y"] + p2["y"]) / 2.0
                        poly_cx = sum(p["x"] for p in local_pts) / len(local_pts)
                        poly_cy = sum(p["y"] for p in local_pts) / len(local_pts)
                        vx = poly_cx - midx
                        vy = poly_cy - midy
                        n1x, n1y = uy, -ux
                        n2x, n2y = -uy, ux
                        if (n1x * vx + n1y * vy) >= (n2x * vx + n2y * vy):
                            outx, outy = -n1x, -n1y
                        else:
                            outx, outy = -n2x, -n2y

                        f_s = {"x": p1["x"] + fence_offset * outx, "y": p1["y"] + fence_offset * outy}
                        f_e = {"x": p2["x"] + fence_offset * outx, "y": p2["y"] + fence_offset * outy}
                        self.autocad.create_lwpolyline([f_s, f_e], layer="CADA_CWF", closed=False, linetype_scale=3.0)

                        label = "C.W.F" if kind == "CWF" else "D.C.W.F"
                        fence_text_h = max(0.25, 0.5 * float(bearing_road_height))
                        tx = (f_s["x"] + f_e["x"]) / 2.0 + (fence_text_h * 1.2) * outx
                        ty = (f_s["y"] + f_e["y"]) / 2.0 + (fence_text_h * 1.2) * outy
                        rot_rad = math.atan2(uy, ux)
                        deg = math.degrees(rot_rad) % 360
                        if 90 < deg <= 270:
                            rot_rad += math.pi
                        fence_fmt = f"{{\\fVerdana|b0|i0|c0|p34;{label}}}"
                        txt_width = max(10.0, L_bound)
                        self.autocad.add_mtext(
                            fence_fmt,
                            tx,
                            ty,
                            layer="CADA_CWF",
                            rotation_rad=rot_rad,
                            height=float(fence_text_h),
                            width=txt_width,
                            attachment_point=5,
                        )
                except Exception:
                    continue

            # Draw Access Road(s) if requested (supports multiple roads, each beside a traverse leg)
            for road_idx, road_spec in enumerate(roads_to_draw):
                try:
                    ar_lower = road_spec.lower()
                    # 1. Parse Width (support "7m width", "7m road", "width 7m")
                    width = 6.0  # fallback default
                    m_w = (
                        re.search(r"(\d+(?:\.\d+)?)\s*m\s+road", ar_lower)
                        or re.search(r"(\d+(?:\.\d+)?)\s*m\s+width", ar_lower)
                        or re.search(r"width\s+(\d+(?:\.\d+)?)\s*m", ar_lower)
                        or re.search(r"width.*?(\d+(?:\.\d+)?)\s*m", ar_lower)
                    )
                    if m_w:
                        width = float(m_w.group(1))
                    elif "5m road" in ar_lower:
                        width = 5.0
                    
                    # 2. Parse offset (metres from traverse line; user value overrides default).
                    # Do not use a loose "offset.*?(\d+)m" — it often captures road width (e.g. "12m road ... offset 3m").
                    offset = 0.2
                    m_o = (
                        re.search(r"offset\s+of\s+(\d+(?:\.\d+)?)\s*m", ar_lower)
                        or re.search(r"offset\s*[=:]\s*(\d+(?:\.\d+)?)\s*m", ar_lower)
                        or re.search(r"with\s+offset\s+of\s+(\d+(?:\.\d+)?)\s*m", ar_lower)
                        or re.search(r"(\d+(?:\.\d+)?)\s*m\s+offset(?:\s+from|\s+away|\b)", ar_lower)
                    )
                    if m_o:
                        offset = float(m_o.group(1))

                    # 3. Identify Reference Edge
                    # "connecting X - Y", "joining pillars X and Y", "on the side joining pillars X and Y", "on the side of X and Y"
                    ref_match = re.search(
                        r"(?:linking|between|connecting|on|along|joining)\s+(?:the\s+)?(?:side\s+)?(?:of\s+)?(?:boundary\s+)?(?:line\s+)?(?:pillars\s+)?(.*)$",
                        ar_lower,
                    )
                    target_idx = -1
                    if ref_match and pn_list:
                        ref_str = ref_match.group(1).strip()
                        # Normalize separators: "SC/CK 4324 - 4325" or "4324 and 4325"
                        ref_str_norm = re.sub(r"\s+", " ", ref_str)
                        matched_indices = []
                        for idx, p_info in enumerate(pn_list):
                            num = str(p_info.get("number", "")).strip()
                            prefix = str(p_info.get("prefix", "")).strip()
                            if not num:
                                continue
                            # Match full label (e.g. "sc/ck 4324") or number as whole word to avoid "432" matching "4324"
                            full_label = (prefix + " " + num).lower()
                            num_lower = num.lower()
                            if full_label in ref_str_norm.lower():
                                matched_indices.append(idx)
                            elif re.search(r"\b" + re.escape(num_lower) + r"\b", ref_str_norm.lower()):
                                matched_indices.append(idx)
                        
                        if len(matched_indices) >= 2:
                            n_pts = len(local_pts)
                            for i in range(n_pts):
                                j = (i + 1) % n_pts
                                if i in matched_indices and j in matched_indices:
                                    target_idx = i
                                    break
                    if target_idx == -1:
                        # Fallback: first edge (index 0) if we have at least two points
                        if len(local_pts) >= 2:
                            target_idx = 0

                    if target_idx != -1:
                        p1 = local_pts[target_idx]
                        p2 = local_pts[(target_idx + 1) % len(local_pts)]
                        dx = p2["x"] - p1["x"]
                        dy = p2["y"] - p1["y"]
                        L_bound = math.hypot(dx, dy)
                        
                        if L_bound > 1e-6:
                            # Road length = 1.4 × traverse leg length (e.g. 100m leg → 140m road)
                            L_road = 1.4 * L_bound
                            # Minimum plotted length: (L_road / scale_denom) must be >= 0.064 so the road is visible.
                            # If (L_road / chosen_denom) < 0.064, extend road so (new_L_road / chosen_denom) = 0.064
                            # => new_L_road = 0.064 * chosen_denom (e.g. 1:500 → 32m).
                            try:
                                scale_denom = float(chosen_denom) if chosen_denom and float(chosen_denom) > 1e-6 else 500.0
                            except Exception:
                                scale_denom = 500.0
                            min_plotted_ratio = 0.064
                            ratio = L_road / scale_denom
                            if ratio < min_plotted_ratio:
                                L_road = min_plotted_ratio * scale_denom
                                extension_total = max(0.0, L_road - L_bound)
                            else:
                                extension_total = 0.4 * L_bound
                            ext_side = extension_total / 2.0
                            
                            # Unit vector along edge
                            ux, uy = dx / L_bound, dy / L_bound
                            
                            # Normal vector (outward)
                            # Reuse centroid logic
                            midx = (p1["x"] + p2["x"]) / 2.0
                            midy = (p1["y"] + p2["y"]) / 2.0
                            poly_cx = sum(p["x"] for p in local_pts) / len(local_pts)
                            poly_cy = sum(p["y"] for p in local_pts) / len(local_pts)
                            vx = poly_cx - midx
                            vy = poly_cy - midy
                            
                            n1x, n1y = uy, -ux
                            n2x, n2y = -uy, ux
                            
                            # Dot product with centroid vector: interior normal points TOWARD centroid
                            # We want OUTWARD normal
                            if (n1x * vx + n1y * vy) >= (n2x * vx + n2y * vy):
                                # n1 is interior
                                outx, outy = -n1x, -n1y
                            else:
                                # n2 is interior
                                outx, outy = -n2x, -n2y
                                
                            # Define road line points
                            # Line 1: offset
                            # Start: p1 - ext_side*u + offset*out
                            # End: p2 + ext_side*u + offset*out
                            # Actually centered: Midpoint + (L_bound/2 + ext_side)*u ...
                            # Let's use p1 and p2 as basis
                            # Road Start projected on line: p1 - ext_side * u
                            # Road End projected on line: p2 + ext_side * u
                            
                            rsx = p1["x"] - ext_side * ux
                            rsy = p1["y"] - ext_side * uy
                            rex = p2["x"] + ext_side * ux
                            rey = p2["y"] + ext_side * uy
                            
                            # Draw Line 1 (Offset)
                            l1_s = {"x": rsx + offset * outx, "y": rsy + offset * outy}
                            l1_e = {"x": rex + offset * outx, "y": rey + offset * outy}
                            self.autocad.create_lwpolyline([l1_s, l1_e], layer="CADA_ROAD", closed=False, linetype_scale=3.0)
                            
                            # Draw Line 2 (Offset + Width)
                            l2_s = {"x": rsx + (offset + width) * outx, "y": rsy + (offset + width) * outy}
                            l2_e = {"x": rex + (offset + width) * outx, "y": rey + (offset + width) * outy}
                            self.autocad.create_lwpolyline([l2_s, l2_e], layer="CADA_ROAD", closed=False, linetype_scale=3.0)

                            # Road title: per-spec titled '...', else first road uses access_road_title
                            m_spec_title = re.search(r"titled\s+['\"]([^'\"]+)['\"]", ar_lower)
                            if m_spec_title:
                                road_title = m_spec_title.group(1).strip()
                            elif road_idx == 0 and access_road_title:
                                road_title = access_road_title.strip()
                            else:
                                road_title = "ACCESS    ROAD"
                            road_title = road_title.strip() or "ACCESS    ROAD"
                            try:
                                from agent.pdf_survey_plan import normalize_access_road_title

                                road_title = normalize_access_road_title(road_title)
                            except Exception:
                                pass
                            if road_idx == 0:
                                geometry["access_road_title"] = road_title

                            # Position text perfectly centered (vertically and horizontally) within the road
                            # cx, cy = geometric center of road (midpoint between l1 and l2, offset by width/2 outward)
                            cx = (l1_s["x"] + l1_e["x"]) / 2.0 + (width / 2.0) * outx
                            cy = (l1_s["y"] + l1_e["y"]) / 2.0 + (width / 2.0) * outy

                            rot_rad = math.atan2(uy, ux)
                            deg = math.degrees(rot_rad) % 360
                            if 90 < deg <= 270:
                                rot_rad += math.pi

                            # Same text height as bearing/distances (template-matched)
                            road_title_height = bearing_road_height
                            road_title_fmt = f"{{\\fVerdana|b0|i0|c0|p34;{road_title}}}"
                            # Width = length of drawn road: single line when title fits, wrap only when it exceeds road length
                            L_road = math.hypot(l1_e["x"] - l1_s["x"], l1_e["y"] - l1_s["y"])
                            txt_width = max(10.0, L_road)
                            self.autocad.add_mtext(
                                road_title_fmt,
                                cx,
                                cy,
                                layer="CADA_ROAD",
                                rotation_rad=rot_rad,
                                height=road_title_height,
                                width=txt_width,
                                attachment_point=5,
                            )
                except Exception:
                    pass

            # Move ENTIRE plan so the primary pillar sits at the exact coordinate specified by user
            dx_all = e0 - base_x
            dy_all = n0 - base_y
            self.autocad.move_all_modelspace(dx_all, dy_all)
            time.sleep(0.15)

            # Legacy post-move overlays only if local multi-parcel draw did not run
            # (keeps single-parcel path untouched; avoids double-draw / coord-space mismatch).
            if (extra_parcels or main_parcel_label) and not geometry.get("multi_parcel_drawn_local"):
                try:
                    overlay_count = 0
                    label_h = None
                    try:
                        label_h = float((geometry.get("scale_debug") or {}).get("bearing_height") or 0.0)
                    except Exception:
                        label_h = None
                    if not label_h or label_h <= 0:
                        label_h = max(1.5, float(scale_k) * 2.5)

                    def _draw_parcel_overlay(label: str, ring: List[Dict[str, Any]]) -> None:
                        nonlocal overlay_count
                        if not ring or len(ring) < 3:
                            return
                        xy = [{"x": float(p["e"]), "y": float(p["n"])} for p in ring]
                        pl_r = self.autocad.create_lwpolyline(xy, layer="CADA_BOUNDARY", closed=True)
                        if pl_r.get("success"):
                            overlay_count += 1
                        blk = profile.get("blocks", {}).get("pillars", {}).get("block_name") or "PEG_SYMBOL"
                        for p in ring:
                            try:
                                self.autocad.insert_block(
                                    str(blk),
                                    float(p["e"]),
                                    float(p["n"]),
                                    layer="CADA_PILLARS",
                                    xscale=float(scale_k),
                                    yscale=float(scale_k),
                                    zscale=float(scale_k),
                                )
                            except Exception:
                                pass
                        lab = (label or "").strip()
                        if lab:
                            cx = sum(float(p["e"]) for p in ring) / float(len(ring))
                            cy = sum(float(p["n"]) for p in ring) / float(len(ring))
                            self.autocad.add_mtext(
                                lab,
                                cx,
                                cy,
                                layer="CADA_TEXT",
                                height=float(label_h),
                                width=max(25.0, float(label_h) * 14.0),
                                attachment_point=5,
                            )

                    if main_parcel_label and coord_pairs and len(coord_pairs) >= 3:
                        cx = sum(float(p["e"]) for p in coord_pairs) / float(len(coord_pairs))
                        cy = sum(float(p["n"]) for p in coord_pairs) / float(len(coord_pairs))
                        self.autocad.add_mtext(
                            main_parcel_label,
                            cx,
                            cy,
                            layer="CADA_TEXT",
                            height=float(label_h),
                            width=max(20.0, float(label_h) * 12.0),
                            attachment_point=5,
                        )
                    for ep in extra_parcels:
                        if not isinstance(ep, dict):
                            continue
                        _draw_parcel_overlay(
                            str(ep.get("label") or ""),
                            list(ep.get("points") or []),
                        )
                    if overlay_count:
                        geometry["extra_parcel_boundaries"] = int(overlay_count)
                        geometry["boundary_redrawn"] = True
                except Exception as overlay_exc:
                    logger.warning("Multi-parcel overlay failed: %s", overlay_exc)

            # Re-center the SHEET (border + title block/tables/north arrow/etc) around the plotted
            # land boundary WITHOUT moving survey geometry layers.
            try:
                boundary_bb = self.autocad.get_modelspace_bbox(layers=["CADA_BOUNDARY"])
                # Prefer the actual border geometry layer so we move tables AND their boxes together.
                frame_bb = self.autocad.get_modelspace_bbox(layers=["CADA_BORDER"], prefer_largest=True)
                if not frame_bb.get("success"):
                    # Next best: a border block reference (often named BORDER_*).
                    frame_bb = self.autocad.get_modelspace_bbox(block_name_contains="BORDER", prefer_largest=True)
                if not frame_bb.get("success"):
                    # fallback: union bbox of likely sheet content
                    frame_bb = self.autocad.get_modelspace_bbox(
                        layers=["CADA_BORDER", "CADA_INTERIORBORDER", "CADA_SCALEBAR", "CADA_TITLEBLOCK", "CADA_PLANNUMBER", "CADA_CERTIFICATION", "CADA_SURVEYOR", "CADA_NORTHARROW","CADA_EASTARROW", "CADA_COORDINATES", "CADA_NORTHCOORDINATES", "CADA_EASTCOORDINATES", "TITLE", "text"],
                        prefer_largest=False,
                    )
                if boundary_bb.get("success") and frame_bb.get("success"):
                    bc = boundary_bb.get("center") or {}
                    fc = frame_bb.get("center") or {}
                    # Multi-parcel plot-then-fit: align the usable plot WINDOW (not full frame
                    # center) to the land so parcels stay clear of the title block.
                    use_plot_window = bool(geometry.get("multi_parcel_sheet_fitted"))
                    if use_plot_window:
                        interior_rc = self.autocad.get_modelspace_bbox(layers=["CADA_INTERIORBORDER"])
                        if not interior_rc.get("success"):
                            interior_rc = self.autocad.get_modelspace_bbox(
                                layers=["CADA_INTERIORBOUNDARY"]
                            )
                        title_rc = self.autocad.get_modelspace_bbox(layers=["CADA_TITLEBLOCK"])
                        win_rc = _usable_plot_window_from_bboxes(interior_rc, title_rc)
                        if win_rc:
                            fc = {"x": float(win_rc["center_x"]), "y": float(win_rc["center_y"])}
                    movable_layers = _cadastral_sheet_translate_layers(profile)
                    # Primary crosshair + N/E arrows + coords are excluded from sheet recenter
                    # so CADA_PRIMARYPILLAR_ARROWS stays on the primary peg after move_all.

                    # Apply the move and then do a quick correction pass (AutoCAD bbox can shift slightly).
                    total_dx_sheet = 0.0
                    total_dy_sheet = 0.0

                    for _ in range(2):
                        dx_sheet = float(bc.get("x", 0.0)) - float(fc.get("x", 0.0))
                        dy_sheet = float(bc.get("y", 0.0)) - float(fc.get("y", 0.0))
                        if abs(dx_sheet) < 0.05 and abs(dy_sheet) < 0.05:
                            break
                        self.autocad.move_modelspace_by_layers(dx_sheet, dy_sheet, movable_layers)
                        total_dx_sheet += dx_sheet
                        total_dy_sheet += dy_sheet

                        # Recompute target frame/window center after the move
                        if use_plot_window:
                            interior_rc = self.autocad.get_modelspace_bbox(
                                layers=["CADA_INTERIORBORDER"]
                            )
                            if not interior_rc.get("success"):
                                interior_rc = self.autocad.get_modelspace_bbox(
                                    layers=["CADA_INTERIORBOUNDARY"]
                                )
                            title_rc = self.autocad.get_modelspace_bbox(
                                layers=["CADA_TITLEBLOCK"]
                            )
                            win_rc = _usable_plot_window_from_bboxes(interior_rc, title_rc)
                            if win_rc:
                                fc = {
                                    "x": float(win_rc["center_x"]),
                                    "y": float(win_rc["center_y"]),
                                }
                                continue
                        frame_bb = self.autocad.get_modelspace_bbox(layers=["CADA_BORDER"], prefer_largest=True)
                        if not frame_bb.get("success"):
                            frame_bb = self.autocad.get_modelspace_bbox(block_name_contains="BORDER", prefer_largest=True)
                        if not frame_bb.get("success"):
                            frame_bb = self.autocad.get_modelspace_bbox(layers=movable_layers, prefer_largest=False)
                        fc = (frame_bb.get("center") or {}) if frame_bb.get("success") else fc

                    # Identify coordinate text handles (layer-scoped — avoid full ModelSpace dump)
                    easting_handles = []
                    northing_handles = []
                    try:
                        all_ents_res = self.autocad.get_entities_on_layers(
                            [
                                "CADA_COORDINATES",
                                "CADA_EASTCOORDINATES",
                                "CADA_NORTHCOORDINATES",
                            ]
                        )
                        if all_ents_res.get("success"):
                            for ent in all_ents_res.get("entities", []):
                                lyr = str(ent.get("layer") or "").upper()
                                txt = str(ent.get("text_content") or "").upper()
                                h = ent.get("handle")
                                if not h: continue

                                # Check legacy layer
                                if lyr == "CADA_COORDINATES":
                                    if ".E" in txt:
                                        easting_handles.append(h)
                                    elif ".N" in txt:
                                        northing_handles.append(h)
                                # Check new split layers
                                elif lyr == "CADA_EASTCOORDINATES":
                                    # Assuming this layer contains Easting text (.E)
                                    easting_handles.append(h)
                                elif lyr == "CADA_NORTHCOORDINATES":
                                    # Assuming this layer contains Northing text (.N)
                                    northing_handles.append(h)
                    except Exception:
                        pass

                    # Align CADA_NORTHARROW and Easting text: Align X with primary pillar (e0), move Y with sheet
                    try:
                        na_bb = self.autocad.get_modelspace_bbox(layers=["CADA_NORTHARROW"])
                        if na_bb.get("success"):
                            na_c = na_bb.get("center") or {}
                            na_x = float(na_c.get("x", 0.0))
                            dx = e0 - na_x
                            dy = total_dy_sheet
                            self.autocad.move_modelspace_by_layers(dx, dy, ["CADA_NORTHARROW"])
                            if easting_handles:
                                self.autocad.move_entities_by_handles(dx, dy, easting_handles)
                                # Fine-tune block removed to prevent displacement issues
                                # The text moves rigidly with the arrow layer, preserving template relative positions.
                    except Exception:
                        pass

                    # Align CADA_EASTARROW and Northing text: Align Y with primary pillar (n0), move X with sheet
                    try:
                        ea_bb = self.autocad.get_modelspace_bbox(layers=["CADA_EASTARROW"])
                        if ea_bb.get("success"):
                            ea_c = ea_bb.get("center") or {}
                            ea_y = float(ea_c.get("y", 0.0))
                            dx = total_dx_sheet
                            dy = n0 - ea_y
                            self.autocad.move_modelspace_by_layers(dx, dy, ["CADA_EASTARROW"])
                            if northing_handles:
                                self.autocad.move_entities_by_handles(dx, dy, northing_handles)
                                # Fine-tune block removed to prevent displacement issues
                                # The text moves rigidly with the arrow layer, preserving template relative positions.
                    except Exception:
                        pass

                    # Snap arrows + coordinate texts onto the (scaled) interior border edges (template-neat behavior)
                    try:
                        interior_bb = self.autocad.get_modelspace_bbox(layers=["CADA_INTERIORBORDER"])
                        if not interior_bb.get("success"):
                            interior_bb = self.autocad.get_modelspace_bbox(layers=["CADA_INTERIORBOUNDARY"])
                        if interior_bb.get("success"):
                            imin_x = float(interior_bb.get("minx", 0.0))
                            imax_x = float(interior_bb.get("maxx", 0.0))
                            imin_y = float(interior_bb.get("miny", 0.0))
                            imax_y = float(interior_bb.get("maxy", 0.0))
                            bcx = float((bc or {}).get("x", 0.0))
                            bcy = float((bc or {}).get("y", 0.0))

                            # East arrow should sit on left/right interior border (keep its Y already aligned to n0)
                            ea_bb2 = self.autocad.get_modelspace_bbox(layers=["CADA_EASTARROW"])
                            if ea_bb2.get("success"):
                                eac = ea_bb2.get("center") or {}
                                ea_cx = float(eac.get("x", 0.0))
                                left_side = ea_cx < bcx
                                if left_side:
                                    dx2 = imin_x - float(ea_bb2.get("minx", 0.0))
                                else:
                                    dx2 = imax_x - float(ea_bb2.get("maxx", 0.0))
                                if abs(dx2) > 1e-6:
                                    self.autocad.move_modelspace_by_layers(dx2, 0.0, ["CADA_EASTARROW"])
                                    if northing_handles:
                                        self.autocad.move_entities_by_handles(dx2, 0.0, northing_handles)

                            # North arrow should sit on top/bottom interior border (keep its X already aligned to e0)
                            na_bb2 = self.autocad.get_modelspace_bbox(layers=["CADA_NORTHARROW"])
                            if na_bb2.get("success"):
                                nac = na_bb2.get("center") or {}
                                na_cy = float(nac.get("y", 0.0))
                                top_side = na_cy > bcy
                                if top_side:
                                    dy2 = imax_y - float(na_bb2.get("maxy", 0.0))
                                else:
                                    dy2 = imin_y - float(na_bb2.get("miny", 0.0))
                                if abs(dy2) > 1e-6:
                                    self.autocad.move_modelspace_by_layers(0.0, dy2, ["CADA_NORTHARROW"])
                                    if easting_handles:
                                        self.autocad.move_entities_by_handles(0.0, dy2, easting_handles)
                    except Exception:
                        pass

                # Align coordinate text and north arrow lines with CADA_PRIMARYPILLAR_ARROWS lines:
                # - Horizontal arrows: shift coordinate text horizontally to align with primary pillar X
                # - Vertical arrows: shift coordinate text vertically to align with primary pillar Y
                try:
                    prim_x, prim_y = float(e0), float(n0)
                    all_ents = self.autocad.get_entities_on_layers(
                        [
                            "CADA_PRIMARYPILLAR_ARROWS",
                            "CADA_COORDINATES",
                            "CADA_EASTCOORDINATES",
                            "CADA_NORTHCOORDINATES",
                        ]
                    )
                    ents = (all_ents.get("entities") or []) if all_ents.get("success") else []

                    def _bbox_from_ent(ent: Dict[str, Any]) -> Optional[Dict[str, float]]:
                        coords = ent.get("coordinates") or {}
                        pts = []
                        if ent.get("type") == "LINE":
                            s = coords.get("start") or {}
                            e = coords.get("end") or {}
                            pts = [(s.get("x"), s.get("y")), (e.get("x"), e.get("y"))]
                        elif ent.get("type") in ("LWPOLYLINE", "POLYLINE"):
                            for p in (coords.get("points") or []):
                                pts.append((p.get("x"), p.get("y")))
                        elif ent.get("type") in ("TEXT", "MTEXT", "INSERT", "TABLE"):
                            ip = ent.get("insertion_point") or {}
                            pts = [(ip.get("x"), ip.get("y"))]
                        else:
                            c = ent.get("center") or {}
                            if c:
                                pts = [(c.get("x"), c.get("y"))]
                        pts = [(float(x), float(y)) for (x, y) in pts if x is not None and y is not None]
                        if not pts:
                            return None
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        return {"minx": min(xs), "miny": min(ys), "maxx": max(xs), "maxy": max(ys)}

                    # Find entities on CADA_PRIMARYPILLAR_ARROWS layer
                    arrow_handles = []
                    for ent in ents:
                        lyr = str(ent.get("layer") or "").upper()
                        if lyr == "CADA_PRIMARYPILLAR_ARROWS":
                            h = str(ent.get("handle") or "").strip()
                            if h:
                                arrow_handles.append(h)

                    if arrow_handles:
                        # Crosshair = horizontal arm(s) + vertical arm(s). Align by
                        # intersection of those arms — NOT the combined bbox centre
                        # (unequal arm lengths make bbox-centre ≠ primary peg).
                        horiz_ys: List[float] = []
                        vert_xs: List[float] = []
                        for ent in ents:
                            h = str(ent.get("handle") or "").strip()
                            if not h or h.upper() not in {ah.upper() for ah in arrow_handles}:
                                continue
                            if str(ent.get("type") or "").upper() != "LINE":
                                continue
                            coords = ent.get("coordinates") or {}
                            s = coords.get("start") or {}
                            e = coords.get("end") or {}
                            try:
                                x1, y1 = float(s.get("x")), float(s.get("y"))
                                x2, y2 = float(e.get("x")), float(e.get("y"))
                            except Exception:
                                continue
                            dx_l, dy_l = x2 - x1, y2 - y1
                            if abs(dx_l) >= abs(dy_l):
                                horiz_ys.append(0.5 * (y1 + y2))
                            else:
                                vert_xs.append(0.5 * (x1 + x2))

                        if horiz_ys and vert_xs:
                            horiz_ys.sort()
                            vert_xs.sort()
                            ix = float(vert_xs[len(vert_xs) // 2])
                            iy = float(horiz_ys[len(horiz_ys) // 2])
                            dx_cross = prim_x - ix
                            dy_cross = prim_y - iy
                            if abs(dx_cross) > 1e-6 or abs(dy_cross) > 1e-6:
                                move_h = [str(h) for h in arrow_handles]
                                if east_h:
                                    move_h.append(str(east_h))
                                if north_h:
                                    move_h.append(str(north_h))
                                for ent in ents:
                                    lyr = str(ent.get("layer") or "").upper()
                                    txt = str(ent.get("text_content") or "").upper()
                                    h = str(ent.get("handle") or "").strip()
                                    if not h:
                                        continue
                                    if (lyr == "CADA_COORDINATES" and (".E" in txt or ".N" in txt)) or (
                                        lyr in ("CADA_EASTCOORDINATES", "CADA_NORTHCOORDINATES")
                                    ):
                                        move_h.append(h)
                                self.autocad.move_entities_by_handles(dx_cross, dy_cross, move_h)
                                geometry["primary_crosshair_snapped"] = True
                                geometry["primary_crosshair_dx"] = float(dx_cross)
                                geometry["primary_crosshair_dy"] = float(dy_cross)
                        else:
                            # Fallback: previous bbox-centre snap (templates without LINE arms).
                            arrow_bb = None
                            for ent in ents:
                                h = str(ent.get("handle") or "").upper()
                                if h not in [ah.upper() for ah in arrow_handles]:
                                    continue
                                bb = _bbox_from_ent(ent)
                                if not bb:
                                    continue
                                if arrow_bb is None:
                                    arrow_bb = dict(bb)
                                else:
                                    arrow_bb["minx"] = min(arrow_bb["minx"], bb["minx"])
                                    arrow_bb["miny"] = min(arrow_bb["miny"], bb["miny"])
                                    arrow_bb["maxx"] = max(arrow_bb["maxx"], bb["maxx"])
                                    arrow_bb["maxy"] = max(arrow_bb["maxy"], bb["maxy"])

                            if arrow_bb:
                                arrow_x_center = 0.5 * (arrow_bb["minx"] + arrow_bb["maxx"])
                                arrow_y_center = 0.5 * (arrow_bb["miny"] + arrow_bb["maxy"])
                                dx_horiz = prim_x - arrow_x_center
                                dy_vert = prim_y - arrow_y_center
                                if abs(dx_horiz) > 1e-6 or abs(dy_vert) > 1e-6:
                                    move_h = [str(h) for h in arrow_handles]
                                    if east_h:
                                        move_h.append(str(east_h))
                                    if north_h:
                                        move_h.append(str(north_h))
                                    self.autocad.move_entities_by_handles(dx_horiz, dy_vert, move_h)
                except Exception:
                    pass
            except Exception:
                pass

            # Zoom to extents of generated plan
            try:
                self.autocad.execute_command("ZOOM E")
            except Exception:
                pass

        # Plan number must be fitted last: sheet scaling changes final cell width vs text height.
        if plan_h:
            try:
                _cad_checkpoint()
                sd = geometry.get("scale_debug") if isinstance(geometry.get("scale_debug"), dict) else {}
                template_ref_denom = int(
                    geometry.get("template_native_denom")
                    or sd.get("template_denom")
                    or profile.get("template_scale_denom")
                    or _CADASTRAL_TEMPLATE_REF_DENOM
                )
                output_denom = int(
                    geometry.get("output_plan_denom")
                    or sd.get("chosen_denom")
                    or output_plan_denom
                    or user_scale_denom
                    or template_ref_denom
                )
                output_sk = float(
                    geometry.get("output_scale_k")
                    or sd.get("k")
                    or output_scale_k
                    or 1.0
                )
                ideal_plan_h = _ideal_plan_number_text_height(
                    template_nominal_h=template_plan_nominal_h,
                    template_denom=template_ref_denom,
                    chosen_denom=output_denom,
                    profile_ref=float(
                        (profile.get("text_heights") or {}).get("plan_number")
                        or _CADASTRAL_TEMPLATE_PLANNUMBER_REF_H
                    ),
                )
                fit_debug = _apply_plan_number_table_cell(
                    self.autocad,
                    plan_h=plan_h,
                    plan_number=plan_number,
                    get_cell=_get_cell,
                    set_cell=_set_cell,
                    ideal_text_height=ideal_plan_h,
                    cell_text_style=template_plan_text_style,
                    sheet_scale_k=float(output_sk),
                )
                if fit_debug:
                    geometry["plan_number_fit"] = {
                        **fit_debug,
                        "template_nominal_h": template_plan_nominal_h,
                        "template_denom": int(template_ref_denom),
                        "chosen_denom": int(output_denom),
                        "ideal_height": float(ideal_plan_h),
                        "sheet_scale_k": float(output_sk),
                    }
            except Exception as plan_fit_exc:
                logger.warning("Plan number table fit failed: %s", plan_fit_exc)

        _cad_checkpoint()

        try:
            self.autocad.execute_command("REGEN")
        except Exception:
            pass
        time.sleep(0.15)

        # Save the intended output file (not whichever drawing tab AutoCAD had focused).
        self._ensure_output_saved(str(outp))
        try:
            self.autocad.execute_command("ZOOM E")
        except Exception:
            pass

        # Post-verify counts from ModelSpace so we return truthful geometry even if some ops were skipped.
        try:
            ms = self.autocad.doc.ModelSpace
            cnt_p = cnt_b = cnt_bd = 0
            for i in range(ms.Count):
                e = ms.Item(i)
                layer = str(getattr(e, "Layer", "")).upper()
                on = str(getattr(e, "ObjectName", ""))
                if layer == "CADA_PILLARS" and "BlockReference" in on:
                    cnt_p += 1
                if layer == "CADA_BOUNDARY":
                    cnt_b += 1
                if layer == "CADA_BEARING_DIST" and (on == "AcDbMText" or on == "AcDbText"):
                    cnt_bd += 1
            geometry["pillar_inserts"] = cnt_p
            geometry["boundary_entities"] = cnt_b
            geometry["bearing_mtext"] = cnt_bd
        except Exception:
            pass

        if coordinates and not geometry.get("boundary_redrawn"):
            _release_output_tab(already_saved=True)
            return {
                "success": False,
                "error": (
                    "Parcel geometry was not replotted from your coordinates/traverse — "
                    "the output would have kept the template land drawing. "
                    "Check pillar numbers, bearing/distance legs, and coordinate format."
                ),
                "geometry": geometry,
                "profile_path": profile_path,
            }

        out_result = {
            "success": True,
            "output_dwg": str(outp),
            "geometry": geometry,
            "profile_path": profile_path,
            "template_path": str(Path(template).resolve()) if template else None,
        }
        if geometry.get("access_road_title"):
            out_result["access_road_title"] = geometry["access_road_title"]
        _release_output_tab(already_saved=True)
        return out_result

    # ==========================================================================
    # IN-SESSION CAD MODIFICATIONS (same output file, template always read-only)
    # ==========================================================================

    def _extract_dwg_path_from_query(self, query: str) -> Optional[str]:
        """Extract a .dwg file path from the query (quoted or path-like). Returns resolved path or None."""
        import re
        from pathlib import Path
        q = query or ""
        # Quoted paths: 'path.dwg' or "path.dwg"
        for pat in [r"['\"]([^'\"]+?\.dwg)['\"]", r"(?:in|to|file|open|modify)\s+['\"]?([^\s'\"]+\.dwg)['\"]?", r"([A-Za-z]:\\[^\s]+\.dwg)", r"([^\s<>|]+\.dwg)"]:
            m = re.search(pat, q, re.IGNORECASE)
            if m:
                raw = (m.group(1) or "").strip().strip("'\"")
                if not raw:
                    continue
                p = Path(raw)
                if not p.is_absolute():
                    p = (Path.cwd() / p).resolve()
                if p.suffix.lower() == ".dwg":
                    return str(p)
        return None

    def _ensure_protected_templates_loaded(self) -> None:
        """Load all known survey plan template paths from template_profiles so they are never written."""
        from pathlib import Path
        import json as _json
        # Scan the writable profiles dir plus read-only seed dirs (bundled + dev)
        # so templates shipped with the app are also protected from writes.
        scan_dirs = [self._cad_template_profiles_dir()] + list(self._cad_template_profiles_seed_dirs())
        for profile_dir in scan_dirs:
            if not profile_dir.exists():
                continue
            for prof_path in profile_dir.glob("*.json"):
                try:
                    data = _json.loads(prof_path.read_text(encoding="utf-8"))
                    if isinstance(data.get("templates"), list):
                        for ent in data.get("templates") or []:
                            tp = str((ent or {}).get("path") or "")
                            if tp:
                                self._protected_template_paths.add(str(Path(tp).resolve()))
                        continue
                    tp = (data.get("template") or {}).get("path") or ""
                    if tp:
                        self._protected_template_paths.add(str(Path(tp).resolve()))
                except Exception:
                    continue

    def _is_protected_template_path(self, dwg_path: str) -> bool:
        """True if dwg_path is a protected survey plan template (must never be written)."""
        if not dwg_path:
            return False
        self._ensure_protected_templates_loaded()
        try:
            from pathlib import Path
            # Must compare str to str (protected set stores str(Path.resolve()))
            return str(Path(dwg_path).resolve()) in self._protected_template_paths
        except Exception:
            return False

    def _safe_save_active_drawing(self) -> None:
        """
        Save the active drawing only if it is NOT a protected survey plan template.
        STRICT: Never write to the template (read-only to avoid corruption).
        If the active document is a template, save is skipped and a warning is logged.
        """
        active_path = self.autocad.get_active_document_path() if getattr(self.autocad, "get_active_document_path", None) else None
        if active_path and self._is_protected_template_path(active_path):
            logger.warning("Survey plan template is read-only; save skipped to avoid corruption.")
            return
        try:
            self.autocad.save_active_drawing()
        except Exception as e:
            logger.warning("save_active_drawing failed: %s", e)

    def _ensure_output_saved(self, output_dwg_path: str) -> None:
        """
        Activate the specific output DWG and QSAVE it. Prevents writing the wrong tab when
        multiple drawings are open (e.g. template + Check16 + Check17).
        """
        from pathlib import Path

        try:
            p = str(Path(output_dwg_path).resolve())
        except Exception:
            p = str(output_dwg_path)
        if self._is_protected_template_path(p):
            logger.warning("Refusing to save protected template path: %s", p)
            return
        try:
            r = self.autocad.open_drawing(p, read_only=False)
            if not r.get("success"):
                logger.warning("ensure_output_saved: could not activate %s: %s", p, r.get("error"))
        except Exception as e:
            logger.warning("ensure_output_saved: open_drawing failed: %s", e)
        ap = self.autocad.get_active_document_path() if getattr(self.autocad, "get_active_document_path", None) else None
        if ap and self._is_protected_template_path(ap):
            logger.warning("Active document is a protected template; save skipped: %s", ap)
            return
        try:
            self.autocad.save_active_drawing()
        except Exception as e:
            logger.warning("ensure_output_saved: save failed: %s", e)

    def _is_template_path(self, dwg_path: str) -> bool:
        """True if dwg_path is the template path from the last-used profile (template is read-only)."""
        if not dwg_path or not self._last_cadastral_profile_path:
            return False
        try:
            import json as _json
            from pathlib import Path
            prof = Path(self._last_cadastral_profile_path).resolve()
            if not prof.exists():
                return False
            profile = _json.loads(prof.read_text(encoding="utf-8"))
            template_path = (profile.get("template") or {}).get("path") or ""
            if not template_path:
                return False
            return Path(dwg_path).resolve() == Path(template_path).resolve()
        except Exception:
            return False

    def _should_fastpath_cad_modification(self, query: str) -> bool:
        """True when the user asks to modify a cadastral plan in the same session (e.g. add road, change title)."""
        q = (query or "").lower()
        if ".dwg" not in q:
            if not self._last_cadastral_output_dwg:
                return False
        mod_keywords = [
            "add another road", "add a road", "add road", "add access road",
            "change the title", "change title", "set title", "update title",
            "modify the plan", "modify plan", "edit the plan", "edit plan",
            "add road on the other side", "add road on the other side of the boundary",
            "change the plan title", "update the plan title",
        ]
        if not any(k in q for k in mod_keywords):
            return False
        return bool(self._last_cadastral_output_dwg or ".dwg" in q)

    def _run_cad_modification_pipeline(self, query: str) -> Dict[str, Any]:
        """
        Apply modifications to an existing cadastral plan (output DWG) in the same session.
        Template file is never written to; only the output/working file is modified.
        Works even if the CAD file is already open (activates existing document).
        """
        import json as _json
        import math
        import re
        from pathlib import Path

        # Resolve target file: explicit path in query or last-generated output
        target = self._extract_dwg_path_from_query(query) or (self._last_cadastral_output_dwg and str(Path(self._last_cadastral_output_dwg).resolve()))
        if not target:
            return {"success": False, "error": "No plan file specified and no plan was generated in this session. Generate a plan first or specify the output .dwg file."}
        target_p = Path(target).resolve()
        if not target_p.exists():
            return {"success": False, "error": f"Plan file not found: {target_p}"}
        if self._is_template_path(target):
            return {"success": False, "error": "The template file is read-only and cannot be modified. Use the output plan file or the plan we just generated."}

        # Same path/name already on disk — confirm before modifying (never the template).
        if target_p.exists():
            if not _confirm_overwrite_existing_dwg(str(target_p), mode="modify"):
                return {
                    "success": False,
                    "cancelled": True,
                    "error": (
                        f"Left existing drawing unchanged: {str(target_p)}\n"
                        "Modification was declined."
                    ),
                }

        if not self.autocad.is_connected and not self.autocad.connect():
            return {"success": False, "error": "Could not connect to AutoCAD via COM"}

        # Save/close other unsaved drawings (never the protected template, keep the target open).
        try:
            self._ensure_protected_templates_loaded()
            exclude = list(self._protected_template_paths) + [str(target_p)]
            self.autocad.save_and_close_other_drawings(
                exclude_paths=exclude,
                close_after_save=True,
            )
        except Exception as e:
            logger.warning("save_and_close_other_drawings (modification) failed: %s", e)

        opened = self.autocad.open_drawing(str(target_p), read_only=False)
        if not opened.get("success"):
            return {"success": False, "error": opened.get("error", "Failed to open plan drawing")}

        # Load profile: from last session or build minimal from current drawing
        profile = None
        if self._last_cadastral_output_dwg and Path(self._last_cadastral_output_dwg).resolve() == target_p and self._last_cadastral_profile_path:
            try:
                prof_path = Path(self._last_cadastral_profile_path).resolve()
                if prof_path.exists():
                    profile = _json.loads(prof_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        if not profile:
            tables = self.autocad.list_tables().get("tables") or []
            by_layer = {}
            for t in tables:
                by_layer.setdefault(str(t.get("layer") or ""), []).append(t)
            profile = {
                "tables": {
                    "title_block": {"handle": (by_layer.get("CADA_TITLEBLOCK", [{}])[0] or {}).get("handle")},
                    "plan_number": {"handle": (by_layer.get("CADA_PLANNUMBER", [{}])[0] or {}).get("handle")},
                    "surveyor": {"handle": (by_layer.get("CADA_SURVEYOR", [{}])[0] or {}).get("handle")},
                    "certification": {"handle": (by_layer.get("CADA_CERTIFICATION", [{}])[0] or {}).get("handle")},
                }
            }

        def _get_cell(h: str, r: int, c: int = 0) -> str:
            if not h:
                return ""
            res = self.autocad.get_table_cell_text(h, r, c)
            return str(res.get("text") or "") if res.get("success") else ""

        def _set_cell(h: str, r: int, c: int, val: str):
            if h:
                self.autocad.set_table_cell_text(h, r, c, val)

        def _mtxt_replace(raw: str, new_content: str) -> str:
            raw = raw or ""
            if raw.startswith("{") and raw.endswith("}") and ";" in raw:
                idx = raw.rfind(";")
                return raw[: idx + 1] + new_content + "}"
            return new_content

        modifications_done = []
        q = query or ""

        # --- Change title (buyer/title block row 2) ---
        title_match = re.search(r"(?:change|set|update)\s+(?:the\s+)?title\s+(?:to\s+)?['\"]?([^'\"]+)['\"]?|title\s+(?:as|to)\s+['\"]?([^'\"]+)['\"]?", q, re.IGNORECASE)
        if title_match:
            new_title = (title_match.group(1) or title_match.group(2) or "").strip()
            if new_title:
                tables = profile.get("tables", {})
                title_h = str((tables.get("title_block") or {}).get("handle") or "")
                if title_h:
                    cur = _get_cell(title_h, 2, 0)
                    owner_max_chars = 38
                    try:
                        ext_owner = self.autocad.get_table_cell_extents(title_h, 2, 0, outer=True)
                        step_owner = self.autocad.get_table_cell_mtext_line_step(
                            title_h, 2, 0, cur
                        )
                        if ext_owner.get("success"):
                            ow = float(ext_owner.get("maxx", 0.0)) - float(
                                ext_owner.get("minx", 0.0)
                            )
                            oth = float((step_owner or {}).get("text_height") or 0.0)
                            if ow > 1.0 and oth > 0.05:
                                owner_max_chars = max(
                                    16,
                                    int(ow / (oth * _CARTO_CHAR_WIDTH_FACTOR)),
                                )
                    except Exception:
                        pass
                    _set_cell(
                        title_h,
                        2,
                        0,
                        _mtxt_replace(
                            cur,
                            _format_buyer_name_for_titleblock(
                                new_title, max_chars_per_line=owner_max_chars
                            ),
                        ),
                    )
                    modifications_done.append("title")
                else:
                    modifications_done.append("title_skip_no_handle")

        # --- Add (another) access road ---
        add_road = any(phrase in q.lower() for phrase in ["add another road", "add a road", "add road", "add access road", "add road on the other side"])
        if add_road:
            # Parse width, offset, pillar ref from query
            width = 7.0
            offset = 0.0
            m_w = re.search(r"(?:width|wide)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*m", q, re.IGNORECASE)
            if m_w:
                width = float(m_w.group(1))
            m_off = re.search(r"offset\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*m", q, re.IGNORECASE)
            if m_off:
                offset = float(m_off.group(1))
            # Boundary points from current drawing (get_all_entities returns "points" for LWPOLYLINE)
            all_ents = self.autocad.get_all_entities()
            entities = (all_ents.get("entities") or []) if all_ents.get("success") else []
            local_pts = []
            for ent in entities:
                if str(ent.get("layer") or "").upper() != "CADA_BOUNDARY":
                    continue
                if str(ent.get("type") or "").upper() not in ("LWPOLYLINE", "POLYLINE"):
                    continue
                points = ent.get("points") or []
                if isinstance(points, list) and len(points) >= 3:
                    for p in points:
                        if isinstance(p, dict) and "x" in p and "y" in p:
                            local_pts.append({"x": float(p["x"]), "y": float(p["y"])})
                    break
            if len(local_pts) >= 3:
                # Edge index: use 1 for "another" road, 0 otherwise
                target_idx = 1 if "another" in q.lower() or "other side" in q.lower() else 0
                if target_idx >= len(local_pts):
                    target_idx = 0
                p1 = local_pts[target_idx]
                p2 = local_pts[(target_idx + 1) % len(local_pts)]
                dx = p2["x"] - p1["x"]
                dy = p2["y"] - p1["y"]
                L_bound = math.hypot(dx, dy)
                if L_bound > 1e-6:
                    # Road length = 1.4 × traverse leg length
                    extension_total = 0.4 * L_bound
                    ext_side = extension_total / 2.0
                    ux, uy = dx / L_bound, dy / L_bound
                    midx = (p1["x"] + p2["x"]) / 2.0
                    midy = (p1["y"] + p2["y"]) / 2.0
                    poly_cx = sum(p["x"] for p in local_pts) / len(local_pts)
                    poly_cy = sum(p["y"] for p in local_pts) / len(local_pts)
                    vx, vy = poly_cx - midx, poly_cy - midy
                    n1x, n1y = uy, -ux
                    n2x, n2y = -uy, ux
                    if (n1x * vx + n1y * vy) >= (n2x * vx + n2y * vy):
                        outx, outy = -n1x, -n1y
                    else:
                        outx, outy = -n2x, -n2y
                    rsx = p1["x"] - ext_side * ux
                    rsy = p1["y"] - ext_side * uy
                    rex = p2["x"] + ext_side * ux
                    rey = p2["y"] + ext_side * uy
                    l1_s = {"x": rsx + offset * outx, "y": rsy + offset * outy}
                    l1_e = {"x": rex + offset * outx, "y": rey + offset * outy}
                    l2_s = {"x": rsx + (offset + width) * outx, "y": rsy + (offset + width) * outy}
                    l2_e = {"x": rex + (offset + width) * outx, "y": rey + (offset + width) * outy}
                    self.autocad.create_lwpolyline([l1_s, l1_e], layer="CADA_ROAD", closed=False, linetype_scale=3.0)
                    self.autocad.create_lwpolyline([l2_s, l2_e], layer="CADA_ROAD", closed=False, linetype_scale=3.0)
                    road_title = "ACCESS    ROAD"
                    m_title = re.search(r"(?:title|labeled|named)\s+['\"]([^'\"]+)['\"]", q, re.IGNORECASE)
                    if m_title:
                        road_title = m_title.group(1).strip()
                    cx = (l1_s["x"] + l1_e["x"]) / 2.0 + (width / 2.0) * outx
                    cy = (l1_s["y"] + l1_e["y"]) / 2.0 + (width / 2.0) * outy
                    rot_rad = math.atan2(uy, ux)
                    deg = math.degrees(rot_rad) % 360
                    if 90 < deg <= 270:
                        rot_rad += math.pi
                    road_title_fmt = f"{{\\fVerdana|b0|i0|c0|p34;{road_title}}}"
                    L_road = math.hypot(l1_e["x"] - l1_s["x"], l1_e["y"] - l1_s["y"])
                    txt_width = max(10.0, L_road)
                    road_h = float((profile.get("text_heights") or {}).get("bearing_dist_road") or 1.2)
                    try:
                        hr = self.autocad.get_sample_text_height(layers=["CADA_BEARING_DIST", "CADA_ROAD"])
                        if hr.get("success") and hr.get("height"):
                            road_h = float(hr["height"])
                    except Exception:
                        pass
                    self.autocad.add_mtext(road_title_fmt, cx, cy, layer="CADA_ROAD", rotation_rad=rot_rad, height=road_h, width=txt_width, attachment_point=5)
                    modifications_done.append("access_road")
            elif add_road:
                return {"success": False, "error": "Could not find boundary (CADA_BOUNDARY) in the plan to add the road."}

        if not modifications_done:
            return {"success": False, "error": "Could not parse a modification from your request (e.g. 'change title to X' or 'add another road on the other side')."}

        # STRICT: Never save if active doc is template (read-only to avoid corruption).
        self._safe_save_active_drawing()
        try:
            self.autocad.execute_command("ZOOM E")
        except Exception:
            pass
        return {"success": True, "output_dwg": str(target_p), "modifications": modifications_done}

    def _run_docx_report_pipeline(
        self,
        query: str,
        output_doc_path: str,
        llm: BaseChatModel,
        model_name_used: str,
    ) -> Dict[str, Any]:
        """
        Deterministic report generation pipeline to avoid tool-loop recursion:
        - (Optional) fetch a small set of internet sources (permissioned)
        - single LLM call to draft report
        - save to output_doc_path
        """
        from pathlib import Path

        output_path = self._resolve_user_output_path(query, output_doc_path)

        # Internet sourcing (permissioned)
        internet_block = ""
        if getattr(self, "_internet_permission_granted", False):
            try:
                # Build targeted searches from the user query (best-effort, no extra LLM call).
                ql = (query or "").lower()
                searches = []
                # Domain-specific boosts
                if "nigeria" in ql or "surcon" in ql or "surveying" in ql:
                    searches.extend(
                        [
                            "Surveyors Council of Nigeria SURCON licensing requirements",
                            "Surveyors Council of Nigeria Act",
                            "Nigerian Institution of Surveyors history",
                            "practice of surveying in Nigeria history",
                            "SURCON register of surveyors Nigeria licensing process",
                        ]
                    )
                # Also include the raw query (truncated)
                searches.append((query or "")[:180])
                # De-dup / sanitize
                searches = [s.strip() for s in searches if s and s.strip()]
                seen_s = set()
                searches = [s for s in searches if not (s.lower() in seen_s or seen_s.add(s.lower()))]
                searches = searches[:6]
                hits = []
                for s in searches:
                    res = _internet_search(s)
                    if res.get("success"):
                        for r in (res.get("results") or [])[:3]:
                            hits.append(
                                f"- {r.get('title','')}\n  - {r.get('url','')}\n  - {r.get('snippet','')}"
                            )
                if hits:
                    internet_block = (
                        "\n\nINTERNET-SOURCED (EXTERNAL) REFERENCES (permission granted):\n"
                        + "\n".join(hits[:12])
                        + "\n\nIMPORTANT: Treat as external; prefer official government / SURCON / NIS sources where possible.\n"
                    )
            except Exception as e:
                internet_block = f"\n\n[Internet lookup failed: {e}]\n"

        prompt = (
            "Write a well-structured report for a professional audience.\n"
            "REQUIREMENTS:\n"
            "- Follow the USER REQUEST exactly.\n"
            "- Include a section titled exactly: 'Internet-sourced (external) information' listing any external links used.\n"
            "- Use citations inline like [1], [2] where appropriate, matching the links you list.\n"
            "- If internet results are missing/empty, clearly state that live sources could not be retrieved and proceed with an offline explanation.\n"
            "- Keep the report structured with headings and bullet points.\n"
            "- Output should be suitable for saving into a Word document.\n\n"
            f"USER REQUEST:\n{query}\n"
            f"{internet_block}\n"
        )

        report_msg, err, timed_out = self._run_with_timeout(
            180, lambda: llm.invoke([HumanMessage(content=prompt)])
        )
        if timed_out:
            return {
                "success": False,
                "error": "LLM report call timed out after 180 seconds",
                "response": "LLM report call timed out. Try increasing AGENT_QUERY_TIMEOUT or using a smaller scope.",
                "output_path": str(output_path),
            }
        if err:
            return {
                "success": False,
                "error": str(err),
                "response": f"LLM report call failed: {err}",
                "output_path": str(output_path),
            }

        report_text = report_msg.content if hasattr(report_msg, "content") else str(report_msg)
        title = f"Report - {output_path.stem}"
        create_result = self.document_processor.create_word_document(str(output_path), report_text, title=title)
        if not create_result.get("success"):
            return {
                "success": False,
                "error": create_result.get("error", "Failed to create report document"),
                "response": str(create_result),
                "output_path": str(output_path),
            }

        return {
            "success": True,
            "response": (
                f"✅ Created report document:\n"
                f"- Output: {str(output_path)}\n"
                f"- Model: {model_name_used}\n"
                f"- Internet used: {'yes' if getattr(self, '_internet_permission_granted', False) else 'no'}\n"
            ),
            "output_path": str(output_path),
            "model_name": model_name_used,
        }

    def _run_large_doc_summary_pipeline(
        self,
        query: str,
        input_doc_path: str,
        output_doc_path: str,
        llm: BaseChatModel,
        model_name_used: str
    ) -> Dict[str, Any]:
        """
        Large-document summarization pipeline (fast path).
        
        Steps:
        - Preflight estimation (already done by caller)
        - Extract only relevant sections by keywords (streaming, no full text)
        - Pull relevant tables (coordinates/control points) and include them as context
        - Ask the LLM ONCE to write the final 3-page style summary
        - Save to output_doc_path
        """
        from pathlib import Path

        input_path = Path(input_doc_path)
        output_path = Path(output_doc_path)

        # Keyword set tuned for survey QA/QC extraction
        keywords = [
            "Location", "Personnel", "Personnels", "Contractor", "Client",
            "Purpose", "Scope", "Date", "Duration", "Equipment", "Equipments",
            "Quantities", "Quantity", "Achieved", "Surveyed",
            "Control", "Control Point", "Control Points",
            "Coordinate", "Coordinates", "Easting", "Northing", "UTM",
            "Check", "QC", "QA", "In-situ", "Insitu", "Verification", "Validation"
        ]

        section_result = self.document_processor.extract_sections_by_keywords(
            str(input_path), keywords=keywords, context_lines=12
        )

        # Avoid full table extraction on huge Word docs (python-docx can be very slow).
        # Instead, pull coordinate-like snippets from extracted text.
        relevant_tables: list[dict[str, Any]] = []

        extracted_text = ""
        if section_result.get("success"):
            extracted_text = section_result.get("extracted_text", "") or ""

        # Hard cap extracted text to avoid blowing context on pathological docs
        if len(extracted_text) > 200_000:
            extracted_text = extracted_text[:200_000] + "\n\n[TRUNCATED: extracted_text too long]\n"

        # Quick coordinate snippets
        coord_snippets = []
        try:
            import re
            patterns = [
                r"\bE(?:asting)?[:\s]*\d{5,}\.?\d*\s+N(?:orthing)?[:\s]*\d{5,}\.?\d*\b",
                r"\b\d{5,}\.?\d*\s*,\s*\d{5,}\.?\d*\b",
                r"\b(?:X|E)[:\s]*\d{5,}\.?\d*\s+(?:Y|N)[:\s]*\d{5,}\.?\d*\b",
            ]
            for pat in patterns:
                coord_snippets.extend(re.findall(pat, extracted_text, flags=re.IGNORECASE))
            coord_snippets = list(dict.fromkeys(coord_snippets))[:30]
        except Exception:
            coord_snippets = []

        prompt = (
            "You are a senior QA/QC Surveyor. Produce a professional 3-page-style summary (concise but complete) "
            "from the extracted sections and tables provided.\n\n"
            "REQUIREMENTS:\n"
            "- Extract and summarize: Location, Personnel involved, Contractor and Client, Purpose/scope, "
            "date and duration, equipment used, quantities achieved/surveyed, control points used, "
            "coordinates, and in-situ/verification checks.\n"
            "- If a field is not present in the extracted content, write 'Not stated in the provided extract' (do NOT guess).\n"
            "- Prefer structured headings and bullet points.\n"
            "- Keep it within ~1200-1800 words (approx 3 pages).\n\n"
            f"SOURCE DOCUMENT: {input_path.name}\n\n"
            "EXTRACTED TEXT (partial, keyword-based):\n"
            "----------------\n"
            f"{extracted_text}\n\n"
            "COORDINATE / CONTROL-POINT SNIPPETS (auto-detected):\n"
            "----------------\n"
            f"{coord_snippets}\n"
        )

        summary_msg, err, timed_out = self._run_with_timeout(
            120, lambda: llm.invoke([HumanMessage(content=prompt)])
        )
        if timed_out:
            return {
                "success": False,
                "error": "LLM summary call timed out after 120 seconds",
                "response": "LLM summary call timed out. Try increasing AGENT_QUERY_TIMEOUT or using a smaller keyword extract.",
                "output_path": str(output_path),
            }
        if err:
            return {
                "success": False,
                "error": str(err),
                "response": f"LLM summary call failed: {err}",
                "output_path": str(output_path),
            }
        summary_text = summary_msg.content if hasattr(summary_msg, "content") else str(summary_msg)

        title = f"Summary - {input_path.stem}"
        create_result = self.document_processor.create_word_document(str(output_path), summary_text, title=title)

        if not create_result.get("success"):
            return {
                "success": False,
                "error": create_result.get("error", "Failed to create summary document"),
                "response": str(create_result),
                "output_path": str(output_path)
            }

        response_text = (
            f"✅ Created summary document:\n"
            f"- Source: {str(input_path)}\n"
            f"- Output: {str(output_path)}\n"
            f"- Model: {model_name_used}\n"
        )

        return {
            "success": True,
            "response": response_text,
            "output_path": str(output_path),
            "model_name": model_name_used
        }
    
    def _vs_search(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Dispatch vector-store search using the configured search mode.

        Modes (set via VECTOR_SEARCH_MODE env var / settings):
          hybrid   – RRF fusion of cosine-ANN + BM25/ts_rank (best recall, default)
          semantic – pure cosine-ANN via pgvector
          keyword  – full-text ts_rank only (no embedding needed)

        Falls back to semantic if the backend does not support hybrid.
        """
        if self.vector_store is None:
            return []
        mode = getattr(self, "_vs_search_mode", "hybrid")
        if mode == "hybrid" and hasattr(self.vector_store, "hybrid_search"):
            alpha = getattr(self, "_vs_hybrid_alpha", 0.6)
            return self.vector_store.hybrid_search(
                query=query,
                collection=collection,
                top_k=top_k,
                where=where,
                alpha=alpha,
            )
        # semantic or keyword fallback
        return self.vector_store.search(
            query=query,
            collection=collection,
            top_k=top_k,
            where=where,
        )

    def _retrieve_relevant_context(
        self,
        query: str,
        session_id: Optional[str] = None,
        collections: Optional[List[str]] = None,
    ) -> str:
        """
        Retrieve relevant context from the vector store for the given query.

        Prioritizes:
        1. Recent conversations from the current session (if session_id provided)
        2. Hybrid search (semantic + keyword RRF) across all conversations
        3. Relevant documents and drawings
        4. Survey coordinate data

        Args:
            query: The user's query to find relevant context for
            session_id: Session ID to prioritize recent session context

        Returns:
            Formatted context string to inject into the conversation
        """
        if self.vector_store is None:
            return ""
        
        if not getattr(self.settings, 'auto_context_retrieval', True):
            return ""
        
        top_k = getattr(self.settings, 'context_retrieval_top_k', 5)
        threshold = getattr(self.settings, 'context_score_threshold', 0.3)

        allowed = set(collections or [])
        allow_all = not allowed
        
        context_parts = []
        recent_convs = []  # Initialize to avoid NameError
        
        try:
            # PRIORITY 1: Get recent conversations from current session
            if session_id and (allow_all or COLLECTION_CONVERSATIONS in allowed):
                recent_convs = self.vector_store.get_recent_conversations(
                    session_id=session_id,
                    limit=10  # Get last 10 messages from this session
                )
                
                if recent_convs:
                    # Format recent conversation history (most recent first)
                    context_parts.append("**RECENT CONVERSATION HISTORY (Current Session):**")
                    for conv in recent_convs[:5]:  # Show last 5 messages
                        role = conv.get('metadata', {}).get('role', 'unknown')
                        content = conv.get('content', '')
                        # Truncate long content
                        if len(content) > 400:
                            content = content[:400] + "..."
                        context_parts.append(f"  [{role.upper()}]: {content}")
                    context_parts.append("")  # Empty line separator
            
            relevant_convs = []
            other_session_convs = []
            if allow_all or COLLECTION_CONVERSATIONS in allowed:
                # PRIORITY 2: Semantic / hybrid search for relevant conversations
                # Strategy: Get more results, then separate by session
                conv_results = self._vs_search(
                    query=query,
                    collection=COLLECTION_CONVERSATIONS,
                    top_k=top_k * 2,
                )
                
                # Separate current session from other sessions
                if session_id:
                    current_session_convs = [
                        r for r in conv_results 
                        if r.get('metadata', {}).get('session_id') == session_id
                        and r.get('score', 0) >= threshold
                    ]
                    other_session_convs = [
                        r for r in conv_results 
                        if r.get('metadata', {}).get('session_id') != session_id
                        and r.get('score', 0) >= threshold
                    ]
                    # Prioritize current session, but also include highly relevant from other sessions
                    relevant_convs = current_session_convs[:3] + other_session_convs[:2]
                else:
                    relevant_convs = [r for r in conv_results if r.get('score', 0) >= threshold][:5]
                
                # Show semantic results if we have them and didn't already show recent session history
                if relevant_convs and not (session_id and recent_convs):
                    context_parts.append("**Relevant Past Conversations:**")
                    for i, result in enumerate(relevant_convs[:3], 1):
                        role = result.get('metadata', {}).get('role', 'unknown')
                        score = result.get('score', 0)
                        result_session = result.get('metadata', {}).get('session_id', 'unknown')
                        is_current_session = session_id and result_session == session_id
                        session_label = " (current session)" if is_current_session else " (past session)"
                        content_preview = result.get('content', '')[:300]
                        context_parts.append(
                            f"  {i}. [{role}]{session_label} (relevance: {score:.2f}): {content_preview}..."
                        )
                elif other_session_convs and session_id and recent_convs:
                    # We have recent session history, but also show highly relevant from other sessions
                    highly_relevant_other = [r for r in other_session_convs if r.get('score', 0) >= 0.7][:2]
                    if highly_relevant_other:
                        context_parts.append("\n**Highly Relevant from Past Sessions:**")
                        for i, result in enumerate(highly_relevant_other, 1):
                            role = result.get('metadata', {}).get('role', 'unknown')
                            score = result.get('score', 0)
                            content_preview = result.get('content', '')[:300]
                            context_parts.append(
                                f"  {i}. [{role}] (relevance: {score:.2f}): {content_preview}..."
                            )
            
            relevant_docs = []
            if allow_all or COLLECTION_DOCUMENTS in allowed:
                # PRIORITY 3: Search documents for relevant information
                doc_results = self._vs_search(
                    query=query,
                    collection=COLLECTION_DOCUMENTS,
                    top_k=top_k,
                )
                relevant_docs = [r for r in doc_results if r.get("score", 0) >= threshold]

            if relevant_docs:
                context_parts.append("\n**Relevant Documents:**")
                for i, result in enumerate(relevant_docs[:3], 1):
                    source = result.get("metadata", {}).get("source", "unknown")
                    score = result.get("score", 0)
                    content_preview = result.get("content", "")[:300]
                    context_parts.append(
                        f"  {i}. [Source: {source}] (relevance: {score:.2f}): {content_preview}..."
                    )

            relevant_draws = []
            if allow_all or COLLECTION_DRAWINGS in allowed:
                # PRIORITY 4: Search drawings for relevant CAD data
                draw_results = self._vs_search(
                    query=query,
                    collection=COLLECTION_DRAWINGS,
                    top_k=3,
                )
                relevant_draws = [r for r in draw_results if r.get("score", 0) >= threshold]

            relevant_coords = []
            if allow_all or COLLECTION_COORDINATES in allowed:
                coord_results = self._vs_search(
                    query=query,
                    collection=COLLECTION_COORDINATES,
                    top_k=top_k,
                )
                relevant_coords = [r for r in coord_results if r.get("score", 0) >= threshold]
                if relevant_coords:
                    context_parts.append("\n**Relevant Coordinate Data:**")
                    for i, result in enumerate(relevant_coords[:3], 1):
                        src = result.get("metadata", {}).get("source", "unknown")
                        score = result.get("score", 0)
                        content_preview = result.get("content", "")[:300]
                        context_parts.append(f"  {i}. [Source: {src}] (relevance: {score:.2f}): {content_preview}...")
            
            if relevant_draws:
                context_parts.append("\n**Relevant Drawing Data:**")
                for i, result in enumerate(relevant_draws[:2], 1):
                    drawing = result.get('metadata', {}).get('drawing_name', 'unknown')
                    entity_type = result.get('metadata', {}).get('entity_type', 'unknown')
                    score = result.get('score', 0)
                    context_parts.append(f"  {i}. [Drawing: {drawing}, Type: {entity_type}] (relevance: {score:.2f})")
            
            if context_parts:
                total_convs = len(recent_convs) if session_id and recent_convs else len(relevant_convs)
                logger.info(
                    f"✓ Retrieved {total_convs} conversations, {len(relevant_docs)} documents, "
                    f"{len(relevant_draws)} drawings, {len(relevant_coords)} coordinates as context"
                )
                return "\n".join(context_parts)
            
        except Exception as e:
            logger.warning(f"⚠ Context retrieval failed: {e}")
        
        return ""
    
    def _store_conversation(
        self, 
        query: str, 
        response: str, 
        session_id: str,
        llm_used: str = "primary"
    ) -> None:
        """
        Store a conversation exchange in the vector store for future context.
        
        Args:
            query: The user's original query
            response: The agent's response
            session_id: Session identifier for grouping conversations
            llm_used: Which LLM was used for the response
        """
        if self.vector_store is None:
            return
        
        if not getattr(self.settings, 'auto_store_conversations', True):
            return
        
        try:
            timestamp = datetime.now().isoformat()
            
            # Store the user query
            self.vector_store.add_conversation(
                role="user",
                content=query,
                session_id=session_id,
                metadata={
                    "timestamp": timestamp,
                    "type": "query"
                }
            )
            
            # Store the assistant response
            self.vector_store.add_conversation(
                role="assistant",
                content=response,
                session_id=session_id,
                metadata={
                    "timestamp": timestamp,
                    "type": "response",
                    "llm_used": llm_used
                }
            )
            
            logger.debug(f"✓ Stored conversation in vector store (session: {session_id[:8]}...)")
            
        except Exception as e:
            logger.warning(f"⚠ Failed to store conversation: {e}")
    
    def set_session_id(self, session_id: str) -> None:
        """
        Set a persistent session ID for conversation continuity.
        
        Using the same session ID across queries allows the agent to maintain
        context and retrieve relevant past conversations.
        
        Args:
            session_id: Unique identifier for this conversation session
        """
        self._current_session_id = session_id
        logger.info(f"Session ID set: {session_id[:8]}...")
    
    def get_session_id(self) -> str:
        """
        Get the current session ID, generating one if not set.
        
        Returns:
            The current session ID
        """
        if self._current_session_id is None:
            self._current_session_id = str(uuid.uuid4())
        return self._current_session_id

    @staticmethod
    def _is_affirmative_reply(text: str) -> bool:
        normalized = " ".join((text or "").strip().lower().split())
        return normalized in {
            "yes",
            "y",
            "yes.",
            "ok",
            "okay",
            "proceed",
            "continue",
            "allow",
            "go ahead",
            "approved",
            "permission granted",
        }

    @staticmethod
    def _is_negative_reply(text: str) -> bool:
        normalized = " ".join((text or "").strip().lower().split())
        return normalized in {
            "no",
            "n",
            "no.",
            "deny",
            "denied",
            "don't",
            "do not",
            "stop",
            "cancel",
            "permission denied",
        }

    @staticmethod
    def _is_affirmative_permission_reply(text: str) -> bool:
        """Lenient yes-detector used ONLY when the previous assistant turn asked
        for internet permission. Accepts natural replies like "yes please",
        "yes, you may search the internet", "sure", "go ahead and search"."""
        n = " ".join((text or "").strip().lower().split()).rstrip(".!")
        if not n:
            return False
        if SurvyAIAgent._is_affirmative_reply(n):
            return True
        starters = ("yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please do",
                    "go ahead", "affirmative", "do it", "please")
        if any(n == s or n.startswith(s + " ") or n.startswith(s + ",") for s in starters):
            return True
        grants = ("you may search", "you can search", "search the internet",
                  "search online", "permission granted", "go ahead and search",
                  "please search", "browse the web", "look it up online")
        return any(g in n for g in grants)

    @staticmethod
    def _is_negative_permission_reply(text: str) -> bool:
        """Lenient no-detector for the same conversational permission context."""
        n = " ".join((text or "").strip().lower().split()).rstrip(".!")
        if not n:
            return False
        if SurvyAIAgent._is_negative_reply(n):
            return True
        starters = ("no", "nope", "nah", "don't", "do not", "please don't",
                    "negative", "cancel", "stop")
        if any(n == s or n.startswith(s + " ") or n.startswith(s + ",") for s in starters):
            return True
        denies = ("do not search", "don't search", "no internet", "without internet",
                  "stay offline", "use offline")
        return any(d in n for d in denies)

    def _extract_last_assistant_turn(self, query: str) -> str:
        """Return the most recent assistant message from injected GUI history."""
        block = self._extract_history_block(query)
        idx = block.rfind("Assistant:")
        if idx == -1:
            return ""
        text = block[idx + len("Assistant:"):].strip()
        for marker in ("\nUser:", "\n--- End of History", "\n--- Exchange"):
            pos = text.find(marker)
            if pos != -1:
                text = text[:pos].strip()
        return text

    def _extract_last_user_turn(self, query: str) -> str:
        """Return the most recent user message from injected GUI history."""
        block = self._extract_history_block(query)
        idx = block.rfind("User:")
        if idx == -1:
            return ""
        text = block[idx + len("User:"):].strip()
        for marker in ("\nAssistant:", "\n--- End of History", "\n--- Exchange"):
            pos = text.find(marker)
            if pos != -1:
                text = text[:pos].strip()
        return text

    @staticmethod
    def _score_optional_offer(offer: str) -> int:
        """Prefer analysis/search/deeper-extract offers over invented CAD plots."""
        ol = (offer or "").lower()
        if not ol:
            return -999
        score = 0
        if any(
            k in ol
            for k in (
                "search",
                "internet",
                "deeper",
                "more detail",
                "extract",
                "read",
                "dig",
                "look up",
                "look carefully",
                "more thorough",
                "full extraction",
                "retrieve",
                "transformation",
                "crs",
                "epsg",
                "fit comparison",
                "gis-based fit",
                "geometric fit",
                "footprint",
                "large enough",
            )
        ):
            score += 10
        if any(
            k in ol
            for k in (
                ".dwg",
                "cadastral",
                "cad plan",
                "cad drawing",
                "plot a",
                "plot the",
                "generate a plan",
                "generate cad",
                "autocad",
                "survey plan",
            )
        ):
            score -= 8
        return score

    def _extract_offer_from_assistant_text(self, assistant_text: str) -> str:
        """Pull the best optional next-step offer from the last assistant turn.

        When several offers appear, prefer search / deeper extract / analysis over
        CAD plotting so a bare "go ahead" after Excel inspect does not invent a DWG.
        """
        t = (assistant_text or "").strip()
        if not t:
            return ""
        patterns = (
            r"(?is)\bif you want,?\s+i can(?: also| still)?\s+[^.?\n]+[.?\n]",
            r"(?is)\bi can next\s+[^.?\n]+[.?\n]",
            r"(?is)\bi can still\s+[^.?\n]+[.?\n]",
            r"(?is)\bwould you like me to\s+[^.?\n]+[.?\n]",
            r"(?is)\bshall i\s+[^.?\n]+[.?\n]",
            r"(?is)\bi can(?: also| still)?\s+(?:try|do|run|perform|search|extract|read|dig)[^.?\n]+[.?\n]",
        )
        offers: List[str] = []
        seen: set[str] = set()
        for pat in patterns:
            for m in re.finditer(pat, t):
                cleaned = re.sub(r"\s+", " ", m.group(0)).strip()
                key = cleaned.lower()
                if cleaned and key not in seen:
                    seen.add(key)
                    offers.append(cleaned)
        if not offers:
            return ""
        offers.sort(key=self._score_optional_offer, reverse=True)
        return offers[0]

    def _is_bare_affirmative_reply(self, text: str) -> bool:
        """True for short replies like 'yes' / 'go ahead' with no new task wording."""
        t = " ".join((text or "").strip().lower().split())
        if not t or len(t.split()) > 5:
            return False
        return self._is_affirmative_reply(t)

    def _last_assistant_offered_deeper_analysis(self, query: str) -> bool:
        """True when the last assistant turn offered deeper extract/search (not CAD)."""
        last = (self._extract_last_assistant_turn(query) or "").lower()
        if not last:
            return False
        markers = (
            "deeper",
            "dig deeper",
            "more detail",
            "more details",
            "full extraction",
            "document_get_text_force",
            "look more carefully",
            "more thorough",
            "search the internet",
            "try a deeper",
            "extract more",
            "read the spreadsheet again",
            "force extraction",
        )
        if not any(m in last for m in markers):
            return False
        # If the only concrete offer is clearly a CAD plot, do not treat as deeper-analysis.
        offer = self._extract_offer_from_assistant_text(last)
        if offer and explicit_cadastral_plot_intent(offer) and self._score_optional_offer(offer) < 0:
            return False
        return True

    def _resolve_affirmative_to_last_offer(
        self, raw_query: str, routing_query: str
    ) -> Optional[str]:
        """Expand bare 'yes' into the assistant's last optional offer (anti-context-loss)."""
        rq = (routing_query or "").strip()
        if not self._is_bare_affirmative_reply(rq):
            return None
        if self._last_assistant_asked_internet_permission(raw_query):
            return None
        # Let GIS fit affirmation reconstruction own this turn (keeps session artifacts).
        if self._last_assistant_offered_gis_fit(raw_query):
            return None
        if self._last_assistant_offered_session_docx_save(raw_query):
            return None
        if self._is_pdf_replot_affirmation(rq, raw_query):
            return None
        last = self._extract_last_assistant_turn(raw_query)
        if not last:
            return None
        prior_user = self._extract_last_user_turn(raw_query)
        offer = self._extract_offer_from_assistant_text(last)

        # After an Excel/CSV inspect turn, bare "go ahead" means deepen analysis —
        # never invent a CAD/.dwg deliverable the user did not request.
        if (
            prior_user
            and is_tabular_inspect_only_request(prior_user)
            and (
                self._last_assistant_offered_deeper_analysis(raw_query)
                or not (offer and explicit_cadastral_plot_intent(offer))
                or self._score_optional_offer(offer or "") >= 0
            )
        ):
            if offer and not explicit_cadastral_plot_intent(offer):
                return (
                    f"The user replied affirmatively ({rq!r}) to your last offer: {offer} "
                    "Proceed with ONLY that offer using the same spreadsheet/files from the "
                    "immediately preceding exchange. Do NOT create, plot, or save any CAD/.dwg "
                    "file unless the user explicitly asked for a CAD plan."
                )
            return (
                f"The user replied affirmatively ({rq!r}) after a spreadsheet read/extract turn. "
                "Continue with a deeper extraction/analysis of that same workbook using "
                "document/Excel read tools (or internet search only if you already asked and "
                "they granted permission). Do NOT call excel_cadastral_plot / "
                "cadastral_compose_and_plot / cadastral_generate_from_prompt and do NOT invent "
                "a .dwg filename or CAD drawing."
            )

        if offer and explicit_cadastral_plot_intent(offer) and prior_user and is_tabular_inspect_only_request(prior_user):
            return (
                f"The user replied affirmatively ({rq!r}) after inspecting a spreadsheet. "
                "Do NOT treat that as permission to plot CAD. Deepen the spreadsheet "
                "extraction/analysis only. Call a cadastral plot tool only after an explicit "
                "user request to generate/create/plot a .dwg/CAD plan."
            )

        if offer:
            return (
                f"The user replied affirmatively ({rq!r}) to your last offer: {offer} "
                "Proceed with ONLY that offer using the same files and workspace from the "
                "immediately preceding exchange. Do NOT switch to unrelated workflows "
                "(e.g. do not run volume/CutFill if the active task is coordinate conversion; "
                "do not plot CAD if the offer was only deeper extract/search)."
            )
        return (
            f"The user replied affirmatively ({rq!r}) to your immediately prior message. "
            "Execute ONLY the optional next step you most recently offered there. "
            "Do NOT resume older unrelated workflows from earlier in the session. "
            "Do NOT invent CAD/.dwg outputs the user did not request."
        )

    @staticmethod
    def _extract_history_block(query: str) -> str:
        """Return only the injected conversation-history portion of a GUI query
        (everything before the current request marker)."""
        marker = "NOW, the user wants you to continue with this new request:"
        if marker in (query or ""):
            return query.split(marker)[0]
        return query or ""

    def _last_assistant_asked_internet_permission(self, query: str) -> bool:
        """True when the most recent assistant turn in the injected history asked
        the user for permission to search the internet."""
        block = self._extract_history_block(query)
        idx = block.rfind("Assistant:")
        if idx == -1:
            return False
        last_assistant = block[idx + len("Assistant:"):].lower()
        markers = (
            "search the internet", "search online", "browse the web",
            "may i search", "you may search", "permission to search",
            "need explicit permission", "(yes/no)", "up-to-date information",
            "latest official appointment", "latest appointment",
        )
        return any(m in last_assistant for m in markers)

    @staticmethod
    def _parse_option_selection(text: str) -> Optional[int]:
        """Return 1-based option index when the user selects a numbered menu item."""
        t = " ".join((text or "").strip().lower().split())
        if not t:
            return None
        patterns = (
            r"\b(?:go\s+ahead\s+with|proceed\s+with|use|choose|pick|select|with)\s+"
            r"(?:option|choice|method|approach)\s*(?:#|no\.?|number)?\s*(\d+)\b",
            r"\b(?:option|choice|method|approach)\s*(?:#|no\.?|number)?\s*(\d+)\b",
            r"^\s*(\d+)\s*[.)]?\s*$",
        )
        for pat in patterns:
            m = re.search(pat, t, flags=re.IGNORECASE)
            if m:
                try:
                    n = int(m.group(1))
                except Exception:
                    continue
                if 1 <= n <= 20:
                    return n
        return None

    @staticmethod
    def _extract_numbered_options(assistant_text: str) -> Dict[int, str]:
        """Parse '1. … / 2. …' style menus from the last assistant turn."""
        text = assistant_text or ""
        options: Dict[int, str] = {}
        for m in re.finditer(
            r"(?m)^\s*(?:[-*]\s*)?(\d{1,2})[.)]\s+([^\n]+)",
            text,
        ):
            try:
                n = int(m.group(1))
            except Exception:
                continue
            body = re.sub(r"\s+", " ", (m.group(2) or "").strip())
            if body and 1 <= n <= 20:
                options[n] = body
        # Fallback: inline "1. foo 2. bar" when not line-broken.
        if len(options) < 2:
            for m in re.finditer(
                r"(?:^|[\s;])(\d{1,2})[.)]\s+([^0-9].+?)(?=(?:[\s;]\d{1,2}[.)]\s+)|\Z)",
                text,
                flags=re.S,
            ):
                try:
                    n = int(m.group(1))
                except Exception:
                    continue
                body = re.sub(r"\s+", " ", (m.group(2) or "").strip())
                if body and 1 <= n <= 20:
                    options[n] = body
        return options

    @staticmethod
    def _looks_like_gis_followup_analysis(text: str) -> bool:
        """True for fit/containment / open-result follow-ups (not the original convert/plot job)."""
        tl = (text or "").lower()
        if not tl:
            return False
        # Living Atlas / pretrained land-cover is a different follow-up class — never
        # rewrite it into a landmark-fit instruction just because it says "deep learning".
        try:
            from survyai.gis_session_intent import looks_like_living_atlas_or_landcover_dl
            if looks_like_living_atlas_or_landcover_dl(text):
                return False
        except Exception:
            pass
        fit_markers = (
            "would fit",
            "fit within",
            "fit inside",
            "fit the",
            "fit comparison",
            "gis-based fit",
            "gis based fit",
            "practical gis",
            "footprint",
            "contain",
            "within each",
            "open arcgis result",
            "open arcgis",
            "ascertain if",
            "can it fit",
            "would they fit",
            "fit the eiffel",
            "structure within",
            "large enough",
            "footprint approximation",
        )
        # Bare "deep learning" alone is too broad (matches land-cover asks). Require a
        # fit/open-result companion phrase when that is the only DL cue.
        if "deep learning" in tl and any(
            p in tl
            for p in (
                "fit",
                "ascertain",
                "footprint",
                "open arcgis",
                "would",
                "large enough",
                "tower",
                "landmark",
            )
        ):
            return True
        if any(p in tl for p in fit_markers):
            return True
        has_ref = any(
            k in tl
            for k in (
                "tower",
                "building",
                "stadium",
                "structure",
                "monument",
                "bridge",
                "arena",
            )
        )
        has_parcel = any(
            k in tl for k in ("parcel", "polygon", "each of them", "each owner", "owner polygon")
        )
        return has_ref and has_parcel

    def _last_assistant_offered_gis_fit(self, query: str) -> bool:
        """True when the last assistant turn offered a practical GIS/parcel fit test."""
        last = (self._extract_last_assistant_turn(query) or "").lower()
        if not last:
            return False
        markers = (
            "gis-based fit",
            "gis based fit",
            "fit comparison",
            "practical gis",
            "footprint approximation",
            "parcel-vs-",
            "parcels are large enough",
            "which parcels",
            "geometric fit",
            "eiffel",
            "fit test",
            "fit comparison next",
            "fit check",
            "verified parcel",
            "gis session",
            "session files",
            "using the verified",
            "parcel data already",
            "i can still do this next",
            "if you want, i can still",
            "result_polygon_layer_count",
        )
        return any(m in last for m in markers)

    def _resolve_gis_fit_offer_affirmation(
        self, raw_query: str, routing_query: str
    ) -> Optional[str]:
        """
        Bind 'yes do a practical GIS-based fit comparison' (and similar) to the
        prior parcel/landmark fit goal + verified session GIS artifacts.
        Prevents the model from answering with a tools-comparison essay.
        """
        rq = (routing_query or "").strip()
        if not rq:
            return None
        # Affirmation of a fit offer, or explicit fit-comparison wording.
        affirming = (
            self._is_affirmative_permission_reply(rq)
            or rq.lower().startswith("yes")
            or rq.lower().startswith("ok")
            or rq.lower().startswith("sure")
            or rq.lower().startswith("please")
            or rq.lower().startswith("proceed")
        )
        fit_words = self._looks_like_gis_followup_analysis(rq)
        offered = self._last_assistant_offered_gis_fit(raw_query)
        if not ((affirming and offered) or (fit_words and (
            affirming or offered or self._extract_verified_gis_artifacts_from_history(raw_query)
        ))):
            return None
        artifacts = self._extract_verified_gis_artifacts_from_history(raw_query)
        # Recover the original landmark/parcel ask from history when present.
        analysis_ask = ""
        block = self._extract_history_block(raw_query)
        for line in block.splitlines():
            s = line.strip()
            if s.startswith("User:"):
                u = s[len("User:"):].strip()
                if self._looks_like_gis_followup_analysis(u) and "fit comparison" not in u.lower():
                    analysis_ask = u
        if not analysis_ask:
            analysis_ask = (
                "Perform a geometric fit test of the reference structure against each "
                "owner/parcel polygon from the open ArcGIS result in this conversation."
            )
        option_note = ""
        if fit_words:
            option_note = rq
        elif offered:
            option_note = (
                "Practical GIS-based geometric fit comparison of the reference footprint "
                "against each existing owner/parcel polygon (NOT a comparison of GIS software tools)."
            )
        return self._build_gis_fit_continuation(
            analysis_ask=analysis_ask,
            artifacts=artifacts,
            option_note=option_note,
        )

    def _extract_gis_paths_from_text(self, text: str) -> List[str]:
        """Extract GIS/file paths from a free-text blob (history or current message)."""
        paths: List[str] = []
        seen: set[str] = set()
        patterns = (
            r"`([A-Za-z]:\\[^`\n]+?\.(?:aprx|gdb|xlsx|csv|shp|geojson|gpkg|docx))`",
            r"`(/(?:[^`\n]+?\.(?:aprx|gdb|xlsx|csv|shp|geojson|gpkg|docx)))`",
            r"(?<![`\w])([A-Za-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]*?\.(?:aprx|gdb|xlsx|csv|shp|geojson|gpkg|docx))",
            r"(?<![`\w])(/(?:[\w.\-]+/)*[\w.\-]+\.(?:aprx|gdb|xlsx|csv|shp|geojson|gpkg|docx))",
        )
        for pat in patterns:
            for m in re.finditer(pat, text or "", flags=re.IGNORECASE):
                p = (m.group(1) if m.lastindex else m.group(0)).rstrip(".,);:]")
                key = p.lower()
                if key not in seen and len(p) > 4:
                    seen.add(key)
                    paths.append(p)
        return paths[:20]

    def _extract_verified_gis_artifacts_from_history(self, query: str) -> List[str]:
        """Collect verified GIS output paths mentioned in recent assistant turns."""
        return self._extract_gis_paths_from_text(self._extract_history_block(query))

    def _assistant_asked_for_gis_paths(self, query: str) -> bool:
        """True when the last assistant turn asked the user for .aprx / parcel layer paths."""
        last = (self._extract_last_assistant_turn(query) or "").lower()
        if not last:
            return False
        markers = (
            "project path",
            "parcel layer",
            "polygon layer",
            "layer/file",
            "layer file",
            "need the arcgis",
            "need either",
            "send me the",
            "provide the open",
            "arcgis project path",
            "open arcgis project path",
            "relevant polygon",
            "cannot be inspected",
            "arcpy isn't available",
            "arcpy isn",
            "arcpy not available",
        )
        return any(m in last for m in markers)

    def _looks_like_gis_path_provision(self, text: str) -> bool:
        """True when the user is supplying GIS file paths (often after being asked)."""
        t = (text or "").strip()
        if not t:
            return False
        paths = self._extract_gis_paths_from_text(t)
        if not paths:
            return False
        # At least one spatial/project artifact (not only a random .docx).
        spatial = any(
            p.lower().endswith(ext)
            for p in paths
            for ext in (".aprx", ".gdb", ".shp", ".xlsx", ".csv", ".geojson", ".gpkg")
        )
        if not spatial:
            return False
        tl = t.lower()
        provision_cues = (
            "here are the details",
            "project to use",
            "owner polygons",
            "arcgis pro project",
            "use these",
            "use this",
            "path:",
            ".aprx",
            ".shp",
            ".gdb",
        )
        # Path-heavy paste (2+ GIS files) counts even without cue phrases.
        return len(paths) >= 2 or any(c in tl for c in provision_cues)

    def _recover_pending_gis_fit_ask(self, query: str) -> str:
        """Recover the last parcel/landmark fit ask from injected history."""
        block = self._extract_history_block(query)
        for line in reversed(block.splitlines()):
            s = line.strip()
            if s.startswith("User:"):
                u = s[len("User:"):].strip()
                if self._looks_like_gis_followup_analysis(u) and "fit comparison" not in u.lower():
                    return u
        last = (self._extract_last_assistant_turn(query) or "").lower()
        if "eiffel" in last or "fit test" in last or "fit comparison" in last:
            return (
                "Perform a geometric fit test of the reference structure (e.g. Eiffel Tower) "
                "against each owner/parcel polygon from the open ArcGIS result in this conversation."
            )
        return (
            "Perform a geometric fit test of the reference structure against each "
            "owner/parcel polygon from the open ArcGIS / GIS result in this conversation."
        )

    def _enrich_gis_paths_for_pending_fit(
        self, raw_query: str, routing_query: str
    ) -> Optional[str]:
        """
        When the user pastes .aprx/.shp/.gdb paths after the agent asked for them
        (or after offering a fit test), bind those paths to the pending FIT analysis —
        never re-run the full convert/plot/buffer workflow.
        """
        current = (routing_query or "").strip()
        if not current or current.startswith("CONTINUATION (retain session GIS context"):
            return None
        if not self._looks_like_gis_path_provision(current):
            return None
        # Don't hijack an explicit full GIS job restatement.
        try:
            from survyai.gis_session_intent import looks_like_full_gis_workflow_request
            if looks_like_full_gis_workflow_request(current):
                return None
        except Exception:
            pass
        # Require a fit-related prior turn (asked for paths / offered fit / prior fit ask).
        hist_users = [
            s[len("User:"):].strip()
            for s in self._extract_history_block(raw_query).splitlines()
            if s.strip().startswith("User:")
        ]
        if not (
            self._assistant_asked_for_gis_paths(raw_query)
            or self._last_assistant_offered_gis_fit(raw_query)
            or any(self._looks_like_gis_followup_analysis(u) for u in hist_users)
        ):
            return None

        artifacts = self._extract_gis_paths_from_text(current)
        for p in self._extract_verified_gis_artifacts_from_history(raw_query):
            if p not in artifacts:
                artifacts.append(p)
        analysis_ask = self._recover_pending_gis_fit_ask(raw_query)
        return self._build_gis_fit_continuation(
            analysis_ask=analysis_ask,
            artifacts=artifacts,
            option_note=(
                "User supplied the ArcGIS/GIS file paths below — run the geometric fit "
                "analysis now. Do NOT re-execute the convert/plot/buffer workflow."
            ),
        )

    def _build_gis_fit_continuation(
        self,
        *,
        analysis_ask: str,
        artifacts: Sequence[str],
        option_note: str = "",
    ) -> str:
        """Instruction block that keeps verified GIS outputs in scope for fit tests."""
        parts = [
            "CONTINUATION (retain session GIS context — do not restart as a blank task):",
            f"Original analysis request: {analysis_ask or '(see conversation history)'}",
        ]
        if option_note:
            parts.append(f"User-selected approach: {option_note}")
        if artifacts:
            parts.append(
                "Already-verified GIS outputs from this conversation (REUSE these; "
                "do NOT ask the user to re-send parcel layers):\n- "
                + "\n- ".join(list(artifacts)[:20])
            )
        else:
            parts.append(
                "Search conversation history for verified .aprx / .gdb / converted .xlsx / "
                "Owner_Areas*.csv / .shp paths from the earlier successful GIS run and use those."
            )
        parts.append(
            "OBSERVE → THINK → ACT as an expert surveyor/GIS analyst:\n"
            "1) OBSERVE: parcels already exist as session files (converted Excel Owner+X/Y, "
            "owner polygon .shp/.gdb, and/or .aprx). A CURRENT open map with 0 layers is NOT "
            "a reason to stop. ArcGIS Pro may already be open — still use the file paths above.\n"
            "2) THINK: interpret 'deep learning' as rigorous spatial analysis unless a model was "
            "supplied. Source reference footprint/base dimensions (ask once if the user has a "
            "file; else ask internet permission for published dimensions).\n"
            "3) ACT: run geometric fit (AABB or oriented rectangle of the reference vs each "
            "owner polygon) in the project CRS.\n"
            "4) Prefer geopandas_execute on parcel .shp or converted Excel (Owner + X/Y). Use "
            "arcgis_execute_python_code with explicit .aprx/.gdb/.shp paths (Pro's Python has "
            "arcpy). NEVER gate on arcgis_get_project_info / local `import arcpy` / "
            "ArcGISProject('CURRENT') layer counts when session files exist.\n"
            "5) Print RESULT_FIT / RESULT_PARCEL_AREA / RESULT_REF_FOOTPRINT per owner.\n"
            "6) Do NOT ask for parcel/.aprx paths again when listed above. Do NOT invent a DL "
            "model. Do NOT write a tools-comparison essay (GeoPandas vs ArcGIS vs AutoCAD). "
            "Do NOT re-run the full Excel→convert→buffer→overlap job unless the user asked to."
        )
        return "\n".join(parts)

    def _reconstruct_gis_followup_intent(
        self, query: str, *, selected_option_text: str = ""
    ) -> Optional[str]:
        """
        Rebuild a multi-turn GIS follow-up (option pick + prior analysis ask +
        verified layers) so permission/option affirmations do not lose context.
        """
        block = self._extract_history_block(query)
        users: List[str] = []
        for line in block.splitlines():
            s = line.strip()
            if s.startswith("User:"):
                users.append(s[len("User:"):].strip())
        analysis_ask = ""
        option_reply = ""
        for u in reversed(users):
            if not u or self._is_affirmative_permission_reply(u) or self._is_negative_permission_reply(u):
                continue
            if self._parse_option_selection(u) is not None and not option_reply:
                option_reply = u
                continue
            if self._looks_like_gis_followup_analysis(u):
                analysis_ask = u
                break
        # Only reconstruct when this really is a GIS follow-up / option chain.
        if not analysis_ask and not (
            option_reply and self._extract_verified_gis_artifacts_from_history(query)
        ) and not selected_option_text:
            return None

        artifacts = self._extract_verified_gis_artifacts_from_history(query)
        option_note = selected_option_text or ""
        if not option_note and option_reply:
            n = self._parse_option_selection(option_reply)
            last = self._extract_last_assistant_turn(query)
            options = self._extract_numbered_options(last)
            if n and n not in options:
                for m in re.finditer(r"Assistant:(.*?)(?=\nUser:|\Z)", block, flags=re.S | re.I):
                    cand = self._extract_numbered_options(m.group(1) or "")
                    if n in cand:
                        options = cand
                        break
            if n and n in options:
                option_note = options[n]
            else:
                option_note = option_reply

        return self._build_gis_fit_continuation(
            analysis_ask=analysis_ask or "(see conversation history)",
            artifacts=artifacts,
            option_note=option_note,
        )

    def _enrich_current_gis_fit_followup(
        self, raw_query: str, routing_query: str
    ) -> Optional[str]:
        """
        First-turn enrichment: when the CURRENT ask is a parcel/landmark fit follow-up
        and this chat already verified GIS outputs, bind those paths into the task.
        """
        current = (routing_query or "").strip()
        if not current or not self._looks_like_gis_followup_analysis(current):
            return None
        # Already enriched / continuation block present.
        if current.startswith("CONTINUATION (retain session GIS context"):
            return None
        artifacts = self._extract_verified_gis_artifacts_from_history(raw_query)
        for p in self._extract_gis_paths_from_text(current):
            if p not in artifacts:
                artifacts.append(p)
        if not artifacts:
            return None
        return self._build_gis_fit_continuation(
            analysis_ask=current,
            artifacts=artifacts,
            option_note="",
        )

    def _enrich_repeated_full_gis_workflow(
        self, raw_query: str, routing_query: str
    ) -> Optional[str]:
        """Force tool re-execution when the user restates a full GIS job after a prior success."""
        from survyai.gis_session_intent import (
            build_gis_workflow_rerun_instruction,
            looks_like_full_gis_workflow_request,
            prior_assistant_reported_gis_success,
        )

        current = (routing_query or "").strip()
        if not current or not looks_like_full_gis_workflow_request(current):
            return None
        if current.startswith("RE-EXECUTE FULL GIS WORKFLOW"):
            return None
        hist = self._extract_history_block(raw_query)
        if not prior_assistant_reported_gis_success(hist):
            return None
        return build_gis_workflow_rerun_instruction(current)

    def _underlying_question_from_history(self, query: str) -> Optional[str]:
        """Recover the real question the assistant was about to answer when it
        asked for internet permission (the last substantive user turn), so an
        affirmative "yes" resolves to that question rather than the bare "yes".

        When the last user turn is an option pick (e.g. "go ahead with option 2")
        after a GIS fit follow-up, reconstruct the full analysis intent + verified layers.
        Do NOT rewrite Living Atlas / land-cover / pretrained-model asks into fit jobs.
        """
        block = self._extract_history_block(query)
        users: List[str] = []
        for line in block.splitlines():
            s = line.strip()
            if s.startswith("User:"):
                users.append(s[len("User:"):].strip())
        last_user: Optional[str] = None
        for u in reversed(users):
            if u and len(u) >= 8 and not self._is_affirmative_permission_reply(u) \
                    and not self._is_negative_permission_reply(u):
                last_user = u
                break
        if not last_user:
            return users[0] if users else None

        try:
            from survyai.gis_session_intent import looks_like_living_atlas_or_landcover_dl
            if looks_like_living_atlas_or_landcover_dl(last_user):
                return last_user
        except Exception:
            pass

        if self._parse_option_selection(last_user) is not None or self._looks_like_gis_followup_analysis(
            last_user
        ):
            reconstructed = self._reconstruct_gis_followup_intent(query)
            if reconstructed:
                return reconstructed
        return last_user

    @staticmethod
    def _response_is_internet_permission_ask_only(text: str) -> bool:
        """True when the model output is essentially only the internet permission ask."""
        t = (text or "").strip().lower()
        if not t:
            return False
        if "may i search the internet" not in t and "[internet_permission_request]" not in t:
            return False
        # If the reply already contains substantive GIS/work content, keep it.
        if any(
            k in t
            for k in (
                "result_",
                ".aprx",
                ".gdb",
                "living atlas",
                "grassland",
                "proceeding",
                "i will",
                "internet-sourced",
                "verified",
            )
        ):
            return False
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        return len(lines) <= 8

    @staticmethod
    def _wants_external_catalog_or_web(text: str) -> bool:
        """True when the task needs web/Living Atlas/catalog lookup (not only local files)."""
        tl = (text or "").lower()
        if not tl:
            return False
        try:
            from survyai.gis_session_intent import looks_like_living_atlas_or_landcover_dl
            if looks_like_living_atlas_or_landcover_dl(text):
                return True
        except Exception:
            pass
        keys = (
            "search the internet",
            "search online",
            "web search",
            "browse the web",
            "living atlas",
            "pretrained",
            "pre-trained",
            "already-trained",
            "trained model",
            "look it up online",
        )
        return any(k in tl for k in keys)

    def _internet_search_seed_from_intent(self, intent_text: str) -> str:
        """Prefer a compact web-search seed (reference object dims) over option prose."""
        text = intent_text or ""
        # Pull an explicit "Original analysis request:" line when present.
        m = re.search(
            r"Original analysis request:\s*(.+?)(?:\nUser-selected|\nAlready-verified|\nExecute as|\Z)",
            text,
            flags=re.S | re.I,
        )
        ask = (m.group(1).strip() if m else text).strip()
        ask_l = ask.lower()
        # Generic landmark/structure fit phrasing → search for footprint/dimensions.
        fit = re.search(
            r"(?:fit|fitting|contain|inside|within).{0,40}?"
            r"(?:the\s+)?([A-Z][\w]*(?:\s+[A-Z][\w]*){0,4}|eiffel\s+tower|"
            r"[A-Za-z][\w\-]*(?:\s+[A-Za-z][\w\-]*){0,3}\s+"
            r"(?:tower|building|stadium|arena|bridge|monument|footprint|structure))",
            ask,
            flags=re.I,
        )
        if fit:
            obj = re.sub(r"\s+", " ", fit.group(1)).strip(" .")
            if obj:
                return f"{obj} footprint dimensions meters base width length"
        if any(k in ask_l for k in ("footprint", "dimension", "fit", "tower", "structure")):
            # Strip GIS-local noise for search.
            cleaned = re.sub(
                r"\b(arcgis|polygon|parcel|owner|buffer|overlap|deep learning analysis)\b",
                " ",
                ask,
                flags=re.I,
            )
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned:
                return f"{cleaned} footprint dimensions meters"
        return ask

    @staticmethod
    def _extract_clean_question(text: str) -> str:
        """Strip injected history, permission tags, and boilerplate to recover the
        substantive question the user wants answered."""
        q = (text or "").strip()
        if not q:
            return ""
        marker = "NOW, the user wants you to continue with this new request:"
        if marker in q:
            q = q.split(marker)[-1].strip()
        for tag in (
            "[INTERNET_PERMISSION_GRANTED]",
            "[internet_permission_granted]",
            "[INTERNET_PERMISSION_DENIED]",
            "[internet_permission_denied]",
            "[INTERNET_PERMISSION_REQUEST]",
        ):
            q = q.replace(tag, "").strip()
        # Drop a lone permission reply if it slipped through.
        if SurvyAIAgent._is_affirmative_permission_reply(q) or SurvyAIAgent._is_negative_permission_reply(q):
            return ""
        return q.strip()

    @staticmethod
    def _is_current_fact_question(text: str) -> bool:
        """True when the question needs live/up-to-date external facts (office holders, etc.)."""
        import re as _re

        ql = (text or "").strip().lower()
        if not ql:
            return False
        patterns = (
            r"\bwho(?:'s| is| are)\s+the\b",
            r"\bwho(?:'s| is| are)\b.*\b(current|present|now|today|currently|sitting|incumbent)\b",
            r"\b(current|present|sitting|incumbent)\s+(surveyor[\s-]?general|president|vice[\s-]?president|governor|minister|chairman|chairperson|ceo|director[\s-]?general|head|office\s*holder|commissioner|secretary)\b",
            r"\bas of (today|now|this (year|month)|20\d{2})\b",
            r"\b(latest|most recent|up[\s-]?to[\s-]?date)\b.*\b(who|name|appointed|appointment|holder|office)\b",
            r"\bwho (won|leads|heads|holds|is leading|currently)\b",
        )
        return any(_re.search(p, ql) for p in patterns)

    def _optimize_internet_search_queries(self, question: str) -> List[str]:
        """Domain-agnostic search-query generation (rule-based, no LLM, no hard-coded topics).

        Used as a cheap default and as the fallback for the LLM-based rewriter.
        """
        q = self._extract_clean_question(question) or (question or "").strip()
        if not q:
            return []
        return _rule_based_query_variants(q, max_variants=6)

    def _rewrite_search_queries_with_llm(
        self,
        question: str,
        llm: Optional[BaseChatModel],
        *,
        max_queries: int = 5,
    ) -> List[str]:
        """Query understanding + rewriting, the way enterprise web assistants do it.

        Generates several diverse, high-recall search intents from ONE question.
        Fully domain-agnostic: the model is instructed to infer entities, synonyms,
        and likely authoritative phrasings for whatever the topic is. Falls back to
        deterministic rule-based variants if the LLM is unavailable or misbehaves.
        """
        clean = self._extract_clean_question(question) or (question or "").strip()
        if not clean:
            return []
        rule_based = _rule_based_query_variants(clean, max_variants=max_queries)
        if llm is None:
            return rule_based

        system = (
            "You rewrite a user question into diverse web-search queries for a retrieval "
            "engine. Output ONLY a compact JSON array of 3-5 short query strings (no prose). "
            "Rules: cover different phrasings and likely authoritative sources; expand acronyms; "
            "include the most specific entity/title; add a recency term (e.g. current year) ONLY "
            "if the question asks for current/latest facts. Do NOT invent facts or names."
        )
        user = f"Question: {clean}\nReturn JSON array of search queries only."
        try:
            msg = self._run_with_timeout(
                25, lambda: llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            )[0]
            text = msg.content if hasattr(msg, "content") else str(msg)
            text = str(text or "").strip()
            # Extract the JSON array even if wrapped in markdown/code fences.
            m = re.search(r"\[.*\]", text, re.S)
            arr = json.loads(m.group(0)) if m else json.loads(text)
            llm_queries = [str(x).strip() for x in arr if str(x).strip()]
        except Exception as exc:
            logger.info("LLM query rewriting unavailable (%s) — using rule-based variants.", exc)
            return rule_based

        merged: List[str] = []
        seen: set = set()
        for cand in [clean] + llm_queries + rule_based:
            cand = " ".join((cand or "").split()).strip().strip('"').strip()
            if cand and cand.lower() not in seen:
                seen.add(cand.lower())
                merged.append(cand)
        return merged[:max(3, max_queries)]

    def _assess_prompt_action(
        self,
        *,
        raw_query: str,
        routing_query: str,
        permission_granted: bool,
    ) -> PromptActionAssessment:
        """Deterministic pre-LLM assessment: what is this prompt and what should we do?"""
        clean = self._extract_clean_question(routing_query) or self._extract_clean_question(raw_query)
        routing = (routing_query or "").strip()
        intent = self._classify_query_intent(clean or routing)

        # 1) Short permission reply to a prior assistant internet ask.
        if self._last_assistant_asked_internet_permission(raw_query):
            if self._is_affirmative_permission_reply(routing) or self._is_affirmative_permission_reply(clean):
                underlying = self._underlying_question_from_history(raw_query) or clean
                search_q = (self._optimize_internet_search_queries(underlying) or [underlying])[0]
                return PromptActionAssessment(
                    kind="permission_affirm",
                    effective_query=underlying,
                    needs_internet=True,
                    internet_search_query=search_q,
                    min_complexity="average",
                    reason="User affirmed a prior internet-permission request.",
                )
            if self._is_negative_permission_reply(routing) or self._is_negative_permission_reply(clean):
                underlying = self._underlying_question_from_history(raw_query) or clean
                return PromptActionAssessment(
                    kind="permission_deny",
                    effective_query=underlying,
                    needs_internet=False,
                    min_complexity="simple",
                    reason="User declined a prior internet-permission request.",
                )

        # 2) Current-fact / office-holder lookup (needs web when interactive).
        if clean and self._is_current_fact_question(clean):
            search_q = (self._optimize_internet_search_queries(clean) or [clean])[0]
            return PromptActionAssessment(
                kind="current_fact_lookup",
                effective_query=clean,
                needs_internet=True,
                internet_search_query=search_q,
                min_complexity="average",
                reason="Question asks for a current real-world fact likely requiring web sources.",
            )

        # 3) General knowledge (offline-capable unless permission already granted for web).
        if intent == "knowledge" and clean:
            return PromptActionAssessment(
                kind="general_knowledge",
                effective_query=clean,
                needs_internet=permission_granted and self._is_current_fact_question(clean),
                internet_search_query=clean if permission_granted else None,
                min_complexity="average" if permission_granted else "simple",
                reason="Informational question.",
            )

        if intent == "task" or looks_like_file_driven_task(clean or routing):
            from survyai.prompt_router import is_heavy_geospatial_task

            q_task = clean or routing
            heavy = is_heavy_geospatial_task(q_task)
            return PromptActionAssessment(
                kind="file_task",
                effective_query=q_task,
                needs_internet=False,
                min_complexity="complex" if heavy else "average",
                reason="File-driven or operational task.",
            )
        if intent == "continuation":
            return PromptActionAssessment(
                kind="continuation",
                effective_query=clean or routing,
                needs_internet=False,
                min_complexity="average",
                reason="In-session continuation of a prior task.",
            )

        if self._is_bare_affirmative_reply(clean or routing):
            resolved = self._resolve_affirmative_to_last_offer(raw_query, routing) or (clean or routing)
            return PromptActionAssessment(
                kind="continuation",
                effective_query=resolved,
                needs_internet=False,
                min_complexity="average",
                reason="Affirmative reply bound to last assistant offer.",
            )

        return PromptActionAssessment(
            kind="other",
            effective_query=clean or routing,
            needs_internet=False,
            min_complexity="average",
            reason="No specialised action detected.",
        )

    def _format_evidence_pack(self, evidence: List[Dict[str, Any]]) -> str:
        """Render ranked, trust-scored evidence as a numbered, citable pack for the LLM."""
        lines: List[str] = []
        for i, e in enumerate(evidence, 1):
            title = (e.get("title") or "").strip()
            url = (e.get("url") or "").strip()
            domain = (e.get("domain") or "").strip()
            trust = e.get("trust")
            body = (e.get("extracted") or e.get("snippet") or "").strip()
            if not (title or body):
                continue
            trust_str = f"{trust:.2f}" if isinstance(trust, (int, float)) else "n/a"
            lines.append(
                f"[{i}] {title}\n"
                f"    source: {url} (domain: {domain}, trust: {trust_str})\n"
                f"    evidence: {body[:900]}"
            )
        return "\n".join(lines)

    def _run_factual_web_lookup_pipeline(
        self,
        *,
        question: str,
        llm: BaseChatModel,
        model_name_used: Optional[str],
        search_queries: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Enterprise-style web answer: evidence retrieval → grounded synthesis.

        Stages (all in `utils.internet.research`): query variants → multi-source
        over-fetch → trust+relevance re-rank → page reading → evidence pack +
        cross-source confidence. The LLM then synthesises ONLY from the evidence
        and must cite sources. Total LLM cost: 1 rewrite call (cheap) + 1 synthesis.
        """
        effective = self._extract_clean_question(question) or (question or "").strip()
        if not effective:
            return {"success": False, "error": "empty_question", "response": ""}

        # Stage 1: query understanding + rewriting (LLM, cheap; rule-based fallback).
        variants = search_queries or self._rewrite_search_queries_with_llm(effective, llm)
        if not variants:
            variants = [effective]

        # Stages 2-6: retrieval, ranking, page reading, evidence + confidence.
        _max_sources = int(getattr(self.settings, "web_research_max_sources", 8) or 8)
        _fetch_pages = int(getattr(self.settings, "web_research_fetch_pages", 4) or 0)
        try:
            pack = _web_research(
                effective,
                query_variants=variants,
                max_sources=_max_sources,
                fetch_pages=_fetch_pages,
                read_pages=_fetch_pages > 0,
            )
        except Exception as exc:
            logger.warning("Web research pipeline failed: %s", exc)
            pack = {"success": False, "error": str(exc)}

        evidence = pack.get("evidence") or []
        if not pack.get("success") or not evidence:
            return {
                "success": False,
                "error": pack.get("error", "no_web_results"),
                "response": (
                    "I searched the web across multiple sources but could not retrieve "
                    "reliable evidence for that question. You can try rephrasing it or "
                    "naming the specific entity/organisation."
                ),
            }

        confidence = float(pack.get("confidence") or 0.0)
        evidence_block = self._format_evidence_pack(evidence)
        confidence_note = (
            "Multiple independent, trustworthy sources corroborate the evidence."
            if confidence >= 0.6 and pack.get("distinct_domains", 0) >= 2
            else (
                "Evidence is limited or comes from few sources — flag remaining "
                "uncertainty explicitly and avoid overstating confidence."
            )
        )

        # Stage 7-9: grounded synthesis with citation enforcement + verification mindset.
        synth_system = (
            "You are SurvyAI answering a factual question. You are given an EVIDENCE PACK: "
            "numbered, trust-scored web sources with extracted on-page text. Follow STRICTLY:\n"
            "1. Treat search as retrieval of EVIDENCE, not answers. Reason over the evidence.\n"
            "2. Lead with a direct, concise answer (name/title/date/value as applicable).\n"
            "3. EVERY factual claim must be grounded in the evidence and cite its source "
            "number(s) inline like [1], [2]. Do NOT state facts that lack supporting evidence.\n"
            "4. Prefer higher-trust sources (official/government/primary) over blogs/forums "
            "when they conflict, and say so if sources disagree.\n"
            "5. If the evidence is insufficient or only weakly supports an answer, say that "
            "plainly rather than guessing.\n"
            "6. End with a section titled exactly: 'Internet-sourced (external) information' "
            "listing the URLs you actually relied on (matching your [n] citations).\n"
            "7. NEVER ask for permission to search — the search is already complete. NEVER ask "
            "the user to choose between options or restate the question."
        )
        synth_user = (
            f"QUESTION:\n{effective}\n\n"
            f"EVIDENCE PACK (ranked by relevance × trust):\n{evidence_block}\n\n"
            f"RETRIEVAL CONFIDENCE: {confidence:.2f} — {confidence_note}\n\n"
            "Write the grounded, cited answer now."
        )
        try:
            report_msg = llm.invoke([
                SystemMessage(content=synth_system),
                HumanMessage(content=synth_user),
            ])
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "response": f"Evidence was found but synthesis failed: {exc}",
            }

        body = report_msg.content if hasattr(report_msg, "content") else str(report_msg)
        return {
            "success": True,
            "response": str(body or "").strip(),
            "model_name": model_name_used,
            "search_queries": variants,
            "result_count": len(evidence),
            "confidence": confidence,
            "sources": pack.get("sources") or [],
        }

    # ==========================================================================
    # COMPLEXITY DETECTION
    # ==========================================================================
    
    def _parse_user_model_tier_override(self, query: str) -> Optional[Literal["simple", "average", "complex"]]:
        """
        If the user explicitly asks for a cheaper/faster or stronger model tier, honor it
        (commercial workflows: surveyors can opt into depth or speed in natural language).
        """
        import re

        q = (query or "").strip()
        if not q:
            return None
        ql = q.lower()

        complex_pats = (
            r"\buse\s+(the\s+)?(most\s+capable|strongest|best|highest[\s-]quality)\s+model\b",
            r"\buse\s+(the\s+)?complex\s+model\b",
            r"\bcomplex\s+reasoning\b",
            r"\bdeep\s+reasoning\b",
            r"\bmaximum\s+(reasoning|quality|depth)\b",
            r"\btier\s*:\s*complex\b",
            r"\bsmartest\s+model\b",
        )
        average_pats = (
            r"\buse\s+(the\s+)?(standard|balanced|default)\s+model\b",
            r"\buse\s+(the\s+)?mini\s+model\b",
            r"\btier\s*:\s*(average|standard)\b",
        )
        simple_pats = (
            r"\buse\s+(the\s+)?(simple|fast|nano|cheapest|lowest[\s-]cost|smallest)\s+model\b",
            r"\bquick\s+answer\s+only\b",
            r"\bfaster\s+cheaper\s+model\b",
            r"\btier\s*:\s*simple\b",
        )
        for pat in complex_pats:
            if re.search(pat, ql, flags=re.IGNORECASE):
                return "complex"
        for pat in average_pats:
            if re.search(pat, ql, flags=re.IGNORECASE):
                return "average"
        for pat in simple_pats:
            if re.search(pat, ql, flags=re.IGNORECASE):
                return "simple"
        return None

    def _classify_query_intent(self, current_query: str) -> str:
        """Lightweight intent classifier for the CURRENT user turn.

        Implements the "intent classification on the latest turn" pattern used by
        modern assistants so that conversation history can NEVER, by itself, drive
        tool selection or destructive operations.

        Returns one of:
        - "task"          : current message itself requests a file/CAD/GIS/report op
        - "continuation"  : short follow-up that refines/continues the prior task
                            (e.g. "add a road", "change the title", "now export it")
        - "knowledge"     : a self-contained informational/explanatory question
        - "other"         : anything that doesn't clearly match the above

        This is heuristic and dependency-free (no extra LLM call) to keep latency
        low; it is intentionally conservative — when unsure it returns "other",
        which routes to the normal LLM path (never a destructive fast-path).
        """
        q = (current_query or "").strip()
        ql = q.lower()
        if not ql:
            return "other"

        # 1) Explicit task signals in the CURRENT message (paths / file types / verbs).
        if looks_like_file_driven_task(q):
            return "task"
        task_verbs = (
            "plot", "draw", "generate", "create plan", "create a plan", "cadastral",
            "export", "convert", "compute volume", "cutfill", "cut fill", "idw",
            "open the drawing", "open drawing", "batch", "replot",
        )
        if any(v in ql for v in task_verbs):
            return "task"

        # 2) Continuation / refinement: short, action-oriented follow-ups that lean
        #    on the active task (anaphora or in-session CAD modification keywords).
        mod_keywords = (
            "add another road", "add a road", "add road", "add access road",
            "change the title", "change title", "set title", "update title",
            "modify the plan", "modify plan", "edit the plan", "edit plan",
            "the other side", "change the plan title", "update the plan title",
        )
        word_count = len(ql.split())
        anaphora = any(
            p in f" {ql} " for p in (
                " it ", " this ", " that ", " these ", " those ", " the same ",
                " the plan", " the drawing", " the report", " the output", " above ",
            )
        )
        if word_count <= 14 and (any(k in ql for k in mod_keywords) or anaphora):
            return "continuation"

        # 3) Knowledge / informational question (self-contained).
        knowledge_starts = (
            "what is", "what are", "who", "when", "where", "why", "how", "explain",
            "describe", "tell me about", "give a brief", "give me a brief",
            "history of", "summary of the history", "define", "compare",
        )
        if any(ql.startswith(s) or f" {s}" in f" {ql}" for s in knowledge_starts):
            return "knowledge"

        return "other"

    def _should_direct_answer_non_file_prompt(self, routing_query: str, prompt_action: Any, intent: str) -> bool:
        """Bypass LangGraph/tools for self-contained non-file knowledge prompts."""
        q = (routing_query or "").strip()
        if not q:
            return False
        if looks_like_file_driven_task(q) or self._extract_document_paths(q):
            return False
        if prompt_action.kind in ("current_fact_lookup", "permission_affirm"):
            return False
        return bool(getattr(self.settings, "fast_mode_non_file_prompts", False) and intent == "knowledge")

    def _run_direct_knowledge_answer(
        self,
        *,
        question: str,
        llm: Any,
        model_name_used: Optional[str],
        timeout_seconds: int = 60,
    ) -> Dict[str, Any]:
        """Single LLM call for explanatory Q&A; no tools, no graph, no stale task state."""
        from langchain_core.messages import HumanMessage, SystemMessage

        if self._llm_is_ollama(llm):
            ok, ram_err, _num_ctx = _ollama_ram_policy(
                model_name_used or self._resolve_provider_model_name("ollama")
            )
            if not ok:
                return {"success": False, "error": ram_err, "model_name": model_name_used}
            timeout_seconds = min(int(timeout_seconds or 60), 90)

        system = (
            "You are SurvyAI, a professional surveying, geospatial, and CAD assistant. "
            "Answer only the user's current question. Do not continue previous CAD/file tasks, "
            "do not propose file operations, and do not mention unrelated prior work. "
            "For surveying history/principles, be accurate, practical, and concise."
        )
        user = (question or "").strip()
        msg, err, timed_out = self._run_with_timeout(
            timeout_seconds,
            lambda: llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]),
            llm_model_name=model_name_used,
            serialize_llm=True,
        )
        if timed_out:
            return {
                "success": False,
                "error": f"Direct knowledge LLM call timed out after {timeout_seconds} seconds.",
            }
        if err:
            return {"success": False, "error": str(err)}
        raw = getattr(msg, "content", msg)
        if isinstance(raw, list):
            raw = "\n".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in raw
            )
        return {
            "success": True,
            "response": str(raw or "").strip(),
            "model_name": model_name_used,
        }

    @staticmethod
    def _is_retry_request(current_query: str) -> bool:
        q = (current_query or "").strip().lower().strip(".! ")
        return q in {
            "try again",
            "retry",
            "rerun",
            "run it again",
            "do it again",
            "try once more",
            "repeat the last request",
            "repeat last request",
        }

    @classmethod
    def _is_retry_request_from_routing_context(
        cls,
        *,
        raw_query: str,
        extracted_query: str,
        routing_query: str,
    ) -> bool:
        """
        Detect short retry follow-ups without losing the history blob needed to rerun.

        The GUI may wrap the current user message in continuation/history text.
        Routing normally uses the extracted current request, but retry detection
        should also inspect marker tails and final user/request lines in case the
        extraction was empty or trimmed differently.
        """
        candidates: List[str] = []
        for item in (routing_query, extracted_query):
            if item:
                candidates.append(item)

        raw = raw_query or ""
        marker = "NOW, the user wants you to continue with this new request:"
        if marker in raw:
            candidates.append(raw.split(marker)[-1])
        elif raw:
            candidates.append(raw)

        # Common transcript/context line forms: "User: retry" or "New request: retry".
        for line in reversed([ln.strip() for ln in raw.splitlines() if ln.strip()]):
            m = re.match(
                r"^(?:user|human|current\s+user|new\s+request|request)\s*:\s*(.+)$",
                line,
                flags=re.IGNORECASE,
            )
            if m:
                candidates.append(m.group(1))
                break
        if raw:
            tail_lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            if tail_lines:
                candidates.append(tail_lines[-1])

        for candidate in candidates:
            text = str(candidate or "").strip().strip("'\"")
            if cls._is_retry_request(text):
                return True
        return False

    def _detect_task_complexity(self, query: str) -> Literal["simple", "average", "complex"]:
        """
        Analyze query to determine task complexity level.
        
        This heuristic-based approach categorizes queries into:
        - simple: Basic questions, lookups, single operations
        - average: Multi-step tasks, calculations, coordinate conversions
        - complex: Multi-tool workflows, complex reasoning, analysis tasks
        
        Args:
            query: The user's query string
            
        Returns:
            One of "simple", "average", or "complex"
        """
        import re

        query_lower = (query or "").lower()
        raw = query or ""

        # Production GIS volumetric / raster / multi-path jobs → complex tier.
        if len(re.findall(r"[a-zA-Z]:\\", raw)) >= 3:
            return "complex"

        # Cheap accurate tier: short historic / definitional lookups (no tool orchestration).
        _historic_simple = (
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
        _toolish = (
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
        if (
            any(s in query_lower for s in _historic_simple)
            and not any(s in query_lower for s in _toolish)
            and len(query_lower.split()) <= 40
        ):
            return "simple"

        # Unordered / multi-step geospatial with unclear sequence → complex (max accuracy).
        _unordered = (
            "in any order",
            "no particular order",
            "not sure which first",
            "whichever first",
            "multiple difficult",
            "several difficult",
            "no order",
            "unordered",
        )
        if any(s in query_lower for s in _unordered) and any(
            s in query_lower
            for s in ("gis", "geospatial", "arcgis", "survey", "cadastral", "coordinate", "polygon")
        ):
            return "complex"

        _gis_heavy = (
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
        _gis_tool_orchestration = (
            "polygon",
            "buffer",
            "overlap",
            "symbology",
            "fill color",
            "tight-fitting",
            "tight fitting",
            "owner",
            "convert",
            "mid-belt",
            "midbelt",
            "utm",
            "arcgis",
            "arcgis pro",
            "geodatabase",
            ".gdb",
            "feature class",
        )
        # Heavy raster / DL / volume analysis needs the complex tier.
        if any(s in query_lower for s in _gis_heavy) and (
            "arcgis" in query_lower or "arcpy" in query_lower or "raster" in query_lower
        ):
            return "complex"
        # Multi-step Excel→CRS→polygons→buffers→overlaps is primarily tool orchestration:
        # prefer the balanced (average) tier for cost/speed, escalate on failure.
        gis_hits = sum(1 for s in _gis_tool_orchestration if s in query_lower)
        if gis_hits >= 3 or (
            ("arcgis" in query_lower or "arcgis pro" in query_lower)
            and any(s in query_lower for s in ("buffer", "overlap", "polygon", "convert"))
        ):
            return "average"
        if "arcpy" in query_lower and any(s in query_lower for s in ("raster", "idw", "cut", "fill", "gdb")):
            return "complex"

        # Simple task indicators (basic lookups, single operations)
        simple_indicators = [
            "what is", "what are", "tell me", "explain", "define",
            "show me", "list", "find", "search", "lookup",
            "is", "are", "can you", "does", "do you know"
        ]
        
        # Complex task indicators (multi-step, analysis, complex reasoning)
        complex_indicators = [
            "analyze", "compare", "calculate total", "compute volume",
            "multiple", "several", "all", "combine", "integrate",
            "correlate", "cross-reference", "relationship between",
            "create project", "generate report", "perform analysis",
            "extract and", "retrieve and", "process and",
            "coordinate system", "projection", "transformation",
            "calculate area", "measure distance", "boundary analysis"
        ]
        
        # Count keywords
        simple_count = sum(1 for indicator in simple_indicators if indicator in query_lower)
        complex_count = sum(1 for indicator in complex_indicators if indicator in query_lower)
        
        # Check for multiple tool usage (indicates complexity)
        tool_indicators = ["autocad", "arcgis", "excel", "coordinate", "convert", "extract"]
        tool_count = sum(1 for indicator in tool_indicators if indicator in query_lower)
        
        # Check query length (longer queries often more complex)
        word_count = len(query.split())
        
        # Determine complexity
        # Length alone is a weak signal: a 40-word definition is not flagship GIS.
        if complex_count >= 2 or tool_count >= 3:
            return "complex"
        elif complex_count >= 1 or tool_count >= 2 or word_count > 15:
            return "average"
        elif simple_count >= 1 and complex_count == 0 and tool_count <= 1:
            return "simple"
        else:
            # Default to average for ambiguous cases
            return "average"

    # ==========================================================================
    # AGENTIC RAG ROUTING
    # ==========================================================================

    def _decide_rag_route(self, query: str, interactive_mode: bool = False) -> RAGRouteDecision:
        """
        Agentic RAG router: decide whether to:
        - call LLM directly (no augmentation),
        - retrieve local context from VectorStore,
        - run permissioned internet search,
        - or do both (hybrid).

        This is intentionally lightweight and defaults to HEURISTIC routing to avoid
        extra LLM calls. (An LLM-router can be added later behind a settings flag.)
        """
        q = (query or "").strip()
        ql = q.lower()

        clean_routing = self._extract_clean_question(q) or q
        if self._is_bare_affirmative_reply(clean_routing) or self._is_bare_affirmative_reply(q):
            return RAGRouteDecision(
                route="llm_only",
                use_vector=False,
                use_internet=False,
                reason="Bare affirmative reply; use last-exchange context only (no vector/internet).",
            )

        # If query includes explicit input file paths, prefer tool/document pipelines over RAG.
        # (RAG is still allowed if user asks "based on our previous conversation" etc.)
        file_driven = looks_like_file_driven_task(q) or bool(self._extract_document_paths(q)) or bool(
            self._extract_image_paths(q)
        )

        wants_memory = any(k in ql for k in [
            "previous", "earlier", "last time", "as we discussed", "from our conversation",
            "continue", "resume", "what did you say", "you mentioned", "same as before",
        ])

        # Internet-needed signals (permissioned)
        # Expanded to catch more cases where internet search would be helpful
        # BUT: Exclude CAD/AutoCAD continuation tasks - these don't need internet
        cad_continuation_indicators = [
            "add another", "add a", "now add", "also add", "continue with",
            "draw another", "create another", "add to the", "modify the",
            "update the", "change the", "edit the", "in the same", "same drawing",
            "same file", "same dwg", "same dxf", "to the drawing", "to the plan"
        ]
        is_cad_continuation = any(indicator in ql for indicator in cad_continuation_indicators)
        
        # Also check if previous context mentions CAD/AutoCAD work
        has_cad_context = any(
            keyword in ql for keyword in [
                "autocad", "cad", "dwg", "dxf", "drawing", "plan", "survey plan",
                "pillar", "coordinates", "boundary", "road", "access road"
            ]
        )
        
        # Internet signals: ONLY phrases that unambiguously indicate the user needs
        # live external data (standards, citations, current events).
        # Deliberately EXCLUDE generic starters ("what is the", "what are the",
        # "problems", "issues", "challenges") that match every analytical question
        # and cause false-positive internet permission dialogs on follow-up reasoning
        # questions like "which is more correct?" or "why do they differ?".
        internet_signals = [
            "according to api", "api mpms", "api 653", "api standard", "astm", "iso",
            "latest version", "current version", "updated standard",
            "as of 2023", "as of 2024", "as of 2025", "as of 2026",
            "cite", "citations", "references", "journal", "peer-reviewed", "paper", "studies",
            "who said", "source", "link", "url",
            "search the internet", "search online", "look it up online",
            "find information", "look up on the web",
            "standards", "regulations", "requirements",
            "country-specific", "national standard", "local standard",
            # External geospatial catalogs / pretrained models
            "living atlas", "livingatlas", "pretrained", "pre-trained", "pre trained",
            "already-trained", "already trained", "trained model", "deep learning model",
            "esri land cover", "land cover model", "landcover model",
        ]
        wants_internet = any(s in ql for s in internet_signals)
        try:
            from survyai.gis_session_intent import looks_like_living_atlas_or_landcover_dl
            if looks_like_living_atlas_or_landcover_dl(q):
                wants_internet = True
        except Exception:
            pass

        # Current real-world facts (office holders, latest appointments, "as of now"
        # / "who is the …") almost always need live data. Detect them so the
        # deterministic permission dialog fires up-front instead of the LLM looping
        # on free-text permission requests.
        import re as _re
        current_fact_patterns = (
            r"\bwho(?:'s| is| are)\s+the\b",
            r"\bwho(?:'s| is| are)\b.*\b(current|present|now|today|currently|sitting|incumbent)\b",
            r"\b(current|present|sitting|incumbent)\s+(surveyor[\s-]?general|president|vice[\s-]?president|governor|minister|chairman|chairperson|ceo|director[\s-]?general|head|office\s*holder|commissioner|secretary)\b",
            r"\bas of (today|now|this (year|month)|20\d{2})\b",
            r"\b(latest|most recent|up[\s-]?to[\s-]?date)\b",
            r"\bwho (won|leads|heads|holds|is leading|currently)\b",
        )
        if any(_re.search(p, ql) for p in current_fact_patterns):
            wants_internet = True

        # Override: If this looks like CAD continuation, don't ask for internet
        # Also check if the query mentions continuation context
        has_continuation_context = (
            "=== CONTINUATION OF PREVIOUS WORK" in q or
            "=== CONVERSATION CONTEXT" in q or
            "--- Exchange" in q or
            "PREVIOUS CONVERSATION" in q.upper()
        )

        # Analytical follow-up signals: questions reasoning over already-computed
        # results from this conversation.  These should NEVER trigger internet search
        # because the answer lies in the conversation history + LLM domain knowledge.
        _result_reasoning_signals = [
            "their difference", "the difference", "why differ", "why do they differ",
            "more correct", "more accurate", "more reliable", "which is better",
            "most valid", "valid reason", "likely correct", "likely more",
            "explain the difference", "account for", "reason for",
        ]
        is_result_followup = any(s in ql for s in _result_reasoning_signals)

        if is_result_followup or (is_cad_continuation and has_cad_context) or (has_continuation_context and has_cad_context):
            wants_internet = False
            if is_result_followup:
                logger.info(
                    "🔍 Detected analytical follow-up about prior results — skipping internet; "
                    "injecting conversation context instead."
                )
                wants_memory = True  # Retrieve prior conversation to supply the numbers
            else:
                logger.info("🔧 Detected CAD continuation task - skipping internet search request")
                wants_memory = True

        # Local retrieval signals: user is asking about *their* stored materials,
        # prior runs, prior outputs, or asks to "search my documents".
        vector_signals = [
            "from my documents", "my document", "my drawing", "my dwg", "my dxf",
            "search the database", "search the vector store", "retrieve", "stored",
            "chroma", "vectordb", "what did we store", "our chat", "conversation history",
            "use the context", "based on stored",
        ]
        wants_vector = wants_memory or any(s in ql for s in vector_signals)

        # If it's a file-driven workflow (doc/dwg/xlsx), don't auto-retrieve unless user
        # explicitly wants memory/context. This prevents irrelevant injection.
        # Still allow internet when Living Atlas / pretrained model / web catalog is asked.
        if file_driven and not wants_vector and not wants_internet:
            return RAGRouteDecision(
                route="llm_only",
                use_vector=False,
                use_internet=False,
                reason="File-driven workflow detected; avoiding automatic RAG injection.",
            )

        # Choose route
        if wants_internet and wants_vector:
            return RAGRouteDecision(
                route="hybrid",
                use_vector=True,
                vector_collections=[COLLECTION_CONVERSATIONS, COLLECTION_DOCUMENTS, COLLECTION_DRAWINGS, COLLECTION_COORDINATES],
                use_internet=True,
                internet_query=query,
                reason="User requested external standards/citations and also referenced prior/stored context.",
            )
        if wants_internet:
            return RAGRouteDecision(
                route="internet",
                use_vector=False,
                use_internet=True,
                internet_query=query,
                reason="Query asks for standards/citations/current information likely requiring web search.",
            )
        if wants_vector:
            return RAGRouteDecision(
                route="vector",
                use_vector=True,
                vector_collections=[COLLECTION_CONVERSATIONS, COLLECTION_DOCUMENTS, COLLECTION_DRAWINGS, COLLECTION_COORDINATES],
                use_internet=False,
                reason="Query references prior/stored context; local retrieval likely helpful.",
            )

        # Default: LLM only (no augmentation)
        return RAGRouteDecision(
            route="llm_only",
            use_vector=False,
            use_internet=False,
            reason="No strong signal for retrieval or web search.",
        )

    def _get_model_tier(self, model_name: Optional[str]) -> str:
        """
        Determine the tier of a model (nano, mini, or complex).
        
        Args:
            model_name: Model name (e.g., "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol")
            
        Returns:
            "nano", "mini", or "complex"
        """
        from survyai.openai_models import infer_tier

        return infer_tier(model_name)
    
    def _active_llm_provider(self, use_fallback: bool = False) -> str:
        """Concrete provider id for the current primary/fallback selection."""
        if use_fallback:
            raw = str(getattr(self.settings, "fallback_llm", "") or "").strip().lower()
        else:
            raw = str(getattr(self.settings, "primary_llm", "") or "").strip().lower()
        if raw in ("", "auto"):
            try:
                from survyai.llm_routing import resolve_primary_llm_selection

                return resolve_primary_llm_selection("auto")
            except Exception:
                return "openai"
        return raw

    def _escalate_model_tier(self, current_tier: str, provider: Optional[str] = None) -> Optional[str]:
        """
        Get the next higher tier model for escalation (same provider).

        Args:
            current_tier: Current tier ("nano", "mini", or "complex")
            provider: Provider id (default: primary LLM)

        Returns:
            Model name for next tier, or None if already at highest tier
        """
        from survyai.openai_models import escalate_tier_model
        from survyai.provider_models import escalate_provider_from_model

        prov = (provider or self._active_llm_provider(False) or "openai").strip().lower()
        if prov == "openai":
            return escalate_tier_model(
                current_tier if current_tier in ("nano", "mini", "complex") else "mini",
                mini=getattr(self.settings, "openai_model_mini", None),
                complex_model=getattr(self.settings, "openai_model_complex", None),
            )
        # For non-OpenAI, map nano/mini/complex → simple/average/complex model ids.
        tier_to_complexity = {"nano": "simple", "mini": "average", "complex": "complex"}
        c = tier_to_complexity.get(current_tier, "average")
        # Build a synthetic "current" model at this tier, then escalate one step.
        current = self._get_model_for_complexity(prov, c)  # type: ignore[arg-type]
        return escalate_provider_from_model(prov, current, settings=self.settings)

    def _next_provider_failover_model(
        self,
        provider: str,
        current_model: Optional[str],
        complexity: str,
        *,
        elevated_average: bool = False,
    ) -> Optional[str]:
        """Next alternate model for the active provider after quota / proxy failure."""
        from survyai.provider_models import next_provider_failover_model

        tried = list(getattr(self, "_openai_models_tried_this_query", []) or [])
        if current_model and current_model not in tried:
            tried.append(current_model)
        return next_provider_failover_model(
            provider,
            current_model,
            complexity=complexity if complexity in ("simple", "average", "complex") else "average",
            tried=tried,
            settings=self.settings,
            elevated_average=elevated_average,
        )

    def _next_openai_failover_model(
        self,
        current_model: Optional[str],
        complexity: str,
    ) -> Optional[str]:
        """Back-compat wrapper: OpenAI-only failover."""
        return self._next_provider_failover_model("openai", current_model, complexity)

    def _switch_model_and_retry(
        self,
        query: str,
        original_query: str,
        current_model: Optional[str],
        current_llm: BaseChatModel,
        complexity: str,
        enhanced_system_prompt: str,
        initial_messages: List[BaseMessage],
        current_session_id: str,
        use_fallback: bool,
        interactive_mode: bool,
        context_retrieved: bool,
        switch_reason: str,
        tools_to_bind: List[BaseTool],
        target_model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Dict:
        """
        Switch to another model on the same paid provider and retry the query.

        If ``target_model`` is set (quota/proxy failover), use it. Otherwise escalate
        capability tier (simple→average→complex / nano→mini→complex).
        """
        prov = (provider or self._active_llm_provider(use_fallback) or "openai").strip().lower()
        if target_model:
            escalated_model = target_model
        else:
            from survyai.provider_models import PAID_PROVIDERS, escalate_provider_from_model

            if prov in PAID_PROVIDERS and prov != "openai":
                escalated_model = escalate_provider_from_model(
                    prov, current_model, settings=self.settings
                )
            else:
                current_tier = self._get_model_tier(current_model)
                escalated_model = self._escalate_model_tier(current_tier, provider=prov)

        if not escalated_model:
            logger.warning("⚠ No alternate %s model available for retry", prov)
            return {
                "query": original_query,
                "response": (
                    f"Query failed with {switch_reason}. "
                    f"No alternate {prov} model remains to try. "
                    "Wait for quota reset, or set Primary LLM to Ollama for offline work."
                ),
                "success": False,
                "error": switch_reason,
                "llm_used": "fallback" if use_fallback else "primary",
                "model_name": current_model,
                "session_id": current_session_id,
                "llm_cost_usd": 0.0,
            }

        tried = list(getattr(self, "_openai_models_tried_this_query", []) or [])
        if current_model and current_model not in tried:
            tried.append(current_model)
        if escalated_model not in tried:
            tried.append(escalated_model)
        self._openai_models_tried_this_query = tried

        logger.info(
            f"🔄 Switching {prov} from {current_model} to {escalated_model} "
            f"(reason: {switch_reason}; tried={tried})"
        )

        self._model_switched_this_query = True

        try:
            new_llm = self._initialize_llm(prov, model_name=escalated_model)
            if prov == "openai":
                self._current_openai_model = escalated_model
            elif prov == "gemini":
                self._current_gemini_model = escalated_model

            self.llm_with_tools = new_llm.bind_tools(tools_to_bind)
            self.graph = self._build_graph()
            self.app = self.graph.compile(checkpointer=self.memory)

            logger.info(f"✓ Switched to {escalated_model} ({prov}) - retrying query")

            thread_id = f"{current_session_id}:retry:{uuid.uuid4().hex}"
            max_iterations = getattr(self.settings, 'agent_max_iterations', 20)
            recursion_limit = getattr(self.settings, "agent_recursion_limit", max(50, (max_iterations * 3)))
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": recursion_limit,
            }

            initial_state = {"messages": initial_messages}
            result = self.app.invoke(initial_state, config=config)

            response_text = self._extract_response(result)
            input_hint, _ = estimate_message_tokens(initial_messages, escalated_model)
            llm_cost_usd = self._estimate_llm_cost_usd_from_graph_result(
                result,
                escalated_model,
                response_text,
                initial_messages_token_hint=input_hint,
            )

            llm_used = "fallback" if use_fallback else "primary"
            self._store_conversation(
                query=query,
                response=response_text,
                session_id=current_session_id,
                llm_used=llm_used
            )

            logger.info(f"✓ Model switch successful - query completed with {escalated_model}")

            return {
                "query": query,
                "response": response_text,
                "llm_used": llm_used,
                "model_name": escalated_model,
                "complexity": complexity,
                "success": True,
                "session_id": current_session_id,
                "context_retrieved": context_retrieved,
                "model_switched": True,
                "original_model": current_model,
                "switch_reason": switch_reason,
                "llm_cost_usd": llm_cost_usd,
            }

        except Exception as e:
            logger.error(f"❌ Model switch failed: {e}")
            return {
                "query": original_query,
                "response": f"Query failed and model switch also failed: {e}",
                "success": False,
                "error": str(e),
                "llm_used": "fallback" if use_fallback else "primary",
                "model_name": current_model,
                "session_id": current_session_id,
                "llm_cost_usd": 0.0,
            }

    def _route_task_with_cheap_model(
        self,
        *,
        provider: str,
        query: str,
        heuristic_complexity: Literal["simple", "average", "complex"],
        kind: str = "",
        file_driven: bool = False,
        needs_internet: bool = False,
    ) -> Optional[Any]:
        """
        Ask the cheapest paid model for ``provider`` to pick the execution model.

        Returns a ``PromptRouteDecision`` on success, or None so the caller can
        keep heuristic complexity routing (no behavior change on failure).
        """
        from dataclasses import replace

        from survyai.prompt_router import (
            ROUTER_CACHE_MAX,
            ROUTER_TIMEOUT_SECONDS,
            apply_route_floors,
            build_router_messages,
            execution_candidates_for_provider,
            message_content_to_text,
            parse_router_response,
            router_models_to_try,
            truncate_router_query,
        )

        p = str(provider or "").strip().lower()
        candidates = execution_candidates_for_provider(p, self.settings)
        if not candidates:
            return None

        knowledge_only = (not file_driven) and str(kind or "").strip().lower() in {
            "general_knowledge",
            "other",
            "",
        }
        cache_key = (
            p,
            truncate_router_query(query).lower(),
            bool(file_driven),
            str(kind or ""),
        )
        cache = getattr(self, "_prompt_router_cache", None)
        if cache is None:
            cache = {}
            self._prompt_router_cache = cache
        hit = cache.get(cache_key)
        if hit is not None:
            logger.info("Prompt router cache hit (%s → %s)", p, getattr(hit, "model", ""))
            return hit

        system, user = build_router_messages(
            provider=p,
            query=query,
            candidates=candidates,
            heuristic_complexity=heuristic_complexity,
            kind=kind,
            file_driven=file_driven,
            needs_internet=needs_internet,
        )
        last_err: Optional[str] = None
        for router_model in router_models_to_try(p, self.settings):
            try:
                llm = self._initialize_llm(p, model_name=router_model)
            except Exception as exc:
                last_err = str(exc)
                logger.info(
                    "Prompt router could not init %s/%s (%s) — trying next cheap model.",
                    p,
                    router_model,
                    exc,
                )
                continue
            try:
                def _invoke_router(llm=llm, sys=system, usr=user):
                    return llm.invoke(
                        [SystemMessage(content=sys), HumanMessage(content=usr)]
                    )

                msg, err, timed_out = self._run_with_timeout(
                    ROUTER_TIMEOUT_SECONDS,
                    _invoke_router,
                    llm_model_name=router_model,
                    serialize_llm=False,
                )
            except Exception as exc:
                last_err = str(exc)
                logger.info("Prompt router invoke failed on %s (%s).", router_model, exc)
                continue
            if timed_out:
                last_err = f"timed out after {ROUTER_TIMEOUT_SECONDS}s"
                logger.info(
                    "Prompt router %s timed out — trying next cheap model or heuristic.",
                    router_model,
                )
                continue
            if err or msg is None:
                last_err = str(err or "empty router response")
                logger.info("Prompt router %s error: %s", router_model, last_err)
                continue
            decision = parse_router_response(
                message_content_to_text(msg),
                candidates,
                heuristic_complexity=heuristic_complexity,
            )
            if decision is None:
                last_err = "unparseable router JSON"
                logger.info("Prompt router %s returned unusable JSON — trying next.", router_model)
                continue
            decision = apply_route_floors(
                decision,
                candidates,
                file_driven=file_driven,
                knowledge_only=knowledge_only,
            )
            resolved = replace(decision, router_model=router_model, source="llm")
            try:
                if len(cache) >= ROUTER_CACHE_MAX:
                    cache.pop(next(iter(cache)))
                cache[cache_key] = resolved
            except Exception:
                pass
            return resolved

        if last_err:
            logger.info(
                "Prompt router unavailable for %s (%s) — using heuristic complexity routing.",
                p,
                last_err,
            )
        return None

    def _get_model_for_complexity(
        self,
        provider: str,
        complexity: Literal["simple", "average", "complex"],
        *,
        elevated_average: bool = False,
    ) -> str:
        """Select the exact model id for any paid provider + complexity bucket."""
        from survyai.provider_models import resolve_provider_model_for_complexity

        selected = resolve_provider_model_for_complexity(
            provider,
            complexity,
            settings=self.settings,
            enable_tiered=bool(getattr(self.settings, "enable_tiered_models", True)),
            elevated_average=elevated_average,
        )
        logger.info(
            "Selected %s model '%s' for complexity '%s'%s",
            provider,
            selected,
            complexity,
            " (elevated average)" if elevated_average and complexity == "average" else "",
        )
        return selected

    def _get_openai_model_for_complexity(
        self,
        complexity: Literal["simple", "average", "complex"],
        *,
        elevated_average: bool = False,
    ) -> str:
        """Back-compat: OpenAI complexity → model id."""
        return self._get_model_for_complexity(
            "openai", complexity, elevated_average=elevated_average
        )

    def _try_openai_tier_llm(
        self,
        complexity: Literal["simple", "average", "complex"],
    ) -> Tuple[Optional[BaseChatModel], Optional[str]]:
        """Initialize a tiered model for the primary paid provider when enabled."""
        model_name: Optional[str] = None
        prov = str(getattr(self.settings, "primary_llm", "") or "").strip().lower()
        try:
            from survyai.provider_models import PAID_PROVIDERS

            if prov in PAID_PROVIDERS and getattr(self.settings, "enable_tiered_models", True):
                model_name = self._get_model_for_complexity(prov, complexity)
                return self._initialize_llm(prov, model_name=model_name), model_name
        except Exception:
            pass
        primary = getattr(self, "llm_primary", None)
        if primary is not None:
            model_name = (
                getattr(self, "_current_openai_model", None)
                or getattr(self, "_current_gemini_model", None)
                or self._resolve_provider_model_name(prov or "openai")
            )
            return primary, model_name
        return None, None
    
    # ==========================================================================
    # LLM INITIALIZATION
    # ==========================================================================

    def _resolve_provider_model_name(self, provider: Optional[str]) -> str:
        """Map provider id (e.g. 'ollama') to the concrete model id for TPM/cost/logging."""
        p = str(provider or "").strip().lower()
        if p == "ollama":
            return str(getattr(self.settings, "ollama_model", None) or "llama3.2:1b").strip() or "llama3.2:1b"
        if p == "deepseek":
            return str(getattr(self.settings, "deepseek_model", None) or "deepseek-chat").strip() or "deepseek-chat"
        if p == "claude":
            return str(
                getattr(self.settings, "claude_model", None) or "claude-3-5-sonnet-20241022"
            ).strip()
        if p == "gemini":
            return (
                self._current_gemini_model
                or str(getattr(self.settings, "gemini_model", None) or "gemini-2.0-flash")
            )
        if p == "openai":
            return (
                self._current_openai_model
                or str(getattr(self.settings, "openai_model", None) or "gpt-4o-mini")
            )
        return str(provider or "unknown")

    def _llm_is_ollama(self, llm: Optional[Any] = None) -> bool:
        """True when the chat model talks to a local Ollama OpenAI-compatible endpoint."""
        target = llm
        if target is None:
            target = getattr(self, "llm_with_tools", None) or getattr(self, "llm_primary", None)
        try:
            base = str(
                getattr(target, "openai_api_base", None)
                or getattr(target, "base_url", None)
                or ""
            )
            if "11434" in base or "ollama" in base.lower():
                return True
        except Exception:
            pass
        return False

    def _release_ollama_model(self) -> None:
        """Best-effort unload so a timed-out local run cannot keep thrashing host RAM."""
        try:
            from survyai.ollama_support import release_ollama_model

            base = str(getattr(self.settings, "ollama_base_url", None) or "http://localhost:11434")
            model = str(getattr(self.settings, "ollama_model", None) or "").strip()
            if model:
                release_ollama_model(base, model)
        except Exception:
            pass

    def apply_runtime_auth(self, settings: Any) -> None:
        """
        Hot-swap rotating cloud auth onto this agent without rebuilding tools/graph.

        Updates ``settings``, proxy enablement, and in-place ``access_token`` on
        cached SurvyAIProxyChatModel clients (re-keys the LLM cache).
        """
        if settings is None:
            return
        self.settings = settings
        self._cloud_proxy_enabled = bool(
            getattr(self.settings, "survyai_llm_proxy_enabled", False)
            and str(getattr(self.settings, "survyai_api_base_url", "") or "").strip()
            and str(getattr(self.settings, "survyai_access_token", "") or "").strip()
        )
        new_tok = str(getattr(self.settings, "survyai_access_token", "") or "").strip()
        new_base = str(getattr(self.settings, "survyai_api_base_url", "") or "").strip()
        new_dev = str(getattr(self.settings, "survyai_device_id", "") or "").strip()
        new_path = (
            str(getattr(self.settings, "survyai_llm_proxy_path", "/v1/llm/chat") or "/v1/llm/chat").strip()
            or "/v1/llm/chat"
        )
        cache = getattr(self, "_llm_client_cache", None) or {}
        rebuilt: Dict[tuple, Any] = {}
        for key, llm in list(cache.items()):
            transport = ""
            provider = ""
            if isinstance(key, tuple) and len(key) >= 2:
                provider = str(key[0] or "")
                transport = str(key[1] or "")
            if transport == "proxy" and llm is not None:
                try:
                    if hasattr(llm, "access_token"):
                        llm.access_token = new_tok
                    if hasattr(llm, "base_url"):
                        llm.base_url = new_base
                    if hasattr(llm, "device_id"):
                        llm.device_id = new_dev
                    if hasattr(llm, "proxy_path"):
                        llm.proxy_path = new_path
                except Exception:
                    # Drop broken proxy client; next call will recreate.
                    continue
                try:
                    new_key = (
                        key[0],
                        key[1],
                        key[2],
                        key[3],
                        key[4],
                        self._llm_auth_fingerprint(provider, proxy=True),
                    )
                    rebuilt[new_key] = llm
                except Exception:
                    rebuilt[key] = llm
            else:
                rebuilt[key] = llm
        self._llm_client_cache = rebuilt
        self._openai_llm_cache = rebuilt
        logger.debug("Hot-swapped runtime cloud auth onto warm agent (proxy cache re-keyed)")

    def _llm_auth_fingerprint(self, llm_type: str, *, proxy: bool) -> str:
        """Short stable fingerprint so token/key rotation invalidates client cache."""
        if proxy:
            tok = str(getattr(self.settings, "survyai_access_token", "") or "").strip()
            return f"tok:{tok[-12:]}" if tok else "tok:"
        if llm_type == "openai":
            key = str(getattr(self.settings, "openai_api_key", "") or "").strip()
        elif llm_type == "claude":
            key = str(getattr(self.settings, "anthropic_api_key", "") or "").strip()
        elif llm_type == "gemini":
            key = str(getattr(self.settings, "google_api_key", "") or "").strip()
        elif llm_type == "deepseek":
            key = str(getattr(self.settings, "deepseek_api_key", "") or "").strip()
        else:
            key = str(getattr(self.settings, "ollama_base_url", "") or "").strip()
        return f"key:{key[-12:]}" if key else "key:"

    def _llm_client_cache_key(
        self,
        llm_type: str,
        *,
        transport: str,
        model_name: str,
        max_tokens: int,
    ) -> tuple:
        return (
            str(llm_type or "").strip().lower(),
            str(transport or "direct").strip().lower(),
            str(model_name or "").strip(),
            float(getattr(self.settings, "agent_temperature", 0.3) or 0.0),
            int(max_tokens or 0),
            self._llm_auth_fingerprint(
                llm_type, proxy=(str(transport).lower() == "proxy")
            ),
        )

    def _get_cached_llm(self, cache_key: tuple) -> Optional[BaseChatModel]:
        cache = getattr(self, "_llm_client_cache", None)
        if cache is None:
            cache = getattr(self, "_openai_llm_cache", None) or {}
            self._llm_client_cache = cache
        return cache.get(cache_key)

    def _store_cached_llm(self, cache_key: tuple, llm: BaseChatModel) -> BaseChatModel:
        try:
            cache = getattr(self, "_llm_client_cache", None)
            if cache is None:
                cache = {}
                self._llm_client_cache = cache
                self._openai_llm_cache = cache
            cache[cache_key] = llm
        except Exception:
            pass
        return llm

    def _resolve_proxy_model_name(
        self, llm_type: str, model_name: Optional[str] = None
    ) -> str:
        requested = str(model_name or "").strip()
        if requested:
            return requested
        if llm_type == "openai":
            return str(getattr(self.settings, "openai_model", "gpt-5.6-terra") or "gpt-5.6-terra")
        if llm_type == "claude":
            return str(
                getattr(self.settings, "claude_model", "claude-3-5-sonnet-20241022")
                or "claude-3-5-sonnet-20241022"
            )
        if llm_type == "gemini":
            return str(getattr(self.settings, "gemini_model", "gemini-2.0-flash") or "gemini-2.0-flash")
        return str(getattr(self.settings, "deepseek_model", None) or "deepseek-chat")

    def _make_cloud_proxy_llm(
        self, llm_type: str, model_name: Optional[str] = None
    ) -> BaseChatModel:
        resolved_model = self._resolve_proxy_model_name(llm_type, model_name)
        max_tokens = int(getattr(self.settings, "agent_max_tokens", 0) or 0)
        cache_key = self._llm_client_cache_key(
            llm_type,
            transport="proxy",
            model_name=resolved_model,
            max_tokens=max_tokens,
        )
        cached = self._get_cached_llm(cache_key)
        if cached is not None:
            logger.debug("Using cached proxy LLM (%s / %s)", llm_type, resolved_model)
            return cached
        logger.debug("Initializing %s via SurvyAI cloud LLM proxy (%s)", llm_type, resolved_model)
        llm = SurvyAIProxyChatModel(
            base_url=str(getattr(self.settings, "survyai_api_base_url", "") or "").strip(),
            access_token=str(getattr(self.settings, "survyai_access_token", "") or "").strip(),
            device_id=str(getattr(self.settings, "survyai_device_id", "") or "").strip(),
            provider=llm_type,
            model_name=str(resolved_model).strip(),
            temperature=float(self.settings.agent_temperature),
            max_tokens=max_tokens,
            proxy_path=str(getattr(self.settings, "survyai_llm_proxy_path", "/v1/llm/chat") or "/v1/llm/chat").strip() or "/v1/llm/chat",
        )
        return self._store_cached_llm(cache_key, llm)
    
    def _initialize_llm(self, llm_type: str, model_name: Optional[str] = None) -> BaseChatModel:
        """
        Initialize a Language Model based on the specified type.
        
        Supported LLM types:
        - "gemini": Google's Gemini models (gemini-2.0-flash, gemini-1.5-flash, gemini-pro-latest)
        - "deepseek": DeepSeek's models via OpenAI-compatible API
        - "claude": Anthropic's Claude models (Opus, Sonnet, Haiku)
        - "openai": OpenAI's models (GPT-4, GPT-4o, GPT-4o-Turbo, GPT-5, GPT-5-nano, GPT-5-mini, GPT-5.1)
        
        Args:
            llm_type: One of "gemini", "deepseek", "claude", or "openai"
            model_name: Optional specific model name (for OpenAI tiered models). If None, uses default from settings.
            
        Returns:
            BaseChatModel: An initialized LLM ready for use
            
        Raises:
            ValueError: If llm_type is not recognized or API key is missing
            Exception: If API connection fails
        """
        try:
            if self._cloud_proxy_enabled and llm_type in {"openai", "claude", "gemini", "deepseek"}:
                return self._make_cloud_proxy_llm(llm_type, model_name=model_name)
            if llm_type == "deepseek":
                # DeepSeek uses an OpenAI-compatible API
                # We use ChatOpenAI with a custom base_url
                if not self.settings.deepseek_api_key:
                    raise ValueError("DEEPSEEK_API_KEY is required but not set")

                resolved = (
                    str(model_name or "").strip()
                    or str(getattr(self.settings, "deepseek_model", None) or "").strip()
                    or "deepseek-chat"
                )
                max_tokens = int(getattr(self.settings, "agent_max_tokens", 0) or 0)
                cache_key = self._llm_client_cache_key(
                    "deepseek",
                    transport="direct",
                    model_name=resolved,
                    max_tokens=max_tokens,
                )
                cached = self._get_cached_llm(cache_key)
                if cached is not None:
                    logger.debug("Using cached DeepSeek LLM (%s)", resolved)
                    return cached
                logger.debug("Initializing DeepSeek LLM (%s)", resolved)
                llm = ChatOpenAI(
                    model=resolved,
                    api_key=self.settings.deepseek_api_key,
                    base_url=self.settings.deepseek_base_url,
                    temperature=self.settings.agent_temperature,
                    max_tokens=max_tokens,
                )
                return self._store_cached_llm(cache_key, llm)
                
            elif llm_type == "gemini":
                # Google Gemini - honor caller model_name (tiered routing); fall back to settings.
                if not self.settings.google_api_key:
                    raise ValueError("GOOGLE_API_KEY is required but not set")

                resolved = (
                    str(model_name or "").strip()
                    or str(getattr(self.settings, "gemini_model", "gemini-2.0-flash") or "gemini-2.0-flash")
                )

                # Query available models from Google's API (cached for process lifetime)
                available = self._list_gemini_models()

                # If configured model isn't available, pick a fallback
                if available and resolved not in available:
                    # Preference order for fallback models (flash models have better free tier limits)
                    preferred = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-flash-exp", "gemini-pro-latest"]
                    selected = next((m for m in preferred if m in available), available[0])
                    logger.warning(f"Model '{resolved}' not available; using '{selected}'")
                    resolved = selected

                max_tokens = int(getattr(self.settings, "agent_max_tokens", 0) or 0)
                cache_key = self._llm_client_cache_key(
                    "gemini",
                    transport="direct",
                    model_name=resolved,
                    max_tokens=max_tokens,
                )
                cached = self._get_cached_llm(cache_key)
                if cached is not None:
                    logger.debug("Using cached Gemini LLM (%s)", resolved)
                    self._current_gemini_model = resolved
                    return cached

                logger.debug("Initializing Gemini LLM with model: %s", resolved)

                llm = ChatGoogleGenerativeAI(
                    model=resolved,
                    google_api_key=self.settings.google_api_key,
                    temperature=self.settings.agent_temperature,
                    max_output_tokens=max_tokens,
                )
                self._current_gemini_model = resolved
                return self._store_cached_llm(cache_key, llm)
                
            elif llm_type == "claude":
                # Anthropic Claude models (Opus, Sonnet, Haiku)
                if not self.settings.anthropic_api_key or not self.settings.anthropic_api_key.strip():
                    raise ValueError(
                        "ANTHROPIC_API_KEY is required but not set. "
                        "Please set ANTHROPIC_API_KEY in your .env file or environment variables."
                    )

                resolved = (
                    str(model_name or "").strip()
                    or str(
                        getattr(self.settings, "claude_model", "claude-3-5-sonnet-20241022")
                        or "claude-3-5-sonnet-20241022"
                    )
                )
                
                # Model-specific max output-token limits for Claude
                # https://docs.anthropic.com/en/docs/about-claude/models
                claude_max_tokens_limits = {
                    # Claude 3.5 family
                    "claude-3-5-sonnet-20241022": 8192,
                    "claude-3-5-sonnet-20240620": 8192,
                    "claude-3-5-haiku-20241022": 8192,
                    # Claude 3 family
                    "claude-3-opus-20240229": 4096,
                    "claude-3-sonnet-20240229": 4096,
                    "claude-3-haiku-20240307": 4096,
                    # Claude 3.7 / future (conservative fallback)
                    "claude-3-7-sonnet-20250219": 16000,
                }
                
                # Cap max_tokens to model's actual API limit — log at DEBUG,
                # because clamping is the expected behaviour when agent_max_tokens is a
                # generous ceiling rather than an exact target.
                model_max = claude_max_tokens_limits.get(resolved, 4096)
                requested_tokens = self.settings.agent_max_tokens
                actual_max_tokens = min(requested_tokens, model_max)

                cache_key = self._llm_client_cache_key(
                    "claude",
                    transport="direct",
                    model_name=resolved,
                    max_tokens=actual_max_tokens,
                )
                cached = self._get_cached_llm(cache_key)
                if cached is not None:
                    logger.debug("Using cached Claude LLM (%s)", resolved)
                    return cached
                
                if requested_tokens > model_max:
                    logger.debug(
                        f"Clamping max_tokens from {requested_tokens} → {actual_max_tokens} "
                        f"(model '{resolved}' limit)."
                    )
                
                logger.debug("Initializing Claude LLM with model: %s", resolved)
                
                llm = ChatAnthropic(
                    model=resolved,
                    anthropic_api_key=self.settings.anthropic_api_key,
                    temperature=self.settings.agent_temperature,
                    max_tokens=actual_max_tokens
                )
                return self._store_cached_llm(cache_key, llm)
                
            elif llm_type == "openai":
                # OpenAI models (GPT-4, GPT-4o, GPT-4o-mini, GPT-5, GPT-5-mini, GPT-5-nano, GPT-5.1)
                if not self.settings.openai_api_key or not self.settings.openai_api_key.strip():
                    raise ValueError(
                        "OPENAI_API_KEY is required but not set. "
                        "Please set OPENAI_API_KEY in your .env file or environment variables."
                    )
                
                # Use provided model_name or fallback to settings
                if not str(model_name or "").strip():
                    model_name = getattr(self.settings, "openai_model", "gpt-5.6-terra")
                model_name = str(model_name).strip()
                
                # Model-specific max output-token limits for OpenAI.
                # Conservative values aligned with known API limits.
                # All unknown / future models fall back to 16384 (safe ceiling for
                # current GPT-4o / GPT-5 class models).
                openai_max_tokens_limits = {
                    # GPT-4 series
                    "gpt-4": 8192,
                    "gpt-4-turbo": 4096,
                    "gpt-4o": 16384,
                    "gpt-4o-2024-08-06": 16384,
                    "gpt-4o-mini": 16384,
                    "gpt-4o-mini-2024-07-18": 16384,
                    "gpt-4.1": 32768,
                    "gpt-4.1-mini": 32768,
                    # GPT-5 series
                    "gpt-5-nano": 16384,
                    "gpt-5-mini": 16384,
                    "gpt-5": 16384,
                    "gpt-5.1": 16384,
                    "gpt-5.2": 16384,
                    # GPT-5.4 family
                    "gpt-5.4-nano": 16384,
                    "gpt-5.4-mini": 16384,
                    "gpt-5.4": 16384,
                    "gpt-5.4-pro": 16384,
                    # GPT-5.5 family
                    "gpt-5.5": 32768,
                    "gpt-5.5-pro": 32768,
                    "gpt-5.5-nano": 16384,
                    "gpt-5.5-mini": 16384,
                    # GPT-5.6 family (OpenAI Models docs)
                    "gpt-5.6": 32768,
                    "gpt-5.6-sol": 32768,
                    "gpt-5.6-terra": 32768,
                    "gpt-5.6-luna": 32768,
                }

                # Cap max_tokens to model's actual API limit — DEBUG, not WARNING,
                # because clamping is the expected, harmless behaviour.
                model_max = openai_max_tokens_limits.get(model_name, 16384)
                requested_tokens = self.settings.agent_max_tokens
                actual_max_tokens = min(requested_tokens, model_max)

                cache_key = self._llm_client_cache_key(
                    "openai",
                    transport="direct",
                    model_name=model_name,
                    max_tokens=actual_max_tokens,
                )
                cached = self._get_cached_llm(cache_key)
                if cached is not None:
                    logger.debug("Using cached OpenAI LLM (%s)", model_name)
                    return cached
                
                if requested_tokens > model_max:
                    logger.debug(
                        f"Clamping max_tokens from {requested_tokens} → {actual_max_tokens} "
                        f"(model '{model_name}' limit)."
                    )
                
                logger.debug("Initializing OpenAI LLM with model: %s", model_name)
                
                llm = ChatOpenAI(
                    model=model_name,
                    api_key=self.settings.openai_api_key,
                    temperature=self.settings.agent_temperature,
                    max_tokens=actual_max_tokens
                )
                return self._store_cached_llm(cache_key, llm)

            elif llm_type == "ollama":
                base = (
                    getattr(self.settings, "ollama_base_url", None) or "http://localhost:11434"
                ).strip().rstrip("/")
                model_name = (
                    str(model_name or "").strip()
                    or getattr(self.settings, "ollama_model", None)
                    or "llama3.2:1b"
                )
                ok, ram_err, num_ctx = _ollama_ram_policy(model_name)
                if not ok:
                    raise RuntimeError(ram_err)
                num_predict = int(getattr(self.settings, "ollama_num_predict", 512) or 512)
                # On low-RAM hosts, shrink generation budget further.
                if num_ctx <= 1024:
                    num_predict = min(num_predict, 256)
                elif num_ctx <= 2048:
                    num_predict = min(num_predict, 384)
                requested_tokens = int(self.settings.agent_max_tokens or num_predict)
                actual_max_tokens = min(requested_tokens, num_predict)
                # Hard HTTP timeout is critical: without it, SurvyAI's join-timeout
                # leaves Ollama generating forever in a daemon thread (RAM/CPU thrash).
                invoke_budget = int(getattr(self.settings, "llm_invoke_timeout_seconds", 180) or 180)
                http_timeout = max(30, min(invoke_budget, 90))
                cache_key = self._llm_client_cache_key(
                    "ollama",
                    transport="direct",
                    model_name=f"{model_name}@{base}:{num_ctx}",
                    max_tokens=actual_max_tokens,
                )
                cached = self._get_cached_llm(cache_key)
                if cached is not None:
                    logger.debug("Using cached Ollama LLM (%s)", model_name)
                    self._ollama_num_ctx = int(num_ctx)
                    return cached
                logger.debug(
                    "Initializing Ollama LLM with model: %s at %s/v1 "
                    "(max_tokens=%s, num_ctx=%s, http_timeout=%ss)",
                    model_name,
                    base,
                    actual_max_tokens,
                    num_ctx,
                    http_timeout,
                )
                kwargs: Dict[str, Any] = {
                    "model": model_name,
                    "api_key": "ollama",
                    "base_url": f"{base}/v1",
                    "temperature": self.settings.agent_temperature,
                    "max_tokens": actual_max_tokens,
                    "timeout": http_timeout,
                    "max_retries": 1,
                    # Cap KV-cache / context and unload immediately after each call
                    # so Ollama cannot keep a large resident footprint between prompts.
                    "extra_body": {
                        "keep_alive": 0,
                        "options": {
                            "num_ctx": int(num_ctx),
                            "num_predict": int(actual_max_tokens),
                        },
                    },
                }
                self._ollama_num_ctx = int(num_ctx)
                return self._store_cached_llm(cache_key, ChatOpenAI(**kwargs))
                
            else:
                raise ValueError(
                    f"Unknown LLM type: {llm_type}. "
                    "Supported types: gemini, deepseek, claude, openai, ollama"
                )
                
        except Exception as e:
            logger.error(f"Error initializing {llm_type} LLM: {e}")
            raise
    
    def _list_gemini_models(self) -> List[str]:
        """
        Query Google's API to get a list of available Gemini models.
        
        This is useful because:
        1. Model availability varies by region and API version
        2. We can automatically fall back to available models
        3. Prevents errors from trying to use unavailable models
        
        Returns:
            List[str]: Names of available chat-capable Gemini models
            
        Note:
            Returns empty list if the API call fails (network error, etc.)
            Successful results are cached for the agent process lifetime.
        """
        cached = getattr(self, "_gemini_models_cache", None)
        if cached is not None:
            return list(cached)
        try:
            import requests
            
            api_key = getattr(self.settings, "google_api_key", None)
            if not api_key:
                self._gemini_models_cache = []
                return []
            
            # Google's model listing endpoint
            url = "https://generativelanguage.googleapis.com/v1beta/models"
            response = requests.get(url, params={"key": api_key}, timeout=10)
            response.raise_for_status()
            
            models = []
            for model in response.json().get("models", []):
                name = model.get("name", "").split("/")[-1]
                
                # Filter to chat-capable models only
                # Exclude embedding, image, TTS, and vision-only models
                if name.startswith("gemini") and not any(
                    x in name.lower() for x in ["embedding", "image", "tts", "vision"]
                ):
                    # Check if model supports text generation
                    methods = model.get("supportedGenerationMethods", [])
                    if not methods or "generateContent" in methods:
                        models.append(name)

            self._gemini_models_cache = models
            return list(models)
            
        except Exception as e:
            logger.debug(f"Could not list Gemini models: {e}")
            # Do not cache hard failures forever — allow a later retry.
            return []
    
    # ==========================================================================
    # TOOL CREATION
    # ==========================================================================
    
    def _filter_tools_by_feature_flags(self, tools: List[BaseTool]) -> List[BaseTool]:
        """
        Phase 2: omit tools not allowed for the current license (builder vs pro + flags).

        Core tools (Excel, documents, coordinates, filesystem, etc.) are never removed.
        """
        ff = self.feature_flags
        removed: List[str] = []
        kept: List[BaseTool] = []
        for t in tools:
            name = getattr(t, "name", "") or ""
            if ff.is_tool_allowed(name):
                kept.append(t)
            else:
                removed.append(name)
        if removed:
            logger.info(
                "License/feature filter (%s): omitted %d tool(s): %s",
                ff.license_mode,
                len(removed),
                ", ".join(removed[:40]) + (" ..." if len(removed) > 40 else ""),
            )
        return kept
    
    def _create_tools(self) -> List[BaseTool]:
        """
        Create the tools that the agent can use.
        
        Tools are the "hands" of the agent - they allow it to interact with
        external systems like AutoCAD, Excel files, etc.
        
        Each tool has:
        - name: Unique identifier the LLM uses to call it
        - description: Tells the LLM what the tool does and when to use it
        - func: The Python function that executes the tool
        - args_schema: Pydantic model defining the expected arguments
        
        Returns:
            List[BaseTool]: List of configured tools
        """
        # ==================================================================
        # AUTOCAD TOOLS
        # ==================================================================
        # These tools interface with AutoCAD via COM API
        
        # --- Tool 1: Open Drawing ---
        class AutoCADOpenInput(BaseModel):
            """Input schema for opening AutoCAD drawings."""
            file_path: str = Field(
                description="Path to DWG or DXF file to open in AutoCAD"
            )
        
        def autocad_open(file_path: str) -> str:
            """
            Open a CAD drawing file in AutoCAD.
            
            This must be called before any other AutoCAD operations.
            Establishes connection to AutoCAD if not already connected.
            Falls back to ezdxf if AutoCAD is not available.
            STRICT: Survey plan templates are always opened read-only and never written.
            """
            from pathlib import Path
            # Try AutoCAD first
            if not self.autocad.is_connected:
                connected = self.autocad.connect()
                if not connected:
                    # Fall back to ezdxf
                    if self.dxf_fallback.is_available:
                        logger.warning("AutoCAD not available, using ezdxf fallback")
                        result = self.dxf_fallback.open_drawing(file_path)
                        if result.get("success"):
                            return str(result)
                        return f"Fallback failed: {result.get('error', 'Unknown error')}"
                    return "AutoCAD not available and ezdxf fallback not installed. Please open AutoCAD manually first."
            # STRICT: Never open survey plan template for writing (read-only to avoid corruption).
            resolved = str(Path(file_path).resolve()) if file_path else ""
            if self._is_protected_template_path(resolved):
                return str(self.autocad.open_drawing(file_path, read_only=True))
            return str(self.autocad.open_drawing(file_path))

        # ==================================================================
        # INTERNET SEARCH TOOL (Permissioned)
        # ==================================================================
        class InternetSearchInput(BaseModel):
            """Input schema for internet search (requires user permission)."""
            query: str = Field(description="What to search for on the internet")
            max_results: int = Field(5, description="Max results to return (1-10)")

        def internet_search(query: str, max_results: int = 5) -> str:
            """
            Search the internet for up-to-date information (permission required).

            Returns structured JSON with results (title, url, snippet).
            
            NOTE: If internet results were already injected into the system prompt by the router,
            this tool will return a note indicating that search was already performed.
            """
            # Check if internet was already searched by router (shouldn't happen, but safety check)
            if getattr(self, "_internet_already_searched_this_query", False):
                return json.dumps(
                    {
                        "success": True,
                        "source": "internet",
                        "note": "Internet search was already performed by the router. Results are in the system prompt context above.",
                        "message": "Please use the internet search results already provided in the conversation context.",
                    },
                    indent=2,
                )
            
            # Clamp
            try:
                max_results_i = int(max_results)
            except Exception:
                max_results_i = 5
            max_results_i = max(1, min(10, max_results_i))

            # CRITICAL: Permission should be checked BEFORE tool is called (by router).
            # If tool is called without permission, return a proper ToolMessage-compatible response
            # (not a permission request string, which breaks the tool call sequence).
            if not getattr(self, "_internet_permission_granted", False):
                # Return a proper error response (not a permission request string)
                return json.dumps(
                    {
                        "success": False,
                        "source": "internet",
                        "error": "Permission not granted. Internet search requires explicit user permission.",
                        "note": "Use the internet_search tool only after user has granted permission via interactive mode.",
                    },
                    indent=2,
                )

            result = _internet_search(query)
            if not result.get("success"):
                return json.dumps(
                    {
                        "success": False,
                        "source": "internet",
                        "provider": result.get("providers_attempted", ["duckduckgo_instant_answer", "wikipedia"]),
                        "query": query,
                        "error": result.get("error", "Unknown error"),
                        "note": "INTERNET_SOURCED",
                    },
                    indent=2,
                )

            # Trim results
            results = (result.get("results", []) or [])[:max_results_i]
            payload = {
                "success": True,
                "source": "internet",
                "providers": result.get("providers", ["duckduckgo_instant_answer", "wikipedia"]),
                "query": result.get("query", query),
                "results": results,
                "note": "INTERNET_SOURCED",
            }
            return json.dumps(payload, indent=2)
        
        # --- Tool 2: Calculate Area ---
        class AutoCADAreaInput(BaseModel):
            """Input schema for area calculation."""
            layer: Optional[str] = Field(
                None, 
                description="Filter by layer name (optional)"
            )
            color: Optional[str] = Field(
                None, 
                description="Filter by color name (e.g., 'red', 'blue')"
            )
        
        def autocad_calculate_area(
            layer: Optional[str] = None, 
            color: Optional[str] = None
        ) -> str:
            """
            Calculate the total area of closed shapes in the drawing.
            
            Uses AutoCAD's native area calculation for maximum precision.
            Can filter by layer or color to calculate specific regions.
            Falls back to ezdxf if AutoCAD is not available.
            """
            # Try AutoCAD first - let the method handle connection checking
            result = self.autocad.calculate_area(layer=layer, color=color)
            if result.get("success") or "error" not in str(result) or "not connected" not in str(result).lower():
                return str(result)
            
            # Fallback to ezdxf
            if self.dxf_fallback.is_available and self.dxf_fallback.doc:
                logger.info("Using ezdxf fallback for area calculation")
                return str(self.dxf_fallback.calculate_area(layer=layer, color=color))
            
            return str(result)  # Return the error from AutoCAD method
        
        # --- Tool 3: Search Text ---
        class AutoCADTextSearchInput(BaseModel):
            """Input schema for text search."""
            pattern: str = Field(
                description="Text pattern to search for (supports regex)"
            )
            case_sensitive: bool = Field(
                False, 
                description="Whether search should be case-sensitive"
            )
        
        def autocad_search_text(
            pattern: str, 
            case_sensitive: bool = False
        ) -> str:
            """
            Search for text entities matching a pattern.
            
            Useful for finding owner names, survey titles, and annotations.
            Supports regular expressions for flexible matching.
            """
            # Try AutoCAD first - let the method handle connection checking
            result = self.autocad.search_text(pattern, case_sensitive)
            if isinstance(result, dict) and result.get("success"):
                return str(result)
            
            # Fallback: get all text and filter
            if self.dxf_fallback.is_available and self.dxf_fallback.doc:
                import re
                fallback_result = self.dxf_fallback.get_all_text()
                if fallback_result.get("success"):
                    flags = 0 if case_sensitive else re.IGNORECASE
                    matches = []
                    for text in fallback_result.get("texts", []):
                        content = text.get("content", "")
                        try:
                            if re.search(pattern, content, flags):
                                matches.append(text)
                        except re.error:
                            if pattern.lower() in content.lower():
                                matches.append(text)
                    return str({"success": True, "matches_found": len(matches), "matches": matches})
            
            return str(result)  # Return the error from AutoCAD method
        
        # --- Tool 4: Get All Text ---
        class AutoCADGetTextInput(BaseModel):
            """Input schema for getting all text (no parameters needed)."""
            pass
        
        def autocad_get_all_text() -> str:
            """
            Extract all text content from the current drawing.
            
            Returns all TEXT and MTEXT entities with their content,
            layer, color, and position information.
            """
            # Try AutoCAD first - let the method handle connection checking
            result = self.autocad.get_all_text()
            if result.get("success") or (isinstance(result, dict) and "error" not in result):
                return str(result)
            
            # Fallback to ezdxf
            if self.dxf_fallback.is_available and self.dxf_fallback.doc:
                return str(self.dxf_fallback.get_all_text())
            
            return str(result)  # Return the error from AutoCAD method
        
        # --- Tool 5: Get All Entities (AI-Driven) ---
        class AutoCADGetAllEntitiesInput(BaseModel):
            """Input schema for getting all entities (no parameters needed)."""
            pass
        
        def autocad_get_all_entities() -> str:
            """
            Get ALL entities from the drawing with complete properties.
            
            [AI-DRIVEN TOOL] This returns raw data for agent reasoning.
            Returns all entities with their full properties (type, layer, color,
            coordinates, area, closed status, etc.) without any filtering.
            
            The agent should reason about which entities match criteria:
            - Identify red entities by checking color property (color_code=1 or color='red')
            - Identify closed shapes by checking 'closed' property and entity type
            - Filter by layer, coordinates, or other properties using reasoning
            
            Use this for AI-driven extraction where the agent reasons about
            what to extract rather than using hardcoded filters.
            """
            # Let the method handle connection/document checking internally
            return str(self.autocad.get_all_entities())
        
        # --- Tool 6: Get Entities Summary (AI-Driven) ---
        class AutoCADEntitiesSummaryInput(BaseModel):
            """Input schema for getting entities summary (no parameters needed)."""
            pass
        
        def autocad_get_entities_summary() -> str:
            """
            Get a summary of all entities for quick analysis.
            
            [AI-DRIVEN TOOL] Returns lightweight summary (counts by type, color, layer)
            that the agent can use to reason about the drawing structure before
            calling get_all_entities() for detailed extraction.
            
            Use this first to understand what's in the drawing, then reason
            about which entities to extract in detail.
            """
            # Let the method handle connection/document checking internally
            return str(self.autocad.get_entities_summary())
        
        # --- Tool 7: Get Entity by Handle (AI-Driven) ---
        class AutoCADGetEntityInput(BaseModel):
            """Input schema for getting entity by handle."""
            handle: str = Field(
                description="Entity handle (unique identifier from get_all_entities)"
            )
        
        def autocad_get_entity(handle: str) -> str:
            """
            Get detailed information about a specific entity by its handle.
            
            [AI-DRIVEN TOOL] Use this to get detailed properties of an entity
            that was identified from get_all_entities(). Handles are unique
            identifiers for entities in AutoCAD.
            """
            # Let the method handle connection/document checking internally
            return str(self.autocad.get_entity_by_handle(handle))
        
        # --- Tool 8: Calculate Entity Area (AI-Driven) ---
        class AutoCADEntityAreaInput(BaseModel):
            """Input schema for calculating area of a specific entity."""
            handle: str = Field(
                description="Entity handle (unique identifier from get_all_entities)"
            )
        
        def autocad_calculate_entity_area(handle: str) -> str:
            """
            Calculate the area of a specific entity by handle.
            
            [AI-DRIVEN TOOL] Use this to calculate area of entities identified
            from get_all_entities(). The agent should first identify which
            entities are closed shapes (check 'closed' property and entity type),
            then call this method for each one.
            
            Workflow:
            1. Call get_all_entities() to get all entities
            2. Reason about which entities are closed shapes (LWPOLYLINE with closed=True, CIRCLE, HATCH)
            3. For each matching entity, call this method with its handle
            4. Sum the areas if needed
            """
            # Let the method handle connection/document checking internally
            return str(self.autocad.calculate_entity_area(handle))
        
        # --- Tool 9: Get Entities (Backward Compatibility) ---
        class AutoCADEntitiesInput(BaseModel):
            """Input schema for entity retrieval."""
            entity_type: Optional[str] = Field(
                None, 
                description="Entity type: LINE, POLYLINE, CIRCLE, TEXT, etc."
            )
            layer: Optional[str] = Field(
                None, 
                description="Filter by layer name"
            )
            color: Optional[str] = Field(
                None, 
                description="Filter by color name"
            )
        
        def autocad_get_entities(
            entity_type: Optional[str] = None,
            layer: Optional[str] = None,
            color: Optional[str] = None
        ) -> str:
            """
            Retrieve entities from the drawing with optional filters.
            
            [BACKWARD COMPATIBILITY] This uses get_all_entities() internally.
            For AI-driven extraction, prefer using get_all_entities() directly
            and let the agent reason about filtering.
            """
            # Let the method handle connection/document checking internally
            return str(self.autocad.get_entities_by_type(entity_type, layer, color))
        
        # --- Tool 6: Get Drawing Info ---
        class AutoCADInfoInput(BaseModel):
            """Input schema for drawing info (no parameters needed)."""
            pass
        
        def autocad_get_info() -> str:
            """
            Get metadata about the current drawing.
            
            Returns drawing name, path, units, layers, and entity counts.
            """
            return str(self.autocad.get_drawing_info())
        
        # --- Tool 7: Execute Command ---
        class AutoCADCommandInput(BaseModel):
            """Input schema for raw AutoCAD commands."""
            command: str = Field(
                description="AutoCAD command to execute (e.g., 'ZOOM E', 'REGEN')"
            )
        
        def autocad_execute_command(command: str) -> str:
            """
            Execute a raw AutoCAD command.
            
            Use for operations not covered by other specialized tools.
            Commands are sent directly to AutoCAD's command line.
            """
            return str(self.autocad.execute_command(command))

        # --- Tool: Dump All Tables ---
        class AutoCADDumpTablesInput(BaseModel):
            """No parameters needed – reads all TABLE objects in the active drawing."""
            pass

        def autocad_dump_all_tables() -> str:
            """
            Read every TABLE object in the drawing and return ALL cell text.

            This is the primary tool for extracting title-block metadata that is
            stored in AutoCAD TABLE objects, including:
            - Owner / buyer name
            - Land location, LGA, State
            - Plan number
            - Certification date
            - Surveyor name and address
            - CRS / coordinate origin
            - Pillar numbers

            Returns a list of tables; each table has a 'grid' key containing a
            2-D list of strings (row × column). Inspect every cell — the label is
            usually in one column and the value in the adjacent column.

            WORKFLOW for survey plan extraction:
            1. autocad_dump_all_tables() → scan grid for owner, location, plan no, etc.
            2. autocad_get_all_text() → capture TEXT/MTEXT annotations not in tables
            3. autocad_extract_boundary_area() → get the actual plot boundary area
            """
            result = self.autocad.dump_all_tables()
            # Fallback: if AutoCAD COM not connected, try ezdxf (limited TABLE support)
            if not result.get("success") and self.dxf_fallback.is_available and self.dxf_fallback.doc:
                return str({"success": False,
                            "error": result.get("error", "AutoCAD not connected"),
                            "note": "TABLE cell reading requires AutoCAD COM. ezdxf does not support TABLE cell text."})
            return str(result)

        # --- Tool: Extract Boundary Area (smart, avoids border frames) ---
        class AutoCADExtractBoundaryAreaInput(BaseModel):
            """No parameters needed – uses heuristics to identify the real plot boundary."""
            pass

        def autocad_extract_boundary_area() -> str:
            """
            Intelligently identify and measure the ACTUAL survey plot boundary area.

            Unlike autocad_calculate_area() which returns ALL closed polylines
            (including interior border frames and sheet borders), this tool applies
            a priority strategy to isolate the true land parcel outline:

            1. Prefers closed polylines on a layer whose name contains 'BOUNDARY'
               (but NOT 'INTERIOR' or 'BORDER') — e.g. CADA_BOUNDARY.
            2. Falls back to red-coloured polylines (survey convention: boundaries
               are 'verged in red').
            3. If neither above applies, excludes axis-aligned rectangular shapes
               (which are sheet borders / interior frames) and returns the SMALLEST
               remaining irregular closed polyline — almost always the land parcel.

            Returns the area in sq meters, hectares, acres, sq feet, plus the
            'strategy_used' field so the reasoning is transparent.

            USE THIS TOOL (not autocad_calculate_area) for survey plan extraction.
            If the result looks wrong, override with:
                autocad_calculate_area(layer='<correct_layer_name>')
            """
            result = self.autocad.calculate_boundary_area()
            if not result.get("success") and self.dxf_fallback.is_available and self.dxf_fallback.doc:
                return str(self.dxf_fallback.calculate_area())
            return str(result)

        # ==================================================================
        # EXCEL TOOLS
        # ==================================================================

        class ExcelInspectInput(BaseModel):
            """Input for inspecting Excel workbook structure."""
            file_path: str = Field(description="Path to the Excel file (.xlsx, .xls, .xlsm)")

        def excel_inspect_workbook(file_path: str) -> str:
            """
            Inspect an Excel workbook: list all sheet names and each sheet's column headers.
            MANDATORY FIRST STEP when the user refers to named data (e.g. 'Pre-fill', 'Post-fill',
            'coordinates', 'X/Y/Z'): call this to discover actual sheet and column names, then
            reason to map user terms to real names (e.g. 'Pre Fill' -> 'Pre_fill_2024', X/Y/Z -> EASTING, NORTHING, RL).
            Also reports ownership/family-block layout when present — if ownership.hint says normalize,
            call excel_normalize_ownership_workbook before CRS conversion or ArcGIS import.
            Only after this deep research should you call ArcGIS/Excel tools or report that data was not found.
            """
            import json
            out = self.excel_processor.inspect_workbook(file_path)
            return json.dumps(out, indent=2, default=str)

        class ExcelNormalizeOwnershipInput(BaseModel):
            """Normalize family/owner-block Excel into headed Easting/Northing/Pillar/Owner."""
            file_path: Optional[str] = Field(
                default=None,
                description="Path or basename of the ownership workbook (optional if sole Excel in workspace)",
            )
            output_name: Optional[str] = Field(
                default=None,
                description="Output .xlsx filename (e.g. ownership_normalized.xlsx). Defaults if omitted.",
            )
            user_request: Optional[str] = Field(
                default=None,
                description="Optional user wording (helps resolve which workbook / output name).",
            )

        def excel_normalize_ownership_workbook(
            file_path: Optional[str] = None,
            output_name: Optional[str] = None,
            user_request: Optional[str] = None,
        ) -> str:
            """
            Parse family/owner-block or headed ownership Excel into a clean table with columns
            Easting, Northing, Pillar, Owner. Required before excel_coordinate_converter or ArcGIS
            XY import when inspect shows Unnamed columns or owner-name 'headers'.
            Does NOT plot CAD — use for CRS conversion, GIS polygons/buffers/overlaps, etc.
            """
            import json
            from pathlib import Path

            from agent.excel_cadastral import normalize_ownership_workbook

            result = normalize_ownership_workbook(
                file_path,
                workspace=Path.cwd(),
                dest_name=output_name,
                query=user_request or "",
            )
            return json.dumps(result, indent=2, default=str)

        class ExcelInput(BaseModel):
            """Input schema for Excel processing."""
            file_path: str = Field(description="Path to Excel file")
            x_column: Optional[str] = Field(
                None,
                description="Column name containing X coordinates"
            )
            y_column: Optional[str] = Field(
                None,
                description="Column name containing Y coordinates"
            )

        def excel_processor_func(**kwargs) -> str:
            """
            Extract coordinate data from Excel spreadsheets.

            Supports .xlsx and .xls formats. Can automatically detect
            coordinate columns or use specified column names.
            """
            return str(self.excel_processor.process_file(**kwargs))

        class CsvToExcelInput(BaseModel):
            """Input schema for CSV to Excel conversion."""
            csv_path: str = Field(description="Path to the CSV file to convert")
            output_excel_path: Optional[str] = Field(
                None,
                description="Path for the output .xlsx file. If omitted, same folder as CSV, same name with .xlsx extension."
            )

        def csv_to_excel(csv_path: str, output_excel_path: Optional[str] = None) -> str:
            """
            Convert a CSV file to an Excel workbook (.xlsx).

            CRITICAL for workflows that start with CSV: ArcGIS ExcelToTable and many coordinate/import
            tools accept only .xlsx/.xls. If the user provides a .csv (e.g. Coords.csv), call this
            FIRST to create Coords.xlsx in the same folder, then use the Excel path for
            excel_coordinate_convert, arcgis_import_xy_points_from_excel, etc.
            """
            return str(self.excel_processor.csv_to_excel(csv_path, output_excel_path))

        class CadastralGenerateFromPromptInput(BaseModel):
            """Fully composed cadastral CAD prompt (inline coordinates required)."""
            cadastral_prompt: str = Field(
                description=(
                    "Complete cadastral prompt including Generate 'Out.dwg', buyer/location/LGA/state, "
                    "origin_crs, plan number, certification date, surveyor fields, pillar numbers, and "
                    "coordinates for the points as (EmE, NmN) pairs and/or bearing/distance legs."
                )
            )

        def cadastral_generate_from_prompt(cadastral_prompt: str) -> str:
            """
            Run the deterministic cadastral CAD pipeline on a fully composed prompt.

            Use AFTER reading Excel / extracting metadata from a reference DWG, when you have
            built a valid prompt with real EmE/NmN (or bearing/distance) geometry. Do NOT call
            this with 'coordinates … extracted from excel' placeholders.
            """
            import json

            result = self._run_cadastral_cad_prompt_pipeline(
                cadastral_prompt or "",
                skip_intent_assessment=False,
            )
            if result.get("success"):
                self._last_cadastral_output_dwg = result.get("output_dwg")
                self._last_cadastral_profile_path = result.get("profile_path")
            return json.dumps(result, indent=2, default=str)

        class ExcelCadastralPlotInput(BaseModel):
            """Natural-language Excel → cadastral request (multi-parcel OR per-owner plans)."""
            user_request: str = Field(
                description=(
                    "The user's full request: Excel ownership parcels, output .dwg name(s), "
                    "optional reference plan for location/surveyor/plan number, certification date, CRS. "
                    "May ask for one multi-parcel DWG or separate DWGs per owner."
                )
            )

        def excel_cadastral_plot(user_request: str) -> str:
            """
            Parse family/ownership parcels from Excel (any workbook name), optionally write a
            normalized copy when the user asks (any filename), pull title-block metadata from a
            named reference DWG when present, then plot:
            - SEPARATE owner plans when the user asks for different/unique CAD plans per owner
              (each buyer's coordinates → that buyer's .dwg; plan numbers increment from the
              user-stated start-from / plan-number base when present, else from the reference), OR
            - one MULTI-PARCEL DWG when they want all owners on a single sheet with letter tags.
            Field sources are independent: an explicit start-from plan number overrides the
            reference DWG plan no. even when location/surveyor are taken from that DWG.
            Plotting may use the input file directly.

            REFUSES inspect-only / deeper-extract requests that do not explicitly ask for a
            CAD/.dwg deliverable — never invent output drawings.
            """
            import json

            req = user_request or ""
            if not explicit_cadastral_plot_intent(req):
                return json.dumps(
                    {
                        "success": False,
                        "refused_reason": "missing_explicit_cadastral_plot_intent",
                        "error": (
                            "excel_cadastral_plot refused: the request does not explicitly ask "
                            "to generate/create/plot a CAD/.dwg plan. Use document/Excel read "
                            "tools for extract/inspect/deeper-search turns. Only call this tool "
                            "after a clear CAD deliverable request."
                        ),
                    },
                    indent=2,
                )
            result = self._run_excel_cadastral_pipeline(req)
            return json.dumps(result, indent=2, default=str)

        class CadastralComposeAndPlotInput(BaseModel):
            """Complex / multi-format deferred cadastral composition request."""
            user_request: str = Field(
                description=(
                    "Full user request where coordinates/bearings live in Excel/CSV/TXT/DOCX "
                    "(or a complex natural-language mix), plus Generate … .dwg and optional "
                    "reference plan for location/surveyor/plan number."
                )
            )

        def cadastral_compose_and_plot(user_request: str) -> str:
            """
            Intelligent composition: discover Excel/CSV/TXT/DOCX sources, extract geometry
            (deterministic first, then one RAG-informed LLM pass if needed), compose a valid
            cadastral prompt, and plot. Prefer for complex deferred-file prompts; use
            cadastral_generate_from_prompt only when EmE/NmN are already fully composed.

            REFUSES inspect-only / deeper-extract requests without an explicit CAD/.dwg ask.
            """
            import json

            req = user_request or ""
            if not explicit_cadastral_plot_intent(req):
                return json.dumps(
                    {
                        "success": False,
                        "refused_reason": "missing_explicit_cadastral_plot_intent",
                        "error": (
                            "cadastral_compose_and_plot refused: no explicit generate/create/"
                            "plot CAD/.dwg request. Continue with file read/extract tools only."
                        ),
                    },
                    indent=2,
                )
            result = self._run_intelligent_cadastral_pipeline(req)
            return json.dumps(result, indent=2, default=str)
        
        # ==================================================================
        # DOCUMENT PROCESSING TOOLS (Atomic, AI-driven extraction)
        # ==================================================================
        
        class DocumentMetadataInput(BaseModel):
            """Input schema for document metadata."""
            file_path: str = Field(description="Path to PDF or Word document")
        
        class DocumentTextInput(BaseModel):
            """Input schema for text extraction."""
            file_path: str = Field(description="Path to PDF or Word document")
            preserve_structure: bool = Field(
                default=True,
                description="Preserve paragraph breaks and document structure"
            )
        
        class DocumentTablesInput(BaseModel):
            """Input schema for table extraction."""
            file_path: str = Field(description="Path to PDF or Word document")
            page_number: Optional[int] = Field(
                default=None,
                description="Specific page number (for PDF) or None for all pages"
            )
        
        class DocumentSectionInput(BaseModel):
            """Input schema for section extraction."""
            file_path: str = Field(description="Path to PDF or Word document")
            section_title: Optional[str] = Field(
                default=None,
                description="Title of section to extract (e.g., 'Signature', 'Summary')"
            )
            start_keyword: Optional[str] = Field(
                default=None,
                description="Keyword marking start of section"
            )
            end_keyword: Optional[str] = Field(
                default=None,
                description="Keyword marking end of section"
            )
        
        class DocumentSearchInput(BaseModel):
            """Input schema for text search."""
            file_path: str = Field(description="Path to PDF or Word document")
            pattern: str = Field(description="Text pattern to search for")
            case_sensitive: bool = Field(default=False, description="Case-sensitive search")
            use_regex: bool = Field(default=False, description="Use regex pattern")
            context_lines: int = Field(default=2, description="Lines of context around matches")
        
        class DocumentStructuredDataInput(BaseModel):
            """Input schema for structured data extraction."""
            file_path: str = Field(description="Path to PDF or Word document")
            data_types: Optional[List[str]] = Field(
                default=None,
                description="Data types to extract: dates, names, numbers, emails, coordinates, depths, or 'all'"
            )
        
        class DocumentCreateInput(BaseModel):
            """Input schema for document creation."""
            file_path: str = Field(
                description=(
                    "Filename or path for the Word document (.docx). "
                    "When the user gives only a name (no folder), use that filename only — "
                    "SurvyAI saves it in the active workspace automatically."
                )
            )
            content: str = Field(description="Text content to write to the document")
            title: Optional[str] = Field(
                default=None,
                description="Optional title for the document"
            )
        
        class DocumentCreateStructuredInput(BaseModel):
            """Input schema for structured document creation."""
            file_path: str = Field(
                description=(
                    "Filename or path for the Word document (.docx). "
                    "Bare filenames are saved in the active SurvyAI workspace."
                )
            )
            title: str = Field(description="Document title")
            sections: List[Dict] = Field(
                description="List of sections, each with heading, level, content, and optional table"
            )
            metadata: Optional[Dict] = Field(
                default=None,
                description="Optional metadata (author, date, etc.)"
            )
        
        class DocumentReadInput(BaseModel):
            """Input schema for reading existing documents."""
            file_path: str = Field(description="Path to existing Word document to read")
        
        class DocumentUpdateInput(BaseModel):
            """Input schema for updating existing documents."""
            file_path: str = Field(description="Path to document to update (will be created if doesn't exist)")
            new_content: str = Field(description="New content to write")
            title: Optional[str] = Field(default=None, description="Optional title")
            overwrite: bool = Field(default=True, description="If True, replace entire document; if False, append")
        
        class DocumentStructureInput(BaseModel):
            """Input schema for document structure analysis."""
            file_path: str = Field(description="Path to PDF or Word document")
        
        class DocumentResourceEstimationInput(BaseModel):
            """Input schema for resource estimation."""
            file_path: str = Field(description="Path to PDF or Word document")
            model_name: Optional[str] = Field(
                default=None,
                description="LLM model name for cost estimation (defaults to current model)"
            )
        
        class DocumentExtractSectionsInput(BaseModel):
            """Input schema for keyword-based section extraction."""
            file_path: str = Field(description="Path to PDF or Word document")
            keywords: List[str] = Field(description="List of keywords to search for (e.g., ['Location', 'Personnel', 'Contractor'])")
            context_lines: int = Field(default=5, description="Number of lines of context around matches")
        
        def document_get_metadata(file_path: str) -> str:
            """
            Get document metadata (file info, structure, table presence).
            
            Use this first to understand document structure before extraction.
            Returns: file type, page count, table presence, etc.
            """
            result = self.document_processor.get_document_metadata(file_path)
            return str(result)
        
        def document_get_text(file_path: str, preserve_structure: bool = True) -> str:
            """
            Extract all text content from the document.
            
            Returns raw text that you can analyze. Use this for general text extraction
            or when you need to search through the entire document content.
            """
            # Always preflight cost/size first so the user is informed before expensive processing
            model_for_cost = None
            if self.settings.primary_llm == "openai":
                model_for_cost = self._current_openai_model or getattr(self.settings, "openai_model_mini", "gpt-5.4-mini")
            elif self.settings.primary_llm == "gemini":
                model_for_cost = self._current_gemini_model or getattr(self.settings, "gemini_model", "gemini-2.0-flash")
            else:
                model_for_cost = getattr(self.settings, "openai_model_mini", "gpt-5.4-mini")

            est = self.document_processor.get_resource_estimation(file_path, model_for_cost)
            if est.get("success") and est.get("warnings"):
                return str({
                    "success": False,
                    "blocked": True,
                    "reason": "Document appears large; full text extraction is blocked by default to prevent timeouts/cost blowups.",
                    "resource_estimation": est,
                    "next_steps": [
                        "Use document_extract_sections_by_keywords(file_path, keywords=[...]) to pull only relevant parts",
                        "If you explicitly want full extraction, confirm and then use document_get_text_force(file_path)"
                    ],
                    "question": "Proceed with FULL text extraction via document_get_text_force? (yes/no)"
                })

            result = self.document_processor.get_full_text(file_path, preserve_structure)
            return str({
                "resource_estimation": est,
                "result": result
            })

        def document_get_text_force(file_path: str, preserve_structure: bool = True) -> str:
            """
            Force full text extraction from the document, even if large.
            
            Use ONLY after the user explicitly confirms they want full extraction.
            """
            result = self.document_processor.get_full_text_force(file_path, preserve_structure)
            return str(result)
        
        def document_get_tables(file_path: str, page_number: Optional[int] = None) -> str:
            """
            Extract all tables from the document.
            
            Returns structured table data with headers and rows. Use this when
            the document contains tabular data (e.g., feature lists, measurements).
            """
            result = self.document_processor.get_tables(file_path, page_number)
            return str(result)
        
        def document_get_section(
            file_path: str,
            section_title: Optional[str] = None,
            start_keyword: Optional[str] = None,
            end_keyword: Optional[str] = None
        ) -> str:
            """
            Extract text from a specific section of the document.
            
            Use this to extract specific sections like signatures, summaries,
            or findings. Provide either section_title or start_keyword/end_keyword.
            """
            result = self.document_processor.get_text_by_section(
                file_path, start_keyword, end_keyword, section_title
            )
            return str(result)
        
        def document_search_text(
            file_path: str,
            pattern: str,
            case_sensitive: bool = False,
            use_regex: bool = False,
            context_lines: int = 2
        ) -> str:
            """
            Search for specific text patterns in the document.
            
            Use this to find specific information like dates, names, or keywords.
            Supports regex patterns for flexible searching.
            """
            result = self.document_processor.search_text(
                file_path, pattern, case_sensitive, use_regex, context_lines
            )
            return str(result)
        
        def document_extract_structured_data(
            file_path: str,
            data_types: Optional[List[str]] = None
        ) -> str:
            """
            Extract structured data types (dates, names, numbers, emails, etc.).
            
            Use this to quickly extract common data types. Specify data_types as
            a list: ['dates', 'names', 'numbers', 'emails', 'coordinates', 'depths'] or ['all'].
            """
            result = self.document_processor.extract_structured_data(file_path, data_types)
            return str(result)
        
        def document_create_word(
            file_path: str,
            content: str,
            title: Optional[str] = None
        ) -> str:
            """
            Create a new Word document (.docx) with the specified content.
            
            CRITICAL CONTEXT RULES:
            - Use ONLY the data you JUST extracted and displayed in THIS conversation
            - When user says "save the summary", use the summary you JUST showed them above in YOUR CURRENT RESPONSE
            - NEVER use data from previous conversations or different documents
            - The 'content' parameter should be the text you displayed in your IMMEDIATELY PRECEDING response
            - CONTEXT ISOLATION: Each conversation is independent - do NOT mix data from different documents
            
            Use this when the user explicitly asks to save, export, or create a document file.
            When user says "save as [filename]" or "export as [filename]", use this tool immediately.
            
            Args:
                file_path: Filename or path for the .docx. Bare filenames are saved in the active
                    SurvyAI workspace (current working directory). Prefer workspace + filename
                    when the user does not give a full folder path.
                content: Text content to write - MUST be from CURRENT conversation response, not previous ones
                title: Optional document title
                
            Returns: Success message with file path
            """
            try:
                resolved = str(self._resolve_output_path_for_write(file_path))
            except Exception:
                resolved = file_path
            result = self.document_processor.create_word_document(resolved, content, title)
            return str(result)
        
        def document_create_structured_word(
            file_path: str,
            title: str,
            sections: List[Dict],
            metadata: Optional[Dict] = None
        ) -> str:
            """
            Create a Word document from structured data with sections and tables.
            
            Use this for creating professional reports with multiple sections, headings, and tables.
            More advanced than document_create_word - use when you have structured data.
            
            Args:
                file_path: Filename or path for the .docx (bare names → active workspace)
                title: Document title
                sections: List of section dicts with 'heading', 'level', 'content', optional 'table'
                metadata: Optional metadata dict
                
            Returns: Success message with file path
            """
            try:
                resolved = str(self._resolve_output_path_for_write(file_path))
            except Exception:
                resolved = file_path
            result = self.document_processor.create_word_document_from_structure(
                resolved, title, sections, metadata
            )
            return str(result)
        
        def document_read_word(file_path: str) -> str:
            """
            Read an existing Word document to get its content.
            
            CRITICAL: Use this PROACTIVELY when:
            - User asks to modify/update/shorten a document you JUST created in this conversation
            - User says "the same document" or "the same file" - use the file path from your previous response
            - You need to read a document before updating it
            
            REMEMBER: If you just created a file and mentioned its path in your response, use that path here.
            Don't ask the user for the path - you already know it from the conversation context.
            
            Args:
                file_path: Path to existing Word document (remember this from when you created it)
                
            Returns: Document content (text, paragraphs, tables)
            """
            result = self.document_processor.read_existing_word_document(file_path)
            return str(result)
        
        def document_update_word(
            file_path: str,
            new_content: str,
            title: Optional[str] = None,
            overwrite: bool = True
        ) -> str:
            """
            Update an existing Word document with new content.
            
            CRITICAL: Use this PROACTIVELY when:
            - User asks to modify/update/shorten a document you JUST created
            - User says "save in the same document" or "update the same file"
            - After reading a document with document_read_word, use this to save the modified content
            
            REMEMBER: Use the same file_path you used when creating the document - it's in your previous response.
            Don't ask the user for the path - you already know it from the conversation context.
            
            Args:
                file_path: Path to document (remember this from when you created it, or from document_read_word)
                new_content: New content to write (use actual extracted/condensed data, not placeholders)
                title: Optional title
                overwrite: If True, replace entire document; if False, append
                
            Returns: Success message with file path
            """
            try:
                resolved = str(self._resolve_output_path_for_write(file_path))
            except Exception:
                resolved = file_path
            result = self.document_processor.update_word_document(
                resolved, new_content, title, overwrite
            )
            return str(result)
        
        def document_get_structure(file_path: str) -> str:
            """
            Analyze document structure (headings, sections, organization).
            
            CRITICAL FOR LARGE DOCUMENTS (>100 pages or >50K words):
            - ALWAYS call this FIRST for large documents before extracting text
            - Use this to understand document organization
            - Then use document_extract_sections_by_keywords to extract only relevant sections
            - This prevents processing the entire document and saves tokens/costs
            
            Returns: Document structure with sections, headings, and outline.
            """
            result = self.document_processor.get_document_structure(file_path)
            return str(result)
        
        def document_get_resource_estimation(
            file_path: str,
            model_name: Optional[str] = None
        ) -> str:
            """
            Estimate resource requirements and costs for processing a document.
            
            CRITICAL FOR LARGE DOCUMENTS:
            - ALWAYS call this FIRST when user requests processing a document
            - Shows file size, estimated tokens, and cost
            - Provides warnings and recommendations for large documents
            - Use this to inform the user before processing
            
            Args:
                file_path: Path to document
                model_name: Optional model name (defaults to current model)
            
            Returns: Resource estimation with warnings, costs, and recommendations.
            """
            if not model_name:
                if self.settings.primary_llm == "openai":
                    model_name = self._current_openai_model or getattr(self.settings, "openai_model_mini", "gpt-5.4-mini")
                elif self.settings.primary_llm == "gemini":
                    model_name = self._current_gemini_model or getattr(self.settings, "gemini_model", "gemini-2.0-flash")
                else:
                    model_name = getattr(self.settings, "openai_model_mini", "gpt-5.4-mini")
            
            result = self.document_processor.get_resource_estimation(file_path, model_name)
            return str(result)
        
        def document_extract_sections_by_keywords(
            file_path: str,
            keywords: List[str],
            context_lines: int = 5
        ) -> str:
            """
            Extract only document sections matching specific keywords.
            
            CRITICAL FOR LARGE DOCUMENTS:
            - Use this instead of document_get_text for documents >100 pages
            - Extracts only relevant sections, saving tokens and costs
            - Much faster than processing entire document
            
            Workflow for large documents:
            1. document_get_resource_estimation → Check if document is large
            2. document_get_structure → Understand document organization
            3. document_extract_sections_by_keywords → Extract only relevant sections
            4. Process extracted sections instead of full document
            
            Args:
                file_path: Path to document
                keywords: List of keywords to search for (e.g., ["Location", "Personnel", "Contractor"])
                context_lines: Number of lines of context around matches (default: 5)
            
            Returns: Extracted sections matching keywords with context.
            """
            result = self.document_processor.extract_sections_by_keywords(
                file_path, keywords, context_lines
            )
            return str(result)
        
        # ==================================================================
        # COORDINATE CONVERSION TOOL
        # ==================================================================
        
        class CoordConvertInput(BaseModel):
            """Input schema for coordinate conversion."""
            x: float = Field(description="X coordinate (Easting or Longitude)")
            y: float = Field(description="Y coordinate (Northing or Latitude)")
            source_crs: str = Field(
                "WGS84", 
                description="Source coordinate reference system"
            )
            target_crs: str = Field(
                "WGS84", 
                description="Target coordinate reference system"
            )
            use_geographic_calculator: bool = Field(
                False,
                description="If True, attempt to use Geographic Calculator COM interface. "
                           "If False (default), use pyproj. Only set to True if user explicitly "
                           "requests Geographic Calculator in their query."
            )
        
        def coordinate_convert(**kwargs) -> str:
            """
            Convert coordinates between different reference systems using pyproj (default).
            
            Uses pyproj by default for fast, reliable coordinate conversions. Supports:
            - WGS84, UTM zones, State Plane
            - Nigerian coordinate systems (Minna NTM, etc.)
            - Many other coordinate reference systems from EPSG database
            
            If user explicitly requests "Geographic Calculator" in their query, will attempt
            to use Geographic Calculator COM interface, but always falls back to pyproj if COM
            is unavailable or fails.
            
            The system automatically resolves informal CRS names (e.g., "MINNA_NTM_MIDBELT")
            to proper EPSG codes for accurate conversions.
            """
            # Detect if user explicitly requested Geographic Calculator
            # This is a simple heuristic - in practice, the LLM should detect this from context
            use_geocalc = kwargs.pop('use_geographic_calculator', False)
            return str(self.blue_marble.convert_coordinate(use_geographic_calculator=use_geocalc, **kwargs))

        class CoordConvertAutoInput(BaseModel):
            """
            Input schema for survey-aware coordinate conversion from free-form text.

            This is designed for real-world survey formats (DMS/DM/decimal degrees, hemisphere letters,
            and projected coordinates with E/N or X/Y labels).
            """
            text: str = Field(
                description=(
                    "Free-form text containing coordinates (and optionally CRS names/codes). "
                    "Examples: '6°12\\'30.5\"N 3°21\\'10\"E', 'E 512345.12 N 6789012.34', "
                    "'lat 6 12 30 N lon 3 21 10 E', '6.1234, 3.4567'."
                )
            )
            source_crs: Optional[str] = Field(
                default=None,
                description=(
                    "Optional source CRS name/code. If omitted, SurvyAI will try to infer from text "
                    "(e.g., 'WGS84', 'UTM Zone 32N', 'EPSG:4326')."
                ),
            )
            target_crs: Optional[str] = Field(
                default=None,
                description=(
                    "Optional target CRS name/code. If omitted, SurvyAI will try to infer from text. "
                    "If still unknown, defaults to WGS84."
                ),
            )
            use_geographic_calculator: bool = Field(
                default=False,
                description="If True, attempt to use Geographic Calculator COM interface; otherwise use pyproj.",
            )

        def coordinate_convert_auto(
            text: str,
            source_crs: Optional[str] = None,
            target_crs: Optional[str] = None,
            use_geographic_calculator: bool = False,
        ) -> str:
            """
            Survey-aware coordinate conversion from free-form text.

            What it does:
            - Auto-detect coordinates inside text (supports DMS like 6°12'30\"N, hemisphere letters, E/N labels)
            - Normalizes geodetic coordinates to decimal degrees
            - Attempts to infer CRS names/codes from the text ("from ... to ...", EPSG/WKID, UTM zone hints)
            - Converts each detected coordinate pair using the existing BlueMarbleConverter (pyproj by default)
            """
            try:
                parsed = infer_crs_from_text(text)
                src = (source_crs or parsed.get("source_crs") or "WGS84").strip()
                dst = (target_crs or parsed.get("target_crs") or "WGS84").strip()

                points = extract_points(text, max_points=20)
                if not points:
                    return (
                        "✗ No coordinates detected. Provide coordinates in one of these formats:\n"
                        "- Geodetic (DMS/DM): 6°12'30.5\"N 3°21'10\"E\n"
                        "- Geodetic (decimal): 6.1234N, 3.4567E or 6.1234, 3.4567\n"
                        "- Projected: E 512345.12 N 6789012.34 or X=512345.12 Y=6789012.34\n"
                        "Also include CRS hints like 'from WGS84 to UTM Zone 32N' or 'EPSG:4326 to EPSG:32632'."
                    )

                results = []
                for p in points:
                    r = self.blue_marble.convert_coordinate(
                        x=p.x,
                        y=p.y,
                        source_crs=src,
                        target_crs=dst,
                        use_geographic_calculator=use_geographic_calculator,
                    )
                    results.append(
                        {
                            "parsed": {
                                "x": p.x,
                                "y": p.y,
                                "kind": p.kind,
                                "source_text": p.source_text,
                                "notes": p.notes,
                            },
                            "conversion": r,
                        }
                    )

                payload = {
                    "success": True,
                    "source_crs": src,
                    "target_crs": dst,
                    "count": len(results),
                    "results": results,
                }
                return json.dumps(payload, indent=2, ensure_ascii=False)
            except Exception as e:
                return f"✗ Auto coordinate conversion failed: {e}"
        
        # Excel batch coordinate conversion tool
        class ExcelCoordConvertInput(BaseModel):
            """Input schema for Excel coordinate conversion."""
            excel_path: str = Field(description="Path to Excel file containing coordinates")
            x_column: str = Field(
                default="X",
                description="Name of column containing X/Easting coordinates"
            )
            y_column: str = Field(
                default="Y",
                description="Name of column containing Y/Northing coordinates"
            )
            source_crs: str = Field(
                default="WGS84",
                description="Source coordinate reference system"
            )
            target_crs: str = Field(
                default="WGS84",
                description="Target coordinate reference system"
            )
            source_zone: Optional[int] = Field(
                default=None,
                description="Source UTM zone (if applicable)"
            )
            target_zone: Optional[int] = Field(
                default=None,
                description="Target UTM zone (if applicable)"
            )
            output_path: Optional[str] = Field(
                default=None,
                description=(
                    "Output file path. "
                    "CRITICAL: If not specified, automatically saves in same folder as excel_path "
                    "with '_converted' suffix. This ensures outputs are created alongside input files."
                )
            )
            sheet_name: Optional[str] = Field(
                default=None,
                description="Sheet name to process (default: first sheet)"
            )
            use_geographic_calculator: bool = Field(
                default=False,
                description="If True, attempt to use Geographic Calculator COM interface. "
                           "If False (default), use pyproj. Only set to True if user explicitly "
                           "requests Geographic Calculator in their query."
            )
            user_request: Optional[str] = Field(
                default=None,
                description=(
                    "Optional full user wording. Used only to infer a save-as Excel output name "
                    "when output_path is omitted (e.g. save file as 'coords_midbelt' excel file)."
                ),
            )
        
        def excel_coordinate_convert(**kwargs) -> str:
            """
            Convert coordinates in an Excel file using pyproj (default).
            
            Reads coordinates from specified columns, converts them using pyproj (default),
            and saves results to a new Excel file with converted coordinates added as new columns.
            
            If user explicitly requests "Geographic Calculator" in their query, will attempt
            to use Geographic Calculator COM interface, but always falls back to pyproj if COM
            is unavailable or fails.
            
            The system automatically resolves informal CRS names (e.g., "MINNA_NTM_MIDBELT")
            to proper EPSG codes for accurate conversions.

            When the sheet is a family/owner-block layout (no Easting/Northing headers),
            SurvyAI auto-normalizes once, then converts — do not re-parse inside ArcPy.
            """
            try:
                from pathlib import Path

                from agent.excel_cadastral import (
                    extract_requested_excel_output_name,
                    normalize_ownership_workbook,
                )

                # Extract use_geographic_calculator if provided, default to False
                use_geocalc = kwargs.pop("use_geographic_calculator", False)
                user_request = str(kwargs.pop("user_request", None) or "")
                # Alias cleanup — convert_excel_file only accepts excel_path.
                if not kwargs.get("excel_path") and kwargs.get("file_path"):
                    kwargs["excel_path"] = kwargs.pop("file_path")
                else:
                    kwargs.pop("file_path", None)

                if not kwargs.get("output_path") and user_request:
                    requested_out = extract_requested_excel_output_name(user_request)
                    if requested_out:
                        src = Path(str(kwargs.get("excel_path") or ""))
                        kwargs["output_path"] = str(
                            (src.parent / requested_out) if src.parent.exists() else requested_out
                        )

                # Keep only kwargs accepted by convert_excel_file
                allowed = {
                    "excel_path",
                    "x_column",
                    "y_column",
                    "source_crs",
                    "target_crs",
                    "source_zone",
                    "target_zone",
                    "output_path",
                    "sheet_name",
                    "output_schema",
                    "output_x_column",
                    "output_y_column",
                    "include_crs_metadata",
                    "include_error_column",
                }
                kwargs = {k: v for k, v in kwargs.items() if k in allowed and v is not None}

                try:
                    result = self.blue_marble.convert_excel_file(
                        use_geographic_calculator=use_geocalc, **kwargs
                    )
                except Exception as first_exc:
                    msg = str(first_exc).lower()
                    needs_normalize = any(
                        s in msg
                        for s in (
                            "no valid coordinates",
                            "column not found",
                            "could not find",
                            "not found in",
                            "no coordinates",
                        )
                    )
                    if not needs_normalize:
                        raise

                    excel_path = kwargs.get("excel_path") or ""
                    # Intermediate headed table only — never steal the user-named converted file.
                    norm = normalize_ownership_workbook(
                        excel_path,
                        workspace=Path.cwd(),
                        dest_name="ownership_normalized.xlsx",
                        query="",
                    )
                    if not norm.get("success"):
                        raise first_exc
                    retry_kwargs = dict(kwargs)
                    retry_kwargs["excel_path"] = norm["output_path"]
                    retry_kwargs["x_column"] = "Easting"
                    retry_kwargs["y_column"] = "Northing"
                    # Preserve Owner through conversion when the converter keeps extra columns.
                    result = self.blue_marble.convert_excel_file(
                        use_geographic_calculator=use_geocalc, **retry_kwargs
                    )
                    result = dict(result)
                    result["normalized_input"] = norm.get("output_path")
                    result["owners"] = norm.get("owners")
                return (
                    f"✓ Excel coordinate conversion completed successfully!\n\n"
                    f"Input file: {result['input_file']}\n"
                    f"Output file: {result['output_file']}\n"
                    f"Total coordinates: {result['total_coordinates']}\n"
                    f"Successful conversions: {result['successful_conversions']}\n"
                    f"Failed conversions: {result['failed_conversions']}\n"
                    f"Conversion method: {result['method']}\n"
                    f"Source CRS: {result['source_crs']}\n"
                    f"Target CRS: {result['target_crs']}\n"
                    f"Output columns: {', '.join(result['output_columns'])}"
                    + (
                        f"\nNormalized ownership workbook first: {result.get('normalized_input')}"
                        if result.get("normalized_input")
                        else ""
                    )
                    + (
                        f"\nOwners preserved: {', '.join(result.get('owners') or [])}"
                        if result.get("owners")
                        else ""
                    )
                )
            except Exception as e:
                return (
                    f"✗ Excel coordinate conversion failed: {str(e)}\n"
                    "If the sheet uses owner title rows + E/N blocks without headers, call "
                    "excel_normalize_ownership_workbook first, then retry with "
                    "x_column=Easting, y_column=Northing."
                )

        class ExcelConvertAndAreaInput(BaseModel):
            """Convert coordinates in an Excel file (including DMS) and compute area automatically."""
            excel_path: str = Field(description="Path to Excel file containing boundary coordinates")
            source_crs: str = Field(default="WGS84", description="Source CRS (e.g., WGS84, EPSG:4326)")
            target_crs: str = Field(default="WGS84", description="Target CRS (e.g., Minna Nigerian NTM MidBelt)")
            x_column: str = Field(default="Long.", description="Longitude/Easting column name (tabs/whitespace tolerated)")
            y_column: str = Field(default="Lat.", description="Latitude/Northing column name (tabs/whitespace tolerated)")
            output_filename: str = Field(
                default="converted1.xlsx",
                description="Output filename to save in the same folder as the input Excel file",
            )
            area_on: Literal["best", "source", "target"] = Field(
                default="best",
                description="Where to compute area: best (auto), source (original coords), or target (converted coords).",
            )

        def excel_convert_and_area(
            excel_path: str,
            source_crs: str = "WGS84",
            target_crs: str = "WGS84",
            x_column: str = "Long.",
            y_column: str = "Lat.",
            output_filename: str = "converted1.xlsx",
            area_on: str = "best",
        ) -> str:
            """
            One-shot workflow:
            - Reads Excel
            - Parses DMS/DM/decimal values in Lat/Long columns
            - Converts source->target CRS
            - Saves output to same folder as input
            - Computes area using best available method
            """
            try:
                from pathlib import Path
                import pandas as pd
                from utils.coordinate_parsing import parse_angle

                inp = Path(excel_path)
                out_path = (inp.parent / Path(output_filename).name).resolve()

                # Run conversion using the improved converter (handles DMS + messy headers)
                conv = self.blue_marble.convert_excel_file(
                    excel_path=str(inp),
                    x_column=x_column,
                    y_column=y_column,
                    source_crs=source_crs,
                    target_crs=target_crs,
                    output_path=str(out_path),
                    use_geographic_calculator=False,
                    # Default to a clean output: original columns + one converted X/Y pair
                    output_schema="clean",
                )

                # Compute area:
                # - for WGS84-like/geodetic source: geodesic (best)
                # - for projected: planar (best)
                # Read back the saved file so we can access converted columns reliably
                df_out = pd.read_excel(out_path, sheet_name=0)
                x_conv_col = conv.get("output_x_column") or "X"
                y_conv_col = conv.get("output_y_column") or "Y"

                # Build point lists (assume row order is vertex order; close polygon automatically)
                def _series_to_points(df, xcol, ycol, allow_dms: bool = False):
                    pts = []
                    for _, r in df.iterrows():
                        try:
                            rx = r.get(xcol)
                            ry = r.get(ycol)
                            if allow_dms:
                                xv = parse_angle(str(rx)) if rx is not None else None
                                yv = parse_angle(str(ry)) if ry is not None else None
                                if xv is None or yv is None:
                                    continue
                                xv = float(xv)
                                yv = float(yv)
                            else:
                                xv = float(rx)
                                yv = float(ry)
                            pts.append((xv, yv))
                        except Exception:
                            continue
                    return pts

                # Source columns may be DMS strings (lat/long), so allow parsing there.
                src_pts = _series_to_points(df_out, conv["x_column"], conv["y_column"], allow_dms=True)
                tgt_pts = _series_to_points(df_out, x_conv_col, y_conv_col, allow_dms=False)

                if not src_pts:
                    return f"✗ No usable source points found for area calculation in {out_path}"

                # Decide which set to use
                area_choice = (area_on or "best").lower()
                if area_choice == "target" and tgt_pts:
                    area_res = best_area(tgt_pts, crs_hint=target_crs)
                elif area_choice == "source":
                    area_res = best_area(src_pts, crs_hint=source_crs)
                else:
                    # best: prefer geodesic if source looks like lon/lat, else planar on target if present
                    try_src = best_area(src_pts, crs_hint=source_crs)
                    if try_src.method.startswith("geodesic"):
                        area_res = try_src
                    elif tgt_pts:
                        area_res = best_area(tgt_pts, crs_hint=target_crs)
                    else:
                        area_res = try_src

                summary = {
                    "success": True,
                    "input_file": str(inp),
                    "output_file": str(out_path),
                    "source_crs": source_crs,
                    "target_crs": target_crs,
                    "converted_points": conv.get("total_coordinates"),
                    "area": {
                        "method": area_res.method,
                        "m2": area_res.area_m2,
                        "hectares": area_res.hectares,
                        "ft2": area_res.ft2,
                        "acres": area_res.acres,
                        "perimeter_m": area_res.perimeter_m,
                        "computed_on": area_choice,
                    },
                }
                return json.dumps(summary, indent=2, ensure_ascii=False)
            except Exception as e:
                return f"✗ Excel convert+area failed: {e}"
        
        # ==================================================================
        # BUILD TOOL LIST
        # ==================================================================
        # After the list is built, `self._filter_tools_by_feature_flags` applies
        # SURVYAI_LICENSE_MODE + SURVYAI_FEATURE_* (Phase 2).
        
        tools = [
            # AutoCAD tools
            StructuredTool(
                name="autocad_open_drawing",
                description=(
                    "Open a DWG or DXF file in AutoCAD. "
                    "MUST be called before any other AutoCAD operations. "
                    "STRICT: Survey plan templates (e.g. survey_plan_template2.dwg) are always opened read-only and must never be written to avoid corruption."
                ),
                func=autocad_open,
                args_schema=AutoCADOpenInput
            ),
            # Internet search (permissioned)
            StructuredTool(
                name="internet_search",
                description=(
                    "Search the internet for up-to-date information. "
                    "MUST ask user permission before using. "
                    "All results returned are internet-sourced and must be clearly highlighted in your response."
                ),
                func=internet_search,
                args_schema=InternetSearchInput
            ),
            StructuredTool(
                name="autocad_calculate_area",
                description=(
                    "Calculate area of closed shapes (polylines, circles, hatches). "
                    "Use color='red' for boundaries 'verged in red'. "
                    "Returns area in sq meters, sq feet, hectares, and acres."
                ),
                func=autocad_calculate_area,
                args_schema=AutoCADAreaInput
            ),
            StructuredTool(
                name="autocad_search_text",
                description=(
                    "Search for text matching a pattern. "
                    "Use patterns like 'property of' or 'plan shewing' "
                    "to find owner names and survey titles."
                ),
                func=autocad_search_text,
                args_schema=AutoCADTextSearchInput
            ),
            StructuredTool(
                name="autocad_get_all_text",
                description=(
                    "Get all text content from the drawing. "
                    "Use to find titles, names, annotations, and labels."
                ),
                func=autocad_get_all_text,
                args_schema=AutoCADGetTextInput
            ),
            # AI-Driven atomic tools (preferred for reasoning)
            StructuredTool(
                name="autocad_get_all_entities",
                description=(
                    "[AI-DRIVEN] Get ALL entities with complete properties for agent reasoning. "
                    "Returns raw data (type, layer, color, coordinates, area, closed status) "
                    "without filtering. Agent should reason about which entities match criteria. "
                    "Use this for AI-driven extraction where the agent reasons about what to extract."
                ),
                func=autocad_get_all_entities,
                args_schema=AutoCADGetAllEntitiesInput
            ),
            StructuredTool(
                name="autocad_get_entities_summary",
                description=(
                    "[AI-DRIVEN] Get lightweight summary (counts by type, color, layer) "
                    "to understand drawing structure before detailed extraction."
                ),
                func=autocad_get_entities_summary,
                args_schema=AutoCADEntitiesSummaryInput
            ),
            StructuredTool(
                name="autocad_get_entity",
                description=(
                    "[AI-DRIVEN] Get detailed properties of a specific entity by handle. "
                    "Use after identifying entities from get_all_entities()."
                ),
                func=autocad_get_entity,
                args_schema=AutoCADGetEntityInput
            ),
            StructuredTool(
                name="autocad_calculate_entity_area",
                description=(
                    "[AI-DRIVEN] Calculate area of a specific entity by handle. "
                    "Agent should first identify closed shapes from get_all_entities(), "
                    "then call this for each one. Workflow: get_all_entities() -> "
                    "reason about closed shapes -> calculate_entity_area(handle) for each."
                ),
                func=autocad_calculate_entity_area,
                args_schema=AutoCADEntityAreaInput
            ),
            # Backward compatibility tools
            StructuredTool(
                name="autocad_get_entities",
                description=(
                    "[BACKWARD COMPATIBILITY] Get entities with filters. "
                    "For AI-driven extraction, prefer autocad_get_all_entities() "
                    "and let the agent reason about filtering."
                ),
                func=autocad_get_entities,
                args_schema=AutoCADEntitiesInput
            ),
            StructuredTool(
                name="autocad_get_info",
                description="Get drawing metadata: units, layers, entity counts.",
                func=autocad_get_info,
                args_schema=AutoCADInfoInput
            ),
            StructuredTool(
                name="autocad_command",
                description=(
                    "Execute a raw AutoCAD command. "
                    "Use for operations not covered by other tools."
                ),
                func=autocad_execute_command,
                args_schema=AutoCADCommandInput
            ),
            StructuredTool(
                name="autocad_dump_all_tables",
                description=(
                    "Read ALL AutoCAD TABLE objects and return every cell's text content. "
                    "USE THIS FIRST when extracting survey plan metadata: owner name, "
                    "land location, LGA, state, plan number, certification date, "
                    "surveyor name/address, CRS/origin, pillar numbers. "
                    "Returns a 'grid' (2-D list) for each table — scan all cells for labels and values. "
                    "REQUIRES AutoCAD COM (active drawing open). ezdxf does not support TABLE cell text."
                ),
                func=autocad_dump_all_tables,
                args_schema=AutoCADDumpTablesInput
            ),
            StructuredTool(
                name="autocad_extract_boundary_area",
                description=(
                    "Identify and measure the ACTUAL survey plot boundary area using smart heuristics. "
                    "Prefers the CADA_BOUNDARY layer, then red polylines, then smallest non-rectangular "
                    "closed polyline (excludes sheet borders / interior border frames). "
                    "ALWAYS USE THIS instead of autocad_calculate_area() for survey plan area extraction — "
                    "autocad_calculate_area() without a layer filter includes border frames and gives wrong results. "
                    "Returns area in sq meters, hectares, acres, sq feet plus 'strategy_used' for transparency."
                ),
                func=autocad_extract_boundary_area,
                args_schema=AutoCADExtractBoundaryAreaInput
            ),

            # Other tools
            StructuredTool(
                name="excel_inspect_workbook",
                description=(
                    "Inspect Excel workbook structure: list all sheet names and each sheet's column headers. "
                    "MANDATORY FIRST when the user refers to named sheets or data (e.g. 'Pre-fill', 'Post-fill', "
                    "'coordinates', 'X/Y/Z'): discover actual names, then reason to map user intent to real sheet/column names. "
                    "Also returns ownership.layout_kind / ownership.hint when the sheet is family/owner-block "
                    "(owner title rows + E/N/pillar). If hint says normalize, call excel_normalize_ownership_workbook "
                    "BEFORE excel_coordinate_converter or ArcGIS XY import. "
                    "Only report errors or ask for names after this deep research."
                ),
                func=excel_inspect_workbook,
                args_schema=ExcelInspectInput
            ),
            StructuredTool(
                name="excel_processor",
                description="Extract coordinate data from Excel files (.xlsx, .xls).",
                func=excel_processor_func,
                args_schema=ExcelInput
            ),
            StructuredTool(
                name="csv_to_excel",
                description=(
                    "Convert a CSV file to an Excel file (.xlsx). Use this FIRST when the user provides a .csv "
                    "but downstream steps need Excel (e.g. coordinate conversion, ArcGIS import). "
                    "Output defaults to same folder as CSV with .xlsx extension. "
                    "Parameters: csv_path (required), output_excel_path (optional)."
                ),
                func=csv_to_excel,
                args_schema=CsvToExcelInput,
            ),
            StructuredTool(
                name="excel_normalize_ownership_workbook",
                description=(
                    "Normalize family/owner-block Excel (owner title rows + Easting/Northing/pillar blocks, "
                    "often shown as Unnamed columns by excel_inspect_workbook) into a headed workbook with "
                    "columns Easting, Northing, Pillar, Owner. "
                    "Call this BEFORE excel_coordinate_converter, arcgis_import_xy_points, or ArcPy polygon/"
                    "buffer/overlap workflows when the sheet is not already headed. "
                    "Does NOT create CAD/.dwg plans — for GIS/CRS/analysis preparation only. "
                    "Never re-implement family-block parsing inside arcgis_execute_python_code."
                ),
                func=excel_normalize_ownership_workbook,
                args_schema=ExcelNormalizeOwnershipInput,
            ),
            StructuredTool(
                name="excel_cadastral_plot",
                description=(
                    "Excel family/ownership parcels → cadastral DWG(s). "
                    "Use ONLY when the user explicitly wants Generate/create/plot … .dwg / CAD plan(s) "
                    "and Easting/Northing/pillar values are in a spreadsheet (owner blocks "
                    "separated by blank rows), optionally with metadata from an existing plan DWG. "
                    "Do NOT use for read/extract/inspect/summarize Excel turns, or bare 'yes/go ahead' "
                    "after those turns (that means deeper extract/search, not CAD). "
                    "SEMANTICS (critical): "
                    "(1) SEPARATE plans — phrases like 'different CAD plans', 'each owner … only that "
                    "plan', 'buyer name as the name of that particular CAD drawing', plan-number "
                    "increment → one .dwg per owner named after the buyer. "
                    "(2) MULTI-PARCEL — 'all owner names', letter tags like 'AMADI (B)', parcels "
                    "marked within one plan → a single multi-parcel DWG. "
                    "(3) PLAN NUMBER — 'start from plan number …' / 'plan number: …' is the first "
                    "plan and increments per owner; it overrides the reference DWG plan no. "
                    "Taking location/surveyor from a reference plan does not imply taking its "
                    "plan number when a start-from was stated. "
                    "Chooses the best ownership workbook by content (real owner/family rows), not by "
                    "filename. Writes a normalized Easting/Northing/Pillar/Owner copy only when asked. "
                    "Prefer this over inventing coordinates or calling cadastral_generate_from_prompt "
                    "with placeholders. Never invent a default DWG name the user did not request."
                ),
                func=excel_cadastral_plot,
                args_schema=ExcelCadastralPlotInput,
            ),
            StructuredTool(
                name="cadastral_compose_and_plot",
                description=(
                    "INTELLIGENT multi-format cadastral composition for complex prompts that "
                    "explicitly ask to Generate/create/plot a .dwg/CAD plan: "
                    "reads coordinates/bearings from Excel, CSV, TXT, or DOCX (deterministic extract first, "
                    "then one RAG-informed LLM compose when structure is ambiguous), pulls metadata from a "
                    "named reference DWG when asked, optionally writes a normalized workbook when requested, and plots the DWG. "
                    "Use when the user did NOT paste inline EmE/NmN traverse text. "
                    "Do NOT use for simple conventional Generate … prompts that already include coordinates/bearings. "
                    "Do NOT use for spreadsheet read/extract/deeper-search-only turns."
                ),
                func=cadastral_compose_and_plot,
                args_schema=CadastralComposeAndPlotInput,
            ),
            StructuredTool(
                name="cadastral_generate_from_prompt",
                description=(
                    "Plot a cadastral plan from a FULLY COMPOSED prompt that already contains real "
                    "coordinates for the points as (EmE, NmN) and/or bearing/distance legs. "
                    "Use after you read Excel/CSV/TXT/DOCX / a reference DWG and assembled the canonical cadastral fields. "
                    "Never pass 'extract coordinates from excel/csv/file' without numeric EmE/NmN values."
                ),
                func=cadastral_generate_from_prompt,
                args_schema=CadastralGenerateFromPromptInput,
            ),
            # Document processing tools (atomic, AI-driven)
            StructuredTool(
                name="document_get_metadata",
                description=(
                    "Get document metadata (file info, structure, table presence). "
                    "Use this first to understand document structure before extraction."
                ),
                func=document_get_metadata,
                args_schema=DocumentMetadataInput
            ),
            StructuredTool(
                name="document_get_text",
                description=(
                    "⚠️⚠️⚠️ DO NOT USE FOR LARGE DOCUMENTS! ⚠️⚠️⚠️\n"
                    "This tool AUTOMATICALLY FAILS for documents >50 pages or >25K words.\n"
                    "MANDATORY: Call document_get_resource_estimation(file_path) FIRST for ALL documents.\n"
                    "If document is large, use document_extract_sections_by_keywords() instead.\n"
                    "This tool ONLY works for small documents (<50 pages, <25K words).\n"
                    "If you truly need full extraction for a large document, use document_get_text_force after user confirmation."
                ),
                func=document_get_text,
                args_schema=DocumentTextInput
            ),
            StructuredTool(
                name="document_get_text_force",
                description=(
                    "FORCE full document text extraction, even for large documents. "
                    "Use ONLY after the user explicitly confirms they want full extraction "
                    "(it can be slow/expensive and may exceed limits in multi-step workflows)."
                ),
                func=document_get_text_force,
                args_schema=DocumentTextInput
            ),
            StructuredTool(
                name="document_get_tables",
                description=(
                    "Extract all tables from the document as structured data. "
                    "Use when document contains tabular data (feature lists, measurements, etc.)."
                ),
                func=document_get_tables,
                args_schema=DocumentTablesInput
            ),
            StructuredTool(
                name="document_get_section",
                description=(
                    "Extract text from a specific section (e.g., 'Signature', 'Summary', 'Findings'). "
                    "Use to extract specific document sections by title or keywords."
                ),
                func=document_get_section,
                args_schema=DocumentSectionInput
            ),
            StructuredTool(
                name="document_search_text",
                description=(
                    "Search for specific text patterns in the document. "
                    "Use to find dates, names, keywords, or any specific information. Supports regex."
                ),
                func=document_search_text,
                args_schema=DocumentSearchInput
            ),
            StructuredTool(
                name="document_extract_structured_data",
                description=(
                    "Extract structured data types: dates, names, numbers, emails, coordinates, depths. "
                    "Use to quickly extract common data types. Specify data_types as list or use 'all'."
                ),
                func=document_extract_structured_data,
                args_schema=DocumentStructuredDataInput
            ),
            StructuredTool(
                name="document_create_word",
                description=(
                    "Create a new Word document (.docx) with specified content. "
                    "CRITICAL: Use this IMMEDIATELY when user asks to 'save', 'export', or 'create' a document file. "
                    "When user says 'save as [filename]' or 'export as [filename]', use this tool right away - do not ask for confirmation again. "
                    "CONTEXT RULE: Use ONLY the data you JUST extracted and displayed in YOUR CURRENT RESPONSE - NEVER use data from previous conversations or different documents. "
                    "The 'content' parameter must be the summary/data you just showed the user above."
                ),
                func=document_create_word,
                args_schema=DocumentCreateInput
            ),
            StructuredTool(
                name="document_create_structured_word",
                description=(
                    "Create a Word document from structured data with sections, headings, and tables. "
                    "Use for professional reports. When user asks to save/export a structured document, use this."
                ),
                func=document_create_structured_word,
                args_schema=DocumentCreateStructuredInput
            ),
            StructuredTool(
                name="document_read_word",
                description=(
                    "Read an existing Word document to get its content. "
                    "CRITICAL: Use this PROACTIVELY when user asks to modify/update a document you JUST created. "
                    "Remember the file path from your previous response where you said 'saved as [path]'. "
                    "Don't ask user for the path - you already know it from conversation context. "
                    "Workflow: User says 'make it shorter' → Use this tool with the path you just used → Condense → document_update_word"
                ),
                func=document_read_word,
                args_schema=DocumentReadInput
            ),
            StructuredTool(
                name="document_update_word",
                description=(
                    "Update an existing Word document with new content. "
                    "CRITICAL: Use this PROACTIVELY when user asks to modify/update/shorten a document you JUST created. "
                    "Use the same file_path from when you created it (it's in your previous response). "
                    "Don't ask user for path or confirmation - use the path you already know. "
                    "Workflow: After document_read_word → Process content → Use this tool with same path and new content"
                ),
                func=document_update_word,
                args_schema=DocumentUpdateInput
            ),
            StructuredTool(
                name="document_get_structure",
                description=(
                    "Analyze document structure (headings, sections, organization). "
                    "CRITICAL FOR LARGE DOCUMENTS: ALWAYS call this FIRST for documents >100 pages or >50K words. "
                    "Use this to understand document organization before extracting text. "
                    "Then use document_extract_sections_by_keywords to extract only relevant sections."
                ),
                func=document_get_structure,
                args_schema=DocumentStructureInput
            ),
            StructuredTool(
                name="document_get_resource_estimation",
                description=(
                    "Estimate resource requirements and costs for processing a document. "
                    "CRITICAL FOR LARGE DOCUMENTS: ALWAYS call this FIRST when user requests processing a document. "
                    "Shows file size, estimated tokens, cost, warnings, and recommendations. "
                    "Use this to inform the user before processing large documents."
                ),
                func=document_get_resource_estimation,
                args_schema=DocumentResourceEstimationInput
            ),
            StructuredTool(
                name="document_extract_sections_by_keywords",
                description=(
                    "Extract only document sections matching specific keywords. "
                    "CRITICAL FOR LARGE DOCUMENTS: Use this instead of document_get_text for documents >50 pages. "
                    "Extracts only relevant sections, saving tokens and costs. Much faster than processing entire document. "
                    "Workflow: document_get_resource_estimation → document_get_structure → this tool → process extracted sections."
                ),
                func=document_extract_sections_by_keywords,
                args_schema=DocumentExtractSectionsInput
            ),
            StructuredTool(
                name="coordinate_converter",
                description=(
                    "Convert coordinates between reference systems using pyproj (default). "
                    "Supports WGS84, UTM zones, State Plane, Nigerian coordinate systems (Minna NTM, etc.), "
                    "and many other CRS from EPSG database. "
                    "Uses pyproj by default for fast, reliable conversions. "
                    "Only use Geographic Calculator COM if user explicitly requests it in their query "
                    "(set use_geographic_calculator=True). Always falls back to pyproj if COM is unavailable."
                ),
                func=coordinate_convert,
                args_schema=CoordConvertInput
            ),
            StructuredTool(
                name="coordinate_converter_auto",
                description=(
                    "Survey-aware coordinate conversion from FREE-FORM TEXT. "
                    "Use this when coordinates may contain degrees/minutes/seconds (DMS), hemisphere letters "
                    "(N/S/E/W), or projected labels (Easting/Northing, X/Y). "
                    "Automatically extracts coordinates, converts DMS/DM to decimal degrees, infers CRS hints "
                    "from text (e.g., 'from WGS84 to UTM Zone 32N', 'EPSG:4326 to EPSG:32632'), then converts "
                    "using pyproj by default (or Geographic Calculator COM if explicitly requested)."
                ),
                func=coordinate_convert_auto,
                args_schema=CoordConvertAutoInput,
            ),
            StructuredTool(
                name="excel_coordinate_converter",
                description=(
                    "Convert coordinates in an Excel file using pyproj (default). "
                    "Reads coordinates from specified columns, converts them using pyproj, "
                    "and saves results with converted coordinates as new columns. "
                    "Supports specialized coordinate systems like Nigerian NTM (Minna MidBelt / EPSG:26392, etc.). "
                    "Uses pyproj by default for fast, reliable conversions. "
                    "If excel_inspect_workbook reports ownership.family_blocks / Unnamed columns, call "
                    "excel_normalize_ownership_workbook first (or pass the raw file — SurvyAI will auto-normalize "
                    "on 'no valid coordinates' and retry with Easting/Northing). "
                    "For GIS follow-on (polygons/buffers/overlaps), convert the normalized/headed workbook, "
                    "keep the Owner column, then use geopandas_execute or arcgis_execute_python_code on the "
                    "converted file — never use arcgis_fill_volume_* for polygon/buffer/overlap work. "
                    "Only use Geographic Calculator COM if user explicitly requests it "
                    "(set use_geographic_calculator=True). "
                    "Parameters: excel_path (required), x_column (default: 'X'; use 'Easting' after normalize), "
                    "y_column (default: 'Y'; use 'Northing' after normalize), "
                    "source_crs, target_crs, source_zone, target_zone (optional), output_path (optional), "
                    "user_request (optional; used to infer save-as output name when output_path omitted), "
                    "sheet_name (optional), use_geographic_calculator (default: False)."
                ),
                func=excel_coordinate_convert,
                args_schema=ExcelCoordConvertInput
            ),
            StructuredTool(
                name="excel_convert_and_area",
                description=(
                    "One-shot Excel workflow for surveyors: reads an Excel file, auto-handles DMS Lat/Long (text), "
                    "converts coordinates from source CRS to target CRS, saves output in the same folder, and computes "
                    "area using the best available method (geodesic if geodetic/WGS84-like, otherwise planar). "
                    "Use this instead of asking the user to rename headers or pre-clean DMS."
                ),
                func=excel_convert_and_area,
                args_schema=ExcelConvertAndAreaInput,
            ),
        ]
        
        # ==================================================================
        # GEOGRAPHIC CALCULATOR CLI TOOLS
        # ==================================================================
        # Always register the check tool so the agent can verify availability
        # The execute tool is only registered if CLI is available
        
        # --- Tool: Check Geographic Calculator Availability ---
        # This tool should ALWAYS be available, even if CLI is not installed
        class GeoCalcCheckInput(BaseModel):
            """Input schema for checking Geographic Calculator availability (no parameters)."""
            pass
        
        def geocalc_check_availability() -> str:
            """
            Check if Geographic Calculator CLI is available on the system.
            
            Returns information about the installation status, version, and executable path.
            This is a read-only check that doesn't require user permission.
            """
            # Re-scan on every check so the agent can pick up installs/changes immediately.
            try:
                self.geocalc_cli.refresh()
            except Exception:
                pass

            # Check COM interface availability (optional method)
            # NOTE: We do NOT auto-connect here; we only report current state.
            com_available = self.blue_marble.is_available
            com_status = "✓ Available" if com_available else "✗ Not available"
            
            if not self.geocalc_cli.is_available:
                # Check if GUI is installed (even if CLI isn't found)
                from pathlib import Path
                gui_path = Path(r"C:\Program Files\Blue Marble Geo\Geographic Calculator\Geographic Calculator.exe")
                if gui_path.exists():
                    return (
                        f"Geographic Calculator Status:\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"GUI Installation: ✓ Found at {gui_path}\n"
                        f"COM Interface: {com_status}\n"
                        f"CLI Component: ✗ Not found\n\n"
                        f"RECOMMENDATION: Use pyproj by default; COM is optional\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"SurvyAI uses pyproj as the main conversion method (fast, reliable, no external COM).\n"
                        f"If your Geographic Calculator automation is installed/registered, COM can be used\n"
                        f"when explicitly requested for specialized workflows.\n\n"
                        f"Available Tools:\n"
                        f"  • coordinate_converter - Convert individual coordinates (pyproj default)\n"
                        f"  • coordinate_converter_auto - Parse DMS/text coords then convert (pyproj default)\n"
                        f"  • excel_coordinate_converter - Batch convert Excel (pyproj default, auto DMS)\n"
                        f"  • excel_convert_and_area - Convert Excel + compute area (best method)\n\n"
                        f"To use COM interface (optional):\n"
                        f"  - Re-run the Geographic Calculator installer and enable Automation/COM if available,\n"
                        f"    then request COM explicitly with use_geographic_calculator=True.\n\n"
                        f"Note: CLI component (GeographicCalculatorCMD.exe) is optional and\n"
                        f"not required for coordinate conversions. pyproj is sufficient."
                    )
                
                return (
                    "Geographic Calculator CLI (GeographicCalculatorCMD.exe) is not found on this system.\n\n"
                    "The system performed a thorough search but could not locate the executable.\n"
                    "Checks include: environment variable (GEOGRAPHIC_CALCULATOR_CMD_PATH), settings/.env, "
                    "Windows 'App Paths' registry, vendor registry keys, Uninstall registry entries, PATH/where.exe, "
                    "common install locations, and a targeted Program Files scan.\n\n"
                    "If Geographic Calculator is installed in a non-standard location, set the full path via "
                    "GEOGRAPHIC_CALCULATOR_CMD_PATH (or add its folder to PATH) and re-run this check."
                )
            
            version = self.geocalc_cli.get_version()
            cmd_path = self.geocalc_cli.cmd_path
            
            info = {
                "available": True,
                "status": "Geographic Calculator CLI is installed and ready to use",
                "executable_path": str(cmd_path) if cmd_path else "Unknown",
                "version": version if version else "Unknown (could not determine version)",
            }
            
            # Format the response nicely
            response = (
                "✓ Geographic Calculator CLI is available on your system!\n\n"
                f"Executable Path: {info['executable_path']}\n"
                f"Version: {info['version']}\n\n"
                "You can now execute Geographic Calculator jobs using the "
                "geographic_calculator_execute_job tool. Job files (.gpj, .gpp, .gpw) must be "
                "created in Geographic Calculator GUI first."
            )
            return response
        
        # Always add the check tool
        tools.append(
            StructuredTool(
                name="geographic_calculator_check",
                description=(
                    "MANDATORY TOOL FOR GEOGRAPHIC CALCULATOR QUERIES: "
                    "Check if Geographic Calculator CLI (GeographicCalculatorCMD.exe) is installed on the system. "
                    "Returns installation status, version information, and executable path if available. "
                    "CRITICAL INSTRUCTIONS: "
                    "1. When user asks about Geographic Calculator availability, installation status, or file path, "
                    "   you MUST call this tool IMMEDIATELY - do NOT ask for permission, do NOT provide menus, do NOT ask for more information. "
                    "2. This is a read-only check that does not access or modify files - no permission needed. "
                    "3. If user says 'yes' or grants permission after you ask, IMMEDIATELY call this tool - do NOT provide menus, do NOT ask for more info, do NOT give unrelated responses. "
                    "4. Just call this tool right away when asked about Geographic Calculator availability."
                ),
                func=geocalc_check_availability,
                args_schema=GeoCalcCheckInput
            )
        )
        
        if self.geocalc_cli.is_available:
            # --- Tool: Execute Geographic Calculator Job ---
            class GeoCalcJobInput(BaseModel):
                """Input schema for executing Geographic Calculator jobs."""
                job_path: str = Field(
                    description="Path to Geographic Calculator job/project/workspace file (.gpj, .gpp, or .gpw)"
                )
                close_after_done: bool = Field(
                    True,
                    description="Close Geographic Calculator after execution completes"
                )
                continue_after_error: bool = Field(
                    False,
                    description="Continue processing even if errors occur"
                )
            
            def geocalc_execute_job(
                job_path: str,
                close_after_done: bool = True,
                continue_after_error: bool = False
            ) -> str:
                """
                Execute a Geographic Calculator job, project, or workspace file.
                
                This tool runs pre-configured Geographic Calculator operations
                such as coordinate conversions, transformations, and batch processing.
                The job file must be created in Geographic Calculator GUI first.
                """
                result = self.geocalc_cli.execute_job(
                    job_path=job_path,
                    close_after_done=close_after_done,
                    continue_after_error=continue_after_error
                )
                return str(result)
            
            # Add Geographic Calculator execute job tool (only if CLI is available)
            tools.append(
                StructuredTool(
                    name="geographic_calculator_execute_job",
                    description=(
                        "Execute a Geographic Calculator job, project, or workspace file via command-line. "
                        "Use this when the user wants to run pre-configured Geographic Calculator "
                        "operations like coordinate conversions, transformations, or batch processing. "
                        "The job file (.gpj, .gpp, or .gpw) must be created in Geographic Calculator GUI first. "
                        "This tool executes GeographicCalculatorCMD.exe with the specified job file. "
                        "Parameters: job_path (required), close_after_done (default: true), continue_after_error (default: false)."
                    ),
                    func=geocalc_execute_job,
                    args_schema=GeoCalcJobInput
                )
            )
            
            logger.info("✓ Geographic Calculator execute job tool registered")
        else:
            logger.info("⚠ Geographic Calculator CLI not available - execute job tool not registered (check tool is always available)")
        
        logger.info("✓ Geographic Calculator check tool registered (always available)")
        
        # ==================================================================
        # VECTOR STORE TOOLS (if available)
        # ==================================================================
        
        if self.vector_store is not None:
            # --- Vector Search Tool ---
            class VectorSearchInput(BaseModel):
                """Input schema for vector search."""
                query: str = Field(
                    description="Search query text for semantic search"
                )
                collection: str = Field(
                    default="documents",
                    description="Collection to search: documents, drawings, or coordinates"
                )
                top_k: int = Field(
                    default=5,
                    description="Number of results to return (1-20)"
                )
            
            def vector_search(query: str, collection: str = "documents", top_k: int = 5) -> str:
                """
                Search for relevant documents using semantic similarity.
                
                Finds documents, drawings, or coordinates that are semantically
                similar to the query text.
                """
                try:
                    results = self.vector_store.search(
                        query=query,
                        collection=collection,
                        top_k=min(top_k, 20)
                    )
                    if not results:
                        return "No matching documents found."
                    
                    output = []
                    for i, r in enumerate(results, 1):
                        score = r.get('score', 0)
                        content = r.get('content', '')[:500]  # Truncate long content
                        metadata = r.get('metadata', {})
                        output.append(
                            f"{i}. [Score: {score:.2f}]\n"
                            f"   Content: {content}\n"
                            f"   Metadata: {metadata}"
                        )
                    return "\n\n".join(output)
                except Exception as e:
                    return f"Search error: {str(e)}"
            
            # --- Vector Store Document Tool ---
            class VectorStoreInput(BaseModel):
                """Input schema for storing documents."""
                content: str = Field(
                    description="Text content to store in the vector database"
                )
                collection: str = Field(
                    default="documents",
                    description="Collection to store in: documents, drawings, or coordinates"
                )
                source: str = Field(
                    default="",
                    description="Source of the content (file name, etc.)"
                )
            
            def vector_store_document(
                content: str, 
                collection: str = "documents",
                source: str = ""
            ) -> str:
                """
                Store a document in the vector database for later retrieval.
                
                Useful for saving extracted text, survey data, or any content
                that should be searchable later.
                """
                try:
                    metadata = {"source": source} if source else {}
                    doc_id = self.vector_store.add_text(
                        text=content,
                        metadata=metadata,
                        collection=collection
                    )
                    return f"✓ Document stored successfully (ID: {doc_id})"
                except Exception as e:
                    return f"Storage error: {str(e)}"
            
            # --- Vector Store Stats Tool ---
            class VectorStoreStatsInput(BaseModel):
                """Input schema for vector store stats (no parameters)."""
                pass
            
            def vector_store_stats() -> str:
                """Get statistics about the vector database."""
                try:
                    stats = self.vector_store.get_stats()
                    return (
                        f"Vector Store Statistics:\n"
                        f"  - Provider: {stats['embedding_provider']}\n"
                        f"  - Model: {stats['embedding_model']}\n"
                        f"  - Dimension: {stats['embedding_dimension']}\n"
                        f"  - Total Documents: {stats['total_documents']}\n"
                        f"  - Collections:\n" +
                        "\n".join(f"    - {k}: {v} docs" for k, v in stats['collections'].items())
                    )
                except Exception as e:
                    return f"Error getting stats: {str(e)}"
            
            # Add vector store tools
            tools.extend([
                StructuredTool(
                    name="semantic_search",
                    description=(
                        "Search for relevant documents using semantic similarity. "
                        "Use this to find previously stored information about surveys, "
                        "drawings, coordinates, or any text content. "
                        "Collections: documents, drawings, coordinates."
                    ),
                    func=vector_search,
                    args_schema=VectorSearchInput
                ),
                StructuredTool(
                    name="store_document",
                    description=(
                        "Store text content in the vector database for later retrieval. "
                        "Use this to save extracted text, survey data, or important information "
                        "that should be searchable in future queries."
                    ),
                    func=vector_store_document,
                    args_schema=VectorStoreInput
                ),
                StructuredTool(
                    name="vector_store_stats",
                    description="Get statistics about the vector database (document counts, etc.).",
                    func=vector_store_stats,
                    args_schema=VectorStoreStatsInput
                ),
            ])
            
            logger.info(f"✓ Added {3} vector store tools")

        # ==================================================================
        # GEOPANDAS DYNAMIC GIS EXECUTION TOOL
        # ==================================================================
        # Always available (no ArcGIS licence required).
        # Used for: spatial join, point-in-polygon, buffer, clip, dissolve,
        #           attribute export to Excel/CSV, any ad-hoc vector analysis.

        class GeoPandasExecuteInput(BaseModel):
            """Schema for the geopandas_execute tool."""
            code: str = Field(
                description=(
                    "Complete Python script using GeoPandas / Shapely / pandas / ezdxf. "
                    "The following helpers are pre-injected — call them directly:\n"
                    "  read_csv_points(path, e_col=None, n_col=None, crs=None) → GeoDataFrame\n"
                    "  read_dwg_polygons(dwg_path, layer_filter=None, crs=None) → GeoDataFrame\n"
                    "  read_shapefile_or_geojson(path, crs=None) → GeoDataFrame\n"
                    "  points_within_polygon(points_gdf, polygon_gdf) → GeoDataFrame\n"
                    "  merge_point_attributes(points_gdf, polygon_gdf) → GeoDataFrame\n"
                    "  export_to_excel(gdf, output_path, sheet_name='Results') → str\n"
                    "  export_to_csv(gdf, output_path) → str\n"
                    "  export_to_shapefile(gdf, output_path) → str\n"
                    "  result_log(key, value) — emit RESULT_KEY: value lines for structured output.\n"
                    "Always call result_log for key metrics (row counts, output file paths)."
                )
            )
            description: str = Field(
                description="One-sentence description of what this script does (for audit log)."
            )
            working_dir: Optional[str] = Field(
                default=None,
                description=(
                    "Working directory for the script (default: same folder as the first input file, "
                    "or the SurvyAI workspace). Scripts are saved here for audit."
                )
            )
            expected_output_files: Optional[List[str]] = Field(
                default=None,
                description=(
                    "List of output file paths the script should create. "
                    "The tool reports success only if all listed files exist after execution."
                )
            )

        def geopandas_execute(
            code: str,
            description: str,
            working_dir: Optional[str] = None,
            expected_output_files: Optional[List[str]] = None,
        ) -> str:
            """Execute dynamic GeoPandas GIS analysis code in a subprocess."""
            # Infer working dir from first expected output if not given
            if not working_dir and expected_output_files:
                try:
                    working_dir = str(Path(expected_output_files[0]).parent.resolve())
                except Exception:
                    pass
            result = self.geopandas_executor.execute_script(
                code=code,
                script_name=description[:50] if description else None,
                working_dir=working_dir,
                expected_output_files=expected_output_files,
            )
            return self.geopandas_executor.format_result(result)

        tools.append(
            StructuredTool(
                name="geopandas_execute",
                description=(
                    "Execute arbitrary GeoPandas / Shapely / ezdxf Python code for dynamic GIS analysis "
                    "WITHOUT requiring ArcGIS Pro. Use this for: spatial join, point-in-polygon selection, "
                    "buffer, clip, dissolve, intersect, union, attribute filtering, coordinate transformation, "
                    "reading DWG/DXF polygons, reading CSV/Excel points, and exporting results to Excel/CSV/shapefile. "
                    "Pre-injected helpers handle DWG polygon reading, CSV point loading, spatial join, and Excel export — "
                    "call them directly in your code. Always emit result_log() lines for key metrics. "
                    "WHEN TO USE: prefer this over arcgis_execute_python_code when (a) visualization in ArcGIS Pro is "
                    "not needed, (b) the task is purely vector analysis (join, filter, select, export), or "
                    "(c) faster execution without ArcGIS startup overhead is preferred. "
                    "Use arcgis_execute_python_code when raster operations (IDW, CutFill, TIN), "
                    "ArcGIS-specific outputs, or map visualization are required."
                ),
                func=geopandas_execute,
                args_schema=GeoPandasExecuteInput,
            )
        )
        logger.info("✓ Added geopandas_execute tool (dynamic GIS analysis without ArcGIS)")

        # ==================================================================
        # ARCGIS PRO TOOLS
        # ==================================================================
        
        # Add ArcGIS tools if ArcGIS Pro is installed
        if self.arcgis_processor.is_installed or self.arcgis_processor.is_available:

            # --- Tool: filesystem stat/exists (verification gate) ---
            class FilesystemStatInput(BaseModel):
                """Input schema for checking file existence and size."""
                paths: List[str] = Field(description="List of file/folder paths to check")

            def filesystem_stat(paths: List[str]) -> str:
                from pathlib import Path
                out = []
                for p in paths or []:
                    try:
                        pp = Path(p)
                        exists = pp.exists()
                        info = {"path": str(pp), "exists": exists}
                        if exists:
                            try:
                                st = pp.stat()
                                info.update({"is_dir": pp.is_dir(), "size_bytes": st.st_size, "mtime": st.st_mtime})
                            except Exception:
                                pass
                        out.append(info)
                    except Exception as e:
                        out.append({"path": str(p), "exists": False, "error": str(e)})
                return json.dumps({"items": out}, indent=2, ensure_ascii=False)
            
            # --- Tool: Launch ArcGIS Pro ---
            class ArcGISLaunchInput(BaseModel):
                """Input schema for launching ArcGIS Pro."""
                pass
            
            def arcgis_launch() -> str:
                """
                Launch ArcGIS Pro application.
                Opens ArcGIS Pro on the user's computer.
                """
                result = self.arcgis_processor.launch_arcgis_pro()
                return str(result)
            
            # --- Tool: Create Project ---
            class ArcGISCreateProjectInput(BaseModel):
                """Input schema for creating an ArcGIS Pro project."""
                project_name: str = Field(
                    description="Name of the project (without .aprx extension)"
                )
                project_path: Optional[str] = Field(
                    None,
                    description="Directory to save the project (default: Documents/ArcGIS/Projects)"
                )
                coordinate_system: Optional[str] = Field(
                    None,
                    description=(
                        "Coordinate system for the project. "
                        "Examples: 'WGS84', 'UTM Zone 32N', 'EPSG:4326', '32632'"
                    )
                )
                template: str = Field(
                    default="MAP",
                    description="Project template: MAP, CATALOG, GLOBAL_SCENE, or LOCAL_SCENE"
                )
            
            def arcgis_create_project(
                project_name: str,
                project_path: Optional[str] = None,
                coordinate_system: Optional[str] = None,
                template: str = "MAP"
            ) -> str:
                """
                Create a new ArcGIS Pro project with specified settings.
                
                Can set the coordinate system for the map (e.g., UTM Zone 32N).
                """
                # Use default coordinate system if not specified
                if not coordinate_system:
                    coordinate_system = getattr(
                        self.settings, 
                        'arcgis_default_coordinate_system', 
                        None
                    )
                
                # Use default project path if not specified
                if not project_path:
                    project_path = getattr(
                        self.settings,
                        'arcgis_default_project_path',
                        None
                    ) or None
                
                result = self.arcgis_processor.create_project(
                    project_name=project_name,
                    project_path=project_path,
                    coordinate_system=coordinate_system,
                    template=template
                )
                return str(result)
            
            # --- Tool: Open Project ---
            class ArcGISOpenProjectInput(BaseModel):
                """Input schema for opening an ArcGIS Pro project."""
                project_path: str = Field(
                    description="Path to the .aprx project file"
                )
            
            def arcgis_open_project(project_path: str) -> str:
                """
                Open an existing ArcGIS Pro project.
                """
                result = self.arcgis_processor.open_project(project_path)
                return str(result)
            
            # --- Tool: Set Coordinate System ---
            class ArcGISSetCRSInput(BaseModel):
                """Input schema for setting coordinate system."""
                coordinate_system: str = Field(
                    description=(
                        "Coordinate system to set. "
                        "Examples: 'WGS84', 'UTM Zone 32N', 'EPSG:4326', '32632', 'British National Grid'"
                    )
                )
                map_name: Optional[str] = Field(
                    None,
                    description="Name of the map to modify (default: first map)"
                )
            
            def arcgis_set_coordinate_system(
                coordinate_system: str,
                map_name: Optional[str] = None
            ) -> str:
                """
                Set the coordinate system for a map in the current project.
                """
                result = self.arcgis_processor.set_map_coordinate_system(
                    coordinate_system=coordinate_system,
                    map_name=map_name
                )
                return str(result)
            
            # --- Tool: Get Project Info ---
            class ArcGISProjectInfoInput(BaseModel):
                """Input schema for getting project info."""
                project_path: Optional[str] = Field(
                    None,
                    description=(
                        "Optional full path to a .aprx project. Prefer this over relying on "
                        "an in-process CURRENT project. If local arcpy is unavailable, do not "
                        "stop — use arcgis_execute_python_code or geopandas_execute with "
                        "session parcel paths instead."
                    ),
                )
            
            def arcgis_get_project_info(project_path: Optional[str] = None) -> str:
                """
                Get information about an ArcGIS Pro project.

                Prefer passing project_path (.aprx). Do NOT treat a local-process
                'arcpy not available' result as a hard stop — for analysis use
                arcgis_execute_python_code (Pro's Python) or geopandas_execute on
                verified .shp/.xlsx/.gdb paths from the session instead.
                """
                result = self.arcgis_processor.get_project_info(project_path=project_path)
                if isinstance(result, dict) and not result.get("success"):
                    err = str(result.get("error") or "")
                    if "arcpy" in err.lower() or "not available" in err.lower():
                        result = dict(result)
                        result["guidance"] = (
                            "Local SurvyAI missing arcpy is normal and NOT a blocker. "
                            "Do not ask the user for paths again if .aprx/.shp/.gdb/.xlsx "
                            "are already in this conversation. Proceed with "
                            "arcgis_execute_python_code(project_path=...) or "
                            "geopandas_execute on the parcel shapefile/Excel paths."
                        )
                        result["next_tools"] = [
                            "arcgis_execute_python_code",
                            "geopandas_execute",
                        ]
                return str(result)
            
            # --- Tool: List Coordinate Systems ---
            class ArcGISListCRSInput(BaseModel):
                """Input schema for listing coordinate systems."""
                filter_text: Optional[str] = Field(
                    None,
                    description="Optional text to filter coordinate systems (e.g., 'UTM', 'WGS')"
                )
            
            def arcgis_list_coordinate_systems(filter_text: Optional[str] = None) -> str:
                """
                List available coordinate systems with their WKID codes.
                Useful for finding the correct coordinate system name.
                """
                result = self.arcgis_processor.list_coordinate_systems(filter_text)
                return str(result)

            # --- Tool: Import XY Points from Excel ---
            class ArcGISImportXYPointsInput(BaseModel):
                """Input schema for importing XY points from an Excel file."""
                project_path: str = Field(
                    description="Path to the .aprx project file to import points into"
                )
                excel_path: str = Field(
                    description="Path to the Excel file containing coordinates"
                )
                x_field: str = Field(
                    description="Name of the X/Easting field in the Excel file (e.g., 'Long._converted' or 'Easting')"
                )
                y_field: str = Field(
                    description="Name of the Y/Northing field in the Excel file (e.g., 'Lat._converted' or 'Northing')"
                )
                coordinate_system: str = Field(
                    description=(
                        "Coordinate system of the X/Y fields. "
                        "Examples: 'Minna / Nigeria Mid Belt', 'UTM Zone 32N', 'EPSG:26392'"
                    )
                )
                sheet_name: Optional[str] = Field(
                    None,
                    description="Optional Excel sheet name (default: first sheet)"
                )
                layer_name: Optional[str] = Field(
                    None,
                    description="Optional output layer/feature class name (default: <project>_points)"
                )

            def arcgis_import_xy_points(
                project_path: str,
                excel_path: str,
                x_field: str,
                y_field: str,
                coordinate_system: str,
                sheet_name: Optional[str] = None,
                layer_name: Optional[str] = None,
            ) -> str:
                """
                Import XY points from an Excel file into an ArcGIS Pro project.

                Creates a point feature class in a file geodatabase within the project folder,
                adds it to the map, and saves the project.
                """
                result = self.arcgis_processor.import_xy_points_from_excel(
                    project_path=project_path,
                    excel_path=excel_path,
                    x_field=x_field,
                    y_field=y_field,
                    coordinate_system=coordinate_system,
                    sheet_name=sheet_name,
                    layer_name=layer_name,
                )
                return str(result)

            # --- Tool: Verified workflow from Excel -> points -> hull -> traverse -> CSV ---
            class ArcGISExcelHullTraverseInput(BaseModel):
                """Input schema for a verified Excel->ArcGIS hull+traverse workflow."""
                excel_path: str = Field(description="Path to the Excel file containing coordinates")
                project_name: str = Field(description="ArcGIS project name (folder + .aprx will be created)")
                project_folder: Optional[str] = Field(
                    None,
                    description=(
                        "Folder to create the project in. "
                        "CRITICAL: If not specified, automatically uses the same folder as excel_path. "
                        "This ensures projects are created alongside input files when user doesn't specify location."
                    )
                )
                coordinate_system: str = Field(
                    description="Coordinate system for the imported points (e.g., 'Minna / Nigeria Mid Belt', 'EPSG:26392')"
                )
                output_csv: str = Field(
                    description=(
                        "Output CSV path. Can be absolute path or just filename. "
                        "CRITICAL: If just a filename (not absolute), automatically saved in same folder as excel_path. "
                        "This ensures outputs are created alongside input files when user doesn't specify location."
                    )
                )
                sheet_name: Optional[str] = Field(None, description="Optional Excel sheet name (default: first)")
                close_traverse: bool = Field(True, description="If True, closes traverse last->first")
                clean_project_layers: bool = Field(
                    True,
                    description="If True, removes template/sample layers so only user data appears"
                )

            def arcgis_excel_hull_traverse(**kwargs) -> str:
                """
                VERIFIED ArcGIS workflow:
                - Creates/opens a clean project in the requested folder
                - Imports points from Excel (prefers X/Y)
                - Creates convex hull + computes area
                - Computes traverse distances/bearings
                - Exports CSV and verifies it exists
                """
                # Auto-infer missing path parameters from input file location
                excel_path = kwargs.get("excel_path")
                if excel_path:
                    # If project_folder not specified, use same folder as Excel file
                    if not kwargs.get("project_folder"):
                        inferred_folder = self._infer_output_path_from_input(excel_path, output_type="folder")
                        if inferred_folder:
                            kwargs["project_folder"] = inferred_folder
                            logger.info(f"Auto-inferred project_folder from excel_path: {inferred_folder}")
                    
                    # If output_csv is just a filename (not absolute path), resolve to same folder as Excel
                    output_csv = kwargs.get("output_csv")
                    if output_csv:
                        from pathlib import Path
                        csv_path = Path(output_csv)
                        if not csv_path.is_absolute():
                            inferred_csv = self._infer_output_path_from_input(excel_path, output_filename=output_csv)
                            if inferred_csv:
                                kwargs["output_csv"] = inferred_csv
                                logger.info(f"Auto-inferred output_csv location: {inferred_csv}")
                
                res = self.arcgis_processor.excel_points_convex_hull_traverse(**kwargs)
                return json.dumps(res, indent=2, ensure_ascii=False)

            # --- Tool: Fill volume (IDW + Cut Fill) - hardened workflow ---
            class ArcGISFillVolumeIDWCutfillInput(BaseModel):
                """Input for verified fill-volume workflow: Excel -> IDW rasters -> Cut Fill -> volume + results_fill.xlsx."""
                excel_path: str = Field(description="Path to the Excel file (one sheet with X, Y, pre and post elevation columns)")
                sheet_name: str = Field(description="Exact sheet name (from excel_inspect_workbook)")
                x_field: str = Field(description="Easting/X column name (e.g. Eastings, EASTING)")
                y_field: str = Field(description="Northing/Y column name (e.g. Northings, NORTHING)")
                post_z_field: str = Field(description="Post-fill elevation column (e.g. 'post fill', Post)")
                pre_z_field: str = Field(description="Pre-fill elevation column (e.g. 'pre fill', Pre)")
                coordinate_system: str = Field(
                    default="Minna / Nigeria Mid Belt",
                    description="Coordinate system (e.g. Nigerian Mid-Belt, EPSG:26392)",
                )
                output_excel_path: Optional[str] = Field(
                    None,
                    description="Output Excel path (default: same folder as input, results_fill.xlsx)",
                )

            def arcgis_fill_volume_idw_cutfill(
                excel_path: str,
                sheet_name: str,
                x_field: str,
                y_field: str,
                post_z_field: str,
                pre_z_field: str,
                coordinate_system: str = "Minna / Nigeria Mid Belt",
                output_excel_path: Optional[str] = None,
            ) -> str:
                """
                VERIFIED fill-volume workflow: no ArcGISProject('CURRENT'), ExcelToTable uses 3rd positional sheet.
                Use this when the user asks for fill volume from Pre-fill/Post-fill data, IDW rasters, Cut Fill, metric, results_fill.xlsx.
                Call excel_inspect_workbook first to get sheet and column names, then call this with the resolved names.
                """
                res = self.arcgis_processor.compute_fill_volume_idw_cutfill(
                    excel_path=excel_path,
                    sheet_name=sheet_name,
                    x_field=x_field,
                    y_field=y_field,
                    post_z_field=post_z_field,
                    pre_z_field=pre_z_field,
                    coordinate_system=coordinate_system,
                    output_excel_path=output_excel_path,
                )
                return json.dumps(res, indent=2, ensure_ascii=False)

            # --- Tool: PRE/POST CSV (or Excel) + DWG boundary -> IDW -> CutFill (verified) ---
            class ArcGISPrePostCSVDWGCutfillInput(BaseModel):
                """Inputs for separate PRE/POST tabular files plus a DWG boundary -> IDW surfaces -> CutFill -> CSV."""

                pre_csv_path: str = Field(description="Path to PRE survey points (.csv or .xlsx with E, N, Z)")
                post_csv_path: str = Field(description="Path to POST survey points (.csv or .xlsx with E, N, Z)")
                boundary_dwg_path: str = Field(description="DWG with boundary polygon or closed polylines")
                workspace_folder: Optional[str] = Field(
                    None,
                    description="Folder for .aprx, GDB, and CSV copies (default: current workspace)",
                )
                output_csv_path: Optional[str] = Field(
                    None,
                    description="Volume/metrics CSV path (default: Adibawa_VolumeResult.csv in workspace)",
                )
                project_name: str = Field(
                    default="BorrowPit_Volume_Project",
                    description="ArcGIS Pro project base name (creates project_name/project_name.aprx)",
                )
                coordinate_system: Optional[str] = Field(
                    None,
                    description="Optional CRS (e.g. EPSG:26392, Minna / Nigeria Mid Belt); else inferred from DWG boundary",
                )

            def arcgis_pre_post_csv_dwg_cutfill(
                pre_csv_path: str,
                post_csv_path: str,
                boundary_dwg_path: str,
                workspace_folder: Optional[str] = None,
                output_csv_path: Optional[str] = None,
                project_name: str = "BorrowPit_Volume_Project",
                coordinate_system: Optional[str] = None,
            ) -> str:
                """
                VERIFIED workflow: two point files + DWG boundary -> points in GDB -> IDW (Z) clipped to boundary
                -> CutFill + dz raster -> metrics CSV. Adds boundary, points, rasters, and cutfill to the map,
                finalizes visualization, opens ArcGIS Pro.
                """
                res = self.arcgis_processor.compute_pre_post_csv_dwg_cutfill(
                    pre_csv_path=pre_csv_path,
                    post_csv_path=post_csv_path,
                    boundary_dwg_path=boundary_dwg_path,
                    workspace_folder=workspace_folder,
                    output_csv_path=output_csv_path,
                    project_name=project_name,
                    coordinate_system=coordinate_system,
                )
                return json.dumps(res, indent=2, ensure_ascii=False)

            # --- Tool: PRE/POST CSV + DWG -> CreateTin -> volume CSV (verified; IDW fallback on failure) ---
            class ArcGISPrePostCSVDWGTinVolumeInput(BaseModel):
                """TIN-based PRE/POST surfaces with DWG boundary; falls back to IDW workflow if CreateTin fails."""

                pre_csv_path: str = Field(description="Path to PRE survey points (.csv or .xlsx with E, N, Z)")
                post_csv_path: str = Field(description="Path to POST survey points (.csv or .xlsx with E, N, Z)")
                boundary_dwg_path: str = Field(description="DWG with boundary polygon or closed polylines")
                workspace_folder: Optional[str] = Field(
                    None,
                    description="Folder for .aprx, GDB, CSV copies (default: current workspace)",
                )
                output_csv_path: Optional[str] = Field(
                    None,
                    description="Volume CSV path (default: Adibawa_VolumeResult2.csv in workspace)",
                )
                project_name: str = Field(
                    default="Adibawa_TIN_Volume",
                    description="ArcGIS Pro project base name",
                )
                coordinate_system: Optional[str] = Field(
                    default="EPSG:26392",
                    description="Projected CRS for TIN (default EPSG:26392 Minna Mid Belt)",
                )
                cad_reference_scale: str = Field(
                    default="1000",
                    description="CADToGeodatabase reference scale (e.g. 1000); adjust if DWG import fails",
                )
                fallback_to_idw_on_failure: bool = Field(
                    default=True,
                    description="If True, run arcgis_pre_post_csv_dwg_cutfill when TIN/CreateTin fails",
                )

            def arcgis_pre_post_csv_dwg_tin_volume(
                pre_csv_path: str,
                post_csv_path: str,
                boundary_dwg_path: str,
                workspace_folder: Optional[str] = None,
                output_csv_path: Optional[str] = None,
                project_name: str = "Adibawa_TIN_Volume",
                coordinate_system: Optional[str] = "EPSG:26392",
                cad_reference_scale: str = "1000",
                fallback_to_idw_on_failure: bool = True,
            ) -> str:
                """
                VERIFIED TIN workflow (3D Analyst): FeatureTo3DByAttribute -> CreateTin (retry without clip)
                -> TinRaster -> dz -> zonal volume CSV. Requires Spatial Analyst + 3D Analyst.
                On failure, optionally runs the IDW CutFill workflow and writes the same output CSV path.
                """
                res = self.arcgis_processor.compute_pre_post_csv_dwg_tin_volume(
                    pre_csv_path=pre_csv_path,
                    post_csv_path=post_csv_path,
                    boundary_dwg_path=boundary_dwg_path,
                    workspace_folder=workspace_folder,
                    output_csv_path=output_csv_path,
                    project_name=project_name,
                    coordinate_system=coordinate_system,
                    cad_reference_scale=cad_reference_scale,
                    fallback_to_idw_on_failure=fallback_to_idw_on_failure,
                )
                return json.dumps(res, indent=2, ensure_ascii=False)

            # --- Tool: Execute Python Code ---
            class ArcGISExecutePythonCodeInput(BaseModel):
                """Input schema for executing dynamically generated Python/arcpy code."""
                python_code: str = Field(
                    description=(
                        "Complete Python/arcpy code for FULL AUTOMATION. Generate code that:\n"
                        "1. Performs the complete workflow without manual steps\n"
                        "2. Calculates and prints results in structured format for parsing\n"
                        "3. Handles field discovery, imports, analysis, and result extraction\n\n"
                        "CRITICAL PATTERNS:\n"
                        "FIELD DISCOVERY (prefer value-safe X/Y):\n"
                        "  fields = [f.name for f in arcpy.ListFields(table)]\n"
                        "  # prefer canonical X/Y if present; else fall back to first numeric-looking pair\n"
                        "  x_field = 'X' if 'X' in fields else next((f for f in fields if 'east' in f.lower() or 'lon' in f.lower()), None)\n"
                        "  y_field = 'Y' if 'Y' in fields else next((f for f in fields if 'north' in f.lower() or 'lat' in f.lower()), None)\n"
                        "RESULT OUTPUT: Print results with clear labels: print('RESULT_AREA:', area); print('RESULT_BEARING_1_2:', bearing)\n"
                        "AREA CALCULATION: Use arcpy.da.SearchCursor with SHAPE@ token: area = row[0].area\n"
                        "BEARING/DISTANCE: Use geometry methods: bearing = math.degrees(math.atan2(dy, dx)); distance = math.sqrt(dx**2 + dy**2)\n"
                        "POLYGON CREATION (convex hull): use MinimumBoundingGeometry with a valid group option for points:\n"
                        "  arcpy.management.MinimumBoundingGeometry(points_fc, polygon_fc, 'CONVEX_HULL', group_option='ALL')\n"
                        "TRAVERSE ANALYSIS: Loop through points, calculate bearing/distance between consecutive points\n\n"
                        "IMPORTANT (ArcGIS Excel field types): Avoid XYTableToPoint directly on ExcelToTable outputs because ArcGIS often imports numeric-looking columns as TEXT, causing ERROR 000308.\n"
                        "IMPORTANT (ArcGISProject context): Do NOT set aprx.activeMap or rely on UI-only properties unless you are using arcpy.mp.ArcGISProject('CURRENT') inside the ArcGIS Pro Python Window.\n"
                        "When running headless (propy.bat) or opening an .aprx by path, use aprx.listMaps()[0] and DO NOT attempt to activate maps/views.\n"
                        "IMPORTANT (Headless automation): Do NOT use arcpy.mp.ArcGISProject('CURRENT') in scripts intended to be executed automatically via propy.bat. Always open a project by explicit .aprx path (or create a new project by path).\n"
                        "Preferred: Create the point feature class yourself with CreateFeatureclass + InsertCursor and cast coordinates with float().\n"
                        "Example:\n"
                        "  arcpy.management.CreateFeatureclass(gdb, name, 'POINT', spatial_reference=sr)\n"
                        "  with arcpy.da.SearchCursor(excel_table, ['OID@', x_field, y_field]) as sc:\n"
                        "      with arcpy.da.InsertCursor(fc, ['SrcOID','SHAPE@XY']) as ic:\n"
                        "          for oid, x_raw, y_raw in sc:\n"
                        "              x=float(str(x_raw).replace(',',''))\n"
                        "              y=float(str(y_raw).replace(',',''))\n"
                        "              ic.insertRow((oid, (x,y)))\n\n"
                        "CODE STRUCTURE:\n"
                        "1. Import libraries (arcpy, math, os)\n"
                        "2. Open project, discover fields, import data\n"
                        "3. Create analysis features (polygons, lines)\n"
                        "4. Perform calculations (area, bearings, distances)\n"
                        "5. Print results with RESULT_ prefix for parsing\n"
                        "6. Save project\n\n"
                        "ZOOM/EXTENT (headless-safe): DO NOT rely on Layer.getExtent(). Prefer:\n"
                        "  ext = arcpy.Describe(points_fc).extent\n"
                        "  try: mp.defaultCamera.setExtent(ext)\n"
                        "  except: pass\n\n"
                        "EXAMPLE RESULT FORMAT:\n"
                        "print('RESULT_AREA:', 12345.67, 'square_meters')\n"
                        "print('RESULT_BEARING_P1_P2:', 45.5, 'degrees')\n"
                        "print('RESULT_DISTANCE_P1_P2:', 123.45, 'meters')"
                    )
                )
                project_path: Optional[str] = Field(
                    None,
                    description=(
                        "Path to an existing .aprx file. If omitted, SurvyAI auto-creates a project in the workspace "
                        "so geoprocessing runs headlessly and ArcGIS Pro can open afterward with layers loaded "
                        "(same end-to-end pattern as verified volume workflows)."
                    ),
                )
                workspace_folder: Optional[str] = Field(
                    None,
                    description=(
                        "When project_path is omitted: folder where the auto-created .aprx should live "
                        "(default: current SurvyAI workspace / cwd)."
                    ),
                )
                auto_project_name: Optional[str] = Field(
                    None,
                    description="When project_path is omitted: optional stem for the auto-created project name.",
                )
                coordinate_system: Optional[str] = Field(
                    None,
                    description=(
                        "When project_path is omitted: optional coordinate system for the auto-created project "
                        "(e.g. the user's stated CRS). Otherwise settings default may apply."
                    ),
                )
                script_name: Optional[str] = Field(
                    None,
                    description="Optional script filename (default: auto-generated timestamp-based name)"
                )
                execute_automatically: bool = Field(
                    True,
                    description=(
                        "If True, execute automatically. Generated ArcGIS workflows should run deterministically first "
                        "and ArcGIS Pro should open after outputs are ready for review. "
                        "If False, save the script and provide instructions."
                    )
                )
            
            def arcgis_execute_python_code(
                python_code: str,
                project_path: Optional[str] = None,
                workspace_folder: Optional[str] = None,
                auto_project_name: Optional[str] = None,
                coordinate_system: Optional[str] = None,
                script_name: Optional[str] = None,
                execute_automatically: bool = True,
            ) -> str:
                """
                Execute dynamically generated Python/arcpy code.
                
                This tool allows you to generate arcpy code on-the-fly based on user requests and execute it.
                Use this for complex, multi-step operations that require custom arcpy code generation.
                The code is automatically executed and saved to the project's scripts folder for reference.
                
                For operations like "import points and zoom to extent", generate complete arcpy code that:
                - Imports arcpy and opens the project
                - Performs the import operation
                - Gets the layer extent
                - Zooms to the extent using arcpy techniques
                - Saves the project
                
                For volumetrics (IDW, Cut Fill): use a **projected CRS** with known Z units; **add all outputs
                to the active map** (addDataFromPath) and **save the project** so layers appear when Pro opens.
                """
                result = self.arcgis_processor.execute_python_code(
                    python_code=python_code,
                    project_path=project_path,
                    script_name=script_name,
                    execute_automatically=execute_automatically,
                    workspace_folder=workspace_folder,
                    auto_project_name=auto_project_name,
                    coordinate_system=coordinate_system,
                )
                return json.dumps(result, ensure_ascii=True, default=str)
            
            # --- Tool: Finalize Project Visualization ---
            class ArcGISFinalizeVisualizationInput(BaseModel):
                """Input schema for finalizing project visualization."""
                project_path: str = Field(
                    description="Path to the .aprx project file to finalize"
                )
                load_basemap: bool = Field(
                    default=True,
                    description="If True, add 'Imagery Hybrid' basemap to all maps"
                )
                basemap_name: str = Field(
                    default="Imagery Hybrid",
                    description="Name of basemap to add (default: 'Imagery Hybrid')"
                )
                load_geodatabase: bool = Field(
                    default=True,
                    description="If True, load native geodatabase and all feature classes"
                )
            
            def arcgis_finalize_visualization(
                project_path: str,
                load_basemap: bool = True,
                basemap_name: str = "Imagery Hybrid",
                load_geodatabase: bool = True,
            ) -> str:
                """
                Finalize ArcGIS Pro project visualization after user operations complete.
                
                This function is called AFTER all user-requested operations have been executed
                to ensure the project is visually ready for inspection:
                - Adds 'Imagery Hybrid' basemap to all maps
                - Loads the native geodatabase (project_dir/project_name.gdb) and all its feature classes
                
                IMPORTANT: This should be called AFTER user operations complete, not during project creation,
                so users can visually verify that their instructions were properly carried out.
                """
                result = self.arcgis_processor.finalize_project_visualization(
                    project_path=project_path,
                    load_basemap=load_basemap,
                    basemap_name=basemap_name,
                    load_geodatabase=load_geodatabase,
                )
                return json.dumps(result, ensure_ascii=True, default=str)
            
            # Add ArcGIS tools to the list
            tools.extend([
                StructuredTool(
                    name="filesystem_stat",
                    description=(
                        "Verify whether files/folders exist and their sizes. "
                        "Use this to confirm outputs were actually created before claiming success."
                    ),
                    func=filesystem_stat,
                    args_schema=FilesystemStatInput,
                ),
                StructuredTool(
                    name="arcgis_launch",
                    description=(
                        "Launch ArcGIS Pro. For automated geoprocessing, prefer arcgis_execute_python_code or a "
                        "verified ArcGIS tool first — they run via propy.bat, finalize the project, and open ArcGIS Pro "
                        "when done. Use arcgis_launch alone only when the user explicitly wants Pro opened without a "
                        "scripted workflow, or after a tool result says Pro was not launched."
                    ),
                    func=arcgis_launch,
                    args_schema=ArcGISLaunchInput
                ),
                StructuredTool(
                    name="arcgis_create_project",
                    description=(
                        "Create a new ArcGIS Pro project. "
                        "Can specify project name, location, coordinate system (e.g., 'UTM Zone 32N', 'WGS84'), "
                        "and template (MAP, CATALOG, GLOBAL_SCENE, LOCAL_SCENE)."
                    ),
                    func=arcgis_create_project,
                    args_schema=ArcGISCreateProjectInput
                ),
                StructuredTool(
                    name="arcgis_open_project",
                    description=(
                        "Open an existing ArcGIS Pro project (.aprx file). "
                        "Provide the full path to the project file."
                    ),
                    func=arcgis_open_project,
                    args_schema=ArcGISOpenProjectInput
                ),
                StructuredTool(
                    name="arcgis_set_coordinate_system",
                    description=(
                        "DEPRECATED: Use arcgis_execute_python_code for complex operations. "
                        "This tool only sets coordinate system and requires manual steps. "
                        "For complete workflows, use arcgis_execute_python_code."
                    ),
                    func=arcgis_set_coordinate_system,
                    args_schema=ArcGISSetCRSInput
                ),
                StructuredTool(
                    name="arcgis_get_project_info",
                    description=(
                        "Optionally inspect an ArcGIS Pro .aprx (pass project_path). "
                        "Local SurvyAI often has no arcpy — that failure is NOT a blocker. "
                        "For parcel fit / buffer / overlap analysis on known .aprx/.shp/.gdb/"
                        ".xlsx paths, prefer arcgis_execute_python_code or geopandas_execute; "
                        "do not ask the user to re-send paths already listed in the chat."
                    ),
                    func=arcgis_get_project_info,
                    args_schema=ArcGISProjectInfoInput
                ),
                StructuredTool(
                    name="arcgis_list_coordinate_systems",
                    description=(
                        "List available coordinate systems with their WKID/EPSG codes. "
                        "Filter by text (e.g., 'UTM', 'WGS') to find specific systems."
                    ),
                    func=arcgis_list_coordinate_systems,
                    args_schema=ArcGISListCRSInput
                ),
                StructuredTool(
                    name="arcgis_import_xy_points",
                    description=(
                        "Import XY points from an Excel file into an ArcGIS Pro project and add them to the map. "
                        "Generates a Python script, saves it to the project's scripts folder, launches ArcGIS Pro, "
                        "and provides instructions to run the script in ArcGIS Pro's Python window. "
                        "Use this after coordinate conversion (e.g., WGS84 -> Minna / Nigeria Mid Belt) to create a "
                        "proper point feature class with the correct coordinate system. "
                        "The script will create a file geodatabase, import points, set coordinate system, and add the layer to the map."
                    ),
                    func=arcgis_import_xy_points,
                    args_schema=ArcGISImportXYPointsInput
                ),
                StructuredTool(
                    name="arcgis_execute_python_code",
                    description=(
                        "*** PRIMARY TOOL for FULLY AUTOMATED ArcGIS workflows *** "
                        "Generate and execute complete ArcPy code that performs entire workflows WITHOUT user intervention. "
                        "Use for requests like: 'import points, create polygon, calculate area and bearings', "
                        "'analyze survey data and return traverse calculations', 'process coordinates and generate reports'. "
                        "The code executes automatically via propy.bat and returns computational results. "
                        "Generate code that: imports data, performs analysis, calculates results (areas, bearings, distances), "
                        "prints results with RESULT_ prefix for parsing, saves project. "
                        "For IDW/CutFill/volume: set projected CRS, add EVERY output layer to the map (addDataFromPath), "
                        "project.save(). NO MANUAL STEPS - user sees final results and a populated map."
                    ),
                    func=arcgis_execute_python_code,
                    args_schema=ArcGISExecutePythonCodeInput
                ),
                StructuredTool(
                    name="arcgis_excel_hull_traverse",
                    description=(
                        "*** VERIFIED end-to-end workflow *** "
                        "Use this for tasks like: 'import Excel points, create convex hull, compute area, "
                        "compute traverse distances/bearings, export results'. "
                        "This tool verifies inserted point counts and output files on disk, and avoids "
                        "adding any non-user data to the project."
                    ),
                    func=arcgis_excel_hull_traverse,
                    args_schema=ArcGISExcelHullTraverseInput,
                ),
                StructuredTool(
                    name="arcgis_fill_volume_idw_cutfill",
                    description=(
                        "*** VERIFIED fill-volume workflow *** "
                        "Excel -> IDW rasters (pre + post) -> Cut Fill -> fill volume (m³) -> results_fill.xlsx. "
                        "Creates ArcGIS Pro project, adds all layers (pre_idw, post_idw, cutfill, points, post_hull) to map, and opens ArcGIS Pro—as if a GIS analyst did it manually. "
                        "Use excel_inspect_workbook first for sheet/column names. Report project path and layers to user before final volume."
                    ),
                    func=arcgis_fill_volume_idw_cutfill,
                    args_schema=ArcGISFillVolumeIDWCutfillInput,
                ),
                StructuredTool(
                    name="arcgis_pre_post_csv_dwg_cutfill",
                    description=(
                        "*** VERIFIED borrow-pit / two-surface workflow *** "
                        "Use when the user provides SEPARATE PRE and POST point files (.csv or .xlsx) and a DWG boundary. "
                        "Runs: copy tabular inputs -> CADToGeodatabase -> PRE/POST points -> IDW (Z) -> CutFill -> volume CSV. "
                        "Populates the ArcGIS map with boundary, PRE/POST points, both IDW rasters, dz, and cutfill, then opens Pro."
                    ),
                    func=arcgis_pre_post_csv_dwg_cutfill,
                    args_schema=ArcGISPrePostCSVDWGCutfillInput,
                ),
                StructuredTool(
                    name="arcgis_pre_post_csv_dwg_tin_volume",
                    description=(
                        "*** VERIFIED TIN-based borrow-pit workflow (3D Analyst) *** "
                        "Same inputs as arcgis_pre_post_csv_dwg_cutfill but builds PRE/POST TINs, TinRaster, then dz volume. "
                        "Retries CreateTin without hard clip if ERROR 999999; if still failing, falls back to the IDW workflow "
                        "so the user still gets Adibawa_VolumeResult2.csv (or chosen output path)."
                    ),
                    func=arcgis_pre_post_csv_dwg_tin_volume,
                    args_schema=ArcGISPrePostCSVDWGTinVolumeInput,
                ),
                StructuredTool(
                    name="arcgis_finalize_visualization",
                    description=(
                        "Finalize ArcGIS Pro project visualization AFTER user operations complete. "
                        "Adds 'Imagery Hybrid' basemap and loads the project geodatabase feature classes and rasters (IDW, CutFill, etc.). "
                        "This should be called AFTER all user-requested operations are done, so users can "
                        "visually verify that their instructions were properly carried out. "
                        "NOTE: This is automatically called by arcgis_execute_python_code and arcgis_excel_hull_traverse, "
                        "but can be called manually if needed."
                    ),
                    func=arcgis_finalize_visualization,
                    args_schema=ArcGISFinalizeVisualizationInput,
                ),
            ])
            
            logger.info(
                "✓ Added ArcGIS bundle: filesystem_stat + 13 ArcGIS tools "
                "(launch, projects, import_xy, execute_python_code, verified workflows, finalize)"
            )
        else:
            logger.info("⚠ ArcGIS Pro not installed - ArcGIS tools not available")
        
        tools = self._filter_tools_by_feature_flags(tools)
        logger.info(f"Created {len(tools)} tools for the agent (after license feature filter)")
        return tools
    
    # ==========================================================================
    # LLM INVOCATION HELPERS
    # ==========================================================================

    def _reset_pipeline_llm_cost(self) -> None:
        """Reset per-query LLM cost accumulator (fast-path + direct invoke tracking)."""
        self._pipeline_llm_cost_usd = 0.0

    def _track_llm_invoke_result(self, msg: Any, model_name: Optional[str] = None) -> None:
        """Add one direct LLM response (AIMessage) to the current query's cost tally."""
        try:
            from langchain_core.messages import AIMessage
            from utils.cost_estimator import (
                estimate_token_cost_usd,
                extract_message_token_usage,
                is_local_free_model,
            )
        except ImportError:
            return
        if not isinstance(msg, AIMessage):
            return
        # Prefer the bound active model; never default Ollama runs to OpenAI pricing.
        mn = (model_name or "").strip() or str(getattr(self, "_active_model_name", "") or "").strip()
        if not mn and self._llm_is_ollama():
            mn = self._resolve_provider_model_name("ollama")
        if not mn:
            mn = (
                getattr(self, "_current_openai_model", None)
                or getattr(self.settings, "openai_model", None)
                or getattr(self.settings, "openai_model_mini", "gpt-5.4-mini")
            )
        if self._llm_is_ollama() or is_local_free_model(str(mn)):
            return
        usage = extract_message_token_usage(msg)
        if not usage:
            return
        cost = estimate_token_cost_usd(
            str(mn),
            int(usage.get("input_tokens") or 0),
            int(usage.get("output_tokens") or 0),
            cached_input_tokens=int(usage.get("cached_input_tokens") or 0),
        )
        if cost > 0:
            self._pipeline_llm_cost_usd = round(
                float(self._pipeline_llm_cost_usd or 0.0) + float(cost), 6
            )

    def finalize_query_result_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attach llm_cost_usd for Credits & Usage when fast paths omit graph message totals.

        Prefers an explicit graph estimate when present; otherwise uses the per-query
        accumulator fed by ``_run_with_timeout`` / direct ``llm.invoke`` calls.
        Local/Ollama runs are always $0.
        """
        out = dict(result or {})
        try:
            from utils.cost_estimator import is_local_free_model
        except ImportError:
            is_local_free_model = lambda _n: False  # type: ignore
        mn = str(out.get("model_name") or getattr(self, "_active_model_name", "") or "").strip()
        if self._llm_is_ollama() or is_local_free_model(mn):
            out["llm_cost_usd"] = 0.0
            return out
        existing = float(out.get("llm_cost_usd") or 0.0)
        tracked = float(self._pipeline_llm_cost_usd or 0.0)
        if existing > 0:
            out["llm_cost_usd"] = round(existing, 6)
        elif tracked > 0:
            out["llm_cost_usd"] = round(tracked, 6)
        else:
            out.setdefault("llm_cost_usd", 0.0)
        return out

    def _llm_run_with_timeout(self, model_name: Optional[str] = None) -> Callable[..., Any]:
        """Return a ``run_with_timeout`` callable that tags LLM costs with ``model_name``."""

        def _runner(timeout_seconds: int, fn: Callable[[], Any]) -> Tuple[Optional[Any], Optional[Exception], bool]:
            return self._run_with_timeout(
                timeout_seconds, fn, llm_model_name=model_name, serialize_llm=True
            )

        return _runner

    def _run_with_timeout(
        self,
        timeout_seconds: int,
        fn: Callable[[], Any],
        *,
        llm_model_name: Optional[str] = None,
        serialize_llm: bool = False,
    ) -> Tuple[Optional[Any], Optional[Exception], bool]:
        """
        Run a callable in a daemon thread with a timeout.
        Returns (result, error, timed_out). Exactly one of result or error is set when not timed_out.

        When ``serialize_llm`` is True, acquires an invoke gate so timed-out Ollama/HTTP
        calls cannot stack concurrent generations (a common cause of host lockups).
        The gate stays held until the worker thread finishes.
        """
        result_container: List[Any] = [None]
        err_container: List[Optional[Exception]] = [None]
        gate = getattr(self, "_llm_invoke_gate", None) if serialize_llm else None
        acquired = False
        started = False
        if gate is not None:
            acquired = bool(gate.acquire(timeout=max(1.0, float(timeout_seconds))))
            if not acquired:
                return (
                    None,
                    TimeoutError(
                        "Another LLM call is still finishing. Wait a moment and try again "
                        "(this protects your PC from stacked local-model load)."
                    ),
                    True,
                )

        def run():
            try:
                result_container[0] = fn()
            except Exception as e:
                err_container[0] = e
            finally:
                if gate is not None and acquired:
                    try:
                        gate.release()
                    except RuntimeError:
                        pass

        try:
            t = threading.Thread(target=run, daemon=True)
            t.start()
            started = True
            t.join(timeout=timeout_seconds)
            timed_out = t.is_alive()
            if timed_out and serialize_llm:
                # Give HTTP-layer timeouts a short window to abort the orphaned call
                # instead of leaving Ollama generating indefinitely.
                t.join(timeout=min(20.0, float(timeout_seconds)))
                if self._llm_is_ollama():
                    self._release_ollama_model()
                if t.is_alive():
                    logger.error(
                        "LLM call still running after timeout — further LLM invokes are "
                        "blocked until it finishes (protects the PC from overload)."
                    )
            result = result_container[0]
            error = err_container[0]
            if result is not None and error is None and not timed_out:
                try:
                    self._track_llm_invoke_result(result, llm_model_name)
                except Exception:
                    pass
            return (result, error, timed_out)
        finally:
            if gate is not None and acquired and not started:
                try:
                    gate.release()
                except RuntimeError:
                    pass

    def _invoke_llm_with_retry(self, messages: List[Any]) -> Any:
        """Invoke LLM with timeout protection; raises TimeoutError or the LLM exception on failure."""
        timeout_seconds = int(getattr(self.settings, "llm_invoke_timeout_seconds", 180) or 180)
        if timeout_seconds < 60:
            timeout_seconds = 60
        # Local Ollama on CPU can peg the machine; keep per-call budget tighter.
        if self._llm_is_ollama():
            ok, ram_err, _num_ctx = _ollama_ram_policy(self._resolve_provider_model_name("ollama"))
            if not ok:
                raise RuntimeError(ram_err)
            timeout_seconds = min(timeout_seconds, 90)
        active_model = str(getattr(self, "_active_model_name", "") or "").strip()
        if not active_model and self._llm_is_ollama():
            active_model = self._resolve_provider_model_name("ollama")
        result, error, timed_out = self._run_with_timeout(
            timeout_seconds,
            lambda: self.llm_with_tools.invoke(messages),
            llm_model_name=active_model or None,
            serialize_llm=True,
        )
        if timed_out:
            logger.error(f"LLM invocation timed out after {timeout_seconds} seconds")
            raise TimeoutError(
                f"LLM call timed out after {timeout_seconds} seconds. "
                "The query may be too complex or the document too large. "
                "Try breaking the task into smaller steps."
            )
        if error:
            if "429" in str(error) or "rate limit" in str(error).lower() or "tpm" in str(error).lower():
                logger.warning("Rate limit error detected, will be handled by caller")
            raise error
        return result

    def _ensure_app_bound(self, llm: BaseChatModel, model_name: Optional[str], tools_to_bind: List[BaseTool]) -> None:
        """
        Bind tools and (re)compile the graph only when necessary.
        This is a pure efficiency optimization: behavior is unchanged.
        """
        model_sig = model_name or getattr(llm, "model", None) or "unknown"
        tool_sig = tuple(sorted([t.name for t in tools_to_bind]))

        if self._app_signature == (model_sig, tool_sig) and getattr(self, "app", None) is not None:
            return

        self._current_tools = tools_to_bind
        self._active_model_name = str(model_name or model_sig or "").strip()
        self.llm_with_tools = llm.bind_tools(tools_to_bind)
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.memory)
        self._app_signature = (model_sig, tool_sig)
    
    # ==========================================================================
    # LANGGRAPH CONSTRUCTION
    # ==========================================================================
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph that defines the agent's behavior.
        
        The graph has two main nodes:
        1. agent_node: Calls the LLM to reason about the query
        2. tools_node: Executes any tools the LLM requests
        
        The flow is:
        - Start → agent_node
        - agent_node → (if tool calls) → tools_node → agent_node
        - agent_node → (if no tool calls) → END
        
        Returns:
            StateGraph: The configured graph (not yet compiled)
        """
        
        # ==================================================================
        # Define the agent node
        # ==================================================================
        
        def agent_node(state: AgentState) -> Dict:
            """
            The agent node - where the LLM does its reasoning.
            
            This node:
            1. Takes the current conversation state
            2. Sends messages to the LLM (with system prompt)
            3. Returns the LLM's response (which may include tool calls)
            
            Args:
                state: Current conversation state with message history
                
            Returns:
                Dict with new messages to add to state
            """
            # Get current messages from state
            messages = list(state["messages"])
            
            # Ensure system prompt is at the start
            # This guides the LLM's behavior throughout the conversation
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=self._system_prompt)] + messages
            
            # Get current model name for token estimation
            current_model = getattr(self, '_current_openai_model', None) or \
                          getattr(self, '_current_gemini_model', None) or \
                          getattr(self.settings, 'openai_model', 'gpt-4o-mini')
            
            # Estimate tokens and check if we need chunking
            input_tokens, output_tokens_estimate = estimate_message_tokens(messages, current_model)
            token_estimate = check_tpm_limit(input_tokens, output_tokens_estimate, current_model)
            
            # Check if there are any pending tool calls in the conversation
            # We cannot chunk messages that contain tool calls because tool calls must be
            # immediately followed by tool responses (OpenAI API requirement)
            has_pending_tool_calls = False
            for msg in messages:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    has_pending_tool_calls = True
                    break
                # Also check if we have ToolMessages that might be part of an active sequence
                if isinstance(msg, ToolMessage):
                    # If there's a ToolMessage, there was likely a tool call before it
                    # Check if the previous message was an AIMessage with tool_calls
                    msg_idx = messages.index(msg)
                    if msg_idx > 0 and isinstance(messages[msg_idx - 1], AIMessage):
                        if messages[msg_idx - 1].tool_calls:
                            has_pending_tool_calls = True
                            break
            
            # If tokens exceed limit AND there are no pending tool calls, chunk the messages
            # Otherwise, proceed normally and let rate limit errors be handled by retries
            if token_estimate.exceeds_tpm and not has_pending_tool_calls:
                logger.warning(
                    f"Token limit exceeded in agent_node: {token_estimate.total_tokens:,} tokens. "
                    f"Chunking into {token_estimate.chunks_needed} chunks."
                )
                # Chunk with 90% of TPM limit for safety
                safe_limit = int(token_estimate.tpm_limit * 0.9)
                message_chunks = chunk_messages(messages, safe_limit, current_model)
                
                # Process chunks with delays
                all_responses = []
                for chunk_idx, chunk in enumerate(message_chunks):
                    if chunk_idx > 0:
                        # Wait 61 seconds between chunks to reset rate limit window
                        wait_for_rate_limit(61)
                    
                    logger.info(f"Processing chunk {chunk_idx + 1}/{len(message_chunks)}")
                    try:
                        response = self._invoke_llm_with_retry(chunk)
                        all_responses.append(response)
                    except Exception as e:
                        # If rate limit error, wait and retry
                        if "429" in str(e) or "rate limit" in str(e).lower():
                            logger.warning(f"Rate limit hit on chunk {chunk_idx + 1}, waiting and retrying...")
                            wait_for_rate_limit(61)
                            response = self._invoke_llm_with_retry(chunk)
                            all_responses.append(response)
                        else:
                            raise
                
                # Combine responses (for now, use the last response)
                # In a more sophisticated implementation, we could merge responses
                response = all_responses[-1] if all_responses else None
                if not response:
                    raise RuntimeError("No response received from chunked LLM calls")
            elif token_estimate.exceeds_tpm and has_pending_tool_calls:
                # Cannot chunk due to tool calls - proceed normally and handle rate limits via retries
                logger.warning(
                    f"Token limit exceeded ({token_estimate.total_tokens:,} tokens) but tool calls detected. "
                    f"Proceeding without chunking - rate limits will be handled by retries."
                )
                response = self._invoke_llm_with_retry(messages)
            else:
                # Normal path: no chunking needed
                response = self._invoke_llm_with_retry(messages)
            
            # Return the response to be added to state
            return {"messages": [response]}
        
        # ==================================================================
        # Define the routing function
        # ==================================================================
        
        def should_continue(state: AgentState) -> Literal["tools", "end"]:
            """
            Determine whether to execute tools or end the conversation.
            
            This function looks at the last message from the agent:
            - If it contains tool_calls → route to "tools" node
            - Otherwise → route to END (conversation complete)
            
            Also enforces max_iterations limit to prevent infinite loops.
            
            Args:
                state: Current conversation state
                
            Returns:
                "tools" if tools should be executed, "end" otherwise
            """
            messages = state["messages"]
            last_message = messages[-1]
            
            # Check iteration count to prevent infinite loops
            max_iterations = getattr(self.settings, 'agent_max_iterations', 20)
            tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
            iteration_count = len(tool_messages)

            if iteration_count >= max_iterations:
                logger.warning(
                    f"Max iterations ({max_iterations}) reached. "
                    "Stopping to prevent infinite loop. "
                    "The query may be too complex or require manual intervention."
                )
                return "end"

            # Same-error stop: if last two tool results look like the same failure, end to avoid runaway cost
            if iteration_count >= 2:
                last_two = tool_messages[-2:]
                contents = []
                for tm in last_two:
                    c = getattr(tm, "content", None) or ""
                    if isinstance(c, list):
                        c = " ".join(str(part.get("text", part)) for part in c if isinstance(part, dict))
                    else:
                        c = str(c)
                    contents.append(c[:300].lower().strip())
                if contents[0] and contents[1] and (
                    contents[0] == contents[1]
                    or (contents[0].split()[:20] == contents[1].split()[:20] and ("error" in contents[0] or "failed" in contents[0]))
                ):
                    logger.warning(
                        "Same or very similar tool error repeated; stopping loop to prevent runaway cost. "
                        "Report what was tried and suggest next step."
                    )
                    return "end"

            # Check if the AI wants to use tools
            # AIMessage has a tool_calls attribute when tools are requested
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
                logger.info(f"Iteration {iteration_count + 1}/{max_iterations}: Agent requested tools: {', '.join(tool_names)}")
                # Warn if internet_search is being called (should be removed if already searched)
                if "internet_search" in tool_names:
                    logger.warning("⚠ Agent is trying to call internet_search - this may indicate a loop if search was already done")
                return "tools"
            
            # No tool calls - we're done
            logger.info(f"Agent completed after {iteration_count} tool iterations")
            return "end"
        
        # ==================================================================
        # Build the graph
        # ==================================================================
        
        # Create a new graph with our state schema
        graph = StateGraph(AgentState)
        
        # Add the agent node (LLM reasoning)
        graph.add_node("agent", agent_node)
        
        # Add the tools node (tool execution)
        # ToolNode is a pre-built node that handles tool execution.
        #
        # NOTE: We intentionally avoid wrapping ToolNode here because LangGraph/LangChain
        # passes runtime config through the graph, and wrappers can accidentally drop or
        # invalidate required config keys (causing errors like "Missing required config key ... for 'tools'").
        # Use filtered tools if available (e.g., when internet_search was removed), otherwise use all tools
        tools_for_node = getattr(self, "_current_tools", None) or self.tools
        tools_node = ToolNode(tools_for_node)
        graph.add_node("tools", tools_node)
        
        # Set the entry point - where the graph starts
        graph.set_entry_point("agent")
        
        # Add conditional routing from agent
        # After the agent runs, we check if tools should be called
        graph.add_conditional_edges(
            "agent",           # From node
            should_continue,   # Routing function
            {
                "tools": "tools",   # If should_continue returns "tools"
                "end": END          # If should_continue returns "end"
            }
        )
        
        # After tools run, go back to agent to process results
        graph.add_edge("tools", "agent")
        
        logger.info("LangGraph built successfully")
        return graph
    
    # ==========================================================================
    # QUERY PROCESSING
    # ==========================================================================
    
    def process_query(
        self, 
        query: str, 
        use_fallback: bool = False,
        session_id: Optional[str] = None,
        interactive_mode: bool = False
    ) -> Dict:
        """
        Process a user query through the agent.
        
        This is the main entry point for using the agent. It:
        1. Retrieves relevant context from vector store (if enabled)
        2. Creates a new conversation thread
        3. Invokes the LangGraph with the query and context
        4. Extracts and returns the final response
        5. Stores the conversation for future context (if enabled)
        6. Handles errors and fallback to secondary LLM
        7. Checks token limits and handles chunking with rate limiting
        
        Args:
            query: The user's question or request
            use_fallback: If True, use the fallback LLM instead of primary
            session_id: Optional session ID for conversation continuity.
                       If not provided, uses the current session or creates new.
            interactive_mode: If True, will ask user for approval when token limits are exceeded
            
        Returns:
            Dict containing:
            - query: The original query
            - response: The agent's response text
            - llm_used: Which LLM was used ("primary" or "fallback")
            - success: Whether the query was processed successfully
            - error: Error message if success is False
            - session_id: The session ID used for this query
            - context_retrieved: Whether context was retrieved from vector store
            
        Example:
            >>> result = agent.process_query("What is the area of this survey?")
            >>> print(result["response"])
            "The total area of the survey is 1,500 square meters..."
        """
        # Set interactive mode flag
        self._interactive_mode = interactive_mode
        self._reset_pipeline_llm_cost()

        # Get or set session ID as early as possible so permission handling and
        # short continuation replies can remain anchored to the active conversation.
        if session_id:
            self.set_session_id(session_id)
        current_session_id = self.get_session_id()
        
        # Reset model switch flag for new query
        self._model_switched_this_query = False
        self._openai_models_tried_this_query = []
        self._force_internet_search_this_query = False
        # Bound early so OCR/excel/CAD fast-paths can return before tier selection.
        model_name_used: Optional[str] = None

        # Handle internet permission markers from interactive CLI
        # IMPORTANT: Extract and set permission BEFORE routing/processing
        q_upper = (query or "").upper()
        original_query = query

        # Permission is per-request unless explicitly granted this turn (tag, dialog,
        # or affirmative reply to a permission ask in the same conversation).
        _perm_carry = (
            "[INTERNET_PERMISSION_GRANTED]" in q_upper
            or "[INTERNET_PERMISSION_DENIED]" in q_upper
            or self._last_assistant_asked_internet_permission(query)
            or current_session_id in self._pending_permission_requests
        )
        if not _perm_carry:
            self._internet_permission_granted = False

        # Stale pending requests from a prior deterministic dialog must not block
        # conversational grant detection on this turn.
        if "[INTERNET_PERMISSION_GRANTED]" in q_upper or "[INTERNET_PERMISSION_DENIED]" in q_upper:
            self._pending_permission_requests.pop(current_session_id, None)

        pending_permission = self._pending_permission_requests.get(current_session_id)
        if pending_permission and isinstance(query, str):
            stripped_reply = query.strip()
            # Use the lenient permission affirm detector so "yes go ahead" counts.
            if self._is_affirmative_permission_reply(stripped_reply):
                if pending_permission.get("kind") == "internet":
                    self._internet_permission_granted = True
                    self._force_internet_search_this_query = True
                    query = f"[INTERNET_PERMISSION_GRANTED]\n{pending_permission.get('query', '')}".strip()
                    q_upper = query.upper()
                    original_query = str(pending_permission.get("query") or original_query)
                    logger.info("✓ Applied plain-text approval reply to pending internet permission request")
                self._pending_permission_requests.pop(current_session_id, None)
            elif self._is_negative_permission_reply(stripped_reply):
                if pending_permission.get("kind") == "internet":
                    self._internet_permission_granted = False
                    query = f"[INTERNET_PERMISSION_DENIED]\n{pending_permission.get('query', '')}".strip()
                    q_upper = query.upper()
                    original_query = str(pending_permission.get("query") or original_query)
                    logger.info("✓ Applied plain-text denial reply to pending internet permission request")
                self._pending_permission_requests.pop(current_session_id, None)
        
        # FIRST: Extract actual query if this is a continuation (has context markers)
        # This must happen BEFORE permission tag handling so we route correctly
        actual_query_for_routing = query
        if (
            "=== CONTINUATION OF PREVIOUS WORK" in query
            or "=== CONVERSATION CONTEXT" in query
            or "--- Exchange" in query
        ):
            # Extract the actual current query from the context-enhanced query
            if "NOW, the user wants you to continue with this new request:" in query:
                parts = query.split("NOW, the user wants you to continue with this new request:")
                if len(parts) > 1:
                    actual_query_for_routing = parts[-1].strip()
                    logger.info(f"🔍 Detected continuation query - extracted actual request: {actual_query_for_routing[:100]}...")
            elif "\n\n" in query:
                # Fallback: get the last part after double newline
                parts = query.split("\n\n")
                actual_query_for_routing = parts[-1].strip()
                logger.info(f"🔍 Detected continuation query - using last part: {actual_query_for_routing[:100]}...")

        # ==================================================================
        # CONVERSATIONAL INTERNET-PERMISSION GRANT (anti-loop)
        # ------------------------------------------------------------------
        # The LLM sometimes asks for internet permission in free text (outside
        # the deterministic router path). In that case there is no pending
        # permission request, so a bare "yes" was never recognised and the model
        # kept re-asking forever. Here we detect that the PREVIOUS assistant turn
        # asked for internet permission and the CURRENT user message affirms (or
        # denies) it, then resolve the grant against the original question.
        # ==================================================================
        if (
            not pending_permission
            and "[INTERNET_PERMISSION_GRANTED]" not in q_upper
            and "[INTERNET_PERMISSION_DENIED]" not in q_upper
            and self._last_assistant_asked_internet_permission(query)
        ):
            current_msg = (actual_query_for_routing or query or "").strip()
            if self._is_affirmative_permission_reply(current_msg):
                underlying = self._underlying_question_from_history(query) or current_msg
                query = f"[INTERNET_PERMISSION_GRANTED]\n{underlying}"
                actual_query_for_routing = f"[INTERNET_PERMISSION_GRANTED]\n{underlying}"
                q_upper = query.upper()
                original_query = underlying
                self._force_internet_search_this_query = True
                logger.info(
                    "✓ Affirmative reply to a conversational internet-permission "
                    "request detected — granting permission and forcing search for: "
                    f"{underlying[:120]}"
                )
            elif self._is_negative_permission_reply(current_msg):
                underlying = self._underlying_question_from_history(query) or current_msg
                query = f"[INTERNET_PERMISSION_DENIED]\n{underlying}"
                actual_query_for_routing = f"[INTERNET_PERMISSION_DENIED]\n{underlying}"
                q_upper = query.upper()
                original_query = underlying
                logger.info(
                    "✓ Negative reply to a conversational internet-permission "
                    "request detected — answering offline for: "
                    f"{underlying[:120]}"
                )

        # Bare "yes" / "go ahead" → bind to the LAST assistant optional offer only.
        # Numbered option picks ("go ahead with option 2") → bind to that menu item + GIS context.
        if "[INTERNET_PERMISSION_GRANTED]" not in q_upper and "[INTERNET_PERMISSION_DENIED]" not in q_upper:
            option_n = self._parse_option_selection(actual_query_for_routing)
            option_resolution = None
            if option_n is not None and not self._last_assistant_asked_internet_permission(query):
                last = self._extract_last_assistant_turn(query)
                options = self._extract_numbered_options(last)
                selected = options.get(option_n) or ""
                option_resolution = self._reconstruct_gis_followup_intent(
                    query, selected_option_text=selected or actual_query_for_routing
                )
                if not option_resolution and selected:
                    option_resolution = (
                        f"The user selected option {option_n} from your last message: {selected} "
                        "Proceed with ONLY that approach. Reuse verified files/layers already "
                        "produced in this conversation — do not ask the user to re-send them."
                    )
            offer_resolution = option_resolution
            if not offer_resolution:
                # Prefer GIS fit binding over generic affirmative expansion so
                # "Go ahead" after a fit offer keeps session .aprx/.gdb/.xlsx paths.
                offer_resolution = self._resolve_gis_fit_offer_affirmation(
                    query, actual_query_for_routing
                )
            if not offer_resolution:
                offer_resolution = self._resolve_affirmative_to_last_offer(
                    query, actual_query_for_routing
                )
            if not offer_resolution:
                # User pasted .aprx/.shp paths after being asked — bind to pending FIT.
                offer_resolution = self._enrich_gis_paths_for_pending_fit(
                    query, actual_query_for_routing
                )
            if not offer_resolution:
                # User restated a full convert/plot/buffer job after a prior success → re-run.
                offer_resolution = self._enrich_repeated_full_gis_workflow(
                    query, actual_query_for_routing
                )
            if not offer_resolution:
                # First-turn GIS fit follow-up ("would X fit in the open ArcGIS result?")
                # — bind verified .aprx/.gdb/Excel paths from this chat immediately.
                offer_resolution = self._enrich_current_gis_fit_followup(
                    query, actual_query_for_routing
                )
            if offer_resolution:
                actual_query_for_routing = offer_resolution
                marker = "NOW, the user wants you to continue with this new request:"
                if marker in query:
                    query = (
                        query.split(marker)[0].rstrip()
                        + "\n"
                        + marker
                        + "\n"
                        + offer_resolution
                    )
                else:
                    query = offer_resolution
                logger.info(
                    "Affirmative/option/GIS-fit reply resolved with session context: %s",
                    offer_resolution[:180],
                )

        if "[INTERNET_PERMISSION_GRANTED]" in q_upper:
            self._internet_permission_granted = True
            self._force_internet_search_this_query = True
            self._pending_permission_requests.pop(current_session_id, None)
            # Clean the query to remove permission tags for cleaner processing
            query = query.replace("[INTERNET_PERMISSION_GRANTED]", "").replace("[internet_permission_granted]", "").strip()
            actual_query_for_routing = actual_query_for_routing.replace("[INTERNET_PERMISSION_GRANTED]", "").replace("[internet_permission_granted]", "").strip()
            logger.info("✓ Internet permission granted - permission tag removed from query")
        if "[INTERNET_PERMISSION_DENIED]" in q_upper:
            self._internet_permission_granted = False
            self._pending_permission_requests.pop(current_session_id, None)
            query = query.replace("[INTERNET_PERMISSION_DENIED]", "").replace("[internet_permission_denied]", "").strip()
            actual_query_for_routing = actual_query_for_routing.replace("[INTERNET_PERMISSION_DENIED]", "").replace("[internet_permission_denied]", "").strip()
            logger.info("✓ Internet permission denied - permission tag removed from query")

        # After an internet-permission grant/deny, the offer-resolution block above is
        # skipped (tag present). Still bind session GIS artifacts for fit / open-result
        # asks so Allow → search does not forget .aprx/.shp paths.
        if (
            ("CONTINUATION (retain session GIS context" not in (actual_query_for_routing or ""))
            and ("RE-EXECUTE FULL GIS WORKFLOW" not in (actual_query_for_routing or ""))
        ):
            post_perm_enrich = (
                self._enrich_gis_paths_for_pending_fit(query, actual_query_for_routing)
                or self._enrich_current_gis_fit_followup(query, actual_query_for_routing)
            )
            if post_perm_enrich:
                actual_query_for_routing = post_perm_enrich
                marker = "NOW, the user wants you to continue with this new request:"
                # Prefer preserving any history already on `query` / `original_query`.
                hist_src = query
                if marker not in hist_src and marker in (original_query or ""):
                    hist_src = original_query
                if marker in hist_src:
                    query = (
                        hist_src.split(marker)[0].rstrip()
                        + "\n"
                        + marker
                        + "\n"
                        + post_perm_enrich
                    )
                elif "=== CONVERSATION CONTEXT" in (hist_src or ""):
                    query = hist_src.rstrip() + "\n" + marker + "\n" + post_perm_enrich
                else:
                    query = post_perm_enrich
                logger.info(
                    "Post-permission GIS fit enrichment applied: %s",
                    post_perm_enrich[:180],
                )
        
        try:
            logger.info(f"Processing query: {query[:200]}...")
            
            # ==================================================================
            # EARLY ROUTER CHECK: Ask for internet permission BEFORE any processing
            # This prevents loops and ensures proactive permission requests
            # Use the extracted actual query for routing (not the context-enhanced one)
            # ==================================================================
            early_rag_decision = self._decide_rag_route(actual_query_for_routing, interactive_mode=interactive_mode)
            # Safety override: explicit local file/tool workflows should not trigger
            # internet permission prompts unless the user explicitly asks for web /
            # Living Atlas / pretrained land-cover catalog lookup.
            local_file_driven = looks_like_file_driven_task(actual_query_for_routing)
            explicit_web_intent = self._wants_external_catalog_or_web(actual_query_for_routing)
            if local_file_driven and not explicit_web_intent:
                if early_rag_decision.use_internet:
                    logger.info("🔧 File-driven workflow detected - suppressing internet permission prompt")
                early_rag_decision.use_internet = False
                if early_rag_decision.route in ("internet", "hybrid"):
                    early_rag_decision.route = "vector" if early_rag_decision.use_vector else "llm_only"
            # User already affirmed an internet-permission request this turn: force the
            # search to run instead of letting the router (or the LLM) ask again.
            if getattr(self, "_internet_permission_granted", False) and (
                getattr(self, "_force_internet_search_this_query", False)
                or early_rag_decision.use_internet
                or explicit_web_intent
                or self._is_current_fact_question(
                    self._extract_clean_question(actual_query_for_routing) or actual_query_for_routing
                )
            ):
                early_rag_decision.use_internet = True
                clean_q = self._extract_clean_question(actual_query_for_routing) or actual_query_for_routing
                # For GIS follow-ups, search the reference object dimensions — not "option 2" prose.
                # For Living Atlas / land-cover, keep a catalog-oriented seed.
                search_seed = self._internet_search_seed_from_intent(clean_q) or clean_q
                try:
                    from survyai.gis_session_intent import looks_like_living_atlas_or_landcover_dl
                    if looks_like_living_atlas_or_landcover_dl(clean_q):
                        search_seed = (
                            "Esri Living Atlas pretrained land cover deep learning model "
                            "grassland classification ArcGIS Pro"
                        )
                except Exception:
                    pass
                if not getattr(early_rag_decision, "internet_query", None):
                    variants = self._optimize_internet_search_queries(search_seed)
                    early_rag_decision.internet_query = variants[0] if variants else search_seed
                else:
                    # Prefer a tighter seed when the router kept option-selection wording.
                    routed = str(early_rag_decision.internet_query or "")
                    if self._parse_option_selection(routed) is not None or "option" in routed.lower():
                        early_rag_decision.internet_query = search_seed
                if early_rag_decision.route == "llm_only":
                    early_rag_decision.route = "internet"
                elif early_rag_decision.route == "vector":
                    early_rag_decision.route = "hybrid"
                logger.info("🔎 Forcing internet search — permission already granted.")
            if early_rag_decision.use_internet and not getattr(self, "_internet_permission_granted", False):
                if interactive_mode:
                    logger.info("🔍 Router detected internet need - requesting permission BEFORE processing")
                    self._pending_permission_requests[current_session_id] = {
                        "kind": "internet",
                        "query": original_query,
                        "actual_query": actual_query_for_routing,
                    }
                    return {
                        "query": original_query,
                        "response": (
                            "PERMISSION REQUIRED: INTERNET SEARCH\n\n"
                            "This query appears to require up-to-date external information (standards, citations, current data).\n"
                            "May I search the internet for up-to-date information? (yes/no)\n\n"
                            "[INTERNET_PERMISSION_REQUEST]"
                        ),
                        "success": False,
                        "error": "internet_permission_required",
                        "llm_used": "fallback" if use_fallback else "primary",
                        "model_name": None,  # Not initialized yet
                        "session_id": current_session_id,
                    }
                else:
                    # Non-interactive: proceed without internet (user can't grant permission)
                    logger.warning("⚠ Internet needed but non-interactive mode - proceeding without internet search")
            
            logger.info(f"Current primary LLM setting: {self.settings.primary_llm}")

            # ==================================================================
            # TASK-SCOPED ROUTING (context-leak prevention)
            # ------------------------------------------------------------------
            # CRITICAL: All intent/task classification (complexity, fast-paths,
            # file-driven detection, tool selection) must run on the CURRENT user
            # request only — never on the history-enriched `query` blob.
            #
            # The GUI prepends recent conversation history to `query` for
            # continuity.  If we classify on that blob, stale context (e.g. a
            # previous CAD plan + "add the road") leaks into routing and the agent
            # fires a destructive tool pipeline on an unrelated new question.
            # `actual_query_for_routing` already holds the extracted current
            # request (see continuation-marker extraction above); fall back to the
            # raw query when there is no injected history.
            # ==================================================================
            routing_query = (actual_query_for_routing or query or "").strip() or query
            retry_requested = self._is_retry_request_from_routing_context(
                raw_query=query,
                extracted_query=actual_query_for_routing,
                routing_query=routing_query,
            )
            tool_routing_query = query if retry_requested else routing_query
            if tool_routing_query != routing_query:
                logger.info("Explicit retry request detected; allowing tool routing against reference history.")

            _pre_intent = self._classify_query_intent(routing_query)

            # FAST PATH (early): survey plan PDF -> CAD DWG — current turn only (not knowledge).
            if _pre_intent != "knowledge" and self._should_fastpath_pdf_survey_replot(tool_routing_query, routing_query):
                logger.info("PDF survey replot fast-path triggered (early router)")
                fast = self._run_pdf_survey_replot_pipeline(tool_routing_query)
                llm_used = "fallback" if use_fallback else "primary"
                return {
                    "query": query,
                    "response": fast.get("response") or fast.get("error") or str(fast),
                    "llm_used": llm_used,
                    "model_name": fast.get("model_name"),
                    "complexity": "complex",
                    "success": bool(fast.get("success")),
                    "session_id": current_session_id,
                    "context_retrieved": False,
                    "output_path": fast.get("output_path"),
                    "error": fast.get("error") if not fast.get("success") else None,
                }

            # FAST PATH (early): extract key details from a survey-plan PDF (vision for scans).
            if _pre_intent != "knowledge" and self._should_fastpath_pdf_plan_key_details(
                tool_routing_query
            ):
                logger.info("PDF survey key-details (vision) fast-path triggered")
                fast = self._run_pdf_plan_key_details_pipeline(tool_routing_query)
                llm_used = "fallback" if use_fallback else "primary"
                return {
                    "query": query,
                    "response": fast.get("response") or fast.get("error") or str(fast),
                    "llm_used": llm_used,
                    "model_name": fast.get("model_name") or model_name_used,
                    "complexity": "complex",
                    "success": bool(fast.get("success")),
                    "session_id": current_session_id,
                    "context_retrieved": False,
                    "error": fast.get("error") if not fast.get("success") else None,
                }

            # VISION OCR (images): preprocess only — LangGraph stays text-only.
            self._vision_ocr_context = None
            if getattr(self.settings, "enable_vision_ocr", True):
                from agent.vision_ocr import (
                    export_ocr_extraction_to_docx,
                    export_ocr_extraction_to_excel,
                    extract_image_paths_from_query,
                    format_last_ocr_for_agent,
                    is_ocr_export_request,
                    is_ocr_followup_request,
                    is_ocr_only_request,
                    is_ocr_word_export_request,
                    load_last_ocr_extraction,
                    resolve_ocr_export_path,
                    resolve_ocr_word_export_path,
                    save_last_ocr_extraction,
                    should_fastpath_image_survey_replot,
                    user_requested_save,
                )
                from survyai.provider_models import (
                    provider_supports_vision,
                    vision_unsupported_user_message,
                )

                _image_paths = extract_image_paths_from_query(tool_routing_query)
                _ocr_user_text = self._vision_ocr_user_text(tool_routing_query)
                _ws = Path.cwd()

                # Follow-up with no new images: export prior extraction or inject it for the agent.
                if not _image_paths and _pre_intent != "knowledge":
                    _last = load_last_ocr_extraction(_ws)
                    if _last and is_ocr_export_request(_ocr_user_text):
                        llm_used = "fallback" if use_fallback else "primary"
                        try:
                            out_path = resolve_ocr_export_path(_ocr_user_text, _ws)
                            structured = _last.get("structured") or {}
                            # Prefer applying the chat table if structured rows were lost
                            if isinstance(_last.get("user_table"), str) and _last.get("user_table").strip():
                                structured = dict(structured)
                                structured.setdefault("user_table", _last["user_table"])
                            written = export_ocr_extraction_to_excel(
                                structured,
                                out_path,
                                image_paths=_last.get("image_paths") or [],
                            )
                            n_rows = len(
                                [r for r in (structured.get("rows") or []) if isinstance(r, dict)]
                            )
                            return {
                                "query": query,
                                "response": (
                                    f"Saved the latest OCR extraction to `{written}`.\n"
                                    f"- Open the **Extraction** sheet first (metadata + {n_rows} observation row(s)).\n"
                                    f"- Sheets: Extraction, Observations, Metadata.\n\n"
                                    + (str(_last.get("user_table") or "").strip())
                                ).strip(),
                                "llm_used": llm_used,
                                "model_name": model_name_used,
                                "complexity": "simple",
                                "success": True,
                                "session_id": current_session_id,
                                "context_retrieved": False,
                                "output_path": str(written),
                            }
                        except Exception as export_exc:
                            logger.exception("OCR Excel export failed")
                            return {
                                "query": query,
                                "response": f"Could not save OCR extraction to Excel: {export_exc}",
                                "llm_used": llm_used,
                                "model_name": model_name_used,
                                "complexity": "simple",
                                "success": False,
                                "session_id": current_session_id,
                                "context_retrieved": False,
                                "error": str(export_exc),
                            }
                    if _last and is_ocr_word_export_request(_ocr_user_text):
                        llm_used = "fallback" if use_fallback else "primary"
                        try:
                            out_path = resolve_ocr_word_export_path(_ocr_user_text, _ws)
                            structured = _last.get("structured") if isinstance(_last.get("structured"), dict) else {}
                            written = export_ocr_extraction_to_docx(
                                structured,
                                out_path,
                                image_paths=_last.get("image_paths") or [],
                                user_table=str(_last.get("user_table") or ""),
                            )
                            return {
                                "query": query,
                                "response": (
                                    f"Saved the latest OCR extraction to Word.\n"
                                    f"- Output: `{written}`\n"
                                    f"- Content: the full extraction (headings, panels, tables) — "
                                    f"not an essay rewrite.\n"
                                ).strip(),
                                "llm_used": llm_used,
                                "model_name": model_name_used,
                                "complexity": "simple",
                                "success": True,
                                "session_id": current_session_id,
                                "context_retrieved": False,
                                "output_path": str(written),
                            }
                        except Exception as export_exc:
                            logger.exception("OCR Word export failed")
                            return {
                                "query": query,
                                "response": f"Could not save OCR extraction to Word: {export_exc}",
                                "llm_used": llm_used,
                                "model_name": model_name_used,
                                "complexity": "simple",
                                "success": False,
                                "session_id": current_session_id,
                                "context_retrieved": False,
                                "error": str(export_exc),
                            }
                    if _last and is_ocr_followup_request(_ocr_user_text):
                        self._vision_ocr_context = format_last_ocr_for_agent(_last)

                if _image_paths and _pre_intent != "knowledge":
                    _prov = self._active_llm_provider(use_fallback)
                    if not provider_supports_vision(_prov):
                        llm_used = "fallback" if use_fallback else "primary"
                        return {
                            "query": query,
                            "response": vision_unsupported_user_message(_prov),
                            "llm_used": llm_used,
                            "model_name": model_name_used,
                            "complexity": "average",
                            "success": False,
                            "session_id": current_session_id,
                            "context_retrieved": False,
                            "error": "vision_unsupported_provider",
                        }

                    if should_fastpath_image_survey_replot(
                        tool_routing_query, routing_query
                    ) and getattr(self.settings, "pdf_survey_replot_enabled", True):
                        logger.info("Image survey replot fast-path triggered")
                        fast = self._run_image_survey_replot_pipeline(tool_routing_query)
                        llm_used = "fallback" if use_fallback else "primary"
                        return {
                            "query": query,
                            "response": fast.get("response") or fast.get("error") or str(fast),
                            "llm_used": llm_used,
                            "model_name": fast.get("model_name") or model_name_used,
                            "complexity": "complex",
                            "success": bool(fast.get("success")),
                            "session_id": current_session_id,
                            "context_retrieved": False,
                            "output_path": fast.get("output_path"),
                            "error": fast.get("error") if not fast.get("success") else None,
                        }

                    logger.info(
                        "Vision OCR preprocessing for %d image(s)", len(_image_paths)
                    )
                    ocr_result = self._run_vision_ocr_for_query(tool_routing_query)
                    user_text_for_mode = _ocr_user_text
                    if getattr(ocr_result, "success", False) and getattr(ocr_result, "structured", None):
                        save_last_ocr_extraction(
                            ocr_result.structured,
                            workspace=_ws,
                            image_paths=getattr(ocr_result, "image_paths", None) or _image_paths,
                            document_type=getattr(ocr_result, "document_type", "") or "",
                            source="vision_ocr",
                        )
                    if is_ocr_only_request(user_text_for_mode):
                        llm_used = "fallback" if use_fallback else "primary"
                        response_text = (
                            ocr_result.format_for_user()
                            if getattr(ocr_result, "format_for_user", None)
                            else str(ocr_result)
                        )
                        if (
                            ocr_result.success
                            and ocr_result.structured
                            and (
                                is_ocr_export_request(user_text_for_mode)
                                or (
                                    user_requested_save(user_text_for_mode)
                                    and any(
                                        m in user_text_for_mode.lower()
                                        for m in ("excel", ".xlsx", ".xls", "spreadsheet", "workbook")
                                    )
                                )
                            )
                        ):
                            try:
                                out_path = resolve_ocr_export_path(user_text_for_mode, _ws)
                                export_ocr_extraction_to_excel(
                                    ocr_result.structured,
                                    out_path,
                                    image_paths=getattr(ocr_result, "image_paths", None) or _image_paths,
                                )
                                response_text += (
                                    f"\n\nSaved extraction to `{out_path}` "
                                    f"(open the Extraction sheet for metadata + observation rows)."
                                )
                            except Exception as save_exc:
                                logger.debug("OCR Excel save skipped: %s", save_exc)
                                response_text += f"\n\nCould not save Excel: {save_exc}"
                        elif (
                            ocr_result.success
                            and ocr_result.structured
                            and is_ocr_word_export_request(user_text_for_mode)
                        ):
                            try:
                                out_path = resolve_ocr_word_export_path(user_text_for_mode, _ws)
                                export_ocr_extraction_to_docx(
                                    ocr_result.structured,
                                    out_path,
                                    image_paths=getattr(ocr_result, "image_paths", None) or _image_paths,
                                    user_table=response_text,
                                )
                                response_text += (
                                    f"\n\nSaved extraction to Word: `{out_path}`."
                                )
                            except Exception as save_exc:
                                logger.debug("OCR Word save skipped: %s", save_exc)
                                response_text += f"\n\nCould not save Word: {save_exc}"
                        elif (
                            ocr_result.success
                            and user_requested_save(user_text_for_mode)
                            and ocr_result.structured
                        ):
                            try:
                                out_json = _ws / "ocr_result.json"
                                out_json.write_text(
                                    json.dumps(ocr_result.structured, indent=2, ensure_ascii=False),
                                    encoding="utf-8",
                                )
                                response_text += f"\n\nSaved structured OCR to `{out_json}`."
                            except Exception as save_exc:
                                logger.debug("OCR save skipped: %s", save_exc)
                        return {
                            "query": query,
                            "response": response_text,
                            "llm_used": llm_used,
                            "model_name": getattr(ocr_result, "model_name", None) or model_name_used,
                            "complexity": "average",
                            "success": bool(getattr(ocr_result, "success", False)),
                            "session_id": current_session_id,
                            "context_retrieved": False,
                            "error": getattr(ocr_result, "error", None)
                            if not getattr(ocr_result, "success", False)
                            else None,
                            "ocr_review": getattr(ocr_result, "ocr_review", None) or None,
                        }

                    if getattr(ocr_result, "success", False):
                        self._vision_ocr_context = ocr_result.format_for_agent_context()
                        # Same-turn: scan + save to Excel without requiring a second message
                        if is_ocr_export_request(user_text_for_mode) and ocr_result.structured:
                            try:
                                out_path = resolve_ocr_export_path(user_text_for_mode, _ws)
                                export_ocr_extraction_to_excel(
                                    ocr_result.structured,
                                    out_path,
                                    image_paths=getattr(ocr_result, "image_paths", None) or _image_paths,
                                )
                                self._vision_ocr_context += (
                                    f"\n\nAlready saved this extraction to `{out_path}`. "
                                    "Tell the user the file path; do not ask them for an existing Excel file."
                                )
                            except Exception as save_exc:
                                logger.debug("OCR Excel save (continue path) skipped: %s", save_exc)
                        elif is_ocr_word_export_request(user_text_for_mode) and ocr_result.structured:
                            try:
                                out_path = resolve_ocr_word_export_path(user_text_for_mode, _ws)
                                export_ocr_extraction_to_docx(
                                    ocr_result.structured,
                                    out_path,
                                    image_paths=getattr(ocr_result, "image_paths", None) or _image_paths,
                                    user_table=ocr_result.format_for_user(),
                                )
                                self._vision_ocr_context += (
                                    f"\n\nAlready saved this extraction to Word `{out_path}`. "
                                    "Tell the user the file path; do not rewrite it as an essay."
                                )
                            except Exception as save_exc:
                                logger.debug("OCR Word save (continue path) skipped: %s", save_exc)
                    elif getattr(ocr_result, "error", None):
                        # Soft-fail: continue agent loop with a note (path may still be useful).
                        self._vision_ocr_context = (
                            f"VISION OCR RESULT: failed — {ocr_result.error}. "
                            "If the user asked about this image, explain the failure clearly."
                        )

            prompt_action = self._assess_prompt_action(
                raw_query=query,
                routing_query=routing_query,
                permission_granted=bool(getattr(self, "_internet_permission_granted", False)),
            )
            logger.info(
                "🎯 Prompt assessment: kind=%s needs_internet=%s effective_query='%s' reason=%s",
                prompt_action.kind,
                prompt_action.needs_internet,
                (prompt_action.effective_query or "")[:120],
                prompt_action.reason,
            )

            intent = self._classify_query_intent(routing_query)
            if intent == "other" and _pre_intent != intent:
                intent = _pre_intent
            logger.info(f"🧭 Current-turn intent: {intent} | routing_query='{routing_query[:120]}'")

            # Tiered model selection: heuristics + explicit user phrasing + desktop fast-mode (non-file only)
            route_query = prompt_action.effective_query or routing_query
            complexity = self._detect_task_complexity(route_query)
            tier_override = self._parse_user_model_tier_override(routing_query)
            fast_mode_forced_simple = False
            if tier_override is not None:
                complexity = tier_override
                logger.info(f"Model tier from user request: {complexity}")
            elif (
                getattr(self.settings, "fast_mode_non_file_prompts", False)
                and not looks_like_file_driven_task(routing_query)
                and len(self._extract_document_paths(routing_query)) == 0
                and len(self._extract_image_paths(routing_query)) == 0
                and prompt_action.kind not in ("current_fact_lookup", "permission_affirm")
            ):
                complexity = "simple"
                fast_mode_forced_simple = True
                logger.info("Model tier: simple (fast_mode_non_file_prompts, non-file query)")
            else:
                logger.info(f"Detected task complexity (heuristic): {complexity}")

            from survyai.prompt_router import heuristic_route_confidence, should_use_llm_prompt_router
            from survyai.provider_models import PAID_PROVIDERS, is_elevated_average_task

            llm_to_use = None
            model_name_used = None
            llm_routed_model = None
            elevated_average = False
            active_provider = self._active_llm_provider(use_fallback)
            file_driven_route = looks_like_file_driven_task(
                route_query
            ) or bool(self._extract_document_paths(routing_query)) or bool(
                self._extract_image_paths(routing_query)
            )

            used_llm_router = False
            route_confidence = heuristic_route_confidence(
                route_query,
                complexity=complexity,
                file_driven=file_driven_route,
                intent=intent,
                kind=str(getattr(prompt_action, "kind", "") or ""),
            )
            if should_use_llm_prompt_router(
                provider=active_provider,
                enable_tiered=bool(getattr(self.settings, "enable_tiered_models", True)),
                enable_llm_prompt_router=bool(
                    getattr(self.settings, "enable_llm_prompt_router", True)
                ),
                user_tier_override=tier_override,
                fast_mode_forced_simple=fast_mode_forced_simple,
                heuristic_confidence=route_confidence,
            ):
                route = self._route_task_with_cheap_model(
                    provider=active_provider,
                    query=route_query,
                    heuristic_complexity=complexity,
                    kind=str(getattr(prompt_action, "kind", "") or ""),
                    file_driven=file_driven_route,
                    needs_internet=bool(getattr(prompt_action, "needs_internet", False)),
                )
                if route is not None:
                    used_llm_router = True
                    complexity = route.complexity
                    elevated_average = bool(route.elevated_average)
                    llm_routed_model = route.model
                    logger.info(
                        "Cheap-model router (%s) chose %s (complexity=%s, elevated=%s): %s",
                        route.router_model or "unknown",
                        route.model,
                        route.complexity,
                        route.elevated_average,
                        route.reason or "n/a",
                    )
            elif (
                route_confidence in ("simple", "complex")
                and not fast_mode_forced_simple
                and tier_override is None
            ):
                logger.info(
                    "Skipping LLM prompt router (confident %s heuristic) — saved a round-trip",
                    route_confidence,
                )

            if not used_llm_router:
                # Assessment may require a stronger tier (e.g. factual web synthesis).
                _tier_rank = {"simple": 0, "average": 1, "complex": 2}
                if _tier_rank.get(prompt_action.min_complexity, 0) > _tier_rank.get(complexity, 0):
                    complexity = prompt_action.min_complexity
                    logger.info("Model tier raised by prompt assessment: %s", complexity)
                elevated_average = bool(
                    complexity == "average"
                    and is_elevated_average_task(route_query)
                )
                if elevated_average:
                    logger.info(
                        "Elevated-average reasoning detected → prefer stronger mid-tier "
                        "(e.g. gpt-5.5 on OpenAI) for accuracy"
                    )

            logger.info(f"Final task complexity for model selection: {complexity}")

            # Determine which LLM and model to use (tiered for all paid providers)
            if use_fallback:
                logger.warning(
                    f"⚠ Using fallback LLM ({self.settings.fallback_llm}) instead of "
                    f"primary ({self.settings.primary_llm})"
                )
                llm_to_use = self.llm_fallback
                if active_provider in PAID_PROVIDERS and getattr(
                    self.settings, "enable_tiered_models", True
                ):
                    model_name = llm_routed_model or self._get_model_for_complexity(
                        active_provider,
                        complexity,
                        elevated_average=elevated_average,
                    )
                    llm_to_use = self._initialize_llm(active_provider, model_name=model_name)
                    model_name_used = model_name
                    if active_provider == "openai":
                        self._current_openai_model = model_name
                    elif active_provider == "gemini":
                        self._current_gemini_model = model_name
                    logger.info(
                        "Using %s fallback model: %s (complexity: %s)",
                        active_provider,
                        model_name,
                        complexity,
                    )
                else:
                    model_name_used = self._resolve_provider_model_name(active_provider)
            else:
                if active_provider in PAID_PROVIDERS and getattr(
                    self.settings, "enable_tiered_models", True
                ):
                    model_name = llm_routed_model or self._get_model_for_complexity(
                        active_provider,
                        complexity,
                        elevated_average=elevated_average,
                    )
                    llm_to_use = self._initialize_llm(active_provider, model_name=model_name)
                    model_name_used = model_name
                    if active_provider == "openai":
                        self._current_openai_model = model_name
                    elif active_provider == "gemini":
                        self._current_gemini_model = model_name
                    logger.info(
                        "✓ Using %s model: %s (complexity: %s)",
                        active_provider,
                        model_name,
                        complexity,
                    )
                else:
                    # Untiered paid provider, or local Ollama
                    llm_to_use = self.llm_primary
                    model_name_used = self._resolve_provider_model_name(active_provider)
                    if active_provider == "openai":
                        self._current_openai_model = model_name_used
                    elif active_provider == "gemini":
                        self._current_gemini_model = model_name_used
                    logger.info(
                        "✓ Using primary LLM: %s (model: %s)",
                        active_provider,
                        model_name_used,
                    )

            # Host RAM hard-cap for local Ollama before any heavy prompt/tool work.
            if self._llm_is_ollama(llm_to_use) or str(
                self.settings.fallback_llm if use_fallback else self.settings.primary_llm
            ).lower() == "ollama":
                ollama_model = model_name_used or self._resolve_provider_model_name("ollama")
                ok, ram_err, _num_ctx = _ollama_ram_policy(ollama_model)
                if not ok:
                    return {
                        "query": query,
                        "response": ram_err,
                        "success": False,
                        "error": ram_err,
                        "llm_used": "fallback" if use_fallback else "primary",
                        "model_name": ollama_model,
                        "session_id": current_session_id,
                    }

            # FAST PATH: standalone knowledge/explanation prompts.
            # This is intentionally before any tool-enabled graph execution to avoid
            # stale CAD/file context and multi-minute planner loops for simple Q&A.
            if self._should_direct_answer_non_file_prompt(routing_query, prompt_action, intent):
                direct = self._run_direct_knowledge_answer(
                    question=prompt_action.effective_query or routing_query,
                    llm=llm_to_use,
                    model_name_used=model_name_used,
                    timeout_seconds=60,
                )
                llm_used = "fallback" if use_fallback else "primary"
                return {
                    "query": query,
                    "response": direct.get("response", "") or direct.get("error", ""),
                    "llm_used": llm_used,
                    "model_name": direct.get("model_name", model_name_used),
                    "complexity": complexity,
                    "success": bool(direct.get("success")),
                    "session_id": current_session_id,
                    "context_retrieved": False,
                    "error": direct.get("error") if not direct.get("success") else None,
                }

            # FAST PATH: learn/register a cadastral CAD template only.
            # This must not go through the LLM; users often do this once after a
            # fresh install so later plan-generation prompts can omit the path.
            if self._should_fastpath_cadastral_template_registration(tool_routing_query):
                fast = self._run_cadastral_template_registration_pipeline(tool_routing_query)
                llm_used = "fallback" if use_fallback else "primary"
                return {
                    "query": query,
                    "response": fast.get("response", str(fast)),
                    "llm_used": llm_used,
                    "model_name": model_name_used,
                    "complexity": complexity,
                    "success": bool(fast.get("success")),
                    "session_id": current_session_id,
                    "context_retrieved": False,
                    "output_path": fast.get("template_path"),
                    "error": fast.get("error") if not fast.get("success") else None,
                }

            # FAST PATH: factual web lookup (current office holders, who-is questions)
            # Runs after permission is granted — one search + one synthesis call, no tool loop.
            if (
                getattr(self, "_internet_permission_granted", False)
                and prompt_action.needs_internet
                and prompt_action.kind in ("current_fact_lookup", "permission_affirm", "general_knowledge")
                and llm_to_use is not None
            ):
                # search_queries=None → the pipeline runs LLM-based query rewriting
                # (with rule-based fallback) for the best, domain-agnostic retrieval.
                fast = self._run_factual_web_lookup_pipeline(
                    question=prompt_action.effective_query or routing_query,
                    llm=llm_to_use,
                    model_name_used=model_name_used,
                    search_queries=None,
                )
                llm_used = "fallback" if use_fallback else "primary"
                if fast.get("success"):
                    return {
                        "query": query,
                        "response": fast.get("response", ""),
                        "llm_used": llm_used,
                        "model_name": fast.get("model_name", model_name_used),
                        "complexity": complexity,
                        "success": True,
                        "session_id": current_session_id,
                        "context_retrieved": False,
                        "internet_searched": True,
                    }
                # Permission was granted — do NOT fall through to the LangGraph loop
                # (it would ask for permission again). Return the search failure plainly.
                if prompt_action.kind in ("permission_affirm", "current_fact_lookup"):
                    return {
                        "query": query,
                        "response": fast.get("response", "Web search returned no usable results."),
                        "llm_used": llm_used,
                        "model_name": model_name_used,
                        "complexity": complexity,
                        "success": False,
                        "error": fast.get("error", "no_web_results"),
                        "session_id": current_session_id,
                        "context_retrieved": False,
                    }
                logger.warning(
                    "Factual web lookup fast-path did not return results: %s",
                    fast.get("error"),
                )

            # FAST PATH: multi-DWG survey plan extract → Word (per-file loaders, no agent loop)
            if self._should_fastpath_dwg_plan_extract_to_docx(tool_routing_query):
                logger.info("DWG plan extract → Word fast-path triggered")
                fast = self._run_dwg_plan_extract_to_docx_pipeline(tool_routing_query)
                llm_used = "fallback" if use_fallback else "primary"
                return {
                    "query": query,
                    "response": fast.get("response", ""),
                    "llm_used": llm_used,
                    "model_name": model_name_used,
                    "complexity": complexity,
                    "success": bool(fast.get("success")),
                    "session_id": self.get_session_id(),
                    "context_retrieved": False,
                    "output_path": fast.get("output_path"),
                    "error": fast.get("error") if not fast.get("success") else None,
                }

            # FAST PATH: save prior session answer (essay/report) to .docx
            if self._should_fastpath_save_session_docx(tool_routing_query, query):
                fast = self._run_save_session_docx_pipeline(
                    query=query,
                    routing_query=routing_query,
                    llm=llm_to_use,
                    model_name_used=model_name_used,
                )
                llm_used = "fallback" if use_fallback else "primary"
                return {
                    "query": query,
                    "response": fast.get("response", ""),
                    "llm_used": llm_used,
                    "model_name": fast.get("model_name", model_name_used),
                    "complexity": complexity,
                    "success": bool(fast.get("success")),
                    "session_id": self.get_session_id(),
                    "context_retrieved": False,
                    "output_path": fast.get("output_path"),
                    "error": fast.get("error") if not fast.get("success") else None,
                }

            # FAST PATH: report generation to .docx (avoids LangGraph recursion/tool loops)
            # Never hijack operational GIS/CAD/Excel convert-plot-buffer workflows.
            if (
                self._should_fastpath_docx_report(tool_routing_query)
                and not self._looks_like_operational_workflow_request(tool_routing_query)
            ):
                out_candidate = self._extract_any_output_docx(tool_routing_query) or "Report.docx"
                fast = self._run_docx_report_pipeline(
                    query=tool_routing_query,
                    output_doc_path=out_candidate,
                    llm=llm_to_use,
                    model_name_used=model_name_used or getattr(self.settings, "openai_model_mini", "gpt-5.4-mini"),
                )
                llm_used = "fallback" if use_fallback else "primary"
                return {
                    "query": query,
                    "response": fast.get("response", ""),
                    "llm_used": llm_used,
                    "model_name": fast.get("model_name", model_name_used),
                    "complexity": complexity,
                    "success": bool(fast.get("success")),
                    "session_id": self.get_session_id(),
                    "context_retrieved": False,
                    "output_path": fast.get("output_path"),
                    "error": fast.get("error") if not fast.get("success") else None,
                }

            # FAST PATH: Excel ownership workbook → multi-parcel OR per-owner cadastral DWG(s)
            if self._should_fastpath_excel_cadastral(tool_routing_query):
                fast = self._run_excel_cadastral_pipeline(tool_routing_query)
                llm_used = "fallback" if use_fallback else "primary"
                if fast.get("success"):
                    if fast.get("mode") == "separate_owner_plans":
                        resp_lines = [
                            "✅ Separate owner cadastral plans generated from Excel coordinates "
                            "(one DWG per owner — not a multi-parcel layout).",
                            f"- Plans succeeded: {fast.get('plans_success')}/{fast.get('plans_total')}",
                            f"- Source Excel: {fast.get('excel_path')}",
                            f"- Duplicate workbook: {fast.get('dup_xlsx')}",
                            f"- Owners: {fast.get('buyer_name')}",
                        ]
                        for plan in (fast.get("plans") or [])[:25]:
                            status = "OK" if plan.get("success") else "FAIL"
                            resp_lines.append(
                                f"  • [{status}] {plan.get('owner_name')}: "
                                f"{plan.get('output_dwg')}"
                                + (
                                    f" (plan no. {plan.get('plan_number')})"
                                    if plan.get("plan_number")
                                    else ""
                                )
                            )
                    else:
                        resp_lines = [
                            "✅ Multi-parcel cadastral plan generated from Excel coordinates.",
                            f"- Output: {fast.get('output_dwg')}",
                            f"- Source Excel: {fast.get('excel_path')}",
                            f"- Duplicate workbook: {fast.get('dup_xlsx')}",
                            f"- Parcels plotted: {fast.get('parcel_count')}",
                            f"- Buyer/owner labels: {fast.get('buyer_name')}",
                        ]
                    if fast.get("reference_dwg"):
                        resp_lines.append(f"- Metadata source plan: {fast.get('reference_dwg')}")
                    for note in fast.get("notes") or []:
                        resp_lines.append(f"- Note: {note}")
                    if fast.get("access_road_title"):
                        resp_lines.append(f"- Access road title (as plotted): {fast.get('access_road_title')!r}")
                    resp_lines.append(
                        "\nYou can request modifications in this session "
                        "(e.g. add a road, change the title) without re-prompting."
                    )
                    return {
                        "query": query,
                        "response": "\n".join(resp_lines) + "\n",
                        "llm_used": llm_used,
                        "model_name": model_name_used,
                        "complexity": complexity,
                        "success": True,
                        "session_id": self.get_session_id(),
                        "context_retrieved": False,
                        "output_path": fast.get("output_dwg"),
                    }
                # User declined overwrite / file still locked: stop here (do not re-prompt via compose).
                err_txt = str(fast.get("error") or "")
                if (
                    fast.get("cancelled")
                    or fast.get("locked")
                    or "Kept existing drawing" in err_txt
                    or "still locked" in err_txt.lower()
                ):
                    return {
                        "query": query,
                        "response": err_txt or "CAD generation stopped.",
                        "llm_used": llm_used,
                        "model_name": model_name_used,
                        "complexity": complexity,
                        "success": False,
                        "session_id": self.get_session_id(),
                        "context_retrieved": False,
                        "error": err_txt,
                    }
                # Excel failed → try intelligent multi-format compose before full agent loop.
                logger.warning(
                    "Excel cadastral fastpath failed (%s); trying intelligent composition.",
                    fast.get("error"),
                )
                intel = self._run_intelligent_cadastral_pipeline(tool_routing_query)
                if intel.get("success"):
                    if intel.get("mode") == "separate_owner_plans":
                        resp_lines = [
                            "✅ Separate owner cadastral plans generated via intelligent composition "
                            "(one DWG per owner).",
                            f"- Plans succeeded: {intel.get('plans_success')}/{intel.get('plans_total')}",
                            f"- Sources: {', '.join(intel.get('source_files') or [])}",
                            f"- Owners: {intel.get('buyer_name')}",
                        ]
                        for plan in (intel.get("plans") or [])[:25]:
                            status = "OK" if plan.get("success") else "FAIL"
                            resp_lines.append(
                                f"  • [{status}] {plan.get('owner_name')}: {plan.get('output_dwg')}"
                            )
                    else:
                        resp_lines = [
                            "✅ Cadastral plan generated via intelligent file composition.",
                            f"- Output: {intel.get('output_dwg')}",
                            f"- Sources: {', '.join(intel.get('source_files') or [])}",
                            f"- Compose mode: {intel.get('compose_source')}",
                            f"- Parcels plotted: {intel.get('parcel_count')}",
                            f"- Buyer/owner labels: {intel.get('buyer_name')}",
                        ]
                    if intel.get("dup_xlsx"):
                        resp_lines.append(f"- Duplicate workbook: {intel.get('dup_xlsx')}")
                    if intel.get("reference_dwg"):
                        resp_lines.append(f"- Metadata source plan: {intel.get('reference_dwg')}")
                    for note in intel.get("notes") or []:
                        resp_lines.append(f"- Note: {note}")
                    return {
                        "query": query,
                        "response": "\n".join(resp_lines) + "\n",
                        "llm_used": llm_used,
                        "model_name": model_name_used,
                        "complexity": complexity,
                        "success": True,
                        "session_id": self.get_session_id(),
                        "context_retrieved": False,
                        "output_path": intel.get("output_dwg"),
                    }
                logger.warning(
                    "Intelligent cadastral composition also failed (%s); falling through to agent tools.",
                    intel.get("error"),
                )

            # FAST PATH: complex / CSV|TXT|DOCX deferred coords → RAG+LLM compose → CAD plot
            # (Simple inline cadastral prompts never enter here — deferred detection is False.)
            if self._should_fastpath_intelligent_cadastral(tool_routing_query):
                fast = self._run_intelligent_cadastral_pipeline(tool_routing_query)
                llm_used = "fallback" if use_fallback else "primary"
                if fast.get("success"):
                    if fast.get("mode") == "separate_owner_plans":
                        resp_lines = [
                            "✅ Separate owner cadastral plans generated via intelligent composition "
                            "(one DWG per owner).",
                            f"- Plans succeeded: {fast.get('plans_success')}/{fast.get('plans_total')}",
                            f"- Sources: {', '.join(fast.get('source_files') or [])}",
                            f"- Owners: {fast.get('buyer_name')}",
                        ]
                        for plan in (fast.get("plans") or [])[:25]:
                            status = "OK" if plan.get("success") else "FAIL"
                            resp_lines.append(
                                f"  • [{status}] {plan.get('owner_name')}: {plan.get('output_dwg')}"
                            )
                    else:
                        resp_lines = [
                            "✅ Cadastral plan generated via intelligent file composition.",
                            f"- Output: {fast.get('output_dwg')}",
                            f"- Sources: {', '.join(fast.get('source_files') or [])}",
                            f"- Compose mode: {fast.get('compose_source')}",
                            f"- Parcels plotted: {fast.get('parcel_count')}",
                            f"- Buyer/owner labels: {fast.get('buyer_name')}",
                        ]
                    if fast.get("dup_xlsx"):
                        resp_lines.append(f"- Duplicate workbook: {fast.get('dup_xlsx')}")
                    if fast.get("reference_dwg"):
                        resp_lines.append(f"- Metadata source plan: {fast.get('reference_dwg')}")
                    for note in fast.get("notes") or []:
                        resp_lines.append(f"- Note: {note}")
                    if fast.get("access_road_title"):
                        resp_lines.append(f"- Access road title (as plotted): {fast.get('access_road_title')!r}")
                    resp_lines.append(
                        "\nYou can request modifications in this session "
                        "(e.g. add a road, change the title) without re-prompting."
                    )
                    return {
                        "query": query,
                        "response": "\n".join(resp_lines) + "\n",
                        "llm_used": llm_used,
                        "model_name": model_name_used,
                        "complexity": complexity,
                        "success": True,
                        "session_id": self.get_session_id(),
                        "context_retrieved": False,
                        "output_path": fast.get("output_dwg"),
                    }
                logger.warning(
                    "Intelligent cadastral fastpath failed (%s); falling through to agent tools.",
                    fast.get("error"),
                )

            # FAST PATH: cadastral CAD prompt (template DWG -> output DWG with parcel replot)
            if self._should_fastpath_cadastral_cad_batch(tool_routing_query):
                fast = self._run_cadastral_cad_batch_pipeline(tool_routing_query)
                llm_used = "fallback" if use_fallback else "primary"
                if fast.get("success"):
                    res = fast.get("results") or []
                    lines = [
                        "✅ Batch cadastral plotting completed.",
                        f"- Plans requested: {fast.get('plans_total')}",
                        f"- Successful: {fast.get('plans_success')}",
                        f"- Failed: {fast.get('plans_failed')}",
                        "",
                        "Outputs:",
                    ]
                    for item in res:
                        idx = item.get("_plan_index")
                        if item.get("success"):
                            lines.append(f"- Plan {idx}: {item.get('output_dwg')}")
                        else:
                            err = item.get("error") or "Failed"
                            lines.append(f"- Plan {idx}: FAILED ({err})")
                    lines.append("\nYou can request modifications in this session for the last successful plan (e.g. add road, change title).")
                    return {
                        "query": query,
                        "response": "\n".join(lines) + "\n",
                        "llm_used": llm_used,
                        "model_name": model_name_used,
                        "complexity": complexity,
                        "success": True,
                        "session_id": self.get_session_id(),
                        "context_retrieved": False,
                        "output_path": None,
                    }
                return {
                    "query": query,
                    "response": str(fast),
                    "llm_used": llm_used,
                    "model_name": model_name_used,
                    "complexity": complexity,
                    "success": False,
                    "session_id": self.get_session_id(),
                    "context_retrieved": False,
                    "output_path": None,
                    "error": fast.get("error") if isinstance(fast, dict) else "Batch cadastral pipeline failed",
                }

            if self._should_fastpath_cadastral_cad(tool_routing_query):
                fast = self._run_cadastral_cad_prompt_pipeline(tool_routing_query)
                llm_used = "fallback" if use_fallback else "primary"
                if fast.get("success"):
                    self._last_cadastral_output_dwg = fast.get("output_dwg")
                    self._last_cadastral_profile_path = fast.get("profile_path")
                    resp_lines = [
                        "✅ Cadastral plan generated from template.",
                        f"- Output: {fast.get('output_dwg')}",
                        f"- Geometry: {fast.get('geometry')}",
                    ]
                    if fast.get("access_road_title"):
                        resp_lines.append(f"- Access road title (as plotted): {fast.get('access_road_title')!r}")
                    try:
                        bow = (fast.get("geometry") or {}).get("bowditch") if isinstance(fast, dict) else None
                        if isinstance(bow, dict) and bow.get("mode") == "bearing_distance":
                            if bow.get("applied"):
                                resp_lines.append(
                                    "- Bowditch adjustment applied (misclosure > 1cm): "
                                    f"misclosure={bow.get('misclosure_m'):.3f}m "
                                    f"(E={bow.get('misclosure_e_m'):.3f}m, N={bow.get('misclosure_n_m'):.3f}m), "
                                    f"max point shift={bow.get('max_point_shift_m'):.3f}m."
                                )
                                prev = bow.get("adjusted_points_preview") or []
                                if prev:
                                    resp_lines.append(f"- Adjusted points preview (first {len(prev)}): {prev}")
                            else:
                                resp_lines.append(
                                    "- Bowditch adjustment not applied: "
                                    f"misclosure={bow.get('misclosure_m'):.3f}m (<= 0.010m threshold)."
                                )
                    except Exception:
                        pass
                    resp_lines.append("\nYou can request modifications in this session (e.g. add another road, change the title) without closing or re-prompting.")
                    return {
                        "query": query,
                        "response": "\n".join(resp_lines) + "\n",
                        "llm_used": llm_used,
                        "model_name": model_name_used,
                        "complexity": complexity,
                        "success": True,
                        "session_id": self.get_session_id(),
                        "context_retrieved": False,
                        "output_path": fast.get("output_dwg"),
                    }
                # Do not hard-fail when the prompt only *mentioned* coordinates but left
                # values in Excel/files — let the agent compose a valid cadastral prompt.
                err_txt = str((fast or {}).get("error") or "")
                if coordinates_deferred_to_external_source(tool_routing_query) or (
                    "could not parse coordinates" in err_txt.lower()
                    and any(
                        k in (tool_routing_query or "").lower()
                        for k in ("excel", ".xlsx", ".xls", "spreadsheet")
                    )
                ):
                    logger.warning(
                        "Cadastral fastpath failed on deferred/Excel coords (%s); "
                        "falling through to agent tools.",
                        err_txt,
                    )
                else:
                    return {
                        "query": query,
                        "response": str(fast),
                        "llm_used": llm_used,
                        "model_name": model_name_used,
                        "complexity": complexity,
                        "success": False,
                        "session_id": self.get_session_id(),
                        "context_retrieved": False,
                        "output_path": None,
                        "error": fast.get("error") if isinstance(fast, dict) else "Fastpath cadastral pipeline failed",
                    }

            # FAST PATH: in-session CAD plan modifications (add road, change title, etc.)
            # Template remains read-only; modifications apply to the output plan file (even if open).
            # CRITICAL: gate on routing_query (current request only) so injected
            # conversation history can no longer trigger this.  This is the exact
            # path that previously fired on a knowledge question because the
            # injected history contained "add the road".  The `intent != knowledge`
            # check is belt-and-suspenders against clearly informational questions.
            if intent != "knowledge" and self._should_fastpath_cad_modification(tool_routing_query):
                mod = self._run_cad_modification_pipeline(tool_routing_query)
                llm_used = "fallback" if use_fallback else "primary"
                if mod.get("success"):
                    resp_lines = [
                        "✅ Plan updated.",
                        f"- File: {mod.get('output_dwg')}",
                        f"- Modifications: {mod.get('modifications', [])}",
                    ]
                    if mod.get("save_warning"):
                        resp_lines.append(f"- Note: {mod.get('save_warning')}")
                    return {
                        "query": query,
                        "response": "\n".join(resp_lines) + "\n",
                        "llm_used": llm_used,
                        "model_name": model_name_used,
                        "complexity": complexity,
                        "success": True,
                        "session_id": self.get_session_id(),
                        "context_retrieved": False,
                        "output_path": mod.get("output_dwg"),
                    }
                return {
                    "query": query,
                    "response": str(mod.get("error", mod)),
                    "llm_used": llm_used,
                    "model_name": model_name_used,
                    "complexity": complexity,
                    "success": False,
                    "session_id": self.get_session_id(),
                    "context_retrieved": False,
                    "output_path": None,
                    "error": mod.get("error") if isinstance(mod, dict) else "CAD modification failed",
                }

            # AUTOMATIC DOCUMENT PRE-PROCESSING: Detect document paths and get resource estimation
            # This prevents the agent from trying to process large documents without knowing the cost/size.
            # Use routing_query so only documents referenced in the CURRENT request are
            # pre-processed (prevents stale docs from prior turns leaking in).
            document_paths = self._extract_document_paths(tool_routing_query)
            document_preflight_info = []
            
            if document_paths:
                logger.info(f"Detected {len(document_paths)} document path(s) in query - running pre-flight checks")
                for doc_path in document_paths:
                    try:
                        path_obj = Path(doc_path)
                        file_size_mb = path_obj.stat().st_size / (1024 * 1024) if path_obj.exists() else 0
                        model_for_est = model_name_used or getattr(self.settings, "openai_model_mini", "gpt-5.4-mini")
                        est = self.document_processor.get_resource_estimation(doc_path, model_for_est)
                        if est.get("success"):
                            doc_info = {
                                "path": doc_path,
                                "file_size_mb": est.get("file_size_mb", file_size_mb),
                                "page_count": est.get("page_count", 0),
                                "word_count": est.get("word_count", 0),
                                "estimated_tokens": est.get("estimated_tokens", 0),
                                "warnings": est.get("warnings", []),
                                "recommendations": est.get("recommendations", []),
                                "estimated_cost": est.get("estimated_cost", {}),
                                "estimation": est,
                            }
                            document_preflight_info.append(doc_info)
                            logger.info(f"Pre-flight check for {doc_path}: {doc_info['page_count']} pages, {doc_info['word_count']} words, ~{doc_info['estimated_tokens']} tokens")
                        else:
                            # Fallback: estimation failed (e.g., very large/table-heavy doc) - use file size for routing
                            doc_info = {
                                "path": doc_path,
                                "file_size_mb": file_size_mb,
                                "page_count": 0,
                                "word_count": 0,
                                "estimated_tokens": 0,
                                "estimation": {"success": False},
                            }
                            document_preflight_info.append(doc_info)
                            logger.info(f"Pre-flight fallback for {doc_path}: {file_size_mb:.2f} MB (estimation unavailable)")
                    except Exception as e:
                        logger.warning(f"Could not get resource estimation for {doc_path}: {e}")
                        # Still add minimal info so file-size-based fast-path can trigger
                        try:
                            path_obj = Path(doc_path)
                            file_size_mb = path_obj.stat().st_size / (1024 * 1024) if path_obj.exists() else 0
                            document_preflight_info.append({
                                "path": doc_path,
                                "file_size_mb": file_size_mb,
                                "page_count": 0,
                                "word_count": 0,
                                "estimated_tokens": 0,
                                "estimation": {"success": False},
                            })
                        except Exception:
                            pass

                # FAST PATH: if this is a large-document summarize/save request, run a deterministic pipeline
                # Even if multiple .docx are mentioned (e.g., output filename), pick the *largest* doc as the input.
                if document_preflight_info:
                    # Pick largest doc by tokens, then by file size (for estimation failures)
                    primary_doc = max(
                        document_preflight_info,
                        key=lambda d: (int(d.get("estimated_tokens") or 0), float(d.get("file_size_mb") or 0))
                    )
                    if self._should_fastpath_large_doc_summary(tool_routing_query, primary_doc):
                        input_doc = primary_doc["path"]
                        output_doc = self._extract_requested_output_docx(query, input_doc) or str(
                            (Path(input_doc).parent / f"Summary_{Path(input_doc).stem}.docx").resolve()
                        )
                        logger.info("Using fast-path large document summary pipeline")
                        fast_result = self._run_large_doc_summary_pipeline(
                            query=tool_routing_query,
                            input_doc_path=input_doc,
                            output_doc_path=output_doc,
                            llm=llm_to_use,
                            model_name_used=model_name_used or getattr(self.settings, "openai_model_mini", "gpt-5.4-mini")
                        )
                        if fast_result.get("success"):
                            # Store conversation for future context
                            llm_used = "fallback" if use_fallback else "primary"
                            self._store_conversation(
                                query=query,
                                response=fast_result.get("response", ""),
                                session_id=current_session_id,
                                llm_used=llm_used
                            )
                            return {
                                "query": query,
                                "response": fast_result.get("response", ""),
                                "llm_used": llm_used,
                                "model_name": fast_result.get("model_name", model_name_used),
                                "complexity": complexity,
                                "success": True,
                                "session_id": current_session_id,
                                "context_retrieved": False
                            }
            
            # Agentic RAG routing: decide whether to use VectorStore retrieval and/or Internet search.
            # Use cleaned query (without permission tags) for routing decision
            # NOTE: We already checked permission above, so if we reach here and internet is needed, permission is granted
            rag_decision = early_rag_decision  # Reuse the early decision

            internet_block = ""
            internet_already_searched = False  # Track if we already did internet search
            if rag_decision.use_internet:
                # Permission should already be granted (checked above), but double-check
                if getattr(self, "_internet_permission_granted", False):
                    try:
                        search_question = self._extract_clean_question(
                            rag_decision.internet_query or actual_query_for_routing or query
                        ) or (rag_decision.internet_query or query)
                        logger.info(f"🔍 Executing multi-stage web research for: {search_question}")
                        variants = self._rewrite_search_queries_with_llm(search_question, llm_to_use)
                        _max_sources = int(getattr(self.settings, "web_research_max_sources", 8) or 8)
                        _fetch_pages = int(getattr(self.settings, "web_research_fetch_pages", 4) or 0)
                        pack = _web_research(
                            search_question,
                            query_variants=variants,
                            max_sources=_max_sources,
                            fetch_pages=_fetch_pages,
                            read_pages=_fetch_pages > 0,
                        )
                        if pack.get("success") and (pack.get("evidence") or []):
                            internet_block = self._format_evidence_pack(pack.get("evidence") or [])
                            internet_block = (
                                "\n\n---\n**INTERNET EVIDENCE PACK (EXTERNAL, PERMISSION GRANTED — "
                                f"retrieval confidence {float(pack.get('confidence') or 0.0):.2f}):**\n"
                                "Ground every external claim in these numbered sources and cite [n]. "
                                "Add a section titled exactly: \"Internet-sourced (external) information\" "
                                "listing the URLs used.\n\n"
                                + internet_block
                                + "\n---\n"
                            )
                            internet_already_searched = True
                            self._internet_already_searched_this_query = True
                            logger.info(
                                "✓ Web research completed: %d evidence items, %d domains, confidence %.2f",
                                len(pack.get("evidence") or []),
                                pack.get("distinct_domains", 0),
                                float(pack.get("confidence") or 0.0),
                            )
                        else:
                            logger.warning(f"⚠ Web research returned no usable evidence: {pack.get('error', 'Unknown error')}")
                    except Exception as e:
                        logger.warning(f"⚠ Web research failed during routing: {e}")
                else:
                    # This shouldn't happen (we checked above), but log it
                    logger.warning("⚠ Internet needed but permission not granted - this should have been caught earlier")
            else:
                # Reset flag if internet is not needed
                self._internet_already_searched_this_query = False

            retrieved_context = ""
            if rag_decision.use_vector:
                retrieved_context = self._retrieve_relevant_context(
                    query, current_session_id, collections=rag_decision.vector_collections or None
                )
            context_retrieved = bool(retrieved_context)
            
            # Build enhanced system prompt with routed augmentation + document pre-flight info
            enhanced_system_prompt = self._system_prompt
            active_workspace = Path.cwd().resolve()
            enhanced_system_prompt += (
                "\n\n---\n"
                "**ACTIVE SURVYAI WORKSPACE (DEFAULT OUTPUT LOCATION):**\n"
                f"- Path: `{active_workspace}`\n"
                "- Use this when the user does not name another folder or full output file path.\n"
                "- If the user names an explicit folder (e.g. \"in the folder 'C:/path/to/dir'\") or a full "
                "output path, that overrides this workspace.\n"
                "- If output location is ambiguous, prefer this workspace.\n"
                "---\n"
            )
            vision_ctx = getattr(self, "_vision_ocr_context", None)
            if vision_ctx:
                enhanced_system_prompt += (
                    "\n\n---\n"
                    f"**{vision_ctx}**\n"
                    "---\n"
                )
            # Session GIS artifacts for fit / open-result follow-ups (anti-context-loss).
            try:
                _gis_arts = self._extract_verified_gis_artifacts_from_history(query)
                _fit_now = self._looks_like_gis_followup_analysis(
                    self._extract_clean_question(actual_query_for_routing) or actual_query_for_routing
                ) or "CONTINUATION (retain session GIS context" in (actual_query_for_routing or "")
                if _gis_arts and _fit_now:
                    enhanced_system_prompt += (
                        "\n\n---\n"
                        "**SESSION GIS ARTIFACTS (VERIFIED EARLIER IN THIS CHAT — REUSE, DO NOT RE-ASK):**\n"
                        + "\n".join(f"- `{p}`" for p in _gis_arts)
                        + "\n- Prefer geopandas_execute on converted Excel Owner+X/Y or these GDB/CSV paths.\n"
                        "- arcgis_execute_python_code uses Pro's Python — local missing arcpy is NOT a blocker.\n"
                        "- Do NOT ask for .aprx/parcel paths already listed here.\n"
                        "---\n"
                    )
            except Exception as _gis_inj_exc:
                logger.debug("Session GIS artifact injection skipped: %s", _gis_inj_exc)

            if document_preflight_info:
                doc_context = "\n\n---\n**DOCUMENT PRE-FLIGHT ANALYSIS (AUTOMATIC):**\n"
                doc_context += "The following document(s) were detected in the user's query. Resource estimation has been performed:\n\n"
                for doc_info in document_preflight_info:
                    doc_context += f"**Document: {doc_info['path']}**\n"
                    doc_context += f"  • Size: {doc_info['file_size_mb']:.2f} MB\n"
                    doc_context += f"  • Pages: {doc_info['page_count']}\n"
                    doc_context += f"  • Words: {doc_info['word_count']:,}\n"
                    doc_context += f"  • Estimated tokens: {doc_info['estimated_tokens']:,}\n"
                    if doc_info.get('estimated_cost', {}).get('total_cost'):
                        doc_context += f"  • Estimated cost: ${doc_info['estimated_cost']['total_cost']:.4f}\n"
                    if doc_info.get('warnings'):
                        doc_context += f"  • Warnings: {'; '.join(doc_info['warnings'])}\n"
                    if doc_info.get('recommendations'):
                        doc_context += f"  • Recommendations: {'; '.join(doc_info['recommendations'])}\n"
                    doc_context += "\n"
                doc_context += "**CRITICAL INSTRUCTIONS (MANDATORY - TPM/RATE LIMIT PROTECTION):**\n"
                doc_context += "1. For ANY document with >50 pages, >25K words, or >50K estimated tokens: use document_extract_sections_by_keywords() ONLY. NEVER use document_get_text() or document_get_full_text().\n"
                doc_context += "2. For summarize/summary requests that save to .docx: extract sections by keywords (e.g. Location, Personnel, Purpose, Coordinates, Control Points, Projects) then write the summary from the extracted content only.\n"
                doc_context += "3. Full text extraction causes TPM overflow (500K limit) and will fail with 429 rate limit. ALWAYS prefer section extraction for large docs.\n"
                doc_context += "4. Process only the extracted sections - never attempt to load the entire document into context.\n"
                doc_context += "---\n"
                enhanced_system_prompt += doc_context
                logger.info("✓ Document pre-flight info injected into system prompt")
            
            if retrieved_context:
                enhanced_system_prompt += (
                    f"\n\n---\n"
                    f"**CONTEXT FROM PREVIOUS SESSIONS AND STORED DOCUMENTS:**\n"
                    f"The following context may be relevant to the user's query:\n\n"
                    f"{retrieved_context}\n"
                    f"---\n"
                    f"Use this context to provide more informed and consistent responses. "
                    f"If the context doesn't seem relevant to the current query, you may ignore it."
                )
                logger.info("✓ Context injected into system prompt")

            if internet_block:
                enhanced_system_prompt += internet_block
                # Add STRONG instruction to LLM: internet search already done, permission already granted, tool removed
                enhanced_system_prompt += (
                    "\n\n**CRITICAL - READ CAREFULLY:**\n"
                    "1. Internet search has ALREADY been performed with user permission.\n"
                    "2. The results are included above in the 'INTERNET SEARCH RESULTS' section.\n"
                    "3. The internet_search tool has been REMOVED from your available tools.\n"
                    "4. DO NOT ask for permission - permission was already granted.\n"
                    "5. DO NOT mention needing to search - the search is already done.\n"
                    "6. Use the information provided above to answer the user's query directly.\n"
                    "7. If you reference external information, cite it from the results above."
                )
                logger.info("✓ Internet results injected into system prompt (permission granted)")
            elif getattr(self, "_internet_permission_granted", False) or getattr(
                self, "_force_internet_search_this_query", False
            ):
                # Permission was granted this turn but evidence pack was empty / failed.
                # Still forbid re-asking — the LLM must call internet_search or proceed
                # with ArcGIS Living Atlas / ArcPy online content / honest offline fallback.
                enhanced_system_prompt += (
                    "\n\n---\n"
                    "**INTERNET PERMISSION STATUS: GRANTED FOR THIS TURN (ANTI-LOOP)**\n"
                    "- The user already answered YES to internet search permission.\n"
                    "- NEVER output: \"May I search the internet for up-to-date information? (yes/no)\"\n"
                    "- NEVER claim permission is missing or inactive.\n"
                    "- If you still need external catalog info: call `internet_search` immediately "
                    "OR use ArcGIS Living Atlas / ArcGIS Online / pretrained land-cover tools via "
                    "`arcgis_execute_python_code` with the session `.aprx`/`.gdb` paths.\n"
                    "- If Living Atlas / imagery / the pretrained model is unavailable after a real "
                    "attempt, state that honestly and fall back to verified parcel geometry — "
                    "do not invent deep-learning class areas.\n"
                    "---\n"
                )
                logger.info("✓ Internet-permission-granted anti-loop instructions injected (no evidence pack)")

            # Strong ArcGIS routing augmentation for the common "two files + DWG mask + IDW/CutFill" case.
            ql_for_arcgis = (actual_query_for_routing or query or "").lower()
            tabular_pair = ql_for_arcgis.count(".csv") >= 2 or ql_for_arcgis.count(".xlsx") >= 2
            wants_cutfill = "cutfill" in ql_for_arcgis or "cut fill" in ql_for_arcgis
            wants_tin = "tin" in ql_for_arcgis
            wants_idw = "idw" in ql_for_arcgis
            wants_volume = "volume" in ql_for_arcgis
            needs_custom_arcgis_workflow = (
                tabular_pair
                and ".dwg" in ql_for_arcgis
                and (wants_cutfill or wants_volume or wants_tin or wants_idw)
            )
            if needs_custom_arcgis_workflow:
                tin_hint = (
                    "- The user asked for **TIN** surfaces: prefer `arcgis_pre_post_csv_dwg_tin_volume` "
                    "(CreateTin + CutFill; falls back to IDW if TIN fails).\n"
                    if wants_tin
                    else ""
                )
                idw_hint = (
                    "- Prefer `arcgis_pre_post_csv_dwg_cutfill` for separate PRE/POST .csv or .xlsx "
                    "plus a boundary .dwg when IDW/CutFill is requested.\n"
                    if wants_idw or (wants_cutfill and not wants_tin)
                    else ""
                )
                enhanced_system_prompt += (
                    "\n\n---\n"
                    "**CRITICAL ARCGIS ROUTING FOR THIS QUERY:**\n"
                    "- The user provided separate PRE and POST tabular sources plus a DWG polygon/boundary "
                    "and wants surface/volume output in ArcGIS Pro.\n"
                    f"- Put CSV copies, ArcGIS project/GDB, and volume result CSV under the active workspace: "
                    f"`{active_workspace}` (or omit workspace_folder so tools default to Path.cwd()).\n"
                    f"{tin_hint}{idw_hint}"
                    "- Do NOT use `arcgis_fill_volume_idw_cutfill` when PRE and POST are in separate files "
                    "or when a DWG defines the mask/extent.\n"
                    "- You MUST call a tool — do NOT claim success without verified tool output and on-disk files.\n"
                    "- Only use `arcgis_execute_python_code` if no verified tool matches (e.g. unusual outputs).\n"
                    "- If you use `arcgis_execute_python_code`, follow the same pattern: headless ArcPy, RESULT_* stdout, save project.\n"
                    "- If ArcGIS returns `ERROR 010092: Invalid output extent`, repair extent/mask from the boundary and retry.\n"
                    "---\n"
                )
                logger.info("✓ ArcGIS custom-workflow routing instructions injected into system prompt")

            offer_text = (actual_query_for_routing or routing_query or "").lower()
            if any(
                k in offer_text
                for k in (
                    "coordinate conversion", "crs", "epsg", "transformation",
                    "pyproj", "converted_points", "wgs84", "wgs 84",
                )
            ) and not any(
                k in offer_text for k in ("cutfill", "cut fill", "pre/post", "pre and post", "volume workflow")
            ):
                enhanced_system_prompt += (
                    "\n\n---\n"
                    "**ACTIVE WORKFLOW: COORDINATE CONVERSION / CRS METADATA**\n"
                    "- Continue the coordinate-conversion thread only.\n"
                    "- Do NOT call `arcgis_fill_volume_idw_cutfill`, `arcgis_pre_post_csv_dwg_cutfill`, "
                    "or other volume/CutFill tools — converted XY files are not PRE/POST elevation surfaces.\n"
                    "- To document transformation parameters, use pyproj/CRS introspection or update the "
                    "existing Excel output — do not invent elevation columns.\n"
                    "---\n"
                )
                logger.info("✓ Coordinate-conversion workflow guard injected into system prompt")
            
            # Bind tools to the selected LLM and rebuild graph
            # CRITICAL: If internet was already searched, conditionally remove internet_search tool
            # to prevent the LLM from calling it again and causing loops
            tools_to_bind = self.tools
            if internet_already_searched:
                # Filter out internet_search tool to prevent redundant calls
                tools_to_bind = [t for t in self.tools if t.name != "internet_search"]
                logger.info("✓ Removed internet_search tool (already searched) to prevent loops")

            # Task-scoped packs: simple knowledge/historic Q&A → no tool schemas + short prompt.
            # GIS/CAD/file/complex stays on the full pack (accuracy first).
            try:
                from survyai.tool_scoping import lite_system_prompt, select_tool_scope

                _scope_q = actual_query_for_routing or routing_query or query or ""
                _tool_scope = select_tool_scope(
                    _scope_q,
                    complexity=complexity if "complexity" in locals() else "average",
                    intent=intent if "intent" in locals() else "other",
                    prompt_action=prompt_action if "prompt_action" in locals() else None,
                    file_driven=bool(
                        looks_like_file_driven_task(_scope_q)
                        or self._extract_document_paths(_scope_q)
                        or self._extract_image_paths(_scope_q)
                    ),
                )
            except Exception:
                _tool_scope = "full"
            if _tool_scope == "lite":
                tools_to_bind = []
                enhanced_system_prompt = lite_system_prompt(base_full_prompt=self._system_prompt)
                logger.info("✓ Tool/prompt scope: lite (simple knowledge — no tool schemas)")

            # Ollama + full tool schemas massively inflate prompt/KV-cache RAM.
            # For non-file prompts, run tool-free to protect the host.
            if self._llm_is_ollama(llm_to_use) and not looks_like_file_driven_task(
                actual_query_for_routing or routing_query or query or ""
            ):
                tools_to_bind = []
                logger.info("✓ Ollama non-file prompt: binding no tools (RAM-safe)")

            # Bind tools and compile graph only if (model, toolset) changed
            self._ensure_app_bound(llm_to_use, model_name_used, tools_to_bind)
            
            try:
                # Use a per-invocation thread_id so LangGraph's MemorySaver does NOT
                # accumulate tool-call history across separate queries.  Accumulated
                # history causes (a) massive input-token counts that slow every
                # subsequent query and (b) the LLM replaying an old tool workflow
                # instead of answering the new question.
                # Cross-query conversation continuity is already provided by
                # _build_continuation_query in the GUI layer (injected via the system
                # prompt), so per-query isolation here is safe.
                thread_id = f"{current_session_id}:q:{uuid.uuid4().hex}"
                # Increase recursion_limit to avoid premature GRAPH_RECURSION_LIMIT on complex tool workflows.
                max_iterations = getattr(self.settings, 'agent_max_iterations', 20)
                recursion_limit = getattr(self.settings, "agent_recursion_limit", max(50, (max_iterations * 3)))
                if _tool_scope == "lite" or not tools_to_bind:
                    # Knowledge / tool-free turns should finish in 1–2 LLM steps.
                    recursion_limit = min(int(recursion_limit), 8)
                config = {
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": recursion_limit,
                }
                
                # Prepare initial state with enhanced system prompt and user query
                initial_messages = [
                    SystemMessage(content=enhanced_system_prompt),
                    HumanMessage(content=query)
                ]
                
                # Pre-flight token check: Estimate tokens and check TPM limits
                input_tokens, output_tokens_estimate = estimate_message_tokens(
                    initial_messages, model_name_used
                )
                token_estimate = check_tpm_limit(
                    input_tokens, output_tokens_estimate, model_name_used
                )
                
                # Check if user has already approved chunking (from interactive mode)
                user_approved_chunking = "[USER APPROVED:" in query.upper()
                
                # If tokens exceed TPM limit, handle chunking or ask for approval
                if token_estimate.exceeds_tpm and not user_approved_chunking:
                    logger.warning(
                        f"Token limit exceeded: {token_estimate.total_tokens:,} tokens "
                        f"(limit: {token_estimate.tpm_limit:,}) for model {model_name_used}"
                    )
                    
                    # In interactive mode, ask for user approval
                    if interactive_mode:
                        warning_msg = format_token_warning(token_estimate, model_name_used)
                        # Return a special response that the CLI can detect and prompt for
                        return {
                            "query": query,
                            "response": warning_msg,
                            "success": False,
                            "error": "token_limit_exceeded",
                            "token_estimate": {
                                "total_tokens": token_estimate.total_tokens,
                                "tpm_limit": token_estimate.tpm_limit,
                                "chunks_needed": token_estimate.chunks_needed,
                                "estimated_cost": token_estimate.estimated_cost,
                            },
                            "llm_used": "fallback" if use_fallback else "primary",
                            "model_name": model_name_used,
                            "session_id": current_session_id,
                        }
                    else:
                        # Non-interactive mode: automatically proceed with chunking
                        logger.info(
                            f"Proceeding with automatic chunking: {token_estimate.chunks_needed} chunks "
                            f"with 61s delays"
                        )
                elif token_estimate.exceeds_tpm and user_approved_chunking:
                    # User approved, proceed with chunking
                    logger.info(
                        f"User approved chunking: {token_estimate.chunks_needed} chunks "
                        f"with 61s delays"
                    )
                
                initial_state = {"messages": initial_messages}
                
                # Run LangGraph on this OS thread. Do NOT run `self.app.invoke` in a helper thread:
                # AutoCAD/Carlson COM (`pythoncom`, `Dispatch`, `GetActiveObject`) is apartment-bound;
                # a background thread triggers RPC_E_WRONG_THREAD (-2147417842) when tools call AutoCAD.
                # Time limits: use GUI Cancel (subprocess terminate) or adjust agent_query_timeout in settings;
                # there is no reliable cross-platform interrupt of an in-flight `invoke` from another thread.
                base_timeout = getattr(self.settings, 'agent_query_timeout', 300)
                arcgis_ui_timeout = getattr(self.settings, 'arcgis_ui_execution_timeout', base_timeout)
                _query_lower = (query or "").lower()
                _arcgis_indicators = [
                    "arcgis", "arcgis pro", "arcpy", "idw", "cutfill", "cut fill",
                    ".dwg", ".gdb", ".aprx", "raster", "surface", "volume",
                    "point feature", "point features", "polygon", "cadtogeodatabase",
                ]
                is_arcgis_workflow = any(tok in _query_lower for tok in _arcgis_indicators)

                # ArcGIS Pro UI workflows can take materially longer than pure LLM/tool queries.
                # Log a larger budget hint for troubleshooting (graph itself is not join-timeout limited).
                overall_timeout = base_timeout
                if is_arcgis_workflow:
                    overall_timeout = max(base_timeout, arcgis_ui_timeout + 300)
                max_iterations = getattr(self.settings, 'agent_max_iterations', 20)
                result_container = [None]
                exception_container = [None]
                
                try:
                    logger.info(
                        f"Starting graph execution (budget hint: {overall_timeout}s, max iterations: {max_iterations})"
                    )
                    logger.info("Processing query - this may take a moment for large documents...")
                    result_container[0] = self.app.invoke(initial_state, config=config)
                    logger.info("Graph execution completed successfully")
                except Exception as e:
                    logger.error(f"Error during graph execution: {e}")
                    exception_container[0] = e
                
                if exception_container[0]:
                    error = exception_container[0]
                    error_str = str(error).lower()
                    
                    # Detect if model is struggling and should be switched
                    should_switch_model = False
                    switch_reason = None
                    
                    # Check for recursion limit (model can't handle complexity)
                    if "recursion limit" in error_str or "graph_recursion_limit" in error_str:
                        should_switch_model = True
                        switch_reason = "recursion_limit"
                        logger.warning("🔄 Model hit recursion limit - considering model switch")
                    
                    # Check for token/TPM limits (model too small for task)
                    if "tokens per min" in error_str or "tpm" in error_str or "token limit" in error_str:
                        # Only switch if we're not already on the highest tier
                        current_tier = self._get_model_tier(model_name_used)
                        if current_tier != "complex":
                            should_switch_model = True
                            switch_reason = "token_limit"
                            logger.warning(f"🔄 Model hit token limit (tier: {current_tier}) - considering model switch")
                    
                    # If model switch is needed and we haven't already switched
                    if should_switch_model and not getattr(self, "_model_switched_this_query", False):
                        logger.info(f"🔄 Attempting dynamic model switch (reason: {switch_reason})")
                        return self._switch_model_and_retry(
                            query=query,
                            original_query=original_query,
                            current_model=model_name_used,
                            current_llm=llm_to_use,
                            complexity=complexity,
                            enhanced_system_prompt=enhanced_system_prompt,
                            initial_messages=initial_messages,
                            current_session_id=current_session_id,
                            use_fallback=use_fallback,
                            interactive_mode=interactive_mode,
                            context_retrieved=context_retrieved,
                            switch_reason=switch_reason,
                            tools_to_bind=tools_to_bind if 'tools_to_bind' in locals() else self.tools,
                            provider=active_provider if "active_provider" in locals() else self._active_llm_provider(use_fallback),
                        )
                    
                    # If we can't switch or already switched, raise the error
                    raise error
                
                result = result_container[0]
                if result is None:
                    raise RuntimeError("Graph execution returned no result")
                
                # Extract the final response from messages
                response_text = self._extract_response(result)
                tools_used = self._graph_result_used_tools(result)
                graph_success = True
                if (
                    getattr(self, "_internet_permission_granted", False)
                    and self._response_is_internet_permission_ask_only(response_text)
                ):
                    logger.warning(
                        "Suppressed internet-permission re-ask after grant (anti-loop)"
                    )
                    response_text = (
                        "Internet permission was already granted for this request — I will not ask again.\n\n"
                        "Please resend your Living Atlas / land-cover (or other analysis) goal once. "
                        "I will proceed under that already-granted permission using session `.aprx` / `.gdb` "
                        "parcels plus internet/Living Atlas catalog access, without another yes/no prompt. "
                        "If Living Atlas/imagery/model access fails after a real attempt, I will say so "
                        "honestly instead of inventing deep-learning class areas."
                    )
                if self._response_looks_like_unverified_task_completion(
                    routing_query, response_text, tools_used
                ):
                    logger.warning(
                        "Blocked unverified task completion (file-driven task, no tools invoked)"
                    )
                    graph_success = False
                    response_text = (
                        "I could not verify that the requested file operations completed because "
                        "no automation tools were executed. I will not report a fabricated result.\n\n"
                        "Please retry this request — I should run the appropriate ArcGIS, CAD, Excel, "
                        "or document tools and confirm outputs exist on disk before reporting success."
                    )
                llm_cost_usd = self._estimate_llm_cost_usd_from_graph_result(
                    result,
                    model_name_used,
                    response_text,
                    initial_messages_token_hint=input_tokens,
                )
                
                # Store conversation in vector store for future context
                llm_used = "fallback" if use_fallback else "primary"
                self._store_conversation(
                    query=query,
                    response=response_text,
                    session_id=current_session_id,
                    llm_used=llm_used
                )
                
                # Format model info for display
                model_display = f"{llm_used}"
                if model_name_used:
                    model_display = f"{model_display} ({model_name_used})"
                
                return {
                    "query": query,
                    "response": response_text,
                    "llm_used": llm_used,
                    "model_name": model_name_used,  # Include actual model name
                    "complexity": complexity,  # Include detected complexity
                    "success": graph_success,
                    "session_id": current_session_id,
                    "context_retrieved": context_retrieved,
                    "llm_cost_usd": llm_cost_usd,
                    "error": "unverified_completion" if not graph_success else None,
                }
                
            except Exception as e:
                error_str = str(e).lower()
                logger.error(f"Error with {'fallback' if use_fallback else 'primary'} LLM: {e}")
                
                # Detect TPM / rate-limit errors (distinct from actual quota exhaustion)
                is_tpm_error = (
                    "tokens per min" in error_str or
                    "tokens per minute" in error_str or
                    "tpm" in error_str
                )

                active_provider = self.settings.fallback_llm if use_fallback else self.settings.primary_llm
                is_ollama_connection_error = (
                    str(active_provider).lower() == "ollama"
                    and (
                        "connection error" in error_str
                        or "connection refused" in error_str
                        or "failed to establish" in error_str
                        or "localhost:11434" in error_str
                        or "ollama" in error_str
                    )
                )
                if is_ollama_connection_error:
                    return {
                        "query": query,
                        "response": (
                            "Local Ollama is selected, but SurvyAI could not reach the Ollama server.\n\n"
                            "What to do:\n"
                            "1. Open Ollama and make sure it is running.\n"
                            f"2. Confirm the model is installed: `ollama pull {getattr(self.settings, 'ollama_model', 'llama3.2:1b')}`\n"
                            f"3. Confirm the server URL is `{getattr(self.settings, 'ollama_base_url', 'http://localhost:11434')}`.\n\n"
                            "CAD fast-path tasks can still run without an LLM once the request is parsed, "
                            "but general chat/model reasoning needs a running local model or a signed-in cloud proxy."
                        ),
                        "success": False,
                        "error": "ollama_connection_error",
                        "llm_used": "fallback" if use_fallback else "primary",
                        "model_name": getattr(self.settings, "ollama_model", "llama3.2:1b"),
                        "complexity": complexity if 'complexity' in locals() else None,
                    }

                if is_tpm_error:
                    current_model = model_name_used or self._current_openai_model or getattr(self.settings, "openai_model", "unknown")
                    logger.warning(f"TPM/rate-limit exceeded for model: {current_model}")
                    return {
                        "query": query,
                        "response": (
                            f"⚠️ **Rate Limit / TPM Exceeded**\n\n"
                            f"The request exceeded the tokens-per-minute (TPM) limit for `{current_model}`.\n\n"
                            f"**What to do:**\n"
                            f"- For large documents, the agent should use section extraction / the fast-path summarizer (not full text).\n"
                            f"- If you rerun the same request now, it should route into the large-document pipeline and avoid huge tool outputs.\n\n"
                            f"**Error:** {str(e)[:300]}..."
                        ),
                        "success": False,
                        "error": "tpm_rate_limit_exceeded",
                        "llm_used": "fallback" if use_fallback else "primary",
                        "model_name": model_name_used if 'model_name_used' in locals() else current_model,
                        "complexity": complexity if 'complexity' in locals() else None
                    }

                # Detect quota exhaustion / model unavailable (429, quota, proxy 5xx)
                is_quota_error = (
                    "429" in str(e)
                    or "quota" in error_str
                    or "rate limit" in error_str
                    or "resourceexhausted" in error_str
                )
                is_proxy_or_model_dead = (
                    "model_not_found" in error_str
                    or "does not exist" in error_str
                    or "invalid model" in error_str
                    or (
                        ("/v1/llm/chat" in error_str or "hosted llm proxy" in error_str)
                        and any(code in error_str for code in ("500", "502", "503"))
                    )
                )

                if is_quota_error or is_proxy_or_model_dead:
                    # Determine current model name based on LLM type
                    if model_name_used:
                        current_model = model_name_used
                    elif self.settings.primary_llm == "gemini" or (use_fallback and self.settings.fallback_llm == "gemini"):
                        current_model = self._current_gemini_model or "unknown"
                    elif self.settings.primary_llm == "openai" or (use_fallback and self.settings.fallback_llm == "openai"):
                        current_model = self._current_openai_model or getattr(self.settings, "openai_model", "unknown")
                    else:
                        current_model = "unknown"

                    logger.warning(
                        "Model unavailable/quota for %s (quota=%s proxy=%s)",
                        current_model,
                        is_quota_error,
                        is_proxy_or_model_dead,
                    )

                    # Intelligent same-provider failover before giving up.
                    from survyai.provider_models import PAID_PROVIDERS as _PAID

                    _failover_provider = self._active_llm_provider(use_fallback)
                    using_paid_tier = _failover_provider in _PAID
                    if (
                        using_paid_tier
                        and current_model
                        and current_model != "unknown"
                        and "initial_messages" in locals()
                    ):
                        nxt = self._next_provider_failover_model(
                            _failover_provider,
                            current_model,
                            complexity if "complexity" in locals() else "average",
                            elevated_average=bool(
                                locals().get("elevated_average", False)
                            ),
                        )
                        tried_n = len(getattr(self, "_openai_models_tried_this_query", []) or [])
                        if nxt and tried_n < 6:
                            logger.info(
                                "🔄 %s failover %s → %s after %s",
                                _failover_provider,
                                current_model,
                                nxt,
                                "quota" if is_quota_error else "proxy/model error",
                            )
                            return self._switch_model_and_retry(
                                query=query,
                                original_query=original_query if "original_query" in locals() else query,
                                current_model=current_model,
                                current_llm=llm_to_use if "llm_to_use" in locals() else self.llm_primary,
                                complexity=complexity if "complexity" in locals() else "average",
                                enhanced_system_prompt=enhanced_system_prompt if "enhanced_system_prompt" in locals() else self._system_prompt,
                                initial_messages=initial_messages,
                                current_session_id=current_session_id,
                                use_fallback=use_fallback,
                                interactive_mode=interactive_mode if "interactive_mode" in locals() else False,
                                context_retrieved=context_retrieved if "context_retrieved" in locals() else False,
                                switch_reason="quota_or_proxy_failover",
                                tools_to_bind=tools_to_bind if "tools_to_bind" in locals() else self.tools,
                                target_model=nxt,
                                provider=_failover_provider,
                            )

                    # No alternate model left — surface a clear message.
                    return {
                        "query": query,
                        "response": (
                            f"⚠️ **API Quota / Model Unavailable**\n\n"
                            f"The hosted model `{current_model}` could not complete this request "
                            f"(quota exhausted or upstream proxy error).\n\n"
                            f"SurvyAI already tried alternate {_failover_provider} models in the "
                            f"failover chain when available.\n\n"
                            f"**Options:**\n"
                            f"1. **Wait for quota reset** — provider quotas often reset daily\n"
                            f"2. **Retry later** — or Sign in again if the cloud session expired\n"
                            f"3. **Use Ollama** — set Primary LLM to a local model for offline work\n\n"
                            f"**Tried / current:** {current_model}\n"
                            f"**Error:** {str(e)[:240]}..."
                        ),
                        "success": False,
                        "error": "quota_exhausted",
                        "llm_used": "fallback" if use_fallback else "primary",
                        "model_name": model_name_used if "model_name_used" in locals() else current_model,
                        "complexity": complexity if "complexity" in locals() else None,
                    }
                
                # Try fallback if primary failed (and it's not a quota error)
                # Check if Gemini fallback is disabled
                if not use_fallback:
                    if getattr(self.settings, 'disable_gemini_fallback', False) and self.settings.fallback_llm == "gemini":
                        logger.warning("⚠️ Fallback to Gemini is disabled. Using GPT models only.")
                        # Return error instead of falling back to Gemini
                        return {
                            "query": query,
                            "response": (
                                f"❌ **Error with primary LLM ({self.settings.primary_llm})**\n\n"
                                f"Error: {str(e)[:500]}\n\n"
                                f"**Note:** Fallback to Gemini is disabled (DISABLE_GEMINI_FALLBACK=True). "
                                f"Please ensure your OpenAI API key is valid and configured.\n"
                                f"Check your .env file for OPENAI_API_KEY."
                            ),
                            "success": False,
                            "error": str(e),
                            "llm_used": "primary",
                            "model_name": model_name_used if 'model_name_used' in locals() else "unknown"
                        }
                    else:
                        logger.info(f"Attempting with fallback LLM ({self.settings.fallback_llm})")
                        return self.process_query(query, use_fallback=True)
                raise
        
        except Exception as e:
            logger.error(f"Error in process_query: {e}")
            err = str(e)
            low = err.lower()
            if (
                "/v1/llm/chat" in low
                or "hosted llm proxy" in low
                or ("unauthorized" in low and "token" in low)
            ):
                response = (
                    f"{err}\n\n"
                    "What to try:\n"
                    "1. Account → Sign in again (refresh hosted LLM access).\n"
                    "2. Confirm your Pro subscription still has API credits.\n"
                    "3. Retry this prompt.\n"
                    "4. If hosted LLM stays down, set Primary LLM to Ollama for local/offline work "
                    "(needs enough free RAM)."
                )
            else:
                response = f"Error processing query: {err}"
            return {
                "query": query,
                "response": response,
                "success": False,
                "error": err,
                "llm_cost_usd": 0.0,
            }
    
    def _estimate_llm_cost_usd_from_graph_result(
        self,
        result: Dict[str, Any],
        model_name: Optional[str],
        response_text: str,
        initial_messages_token_hint: Optional[int] = None,
    ) -> float:
        """
        Provider-reported USD cost for the graph run (Credits & Usage / local tracking).

        Raw provider cost: per AIMessage usage with cached-input pricing, summed across
        the run. If the provider omits usage metadata, return 0 so customer credits
        are not debited from estimates.
        """
        mn = (model_name or "").strip() or str(getattr(self, "_active_model_name", "") or "").strip()
        if not mn:
            mn = getattr(self.settings, "openai_model_mini", "gpt-5.4-mini")
        try:
            from utils.cost_estimator import is_local_free_model, summarize_graph_llm_usage

            if self._llm_is_ollama() or is_local_free_model(mn):
                return 0.0
        except Exception:
            from utils.cost_estimator import summarize_graph_llm_usage
        try:
            usage = summarize_graph_llm_usage(
                list((result or {}).get("messages") or []),
                mn,
                response_text=response_text or "",
                initial_messages_token_hint=initial_messages_token_hint,
                infer_missing_cached=False,
            )
            if usage.get("estimated"):
                return 0.0
            return float(usage.get("cost_usd") or 0.0)
        except Exception:
            return 0.0
    
    @staticmethod
    def _graph_result_used_tools(result: Dict) -> bool:
        """True if the LangGraph run invoked at least one tool."""
        for message in (result or {}).get("messages") or []:
            if isinstance(message, ToolMessage):
                return True
        return False

    def _response_looks_like_unverified_task_completion(
        self,
        routing_query: str,
        response: str,
        tools_used: bool,
    ) -> bool:
        """
        Detect when the LLM claims a file/GIS/CAD task finished without running tools.
        """
        if tools_used:
            return False
        if self._classify_query_intent(routing_query) != "task":
            return False
        if not looks_like_file_driven_task(routing_query):
            return False

        rl = (response or "").lower()
        if not rl.strip():
            return True

        completion_phrases = (
            "saved to", "saved as", "saved essay", "output:", "created the",
            "successfully created", "successfully saved", "successfully exported",
            "successfully generated", "successfully computed", "successfully calculated",
            "volume:", "cut fill", "cutfill", "task complete", "done.",
            "exported to", "written to", "file has been",
        )
        if any(p in rl for p in completion_phrases):
            return True

        action_verbs = ("saved", "created", "exported", "generated", "computed", "calculated")
        evidence_tokens = ("output", "path", "file", "volume", "result", ".csv", ".dwg", ".docx", ".xlsx")
        if any(v in rl for v in action_verbs) and any(t in rl for t in evidence_tokens):
            return True

        return False

    def _extract_response(self, result: Dict) -> str:
        """
        Extract the final text response from the graph result.
        
        The result contains a list of messages. We want the last AI message
        that contains the final response to the user (not tool calls).
        
        Args:
            result: The result from invoking the LangGraph
            
        Returns:
            str: The extracted response text
        """
        import ast

        def _stringify_content(content: Any) -> str:
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                text_parts: List[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = str(item.get("text", "")).strip()
                        if text:
                            text_parts.append(text)
                    elif isinstance(item, str):
                        text = item.strip()
                        if text:
                            text_parts.append(text)
                return "\n".join(text_parts).strip()
            return str(content).strip() if content is not None else ""

        def _maybe_parse_tool_payload(raw_text: str) -> Optional[Dict[str, Any]]:
            if not raw_text:
                return None
            try:
                parsed = json.loads(raw_text)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                pass
            try:
                parsed = ast.literal_eval(raw_text)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None

        def _summarize_tool_payload(payload: Dict[str, Any]) -> str:
            success = bool(payload.get("success", False))
            lines: List[str] = []

            message = str(payload.get("message") or "").strip()
            note = str(payload.get("note") or "").strip()
            error = str(payload.get("error") or "").strip()
            instructions = str(payload.get("instructions") or "").strip()
            script_path = str(payload.get("script_path") or "").strip()
            project_path = str(payload.get("project_path") or "").strip()
            stdout = str(payload.get("stdout") or "").strip()
            results = payload.get("results")

            if success:
                lines.append(message or "Task completed successfully.")
            else:
                lines.append(message or "Task did not complete successfully.")

            if isinstance(results, dict) and results:
                preferred_keys = [
                    "net_volume",
                    "net_volume_cum",
                    "net_volume_m3",
                    "fill_volume",
                    "fill_volume_cum",
                    "cut_volume",
                    "cut_volume_cum",
                    "footprint_area",
                    "boundary_area_sqm",
                    "output_csv",
                    "pre_csv_copy",
                    "post_csv_copy",
                    "project_path",
                ]
                seen = set()
                for key in preferred_keys + list(results.keys()):
                    if key in seen or key not in results:
                        continue
                    seen.add(key)
                    val = results.get(key)
                    if isinstance(val, dict):
                        formatted = val.get("formatted") or val.get("text")
                        if formatted:
                            lines.append(f"- {key.replace('_', ' ').title()}: {formatted}")
                    elif val not in (None, ""):
                        lines.append(f"- {key.replace('_', ' ').title()}: {val}")

            if error:
                lines.append(f"Error: {error}")
            if note:
                lines.append(note)
            if instructions and not success:
                lines.append(instructions)
            if script_path:
                lines.append(f"Script: {script_path}")
            if project_path:
                lines.append(f"Project: {project_path}")

            if stdout:
                result_lines = [line.strip() for line in stdout.splitlines() if line.strip().startswith("RESULT_")]
                for line in result_lines[:8]:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        key = parts[0].replace("RESULT_", "").replace("_", " ").title()
                        value = parts[1].strip()
                        lines.append(f"- {key}: {value}")
                if not result_lines and not results and success:
                    lines.append(stdout[:1500].strip())

            compact = [line.strip() for line in lines if line and line.strip()]
            return "\n".join(compact).strip()

        messages = result.get("messages", [])
        
        # Find the last AI message with actual content (not just tool calls)
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                # Check if this message has text content
                text = _stringify_content(message.content)
                if text:
                    return text

        # If the graph ended immediately after a tool call (for example because a
        # same-error guard stopped another retry), there may be no final AI message.
        # In that case, surface the last tool result instead of returning an empty response.
        for message in reversed(messages):
            if isinstance(message, ToolMessage):
                tool_text = _stringify_content(getattr(message, "content", ""))
                if not tool_text:
                    continue
                payload = _maybe_parse_tool_payload(tool_text)
                if payload:
                    summary = _summarize_tool_payload(payload)
                    if summary:
                        return summary
                return tool_text

        return "Task finished, but no final assistant message was produced. Check the latest tool output in the saved logs or output files."


# ==============================================================================
# MODULE-LEVEL EXPORTS
# ==============================================================================

# When someone does "from agent import ...", these are available
__all__ = ["SurvyAIAgent", "SYSTEM_PROMPT"]
