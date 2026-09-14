"""
Vector-assisted assessment of cadastral plan *extras* (access roads, fences, etc.).

The deterministic regex parser in agent.py remains the baseline.  This module
retrieves similar past cadastral prompts from the vector store, then uses a
cheap LLM pass to interpret varied natural-language phrasing before merging
results back into the plotting pipeline.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from tools.vector_store import COLLECTION_DOCUMENTS

CADASTRAL_EXTRAS_DOC_TYPE = "cadastral_plan_extras"


class CadastralAccessRoadSpec(BaseModel):
    """One access road beside a traverse leg between two pillars."""

    width_m: float = Field(..., gt=0)
    pillar_a: str = ""
    pillar_b: str = ""
    offset_m: Optional[float] = None
    title: Optional[str] = None


class CadastralFenceSpec(BaseModel):
    """Concrete wall fence along one or more consecutive traverse legs."""

    kind: str = "CWF"  # CWF or DCWF
    pillar_chain: List[str] = Field(default_factory=list)


class CadastralPlanExtrasAssessment(BaseModel):
    """Structured interpretation of non-core cadastral plotting instructions."""

    access_roads: List[CadastralAccessRoadSpec] = Field(default_factory=list)
    fences: List[CadastralFenceSpec] = Field(default_factory=list)
    access_road_title: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    source: str = "none"
    notes: str = ""


def _normalize_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _road_merge_key(spec: str) -> str:
    """Identity for an access road: width + sorted pillar names (order-independent)."""
    s = _normalize_key(spec)
    wm = re.search(r"(\d+(?:\.\d+)?)\s*m", s)
    width = wm.group(1) if wm else ""
    m = re.search(
        r"(?:side of|joining pillars|(?:boundary line )?connecting)\s+(.+)$",
        s,
    )
    if m:
        tail = re.sub(r"\s+offset\b.*$", "", m.group(1)).strip()
        parts = [
            p.strip(" .,;")
            for p in re.split(r"\s+and\s+", tail)
            if p.strip(" .,;")
        ]
        if len(parts) >= 2:
            return width + "|" + "|".join(sorted(parts[:2]))
    return s


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    from survyai.provider_models import extract_llm_json_object

    return extract_llm_json_object(text)


def _coerce_access_roads(raw: Any) -> List[CadastralAccessRoadSpec]:
    roads: List[CadastralAccessRoadSpec] = []
    if not isinstance(raw, list):
        return roads
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            width = float(item.get("width_m") or item.get("width") or 0)
            if width <= 0:
                continue
            pa = str(item.get("pillar_a") or item.get("from_pillar") or "").strip()
            pb = str(item.get("pillar_b") or item.get("to_pillar") or "").strip()
            if not pa or not pb:
                pillars = item.get("pillars") or item.get("pillar_pair")
                if isinstance(pillars, list) and len(pillars) >= 2:
                    pa, pb = str(pillars[0]).strip(), str(pillars[1]).strip()
            if not pa or not pb:
                continue
            offset = item.get("offset_m") or item.get("offset")
            roads.append(
                CadastralAccessRoadSpec(
                    width_m=width,
                    pillar_a=pa,
                    pillar_b=pb,
                    offset_m=float(offset) if offset not in (None, "") else None,
                    title=(str(item.get("title")).strip() if item.get("title") else None),
                )
            )
        except Exception:
            continue
    return roads


def _coerce_fences(raw: Any) -> List[CadastralFenceSpec]:
    fences: List[CadastralFenceSpec] = []
    if not isinstance(raw, list):
        return fences
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind_raw = str(item.get("kind") or "CWF").upper()
        kind = "DCWF" if "D" in kind_raw and "CWF" in kind_raw else "CWF"
        if re.search(r"dwarf|d\.c\.w\.f", str(item.get("kind") or ""), re.I):
            kind = "DCWF"
        chain = item.get("pillar_chain") or item.get("pillars") or []
        if isinstance(chain, str):
            parts = re.split(r"\s+to\s+|\s+and\s+|,\s*", chain, flags=re.I)
            chain = [p.strip() for p in parts if p.strip()]
        if not isinstance(chain, list) or len(chain) < 2:
            continue
        fences.append(
            CadastralFenceSpec(
                kind=kind,
                pillar_chain=[str(p).strip() for p in chain if str(p).strip()],
            )
        )
    return fences


def retrieve_similar_cadastral_extras(
    query: str,
    *,
    vector_store: Any,
    search_fn: Callable[..., List[Dict[str, Any]]],
    top_k: int = 4,
    score_threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    """Hybrid/semantic search for prior cadastral plotting instructions."""
    if vector_store is None:
        return []

    focus = query
    m = re.search(
        r"(add\s+.+)$",
        query,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        focus = m.group(1).strip()

    search_text = (
        "cadastral survey plan plotting instructions access road fence concrete wall "
        f"{focus}"
    )
    hits: List[Dict[str, Any]] = []
    try:
        typed = search_fn(
            query=search_text,
            collection=COLLECTION_DOCUMENTS,
            top_k=top_k,
            where={"doc_type": CADASTRAL_EXTRAS_DOC_TYPE},
        )
        hits.extend(typed or [])
    except Exception:
        pass

    if len(hits) < top_k:
        try:
            general = search_fn(
                query=search_text,
                collection=COLLECTION_DOCUMENTS,
                top_k=top_k,
            )
            seen = {h.get("id") for h in hits}
            for h in general or []:
                if h.get("id") not in seen:
                    hits.append(h)
        except Exception:
            pass

    filtered: List[Dict[str, Any]] = []
    for h in hits:
        score = float(h.get("score") or 0.0)
        if score >= score_threshold:
            filtered.append(h)
    return filtered[:top_k]


def _format_context_examples(hits: Sequence[Dict[str, Any]]) -> str:
    if not hits:
        return "(no similar stored examples yet)"
    lines: List[str] = []
    for i, hit in enumerate(hits, 1):
        content = str(hit.get("content") or "").strip()
        if len(content) > 1200:
            content = content[:1200] + "..."
        lines.append(f"Example {i} (score={hit.get('score', 0):.2f}):\n{content}")
    return "\n\n".join(lines)


def assess_cadastral_plan_extras(
    query: str,
    *,
    pillar_numbers: Sequence[str],
    vector_store: Any,
    search_fn: Callable[..., List[Dict[str, Any]]],
    llm: Any,
    run_with_timeout: Callable[..., Any],
    score_threshold: float = 0.25,
) -> CadastralPlanExtrasAssessment:
    """
    Vector retrieval + cheap LLM structured interpretation of plan extras.

    Returns an empty assessment when vector store or LLM is unavailable.
    """
    pillars = [str(p).strip() for p in pillar_numbers if str(p).strip()]
    hits = retrieve_similar_cadastral_extras(
        query,
        vector_store=vector_store,
        search_fn=search_fn,
        score_threshold=score_threshold,
    )
    context_block = _format_context_examples(hits)

    if llm is None:
        return CadastralPlanExtrasAssessment(source="unavailable", notes="LLM not configured")

    system = (
        "You interpret cadastral SURVEY PLAN plotting instructions for an AI agent.\n"
        "Extract ONLY optional plot extras from the user prompt:\n"
        "  - access_roads: width in metres + the TWO pillar labels for each road side\n"
        "  - fences: ONLY when the prompt explicitly mentions CWF, DCWF, C.W.F., D.C.W.F., WF, Fence, Wall Fence, or Concrete Wall Fence — never infer from line symbology alone\n"
        "  - access_road_title: optional custom road label for the first road\n"
        "Do NOT invent pillars not mentioned. Use exact pillar spellings from the prompt.\n"
        "Multiple access roads are common (e.g. two different widths on different sides).\n"
        "Output ONLY compact JSON with keys:\n"
        "  access_roads: [{width_m, pillar_a, pillar_b, offset_m?, title?}]\n"
        "  fences: [{kind: 'CWF'|'DCWF', pillar_chain: [..]}]\n"
        "  access_road_title: string or null\n"
        "  confidence: 0.0-1.0\n"
        "  notes: short string\n"
        "No markdown, no prose outside JSON."
    )
    user = (
        f"Known pillar numbers for this plan: {', '.join(pillars) or '(not listed)'}\n\n"
        f"Similar past instructions from vector memory:\n{context_block}\n\n"
        f"Current user prompt:\n{query}\n\n"
        "Return JSON only."
    )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        msg = run_with_timeout(
            30,
            lambda: llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]),
        )[0]
        from survyai.provider_models import llm_visible_text_from_content

        text = llm_visible_text_from_content(getattr(msg, "content", msg))
        payload = _extract_json_object(str(text or ""))
        if not payload:
            return CadastralPlanExtrasAssessment(
                source="llm_parse_failed",
                notes="Could not parse LLM JSON for cadastral extras.",
            )

        roads = _coerce_access_roads(payload.get("access_roads"))
        fences = _coerce_fences(payload.get("fences"))
        title = payload.get("access_road_title")
        confidence = float(payload.get("confidence") or 0.0)
        notes = str(payload.get("notes") or "").strip()

        source = "vector_llm" if hits else "llm_only"
        return CadastralPlanExtrasAssessment(
            access_roads=roads,
            fences=fences,
            access_road_title=(str(title).strip() if title else None),
            confidence=max(0.0, min(1.0, confidence)),
            source=source,
            notes=notes,
        )
    except Exception as exc:
        return CadastralPlanExtrasAssessment(
            source="error",
            notes=f"Cadastral extras assessment failed: {exc}",
        )


def parse_cadastral_geometry_blob_with_llm(
    query: str,
    *,
    pillar_numbers: Sequence[str],
    llm: Any,
    run_with_timeout: Callable[..., Any],
    vector_store: Any = None,
    search_fn: Optional[Callable[..., List[Dict[str, Any]]]] = None,
    score_threshold: float = 0.25,
    timeout_s: int = 35,
) -> str:
    """
    LLM fallback: extract the coordinate/traverse text block from a cadastral prompt.

    Used when regex cannot match varied phrasing such as 'coordinates for the point:'
    or bare '291200.165mE, 537230.450mN' without parentheses.
    """
    if llm is None:
        return ""

    pillars = [str(p).strip() for p in pillar_numbers if str(p).strip()]
    context_block = "(no similar stored examples yet)"
    if search_fn is not None:
        try:
            hits = retrieve_similar_cadastral_extras(
                query,
                vector_store=vector_store,
                search_fn=search_fn,
                score_threshold=score_threshold,
            )
            context_block = _format_context_examples(hits)
        except Exception:
            pass

    system = (
        "You extract coordinate and traverse geometry from Nigerian cadastral CAD prompts.\n"
        "Return ONLY JSON:\n"
        '  {"coordinates_blob": "...", "confidence": 0.0-1.0}\n'
        "coordinates_blob must contain ONLY the geometry portion:\n"
        "  - anchor coordinate(s) as EmE, NmN (with or without parentheses)\n"
        "  - and/or bearing/distance traverse legs\n"
        "Do NOT include access roads, fences, buyer name, or other metadata.\n"
        "Preserve numeric values exactly as in the user prompt."
    )
    user = (
        f"Pillar numbers: {', '.join(pillars) or '(not listed)'}\n\n"
        f"Similar past prompts:\n{context_block}\n\n"
        f"Current prompt:\n{query}\n\n"
        "Return JSON only."
    )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        msg, err, timed_out = run_with_timeout(
            timeout_s,
            lambda: llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]),
        )
        if timed_out or err or msg is None:
            return ""
        from survyai.provider_models import llm_visible_text_from_content

        text = llm_visible_text_from_content(getattr(msg, "content", msg))
        payload = _extract_json_object(str(text or ""))
        if not payload:
            return ""
        blob = str(payload.get("coordinates_blob") or "").strip()
        confidence = float(payload.get("confidence") or 0.0)
        if blob and confidence >= 0.25:
            from agent.pdf_survey_plan import _trim_coordinates_blob

            return _trim_coordinates_blob(blob)
        return ""
    except Exception:
        return ""


def access_road_to_spec(road: CadastralAccessRoadSpec) -> str:
    """Convert structured road to the legacy string format used by the plotter."""
    spec = f"{road.width_m:g}m width on the side of {road.pillar_a} and {road.pillar_b}"
    if road.offset_m is not None:
        spec += f" offset {road.offset_m:g}m"
    return spec


def fence_to_dict(fence: CadastralFenceSpec) -> Dict[str, str]:
    """Convert structured fence to the legacy dict used by the plotter."""
    chain = " to ".join(fence.pillar_chain)
    if fence.kind == "DCWF":
        label = "Dwarf Concrete Wall Fence"
    else:
        label = "Concrete wall fence"
    return {"kind": fence.kind, "spec": f"{label} on the sides joining {chain}"}


def merge_access_roads(
    regex_specs: List[str],
    assessed: List[CadastralAccessRoadSpec],
    *,
    confidence: float = 0.0,
    min_confidence: float = 0.35,
) -> List[str]:
    """
    Union regex and assessed roads. Regex results are always kept.
    Assessed roads are added when they are new and either confidence is high
    enough, or regex found none (LLM is the only source).
    """
    merged: List[str] = list(regex_specs or [])
    seen = {_road_merge_key(s) for s in merged}

    for road in assessed:
        spec = access_road_to_spec(road)
        key = _road_merge_key(spec)
        if not key or key in seen:
            continue
        add = confidence >= min_confidence or not regex_specs
        if add:
            merged.append(spec)
            seen.add(key)
    return merged


def merge_fences(
    regex_fences: List[Dict[str, str]],
    assessed: List[CadastralFenceSpec],
    *,
    confidence: float = 0.0,
    min_confidence: float = 0.35,
    query: str = "",
) -> List[Dict[str, str]]:
    merged: List[Dict[str, str]] = list(regex_fences or [])
    seen = {_normalize_key(f.get("spec", "")) for f in merged}

    try:
        from agent.pdf_survey_plan import query_has_explicit_fence_label
    except Exception:
        query_has_explicit_fence_label = None  # type: ignore

    if query_has_explicit_fence_label and not query_has_explicit_fence_label(query):
        return merged

    for fence in assessed:
        item = fence_to_dict(fence)
        key = _normalize_key(item.get("spec", ""))
        if not key or key in seen:
            continue
        add = confidence >= min_confidence and query_has_explicit_fence_label(query)
        if add:
            merged.append(item)
            seen.add(key)
    return merged


def store_cadastral_plan_extras(
    vector_store: Any,
    *,
    query: str,
    output_dwg: str,
    access_roads: List[str],
    fences: List[Dict[str, str]],
    pillar_numbers: str,
) -> None:
    """Persist a successful plot's extras so future prompts can be matched semantically."""
    if vector_store is None:
        return
    payload = {
        "prompt_excerpt": (query or "")[:2500],
        "output_dwg": output_dwg,
        "pillar_numbers": pillar_numbers,
        "access_roads": access_roads,
        "fences": fences,
        "instruction_summary": (
            f"Plotted {output_dwg} with {len(access_roads)} access road(s) and "
            f"{len(fences)} fence segment(s)."
        ),
    }
    content = json.dumps(payload, ensure_ascii=False)
    metadata = {
        "doc_type": CADASTRAL_EXTRAS_DOC_TYPE,
        "output_dwg": output_dwg,
        "access_road_count": len(access_roads),
        "fence_count": len(fences),
    }
    try:
        vector_store.add_documents(
            [{"content": content, "metadata": metadata}],
            collection=COLLECTION_DOCUMENTS,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Traverse adjustment intent (Bowditch / compass rule)
# ---------------------------------------------------------------------------
# Surveyors name the same method in many ways. Detect the *method*, not one
# prompt template. Generic "adjust/close the traverse" is NOT Bowditch — the
# CAD default remains bearing-only adjustment unless this method is requested.

_BOWDITCH_TERM = (
    r"(?:bowditch|bowdich|bodwitch)(?:['’]s)?"
    r"|compass\s+(?:rule|method|adjustment)"
)
# Negation must attach to the method itself. Do not let
# "do not auto-adjust traverse" swallow a later "use Bowditch".
_BOWDITCH_NEGATION = re.compile(
    r"(?:"
    r"(?:do\s+not|don['’]?t|never)\s+(?:please\s+)?(?:use|apply|perform|run)\s+(?:the\s+)?"
    r"|without\s+(?:please\s+)?(?:using\s+|applying\s+)?(?:the\s+)?"
    r"|\b(?:no|not)\s+(?:the\s+)?"
    r")(?:" + _BOWDITCH_TERM + r")",
    flags=re.IGNORECASE,
)
_BOWDITCH_REQUEST = re.compile(
    r"\b(?:" + _BOWDITCH_TERM + r")\b",
    flags=re.IGNORECASE,
)
_BEARING_ADJUST_REQUEST = re.compile(
    r"\b(?:bearing\s+adjustment|hold\s+distances?\s+constant|distances?\s+held\s+constant)\b",
    flags=re.IGNORECASE,
)


def user_requests_bowditch_adjustment(*texts: Optional[str]) -> bool:
    """
    True when the user asked for Bowditch / compass-rule closure.

    Searches the whole message (title block, access-road tail, compose scope,
    etc.). The coordinates blob is often trimmed before 'Add an access…', so
    callers must pass the original prompt — not only the geometry excerpt.
    """
    blob = "\n".join(str(t) for t in texts if t and str(t).strip())
    if not blob:
        return False
    if _BOWDITCH_NEGATION.search(blob):
        return False
    return bool(_BOWDITCH_REQUEST.search(blob))


def user_requests_bearing_adjustment(*texts: Optional[str]) -> bool:
    """True when the user asked for bearing-only adjustment (distances held)."""
    blob = "\n".join(str(t) for t in texts if t and str(t).strip())
    if not blob:
        return False
    return bool(_BEARING_ADJUST_REQUEST.search(blob))


def preferred_traverse_adjustment_method(*texts: Optional[str]) -> str:
    """
    Return 'bowditch' or 'bearing_adjustment'.

    Default is bearing adjustment (distances held). If both methods are named,
    the last explicit mention wins so Automated CAD checkboxes stay strict.
    """
    blob = "\n".join(str(t) for t in texts if t and str(t).strip())
    wants_bow = user_requests_bowditch_adjustment(*texts)
    wants_brg = user_requests_bearing_adjustment(*texts)
    if wants_bow and wants_brg and blob:
        last = "bearing_adjustment"
        markers: List[tuple[int, str]] = []
        for m in _BOWDITCH_REQUEST.finditer(blob):
            markers.append((m.start(), "bowditch"))
        for m in _BEARING_ADJUST_REQUEST.finditer(blob):
            markers.append((m.start(), "bearing_adjustment"))
        if markers:
            markers.sort(key=lambda item: item[0])
            last = markers[-1][1]
        return last
    if wants_bow:
        return "bowditch"
    return "bearing_adjustment"


def bowditch_instruction_for_subprompt(*texts: Optional[str]) -> str:
    """Canonical sentence so composed CAD sub-prompts keep an explicit request."""
    if user_requests_bowditch_adjustment(*texts):
        return "Use Bowditch adjustment method to close the traverse."
    return ""


def _fmt_metres(value: Any, *, digits: int = 3) -> str:
    try:
        return f"{float(value):.{int(digits)}f} m"
    except (TypeError, ValueError):
        return "—"


def _fmt_en_m(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_bearing_dms(deg: Any) -> str:
    try:
        d = float(deg) % 360.0
    except (TypeError, ValueError):
        return "—"
    whole = int(d)
    minutes = int(round((d - whole) * 60.0))
    if minutes == 60:
        whole = (whole + 1) % 360
        minutes = 0
    return f"{whole}° {minutes:02d}′"


def _shoelace_area_m2(points: Sequence[Any]) -> Optional[float]:
    coords: List[Tuple[float, float]] = []
    for p in points or []:
        if not isinstance(p, dict):
            continue
        try:
            coords.append((float(p.get("e", p.get("x"))), float(p.get("n", p.get("y")))))
        except (TypeError, ValueError):
            continue
    if len(coords) < 3:
        return None
    if coords[0] != coords[-1]:
        coords = list(coords) + [coords[0]]
    acc = 0.0
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        acc += x1 * y2 - x2 * y1
    return abs(acc) / 2.0


def format_traverse_adjustment_chat_lines(bow: Optional[Dict[str, Any]]) -> List[str]:
    """User-visible summary that names the method actually applied."""
    if not isinstance(bow, dict) or bow.get("mode") != "bearing_distance":
        return []
    method = str(bow.get("method") or "")
    mis_s = _fmt_metres(bow.get("misclosure_m"))
    me = _fmt_metres(bow.get("misclosure_e_m"))
    mn = _fmt_metres(bow.get("misclosure_n_m"))
    shift = _fmt_metres(bow.get("max_point_shift_m"))

    if method == "pdf_auto_adjust_forbidden":
        return [
            f"• Traverse left unadjusted (PDF-derived geometry). Misclosure: {mis_s}."
        ]
    if method == "bearing_adjustment_failed":
        return [
            f"• Bearing adjustment failed; traverse plotted unadjusted. Misclosure: {mis_s}."
        ]
    if not bow.get("applied"):
        return [
            f"• Traverse already closed within 1 cm (misclosure {mis_s}); no adjustment applied."
        ]
    if method == "bowditch":
        return [
            "• Method: Bowditch (compass rule) — bearings and distances both adjusted",
            f"• Misclosure: {mis_s}  (E {me}, N {mn})",
            f"• Largest station shift: {shift}",
        ]
    if method == "bearing_adjustment":
        return [
            "• Method: Bearing adjustment (distances held constant)",
            f"• Misclosure: {mis_s}  (E {me}, N {mn})",
            f"• Largest station shift: {shift}",
        ]
    return [
        f"• Method: {method or 'traverse adjustment'}",
        f"• Misclosure: {mis_s}",
        f"• Largest station shift: {shift}",
    ]


CAD_FOLLOWUP_FOOTER = (
    "You can ask for changes in this conversation — add a road, change the title, "
    "or analyse the plot — without starting over."
)


def format_cadastral_plot_success_message(
    *,
    output_dwg: Optional[str] = None,
    geometry: Optional[Dict[str, Any]] = None,
    access_road_title: Optional[str] = None,
    opener: str = "Cadastral plan ready.",
) -> str:
    """Professional user-facing summary of a successful cadastral plot (no raw dicts)."""
    geom = geometry if isinstance(geometry, dict) else {}
    bow = geom.get("bowditch") if isinstance(geom.get("bowditch"), dict) else {}
    lines: List[str] = [opener.strip() or "Cadastral plan ready.", ""]

    if output_dwg:
        lines.extend(["File", str(output_dwg).strip(), ""])

    plot_bits: List[str] = []
    denom = geom.get("output_plan_denom") or (geom.get("scale_debug") or {}).get("chosen_denom")
    try:
        if denom:
            plot_bits.append(f"• Scale: 1:{int(denom)}")
    except (TypeError, ValueError):
        pass
    stations = bow.get("adjusted_points_preview") or []
    n_st = 0
    try:
        n_st = int(geom.get("pillar_inserts") or 0)
    except (TypeError, ValueError):
        n_st = 0
    if not n_st:
        n_st = len(stations) if isinstance(stations, list) else 0
    if n_st:
        plot_bits.append(f"• Stations plotted: {n_st}")
    road = (access_road_title or geom.get("access_road_title") or "").strip()
    if road:
        plot_bits.append(f"• Access road label: {road}")
    area = _shoelace_area_m2(stations) if isinstance(stations, list) else None
    if area and area > 1e-3:
        plot_bits.append(f"• Approx. area: {area:,.0f} m²")
    peri = bow.get("perimeter_m")
    try:
        if peri and float(peri) > 0:
            plot_bits.append(f"• Perimeter: {_fmt_metres(peri, digits=2)}")
    except (TypeError, ValueError):
        pass
    if plot_bits:
        lines.append("Plot")
        lines.extend(plot_bits)
        lines.append("")

    adj_lines = format_traverse_adjustment_chat_lines(bow if bow else None)
    if adj_lines:
        lines.append("Traverse")
        lines.extend(adj_lines)
        lines.append("")

    if isinstance(stations, list) and stations:
        lines.append("Stations (as plotted)")
        for i, p in enumerate(stations, start=1):
            if not isinstance(p, dict):
                continue
            lines.append(f"{i}.  {_fmt_en_m(p.get('e'))} E,  {_fmt_en_m(p.get('n'))} N")
        lines.append("")

    legs = bow.get("adjusted_legs_preview") or []
    if isinstance(legs, list) and legs:
        lines.append("Legs (as plotted)")
        for i, leg in enumerate(legs, start=1):
            if not isinstance(leg, dict):
                continue
            lines.append(
                f"{i}.  {_fmt_bearing_dms(leg.get('bearing_deg'))}  ·  "
                f"{_fmt_metres(leg.get('distance'), digits=2)}"
            )
        lines.append("")

    lines.append(CAD_FOLLOWUP_FOOTER)
    return "\n".join(lines).rstrip() + "\n"


def format_cadastral_status_message(
    *,
    opener: str,
    file_path: Optional[str] = None,
    bullets: Optional[Sequence[str]] = None,
    extra_lines: Optional[Sequence[str]] = None,
    include_footer: bool = True,
) -> str:
    """Sectioned success copy for Excel / compose / batch cadastral results."""
    lines: List[str] = [(opener or "Cadastral plan ready.").strip(), ""]
    if file_path:
        lines.extend(["File", str(file_path).strip(), ""])
    bits: List[str] = []
    for item in bullets or []:
        bit = str(item or "").strip()
        if not bit or bit.endswith(":") or re.search(r":\s*None\s*$", bit):
            continue
        bits.append(bit if bit.startswith("•") else f"• {bit}")
    if bits:
        lines.append("Plot")
        lines.extend(bits)
        lines.append("")
    extras = [str(x).rstrip() for x in (extra_lines or [])]
    extras = [x for x in extras if x]
    if extras:
        lines.extend(extras)
        lines.append("")
    if include_footer:
        lines.append(CAD_FOLLOWUP_FOOTER)
    return "\n".join(lines).rstrip() + "\n"


_ORDINAL_WORDS = {
    "first": 0,
    "1st": 0,
    "second": 1,
    "2nd": 1,
    "third": 2,
    "3rd": 2,
    "fourth": 3,
    "4th": 3,
    "fifth": 4,
    "5th": 4,
    "sixth": 5,
    "6th": 5,
    "last": -1,
    "final": -1,
}
_OWNER_ROLE = r"(?:buyer|owner|purchaser|allottee|title(?:\s+holder)?|person)"
_SHARE_TOKEN = (
    r"(?:half|one[\s-]?half|1\s*/\s*2|two[\s-]?thirds|2\s*/\s*3|"
    r"one[\s-]?third|1\s*/\s*3|three[\s-]?quarters|3\s*/\s*4|"
    r"quarter|one[\s-]?quarter|1\s*/\s*4|"
    r"two[\s-]?fifths?|2\s*/\s*5|three[\s-]?fifths?|3\s*/\s*5|"
    r"four[\s-]?fifths?|4\s*/\s*5|one[\s-]?fifth|1\s*/\s*5|"
    r"\d+(?:\.\d+)?\s*%)"
)
_AREA_UNIT = r"(?:sq\.?\s*m(?:et(?:re|er)s?)?|square\s*met(?:re|er)s?|m[²2]|sqm)"
_PLACEMENT_CARD = {
    "north": "N",
    "northern": "N",
    "northerly": "N",
    "south": "S",
    "southern": "S",
    "southerly": "S",
    "east": "E",
    "eastern": "E",
    "easterly": "E",
    "west": "W",
    "western": "W",
    "westerly": "W",
}


def _ordinal_to_index(word: str, n: int) -> Optional[int]:
    key = re.sub(r"[^a-z0-9]", "", (word or "").lower())
    if key not in _ORDINAL_WORDS:
        return None
    idx = _ORDINAL_WORDS[key]
    if idx < 0:
        idx = n - 1
    return idx if 0 <= idx < n else None


def _share_token_to_frac(text: str) -> Optional[float]:
    t = (text or "").lower()
    if re.search(r"remainder|remaining|\brest\b", t) and not re.search(
        r"\b(?:parcel|plot|land|holding)\b", t
    ):
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", t)
    if m:
        try:
            return float(m.group(1)) / 100.0
        except (TypeError, ValueError):
            return None
    if re.search(r"two[\s-]?thirds|2\s*/\s*3", t):
        return 2.0 / 3.0
    if re.search(r"three[\s-]?quarters|3\s*/\s*4", t):
        return 0.75
    if re.search(r"four[\s-]?fifths?|4\s*/\s*5", t):
        return 0.8
    if re.search(r"three[\s-]?fifths?|3\s*/\s*5", t):
        return 0.6
    if re.search(r"two[\s-]?fifths?|2\s*/\s*5", t):
        return 0.4
    if re.search(r"one[\s-]?fifth|1\s*/\s*5", t):
        return 0.2
    if re.search(r"\bhalf\b|one[\s-]?half|1\s*/\s*2", t):
        return 0.5
    if re.search(r"one[\s-]?third|1\s*/\s*3", t):
        return 1.0 / 3.0
    if re.search(r"\bquarter\b|one[\s-]?quarter|1\s*/\s*4", t):
        return 0.25
    return None


def _match_owner_index(text: str, owners: Sequence[str]) -> Optional[int]:
    blob = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    blob_parts = set(p for p in blob.split() if len(p) >= 3)
    if not blob_parts:
        return None
    best_i: Optional[int] = None
    best_score = 0
    for i, name in enumerate(owners):
        parts = [p for p in re.split(r"\s+", str(name or "").lower()) if len(p) >= 3]
        if not parts:
            continue
        score = sum(1 for p in parts if p in blob_parts)
        if parts[-1] in blob_parts:
            score += 1
        if score > best_score:
            best_score = score
            best_i = i
    return best_i if best_score >= 1 else None


def _owner_index_from_clause(clause: str, owners: Sequence[str]) -> Optional[int]:
    n = len(owners)
    m = re.search(
        rf"\b({ '|'.join(_ORDINAL_WORDS) })\s+(?:title\s+)?{_OWNER_ROLE}",
        clause,
        flags=re.IGNORECASE,
    )
    if m:
        idx = _ordinal_to_index(m.group(1), n)
        if idx is not None:
            return idx
    m = re.search(
        rf"\b{_OWNER_ROLE}\s*(?:no\.?|number|#)?\s*(\d+)\b",
        clause,
        flags=re.IGNORECASE,
    )
    if m:
        try:
            i = int(m.group(1)) - 1
            if 0 <= i < n:
                return i
        except (TypeError, ValueError):
            pass
    return _match_owner_index(clause, owners)


def _placement_from_clause(clause: str) -> Optional[Dict[str, Any]]:
    c = clause or ""
    if re.search(r"\b(middle|centre|center|between)\b", c, flags=re.IGNORECASE):
        return {"kind": "middle"}
    if re.search(r"\b(rear|back|behind|far\s+end)\b", c, flags=re.IGNORECASE):
        return {"kind": "rear"}
    if re.search(
        r"\b(front|frontage|roadside|along\s+the\s+(?:primary\s+)?(?:access|road))\b",
        c,
        flags=re.IGNORECASE,
    ):
        return {"kind": "front"}
    m_card = re.search(
        r"\b(north(?:ern|erly)?|south(?:ern|erly)?|east(?:ern|erly)?|west(?:ern|erly)?)\b",
        c,
        flags=re.IGNORECASE,
    )
    if m_card:
        key = re.sub(r"[^a-z]", "", m_card.group(1).lower())
        card = _PLACEMENT_CARD.get(key)
        if card:
            return {"kind": "cardinal", "dir": card}
    pillars = re.findall(
        r"\b([A-Z]{1,4}\s*/\s*[A-Z]{1,4}\s*\d{2,6})\b",
        c,
        flags=re.IGNORECASE,
    )
    if len(pillars) >= 2:
        return {
            "kind": "pillars",
            "a": re.sub(r"\s+", "", pillars[0].upper()),
            "b": re.sub(r"\s+", "", pillars[1].upper()),
        }
    return None


def query_has_share_or_placement_language(query: str) -> bool:
    q = query or ""
    return bool(
        re.search(
            r"\b(half|percent|%|share|remainder|remaining|twice|fifth|fifths|"
            r"more than|less than|square|sqm|sq\.?\s*m|"
            r"first|second|third|fourth|1st|2nd|3rd|4th|"
            r"north|south|east|west|frontage|roadside|middle|"
            r"along|joining|beside|next to)\b"
            r"|m[²2]\b",
            q,
            flags=re.IGNORECASE,
        )
    )


def share_plan_needs_llm(plan: Optional[Dict[str, Any]], query: str) -> bool:
    """True when share/placement wording is present but the CAD draft is still ambiguous."""
    if not query_has_share_or_placement_language(query):
        return False
    src = str((plan or {}).get("source") or "")
    if src in ("equal", "partial", "unsolved"):
        return True
    if (plan or {}).get("needs_llm"):
        return True
    return bool(
        re.search(
            r"more than|less than|square|sqm|m[²2]|fifth|twice|double",
            query or "",
            flags=re.IGNORECASE,
        )
        and not (plan or {}).get("relations_resolved")
    )


def _parse_area_relations(
    query: str,
    owners: Sequence[str],
    ordinal_ref: str,
) -> List[Tuple[int, int, float]]:
    """(i, j, delta_m2) meaning area_i = area_j + delta (delta may be negative)."""
    names = list(owners)
    rels: List[Tuple[int, int, float]] = []
    pat = re.compile(
        r"(" + ordinal_ref + r").{0,48}?"
        r"(\d+(?:\.\d+)?)\s*" + _AREA_UNIT + r".{0,24}?"
        r"(more|less)\s+than.{0,28}?"
        r"(" + ordinal_ref + r")",
        flags=re.IGNORECASE,
    )
    for m in pat.finditer(query or ""):
        i = _owner_index_from_clause(m.group(1), names)
        j = _owner_index_from_clause(m.group(4), names)
        try:
            delta = float(m.group(2))
        except (TypeError, ValueError):
            continue
        if str(m.group(3)).lower().startswith("less"):
            delta = -delta
        if i is None or j is None or i == j:
            continue
        rels.append((i, j, delta))
    twice = re.compile(
        r"(" + ordinal_ref + r").{0,40}?(?:twice|double).{0,28}?"
        r"(" + ordinal_ref + r")",
        flags=re.IGNORECASE,
    )
    for m in twice.finditer(query or ""):
        i = _owner_index_from_clause(m.group(1), names)
        j = _owner_index_from_clause(m.group(2), names)
        if i is None or j is None or i == j:
            continue
        # area_i = 2 * area_j  →  area_i - area_j = area_j  (needs parent solve)
        rels.append((i, j, float("inf")))  # sentinel: twice
    return rels


def _solve_share_areas(
    n: int,
    assigned_frac: Dict[int, float],
    relations: Sequence[Tuple[int, int, float]],
    parent_area: float,
) -> Optional[List[float]]:
    if parent_area <= 1e-6 or n < 2:
        return None
    areas = [None] * n  # type: List[Optional[float]]
    for i, frac in assigned_frac.items():
        if 0 <= i < n:
            areas[i] = max(0.0, float(frac) * parent_area)
    twice_pairs = [(i, j) for i, j, d in relations if d == float("inf")]
    delta_pairs = [(i, j, d) for i, j, d in relations if d != float("inf")]

    changed = True
    guard = 0
    while changed and guard < 12:
        changed = False
        guard += 1
        for i, j, d in delta_pairs:
            ai, aj = areas[i], areas[j]
            if ai is None and aj is not None:
                areas[i] = aj + d
                changed = True
            elif aj is None and ai is not None:
                areas[j] = ai - d
                changed = True
        for i, j in twice_pairs:
            ai, aj = areas[i], areas[j]
            if ai is None and aj is not None:
                areas[i] = 2.0 * aj
                changed = True
            elif aj is None and ai is not None:
                areas[j] = 0.5 * ai
                changed = True
        unknown = [k for k in range(n) if areas[k] is None]
        rem = parent_area - sum(a for a in areas if a is not None)
        if len(unknown) == 1 and rem > 1e-6:
            areas[unknown[0]] = rem
            changed = True
        elif len(unknown) == 2 and rem > 1e-6:
            a, b = unknown
            pair_d = next((d for i, j, d in delta_pairs if {i, j} == {a, b}), None)
            pair_t = next(((i, j) for i, j in twice_pairs if {i, j} == {a, b}), None)
            if pair_d is not None:
                # area_i = area_j + d
                i, j, d = next(p for p in delta_pairs if {p[0], p[1]} == {a, b})
                # i + j = rem, i = j + d
                other = rem - d
                if i == a:
                    areas[j] = other / 2.0
                    areas[i] = areas[j] + d
                else:
                    areas[j] = other / 2.0
                    areas[i] = areas[j] + d
                changed = True
            elif pair_t is not None:
                hi, lo = pair_t
                # hi + lo = rem, hi = 2*lo → 3*lo = rem
                areas[lo] = rem / 3.0
                areas[hi] = 2.0 * areas[lo]
                changed = True

    if any(a is None or a <= 1e-6 for a in areas):
        unknown = [k for k in range(n) if areas[k] is None]
        rem = parent_area - sum(a or 0.0 for a in areas)
        if unknown and rem > 1e-6 and not relations:
            each = rem / float(len(unknown))
            for k in unknown:
                areas[k] = each
        else:
            return None
    if abs(sum(float(a or 0.0) for a in areas) - parent_area) > max(1.0, 0.03 * parent_area):
        return None
    if any(float(a or 0.0) <= 1e-6 for a in areas):
        return None
    return [float(a) / parent_area for a in areas]


def parse_owner_share_plan(
    query: str,
    owners: Sequence[str],
    *,
    parent_area_m2: Optional[float] = None,
) -> Dict[str, Any]:
    """Map title owners to shares and optional sit-in-parcel placement.

    ``second owner`` / ``2nd buyer`` is the 2nd name on the title, not the
    first name and not a newly invented person.
    """
    names = [str(o).strip() for o in owners if str(o).strip()]
    n = max(1, len(names))
    if n == 1:
        return {
            "fractions": [1.0],
            "order": [0],
            "placements": {},
            "source": "single",
        }
    q = query or ""
    assigned: Dict[int, float] = {}
    ordinal_ref = (
        r"(?:the\s+)?(?:" + "|".join(_ORDINAL_WORDS) + r")\s+(?:title\s+)?" + _OWNER_ROLE
        + r"|" + _OWNER_ROLE + r"\s*(?:no\.?|number|#)?\s*\d+"
    )
    share_re = re.compile(
        r"(" + ordinal_ref + r").{0,72}?(" + _SHARE_TOKEN + r")",
        flags=re.IGNORECASE,
    )
    place_re = re.compile(
        r"(" + ordinal_ref + r").{0,80}?"
        r"(north(?:ern|erly)?|south(?:ern|erly)?|east(?:ern|erly)?|"
        r"west(?:ern|erly)?|frontage|roadside|rear|back|middle|centre|center|"
        r"joining|along|beside|next\s+to)",
        flags=re.IGNORECASE,
    )
    place_word = (
        r"north(?:ern|erly)?|south(?:ern|erly)?|east(?:ern|erly)?|"
        r"west(?:ern|erly)?|frontage|roadside|rear|back|middle|centre|center|"
        r"joining|along|beside|next\s+to"
    )

    for m in share_re.finditer(q):
        clause = m.group(0)
        if re.search(r"half\s+of\s+the\s+(?:land\s+)?remainder", clause, flags=re.IGNORECASE):
            continue
        idx = _owner_index_from_clause(m.group(1), names)
        frac = _share_token_to_frac(m.group(2))
        if idx is None or frac is None or idx in assigned:
            continue
        assigned[idx] = min(0.90, max(0.08, float(frac)))

    _skip_name = {"owner", "buyer", "purchaser", "mister", "miss", "mrs", "mr", "surv", "chief"}
    for i, name in enumerate(names):
        if i in assigned:
            continue
        tokens = [
            p
            for p in re.split(r"\s+", name)
            if len(p) >= 3 and p.lower().strip(".") not in _skip_name
        ]
        if not tokens:
            continue
        last = re.escape(tokens[-1])
        m = re.search(
            r"\b" + last + r"\b.{0,48}?(" + _SHARE_TOKEN + r")",
            q,
            flags=re.IGNORECASE,
        )
        if not m:
            continue
        if re.search(r"half\s+of\s+the\s+(?:land\s+)?remainder", m.group(0), flags=re.IGNORECASE):
            continue
        frac = _share_token_to_frac(m.group(1))
        if frac is None:
            continue
        assigned[i] = min(0.90, max(0.08, float(frac)))

    relations = _parse_area_relations(q, names, ordinal_ref)
    try:
        parent_area = float(parent_area_m2) if parent_area_m2 not in (None, "") else 0.0
    except (TypeError, ValueError):
        parent_area = 0.0
    relations_resolved = False
    solved = None
    if relations and parent_area > 1e-6:
        solved = _solve_share_areas(n, assigned, relations, parent_area)
        relations_resolved = solved is not None

    placements: Dict[int, Dict[str, Any]] = {}
    for m in place_re.finditer(q):
        idx = _owner_index_from_clause(m.group(1), names)
        hint = _placement_from_clause(m.group(0))
        if idx is None or hint is None or idx in placements:
            continue
        placements[idx] = hint
    for i, name in enumerate(names):
        if i in placements:
            continue
        tokens = [
            p
            for p in re.split(r"\s+", name)
            if len(p) >= 3 and p.lower().strip(".") not in _skip_name
        ]
        if not tokens:
            continue
        last = re.escape(tokens[-1])
        m = re.search(
            r"\b" + last + r"\b.{0,64}?(" + place_word + r")",
            q,
            flags=re.IGNORECASE,
        )
        if not m:
            continue
        hint = _placement_from_clause(m.group(0))
        if hint:
            placements[i] = hint

    fractions = [0.0] * n
    needs_llm = False
    if solved:
        fractions = list(solved)
        source = "named"
        relations_resolved = True
    elif assigned and not relations:
        used = sum(assigned.values())
        used = min(0.92, used)
        unset = [i for i in range(n) if i not in assigned]
        rem = max(0.08, 1.0 - used)
        if unset:
            each = rem / float(len(unset))
            for i in unset:
                fractions[i] = each
        for i, frac in assigned.items():
            fractions[i] = float(frac)
        total = sum(fractions) or 1.0
        fractions = [f / total for f in fractions]
        source = "named"
    elif assigned and relations and not relations_resolved:
        eq = 1.0 / float(n)
        fractions = [eq] * n
        source = "unsolved"
        needs_llm = True
    else:
        eq = 1.0 / float(n)
        fractions = [eq] * n
        source = "equal"
        if relations:
            needs_llm = True
    return {
        "fractions": fractions,
        "order": list(range(n)),
        "placements": placements,
        "source": source,
        "needs_llm": needs_llm,
        "relations_resolved": relations_resolved,
    }


def parse_owner_share_fractions(
    query: str,
    n_owners: int,
    owners: Optional[Sequence[str]] = None,
) -> List[float]:
    """Shares aligned to title order. Prefer ``parse_owner_share_plan`` when names exist."""
    n = max(1, int(n_owners or 1))
    names = list(owners) if owners else [f"Owner {i + 1}" for i in range(n)]
    if len(names) < n:
        names.extend(f"Owner {i + 1}" for i in range(len(names), n))
    return list(parse_owner_share_plan(query, names[:n]).get("fractions") or [1.0 / n] * n)


def interpret_owner_share_plan_with_llm(
    query: str,
    owners: Sequence[str],
    *,
    llm: Any,
    run_with_timeout: Callable[..., Any],
    timeout_s: int = 22,
    parent_area_m2: Optional[float] = None,
    draft: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Simple/medium structured plan for the CAD frontage-strip subdivider."""
    names = [str(o).strip() for o in owners if str(o).strip()]
    if llm is None or len(names) < 2:
        return None
    numbered = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(names))
    try:
        area = float(parent_area_m2) if parent_area_m2 not in (None, "") else 0.0
    except (TypeError, ValueError):
        area = 0.0
    draft = draft or {}
    system = (
        "You convert a surveyor's ownership request into CAD instructions.\n"
        "The CAD section will call a frontage-strip subdivider. It needs:\n"
        "  shares — one fraction per TITLE owner, same order as the numbered list, summing to 1\n"
        "  areas_m2 — optional same-length list if you reasoned in square metres\n"
        "  order — optional 1-based owner numbers along the primary access (omit if title order)\n"
        "  placements — optional [{owner: 1-based, where: north|south|east|west|front|rear|middle}]\n"
        "  notes — one sentence of the arithmetic\n"
        "  confidence — 0-1\n"
        "'Second owner' / '2nd buyer' is item 2 on the title list. Never invent owners.\n"
        "Do the arithmetic: two-fifths = 0.4; 'B has 30 m² more than C' means area_B = area_C + 30 "
        "after the stated shares are taken from the parent area.\n"
        "Return equal shares ONLY if the user asked for equal. "
        "If a constraint is impossible, still return the closest legal positive shares and lower confidence.\n"
        "JSON only. No markdown."
    )
    user = (
        f"Title owners in order:\n{numbered}\n\n"
        f"Parent parcel area (as plotted): {area:.3f} m²\n"
        f"CAD draft so far: source={draft.get('source')!s}, "
        f"shares={draft.get('fractions')!s}\n\n"
        f"User request:\n{query}\n\n"
        "Plan the split and return JSON for the CAD section."
    )
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        packed = run_with_timeout(
            timeout_s,
            lambda: llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]),
        )
        msg = packed[0] if isinstance(packed, (tuple, list)) else packed
        if msg is None:
            return None
        from survyai.provider_models import llm_visible_text_from_content

        text = llm_visible_text_from_content(getattr(msg, "content", msg))
        payload = _extract_json_object(str(text or ""))
        if not payload:
            return None
        try:
            conf = float(payload.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf < 0.25:
            return None
        raw_shares = payload.get("shares") or payload.get("fractions")
        raw_areas = payload.get("areas_m2") or payload.get("areas")
        fractions: List[float] = []
        if isinstance(raw_areas, list) and len(raw_areas) == len(names) and area > 1e-6:
            try:
                av = [max(0.0, float(x)) for x in raw_areas]
            except (TypeError, ValueError):
                av = []
            s = sum(av)
            if s > 1e-6:
                fractions = [x / s for x in av]
        if not fractions and isinstance(raw_shares, list) and len(raw_shares) == len(names):
            try:
                fractions = [max(0.0, float(x)) for x in raw_shares]
            except (TypeError, ValueError):
                fractions = []
            s = sum(fractions)
            if s <= 1e-6:
                fractions = []
            else:
                fractions = [x / s for x in fractions]
        order: List[int] = list(range(len(names)))
        raw_order = payload.get("order") or payload.get("along_frontage")
        if isinstance(raw_order, list) and len(raw_order) == len(names):
            try:
                idx = [int(x) - 1 if int(x) >= 1 else int(x) for x in raw_order]
            except (TypeError, ValueError):
                idx = []
            if sorted(idx) == list(range(len(names))):
                order = idx
        placements: Dict[int, Dict[str, Any]] = {}
        raw_pl = payload.get("placements")
        if isinstance(raw_pl, list):
            for item in raw_pl:
                if not isinstance(item, dict):
                    continue
                try:
                    oi = int(item.get("owner") or item.get("index") or 0)
                except (TypeError, ValueError):
                    continue
                if oi >= 1:
                    oi -= 1
                if not (0 <= oi < len(names)):
                    continue
                where = str(item.get("where") or item.get("side") or "").strip().lower()
                hint = _placement_from_clause(where)
                if hint:
                    placements[oi] = hint
        if not fractions:
            return None
        return {
            "fractions": fractions,
            "order": order,
            "placements": placements,
            "source": "llm",
            "confidence": conf,
            "notes": str(payload.get("notes") or "").strip(),
        }
    except Exception:
        return None


def merge_owner_share_plans(
    base: Dict[str, Any],
    extra: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    out = dict(base or {})
    if not extra:
        return out
    extra_fracs = extra.get("fractions") or []
    base_fracs = out.get("fractions") or []
    if extra_fracs and len(extra_fracs) == len(base_fracs):
        unclear = str(out.get("source") or "") in ("equal", "partial", "unsolved") or out.get(
            "needs_llm"
        )
        named_ok = str(out.get("source") or "") == "named" and not out.get("needs_llm")
        # Keep a solved regex plan unless the LLM is quite sure. Prefer LLM when CAD is unclear.
        if unclear or (not named_ok and float(extra.get("confidence") or 0) >= 0.45):
            out["fractions"] = list(extra_fracs)
            out["source"] = extra.get("source") or out.get("source")
            out["needs_llm"] = False
            out["relations_resolved"] = True
        elif named_ok and float(extra.get("confidence") or 0) >= 0.80:
            out["fractions"] = list(extra_fracs)
            out["source"] = extra.get("source") or out.get("source")
    if extra.get("order") and sorted(extra["order"]) == list(range(len(out.get("fractions") or []))):
        if extra.get("source") == "llm" and extra.get("order") != list(range(len(base_fracs))):
            out["order"] = list(extra["order"])
    merged_pl = dict(out.get("placements") or {})
    merged_pl.update(extra.get("placements") or {})
    out["placements"] = merged_pl
    return out


def resolve_frontage_owner_order(
    plan: Dict[str, Any],
    ring: Sequence[Any],
    front_edge_index: int,
    *,
    pn_list: Optional[Sequence[Dict[str, str]]] = None,
) -> List[int]:
    """Title order along the access, unless the user sat an owner on a side."""
    fracs = list(plan.get("fractions") or [])
    n = len(fracs)
    order = list(plan.get("order") or list(range(n)))
    if sorted(order) != list(range(n)):
        order = list(range(n))
    placements = plan.get("placements") or {}
    if not placements:
        return order
    pts = _ring_xy(ring)
    npts = len(pts)
    if npts < 2:
        return order
    front = int(front_edge_index) % npts
    f0, f1 = pts[front], pts[(front + 1) % npts]

    def _slot_for(hint: Dict[str, Any]) -> Optional[int]:
        kind = str(hint.get("kind") or "")
        if kind == "front":
            return 0
        if kind == "rear":
            return n - 1
        if kind == "middle":
            return n // 2
        if kind == "cardinal":
            d = str(hint.get("dir") or "")
            if d == "N":
                start = f1[1] >= f0[1]
            elif d == "S":
                start = f1[1] <= f0[1]
            elif d == "E":
                start = f1[0] >= f0[0]
            elif d == "W":
                start = f1[0] <= f0[0]
            else:
                return None
            return n - 1 if start else 0
        if kind == "pillars" and pn_list:
            want = {
                re.sub(r"\s+", "", str(hint.get("a") or "")),
                re.sub(r"\s+", "", str(hint.get("b") or "")),
            }
            labels = []
            for rec in pn_list:
                pref = str((rec or {}).get("prefix") or "").replace(" ", "")
                num = str((rec or {}).get("number") or "").replace(" ", "")
                labels.append(f"{pref}{num}".upper())
            hit = -1
            for i, lab in enumerate(labels):
                nxt = labels[(i + 1) % len(labels)] if labels else ""
                pair = {lab, nxt}
                if want <= pair or any(w and w in pair for w in want if w):
                    if want.issubset({lab, nxt}) or (
                        any(w.endswith(lab[-4:]) for w in want if lab)
                        and any(w.endswith(nxt[-4:]) for w in want if nxt)
                    ):
                        hit = i
                        break
            if hit < 0:
                return None
            if hit == front:
                return None
            if hit == (front - 1) % npts:
                return 0
            if hit == (front + 1) % npts:
                return n - 1
        return None

    for owner_i, hint in placements.items():
        try:
            oi = int(owner_i)
        except (TypeError, ValueError):
            continue
        if oi not in order:
            continue
        slot = _slot_for(hint if isinstance(hint, dict) else {})
        if slot is None:
            continue
        order.remove(oi)
        slot = max(0, min(len(order), slot))
        order.insert(slot, oi)
    return order


def _xy_from_pt(pt: Any) -> Optional[Tuple[float, float]]:
    if isinstance(pt, dict):
        try:
            return (
                float(pt.get("x", pt.get("e", pt.get("easting")))),
                float(pt.get("y", pt.get("n", pt.get("northing")))),
            )
        except (TypeError, ValueError):
            return None
    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
        try:
            return (float(pt[0]), float(pt[1]))
        except (TypeError, ValueError):
            return None
    return None


def _ring_xy(points: Sequence[Any]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for p in points or []:
        xy = _xy_from_pt(p)
        if xy is None:
            continue
        if out and abs(out[-1][0] - xy[0]) < 1e-9 and abs(out[-1][1] - xy[1]) < 1e-9:
            continue
        out.append(xy)
    if len(out) >= 2 and abs(out[0][0] - out[-1][0]) < 1e-9 and abs(out[0][1] - out[-1][1]) < 1e-9:
        out = out[:-1]
    return out


def _ring_area(points: Sequence[Any]) -> float:
    ring = _ring_xy(points)
    if len(ring) < 3:
        return 0.0
    acc = 0.0
    for i, (x1, y1) in enumerate(ring):
        x2, y2 = ring[(i + 1) % len(ring)]
        acc += x1 * y2 - x2 * y1
    return abs(acc) * 0.5


def _lerp_xy(a: Tuple[float, float], b: Tuple[float, float], t: float) -> Tuple[float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _orient2d(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
    d: Tuple[float, float],
    *,
    eps: float = 1e-9,
) -> bool:
    """True when open segments AB and CD properly intersect (shared endpoints ignored)."""
    o1 = _orient2d(a, b, c)
    o2 = _orient2d(a, b, d)
    o3 = _orient2d(c, d, a)
    o4 = _orient2d(c, d, b)
    if abs(o1) < eps or abs(o2) < eps or abs(o3) < eps or abs(o4) < eps:
        return False
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def _lot_ring_from_front_rear(
    f0: Tuple[float, float],
    f1: Tuple[float, float],
    r0: Tuple[float, float],
    r1: Tuple[float, float],
    t0: float,
    t1: float,
) -> List[Dict[str, float]]:
    fa = _lerp_xy(f0, f1, t0)
    fb = _lerp_xy(f0, f1, t1)
    ra = _lerp_xy(r0, r1, t0)
    rb = _lerp_xy(r0, r1, t1)
    return [
        {"x": fa[0], "y": fa[1]},
        {"x": fb[0], "y": fb[1]},
        {"x": rb[0], "y": rb[1]},
        {"x": ra[0], "y": ra[1]},
    ]


def pick_frontage_edge_index(
    ring: Sequence[Any],
    preferred_edge_indices: Optional[Sequence[int]] = None,
) -> int:
    """Street frontage: first named access-road edge, else the longest traverse leg."""
    pts = _ring_xy(ring)
    n = len(pts)
    if n < 3:
        return 0
    for idx in preferred_edge_indices or []:
        try:
            i = int(idx) % n
        except (TypeError, ValueError):
            continue
        return i
    best_i = 0
    best_l = -1.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        leng = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if leng > best_l:
            best_l = leng
            best_i = i
    return best_i


def subdivide_parcel_frontage_strips(
    ring: Sequence[Any],
    fractions: Sequence[float],
    *,
    front_edge_index: int = 0,
) -> Dict[str, Any]:
    """Area-true lots from the frontage to the opposite side. Internals never cross.

    Each owner keeps a strip of the primary access frontage — the practical
    way to share a rectangular/parallelogram cadastral plot.
    """
    pts = _ring_xy(ring)
    n = len(pts)
    empty = {"lots": [], "cuts": [], "front_edge_index": 0, "ok": False}
    if n < 4 or not fractions:
        return empty
    front = int(front_edge_index) % n
    # Opposite edge for a typical 4-sided traverse; for n>4 use the far edge.
    rear = (front + (n // 2)) % n
    f0, f1 = pts[front], pts[(front + 1) % n]
    # Same parametric direction along the rear as along the frontage.
    r0, r1 = pts[(rear + 1) % n], pts[rear]
    parent_area = _ring_area(pts)
    if parent_area <= 1e-6:
        return empty

    clean = [max(0.0, float(f)) for f in fractions if float(f) > 1e-6]
    if not clean:
        return empty
    s = sum(clean)
    clean = [f / s for f in clean]

    ts: List[float] = [0.0]
    t_prev = 0.0
    for i, frac in enumerate(clean):
        if i == len(clean) - 1:
            ts.append(1.0)
            break
        target = frac * parent_area
        lo, hi = t_prev + 1e-5, 1.0 - 1e-5
        mid = t_prev
        for _ in range(42):
            mid = 0.5 * (lo + hi)
            area = _ring_area(_lot_ring_from_front_rear(f0, f1, r0, r1, t_prev, mid))
            if area < target:
                lo = mid
            else:
                hi = mid
        t_prev = 0.5 * (lo + hi)
        ts.append(t_prev)

    lots: List[Dict[str, Any]] = []
    cuts: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for i in range(len(clean)):
        t0, t1 = ts[i], ts[i + 1]
        lot = _lot_ring_from_front_rear(f0, f1, r0, r1, t0, t1)
        lots.append(
            {
                "points": lot,
                "area_m2": _ring_area(lot),
                "share": clean[i],
            }
        )
        if 1e-4 < t1 < 1.0 - 1e-4:
            fa = _lerp_xy(f0, f1, t1)
            ra = _lerp_xy(r0, r1, t1)
            cuts.append((fa, ra))

    for i, (a1, b1) in enumerate(cuts):
        for a2, b2 in cuts[i + 1 :]:
            if _segments_cross(a1, b1, a2, b2):
                return empty
    return {
        "lots": lots,
        "cuts": [{"x1": a[0], "y1": a[1], "x2": b[0], "y2": b[1]} for a, b in cuts],
        "front_edge_index": front,
        "ok": len(lots) == len(clean),
    }
