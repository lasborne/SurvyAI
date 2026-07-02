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
from typing import Any, Callable, Dict, List, Optional, Sequence

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


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    for candidate in (raw,):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


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
        text = msg.content if hasattr(msg, "content") else str(msg)
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
        text = msg.content if hasattr(msg, "content") else str(msg)
        if isinstance(text, list):
            text = "\n".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in text
            )
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
    enough or the assessment found more roads than regex alone.
    """
    merged: List[str] = list(regex_specs or [])
    seen = {_normalize_key(s) for s in merged}

    for road in assessed:
        spec = access_road_to_spec(road)
        key = _normalize_key(spec)
        if not key or key in seen:
            continue
        add = (
            confidence >= min_confidence
            or not regex_specs
            or len(assessed) > len(regex_specs)
        )
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
