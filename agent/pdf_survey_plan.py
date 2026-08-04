"""
Extract survey/cadastral plan data from PDF files and build cadastral CAD prompts.

PDF text layers are often fragmented; this module:
1. Reconstructs layout-ordered text (pdfplumber word positions + tables)
2. Optionally renders page images for vision-capable LLMs (PyMuPDF)
3. Uses a complex-tier LLM pass to produce structured traverse + metadata
4. Emits a sub-prompt compatible with _run_cadastral_cad_prompt_pipeline
"""

from __future__ import annotations

import base64
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import calendar
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

from pydantic import BaseModel, Field

from utils.logger import get_logger

logger = get_logger(__name__)

_T = TypeVar("_T")


def _parallel_invoke(*callables: Callable[[], _T]) -> List[_T]:
    """Run independent callables concurrently; single-call fast path avoids thread overhead."""
    if not callables:
        return []
    if len(callables) == 1:
        return [callables[0]()]
    workers = min(len(callables), 8)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn) for fn in callables]
        return [f.result() for f in futures]


def _load_pdf_extraction_sources(
    pdf_path: str,
    *,
    vision_max_pages: int = 1,
) -> tuple[str, str, List[str]]:
    """Load layout text, plain text, and vision images from a PDF in parallel."""
    max_pages = max(1, int(vision_max_pages or 1))
    layout_text, plain_text, images = _parallel_invoke(
        lambda: extract_layout_text_from_pdf(pdf_path),
        lambda: extract_plain_text_from_pdf(pdf_path),
        lambda: render_pdf_pages_base64(pdf_path, max_pages=max_pages),
    )
    return layout_text, plain_text, images


class SurveyTraverseLeg(BaseModel):
    from_pillar: str = ""
    to_pillar: str = ""
    bearing_deg: int = 0
    bearing_min: int = 0
    distance_m: float = 0.0


class SurveyPlanExtraction(BaseModel):
    """Structured fields read from a survey plan PDF."""

    buyer_name: str = ""
    location: str = ""
    lga: str = ""
    state: str = ""
    origin_crs: str = ""
    plan_number: str = ""
    surveyor_name: str = ""
    surveyor_address: str = ""
    area_sq_m: Optional[float] = None
    scale_denom: Optional[int] = None
    pillar_numbers: List[str] = Field(default_factory=list)
    anchor_easting: Optional[float] = None
    anchor_northing: Optional[float] = None
    anchor_pillar: str = ""
    traverse_legs: List[SurveyTraverseLeg] = Field(default_factory=list)
    access_roads: List[str] = Field(default_factory=list)
    access_road_title: str = ""
    fences: List[str] = Field(default_factory=list)
    grid_easting_pillar: str = ""
    grid_northing_pillar: str = ""
    absolute_parcel_coords: List[Dict[str, float]] = Field(default_factory=list)
    certification_date: str = ""
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    source: str = "none"
    notes: str = ""


class SurveyPlanOverrides(BaseModel):
    """Partial plan fields the user asked to change on top of PDF extraction."""

    buyer_name: str | None = None
    location: str | None = None
    lga: str | None = None
    state: str | None = None
    origin_crs: str | None = None
    plan_number: str | None = None
    surveyor_name: str | None = None
    surveyor_address: str | None = None
    certification_date: str | None = None
    scale_denom: int | None = None
    area_sq_m: float | None = None
    pillar_numbers: list[str] | None = None
    anchor_easting: float | None = None
    anchor_northing: float | None = None
    anchor_pillar: str | None = None
    traverse_legs: list[SurveyTraverseLeg] | None = None
    access_roads: list[str] | None = None
    fences: list[str] | None = None
    access_road_title: str | None = None
    override_fields: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""


SurveyPlanOverrides.model_rebuild()


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
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


def extract_pdf_paths_from_text(text: str) -> List[str]:
    """Find PDF paths in a query string (does not require the file to exist)."""
    found: List[str] = []
    patterns = [
        r"([A-Za-z]:\\[^\r\n\"<>|]+?\.pdf)",
        r"((?:/|\\)[^\r\n\"<>|]+?\.pdf)",
        r"(?<![A-Za-z0-9_/\\])([A-Za-z0-9][A-Za-z0-9_\- ]*?\.pdf)(?![A-Za-z0-9])",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text or "", flags=re.IGNORECASE):
            raw = (m.group(1) or "").strip().strip("\"'").rstrip(").,;")
            if not raw:
                continue
            try:
                resolved = str(Path(raw).resolve())
            except Exception:
                resolved = raw
            if resolved not in found:
                found.append(resolved)
    return found


def _is_plausible_utm_easting(value: float) -> bool:
    return 50000.0 <= float(value) <= 950000.0


def _is_plausible_utm_northing(value: float) -> bool:
    return 50000.0 <= float(value) <= 15000000.0


def _collapse_spaced_coordinate_text(text: str) -> str:
    """Join split thousands in UTM labels (e.g. '537 935.100m.N' -> '537935.100m.N')."""
    if not text:
        return ""
    out = text
    for _ in range(3):
        collapsed = re.sub(
            r"(\d{3})\s+(\d{3}(?:\.\d+)?)(\s*m\.?\s*[EN]\b)",
            r"\1\2\3",
            out,
            flags=re.IGNORECASE,
        )
        collapsed = re.sub(
            r"(\d{3})\s+(\d{3}(?:\.\d+)?)(?=\s*m\.?\s*[EN]\b)",
            r"\1\2",
            collapsed,
            flags=re.IGNORECASE,
        )
        if collapsed == out:
            break
        out = collapsed
    return out


def _absolute_coords_are_plausible_utm(coords: Sequence[Dict[str, float]]) -> bool:
    if not coords or len(coords) < 3:
        return False
    for p in coords:
        e = float(p.get("e", 0.0))
        n = float(p.get("n", 0.0))
        if not _is_plausible_utm_easting(e) or not _is_plausible_utm_northing(n):
            return False
    return True


def _coordinate_source_text(extraction: SurveyPlanExtraction, combined_text: str = "") -> str:
    parts = [combined_text or "", extraction.notes or ""]
    return _collapse_spaced_coordinate_text("\n".join(p for p in parts if p))


def enrich_extraction_coordinates(
    extraction: SurveyPlanExtraction,
    combined_text: str = "",
) -> SurveyPlanExtraction:
    """
    Resolve full UTM coordinates from notes, PDF text, anchors, and traverse geometry.

    Scanned plans often yield northing-only grid anchoring with relative eastings; this
    repairs that using labelled E/N values in extraction notes or combined PDF text.
    """
    source = _coordinate_source_text(extraction, combined_text)
    best_e, best_n, _, _ = extract_grid_coordinates_from_text(source)

    if extraction.anchor_easting is None or not _is_plausible_utm_easting(extraction.anchor_easting):
        if best_e is not None:
            extraction.anchor_easting = best_e
    if extraction.anchor_northing is None or not _is_plausible_utm_northing(extraction.anchor_northing):
        if best_n is not None:
            extraction.anchor_northing = best_n

    if extraction.absolute_parcel_coords and not _absolute_coords_are_plausible_utm(
        extraction.absolute_parcel_coords
    ):
        extraction.absolute_parcel_coords = []

    pillars = _filter_plausible_pillars(extraction.pillar_numbers)
    legs = _filter_plausible_legs(extraction.traverse_legs)
    if not extraction.absolute_parcel_coords and len(pillars) >= 3 and len(legs) >= 3:
        grid_e = extraction.anchor_easting if _is_plausible_utm_easting(extraction.anchor_easting or 0) else best_e
        grid_n = extraction.anchor_northing if _is_plausible_utm_northing(extraction.anchor_northing or 0) else best_n
        grid_e_pillar = extraction.grid_easting_pillar or extraction.anchor_pillar or ""
        grid_n_pillar = extraction.grid_northing_pillar or extraction.anchor_pillar or ""
        if not grid_e_pillar and grid_e is not None and pillars:
            rel = _compute_relative_traverse_vertices(pillars, legs)
            if rel:
                grid_e_pillar = pillars[_pick_primary_pillar_index(rel)]
        if not grid_n_pillar and grid_n is not None and pillars:
            rel = _compute_relative_traverse_vertices(pillars, legs)
            if rel:
                ns = [float(p["n"]) for p in rel]
                max_n = max(ns)
                spread = float(max(ns) - min(ns)) if len(ns) > 1 else 0.0
                eps_n = max(0.01, spread * 1e-9)
                north_idx = int(
                    max(
                        [i for i in range(len(rel)) if abs(float(rel[i]["n"]) - max_n) <= eps_n],
                        key=lambda i: float(rel[i]["e"]),
                    )
                )
                grid_n_pillar = pillars[north_idx]

        abs_try = _compute_absolute_parcel_coordinates(
            extraction,
            grid_e=grid_e,
            grid_e_pillar=grid_e_pillar,
            grid_n=grid_n,
            grid_n_pillar=grid_n_pillar,
        )
        if abs_try and _absolute_coords_are_plausible_utm(abs_try):
            extraction.absolute_parcel_coords = abs_try
            primary_idx = _pick_primary_pillar_index(abs_try)
            extraction.anchor_easting = float(abs_try[primary_idx]["e"])
            extraction.anchor_northing = float(abs_try[primary_idx]["n"])
            if primary_idx < len(pillars):
                extraction.anchor_pillar = pillars[primary_idx]

    return extraction


# Nigerian cadastral pillar labels (CADA_PILLARNUMBERS top/bottom cells).
# Classic: SC/AS 2457, SP/RV 33567. Longer district: SC/AKAB 19155.
# Alphanumeric peg tokens: SC/DT AS3459RP, SC/RV OA94567KL, SC/EN IL3456PX.
# Also seen: RV/SP 2345 (series before slash is not always SC/SP).
_PILLAR_SERIES = r"(?:SC|SP|RV|RP)"
_PILLAR_DISTRICT = r"[A-Z]{1,6}"
_PILLAR_PEG_TOKEN = r"[A-Z0-9]{3,12}"
_PILLAR_ID_TEXT_RE = re.compile(
    rf"(?:{_PILLAR_SERIES})\s*/?\s*{_PILLAR_DISTRICT}\s*{_PILLAR_PEG_TOKEN}",
    re.IGNORECASE,
)
_PILLAR_PREFIX_TOKEN_RE = re.compile(
    rf"^({_PILLAR_SERIES})\s*/?\s*({_PILLAR_DISTRICT})$",
    re.IGNORECASE,
)
_PILLAR_NUMBER_TOKEN_RE = re.compile(rf"^({_PILLAR_PEG_TOKEN})$", re.IGNORECASE)


def split_cadastral_pillar_label(raw: str) -> Optional[Dict[str, str]]:
    """
    Split a Nigerian cadastral pillar id into CADA_PILLARNUMBERS table cells.

    Returns ``{{"prefix": "SC/DT", "number": "AS3459RP"}}`` (top / bottom rows).
    Accepts classic digit pegs and longer alphanumeric peg tokens (up to ~9+ chars).
    """
    text = re.sub(r"\s+", " ", (raw or "").strip())
    if not text:
        return None
    # Spaced form: "SC/AKAB 19155", "SC/DT AS3459RP", "RV/SP 2345"
    m = re.match(
        rf"^({_PILLAR_SERIES})\s*/\s*({_PILLAR_DISTRICT})\s+({_PILLAR_PEG_TOKEN})$",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        prefix = f"{m.group(1).upper()}/{m.group(2).upper()}"
        number = m.group(3).upper()
        if not re.search(r"\d", number):
            return None
        return {"prefix": prefix, "number": number}

    compact = re.sub(r"\s+", "", text.upper())
    m_head = re.match(rf"^({_PILLAR_SERIES})/?(.*)$", compact)
    if not m_head:
        # Last resort: any "XX/YYY rest" with a digit somewhere in rest
        m = re.match(
            r"^([A-Za-z]{1,4}\s*/\s*[A-Za-z]{1,6})\s+([A-Za-z0-9]{3,12})$",
            text,
        )
        if m and re.search(r"\d", m.group(2)):
            prefix = re.sub(r"\s+", "", m.group(1)).upper()
            return {"prefix": prefix, "number": m.group(2).upper()}
        return None

    series = m_head.group(1)
    rest = m_head.group(2) or ""
    digit_hit: Optional[Dict[str, str]] = None
    alpha_hit: Optional[Dict[str, str]] = None
    alpha_peg_re = re.compile(r"^[A-Z]{2,4}\d{3,9}[A-Z]{0,4}$")
    for dlen in range(1, min(6, max(0, len(rest) - 2)) + 1):
        district = rest[:dlen]
        peg = rest[dlen:]
        if not re.fullmatch(r"[A-Z]{1,6}", district):
            continue
        if not re.fullmatch(r"[A-Z0-9]{3,12}", peg) or not re.search(r"\d", peg):
            continue
        cand = {"prefix": f"{series}/{district}", "number": peg}
        if peg.isdigit() and 3 <= len(peg) <= 9:
            # Prefer longest district for classic digit pegs (SC/BV 6015 over SC/B V6015).
            digit_hit = cand
        elif alpha_peg_re.match(peg):
            # Prefer longest district for alphanumeric pegs (SC/DT AS3459RP).
            alpha_hit = cand
    if digit_hit:
        return digit_hit
    if alpha_hit:
        return alpha_hit

    m = re.match(
        r"^([A-Za-z]{1,4}\s*/\s*[A-Za-z]{1,6})\s+([A-Za-z0-9]{3,12})$",
        text,
    )
    if m and re.search(r"\d", m.group(2)):
        prefix = re.sub(r"\s+", "", m.group(1)).upper()
        return {"prefix": prefix, "number": m.group(2).upper()}
    return None


def _parse_pillar_token(tok: str, next_tok: str = "") -> Optional[str]:
    """Parse one or two PDF word tokens into a normalized pillar id."""
    raw = (tok or "").strip()
    if not raw:
        return None
    split = split_cadastral_pillar_label(raw)
    if split:
        return _normalize_pillar_id(f"{split['prefix']} {split['number']}")
    if _PILLAR_PREFIX_TOKEN_RE.match(raw):
        nxt = (next_tok or "").strip()
        if _PILLAR_NUMBER_TOKEN_RE.match(nxt) and re.search(r"\d", nxt):
            return _normalize_pillar_id(f"{raw} {nxt}")
    m2 = _PILLAR_ID_TEXT_RE.match(raw)
    if m2:
        return _normalize_pillar_id(m2.group(0))
    return None


def _order_pillars_clockwise(
    positions: Dict[str, tuple[float, float]],
    pillars: Sequence[str],
) -> List[str]:
    """Order pillar labels clockwise around the parcel centroid (PDF page space)."""
    pts = [(p, positions[p]) for p in pillars if p in positions]
    if len(pts) < 3:
        return list(pillars)
    cx = sum(x for _, (x, _y) in pts) / len(pts)
    cy = sum(y for _, (_x, y) in pts) / len(pts)

    def _angle(item: tuple[str, tuple[float, float]]) -> float:
        _p, (x, y) = item
        return math.atan2(y - cy, x - cx)

    ordered = [p for p, _ in sorted(pts, key=_angle)]
    if len(ordered) < len(pillars):
        for p in pillars:
            if p not in ordered:
                ordered.append(p)
    return ordered


def extract_pillars_from_pdf_page(page: Any) -> List[str]:
    """
    Read pillar labels from PDF word geometry (handles split tokens like SC/BV + 6015).
    """
    try:
        words = page.get_text("words") or []
    except Exception:
        return []
    if not words:
        return []

    found: Dict[str, tuple[float, float]] = {}
    i = 0
    while i < len(words):
        w = words[i]
        tok = str(w[4] or "").strip()
        nxt = str(words[i + 1][4] or "").strip() if i + 1 < len(words) else ""
        pid = _parse_pillar_token(tok, nxt)
        if pid and _is_plausible_pillar_id(pid):
            cx = (float(w[0]) + float(w[2])) / 2.0
            cy = (float(w[1]) + float(w[3])) / 2.0
            if _PILLAR_PREFIX_TOKEN_RE.match(tok) and _PILLAR_NUMBER_TOKEN_RE.match(nxt):
                w2 = words[i + 1]
                cx = (cx + (float(w2[0]) + float(w2[2])) / 2.0) / 2.0
                cy = (cy + (float(w2[1]) + float(w2[3])) / 2.0) / 2.0
                i += 1
            found[pid] = (cx, cy)
        i += 1

    if len(found) < 3:
        return []
    return _order_pillars_clockwise(found, list(found.keys()))


def _extraction_geometry_is_usable(extraction: SurveyPlanExtraction) -> bool:
    pillars = _filter_plausible_pillars(extraction.pillar_numbers)
    legs = _filter_plausible_legs(extraction.traverse_legs)
    if len(pillars) < 3 or len(legs) < 3:
        return False
    if len(pillars) != len(legs):
        return False
    return True


def repair_survey_extraction_from_pdf(
    extraction: SurveyPlanExtraction,
    pdf_path: Optional[str],
    combined_text: str,
) -> SurveyPlanExtraction:
    """
    Proactively repair weak LLM/heuristic extraction using PDF geometry and text.

    Only replaces missing or inconsistent pillars/traverse — never overwrites a
    complete, self-consistent extraction.
    """
    if _extraction_geometry_is_usable(extraction):
        return extraction

    pillars = _filter_plausible_pillars(extraction.pillar_numbers)
    legs = _filter_plausible_legs(extraction.traverse_legs)

    if pdf_path and Path(pdf_path).exists():
        try:
            import fitz

            doc = fitz.open(str(pdf_path))
            page = doc[0]
            pdf_pillars = extract_pillars_from_pdf_page(page)
            if len(pdf_pillars) >= 3:
                if len(pillars) < 3 or len(pillars) != len(pdf_pillars):
                    extraction.pillar_numbers = pdf_pillars
                    pillars = pdf_pillars
                    extraction.notes = (
                        f"{extraction.notes or ''} | pillars from PDF geometry".strip(" |")
                    )
            doc.close()
        except Exception as exc:
            logger.debug("PDF pillar repair failed: %s", exc)

    if len(pillars) < 3:
        norm_text = re.sub(r"\s+", " ", (combined_text or "").replace("\n", " "))
        for m in _PILLAR_ID_TEXT_RE.finditer(norm_text):
            pid = _normalize_pillar_id(m.group(0))
            if _is_plausible_pillar_id(pid) and pid not in pillars:
                pillars.append(pid)
        if len(pillars) >= 3:
            extraction.pillar_numbers = pillars[: min(len(pillars), 12)]

    if len(pillars) >= 3 and (
        len(legs) < 3 or len(legs) != len(pillars) or not _extraction_geometry_is_usable(extraction)
    ):
        pdf_legs = extract_boundary_legs_from_pdf(pdf_path, pillars) if pdf_path else []
        if len(pdf_legs) >= 3 and len(pdf_legs) == len(pillars):
            extraction.traverse_legs = pdf_legs
            legs = pdf_legs
            extraction.notes = (
                f"{extraction.notes or ''} | traverse from PDF geometry".strip(" |")
            )

    if _extraction_geometry_is_usable(extraction):
        n = len(extraction.pillar_numbers)
        for i, leg in enumerate(extraction.traverse_legs):
            if not leg.from_pillar and i < n:
                leg.from_pillar = extraction.pillar_numbers[i]
            if not leg.to_pillar and i < n:
                leg.to_pillar = extraction.pillar_numbers[(i + 1) % n]

    return extraction


def validate_extraction_for_replot(extraction: SurveyPlanExtraction) -> List[str]:
    """Return human-readable issues that block a safe PDF→CAD replot."""
    issues: List[str] = []
    pillars = _filter_plausible_pillars(extraction.pillar_numbers)
    legs = _filter_plausible_legs(extraction.traverse_legs)
    if len(pillars) < 3:
        issues.append("fewer than three pillar numbers extracted")
    if len(legs) < 3:
        issues.append("fewer than three traverse legs extracted")
    elif len(pillars) >= 3 and len(legs) != len(pillars):
        issues.append(
            f"pillar count ({len(pillars)}) does not match traverse legs ({len(legs)})"
        )
    if extraction.anchor_easting is None and extraction.anchor_northing is None:
        if not extraction.absolute_parcel_coords:
            issues.append("no anchor coordinates or absolute parcel coordinates")
    elif extraction.anchor_easting is None or extraction.anchor_northing is None:
        if not extraction.absolute_parcel_coords:
            issues.append("incomplete anchor easting/northing pair")
    if (
        extraction.absolute_parcel_coords
        and len(extraction.absolute_parcel_coords) >= 3
        and not _absolute_coords_are_plausible_utm(extraction.absolute_parcel_coords)
    ):
        issues.append("absolute coordinates are not valid UTM eastings/northings")
    elif (
        not extraction.absolute_parcel_coords
        and _extraction_geometry_is_usable(extraction)
        and (
            extraction.anchor_easting is None
            or extraction.anchor_northing is None
            or not _is_plausible_utm_easting(extraction.anchor_easting)
            or not _is_plausible_utm_northing(extraction.anchor_northing)
        )
    ):
        issues.append("traverse data present but coordinates could not be resolved")
    return issues


def validate_subprompt_geometry(subprompt: str) -> List[str]:
    """Ensure the cadastral sub-prompt carries real geometry, not template placeholders."""
    issues: List[str] = []
    if not re.search(r"pillar\s+numbers\s*[:=]", subprompt, re.I):
        issues.append("subprompt missing pillar numbers")
    if not re.search(r"coordinates\s+for\s+the\s+points\s*=", subprompt, re.I):
        issues.append("subprompt missing coordinates")
    elif not re.search(r"\(\s*\d{5,7}(?:\.\d+)?\s*m?\s*[eE]", subprompt):
        if not re.search(r"\d{5,7}(?:\.\d+)?\s*m?\s*[eE]\s*,", subprompt, re.I):
            issues.append("subprompt coordinates look incomplete")
    return issues


# Stop metadata captures at the next labelled field — newline OR comma-separated (prompt-to-CAD).
# Includes scale / Plot-using so surveyor name/address never swallow plan-scale lines.
CADASTRAL_FIELD_BOUNDARY = (
    r"(?=(?:,\s*|\n\s*)"
    r"(?:location|local\s+(?:govt\.?|government)\s+area|state|"
    r"crs_?origin|origin_?crs|plan\s*(?:no\.?|number)|Surveyor\s+name|"
    r"Surveyor\s+company\s+and\s+address|Surveyor\s+company|pillar\s+numbers?|"
    r"coordinates\s+for|title\s+as|Plot\s+using(?:\s+scale)?|date\s+on\s+the|"
    r"buyer\s*'?s?\s*name|"
    r"scale\s*[:=]\s*1\s*[:/]\s*\d+|"
    r"scale\s+1\s*[:/]\s*\d+"
    r")(?:\s*[:=]|\b)"
    r"|,\s*(?:Add\s+(?:an?\s+)?access|Add\s+\d+)\b"
    r"|\Z)"
)

_COORDINATES_FOR_STOP = r"coordinates\s+for\s+the\s+point(?:s)?\s*(?:[^\n:=]+?\s*)?[:=]"

# Shared with agent.cadastral parser (pillar list capture stops here).
CADASTRAL_COORDINATES_FOR_STOP = _COORDINATES_FOR_STOP


def _trim_coordinates_blob(blob: str) -> str:
    """Drop access-road / fence tail accidentally captured after traverse legs."""
    s = (blob or "").strip()
    if not s:
        return ""
    for pat in (
        r"\.\s*Add\s+(?:\d+\s+)?(?:Concrete\s+wall\s+fence|Dwarf\s+Concrete|an?\s+access)",
        r";\s*Add\s+(?:\d+\s+)?(?:Concrete|an?\s+access|another)",
        r"\.\s*Generate\b",
        r";\s*Generate\b",
    ):
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            s = s[: m.start()].strip().rstrip(",;")
    return s.strip()


def extract_coordinates_blob_from_cadastral_query(text: str) -> str:
    """Extract traverse/coordinate text after 'coordinates for the point(s)'."""
    source = text or ""
    blob = ""

    m = re.search(
        rf"{_COORDINATES_FOR_STOP}\s*(.+)$",
        source,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        blob = (m.group(1) or "").strip()
    if not blob:
        m = re.search(
            r"coordinates\s+for\s+the\s+points?\s*=\s*(.+)$",
            source,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m:
            blob = (m.group(1) or "").strip()
    if not blob:
        m = re.search(
            r"(?:plot|draw|use)\s+(?:using\s+)?coordinate[s]?\s*[:=]?\s*(.+)$",
            source,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m:
            blob = (m.group(1) or "").strip()

    return _trim_coordinates_blob(blob)


def cadastral_query_has_coordinates(text: str) -> bool:
    """True when the prompt appears to include coordinate/traverse instructions."""
    ql = (text or "").lower()
    if re.search(r"coordinates\s+for\s+the\s+points?\b", ql):
        return True
    if re.search(r"(?:plot|draw|use)\s+(?:using\s+)?coordinate[s]?\b", ql):
        return True
    return bool(re.search(r"\d{5,7}(?:\.\d+)?\s*m?\s*[eE]\s*[,; ]+\s*\d{5,7}(?:\.\d+)?\s*m?\s*[nN]", text or ""))


def resolve_cadastral_coordinates_blob(
    text: str,
    *,
    pillar_list: Optional[Sequence[str]] = None,
    llm: Any = None,
    run_with_timeout: Optional[Callable[..., Any]] = None,
    vector_store: Any = None,
    search_fn: Optional[Callable[..., Any]] = None,
    timeout_s: int = 35,
) -> str:
    """
    Resolve coordinate/traverse blob from varied cadastral prompt phrasing.

    Fast regex runs first; when that fails but coordinates are implied, a cheap
    LLM pass (optionally informed by similar vector-store prompts) extracts the blob.
    """
    blob = extract_coordinates_blob_from_cadastral_query(text)
    if blob:
        return blob

    if not cadastral_query_has_coordinates(text):
        return ""

    if llm is None or run_with_timeout is None:
        return ""

    try:
        from agent.cadastral_intent import parse_cadastral_geometry_blob_with_llm
    except Exception:
        return ""

    return parse_cadastral_geometry_blob_with_llm(
        text,
        pillar_numbers=list(pillar_list or []),
        llm=llm,
        run_with_timeout=run_with_timeout,
        vector_store=vector_store,
        search_fn=search_fn,
        timeout_s=timeout_s,
    )


def _pick_cadastral_value(text: str, patterns: Sequence[str]) -> str:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return (m.group(1) or "").strip()
    return ""


_METADATA_JUNK_RE = re.compile(
    r"coordinates\s+for|bearing\s+\d|pillar\s+numbers|traverse|add\s+access|"
    r"\d{1,3}\s*deg\s+\d|surveyor\s+name|plan\s+number|certification|"
    r"crs_?origin|origin_?crs|plot\s+using\s+scale|title\s+as|"
    r"\bscale\s*[:=]\s*1\s*[:/]|\bscale\s+1\s*[:/]|"
    r"local\s+(?:govt\.?|government)\s+area\s*[:=]|\bstate\s*[:=]",
    re.IGNORECASE,
)


def trim_metadata_field(text: str, *, max_len: int = 160) -> str:
    """Keep a single title-block line; drop traverse/prompt leakage."""
    s = re.sub(r"\s+", " ", (text or "").strip())
    if not s:
        return ""
    m = _METADATA_JUNK_RE.search(s)
    if m:
        s = s[: m.start()].strip(" ,;:-")
    if not s:
        return ""
    if len(s) > max_len:
        s = s[:max_len].rsplit(" ", 1)[0].strip()
    return s


def sanitize_metadata_field(text: str, *, max_len: int = 160) -> str:
    """Keep title-block fields short and free of traverse/coordinate dumps."""
    return trim_metadata_field(text, max_len=max_len)


def scrub_surveyor_metadata_value(text: str, *, max_len: int = 200) -> str:
    """
    Surveyor name/address must never retain plan-scale or other title-block fields.

    Guards against greedy captures that swallowed ``Plot using scale 1:250`` /
    ``scale: 1:250`` when those lines sat between surveyor address and pillars.
    """
    s = (text or "").strip()
    if not s:
        return ""
    # Cut at first leaked scale / next-field token even across newlines.
    s = re.split(
        r"(?i)(?:\n|,)\s*(?:Plot\s+using\s+scale|scale\s*[:=]\s*1\s*[:/]|scale\s+1\s*[:/]|"
        r"pillar\s+numbers|coordinates\s+for|plan\s*(?:no\.?|number)|"
        r"date\s+on\s+the|buyer\s*'?s?\s*name|Generate\b)",
        s,
        maxsplit=1,
    )[0].strip(" ,;:-")
    return sanitize_metadata_field(s, max_len=max_len)


_LGA_TRAILING_RE = re.compile(
    r"\s*[,\-]?\s*(?:"
    r"local\s+government\s+area|"
    r"local\s+govt\.?\s*(?:area)?|"
    r"l\.?\s*g\.?\s*a\.?|"
    r"lga"
    r")\s*$",
    re.IGNORECASE,
)
_LGA_AS_PRINTED_RE = re.compile(r"\s*\(?\s*as\s+printed\s*\)?\s*", re.IGNORECASE)
_LGA_PAREN_RE = re.compile(
    r"\(([^)]*(?:local|govt|government|l\.?\s*g\.?\s*a|lga)[^)]*)\)",
    re.IGNORECASE,
)


def normalize_lga_name(raw: str) -> str:
    """
    Return only the bare Local Government Area name.

    Template CAD already carries the ``LOCAL GOVERNMENT AREA`` label — callers
    must not pass through suffixes like ``LGA``, ``Local Govt. Area``, or
    ``as printed``.

    Examples:
      ``Iro LGA`` → ``Iro``
      ``Boki Local Govt. area`` → ``Boki``
      ``Obio/Akpor Local government area`` → ``Obio/Akpor``
      ``Khana L.G.A`` → ``Khana``
      ``EMUOHA LOCAL GOVT. AREA AS PRINTED`` → ``EMUOHA``
      ``ODUOHA (EMUOHA LOCAL GOVT. AREA AS PRINTED)`` → ``EMUOHA``
    """
    s = re.sub(r"\s+", " ", (raw or "").strip())
    if not s:
        return ""
    # Prefer parenthetical core when it looks like an LGA phrase.
    m_par = _LGA_PAREN_RE.search(s)
    if m_par:
        s = m_par.group(1).strip()
    s = _LGA_AS_PRINTED_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;:-()")
    # Strip trailing LGA labels (may appear more than once after dirty extracts).
    for _ in range(3):
        nxt = _LGA_TRAILING_RE.sub("", s).strip(" ,;:-")
        if nxt == s:
            break
        s = nxt
    return s.strip()


def ensure_surveyor_professional_title(name: str) -> str:
    """
    Ensure a Nigerian cadastral surveyor display name keeps a professional title.

    Bare names become ``SURV. <NAME>``. Existing ``SURV.`` / ``SURVEYOR`` titles
    are normalized to ``SURV.`` without inventing a different person.
    """
    s = re.sub(r"\s+", " ", (name or "").strip())
    if not s:
        return ""
    # Prefer ``surveyor`` before ``surv``; require ``surv.`` or a word-boundary ``surv``
    # so "Surveyor X" / "SURV. X" normalize cleanly and "Survive" is left alone.
    m = re.match(r"^surveyor\b\.?\s*", s, flags=re.IGNORECASE)
    if not m:
        m = re.match(r"^surv(?:\.|\b)\s*", s, flags=re.IGNORECASE)
    if m:
        rest = s[m.end() :].strip().lstrip(".").strip()
        return f"SURV. {rest}".strip() if rest else "SURV."
    return f"SURV. {s}"


_LGA_LINE_RE = re.compile(
    r"(?:"
    r"local\s+government\s+area|"
    r"local\s+govt\.?\s*(?:area)?|"
    r"l\.?\s*g\.?\s*a\.?|"
    r"\blga\b"
    r")",
    re.IGNORECASE,
)


def extract_location_from_text(text: str) -> str:
    """
    Extract location from Nigerian plan text (everything after AT until LGA).

    Collects all location clauses (e.g. site line + community line). Does not
    include the Local Government Area line — that is a separate title-block field.
    """
    raw = text or ""
    # Prefer structured AT … until LGA (accept LOCAL GOVT. AREA / L.G.A / LGA).
    m_block = re.search(
        r"\bAT\b\s*(.*?)(?=\n\s*[^\n]*(?:local\s+government\s+area|local\s+govt\.?\s*(?:area)?|"
        r"l\.?\s*g\.?\s*a\.?|\blga\b)|\Z)",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_block:
        block = m_block.group(1) or ""
        parts: List[str] = []
        for line in re.split(r"[\n\r]+", block):
            line = re.sub(r"\s+", " ", line).strip(" ,;:-")
            if not line or re.search(r"\.{3,}", line):
                continue
            if _LGA_LINE_RE.search(line) and not re.search(
                r"\b(?:in|at|near|along)\b", line, flags=re.IGNORECASE
            ):
                # Pure LGA line — stop (should already be excluded by lookahead).
                continue
            # Drop a trailing LGA phrase glued onto the last location line.
            line = _LGA_TRAILING_RE.sub("", line).strip(" ,;:-")
            if line:
                parts.append(line)
        if parts:
            loc = ", ".join(parts)
            if len(loc) <= 200:
                return loc.upper()

    patterns = [
        re.compile(
            r"\bAT\s+(.+?)(?:\s*,\s*LOCAL\s+GOV(?:ERNMENT|T\.?)\s*AREA|\s+LOCAL\s+GOVERNMENT\s+AREA|"
            r"\s+L\.?\s*G\.?\s*A\.?|\s+LGA\b|$)",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"\bLOCATION\s*[:=]\s*(.+?)(?:\n|,\s*LOCAL\s+GOVERNMENT|LGA\b|STATE\b|$)",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            rf"location\s*[:=]\s*(.+?){CADASTRAL_FIELD_BOUNDARY}",
            re.IGNORECASE | re.DOTALL,
        ),
    ]
    for pat in patterns:
        m = pat.search(raw)
        if m:
            loc = re.sub(r"\s+", " ", m.group(1)).strip(" ,;:-")
            loc = _LGA_TRAILING_RE.sub("", loc).strip(" ,;:-")
            if loc and len(loc) <= 200:
                return loc.upper()
    return ""


def extract_scale_denom_from_text(text: str) -> Optional[int]:
    """Prefer SCALE:- label; ignore secondary 'SCALE: to 1:xxxx' lines."""
    raw = text or ""
    m = re.search(r"SCALE\s*:?-?\s*1\s*:\s*(\d+)", raw, re.IGNORECASE)
    if not m:
        m = re.search(r"plot\s+using\s+scale\s+1\s*:\s*(\d+)", raw, re.IGNORECASE)
    if not m:
        for line in raw.splitlines():
            if re.search(r"\bSCALE\b\s*:.*\bto\b", line, re.IGNORECASE):
                continue
            m2 = re.search(r"1\s*:\s*(\d+)", line)
            if m2:
                m = m2
                break
    if not m:
        return None
    try:
        d = int(m.group(1))
        return d if d > 0 else None
    except Exception:
        return None


# Common Nigerian cadastral plotting scales SurvyAI will honour from prompts.
_USER_SCALE_ALLOWED_DENOMS = frozenset(
    {250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 25000}
)

_USER_SCALE_REQUEST_PATTERNS: tuple[str, ...] = (
    # "Plot using scale 1:250" / "plot at scale of 1:250"
    r"plot\s+(?:using\s+|at\s+)?(?:a\s+)?scale\s*(?:of\s*)?1\s*[:/]\s*(\d+)",
    # "scale: 1:250" / "scale = 1:250" / "scale 1:250" (comma- or line-delimited fields)
    r"(?:^|[\n,;])\s*scale\s*[:=]\s*1\s*[:/]\s*(\d+)",
    r"(?:^|[\n,;])\s*scale\s+1\s*[:/]\s*(\d+)",
    # Mid-sentence field: "... origin_crs: UTM Zone 32N, scale: 1:250, plan number: ..."
    r"\bscale\s*[:=]\s*1\s*[:/]\s*(\d+)",
    # "use scale 1:250" / "using a scale of 1:250" / "at a scale of 1:250"
    r"(?:use|using|with|at)\s+(?:a\s+)?scale\s*(?:of\s*)?1\s*[:/]\s*(\d+)",
    # "scale should (now) be 1:250" / "change the scale to 1:250"
    r"scale\s+should\s+(?:now\s+)?be\s+1\s*[:/]\s*(\d+)",
    r"(?:change|update|set)\s+(?:the\s+)?scale\s+to\s+1\s*[:/]\s*(\d+)",
    # Title-block style inside a user prompt: "SCALE:- 1:250"
    r"\bSCALE\s*:?-?\s*1\s*:\s*(\d+)",
)


def extract_user_requested_scale_denom(text: str) -> Optional[int]:
    """
    Extract an explicit user-requested plan scale denominator from free text.

    Recognises varied prompt styles (``scale: 1:250``, ``Plot using scale 1:250``,
    ``at a scale of 1:250``, etc.). Ignores secondary ``SCALE: to 1:xxxx`` lines.
    Returns only common survey denoms (250…25000).
    """
    raw = text or ""
    if not raw.strip():
        return None

    def _accept(raw_denom: str) -> Optional[int]:
        try:
            d = int(raw_denom)
        except Exception:
            return None
        if d in _USER_SCALE_ALLOWED_DENOMS:
            return d
        return None

    for pat in _USER_SCALE_REQUEST_PATTERNS:
        for m in re.finditer(pat, raw, flags=re.IGNORECASE | re.MULTILINE):
            # Skip secondary title-block labels like "SCALE: to 1:1000" on this line only.
            line_start = raw.rfind("\n", 0, m.start()) + 1
            line_end = raw.find("\n", m.end())
            line = raw[line_start : line_end if line_end != -1 else len(raw)]
            if re.search(r"\bscale\b\s*:\s*to\b", line, flags=re.IGNORECASE):
                continue
            got = _accept(m.group(1))
            if got is not None:
                return got
    return None


# Plan-number token: RV/018/2026/SP, AB-1234, etc. (not a bare year or single word).
_PLAN_NUMBER_TOKEN = r"([A-Za-z][A-Za-z0-9]*(?:[/\-][A-Za-z0-9]+){1,6})"

_USER_PLAN_NUMBER_START_PATTERNS: tuple[str, ...] = (
    # Highest priority: explicit starting / base plan number for a batch.
    rf"start(?:ing)?\s+from\s+plan\s*(?:no\.?|number|#)?\s*[:=]?\s*['\"]?{_PLAN_NUMBER_TOKEN}['\"]?",
    rf"start(?:ing)?\s+with\s+plan\s*(?:no\.?|number|#)?\s*[:=]?\s*['\"]?{_PLAN_NUMBER_TOKEN}['\"]?",
    rf"begin(?:ning)?\s+(?:from|with)\s+plan\s*(?:no\.?|number|#)?\s*[:=]?\s*['\"]?{_PLAN_NUMBER_TOKEN}['\"]?",
    rf"(?:base|initial|first)\s+plan\s*(?:no\.?|number|#)\s*[:=]?\s*['\"]?{_PLAN_NUMBER_TOKEN}['\"]?",
    rf"plan\s*(?:no\.?|number|#)\s+(?:to\s+)?start(?:s|ing)?\s+(?:from|at|with)\s*[:=]?\s*['\"]?{_PLAN_NUMBER_TOKEN}['\"]?",
    rf"start(?:ing)?\s+from\s*['\"]{_PLAN_NUMBER_TOKEN}['\"]",
)

_USER_PLAN_NUMBER_FIELD_PATTERNS: tuple[str, ...] = (
    rf"use\s+plan\s*(?:no\.?|number|#)\s*[:=]?\s*['\"]?{_PLAN_NUMBER_TOKEN}['\"]?",
    rf"using\s+plan\s*(?:no\.?|number|#)\s*[:=]?\s*['\"]?{_PLAN_NUMBER_TOKEN}['\"]?",
    rf"plan\s*(?:no\.?|number|#)\s*[:=]\s*['\"]?{_PLAN_NUMBER_TOKEN}['\"]?",
    # "… and plan number to RV/NEW/002" / "change the plan number to …"
    rf"plan\s*(?:no\.?|number|#)\s+to\s+['\"]?{_PLAN_NUMBER_TOKEN}['\"]?",
    rf"plan\s*(?:no\.?|number|#)\s+should\s+(?:now\s+)?be\s+['\"]?{_PLAN_NUMBER_TOKEN}['\"]?",
    rf"plan\s*(?:no\.?|number|#)\s+now\s+is\s+['\"]?{_PLAN_NUMBER_TOKEN}['\"]?",
    rf"(?:change|update|set)\s+(?:the\s+)?plan\s*(?:no\.?|number|#)\s+to\s+['\"]?{_PLAN_NUMBER_TOKEN}['\"]?",
)

_TAKE_PLAN_FROM_REFERENCE_RE = re.compile(
    r"take\s+(?:the\s+)?plan\s*(?:no\.?|number|#)\s+from\s+(?:the\s+)?"
    r"(?:existing|reference|current|same)\b",
    flags=re.IGNORECASE,
)


def extract_user_requested_plan_number(text: str) -> Optional[str]:
    """
    Extract an explicit user-requested plan number (or batch starting number) from free text.

    Precedence (semantic, not keyword-only):
    1. ``start from plan number 'RV/018/2026/SP'`` / ``starting with plan no …``
    2. Field-style ``plan number: …`` / ``use plan number …`` / override phrasing
    3. Never treat ``e.g. if plan A is '…'`` illustrations as the base when the user
       also said to take the plan number from an existing/reference CAD plan.

    Returns a normalized plan number string, or None when the prompt does not state one.
    """
    raw = text or ""
    if not raw.strip():
        return None

    def _from_patterns(patterns: tuple[str, ...]) -> Optional[str]:
        for pat in patterns:
            m = re.search(pat, raw, flags=re.IGNORECASE)
            if not m:
                continue
            token = (m.group(1) or "").strip().strip("'\"")
            # Skip obvious "e.g. if plan A is …" illustration hits for field patterns
            # when the match sits inside an e.g./for example clause.
            window_start = max(0, m.start() - 48)
            prefix = raw[window_start : m.start()]
            if re.search(r"\b(?:e\.g\.|eg\.|for\s+example)\b", prefix, flags=re.IGNORECASE):
                continue
            normalized = normalize_plan_number(token)
            if normalized:
                return normalized
        return None

    started = _from_patterns(_USER_PLAN_NUMBER_START_PATTERNS)
    if started:
        return started

    fielded = _from_patterns(_USER_PLAN_NUMBER_FIELD_PATTERNS)
    if fielded:
        return fielded

    # Illustrative "e.g. if plan A is 'RV/…'" only when the user did NOT redirect
    # plan number to the reference DWG.
    if not _TAKE_PLAN_FROM_REFERENCE_RE.search(raw):
        m_eg = re.search(
            rf"(?:e\.g\.|eg\.|for\s+example)\s+"
            rf"(?:if\s+)?plan\s+[A-Za-z]\s+is\s+['\"]?{_PLAN_NUMBER_TOKEN}['\"]?",
            raw,
            flags=re.IGNORECASE,
        )
        if m_eg:
            return normalize_plan_number(m_eg.group(1).strip().strip("'\"")) or None
    return None


def normalize_plan_number(raw: str) -> str:
    """Normalize Nigerian plan numbers (e.g. RV11242026012 -> RV/1124/2026/012)."""
    s = (raw or "").strip().upper()
    if not s:
        return ""
    if "/" in s and len(s) <= 32:
        return s
    compact = re.sub(r"[^A-Z0-9]", "", s)
    m = re.match(r"^(RV|RP|TR|TP)(\d{4})(\d{4})(\d{2,4})$", compact)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}/{m.group(4)}"
    return s[:32]


def _prefer_labelled_easting(raw: str, eastings: List[float]) -> Optional[float]:
    labelled: List[float] = []
    for e in eastings:
        stem = rf"{int(e)}(?:\.\d+)?"
        if re.search(rf"{stem}\s*m\.?\s*E\b", raw, re.IGNORECASE):
            labelled.append(e)
        elif re.search(rf"\bE\.?\s*m?\.?\s*{stem}\b", raw, re.IGNORECASE):
            labelled.append(e)
    if labelled:
        # When several labelled eastings appear (title block + plan grid), prefer the
        # plan-grid value — typically the larger labelled easting on Nigerian sheets.
        return max(labelled)
    return None


def _prefer_labelled_northing(raw: str, northings: List[float]) -> Optional[float]:
    labelled: List[float] = []
    for n in northings:
        stem = rf"{int(n)}(?:\.\d+)?"
        if re.search(rf"{stem}\s*m\.?\s*N\b", raw, re.IGNORECASE):
            labelled.append(n)
        elif re.search(rf"\bN\.?\s*m?\.?\s*{stem}\b", raw, re.IGNORECASE):
            labelled.append(n)
    if labelled:
        return max(labelled)
    return None


def _nearest_pillar_to_point(
    positions: Dict[str, tuple[float, float]],
    pt: tuple[float, float],
    *,
    max_dist: float = 140.0,
) -> Optional[str]:
    if not positions:
        return None
    best_pillar: Optional[str] = None
    best_d = 1e18
    for pillar, pos in positions.items():
        d = math.hypot(pos[0] - pt[0], pos[1] - pt[1])
        if d < best_d:
            best_d = d
            best_pillar = pillar
    if best_pillar is None or best_d > max_dist:
        return None
    return best_pillar


def _pillar_list_index(pillars: Sequence[str], pillar_ref: str) -> Optional[int]:
    if not pillar_ref or not pillars:
        return None
    ref = re.sub(r"\s+", " ", pillar_ref.strip().upper())
    for i, pillar in enumerate(pillars):
        pu = re.sub(r"\s+", " ", pillar.strip().upper())
        if pu == ref or pu.replace(" ", "") == ref.replace(" ", ""):
            return i
    nums = re.findall(r"\d+", ref)
    if not nums:
        return None
    num = nums[-1]
    candidates = [i for i, p in enumerate(pillars) if num in p]
    if len(candidates) == 1:
        return candidates[0]
    for i in candidates:
        prefix = re.sub(r"\s+", "", pillars[i].upper().split()[0]) if " " in pillars[i] else ""
        if prefix and prefix[:2] in ref.replace(" ", ""):
            return i
    return candidates[0] if len(candidates) == 1 else None


def _extract_grid_labels_from_pdf_page(
    page: Any,
) -> tuple[List[tuple[float, float, float]], List[tuple[float, float, float]]]:
    """
    Return labelled grid eastings/northings as (value, cx, cy) in PDF page space.
    """
    eastings: List[tuple[float, float, float]] = []
    northings: List[tuple[float, float, float]] = []
    seen_e: set[float] = set()
    seen_n: set[float] = set()

    def _add(axis: str, val: float, cx: float, cy: float) -> None:
        if axis == "E":
            if not _is_plausible_utm_easting(val) or val in seen_e:
                return
            seen_e.add(val)
            eastings.append((val, cx, cy))
        else:
            if not _is_plausible_utm_northing(val) or val in seen_n:
                return
            seen_n.add(val)
            northings.append((val, cx, cy))

    try:
        words = page.get_text("words") or []
    except Exception:
        words = []

    for w in words:
        tok = str(w[4] or "").strip()
        cx = (float(w[0]) + float(w[2])) / 2.0
        cy = (float(w[1]) + float(w[3])) / 2.0
        me = re.match(r"^(\d{5,7}(?:\.\d+)?)\s*m\.?\s*E\.?$", tok, re.IGNORECASE)
        if me:
            _add("E", float(me.group(1)), cx, cy)
            continue
        mn = re.match(r"^(\d{5,7}(?:\.\d+)?)\s*m\.?\s*N\.?$", tok, re.IGNORECASE)
        if mn:
            _add("N", float(mn.group(1)), cx, cy)

    try:
        plain = page.get_text() or ""
    except Exception:
        plain = ""

    for pat, axis in (
        (re.compile(r"(\d{5,7}(?:\.\d+)?)\s*m\.?\s*E\b", re.IGNORECASE), "E"),
        (re.compile(r"(\d{5,7}(?:\.\d+)?)\s*m\.?\s*N\b", re.IGNORECASE), "N"),
    ):
        for m in pat.finditer(plain):
            val = float(m.group(1))
            snippet = m.group(0)
            cx = cy = 0.0
            found_pos = False
            try:
                rects = page.search_for(snippet) or page.search_for(snippet.upper())
                if not rects:
                    rects = page.search_for(f"{val:.3f}")
            except Exception:
                rects = []
            if rects:
                r = rects[0]
                cx = (float(r.x0) + float(r.x1)) / 2.0
                cy = (float(r.y0) + float(r.y1)) / 2.0
                found_pos = True
            if found_pos:
                _add(axis, val, cx, cy)

    return eastings, northings


def _compute_relative_traverse_vertices(
    pillars: Sequence[str],
    legs: Sequence[SurveyTraverseLeg],
) -> Optional[List[Dict[str, float]]]:
    if len(pillars) < 3 or len(legs) < 3:
        return None
    pts: List[Dict[str, float]] = [{"e": 0.0, "n": 0.0}]
    ce, cn = 0.0, 0.0
    for leg in legs:
        bdeg = float(leg.bearing_deg) + float(leg.bearing_min) / 60.0
        br = math.radians(bdeg)
        de = float(leg.distance_m) * math.sin(br)
        dn = float(leg.distance_m) * math.cos(br)
        ce += de
        cn += dn
        pts.append({"e": float(ce), "n": float(cn)})
    if len(pts) == len(legs) + 1 and len(legs) >= 3:
        dx0 = pts[-1]["e"] - pts[0]["e"]
        dy0 = pts[-1]["n"] - pts[0]["n"]
        if math.hypot(dx0, dy0) <= 0.25:
            pts = pts[:-1]
    if len(pts) != len(pillars):
        return None
    return pts


def _is_plausible_pillar_id(pillar: str) -> bool:
    """
    Lightweight gate for PDF/layout pillar candidates.

    Accepts classic SC/SP (and RV/RP) labels with digit pegs up to 9 digits, and
    alphanumeric peg tokens such as AS3459RP / OA94567KL.
    """
    split = split_cadastral_pillar_label(pillar or "")
    if not split:
        # Fall back for already-normalized ids
        if not re.search(r"^(?:SC|SP|RV|RP)\s*/", pillar or "", re.IGNORECASE):
            return False
        nums = re.findall(r"\d+", pillar or "")
        if not nums:
            return False
        try:
            num = int(nums[-1])
        except Exception:
            return False
        return 100 <= num <= 999_999_999
    number = split["number"]
    nums = re.findall(r"\d+", number)
    if not nums:
        return False
    try:
        num = int(max(nums, key=len))
    except Exception:
        return False
    # 3–9 digit runs are common; reject tiny/noise values.
    if num < 100 or num > 999_999_999:
        return False
    if len(max(nums, key=len)) > 9:
        return False
    return True


def _filter_plausible_pillars(pillars: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for p in pillars:
        p = _normalize_pillar_id(p)
        if not _is_plausible_pillar_id(p) or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _is_plausible_boundary_leg(leg: SurveyTraverseLeg) -> bool:
    try:
        bd = int(leg.bearing_deg)
        bm = int(leg.bearing_min)
        dist = float(leg.distance_m)
    except Exception:
        return False
    if bd < 0 or bd > 360 or bm < 0 or bm >= 60:
        return False
    if bd == 360 and bm > 0:
        return False
    if dist < 0.5 or dist > 250.0:
        return False
    return True


def _filter_plausible_legs(legs: Sequence[SurveyTraverseLeg]) -> List[SurveyTraverseLeg]:
    return [lg for lg in legs if _is_plausible_boundary_leg(lg)]


def _match_grid_coordinate_block(
    e_labels: Sequence[tuple[float, float, float]],
    n_labels: Sequence[tuple[float, float, float]],
    positions: Dict[str, tuple[float, float]],
    pillars: Sequence[str],
    *,
    anchor_hint: str = "",
) -> tuple[Optional[float], Optional[float], str]:
    """
    When E and N labels sit together in the coordinate block, assign both to the same pillar.
    Prefer the LLM/anchor pillar hint when supplied.
    """
    if anchor_hint and _pillar_list_index(pillars, anchor_hint) is not None:
        best_e = best_n = None
        for ev, _ex, _ey in e_labels:
            best_e = ev
            break
        for nv, _nx, _ny in n_labels:
            best_n = nv
            break
        if best_e is not None and best_n is not None:
            return best_e, best_n, anchor_hint

    best_pair: Optional[tuple[float, float, float, float, float]] = None
    for ev, ex, ey in e_labels:
        for nv, nx, ny in n_labels:
            gap = math.hypot(ex - nx, ey - ny)
            if gap > 80.0:
                continue
            cx, cy = (ex + nx) / 2.0, (ey + ny) / 2.0
            pillar = _nearest_pillar_to_point(positions, (cx, cy), max_dist=220.0) or ""
            if not pillar:
                continue
            d = math.hypot(positions[pillar][0] - cx, positions[pillar][1] - cy)
            score = d + gap * 0.25
            if best_pair is None or score < best_pair[0]:
                best_pair = (score, ev, nv, cx, cy)
    if best_pair:
        _, ev, nv, cx, cy = best_pair
        pillar = _nearest_pillar_to_point(positions, (cx, cy), max_dist=220.0) or anchor_hint
        return ev, nv, pillar or anchor_hint
    return None, None, ""


def _compute_absolute_parcel_coordinates(
    extraction: SurveyPlanExtraction,
    *,
    grid_e: Optional[float],
    grid_e_pillar: str,
    grid_n: Optional[float],
    grid_n_pillar: str,
) -> Optional[List[Dict[str, float]]]:
    """Anchor a bearing/distance traverse to PDF grid labels on the correct pillars."""
    pillars = extraction.pillar_numbers
    legs = extraction.traverse_legs
    rel = _compute_relative_traverse_vertices(pillars, legs)
    if not rel:
        return None

    pts = [dict(p) for p in rel]

    # Full E+N at one pillar (common on Nigerian plans) — highest priority.
    anchor_p = (extraction.anchor_pillar or "").strip()
    anchor_e = extraction.anchor_easting
    anchor_n = extraction.anchor_northing
    if anchor_p and anchor_e is not None and anchor_n is not None:
        i_anchor = _pillar_list_index(pillars, anchor_p)
        if i_anchor is not None:
            de = float(anchor_e) - float(pts[i_anchor]["e"])
            dn = float(anchor_n) - float(pts[i_anchor]["n"])
            for p in pts:
                p["e"] = float(p["e"]) + de
                p["n"] = float(p["n"]) + dn
            return pts

    i_e = _pillar_list_index(pillars, grid_e_pillar) if grid_e_pillar else None
    i_n = _pillar_list_index(pillars, grid_n_pillar) if grid_n_pillar else None

    if i_e is not None and grid_e is not None:
        de = float(grid_e) - float(pts[i_e]["e"])
        dn = 0.0
        if i_n is not None and grid_n is not None and i_n == i_e:
            dn = float(grid_n) - float(pts[i_n]["n"])
        for p in pts:
            p["e"] = float(p["e"]) + de
            p["n"] = float(p["n"]) + dn
    elif i_n is not None and grid_n is not None:
        dn = float(grid_n) - float(pts[i_n]["n"])
        for p in pts:
            p["n"] = float(p["n"]) + dn

    return pts


def _pick_primary_pillar_index(coords: Sequence[Dict[str, float]]) -> int:
    if not coords:
        return 0
    es = [float(p.get("e", 0.0)) for p in coords]
    min_e = min(es)
    spread = float(max(es) - min_e) if len(es) > 1 else 0.0
    eps_e = max(0.01, spread * 1e-9)
    near_west = [i for i in range(len(coords)) if abs(float(coords[i].get("e", 0.0)) - min_e) <= eps_e]
    return int(min(near_west, key=lambda i: float(coords[i].get("n", 0.0))))


def apply_pdf_grid_coordinates(
    extraction: SurveyPlanExtraction,
    pdf_path: Optional[str],
    combined_text: str,
) -> SurveyPlanExtraction:
    """
    Match PDF grid E/N labels to pillars and compute absolute parcel coordinates.

    When E and N grid lines pass through different pillars, anchor the traverse using
    the labelled easting (preferred) so the western primary pillar shows the correct E.
    """
    pillars = extraction.pillar_numbers
    if len(pillars) < 3 or len(extraction.traverse_legs) < 3:
        return extraction

    grid_e: Optional[float] = None
    grid_n: Optional[float] = None
    grid_e_pillar = ""
    grid_n_pillar = ""
    positions: Dict[str, tuple[float, float]] = {}

    if pdf_path and Path(pdf_path).exists():
        try:
            import fitz

            doc = fitz.open(str(pdf_path))
            page = doc[0]
            positions = _pillar_label_positions(page, pillars)
            e_labels, n_labels = _extract_grid_labels_from_pdf_page(page)
            doc.close()

            block_e, block_n, block_pillar = _match_grid_coordinate_block(
                e_labels,
                n_labels,
                positions,
                pillars,
                anchor_hint=extraction.anchor_pillar or "",
            )
            if block_e is not None and block_n is not None and block_pillar:
                grid_e, grid_n = block_e, block_n
                grid_e_pillar = grid_n_pillar = block_pillar
            else:
                if e_labels:
                    scored_e = []
                    for val, cx, cy in e_labels:
                        pillar = _nearest_pillar_to_point(positions, (cx, cy))
                        if pillar:
                            d = abs(positions[pillar][0] - cx) + abs(positions[pillar][1] - cy) * 0.35
                        else:
                            d = 9999.0
                        scored_e.append((d, val, pillar or ""))
                    scored_e.sort(key=lambda t: t[0])
                    grid_e = scored_e[0][1]
                    grid_e_pillar = scored_e[0][2]
                if n_labels:
                    scored_n = []
                    for val, cx, cy in n_labels:
                        pillar = _nearest_pillar_to_point(positions, (cx, cy))
                        if pillar:
                            d = abs(positions[pillar][1] - cy) + abs(positions[pillar][0] - cx) * 0.35
                        else:
                            d = 9999.0
                        scored_n.append((d, val, pillar or ""))
                    scored_n.sort(key=lambda t: t[0])
                    grid_n = scored_n[0][1]
                    grid_n_pillar = scored_n[0][2]
        except Exception as exc:
            logger.debug("PDF grid label extraction failed: %s", exc)

    if grid_e is None or grid_n is None:
        best_e, best_n, all_e, all_n = extract_grid_coordinates_from_text(combined_text)
        if grid_e is None:
            grid_e = best_e
        if grid_n is None:
            grid_n = best_n

    if not grid_e_pillar and grid_e is not None:
        rel = _compute_relative_traverse_vertices(pillars, extraction.traverse_legs)
        if rel:
            grid_e_pillar = pillars[_pick_primary_pillar_index(rel)]
        elif positions:
            west_idx = None
            west_e = 1e18
            for i, pillar in enumerate(pillars):
                if pillar not in positions:
                    continue
                px = positions[pillar][0]
                if px < west_e:
                    west_e = px
                    west_idx = i
            if west_idx is not None:
                grid_e_pillar = pillars[west_idx]

    if not grid_n_pillar and grid_n is not None:
        rel = _compute_relative_traverse_vertices(pillars, extraction.traverse_legs)
        if rel:
            ns = [float(p["n"]) for p in rel]
            max_n = max(ns)
            spread = float(max(ns) - min(ns)) if len(ns) > 1 else 0.0
            eps_n = max(0.01, spread * 1e-9)
            north_idx = int(
                max(
                    [i for i in range(len(rel)) if abs(float(rel[i]["n"]) - max_n) <= eps_n],
                    key=lambda i: float(rel[i]["e"]),
                )
            )
            grid_n_pillar = pillars[north_idx]
        elif positions:
            north_idx = None
            north_y = 1e18
            for i, pillar in enumerate(pillars):
                if pillar not in positions:
                    continue
                py = positions[pillar][1]
                if py < north_y:
                    north_y = py
                    north_idx = i
            if north_idx is not None:
                grid_n_pillar = pillars[north_idx]

    abs_coords = _compute_absolute_parcel_coordinates(
        extraction,
        grid_e=grid_e,
        grid_e_pillar=grid_e_pillar,
        grid_n=grid_n,
        grid_n_pillar=grid_n_pillar,
    )
    if not abs_coords:
        if grid_e is not None:
            extraction.anchor_easting = grid_e
        if grid_n is not None:
            extraction.anchor_northing = grid_n
        return extraction

    if not _absolute_coords_are_plausible_utm(abs_coords):
        if grid_e is not None:
            extraction.anchor_easting = grid_e
        if grid_n is not None:
            extraction.anchor_northing = grid_n
        return extraction

    extraction.grid_easting_pillar = grid_e_pillar or extraction.grid_easting_pillar
    extraction.grid_northing_pillar = grid_n_pillar or extraction.grid_northing_pillar
    extraction.absolute_parcel_coords = abs_coords

    primary_idx = _pick_primary_pillar_index(abs_coords)
    primary = abs_coords[primary_idx]
    extraction.anchor_easting = float(primary["e"])
    extraction.anchor_northing = float(primary["n"])
    extraction.anchor_pillar = pillars[primary_idx] if primary_idx < len(pillars) else extraction.anchor_pillar

    return extraction


def is_explicit_fence_label(text: str) -> Optional[tuple[str, str]]:
    """
    Return (kind, title) only when text contains an explicit CWF/DCWF-style label.

    Parallel offset lines alone or road/railway symbology must NOT match without a fence label.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    compact = re.sub(r"\s+", "", raw.upper())

    if re.match(r"^D\.?C\.?W\.?F\.?$", compact) or compact == "DCWF":
        return "DCWF", "Dwarf Concrete wall fence"
    if re.search(r"dwarf\s+concrete\s+wall\s+fence|short\s+wall\s+fence|dwarf\s+wall\s+fence", raw, re.I):
        return "DCWF", "Dwarf Concrete wall fence"

    if re.match(r"^C\.?W\.?F\.?$", compact) or compact == "CWF":
        return "CWF", "Concrete wall fence"
    if re.search(r"concrete\s+wall\s+fence", raw, re.I):
        return "CWF", "Concrete wall fence"
    if re.search(r"\bwall\s+fence\b", raw, re.I):
        return "CWF", "Concrete wall fence"
    if compact == "WF":
        return "CWF", "Concrete wall fence"
    if compact == "FENCE" or re.match(r"^FENCE\.?$", raw, re.I):
        return "CWF", "Concrete wall fence"

    return None


def query_has_explicit_fence_label(text: str) -> bool:
    """True when any word/phrase in text is an explicit CWF/DCWF label."""
    if not text:
        return False
    if is_explicit_fence_label(text):
        return True
    for m in re.finditer(r"\S+", text):
        if is_explicit_fence_label(m.group(0)):
            return True
    for m in re.finditer(
        r"dwarf\s+concrete\s+wall\s+fence|concrete\s+wall\s+fence|\bwall\s+fence\b|\bfence\b",
        text,
        re.IGNORECASE,
    ):
        if is_explicit_fence_label(m.group(0)):
            return True
    return False


def _fence_kind_from_token(token: str) -> Optional[tuple[str, str]]:
    return is_explicit_fence_label(token)


def infer_fences_from_pdf(
    extraction: SurveyPlanExtraction,
    *,
    pdf_path: Optional[str] = None,
    combined_text: str = "",
) -> List[str]:
    """Detect fences ONLY from explicit C.W.F. / D.C.W.F. (or equivalent) text on the PDF."""
    pillars = extraction.pillar_numbers
    if len(pillars) < 2:
        return []

    found: List[str] = []
    seen_edges: set[tuple[str, str]] = set()
    positions: Dict[str, tuple[float, float]] = {}

    if pdf_path and Path(pdf_path).exists():
        try:
            import fitz

            doc = fitz.open(str(pdf_path))
            page = doc[0]
            positions = _pillar_label_positions(page, pillars)

            words = page.get_text("words") or []
            for i, w in enumerate(words):
                tok = str(w[4] or "").strip()
                kind = _fence_kind_from_token(tok)
                cx = (float(w[0]) + float(w[2])) / 2.0
                cy = (float(w[1]) + float(w[3])) / 2.0
                if not kind and i + 1 < len(words):
                    nxt = str(words[i + 1][4] or "").strip()
                    phrase = f"{tok} {nxt}"
                    kind = is_explicit_fence_label(phrase)
                    if kind:
                        w2 = words[i + 1]
                        cx = (float(w[0]) + float(w2[2])) / 2.0
                        cy = (float(w[1]) + float(w2[3])) / 2.0
                if not kind:
                    continue
                edge = _nearest_edge_to_label(positions, pillars, (cx, cy))
                if not edge:
                    continue
                a, b = edge
                key = tuple(sorted((a, b)))
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                _, title = kind
                found.append(f"Add {title} on the sides joining {a} and {b}")

            for probe in (
                "D.C.W.F.",
                "D.C.W.F",
                "DCWF",
                "C.W.F.",
                "C.W.F",
                "CWF",
                "WF",
                "WALL FENCE",
                "Wall Fence",
                "FENCE",
                "Fence",
                "CONCRETE WALL FENCE",
            ):
                if not is_explicit_fence_label(probe):
                    continue
                try:
                    rects = page.search_for(probe) or page.search_for(probe.upper())
                except Exception:
                    rects = []
                kind = _fence_kind_from_token(probe)
                if not kind:
                    continue
                for r in rects or []:
                    cx = (float(r.x0) + float(r.x1)) / 2.0
                    cy = (float(r.y0) + float(r.y1)) / 2.0
                    edge = _nearest_edge_to_label(positions, pillars, (cx, cy))
                    if not edge:
                        continue
                    a, b = edge
                    key = tuple(sorted((a, b)))
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    _, title = kind
                    found.append(f"Add {title} on the sides joining {a} and {b}")

            doc.close()
        except Exception as exc:
            logger.debug("PDF fence extraction failed: %s", exc)

    for spec in infer_fences_from_boundary_text(combined_text):
        m = re.search(
            r"joining\s+(SC/[A-Z]{1,3}\s*\d{3,5})\s+and\s+(SC/[A-Z]{1,3}\s*\d{3,5})",
            spec,
            re.IGNORECASE,
        )
        if not m:
            continue
        key = tuple(sorted((_normalize_pillar_id(m.group(1)), _normalize_pillar_id(m.group(2)))))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        found.append(spec)

    return found


def _fence_spec_for_pillar_pair(a: str, b: str, *, kind: str = "CWF") -> str:
    title = "Dwarf Concrete Wall Fence" if kind.upper() == "DCWF" else "Concrete wall fence"
    pa, pb = _normalize_pillar_id(a), _normalize_pillar_id(b)
    return f"Add {title} on the sides joining {pa} and {pb}"


def infer_fences_from_boundary_text(text: str) -> List[str]:
    """
    Build fence plot specs from explicit C.W.F./DCWF labels plus pillar boundary pairs in text.

    Handles notes like: "C.W.F. along boundaries SC/CJ 2140-SC/CJ 2141 and SC/CJ 2142-SC/CJ 2143".
    """
    source = (text or "").strip()
    if not source or not query_has_explicit_fence_label(source):
        return []

    found: List[str] = []
    seen: set[str] = set()
    default_kind = "DCWF" if re.search(r"d\.?\s*c\.?\s*w\.?\s*f|\bdcwf\b", source, re.I) else "CWF"

    dash_pat = r"(SC/[A-Z]{1,3}\s*\d{3,5})\s*[-–]\s*(SC/[A-Z]{1,3}\s*\d{3,5})"
    for chunk in re.split(r"\s+and\s+", source, flags=re.IGNORECASE):
        for m in re.finditer(dash_pat, chunk, flags=re.IGNORECASE):
            spec = _fence_spec_for_pillar_pair(m.group(1), m.group(2), kind=default_kind)
            key = spec.lower()
            if key not in seen:
                seen.add(key)
                found.append(spec)

    if found:
        return found

    fallback_patterns = (
        r"between\s+(SC/[A-Z]{1,3}\s*\d{3,5})\s+and\s+(SC/[A-Z]{1,3}\s*\d{3,5})",
        r"joining\s+(SC/[A-Z]{1,3}\s*\d{3,5})\s+and\s+(SC/[A-Z]{1,3}\s*\d{3,5})",
    )
    for pat in fallback_patterns:
        for m in re.finditer(pat, source, flags=re.IGNORECASE):
            spec = _fence_spec_for_pillar_pair(m.group(1), m.group(2), kind=default_kind)
            key = spec.lower()
            if key not in seen:
                seen.add(key)
                found.append(spec)

    return found


def merge_fence_spec_strings(specs: Sequence[str]) -> List[str]:
    """Union fence spec strings, preserving order."""
    merged: List[str] = []
    seen: set[str] = set()
    for spec in specs or []:
        s = (spec or "").strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(s)
    return merged


def parse_fence_specs_from_text(text: str) -> List[Dict[str, str]]:
    """Parse fence instructions from a cadastral sub-prompt or user query."""
    fences: List[Dict[str, str]] = []
    seen: set[str] = set()
    for seg in re.split(
        r"\n+|(?<=[.;])\s+|;\s+(?=Add\s+)|\s+and\s+add\s+|\s+also\s+add\s+",
        text or "",
        flags=re.IGNORECASE,
    ):
        seg = (seg or "").strip().strip(";")
        if not seg:
            continue
        seg_l = seg.lower()
        if not re.search(
            r"c\.w\.f|d\.c\.w\.f|\bdcwf\b|\bcwf\b|\bwf\b|"
            r"concrete\s+wall\s+fence|\bwall\s+fence\b|\bfence\b|"
            r"dwarf\s+concrete\s+wall\s+fence",
            seg_l,
            re.IGNORECASE,
        ):
            continue
        if not re.search(r"side\s+of|joining|connecting|along|between|linking|sides", seg_l, re.IGNORECASE):
            continue
        kind = "DCWF" if re.search(
            r"d\.c\.w\.f|\bdwarf\s+concrete\s+wall\s+fence|\bdcwf\b",
            seg_l,
            re.IGNORECASE,
        ) else "CWF"
        for expanded in _expand_fence_segment_specs(seg, kind=kind):
            key = expanded["spec"].lower()[:200]
            if key in seen:
                continue
            seen.add(key)
            fences.append(expanded)
    return fences


def _expand_fence_segment_specs(seg: str, *, kind: str) -> List[Dict[str, str]]:
    """Expand chained pillar fences (A to B to C) into per-leg plot specs."""
    seg_l = seg.lower()
    label = "Dwarf Concrete wall fence" if kind == "DCWF" else "Concrete wall fence"
    m = re.search(
        r"(?:joining|connecting|along|between|side\s+of|sides)\s+(.+)$",
        seg,
        re.IGNORECASE,
    )
    if not m:
        return [{"kind": kind, "spec": seg}]
    chain_text = (m.group(1) or "").strip()
    if not chain_text:
        return [{"kind": kind, "spec": seg}]
    pillars = re.findall(r"(?:SC|SP)/[A-Z]{1,3}\s*\d{3,5}", chain_text, re.IGNORECASE)
    if len(pillars) < 3 or not re.search(r"\s+to\s+", chain_text, re.IGNORECASE):
        return [{"kind": kind, "spec": seg}]
    out: List[Dict[str, str]] = []
    for i in range(len(pillars) - 1):
        pa = _normalize_pillar_id(pillars[i])
        pb = _normalize_pillar_id(pillars[i + 1])
        out.append(
            {
                "kind": kind,
                "spec": f"Add {label} on the sides joining {pa} and {pb}",
            }
        )
    return out or [{"kind": kind, "spec": seg}]


def filter_user_facing_extraction_notes(notes: str, override_fields: Sequence[str]) -> str:
    """Drop LLM extraction caveats that contradict applied user overrides."""
    raw = (notes or "").strip()
    if not raw:
        return ""
    if not override_fields:
        return raw
    parts = re.split(r"(?<=[.!])\s+", raw)
    kept: List[str] = []
    for part in parts:
        pl = part.lower()
        if any(
            phrase in pl
            for phrase in (
                "user-requested",
                "were ignored",
                "was ignored",
                "changes were ignored",
                "not applied",
                "cannot be executed",
                "extracted original visible",
                "downstream",
            )
        ):
            continue
        kept.append(part.strip())
    return " ".join(p for p in kept if p).strip()


def extract_grid_coordinates_from_text(
    text: str,
) -> tuple[Optional[float], Optional[float], List[float], List[float]]:
    """
    Extract UTM-style easting/northing from survey plan PDF text.

    Labelled values (`292132.704m.E`, `537097.737m.N`) take priority over blind scans.
    """
    raw = _collapse_spaced_coordinate_text(text or "")
    eastings: List[float] = []
    northings: List[float] = []

    def _add_e(val: float, *, labelled: bool = False) -> None:
        if not _is_plausible_utm_easting(val):
            return
        if val not in eastings:
            eastings.append(val)

    def _add_n(val: float, *, labelled: bool = False) -> None:
        if not _is_plausible_utm_northing(val):
            return
        if val not in northings:
            northings.append(val)

    pair_patterns = [
        re.compile(
            r"(\d{5,7}(?:\.\d+)?)\s*m?\s*E\s*[,;]?\s*(\d{5,7}(?:\.\d+)?)\s*m?\s*N",
            re.IGNORECASE,
        ),
    ]
    for pat in pair_patterns:
        for m in pat.finditer(raw):
            _add_e(float(m.group(1)), labelled=True)
            _add_n(float(m.group(2)), labelled=True)

    e_label_patterns = [
        re.compile(r"(\d{5,7}(?:\.\d+)?)\s*m\.?\s*E\b", re.IGNORECASE),
    ]
    n_label_patterns = [
        re.compile(r"(\d{5,7}(?:\.\d+)?)\s*m\.?\s*N\b", re.IGNORECASE),
    ]
    for pat in e_label_patterns:
        for m in pat.finditer(raw):
            _add_e(float(m.group(1)), labelled=True)
    for pat in n_label_patterns:
        for m in pat.finditer(raw):
            _add_n(float(m.group(1)), labelled=True)

    # Blind scan only when an axis is still missing; reject common OCR fragments.
    if not eastings or not northings:
        for m in re.finditer(r"(\d{5,7}(?:\.\d+)?)", raw):
            val = float(m.group(1))
            has_decimal = "." in m.group(1)
            if not eastings and 200000.0 <= val <= 400000.0:
                if has_decimal or val >= 280000.0:
                    _add_e(val)
            elif not northings and val >= 450000.0:
                _add_n(val)

    best_e = _prefer_labelled_easting(raw, eastings)
    if best_e is None:
        best_e = eastings[0] if len(eastings) == 1 else (max(eastings) if eastings else None)
    best_n = _prefer_labelled_northing(raw, northings)
    if best_n is None:
        best_n = northings[0] if len(northings) == 1 else (max(northings) if northings else None)

    # Typical Nigerian plans: northing > easting when both present.
    if best_e and best_n and best_e > best_n:
        best_e, best_n = best_n, best_e
        eastings, northings = northings, eastings

    return best_e, best_n, eastings, northings


def finalize_survey_extraction(
    extraction: SurveyPlanExtraction,
    layout_text: str,
    *,
    plain_text: str = "",
    pdf_path: Optional[str] = None,
    llm: Any = None,
    run_with_timeout: Optional[Callable[..., Any]] = None,
    pre_rendered_images: Optional[List[str]] = None,
) -> SurveyPlanExtraction:
    """Fill or correct anchor coordinates from all available PDF text sources."""
    combined = f"{layout_text}\n{plain_text}"
    extraction = repair_survey_extraction_from_pdf(extraction, pdf_path, combined)

    if pdf_path:
        apply_pdf_grid_coordinates(extraction, pdf_path, combined)
        if not extraction.absolute_parcel_coords:
            best_e, best_n, all_e, all_n = extract_grid_coordinates_from_text(combined)
            abs_try = _compute_absolute_parcel_coordinates(
                extraction,
                grid_e=best_e,
                grid_e_pillar=extraction.grid_easting_pillar,
                grid_n=best_n,
                grid_n_pillar=extraction.grid_northing_pillar,
            )
            if abs_try and _absolute_coords_are_plausible_utm(abs_try):
                extraction.absolute_parcel_coords = abs_try
                primary_idx = _pick_primary_pillar_index(abs_try)
                extraction.anchor_easting = float(abs_try[primary_idx]["e"])
                extraction.anchor_northing = float(abs_try[primary_idx]["n"])
                extraction.anchor_pillar = extraction.pillar_numbers[primary_idx]
            else:
                if extraction.anchor_easting is None:
                    extraction.anchor_easting = best_e
                if extraction.anchor_northing is None:
                    extraction.anchor_northing = best_n
    else:
        best_e, best_n, all_e, all_n = extract_grid_coordinates_from_text(combined)
        e = extraction.anchor_easting
        n = extraction.anchor_northing
        if e is None or not _is_plausible_utm_easting(e):
            e = best_e
        if n is None or not _is_plausible_utm_northing(n):
            n = best_n
        if all_e and (e is None or e not in all_e):
            e = best_e if best_e is not None else e
        if all_n and (n is None or n not in all_n):
            n = best_n if best_n is not None else n
        extraction.anchor_easting = e
        extraction.anchor_northing = n

    if not extraction.anchor_pillar and extraction.pillar_numbers:
        extraction.anchor_pillar = extraction.pillar_numbers[0]

    e = extraction.anchor_easting
    n = extraction.anchor_northing
    if e and n and len(extraction.traverse_legs) >= 3:
        extraction.confidence = max(float(extraction.confidence or 0), 0.65)

    extraction = enrich_extraction_coordinates(extraction, combined)

    return prepare_extraction_for_cadastral(
        extraction,
        combined,
        pdf_path=pdf_path,
        llm=llm,
        run_with_timeout=run_with_timeout,
        pre_rendered_images=pre_rendered_images,
    )


def _pdf_page_line_segments(page: Any) -> List[tuple[float, float, float, float, float]]:
    """Return line segments as (x1, y1, x2, y2, length_pt) from a PyMuPDF page."""
    segs: List[tuple[float, float, float, float, float]] = []
    try:
        for d in page.get_drawings():
            for item in d.get("items") or []:
                if not item or item[0] != "l":
                    continue
                p1, p2 = item[1], item[2]
                x1, y1 = float(p1.x), float(p1.y)
                x2, y2 = float(p2.x), float(p2.y)
                ln = math.hypot(x2 - x1, y2 - y1)
                if ln >= 3.0:
                    segs.append((x1, y1, x2, y2, ln))
    except Exception:
        pass
    return segs


def _pdf_distance_labels(page: Any) -> List[tuple[float, float, float]]:
    """Words that look like metric distances (e.g. 9.60m) with page positions."""
    out: List[tuple[float, float, float]] = []
    try:
        for w in page.get_text("words") or []:
            tok = str(w[4] or "").strip()
            m = re.match(r"^(\d+(?:\.\d+)?)\s*m\.?$", tok, re.IGNORECASE)
            if m:
                cx = (float(w[0]) + float(w[2])) / 2.0
                cy = (float(w[1]) + float(w[3])) / 2.0
                out.append((cx, cy, float(m.group(1))))
    except Exception:
        pass
    return out


def _pdf_scale_bar_m_per_pt(page: Any, scale_denom: Optional[int]) -> Optional[float]:
    """Estimate metres/pt from scale-bar tick labels (e.g. 0, 5, 10, 15 m)."""
    try:
        ticks: List[tuple[float, float]] = []
        for w in page.get_text("words") or []:
            tok = str(w[4] or "").strip()
            if re.match(r"^\d+(?:\.\d+)?$", tok):
                val = float(tok)
                if 0.0 <= val <= 30.0:
                    cx = (float(w[0]) + float(w[2])) / 2.0
                    cy = (float(w[1]) + float(w[3])) / 2.0
                    ticks.append((cx, val))
        ticks.sort(key=lambda t: t[0])
        for i in range(len(ticks) - 1):
            x0, v0 = ticks[i]
            x1, v1 = ticks[i + 1]
            dv = v1 - v0
            dx = abs(x1 - x0)
            if dv >= 2.0 and dx >= 8.0:
                return dv / dx
    except Exception:
        pass
    if scale_denom and int(scale_denom) > 0:
        return (25.4 / 72.0) * (float(scale_denom) / 1000.0)
    return None


def calibrate_pdf_meters_per_point(
    page: Any,
    extraction: SurveyPlanExtraction,
    *,
    scale_denom: Optional[int] = None,
) -> Optional[float]:
    """
    Ground metres per PDF point by matching labelled traverse distances to drawn segments.
    Falls back to scale-bar ticks, then nominal map scale.
    """
    segs = _pdf_page_line_segments(page)
    labels = _pdf_distance_labels(page)
    ratios: List[float] = []

    if labels and segs:
        for cx, cy, dm in labels:
            best = min(
                segs,
                key=lambda s: math.hypot(((s[0] + s[2]) / 2.0) - cx, ((s[1] + s[3]) / 2.0) - cy),
            )
            mx = (best[0] + best[2]) / 2.0
            my = (best[1] + best[3]) / 2.0
            if math.hypot(mx - cx, my - cy) > 90.0:
                continue
            pt_len = best[4]
            if pt_len >= 8.0 and 0.5 <= dm <= 500.0:
                ratios.append(dm / pt_len)

    if ratios:
        ratios.sort()
        return ratios[len(ratios) // 2]

    leg_ms = sorted({float(leg.distance_m) for leg in extraction.traverse_legs if leg.distance_m > 0})
    if leg_ms and segs:
        seg_lens = sorted(s[4] for s in segs if s[4] >= 12.0)
        if seg_lens:
            n = min(len(leg_ms), len(seg_lens))
            for i in range(n):
                ratios.append(leg_ms[i] / seg_lens[-(n - i)])
            if ratios:
                ratios.sort()
                return ratios[len(ratios) // 2]

    return _pdf_scale_bar_m_per_pt(page, scale_denom)


def _seg_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.atan2(y2 - y1, x2 - x1)


def _point_line_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    ln = math.hypot(dx, dy)
    if ln < 1e-9:
        return math.hypot(px - x1, py - y1)
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / ln


def _segments_parallel(a: tuple, b: tuple, *, angle_tol: float = 0.12) -> bool:
    ax1, ay1, ax2, ay2, _ = a
    bx1, by1, bx2, by2, _ = b
    da = _seg_angle(ax1, ay1, ax2, ay2)
    db = _seg_angle(bx1, by1, bx2, by2)
    diff = abs(da - db)
    diff = min(diff, abs(diff - math.pi))
    return diff <= angle_tol


def normalize_access_road_title(raw: str) -> str:
    """Normalize road labels to CAD title-block style."""
    s = re.sub(r"\s+", " ", (raw or "").strip().upper())
    if not s:
        return "ACCESS    ROAD"
    compact = s.replace(" ", "")
    if compact in {"ACCESSCLOSE", "ACCESS/CLOSE"} or s == "ACCESS CLOSE":
        return "ACCESS CLOSE"
    if "ACCESS" in s and "ROAD" in s:
        return "ACCESS    ROAD"
    if s == "CLOSE":
        return "ACCESS CLOSE"
    return s


def extract_access_road_title_from_text(text: str) -> str:
    """Read access-road label from PDF/layout text (not geometry)."""
    raw = text or ""
    for pat, title in (
        (r"ACCESS\s+ROAD", "ACCESS    ROAD"),
        (r"ACCESS\s{2,}ROAD", "ACCESS    ROAD"),
        (r"ACCESS\s*/\s*CLOSE", "ACCESS CLOSE"),
        (r"ACCESS\s+CLOSE", "ACCESS CLOSE"),
    ):
        if re.search(pat, raw, re.IGNORECASE):
            return title
    return ""


def _pillar_label_positions(page: Any, pillars: Sequence[str]) -> Dict[str, tuple[float, float]]:
    """Map pillar id → page centre from PDF text search / word boxes."""
    positions: Dict[str, tuple[float, float]] = {}
    if not pillars:
        return positions

    for pillar in pillars:
        variants = {
            pillar,
            re.sub(r"\s+", " ", pillar.strip()),
            pillar.replace(" ", ""),
        }
        for variant in variants:
            if not variant:
                continue
            try:
                rects = page.search_for(variant) or page.search_for(variant.upper())
            except Exception:
                rects = []
            if rects:
                r = rects[0]
                positions[pillar] = ((float(r.x0) + float(r.x1)) / 2.0, (float(r.y0) + float(r.y1)) / 2.0)
                break

    missing = [p for p in pillars if p not in positions]
    if missing:
        try:
            words = page.get_text("words") or []
            i = 0
            while i < len(words):
                w = words[i]
                tok = str(w[4] or "").strip()
                nxt = str(words[i + 1][4] or "").strip() if i + 1 < len(words) else ""
                pid = _parse_pillar_token(tok, nxt)
                if pid and pid in missing:
                    cx = (float(w[0]) + float(w[2])) / 2.0
                    cy = (float(w[1]) + float(w[3])) / 2.0
                    if _PILLAR_PREFIX_TOKEN_RE.match(tok) and _PILLAR_NUMBER_TOKEN_RE.match(nxt):
                        w2 = words[i + 1]
                        cx = (cx + (float(w2[0]) + float(w2[2])) / 2.0) / 2.0
                        cy = (cy + (float(w2[1]) + float(w2[3])) / 2.0) / 2.0
                        i += 1
                    positions[pid] = (cx, cy)
                    missing = [p for p in pillars if p not in positions]
                i += 1
        except Exception:
            pass

    if missing:
        try:
            words = page.get_text("words") or []
            for pillar in missing:
                nums = re.findall(r"\d+", pillar)
                if not nums:
                    continue
                num_s = nums[-1]
                prefix = re.sub(r"\s+", "", pillar.upper().split()[0]) if " " in pillar else ""
                for w in words:
                    tok = str(w[4] or "").strip().upper().replace(" ", "")
                    if num_s not in tok:
                        continue
                    if prefix and prefix[:2] not in tok and not any(
                        part in tok for part in re.split(r"/", prefix) if len(part) >= 2
                    ):
                        continue
                    cx = (float(w[0]) + float(w[2])) / 2.0
                    cy = (float(w[1]) + float(w[3])) / 2.0
                    positions[pillar] = (cx, cy)
                    break
        except Exception:
            pass
    return positions


def _extract_all_access_road_labels_from_pdf(page: Any) -> List[tuple[str, tuple[float, float]]]:
    """
    Find every printed access-road / ACCESS CLOSE label and its centre on the plan page.

    Distinguishes ACCESS ROAD from ACCESS CLOSE / ACCESS/CLOSE — never treat a lone
    'ACCESS' token as CLOSE (that caused ACCESS ROAD plans to be mis-labelled).
    """
    found: List[tuple[str, tuple[float, float]]] = []
    clusters: List[tuple[float, float]] = []

    def _is_new_cluster(pt: tuple[float, float], *, tol: float = 45.0) -> bool:
        for cx, cy in clusters:
            if math.hypot(pt[0] - cx, pt[1] - cy) <= tol:
                return False
        clusters.append(pt)
        return True

    phrase_map = (
        ("ACCESS    ROAD", "ACCESS    ROAD"),
        ("ACCESS  ROAD", "ACCESS    ROAD"),
        ("ACCESS ROAD", "ACCESS    ROAD"),
        ("ACCESS CLOSE", "ACCESS CLOSE"),
        ("ACCESS/CLOSE", "ACCESS CLOSE"),
    )
    for phrase, title in phrase_map:
        try:
            rects = page.search_for(phrase) or []
        except Exception:
            rects = []
        for r in rects:
            pt = (
                (float(r.x0) + float(r.x1)) / 2.0,
                (float(r.y0) + float(r.y1)) / 2.0,
            )
            if _is_new_cluster(pt):
                found.append((title, pt))

    # Vertical / split-word labels: ACCESS above ROAD (or CLOSE) on the same column
    try:
        words = page.get_text("words") or []
        access_boxes: List[tuple[float, float, float, float]] = []
        road_boxes: List[tuple[float, float, float, float]] = []
        close_boxes: List[tuple[float, float, float, float]] = []
        for w in words:
            tok = re.sub(r"\s+", "", str(w[4] or "")).upper()
            box = (float(w[0]), float(w[1]), float(w[2]), float(w[3]))
            if tok == "ACCESS":
                access_boxes.append(box)
            elif tok == "ROAD":
                road_boxes.append(box)
            elif tok in ("CLOSE", "ACCESS/CLOSE", "ACCESSCLOSE"):
                close_boxes.append(box)

        def _pair_center(a: tuple, b: tuple) -> tuple[float, float]:
            return ((a[0] + a[2] + b[0] + b[2]) / 4.0, (a[1] + a[3] + b[1] + b[3]) / 4.0)

        for ab in access_boxes:
            for rb in road_boxes:
                ax = (ab[0] + ab[2]) / 2.0
                rx = (rb[0] + rb[2]) / 2.0
                if abs(ax - rx) <= 18.0 and abs(ab[1] - rb[1]) <= 80.0:
                    pt = _pair_center(ab, rb)
                    if _is_new_cluster(pt):
                        found.append(("ACCESS    ROAD", pt))
            for cb in close_boxes:
                ax = (ab[0] + ab[2]) / 2.0
                cx = (cb[0] + cb[2]) / 2.0
                if abs(ax - cx) <= 18.0 and abs(ab[1] - cb[1]) <= 80.0:
                    pt = _pair_center(ab, cb)
                    if _is_new_cluster(pt):
                        found.append(("ACCESS CLOSE", pt))
    except Exception:
        pass
    return found


def _extract_access_road_label_from_pdf(page: Any) -> Optional[tuple[str, tuple[float, float]]]:
    """Return the first access-road label on the page (legacy single-road helper)."""
    labels = _extract_all_access_road_labels_from_pdf(page)
    return labels[0] if labels else None


def _detect_close_label_in_pdf(page: Any) -> Optional[tuple[float, float]]:
    """Legacy helper — returns label centre only for ACCESS/CLOSE-style roads."""
    found = _extract_access_road_label_from_pdf(page)
    if not found:
        return None
    title, pt = found
    if title == "ACCESS CLOSE":
        return pt
    return None


def _resolve_two_pillar_refs(
    ref_a: str,
    ref_b: str,
    pillars: Sequence[str],
) -> Optional[tuple[str, str]]:
    """Map free-text pillar references to known pillar labels."""

    def _match_one(ref: str) -> Optional[str]:
        r = re.sub(r"\s+", " ", (ref or "").strip().upper())
        if not r:
            return None
        for pillar in pillars:
            pu = pillar.upper()
            if pu == r or pu.replace(" ", "") == r.replace(" ", ""):
                return pillar
        nums = re.findall(r"\d+", r)
        if not nums:
            return None
        num = nums[-1]
        candidates = [p for p in pillars if num in p]
        if len(candidates) == 1:
            return candidates[0]
        for pillar in candidates:
            prefix = re.sub(r"\s+", "", pillar.upper().split()[0]) if " " in pillar else ""
            if prefix and prefix[:2] in r.replace(" ", ""):
                return pillar
        return candidates[0] if len(candidates) == 1 else None

    a = _match_one(ref_a)
    b = _match_one(ref_b)
    if a and b and a != b:
        return a, b
    return None


def _parse_pillar_pair_from_road_specs(
    specs: Sequence[str],
    pillars: Sequence[str],
) -> Optional[tuple[str, str]]:
    """Parse pillar pair from LLM/CAD road spec strings."""
    side_re = re.compile(
        r"(?:on\s+(?:the\s+)?side\s+of|between|joining|connecting|along)\s+(.+?)\s+and\s+(.+?)"
        r"(?:\s*[;,.]|\s+with\s+|\s+offset|\s+width|\s*$)",
        re.IGNORECASE,
    )
    for spec in specs or []:
        m = side_re.search(spec or "")
        if not m:
            continue
        pair = _resolve_two_pillar_refs(m.group(1), m.group(2), pillars)
        if pair:
            return pair
    return None


def _parcel_centroid_pdf(positions: Dict[str, tuple[float, float]]) -> Optional[tuple[float, float]]:
    if not positions:
        return None
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _outward_normal_pdf(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    centroid: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Unit tangent (ux,uy) and outward unit normal (nx,ny) for edge A→B in PDF space."""
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    n1x, n1y = uy, -ux
    n2x, n2y = -uy, ux
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    vx, vy = centroid[0] - mx, centroid[1] - my
    if (n1x * vx + n1y * vy) >= (n2x * vx + n2y * vy):
        return ux, uy, -n1x, -n1y
    return ux, uy, -n2x, -n2y


def _score_road_geometry_at_edge(
    segs: Sequence[tuple],
    ax: float,
    ay: float,
    bx: float,
    by: float,
    *,
    label_pt: Optional[tuple[float, float]] = None,
    centroid: Optional[tuple[float, float]] = None,
) -> float:
    """
    Score how strongly PDF line work indicates a road beside traverse edge A→B.
    Rewards parallel line pairs offset outward from the boundary (typical road symbol).
    """
    if not segs or not centroid:
        return 0.0
    ux, uy, nx, ny = _outward_normal_pdf(ax, ay, bx, by, centroid)
    edge_angle = math.atan2(uy, ux)
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    parallel_count = 0
    pair_bonus = 0.0
    label_bonus = 0.0

    near_segs: List[tuple[float, float, float, float, float]] = []
    for seg in segs:
        x1, y1, x2, y2, slen = seg
        if slen < 6.0:
            continue
        sang = math.atan2(y2 - y1, x2 - x1)
        diff = abs(sang - edge_angle)
        diff = min(diff, abs(diff - math.pi))
        if diff > 0.18:
            continue
        smx, smy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        perp = _point_line_distance(smx, smy, ax, ay, bx, by)
        if perp < 2.0 or perp > 120.0:
            continue
        vx, vy = smx - mx, smy - my
        if (vx * nx + vy * ny) < -4.0:
            continue
        parallel_count += 1
        near_segs.append(seg)
        if label_pt and math.hypot(smx - label_pt[0], smy - label_pt[1]) < 90.0:
            label_bonus += 3.0

    for i, a in enumerate(near_segs):
        ax1, ay1, ax2, ay2, _ = a
        for b in near_segs[i + 1 :]:
            if not _segments_parallel(a, b, angle_tol=0.15):
                continue
            bx1, by1, bx2, by2, _ = b
            gap = _point_line_distance(bx1, by1, ax1, ay1, ax2, ay2)
            if 2.0 <= gap <= 70.0:
                pair_bonus += 5.0

    return float(parallel_count) + pair_bonus + label_bonus


def _detect_road_edge_from_pdf_geometry(
    page: Any,
    pillars: Sequence[str],
    *,
    label_pt: Optional[tuple[float, float]] = None,
) -> Optional[tuple[str, str]]:
    """Pick the boundary side with the strongest road line-work on the PDF page."""
    positions = _pillar_label_positions(page, pillars)
    if len(positions) < 2:
        return None
    centroid = _parcel_centroid_pdf(positions)
    if not centroid:
        return None
    segs = _pdf_page_line_segments(page)
    best_edge: Optional[tuple[str, str]] = None
    best_score = 0.0
    n = len(pillars)
    for i in range(n):
        a = pillars[i]
        b = pillars[(i + 1) % n]
        if a not in positions or b not in positions:
            continue
        ax, ay = positions[a]
        bx, by = positions[b]
        score = _score_road_geometry_at_edge(
            segs, ax, ay, bx, by, label_pt=label_pt, centroid=centroid
        )
        if score > best_score:
            best_score = score
            best_edge = (a, b)
    return best_edge if best_score >= 4.0 else None


def _nearest_edge_to_label(
    positions: Dict[str, tuple[float, float]],
    pillars: Sequence[str],
    label_pt: tuple[float, float],
) -> Optional[tuple[str, str]]:
    """Pick the traverse edge whose midpoint is closest to the road label on the PDF."""
    if len(pillars) < 2 or not positions:
        return None
    best_edge: Optional[tuple[str, str]] = None
    best_d = 1e18
    n = len(pillars)
    for i in range(n):
        a = pillars[i]
        b = pillars[(i + 1) % n]
        if a not in positions or b not in positions:
            continue
        mx = (positions[a][0] + positions[b][0]) / 2.0
        my = (positions[a][1] + positions[b][1]) / 2.0
        d = math.hypot(mx - label_pt[0], my - label_pt[1])
        if d < best_d:
            best_d = d
            best_edge = (a, b)
    return best_edge


def _road_width_from_parallel_segments(
    segs: Sequence[tuple],
    m_per_pt: float,
    *,
    near_pt: Optional[tuple[float, float]] = None,
) -> float:
    """Measure perpendicular spacing between parallel line pairs (road edges)."""
    if m_per_pt <= 0 or len(segs) < 2:
        return 0.0
    best_w = 0.0
    for i, a in enumerate(segs):
        ax1, ay1, ax2, ay2, alen = a
        if alen < 8.0:
            continue
        amx, amy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
        if near_pt and math.hypot(amx - near_pt[0], amy - near_pt[1]) > 160.0:
            continue
        for b in segs[i + 1 :]:
            if not _segments_parallel(a, b):
                continue
            bx1, by1, bx2, by2, blen = b
            if blen < 8.0:
                continue
            gap_pt = _point_line_distance(bx1, by1, ax1, ay1, ax2, ay2)
            if 1.5 <= gap_pt <= 80.0:
                w_m = gap_pt * m_per_pt
                if 2.5 <= w_m <= 18.0 and w_m > best_w:
                    best_w = w_m
    return best_w


def _road_pair_key(pair: tuple[str, str]) -> tuple[str, str]:
    a, b = (pair[0] or "").upper(), (pair[1] or "").upper()
    return tuple(sorted((a, b)))


def _pair_from_access_road_spec(spec: str) -> Optional[tuple[str, str]]:
    """Parse the two pillar labels from a cadastral access-road spec string."""
    s = spec or ""
    m = re.search(
        r"(?:on\s+the\s+side\s+of|joining\s+pillars|connecting|between)\s+(.+?)\s+and\s+(.+?)"
        r"(?:\s+offset|\s*$|\s*;|\s*,|\s+with\b)",
        s,
        re.IGNORECASE,
    )
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


def _format_access_road_spec(
    width_m: float,
    pair: tuple[str, str],
    *,
    title: str = "",
) -> str:
    width_s = f"{width_m:.1f}".rstrip("0").rstrip(".")
    base = f"an access road of width {width_s}m on the side of {pair[0]} and {pair[1]}"
    norm_title = normalize_access_road_title(title)
    if norm_title and norm_title not in ("ACCESS    ROAD",):
        return (
            f"an access road titled '{norm_title}' of width {width_s}m "
            f"on the side of {pair[0]} and {pair[1]}"
        )
    return base


def _merge_access_road_specs(
    primary: Sequence[str],
    supplemental: Sequence[str],
) -> List[str]:
    """Merge road specs, deduplicating by boundary-side pillar pair (primary wins)."""
    merged: List[str] = []
    seen_pairs: set[tuple[str, str]] = set()
    seen_specs: set[str] = set()

    def _add(spec: str) -> None:
        pair = _pair_from_access_road_spec(spec)
        if not pair:
            key = spec.strip().lower()
            if key in seen_specs:
                return
            seen_specs.add(key)
            merged.append(spec)
            return
        key = _road_pair_key(pair)
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        merged.append(spec)

    for spec in primary:
        if (spec or "").strip():
            _add(spec.strip())
    for spec in supplemental:
        if (spec or "").strip():
            _add(spec.strip())
    return merged


def _coerce_access_road_specs_from_llm(
    raw: Any,
    pillars: Sequence[str],
) -> List[str]:
    """Turn LLM JSON access_roads (strings or objects) into parser-friendly specs."""
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    if not isinstance(raw, list):
        return []

    specs: List[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            specs.append(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        width = _maybe_float(item.get("width_m") or item.get("width"))
        if not width or width <= 0:
            continue
        pa = str(item.get("pillar_a") or item.get("from_pillar") or "").strip()
        pb = str(item.get("pillar_b") or item.get("to_pillar") or "").strip()
        if not pa or not pb:
            chain = item.get("pillars") or item.get("pillar_pair")
            if isinstance(chain, list) and len(chain) >= 2:
                pa, pb = str(chain[0]).strip(), str(chain[1]).strip()
        pair = _resolve_two_pillar_refs(pa, pb, pillars) if pillars else None
        if not pair and pa and pb:
            pair = (pa, pb)
        if not pair:
            continue
        title = str(item.get("title") or item.get("access_road_title") or "").strip()
        specs.append(_format_access_road_spec(float(width), pair, title=title))
    return specs


_ACCESS_ROAD_LLM_SYSTEM = """You are a licensed Nigerian land surveyor reading a cadastral plan image.
Identify every ACCESS ROAD or ACCESS CLOSE / ACCESS/CLOSE drawn beside the parcel boundary.

Return ONLY JSON:
{
  "roads": [
    {
      "access_road_title": "ACCESS ROAD" or "ACCESS CLOSE" exactly as printed,
      "pillar_a": "full pillar label at one end of that road side",
      "pillar_b": "full pillar label at the other end of the same boundary side",
      "width_m": number in metres or null
    }
  ]
}

Rules:
- List one entry per distinct road label/symbol on the plan (a plan may have roads on more than one side).
- pillar_a and pillar_b must be two consecutive parcel corners (clockwise) on the side where that road is drawn.
- Read positions from the drawing — never assume the shortest traverse leg.
- If no access road/CLOSE is visible, return {"roads": []}."""


def _resolve_road_pillar_pair(
    text: str,
    extraction: SurveyPlanExtraction,
    *,
    close_near: Optional[tuple[float, float]] = None,
    road_label: Optional[tuple[str, tuple[float, float]]] = None,
    page: Any = None,
    access_road_specs: Optional[Sequence[str]] = None,
) -> Optional[tuple[str, str]]:
    """
    Resolve which traverse side carries the access road/CLOSE.

    Priority (PDF-first, no traverse heuristics):
    1. Pillar pair named in LLM/CAD road specs
    2. Pillar pair named explicitly in text
    3. Road label position on PDF → nearest boundary edge
    4. PDF line geometry (parallel road edges) beside a boundary side
    5. Never use shortest-leg or other guesses
    """
    pillars = extraction.pillar_numbers
    if len(pillars) < 2:
        return None

    label_pt = close_near
    if road_label:
        label_pt = road_label[1]

    specs = list(access_road_specs or [])
    specs.extend(extraction.access_roads or [])
    pair = _parse_pillar_pair_from_road_specs(specs, pillars)
    if pair:
        return pair

    explicit = _find_access_road_pillar_pair(text, pillars)
    if explicit:
        return explicit

    if page is not None:
        if label_pt:
            pos = _pillar_label_positions(page, pillars)
            edge = _nearest_edge_to_label(pos, pillars, label_pt)
            if edge:
                return edge

        geom_edge = _detect_road_edge_from_pdf_geometry(page, pillars, label_pt=label_pt)
        if geom_edge:
            return geom_edge

    return None


def resolve_access_roads_with_llm(
    pdf_path: str,
    extraction: SurveyPlanExtraction,
    *,
    llm: Any,
    run_with_timeout: Callable[..., Any],
    timeout_s: int = 45,
    pre_rendered_images: Optional[List[str]] = None,
) -> List[tuple[tuple[str, str], str, Optional[float]]]:
    """
    Vision LLM fallback: read every access road/CLOSE from the plan image.
    Returns list of (pillar_pair, title, width_m).
    """
    if not llm or not extraction.pillar_numbers:
        return []
    images = list(pre_rendered_images or [])
    if not images:
        images = render_pdf_pages_base64(pdf_path, max_pages=1)
    if not images:
        return []

    pillars_s = ", ".join(extraction.pillar_numbers)
    user_prompt = (
        f"Pillars in clockwise order: {pillars_s}\n"
        "List every boundary side that shows an access road or ACCESS/CLOSE label/symbol. "
        "Return the two pillar labels at the ends of each side and the printed title."
    )
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        content: List[dict] = [{"type": "text", "text": user_prompt}]
        for b64 in images[:1]:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            )
        messages = [SystemMessage(content=_ACCESS_ROAD_LLM_SYSTEM), HumanMessage(content=content)]
        msg, err, timed_out = run_with_timeout(timeout_s, lambda: llm.invoke(messages))
        if timed_out or err:
            return []
        raw = msg.content if hasattr(msg, "content") else str(msg)
        if isinstance(raw, list):
            raw = "\n".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in raw
            )
        data = _extract_json_object(str(raw))
        if not data:
            return []

        road_items: List[dict] = []
        if isinstance(data.get("roads"), list):
            road_items = [r for r in data["roads"] if isinstance(r, dict)]
        elif data.get("has_road"):
            road_items = [data]

        out: List[tuple[tuple[str, str], str, Optional[float]]] = []
        seen: set[tuple[str, str]] = set()
        for item in road_items:
            pair = _resolve_two_pillar_refs(
                str(item.get("pillar_a") or ""),
                str(item.get("pillar_b") or ""),
                extraction.pillar_numbers,
            )
            if not pair:
                continue
            key = _road_pair_key(pair)
            if key in seen:
                continue
            seen.add(key)
            title = normalize_access_road_title(str(item.get("access_road_title") or ""))
            width = _maybe_float(item.get("width_m"))
            out.append((pair, title, width))
        return out
    except Exception as exc:
        logger.debug("LLM access-road resolution failed: %s", exc)
        return []


def resolve_access_road_with_llm(
    pdf_path: str,
    extraction: SurveyPlanExtraction,
    *,
    llm: Any,
    run_with_timeout: Callable[..., Any],
    timeout_s: int = 45,
) -> Optional[tuple[tuple[str, str], str, Optional[float]]]:
    """Single-road LLM helper (first road only)."""
    roads = resolve_access_roads_with_llm(
        pdf_path,
        extraction,
        llm=llm,
        run_with_timeout=run_with_timeout,
        timeout_s=timeout_s,
    )
    return roads[0] if roads else None


def _road_spec_for_label(
    page: Any,
    extraction: SurveyPlanExtraction,
    text: str,
    *,
    title: str,
    label_pt: tuple[float, float],
    pdf_path: str,
    m_per_pt: float,
    segs: Sequence[tuple],
    llm_width: Optional[float] = None,
) -> Optional[str]:
    """Build one access-road spec for a labelled boundary side."""
    road_label = (title, label_pt)
    pair = _resolve_road_pillar_pair(
        text,
        extraction,
        road_label=road_label,
        close_near=label_pt,
        page=page,
    )
    if not pair:
        positions = _pillar_label_positions(page, extraction.pillar_numbers)
        pair = _nearest_edge_to_label(positions, extraction.pillar_numbers, label_pt)
    if not pair:
        return None

    width_m = float(llm_width) if llm_width and llm_width > 0 else 0.0
    if width_m <= 0.0:
        width_m = _road_width_from_parallel_segments(
            segs, float(m_per_pt or 0.0), near_pt=label_pt
        )
    if width_m <= 0.0:
        width_m = estimate_road_width_from_pdf(
            pdf_path,
            scale_denom=extraction.scale_denom,
            reference_leg_m=_leg_length_for_pair(extraction, pair),
        )
    norm_title = normalize_access_road_title(title) if title else "ACCESS    ROAD"
    if not norm_title:
        norm_title = extract_access_road_title_from_text(text) or "ACCESS    ROAD"
    return _format_access_road_spec(width_m, pair, title=norm_title)


def analyze_pdf_access_roads(
    pdf_path: str,
    extraction: SurveyPlanExtraction,
    text: str,
    *,
    llm: Any = None,
    run_with_timeout: Optional[Callable[..., Any]] = None,
    pre_rendered_images: Optional[List[str]] = None,
) -> Optional[tuple[List[str], str]]:
    """
    Detect every access road/CLOSE from PDF labels + line geometry (+ vision LLM if needed).
    Returns (road_specs, primary_title) compatible with the cadastral CAD parser.
    """
    path = Path(pdf_path)
    if not path.exists():
        return None
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None

    try:
        doc = fitz.open(str(path))
        page = doc[0]
        labels = _extract_all_access_road_labels_from_pdf(page)
        m_per_pt = calibrate_pdf_meters_per_point(
            page,
            extraction,
            scale_denom=extraction.scale_denom,
        )
        segs = _pdf_page_line_segments(page)

        specs: List[str] = []
        used_pairs: set[tuple[str, str]] = set()
        primary_title = ""

        if labels:
            for _idx, (title, label_pt) in enumerate(labels):
                spec = _road_spec_for_label(
                    page,
                    extraction,
                    text,
                    title=title,
                    label_pt=label_pt,
                    pdf_path=pdf_path,
                    m_per_pt=float(m_per_pt or 0.0),
                    segs=segs,
                )
                if not spec:
                    continue
                pair = _pair_from_access_road_spec(spec)
                if not pair:
                    continue
                key = _road_pair_key(pair)
                if key in used_pairs:
                    continue
                used_pairs.add(key)
                if not primary_title:
                    m_t = re.search(r"titled\s+'([^']+)'", spec, re.IGNORECASE)
                    primary_title = m_t.group(1) if m_t else normalize_access_road_title(title)
                specs.append(spec)

        if not specs:
            road_label = _extract_access_road_label_from_pdf(page)
            pair = _resolve_road_pillar_pair(
                text,
                extraction,
                road_label=road_label,
                close_near=road_label[1] if road_label else None,
                page=page,
            )
            if pair:
                label_pt = road_label[1] if road_label else None
                title = road_label[0] if road_label else ""
                spec = _road_spec_for_label(
                    page,
                    extraction,
                    text,
                    title=title,
                    label_pt=label_pt or (0.0, 0.0),
                    pdf_path=pdf_path,
                    m_per_pt=float(m_per_pt or 0.0),
                    segs=segs,
                )
                if spec:
                    specs.append(spec)
                    m_t = re.search(r"titled\s+'([^']+)'", spec, re.IGNORECASE)
                    primary_title = (
                        m_t.group(1) if m_t else normalize_access_road_title(title)
                    )

        if not specs and llm is not None and run_with_timeout is not None:
            llm_roads = resolve_access_roads_with_llm(
                pdf_path,
                extraction,
                llm=llm,
                run_with_timeout=run_with_timeout,
                pre_rendered_images=pre_rendered_images,
            )
            for pair, title, llm_width in llm_roads:
                key = _road_pair_key(pair)
                if key in used_pairs:
                    continue
                used_pairs.add(key)
                width_m = (
                    float(llm_width)
                    if llm_width and llm_width > 0
                    else estimate_road_width_from_pdf(
                        pdf_path,
                        scale_denom=extraction.scale_denom,
                        reference_leg_m=_leg_length_for_pair(extraction, pair),
                    )
                )
                norm_title = normalize_access_road_title(title)
                if not primary_title:
                    primary_title = norm_title
                specs.append(_format_access_road_spec(width_m, pair, title=norm_title))

        doc.close()
        if not specs:
            return None
        if not primary_title:
            primary_title = extract_access_road_title_from_text(text) or "ACCESS    ROAD"
        return specs, normalize_access_road_title(primary_title)
    except Exception as exc:
        logger.debug("PDF access road analysis failed for %s: %s", pdf_path, exc)
        return None


def analyze_pdf_access_road(
    pdf_path: str,
    extraction: SurveyPlanExtraction,
    text: str,
    *,
    llm: Any = None,
    run_with_timeout: Optional[Callable[..., Any]] = None,
) -> Optional[tuple[List[str], str]]:
    """Legacy alias — detects one or more access roads from the PDF."""
    return analyze_pdf_access_roads(
        pdf_path,
        extraction,
        text,
        llm=llm,
        run_with_timeout=run_with_timeout,
    )


def _leg_length_for_pair(
    extraction: SurveyPlanExtraction,
    pair: tuple[str, str],
) -> Optional[float]:
    legs = extraction.traverse_legs
    pillars = extraction.pillar_numbers
    if not legs or not pillars:
        return None
    try:
        i = pillars.index(pair[0])
        j = pillars.index(pair[1])
        if abs(i - j) == 1 or (i == 0 and j == len(pillars) - 1):
            idx = min(i, j)
            if idx < len(legs):
                return float(legs[idx].distance_m)
    except Exception:
        pass
    if legs:
        return min(float(leg.distance_m) for leg in legs if leg.distance_m > 0)
    return None


def estimate_road_width_from_pdf(
    pdf_path: str,
    *,
    scale_denom: Optional[int] = None,
    reference_leg_m: Optional[float] = None,
) -> float:
    """
    Estimate access-road width (metres) from PDF geometry relative to scale / traverse legs.
    Falls back to 6.0 m when geometry cannot be measured reliably.
    """
    default = 6.0
    path = Path(pdf_path)
    if not path.exists():
        return default
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return default

    try:
        doc = fitz.open(str(path))
        page = doc[0]
        words = page.get_text("words") or []

        m_per_pt: Optional[float] = None
        if reference_leg_m and reference_leg_m > 0:
            dist_tokens: List[tuple[float, float, float]] = []
            for w in words:
                token = str(w[4] or "").strip()
                m_dist = re.match(r"^(\d+(?:\.\d+)?)\s*m\.?$", token, re.IGNORECASE)
                if m_dist:
                    cx = (float(w[0]) + float(w[2])) / 2.0
                    cy = (float(w[1]) + float(w[3])) / 2.0
                    dist_tokens.append((cx, cy, float(m_dist.group(1))))
            drawings = page.get_drawings()
            seg_lens: List[float] = []
            for d in drawings:
                for item in d.get("items") or []:
                    if not item or item[0] != "l":
                        continue
                    p1, p2 = item[1], item[2]
                    seg_len = math.hypot(float(p2.x - p1.x), float(p2.y - p1.y))
                    if seg_len >= 20.0:
                        seg_lens.append(seg_len)
            if dist_tokens and seg_lens:
                ref_m = min(reference_leg_m, max(t[2] for t in dist_tokens))
                ref_pt = sorted(seg_lens)[len(seg_lens) // 3]
                if ref_pt > 1.0:
                    m_per_pt = ref_m / ref_pt

        if m_per_pt is None and scale_denom and int(scale_denom) > 0:
            # 1 drawing point ≈ 1/72 inch on paper; ground metres ≈ (pt/72)*25.4mm * scale/1000
            m_per_pt = (25.4 / 72.0) * (float(scale_denom) / 1000.0)

        access_y: Optional[float] = None
        road_label = _extract_access_road_label_from_pdf(page)
        if road_label:
            access_y = road_label[1][1]
        else:
            for w in words:
                tok = str(w[4] or "").upper().replace(" ", "")
                if tok in ("ACCESSROAD", "ACCESS/CLOSE", "ACCESSCLOSE", "CLOSE"):
                    access_y = (float(w[1]) + float(w[3])) / 2.0
                    break

        horiz: List[tuple[float, float]] = []
        for d in page.get_drawings():
            for item in d.get("items") or []:
                if not item or item[0] != "l":
                    continue
                p1, p2 = item[1], item[2]
                dx = abs(float(p2.x - p1.x))
                dy = abs(float(p2.y - p1.y))
                if dx < dy or dx < 8.0:
                    continue
                y = (float(p1.y) + float(p2.y)) / 2.0
                if access_y is not None and abs(y - access_y) > 120.0:
                    continue
                horiz.append((y, dx))

        width_m = 0.0
        if m_per_pt and len(horiz) >= 2:
            horiz.sort(key=lambda t: t[0])
            for i in range(len(horiz) - 1):
                gap_pt = abs(horiz[i + 1][0] - horiz[i][0])
                if 2.0 <= gap_pt <= 60.0:
                    cand = gap_pt * m_per_pt
                    if 3.0 <= cand <= 15.0:
                        width_m = cand
                        break

        doc.close()
        if width_m > 0:
            return round(width_m, 1)
    except Exception as exc:
        logger.debug("Road width estimate failed for %s: %s", pdf_path, exc)
    return default


def _find_access_road_pillar_pair(
    text: str,
    pillars: Sequence[str],
) -> Optional[tuple[str, str]]:
    """Return pillar pair only when explicitly named beside the road in text/LLM output."""
    if len(pillars) < 2:
        return None
    upper = text or ""
    nums = [re.findall(r"\d+", p)[-1] for p in pillars if re.findall(r"\d+", p)]
    for i in range(len(nums) - 1):
        a_num, b_num = nums[i], nums[i + 1]
        if re.search(
            rf"(?:side\s+of|between|joining|connecting)\s+.*{re.escape(a_num)}\s*(?:and|&|-|/)\s*{re.escape(b_num)}",
            upper,
            re.IGNORECASE,
        ) or re.search(
            rf"(?:side\s+of|between|joining|connecting)\s+.*{re.escape(b_num)}\s*(?:and|&|-|/)\s*{re.escape(a_num)}",
            upper,
            re.IGNORECASE,
        ):
            return pillars[i], pillars[i + 1]
    return None


def infer_access_roads_from_text(
    text: str,
    extraction: SurveyPlanExtraction,
    *,
    pdf_path: Optional[str] = None,
    llm: Any = None,
    run_with_timeout: Optional[Callable[..., Any]] = None,
    pre_rendered_images: Optional[List[str]] = None,
) -> tuple[List[str], str]:
    """Build parser-friendly access-road specs and title from PDF geometry/labels (+ LLM if needed)."""
    llm_specs = _coerce_access_road_specs_from_llm(
        extraction.access_roads,
        extraction.pillar_numbers,
    )

    if pdf_path:
        analyzed = analyze_pdf_access_roads(
            pdf_path,
            extraction,
            text,
            llm=llm,
            run_with_timeout=run_with_timeout,
            pre_rendered_images=pre_rendered_images,
        )
        if analyzed:
            pdf_roads, pdf_title = analyzed
            merged = _merge_access_road_specs(pdf_roads, llm_specs)
            return merged, pdf_title
        if llm_specs:
            title = extraction.access_road_title or extract_access_road_title_from_text(text)
            return llm_specs, normalize_access_road_title(title)

    all_labels: List[tuple[str, tuple[float, float]]] = []
    page = None
    if pdf_path and Path(pdf_path).exists():
        try:
            import fitz

            doc = fitz.open(str(pdf_path))
            page = doc[0]
            all_labels = _extract_all_access_road_labels_from_pdf(page)
            doc.close()
        except Exception:
            all_labels = []
            page = None

    has_road_hint = bool(
        all_labels
        or llm_specs
        or extract_access_road_title_from_text(text)
        or re.search(r"access\s+road|access\s*[/\s]*close|access\s+close", text or "", re.IGNORECASE)
    )
    if not has_road_hint:
        return [], ""

    specs: List[str] = []
    used_pairs: set[tuple[str, str]] = set()
    primary_title = ""

    if all_labels and page is not None:
        try:
            import fitz

            doc = fitz.open(str(pdf_path))
            page = doc[0]
            m_per_pt = calibrate_pdf_meters_per_point(
                page,
                extraction,
                scale_denom=extraction.scale_denom,
            )
            segs = _pdf_page_line_segments(page)
            for title, label_pt in all_labels:
                spec = _road_spec_for_label(
                    page,
                    extraction,
                    text,
                    title=title,
                    label_pt=label_pt,
                    pdf_path=pdf_path or "",
                    m_per_pt=float(m_per_pt or 0.0),
                    segs=segs,
                )
                if not spec:
                    continue
                pair = _pair_from_access_road_spec(spec)
                if not pair:
                    continue
                key = _road_pair_key(pair)
                if key in used_pairs:
                    continue
                used_pairs.add(key)
                if not primary_title:
                    m_t = re.search(r"titled\s+'([^']+)'", spec, re.IGNORECASE)
                    primary_title = m_t.group(1) if m_t else normalize_access_road_title(title)
                specs.append(spec)
            doc.close()
        except Exception:
            pass

    if not specs:
        road_label = all_labels[0] if all_labels else None
        if pdf_path and Path(pdf_path).exists() and page is None:
            try:
                import fitz

                doc = fitz.open(str(pdf_path))
                page = doc[0]
                road_label = road_label or _extract_access_road_label_from_pdf(page)
                doc.close()
            except Exception:
                page = None

        pair = _resolve_road_pillar_pair(
            text,
            extraction,
            road_label=road_label,
            close_near=road_label[1] if road_label else None,
            page=page,
        )
        llm_title = ""
        llm_width: Optional[float] = None
        if not pair and pdf_path and llm is not None and run_with_timeout is not None:
            llm_result = resolve_access_road_with_llm(
                pdf_path,
                extraction,
                llm=llm,
                run_with_timeout=run_with_timeout,
            )
            if llm_result:
                pair, llm_title, llm_width = llm_result
        if pair:
            ref_leg_m = _leg_length_for_pair(extraction, pair)
            width_m = (
                float(llm_width)
                if llm_width and llm_width > 0
                else estimate_road_width_from_pdf(
                    pdf_path or "",
                    scale_denom=extraction.scale_denom,
                    reference_leg_m=ref_leg_m,
                )
            )
            title = llm_title or (road_label[0] if road_label else "")
            if not title:
                title = extract_access_road_title_from_text(text)
            if not title:
                title = "ACCESS    ROAD"
            primary_title = normalize_access_road_title(title)
            specs.append(_format_access_road_spec(width_m, pair, title=primary_title))

    merged = _merge_access_road_specs(specs, llm_specs)
    if not merged:
        return [], ""
    if not primary_title:
        primary_title = extraction.access_road_title or extract_access_road_title_from_text(text)
    if not primary_title:
        primary_title = "ACCESS    ROAD"
    return merged, normalize_access_road_title(primary_title)


def prepare_extraction_for_cadastral(
    extraction: SurveyPlanExtraction,
    combined_text: str,
    *,
    pdf_path: Optional[str] = None,
    llm: Any = None,
    run_with_timeout: Optional[Callable[..., Any]] = None,
    pre_rendered_images: Optional[List[str]] = None,
) -> SurveyPlanExtraction:
    """Sanitize metadata, normalize plan number, and infer access roads for CAD plotting."""
    extraction.buyer_name = sanitize_metadata_field(extraction.buyer_name, max_len=140)
    extraction.location = sanitize_metadata_field(extraction.location, max_len=200)
    if not extraction.location:
        extraction.location = extract_location_from_text(combined_text)
        extraction.location = sanitize_metadata_field(extraction.location, max_len=200)
    extraction.lga = sanitize_metadata_field(
        normalize_lga_name(extraction.lga) or extraction.lga, max_len=80
    )
    extraction.state = sanitize_metadata_field(extraction.state, max_len=40)
    extraction.surveyor_name = sanitize_metadata_field(
        ensure_surveyor_professional_title(extraction.surveyor_name or ""),
        max_len=100,
    )
    extraction.surveyor_address = sanitize_metadata_field(extraction.surveyor_address, max_len=120)
    extraction.plan_number = normalize_plan_number(extraction.plan_number)
    if not extraction.scale_denom:
        extraction.scale_denom = extract_scale_denom_from_text(combined_text)

    clean_roads: List[str] = []
    for road in extraction.access_roads:
        r = re.sub(r"\s+", " ", (road or "").strip())
        if not r or _METADATA_JUNK_RE.search(r):
            continue
        if not re.search(r"\d+(?:\.\d+)?\s*m", r, re.IGNORECASE):
            continue
        if not r.lower().startswith("add"):
            r = f"an {r}" if r.lower().startswith("access") else r
        clean_roads.append(r)
    extraction.access_roads = clean_roads

    # PDF labels + line geometry (+ vision LLM only when needed) determine road side/title.
    if pdf_path:
        ext_snapshot = extraction

        def _roads_task() -> tuple[List[str], str]:
            return infer_access_roads_from_text(
                combined_text,
                ext_snapshot,
                pdf_path=pdf_path,
                llm=llm,
                run_with_timeout=run_with_timeout,
                pre_rendered_images=pre_rendered_images,
            )

        def _pdf_fences_task() -> List[str]:
            return infer_fences_from_pdf(
                ext_snapshot,
                pdf_path=pdf_path,
                combined_text=combined_text,
            )

        roads_title, pdf_fences, note_fences, text_fences = _parallel_invoke(
            _roads_task,
            _pdf_fences_task,
            lambda: infer_fences_from_boundary_text(extraction.notes or ""),
            lambda: infer_fences_from_boundary_text(combined_text),
        )
        pdf_roads, pdf_title = roads_title
        if pdf_roads:
            extraction.access_roads = pdf_roads
        elif clean_roads:
            extraction.access_roads = clean_roads
        if pdf_title:
            extraction.access_road_title = pdf_title
        llm_fences = list(extraction.fences or [])
        extraction.fences = merge_fence_spec_strings(
            [*llm_fences, *pdf_fences, *note_fences, *text_fences]
        )
    elif not extraction.access_roads:
        roads_title, note_fences, text_fences = _parallel_invoke(
            lambda: infer_access_roads_from_text(combined_text, extraction, pdf_path=None),
            lambda: infer_fences_from_boundary_text(extraction.notes or ""),
            lambda: infer_fences_from_boundary_text(combined_text),
        )
        roads, title = roads_title
        extraction.access_roads = roads
        if title and not extraction.access_road_title:
            extraction.access_road_title = title
        llm_fences = list(extraction.fences or [])
        extraction.fences = merge_fence_spec_strings(
            [*llm_fences, *note_fences, *text_fences]
        )
    else:
        note_fences, text_fences = _parallel_invoke(
            lambda: infer_fences_from_boundary_text(extraction.notes or ""),
            lambda: infer_fences_from_boundary_text(combined_text),
        )
        llm_fences = list(extraction.fences or [])
        extraction.fences = merge_fence_spec_strings(
            [*llm_fences, *note_fences, *text_fences]
        )

    extraction.access_road_title = _reconcile_access_road_title(
        extraction.access_road_title,
        combined_text,
        pdf_path=pdf_path,
    )

    return extraction


def _reconcile_access_road_title(
    current: str,
    combined_text: str,
    *,
    pdf_path: Optional[str] = None,
) -> str:
    """
    Prefer labels read from the PDF drawing over LLM/heuristic guesses.
    Default to ACCESS    ROAD — never assume ACCESS CLOSE without evidence.
    """
    from_pdf = ""
    if pdf_path and Path(pdf_path).exists():
        try:
            import fitz

            doc = fitz.open(str(pdf_path))
            found = _extract_access_road_label_from_pdf(doc[0])
            doc.close()
            if found:
                from_pdf = found[0]
        except Exception:
            from_pdf = ""

    from_text = extract_access_road_title_from_text(combined_text)
    preferred = from_pdf or from_text
    if preferred:
        return normalize_access_road_title(preferred)

    if current:
        return normalize_access_road_title(current)

    if re.search(r"access\s*[/\s]*close|access\s+close", combined_text, re.IGNORECASE):
        return "ACCESS CLOSE"
    return "ACCESS    ROAD"


def extract_plain_text_from_pdf(pdf_path: str, *, max_pages: int = 3) -> str:
    """Plain pdfplumber extract_text (supplements layout-ordered reconstruction)."""
    import pdfplumber

    path = Path(pdf_path).resolve()
    if not path.exists():
        return ""
    parts: List[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages[:max_pages]:
                t = page.extract_text() or ""
                if t.strip():
                    parts.append(t)
    except Exception as exc:
        logger.warning("Plain PDF text extraction failed for %s: %s", pdf_path, exc)
    return "\n".join(parts)


def extract_heuristics_from_layout_text(layout_text: str) -> SurveyPlanExtraction:
    """
    Regex/layout fallback when LLM extraction is weak — common Nigerian plan PDFs.
    """
    text = layout_text or ""
    upper = text.upper()

    pillars = [
        _normalize_pillar_id(m.group(0))
        for m in _PILLAR_ID_TEXT_RE.finditer(text)
    ]
    # de-dupe preserving order
    seen: set[str] = set()
    pillar_list: List[str] = []
    for p in _filter_plausible_pillars(pillars):
        if p not in seen:
            seen.add(p)
            pillar_list.append(p)

    for line in text.splitlines():
        tokens = line.split()
        for i, tok in enumerate(tokens):
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            pid = _parse_pillar_token(tok, nxt)
            if pid and _is_plausible_pillar_id(pid) and pid not in seen:
                seen.add(pid)
                pillar_list.append(pid)

    anchor_e, anchor_n, _, _ = extract_grid_coordinates_from_text(text)

    legs: List[SurveyTraverseLeg] = []
    leg_patterns = [
        re.compile(
            r"(\d{2,3})\s*°\s*(\d{1,2})\s*['′]?\s*(?:\d{1,2}\s*['′]?\s*)?"
            r"(\d+(?:\.\d+)?)\s*m\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(\d{2,3})\s*°\s*(\d{1,2})\s*['′]?\D{0,40}?(\d+(?:\.\d+)?)\s*m\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(\d{2,3})\s+(\d{1,2})\s*['′]?\s*(\d+(?:\.\d+)?)\s*m\b",
            flags=re.IGNORECASE,
        ),
    ]
    for leg_re in leg_patterns:
        for m in leg_re.finditer(text):
            try:
                leg = SurveyTraverseLeg(
                    bearing_deg=int(m.group(1)),
                    bearing_min=int(m.group(2)),
                    distance_m=float(m.group(3)),
                )
                if _is_plausible_boundary_leg(leg):
                    legs.append(leg)
            except Exception:
                continue
        if len(legs) >= 3:
            break

    legs = _filter_plausible_legs(legs)

    if len(legs) < 3:
        bearings = [
            (int(a), int(b))
            for a, b in re.findall(
                r"(\d{2,3})\s*°\s*(\d{1,2})\s*['′]?",
                text,
                flags=re.IGNORECASE,
            )
        ]
        distances = [
            float(d)
            for d in re.findall(r"(\d+(?:\.\d+)?)\s*m\b", text, flags=re.IGNORECASE)
            if float(d) <= 500.0
        ]
        if len(bearings) >= 3 and len(distances) >= 3:
            n = min(len(bearings), len(distances), 8)
            legs = _filter_plausible_legs(
                [
                    SurveyTraverseLeg(
                        bearing_deg=bd,
                        bearing_min=bm,
                        distance_m=distances[i],
                    )
                    for i, (bd, bm) in enumerate(bearings[:n])
                ]
            )

    # Pair pillars to legs clockwise when counts align
    if pillar_list and legs and len(pillar_list) == len(legs):
        for i, leg in enumerate(legs):
            leg.from_pillar = pillar_list[i]
            leg.to_pillar = pillar_list[(i + 1) % len(pillar_list)]

    buyer = ""
    for pat in (
        r"OWNER[S]?\s*[:=]\s*(.+?)(?:\n|LOCATION|LGA|STATE|$)",
        r"PLAN\s+SHEWING\s+LANDED\s+PROPERTY\s+OF\s+(.+?)(?:\n|\bAT\s+)",
        r"PLAN\s+SHEWING\s+LANDED\s+PROPERTY\s+(.+?)(?:\n|LOCATION|\bAT\s+)",
    ):
        m = re.search(pat, upper, flags=re.IGNORECASE | re.DOTALL)
        if m:
            buyer = re.sub(r"\s+", " ", m.group(1)).strip(" :-")
            if buyer:
                break

    def _field(label: str) -> str:
        m = re.search(
            rf"{label}\s*[:=]\s*(.+?)(?:\n|,|$)",
            text,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""

    location = extract_location_from_text(text) or _field("LOCATION")
    scale_denom = extract_scale_denom_from_text(text)
    plan_m = re.search(
        r"(?:PLAN\s*(?:NO\.?|NUMBER)\s*[:=]?\s*)([A-Z0-9/\-]+)",
        upper,
        flags=re.IGNORECASE,
    )

    surveyor = _field("SURV") or _field("SURVEYOR")
    if not surveyor:
        m_surv = re.search(
            r"\b(SURV\.?\s+[A-Z][A-Z\s\.\-']{2,80})",
            text or "",
            flags=re.IGNORECASE,
        )
        if m_surv:
            surveyor = re.sub(r"\s+", " ", m_surv.group(1)).strip(" ,;:-")

    conf = 0.35
    if anchor_e and anchor_n:
        conf += 0.2
    if len(legs) >= 3:
        conf += 0.25
    if pillar_list:
        conf += 0.1

    return SurveyPlanExtraction(
        buyer_name=buyer,
        location=location,
        lga=_field("LOCAL GOVERNMENT AREA") or _field("LGA"),
        state=_field("STATE"),
        origin_crs=_field("UTM") or ("UTM ZONE 32N" if "UTM" in upper or "ZONE 32" in upper else ""),
        plan_number=normalize_plan_number(plan_m.group(1).strip() if plan_m else _field("PLAN NO")),
        surveyor_name=ensure_surveyor_professional_title(surveyor) if surveyor else "",
        scale_denom=scale_denom,
        pillar_numbers=pillar_list,
        anchor_easting=anchor_e,
        anchor_northing=anchor_n,
        anchor_pillar=pillar_list[0] if pillar_list else "",
        traverse_legs=legs,
        confidence=min(conf, 0.85),
        source="layout_heuristic",
        notes="Heuristic extraction from layout-ordered PDF text",
    )


def merge_survey_extractions(
    primary: SurveyPlanExtraction,
    fallback: SurveyPlanExtraction,
) -> SurveyPlanExtraction:
    """Prefer primary values; fill gaps from fallback."""
    data = primary.model_dump()
    fb = fallback.model_dump()
    fb_legs = _filter_plausible_legs(fallback.traverse_legs)
    primary_legs = _filter_plausible_legs(primary.traverse_legs)
    for key, val in fb.items():
        if key in ("confidence", "source", "notes"):
            continue
        cur = data.get(key)
        if key in ("anchor_easting", "anchor_northing", "traverse_legs", "pillar_numbers"):
            continue
        if cur in (None, "", [], 0.0) and val not in (None, "", [], 0.0):
            data[key] = val
    # Coordinates: prefer plausible UTM values from either source.
    for axis, fb_key in (("e", "anchor_easting"), ("n", "anchor_northing")):
        pe = data.get("anchor_easting")
        pn = data.get("anchor_northing")
        pval = pe if fb_key == "anchor_easting" else pn
        fval = fb.get(fb_key)
        check = _is_plausible_utm_easting if axis == "e" else _is_plausible_utm_northing
        if fval is not None and check(float(fval)):
            if pval is None or not check(float(pval)):
                data[fb_key] = fval
    if not primary_legs and fb_legs:
        data["traverse_legs"] = fb_legs
    elif primary_legs:
        data["traverse_legs"] = primary_legs
    if not data.get("pillar_numbers") and fallback.pillar_numbers:
        data["pillar_numbers"] = fallback.pillar_numbers
    if float(data.get("confidence") or 0) < float(fb.get("confidence") or 0):
        data["confidence"] = fb["confidence"]
    if primary.source != "error" and fb_legs and not primary_legs:
        data["notes"] = (
            f"{primary.notes or ''} | supplemented by {fallback.source}".strip(" |")
        )
    return SurveyPlanExtraction(**data)


def extract_layout_text_from_pdf(pdf_path: str, *, max_pages: int = 3) -> str:
    """Rebuild reading-order text from PDF word positions (more reliable than extract_text)."""
    import pdfplumber

    path = Path(pdf_path).resolve()
    if not path.exists():
        return ""
    lines_out: List[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages[:max_pages]:
                lines_out.append(f"--- PAGE {page.page_number} ---")
                words = page.extract_words(
                    keep_blank_chars=False,
                    use_text_flow=True,
                    extra_attrs=["size"],
                ) or []
                if words:
                    buckets: Dict[float, List[dict]] = {}
                    for w in words:
                        top = round(float(w.get("top", 0.0)), 1)
                        buckets.setdefault(top, []).append(w)
                    for top in sorted(buckets.keys()):
                        row = sorted(buckets[top], key=lambda x: float(x.get("x0", 0.0)))
                        line = " ".join(str(x.get("text") or "").strip() for x in row if x.get("text"))
                        if line.strip():
                            lines_out.append(line)
                for table in page.extract_tables() or []:
                    for row in table or []:
                        if not row:
                            continue
                        cells = [str(c or "").strip() for c in row]
                        if any(cells):
                            lines_out.append("TABLE | " + " | ".join(cells))
    except Exception as exc:
        logger.warning("PDF layout text extraction failed for %s: %s", pdf_path, exc)
    return "\n".join(lines_out)


def render_pdf_pages_base64(
    pdf_path: str,
    *,
    max_pages: int = 2,
    dpi: int = 144,
) -> List[str]:
    """Render PDF pages to PNG base64 strings for vision models (PyMuPDF if available)."""
    path = Path(pdf_path).resolve()
    if not path.exists():
        return []
    images: List[str] = []
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        try:
            zoom = float(dpi) / 72.0
            mat = fitz.Matrix(zoom, zoom)
            for i in range(min(len(doc), max_pages)):
                pix = doc.load_page(i).get_pixmap(matrix=mat, alpha=False)
                images.append(base64.b64encode(pix.tobytes("png")).decode("ascii"))
        finally:
            doc.close()
        return images
    except ImportError:
        logger.info("PyMuPDF not installed; PDF vision pass will use layout text only")
    except Exception as exc:
        logger.warning("PDF page render failed for %s: %s", pdf_path, exc)
    return images


def _coerce_legs(raw: Any) -> List[SurveyTraverseLeg]:
    legs: List[SurveyTraverseLeg] = []
    if not isinstance(raw, list):
        return legs
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            legs.append(
                SurveyTraverseLeg(
                    from_pillar=str(item.get("from_pillar") or item.get("from") or "").strip(),
                    to_pillar=str(item.get("to_pillar") or item.get("to") or "").strip(),
                    bearing_deg=int(item.get("bearing_deg") or item.get("bearing_d") or 0),
                    bearing_min=int(item.get("bearing_min") or item.get("bearing_m") or 0),
                    distance_m=float(item.get("distance_m") or item.get("distance") or 0.0),
                )
            )
        except Exception:
            continue
    return [lg for lg in legs if lg.distance_m > 0]


def _normalize_pillar_id(raw: str) -> str:
    text = re.sub(r"\s+", " ", (raw or "").strip().replace("\n", " "))
    split = split_cadastral_pillar_label(text)
    if split:
        return f"{split['prefix']} {split['number']}"
    compact = re.sub(r"\s+", "", text.upper())
    # Legacy compact SC/BV6015 (3–5 digit) still covered by split; keep soft fallbacks.
    m = re.match(r"^(SC|SP|RV|RP)/?([A-Z]{1,6})([A-Z0-9]{3,12})$", compact)
    if m and re.search(r"\d", m.group(3)):
        return f"{m.group(1)}/{m.group(2)} {m.group(3)}"
    text = (
        text.replace("SCCR", "SC/CR")
        .replace("Sc/Cr", "SC/CR")
        .replace("SCCK", "SC/CK")
        .replace("Sc/Ck", "SC/CK")
        .replace("SCQ", "SC/Q")
        .replace("Sc/Q", "SC/Q")
        .replace("SCBV", "SC/BV")
        .replace("Sc/Bv", "SC/BV")
        .replace("SPRV", "SP/RV")
        .replace("Sp/Rv", "SP/RV")
    )
    text = re.sub(r"\s+", " ", text).strip()
    m2 = re.match(
        r"^(SC|SP|RV|RP)\s*/?\s*([A-Z]{1,6})\s+([A-Z0-9]{3,12})$",
        text,
        re.IGNORECASE,
    )
    if m2 and re.search(r"\d", m2.group(3)):
        return f"{m2.group(1).upper()}/{m2.group(2).upper()} {m2.group(3).upper()}"
    return text


def _pdf_bearing_labels(page: Any) -> List[tuple[float, float, int, int]]:
    """Collect bearing label positions from PDF words (handles split degree/minute tokens)."""
    hits: List[tuple[float, float, int, int]] = []
    try:
        words = page.get_text("words") or []
    except Exception:
        return hits
    i = 0
    while i < len(words):
        w = words[i]
        tok = str(w[4] or "").strip()
        cx = (float(w[0]) + float(w[2])) / 2.0
        cy = (float(w[1]) + float(w[3])) / 2.0
        nxt = str(words[i + 1][4] or "").strip() if i + 1 < len(words) else ""

        m = re.match(r"^(\d{2,3})[°º](\d{1,2})['′]?$", tok)
        if m:
            bd, bm = int(m.group(1)), int(m.group(2))
            if bd <= 360 and bm < 60:
                hits.append((cx, cy, bd, bm))
            i += 1
            continue

        m = re.match(r"^(\d{2,3})[°º]$", tok)
        if m and re.match(r"^(\d{1,2})['′]?$", nxt):
            bd, bm = int(m.group(1)), int(re.match(r"^(\d{1,2})", nxt).group(1))
            if bd <= 360 and bm < 60:
                w2 = words[i + 1]
                cx = (cx + (float(w2[0]) + float(w2[2])) / 2.0) / 2.0
                cy = (cy + (float(w2[1]) + float(w2[3])) / 2.0) / 2.0
                hits.append((cx, cy, bd, bm))
            i += 2
            continue

        m = re.match(r"^(\d{2,3})$", tok)
        if m and re.match(r"^(\d{1,2})['′]$", nxt):
            bd, bm = int(m.group(1)), int(re.match(r"^(\d{1,2})", nxt).group(1))
            if bd <= 360 and bm < 60:
                w2 = words[i + 1]
                cx = (cx + (float(w2[0]) + float(w2[2])) / 2.0) / 2.0
                cy = (cy + (float(w2[1]) + float(w2[3])) / 2.0) / 2.0
                hits.append((cx, cy, bd, bm))
            i += 2
            continue

        i += 1
    return hits


def extract_boundary_legs_from_pdf(
    pdf_path: str,
    pillars: Sequence[str],
) -> List[SurveyTraverseLeg]:
    """
    Match bearing/distance labels on the PDF page to traverse edges using geometry.
    """
    if len(pillars) < 3 or not pdf_path or not Path(pdf_path).exists():
        return []
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        page = doc[0]
        positions = _pillar_label_positions(page, pillars)
        if len(positions) < len(pillars):
            doc.close()
            return []

        dist_labels = _pdf_distance_labels(page)
        bearing_hits = _pdf_bearing_labels(page)

        legs: List[SurveyTraverseLeg] = []
        n = len(pillars)
        for i in range(n):
            a = pillars[i]
            b = pillars[(i + 1) % n]
            if a not in positions or b not in positions:
                continue
            ax, ay = positions[a]
            bx, by = positions[b]
            mx, my = (ax + bx) / 2.0, (ay + by) / 2.0

            best_dist = 1e18
            best_dm: Optional[float] = None
            for cx, cy, dm in dist_labels:
                if dm < 0.5 or dm > 250.0:
                    continue
                d = math.hypot(cx - mx, cy - my)
                if d < best_dist:
                    best_dist = d
                    best_dm = dm

            best_bearing = 1e18
            best_bd, best_bm = 0, 0
            for cx, cy, bd, bm in bearing_hits:
                d = math.hypot(cx - mx, cy - my)
                if d < best_bearing:
                    best_bearing = d
                    best_bd, best_bm = bd, bm

            if best_dm is None or best_bearing > 120.0:
                continue
            legs.append(
                SurveyTraverseLeg(
                    from_pillar=a,
                    to_pillar=b,
                    bearing_deg=best_bd,
                    bearing_min=best_bm,
                    distance_m=float(best_dm),
                )
            )
        doc.close()
        return _filter_plausible_legs(legs) if len(legs) >= 3 else []
    except Exception as exc:
        logger.debug("PDF boundary leg extraction failed: %s", exc)
        return []


def parse_survey_plan_extraction(data: Dict[str, Any]) -> SurveyPlanExtraction:
    pillars_raw = data.get("pillar_numbers") or data.get("pillars") or []
    pillars: List[str] = []
    if isinstance(pillars_raw, list):
        pillars = [_normalize_pillar_id(str(p)) for p in pillars_raw if str(p).strip()]
    elif isinstance(pillars_raw, str):
        pillars = [
            _normalize_pillar_id(p.strip())
            for p in re.split(r"[,\n]+", pillars_raw)
            if p.strip()
        ]

    scale = data.get("scale_denom") or data.get("scale")
    try:
        scale_i = int(scale) if scale is not None else None
    except Exception:
        scale_i = None

    area = data.get("area_sq_m") or data.get("area")
    try:
        area_f = float(area) if area is not None else None
    except Exception:
        area_f = None

    roads_raw = data.get("access_roads") or []
    roads = _coerce_access_road_specs_from_llm(roads_raw, pillars)

    fences_raw = data.get("fences") or []
    fences: List[str] = []
    if isinstance(fences_raw, list):
        fences = [str(f).strip() for f in fences_raw if str(f).strip()]
    elif isinstance(fences_raw, str) and fences_raw.strip():
        fences = [fences_raw.strip()]

    return SurveyPlanExtraction(
        buyer_name=str(data.get("buyer_name") or data.get("owners") or "").strip(),
        location=str(data.get("location") or "").strip(),
        lga=str(data.get("lga") or data.get("local_government_area") or "").strip(),
        state=str(data.get("state") or "").strip(),
        origin_crs=str(data.get("origin_crs") or data.get("crs") or data.get("origin") or "").strip(),
        plan_number=normalize_plan_number(str(data.get("plan_number") or data.get("plan_no") or "").strip()),
        surveyor_name=str(data.get("surveyor_name") or data.get("surveyor") or "").strip(),
        surveyor_address=str(data.get("surveyor_address") or data.get("surveyor_company_address") or "").strip(),
        area_sq_m=area_f,
        scale_denom=scale_i,
        pillar_numbers=pillars,
        anchor_easting=_maybe_float(data.get("anchor_easting") or data.get("easting") or data.get("e")),
        anchor_northing=_maybe_float(data.get("anchor_northing") or data.get("northing") or data.get("n")),
        anchor_pillar=_normalize_pillar_id(str(data.get("anchor_pillar") or "")),
        traverse_legs=_coerce_legs(data.get("traverse_legs") or data.get("legs")),
        access_roads=roads,
        access_road_title=str(data.get("access_road_title") or "").strip(),
        fences=fences,
        certification_date=str(data.get("certification_date") or "").strip(),
        confidence=float(data.get("confidence") or 0.0),
        source=str(data.get("source") or "llm"),
        notes=str(data.get("notes") or "").strip(),
    )


def _maybe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


_EXTRACTION_SYSTEM = """You are a licensed land surveyor reading a Nigerian cadastral/survey plan PDF.
Extract ALL visible plan details into strict JSON. Do not guess geometry that is not on the plan.

Rules:
- Read bearings as DD° MM' from North, clockwise; distances in metres.
- List traverse_legs in clockwise order around the parcel, one entry per boundary line.
- pillar_numbers: ordered list matching the traverse (clockwise from first pillar).
- If one coordinate pair is shown (easting + northing), set anchor_easting, anchor_northing, and anchor_pillar (the pillar that coordinate belongs to). If E and N appear on different grid lines at different pillars, use the pillar where both values apply or the primary/westernmost pillar with a full pair.
- Normalize pillar IDs like SC/CR 5338, SC/Q 573, SC/CK 2285, SC/BV 6015 (not SCCR5338 or SCQ573).
- Do NOT infer concrete wall fence from line work alone. Only note a fence when the plan prints an explicit label such as C.W.F., D.C.W.F., CWF, DCWF, WF, Fence, Wall Fence, or Concrete Wall Fence beside that boundary side.
- fences: array of machine-parseable specs when C.W.F./D.C.W.F. appears on the plan — one per fenced boundary side. Example: "Add Concrete wall fence on the sides joining SC/CJ 2140 and SC/CJ 2141". List every explicit fence label and its pillar pair.
- access_roads: array of machine-parseable specs — one entry per road on the plan. Example: "an access road of width 6m on the side of SC/CR 5340 and SC/CR 5341". A plan may have roads on more than one boundary side; list each separately with the TWO pillar labels for that side. Never put instructions or narrative here. Never assume the shortest traverse leg — read each road position from the drawing.
- access_road_title: copy EXACTLY as printed beside the primary/first road — typically "ACCESS ROAD" or "ACCESS CLOSE" / "ACCESS/CLOSE". When multiple roads use different titles, embed each title in its access_roads spec using: titled 'ACCESS CLOSE' of width ...
- buyer_name, location, lga, state: short title-block strings only — never bearings, coordinates, or traverse data.
- confidence: 0-1 how complete/certain the extraction is.
- If a field is missing on the plan, use "" or null — never invent.
- notes: extraction caveats only (illegible labels, missing width, etc.). Never say the replot or date change cannot be executed — that is handled downstream.

Return ONLY JSON with keys:
buyer_name, location, lga, state, origin_crs, plan_number, surveyor_name, surveyor_address,
area_sq_m, scale_denom, pillar_numbers, anchor_easting, anchor_northing, anchor_pillar,
traverse_legs (list of {from_pillar,to_pillar,bearing_deg,bearing_min,distance_m}),
access_roads, fences, certification_date, confidence, notes"""


def _preflight_heuristic_pdf_geometry(
    pdf_path: str,
    combined_text: str,
) -> SurveyPlanExtraction:
    """Repair layout heuristics with PDF geometry before deciding if vision LLM is needed."""
    heuristic = extract_heuristics_from_layout_text(combined_text)
    return repair_survey_extraction_from_pdf(heuristic, pdf_path, combined_text)


def _heuristic_pdf_extraction_is_replot_ready(
    extraction: SurveyPlanExtraction,
    combined_text: str,
    *,
    pdf_path: Optional[str] = None,
) -> bool:
    """True when repaired heuristics already satisfy safe replot validation (no vision LLM)."""
    ext = enrich_extraction_coordinates(extraction, combined_text)
    if pdf_path and Path(pdf_path).exists():
        apply_pdf_grid_coordinates(ext, pdf_path, combined_text)
        if not ext.absolute_parcel_coords:
            best_e, best_n, _, _ = extract_grid_coordinates_from_text(combined_text)
            abs_try = _compute_absolute_parcel_coordinates(
                ext,
                grid_e=best_e,
                grid_e_pillar=ext.grid_easting_pillar,
                grid_n=best_n,
                grid_n_pillar=ext.grid_northing_pillar,
            )
            if abs_try and _absolute_coords_are_plausible_utm(abs_try):
                ext.absolute_parcel_coords = abs_try
                primary_idx = _pick_primary_pillar_index(abs_try)
                ext.anchor_easting = float(abs_try[primary_idx]["e"])
                ext.anchor_northing = float(abs_try[primary_idx]["n"])
                ext.anchor_pillar = ext.pillar_numbers[primary_idx]
            else:
                if ext.anchor_easting is None:
                    ext.anchor_easting = best_e
                if ext.anchor_northing is None:
                    ext.anchor_northing = best_n
    return not validate_extraction_for_replot(ext)


def extract_survey_plan_from_pdf(
    pdf_path: str,
    *,
    llm: Any,
    run_with_timeout: Callable[..., Any],
    user_notes: str = "",
    timeout_s: int = 120,
    vision_max_pages: int = 1,
) -> SurveyPlanExtraction:
    """Extract structured survey plan data using layout text + optional vision."""
    max_pages = max(1, int(vision_max_pages or 1))
    layout_text, plain_text, images = _load_pdf_extraction_sources(
        pdf_path, vision_max_pages=max_pages
    )
    combined_text = f"{layout_text}\n{plain_text}"

    preflight = _preflight_heuristic_pdf_geometry(pdf_path, combined_text)
    if _heuristic_pdf_extraction_is_replot_ready(
        preflight, combined_text, pdf_path=pdf_path
    ):
        preflight.source = "layout_text"
        preflight.notes = f"{preflight.notes or ''} | heuristic-fast-path".strip(" |")
        logger.info("PDF extraction: heuristic-fast-path (skipping main vision LLM)")
        return finalize_survey_extraction(
            preflight,
            layout_text,
            plain_text=plain_text,
            pdf_path=pdf_path,
            llm=llm,
            run_with_timeout=run_with_timeout,
            pre_rendered_images=images,
        )

    user_prompt = (
        f"PDF path: {pdf_path}\n\n"
        f"LAYOUT-ORDERED TEXT (from PDF):\n{layout_text[:80000]}\n\n"
        f"PLAIN PDF TEXT:\n{plain_text[:40000]}\n\n"
    )
    if user_notes.strip():
        user_prompt += f"USER INSTRUCTIONS:\n{user_notes.strip()}\n\n"
    user_prompt += (
        "Extract the full traverse (all bearings and distances), metadata, and coordinates. "
        "The plan image(s) are attached when available — read labels on the drawing carefully."
    )

    from langchain_core.messages import HumanMessage, SystemMessage

    vision_images = images[:max_pages] if images else []
    if vision_images:
        content: List[dict] = [{"type": "text", "text": user_prompt}]
        for b64 in vision_images:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            )
        messages = [SystemMessage(content=_EXTRACTION_SYSTEM), HumanMessage(content=content)]
    else:
        messages = [
            SystemMessage(content=_EXTRACTION_SYSTEM),
            HumanMessage(content=user_prompt),
        ]

    heuristic = extract_heuristics_from_layout_text(combined_text)

    msg, err, timed_out = run_with_timeout(timeout_s, lambda: llm.invoke(messages))
    if timed_out:
        if heuristic.traverse_legs:
            return finalize_survey_extraction(
                heuristic,
                layout_text,
                plain_text=plain_text,
                pdf_path=pdf_path,
                llm=llm,
                run_with_timeout=run_with_timeout,
                pre_rendered_images=images,
            )
        return SurveyPlanExtraction(source="error", notes="LLM extraction timed out")
    if err:
        if heuristic.traverse_legs:
            heuristic.notes = f"LLM failed ({err}); used layout heuristics"
            return finalize_survey_extraction(
                heuristic,
                layout_text,
                plain_text=plain_text,
                pdf_path=pdf_path,
                llm=llm,
                run_with_timeout=run_with_timeout,
                pre_rendered_images=images,
            )
        return SurveyPlanExtraction(source="error", notes=f"LLM extraction failed: {err}")

    raw = msg.content if hasattr(msg, "content") else str(msg)
    if isinstance(raw, list):
        raw = "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in raw
        )
    data = _extract_json_object(str(raw))
    if not data:
        if heuristic.traverse_legs:
            heuristic.notes = "LLM JSON parse failed; used layout heuristics"
            return finalize_survey_extraction(
                heuristic,
                layout_text,
                plain_text=plain_text,
                pdf_path=pdf_path,
                llm=llm,
                run_with_timeout=run_with_timeout,
                pre_rendered_images=images,
            )
        return SurveyPlanExtraction(
            source="llm_parse_failed",
            notes=f"Could not parse JSON from model output: {str(raw)[:500]}",
        )
    ext = parse_survey_plan_extraction(data)
    ext.source = "vision" if vision_images else "layout_text"
    merged = merge_survey_extractions(ext, heuristic)
    pillars_for_legs = _filter_plausible_pillars(
        merged.pillar_numbers or heuristic.pillar_numbers
    )
    if len(pillars_for_legs) < 3 and pdf_path and Path(pdf_path).exists():
        try:
            import fitz

            doc = fitz.open(str(pdf_path))
            pdf_pillars = extract_pillars_from_pdf_page(doc[0])
            doc.close()
            if len(pdf_pillars) >= 3:
                merged.pillar_numbers = pdf_pillars
                pillars_for_legs = pdf_pillars
        except Exception as exc:
            logger.debug("Pre-finalize PDF pillar scan failed: %s", exc)
    if len(_filter_plausible_legs(merged.traverse_legs)) < 3:
        pdf_legs = extract_boundary_legs_from_pdf(pdf_path, pillars_for_legs)
        if len(pdf_legs) >= 3:
            merged.traverse_legs = pdf_legs
            if not merged.pillar_numbers and pillars_for_legs:
                merged.pillar_numbers = list(pillars_for_legs)
            merged.notes = f"{merged.notes or ''} | traverse from PDF geometry".strip(" |")
    return finalize_survey_extraction(
        merged,
        layout_text,
        plain_text=plain_text,
        pdf_path=pdf_path,
        llm=llm,
        run_with_timeout=run_with_timeout,
        pre_rendered_images=images,
    )


def _format_bearing_leg(leg: SurveyTraverseLeg) -> str:
    return (
        f"bearing {int(leg.bearing_deg)} deg {int(leg.bearing_min)} min "
        f"distance {leg.distance_m:.2f}m"
    )


def build_cadastral_subprompt(
    extraction: SurveyPlanExtraction,
    *,
    output_dwg_path: str,
    certification_date: Optional[str] = None,
    template_path: Optional[str] = None,
    combined_text: str = "",
) -> str:
    """Build a cadastral fast-path prompt from structured PDF extraction."""
    extraction = enrich_extraction_coordinates(extraction, combined_text)

    lines: List[str] = []
    if template_path:
        lines.append(f"template '{template_path}'")
    lines.append(f"Generate '{output_dwg_path}'")

    if extraction.scale_denom:
        lines.append(f"Plot using scale 1:{int(extraction.scale_denom)}")

    if extraction.buyer_name:
        lines.append(f"buyer name: {extraction.buyer_name}")
    if extraction.location:
        lines.append(f"location: {extraction.location}")
    if extraction.lga:
        lines.append(f"local government area: {extraction.lga}")
    if extraction.state:
        lines.append(f"state: {extraction.state}")
    if extraction.origin_crs:
        lines.append(f"crs_origin: {extraction.origin_crs}")
    if extraction.plan_number:
        lines.append(f"plan number: {extraction.plan_number}")
    if extraction.surveyor_name:
        lines.append(f"Surveyor name: {extraction.surveyor_name}")
    if extraction.surveyor_address:
        lines.append(f"Surveyor company and address: {extraction.surveyor_address}")

    pillars = ", ".join(extraction.pillar_numbers)
    if pillars:
        lines.append(f"pillar numbers: {pillars}")

    coord_parts: List[str] = []
    abs_coords = extraction.absolute_parcel_coords
    if abs_coords and len(abs_coords) >= 3 and _absolute_coords_are_plausible_utm(abs_coords):
        for p in abs_coords:
            coord_parts.append(f"({p['e']:.3f}mE, {p['n']:.3f}mN)")
    else:
        ae = extraction.anchor_easting
        an = extraction.anchor_northing
        if (
            ae is not None
            and an is not None
            and _is_plausible_utm_easting(ae)
            and _is_plausible_utm_northing(an)
        ):
            coord_parts.append(f"{ae:.3f}mE, {an:.3f}mN")
        for leg in extraction.traverse_legs:
            coord_parts.append(_format_bearing_leg(leg))
    if coord_parts:
        lines.append("coordinates for the points = " + "; ".join(coord_parts))

    for fence in extraction.fences:
        spec = (fence or "").strip()
        if not spec:
            continue
        if spec.lower().startswith("add "):
            lines.append(spec[0].upper() + spec[1:])
        else:
            lines.append(f"Add {spec}")

    if extraction.access_road_title and len(extraction.access_roads) <= 1:
        lines.append(f"title as '{extraction.access_road_title}'")

    for road in extraction.access_roads:
        spec = road.strip()
        if not spec:
            continue
        low = spec.lower()
        if low.startswith("add "):
            lines.append(spec[0].upper() + spec[1:] if spec else spec)
        elif low.startswith("an access"):
            lines.append(f"Add {spec}")
        elif low.startswith("access"):
            lines.append(f"Add an {spec}")
        else:
            lines.append(f"Add an access road of {spec}")

    cert = (certification_date or extraction.certification_date or "").strip()
    if cert:
        lines.append(f"date on the certification: {cert}")

    return "\n".join(lines)


_AFFIRMATION_REPLIES = frozenset(
    {"proceed", "continue", "yes", "go ahead", "do it", "ok", "okay", "yep", "yeah"}
)


def _is_affirmation_reply(text: str) -> bool:
    body = (text or "").strip().lower()
    if not body:
        return False
    if body in _AFFIRMATION_REPLIES:
        return True
    return body.startswith("proceed")


def extract_dwg_paths_from_text(text: str) -> List[str]:
    """Find DWG paths/names in a query string (does not require the file to exist)."""
    scope = _normalize_dwg_list_separators(text or "")
    found: List[str] = []
    patterns = [
        r"([A-Za-z]:\\[^\r\n\"<>|]+?\.dwg)",
        r"((?:/|\\)[^\r\n\"<>|]+?\.dwg)",
        r"(?<![A-Za-z0-9_/\\:])([A-Za-z0-9][A-Za-z0-9_.\-]*\.dwg)\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, scope, flags=re.IGNORECASE):
            raw = (m.group(1) or "").strip().strip("\"'").rstrip(").,;")
            if not raw or not _is_valid_dwg_basename(Path(raw).name):
                continue
            try:
                resolved = str(Path(raw).resolve()) if re.match(r"[A-Za-z]:\\", raw) or raw.startswith(("/", "\\")) else raw
            except Exception:
                resolved = raw
            key = resolved.lower()
            if not any(existing.lower() == key for existing in found):
                found.append(resolved)
    return _dedupe_dwg_name_variants(found)


# ---------------------------------------------------------------------------
# Multi-DWG plan extract → Word (deterministic; bypasses LangGraph agent loop)
# ---------------------------------------------------------------------------

_DWG_REPLOT_MARKERS = (
    "replot",
    "generate '",
    'generate "',
    "create '",
    'create "',
    "plot using",
    "save strictly as",
)

_DWG_METADATA_LABEL_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("buyer_name", r"buyer|owner|property\s+of|allotee|name\s+of\s+(?:owner|buyer)"),
    ("location", r"location|situate|situated\s+at"),
    ("lga", r"l\.?\s*g\.?\s*a\.?|local\s+government"),
    ("state", r"\bstate\b"),
    ("plan_number", r"plan\s*(?:no|number|#)|file\s*(?:no|number)"),
    ("surveyor_name", r"surveyor(?:\s*name)?|surveyor\s*in\s*charge"),
    ("surveyor_address", r"surveyor(?:'s)?\s*(?:company|address|firm)"),
    ("scale", r"\bscale\b"),
    ("area", r"\barea\b"),
    ("crs_origin", r"crs|coordinate\s+origin|origin\s+of\s+coordinates"),
    ("certification_date", r"certif|date\s+of\s+survey|survey\s+date"),
)

_DWG_TEXT_IMPORTANCE_PATTERNS = (
    r"\d+\s*°",
    r"SC/",
    r"pillar",
    r"bearing",
    r"access",
    r"fence",
    r"easting",
    r"northing",
    r"\barea\b",
    r"\bscale\b",
    r"\d+(?:\.\d+)?\s*m\b",
)


def should_fastpath_dwg_plan_extract_to_docx(query: str) -> bool:
    """True when the user wants plan details extracted from DWG(s) into Word (not replot)."""
    q = (query or "").lower()
    if ".dwg" not in q:
        return False
    if any(m in q for m in _DWG_REPLOT_MARKERS):
        return False
    wants_extract = any(
        k in q
        for k in (
            "extract",
            "important details",
            "plan details",
            "details in the plan",
            "key details",
            "title block",
            "metadata",
        )
    )
    wants_word = any(
        k in q
        for k in (
            "word",
            "docx",
            "microsoft word",
            ".doc",
            "document",
        )
    )
    return bool(wants_extract and wants_word)


_PDF_KEY_DETAIL_EXTRACT_MARKERS = (
    "extract",
    "key details",
    "important details",
    "plan details",
    "details in the plan",
    "title block",
    "metadata",
    "read the plan",
    "survey plan details",
    "all the details",
    "all details",
    "what does the plan",
    "summarise the plan",
    "summarize the plan",
)

_PDF_KEY_DETAIL_PLAN_MARKERS = (
    "survey",
    "cadastral",
    "site plan",
    "deed",
    "pillar",
    "bearing",
    "parcel",
    "landed property",
    "plan shewing",
    "plan showing",
)

_PDF_KEY_DETAIL_EXCLUDE = (
    "replot",
    "plot using",
    "arcgis",
    "arcpy",
    "cutfill",
    "cut fill",
    "geodatabase",
)


def should_fastpath_pdf_plan_key_details(query: str) -> bool:
    """
    True when the user wants key survey-plan details from a PDF (not a CAD replot).

    Scanned/image PDFs often have no text layer — those must use layout+vision
    extraction rather than document_get_text alone.
    """
    q = (query or "").lower().strip()
    if ".pdf" not in q:
        return False
    if any(m in q for m in _PDF_KEY_DETAIL_EXCLUDE):
        return False
    # Replot-to-DWG is handled by the PDF→CAD fast path.
    if ".dwg" in q and any(k in q for k in ("replot", "generate", "create", "draw", "plot")):
        return False
    wants_extract = any(k in q for k in _PDF_KEY_DETAIL_EXTRACT_MARKERS)
    if not wants_extract:
        return False
    looks_like_plan = any(k in q for k in _PDF_KEY_DETAIL_PLAN_MARKERS) or bool(
        re.search(
            r"[\\/][^\\/\"']*(?:plan|survey|cadastral|deed|site)[^\\/\"']*\.pdf",
            q,
            flags=re.IGNORECASE,
        )
    )
    # "extract all the key details" + a .pdf is enough for SurvyAI's survey workflow
    # when the filename or wording already signals a plan document.
    return bool(looks_like_plan or "key details" in q or "important details" in q or "plan details" in q)


def extraction_has_usable_key_details(plan: SurveyPlanExtraction) -> bool:
    """True when vision/layout extraction produced at least one verified plan field."""
    if plan is None:
        return False
    if plan.buyer_name or plan.location or plan.plan_number or plan.surveyor_name:
        return True
    if plan.lga or plan.state or plan.certification_date or plan.origin_crs:
        return True
    if plan.scale_denom or plan.area_sq_m is not None:
        return True
    if plan.pillar_numbers or plan.traverse_legs:
        return True
    if plan.anchor_easting is not None and plan.anchor_northing is not None:
        return True
    if plan.access_roads or plan.fences:
        return True
    return False


def format_survey_plan_key_details_report(
    plan: SurveyPlanExtraction,
    pdf_path: str,
) -> str:
    """Format structured PDF survey extraction as a surveyor-facing key-details report."""
    path = Path(pdf_path)
    section = survey_plan_to_word_section(
        plan,
        source_file=str(path),
        plan_label=path.stem or "Survey plan",
    )
    lines: List[str] = [
        f"## Key details from `{path.name}`",
        "",
        f"**Source:** `{path}`",
        "",
        "### Title block / metadata",
    ]
    content_lines = [
        str(x).strip()
        for x in (section.get("content") or [])
        if str(x).strip() and not str(x).strip().lower().startswith("source file:")
    ]
    if content_lines:
        for item in content_lines:
            lines.append(f"- {item}")
    else:
        lines.append("- (No title-block fields verified)")

    table = section.get("table")
    if isinstance(table, list) and len(table) > 1:
        lines.append("")
        lines.append("### Traverse (bearings & distances)")
        lines.append("")
        lines.append("| From | To | Bearing | Distance (m) |")
        lines.append("|---|---|---|---|")
        for row in table[1:]:
            if not isinstance(row, (list, tuple)) or len(row) < 4:
                continue
            lines.append(
                f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |"
            )

    src = (plan.source or "vision").strip() or "vision"
    conf = float(plan.confidence or 0.0)
    lines.append("")
    lines.append(
        f"_Extraction method: {src}"
        + (f"; confidence ≈ {conf:.0%}" if conf > 0 else "")
        + ". Scanned/image PDFs are read with layout text + vision — "
        "not plain text extraction alone._"
    )
    lines.append("")
    lines.append(
        "If you want, I can **replot this plan to a DWG**, or **save these details to a Word document**."
    )
    return "\n".join(lines)


def run_pdf_plan_key_details_extract(
    *,
    query: str,
    extract_fn: Callable[..., Any],
    full_query: str = "",
) -> Dict[str, Any]:
    """
    Resolve a survey-plan PDF and extract key details via layout + vision.

    ``extract_fn`` should match agent ``_extract_pdf_survey_plan_with_tier_fallback``
    signature: ``(pdf_path, *, user_notes, timeout_s) -> (SurveyPlanExtraction, model_name)``.
    """
    scope = (query or "").strip()
    full = (full_query or scope).strip()
    resolution = resolve_pdf_path_for_replot(scope, full)
    if not resolution.get("success"):
        err = str(resolution.get("error") or "Could not resolve the PDF path.")
        return {
            "success": False,
            "error": err,
            "response": err,
            "needs_user_approval": bool(resolution.get("needs_user_approval")),
            "similar": resolution.get("similar") or [],
        }

    pdf_path = str(resolution["path"])
    try:
        extraction, model_name = extract_fn(
            pdf_path,
            user_notes=scope,
            timeout_s=120,
        )
    except TypeError:
        # Allow simpler callables used in unit tests.
        extraction, model_name = extract_fn(pdf_path)
    except Exception as exc:
        logger.exception("PDF key-details extraction failed for %s", pdf_path)
        return {
            "success": False,
            "error": str(exc),
            "response": (
                f"I found `{pdf_path}` but vision/layout extraction failed:\n{exc}\n\n"
                "Try again, or provide a DWG/DXF or a text-readable PDF if available."
            ),
            "pdf_path": pdf_path,
        }

    if not isinstance(extraction, SurveyPlanExtraction):
        return {
            "success": False,
            "error": "Invalid extraction result",
            "response": f"Extraction returned an unexpected result for `{pdf_path}`.",
            "pdf_path": pdf_path,
            "model_name": model_name,
        }

    if not extraction_has_usable_key_details(extraction):
        notes = (extraction.notes or "").strip()
        return {
            "success": False,
            "error": notes or "no_usable_key_details",
            "response": (
                f"I inspected `{pdf_path}` with **layout text + vision** "
                "(scanned survey-plan route), but could not verify usable key fields "
                "(owner, location, plan number, bearings, coordinates, etc.).\n\n"
                + (f"Notes: {notes}\n\n" if notes else "")
                + "If you have a DWG/DXF of the same plan, I can extract from that instead."
            ),
            "pdf_path": pdf_path,
            "model_name": model_name,
            "extraction": extraction,
        }

    return {
        "success": True,
        "response": format_survey_plan_key_details_report(extraction, pdf_path),
        "pdf_path": pdf_path,
        "model_name": model_name,
        "extraction": extraction,
    }


def _normalize_dwg_list_separators(text: str) -> str:
    """Turn '&' list separators into commas so each .dwg is parsed individually."""
    t = text or ""
    t = re.sub(r"\s*&\s*", ", ", t)
    return t


def _is_valid_dwg_basename(name: str) -> bool:
    """Reject merged filenames (e.g. 'a.dwg & b.dwg') and other parse artifacts."""
    if not name or not name.lower().endswith(".dwg"):
        return False
    if any(ch in name for ch in "&|,;"):
        return False
    if name.lower().count(".dwg") > 1:
        return False
    stem = Path(name).stem
    if not stem or len(stem) > 200:
        return False
    return True


def _dedupe_dwg_name_variants(names: Sequence[str]) -> List[str]:
    """
    Drop shorter basename variants when a longer one ends with the same stem
    (e.g. keep MR.IKECHUKWU_OLEKA.dwg, drop IKECHUKWU_OLEKA.dwg).
    """
    basenames = [Path(n).name for n in names]
    keep: List[str] = []
    for i, name in enumerate(names):
        base = basenames[i].lower()
        stem = Path(base).stem.lower()
        dominated = False
        for j, other_base in enumerate(basenames):
            if i == j:
                continue
            other_stem = Path(other_base).stem.lower()
            if stem != other_stem and other_stem.endswith(stem):
                dominated = True
                break
        if not dominated:
            keep.append(name)
    return keep


def _dwg_path_resolution_candidates(path: Path) -> List[Path]:
    """When an absolute path is missing, try the current Windows user profile folder."""
    candidates = [path]
    if not path.is_absolute():
        return candidates
    try:
        import os

        user = (os.environ.get("USERNAME") or os.environ.get("USER") or "").strip()
        parts = path.parts
        if user and len(parts) >= 3 and parts[0].endswith(":") and parts[1].lower() == "users":
            profile = parts[2]
            if profile.lower() != user.lower():
                alt = Path(parts[0]) / "Users" / user / Path(*parts[3:])
                candidates.append(alt)
    except Exception:
        pass
    return candidates


def _bare_dwg_names(text: str) -> List[str]:
    scope = _normalize_dwg_list_separators(text or "")
    found: List[str] = []
    for m in re.finditer(
        r"(?<![A-Za-z0-9_\\:/])([A-Za-z0-9][A-Za-z0-9_.\-]*\.dwg)\b",
        scope,
        flags=re.IGNORECASE,
    ):
        name = (m.group(1) or "").strip().strip("\"'").rstrip(").,;")
        if name and _is_valid_dwg_basename(name) and name.lower() not in {x.lower() for x in found}:
            found.append(name)
    return _dedupe_dwg_name_variants(found)


def resolve_dwg_paths_from_query(query: str) -> List[str]:
    """
    Resolve every DWG referenced in the query.

    Full paths are kept; bare filenames are resolved against the folder of the
    first absolute path, or CWD when none is given.
    """
    scope = _normalize_dwg_list_separators(query or "")
    resolved: List[str] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        for candidate in _dwg_path_resolution_candidates(path):
            if not _is_valid_dwg_basename(candidate.name):
                continue
            if candidate.exists():
                key = str(candidate.resolve()).lower()
                if key not in seen:
                    seen.add(key)
                    resolved.append(str(candidate.resolve()))
                return
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            resolved.append(str(path.resolve()))

    anchor: Optional[Path] = None
    for raw in extract_dwg_paths_from_text(scope):
        p = Path(raw)
        if p.is_absolute():
            anchor = anchor or p.parent
            _add(p)
        else:
            if not _is_valid_dwg_basename(p.name):
                continue
            candidate = Path.cwd() / p.name
            if candidate.exists():
                _add(candidate)
                anchor = anchor or candidate.parent

    if anchor is None:
        for raw in extract_dwg_paths_from_text(scope):
            p = Path(raw)
            if p.is_absolute():
                anchor = p.parent
                break

    folder = anchor or Path.cwd()
    for name in _bare_dwg_names(scope):
        if not _is_valid_dwg_basename(name):
            continue
        candidate = folder / name
        if any(Path(x).name.lower() == name.lower() for x in resolved):
            continue
        _add(candidate)

    return resolved


def resolve_dwg_extract_output_docx_path(query: str, workspace: Optional[Path] = None) -> Path:
    """Infer the output .docx path from a DWG-extract-to-Word request."""
    ws = (workspace or Path.cwd()).resolve()
    q = query or ""

    patterns = (
        r"word\s+document\s+['\"]([^'\"]+)['\"]",
        r"word\s+document\s+([A-Za-z0-9_\- ]+)",
        r"save\s+(?:it\s+)?(?:in|as)\s+['\"]([^'\"]+)['\"]",
        r"['\"]([^'\"]+\.docx)['\"]",
        r"\b([A-Za-z0-9_\-]+\.docx)\b",
        r"\b(Plan_details[_\-]?Extract)\b",
    )
    for pat in patterns:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if not m:
            continue
        name = (m.group(1) or "").strip().strip("\"'").rstrip(").,;")
        if not name:
            continue
        p = Path(name)
        if p.is_absolute():
            return p if p.suffix.lower() == ".docx" else p.with_suffix(".docx")
        stem = p.stem if p.suffix else name
        return ws / f"{stem}.docx"

    return ws / "Plan_details_Extract.docx"


def _dwg_table_cell_value(row: Sequence[str], col_idx: int) -> str:
    if col_idx + 1 < len(row):
        val = (row[col_idx + 1] or "").strip()
        if val:
            return val
    return ""


def parse_dwg_metadata_from_tables(tables: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """Scan TABLE grids for common Nigerian cadastral title-block labels."""
    metadata: Dict[str, str] = {}
    for table in tables or []:
        grid = table.get("grid") or []
        for row in grid:
            if not row:
                continue
            for col_idx, cell in enumerate(row):
                label = (cell or "").strip()
                if not label:
                    continue
                for key, pattern in _DWG_METADATA_LABEL_PATTERNS:
                    if key in metadata:
                        continue
                    if re.search(pattern, label, flags=re.IGNORECASE):
                        val = _dwg_table_cell_value(row, col_idx)
                        if not val and col_idx > 0:
                            val = (row[col_idx - 1] or "").strip()
                        if val and not re.search(pattern, val, flags=re.IGNORECASE):
                            metadata[key] = val
    return metadata


_DWG_EXTRACTION_SYSTEM = _EXTRACTION_SYSTEM.replace(
    "reading a Nigerian cadastral/survey plan PDF",
    "reading a Nigerian cadastral/survey plan DWG (AutoCAD drawing)",
).replace(
    "The plan image(s) are attached when available — read labels on the drawing carefully.",
    "Use the cleaned layout text and table metadata only — ignore AutoCAD formatting codes.",
)


def clean_autocad_mtext(raw: str) -> str:
    """Strip MTEXT/MText control codes and normalize bearings (%%D -> °)."""
    if not raw:
        return ""
    s = str(raw)
    s = s.replace("\\P", "\n").replace("\\p", "\n")
    s = re.sub(r"%%D", "°", s, flags=re.IGNORECASE)
    s = re.sub(r"%%P", "±", s, flags=re.IGNORECASE)
    s = re.sub(r"\\f[^;\\]*;", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\\H[^;\\]*;", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\\C[^;\\]*;", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\\[A-Za-z]", " ", s)
    s = s.replace("{", " ").replace("}", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()


def build_dwg_layout_text_from_autocad(
    texts: Sequence[Dict[str, Any]],
    tables: Sequence[Dict[str, Any]],
) -> str:
    """Combine cleaned MTEXT and TABLE cells into layout-ordered plain text."""
    parts: List[str] = []
    for item in texts or []:
        cleaned = clean_autocad_mtext((item.get("content") or "").strip())
        if cleaned and len(cleaned) >= 2:
            parts.append(cleaned)
    for table in tables or []:
        for row in table.get("grid") or []:
            cells = [clean_autocad_mtext(str(c or "")) for c in row]
            cells = [c for c in cells if c]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _extract_area_sq_m_from_dwg_text(text: str) -> Optional[float]:
    for pat in (
        r"AREA\s*[:-]\s*([\d,]+(?:\.\d+)?)\s*SQ",
        r"([\d,]+(?:\.\d+)?)\s*SQ\.?\s*M(?:TRS|ETERS)?\b",
        r"([\d,]+(?:\.\d+)?)\s*m²",
    ):
        m = re.search(pat, text or "", flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except Exception:
                continue
    return None


def _dwg_title_block_candidates(layout_text: str) -> List[str]:
    text = layout_text or ""
    if "PLAN SHEWING" not in text.upper():
        return [text] if text.strip() else []
    parts = re.split(
        r"(?=PLAN\s+SHEWING\s+LANDED\s+PROPERTY)",
        text,
        flags=re.IGNORECASE,
    )
    blocks = [p.strip() for p in parts if p.strip() and "PLAN SHEWING" in p.upper()]
    return blocks or [text]


def _score_dwg_title_block(
    block: str,
    *,
    file_stem: str,
    title_area: Optional[float],
) -> float:
    score = 0.0
    if re.search(r"\.{4,}", block):
        score -= 8.0
    buyer_m = re.search(
        r"PLAN\s+SHEWING\s+LANDED\s+PROPERTY\s*(?:OF)?\s*\n?\s*(.+?)\s*\n\s*AT\b",
        block,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if buyer_m:
        buyer = re.sub(r"\s+", " ", buyer_m.group(1)).strip().upper()
        stem_norm = re.sub(r"[._]+", " ", file_stem).upper()
        if stem_norm and (stem_norm in buyer or buyer in stem_norm):
            score += 12.0
        for tok in stem_norm.split():
            if len(tok) > 3 and tok in buyer:
                score += 2.0
    block_area = _extract_area_sq_m_from_dwg_text(block)
    if title_area and block_area:
        rel = abs(block_area - title_area) / max(title_area, 1.0)
        if rel < 0.02:
            score += 10.0
        elif rel < 0.15:
            score += 4.0
    elif block_area and 20.0 <= block_area <= 100000.0:
        score += 2.0
    return score


def _pick_best_dwg_title_block(
    layout_text: str,
    *,
    file_stem: str,
    title_area: Optional[float],
) -> str:
    candidates = _dwg_title_block_candidates(layout_text)
    if len(candidates) == 1:
        return candidates[0]
    return max(
        candidates,
        key=lambda b: _score_dwg_title_block(b, file_stem=file_stem, title_area=title_area),
    )


def parse_dwg_title_block_fields(text: str) -> Dict[str, str]:
    """Parse Nigerian cadastral title-block fields from cleaned DWG MTEXT."""
    raw = text or ""
    fields: Dict[str, str] = {}

    buyer_m = re.search(
        r"PLAN\s+SHEWING\s+LANDED\s+PROPERTY\s*(?:OF)?\s*\n?\s*(.+?)\s*\n\s*AT\b",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if buyer_m:
        fields["buyer_name"] = sanitize_metadata_field(
            re.sub(r"\s+", " ", buyer_m.group(1)).strip(" :-")
        )

    loc_m = re.search(
        r"\bAT\s*\n?\s*(.+?)\s*\n\s*(.+?)\s+(?:LOCAL\s+GOVERNMENT\s+AREA|LOCAL\s+GOVT\.?\s*AREA|"
        r"L\.?\s*G\.?\s*A\.?|LGA)\b",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if loc_m:
        loc_parts = [
            re.sub(r"\s+", " ", loc_m.group(1)).strip(" ,;:-"),
            re.sub(r"\s+", " ", loc_m.group(2)).strip(" ,;:-"),
        ]
        loc_parts = [
            p
            for p in loc_parts
            if p and not re.search(r"\.{3,}", p) and not _LGA_LINE_RE.fullmatch(p)
        ]
        # Drop trailing LGA glued onto last clause
        loc_parts = [_LGA_TRAILING_RE.sub("", p).strip(" ,;:-") for p in loc_parts]
        loc_parts = [p for p in loc_parts if p]
        if loc_parts:
            fields["location"] = sanitize_metadata_field(", ".join(loc_parts), max_len=200)
    else:
        # Fallback: use shared AT…until-LGA extractor
        loc_fb = extract_location_from_text(raw)
        if loc_fb:
            fields["location"] = sanitize_metadata_field(loc_fb, max_len=200)

    lga_m = re.search(
        r"(.+?)\s+(?:LOCAL\s+GOVERNMENT\s+AREA|LOCAL\s+GOVT\.?\s*AREA|L\.?\s*G\.?\s*A\.?|LGA)\b",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if lga_m:
        lga = re.sub(r"\s+", " ", lga_m.group(1)).strip(" ,;:-")
        lga = re.split(r"\n", lga)[-1].strip()
        if lga and not re.search(r"\.{3,}", lga):
            # Store bare LGA name only; CAD template attaches "LOCAL GOVERNMENT AREA".
            bare = normalize_lga_name(lga)
            if bare:
                fields["lga"] = sanitize_metadata_field(bare, max_len=80)

    state_m = re.search(
        r"(.+?)\s+STATE\s*,\s*NIGERIA",
        raw,
        flags=re.IGNORECASE,
    )
    if state_m:
        state = re.sub(r"\s+", " ", state_m.group(1)).strip(" ,;:-")
        state = re.split(r"\n", state)[-1].strip()
        if state and not re.search(r"\.{3,}", state):
            fields["state"] = sanitize_metadata_field(state, max_len=40)

    origin_m = re.search(r"ORIGIN\s*[:-]\s*(.+?)(?:\n|AREA\b|SCALE\b|$)", raw, flags=re.IGNORECASE)
    if origin_m:
        fields["origin_crs"] = sanitize_metadata_field(origin_m.group(1).strip(), max_len=60)

    area = _extract_area_sq_m_from_dwg_text(raw)
    if area is not None:
        fields["area_sq_m"] = str(area)

    scale = extract_scale_denom_from_text(raw)
    if scale:
        fields["scale_denom"] = str(scale)

    cert_m = re.search(
        r"(?:DATE\s+OF\s+CERTIF(?:ICATION)?|CERTIF(?:ICATION)?\s+DATE)\s*[:-]?\s*(.+?)(?:\n|$)",
        raw,
        flags=re.IGNORECASE,
    )
    if cert_m:
        fields["certification_date"] = sanitize_metadata_field(cert_m.group(1).strip(), max_len=40)

    return fields


def _merge_dwg_table_metadata_into_plan(
    plan: SurveyPlanExtraction,
    table_meta: Dict[str, str],
) -> SurveyPlanExtraction:
    data = plan.model_dump()
    mapping = {
        "buyer_name": "buyer_name",
        "location": "location",
        "lga": "lga",
        "state": "state",
        "plan_number": "plan_number",
        "surveyor_name": "surveyor_name",
        "surveyor_address": "surveyor_address",
        "scale": "scale_denom",
        "crs_origin": "origin_crs",
        "certification_date": "certification_date",
        "area": "area_sq_m",
    }
    for src, dst in mapping.items():
        val = (table_meta or {}).get(src)
        if not val:
            continue
        cur = data.get(dst)
        if cur in (None, "", [], 0.0):
            if dst == "scale_denom":
                try:
                    data[dst] = int(str(val).split(":")[-1].strip())
                except Exception:
                    pass
            elif dst == "area_sq_m":
                try:
                    data[dst] = float(str(val).replace(",", ""))
                except Exception:
                    pass
            else:
                data[dst] = val
    return SurveyPlanExtraction(**data)


def extract_survey_plan_from_dwg_layout(
    layout_text: str,
    *,
    file_stem: str,
    table_meta: Optional[Dict[str, str]] = None,
    measured_area: Optional[Dict[str, Any]] = None,
) -> SurveyPlanExtraction:
    """Heuristic + title-block parse for a single DWG."""
    title_area = _extract_area_sq_m_from_dwg_text(layout_text)
    if title_area is None and measured_area:
        try:
            title_area = float(measured_area.get("sq_m")) if measured_area.get("sq_m") else None
        except Exception:
            title_area = None

    focus = _pick_best_dwg_title_block(
        layout_text,
        file_stem=file_stem,
        title_area=title_area,
    )
    heuristic = extract_heuristics_from_layout_text(f"{focus}\n{layout_text}")
    title_fields = parse_dwg_title_block_fields(focus)

    data = heuristic.model_dump()
    for key, val in title_fields.items():
        if key == "area_sq_m":
            try:
                data["area_sq_m"] = float(val)
            except Exception:
                pass
        elif key == "scale_denom":
            try:
                data["scale_denom"] = int(val)
            except Exception:
                pass
        elif key == "location" and val:
            # Prefer fuller multi-clause AT location over a short first-line heuristic.
            cur = str(data.get("location") or "").strip()
            cand = str(val).strip()
            if not cur or (len(cand) > len(cur) + 3) or ("," in cand and "," not in cur):
                data["location"] = cand
        elif val and not data.get(key):
            data[key] = val

    plan = SurveyPlanExtraction(**data)
    plan = _merge_dwg_table_metadata_into_plan(plan, table_meta or {})
    plan.source = "dwg_heuristic"

    if plan.area_sq_m is None and title_area is not None:
        plan.area_sq_m = title_area
    elif plan.area_sq_m is None and measured_area:
        try:
            plan.area_sq_m = float(measured_area.get("sq_m"))
        except Exception:
            pass

    plan = prepare_extraction_for_cadastral(plan, f"{focus}\n{layout_text}")
    return finalize_survey_extraction(plan, focus)


def extract_survey_plan_from_dwg_with_llm(
    layout_text: str,
    *,
    file_stem: str,
    table_meta: Optional[Dict[str, str]] = None,
    measured_area: Optional[Dict[str, Any]] = None,
    llm: Any = None,
    run_with_timeout: Optional[Callable[..., Any]] = None,
    field_context: str = "",
    timeout_s: int = 120,
) -> SurveyPlanExtraction:
    """Layout heuristics + optional LLM refinement for DWG survey plans."""
    base = extract_survey_plan_from_dwg_layout(
        layout_text,
        file_stem=file_stem,
        table_meta=table_meta,
        measured_area=measured_area,
    )
    if llm is None or run_with_timeout is None:
        return base

    from langchain_core.messages import HumanMessage, SystemMessage

    user_prompt = (
        f"DWG file stem: {file_stem}\n\n"
        f"{field_context}\n\n"
        f"CLEANED LAYOUT TEXT:\n{layout_text[:60000]}\n\n"
        f"TABLE METADATA (from AutoCAD TABLE objects):\n{json.dumps(table_meta or {}, indent=2)}\n\n"
        f"PRELIMINARY HEURISTIC EXTRACTION:\n{base.model_dump_json()}\n\n"
        "Return refined JSON with all standard cadastral fields. "
        "Use title-block text for buyer, location, LGA, state, area, scale, CRS, surveyor, plan number, "
        "and certification date. List traverse legs clockwise with bearings (DD° MM') and distances in metres."
    )
    messages = [
        SystemMessage(content=_DWG_EXTRACTION_SYSTEM),
        HumanMessage(content=user_prompt),
    ]
    msg, err, timed_out = run_with_timeout(timeout_s, lambda: llm.invoke(messages))
    if timed_out or err:
        note = "LLM refinement timed out" if timed_out else f"LLM refinement failed: {err}"
        base.notes = f"{base.notes or ''} | {note}".strip(" |")
        return base

    raw = msg.content if hasattr(msg, "content") else str(msg)
    if isinstance(raw, list):
        raw = "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in raw
        )
    data = _extract_json_object(str(raw))
    if not data:
        base.notes = f"{base.notes or ''} | LLM JSON parse failed; kept heuristics".strip(" |")
        return base

    llm_plan = parse_survey_plan_extraction(data)
    merged = merge_survey_extractions(llm_plan, base)
    merged.source = "dwg_llm"
    merged = prepare_extraction_for_cadastral(merged, layout_text)
    return finalize_survey_extraction(merged, layout_text)


def survey_plan_to_word_section(
    plan: SurveyPlanExtraction,
    *,
    source_file: str,
    plan_label: str,
) -> Dict[str, Any]:
    """Build a professional Word section from structured survey plan fields."""
    lines: List[str] = [f"Source file: {source_file}"]

    def _add(label: str, value: Any) -> None:
        if value in (None, "", [], 0.0):
            return
        lines.append(f"{label}: {value}")

    _add("Buyer / Owner", plan.buyer_name)
    _add("Location", plan.location)
    _add("Local Government Area (LGA)", plan.lga)
    _add("State", plan.state)
    _add("Surveyor's Name", plan.surveyor_name)
    _add("Surveyor's Address / Company", plan.surveyor_address)
    _add("Plan Number", plan.plan_number)
    _add("Scale", f"1:{plan.scale_denom}" if plan.scale_denom else "")
    _add("Coordinate Origin / CRS", plan.origin_crs)

    if plan.anchor_easting is not None and plan.anchor_northing is not None:
        coord = f"E {plan.anchor_easting:.3f} m, N {plan.anchor_northing:.3f} m"
        if plan.anchor_pillar:
            coord += f" (at {plan.anchor_pillar})"
        _add("Coordinates", coord)

    if plan.area_sq_m is not None:
        hectares = plan.area_sq_m / 10000.0
        _add("Area", f"{plan.area_sq_m:,.2f} sq m ({hectares:.4f} hectares)")

    if plan.pillar_numbers:
        _add("Pillar / Control Numbers", ", ".join(plan.pillar_numbers))

    if plan.access_roads:
        _add("Access Road(s)", "; ".join(plan.access_roads))
    elif plan.access_road_title:
        _add("Access Road Title", plan.access_road_title)

    if plan.fences:
        _add("Fence Specification(s)", "; ".join(plan.fences))

    _add("Date of Certification", plan.certification_date)

    if plan.notes:
        _add("Extraction Notes", plan.notes)

    section: Dict[str, Any] = {
        "heading": plan_label,
        "level": 1,
        "content": lines,
    }

    if plan.traverse_legs:
        table: List[List[str]] = [["From Pillar", "To Pillar", "Bearing", "Distance (m)"]]
        for leg in plan.traverse_legs:
            bearing = f"{leg.bearing_deg}° {leg.bearing_min:02d}'"
            table.append([
                leg.from_pillar or "—",
                leg.to_pillar or "—",
                bearing,
                f"{leg.distance_m:.2f}" if leg.distance_m is not None else "—",
            ])
        section["table"] = table

    return section


def build_dwg_word_sections_from_extractions(
    extractions: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Turn per-DWG structured survey plans into Word sections."""
    sections: List[Dict[str, Any]] = []
    for item in extractions:
        plan = item.get("survey_plan")
        if isinstance(plan, SurveyPlanExtraction):
            sections.append(
                survey_plan_to_word_section(
                    plan,
                    source_file=str(item.get("file") or ""),
                    plan_label=str(item.get("name") or Path(str(item.get("file") or "plan")).stem),
                )
            )
            continue
        heading = item.get("name") or Path(item.get("file", "plan")).stem
        errors = item.get("errors") or []
        sections.append({
            "heading": heading,
            "level": 1,
            "content": [
                f"Source file: {item.get('file', '')}",
                "Structured extraction unavailable.",
                *(["Notes: " + "; ".join(errors)] if errors else []),
            ],
        })
    return sections


def extract_plan_details_from_open_dwg(
    autocad: Any,
    *,
    file_path: str,
    llm: Any = None,
    run_with_timeout: Optional[Callable[..., Any]] = None,
    field_context: str = "",
) -> Dict[str, Any]:
    """Read structured survey metadata from the drawing active in AutoCAD."""
    stem = Path(file_path).stem
    out: Dict[str, Any] = {
        "file": str(Path(file_path).resolve()),
        "name": stem,
        "errors": [],
    }

    tables: List[Dict[str, Any]] = []
    table_meta: Dict[str, str] = {}
    tables_result = autocad.dump_all_tables()
    if tables_result.get("success"):
        tables = tables_result.get("tables") or []
        table_meta = parse_dwg_metadata_from_tables(tables)
    else:
        out["errors"].append(tables_result.get("error") or "dump_all_tables failed")

    measured_area: Dict[str, Any] = {}
    area_result = autocad.calculate_boundary_area()
    if area_result.get("success"):
        measured_area = {
            "sq_m": area_result.get("area_sq_m") or area_result.get("area"),
            "hectares": area_result.get("area_hectares"),
            "strategy": area_result.get("strategy_used"),
        }
    else:
        out["errors"].append(area_result.get("error") or "calculate_boundary_area failed")

    text_entities: List[Dict[str, Any]] = []
    text_result = autocad.get_all_text()
    if text_result.get("success"):
        text_entities = text_result.get("texts") or []
    else:
        out["errors"].append(text_result.get("error") or "get_all_text failed")

    layout_text = build_dwg_layout_text_from_autocad(text_entities, tables)
    if not layout_text.strip():
        out["errors"].append("No readable text found in drawing")
        out["survey_plan"] = SurveyPlanExtraction(source="error", notes="No readable text in DWG")
        return out

    try:
        plan = extract_survey_plan_from_dwg_with_llm(
            layout_text,
            file_stem=stem,
            table_meta=table_meta,
            measured_area=measured_area,
            llm=llm,
            run_with_timeout=run_with_timeout,
            field_context=field_context,
        )
        out["survey_plan"] = plan
    except Exception as exc:
        logger.exception("DWG survey plan extraction failed for %s", file_path)
        out["errors"].append(str(exc))
        out["survey_plan"] = SurveyPlanExtraction(source="error", notes=str(exc))

    return out


def extract_plan_details_for_dwg(
    autocad: Any,
    dxf_fallback: Any,
    file_path: str,
    *,
    llm: Any = None,
    run_with_timeout: Optional[Callable[..., Any]] = None,
    field_context: str = "",
) -> Dict[str, Any]:
    """Open a DWG (COM or ezdxf fallback) and extract structured plan details.

    Uses resilient AutoCAD open (retry + COM recover) so reference-plan metadata
    is not lost to intermittent ``Call was rejected by callee`` errors while
    AutoCAD is busy/modal. Falls back to ezdxf+ODA only when COM cannot open.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        return {
            "file": str(path),
            "name": path.stem,
            "survey_plan": SurveyPlanExtraction(source="error", notes=f"File not found: {path}"),
            "errors": [f"File not found: {path}"],
        }

    # Warm COM on this thread before the open (critical for Excel→CAD metadata).
    try:
        if autocad is not None and hasattr(autocad, "connect") and not getattr(
            autocad, "is_connected", False
        ):
            autocad.connect()
    except Exception as conn_exc:
        logger.debug("AutoCAD pre-connect before reference open failed: %s", conn_exc)

    opened: Dict[str, Any] = {"success": False, "error": "open not attempted"}
    try:
        if autocad is not None and hasattr(autocad, "open_drawing_resilient"):
            opened = autocad.open_drawing_resilient(
                str(path), read_only=True, attempts=3
            )
        elif autocad is not None and hasattr(autocad, "open_drawing"):
            opened = autocad.open_drawing(str(path), read_only=True)
            if not opened.get("success") and hasattr(autocad, "recover_com_session"):
                try:
                    autocad.recover_com_session(
                        force=True, reason="extract_plan_details_for_dwg retry"
                    )
                except Exception:
                    pass
                opened = autocad.open_drawing(str(path), read_only=False)
    except Exception as open_exc:
        opened = {"success": False, "error": str(open_exc)}

    if opened.get("success"):
        try:
            details = extract_plan_details_from_open_dwg(
                autocad,
                file_path=str(path),
                llm=llm,
                run_with_timeout=run_with_timeout,
                field_context=field_context,
            )
        finally:
            # Release the reference tab so subsequent template/plot opens are not blocked.
            try:
                if autocad is not None and hasattr(autocad, "close_drawing_if_open"):
                    autocad.close_drawing_if_open(str(path), save_changes=False)
            except Exception:
                pass
        return details

    err = opened.get("error") or "open_drawing failed"
    if dxf_fallback and getattr(dxf_fallback, "is_available", False):
        try:
            fb = dxf_fallback.open_drawing(str(path))
            if fb.get("success"):
                texts = dxf_fallback.get_all_text() if hasattr(dxf_fallback, "get_all_text") else {}
                area = (
                    dxf_fallback.calculate_area()
                    if hasattr(dxf_fallback, "calculate_area")
                    else {}
                )
                layout_text = build_dwg_layout_text_from_autocad((texts or {}).get("texts") or [], [])
                measured = {
                    "sq_m": area.get("area_sq_m") or area.get("area"),
                    "hectares": area.get("area_hectares"),
                }
                plan = extract_survey_plan_from_dwg_with_llm(
                    layout_text,
                    file_stem=path.stem,
                    table_meta={},
                    measured_area=measured,
                    llm=llm,
                    run_with_timeout=run_with_timeout,
                    field_context=field_context,
                )
                return {
                    "file": str(path),
                    "name": path.stem,
                    "survey_plan": plan,
                    "errors": [
                        "AutoCAD COM unavailable; used ezdxf text extraction (TABLE cells not supported)."
                    ],
                }
        except Exception as exc:
            err = f"{err}; ezdxf fallback failed: {exc}"

    return {
        "file": str(path),
        "name": path.stem,
        "survey_plan": SurveyPlanExtraction(source="error", notes=err),
        "errors": [err],
    }


def _dwg_sections_to_word_payload(sections: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize sections for document_processor.create_word_document_from_structure."""
    payload: List[Dict[str, Any]] = []
    for sec in sections:
        entry: Dict[str, Any] = {
            "heading": sec.get("heading", ""),
            "level": sec.get("level", 1),
            "content": sec.get("content", ""),
        }
        if sec.get("table"):
            entry["table"] = sec["table"]
        elif sec.get("annotations_table"):
            entry["table"] = sec["annotations_table"]
        payload.append(entry)
    return payload


def run_dwg_plan_extract_to_docx(
    *,
    query: str,
    autocad: Any,
    dxf_fallback: Any,
    document_processor: Any,
    workspace: Optional[Path] = None,
    llm: Any = None,
    run_with_timeout: Optional[Callable[..., Any]] = None,
    field_context: str = "",
) -> Dict[str, Any]:
    """Process each DWG sequentially; extract structured survey fields; write Word doc."""
    ws = (workspace or Path.cwd()).resolve()
    dwg_paths = resolve_dwg_paths_from_query(query)
    if not dwg_paths:
        return {
            "success": False,
            "error": "No .dwg files could be resolved from the request.",
            "response": "Could not find any .dwg paths in your message. List full paths or filenames in the same folder.",
        }

    missing = [p for p in dwg_paths if not Path(p).exists()]
    if missing:
        return {
            "success": False,
            "error": f"Missing DWG file(s): {', '.join(missing)}",
            "response": (
                "These plan files were not found on disk:\n"
                + "\n".join(f"- {m}" for m in missing)
                + "\n\nCheck paths and try again."
            ),
        }

    output_path = resolve_dwg_extract_output_docx_path(query, ws)
    extractions: List[Dict[str, Any]] = []
    for dwg in dwg_paths:
        extractions.append(
            extract_plan_details_for_dwg(
                autocad,
                dxf_fallback,
                dwg,
                llm=llm,
                run_with_timeout=run_with_timeout,
                field_context=field_context,
            )
        )

    sections = _dwg_sections_to_word_payload(build_dwg_word_sections_from_extractions(extractions))
    create_result = document_processor.create_word_document_from_structure(
        str(output_path),
        title="Plan Details Extract",
        sections=sections,
        metadata={
            "Plans processed": str(len(extractions)),
            "Generated by": "SurvyAI cadastral DWG extract pipeline",
        },
    )
    if not create_result.get("success"):
        return {
            "success": False,
            "error": create_result.get("error", "Failed to create Word document"),
            "response": str(create_result),
            "output_path": str(output_path),
            "extractions": extractions,
        }

    lines = [
        "Plan details extracted from DWG file(s) and saved to Word.",
        f"- Output: {output_path}",
        f"- Plans processed: {len(extractions)}",
        "",
        "Per plan:",
    ]
    for ex in extractions:
        plan = ex.get("survey_plan")
        buyer = "(see document)"
        if isinstance(plan, SurveyPlanExtraction):
            buyer = plan.buyer_name or plan.location or buyer
        err = ex.get("errors") or []
        status = "OK" if not err else f"partial ({'; '.join(err[:2])})"
        lines.append(f"- {ex.get('name')}: {buyer} [{status}]")

    return {
        "success": True,
        "response": "\n".join(lines),
        "output_path": str(output_path),
        "extractions": extractions,
        "plans_total": len(extractions),
    }


def _pick_user_requested_dwg(scope_text: str) -> Optional[str]:
    """Return the DWG path/name the user asked to save/generate in *scope_text*."""
    q = scope_text or ""
    patterns = [
        r"(?:save\s+(?:strictly\s+)?as|generate|create|produce|replot\s+(?:as|to)?)\s*['\"]([^'\"]+\.dwg)['\"]",
        r"(?:save\s+(?:strictly\s+)?as|generate|create|produce|replot\s+(?:as|to)?)\s*\"([^\"]+\.dwg)\"",
        r"(?:save\s+(?:strictly\s+)?as|generate|create|produce|replot\s+(?:as|to)?)\s*([^\s'\"]+\.dwg)",
        r"['\"]([^'\"]+\.dwg)['\"]",
    ]
    for pat in patterns:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if m:
            return (m.group(1) or "").strip().strip("\"'").rstrip(").,;")
    dwgs = extract_dwg_paths_from_text(q)
    return dwgs[0] if dwgs else None


def _normalize_resolved_path(raw: str) -> str:
    try:
        return str(Path(raw).resolve())
    except Exception:
        return raw


def find_similar_pdf_paths(requested_path: str, *, limit: int = 5) -> List[str]:
    """List PDFs in the same folder that might match a missing requested file."""
    try:
        req = Path(requested_path)
    except Exception:
        return []
    parent = req.parent
    if not parent.exists() or not parent.is_dir():
        return []
    req_name = req.name.lower()
    stem = req.stem
    similar: List[str] = []
    seen: set[str] = set()

    prefix_m = re.match(r"^(.+?)(\d+)$", stem, flags=re.IGNORECASE)
    if prefix_m:
        base = prefix_m.group(1)
        for candidate in sorted(parent.glob("*.pdf")):
            if candidate.name.lower() == req_name:
                continue
            if candidate.stem.lower().startswith(base.lower()):
                resolved = _normalize_resolved_path(str(candidate))
                if resolved not in seen:
                    seen.add(resolved)
                    similar.append(resolved)

    for candidate in sorted(parent.glob("*.pdf")):
        if candidate.name.lower() == req_name:
            continue
        resolved = _normalize_resolved_path(str(candidate))
        if resolved in seen:
            continue
        seen.add(resolved)
        similar.append(resolved)

    return similar[: max(1, int(limit))]


def resolve_pdf_path_for_replot(
    scope_text: str,
    full_text: str = "",
) -> Dict[str, Any]:
    """
    Resolve the survey-plan PDF path from the user's *current* request.

    Rules:
    - Paths in ``scope_text`` (current user turn) always win over conversation history.
    - The exact user-specified path is used when it exists; history is never substituted.
    - If the requested file is missing, return similar candidates and require user approval.
    - For short affirmations (e.g. "Proceed"), fall back to ``full_text`` for the path.
    """
    scope = (scope_text or "").strip()
    full = (full_text or scope).strip()
    use_full = _is_affirmation_reply(scope) or not extract_pdf_paths_from_text(scope)
    source_text = full if use_full else scope

    pdfs = extract_pdf_paths_from_text(source_text)
    if not pdfs:
        return {"success": False, "error": "No PDF path found in the request."}

    requested = _normalize_resolved_path(pdfs[-1])
    if Path(requested).exists():
        return {"success": True, "path": requested, "requested": requested}

    similar = find_similar_pdf_paths(requested)
    lines = [
        "The survey plan PDF you specified was not found:",
        f"  {requested}",
        "",
        "SurvyAI will not open a different file without your approval.",
    ]
    if similar:
        lines.append("Similar PDFs in the same folder:")
        for item in similar:
            lines.append(f"  - {item}")
        lines.append("")
        lines.append("Reply with the exact path you want to use, or correct the file location.")
    else:
        lines.append("No similar PDF files were found nearby. Check the path and try again.")

    return {
        "success": False,
        "error": "\n".join(lines),
        "requested": requested,
        "similar": similar,
        "needs_user_approval": bool(similar),
    }


def resolve_output_dwg_path(
    query: str,
    pdf_path: str,
    *,
    scope_text: Optional[str] = None,
) -> Optional[str]:
    """Resolve target DWG from the user's current request; default to PDF stem in same folder."""
    scope = (scope_text if scope_text is not None else query) or ""
    full = query or ""
    use_full = _is_affirmation_reply(scope) or not (
        _pick_user_requested_dwg(scope) or extract_dwg_paths_from_text(scope)
    )
    source_text = full if use_full else scope

    dwg_ref = _pick_user_requested_dwg(source_text)
    if dwg_ref:
        dwg_p = Path(dwg_ref)
        if dwg_p.is_absolute():
            return str(dwg_p.resolve())
        return str((Path(pdf_path).parent / dwg_p.name).resolve())

    return str((Path(pdf_path).with_suffix(".dwg")).resolve())


def today_certification_date_str() -> str:
    return _reference_now().strftime("%d-%m-%Y")


def _reference_now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def _relative_certification_date(days_offset: int) -> str:
    return (_reference_now() + timedelta(days=int(days_offset))).strftime("%d-%m-%Y")


def _normalize_cert_date_text(raw: str) -> str:
    s = (raw or "").strip()
    m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$", s)
    if not m:
        return s.replace("/", "-")
    dd, mm, yy = m.group(1).zfill(2), m.group(2).zfill(2), m.group(3)
    if len(yy) == 2:
        yy = "20" + yy
    return f"{dd}-{mm}-{yy}"


_CERT_DATE_TOKEN = r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"

# Explicit numeric certification / plan dates in free-form user prompts.
_USER_CERT_DATE_PATTERNS: tuple[str, ...] = (
    # Field-style: "date= 31/07/2026", "date: 31/07/2026", mid-sentence ", date=…"
    rf"(?:^|[\n,;])\s*date\s*[:=]\s*{_CERT_DATE_TOKEN}",
    rf"\bdate\s*[:=]\s*{_CERT_DATE_TOKEN}",
    # Conventional cadastral phrasing
    rf"date\s+on\s+the\s+certification\s*[:=]\s*{_CERT_DATE_TOKEN}",
    rf"certification\s+date\s*[:=]\s*{_CERT_DATE_TOKEN}",
    rf"date\s+on\s+the\s+(?:plan|certification)\s*(?:to|as|[:=])\s*{_CERT_DATE_TOKEN}",
    rf"date\s+(?:on\s+the\s+(?:plan|certification)\s+)?to\s+{_CERT_DATE_TOKEN}",
    rf"(?:the\s+)?date\s+should\s+(?:now\s+)?be\s+{_CERT_DATE_TOKEN}",
    rf"date\s+should\s+be\s+changed\s+to\s+{_CERT_DATE_TOKEN}",
    rf"(?:change|update|set)\s+(?:the\s+)?(?:plan\s+|certification\s+)?date\s+to\s+{_CERT_DATE_TOKEN}",
)


def extract_user_requested_certification_date(text: str) -> Optional[str]:
    """
    Extract an explicit certification/plan date from free text.

    Accepts varied prompt styles (``date= 31/07/2026``, ``date: 31/07/2026``,
    ``date on the certification: …``, ``the date should now be …``, etc.).
    Returns Nigerian cadastral ``DD-MM-YYYY``, or None when no explicit date is stated.
    """
    raw = text or ""
    if not raw.strip():
        return None
    for pat in _USER_CERT_DATE_PATTERNS:
        m = re.search(pat, raw, flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            continue
        normalized = _normalize_cert_date_text(m.group(1).strip())
        if re.match(r"^\d{2}-\d{2}-\d{4}$", normalized):
            return normalized
    return None


def _add_months(dt: datetime, months: int) -> datetime:
    """Calendar-aware month add/subtract (preserves day-of-month when possible)."""
    month_index = dt.month - 1 + int(months)
    year = dt.year + month_index // 12
    month = month_index % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def _certification_date_change_requested(scope: str) -> bool:
    ql = (scope or "").lower()
    return bool(
        re.search(
            r"\b(change|update|set)\s+the\s+date\b|"
            r"\bdate\s+on\s+the\s+(?:plan|certification)\b|"
            r"\bcertification\s+date\b|"
            r"\b(?:the\s+)?date\s+should\s+(?:now\s+)?be\b|"
            r"\bdate\s+should\s+be\s+changed\s+to\b|"
            r"\bdate\s+should\s+(?:now\s+)?(?:read|show|say)\b|"
            # Field-style dates in CAD prompts: "date= 31/07/2026", "date: 31/07/2026"
            r"\bdate\s*[:=]\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            ql,
        )
    )


def _extract_date_change_target_phrase(scope: str) -> str:
    """Pull the natural-language date target from a change-date instruction."""
    patterns = (
        r"change\s+the\s+date\s+on\s+the\s+plan\s+to\s+(?:the\s+date\s+of\s+)?(.+)$",
        r"change\s+the\s+date\s+to\s+(.+)$",
        r"update\s+the\s+date\s+(?:on\s+the\s+plan\s+)?to\s+(.+)$",
        r"date\s+on\s+the\s+(?:plan|certification)\s+(?:to|as)\s+(.+)$",
        r"set\s+the\s+(?:plan|certification)\s+date\s+to\s+(.+)$",
        r"(?:the\s+)?date\s+should\s+(?:now\s+)?be\s+(.+)$",
    )
    text = (scope or "").strip()
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            phrase = (m.group(1) or "").strip().rstrip(").,;")
            if phrase:
                return phrase
    return ""


def _parse_relative_offset_phrase(phrase: str, reference: datetime) -> Optional[datetime]:
    """
    Parse phrases like '3 months and 1 week before today', '2 weeks ago', '45 days from now'.
    """
    p = (phrase or "").lower().strip()
    if not p:
        return None

    if re.search(r"\btoday(?:['\u2019]s)?(?:\s+date)?\b", p) and not re.search(
        r"\b\d+\s*(?:year|month|week|day)s?\b", p
    ):
        return reference
    if re.search(r"\btomorrow\b", p):
        return reference + timedelta(days=1)
    if re.search(r"\byesterday\b", p):
        return reference - timedelta(days=1)

    backward = bool(
        re.search(r"\b(?:before|ago|earlier\s+than)\b", p)
        or re.search(r"\bprior\s+to\s+today\b", p)
    )
    forward = bool(re.search(r"\b(?:after|from\s+now|later|hence)\b", p))
    sign = -1 if backward and not forward else (1 if forward and not backward else -1)

    years = months = weeks = days = 0
    for m in re.finditer(r"(\d+)\s*(year|month|week|day)s?", p):
        n = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("year"):
            years += n
        elif unit.startswith("month"):
            months += n
        elif unit.startswith("week"):
            weeks += n
        elif unit.startswith("day"):
            days += n

    if not any((years, months, weeks, days)):
        return None

    result = reference
    if sign < 0:
        if years:
            result = _add_months(result, -years * 12)
        if months:
            result = _add_months(result, -months)
        result = result - timedelta(weeks=weeks, days=days)
    else:
        if years:
            result = _add_months(result, years * 12)
        if months:
            result = _add_months(result, months)
        result = result + timedelta(weeks=weeks, days=days)
    return result


_CERT_DATE_LLM_SYSTEM = """You resolve cadastral plan certification dates from natural-language user instructions.

You are given the user's local reference datetime ("today") and their wording for the desired plan date.

Return ONLY JSON:
{
  "date": "DD-MM-YYYY",
  "confidence": 0.0 to 1.0,
  "reason": "brief explanation"
}

Rules:
- Use calendar-aware arithmetic for months and years (not fixed 30-day months).
- "3 months and 1 week before today" means subtract 3 calendar months AND 7 days from the reference date.
- Output Nigerian cadastral style DD-MM-YYYY.
- If the instruction is ambiguous, pick the most likely surveyor intent and lower confidence."""


def resolve_certification_date_with_llm(
    scope: str,
    *,
    llm: Any,
    run_with_timeout: Callable[..., Any],
    reference: Optional[datetime] = None,
    timeout_s: int = 25,
) -> Optional[str]:
    """LLM fallback for non-trivial certification date wording."""
    if not llm or not (scope or "").strip():
        return None
    ref = reference or _reference_now()
    user_prompt = (
        f"Reference datetime (today): {ref.strftime('%d-%m-%Y %H:%M %Z (%A)')}\n"
        f"User instruction:\n{scope.strip()}\n\n"
        "What certification date should be printed on the plan?"
    )
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=_CERT_DATE_LLM_SYSTEM),
            HumanMessage(content=user_prompt),
        ]
        msg, err, timed_out = run_with_timeout(timeout_s, lambda: llm.invoke(messages))
        if timed_out or err:
            logger.debug("Certification date LLM failed: %s", err or "timeout")
            return None
        raw = msg.content if hasattr(msg, "content") else str(msg)
        if isinstance(raw, list):
            raw = "\n".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in raw
            )
        data = _extract_json_object(str(raw))
        if not data:
            return None
        date_s = str(data.get("date") or "").strip()
        if not date_s:
            return None
        return _normalize_cert_date_text(date_s)
    except Exception as exc:
        logger.debug("Certification date LLM parse failed: %s", exc)
        return None


def _resolve_certification_date_fast(scope: str) -> Optional[str]:
    """Cheap deterministic paths — today/tomorrow/yesterday/explicit numeric dates (scope only)."""
    ql = (scope or "").lower()

    # Explicit numeric field dates first (date= / date: / date on the certification: …).
    explicit_field = extract_user_requested_certification_date(scope)
    if explicit_field:
        return explicit_field

    if re.search(r"\btomorrow(?:['\u2019]s)?(?:\s+date)?\b", ql) or re.search(
        r"date\s+(?:on\s+the\s+plan\s+)?to\s+tomorrow\b", ql
    ):
        return _relative_certification_date(1)

    if re.search(r"\byesterday(?:['\u2019]s)?(?:\s+date)?\b", ql) or re.search(
        r"date\s+(?:on\s+the\s+plan\s+)?to\s+yesterday\b", ql
    ):
        return _relative_certification_date(-1)

    if not re.search(r"\b\d+\s*(?:year|month|week|day)s?\s+(?:and\s+)?(?:\d+\s*(?:week|day)s?\s+(?:and\s+)?)?before\s+today\b", ql):
        if any(
            p in ql
            for p in (
                "today's date",
                "todays date",
                "today date",
                "current date",
                "date to today",
            )
        ) or re.search(r"date\s+(?:on\s+the\s+plan\s+)?to\s+today\b", ql):
            return today_certification_date_str()

    return None


def _certification_date_scope(scope: str, full: str) -> str:
    """
    Date instructions always come from the current user turn.

    Conversation history must not override an explicit date phrase in the latest message
    (e.g. prior 'tomorrow' must not win over '2 months before today').
    """
    scope = (scope or "").strip()
    if scope and _certification_date_change_requested(scope):
        return scope
    full = (full or "").strip()
    if _is_affirmation_reply(scope) and full and _certification_date_change_requested(full):
        return full
    return scope


def resolve_certification_date_from_query(
    query: str,
    *,
    scope_text: Optional[str] = None,
    llm: Any = None,
    run_with_timeout: Optional[Callable[..., Any]] = None,
    timeout_s: int = 25,
) -> Optional[str]:
    """
    Resolve certification date from natural-language user instructions.

    Order: relative phrase in current turn → simple literals → LLM fallback.
    Never reads stale date words (tomorrow/today) from conversation history when the
    current message states its own date target.
    """
    scope = _certification_date_scope(scope_text or "", query or "")
    if not scope.strip():
        return None
    ref = _reference_now()

    if _certification_date_change_requested(scope):
        phrase = _extract_date_change_target_phrase(scope) or scope
        parsed = _parse_relative_offset_phrase(phrase, ref)
        if parsed:
            return parsed.strftime("%d-%m-%Y")

    fast = _resolve_certification_date_fast(scope)
    if fast:
        return fast

    if not _certification_date_change_requested(scope):
        return None

    if llm is not None and run_with_timeout is not None:
        llm_date = resolve_certification_date_with_llm(
            scope,
            llm=llm,
            run_with_timeout=run_with_timeout,
            reference=ref,
            timeout_s=timeout_s,
        )
        if llm_date:
            return llm_date

    return None


def _buyer_name_change_requested(scope: str) -> bool:
    ql = (scope or "").lower()
    return bool(
        re.search(
            r"\b(?:buyer|owner)(?:'s)?\s*name\b.*\b(?:should|change|update|set|now)\b|"
            r"\b(?:buyer|owner)(?:'s)?\s*name\s+changes?\s+to\b|"
            r"\b(?:change|update|set)\s+(?:the\s+)?(?:buyer|owner)(?:'s)?\s*name\b|"
            r"\bbuyer\s*name\s*[:=]",
            ql,
        )
    )


def _plan_override_change_requested(scope: str) -> bool:
    """True when the user may be changing any plan field (not only buyer/date)."""
    if _buyer_name_change_requested(scope) or _certification_date_change_requested(scope):
        return True
    ql = (scope or "").lower()
    if re.search(
        r"(?:buyer\s*name|location|plan\s*(?:no\.?|number)|surveyor|pillar\s+numbers?|"
        r"coordinates\s+for|date\s+on|origin_?crs|crs_?origin|scale|bearing|distance|traverse)\s*[:=]",
        ql,
    ):
        return True
    return bool(
        re.search(
            r"\b(?:should\s+(?:now\s+)?be|changes?\s+to|now\s+is|change|update|set|replace|correct|amend|modify)\b",
            ql,
        )
        and re.search(
            r"\b(?:buyer|owner|surveyor|plan\s*(?:no|number)|pillar|location|lga|state|date|"
            r"bearing|distance|traverse|geometry|coordinate|scale|fence|access\s*road|origin|crs|area)\b",
            ql,
        )
    )


def _resolve_buyer_name_fast(scope: str) -> Optional[str]:
    """Cheap deterministic buyer/owner name overrides from natural-language prompts."""
    text = (scope or "").strip()
    if not text:
        return None

    patterns = (
        r"buyers?\s*'?s?\s*name\s*[:=]\s*'([^']+)'",
        r'buyers?\s*\'?s?\s*name\s*[:=]\s*"([^"]+)"',
        rf"buyers?\s*'?s?\s*name\s*[:=]\s*(.+?){CADASTRAL_FIELD_BOUNDARY}",
        r"buyers?\s*'?s?\s*name\s+should\s+(?:now\s+)?be\s+['\"]([^'\"]+)['\"]",
        r"buyers?\s*'?s?\s*name\s+changes?\s+to\s+['\"]([^'\"]+)['\"]",
        r"owners?\s*'?s?\s*name\s+should\s+(?:now\s+)?be\s+['\"]([^'\"]+)['\"]",
        r"(?:change|update|set)\s+(?:the\s+)?(?:buyer|owner)s?(?:'s)?\s*name\s+to\s+['\"]([^'\"]+)['\"]",
        r"(?:change|update|set)\s+(?:the\s+)?(?:buyer|owner)s?(?:'s)?\s*name\s+to\s+([^.,;]+?)(?:\s+and\s+the\s+date|\s*,\s*and\s+|\s*\.|$)",
        r"(?:buyer|owner)s?(?:'s)?\s*name\s+(?:is\s+)?(?:now\s+)?['\"]([^'\"]+)['\"]",
    )
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            name = sanitize_metadata_field((m.group(1) or "").strip(), max_len=140)
            if name:
                return name
    return None


def _parse_coords_blob_fields(blob: str, pillars: List[str]) -> Dict[str, Any]:
    """Parse anchor coordinate and traverse legs from a cadastral coordinates blob."""
    out: Dict[str, Any] = {}
    text = (blob or "").strip()
    if not text:
        return out

    m0 = re.search(
        r"\(\s*([0-9]+(?:\.[0-9]+)?)\s*m?\s*[eE]\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*m?\s*[nN]\s*\)",
        text,
        flags=re.IGNORECASE,
    )
    if not m0:
        m0 = re.search(
            r"([0-9]+(?:\.[0-9]+)?)\s*m?\s*[eE]\s*[,; ]+\s*([0-9]+(?:\.[0-9]+)?)\s*m?\s*[nN]\b",
            text,
            flags=re.IGNORECASE,
        )
    if m0:
        out["anchor_easting"] = float(m0.group(1))
        out["anchor_northing"] = float(m0.group(2))

    leg_re = re.compile(
        r"\bbearing\b\s*(?:(?:=|:|-)|\bis\b)?\s*"
        r"(\d{1,3})\s*(?:deg|degree|degrees|°|d)\s*"
        r"([0-5]?\d)\s*(?:min|mins|minute|minutes|['\u2019])"
        r"(?:[^0-9]{0,80}?)"
        r"(?:distance|dist\.?|measured\s+distance)\s*(?:=|is|:)?\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:m)?\b",
        flags=re.IGNORECASE | re.DOTALL,
    )
    legs: List[SurveyTraverseLeg] = []
    for mm in leg_re.finditer(text):
        legs.append(
            SurveyTraverseLeg(
                bearing_deg=int(mm.group(1)),
                bearing_min=int(mm.group(2)),
                distance_m=float(mm.group(3)),
            )
        )
    legs = _filter_plausible_legs(legs)
    if pillars and legs and len(pillars) == len(legs):
        for i, leg in enumerate(legs):
            leg.from_pillar = pillars[i]
            leg.to_pillar = pillars[(i + 1) % len(pillars)]
    if legs:
        out["traverse_legs"] = legs
    return out


def _resolve_plan_overrides_fast(scope: str) -> SurveyPlanOverrides:
    """Deterministic extraction of user-requested plan field overrides."""
    text = (scope or "").strip()
    overrides = SurveyPlanOverrides()
    if not text:
        return overrides

    def _set(field: str, value: Any) -> None:
        if value is None or value == "" or value == []:
            return
        setattr(overrides, field, value)
        if field not in overrides.override_fields:
            overrides.override_fields.append(field)

    buyer = _resolve_buyer_name_fast(text)
    if buyer:
        _set("buyer_name", buyer)

    cert_date = resolve_certification_date_from_query(text, scope_text=text)
    if cert_date:
        _set("certification_date", cert_date)

    location = _pick_cadastral_value(
        text,
        (
            r"location\s*[:=]\s*'([^']+)'",
            r'location\s*[:=]\s*"([^"]+)"',
            rf"location\s*[:=]\s*(.+?){CADASTRAL_FIELD_BOUNDARY}",
            r"location\s+should\s+(?:now\s+)?be\s+['\"]([^'\"]+)['\"]",
            r"(?:change|update|set)\s+(?:the\s+)?location\s+to\s+['\"]([^'\"]+)['\"]",
        ),
    )
    if location:
        _set("location", sanitize_metadata_field(location, max_len=200))

    lga = _pick_cadastral_value(
        text,
        (
            r"local\s+(?:govt\.?|government)\s+area\s*[:=]\s*'([^']+)'",
            r'local\s+(?:govt\.?|government)\s+area\s*[:=]\s*"([^"]+)"',
            rf"local\s+(?:govt\.?|government)\s+area\s*[:=]\s*(.+?){CADASTRAL_FIELD_BOUNDARY}",
            r"(?:lga|local\s+government\s+area)\s+should\s+(?:now\s+)?be\s+['\"]([^'\"]+)['\"]",
        ),
    )
    if lga:
        _set("lga", sanitize_metadata_field(normalize_lga_name(lga) or lga, max_len=80))

    state = _pick_cadastral_value(
        text,
        (
            r"state\s*[:=]\s*'([^']+)'",
            r'state\s*[:=]\s*"([^"]+)"',
            rf"state\s*[:=]\s*(.+?){CADASTRAL_FIELD_BOUNDARY}",
            r"state\s+should\s+(?:now\s+)?be\s+['\"]([^'\"]+)['\"]",
        ),
    )
    if state:
        _set("state", sanitize_metadata_field(state, max_len=40))

    origin = _pick_cadastral_value(
        text,
        (
            r"(?:crs_?origin|origin_?crs)\s*[:=]\s*'([^']+)'",
            r'(?:crs_?origin|origin_?crs)\s*[:=]\s*"([^"]+)"',
            rf"(?:crs_?origin|origin_?crs)\s*[:=]\s*(.+?){CADASTRAL_FIELD_BOUNDARY}",
        ),
    )
    if origin:
        _set("origin_crs", sanitize_metadata_field(origin, max_len=60))

    plan_no = extract_user_requested_plan_number(text)
    if not plan_no:
        plan_no = _pick_cadastral_value(
            text,
            (
                r"plan\s*(?:no\.?|number)\s*[:=]\s*'([^']+)'",
                r'plan\s*(?:no\.?|number)\s*[:=]\s*"([^"]+)"',
                r"plan\s*(?:no\.?|number)\s*[:=]\s*([A-Z0-9][A-Z0-9/\-]+)",
                r"plan\s*(?:no\.?|number)\s+should\s+(?:now\s+)?be\s+['\"]?([A-Z0-9][A-Z0-9/\-]+)['\"]?",
                r"plan\s*(?:no\.?|number)\s+now\s+is\s+['\"]?([A-Z0-9][A-Z0-9/\-]+)['\"]?",
                r"(?:change|update|set)\s+(?:the\s+)?plan\s*(?:no\.?|number)\s+to\s+['\"]?([A-Z0-9][A-Z0-9/\-]+)['\"]?",
            ),
        )
    if plan_no:
        _set("plan_number", normalize_plan_number(str(plan_no).split("\n")[0].strip()))

    surveyor = _pick_cadastral_value(
        text,
        (
            r"surveyor\s+name\s*[:=]\s*'([^']+)'",
            r'surveyor\s+name\s*[:=]\s*"([^"]+)"',
            rf"surveyor\s+name\s*[:=]\s*(.+?){CADASTRAL_FIELD_BOUNDARY}",
            r"surveyor(?:'s)?\s+name\s+should\s+(?:now\s+)?be\s+['\"]([^'\"]+)['\"]",
            r"(?:change|update|set)\s+(?:the\s+)?surveyor(?:'s)?\s+name\s+to\s+['\"]([^'\"]+)['\"]",
        ),
    )
    if surveyor:
        _set(
            "surveyor_name",
            scrub_surveyor_metadata_value(
                ensure_surveyor_professional_title(surveyor), max_len=100
            ),
        )

    surveyor_addr = _pick_cadastral_value(
        text,
        (
            r"surveyor\s+company\s+and\s+address\s*[:=]\s*'([^']+)'",
            r'surveyor\s+company\s+and\s+address\s*[:=]\s*"([^"]+)"',
            rf"surveyor\s+company\s+and\s+address\s*[:=]\s*(.+?){CADASTRAL_FIELD_BOUNDARY}",
            r"surveyor(?:'s)?\s+(?:company\s+and\s+)?address\s+should\s+(?:now\s+)?be\s+['\"]([^'\"]+)['\"]",
        ),
    )
    if surveyor_addr:
        _set("surveyor_address", scrub_surveyor_metadata_value(surveyor_addr, max_len=200))

    pillar_list: List[str] = []
    m_p = re.search(
        rf"pillar\s+numbers\s*[:=]\s*(.*?)(?={_COORDINATES_FOR_STOP}|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_p:
        raw = (m_p.group(1) or "").strip().rstrip(",").strip()
        pillar_list = [
            _normalize_pillar_id(p.strip().strip("'\""))
            for p in re.split(r"[,\n]+", raw)
            if p.strip()
        ]
    if not pillar_list:
        nl_pillars = _pick_cadastral_value(
            text,
            (
                r"pillar\s+numbers?\s+should\s+(?:now\s+)?be\s+(.+?)(?:\.\s|$|\s+and\s+(?:the\s+)?(?:date|buyer|location))",
                r"(?:change|update|set)\s+(?:the\s+)?pillar\s+numbers?\s+to\s+(.+?)(?:\.\s|$|\s+and\s+)",
            ),
        )
        if nl_pillars:
            pillar_list = [
                _normalize_pillar_id(p.strip().strip("'\""))
                for p in re.split(r"[,\n]+", nl_pillars)
                if p.strip()
            ]
    if pillar_list:
        _set("pillar_numbers", pillar_list)

    scale_m = re.search(r"plot\s+using\s+scale\s+1\s*:\s*(\d+)", text, re.IGNORECASE)
    if not scale_m:
        scale_m = re.search(r"scale\s+should\s+(?:now\s+)?be\s+1\s*:\s*(\d+)", text, re.IGNORECASE)
    if scale_m:
        _set("scale_denom", int(scale_m.group(1)))
    else:
        try:
            sd = extract_user_requested_scale_denom(text)
            if sd:
                _set("scale_denom", int(sd))
        except Exception:
            pass

    coords_blob = extract_coordinates_blob_from_cadastral_query(text)
    if coords_blob:
        geom = _parse_coords_blob_fields(coords_blob, pillar_list)
        for key, val in geom.items():
            _set(key, val)

    return overrides


_PLAN_OVERRIDE_LLM_SYSTEM = """You extract SURVEY PLAN field OVERRIDES from natural-language user instructions.

The user is replotting or editing a Nigerian cadastral/survey plan. They may change ANY detail:
title block (buyer/owner, location, LGA, state, plan number, surveyor, certification date, scale),
geometry (pillar numbers, coordinates, traverse bearings/distances, area),
or extras (access roads, fences).

Return ONLY fields the user explicitly asked to change, set, update, or replace.
Use null for everything else.

Return ONLY JSON:
{
  "buyer_name": string|null,
  "location": string|null,
  "lga": string|null,
  "state": string|null,
  "origin_crs": string|null,
  "plan_number": string|null,
  "surveyor_name": string|null,
  "surveyor_address": string|null,
  "certification_date": "DD-MM-YYYY"|null,
  "scale_denom": integer|null,
  "area_sq_m": number|null,
  "pillar_numbers": [string]|null,
  "anchor_easting": number|null,
  "anchor_northing": number|null,
  "anchor_pillar": string|null,
  "traverse_legs": [{"from_pillar","to_pillar","bearing_deg","bearing_min","distance_m"}]|null,
  "access_roads": [string]|null,
  "fences": [string]|null,
  "access_road_title": string|null,
  "confidence": 0.0 to 1.0,
  "reason": "brief explanation"
}

Rules:
- Exact spellings from the user prompt for names and pillar IDs (SC/XX 1234 style).
- Bearings: DD° MM' from North clockwise; distances in metres.
- traverse_legs: clockwise boundary lines when the user specifies geometry changes.
- certification_date: DD-MM-YYYY Nigerian cadastral style.
- Never invent fields the user did not mention."""


def _plan_overrides_from_llm_dict(data: Dict[str, Any]) -> SurveyPlanOverrides:
    overrides = SurveyPlanOverrides()
    if not data:
        return overrides

    def _set(field: str, value: Any) -> None:
        if value is None or value == "" or value == []:
            return
        setattr(overrides, field, value)
        if field not in overrides.override_fields:
            overrides.override_fields.append(field)

    buyer = sanitize_metadata_field(str(data.get("buyer_name") or "").strip(), max_len=140)
    if buyer:
        _set("buyer_name", buyer)

    for meta_field, max_len in (
        ("location", 120),
        ("lga", 80),
        ("state", 40),
        ("origin_crs", 60),
        ("surveyor_name", 80),
        ("surveyor_address", 200),
        ("access_road_title", 80),
    ):
        val = sanitize_metadata_field(str(data.get(meta_field) or "").strip(), max_len=max_len)
        if meta_field == "lga" and val:
            val = sanitize_metadata_field(normalize_lga_name(val) or val, max_len=max_len)
        if val:
            _set(meta_field, val)

    plan_no = normalize_plan_number(str(data.get("plan_number") or "").strip())
    if plan_no:
        _set("plan_number", plan_no)

    cert = str(data.get("certification_date") or "").strip()
    if cert:
        _set("certification_date", _normalize_cert_date_text(cert))

    scale = data.get("scale_denom") or data.get("scale")
    try:
        if scale is not None:
            _set("scale_denom", int(scale))
    except Exception:
        pass

    area = data.get("area_sq_m") or data.get("area")
    try:
        if area is not None:
            _set("area_sq_m", float(area))
    except Exception:
        pass

    pillars_raw = data.get("pillar_numbers") or data.get("pillars")
    pillars: List[str] = []
    if isinstance(pillars_raw, list):
        pillars = [_normalize_pillar_id(str(p)) for p in pillars_raw if str(p).strip()]
    elif isinstance(pillars_raw, str) and pillars_raw.strip():
        pillars = [
            _normalize_pillar_id(p.strip())
            for p in re.split(r"[,\n]+", pillars_raw)
            if p.strip()
        ]
    if pillars:
        _set("pillar_numbers", pillars)

    ae = _maybe_float(data.get("anchor_easting") or data.get("easting"))
    an = _maybe_float(data.get("anchor_northing") or data.get("northing"))
    if ae is not None:
        _set("anchor_easting", ae)
    if an is not None:
        _set("anchor_northing", an)
    ap = _normalize_pillar_id(str(data.get("anchor_pillar") or ""))
    if ap:
        _set("anchor_pillar", ap)

    legs = _coerce_legs(data.get("traverse_legs") or data.get("legs"))
    if legs:
        if pillars and len(pillars) == len(legs):
            for i, leg in enumerate(legs):
                if not leg.from_pillar:
                    leg.from_pillar = pillars[i]
                if not leg.to_pillar:
                    leg.to_pillar = pillars[(i + 1) % len(pillars)]
        _set("traverse_legs", legs)

    roads_raw = data.get("access_roads")
    if isinstance(roads_raw, list):
        roads = [str(r).strip() for r in roads_raw if str(r).strip()]
        if roads:
            _set("access_roads", roads)

    fences_raw = data.get("fences")
    if isinstance(fences_raw, list):
        fences = [str(f).strip() for f in fences_raw if str(f).strip()]
        if fences:
            _set("fences", fences)

    overrides.confidence = float(data.get("confidence") or 0.0)
    overrides.notes = str(data.get("reason") or data.get("notes") or "").strip()
    return overrides


def _resolve_plan_overrides_with_llm(
    scope: str,
    *,
    llm: Any,
    run_with_timeout: Callable[..., Any],
    base_extraction: Optional[SurveyPlanExtraction] = None,
    reference: Optional[datetime] = None,
    timeout_s: int = 35,
) -> SurveyPlanOverrides:
    """LLM interpretation of any plan field the user wants to override."""
    if not llm or not (scope or "").strip():
        return SurveyPlanOverrides()
    ref = reference or _reference_now()
    baseline = ""
    if base_extraction is not None:
        baseline = (
            "Values extracted from the source PDF (change only what the user asked for):\n"
            + json.dumps(
                {
                    k: v
                    for k, v in base_extraction.model_dump().items()
                    if k not in ("confidence", "source", "notes") and v not in ("", None, [])
                },
                ensure_ascii=False,
                indent=2,
            )[:6000]
        )
    user_prompt = (
        f"Reference datetime (today): {ref.strftime('%d-%m-%Y %H:%M %Z (%A)')}\n"
        f"{baseline}\n\n"
        f"User instruction:\n{scope.strip()}\n\n"
        "What plan field overrides did the user request?"
    )
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=_PLAN_OVERRIDE_LLM_SYSTEM),
            HumanMessage(content=user_prompt),
        ]
        msg, err, timed_out = run_with_timeout(timeout_s, lambda: llm.invoke(messages))
        if timed_out or err:
            logger.debug("Plan override LLM failed: %s", err or "timeout")
            return SurveyPlanOverrides()
        raw = msg.content if hasattr(msg, "content") else str(msg)
        if isinstance(raw, list):
            raw = "\n".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in raw
            )
        data = _extract_json_object(str(raw))
        return _plan_overrides_from_llm_dict(data or {})
    except Exception as exc:
        logger.debug("Plan override LLM parse failed: %s", exc)
        return SurveyPlanOverrides()


def _merge_plan_overrides(
    fast: SurveyPlanOverrides,
    llm_result: SurveyPlanOverrides,
) -> SurveyPlanOverrides:
    """Fast regex wins; LLM fills gaps only."""
    merged = fast.model_copy(deep=True)
    for field in SurveyPlanOverrides.model_fields:
        if field in ("override_fields", "confidence", "notes"):
            continue
        if field in merged.override_fields:
            continue
        val = getattr(llm_result, field, None)
        if val is None or val == "" or val == []:
            continue
        setattr(merged, field, val)
        if field not in merged.override_fields:
            merged.override_fields.append(field)
    if llm_result.confidence > merged.confidence:
        merged.confidence = llm_result.confidence
    if llm_result.notes and not merged.notes:
        merged.notes = llm_result.notes
    return merged


def apply_plan_overrides_to_extraction(
    extraction: SurveyPlanExtraction,
    overrides: SurveyPlanOverrides,
) -> SurveyPlanExtraction:
    """Apply user-requested field changes onto a PDF extraction."""
    if not overrides.override_fields:
        return extraction

    data = extraction.model_dump()
    for field in overrides.override_fields:
        val = getattr(overrides, field, None)
        if val is None or val == "" or val == []:
            continue
        data[field] = val

    result = SurveyPlanExtraction(**data)
    pillars = result.pillar_numbers or []
    legs = result.traverse_legs or []
    if pillars and legs:
        for i, leg in enumerate(legs):
            if not leg.from_pillar and i < len(pillars):
                leg.from_pillar = pillars[i]
            if not leg.to_pillar and pillars:
                leg.to_pillar = pillars[(i + 1) % len(pillars)]
    return result


def resolve_plan_overrides_from_query(
    query: str,
    *,
    scope_text: Optional[str] = None,
    base_extraction: Optional[SurveyPlanExtraction] = None,
    llm: Any = None,
    run_with_timeout: Optional[Callable[..., Any]] = None,
    timeout_s: int = 35,
) -> SurveyPlanOverrides:
    """
    Resolve any user-requested plan field overrides from the current user turn.

    Fast regex runs first for structured and common natural-language phrasing.
    When override intent is detected but regex missed fields, a cheap LLM pass
    interprets the instruction against optional PDF-extracted baseline values.
    """
    scope = (scope_text or query or "").strip()
    if not scope:
        return SurveyPlanOverrides()

    fast = _resolve_plan_overrides_fast(scope)
    if not _plan_override_change_requested(scope):
        return fast

    if llm is None or run_with_timeout is None:
        return fast

    llm_result = _resolve_plan_overrides_with_llm(
        scope,
        llm=llm,
        run_with_timeout=run_with_timeout,
        base_extraction=base_extraction,
        timeout_s=timeout_s,
    )
    return _merge_plan_overrides(fast, llm_result)


def resolve_buyer_name_from_query(
    query: str,
    *,
    scope_text: Optional[str] = None,
    llm: Any = None,
    run_with_timeout: Optional[Callable[..., Any]] = None,
    timeout_s: int = 25,
) -> Optional[str]:
    """Resolve buyer/owner name override (delegates to full plan override resolver)."""
    overrides = resolve_plan_overrides_from_query(
        query,
        scope_text=scope_text,
        llm=llm,
        run_with_timeout=run_with_timeout,
        timeout_s=timeout_s,
    )
    return overrides.buyer_name


class PlanMetadataOverrides(BaseModel):
    """Backward-compatible subset of plan overrides (buyer + certification date)."""

    buyer_name: str | None = None
    certification_date: str | None = None


def resolve_plan_metadata_overrides_from_query(
    query: str,
    *,
    scope_text: Optional[str] = None,
    base_extraction: Optional[SurveyPlanExtraction] = None,
    llm: Any = None,
    run_with_timeout: Optional[Callable[..., Any]] = None,
    timeout_s: int = 25,
) -> PlanMetadataOverrides:
    """Backward-compatible wrapper around ``resolve_plan_overrides_from_query``."""
    overrides = resolve_plan_overrides_from_query(
        query,
        scope_text=scope_text,
        base_extraction=base_extraction,
        llm=llm,
        run_with_timeout=run_with_timeout,
        timeout_s=timeout_s,
    )
    return PlanMetadataOverrides(
        buyer_name=overrides.buyer_name,
        certification_date=overrides.certification_date,
    )


def wants_today_certification_date(query: str) -> bool:
    """Backward-compatible helper — prefer ``resolve_certification_date_from_query``."""
    return resolve_certification_date_from_query(query) == today_certification_date_str()
