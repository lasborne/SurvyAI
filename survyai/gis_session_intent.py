"""
Lightweight GIS session intent helpers shared by GUI context injection and the agent.

Keep this module free of heavy imports (no Qt, no LangChain) so both sides can use it.
"""

from __future__ import annotations

import re
from typing import Optional


def looks_like_full_gis_workflow_request(raw_query: str) -> bool:
    """True for a complete operational GIS job (convert/plot/buffer/overlap), not a follow-up."""
    q = (raw_query or "").strip().lower()
    if not q or len(q.split()) < 12:
        return False
    has_source = any(
        k in q
        for k in (
            "excel",
            ".xlsx",
            ".xls",
            "workbook",
            "coordinates",
            "csv",
        )
    )
    has_convert = any(
        k in q
        for k in (
            "convert",
            "utm",
            "crs",
            "epsg",
            "mid-belt",
            "midbelt",
            "projected",
            "minna",
        )
    )
    has_gis_ops = sum(
        1
        for k in (
            "polygon",
            "buffer",
            "overlap",
            "arcgis",
            "plot",
            "symbology",
            "fill color",
            "geodatabase",
        )
        if k in q
    )
    return bool(has_source and (has_convert or has_gis_ops >= 2) and has_gis_ops >= 1)


def looks_like_living_atlas_or_landcover_dl(raw_query: str) -> bool:
    """True when the user wants Esri Living Atlas / pretrained land-cover DL (not landmark fit)."""
    q = (raw_query or "").strip().lower()
    if not q:
        return False
    markers = (
        "living atlas",
        "livingatlas",
        "land cover",
        "landcover",
        "land-cover",
        "pretrained",
        "pre-trained",
        "pre trained",
        "already-trained",
        "already trained",
        "trained model",
        "trained land",
        "object-detection model",
        "object detection model",
        "deep learning model",
        "dl model",
        "classify grassland",
        "grassland cover",
        "grassland classif",
        "vegetation class",
        "imagery classif",
        "esri land cover",
        "landuse",
        "land use class",
    )
    return any(m in q for m in markers)


def looks_like_gis_path_provision(raw_query: str) -> bool:
    """True when the user is pasting ArcGIS/GIS file paths (often after being asked)."""
    t = (raw_query or "").strip()
    if not t:
        return False
    tl = t.lower()
    exts = (".aprx", ".gdb", ".shp", ".xlsx", ".csv", ".geojson", ".gpkg")
    hits = sum(tl.count(ext) for ext in exts)
    if hits < 1:
        return False
    cues = (
        "here are the details",
        "project to use",
        "owner polygons",
        "arcgis pro project",
        "opened for review",
        "use these",
        "use this path",
    )
    return hits >= 2 or any(c in tl for c in cues)


def looks_like_gis_session_followup(raw_query: str) -> bool:
    """True for GIS continuations that must keep prior ArcGIS/Excel context.

    Must NOT match a full restated convert/plot/buffer workflow — those are
    executable jobs the user may intentionally repeat.
    """
    q = (raw_query or "").strip().lower()
    if not q:
        return False
    if looks_like_full_gis_workflow_request(raw_query):
        return False
    # Living Atlas / land-cover DL on existing parcels must keep .aprx/.gdb context.
    if looks_like_living_atlas_or_landcover_dl(raw_query):
        return True
    # User pasted verified GIS paths after a "need project path" ask.
    if looks_like_gis_path_provision(raw_query):
        return True
    markers = (
        "fit comparison",
        "gis-based fit",
        "gis based fit",
        "practical gis",
        "footprint",
        "deep learning",
        "open arcgis result",
        "open arcgis",
        "would fit",
        "fit within",
        "fit inside",
        "each of them",
        "within each",
        "ascertain if",
        "large enough",
        "footprint approximation",
    )
    if any(m in q for m in markers):
        return True
    words = q.split()
    if words and words[0].strip(".,;:!?") in {"yes", "y", "ok", "okay", "sure", "please", "proceed"}:
        if any(k in q for k in ("fit", "parcel", "polygon", "footprint", "practical gis", "gis-based")):
            return True
    return False


def prior_assistant_reported_gis_success(prior_text: str) -> bool:
    """True when history shows a verified GIS workflow completion."""
    t = (prior_text or "").lower()
    if not t:
        return False
    has_result = "result_" in t or "verified outputs" in t or "verified output" in t
    has_artifact = any(ext in t for ext in (".aprx", ".gdb", ".xlsx", "owner_areas", "overlap"))
    return has_result and has_artifact


def build_gis_workflow_rerun_instruction(current_request: str) -> str:
    """Instruction block forcing tool re-execution for a repeated full GIS job."""
    req = (current_request or "").strip()
    return (
        "RE-EXECUTE FULL GIS WORKFLOW (user restated the job — do NOT no-op):\n"
        "A prior conversation turn may show a successful GIS run for a similar request. "
        "Still EXECUTE the current request with tools end-to-end (inspect/normalize Excel → "
        "CRS convert → owner polygons → buffers → overlaps / symbology as asked). "
        "Do NOT reply 'Already completed for this active GIS session' or only re-list old paths. "
        "Do NOT create a Word Report.docx unless the user explicitly asked for a Word report. "
        "Users often re-run because outputs were deleted, moved, or not useful. "
        "Overwrite same-named outputs or write refreshed siblings; verify RESULT_* from THIS run.\n\n"
        f"Current request:\n{req}"
    )


def extract_current_request_from_enriched(query: str) -> str:
    """Return the CURRENT request portion of a history-enriched query blob."""
    marker = "NOW, the user wants you to continue with this new request:"
    q = query or ""
    if marker in q:
        return q.split(marker)[-1].strip()
    return q.strip()
