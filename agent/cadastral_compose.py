"""
Intelligent cadastral composition for complex / file-deferred prompts.

Architecture (cost-aware):
1. Gate: only when coordinates/bearings are deferred to external files — NOT when
   the user already provided a conventional inline cadastral prompt (fastpath).
2. Discover source files in the workspace (.xlsx/.csv/.txt/.docx/…).
3. Deterministic extractors first (cheap).
4. One RAG recall of similar past composition examples + one cheap LLM pass when
   structure is ambiguous — returns a fully composed cadastral sub-prompt or parcels.
5. Caller plots via the existing deterministic `_run_cadastral_cad_prompt_pipeline`.

This module does not replace the simple CAD fastpath.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from agent.excel_cadastral import (
    FamilyParcel,
    ParcelPoint,
    build_excel_cadastral_subprompt,
    coordinates_deferred_to_external_source,
    find_reference_dwg_from_query,
    format_absolute_coordinates_blob,
    parse_family_parcels_from_excel,
    parse_family_parcels_from_rows,
    write_dup_xlsx_with_headers,
    _letter_for_index,
)
from utils.logger import get_logger

logger = get_logger(__name__)

COORD_SOURCE_SUFFIXES = (".xlsx", ".xls", ".xlsm", ".csv", ".txt", ".docx", ".doc")
CADASTRAL_COMPOSE_DOC_TYPE = "cadastral_compose_example"

_UTM_PAIR_RE = re.compile(
    r"([0-9]{5,7}(?:\.[0-9]+)?)\s*(?:m)?\s*[eE]\s*[,; ]+\s*"
    r"([0-9]{5,7}(?:\.[0-9]+)?)\s*(?:m)?\s*[nN]"
)
_PILLAR_NEAR_RE = re.compile(
    r"([A-Za-z]{1,6}\s*/\s*[A-Za-z]{1,6}\s*[0-9]{2,6}|SC/[A-Za-z0-9]+\s*[0-9]{2,6}|P\s*[0-9]{1,5})",
    re.I,
)


def should_intelligent_cadastral_compose(query: str) -> bool:
    """
    True for complex file-deferred cadastral goals that should NOT use the
    simple inline CAD fastpath.
    """
    q = query or ""
    ql = q.lower()
    if ".dwg" not in ql:
        return False
    if not any(k in ql for k in ("generate", "create", "produce", "plot")):
        return False
    if not coordinates_deferred_to_external_source(q):
        return False
    return True


def discover_coordinate_source_files(
    query: str,
    workspace: str | Path,
    *,
    max_files: int = 6,
) -> List[Path]:
    """Find coordinate/bearing source files mentioned in the query or alone in workspace."""
    q = query or ""
    ws = Path(workspace).resolve()
    found: List[Path] = []
    seen: set[str] = set()

    def _add(p: Path) -> None:
        try:
            rp = p.resolve()
        except Exception:
            return
        key = str(rp).lower()
        if key in seen or not rp.is_file():
            return
        if rp.suffix.lower() not in COORD_SOURCE_SUFFIXES:
            return
        if rp.name.startswith("~$"):
            return
        seen.add(key)
        found.append(rp)

    for m in re.finditer(
        r"['\"]([^'\"]+\.(?:xlsx|xls|xlsm|csv|txt|docx|doc))['\"]",
        q,
        flags=re.IGNORECASE,
    ):
        p = Path(m.group(1).strip())
        if not p.is_absolute():
            p = ws / p
        _add(p)

    for m in re.finditer(
        r"([A-Za-z]:\\[^\s'\"]+\.(?:xlsx|xls|xlsm|csv|txt|docx|doc))",
        q,
        flags=re.IGNORECASE,
    ):
        _add(Path(m.group(1)))

    for m in re.finditer(
        r"\b([A-Za-z0-9 _\-]{1,80}\.(?:xlsx|xls|xlsm|csv|txt|docx|doc))\b",
        q,
        flags=re.IGNORECASE,
    ):
        _add(ws / m.group(1).strip())

    # "only excel/csv/txt/docx file" → sole matching file in workspace
    sole_kinds: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = [
        (("excel", "spreadsheet", "workbook", ".xlsx", ".xls"), (".xlsx", ".xls", ".xlsm")),
        (("csv", ".csv"), (".csv",)),
        (("txt", "text file", ".txt"), (".txt",)),
        (("docx", "word", ".doc"), (".docx", ".doc")),
    ]
    ql = q.lower()
    for markers, suffixes in sole_kinds:
        if not any(m in ql for m in markers):
            continue
        if not any(k in ql for k in ("only", "the", "go to", "open", "read", "from")):
            continue
        cands = [
            p
            for p in ws.iterdir()
            if p.is_file()
            and p.suffix.lower() in suffixes
            and not p.name.startswith("~$")
        ]
        if len(cands) == 1:
            _add(cands[0])
        elif len(cands) > 1 and suffixes == (".xlsx", ".xls", ".xlsm"):
            # Prefer the semantically best ownership workbook (not a filename heuristic).
            try:
                from agent.excel_cadastral import choose_best_ownership_workbook

                choice = choose_best_ownership_workbook(cands)
                if choice.get("success"):
                    _add(Path(choice["path"]))
            except Exception:
                pass

    if not found and ws.is_dir():
        # Last resort: if workspace has exactly one coord-source file, use it.
        cands = [
            p
            for p in ws.iterdir()
            if p.is_file()
            and p.suffix.lower() in COORD_SOURCE_SUFFIXES
            and not p.name.startswith("~$")
        ]
        if len(cands) == 1:
            _add(cands[0])
        else:
            xls = [p for p in cands if p.suffix.lower() in {".xlsx", ".xls", ".xlsm"}]
            if xls:
                try:
                    from agent.excel_cadastral import choose_best_ownership_workbook

                    choice = choose_best_ownership_workbook(xls)
                    if choice.get("success"):
                        _add(Path(choice["path"]))
                except Exception:
                    pass

    return found[:max_files]


def _split_delimited_line(line: str) -> List[str]:
    s = (line or "").strip()
    if not s:
        return []
    if "\t" in s:
        return [c.strip() for c in s.split("\t")]
    if ";" in s and s.count(";") >= s.count(","):
        return [c.strip() for c in s.split(";")]
    if "," in s:
        try:
            return next(csv.reader([s]))
        except Exception:
            return [c.strip() for c in s.split(",")]
    return re.split(r"\s{2,}|\s+", s)


def extract_parcels_from_csv(file_path: str | Path) -> Dict[str, Any]:
    """Parse family parcels from a CSV (same ownership-block conventions as Excel)."""
    path = Path(file_path).resolve()
    if not path.exists():
        return {"success": False, "error": f"CSV not found: {path}", "parcels": []}
    try:
        df = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    except Exception as exc:
        return {"success": False, "error": f"Failed to read CSV: {exc}", "parcels": []}
    parsed = parse_family_parcels_from_rows(
        df.fillna("").values.tolist(),
        source_label=f"CSV '{path.name}'",
    )
    if parsed.get("success"):
        parsed["file_path"] = str(path)
        parsed["source_kind"] = "csv"
    return parsed


def extract_geometry_from_plain_text(text: str) -> Dict[str, Any]:
    """
    Deterministic extract from TXT/DOCX body text.

    Prefers absolute UTM pairs (optionally with nearby pillar IDs). Also detects
    a ready-made coordinates_blob when bearing/distance language is present.
    """
    source = text or ""
    pairs = list(_UTM_PAIR_RE.finditer(source))
    points: List[ParcelPoint] = []
    for m in pairs:
        e = float(m.group(1))
        n = float(m.group(2))
        window = source[max(0, m.start() - 80) : m.end() + 40]
        pillar_m = _PILLAR_NEAR_RE.search(window)
        pillar = pillar_m.group(1).strip() if pillar_m else ""
        points.append(ParcelPoint(e=e, n=n, pillar=pillar))

    has_bearings = bool(
        re.search(r"\bbearing\b", source, re.I)
        and re.search(r"\b(?:distance|dist\.?)\b", source, re.I)
    )

    result: Dict[str, Any] = {
        "success": False,
        "points": points,
        "coordinates_blob": "",
        "parcels": [],
        "source_kind": "text",
    }

    if has_bearings and pairs:
        # Keep a traverse-style blob for the CAD parser.
        start = pairs[0].start()
        # Prefer from first "coordinates" mention if present.
        cm = re.search(r"coordinates\s+for\s+the\s+points?\b", source, re.I)
        if cm:
            start = cm.start()
        blob = source[start:].strip()
        # Trim obvious trailing metadata chatter
        blob = re.split(r"\n\s*(?:Done\.|NOTES?:|Also take)\b", blob, maxsplit=1)[0].strip()
        result["coordinates_blob"] = blob
        result["success"] = True
        return result

    if len(points) >= 3:
        # Try ownership blocks via whitespace/CSV-like lines.
        rows: List[List[str]] = []
        for line in source.splitlines():
            cells = _split_delimited_line(line)
            if cells:
                rows.append(cells)
        parsed = parse_family_parcels_from_rows(rows, source_label="text table")
        if parsed.get("success"):
            result["success"] = True
            result["parcels"] = parsed["parcels"]
            result["parcel_count"] = parsed["parcel_count"]
            return result
        # Single ring from ordered UTM pairs.
        parcel = FamilyParcel(owner_name="Parcel 1", letter="A", points=points)
        result["success"] = True
        result["parcels"] = [parcel]
        result["parcel_count"] = 1
        result["coordinates_blob"] = format_absolute_coordinates_blob(points)
        return result

    return result


def extract_from_coordinate_file(
    file_path: str | Path,
    *,
    document_processor: Any = None,
) -> Dict[str, Any]:
    """Route a source file to the appropriate deterministic extractor."""
    path = Path(file_path).resolve()
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        out = parse_family_parcels_from_excel(path)
        if out.get("success"):
            out["source_kind"] = "excel"
        return out
    if suffix == ".csv":
        return extract_parcels_from_csv(path)
    if suffix == ".txt":
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return {"success": False, "error": f"Failed to read text file: {exc}"}
        out = extract_geometry_from_plain_text(text)
        out["file_path"] = str(path)
        out["raw_text"] = text[:12000]
        return out
    if suffix in {".docx", ".doc"}:
        if document_processor is None:
            return {
                "success": False,
                "error": "Document processor unavailable for Word files.",
                "file_path": str(path),
            }
        try:
            got = document_processor.get_full_text(str(path), preserve_structure=True)
        except Exception as exc:
            return {"success": False, "error": f"Failed to read Word file: {exc}", "file_path": str(path)}
        if not got.get("success"):
            return {
                "success": False,
                "error": got.get("error") or "Word text extraction failed",
                "file_path": str(path),
            }
        text = str(got.get("text") or "")
        out = extract_geometry_from_plain_text(text)
        out["file_path"] = str(path)
        out["raw_text"] = text[:12000]
        # Also try tables if available
        if not out.get("success") and hasattr(document_processor, "get_tables"):
            try:
                tables = document_processor.get_tables(str(path))
                if tables.get("success"):
                    rows: List[List[Any]] = []
                    for table in tables.get("tables") or []:
                        grid = table.get("data") or table.get("rows") or table.get("grid") or []
                        for row in grid:
                            if isinstance(row, (list, tuple)):
                                rows.append(list(row))
                    parsed = parse_family_parcels_from_rows(rows, source_label=f"Word '{path.name}'")
                    if parsed.get("success"):
                        parsed["file_path"] = str(path)
                        parsed["source_kind"] = "docx"
                        return parsed
            except Exception:
                pass
        out["source_kind"] = "docx"
        return out
    return {"success": False, "error": f"Unsupported coordinate source type: {suffix}"}


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
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _retrieve_compose_rag_context(
    query: str,
    *,
    vector_store: Any,
    search_fn: Optional[Callable[..., List[Dict[str, Any]]]],
    score_threshold: float = 0.22,
    top_k: int = 3,
) -> str:
    if search_fn is None:
        return "(no similar stored examples yet)"
    try:
        from tools.vector_store import COLLECTION_DOCUMENTS

        hits = search_fn(
            query[:500],
            COLLECTION_DOCUMENTS,
            top_k=top_k,
        ) or []
        snippets: List[str] = []
        for hit in hits:
            score = float(hit.get("score") or hit.get("similarity") or 0.0)
            if score and score < score_threshold:
                continue
            text = (hit.get("text") or hit.get("content") or "").strip()
            meta = hit.get("metadata") or {}
            if meta.get("doc_type") and meta.get("doc_type") != CADASTRAL_COMPOSE_DOC_TYPE:
                # Still allow general cadastral docs — they help phrasing.
                pass
            if text:
                snippets.append(text[:900])
        if not snippets:
            return "(no similar stored examples yet)"
        return "\n---\n".join(snippets[:top_k])
    except Exception as exc:
        logger.info("Compose RAG recall skipped: %s", exc)
        return "(no similar stored examples yet)"


def _parcels_from_llm_payload(payload: Dict[str, Any]) -> List[FamilyParcel]:
    parcels: List[FamilyParcel] = []
    raw_parcels = payload.get("parcels") or []
    if not isinstance(raw_parcels, list):
        return parcels
    for i, item in enumerate(raw_parcels):
        if not isinstance(item, dict):
            continue
        owner = str(item.get("owner_name") or item.get("owner") or item.get("name") or "").strip()
        letter = str(item.get("letter") or "").strip().upper() or _letter_for_index(i)
        pts_raw = item.get("points") or item.get("coordinates") or []
        points: List[ParcelPoint] = []
        if isinstance(pts_raw, list):
            for pt in pts_raw:
                if not isinstance(pt, dict):
                    continue
                try:
                    e = float(pt.get("e") if pt.get("e") is not None else pt.get("easting") or pt.get("x"))
                    n = float(pt.get("n") if pt.get("n") is not None else pt.get("northing") or pt.get("y"))
                except Exception:
                    continue
                pillar = str(pt.get("pillar") or pt.get("pillar_number") or pt.get("name") or "").strip()
                points.append(ParcelPoint(e=e, n=n, pillar=pillar))
        if len(points) >= 3:
            parcels.append(FamilyParcel(owner_name=owner or f"Parcel {i + 1}", letter=letter, points=points))
    return parcels


def llm_compose_cadastral_plan(
    query: str,
    *,
    file_snippets: Sequence[Dict[str, str]],
    llm: Any,
    run_with_timeout: Callable[..., Any],
    vector_store: Any = None,
    search_fn: Optional[Callable[..., List[Dict[str, Any]]]] = None,
    timeout_s: int = 45,
) -> Dict[str, Any]:
    """
    One RAG-informed LLM call that turns a complex user request + file excerpts
    into either parcels or a ready coordinates_blob + metadata fields.
    """
    if llm is None or run_with_timeout is None:
        return {"success": False, "error": "LLM unavailable for intelligent composition.", "source": "none"}

    rag_block = _retrieve_compose_rag_context(
        query,
        vector_store=vector_store,
        search_fn=search_fn,
    )
    files_block_parts: List[str] = []
    for item in file_snippets[:4]:
        name = item.get("name") or "file"
        kind = item.get("kind") or ""
        body = (item.get("text") or "")[:8000]
        files_block_parts.append(f"### {name} ({kind})\n{body}")
    files_block = "\n\n".join(files_block_parts) if files_block_parts else "(no file text extracted)"

    system = (
        "You are SurvyAI's cadastral composition planner for Nigerian survey plans.\n"
        "Follow OBSERVE → THINK → ACT on every request:\n"
        "OBSERVE: list ownership blocks, coordinate sources, reference DWG metadata asks, "
        "explicit scale, plan-number increment rules, separate-vs-multi layout intent.\n"
        "THINK: map each observed need to one field in the JSON schema; never merge unrelated "
        "fields (scale is NOT part of surveyor name/address; LGA is not location).\n"
        "ACT: return ONLY valid JSON that the CAD pipeline can plot without placeholders.\n"
        "Return ONLY JSON with this schema:\n"
        "{\n"
        '  "confidence": 0.0-1.0,\n'
        '  "buyer_name": string,\n'
        '  "location": string|null,\n'
        '  "lga": string|null,\n'
        '  "state": string|null,\n'
        '  "origin_crs": string|null,\n'
        '  "plan_number": string|null,\n'
        '  "certification_date": string|null,\n'
        '  "surveyor_name": string|null,\n'
        '  "surveyor_address": string|null,\n'
        '  "scale_denom": integer|null,\n'
        '  "pillar_numbers": [string],\n'
        '  "coordinates_blob": string|null,\n'
        '  "parcels": [{"owner_name": string, "letter": "A", '
        '"points": [{"e": number, "n": number, "pillar": string}]}],\n'
        '  "notes": string\n'
        "}\n"
        "Rules:\n"
        "- Prefer parcels[] when multiple ownership rings exist (blank-row separated sets).\n"
        "- Prefer coordinates_blob when the source is a single traverse with bearings/distances.\n"
        "- Owner labels with letters like 'AMADI (B)' ONLY when the user asked for multi-parcel "
        "letter tags on ONE plan. If they asked for separate/different CAD plans per owner "
        "(each owner's coords → only that owner's DWG; buyer name as each drawing name; "
        "incrementing plan numbers), still return parcels[] (one per owner) WITHOUT inventing "
        "a combined multi-parcel layout — the caller plots each parcel as its own DWG.\n"
        "- Preserve numeric coordinates exactly as in the files.\n"
        "- Do not invent coordinates that are not present in the files/prompt.\n"
        "- If metadata must come from a reference DWG and is missing, leave those fields null.\n"
        "- plan_number: set ONLY when the user explicitly stated a plan number or a batch "
        "starting number (e.g. \"start from plan number 'RV/018/2026/SP'\", "
        "\"plan number: RV/…\"). That explicit value is the first plan; later owners "
        "increment from it. Do NOT copy the reference DWG's plan number when the user "
        "asked only for location/LGA/state/surveyor from that plan. Leave null when the "
        "user said to take the plan number from the existing/reference plan (caller fills "
        "it from the DWG). Never invent a plan number. Never put plan numbers inside "
        "surveyor_name or surveyor_address.\n"
        "- scale_denom: integer only (250, 500, 1000, …) when the user explicitly requested a "
        "scale (e.g. 'scale: 1:250'). Leave null when the user did not state a scale — the "
        "plot engine will auto-coarsen for large land or auto-refine to 1:250 for very small "
        "parcels using the same fit criterion. Never put scale text inside surveyor_name or "
        "surveyor_address.\n"
        "- surveyor_name / surveyor_address: person/company/address only — no scale, plan "
        "number, pillar list, or coordinates.\n"
    )
    user = (
        f"Similar past composition examples (RAG):\n{rag_block}\n\n"
        f"User request:\n{query}\n\n"
        f"Source file excerpts:\n{files_block}\n\n"
        "Return JSON only."
    )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        msg, err, timed_out = run_with_timeout(
            timeout_s,
            lambda: llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]),
        )
        if timed_out or err or msg is None:
            return {
                "success": False,
                "error": err or ("LLM timed out" if timed_out else "LLM returned empty"),
                "source": "error",
            }
        text = msg.content if hasattr(msg, "content") else str(msg)
        if isinstance(text, list):
            text = "\n".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in text
            )
        payload = _extract_json_object(str(text or ""))
        if not payload:
            return {"success": False, "error": "LLM compose returned non-JSON.", "source": "llm_parse_failed"}

        confidence = float(payload.get("confidence") or 0.0)
        parcels = _parcels_from_llm_payload(payload)
        blob = str(payload.get("coordinates_blob") or "").strip()
        if not parcels and not blob:
            return {
                "success": False,
                "error": "LLM compose produced neither parcels nor coordinates_blob.",
                "source": "llm_empty",
                "confidence": confidence,
                "payload": payload,
            }
        if confidence < 0.28 and not parcels and len(_UTM_PAIR_RE.findall(blob)) < 1:
            return {
                "success": False,
                "error": f"LLM compose confidence too low ({confidence:.2f}).",
                "source": "llm_low_confidence",
                "confidence": confidence,
                "payload": payload,
            }
        meta = {
                "buyer_name": str(payload.get("buyer_name") or "").strip(),
                "location": str(payload.get("location") or "").strip(),
                "lga": str(payload.get("lga") or "").strip(),
                "state": str(payload.get("state") or "").strip(),
                "origin_crs": str(payload.get("origin_crs") or "").strip(),
                "plan_number": str(payload.get("plan_number") or "").strip(),
                "certification_date": str(payload.get("certification_date") or "").strip(),
                "surveyor_name": str(payload.get("surveyor_name") or "").strip(),
                "surveyor_address": str(payload.get("surveyor_address") or "").strip(),
                "pillar_numbers": [
                    str(p).strip()
                    for p in (payload.get("pillar_numbers") or [])
                    if str(p).strip()
                ],
            }
        try:
            from agent.pdf_survey_plan import (
                extract_user_requested_scale_denom,
                scrub_surveyor_metadata_value,
            )

            meta["surveyor_name"] = scrub_surveyor_metadata_value(
                meta["surveyor_name"], max_len=100
            )
            meta["surveyor_address"] = scrub_surveyor_metadata_value(
                meta["surveyor_address"], max_len=200
            )
            sd_raw = payload.get("scale_denom") or payload.get("scale")
            sd = None
            try:
                if sd_raw is not None and str(sd_raw).strip():
                    sd = int(re.sub(r"[^\d]", "", str(sd_raw).split(":")[-1]) or 0) or None
            except Exception:
                sd = None
            if not sd:
                sd = extract_user_requested_scale_denom(query)
            if sd:
                meta["scale_denom"] = int(sd)
        except Exception:
            pass
        return {
            "success": True,
            "source": "llm_rag",
            "confidence": confidence,
            "parcels": parcels,
            "coordinates_blob": blob,
            "metadata": meta,
            "notes": str(payload.get("notes") or "").strip(),
            "payload": payload,
        }
    except Exception as exc:
        logger.exception("llm_compose_cadastral_plan failed")
        return {"success": False, "error": str(exc), "source": "error"}


def build_subprompt_from_coordinates_blob(
    *,
    output_dwg: str,
    coordinates_blob: str,
    buyer_name: str = "",
    location: str = "",
    lga: str = "",
    state: str = "",
    origin_crs: str = "",
    plan_number: str = "",
    surveyor_name: str = "",
    surveyor_address: str = "",
    certification_date: str = "",
    pillar_numbers: Optional[Sequence[str]] = None,
    template_path: Optional[str] = None,
    scale_denom: Optional[int] = None,
) -> str:
    """Build a conventional cadastral prompt from a coordinates/traverse blob."""
    lines: List[str] = []
    if template_path:
        lines.append(f"template '{template_path}'")
    lines.append(f"Generate '{output_dwg}'")
    if buyer_name:
        lines.append(f"buyer name: {buyer_name}")
    if location:
        lines.append(f"location: {location}")
    if lga:
        lines.append(f"local government area: {lga}")
    if state:
        lines.append(f"state: {state}")
    if origin_crs:
        lines.append(f"origin_crs: {origin_crs}")
    if plan_number:
        lines.append(f"plan number: {plan_number}")
    if certification_date:
        lines.append(f"date on the certification: {certification_date}")
    try:
        sd = int(scale_denom) if scale_denom is not None else None
    except Exception:
        sd = None
    if sd and sd > 0:
        lines.append(f"Plot using scale 1:{sd}")
    if surveyor_name:
        lines.append(f"Surveyor name: {surveyor_name}")
    if surveyor_address:
        lines.append(f"Surveyor company and address: {surveyor_address}")
    pillars = [str(p).strip() for p in (pillar_numbers or []) if str(p).strip()]
    if pillars:
        lines.append("pillar numbers: " + ", ".join(pillars))
    blob = (coordinates_blob or "").strip()
    if blob.lower().startswith("coordinates"):
        lines.append(blob)
    else:
        lines.append("coordinates for the points = " + blob)
    return "\n".join(lines)


def store_compose_example(
    vector_store: Any,
    *,
    query: str,
    composed_subprompt: str,
    source_files: Sequence[str],
) -> None:
    """Best-effort memory write so future RAG recall improves composition."""
    if vector_store is None or not composed_subprompt:
        return
    try:
        from tools.vector_store import COLLECTION_DOCUMENTS

        text = (
            "Cadastral compose example\n"
            f"User request:\n{(query or '')[:1200]}\n\n"
            f"Sources: {', '.join(source_files)}\n\n"
            f"Composed prompt:\n{composed_subprompt[:2500]}"
        )
        meta = {
            "doc_type": CADASTRAL_COMPOSE_DOC_TYPE,
            "source_files": list(source_files)[:8],
        }
        if hasattr(vector_store, "add_texts"):
            vector_store.add_texts([text], metadatas=[meta], collection_name=COLLECTION_DOCUMENTS)
        elif hasattr(vector_store, "store_document"):
            vector_store.store_document(text, metadata=meta, collection=COLLECTION_DOCUMENTS)
    except Exception as exc:
        logger.info("Could not store compose example: %s", exc)


def compose_cadastral_from_files(
    query: str,
    *,
    workspace: str | Path,
    document_processor: Any = None,
    llm: Any = None,
    run_with_timeout: Optional[Callable[..., Any]] = None,
    vector_store: Any = None,
    search_fn: Optional[Callable[..., List[Dict[str, Any]]]] = None,
    prefer_llm: bool = False,
) -> Dict[str, Any]:
    """
    End-to-end composition (no AutoCAD plot): discover files → extract → compose subprompt.
    """
    ws = Path(workspace).resolve()
    sources = discover_coordinate_source_files(query, ws)
    notes: List[str] = []
    if not sources:
        notes.append("No coordinate source file discovered in workspace/query.")
        # Still allow LLM compose from prompt-only complex wording if LLM available.
        if llm is None:
            return {
                "success": False,
                "error": (
                    "No Excel/CSV/TXT/DOCX coordinate source found. "
                    "Place the file in the workspace or quote its path."
                ),
                "notes": notes,
            }

    file_snippets: List[Dict[str, str]] = []
    parcels: List[FamilyParcel] = []
    coordinates_blob = ""
    primary_source: Optional[Path] = None
    deterministic_ok = False

    for path in sources:
        extracted = extract_from_coordinate_file(path, document_processor=document_processor)
        snippet_text = ""
        if extracted.get("raw_text"):
            snippet_text = str(extracted["raw_text"])
        elif path.suffix.lower() == ".csv":
            try:
                snippet_text = path.read_text(encoding="utf-8", errors="replace")[:8000]
            except Exception:
                snippet_text = ""
        elif path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
            try:
                df = pd.read_excel(path, header=None, engine="openpyxl")
                snippet_text = df.head(80).to_csv(index=False, header=False)
            except Exception:
                snippet_text = ""
        file_snippets.append(
            {
                "name": path.name,
                "kind": path.suffix.lower().lstrip("."),
                "text": snippet_text or json.dumps(
                    {
                        "success": extracted.get("success"),
                        "parcel_count": extracted.get("parcel_count"),
                        "error": extracted.get("error"),
                    },
                    default=str,
                ),
            }
        )

        if extracted.get("success") and not prefer_llm:
            if extracted.get("parcels"):
                parcels = list(extracted["parcels"])
                primary_source = path
                deterministic_ok = True
                notes.append(f"Deterministic parcel parse from {path.name}")
                break
            if extracted.get("coordinates_blob"):
                coordinates_blob = str(extracted["coordinates_blob"])
                primary_source = path
                deterministic_ok = True
                notes.append(f"Deterministic traverse blob from {path.name}")
                break

    llm_meta: Dict[str, str] = {}
    llm_result: Optional[Dict[str, Any]] = None
    if (not deterministic_ok or prefer_llm) and llm is not None and run_with_timeout is not None:
        llm_result = llm_compose_cadastral_plan(
            query,
            file_snippets=file_snippets,
            llm=llm,
            run_with_timeout=run_with_timeout,
            vector_store=vector_store,
            search_fn=search_fn,
        )
        if llm_result.get("success"):
            notes.append(
                f"LLM+RAG composition (confidence={float(llm_result.get('confidence') or 0):.2f})"
            )
            if llm_result.get("parcels"):
                parcels = list(llm_result["parcels"])
            if llm_result.get("coordinates_blob"):
                coordinates_blob = str(llm_result["coordinates_blob"])
            llm_meta = dict(llm_result.get("metadata") or {})
            deterministic_ok = True
        else:
            notes.append(llm_result.get("error") or "LLM composition failed")

    if not deterministic_ok and not parcels and not coordinates_blob:
        return {
            "success": False,
            "error": (
                "Could not extract coordinates/bearings from the provided files. "
                "Check that Easting/Northing(/Pillar) rows or EmE/NmN + bearings are present."
            ),
            "notes": notes,
            "source_files": [str(p) for p in sources],
        }

    # Prompt-level overrides (CRS/date) always win when present.
    m_crs = re.search(
        r"(?:origin_crs|crs_origin)\s*[:=]\s*([^,\n]+)",
        query or "",
        flags=re.IGNORECASE,
    )
    origin_crs = (m_crs.group(1).strip().strip("'\"") if m_crs else "") or llm_meta.get("origin_crs", "")
    try:
        from agent.pdf_survey_plan import (
            extract_user_requested_certification_date,
            resolve_certification_date_from_query,
        )

        cert_date = (
            extract_user_requested_certification_date(query or "")
            or resolve_certification_date_from_query(query or "", scope_text=query or "")
            or llm_meta.get("certification_date", "")
            or ""
        )
    except Exception:
        m_cert = re.search(
            r"(?:date\s+on\s+the\s+certification|certification\s+date|\bdate)\s*[:=]\s*"
            r"([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
            query or "",
            flags=re.IGNORECASE,
        )
        cert_date = (m_cert.group(1).strip() if m_cert else "") or llm_meta.get(
            "certification_date", ""
        )

    out_m = re.search(
        r"\b(?:generate|create|produce)\s*[-]?\s*"
        r"(?:cad\s+drawing\s+|cad\s+|drawing\s+|file\s+)?['\"]?"
        r"([^'\"\s]+?\.dwg)",
        query or "",
        flags=re.IGNORECASE,
    )
    out_name = Path(out_m.group(1)).name if out_m else "Composed_Plan.dwg"
    output_dwg = str((ws / out_name).resolve())

    dup_path = None
    extra_parcels: List[Dict[str, Any]] = []
    extent_points: List[Dict[str, Any]] = []
    main_label = ""
    buyer_name = llm_meta.get("buyer_name", "")

    if parcels:
        composed = build_excel_cadastral_subprompt(
            output_dwg=output_dwg,
            parcels=parcels,
            location=llm_meta.get("location", ""),
            lga=llm_meta.get("lga", ""),
            state=llm_meta.get("state", ""),
            origin_crs=origin_crs or "UTM Zone 32N",
            plan_number=llm_meta.get("plan_number", ""),
            surveyor_name=llm_meta.get("surveyor_name", ""),
            surveyor_address=llm_meta.get("surveyor_address", ""),
            certification_date=cert_date,
        )
        if not composed.get("success"):
            return {
                "success": False,
                "error": composed.get("error") or "Failed to build subprompt from parcels.",
                "notes": notes,
            }
        subprompt = composed["subprompt"]
        extra_parcels = list(composed.get("extra_parcels") or [])
        extent_points = list(composed.get("extent_points") or [])
        main_label = str(composed.get("main_parcel_label") or "")
        buyer_name = str(composed.get("buyer_name") or buyer_name)
        # Persist a normalized workbook only when the user asked (any filename).
        try:
            from agent.excel_cadastral import (
                extract_requested_workbook_copy_name,
                query_requests_workbook_copy,
            )

            if query_requests_workbook_copy(query or ""):
                src_for_dup = primary_source or (sources[0] if sources else ws / "coords.csv")
                dup = write_dup_xlsx_with_headers(
                    src_for_dup,
                    dest_name=extract_requested_workbook_copy_name(query or ""),
                    parcels=parcels,
                    query=query or "",
                )
                if dup.get("success"):
                    dup_path = dup.get("output_path")
                else:
                    notes.append(f"Normalized workbook not written: {dup.get('error')}")
        except Exception as exc:
            notes.append(f"Normalized workbook not written: {exc}")
    else:
        subprompt = build_subprompt_from_coordinates_blob(
            output_dwg=output_dwg,
            coordinates_blob=coordinates_blob,
            buyer_name=buyer_name or llm_meta.get("buyer_name", "") or "Buyer",
            location=llm_meta.get("location", ""),
            lga=llm_meta.get("lga", ""),
            state=llm_meta.get("state", ""),
            origin_crs=origin_crs or "UTM Zone 32N",
            plan_number=llm_meta.get("plan_number", ""),
            surveyor_name=llm_meta.get("surveyor_name", ""),
            surveyor_address=llm_meta.get("surveyor_address", ""),
            certification_date=cert_date,
            pillar_numbers=llm_meta.get("pillar_numbers") or [],
        )

    ref_dwg = find_reference_dwg_from_query(query, ws)
    return {
        "success": True,
        "subprompt": subprompt,
        "extra_parcels": extra_parcels,
        "extent_points": extent_points,
        "main_parcel_label": main_label,
        "buyer_name": buyer_name,
        "output_dwg": output_dwg,
        "dup_xlsx": dup_path,
        "source_files": [str(p) for p in sources],
        "primary_source": str(primary_source) if primary_source else None,
        "reference_dwg": str(ref_dwg) if ref_dwg else None,
        "parcels": parcels,
        "coordinates_blob": coordinates_blob,
        "metadata": {
            **llm_meta,
            "origin_crs": origin_crs or llm_meta.get("origin_crs", ""),
            "certification_date": cert_date or llm_meta.get("certification_date", ""),
        },
        "notes": notes,
        "compose_source": (llm_result or {}).get("source") if llm_result else "deterministic",
    }
