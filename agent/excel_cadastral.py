"""Excel → cadastral plan composition helpers (multi-parcel OR per-owner plans).

Supports prompts where Easting/Northing/pillar values live in a workbook
(family/owner blocks separated by blank rows) and must be composed into a
valid cadastral CAD sub-prompt before the deterministic plot pipeline runs.

Two ownership layouts (semantic — never confuse them):
- **Multi-parcel (one DWG):** all owners on a single sheet with letter tags
  like ``AMADI (B)`` marking parcels inside one plan.
- **Separate owner plans (N DWGs):** each owner/family block plots only that
  owner's ring into its own ``.dwg`` named after the buyer; plan numbers
  increment per plan when a base plan number is available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

# File/source markers for deferred coordinate/bearing inputs (Excel, CSV, TXT, Word, …).
_COORD_SOURCE_MARKERS: Tuple[str, ...] = (
    "excel",
    ".xlsx",
    ".xls",
    "spreadsheet",
    "workbook",
    ".csv",
    "csv file",
    "comma-separated",
    ".txt",
    "text file",
    ".docx",
    ".doc",
    "word document",
    "word file",
)

_DEFERRED_COORD_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(
        r"extract(?:ed)?\s+from\s+(?:the\s+)?"
        r"(?:excel|xlsx|xls|csv|txt|docx|doc|spreadsheet|workbook|file|text|word)",
        re.I,
    ),
    re.compile(
        r"(?:coordinates?|pillar\s*(?:point\s*)?names?|easting|northing|bearing).{0,120}"
        r"(?:from|in)\s+(?:the\s+)?"
        r"(?:excel|xlsx|xls|csv|txt|docx|doc|spreadsheet|workbook|file|text|word)",
        re.I,
    ),
    re.compile(
        r"(?:excel|xlsx|xls|csv|txt|docx|doc|spreadsheet|workbook|text\s+file|word\s+file|"
        r"word\s+document).{0,120}"
        r"(?:coordinates?|easting|northing|pillar|bearing)",
        re.I,
    ),
    re.compile(
        r"(?:use|using)\s+these\s+coordinates.{0,80}"
        r"(?:excel|csv|txt|docx|file|plot\s+a\s+cad)",
        re.I,
    ),
    re.compile(
        r"coordinates\s+for\s+the\s+points?\b.{0,80}"
        r"(?:extract|read|taken|gotten|should\s+be)",
        re.I,
    ),
    re.compile(
        r"(?:read|open|go\s+to)\s+(?:the\s+)?(?:only\s+)?"
        r"(?:excel|csv|txt|docx|text|word|spreadsheet).{0,60}"
        r"(?:file|workbook)?",
        re.I,
    ),
)

_INLINE_UTM_PAIR_RE = re.compile(
    r"\d{5,7}(?:\.\d+)?\s*m?\s*[eE]\s*[,; ]+\s*\d{5,7}(?:\.\d+)?\s*m?\s*[nN]"
)

_INLINE_BEARING_DIST_RE = re.compile(
    r"\bbearing\b.{0,40}\b(?:distance|dist\.?)\b|\b(?:distance|dist\.?).{0,40}\bbearing\b",
    re.I,
)


def _mentions_coord_source_file(text: str) -> bool:
    ql = (text or "").lower()
    if any(k in ql for k in _COORD_SOURCE_MARKERS):
        return True
    # Bare "file" only when clearly about extracting coords/bearings from it.
    if re.search(r"\b(?:from|in)\s+(?:the\s+)?file\b", ql) and any(
        k in ql for k in ("coordinate", "easting", "northing", "bearing", "pillar")
    ):
        return True
    return False


def coordinates_deferred_to_external_source(text: str) -> bool:
    """
    True when the user asks to take coordinates/bearings from an external file
    (Excel/CSV/TXT/DOCX/…) rather than embedding EmE/NmN or traverse text inline.

    Simple conventional cadastral prompts with enough inline geometry return False
    so the hard-coded CAD fastpath remains available.
    """
    source = text or ""
    ql = source.lower()
    mentions_source = _mentions_coord_source_file(source)
    looks_deferred = mentions_source and any(p.search(source) for p in _DEFERRED_COORD_PATTERNS)
    if not looks_deferred:
        # Broader catch: coord-source file + generate .dwg + geometry language, no inline pairs.
        if not (
            mentions_source
            and ".dwg" in ql
            and any(k in ql for k in ("coordinate", "easting", "northing", "bearing", "pillar"))
            and any(k in ql for k in ("generate", "create", "produce", "plot"))
        ):
            return False
    # Enough inline UTM pairs → treat as conventional fastpath material.
    if len(_INLINE_UTM_PAIR_RE.findall(source)) >= 3:
        return False
    # Inline bearing/distance traverse with an anchor and no extract-from-file intent.
    if (
        _INLINE_BEARING_DIST_RE.search(source)
        and _INLINE_UTM_PAIR_RE.search(source)
        and not any(p.search(source) for p in _DEFERRED_COORD_PATTERNS[:5])
    ):
        return False
    return True


_TABULAR_SOURCE_MARKERS: Tuple[str, ...] = (
    "excel",
    ".xlsx",
    ".xls",
    "spreadsheet",
    "workbook",
    ".csv",
    "csv file",
    "comma-separated",
)

_INSPECT_ONLY_MARKERS: Tuple[str, ...] = (
    "extract",
    "read",
    "open the",
    "open this",
    "inspect",
    "list the",
    "summarize",
    "summary of",
    "show me",
    "what's in",
    "what is in",
    "contents of",
    "preview",
    "parse",
    "deeper",
    "dig deeper",
    "more detail",
    "full extraction",
    "get text",
    "get tables",
    "key details",
    "look carefully",
    "more thorough",
)

_CAD_DELIVERABLE_MARKERS: Tuple[str, ...] = (
    ".dwg",
    "cad drawing",
    "cad plan",
    "cadastral plan",
    "survey plan",
    "autocad",
    "plot a cad",
    "plot the cad",
    "generate a cad",
    "create a cad",
)


def explicit_cadastral_plot_intent(text: str) -> bool:
    """
    True only when the current request clearly asks for a CAD/DWG deliverable.

    Guardrail: reading/inspecting Excel (or affirming a deeper extract/search)
    must NOT count as plot intent. Used by Excel→CAD fastpaths and tools so
    SurvyAI never invents files like a default multi-owner DWG.
    """
    ql = (text or "").lower()
    if not ql.strip():
        return False
    has_plot_verb = any(
        k in ql for k in ("generate", "create", "produce", "plot", "draw", "replot")
    )
    has_deliverable = any(k in ql for k in _CAD_DELIVERABLE_MARKERS)
    return bool(has_plot_verb and has_deliverable)


def is_tabular_inspect_only_request(text: str) -> bool:
    """
    True when the request is about reading/inspecting a spreadsheet (or deepening
    that extraction), without an explicit CAD/DWG deliverable ask.
    """
    ql = (text or "").lower()
    if not ql.strip():
        return False
    if explicit_cadastral_plot_intent(ql):
        return False
    mentions_table = any(k in ql for k in _TABULAR_SOURCE_MARKERS)
    wants_inspect = any(k in ql for k in _INSPECT_ONLY_MARKERS)
    return bool(mentions_table and wants_inspect)


def default_excel_plot_output_name(
    query: str,
    parcels: Optional[Sequence[FamilyParcel]] = None,
) -> str:
    """
    Choose a DWG basename when the user asked to plot but did not name the file.

    Never invent a fixed brand name (e.g. Excel_Families.dwg). Prefer the first
    real owner/family title; otherwise a neutral cadastral default.
    """
    out_m = re.search(
        r"\b(?:generate|create|produce)\s*[-]?\s*"
        r"(?:cad\s+drawing\s+|cad\s+|drawing\s+|file\s+)?['\"]?"
        r"([^'\"\s]+?\.dwg)",
        query or "",
        flags=re.IGNORECASE,
    )
    if out_m:
        return Path(out_m.group(1)).name
    for parcel in parcels or ():
        name = (getattr(parcel, "owner_name", None) or "").strip()
        if name and not _is_placeholder_owner_name(name):
            return owner_plan_dwg_basename(name)
    return "Cadastral_Plan.dwg"


@dataclass
class ParcelPoint:
    e: float
    n: float
    pillar: str = ""


@dataclass
class FamilyParcel:
    owner_name: str
    letter: str = ""
    points: List[ParcelPoint] = field(default_factory=list)

    @property
    def labeled_name(self) -> str:
        name = (self.owner_name or "").strip()
        if not name:
            return f"({self.letter})" if self.letter else ""
        if self.letter:
            return f"{name} ({self.letter})"
        return name


def _is_empty_row(values: Sequence[Any]) -> bool:
    for v in values:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if str(v).strip() == "":
            continue
        return False
    return True


def _as_float(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    s = str(value).strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


_HEADER_EXACT = {
    "easting",
    "northing",
    "pillar",
    "pillars",
    "pillar no",
    "pillar no.",
    "pillar number",
    "pillar numbers",
    "east",
    "north",
    "x",
    "y",
    "e",
    "n",
    "peg",
    "pegs",
    "owner",
    "owners",
    "buyer",
    "buyers",
    "family",
    "name",
    "parcel",
    "parcel name",
}


def _looks_like_header_row(values: Sequence[Any]) -> bool:
    """
    True for column-title rows like Easting/Northing/Pillar/Owner.

    IMPORTANT: match whole cells only. Never treat words inside owner titles
    (e.g. \"FAMILY\" in \"OKACHI FAMILY LAND\") as a header hit.
    """
    if _as_float(values[0] if values else None) is not None:
        return False
    cells = [str(v or "").strip().lower() for v in list(values)[:6]]
    cells = [c for c in cells if c and c != "nan"]
    if not cells:
        return False
    # Whole-cell exact titles only (avoids matching 'family' inside owner names).
    if any(c in _HEADER_EXACT for c in cells):
        return True
    # Multi-word header cells still OK when the full cell is a known title phrase.
    return any(
        re.fullmatch(
            r"(easting|northing|pillars?|pillar\s*numbers?|pillar\s*no\.?|"
            r"east|north|owner|owners|buyer|buyers|family|name|parcel|parcel\s*name)",
            c,
        )
        for c in cells
    )


def _header_column_map(values: Sequence[Any]) -> Dict[str, int]:
    """Map logical column roles → indices for a header row."""
    out: Dict[str, int] = {}
    for i, v in enumerate(list(values)[:8]):
        c = str(v or "").strip().lower()
        if not c or c == "nan":
            continue
        if c in {"easting", "east", "x", "e"} and "easting" not in out:
            out["easting"] = i
        elif c in {"northing", "north", "y", "n"} and "northing" not in out:
            out["northing"] = i
        elif c in {"pillar", "pillars", "pillar no", "pillar no.", "pillar number",
                   "pillar numbers", "peg", "pegs"} and "pillar" not in out:
            out["pillar"] = i
        elif c in {"owner", "owners", "buyer", "buyers", "family", "name",
                   "parcel", "parcel name"} and "owner" not in out:
            out["owner"] = i
    return out


def _normalize_owner_display_name(raw: str) -> str:
    """
    Normalize Excel owner/family titles for CAD buyer names.

    Strip only a trailing ``Land`` token — keep ``Family`` and all other words.
    Example: ``Greenhouse Family Land`` → ``Greenhouse Family``.
    Also strip trailing letter tags already present: ``AMADI (B)`` → ``AMADI``.
    Never collapse ``X FAMILY`` down to bare ``X``.
    """
    s = re.sub(r"\s+", " ", (raw or "").strip())
    if not s:
        return ""
    # Strip trailing letter tags already present: "AMADI (B)" → "AMADI"
    s = re.sub(r"\s*\(([A-Za-z]{1,3})\)\s*$", "", s).strip()
    # Only remove trailing "Land" (keep "Family"): Greenhouse Family Land → Greenhouse Family
    s = re.sub(r"\s+LAND\s*$", "", s, flags=re.IGNORECASE)
    # Drop synthetic placeholders from earlier bad dup.xlsx runs.
    if re.fullmatch(r"Parcel\s+\d+", s, flags=re.IGNORECASE):
        return ""
    return s.strip() or (raw or "").strip()


def _is_placeholder_owner_name(name: str) -> bool:
    s = (name or "").strip()
    if not s:
        return True
    if re.fullmatch(r"Parcel\s+\d+(?:\s*\([A-Za-z]{1,3}\))?", s, flags=re.IGNORECASE):
        return True
    return False


def _is_owner_name_row(values: Sequence[Any]) -> bool:
    if not values:
        return False
    first = values[0]
    if first is None or (isinstance(first, float) and pd.isna(first)):
        return False
    if _as_float(first) is not None:
        return False
    name = str(first).strip()
    if not name or name.lower() == "nan" or _looks_like_header_row(values):
        return False
    # Owner header: first cell text; remaining cells empty or non-numeric labels.
    rest = list(values[1:3]) if len(values) > 1 else []
    numeric_rest = sum(1 for v in rest if _as_float(v) is not None)
    return numeric_rest == 0


def _collect_name_like_rows(rows: Sequence[Sequence[Any]], start: int = 0) -> List[str]:
    """Collect standalone owner/family title rows (non-numeric first cell)."""
    names: List[str] = []
    for raw in rows[start:]:
        values = list(raw) if raw is not None else []
        if _is_empty_row(values) or not _is_owner_name_row(values):
            continue
        nm = _normalize_owner_display_name(str(values[0]).strip())
        if nm and not _is_placeholder_owner_name(nm):
            names.append(nm)
    return names


def _letter_for_index(idx: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA, ..."""
    n = int(idx)
    if n < 0:
        n = 0
    chars: List[str] = []
    while True:
        chars.append(chr(ord("A") + (n % 26)))
        n = n // 26 - 1
        if n < 0:
            break
    return "".join(reversed(chars))


def _parse_parcels_from_owner_column(
    raw_rows: Sequence[Sequence[Any]],
    colmap: Dict[str, int],
    *,
    start: int = 1,
) -> List[FamilyParcel]:
    """Group a headed Easting/Northing/Pillar/Owner table into parcels."""
    ie = int(colmap.get("easting", 0))
    inn = int(colmap.get("northing", 1))
    ip = colmap.get("pillar")
    io = int(colmap["owner"])
    parcels: List[FamilyParcel] = []
    current_name = ""
    current_pts: List[ParcelPoint] = []

    def _flush() -> None:
        nonlocal current_name, current_pts
        if len(current_pts) >= 3:
            owner = _normalize_owner_display_name(current_name) or f"Parcel {len(parcels) + 1}"
            parcels.append(FamilyParcel(owner_name=owner, points=list(current_pts)))
        current_pts = []

    for raw in raw_rows[start:]:
        values = list(raw) if raw is not None else []
        if _is_empty_row(values):
            _flush()
            current_name = ""
            continue
        e = _as_float(values[ie] if len(values) > ie else None)
        n = _as_float(values[inn] if len(values) > inn else None)
        owner_cell = ""
        if len(values) > io and values[io] is not None:
            owner_cell = str(values[io]).strip()
        owner_norm = _normalize_owner_display_name(owner_cell)
        if e is None or n is None:
            # Standalone name in owner column (or first cell) starts a new block.
            nm = owner_norm or (
                _normalize_owner_display_name(str(values[0]).strip())
                if values and _as_float(values[0]) is None
                else ""
            )
            if nm:
                _flush()
                current_name = nm
            continue
        if owner_norm and owner_norm != _normalize_owner_display_name(current_name):
            if current_pts:
                _flush()
            current_name = owner_norm
        elif not current_name and owner_norm:
            current_name = owner_norm
        pillar = ""
        if ip is not None and len(values) > ip and values[ip] is not None:
            pillar = str(values[ip]).strip()
        current_pts.append(ParcelPoint(e=float(e), n=float(n), pillar=pillar))
    _flush()
    return parcels


def _parse_parcels_family_blocks(raw_rows: Sequence[Sequence[Any]], *, start: int = 0) -> List[FamilyParcel]:
    """Owner/family title row, then Easting/Northing/Pillar rows, blank-separated."""
    parcels: List[FamilyParcel] = []
    current_name = ""
    current_pts: List[ParcelPoint] = []

    def _flush() -> None:
        nonlocal current_name, current_pts
        if len(current_pts) >= 3:
            owner = _normalize_owner_display_name(current_name) or f"Parcel {len(parcels) + 1}"
            parcels.append(FamilyParcel(owner_name=owner, points=list(current_pts)))
        current_pts = []

    for raw in raw_rows[start:]:
        values = list(raw) if raw is not None else []
        if _is_empty_row(values):
            _flush()
            current_name = ""
            continue
        if _is_owner_name_row(values):
            _flush()
            current_name = str(values[0]).strip()
            continue
        e = _as_float(values[0] if len(values) > 0 else None)
        n = _as_float(values[1] if len(values) > 1 else None)
        if e is None or n is None:
            if values and str(values[0]).strip() and str(values[0]).strip().lower() != "nan":
                _flush()
                current_name = str(values[0]).strip()
            continue
        pillar = ""
        if len(values) > 2 and values[2] is not None and str(values[2]).strip():
            pillar = str(values[2]).strip()
        current_pts.append(ParcelPoint(e=float(e), n=float(n), pillar=pillar))
    _flush()
    return parcels


def _parse_blank_separated_point_groups(
    raw_rows: Sequence[Sequence[Any]],
    *,
    start: int = 0,
    names: Optional[Sequence[str]] = None,
) -> List[FamilyParcel]:
    """Fallback: split coordinate blocks by blank rows; attach known names in order."""
    groups: List[List[ParcelPoint]] = []
    cur: List[ParcelPoint] = []
    name_queue = [n for n in (names or []) if n and not _is_placeholder_owner_name(n)]

    def _flush_group() -> None:
        nonlocal cur
        if len(cur) >= 3:
            groups.append(list(cur))
        cur = []

    for raw in raw_rows[start:]:
        values = list(raw) if raw is not None else []
        if _is_empty_row(values):
            _flush_group()
            continue
        if _is_owner_name_row(values):
            _flush_group()
            continue
        e = _as_float(values[0] if len(values) > 0 else None)
        n = _as_float(values[1] if len(values) > 1 else None)
        if e is None or n is None:
            continue
        pillar = ""
        if len(values) > 2 and values[2] is not None and str(values[2]).strip():
            pillar = str(values[2]).strip()
        cur.append(ParcelPoint(e=float(e), n=float(n), pillar=pillar))
    _flush_group()

    parcels: List[FamilyParcel] = []
    for i, pts in enumerate(groups):
        owner = name_queue[i] if i < len(name_queue) else f"Parcel {i + 1}"
        parcels.append(FamilyParcel(owner_name=owner, points=pts))
    return parcels


def parcels_are_placeholder_only(parcels: Sequence[FamilyParcel]) -> bool:
    if not parcels:
        return True
    return all(_is_placeholder_owner_name(p.owner_name) or _is_placeholder_owner_name(p.labeled_name) for p in parcels)


# Back-compat alias used internally.
_parcels_are_placeholder_only = parcels_are_placeholder_only


def parse_family_parcels_from_rows(
    rows: Sequence[Sequence[Any]],
    *,
    source_label: str = "table",
) -> Dict[str, Any]:
    """
    Parse ownership parcels from raw row values (Excel/CSV/text tables).

    Supported layouts:
    - Owner/family name on its own row, then Easting/Northing/Pillar, blank-separated
    - Headed table with an Owner/Buyer column (group by owner)
    """
    raw_rows = [list(r) if r is not None else [] for r in (rows or [])]
    if not raw_rows:
        return {"success": False, "error": f"{source_label} is empty.", "parcels": []}

    start = 0
    colmap: Dict[str, int] = {}
    if raw_rows and _looks_like_header_row(list(raw_rows[0])):
        colmap = _header_column_map(list(raw_rows[0]))
        start = 1

    name_rows = _collect_name_like_rows(raw_rows, start=start)

    parcels: List[FamilyParcel] = []
    # 1) Headed Owner-column workbook (including a previously written dup.xlsx).
    if colmap.get("owner") is not None and colmap.get("easting") is not None and colmap.get("northing") is not None:
        parcels = _parse_parcels_from_owner_column(raw_rows, colmap, start=start)
        # If Owner column is all synthetic "Parcel 1 (A)", treat as unusable names.
        if _parcels_are_placeholder_only(parcels):
            parcels = []

    # 2) Classic family-block layout (title row above each coordinate set).
    if not parcels:
        parcels = _parse_parcels_family_blocks(raw_rows, start=start)

    # 3) Recovery: blank-separated point groups + collected title names.
    if (not parcels or _parcels_are_placeholder_only(parcels)) and name_rows:
        recovered = _parse_blank_separated_point_groups(raw_rows, start=start, names=name_rows)
        if recovered and not _parcels_are_placeholder_only(recovered):
            parcels = recovered
        elif recovered and (not parcels or len(recovered) > len(parcels)):
            parcels = recovered

    # 4) Last resort: one ring from all coordinates (single-parcel sheets only).
    if not parcels:
        pts: List[ParcelPoint] = []
        for raw in raw_rows[start:]:
            values = list(raw) if raw is not None else []
            if _is_empty_row(values) or _is_owner_name_row(values):
                continue
            e = _as_float(values[0] if len(values) > 0 else None)
            n = _as_float(values[1] if len(values) > 1 else None)
            if e is None or n is None:
                continue
            pillar = ""
            if len(values) > 2 and values[2] is not None and str(values[2]).strip():
                pillar = str(values[2]).strip()
            pts.append(ParcelPoint(e=float(e), n=float(n), pillar=pillar))
        if len(pts) >= 3:
            # Prefer blank-separated groups over one mega-parcel when possible.
            grouped = _parse_blank_separated_point_groups(raw_rows, start=start, names=name_rows)
            parcels = grouped if grouped else [FamilyParcel(owner_name="Parcel 1", points=pts)]

    if not parcels:
        return {
            "success": False,
            "error": (
                f"Could not parse ownership parcels from {source_label}. "
                "Expected family/owner name rows, then Easting/Northing/Pillar rows, "
                "separated by blank rows."
            ),
            "parcels": [],
        }

    # If we somehow still have placeholders but title rows exist, rename in order.
    if _parcels_are_placeholder_only(parcels) and name_rows:
        for i, parcel in enumerate(parcels):
            if i < len(name_rows):
                parcel.owner_name = name_rows[i]

    for i, parcel in enumerate(parcels):
        parcel.letter = _letter_for_index(i)

    return {
        "success": True,
        "parcels": parcels,
        "parcel_count": len(parcels),
    }


def parse_family_parcels_from_excel(
    file_path: str | Path,
    *,
    sheet_name: Any = 0,
) -> Dict[str, Any]:
    """
    Parse an Excel workbook of family/ownership parcels.

    Expected layout (with or without a header row):
    - Owner/family name on its own row (top of each set)
    - Following rows: Easting, Northing, Pillar
    - Blank row(s) between ownership parcels
    """
    path = Path(file_path).resolve()
    if not path.exists():
        return {"success": False, "error": f"Excel file not found: {path}", "parcels": []}

    try:
        df = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")
    except Exception as exc:
        return {"success": False, "error": f"Failed to read Excel: {exc}", "parcels": []}

    if df is None or df.empty:
        return {"success": False, "error": "Excel sheet is empty.", "parcels": []}

    parsed = parse_family_parcels_from_rows(
        df.fillna("").values.tolist(),
        source_label=f"Excel '{path.name}'",
    )
    if parsed.get("success"):
        parsed["file_path"] = str(path)
    return parsed


def list_excel_workbooks(folder: str | Path) -> List[Path]:
    """List Excel workbooks in a folder (skips Excel lock temps only)."""
    root = Path(folder).resolve()
    if not root.is_dir():
        return []
    out: List[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        if p.name.startswith("~$"):
            continue
        if p.suffix.lower() not in {".xlsx", ".xls", ".xlsm"}:
            continue
        out.append(p.resolve())
    return out


def assess_ownership_workbook(file_path: str | Path) -> Dict[str, Any]:
    """
    Semantically score a workbook as an ownership/coordinates source.

    Prefers native family-block sheets (owner title rows + E/N/pillar blocks) and
    headed tables with real owner names. Placeholder-only Owner columns score poorly
    regardless of filename.
    """
    path = Path(file_path).resolve()
    result: Dict[str, Any] = {
        "success": False,
        "path": str(path),
        "score": -10_000,
        "parcel_count": 0,
        "real_owner_count": 0,
        "point_count": 0,
        "layout_kind": "unknown",
        "has_family_title_rows": False,
        "has_header_row": False,
        "placeholder_only": True,
        "parcels": [],
    }
    if not path.exists():
        result["error"] = f"Excel file not found: {path}"
        return result
    try:
        df = pd.read_excel(path, sheet_name=0, header=None, engine="openpyxl")
    except Exception as exc:
        result["error"] = f"Failed to read Excel: {exc}"
        return result
    if df is None or df.empty:
        result["error"] = "Excel sheet is empty."
        return result

    raw_rows = df.fillna("").values.tolist()
    has_header = bool(raw_rows and _looks_like_header_row(list(raw_rows[0])))
    colmap = _header_column_map(list(raw_rows[0])) if has_header else {}
    start = 1 if has_header else 0
    title_names = _collect_name_like_rows(raw_rows, start=start)
    parsed = parse_family_parcels_from_rows(raw_rows, source_label=f"Excel '{path.name}'")
    parcels = list(parsed.get("parcels") or []) if parsed.get("success") else []
    real_owners = [
        p for p in parcels if not _is_placeholder_owner_name(p.owner_name) and not _is_placeholder_owner_name(p.labeled_name)
    ]
    point_count = sum(len(p.points) for p in parcels)
    placeholder_only = parcels_are_placeholder_only(parcels) if parcels else True

    if title_names and len(parcels) >= 1:
        layout_kind = "family_blocks"
    elif colmap.get("owner") is not None and not placeholder_only:
        layout_kind = "headed_owner_table"
    elif colmap.get("easting") is not None and colmap.get("northing") is not None:
        layout_kind = "headed_coordinates"
    elif parcels:
        layout_kind = "coordinate_blocks"
    else:
        layout_kind = "unusable"

    score = 0
    score += 25 * len(real_owners)
    score += 3 * min(len(parcels), 40)
    score += min(point_count, 200)
    if layout_kind == "family_blocks":
        score += 80
    elif layout_kind == "headed_owner_table":
        score += 55
    elif layout_kind == "headed_coordinates":
        score += 15
    elif layout_kind == "coordinate_blocks":
        score += 20
    if title_names:
        score += 10 * min(len(title_names), 30)
    if placeholder_only:
        score -= 200
    if not parcels:
        score -= 500

    result.update(
        {
            "success": bool(parcels),
            "score": int(score),
            "parcel_count": len(parcels),
            "real_owner_count": len(real_owners),
            "point_count": int(point_count),
            "layout_kind": layout_kind,
            "has_family_title_rows": bool(title_names),
            "has_header_row": bool(has_header),
            "placeholder_only": bool(placeholder_only),
            "parcels": parcels,
            "owner_names": [p.labeled_name for p in parcels],
            "title_names": title_names,
        }
    )
    if not parcels:
        result["error"] = parsed.get("error") or "No ownership parcels found."
    return result


def choose_best_ownership_workbook(
    candidates: Sequence[str | Path] | str | Path,
    *,
    preferred: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Pick the semantically best ownership/coordinates workbook.

    Filename is irrelevant: a normalized copy with real owners can win; a native
    family-block sheet usually wins; placeholder-only tables lose to siblings.
    """
    if isinstance(candidates, (str, Path)):
        root = Path(candidates)
        cand_paths = list_excel_workbooks(root) if root.is_dir() else [root]
    else:
        cand_paths = [Path(p).resolve() for p in candidates if p]

    assessments = [assess_ownership_workbook(p) for p in cand_paths if Path(p).exists()]
    usable = [a for a in assessments if a.get("success") and not a.get("placeholder_only")]
    pool = usable or [a for a in assessments if a.get("success")] or assessments

    pref = Path(preferred).resolve() if preferred else None
    chosen = None
    if pref is not None:
        for a in assessments:
            if Path(a.get("path") or "").resolve() == pref:
                # Explicit user pick wins when it has usable real owners (or is the only file).
                if a.get("success") and not a.get("placeholder_only"):
                    chosen = a
                elif len(assessments) == 1 and a.get("success"):
                    chosen = a
                break

    if chosen is None and pool:
        chosen = max(pool, key=lambda a: (int(a.get("score") or -10_000), int(a.get("real_owner_count") or 0)))

    if chosen is None:
        return {
            "success": False,
            "error": "No Excel ownership workbook candidates found.",
            "assessments": assessments,
        }
    return {
        "success": True,
        "path": Path(chosen["path"]),
        "assessment": chosen,
        "assessments": assessments,
    }


def better_ownership_excel_nearby(path: str | Path) -> Optional[Path]:
    """Return a better ownership workbook in the same folder, if one exists."""
    src = Path(path).resolve()
    choice = choose_best_ownership_workbook(src.parent, preferred=src)
    if not choice.get("success"):
        return None
    best = Path(choice["path"]).resolve()
    if best == src:
        return None
    src_a = assess_ownership_workbook(src)
    best_a = choice.get("assessment") or {}
    # Only switch when the sibling is meaningfully better (real owners / higher score).
    if src_a.get("placeholder_only") and not best_a.get("placeholder_only"):
        return best
    if int(best_a.get("score") or 0) >= int(src_a.get("score") or 0) + 40:
        return best
    if int(best_a.get("real_owner_count") or 0) > int(src_a.get("real_owner_count") or 0):
        return best
    return None


# Back-compat alias
sibling_source_excel = better_ownership_excel_nearby


def _ensure_excel_basename(name: str) -> Optional[str]:
    """Return a safe Excel basename; append .xlsx when the extension is omitted."""
    raw = (name or "").strip().strip("'\"")
    if not raw:
        return None
    base = Path(raw).name
    if not base or base.lower().startswith("~$"):
        return None
    if Path(base).suffix.lower() not in {".xlsx", ".xls", ".xlsm"}:
        base = f"{base}.xlsx"
    return base


def extract_requested_excel_output_name(query: str) -> Optional[str]:
    """
    Detect a user-requested Excel output filename from natural language.

    Tolerates missing extensions and phrasing like:
    save file as 'coords_midbelt' excel file / save as owners_clean.xlsx
    """
    q = query or ""
    patterns = (
        # Quoted name, optional "excel/workbook/spreadsheet file" trailer
        r"(?:duplicate|copy|save|write|export)\s+(?:(?:it|file|workbook|the\s+file)\s+)?"
        r"(?:and\s+)?(?:save\s+)?(?:it\s+)?"
        r"(?:as|to)\s+['\"]([^'\"]+)['\"]"
        r"(?:\s+(?:excel|workbook|spreadsheet)(?:\s+file)?)?",
        # Unquoted name with explicit Excel extension
        r"(?:duplicate|copy|save|write|export)\s+(?:(?:it|file|workbook|the\s+file)\s+)?"
        r"(?:and\s+)?(?:save\s+)?(?:it\s+)?"
        r"(?:as|to)\s+([A-Za-z0-9 _\-]+\.(?:xlsx|xls|xlsm))\b",
        # Unquoted stem + excel/workbook/spreadsheet file
        r"(?:duplicate|copy|save|write|export)\s+(?:(?:it|file|workbook|the\s+file)\s+)?"
        r"(?:and\s+)?(?:save\s+)?(?:it\s+)?"
        r"(?:as|to)\s+([A-Za-z0-9 _\-]+)\s+(?:excel|workbook|spreadsheet)(?:\s+file)?\b",
        r"(?:save|write|export)\s+(?:a\s+)?(?:copy|duplicate)\s+(?:as|to)\s+"
        r"['\"]?([^'\"\s]+)['\"]?",
    )
    for pat in patterns:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if m:
            name = _ensure_excel_basename(m.group(1))
            if name:
                return name
    return None


def extract_requested_workbook_copy_name(query: str) -> Optional[str]:
    """
    Detect a user-requested normalized/copy workbook filename from natural language.

    Examples: save as 'owners_clean.xlsx', duplicate it as coords.xlsx, copy to book.xlsx.
    """
    return extract_requested_excel_output_name(query)


def query_requests_workbook_copy(query: str) -> bool:
    """True when the user asked to duplicate/normalize the workbook (any output name)."""
    q = (query or "").lower()
    if extract_requested_workbook_copy_name(query):
        return True
    return bool(
        re.search(
            r"\b(duplicate|make\s+a\s+copy|save\s+a\s+copy|normalize|add\s+(?:the\s+)?"
            r"(?:headers?|titles?)|with\s+headers?)\b",
            q,
        )
    )


def resolve_ownership_excel_for_plot(
    query: str,
    workspace: str | Path,
    *,
    preferred: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Resolve the best ownership workbook for plotting from query + workspace semantics.

    Plotting can proceed directly from the chosen input file; a normalized copy is
    optional and only written when the user asked for one (any filename).
    """
    ws = Path(workspace).resolve()
    mentioned: List[Path] = []
    q = query or ""
    for m in re.finditer(r"['\"]([^'\"]+\.(?:xlsx|xls|xlsm))['\"]", q, flags=re.IGNORECASE):
        p = Path(m.group(1).strip())
        if not p.is_absolute():
            p = (ws / p).resolve()
        if p.exists():
            mentioned.append(p)
    for m in re.finditer(r"([A-Za-z]:\\[^\s'\"]+\.(?:xlsx|xls|xlsm))", q, flags=re.IGNORECASE):
        p = Path(m.group(1)).resolve()
        if p.exists():
            mentioned.append(p)

    preferred_path = Path(preferred).resolve() if preferred else (mentioned[0] if mentioned else None)
    # Candidate pool: workspace workbooks + any explicit mentions.
    pool = {str(p.resolve()).lower(): p.resolve() for p in list_excel_workbooks(ws)}
    for p in mentioned:
        pool[str(p.resolve()).lower()] = p.resolve()
    if preferred_path is not None and preferred_path.exists():
        pool[str(preferred_path).lower()] = preferred_path

    if not pool:
        return {
            "success": False,
            "error": (
                "Could not find an Excel ownership workbook in the workspace. "
                "Place the coordinates file in the active workspace (or quote its path)."
            ),
        }

    choice = choose_best_ownership_workbook(list(pool.values()), preferred=preferred_path)
    if not choice.get("success"):
        return {
            "success": False,
            "error": choice.get("error") or "Could not choose an ownership workbook.",
            "assessments": choice.get("assessments") or [],
        }

    assessment = choice["assessment"]
    parcels = list(assessment.get("parcels") or [])
    if parcels_are_placeholder_only(parcels):
        return {
            "success": False,
            "error": (
                "Ownership names resolved only to placeholders like 'Parcel 1 (A)'. "
                "Provide a workbook with real owner/family title rows (or a headed Owner "
                "column with real names), then retry."
            ),
            "path": choice["path"],
            "assessment": assessment,
            "assessments": choice.get("assessments") or [],
        }

    copy_name = extract_requested_workbook_copy_name(q)
    return {
        "success": True,
        "path": Path(choice["path"]),
        "parcels": parcels,
        "assessment": assessment,
        "assessments": choice.get("assessments") or [],
        "write_copy": bool(query_requests_workbook_copy(q)),
        "copy_name": copy_name,
    }


def normalize_ownership_workbook(
    source_path: str | Path | None = None,
    *,
    workspace: str | Path | None = None,
    dest_name: Optional[str] = None,
    query: str = "",
) -> Dict[str, Any]:
    """
    Normalize a family/owner-block (or headed ownership) workbook to
    Easting/Northing/Pillar/Owner for CRS conversion and GIS tools.

    General helper — not CAD-specific. Use whenever inspect shows Unnamed columns
    or owner-like first headers before excel_coordinate_converter / ArcGIS import.
    """
    ws = Path(workspace or Path.cwd()).resolve()
    src: Optional[Path] = None
    if source_path:
        p = Path(source_path)
        if not p.is_absolute():
            p = (ws / p).resolve()
        else:
            p = p.resolve()
        if p.exists():
            src = p
        else:
            # Soft resolve by stem in workspace (e.g. ODUOHA_FAMILY_BOUNDARY1 → *_1.xlsx).
            stem = p.stem.lower().replace(" ", "_")
            for cand in list_excel_workbooks(ws):
                cstem = cand.stem.lower().replace(" ", "_")
                if cstem == stem or stem in cstem or cstem in stem:
                    src = cand
                    break
    if src is None:
        resolved = resolve_ownership_excel_for_plot(query or "", ws, preferred=source_path)
        if not resolved.get("success"):
            return {
                "success": False,
                "error": resolved.get("error")
                or "Could not resolve an ownership Excel workbook to normalize.",
            }
        src = Path(resolved["path"])
        parcels = list(resolved.get("parcels") or [])
    else:
        parcels = None

    out = write_dup_xlsx_with_headers(
        src,
        dest_name=dest_name,
        parcels=parcels,
        query=query or "",
    )
    if out.get("success"):
        out["layout_kind"] = "ownership_normalized"
        out["hint"] = (
            "Use columns Easting/Northing (and Owner) for excel_coordinate_converter "
            "or ArcGIS XY import. Do not re-parse family blocks inside ArcPy."
        )
    return out


def write_dup_xlsx_with_headers(
    source_path: str | Path,
    *,
    dest_name: Optional[str] = None,
    parcels: Optional[Sequence[FamilyParcel]] = None,
    query: str = "",
) -> Dict[str, Any]:
    """
    Write a normalized Easting/Northing/Pillar/Owner workbook.

    Destination name comes from ``dest_name``, else a name requested in ``query``,
    else ``ownership_normalized.xlsx``. Parsing always prefers the semantically best
    nearby ownership source when the provided path has placeholder owners.
    """
    src = Path(source_path).resolve()
    requested = (dest_name or "").strip() or extract_requested_workbook_copy_name(query) or ""
    if not requested:
        requested = "ownership_normalized.xlsx"
    dest = (src.parent / Path(requested).name).resolve()
    try:
        parse_src = src
        # If this path is weak/placeholder, switch to a better nearby ownership sheet.
        if parcels is None or parcels_are_placeholder_only(list(parcels or [])):
            choice = choose_best_ownership_workbook(src.parent, preferred=src)
            if choice.get("success"):
                best = Path(choice["path"]).resolve()
                best_a = choice.get("assessment") or {}
                if not best_a.get("placeholder_only"):
                    parse_src = best
                    if parcels is None or parcels_are_placeholder_only(list(parcels or [])):
                        parcels = list(best_a.get("parcels") or [])

        if parcels is None or parcels_are_placeholder_only(list(parcels or [])):
            parsed = parse_family_parcels_from_excel(parse_src)
            if not parsed.get("success"):
                return parsed
            parcels = parsed["parcels"]
            if parcels_are_placeholder_only(list(parcels or [])):
                return {
                    "success": False,
                    "error": (
                        f"Parsed only placeholder owners from '{parse_src.name}'. "
                        "Use a workbook with real owner/family title rows or a headed "
                        "Owner column containing real names."
                    ),
                }

        rows: List[Dict[str, Any]] = []
        for parcel in parcels:
            for pt in parcel.points:
                rows.append(
                    {
                        "Easting": float(pt.e),
                        "Northing": float(pt.n),
                        "Pillar": pt.pillar or "",
                        "Owner": parcel.labeled_name,
                    }
                )
        out_df = pd.DataFrame(rows, columns=["Easting", "Northing", "Pillar", "Owner"])
        out_df.to_excel(dest, index=False, engine="openpyxl")
        owners = sorted({str(r["Owner"]) for r in rows})
        return {
            "success": True,
            "output_path": str(dest),
            "rows": len(rows),
            "owner_count": len(owners),
            "owners": owners,
            "source_path": str(parse_src),
        }
    except Exception as exc:
        logger.exception("Failed writing %s", dest)
        return {"success": False, "error": str(exc)}


def find_sole_excel_in_folder(folder: str | Path) -> Optional[Path]:
    """Return the best (or only) ownership Excel workbook in folder."""
    choice = choose_best_ownership_workbook(folder)
    if choice.get("success"):
        return Path(choice["path"])
    cands = list_excel_workbooks(folder)
    return cands[0] if len(cands) == 1 else None


def find_excel_from_query(query: str, workspace: str | Path) -> Optional[Path]:
    """Resolve an Excel path from query mentions and semantic ownership quality."""
    resolved = resolve_ownership_excel_for_plot(query, workspace)
    if resolved.get("success"):
        return Path(resolved["path"])
    # Soft fallback: any sole workbook even if owners are weak (caller may recover).
    return find_sole_excel_in_folder(workspace)


def find_reference_dwg_from_query(query: str, workspace: str | Path) -> Optional[Path]:
    """Find a source plan DWG named in the query (not the Generate output)."""
    q = query or ""
    ws = Path(workspace).resolve()
    gen_names = {
        Path(m.group(1)).name.lower()
        for m in re.finditer(
            r"\b(?:generate|create|produce)\s*[-]?\s*"
            r"(?:cad\s+drawing\s+|cad\s+|drawing\s+|file\s+)?['\"]?"
            r"([^'\"\s]+?\.dwg)",
            q,
            flags=re.IGNORECASE,
        )
    }
    candidates: List[Path] = []
    for m in re.finditer(r"['\"]([^'\"]+?\.dwg)['\"]", q, flags=re.IGNORECASE):
        raw = m.group(1).strip()
        p = Path(raw)
        if not p.is_absolute():
            p = (ws / p).resolve()
        if p.name.lower() in gen_names:
            continue
        if p.exists():
            candidates.append(p)
    if candidates:
        return candidates[0]
    # Unquoted basename mention
    for m in re.finditer(r"\b([A-Za-z0-9 _\-]+\.dwg)\b", q, flags=re.IGNORECASE):
        name = m.group(1).strip()
        if name.lower() in gen_names:
            continue
        p = (ws / name).resolve()
        if p.exists():
            return p
    return None


def format_absolute_coordinates_blob(points: Sequence[ParcelPoint]) -> str:
    parts = [f"({pt.e:.3f}mE, {pt.n:.3f}mN)" for pt in points]
    return "; ".join(parts)


_YEAR_TOKEN_RE = re.compile(r"^(?:19|20)\d{2}$")

# Cap runaway Excel batches; same practical ceiling as the inline batch fastpath.
_MAX_SEPARATE_OWNER_PLANS = 25


def wants_multi_parcel_layout(query: str) -> bool:
    """
    True when the user explicitly wants all owners on ONE cadastral sheet
    (letter tags A/B/C… marking parcels within that single plan).
    """
    q = (query or "").lower()
    if not q:
        return False
    markers = (
        "all the owner",
        "all owner names",
        "all the buyers",
        "within the plan showing each owner",
        "mark the parcels within",
        "parcels within the plan",
        "multi-parcel",
        "multiparcel",
        "multi parcel",
        "combined plan",
        "same plan",
        "on the same drawing",
        "on one plan",
        "one cad plan with",
        "single cad plan",
        "letters such as",
        "in bracket after their names",
        "in brackets after their names",
        "letter in bracket",
        "letters in bracket",
    )
    return any(m in q for m in markers)


def wants_separate_owner_plans(query: str) -> bool:
    """
    True when the user wants a distinct CAD .dwg per owner/family block
    (each owner's coordinates plot only that owner's plan).

    Explicit multi-parcel language wins (returns False) so the classic
    "all owners on one sheet with AMADI (B) labels" route stays intact.
    """
    q = (query or "").lower()
    if not q:
        return False
    if wants_multi_parcel_layout(q):
        return False

    patterns: Tuple[str, ...] = (
        r"different\s+cad\s+plans?",
        r"different\s+(?:cad\s+)?(?:drawings?|dwgs?)",
        r"separate\s+(?:cad\s+)?(?:plans?|drawings?|dwgs?)",
        r"individual\s+(?:cad\s+)?(?:plans?|drawings?|dwgs?)",
        r"unique\s+(?:cad\s+)?(?:plans?|drawings?|dwgs?)",
        r"(?:one|a)\s+plan\s+per\s+(?:owner|buyer|family)",
        r"per[- ]owner\s+(?:cad\s+)?(?:plans?|drawings?)",
        r"each\s+(?:owner|buyer|family|set).{0,100}"
        r"(?:only\s+that|its?\s+own|that\s+(?:user|owner|buyer)\s+plan)",
        r"plot\s+only\s+that\s+(?:user|owner|buyer)\s+plan",
        r"each\s+set\s+of\s+coordinates.{0,120}(?:only\s+that|its?\s+own)",
        r"(?:each|every)\s+(?:buyer|owner|family)\s+name\s+as\s+the\s+names?\s+"
        r"of\s+that\s+particular",
        r"buyer\s+name\s+as\s+the\s+names?\s+of\s+that\s+particular\s+cad",
        r"with\s+each\s+(?:buyer|owner)\s+name\s+as\s+the\s+names?",
        r"generate\s+\.dwg\s+files",
        r"\.dwg\s+files?\b.{0,80}each\s+(?:buyer|owner)",
        r"if\s+there\s+are\s+\d+\s+different\s+names",
        r"plot\s+\d+\s+different\s+cad\s+plans?",
        r"\d+\s+different\s+cad\s+plans?",
        r"increment\s+on\s+each\s+plan",
        r"plan\s+[ab]\s+should\s+be",
        r"plan\s+number.{0,40}increment",
    )
    return any(re.search(p, q, flags=re.IGNORECASE) for p in patterns)


def owner_plan_dwg_basename(owner_name: str) -> str:
    """Safe ``Owner_Name.dwg`` basename from a buyer/family title."""
    raw = (owner_name or "").strip()
    raw = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = raw.replace(" ", "_")
    raw = raw.strip("._")
    if not raw:
        raw = "Owner"
    if len(raw) > 80:
        raw = raw[:80].rstrip("._") or "Owner"
    return f"{raw}.dwg"


def increment_plan_number(base: str, offset: int) -> str:
    """
    Increment the first sequential numeric segment in a plan number by ``offset``.

    Preserves zero-padding and skips 4-digit years (19xx/20xx).
    Examples: ``RV/001/2026/SP`` + 1 → ``RV/002/2026/SP``.
    """
    s = (base or "").strip()
    if not s or int(offset) <= 0:
        return s
    tokens = re.split(r"([/\\\-_ ])", s)
    for i, tok in enumerate(tokens):
        if tok.isdigit() and not _YEAR_TOKEN_RE.match(tok):
            n = int(tok) + int(offset)
            tokens[i] = str(n).zfill(len(tok))
            return "".join(tokens)
    # Fallback: last non-year digit run anywhere in the string.
    matches = [
        m for m in re.finditer(r"\d+", s) if not _YEAR_TOKEN_RE.match(m.group(0))
    ]
    if not matches:
        return s
    m = matches[-1]
    n = int(m.group(0)) + int(offset)
    return s[: m.start()] + str(n).zfill(len(m.group(0))) + s[m.end() :]


def build_separate_owner_plan_jobs(
    *,
    parcels: Sequence[FamilyParcel],
    workspace: str | Path,
    location: str = "",
    lga: str = "",
    state: str = "",
    origin_crs: str = "",
    plan_number: str = "",
    surveyor_name: str = "",
    surveyor_address: str = "",
    certification_date: str = "",
    template_path: Optional[str] = None,
    scale_denom: Optional[int] = None,
    max_plans: int = _MAX_SEPARATE_OWNER_PLANS,
) -> Dict[str, Any]:
    """
    Build one single-parcel cadastral sub-prompt job per owner.

    Each job plots only that owner's ring; buyer name is the owner (no A/B/C
    multi-parcel letter tags); output DWG basename is the owner name; plan
    numbers increment from the shared base (owner 0 keeps the base number).
    """
    ws = Path(workspace).resolve()
    usable = [p for p in parcels if p is not None and len(p.points) >= 3]
    if not usable:
        return {"success": False, "error": "No parcels with at least 3 points.", "jobs": []}

    cap = max(1, int(max_plans or _MAX_SEPARATE_OWNER_PLANS))
    used_names: set[str] = set()
    jobs: List[Dict[str, Any]] = []
    for i, parcel in enumerate(usable[:cap]):
        base_name = owner_plan_dwg_basename(parcel.owner_name)
        stem = Path(base_name).stem
        suffix = Path(base_name).suffix or ".dwg"
        candidate = base_name
        n = 2
        while candidate.lower() in used_names:
            candidate = f"{stem}_{n}{suffix}"
            n += 1
        used_names.add(candidate.lower())
        out_path = str((ws / candidate).resolve())
        plan_no = increment_plan_number(plan_number, i) if plan_number else ""
        # Standalone owner plan: keep full normalized owner title (e.g. OKACHI FAMILY),
        # strip multi-parcel letter tags only.
        owner_full = _normalize_owner_display_name(parcel.owner_name) or (
            (parcel.owner_name or "").strip() or f"Owner_{i + 1}"
        )
        solo = FamilyParcel(
            owner_name=owner_full,
            letter="",
            points=list(parcel.points),
        )
        composed = build_excel_cadastral_subprompt(
            output_dwg=out_path,
            parcels=[solo],
            location=location,
            lga=lga,
            state=state,
            origin_crs=origin_crs,
            plan_number=plan_no,
            surveyor_name=surveyor_name,
            surveyor_address=surveyor_address,
            certification_date=certification_date,
            template_path=template_path,
            scale_denom=scale_denom,
        )
        if not composed.get("success"):
            return {
                "success": False,
                "error": composed.get("error")
                or f"Failed to compose subprompt for '{solo.owner_name}'.",
                "jobs": jobs,
            }
        jobs.append(
            {
                "owner_name": solo.owner_name,
                "output_dwg": out_path,
                "plan_number": plan_no,
                "subprompt": composed["subprompt"],
                "buyer_name": composed.get("buyer_name") or solo.owner_name,
                "parcel_index": i,
                "scale_denom": scale_denom,
            }
        )

    return {
        "success": True,
        "jobs": jobs,
        "plan_count": len(jobs),
        "truncated": len(usable) > cap,
        "usable_parcel_count": len(usable),
    }


def _quantize_xy(x: float, y: float, quantize_m: float = 0.001) -> Tuple[float, float]:
    q = float(quantize_m) if quantize_m and quantize_m > 0 else 0.001
    return (round(float(x) / q) * q, round(float(y) / q) * q)


def build_multi_parcel_layout_draw_ops(
    parcels: Sequence[Dict[str, Any]],
    *,
    quantize_m: float = 0.001,
) -> Dict[str, Any]:
    """
    Cartographic draw plan for complex multi-parcel layouts.

    Input parcels: ``[{"label": str, "points": [{"x":..,"y":..}, ...]}, ...]``
    in a single drawing coordinate space (local/template or absolute).

    Returns unique peg insertion points (shared corners once), undirected traverse
    edges (shared boundaries once; crossings allowed), and one label per parcel.
    Does not invent geometry — preserves each parcel's traverse order, including
    the closing edge from last→first when those vertices differ.
    """
    pegs: List[Dict[str, float]] = []
    peg_keys: set = set()
    edges: List[Dict[str, Any]] = []
    edge_keys: set = set()
    labels: List[Dict[str, Any]] = []

    for parcel in parcels or []:
        if not isinstance(parcel, dict):
            continue
        raw_pts = list(parcel.get("points") or [])
        ring: List[Dict[str, float]] = []
        for p in raw_pts:
            if not isinstance(p, dict):
                continue
            try:
                x = float(p.get("x", p.get("e")))
                y = float(p.get("y", p.get("n")))
            except Exception:
                continue
            ring.append({"x": x, "y": y})
        if len(ring) < 2:
            continue

        for p in ring:
            key = _quantize_xy(p["x"], p["y"], quantize_m)
            if key in peg_keys:
                continue
            peg_keys.add(key)
            pegs.append({"x": float(key[0]), "y": float(key[1])})

        n = len(ring)
        # Closed traverse: n edges (including last→first). Open polylines (<3) get n-1.
        edge_count = n if n >= 3 else max(0, n - 1)
        for i in range(edge_count):
            p1 = ring[i]
            p2 = ring[(i + 1) % n]
            k1 = _quantize_xy(p1["x"], p1["y"], quantize_m)
            k2 = _quantize_xy(p2["x"], p2["y"], quantize_m)
            if k1 == k2:
                continue
            ek = (k1, k2) if k1 <= k2 else (k2, k1)
            if ek in edge_keys:
                continue
            edge_keys.add(ek)
            edges.append(
                {
                    "a": {"x": float(ek[0][0]), "y": float(ek[0][1])},
                    "b": {"x": float(ek[1][0]), "y": float(ek[1][1])},
                }
            )

        lab = str(parcel.get("label") or "").strip()
        if lab and len(ring) >= 3:
            cx = sum(p["x"] for p in ring) / float(len(ring))
            cy = sum(p["y"] for p in ring) / float(len(ring))
            labels.append({"label": lab, "x": float(cx), "y": float(cy)})

    return {
        "pegs": pegs,
        "edges": edges,
        "labels": labels,
        "peg_count": len(pegs),
        "edge_count": len(edges),
        "label_count": len(labels),
    }


def build_excel_cadastral_subprompt(
    *,
    output_dwg: str,
    parcels: Sequence[FamilyParcel],
    location: str = "",
    lga: str = "",
    state: str = "",
    origin_crs: str = "",
    plan_number: str = "",
    surveyor_name: str = "",
    surveyor_address: str = "",
    certification_date: str = "",
    template_path: Optional[str] = None,
    scale_denom: Optional[int] = None,
    main_parcel_index: int = 0,
) -> Dict[str, Any]:
    """
    Build a cadastral sub-prompt from Excel parcels + metadata.

    Main parcel geometry drives the title-block traverse; remaining parcels are
    returned as overlay specs (absolute E/N rings + owner labels).
    """
    if not parcels:
        return {"success": False, "error": "No parcels provided."}

    idx = max(0, min(int(main_parcel_index), len(parcels) - 1))
    # Prefer the parcel with the most vertices for a stable primary ring.
    if main_parcel_index == 0 and len(parcels) > 1:
        idx = max(range(len(parcels)), key=lambda i: len(parcels[i].points))

    main = parcels[idx]
    if len(main.points) < 3:
        return {"success": False, "error": f"Main parcel '{main.owner_name}' has fewer than 3 points."}

    buyer = ", ".join(p.labeled_name for p in parcels if p.labeled_name)
    pillars = [pt.pillar for pt in main.points if pt.pillar]
    # Fill missing pillar tokens so counts stay aligned with vertices.
    while len(pillars) < len(main.points):
        pillars.append(f"P{len(pillars) + 1}")

    lines: List[str] = []
    if template_path:
        lines.append(f"template '{template_path}'")
    lines.append(f"Generate '{output_dwg}'")
    lines.append(f"buyer name: {buyer}")
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
        # Place scale BEFORE surveyor so greedy address captures cannot swallow it.
        lines.append(f"Plot using scale 1:{sd}")
    if surveyor_name:
        lines.append(f"Surveyor name: {surveyor_name}")
    if surveyor_address:
        lines.append(f"Surveyor company and address: {surveyor_address}")
    lines.append("pillar numbers: " + ", ".join(pillars))
    lines.append(
        "coordinates for the points = " + format_absolute_coordinates_blob(main.points)
    )

    extras: List[Dict[str, Any]] = []
    for i, parcel in enumerate(parcels):
        if i == idx or len(parcel.points) < 3:
            continue
        extras.append(
            {
                "label": parcel.labeled_name,
                "points": [{"e": pt.e, "n": pt.n, "pillar": pt.pillar} for pt in parcel.points],
            }
        )

    extent_points = [{"e": pt.e, "n": pt.n} for p in parcels for pt in p.points]
    # Always label the main parcel too when multiple owners exist.
    main_label = main.labeled_name if len(parcels) > 1 else ""

    return {
        "success": True,
        "subprompt": "\n".join(lines),
        "extra_parcels": extras,
        "extent_points": extent_points,
        "main_parcel_label": main_label,
        "buyer_name": buyer,
        "main_parcel_index": idx,
        "parcels": parcels,
    }
