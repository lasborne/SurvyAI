"""Pure helpers for chat attachment markers (no Qt / GUI dependency)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

ATTACHMENTS_START = "[SurvyAI attachments]"
ATTACHMENTS_END = "[/SurvyAI attachments]"
ATTACHED_INPUTS_START = "[SurvyAI attached inputs]"
ATTACHED_INPUTS_END = "[/SurvyAI attached inputs]"

IMAGE_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".jfif",
        ".webp",
        ".tif",
        ".tiff",
        ".bmp",
        ".gif",
        ".jp2",
        ".j2k",
    }
)
DOCUMENT_EXTENSIONS = frozenset({".pdf", ".docx", ".doc", ".rtf", ".odt"})
SPREADSHEET_EXTENSIONS = frozenset(
    {".xlsx", ".xls", ".xlsm", ".xlsb", ".csv", ".tsv", ".ods"}
)
PRESENTATION_EXTENSIONS = frozenset({".pptx", ".ppt", ".ppsx", ".odp"})
TEXT_EXTENSIONS = frozenset(
    {".txt", ".md", ".rst", ".log", ".xml", ".html", ".htm", ".ini", ".cfg", ".wkt"}
)
CAD_EXTENSIONS = frozenset(
    {".dwg", ".dxf", ".dwf", ".dwfx", ".dgn", ".dwt", ".dxb", ".sat", ".igs", ".iges", ".stp", ".step"}
)
GIS_EXTENSIONS = frozenset(
    {
        ".shp",
        ".shx",
        ".dbf",
        ".prj",
        ".cpg",
        ".qpj",
        ".sbn",
        ".sbx",
        ".gpkg",
        ".geojson",
        ".json",
        ".kml",
        ".kmz",
        ".gpx",
        ".gml",
        ".mif",
        ".mid",
        ".tab",
        ".aprx",
        ".lyrx",
        ".lyr",
        ".mxd",
        ".qgz",
        ".qgs",
        ".sxd",
        ".las",
        ".laz",
        ".e57",
        ".lasd",
        ".asc",
        ".dem",
        ".xyz",
        ".adf",
        ".ntf",
        ".nitf",
        ".sid",
        ".ecw",
        ".img",
    }
)
SURVEY_DATA_EXTENSIONS = frozenset(
    {
        ".raw",
        ".job",
        ".jxl",
        ".dc",
        ".rw5",
        ".fbk",
        ".gsi",
        ".sdr",
        ".crd",
        ".dat",
        ".pts",
        ".ptx",
        ".obs",
        ".t02",
        ".t04",
    }
)
ARCHIVE_EXTENSIONS = frozenset({".zip"})

ATTACHABLE_EXTENSIONS = (
    IMAGE_EXTENSIONS
    | DOCUMENT_EXTENSIONS
    | SPREADSHEET_EXTENSIONS
    | PRESENTATION_EXTENSIONS
    | TEXT_EXTENSIONS
    | CAD_EXTENSIONS
    | GIS_EXTENSIONS
    | SURVEY_DATA_EXTENSIONS
    | ARCHIVE_EXTENSIONS
)

# Images stay on the OCR size cap; CAD/GIS/point-cloud files are often larger.
DEFAULT_IMAGE_MAX_FILE_MB = 10
DEFAULT_DATA_MAX_FILE_MB = 100

DEFAULT_ATTACHMENTS_ONLY_PROMPT = "Process the attached file(s)."


def normalize_user_path(raw: str) -> str:
    """
    Canonicalize a user-typed or pasted filesystem path.

    Handles:
    - ``file:///C:/Users/...`` and percent-encoded file URIs
    - forward-slash Windows paths (``C:/Users/...``)
    - backslash Windows paths (``C:\\Users\\...``)
    - quoted / trailing-punctuation wrappers
    """
    s = str(raw or "").strip().strip("\"'").rstrip(").,;")
    if not s:
        return ""
    if s.lower().startswith("file:"):
        try:
            parsed = urlparse(s)
            # urlparse('file:///C:/Users/a.pdf').path → '/C:/Users/a.pdf' on Windows
            s = unquote(parsed.path or "")
            if parsed.netloc and not re.match(r"^[A-Za-z]:", s.lstrip("/\\")):
                # file://server/share → \\server\share
                s = f"//{parsed.netloc}{s}"
        except Exception:
            s = unquote(re.sub(r"(?i)^file:(?://+)?", "", s))
        s = s.strip()
    # Strip any leading slashes/backslashes before a Windows drive letter:
    # /C:/Users, ///C:/Users, \C:\Users → C:/Users or C:\Users
    s = re.sub(r"^[/\\]+([A-Za-z]:)", r"\1", s)
    # Repair drive-relative mangling: C:Users\... → C:\Users\...
    s = re.sub(r"^([A-Za-z]:)(?![/\\])", r"\1\\", s)
    if not s:
        return ""
    try:
        p = Path(s)
        if re.match(r"^[A-Za-z]:[/\\]", s):
            return str(p.resolve()) if p.exists() else str(p)
        if p.exists():
            return str(p.resolve())
        if p.is_absolute():
            return str(p)
    except Exception:
        pass
    return s

_KIND_LABELS = {
    "image": "image / scan",
    "spreadsheet": "spreadsheet / table",
    "document": "document",
    "presentation": "presentation",
    "text": "text",
    "cad": "CAD drawing",
    "gis": "GIS / geospatial",
    "survey_data": "survey / instrument data",
    "archive": "archive",
    "file": "file",
}

_MARKER_RE = re.compile(
    r"\[SurvyAI attachments\]\s*(.*?)\s*\[/SurvyAI attachments\]",
    re.IGNORECASE | re.DOTALL,
)
_INPUTS_RE = re.compile(
    r"\[SurvyAI attached inputs\]\s*.*?\s*\[/SurvyAI attached inputs\]",
    re.IGNORECASE | re.DOTALL,
)


def _suffix(path: str | Path) -> str:
    try:
        return Path(path).suffix.lower()
    except Exception:
        return ""


def is_attachable_path(path: str | Path) -> bool:
    return _suffix(path) in ATTACHABLE_EXTENSIONS


def is_image_path(path: str | Path) -> bool:
    return _suffix(path) in IMAGE_EXTENSIONS


def attachment_kind(path: str | Path) -> str:
    ext = _suffix(path)
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in SPREADSHEET_EXTENSIONS:
        return "spreadsheet"
    if ext in CAD_EXTENSIONS:
        return "cad"
    if ext in GIS_EXTENSIONS:
        return "gis"
    if ext in DOCUMENT_EXTENSIONS:
        return "document"
    if ext in PRESENTATION_EXTENSIONS:
        return "presentation"
    if ext in TEXT_EXTENSIONS:
        return "text"
    if ext in SURVEY_DATA_EXTENSIONS:
        return "survey_data"
    if ext in ARCHIVE_EXTENSIONS:
        return "archive"
    return "file"


def attachment_kind_label(path: str | Path) -> str:
    return _KIND_LABELS.get(attachment_kind(path), "file")


def max_file_mb_for_path(path: str | Path, *, image_max_mb: int = DEFAULT_IMAGE_MAX_FILE_MB) -> int:
    """Per-type size cap: keep the image/OCR limit; allow larger CAD/GIS/data files."""
    img_cap = max(1, int(image_max_mb or DEFAULT_IMAGE_MAX_FILE_MB))
    if is_image_path(path):
        return img_cap
    return max(img_cap, DEFAULT_DATA_MAX_FILE_MB)


def default_prompt_for_attachments(paths: Sequence[str]) -> str:
    """Fallback user text when only files are attached (no typed prompt)."""
    kinds = {attachment_kind(p) for p in paths if str(p).strip()}
    if not kinds or kinds <= {"image"}:
        return DEFAULT_ATTACHMENTS_ONLY_PROMPT
    if kinds == {"spreadsheet"}:
        return (
            "Use the attached spreadsheet as the input file. "
            "Inspect its sheets/columns, then perform the requested conversion or analysis."
        )
    if kinds == {"cad"}:
        return "Use the attached CAD drawing as the input file for this request."
    if kinds == {"gis"}:
        return "Use the attached GIS / geospatial file as the input for this request."
    if kinds == {"document"}:
        return "Use the attached document as the input file for this request."
    if kinds == {"survey_data"}:
        return "Use the attached survey / instrument file as the input for this request."
    return "Use the attached file(s) as the input(s) for this request."


def file_dialog_filters() -> str:
    """Qt QFileDialog filter string (grouped by surveying/geospatial use)."""

    def _pat(exts: Iterable[str]) -> str:
        return " ".join(f"*{e}" for e in sorted(exts))

    all_pat = _pat(ATTACHABLE_EXTENSIONS)
    return (
        f"All supported files ({all_pat});;"
        f"Images and rasters ({_pat(IMAGE_EXTENSIONS)});;"
        f"Spreadsheets and tables ({_pat(SPREADSHEET_EXTENSIONS)});;"
        f"CAD drawings ({_pat(CAD_EXTENSIONS)});;"
        f"GIS / geospatial ({_pat(GIS_EXTENSIONS)});;"
        f"Survey / instrument data ({_pat(SURVEY_DATA_EXTENSIONS)});;"
        f"Documents ({_pat(DOCUMENT_EXTENSIONS)});;"
        f"Presentations ({_pat(PRESENTATION_EXTENSIONS)});;"
        f"Text files ({_pat(TEXT_EXTENSIONS)});;"
        f"Archives ({_pat(ARCHIVE_EXTENSIONS)});;"
        "All files (*.*)"
    )


def attach_unsupported_message(suffix: str = "") -> str:
    shown = (suffix or "").strip() or "(no extension)"
    return (
        f"Unsupported type: {shown}\n"
        "Attach images, spreadsheets (.xlsx/.csv), CAD (.dwg/.dxf/.dwf), "
        "GIS (.shp/.gpkg/.kml/.las), documents (.pdf/.docx), "
        "presentations, text, or survey instrument files."
    )


def format_attachments_block(paths: Sequence[str], user_text: str = "") -> str:
    """Build the agent-facing query with an optional attachment marker block."""
    cleaned: List[str] = []
    seen = set()
    for raw in paths:
        p = str(raw or "").strip().strip('"').strip("'")
        if not p:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(p)

    text = (user_text or "").strip()
    if not cleaned:
        return text
    if not text:
        text = default_prompt_for_attachments(cleaned)

    guide_lines = [
        ATTACHED_INPUTS_START,
        "These files were attached with the + button (or drag-and-drop / paste).",
        "They ARE the intended input files. Use these exact paths in tools.",
        "Do not ask the user to paste or re-type these paths.",
    ]
    for p in cleaned:
        name = Path(p).name or p
        guide_lines.append(f"- {name} ({attachment_kind_label(p)}) → {p}")
    guide_lines.append(ATTACHED_INPUTS_END)

    lines = [ATTACHMENTS_START, *cleaned, ATTACHMENTS_END, "", *guide_lines, "", text]
    return "\n".join(lines)


def parse_attachments_block(query: str) -> Tuple[List[str], str]:
    """
    Split a query into (attachment_paths, remaining_user_text).

    Returns empty paths when no marker block is present.
    The guidance block is stripped from remaining text so OCR/mode
    classifiers still see the user's own words.
    """
    q = query or ""
    match = _MARKER_RE.search(q)
    if not match:
        remaining = _INPUTS_RE.sub("", q).strip()
        return [], remaining

    body = match.group(1) or ""
    paths: List[str] = []
    seen = set()
    for line in body.splitlines():
        p = normalize_user_path(line.strip().strip('"').strip("'").rstrip(").,;"))
        if not p:
            continue
        # Skip accidental guidance lines if they ever land inside the marker.
        if p.startswith("- ") or p.lower().startswith("these files were attached"):
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        paths.append(p)

    remaining = (q[: match.start()] + q[match.end() :]).strip()
    remaining = _INPUTS_RE.sub("", remaining).strip()
    return paths, remaining


def collect_attached_paths(
    query: str,
    *,
    suffixes: Optional[Iterable[str]] = None,
    existing_only: bool = False,
) -> List[str]:
    """Return attachment-marker paths, optionally filtered by extension."""
    paths, _ = parse_attachments_block(query or "")
    allow = {str(s).lower() if str(s).startswith(".") else f".{s}".lower() for s in (suffixes or [])}
    out: List[str] = []
    seen = set()
    for raw in paths:
        normalized = normalize_user_path(raw)
        if not normalized:
            continue
        p = Path(normalized)
        if allow and p.suffix.lower() not in allow:
            continue
        if existing_only and not p.is_file():
            continue
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            out.append(str(p.resolve()) if p.is_file() else str(p))
        except Exception:
            out.append(str(p))
    return out


def display_label_for_attachment(path: str) -> str:
    try:
        return Path(path).name or path
    except Exception:
        return path


def format_user_transcript(text: str, attachment_paths: Sequence[str] = ()) -> str:
    """Human-readable user bubble text (filenames, not base64)."""
    body = (text or "").strip()
    labels = [display_label_for_attachment(p) for p in attachment_paths if str(p).strip()]
    if not labels:
        return body
    attach_line = "Attachments: " + ", ".join(labels)
    if not body:
        return attach_line
    return f"{body}\n\n{attach_line}"


__all__ = [
    "ARCHIVE_EXTENSIONS",
    "ATTACHABLE_EXTENSIONS",
    "ATTACHED_INPUTS_END",
    "ATTACHED_INPUTS_START",
    "ATTACHMENTS_END",
    "ATTACHMENTS_START",
    "CAD_EXTENSIONS",
    "DEFAULT_ATTACHMENTS_ONLY_PROMPT",
    "DEFAULT_DATA_MAX_FILE_MB",
    "DEFAULT_IMAGE_MAX_FILE_MB",
    "DOCUMENT_EXTENSIONS",
    "GIS_EXTENSIONS",
    "IMAGE_EXTENSIONS",
    "PRESENTATION_EXTENSIONS",
    "SPREADSHEET_EXTENSIONS",
    "SURVEY_DATA_EXTENSIONS",
    "TEXT_EXTENSIONS",
    "attach_unsupported_message",
    "attachment_kind",
    "attachment_kind_label",
    "collect_attached_paths",
    "default_prompt_for_attachments",
    "display_label_for_attachment",
    "file_dialog_filters",
    "format_attachments_block",
    "format_user_transcript",
    "is_attachable_path",
    "is_image_path",
    "max_file_mb_for_path",
    "normalize_user_path",
    "parse_attachments_block",
]
