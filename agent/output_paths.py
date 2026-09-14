"""
Where newly created files are written.

Default is the active SurvyAI workspace (process cwd), not the folder of an
input PDF/Excel/Word file. Users can still send outputs next to a source file
or into a named folder by saying so.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

_PathLike = Union[str, Path]

_SOURCE_NOUN = (
    r"(?:"
    r"(?:input|source|original)\s+(?:file|document|pdf|dwg|excel|workbook|plan)s?"
    r"|(?:this|the|that)\s+(?:pdf|file|document|docx|excel|workbook|spreadsheet|drawing|plan)"
    r"|the\s+(?:input|source|original)"
    r")"
)

_BESIDE_SOURCE_RE = re.compile(
    r"(?:"
    r"same\s+(?:folder|directory|dir|location)\s+as\s+(?:the\s+)?" + _SOURCE_NOUN + r"\b"
    r"|in\s+(?:the\s+)?(?:same\s+)?(?:input|source|pdf|document)'?s?\s+(?:folder|directory)"
    r"|into\s+(?:the\s+)?(?:input|source|pdf|document)'?s?\s+(?:folder|directory)"
    r"|where\s+(?:the\s+)?" + _SOURCE_NOUN + r"\s+(?:is|lives|sits|was)\b"
    r"|(?:next\s+to|beside|alongside)\s+(?:the\s+)?" + _SOURCE_NOUN + r"\b"
    r"|save\s+(?:it\s+)?(?:in|into|to)\s+(?:the\s+)?same\s+(?:folder|directory)\s+as\s+(?:the\s+)?"
    + _SOURCE_NOUN + r"\b"
    r")",
    flags=re.IGNORECASE,
)

# GUI / conversational workspace: not the input file's directory.
_WORKSPACE_OUTPUT_RE = re.compile(
    r"same\s+(?:folder|directory)\s+as\s+this\s+project\b"
    r"|(?:in|into|to)\s+(?:the\s+)?(?:current\s+(?:folder|directory|workspace)|"
    r"active\s+workspace|workspace(?:\s+folder)?|survyai\s+folder)\b",
    flags=re.IGNORECASE,
)

_EXPLICIT_FOLDER_PATTERNS = (
    r"(?:in|into|to)\s+(?:the\s+)?(?:folder|directory)\s+['\"]([^'\"]+)['\"]",
    r"(?:save|saved|export|create|write|store|generate)\s+(?:\w+\s+){0,10}(?:in|into|to)\s+(?:the\s+)?(?:folder|directory)\s+['\"]([^'\"]+)['\"]",
    r"(?:save|saved|export|create|write)\s+(?:to|in|into)\s+(?:the\s+)?(?:folder|directory)\s+['\"]([^'\"]+)['\"]",
)

_FILE_SUFFIXES = {
    ".dwg", ".dxf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".pdf", ".aprx", ".gdb", ".shp",
}


def _join_texts(*texts: Optional[str]) -> str:
    return "\n".join(str(t) for t in texts if t and str(t).strip())


def user_requests_save_beside_source(*texts: Optional[str]) -> bool:
    """
    True only when the user asked to write next to the *input file*.

    "Current folder", "current workspace", and "same folder as this project"
    mean the SurvyAI workspace, not the source document's directory.
    """
    blob = _join_texts(*texts)
    if not blob:
        return False
    if _WORKSPACE_OUTPUT_RE.search(blob) and not _BESIDE_SOURCE_RE.search(blob):
        return False
    return bool(_BESIDE_SOURCE_RE.search(blob))


def extract_explicit_output_folder(*texts: Optional[str]) -> Optional[Path]:
    """Return a user-named destination directory, if one is present."""
    blob = _join_texts(*texts)
    if not blob:
        return None
    for pat in _EXPLICIT_FOLDER_PATTERNS:
        m = re.search(pat, blob, flags=re.IGNORECASE)
        if not m:
            continue
        raw = (m.group(1) or "").strip().strip("\"'").rstrip(").,;")
        if not raw:
            continue
        low = raw.lower().strip()
        if low in {
            "current",
            "current folder",
            "current directory",
            "current workspace",
            "workspace",
            "the workspace",
            "this folder",
            "this directory",
            "survyai folder",
            "the survyai folder",
        }:
            continue
        folder = Path(raw)
        if folder.suffix.lower() in _FILE_SUFFIXES:
            folder = folder.parent
        try:
            return folder.resolve()
        except Exception:
            return folder
    return None


def resolve_created_output_path(
    output_ref: str,
    *,
    query: str = "",
    source_path: Optional[_PathLike] = None,
    workspace: Optional[_PathLike] = None,
    default_name: Optional[str] = None,
) -> Path:
    """
    Resolve a path for a file SurvyAI is about to create.

    Priority:
      1. Absolute path in output_ref
      2. Explicit folder named in the prompt + filename
      3. Source file's folder, but only if the user asked for that
      4. Active workspace (cwd)
    """
    ws = Path(workspace).resolve() if workspace else Path.cwd().resolve()
    ref = (output_ref or "").strip().strip("\"'").rstrip(").,;")
    if not ref:
        ref = (default_name or "").strip()
    if not ref:
        return ws

    p = Path(ref)
    if p.is_absolute():
        return p

    name = p.name if (len(p.parts) == 1 and "/" not in ref and "\\" not in ref) else None
    if name is None and (len(p.parts) > 1 or "/" in ref or "\\" in ref):
        return (ws / p).resolve()

    filename = name or p.name
    explicit = extract_explicit_output_folder(query)
    if explicit is not None:
        try:
            explicit.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return (explicit / filename).resolve()

    if user_requests_save_beside_source(query) and source_path:
        src = Path(str(source_path))
        dest = src.parent if src.suffix else src
        return (dest / filename).resolve()

    return (ws / filename).resolve()


def join_workspace_path(
    output_path: _PathLike,
    *,
    workspace: Optional[_PathLike] = None,
) -> Path:
    """Join a relative output path to the workspace without dropping parent folders."""
    p = Path(str(output_path).strip().strip("\"'"))
    if p.is_absolute():
        return p
    ws = Path(workspace).resolve() if workspace else Path.cwd().resolve()
    return (ws / p).resolve()


_file_conflict_handler: Optional[Callable[[str, str], bool]] = None

def set_file_conflict_handler(handler: Optional[Callable[[str, str], bool]]) -> None:
    """Install the GUI/CLI callback used before overwriting an existing file."""
    global _file_conflict_handler
    _file_conflict_handler = handler


def _existing_output_kind(path: Path) -> str:
    """Short label for an existing output path (file, folder, or known type)."""
    ext = path.suffix.lower()
    if ext == ".gdb" or (path.exists() and path.is_dir() and ext not in {".gdb"}):
        if ext == ".gdb":
            return "geodatabase"
        return "folder"
    return {
        ".dwg": "drawing",
        ".dxf": "drawing",
        ".docx": "Word document",
        ".doc": "Word document",
        ".xlsx": "Excel workbook",
        ".xls": "Excel workbook",
        ".xlsm": "Excel workbook",
        ".csv": "CSV file",
        ".txt": "text file",
        ".json": "JSON file",
        ".pdf": "PDF",
        ".aprx": "ArcGIS project",
        ".shp": "shapefile",
    }.get(ext, "file")


def confirm_overwrite_existing_file(path: str, *, mode: str = "overwrite") -> bool:
    """Ask before replacing or modifying an existing user file or folder. True = proceed."""
    raw = str(path or "").strip()
    if not raw:
        return True
    try:
        p = Path(raw)
        if not p.exists():
            return True
        # Never treat the workspace or a drive root as a replaceable output.
        try:
            resolved = p.resolve()
            if resolved == Path.cwd().resolve() or resolved.parent == resolved:
                return True
        except Exception:
            pass
    except Exception:
        return True
    handler = _file_conflict_handler
    if callable(handler):
        try:
            return bool(handler(str(p.resolve()), str(mode or "overwrite")))
        except Exception:
            return False
    try:
        import ctypes

        mode_l = (mode or "overwrite").strip().lower()
        kind = _existing_output_kind(p)
        target = "folder" if (p.is_dir() and p.suffix.lower() != ".gdb") else "file"
        if mode_l == "modify":
            text = (
                f"This {kind} already exists:\n\n{p}\n\n"
                f"Do you want to apply changes to this existing {target}?\n\n"
                f"Yes — continue and modify the {target}\n"
                f"No — cancel and leave the {target} unchanged"
            )
            title = f"SurvyAI — Modify existing {kind}"
        else:
            text = (
                f"This {kind} already exists:\n\n{p}\n\n"
                f"Do you want to overwrite it?\n\n"
                f"Yes — overwrite the existing {target}\n"
                f"No — keep the existing {target} unchanged"
            )
            title = f"SurvyAI — {kind.capitalize()} already exists"
        flags = 0x00000004 | 0x00000030 | 0x00010000 | 0x00040000 | 0x00001000
        try:
            ctypes.windll.user32.AllowSetForegroundWindow(0xFFFFFFFF)
        except Exception:
            pass
        result = ctypes.windll.user32.MessageBoxW(0, text, title, flags)
        return int(result) == 6
    except Exception:
        return False


def cancelled_existing_file_write(path: str, *, mode: str = "overwrite") -> Optional[dict]:
    """If *path* exists and the user declines, return a cancelled result; else None."""
    if confirm_overwrite_existing_file(path, mode=mode):
        return None
    return {
        "success": False,
        "cancelled": True,
        "error": f"Left existing path unchanged: {path}",
        "output_path": str(path),
    }
