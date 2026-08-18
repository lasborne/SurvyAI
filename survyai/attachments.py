"""Pure helpers for chat attachment markers (no Qt / GUI dependency)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence, Tuple

ATTACHMENTS_START = "[SurvyAI attachments]"
ATTACHMENTS_END = "[/SurvyAI attachments]"

IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
)
DOCUMENT_EXTENSIONS = frozenset({".pdf", ".docx", ".doc"})
ATTACHABLE_EXTENSIONS = IMAGE_EXTENSIONS | DOCUMENT_EXTENSIONS

DEFAULT_ATTACHMENTS_ONLY_PROMPT = "Process the attached file(s)."

_MARKER_RE = re.compile(
    r"\[SurvyAI attachments\]\s*(.*?)\s*\[/SurvyAI attachments\]",
    re.IGNORECASE | re.DOTALL,
)


def is_attachable_path(path: str | Path) -> bool:
    try:
        return Path(path).suffix.lower() in ATTACHABLE_EXTENSIONS
    except Exception:
        return False


def is_image_path(path: str | Path) -> bool:
    try:
        return Path(path).suffix.lower() in IMAGE_EXTENSIONS
    except Exception:
        return False


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
        text = DEFAULT_ATTACHMENTS_ONLY_PROMPT

    lines = [ATTACHMENTS_START, *cleaned, ATTACHMENTS_END, "", text]
    return "\n".join(lines)


def parse_attachments_block(query: str) -> Tuple[List[str], str]:
    """
    Split a query into (attachment_paths, remaining_user_text).

    Returns empty paths when no marker block is present.
    """
    q = query or ""
    match = _MARKER_RE.search(q)
    if not match:
        return [], q.strip()

    body = match.group(1) or ""
    paths: List[str] = []
    seen = set()
    for line in body.splitlines():
        p = line.strip().strip('"').strip("'").rstrip(").,;")
        if not p:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        paths.append(p)

    remaining = (q[: match.start()] + q[match.end() :]).strip()
    return paths, remaining


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
    "ATTACHABLE_EXTENSIONS",
    "ATTACHMENTS_END",
    "ATTACHMENTS_START",
    "DEFAULT_ATTACHMENTS_ONLY_PROMPT",
    "DOCUMENT_EXTENSIONS",
    "IMAGE_EXTENSIONS",
    "display_label_for_attachment",
    "format_attachments_block",
    "format_user_transcript",
    "is_attachable_path",
    "is_image_path",
    "parse_attachments_block",
]
