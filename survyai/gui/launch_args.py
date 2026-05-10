"""
Parse GUI launch arguments.

Examples:
    python -m survyai.gui
    python -m survyai.gui "prefill this prompt"
    python -m survyai.gui --run "prefill and execute"
    python -m survyai.gui -- --text-that-looks-like-a-flag
"""

from __future__ import annotations

from typing import List, Optional, Tuple


def parse_gui_argv(argv: List[str]) -> Tuple[List[str], Optional[str], bool]:
    """Return (qt_argv, initial_query, auto_run)."""
    auto_run = False
    tokens = list(argv)

    while "--run" in tokens:
        tokens.remove("--run")
        auto_run = True

    if "--" in tokens:
        idx = tokens.index("--")
        qt_argv = tokens[:idx]
        query = " ".join(tokens[idx + 1 :]).strip()
        return qt_argv, (query or None), auto_run

    if "-m" in tokens:
        idx = tokens.index("-m")
        qt_argv = tokens[: idx + 2] if idx + 2 <= len(tokens) else tokens[:]
        query = " ".join(tokens[idx + 2 :]).strip()
        return qt_argv, (query or None), auto_run

    if len(tokens) > 1:
        return [tokens[0]], (" ".join(tokens[1:]).strip() or None), auto_run

    return tokens, None, auto_run


__all__ = ["parse_gui_argv"]
