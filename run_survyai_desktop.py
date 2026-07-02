"""Frozen-app entry point for the SurvyAI Windows desktop application.

This is the script PyInstaller bundles (see survyai.spec). It must stay tiny and
import-light so the splash/startup is fast; all real work happens inside
`survyai.gui.main.run_gui`.
"""

from __future__ import annotations

import multiprocessing
import sys


def _main() -> int:
    # Required so PyInstaller one-file/one-dir builds don't spawn duplicate GUIs
    # if any dependency uses multiprocessing.
    multiprocessing.freeze_support()
    from survyai.gui.main import run_gui

    return run_gui(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(_main())
