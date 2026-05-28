"""
Qt application bootstrap for SurvyAI desktop GUI.

Why this file exists separately from `main_window.py`:
- **Separation of concerns**: `QApplication` lifetime, style, and OS-specific hints
  belong in a tiny entry module; the window class stays testable without starting Qt.
- **High-DPI**: Qt 6 enables scaling by default; we set rounding policy for crisp
  fractional scaling on Windows 10/11.
- **Fusion + QSS**: consistent look across Windows versions without custom controls.
"""

from __future__ import annotations

import sys
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from survyai.gui.launch_args import parse_gui_argv
from survyai.gui.main_window import MainWindow
from survyai.gui.styles import LIGHT_STYLESHEET


def run_gui(argv: Optional[list[str]] = None) -> int:
    """Create the Qt event loop, show the main window, and run until exit."""
    if argv is None:
        argv = sys.argv
    qt_argv, initial_query, auto_run_query = parse_gui_argv(list(argv))

    # Prefer Fusion for predictable styling when we layer QSS on top.
    QApplication.setStyle("Fusion")
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass

    app = QApplication(qt_argv)
    app.setApplicationName("SurvyAI")
    app.setOrganizationName("SurvyAI")

    app.setStyleSheet(LIGHT_STYLESHEET)

    # Windows taskbar grouping / jump lists: explicit app user model id (optional).
    if sys.platform == "win32":
        try:
            from ctypes import windll  # type: ignore[attr-defined]

            windll.shell32.SetCurrentProcessExplicitAppUserModelID("SurvyAI.Desktop.1")
        except Exception:
            pass

    win = MainWindow(initial_query=initial_query, auto_run_query=auto_run_query)
    win.show()
    return app.exec()


__all__ = ["run_gui"]
