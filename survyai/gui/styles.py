"""
Qt stylesheet (QSS) for a readable, professional desktop tool.

Rationale:
- **Light theme by default**: CAD/GIS users often work in bright offices; high
  contrast aids long sessions.
- **System Fusion base**: `QApplication.setStyle("Fusion")` + QSS avoids fully
  custom widgets while still looking intentional.
"""

APPLICATION_STYLESHEET = """
QWidget {
    font-family: "Segoe UI", "Segoe UI Variable", "IBM Plex Sans", sans-serif;
    font-size: 10pt;
}
QMainWindow, QDialog {
    background-color: #f5f6f8;
}
QTextEdit, QPlainTextEdit, QLineEdit, QListWidget {
    background-color: #ffffff;
    border: 1px solid #c8ccd4;
    border-radius: 4px;
    padding: 6px;
    selection-background-color: #1d4ed8;
    selection-color: #ffffff;
}

/* Dropdowns: flat bar look + floating popup list (not legacy Win32 gray). */
QComboBox {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 6px 12px;
    padding-right: 32px;
    min-height: 22px;
    selection-background-color: #1d4ed8;
    selection-color: #ffffff;
}
QComboBox:hover {
    border-color: #94a3b8;
    background-color: #fafbfc;
}
QComboBox:focus {
    border-color: #3b82f6;
    background-color: #ffffff;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 30px;
    border: none;
    border-left: 1px solid #e2e8f0;
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
    background-color: #f1f5f9;
}
QComboBox::drop-down:hover {
    background-color: #e2e8f0;
}
QComboBox::down-arrow {
    width: 12px;
    height: 12px;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 6px 4px;
    outline: none;
    selection-background-color: #1d4ed8;
    selection-color: #ffffff;
}
QComboBox QLineEdit {
    border: none;
    padding: 0px;
    background: transparent;
    selection-background-color: #1d4ed8;
    selection-color: #ffffff;
}
QPushButton {
    background-color: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #1d4ed8;
}
QPushButton:pressed {
    background-color: #1e40af;
}
QPushButton:disabled {
    background-color: #94a3b8;
    color: #e2e8f0;
}
QPushButton#secondaryButton {
    background-color: #e2e8f0;
    color: #1e293b;
}
QPushButton#secondaryButton:hover {
    background-color: #cbd5e1;
}
QPushButton#summaryButton {
    background-color: #ffffff;
    color: #334155;
    border: 1px solid #d8dee9;
    border-radius: 14px;
    padding: 6px 12px;
    min-height: 16px;
}
QPushButton#summaryButton:hover {
    background-color: #eff6ff;
    border: 1px solid #bfdbfe;
}
QLabel#titleLabel {
    font-size: 14pt;
    font-weight: 600;
    color: #0f172a;
}
QLabel#hintLabel {
    color: #64748b;
    font-size: 9pt;
}
QLabel#statusChip {
    background-color: #dbeafe;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 4px 10px;
    font-size: 9pt;
    font-weight: 600;
}
QPushButton#statusChipButton {
    background-color: #dbeafe;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    padding: 6px 12px;
    min-height: 16px;
    font-size: 9pt;
    font-weight: 600;
}
QPushButton#statusChipButton:hover {
    background-color: #bfdbfe;
}
QPushButton#statusChipButton:checked {
    background-color: #fee2e2;
    color: #b91c1c;
    border: 1px solid #fecaca;
}
QGroupBox {
    font-weight: 600;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
}
/* Menu bar + popups: crisp typography, rounded panels, clear hover (Fusion-friendly). */
QMenuBar {
    background-color: transparent;
    spacing: 2px;
    padding: 2px 4px;
}
QMenuBar::item {
    padding: 6px 14px;
    border-radius: 8px;
    background: transparent;
    color: #1e293b;
}
QMenuBar::item:selected {
    background-color: #e8eef7;
}
QMenuBar::item:pressed {
    background-color: #dbeafe;
}

QMenu {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 6px 4px;
}
QMenu::item {
    padding: 9px 32px 9px 16px;
    border-radius: 8px;
    margin: 2px 6px;
    color: #1e293b;
}
QMenu::item:selected {
    background-color: #eff6ff;
    color: #1e40af;
}
QMenu::item:pressed {
    background-color: #dbeafe;
}
QMenu::separator {
    height: 1px;
    margin: 6px 12px;
    background: #e2e8f0;
}

QStatusBar {
    color: #475569;
    font-size: 9pt;
}
QStatusBar::item {
    border: none;
}

/* Primary prompt: grows with content up to max height (see ChatInput in code). */
QPlainTextEdit#chatInput {
    border: 1px solid #cfd5e6;
    border-radius: 12px;
    padding: 10px 12px;
    background: #ffffff;
    font-size: 10.5pt;
    line-height: 1.35;
}

/* Conversation sidebar: card-style rows, clear separation */
QWidget#sidebarConversations {
    background: #eef2f9;
    border: 1px solid #dfe6f2;
    border-radius: 12px;
}
QListWidget#conversationList {
    border: none;
    background: transparent;
    padding: 6px;
    outline: none;
}
QListWidget#conversationList::item {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px 10px 10px 12px;
    margin-bottom: 8px;
    color: #1e293b;
}
QListWidget#conversationList::item:hover {
    background: #f8fafc;
    border-color: #cbd5e1;
}
QListWidget#conversationList::item:selected {
    background: #eff6ff;
    border-color: #93c5fd;
    color: #0f172a;
}

/* Full-page titles (Settings, Diagnostics, …) */
QLabel#pageTitle {
    font-size: 22px;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.3px;
}
QLabel#pageSubtitle {
    font-size: 11pt;
    color: #64748b;
    margin-top: 4px;
}

/* Account / login chip (rounded, professional) */
QToolButton#userMenuButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4f8bff, stop:1 #2f6fed);
    color: #ffffff;
    border: 1px solid #2563eb;
    border-radius: 18px;
    padding: 9px 20px 9px 18px;
    font-weight: 650;
    font-size: 12px;
    min-height: 22px;
}
QToolButton#userMenuButton:hover {
    background: #3b7aed;
    border-color: #1d4ed8;
}
QToolButton#userMenuButton::menu-indicator {
    image: none;
    width: 0px;
    height: 0px;
}
QToolButton#userMenuButtonGuest {
    background: #ffffff;
    color: #1d4ed8;
    border: 2px solid #3b82f6;
    border-radius: 18px;
    padding: 9px 20px 9px 18px;
    font-weight: 650;
    font-size: 12px;
    min-height: 22px;
}
QToolButton#userMenuButtonGuest:hover {
    background: #eff6ff;
    border-color: #2563eb;
}
QToolButton#userMenuButtonGuest::menu-indicator {
    image: none;
    width: 0px;
    height: 0px;
}

/* Subtle credit-usage line under the console prompt (non-intrusive). */
QWidget#creditUsageNoticeShell {
    background: transparent;
    border-top: 1px solid #f1f5f9;
    margin-top: 2px;
}
QLabel#creditUsageNoticeLabel {
    color: #78716c;
    font-size: 9pt;
    font-weight: 400;
    padding: 4px 2px 2px 2px;
}
QToolButton#creditUsageNoticeDismiss {
    color: #9ca3af;
    background: transparent;
    border: none;
    font-size: 12pt;
    font-weight: 500;
    padding: 0px 4px;
    min-width: 18px;
    max-width: 18px;
    min-height: 18px;
    max-height: 18px;
}
QToolButton#creditUsageNoticeDismiss:hover {
    color: #64748b;
    background: #f1f5f9;
    border-radius: 4px;
}
"""
