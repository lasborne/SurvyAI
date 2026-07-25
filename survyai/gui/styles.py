"""
Qt stylesheet (QSS) for SurvyAI Desktop.

Single source of truth for visual design. Applied globally via ``get_stylesheet()``.
Use object names (``#sendButton``, ``#chatTranscript``, …) for component-specific rules.
"""

THEME_LIGHT = "light"
THEME_DARK = "dark"

LIGHT_STYLESHEET = """
/* --- Base typography ------------------------------------------------ */
QWidget {
    font-family: "Segoe UI", "Segoe UI Variable", "IBM Plex Sans", sans-serif;
    font-size: 10pt;
    color: #0f172a;
}
QMainWindow {
    background-color: #f4f6fa;
}
QWidget#centralRoot,
QStackedWidget#appStack,
QWidget#appStackPage,
QScrollArea#appScroll,
QScrollArea#appScroll::viewport,
QWidget#appScrollContent,
QWidget#appPageFooter {
    background-color: #f4f6fa;
}
QDialog {
    background-color: #f4f6fa;
}

/* --- Dialogs & popups ------------------------------------------------ */
QDialog, QMessageBox, QInputDialog {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
}
QDialog QLabel, QMessageBox QLabel, QInputDialog QLabel {
    color: #0f172a;
    font-size: 10.5pt;
    line-height: 1.45;
}
QMessageBox {
    min-width: 380px;
}
QMessageBox QLabel {
    min-width: 280px;
    padding: 4px 2px 8px 2px;
}
QDialog QLineEdit, QMessageBox QLineEdit, QInputDialog QLineEdit {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 8px 12px;
}
QDialogButtonBox QPushButton, QMessageBox QPushButton, QInputDialog QPushButton {
    min-width: 100px;
    min-height: 32px;
    padding: 7px 16px;
    border-radius: 10px;
    font-weight: 600;
}
QMessageBox QPushButton {
    background-color: #2563eb;
    border: 1px solid #1d4ed8;
    color: #ffffff;
}
QMessageBox QPushButton:hover {
    background-color: #1d4ed8;
}
QMessageBox QPushButton[text="Cancel"],
QMessageBox QPushButton[text="No"],
QMessageBox QPushButton[text="&No"] {
    background-color: #ffffff;
    color: #334155;
    border: 1px solid #cbd5e1;
}
QMessageBox QPushButton[text="Cancel"]:hover,
QMessageBox QPushButton[text="No"]:hover,
QMessageBox QPushButton[text="&No"]:hover {
    background-color: #f8fafc;
    border-color: #94a3b8;
}

/* Help / Getting Started dialog */
QDialog#helpDialog {
    background-color: #f8fafc;
}
QLabel#helpDialogTitle {
    font-size: 16pt;
    font-weight: 700;
    color: #0f172a;
    padding-bottom: 2px;
}
QLabel#helpDialogSubtitle {
    font-size: 10pt;
    color: #64748b;
    padding-bottom: 4px;
}
QTextBrowser#helpBrowser {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 16px;
    selection-background-color: #bfdbfe;
}
QCheckBox#helpDontShowAgain {
    color: #475569;
    spacing: 8px;
    padding: 2px 0;
}
QPushButton#primaryButton {
    background-color: #2563eb;
    border: 1px solid #1d4ed8;
    color: #ffffff;
    font-weight: 700;
    min-width: 108px;
    min-height: 34px;
    padding: 8px 18px;
    border-radius: 10px;
}
QPushButton#primaryButton:hover {
    background-color: #1d4ed8;
}

/* --- Tabs (Console / Output History) -------------------------------- */
QTabWidget::pane {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    background: #ffffff;
    top: -1px;
    padding: 4px;
}
QTabBar::tab {
    background: transparent;
    padding: 8px 18px;
    margin-right: 4px;
    border: 1px solid transparent;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    color: #475569;
    font-weight: 500;
}
QTabBar::tab:selected {
    background: #ffffff;
    border-color: #e2e8f0;
    border-bottom-color: #ffffff;
    color: #0f172a;
    font-weight: 600;
}
QTabBar::tab:hover:!selected {
    background: #f1f5f9;
    border-color: #e2e8f0;
    color: #334155;
}

/* --- Form controls -------------------------------------------------- */
QLineEdit, QTextEdit, QPlainTextEdit, QListWidget {
    background-color: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 10px;
    padding: 8px 12px;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QListWidget:focus {
    border-color: #3b82f6;
}
QTextEdit#chatTranscript {
    background-color: #fafbfc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    font-size: 10pt;
    line-height: 1.25;
    padding: 4px 6px;
}
QPlainTextEdit#activityLog {
    background-color: #fafbfc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    font-family: "Cascadia Mono", "Consolas", "Segoe UI Mono", monospace;
    font-size: 9.5pt;
    color: #334155;
}
QListWidget#historyList {
    background-color: #fafbfc;
}
QListWidget#historyList::item {
    padding: 10px 12px;
    border-radius: 8px;
    margin: 2px 4px;
}
QListWidget#historyList::item:selected {
    background-color: #eff6ff;
    color: #0f172a;
}
QListWidget#historyList::item:hover:!selected {
    background-color: #f1f5f9;
}

QComboBox {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 6px 12px;
    padding-right: 32px;
    min-height: 22px;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}
QComboBox:hover {
    border-color: #94a3b8;
    background-color: #fafbfc;
}
QComboBox:focus {
    border-color: #3b82f6;
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
QComboBox QAbstractItemView {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 6px 4px;
    outline: none;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}
QComboBox QLineEdit {
    border: none;
    padding: 0px;
    background: transparent;
}

QCheckBox {
    spacing: 8px;
    color: #334155;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid #cbd5e1;
    background: #ffffff;
}
QCheckBox::indicator:hover {
    border-color: #94a3b8;
}
QCheckBox::indicator:checked {
    background: #2563eb;
    border-color: #2563eb;
}

QSplitter::handle {
    background: #e2e8f0;
    width: 2px;
    height: 2px;
}
QSplitter::handle:hover {
    background: #94a3b8;
}
QSplitter#consoleVerticalSplit::handle {
    height: 3px;
    margin: 0 8px;
}
QSplitter#consoleVerticalSplit::handle:hover {
    background: #94a3b8;
}

/* --- Buttons -------------------------------------------------------- */
QPushButton {
    background-color: #2563eb;
    color: #ffffff;
    border: 1px solid #1d4ed8;
    border-radius: 10px;
    padding: 6px 14px;
    min-height: 20px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #1d4ed8;
}
QPushButton:pressed {
    background-color: #1e40af;
}
QPushButton:disabled {
    background-color: #cbd5e1;
    border-color: #cbd5e1;
    color: #f8fafc;
}
QPushButton#sendButton {
    background-color: #2563eb;
    border-color: #1d4ed8;
    padding: 7px 16px;
    min-height: 22px;
    font-weight: 700;
}
QPushButton#sendButton:hover {
    background-color: #1d4ed8;
}
QPushButton#sendButton:disabled {
    background-color: #cbd5e1;
    border-color: #cbd5e1;
    color: #94a3b8;
}
QPushButton#secondaryButton {
    background-color: #ffffff;
    color: #334155;
    border: 1px solid #cbd5e1;
    font-weight: 600;
}
QPushButton#secondaryButton:hover {
    background-color: #f8fafc;
    border-color: #94a3b8;
}
QPushButton#summaryButton {
    background-color: #ffffff;
    color: #334155;
    border: 1px solid #d8dee9;
    border-radius: 14px;
    padding: 6px 12px;
    min-height: 16px;
    font-weight: 600;
}
QPushButton#summaryButton:hover {
    background-color: #eff6ff;
    border-color: #bfdbfe;
}
QPushButton#statusChipButton {
    background-color: #fff7ed;
    color: #9a3412;
    border: 1px solid #fed7aa;
    border-radius: 12px;
    padding: 6px 12px;
    min-height: 16px;
    font-size: 9pt;
    font-weight: 600;
}
QPushButton#statusChipButton:hover {
    background-color: #ffedd5;
}
QPushButton#statusChipButton:checked {
    background-color: #fee2e2;
    color: #b91c1c;
    border: 1px solid #fecaca;
}

/* --- Labels & headers ----------------------------------------------- */
QLabel#titleLabel {
    font-size: 14pt;
    font-weight: 600;
    color: #0f172a;
}
QLabel#wordmarkLabel {
    color: #0f172a;
    font-weight: 800;
    letter-spacing: -0.2px;
}
QLabel#wordmarkLabel .accent {
    color: #2563eb;
}
QLabel#wordmarkSub {
    color: #64748b;
    font-weight: 600;
    font-size: 9pt;
    padding-left: 4px;
}
QLabel#versionBadge {
    color: #94a3b8;
    font-size: 8pt;
    font-weight: 500;
    padding-left: 6px;
}
QLabel#hintLabel, QLabel#consoleHintLabel {
    color: #64748b;
    font-size: 9pt;
}
QLabel#sectionHeader {
    color: #0f172a;
    font-weight: 700;
    font-size: 9.5pt;
    padding: 1px 2px 4px 2px;
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
QLabel#runStatusLabel {
    color: #334155;
    font-weight: 600;
    font-size: 9pt;
}
QLabel#elapsedLabel {
    color: #64748b;
    font-size: 9pt;
}

QWidget#topBar {
    background-color: #ffffff;
    border: 1px solid #e8ecf4;
    border-radius: 14px;
    padding: 4px;
}
QFrame#topBarDivider {
    background: #e2e8f0;
    max-height: 1px;
    min-height: 1px;
    border: none;
}

/* --- Group boxes & scroll areas ------------------------------------- */
QGroupBox {
    font-weight: 600;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    margin-top: 14px;
    padding-top: 10px;
    background: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
    color: #0f172a;
}
QScrollArea {
    border: none;
    background: transparent;
}
QScrollArea > QWidget > QWidget {
    background: transparent;
}
QScrollArea#appScroll { border: none; }

/* --- Menus ---------------------------------------------------------- */
QMenuBar {
    background-color: transparent;
    spacing: 2px;
    padding: 2px 4px;
}
QMenuBar::item {
    padding: 6px 14px;
    border-radius: 8px;
    background: transparent;
    color: #334155;
}
QMenuBar::item:selected {
    background-color: #f1f5f9;
}
QMenuBar::item:pressed {
    background-color: #e0e7ff;
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
QMenu::separator {
    height: 1px;
    margin: 6px 12px;
    background: #e2e8f0;
}

QStatusBar {
    color: #64748b;
    font-size: 9pt;
    background: #f8fafc;
    border-top: 1px solid #e2e8f0;
}
QStatusBar::item {
    border: none;
}

QToolTip {
    background-color: #1e293b;
    color: #f8fafc;
    border: none;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 9pt;
}

/* --- Chat input & sidebar ------------------------------------------- */
QPlainTextEdit#chatInput {
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 8px 12px;
    background: #ffffff;
    font-size: 10.5pt;
    line-height: 1.4;
}
QPlainTextEdit#chatInput:focus {
    border-color: #3b82f6;
    background: #fafcff;
}
/* --- Scrollbars (light mode) --------------------------------------- */
QScrollBar:vertical {
    background: #f1f5f9;
    width: 10px;
    margin: 2px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #cbd5e1;
    border-radius: 5px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #94a3b8; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: #f1f5f9;
    height: 10px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background: #cbd5e1;
    border-radius: 5px;
    min-width: 24px;
}
QScrollBar::handle:horizontal:hover { background: #94a3b8; }

QWidget#sidebarConversations {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
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
    padding: 10px 12px;
    margin-bottom: 6px;
    color: #334155;
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

/* --- Account menu (primary blue, same radius as QPushButton) ------------- */
QToolButton#userMenuButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3b82f6, stop:1 #2563eb);
    color: #ffffff;
    border: 1px solid #1d4ed8;
    border-radius: 10px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 10pt;
    min-height: 22px;
}
QToolButton#userMenuButton:hover {
    background-color: #2563eb;
    border-color: #1d4ed8;
}
QToolButton#userMenuButton:pressed {
    background-color: #1d4ed8;
}
QToolButton#userMenuButton::menu-indicator {
    image: none;
    width: 0px;
    height: 0px;
}
QToolButton#userMenuButtonGuest {
    background-color: #ffffff;
    color: #1d4ed8;
    border: 2px solid #3b82f6;
    border-radius: 10px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 10pt;
    min-height: 22px;
}
QToolButton#userMenuButtonGuest:hover {
    background-color: #eff6ff;
    border-color: #2563eb;
    color: #1e40af;
}
QToolButton#userMenuButtonGuest::menu-indicator {
    image: none;
    width: 0px;
    height: 0px;
}

/* --- Credit notice -------------------------------------------------- */
QWidget#creditUsageNoticeShell {
    background: transparent;
    border-top: 1px solid #f1f5f9;
    margin-top: 2px;
}
QLabel#creditUsageNoticeLabel {
    color: #78716c;
    font-size: 9pt;
    padding: 4px 2px 2px 2px;
}
QToolButton#creditUsageNoticeDismiss {
    color: #9ca3af;
    background: transparent;
    border: none;
    font-size: 12pt;
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

/* --- Wizard (onboarding) -------------------------------------------- */
QWizard {
    background-color: #f4f6fa;
}
QWizard QLabel {
    color: #334155;
}
"""

DARK_STYLESHEET = """
/* SurvyAI dark — zinc/slate elevation system */
QWidget {
    font-family: "Segoe UI", "Segoe UI Variable", "IBM Plex Sans", sans-serif;
    font-size: 10pt;
    color: #e4e4e7;
}
QMainWindow {
    background-color: #09090b;
}
QWidget#centralRoot,
QStackedWidget#appStack,
QWidget#appStackPage,
QScrollArea#appScroll,
QScrollArea#appScroll::viewport,
QWidget#appScrollContent,
QWidget#appPageFooter {
    background-color: #09090b;
}
QDialog {
    background-color: #0c0c0f;
}

/* --- Dialogs & popups ------------------------------------------------ */
QDialog, QMessageBox, QInputDialog {
    background-color: #0c0c0f;
    border: 1px solid #27272a;
    border-radius: 14px;
}
QDialog QLabel, QMessageBox QLabel, QInputDialog QLabel {
    color: #e4e4e7;
    font-size: 10.5pt;
    line-height: 1.45;
}
QMessageBox {
    min-width: 380px;
}
QMessageBox QLabel {
    min-width: 280px;
    padding: 4px 2px 8px 2px;
}
QDialog QLineEdit, QMessageBox QLineEdit, QInputDialog QLineEdit {
    background-color: #18181b;
    border: 1px solid #3f3f46;
    border-radius: 10px;
    padding: 9px 12px;
    color: #f4f4f5;
}
QDialogButtonBox QPushButton, QMessageBox QPushButton, QInputDialog QPushButton {
    min-width: 100px;
    min-height: 32px;
    padding: 7px 16px;
    border-radius: 10px;
    font-weight: 600;
}
QMessageBox QPushButton {
    background-color: #3b82f6;
    border: 1px solid #2563eb;
    color: #ffffff;
}
QMessageBox QPushButton:hover {
    background-color: #2563eb;
}
QMessageBox QPushButton[text="Cancel"],
QMessageBox QPushButton[text="No"],
QMessageBox QPushButton[text="&No"] {
    background-color: #18181b;
    color: #e4e4e7;
    border: 1px solid #3f3f46;
}
QMessageBox QPushButton[text="Cancel"]:hover,
QMessageBox QPushButton[text="No"]:hover,
QMessageBox QPushButton[text="&No"]:hover {
    background-color: #27272a;
    border-color: #52525b;
}

QDialog#helpDialog {
    background-color: #0c0c0f;
}
QLabel#helpDialogTitle {
    font-size: 16pt;
    font-weight: 700;
    color: #fafafa;
    padding-bottom: 2px;
}
QLabel#helpDialogSubtitle {
    font-size: 10pt;
    color: #a1a1aa;
    padding-bottom: 4px;
}
QTextBrowser#helpBrowser {
    background-color: #141416;
    border: 1px solid #27272a;
    border-radius: 12px;
    padding: 14px 16px;
    color: #e4e4e7;
    selection-background-color: #1d4ed8;
}
QCheckBox#helpDontShowAgain {
    color: #a1a1aa;
    spacing: 8px;
    padding: 2px 0;
}
QPushButton#primaryButton {
    background-color: #3b82f6;
    border: 1px solid #2563eb;
    color: #ffffff;
    font-weight: 700;
    min-width: 108px;
    min-height: 34px;
    padding: 8px 18px;
    border-radius: 10px;
}
QPushButton#primaryButton:hover {
    background-color: #2563eb;
}

QWidget#topBar {
    background-color: #141416;
    border: 1px solid #27272a;
    border-radius: 14px;
}

QTabWidget::pane {
    border: 1px solid #27272a;
    border-radius: 14px;
    background: #141416;
    top: -1px;
    padding: 6px;
}
QTabBar::tab {
    background: transparent;
    padding: 9px 20px;
    margin-right: 6px;
    border: 1px solid transparent;
    border-radius: 10px;
    color: #71717a;
    font-weight: 500;
}
QTabBar::tab:selected {
    background: #1f1f23;
    border: 1px solid #3f3f46;
    color: #fafafa;
    font-weight: 600;
}
QTabBar::tab:hover:!selected {
    background: #1a1a1e;
    color: #a1a1aa;
}

QLineEdit, QTextEdit, QPlainTextEdit, QListWidget {
    background-color: #18181b;
    border: 1px solid #3f3f46;
    border-radius: 10px;
    padding: 9px 12px;
    color: #f4f4f5;
    selection-background-color: #3b82f6;
    selection-color: #ffffff;
}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QListWidget:focus {
    border-color: #60a5fa;
    background-color: #1c1c1f;
}
QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover, QListWidget:hover {
    border-color: #52525b;
}

QTextEdit#chatTranscript {
    background-color: #0c0c0f;
    border: 1px solid #27272a;
    border-radius: 10px;
    color: #e4e4e7;
    font-size: 10pt;
    line-height: 1.25;
    padding: 4px 6px;
}
QPlainTextEdit#activityLog {
    background-color: #0c0c0f;
    border: 1px solid #27272a;
    border-radius: 10px;
    font-family: "Cascadia Mono", "Consolas", "Segoe UI Mono", monospace;
    font-size: 9.5pt;
    color: #a1a1aa;
}
QListWidget#historyList {
    background-color: #0c0c0f;
    border: 1px solid #27272a;
}
QListWidget#historyList::item {
    padding: 10px 12px;
    border-radius: 8px;
    margin: 2px 4px;
    color: #d4d4d8;
}
QListWidget#historyList::item:selected {
    background-color: #1e3a5f;
    color: #fafafa;
}
QListWidget#historyList::item:hover:!selected {
    background-color: #1f1f23;
}

QComboBox {
    background-color: #18181b;
    border: 1px solid #3f3f46;
    border-radius: 9px;
    padding: 7px 12px;
    padding-right: 32px;
    color: #f4f4f5;
}
QComboBox:hover { background-color: #1f1f23; border-color: #52525b; }
QComboBox:focus { border-color: #60a5fa; }
QComboBox::drop-down {
    border-left: 1px solid #3f3f46;
    background-color: #27272a;
    border-top-right-radius: 9px;
    border-bottom-right-radius: 9px;
}
QComboBox QAbstractItemView {
    background-color: #18181b;
    border: 1px solid #3f3f46;
    color: #f4f4f5;
    selection-background-color: #2563eb;
    outline: none;
}

QCheckBox { spacing: 8px; color: #d4d4d8; }
QCheckBox::indicator {
    width: 18px; height: 18px; border-radius: 5px;
    border: 1px solid #52525b; background: #18181b;
}
QCheckBox::indicator:hover { border-color: #71717a; }
QCheckBox::indicator:checked {
    background: #3b82f6; border-color: #3b82f6;
}

QSplitter::handle { background: #27272a; width: 3px; height: 3px; }
QSplitter::handle:hover { background: #3f3f46; }
QSplitter#consoleVerticalSplit::handle {
    height: 3px;
    margin: 0 8px;
}
QSplitter#consoleVerticalSplit::handle:hover {
    background: #52525b;
}

QScrollBar:vertical {
    background: #0c0c0f;
    width: 10px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #3f3f46;
    border-radius: 5px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #52525b; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: #0c0c0f;
    height: 10px;
}
QScrollBar::handle:horizontal {
    background: #3f3f46;
    border-radius: 5px;
    min-width: 24px;
}

QPushButton {
    background-color: #3b82f6;
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 6px 14px;
    min-height: 20px;
    font-weight: 600;
}
QPushButton:hover { background-color: #60a5fa; }
QPushButton:pressed { background-color: #2563eb; }
QPushButton:disabled {
    background-color: #27272a;
    color: #71717a;
}
QPushButton#sendButton {
    background-color: #3b82f6;
    padding: 7px 16px;
    min-height: 22px;
    font-weight: 700;
}
QPushButton#sendButton:disabled {
    background-color: #27272a;
    color: #71717a;
}
QPushButton#secondaryButton {
    background-color: #27272a;
    color: #f4f4f5;
    border: 1px solid #3f3f46;
}
QPushButton#secondaryButton:hover {
    background-color: #3f3f46;
    border-color: #52525b;
}

QLabel#wordmarkLabel { color: #fafafa; font-weight: 800; }
QLabel#wordmarkLabel .accent { color: #60a5fa; }
QLabel#wordmarkSub, QLabel#versionBadge { color: #71717a; }
QLabel#hintLabel, QLabel#consoleHintLabel { color: #a1a1aa; }
QLabel#sectionHeader { color: #fafafa; font-size: 9.5pt; padding: 1px 2px 4px 2px; }
QLabel#pageTitle { color: #fafafa; }
QLabel#pageSubtitle, QLabel#elapsedLabel { color: #a1a1aa; }
QLabel#runStatusLabel { color: #d4d4d8; font-weight: 600; }

QFrame#topBarDivider {
    background: #27272a;
    max-height: 1px;
    min-height: 1px;
    border: none;
}

QGroupBox {
    border: 1px solid #27272a;
    border-radius: 14px;
    margin-top: 16px;
    padding-top: 12px;
    background: #141416;
    color: #e4e4e7;
}
QGroupBox::title {
    color: #fafafa;
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
}

QScrollArea { border: none; background: transparent; }
QScrollArea#appScroll { border: none; }

QMenuBar { background: transparent; padding: 2px 0; }
QMenuBar::item {
    color: #a1a1aa;
    padding: 6px 14px;
    border-radius: 8px;
}
QMenuBar::item:selected { background: #1f1f23; color: #fafafa; }
QMenu {
    background-color: #18181b;
    border: 1px solid #3f3f46;
    border-radius: 12px;
    padding: 6px;
}
QMenu::item {
    color: #e4e4e7;
    padding: 9px 28px 9px 14px;
    border-radius: 8px;
}
QMenu::item:selected { background-color: #1e3a5f; color: #93c5fd; }
QMenu::separator { height: 1px; background: #27272a; margin: 6px 10px; }

QStatusBar {
    color: #71717a;
    background: #0c0c0f;
    border-top: 1px solid #27272a;
}

QToolTip {
    background-color: #27272a;
    color: #fafafa;
    border: 1px solid #3f3f46;
    border-radius: 8px;
    padding: 6px 10px;
}

QPlainTextEdit#chatInput {
    border: 1px solid #3f3f46;
    background: #18181b;
    color: #f4f4f5;
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 10.5pt;
}
QPlainTextEdit#chatInput:focus {
    border-color: #60a5fa;
    background: #1c1c1f;
}

QWidget#sidebarConversations {
    background: #141416;
    border: 1px solid #27272a;
    border-radius: 12px;
}
QListWidget#conversationList {
    border: none;
    background: transparent;
    padding: 6px;
    outline: none;
}
QListWidget#conversationList::item {
    background: #18181b;
    border: 1px solid #27272a;
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 6px;
    color: #d4d4d8;
}
QListWidget#conversationList::item:hover {
    background: #1f1f23;
    border-color: #3f3f46;
}
QListWidget#conversationList::item:selected {
    background: #172554;
    border-color: #3b82f6;
    color: #fafafa;
}

QToolButton#userMenuButton {
    background-color: #3b82f6;
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 9px 16px;
    font-weight: 600;
    font-size: 10pt;
    min-height: 22px;
}
QToolButton#userMenuButton:hover {
    background-color: #60a5fa;
}
QToolButton#userMenuButton:pressed {
    background-color: #2563eb;
}
QToolButton#userMenuButton::menu-indicator {
    image: none;
    width: 0px;
    height: 0px;
}
QToolButton#userMenuButtonGuest {
    background-color: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 9px 16px;
    font-weight: 600;
    font-size: 10pt;
    min-height: 22px;
}
QToolButton#userMenuButtonGuest:hover {
    background-color: #60a5fa;
}
QToolButton#userMenuButtonGuest:pressed {
    background-color: #1d4ed8;
}
QToolButton#userMenuButtonGuest::menu-indicator {
    image: none;
    width: 0px;
    height: 0px;
}

QWidget#themeToggle { background: transparent; }

QWidget#creditUsageNoticeShell { border-top: 1px solid #27272a; }
QLabel#creditUsageNoticeLabel { color: #a1a1aa; }

QWizard { background-color: #09090b; }
QWizard QLabel { color: #d4d4d8; }
"""

APPLICATION_STYLESHEET = LIGHT_STYLESHEET


def get_stylesheet(theme: str) -> str:
    """Return the QSS for ``light`` (default) or ``dark``."""
    if (theme or "").strip().lower() == THEME_DARK:
        return DARK_STYLESHEET
    return LIGHT_STYLESHEET


__all__ = [
    "APPLICATION_STYLESHEET",
    "DARK_STYLESHEET",
    "LIGHT_STYLESHEET",
    "THEME_DARK",
    "THEME_LIGHT",
    "get_stylesheet",
]
