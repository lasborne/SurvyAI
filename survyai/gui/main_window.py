"""
Professional Windows desktop shell for SurvyAI.

This window wraps the existing service layer with the product surfaces expected
from a paid desktop application: onboarding, workspace picker, prompt console,
output history, full-page settings and diagnostics (from the account menu),
cloud sign-in, and safe mode.
"""

from __future__ import annotations

import html
import json
import os
import re
import time
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QDesktopServices, QFont, QTextCursor
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QInputDialog,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from survyai.agent_service import SurvyAIAgentService
from survyai.app_config import merge_settings
from survyai.capabilities import format_capabilities_summary, scan_machine_capabilities
from survyai.credit_gate import credit_limit_enforcement_enabled
from survyai.feature_flags import FeatureFlags
from survyai.gui.manage_pcs_dialog import ManagePcsDialog
from survyai.gui.onboarding import OnboardingWizard, environment_validation_report
from survyai.cloud_user_message import user_facing_cloud_message
from survyai.cloud_api import (
    CloudApiError,
    access_token_expires_at_iso,
    get_billing_plans,
    get_bootstrap,
    get_entitlements,
    get_me,
    login,
    paystack_initialize,
    paystack_subscription_manage_url,
    paystack_verify,
    refresh_tokens,
    register_device,
    register as cloud_register,
)
from survyai.device_identity import compute_machine_fingerprint
from survyai.plan_policy import policy_for_plan
from survyai.gui.state import (
    AccountProfile,
    AppStateStore,
    Conversation,
    DesktopState,
    TaskHistoryEntry,
)
from survyai.gui.worker import AgentRunThread
from survyai.ollama_support import (
    OLLAMA_DOWNLOAD_PAGE,
    install_ollama_with_winget,
    is_ollama_installed,
    list_local_models,
    start_pull_model,
    try_connect_ollama,
)
from survyai.types import AgentRunResult
from survyai.version import __version__

# --- Follow-up vs new-topic heuristics (conversation context injection) -----------------
_FU_STOP = {
    "a", "an", "as", "at", "be", "by", "do", "go", "he", "i", "if", "in", "is",
    "it", "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we", "yes",
}
_FU_GENERIC = {
    "survey", "surveyor", "surveying", "surveys", "data", "file", "files",
    "result", "results", "information", "point", "points", "polygon", "polygons",
    "line", "lines", "plot", "plots", "plan", "plans", "report", "reports",
    "work", "workspace", "project", "output", "input", "document", "pdf",
    "app", "task", "run", "help", "using", "need", "get", "give", "make",
    "with", "from", "into", "about", "your", "this", "that", "these", "those",
    "the", "and", "for", "not", "are", "has", "was", "will", "have", "been",
    "all", "any", "out", "can", "you", "our", "but", "more", "when", "what",
    "where", "which", "user", "done", "save", "created", "contains",
}
_FU_TOPIC = {
    "nigeria", "nigerian", "legal", "legislation", "legislature", "statute", "statutes",
    "regulation", "regulations", "ordinance", "decrees", "constitution", "jurisdiction",
    "breakdown", "chronology", "timeline", "history", "background", "laws", "statutory",
}
_FU_TOKEN = re.compile(r"[a-z0-9][a-z0-9'\-_.]{2,}", re.I)
_FU_YEAR_SINCE = re.compile(
    r"\b(?:since|from|before|after|until)\s+(?:18|19|20)\d{2}\b", re.I
)
_FU_YEAR = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2})\b")


def _fu_sig_tokens(text: str) -> set[str]:
    o: set[str] = set()
    for m in _FU_TOKEN.finditer(text or ""):
        t = m.group(0).lower()
        if len(t) < 4 or t in _FU_STOP or t in _FU_GENERIC:
            continue
        o.add(t)
    return o


def _fu_topic_score(raw: str, prev: str) -> int:
    rlow, hlow = (raw or "").lower(), (prev or "").lower()
    s = 0
    for tok in _FU_TOPIC:
        if len(tok) >= 4 and tok in rlow and tok not in hlow:
            s += 1
    if _FU_YEAR_SINCE.search(raw or "") and not _FU_YEAR_SINCE.search(prev or ""):
        s += 2
    ry, hy = set(_FU_YEAR.findall(raw or "")), set(_FU_YEAR.findall(prev or ""))
    if ry and not (ry & hy):
        for y in sorted(ry):
            if y.isdigit() and int(y) < 2010:
                s += 1
                break
    if "laws" in rlow and "laws" not in hlow and "nigeria" in rlow:
        s += 1
    return min(s, 8)


def _fu_short_affirm(t: str) -> bool:
    t = (t or "").strip().lower()
    if not t:
        return False
    w = t.split()
    if len(w) > 10:
        return False
    aff = {
        "yes", "y", "yep", "yeah", "ok", "okay", "sure", "please", "proceed", "go",
        "agreed", "alright", "right", "correct", "continue", "absolutely", "sounds", "if",
    }
    w0 = w[0].strip(".,;:!?")
    if len(w) <= 2 and w0 in aff:
        return True
    if len(w) <= 4 and w0 in aff and w[-1] == "want":
        return True
    if t in ("yes, i want", "yes i want", "ok do it", "go ahead", "sounds good"):
        return True
    if len(w) <= 3 and {x.strip(".,;:!?") for x in w} <= aff:
        return True
    if len(w) <= 7 and (t.startswith("yes,") or t.startswith("ok,") or t.startswith("please ") or t.startswith("i want you")):
        return True
    return len(w) <= 5 and t.startswith("go ahead")


def _fu_anaphora(t: str) -> bool:
    s = f" {t.lower().strip()} "
    for p in (
        " the above", " the results", " the result", " the file", " the files", " the report",
        " the survey", " the same", "as above", "as discussed", "further to", "additionally",
        "the output", "the workspace", " the plan", " the cad", "word document", "appropriate title",
        "export it", " save it", " the dwg", " this pdf", "as word", " the polygons",
        " these points", " the points",
    ):
        if p in s:
            return True
    for n in (" this ", " these ", " that ", " those ", " it ", " also ", " continue "):
        if n in s:
            return True
    return "as you" in s or "further" in s


def _fu_shared_paths(history: str, raw: str) -> bool:
    h, r = history.lower(), (raw or "").lower()
    for pat in (r"([a-z]:\\[^:\n*\"<>|]{3,}[\w.])", r"([/\w.:/\\\-]+\.(?:pdf|docx?|xlsx?|dwg|dxf|csv))"):
        ih, ir_ = set(re.findall(pat, h, re.I)), set(re.findall(pat, r, re.I))
        if ih and ir_ and (ih & ir_ or any(x in h for x in ir_) or any(x in r for x in ih)):
            return True
    return False


def _should_inject_conversation_context(raw: str, prior_user_assistant_text: str) -> bool:
    """True when the new message is likely a follow-up to the last exchange, not a brand-new task."""
    raw = (raw or "").strip()
    prev = (prior_user_assistant_text or "").strip()
    if not raw or not prev:
        return False
    nw = len(raw.split())
    nt = _fu_topic_score(raw, prev)
    if _fu_short_affirm(raw):
        return True
    if _fu_anaphora(raw):
        return not (nt >= 4 and nw > 8)
    if _fu_shared_paths(prev, raw):
        return not (nt >= 3 and nw > 20)
    sn, sp = _fu_sig_tokens(raw), _fu_sig_tokens(prev)
    inter, uni = sn & sp, sn | sp
    j = len(inter) / max(1, len(uni)) if uni else 0.0
    if nt >= 2 and nw >= 8 and not _fu_shared_paths(prev, raw):
        if nt >= 3:
            return False
        rlow = raw.lower()
        if any(k in rlow for k in ("law", "laws", "nigeria")) and not inter:
            return False
    if len(inter) >= 2 or j >= 0.12:
        return not (nt >= 3)
    if len(inter) == 1 and j >= 0.08 and nt < 2:
        return True
    if nt >= 2 and nw >= 6 and not _fu_anaphora(raw) and not _fu_shared_paths(prev, raw):
        return False
    if nw >= 18 and nt >= 1 and not _fu_anaphora(raw) and not inter and not _fu_shared_paths(prev, raw):
        return False
    nr, np_ = set(re.findall(r"\b\d{2,5}\b", raw)), set(re.findall(r"\b\d{2,5}\b", prev))
    if 6 <= nw < 25 and nr and np_ and (nr & np_) and nt < 2:
        return True
    if nt == 0 and nw <= 6:
        return True
    if nt == 0 and len(inter) >= 1:
        return True
    return nt < 1 and nw < 8


# Central stack: main tabs (Console + History), then full-page Settings, Diagnostics, Credits
_PAGE_MAIN = 0
_PAGE_SETTINGS = 1
_PAGE_DIAGNOSTICS = 2
_PAGE_CREDITS = 3


def _cloud_entitlements_allow_platform_llm(me: dict, ent: dict) -> bool:
    if isinstance(ent, dict) and ent.get("can_use_platform_llm") is True:
        return True
    if isinstance(me, dict) and me.get("can_use_platform_llm") is True:
        return True
    return False


def _email_local_part(email: str) -> str:
    """Part before @ for default display name (e.g. okeke.michael@ymail.com -> okeke.michael)."""
    e = (email or "").strip()
    if "@" not in e:
        return e
    return e.split("@", 1)[0].strip()


class ChatInput(QPlainTextEdit):
    """Enter sends; Shift+Enter inserts a newline.

    Height grows with typed content up to ~2× the base row (then scrolls inside).
    """

    sendRequested = Signal()
    _BASE_HEIGHT = 132
    _MAX_HEIGHT = 260

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("chatInput")
        self.setMinimumHeight(self._BASE_HEIGHT)
        self.setMaximumHeight(self._MAX_HEIGHT)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.textChanged.connect(self._resize_to_content)
        self._resize_to_content()

    def _resize_to_content(self) -> None:
        doc = self.document()
        layout = doc.documentLayout()
        if layout is None:
            self.setFixedHeight(self._BASE_HEIGHT)
            return
        h_doc = layout.documentSize().height()
        try:
            h_px = int(h_doc) + self.frameWidth() * 2 + self.contentsMargins().top() + self.contentsMargins().bottom() + 16
        except Exception:
            fm = self.fontMetrics()
            lines = max(1, doc.blockCount())
            h_px = fm.lineSpacing() * lines + 28
        h_px = max(self._BASE_HEIGHT, min(self._MAX_HEIGHT, h_px))
        self.setFixedHeight(h_px)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (
            event.modifiers() & Qt.ShiftModifier
        ):
            self.sendRequested.emit()
            return
        super().keyPressEvent(event)


class _OllamaSetupDialog(QDialog):
    def __init__(self, parent: QWidget, *, initial_base_url: str, initial_model: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("Local models (Ollama)")
        self.setMinimumWidth(620)

        self._pull_proc = None
        self._pull_timer = QTimer(self)
        self._pull_timer.setInterval(250)
        self._pull_timer.timeout.connect(self._poll_pull_output)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        status_group = QGroupBox("Status")
        status_form = QFormLayout(status_group)
        self._installed_label = QLabel("—")
        self._server_label = QLabel("—")
        self._installed_label.setWordWrap(True)
        self._server_label.setWordWrap(True)
        status_form.addRow("Ollama", self._installed_label)
        status_form.addRow("Server", self._server_label)
        root.addWidget(status_group)

        cfg_group = QGroupBox("Configuration")
        cfg_form = QFormLayout(cfg_group)
        self._base_url_edit = QLineEdit()
        self._base_url_edit.setPlaceholderText("http://localhost:11434")
        self._base_url_edit.setText((initial_base_url or "").strip() or "http://localhost:11434")
        self._base_url_edit.textChanged.connect(self._refresh_status)
        cfg_form.addRow("Base URL", self._base_url_edit)
        root.addWidget(cfg_group)

        models_group = QGroupBox("Models")
        models_outer = QVBoxLayout(models_group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Local models"))
        self._models_combo = QComboBox()
        self._models_combo.setEditable(False)
        row.addWidget(self._models_combo, 1)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("secondaryButton")
        refresh_btn.clicked.connect(self._refresh_models)
        row.addWidget(refresh_btn)
        models_outer.addLayout(row)

        pull_row = QHBoxLayout()
        pull_row.addWidget(QLabel("Pull"))
        self._pull_edit = QLineEdit()
        self._pull_edit.setPlaceholderText("e.g. llama3.2:3b or qwen2.5:7b")
        pull_row.addWidget(self._pull_edit, 1)
        pull_btn = QPushButton("Pull model")
        pull_btn.clicked.connect(self._start_pull_clicked)
        pull_row.addWidget(pull_btn)
        models_outer.addLayout(pull_row)

        self._pull_log = QPlainTextEdit()
        self._pull_log.setReadOnly(True)
        self._pull_log.setPlaceholderText("Pull progress will appear here…")
        self._pull_log.setFixedHeight(160)
        models_outer.addWidget(self._pull_log)

        root.addWidget(models_group, 1)

        install_row = QHBoxLayout()
        self._install_btn = QPushButton("Install via winget")
        self._install_btn.clicked.connect(self._install_clicked)
        install_row.addWidget(self._install_btn)
        hint = QLabel(f"Download page: {OLLAMA_DOWNLOAD_PAGE}")
        hint.setWordWrap(True)
        install_row.addWidget(hint, 1)
        root.addLayout(install_row)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        root.addWidget(self._buttons)

        self._refresh_status()
        self._refresh_models(select=initial_model)

    def base_url(self) -> str:
        return self._base_url_edit.text().strip() or "http://localhost:11434"

    def model(self) -> str:
        return self._models_combo.currentText().strip()

    def _refresh_status(self) -> None:
        st = is_ollama_installed()
        if st.installed:
            self._installed_label.setText(f"Installed ({st.exe_path})")
        else:
            self._installed_label.setText(f"Not installed ({st.reason})")

        ok = try_connect_ollama(base_url=self.base_url(), timeout_seconds=1.0)
        self._server_label.setText("Reachable" if ok else "Not reachable (start Ollama and try again)")
        self._install_btn.setEnabled(not st.installed)

    def _refresh_models(self, *, select: str = "") -> None:
        models = list_local_models()
        self._models_combo.clear()
        if not models:
            self._models_combo.addItem("(no local models yet)")
            self._models_combo.setEnabled(False)
        else:
            self._models_combo.addItems(models)
            self._models_combo.setEnabled(True)
            sel = (select or "").strip()
            if sel and sel in models:
                self._models_combo.setCurrentText(sel)
        self._refresh_status()

    def _install_clicked(self) -> None:
        proc = install_ollama_with_winget()
        if proc is None:
            QMessageBox.warning(
                self,
                "winget not available",
                "Windows Package Manager (winget) was not found.\n\n"
                "Install Ollama from the download page, then click Refresh.",
            )
            return
        QMessageBox.information(
            self,
            "Installing Ollama…",
            "Ollama installer was started.\n\n"
            "After installation completes, click Refresh to detect it and list models.",
        )

    def _start_pull_clicked(self) -> None:
        model = self._pull_edit.text().strip()
        if not model:
            QMessageBox.information(self, "Pull model", "Enter a model name first (e.g. llama3.2:3b).")
            return
        proc = start_pull_model(model)
        if proc is None:
            QMessageBox.warning(self, "Ollama not available", "Install Ollama first, then try again.")
            return
        self._pull_proc = proc
        self._pull_log.appendPlainText(f"$ ollama pull {model}\n")
        self._pull_timer.start()

    def _poll_pull_output(self) -> None:
        proc = self._pull_proc
        if proc is None:
            self._pull_timer.stop()
            return
        try:
            while True:
                if proc.stdout is None:
                    break
                line = proc.stdout.readline()
                if not line:
                    break
                self._pull_log.appendPlainText(line.rstrip())
        except Exception:
            pass

        if proc.poll() is not None:
            code = proc.returncode
            self._pull_timer.stop()
            self._pull_proc = None
            self._pull_log.appendPlainText(f"\nPull finished (exit code {code}).\n")
            self._refresh_models(select=self._pull_edit.text().strip())


class PaystackPlanPickerDialog(QDialog):
    """Subscribe flow: show every plan option at once (no single-line combo)."""

    def __init__(self, parent: QWidget | None, plans: list[dict]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Subscribe with Paystack")
        self.setMinimumWidth(520)
        self._plan_codes: list[str] = []
        root = QVBoxLayout(self)
        intro = QLabel(
            "Choose monthly or annual billing. Both plans are listed below. "
            "You will complete payment securely in your browser."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        self._group = QButtonGroup(self)
        for i, p in enumerate(plans):
            code = str(p.get("plan_code") or "").strip()
            label = str(p.get("label") or p.get("slug") or "Plan")
            self._plan_codes.append(code)
            row = QWidget()
            row_l = QHBoxLayout(row)
            row_l.setContentsMargins(0, 4, 0, 4)
            rb = QRadioButton()
            rb.setText("")
            lbl = QLabel(label)
            lbl.setWordWrap(True)
            lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            row_l.addWidget(rb, 0, Qt.AlignTop)
            row_l.addWidget(lbl, 1)
            self._group.addButton(rb, i)
            root.addWidget(row)
        first = self._group.buttons()
        if first:
            first[0].setChecked(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def selected_plan_code(self) -> str:
        bid = self._group.checkedId()
        if bid < 0 or bid >= len(self._plan_codes):
            return ""
        return self._plan_codes[bid]


class PaystackManageSubscriptionDialog(QDialog):
    """Manage flow: show plan summaries and open Paystack portal (no cramped dropdown)."""

    def __init__(self, parent: QWidget | None, plans: list[dict], portal_url: str) -> None:
        super().__init__(parent)
        self._portal_url = portal_url
        self.setWindowTitle("Manage subscription")
        self.setMinimumWidth(520)
        root = QVBoxLayout(self)
        intro = QLabel(
            "Billing is handled on Paystack. Your current plan options are shown below. "
            "Use the button to open the customer portal (payment method, invoices, cancellation). "
            "To switch between monthly and annual, use Subscribe to Pro… and complete a new checkout."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        if plans:
            for p in plans:
                label = str(p.get("label") or p.get("slug") or "Plan")
                box = QGroupBox()
                bl = QVBoxLayout(box)
                lbl = QLabel(label)
                lbl.setWordWrap(True)
                bl.addWidget(lbl)
                root.addWidget(box)
        else:
            fallback = QLabel("Plan list was unavailable; you can still open the Paystack portal.")
            fallback.setWordWrap(True)
            root.addWidget(fallback)

        btn_box = QDialogButtonBox()
        open_btn = btn_box.addButton("Open Paystack portal", QDialogButtonBox.ButtonRole.AcceptRole)
        open_btn.setDefault(True)
        btn_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        root.addWidget(btn_box)

    def portal_url(self) -> str:
        return self._portal_url


def _paystack_user_message(server_message: str) -> str:
    """Append setup hint when checkout fails because Paystack is not configured on the server."""
    s = (server_message or "").strip()
    low = s.lower()
    if "not configured" in low and "paystack" in low:
        return (
            f"{s}\n\n"
            "On the machine running the cloud API, set PAYSTACK_SECRET_KEY in .env.cloud "
            "(and PAYSTACK_PUBLIC_KEY if your server expects it), then restart the API process."
        )
    return s


def _cloud_error_text_for_user(exc: BaseException) -> str:
    """Map API errors to end-user copy; keep Paystack secret-key hints where relevant."""
    raw = str(exc).strip()
    low = raw.lower()
    if "not configured" in low and "paystack" in low:
        return _paystack_user_message(raw)
    return user_facing_cloud_message(exc)


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        initial_query: Optional[str] = None,
        auto_run_query: bool = False,
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"SurvyAI Desktop — {__version__}")
        self.resize(1420, 900)
        self.setMinimumSize(640, 450)

        self._state_store = AppStateStore()
        self._state: DesktopState = self._state_store.load()
        self._caps = scan_machine_capabilities()
        self._display_feature_flags = FeatureFlags.from_env()
        self._feature_flags = self._display_feature_flags
        self._settings = self._effective_settings()
        self._service = SurvyAIAgentService(
            settings=self._settings,
            feature_flags=self._effective_feature_flags(),
            eager_init=False,
        )

        self._thread: Optional[AgentRunThread] = None
        active_conversation = self._state_store.ensure_conversations(self._state)
        self._session_id = active_conversation.session_id
        self._active_conversation_id = active_conversation.conversation_id
        self._last_query = ""
        self._pending_plain_query: Optional[str] = None
        self._run_started_at = 0.0
        self._run_stage = -1
        self._conversation_list_sync = False
        self._startup_initial_query = (initial_query or "").strip()
        self._startup_auto_run = bool(auto_run_query and self._startup_initial_query)

        self._console_content_panel: Optional[QWidget] = None

        self._build_ui()
        self._apply_chrome()
        self._build_menu()
        self._apply_state_to_ui()
        self._install_status_indicators()
        self._refresh_all_views()
        self._refresh_active_llm_status()

        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(2500)
        self._progress_timer.timeout.connect(self._on_progress_tick)

        self._desktop_state_save_timer = QTimer(self)
        self._desktop_state_save_timer.setSingleShot(True)
        self._desktop_state_save_timer.setInterval(200)
        self._desktop_state_save_timer.timeout.connect(self._flush_desktop_state_save)

        QTimer.singleShot(0, self._finish_startup)
        # Credit strip lays out after first frame; refresh once the prompt row has geometry.
        QTimer.singleShot(0, self._update_credit_usage_notice)
        QTimer.singleShot(0, self._apply_prompt_controls_scale)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        QTimer.singleShot(50, self._update_credit_usage_notice)
        QTimer.singleShot(0, self._apply_prompt_controls_scale)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        QTimer.singleShot(0, self._apply_prompt_controls_scale)

    def _apply_prompt_controls_scale(self) -> None:
        """Resize prompt strip buttons/checkboxes with available width (no fixed strip)."""
        panel = getattr(self, "_console_content_panel", None)
        if panel is None:
            return
        cw = int(panel.contentsRect().width())
        if cw <= 0:
            return

        # Narrower console content -> slightly smaller type and buttons (smooth band).
        lo, hi = 380.0, 1080.0
        t = (cw - lo) / (hi - lo)
        t = max(0.0, min(1.0, t))
        scale = 0.66 + 0.34 * t

        base_pt = self.font().pointSizeF()
        if base_pt <= 0:
            base_pt = 9.0
        pt = max(7.25, min(10.75, base_pt * scale))
        f = QFont(self.font())
        f.setPointSizeF(pt)

        btn_min_h = max(22, int(24 * scale + 6))
        for name in ("_send_btn", "_cancel_btn", "_retry_btn"):
            w = getattr(self, name, None)
            if w is None:
                continue
            w.setFont(f)
            w.setMinimumHeight(btn_min_h)
        for name in ("_fallback_cb", "_fast_mode_cb"):
            w = getattr(self, name, None)
            if w is None:
                continue
            w.setFont(f)

    def _install_status_indicators(self) -> None:
        """Add small always-visible status indicators in the status bar."""
        try:
            sb = self.statusBar()
            self._fast_mode_indicator = QLabel("")
            self._fast_mode_indicator.setToolTip(
                "Fast mode affects only non-file prompts.\n"
                "CAD/doc/ArcGIS/file workflows are unchanged."
            )
            self._fast_mode_indicator.setTextInteractionFlags(Qt.TextSelectableByMouse)
            sb.addPermanentWidget(self._fast_mode_indicator)
            self._refresh_fast_mode_indicator()
        except Exception:
            # Cosmetic only; never block app startup.
            pass

    def _refresh_fast_mode_indicator(self) -> None:
        if not hasattr(self, "_fast_mode_indicator"):
            return
        enabled = bool(getattr(self._state, "fast_mode_non_file_prompts", False))
        self._fast_mode_indicator.setText(f"Fast mode: {'ON' if enabled else 'OFF'}")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _apply_chrome(self) -> None:
        # Visual polish only; keep functional behavior unchanged.
        self.setStyleSheet(
            """
QMainWindow {
  background: #f6f7fb;
}

QTabWidget::pane {
  border: 1px solid #d9dce6;
  border-radius: 10px;
  background: #ffffff;
  top: -1px;
}

QTabBar::tab {
  background: transparent;
  padding: 6px 12px;
  margin-right: 4px;
  border: 1px solid transparent;
  border-top-left-radius: 10px;
  border-top-right-radius: 10px;
  color: #2b2f3a;
}
QTabBar::tab:selected {
  background: #ffffff;
  border-color: #d9dce6;
  border-bottom-color: #ffffff;
  font-weight: 600;
}
QTabBar::tab:hover:!selected {
  background: #eef1f8;
  border-color: #e2e6f2;
}

QGroupBox {
  border: 1px solid #e1e4ee;
  border-radius: 10px;
  margin-top: 12px;
  background: #ffffff;
}
QGroupBox::title {
  subcontrol-origin: margin;
  left: 10px;
  padding: 0 6px;
  color: #2b2f3a;
  font-weight: 600;
}

QScrollArea {
  border: none;
  background: transparent;
}
QScrollArea > QWidget > QWidget {
  background: transparent;
}

QLabel#titleLabel {
  color: #1f2430;
}
QLabel#wordmarkLabel {
  color: #0f172a;
  font-weight: 900;
  letter-spacing: 0.2px;
}
QLabel#wordmarkLabel .accent {
  color: #2f6fed;
}
QLabel#wordmarkSub {
  color: #475569;
  font-weight: 700;
  padding-left: 2px;
}
QLabel#hintLabel {
  color: #4b5162;
}
QLabel#consoleHintLabel {
  color: #5c6378;
  font-size: 11px;
  margin: 0 0 2px 0;
}
QLabel#sectionHeader {
  color: #2b2f3a;
  font-weight: 700;
  padding: 2px 2px 6px 2px;
}

QLineEdit, QTextEdit, QPlainTextEdit, QListWidget {
  border: 1px solid #d9dce6;
  border-radius: 10px;
  padding: 8px 10px;
  background: #ffffff;
  selection-background-color: #1d4ed8;
  selection-color: #ffffff;
}
QComboBox {
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  padding: 6px 12px;
  padding-right: 32px;
  min-height: 22px;
  background: #ffffff;
  selection-background-color: #1d4ed8;
  selection-color: #ffffff;
}
QComboBox:hover {
  border-color: #94a3b8;
  background: #fafbfc;
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
QTextEdit, QPlainTextEdit {
  padding: 10px 12px;
}

QPushButton {
  border: 1px solid #cfd5e6;
  border-radius: 10px;
  padding: 8px 12px;
  background: #2f6fed;
  color: #ffffff;
  font-weight: 600;
}
QPushButton:hover {
  background: #2a63d6;
}
QPushButton:pressed {
  background: #2558bf;
}
QPushButton:disabled {
  background: #cdd3e4;
  border-color: #cdd3e4;
  color: #6a7186;
}

QPushButton#secondaryButton {
  background: #ffffff;
  color: #2b2f3a;
  border: 1px solid #cfd5e6;
  font-weight: 600;
}
QPushButton#secondaryButton:hover {
  background: #eef1f8;
}

QPushButton#summaryButton {
  background: #ffffff;
  color: #2b2f3a;
  border: 1px solid #d9dce6;
  font-weight: 600;
  padding: 8px 10px;
}
QPushButton#summaryButton:hover {
  background: #eef1f8;
}

/* Account chip styling lives in survyai/gui/styles.py (APPLICATION_STYLESHEET). */

QPushButton#statusChipButton {
  background: #fff7ed;
  color: #7a3b00;
  border: 1px solid #ffd7aa;
  font-weight: 700;
  padding: 6px 10px;
}
QPushButton#statusChipButton:checked {
  background: #fee2e2;
  color: #7f1d1d;
  border: 1px solid #fecaca;
}
            """.strip()
        )

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 6, 12, 10)
        root.setSpacing(6)

        top_bar_outer = QVBoxLayout()
        top_bar_outer.setSpacing(4)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)
        title_wrap = QWidget()
        title_row = QHBoxLayout(title_wrap)
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(6)
        title = QLabel('<span class="accent">Survy</span>AI')
        title.setObjectName("wordmarkLabel")
        title.setTextFormat(Qt.TextFormat.RichText)
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        sub = QLabel("Desktop")
        sub.setObjectName("wordmarkSub")
        title_row.addWidget(title)
        title_row.addWidget(sub)
        title_row.addStretch()
        self._back_workspace_btn = QPushButton("Back to workspace")
        self._back_workspace_btn.setObjectName("secondaryButton")
        self._back_workspace_btn.setVisible(False)
        self._back_workspace_btn.setToolTip("Return to Console and Output History")
        self._back_workspace_btn.clicked.connect(self._back_to_workspace)
        top_row.addWidget(self._back_workspace_btn)
        top_row.addWidget(title_wrap, 1)
        self._user_menu_btn = QToolButton()
        self._user_menu_btn.setObjectName("userMenuButtonGuest")
        self._user_menu_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._user_menu_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._user_menu_btn.setAutoRaise(False)
        self._user_menu = QMenu(self._user_menu_btn)
        self._user_menu.setToolTipsVisible(True)
        self._user_menu_btn.setMenu(self._user_menu)
        self._user_menu_btn.setText("Login / Create account \u25be")
        top_row.addWidget(self._user_menu_btn)
        top_bar_outer.addLayout(top_row)

        workspace_row = QHBoxLayout()
        workspace_row.setSpacing(8)
        ws_label = QLabel("Workspace")
        ws_label.setMinimumWidth(64)
        workspace_row.addWidget(ws_label)
        self._workspace_edit = QLineEdit()
        self._workspace_edit.setPlaceholderText("Choose a workspace / project folder")
        self._workspace_edit.setToolTip("Folder used for prompts, exports, and project context.")
        self._workspace_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        workspace_row.addWidget(self._workspace_edit, 1)
        browse_workspace = QPushButton("Browse…")
        browse_workspace.setObjectName("secondaryButton")
        browse_workspace.clicked.connect(self._choose_workspace)
        browse_workspace.setToolTip("Pick a workspace folder.")
        workspace_row.addWidget(browse_workspace)
        open_workspace = QPushButton("Open")
        open_workspace.setObjectName("secondaryButton")
        open_workspace.clicked.connect(self._open_workspace_folder)
        open_workspace.setToolTip("Open the workspace folder in File Explorer.")
        workspace_row.addWidget(open_workspace)
        top_bar_outer.addLayout(workspace_row)

        root.addLayout(top_bar_outer)

        self._central_stack = QStackedWidget()
        self._tabs = QTabWidget()
        self._central_stack.addWidget(self._tabs)
        root.addWidget(self._central_stack, 1)

        self._build_console_tab()
        self._build_history_tab()
        self._settings_page = self._build_settings_page()
        self._diagnostics_page = self._build_diagnostics_page()
        self._credits_page = self._build_credits_page()
        self._central_stack.addWidget(self._settings_page)
        self._central_stack.addWidget(self._diagnostics_page)
        self._central_stack.addWidget(self._credits_page)

    def _build_console_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 6, 10, 10)
        layout.setSpacing(6)

        hint = QLabel(
            "Type in your prompt and click the Send button."
        )
        hint.setObjectName("consoleHintLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        body_split = QSplitter(Qt.Horizontal)

        conversation_panel = QWidget()
        conversation_panel.setObjectName("sidebarConversations")
        conversation_panel.setMinimumWidth(176)
        conversation_layout = QVBoxLayout(conversation_panel)
        conversation_layout.setContentsMargins(10, 10, 10, 10)
        conv_hdr = QLabel("Conversations")
        conv_hdr.setObjectName("sectionHeader")
        conversation_layout.addWidget(conv_hdr)
        self._conversation_list = QListWidget()
        self._conversation_list.setObjectName("conversationList")
        self._conversation_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._conversation_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._conversation_list.currentItemChanged.connect(self._on_conversation_changed)
        conversation_layout.addWidget(self._conversation_list, 1)

        conversation_actions = QHBoxLayout()
        new_conv_btn = QPushButton("New")
        new_conv_btn.setObjectName("secondaryButton")
        new_conv_btn.clicked.connect(self._new_session)
        conversation_actions.addWidget(new_conv_btn)
        delete_conv_btn = QPushButton("Delete")
        delete_conv_btn.setObjectName("secondaryButton")
        delete_conv_btn.clicked.connect(self._delete_selected_conversation)
        conversation_actions.addWidget(delete_conv_btn)
        conversation_layout.addLayout(conversation_actions)
        body_split.addWidget(conversation_panel)

        content_panel = QWidget()
        self._console_content_panel = content_panel
        content_layout = QVBoxLayout(content_panel)
        content_layout.setContentsMargins(0, 0, 0, 0)

        split = QSplitter(Qt.Horizontal)
        self._transcript = QTextEdit()
        self._transcript.setReadOnly(True)
        self._transcript.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self._transcript.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._transcript.setPlaceholderText("Conversation will appear here…")
        self._transcript.setMinimumWidth(200)
        self._transcript.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        split.addWidget(self._transcript)

        activity_panel = QWidget()
        activity_panel.setMinimumWidth(72)
        activity_layout = QVBoxLayout(activity_panel)
        activity_layout.setContentsMargins(0, 0, 0, 0)
        act_hdr = QLabel("Live activity")
        act_hdr.setObjectName("sectionHeader")
        activity_layout.addWidget(act_hdr)
        self._activity_log = QPlainTextEdit()
        self._activity_log.setReadOnly(True)
        self._activity_log.setPlaceholderText("Progress, long-run updates, and troubleshooting notes…")
        activity_layout.addWidget(self._activity_log, 1)
        self._run_status_label = QLabel("Ready")
        self._elapsed_label = QLabel("Elapsed: 0s")
        activity_layout.addWidget(self._run_status_label)
        activity_layout.addWidget(self._elapsed_label)
        split.addWidget(activity_panel)
        split.setSizes([780, 240])
        split.setChildrenCollapsible(True)
        # Favor transcript width (primary reading surface); activity stays secondary.
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 1)
        content_layout.addWidget(split, 1)

        # Prompt + controls beside input: flexible width (no fixed strip); scaled on resize.
        input_row = QHBoxLayout()
        input_row.setSpacing(10)
        self._input = ChatInput()
        self._input.setPlaceholderText("Describe your task… (Enter to send, Shift+Enter for newline)")
        self._input.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self._input.sendRequested.connect(self._on_send_clicked)
        input_row.addWidget(self._input, 1)

        controls = QVBoxLayout()
        controls.setSpacing(4)
        self._send_btn = QPushButton("Send")
        self._send_btn.setDefault(True)
        self._send_btn.clicked.connect(self._on_send_clicked)
        self._send_btn.setToolTip("Send prompt (Enter).")
        self._send_btn.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed
        )
        controls.addWidget(self._send_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("secondaryButton")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._request_cancel_current_run)
        self._cancel_btn.setToolTip("Stop the current run.")
        self._cancel_btn.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed
        )
        controls.addWidget(self._cancel_btn)

        self._retry_btn = QPushButton("Retry last")
        self._retry_btn.setObjectName("secondaryButton")
        self._retry_btn.clicked.connect(self._retry_last_query)
        self._retry_btn.setToolTip("Re-run the last prompt.")
        self._retry_btn.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed
        )
        controls.addWidget(self._retry_btn)

        self._fallback_cb = QCheckBox("Use fallback LLM")
        self._fallback_cb.toggled.connect(self._on_fallback_toggled)
        self._fallback_cb.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed
        )
        controls.addWidget(self._fallback_cb)

        self._fast_mode_cb = QCheckBox("Fast mode (non-file prompts)")
        self._fast_mode_cb.setToolTip(
            "Faster responses for general questions by bypassing tool planning.\n"
            "CAD/doc/ArcGIS/file workflows are unchanged."
        )
        self._fast_mode_cb.toggled.connect(self._on_fast_mode_toggled)
        self._fast_mode_cb.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed
        )
        controls.addWidget(self._fast_mode_cb)
        controls.addStretch()

        controls_wrap = QWidget()
        controls_wrap.setLayout(controls)
        controls_wrap.setMinimumWidth(0)
        controls_wrap.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )
        input_row.addWidget(controls_wrap, 0, Qt.AlignmentFlag.AlignTop)

        content_layout.addLayout(input_row)

        # Subtle, low-contrast line under the prompt: credit-usage planning (50/80/95% or exhausted).
        self._credit_notice_wrap = QWidget()
        self._credit_notice_wrap.setObjectName("creditUsageNoticeShell")
        credit_notice_row = QHBoxLayout(self._credit_notice_wrap)
        credit_notice_row.setContentsMargins(10, 0, 10, 4)
        credit_notice_row.setSpacing(6)
        self._credit_notice_label = QLabel("")
        self._credit_notice_label.setObjectName("creditUsageNoticeLabel")
        self._credit_notice_label.setWordWrap(True)
        self._credit_notice_dismiss_btn = QToolButton()
        self._credit_notice_dismiss_btn.setObjectName("creditUsageNoticeDismiss")
        self._credit_notice_dismiss_btn.setText("\u00d7")
        self._credit_notice_dismiss_btn.setToolTip("Dismiss this reminder")
        self._credit_notice_dismiss_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._credit_notice_dismiss_btn.setVisible(False)
        self._credit_notice_dismiss_btn.setAutoRaise(True)
        self._credit_notice_dismiss_btn.clicked.connect(self._on_credit_notice_dismiss_clicked)
        credit_notice_row.addWidget(self._credit_notice_label, 1)
        credit_notice_row.addWidget(self._credit_notice_dismiss_btn, 0, Qt.AlignmentFlag.AlignTop)
        self._credit_notice_wrap.setVisible(False)
        self._credit_notice_current_band = "none"
        content_layout.addWidget(self._credit_notice_wrap)

        body_split.addWidget(content_panel)
        body_split.setSizes([220, 980])
        body_split.setChildrenCollapsible(True)
        body_split.setStretchFactor(0, 0)
        body_split.setStretchFactor(1, 1)
        layout.addWidget(body_split, 1)
        self._tabs.addTab(tab, "Console")

    def _build_history_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        split = QSplitter(Qt.Horizontal)

        self._history_list = QListWidget()
        self._history_list.currentItemChanged.connect(self._on_history_selection_changed)
        split.addWidget(self._history_list)

        self._history_detail = QPlainTextEdit()
        self._history_detail.setReadOnly(True)
        split.addWidget(self._history_detail)
        split.setSizes([360, 640])
        layout.addWidget(split, 1)

        actions = QHBoxLayout()
        use_query = QPushButton("Use selected query")
        use_query.setObjectName("secondaryButton")
        use_query.clicked.connect(self._reuse_selected_history_query)
        actions.addWidget(use_query)
        retry_selected = QPushButton("Retry selected")
        retry_selected.setObjectName("secondaryButton")
        retry_selected.clicked.connect(self._retry_selected_history_item)
        actions.addWidget(retry_selected)
        actions.addStretch()
        layout.addLayout(actions)
        self._tabs.addTab(tab, "Output History")

    def _build_settings_page(self) -> QWidget:
        """Full-page settings (opened from the account menu, not a main tab)."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(scroll, 1)

        page = QWidget()
        scroll.setWidget(page)

        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(14, 14, 14, 14)
        page_layout.setSpacing(12)

        settings_title = QLabel("Settings")
        settings_title.setObjectName("pageTitle")
        settings_sub = QLabel(
            "Your profile, cloud subscription, and how SurvyAI runs on this computer. "
            "Changes here apply to this desktop session unless noted."
        )
        settings_sub.setObjectName("pageSubtitle")
        settings_sub.setWordWrap(True)
        page_layout.addWidget(settings_title)
        page_layout.addWidget(settings_sub)

        account_group = QGroupBox("Profile")
        account_outer = QVBoxLayout(account_group)
        account_hint = QLabel(
            "Sign in from the account menu (top right). Your name and email are shown here after "
            "you connect to SurvyAI Cloud. Company may be set during onboarding."
        )
        account_hint.setWordWrap(True)
        account_hint.setObjectName("hintLabel")
        account_outer.addWidget(account_hint)
        account_form = QFormLayout()
        account_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        account_form.setFormAlignment(Qt.AlignTop)
        account_form.setHorizontalSpacing(14)
        account_form.setVerticalSpacing(10)
        self._account_name_value = QLabel("—")
        self._account_email_value = QLabel("—")
        self._account_company_value = QLabel("—")
        self._account_email_value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        account_form.addRow("Name", self._account_name_value)
        account_form.addRow("Email", self._account_email_value)
        account_form.addRow("Company", self._account_company_value)
        account_outer.addLayout(account_form)
        page_layout.addWidget(account_group)

        pay_group = QGroupBox("Pro subscription (Paystack)")
        pay_outer = QVBoxLayout(pay_group)
        pay_outer.setSpacing(10)
        pay_hint = QLabel(
            "After you sign in, subscribe or manage billing below. "
            "Use Refresh cloud account after payment if your server has no webhooks yet. "
            "Use Manage PCs… to remove old computers from your device slots."
        )
        pay_hint.setWordWrap(True)
        pay_hint.setObjectName("hintLabel")
        pay_outer.addWidget(pay_hint)
        self._billing_banner = QLabel("")
        self._billing_banner.setWordWrap(True)
        self._billing_banner.setVisible(False)
        pay_outer.addWidget(self._billing_banner)
        pay_row = QHBoxLayout()
        pay_row.setSpacing(8)
        self._paystack_subscribe_btn = QPushButton("Subscribe to Pro…")
        self._paystack_subscribe_btn.clicked.connect(self._on_paystack_subscribe)
        self._paystack_subscribe_btn.setToolTip("Choose monthly or annual, then complete checkout in your browser.")
        pay_row.addWidget(self._paystack_subscribe_btn)
        self._paystack_manage_btn = QPushButton("Manage subscription…")
        self._paystack_manage_btn.setObjectName("secondaryButton")
        self._paystack_manage_btn.clicked.connect(self._on_paystack_manage_subscription)
        self._paystack_manage_btn.setToolTip("Open Paystack portal for card updates, invoices, cancellation.")
        pay_row.addWidget(self._paystack_manage_btn)
        self._paystack_verify_btn = QPushButton("Verify payment reference…")
        self._paystack_verify_btn.setObjectName("secondaryButton")
        self._paystack_verify_btn.clicked.connect(self._on_paystack_verify_reference)
        self._paystack_verify_btn.setToolTip("Use if webhooks are not set up yet: paste the Paystack reference.")
        pay_row.addWidget(self._paystack_verify_btn)
        self._cloud_refresh_license_btn = QPushButton("Refresh cloud account")
        self._cloud_refresh_license_btn.setObjectName("secondaryButton")
        self._cloud_refresh_license_btn.clicked.connect(self._on_refresh_cloud_license)
        self._cloud_refresh_license_btn.setToolTip("Pull latest plan/entitlements from the cloud API.")
        pay_row.addWidget(self._cloud_refresh_license_btn)
        self._manage_pcs_btn = QPushButton("Manage PCs…")
        self._manage_pcs_btn.setObjectName("secondaryButton")
        self._manage_pcs_btn.clicked.connect(self._on_manage_pcs)
        self._manage_pcs_btn.setToolTip(
            "View or remove computers registered to your account for hosted Pro (device slots)."
        )
        pay_row.addWidget(self._manage_pcs_btn)
        pay_row.addStretch()
        pay_outer.addLayout(pay_row)
        page_layout.addWidget(pay_group)

        runtime_group = QGroupBox("Runtime and safety")
        runtime_form = QFormLayout(runtime_group)
        runtime_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        runtime_form.setFormAlignment(Qt.AlignTop)
        runtime_form.setHorizontalSpacing(14)
        runtime_form.setVerticalSpacing(10)
        self._primary_llm_combo = QComboBox()
        self._primary_llm_combo.addItems(["openai", "gemini", "claude", "deepseek", "ollama"])
        self._fallback_llm_combo = QComboBox()
        self._fallback_llm_combo.addItems(["gemini", "openai", "claude", "deepseek", "ollama"])
        self._settings_workspace = QLineEdit()
        self._settings_data_folder = QLineEdit()
        self._settings_workspace.setPlaceholderText("e.g. C:\\Users\\You\\Projects\\SurvyAI")
        self._settings_data_folder.setPlaceholderText("Where SurvyAI stores data and exports")
        runtime_form.addRow("Primary LLM", self._primary_llm_combo)
        runtime_form.addRow("Fallback LLM", self._fallback_llm_combo)
        self._safe_mode_cb = QCheckBox("Enable safe mode (troubleshooting)")
        self._safe_mode_cb.toggled.connect(self._on_safe_mode_toggled)
        runtime_form.addRow("Safe mode", self._safe_mode_cb)
        self._safe_mode_note = QLabel(
            "Safe mode disables advanced external integrations for troubleshooting."
        )
        self._safe_mode_note.setWordWrap(True)
        runtime_form.addRow("", self._safe_mode_note)

        browse_ws = QPushButton("Browse…")
        browse_ws.setObjectName("secondaryButton")
        browse_ws.clicked.connect(self._choose_workspace)
        browse_ws.setToolTip("Pick the workspace folder.")
        runtime_form.addRow("Workspace", self._pair_widget(self._settings_workspace, browse_ws))

        browse_data = QPushButton("Browse…")
        browse_data.setObjectName("secondaryButton")
        browse_data.clicked.connect(self._choose_data_folder)
        browse_data.setToolTip("Pick the data folder.")
        runtime_form.addRow("Data folder", self._pair_widget(self._settings_data_folder, browse_data))

        save_runtime = QPushButton("Apply settings")
        save_runtime.clicked.connect(self._apply_runtime_settings)
        save_runtime.setToolTip("Apply changes for this desktop session.")
        runtime_form.addRow("", save_runtime)
        page_layout.addWidget(runtime_group)

        llm_status = QGroupBox("Active LLMs")
        llm_form = QFormLayout(llm_status)
        llm_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        llm_form.setFormAlignment(Qt.AlignTop)
        llm_form.setHorizontalSpacing(14)
        llm_form.setVerticalSpacing(10)
        self._active_primary_llm_label = QLabel("—")
        self._active_fallback_llm_label = QLabel("—")
        self._active_primary_llm_label.setWordWrap(True)
        self._active_fallback_llm_label.setWordWrap(True)
        llm_form.addRow("Primary", self._active_primary_llm_label)
        llm_form.addRow("Fallback", self._active_fallback_llm_label)
        page_layout.addWidget(llm_status)

        desktop_status = QGroupBox("Desktop status")
        desktop_form = QFormLayout(desktop_status)
        desktop_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        desktop_form.setFormAlignment(Qt.AlignTop)
        desktop_form.setHorizontalSpacing(14)
        desktop_form.setVerticalSpacing(10)
        self._license_settings_label = QLabel("")
        self._license_settings_label.setWordWrap(True)
        self._session_settings_label = QLabel("")
        self._session_settings_label.setWordWrap(True)
        self._machine_settings_label = QLabel("")
        self._machine_settings_label.setWordWrap(True)
        desktop_form.addRow("License", self._license_settings_label)
        desktop_form.addRow("Session", self._session_settings_label)
        desktop_form.addRow("Machine", self._machine_settings_label)
        page_layout.addWidget(desktop_status)

        page_layout.addStretch()
        return tab

    def _build_diagnostics_page(self) -> QWidget:
        """Full-page diagnostics (opened from the account menu, not a main tab)."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(scroll, 1)

        page = QWidget()
        scroll.setWidget(page)
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(14, 14, 14, 14)
        page_layout.setSpacing(12)

        diag_title = QLabel("Diagnostics")
        diag_title.setObjectName("pageTitle")
        diag_sub = QLabel(
            "Technical details for support and troubleshooting. "
            "Export a bundle or open the log folder when asked. Nothing here changes your project data."
        )
        diag_sub.setObjectName("pageSubtitle")
        diag_sub.setWordWrap(True)
        page_layout.addWidget(diag_title)
        page_layout.addWidget(diag_sub)

        self._diagnostics_text = QPlainTextEdit()
        self._diagnostics_text.setReadOnly(True)
        page_layout.addWidget(self._diagnostics_text, 1)

        footer = QWidget()
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(14, 8, 14, 14)
        footer_layout.setSpacing(10)

        actions = QHBoxLayout()
        export_btn = QPushButton("Export diagnostics bundle")
        export_btn.clicked.connect(self._export_diagnostics_bundle)
        actions.addWidget(export_btn)

        open_log = QPushButton("Open log folder")
        open_log.setObjectName("secondaryButton")
        open_log.clicked.connect(self._open_log_folder)
        actions.addWidget(open_log)

        refresh = QPushButton("Refresh diagnostics")
        refresh.setObjectName("secondaryButton")
        refresh.clicked.connect(self._refresh_diagnostics)
        actions.addWidget(refresh)

        actions.addStretch()
        footer_layout.addLayout(actions)
        layout.addWidget(footer)
        return tab

    def _build_credits_page(self) -> QWidget:
        """Full-page Credits & Usage (opened from the account menu)."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(scroll, 1)

        page = QWidget()
        scroll.setWidget(page)
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(14, 14, 14, 14)
        page_layout.setSpacing(12)

        credits_title = QLabel("Credits & usage")
        credits_title.setObjectName("pageTitle")
        credits_sub = QLabel(
            "Your subscription defines a USD credit pool for hosted models — converted from your "
            "payment amount using the exchange rate in effect on the server when you subscribed or renewed. "
            "Usage is shown and deducted in US dollars against that pool."
        )
        credits_sub.setObjectName("pageSubtitle")
        credits_sub.setWordWrap(True)
        page_layout.addWidget(credits_title)
        page_layout.addWidget(credits_sub)

        # --- Summary card ---
        summary_group = QGroupBox("Subscription credits")
        summary_outer = QVBoxLayout(summary_group)
        self._credits_period_note = QLabel("")
        self._credits_period_note.setWordWrap(True)
        self._credits_period_note.setObjectName("hintLabel")
        summary_outer.addWidget(self._credits_period_note)

        summary_form = QFormLayout()
        summary_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        summary_form.setHorizontalSpacing(14)
        summary_form.setVerticalSpacing(10)

        self._credits_total_label = QLabel("—")
        self._credits_used_label = QLabel("—")
        self._credits_remaining_label = QLabel("—")
        self._credits_pct_label = QLabel("")
        self._credits_total_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._credits_used_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._credits_remaining_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        summary_form.addRow("Credit pool (subscription USD)", self._credits_total_label)
        summary_form.addRow("Used (USD)", self._credits_used_label)
        summary_form.addRow("Remaining", self._credits_remaining_label)
        summary_form.addRow("", self._credits_pct_label)
        summary_outer.addLayout(summary_form)
        page_layout.addWidget(summary_group)

        # --- CAD potential card ---
        cad_group = QGroupBox("Estimated CAD plans remaining")
        cad_layout = QVBoxLayout(cad_group)
        self._credits_cad_label = QLabel("—")
        self._credits_cad_label.setWordWrap(True)
        self._credits_cad_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        cad_hint = QLabel(
            "This is an estimate based on average per-run cost and available credit. "
            "Actual cost varies by prompt complexity, document size, and model used."
        )
        cad_hint.setWordWrap(True)
        cad_hint.setObjectName("hintLabel")
        cad_layout.addWidget(self._credits_cad_label)
        cad_layout.addWidget(cad_hint)
        page_layout.addWidget(cad_group)

        # --- Usage history (last N runs from output history) ---
        usage_group = QGroupBox("Recent usage (USD)")
        usage_layout = QVBoxLayout(usage_group)
        self._credits_history_text = QPlainTextEdit()
        self._credits_history_text.setReadOnly(True)
        self._credits_history_text.setMaximumHeight(280)
        usage_layout.addWidget(self._credits_history_text)
        page_layout.addWidget(usage_group)

        # --- Refresh ---
        btn_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh from cloud")
        refresh_btn.clicked.connect(self._on_refresh_credits_from_cloud)
        btn_row.addWidget(refresh_btn)
        btn_row.addStretch()
        page_layout.addLayout(btn_row)

        page_layout.addStretch()
        return tab

    def _build_machine_card(self) -> QGroupBox:
        group = QGroupBox("Machine and environment")
        layout = QVBoxLayout(group)
        self._caps_label = QLabel("")
        self._caps_label.setWordWrap(True)
        self._caps_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._caps_label)
        refresh = QPushButton("Refresh detection")
        refresh.setObjectName("secondaryButton")
        refresh.clicked.connect(self._refresh_capabilities)
        layout.addWidget(refresh)
        return group

    def _build_session_card(self) -> QGroupBox:
        group = QGroupBox("Session")
        layout = QVBoxLayout(group)
        self._session_label = QLabel(self._session_id)
        self._session_label.setWordWrap(True)
        self._session_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._session_status = QLabel("Ready")
        self._session_status.setWordWrap(True)
        layout.addWidget(QLabel("Conversation ID"))
        layout.addWidget(self._session_label)
        layout.addWidget(self._session_status)
        new_conv = QPushButton("New conversation")
        new_conv.setObjectName("secondaryButton")
        new_conv.clicked.connect(self._new_session)
        layout.addWidget(new_conv)
        return group

    @staticmethod
    def _describe_menu_action(action: QAction, help_text: str) -> None:
        """Status bar + hover text for menu items (novice-friendly)."""
        t = (help_text or "").strip()
        if not t:
            return
        action.setStatusTip(t)
        action.setToolTip(t)

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        file_menu.setToolTipsVisible(True)
        file_menu.menuAction().setToolTip(
            "Export files from SurvyAI and exit safely — hover each command below for details."
        )
        file_menu.menuAction().setStatusTip(
            "Exports, diagnostics bundle for support, and quit the application."
        )

        export_transcript = QAction("Export transcript…", self)
        export_transcript.triggered.connect(self._export_transcript)
        self._describe_menu_action(
            export_transcript,
            "Save the current conversation as HTML/text so you can archive it, attach it to email, "
            "or keep a record outside the app.",
        )
        file_menu.addAction(export_transcript)

        export_diag = QAction("Export diagnostics bundle…", self)
        export_diag.triggered.connect(self._export_diagnostics_bundle)
        self._describe_menu_action(
            export_diag,
            "Create a ZIP with logs, desktop settings snapshot, and capability info — send this to "
            "support when something fails or behaves oddly.",
        )
        file_menu.addAction(export_diag)

        file_menu.addSeparator()
        exit_act = QAction("E&xit", self)
        exit_act.triggered.connect(self.close)
        self._describe_menu_action(
            exit_act,
            "Close SurvyAI. If a task is running, cancel it first or wait for it to finish.",
        )
        file_menu.addAction(exit_act)

        account_menu = menubar.addMenu("&Account")
        account_menu.setToolTipsVisible(True)
        account_menu.menuAction().setToolTip(
            "Cloud sign-in, billing, subscription credits, local models, and device registration — "
            "hover each item for what it does."
        )
        account_menu.menuAction().setStatusTip(
            "Sign in, manage Pro subscription, credits, Ollama, settings, and PCs."
        )

        act_cloud = QAction("Sign in or create account…", self)
        act_cloud.triggered.connect(self._cloud_sign_in)
        self._describe_menu_action(
            act_cloud,
            "Connect this SurvyAI desktop to your cloud account so hosted models, credits, and "
            "Paystack billing apply to this PC.",
        )
        account_menu.addAction(act_cloud)

        act_ollama = QAction("Local models (Ollama)…", self)
        act_ollama.triggered.connect(self._open_ollama_setup)
        self._describe_menu_action(
            act_ollama,
            "Run free AI models on your own computer — useful offline or when you prefer not to use "
            "cloud APIs. Install Ollama once, then pick a model here.",
        )
        account_menu.addAction(act_ollama)

        act_settings = QAction("Settings…", self)
        act_settings.triggered.connect(self._show_settings_page)
        self._describe_menu_action(
            act_settings,
            "Workspace folder, primary/fallback LLM, data folder, safe mode, and subscription-related "
            "account fields — full-screen settings panel.",
        )
        account_menu.addAction(act_settings)

        act_credits = QAction("Credits && Usage…", self)
        act_credits.triggered.connect(self._show_credits_page)
        self._describe_menu_action(
            act_credits,
            "See your subscription USD credit pool, dollars used so far, remaining balance, "
            "and estimated runs remaining.",
        )
        account_menu.addAction(act_credits)

        act_diag = QAction("Diagnostics…", self)
        act_diag.triggered.connect(self._show_diagnostics_page)
        self._describe_menu_action(
            act_diag,
            "Technical readout for troubleshooting: environment, capabilities, and recent errors — "
            "pair with exporting a diagnostics bundle under File.",
        )
        account_menu.addAction(act_diag)

        account_menu.addSeparator()

        pay_sub = QAction("Paystack: Subscribe to Pro…", self)
        pay_sub.triggered.connect(self._on_paystack_subscribe)
        self._describe_menu_action(
            pay_sub,
            "Opens Paystack checkout in your browser to subscribe or upgrade to Pro (hosted models and "
            "monthly credits).",
        )
        account_menu.addAction(pay_sub)

        pay_manage = QAction("Paystack: Manage subscription…", self)
        pay_manage.triggered.connect(self._on_paystack_manage_subscription)
        self._describe_menu_action(
            pay_manage,
            "Hosted Paystack portal: update card, view invoices, cancel or change renewal — requires an "
            "active subscription on file.",
        )
        account_menu.addAction(pay_manage)

        pay_refresh = QAction("Refresh cloud account", self)
        pay_refresh.triggered.connect(self._on_refresh_cloud_license)
        self._describe_menu_action(
            pay_refresh,
            "Re-download plan status and credit counters from the server — use after payment or if "
            "numbers look stale.",
        )
        account_menu.addAction(pay_refresh)

        act_manage_pcs = QAction("Manage PCs…", self)
        act_manage_pcs.triggered.connect(self._on_manage_pcs)
        self._describe_menu_action(
            act_manage_pcs,
            "View or remove computers registered for hosted Pro keys — your plan allows a limited "
            "number of devices.",
        )
        account_menu.addAction(act_manage_pcs)

        onboarding = QAction("Run onboarding", self)
        onboarding.triggered.connect(self._run_onboarding)
        self._describe_menu_action(
            onboarding,
            "Launch the first-run wizard again: workspace, validation, and short tour — helpful for "
            "new team members on this PC.",
        )
        account_menu.addAction(onboarding)

        help_menu = menubar.addMenu("&Help")
        help_menu.setToolTipsVisible(True)
        help_menu.menuAction().setToolTip(
            "Documentation, guided tutorial, and product version — hover each link for details."
        )
        help_menu.menuAction().setStatusTip("README, tutorial wizard, and About SurvyAI.")

        readme = QAction("Documentation (README)", self)
        readme.triggered.connect(self._open_readme_docs)
        self._describe_menu_action(
            readme,
            "Opens the project README / user guide so you can read how features, tools, and workflows "
            "are meant to be used.",
        )
        help_menu.addAction(readme)

        tutorial = QAction("First-run tutorial", self)
        tutorial.triggered.connect(self._run_onboarding)
        self._describe_menu_action(
            tutorial,
            "Same guided flow as first launch — workspace checks and orientation if you skipped or "
            "want a refresher.",
        )
        help_menu.addAction(tutorial)

        about = QAction("&About SurvyAI", self)
        about.triggered.connect(self._show_about)
        self._describe_menu_action(
            about,
            "App version, short credits, and legal notices — confirm you are on the build you expect.",
        )
        help_menu.addAction(about)

    # ------------------------------------------------------------------
    # State / configuration helpers
    # ------------------------------------------------------------------

    def _wrap_layout(self, layout) -> QWidget:
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _pair_widget(self, left: QWidget, right: QWidget) -> QWidget:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(left, 1)
        row.addWidget(right)
        return self._wrap_layout(row)

    @Slot()
    def _show_settings_page(self) -> None:
        self._central_stack.setCurrentIndex(_PAGE_SETTINGS)
        self._back_workspace_btn.setVisible(True)

    @Slot()
    def _show_diagnostics_page(self) -> None:
        self._refresh_diagnostics()
        self._central_stack.setCurrentIndex(_PAGE_DIAGNOSTICS)
        self._back_workspace_btn.setVisible(True)

    @Slot()
    def _show_credits_page(self) -> None:
        self._refresh_credits_page()
        self._central_stack.setCurrentIndex(_PAGE_CREDITS)
        self._back_workspace_btn.setVisible(True)

    @Slot()
    def _back_to_workspace(self) -> None:
        self._central_stack.setCurrentIndex(_PAGE_MAIN)
        self._back_workspace_btn.setVisible(False)

    def _user_is_identified(self) -> bool:
        return bool(self._state.cloud_api_base_url.strip() and self._state.cloud_refresh_token.strip())

    def _header_display_name(self) -> str:
        """Greeting suffix after 'Hi, ': local preference or email local-part."""
        me = self._state.cloud_me if isinstance(self._state.cloud_me, dict) else {}
        if self._state.profile.display_name.strip():
            return self._state.profile.display_name.strip()
        dn = str(me.get("display_name") or "").strip()
        if dn:
            return dn
        em = str(me.get("email") or "").strip() or self._state.profile.email.strip()
        local = _email_local_part(em)
        return local or "there"

    def _refresh_user_menu(self) -> None:
        self._user_menu.clear()
        one_chevron = " \u25be"
        if self._user_is_identified():
            self._user_menu_btn.setObjectName("userMenuButton")
            suffix = self._header_display_name()
            label = f"Hi, {suffix}" if suffix else "Hi"
            self._user_menu_btn.setText(label + one_chevron)
            self._user_menu_btn.setToolTip(
                "Quick profile menu — settings, credits and usage, local Ollama models, diagnostics, "
                "registered PCs, and sign out. Hover each item for details."
            )
            act_settings = QAction("Settings", self)
            act_settings.triggered.connect(self._show_settings_page)
            self._describe_menu_action(
                act_settings,
                "Workspace, LLMs, folders, and cloud-related profile fields — same full-screen Settings "
                "as the menu bar.",
            )
            self._user_menu.addAction(act_settings)

            act_credits = QAction("Credits && Usage", self)
            act_credits.triggered.connect(self._show_credits_page)
            self._describe_menu_action(
                act_credits,
                "Your subscription credit pool in USD, how much has been used, and what remains for hosted models.",
            )
            self._user_menu.addAction(act_credits)

            act_ollama = QAction("Local models (Ollama)…", self)
            act_ollama.triggered.connect(self._open_ollama_setup)
            self._describe_menu_action(
                act_ollama,
                "Install or configure Ollama to run free local models on this computer (optional offline use).",
            )
            self._user_menu.addAction(act_ollama)

            act_diag = QAction("Diagnostics", self)
            act_diag.triggered.connect(self._show_diagnostics_page)
            self._describe_menu_action(
                act_diag,
                "Live technical snapshot for support — pair with File → Export diagnostics bundle.",
            )
            self._user_menu.addAction(act_diag)

            act_pcs = QAction("Manage PCs…", self)
            act_pcs.triggered.connect(self._on_manage_pcs)
            self._describe_menu_action(
                act_pcs,
                "Devices allowed on your Pro plan for hosted API keys — remove old PCs to free a slot.",
            )
            self._user_menu.addAction(act_pcs)

            self._user_menu.addSeparator()

            act_out = QAction("Log out", self)
            act_out.triggered.connect(self._sign_out_account)
            self._describe_menu_action(
                act_out,
                "Clear cloud tokens and profile from this app — local workspaces stay on disk.",
            )
            self._user_menu.addAction(act_out)
        else:
            self._user_menu_btn.setObjectName("userMenuButtonGuest")
            self._user_menu_btn.setText("Login / Create account" + one_chevron)
            self._user_menu_btn.setToolTip(
                "Sign in or register for SurvyAI Cloud — then hosted models, credits, and billing apply. "
                "You can still open Settings and Diagnostics without signing in."
            )

            act_in = QAction("Sign in or create account…", self)
            act_in.triggered.connect(self._cloud_sign_in)
            self._describe_menu_action(
                act_in,
                "Create a cloud login or sign in — needed for Pro subscription, hosted LLM keys, and "
                "credit balances.",
            )
            self._user_menu.addAction(act_in)

            act_settings = QAction("Settings", self)
            act_settings.triggered.connect(self._show_settings_page)
            self._describe_menu_action(
                act_settings,
                "Configure workspace, LLMs, and folders — available before cloud sign-in.",
            )
            self._user_menu.addAction(act_settings)

            act_credits = QAction("Credits && Usage", self)
            act_credits.triggered.connect(self._show_credits_page)
            self._describe_menu_action(
                act_credits,
                "After sign-in, shows your subscription pool and usage — meaningful once connected to cloud.",
            )
            self._user_menu.addAction(act_credits)

            act_ollama = QAction("Local models (Ollama)…", self)
            act_ollama.triggered.connect(self._open_ollama_setup)
            self._describe_menu_action(
                act_ollama,
                "Free local AI — no cloud account required for basic offline models.",
            )
            self._user_menu.addAction(act_ollama)

            act_diag = QAction("Diagnostics", self)
            act_diag.triggered.connect(self._show_diagnostics_page)
            self._describe_menu_action(
                act_diag,
                "Environment and capability snapshot — helpful before contacting support.",
            )
            self._user_menu.addAction(act_diag)

    def _finish_startup(self) -> None:
        if not self._state.onboarding_complete:
            self._run_onboarding()
        # Non-blocking post-start prompts (e.g. local models setup).
        QTimer.singleShot(650, self._maybe_prompt_ollama_install)
        if self._startup_initial_query:
            self._input.setPlainText(self._startup_initial_query)
            if self._startup_auto_run:
                QTimer.singleShot(150, self._on_send_clicked)
            else:
                self.statusBar().showMessage(
                    "Prompt loaded from command line — press Send or Enter to run."
                )
        else:
            self.statusBar().showMessage("Ready — choose a workspace and start a task.")

    @Slot()
    def _open_ollama_setup(self) -> None:
        """
        Guided local-models setup:
        - If Ollama isn't installed, offer winget install or open the download page.
        - Always provides a path via the profile dropdown entry.
        """
        dlg = _OllamaSetupDialog(
            self,
            initial_base_url=self._state.ollama_base_url.strip() or getattr(self._settings, "ollama_base_url", ""),
            initial_model=self._state.ollama_model.strip() or getattr(self._settings, "ollama_model", ""),
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        picked_model = dlg.model()
        if not picked_model or picked_model.startswith("("):
            QMessageBox.warning(
                self,
                "Model required",
                "Pull a model first (e.g. llama3.2:3b), then select it from the list.",
            )
            return
        self._state.ollama_base_url = dlg.base_url().strip()
        self._state.ollama_model = picked_model.strip()
        self._state_store.save(self._state)
        self._rebuild_service()
        self._refresh_active_llm_status()
        self.statusBar().showMessage("Ollama settings saved.", 3500)

    def _maybe_prompt_ollama_install(self) -> None:
        """
        Prompt (at most every 3 days) to install Ollama when missing.
        User can permanently dismiss; profile menu keeps a manual entry.
        """
        # Don't prompt if already installed or user dismissed.
        if is_ollama_installed().installed:
            return
        if bool(getattr(self._state, "ollama_prompt_dismissed", False)):
            return

        now = datetime.now(timezone.utc)
        last_raw = str(getattr(self._state, "ollama_last_prompted_at", "") or "").strip()
        if last_raw:
            try:
                last = datetime.fromisoformat(last_raw.replace("Z", "+00:00"))
                if now < last + timedelta(days=3):
                    return
            except Exception:
                # If state is corrupt, fall through and prompt.
                pass

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("Enable free local models")
        box.setText(
            "Want SurvyAI to work offline and without paid API keys?\n\n"
            "Install Ollama to run local models on this PC. You can always do this later "
            "from the profile menu: Local models (Ollama)…"
        )
        cb = QCheckBox("Don't show this again")
        box.setCheckBox(cb)
        install_btn = box.addButton("Install Ollama…", QMessageBox.AcceptRole)
        box.addButton("Not now", QMessageBox.RejectRole)
        box.exec()

        # Persist reminder metadata.
        self._state.ollama_last_prompted_at = now.isoformat()
        if cb.isChecked():
            self._state.ollama_prompt_dismissed = True
        self._state_store.save(self._state)

        if box.clickedButton() == install_btn:
            self._open_ollama_setup()

    def _effective_feature_flags(self) -> FeatureFlags:
        base = self._display_feature_flags
        if not self._state.safe_mode:
            return base
        # Builder mode normally force-enables integrations for developer convenience.
        # For desktop safe mode we intentionally bypass that by using a restrictive
        # runtime flag-set while still keeping the displayed plan from `base`.
        return FeatureFlags(
            license_mode="pro",
            allow_autocad=False,
            allow_arcgis=False,
            allow_blue_marble=False,
            allow_internet_tools=False,
            allow_vector_store=False,
        )

    def _effective_settings(self):
        overrides = {}
        if self._state.preferred_primary_llm:
            overrides["primary_llm"] = self._state.preferred_primary_llm
        if self._state.preferred_fallback_llm:
            overrides["fallback_llm"] = self._state.preferred_fallback_llm
        data_dir = Path(self._state.data_folder or self._state_store.default_data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        overrides["vector_store_path"] = str(data_dir / "vectordb")
        overrides["log_file"] = str(data_dir / "survyai-desktop.log")
        # Cloud overrides: token + base URL + injected platform keys/models from /v1/bootstrap.
        if self._state.cloud_api_base_url.strip() and self._state.cloud_access_token.strip():
            overrides["survyai_api_base_url"] = self._state.cloud_api_base_url.strip()
            overrides["survyai_access_token"] = self._state.cloud_access_token.strip()
            bs = self._state.cloud_bootstrap or {}
            if isinstance(bs, dict):
                # Provider keys
                if str(bs.get("openai_api_key") or "").strip():
                    overrides["openai_api_key"] = str(bs.get("openai_api_key")).strip()
                if str(bs.get("google_api_key") or "").strip():
                    overrides["google_api_key"] = str(bs.get("google_api_key")).strip()
                if str(bs.get("anthropic_api_key") or "").strip():
                    overrides["anthropic_api_key"] = str(bs.get("anthropic_api_key")).strip()
                if str(bs.get("deepseek_api_key") or "").strip():
                    overrides["deepseek_api_key"] = str(bs.get("deepseek_api_key")).strip()
                # Models (single + tiered)
                for k_src, k_dst in [
                    ("openai_model", "openai_model"),
                    ("openai_model_nano", "openai_model_nano"),
                    ("openai_model_mini", "openai_model_mini"),
                    ("openai_model_complex", "openai_model_complex"),
                    ("enable_tiered_models", "enable_tiered_models"),
                    ("gemini_model", "gemini_model"),
                    ("claude_model", "claude_model"),
                    ("deepseek_base_url", "deepseek_base_url"),
                    ("primary_llm", "primary_llm"),
                ]:
                    if k_src in bs and bs.get(k_src) is not None:
                        overrides[k_dst] = bs.get(k_src)
        if self._state.safe_mode:
            overrides["vector_store_enabled"] = False
            overrides["auto_context_retrieval"] = False
            overrides["auto_store_conversations"] = False
        # Desktop-local Ollama settings (do not require editing .env).
        if getattr(self._state, "ollama_base_url", "").strip():
            overrides["ollama_base_url"] = self._state.ollama_base_url.strip()
        if getattr(self._state, "ollama_model", "").strip():
            overrides["ollama_model"] = self._state.ollama_model.strip()
        overrides["fast_mode_non_file_prompts"] = bool(getattr(self._state, "fast_mode_non_file_prompts", False))
        settings = merge_settings(**overrides)

        # Commercial scaffolding (dormant): Free plan forces Ollama-only and disallows live models.
        # This is OFF by default for development; enable via ENFORCE_PLAN_POLICIES=true.
        if bool(getattr(settings, "enforce_plan_policies", False)):
            me = self._state.cloud_me if isinstance(self._state.cloud_me, dict) else {}
            plan_slug = str(me.get("plan_slug") or "free")
            policy = policy_for_plan(plan_slug)
            if policy.slug == "free":
                settings = settings.model_copy(update={"primary_llm": "ollama", "fallback_llm": "ollama"})

        return settings

    def _rebuild_service(self, *, skip_cloud_refresh: bool = False) -> None:
        if not skip_cloud_refresh:
            self._ensure_cloud_token_valid(silent=True)
        self._settings = self._effective_settings()
        self._feature_flags = self._effective_feature_flags()
        self._service = SurvyAIAgentService(
            settings=self._settings,
            feature_flags=self._feature_flags,
            eager_init=False,
        )
        self._refresh_active_llm_status()

    def _ensure_cloud_token_valid(self, *, silent: bool = False) -> bool:
        """
        Refresh the cloud access token when near expiry (or missing expiry metadata).

        Returns True if cloud is not configured, or refresh succeeds / not needed.
        """
        base = self._state.cloud_api_base_url.strip()
        rt = self._state.cloud_refresh_token.strip()
        if not base or not rt:
            return True

        now = datetime.now(timezone.utc)
        exp_raw = self._state.cloud_access_token_expires_at.strip()
        needs_refresh = False
        if not exp_raw:
            needs_refresh = True
        else:
            try:
                exp = datetime.fromisoformat(exp_raw.replace("Z", "+00:00"))
                if now >= exp - timedelta(minutes=2):
                    needs_refresh = True
            except Exception:
                needs_refresh = True

        if not needs_refresh:
            return True

        try:
            tokens = refresh_tokens(base_url=base, refresh_token=rt)
        except CloudApiError as exc:
            if not silent:
                QMessageBox.warning(
                    self,
                    "Session expired",
                    user_facing_cloud_message(exc),
                )
            self._clear_cloud_session()
            return False
        except Exception as exc:
            if not silent:
                QMessageBox.warning(
                    self,
                    "Session expired",
                    user_facing_cloud_message(exc),
                )
            self._clear_cloud_session()
            return False

        self._state.cloud_access_token = tokens.access_token
        self._state.cloud_refresh_token = tokens.refresh_token
        self._state.cloud_access_token_expires_at = access_token_expires_at_iso(
            expires_in_seconds=tokens.expires_in
        )
        self._state_store.save(self._state)
        self._rebuild_service(skip_cloud_refresh=True)
        return True

    def _clear_cloud_session(self) -> None:
        self._state.cloud_api_base_url = ""
        self._state.cloud_access_token = ""
        self._state.cloud_refresh_token = ""
        self._state.cloud_access_token_expires_at = ""
        self._state.cloud_bootstrap = {}
        self._state.cloud_me = {}
        self._state.cloud_device_id = ""
        self._state.cloud_device_fingerprint = ""
        self._state.monthly_credits_usd = 0.0
        self._state.monthly_credits_used_usd = 0.0
        self._state.can_use_platform_llm = False
        self._state.credits_billing_interval = ""
        self._state.credit_banner_anchor_budget_usd = -1.0
        self._state.credit_banner_anchor_used_usd = -1.0
        self._state.credit_banner_dismissed_half = False
        self._state.credit_banner_dismissed_eighty = False
        self._state.credit_banner_dismissed_ninetyfive = False
        self._state_store.save(self._state)
        self._rebuild_service(skip_cloud_refresh=True)
        self._refresh_license_card()
        self._refresh_diagnostics()
        self._update_credit_usage_notice()

    def _register_cloud_device_with_credentials(
        self, *, base_url: str, access_token: str, silent: bool
    ) -> bool:
        """
        POST /v1/devices so Pro bootstrap can enforce max active PCs (default 2).
        Returns False on failure (e.g. device cap reached).
        """
        base = (base_url or "").strip()
        token = (access_token or "").strip()
        if not base or not token:
            return True
        fp = compute_machine_fingerprint()
        if self._state.cloud_device_fingerprint and self._state.cloud_device_fingerprint != fp:
            self._state.cloud_device_id = ""
        self._state.cloud_device_fingerprint = fp
        label = (os.environ.get("COMPUTERNAME") or "").strip() or None
        try:
            dev = register_device(
                base_url=base,
                access_token=token,
                fingerprint=fp,
                label=label,
            )
            did = str(dev.get("id") or "").strip()
            if not did:
                if not silent:
                    QMessageBox.warning(
                        self,
                        "PC registration",
                        "The cloud server did not return a device id. Try Refresh cloud account again.",
                    )
                return False
            self._state.cloud_device_id = did
            self._state_store.save(self._state)
            return True
        except CloudApiError as exc:
            if not silent:
                QMessageBox.warning(self, "PC registration", user_facing_cloud_message(exc))
            return False

    def _register_cloud_device_for_this_pc(self, *, silent: bool) -> bool:
        base, token = self._cloud_base_and_token()
        return self._register_cloud_device_with_credentials(
            base_url=base, access_token=token, silent=silent
        )

    def _set_combo_value(self, combo: QComboBox, value: str) -> None:
        idx = combo.findText(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _apply_state_to_ui(self) -> None:
        self._workspace_edit.setText(self._state.workspace_path)
        self._settings_workspace.setText(self._state.workspace_path)
        self._settings_data_folder.setText(self._state.data_folder)
        self._fallback_cb.setChecked(self._state.use_fallback_llm)
        self._safe_mode_cb.setChecked(self._state.safe_mode)
        self._fast_mode_cb.setChecked(bool(getattr(self._state, "fast_mode_non_file_prompts", False)))
        self._refresh_fast_mode_indicator()
        self._set_combo_value(
            self._primary_llm_combo,
            self._state.preferred_primary_llm or self._settings.primary_llm,
        )
        self._set_combo_value(
            self._fallback_llm_combo,
            self._state.preferred_fallback_llm or self._settings.fallback_llm,
        )

    def _refresh_all_views(self) -> None:
        self._ensure_active_conversation()
        self._refresh_account_views()
        self._refresh_license_card()
        self._refresh_capability_views()
        self._refresh_history_list()
        self._refresh_conversation_list()
        self._render_active_conversation()
        self._refresh_diagnostics()
        self._refresh_credits_page()
        self._session_settings_label.setText(f"{self._session_id}\nStatus: Ready")

    def _refresh_account_views(self) -> None:
        profile = self._state.profile
        me = self._state.cloud_me if isinstance(self._state.cloud_me, dict) else {}
        # Prefer the desktop profile name (set at sign-in, e.g. email local-part) over stale cloud display_name.
        profile_name = (profile.display_name or "").strip()
        cloud_name = str(me.get("display_name") or "").strip()
        email_disp = str(me.get("email") or "").strip() or (profile.email or "").strip()
        local = _email_local_part(email_disp)
        name_disp = profile_name or cloud_name or local or "—"
        email_disp = email_disp or "—"
        company_disp = profile.company or "—"
        self._account_name_value.setText(name_disp)
        self._account_email_value.setText(email_disp)
        self._account_company_value.setText(company_disp)
        self._refresh_user_menu()

    def _refresh_license_card(self) -> None:
        runtime_flags = self._feature_flags
        display_flags = self._display_feature_flags
        plan = "Pro" if display_flags.license_mode == "pro" else "UserMode"
        me = self._state.cloud_me if isinstance(self._state.cloud_me, dict) else {}
        if me.get("plan_slug"):
            plan = str(me.get("plan_slug") or plan)
        source = (
            "Cloud session (API + bootstrap)"
            if getattr(self._settings, "survyai_access_token", "").strip()
            and self._state.cloud_api_base_url.strip()
            else "Local feature flags"
        )
        enabled = []
        if runtime_flags.effective_allow_autocad:
            enabled.append("AutoCAD")
        if runtime_flags.effective_allow_internet_tools:
            enabled.append("Internet")
        if runtime_flags.effective_allow_arcgis:
            enabled.append("ArcGIS")
        if runtime_flags.effective_allow_blue_marble:
            enabled.append("Blue Marble")
        if runtime_flags.effective_allow_vector_store:
            enabled.append("Vector store")
        status = "Troubleshooting (safe mode)" if self._state.safe_mode else "Active"
        sub_status = str(me.get("subscription_status") or "").strip()
        if sub_status:
            status = f"{status} | Cloud subscription: {sub_status}"
        period = me.get("subscription_current_period_end")
        if period:
            status = f"{status} | Renewal / period end: {period}"
        license_text = (
            f"Plan: {plan}\n"
            f"Status: {status}\n"
            f"Source: {source}\n"
            f"Enabled integrations: {', '.join(enabled) if enabled else 'Core assistant only'}"
        )
        cloud_ok = bool(self._state.cloud_api_base_url.strip() and self._state.cloud_access_token.strip())
        pro_like = {"active", "trialing", "non_renewing"}
        st_low = str(me.get("subscription_status") or "").lower()
        plan_low = str(me.get("plan_slug") or "").lower()
        has_pro = plan_low == "pro" and st_low in pro_like
        warn_states = {"past_due", "unpaid", "incomplete"}
        if cloud_ok and st_low in warn_states:
            license_text += (
                "\n\nBilling alert: payment needs attention. "
                "Open Settings → Manage subscription… (opens Paystack in your browser), then Refresh cloud account."
            )
        elif cloud_ok and st_low == "non_renewing":
            license_text += (
                "\n\nBilling notice: subscription is set to cancel at the end of the current billing period."
            )
        self._license_settings_label.setText(license_text)

        self._paystack_subscribe_btn.setEnabled(cloud_ok)
        self._paystack_subscribe_btn.setVisible(not has_pro)
        self._paystack_manage_btn.setEnabled(cloud_ok and bool(me.get("can_manage_paystack_subscription")))
        self._paystack_verify_btn.setEnabled(cloud_ok)
        self._cloud_refresh_license_btn.setEnabled(cloud_ok)
        self._manage_pcs_btn.setEnabled(cloud_ok)

        self._billing_banner.clear()
        self._billing_banner.setVisible(False)

        # Commercial scaffolding (dormant): when plan enforcement is enabled, Free is Ollama-only.
        # Default stays unchanged because `enforce_plan_policies` defaults to False.
        try:
            enforce = bool(getattr(self._settings, "enforce_plan_policies", False))
        except Exception:
            enforce = False
        if enforce and plan_low == "free":
            self._primary_llm_combo.setEnabled(False)
            self._fallback_llm_combo.setEnabled(False)
            self._billing_banner.setText(
                "Free plan: live/cloud models are disabled. SurvyAI runs on local Ollama only.\n\n"
                "Upgrade to Pro to unlock model switching and higher CAD limits."
            )
            self._billing_banner.setStyleSheet(
                "QLabel { color: #1f2937; background: #f3f4f6; border: 1px solid #e5e7eb; "
                "border-radius: 10px; padding: 10px 12px; font-weight: 600; }"
            )
            self._billing_banner.setVisible(True)
        else:
            self._primary_llm_combo.setEnabled(True)
            self._fallback_llm_combo.setEnabled(True)
        if cloud_ok and st_low in warn_states:
            self._billing_banner.setText(
                "Payment problem: your cloud subscription needs a successful payment. "
                "Click “Manage subscription…” to open Paystack in your browser and update your card, "
                "then “Refresh cloud account”."
            )
            self._billing_banner.setStyleSheet(
                "QLabel { color: #92400e; background: #fffbeb; border: 1px solid #fcd34d; "
                "border-radius: 10px; padding: 10px 12px; font-weight: 600; }"
            )
            self._billing_banner.setVisible(True)
        elif cloud_ok and st_low == "non_renewing":
            self._billing_banner.setText(
                "Your Pro subscription will not renew after the current period. "
                "You can change this in “Manage subscription…” (opens in your browser)."
            )
            self._billing_banner.setStyleSheet(
                "QLabel { color: #1e3a8a; background: #eff6ff; border: 1px solid #bfdbfe; "
                "border-radius: 10px; padding: 10px 12px; font-weight: 600; }"
            )
            self._billing_banner.setVisible(True)

    def _refresh_capability_views(self) -> None:
        caps_text = format_capabilities_summary(self._caps)
        self._machine_settings_label.setText(caps_text)

    def _refresh_history_list(self) -> None:
        self._history_list.clear()
        for entry in self._state.output_history:
            title = f"{entry.created_at[:19]} | {'OK' if entry.success else 'Error'}"
            item = QListWidgetItem(title)
            item.setData(Qt.ItemDataRole.UserRole, entry.run_id)
            item.setToolTip(entry.query[:300])
            self._history_list.addItem(item)
        if self._history_list.count() == 0:
            self._history_detail.setPlainText("No output history yet.")

    def _ensure_active_conversation(self) -> Conversation:
        conv = self._state_store.ensure_conversations(self._state)
        active = self._state_store.get_active_conversation(self._state) or conv
        self._active_conversation_id = active.conversation_id
        self._session_id = active.session_id
        return active

    def _active_conversation(self) -> Conversation:
        return self._ensure_active_conversation()

    def _refresh_conversation_list(self) -> None:
        active = self._ensure_active_conversation()
        self._conversation_list_sync = True
        self._conversation_list.clear()
        for conv in self._state.conversations:
            title = conv.title or "New conversation"
            preview = ""
            if conv.messages:
                last = conv.messages[-1].content.strip().replace("\n", " ")
                preview = f"\n{last[:60]}" if last else ""
            item = QListWidgetItem(f"• {title}{preview}")
            item.setData(Qt.ItemDataRole.UserRole, conv.conversation_id)
            item.setToolTip(f"{title}\nSession: {conv.session_id}")
            self._conversation_list.addItem(item)
            if conv.conversation_id == active.conversation_id:
                self._conversation_list.setCurrentItem(item)
        self._conversation_list_sync = False

    def _message_html(self, role: str, text: str, *, error: bool = False) -> str:
        if role == "user":
            safe = html.escape(text)
            return (
                f'<p style="margin:8px 0;"><b style="color:#1e40af;">You</b><br/>'
                f'<span style="color:#0f172a;">{safe.replace(chr(10), "<br/>")}</span></p>'
            )
        if role == "assistant":
            safe = html.escape(text)
            color = "#b91c1c" if error else "#0f172a"
            return (
                f'<p style="margin:8px 0;"><b style="color:#047857;">SurvyAI</b><br/>'
                f'<span style="color:{color};">{safe.replace(chr(10), "<br/>")}</span></p>'
            )
        safe = html.escape(text)
        return f'<p style="margin:4px 0;color:#64748b;font-size:9pt;">{safe}</p>'

    def _render_message(self, role: str, text: str, *, error: bool = False) -> None:
        self._append_html(self._message_html(role, text, error=error))

    def _render_active_conversation(self) -> None:
        conv = self._active_conversation()
        if not conv.messages:
            self._transcript.clear()
            self._transcript.setPlainText("")
            return
        parts = [
            self._message_html(m.role, m.content, error=m.error) for m in conv.messages
        ]
        self._transcript.setHtml("<br/>".join(parts))
        self._transcript.moveCursor(QTextCursor.MoveOperation.End)
        self._transcript.ensureCursorVisible()

    def _schedule_desktop_state_save(self) -> None:
        self._desktop_state_save_timer.start()

    @Slot()
    def _flush_desktop_state_save(self) -> None:
        self._state_store.save(self._state)

    def _store_conversation_message(self, role: str, text: str, *, error: bool = False) -> None:
        conv = self._active_conversation()
        self._state_store.append_conversation_message(
            self._state,
            conversation_id=conv.conversation_id,
            role=role,
            content=text,
            error=error,
        )
        active = self._active_conversation()
        self._session_id = active.session_id
        self._active_conversation_id = active.conversation_id
        self._refresh_conversation_list()

    def _selected_history_entry(self) -> Optional[TaskHistoryEntry]:
        item = self._history_list.currentItem()
        if item is None:
            return None
        run_id = item.data(Qt.ItemDataRole.UserRole)
        for entry in self._state.output_history:
            if entry.run_id == run_id:
                return entry
        return None

    def _refresh_diagnostics(self) -> None:
        env_report = environment_validation_report(self._settings)
        caps_report = format_capabilities_summary(self._caps)
        state_report = json.dumps(
            self._state_store.diagnostics_snapshot(self._state),
            indent=2,
            ensure_ascii=True,
        )
        self._diagnostics_text.setPlainText(
            "Environment validation\n"
            "----------------------\n"
            f"{env_report}\n\n"
            "Machine capabilities\n"
            "--------------------\n"
            f"{caps_report}\n\n"
            "Desktop state snapshot\n"
            "----------------------\n"
            f"{state_report}\n"
        )

    def _refresh_credits_page(self) -> None:
        budget = self._state.monthly_credits_usd
        used = self._state.monthly_credits_used_usd
        remaining = max(budget - used, 0.0)
        markup = self._state.credit_markup_multiplier
        interval = (self._state.credits_billing_interval or "").strip().lower()
        interval_words = (
            "Annual billing period"
            if interval == "annual"
            else ("Monthly billing period" if interval == "monthly" else "Billing period")
        )
        self._credits_period_note.setText(
            f"{interval_words}. The credit pool is the US dollar value from your subscription charge "
            f"at checkout (Nigerian Naira converted to USD on the server at the then-current rate). "
            '"Used" and "Remaining" are all in US dollars on the same basis.'
        )

        self._credits_total_label.setText(f"${budget:,.2f}")
        self._credits_used_label.setText(f"${used:,.2f}")
        self._credits_remaining_label.setText(f"${remaining:,.2f}")

        if budget > 0:
            pct = min(used / budget * 100, 100)
            bar_color = "#22c55e" if pct < 75 else "#f59e0b" if pct < 95 else "#ef4444"
            self._credits_pct_label.setText(
                f'<span style="font-weight:600;color:{bar_color}">{pct:.1f}%</span> of pool consumed'
            )
            self._credits_pct_label.setTextFormat(Qt.TextFormat.RichText)
        else:
            self._credits_pct_label.setText(
                "No subscription credit pool (Free tier or credits not yet synced)."
            )

        avg_cost = self._average_run_cost()
        if avg_cost > 0 and remaining > 0:
            est_runs = int(remaining / avg_cost)
            self._credits_cad_label.setText(
                f"~{est_runs} CAD plan runs remaining "
                f"(average SurvyAI usage ≈ ${avg_cost:,.2f} per similar run, in USD)."
            )
        elif budget <= 0:
            self._credits_cad_label.setText("Subscribe to Pro to receive a monthly credit budget.")
        else:
            self._credits_cad_label.setText(
                "Not enough data yet. Run a few tasks to establish an average cost."
            )

        lines: list[str] = []
        for entry in self._state.output_history[:30]:
            cost_val = float(getattr(entry, "llm_cost_usd", 0.0) or 0.0)
            if cost_val <= 0:
                continue
            billed_usd = cost_val * markup
            model = entry.model_name or "?"
            ts = entry.created_at or ""
            q_preview = (entry.query or "")[:60].replace("\n", " ")
            lines.append(f"[{ts}]  ${billed_usd:,.2f}  ({model})  {q_preview}")
        self._credits_history_text.setPlainText(
            "\n".join(lines) if lines else "No cost data recorded yet."
        )
        self._update_credit_usage_notice()

    def _credit_banner_reset_dismissals_if_period_changed(self) -> None:
        """Clear dismiss flags when the server budget changes or usage rolls back (new period)."""
        b = float(self._state.monthly_credits_usd or 0.0)
        u = float(self._state.monthly_credits_used_usd or 0.0)
        ab = float(getattr(self._state, "credit_banner_anchor_budget_usd", -1.0))
        au = float(getattr(self._state, "credit_banner_anchor_used_usd", -1.0))
        if ab < 0.0 and au < 0.0:
            self._state.credit_banner_anchor_budget_usd = b
            self._state.credit_banner_anchor_used_usd = u
            self._state_store.save(self._state)
            return
        reset = False
        if abs(b - ab) > 1e-4:
            reset = True
        if au >= 0.0 and u < au - 1e-4:
            reset = True
        if reset:
            self._state.credit_banner_dismissed_half = False
            self._state.credit_banner_dismissed_eighty = False
            self._state.credit_banner_dismissed_ninetyfive = False
        self._state.credit_banner_anchor_budget_usd = b
        self._state.credit_banner_anchor_used_usd = u
        if reset:
            self._state_store.save(self._state)

    def _update_credit_usage_notice(self) -> None:
        """Low-contrast strip directly under the console prompt: 50/80/95% (dismissible) or 100% (persistent)."""
        if not hasattr(self, "_credit_notice_wrap"):
            return
        self._credit_banner_reset_dismissals_if_period_changed()

        budget = float(self._state.monthly_credits_usd or 0.0)
        used = float(self._state.monthly_credits_used_usd or 0.0)
        remaining = budget - used
        eps = max(1e-6, abs(budget) * 1e-9)
        self._credit_notice_current_band = "none"

        pro = bool(getattr(self._state, "can_use_platform_llm", False))
        signed_in = bool(self._state.cloud_access_token.strip())
        me_d = self._state.cloud_me if isinstance(self._state.cloud_me, dict) else {}
        plan_slug = str(me_d.get("plan_slug") or "").strip().lower()
        pro_like = pro or plan_slug == "pro"

        # No dollar pool ($0 budget): Pro may still be syncing; everyone else sees the free-plan hint.
        if budget <= 1e-9:
            if signed_in and pro_like:
                self._credit_notice_current_band = "loading"
                self._credit_notice_label.setText(
                    "Credits not loaded on this PC yet — open Account, then Credits and Usage, then "
                    "Refresh from cloud to show balance and usage reminders."
                )
                self._credit_notice_dismiss_btn.setVisible(False)
                self._credit_notice_wrap.setMinimumHeight(26)
                self._credit_notice_wrap.setVisible(True)
            else:
                self._credit_notice_current_band = "free"
                self._credit_notice_label.setText(
                    "You are on the free plan with no subscription credit pool ($0). "
                    "Use local models (Ollama), or sign in and upgrade for hosted usage."
                )
                self._credit_notice_dismiss_btn.setVisible(False)
                self._credit_notice_wrap.setMinimumHeight(26)
                self._credit_notice_wrap.setVisible(True)
            return

        pct = (used / budget) * 100.0 if budget > 0 else 0.0
        exhausted = remaining <= eps

        m50 = "About half of this period's subscription credits (in US dollars) have been used — plan larger jobs accordingly."
        m80 = "About four fifths of this period's credits are used — consider spacing heavy tasks or switching to a local model if needed."
        m95 = (
            "Nearly all of this period's credits are used — you may run out soon; "
            "open Credits and Usage under Account, or use Ollama locally."
        )
        m100 = (
            "You have used 100% of your SurvyAI Pro usage credits for this billing period. "
            "Kindly purchase more credits or renew your plan (Account → Credits and Usage), "
            "or switch to a free local model (Ollama)."
        )

        band = "none"
        text = ""
        show_dismiss = False

        if exhausted:
            band = "100"
            text = m100
            show_dismiss = False
        elif pct >= 95.0 and not self._state.credit_banner_dismissed_ninetyfive:
            band = "95"
            text = m95
            show_dismiss = True
        elif pct >= 80.0 and not self._state.credit_banner_dismissed_eighty:
            band = "80"
            text = m80
            show_dismiss = True
        elif pct >= 50.0 and not self._state.credit_banner_dismissed_half:
            band = "50"
            text = m50
            show_dismiss = True

        self._credit_notice_current_band = band
        if band == "none":
            self._credit_notice_wrap.setVisible(False)
            self._credit_notice_label.setText("")
            self._credit_notice_dismiss_btn.setVisible(False)
            self._credit_notice_wrap.setMinimumHeight(0)
            return

        self._credit_notice_label.setText(text)
        self._credit_notice_dismiss_btn.setVisible(show_dismiss)
        self._credit_notice_wrap.setMinimumHeight(26)
        self._credit_notice_wrap.setVisible(True)

    @Slot()
    def _on_credit_notice_dismiss_clicked(self) -> None:
        band = getattr(self, "_credit_notice_current_band", "none")
        if band == "50":
            self._state.credit_banner_dismissed_half = True
        elif band == "80":
            self._state.credit_banner_dismissed_eighty = True
        elif band == "95":
            self._state.credit_banner_dismissed_ninetyfive = True
        self._state_store.save(self._state)
        self._update_credit_usage_notice()

    def _average_run_cost(self) -> float:
        """Average billed USD per run (same dollars as the Credits page Used column)."""
        markup = self._state.credit_markup_multiplier
        costs: list[float] = []
        for entry in self._state.output_history[:20]:
            c = float(getattr(entry, "llm_cost_usd", 0.0) or 0.0)
            if c > 0:
                costs.append(c * markup)
        return sum(costs) / len(costs) if costs else 0.0

    def _effective_run_llm_id(self) -> str:
        """Primary or fallback LLM id for the next run (matches console / agent thread)."""
        primary = str(getattr(self._settings, "primary_llm", "") or "").strip().lower()
        fallback = str(getattr(self._settings, "fallback_llm", "") or "").strip().lower()
        use_fb = bool(self._state.use_fallback_llm)
        if primary == "ollama" and use_fb:
            use_fb = False
        pick = (fallback if use_fb else primary) or primary
        return pick or ""

    def _platform_credit_exhausted_message(self) -> str:
        return (
            "You have used the API credit included with your SurvyAI Pro subscription for this "
            "billing period. Paid cloud models are paused until you add more capacity or your plan "
            "renews.\n\n"
            "What you can do next:\n"
            "• Open **Account → Credits & Usage** or your Paystack subscription page to purchase or "
            "upgrade.\n"
            "• Switch to a **free local model**: **Account → Local models (Ollama)…** If Ollama is not "
            f"installed, download it from {OLLAMA_DOWNLOAD_PAGE} and pick a model, then set **Primary LLM** "
            "to Ollama under **Settings**.\n\n"
            "SurvyAI stops hosted LLM requests when your remaining balance reaches zero so usage stays "
            "within what your subscription funds."
        )

    def _platform_credit_wall_should_block(self) -> tuple[bool, str]:
        """
        Block starting a run when Pro platform credits are exhausted and the user would hit paid APIs.

        Ollama-only runs are always allowed. Builder bypass via ``SURVYAI_BYPASS_CREDIT_LIMIT`` or
        ``SURVYAI_CREDIT_BYPASS_FINGERPRINT`` (see Diagnostics for this PC's fingerprint).
        """
        if not credit_limit_enforcement_enabled():
            return False, ""
        if not self._state.can_use_platform_llm or not self._state.cloud_access_token.strip():
            return False, ""
        if self._effective_run_llm_id() == "ollama":
            return False, ""
        budget = float(self._state.monthly_credits_usd or 0.0)
        used = float(self._state.monthly_credits_used_usd or 0.0)
        remaining = budget - used
        if budget <= 0:
            return False, ""
        if remaining > 1e-9:
            return False, ""
        return True, self._platform_credit_exhausted_message()

    def _silent_pull_entitlements_from_cloud(self) -> None:
        """Re-sync credit counters from the server (no dialogs)."""
        base, token = self._cloud_base_and_token()
        if not base or not token:
            return
        if not self._ensure_cloud_token_valid(silent=True):
            return
        base, token = self._cloud_base_and_token()
        try:
            ent = get_entitlements(base_url=base, access_token=token)
            ent_d = ent if isinstance(ent, dict) else {}
            self._sync_credits_from_entitlements(ent_d)
            self._state_store.save(self._state)
        except Exception:
            pass

    @Slot()
    def _on_refresh_credits_from_cloud(self) -> None:
        base, token = self._cloud_base_and_token()
        if not base or not token:
            QMessageBox.information(
                self,
                "Not signed in",
                "Sign in from the account menu to sync credits.",
            )
            return
        if not self._ensure_cloud_token_valid(silent=False):
            return
        base, token = self._cloud_base_and_token()
        try:
            ent = get_entitlements(base_url=base, access_token=token)
            ent_d = ent if isinstance(ent, dict) else {}
            self._state.monthly_credits_usd = float(ent_d.get("monthly_credits_usd") or 0)
            self._state.monthly_credits_used_usd = float(ent_d.get("monthly_credits_used_usd") or 0)
            self._state.credit_markup_multiplier = float(ent_d.get("credit_markup_multiplier") or 1.5)
            self._state_store.save(self._state)
            self._refresh_credits_page()
            self.statusBar().showMessage("Credits refreshed from cloud.", 4000)
        except CloudApiError as exc:
            QMessageBox.warning(self, "Credits sync failed", user_facing_cloud_message(exc))

    # ------------------------------------------------------------------
    # Console / history helpers
    # ------------------------------------------------------------------

    def _append_html(self, html_block: str) -> None:
        self._transcript.moveCursor(QTextCursor.MoveOperation.End)
        self._transcript.insertHtml(html_block)
        self._transcript.insertHtml("<br/>")
        self._transcript.ensureCursorVisible()

    def _append_user_message(self, text: str) -> None:
        self._render_message("user", text)

    def _append_assistant_message(self, text: str, *, error: bool = False) -> None:
        self._render_message("assistant", text, error=error)

    def _append_system_line(self, text: str) -> None:
        self._render_message("system", text)

    def _append_activity(self, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self._activity_log.appendPlainText(f"[{stamp}] {text}")

    def _persist_history_from_result(self, result: AgentRunResult) -> None:
        entry = TaskHistoryEntry(
            run_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            workspace_path=self._state.workspace_path,
            session_id=result.session_id or self._session_id,
            query=self._pending_plain_query or result.query or self._last_query,
            response=(result.response or "")[:40000],
            success=bool(result.success),
            error=str(result.error or ""),
            llm_used=str(result.llm_used or ""),
            model_name=str(result.model_name or ""),
            llm_cost_usd=float(result.llm_cost_usd or 0.0),
            cancelled=False,
        )
        self._state_store.add_history_entry(self._state, entry=entry)
        self._refresh_history_list()

    def _persist_cancelled_history(self, query: str, note: str) -> None:
        entry = TaskHistoryEntry(
            run_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            workspace_path=self._state.workspace_path,
            session_id=self._session_id,
            query=query,
            response=note,
            success=False,
            error="cancelled",
            llm_used="",
            model_name="",
            cancelled=True,
        )
        self._state_store.add_history_entry(self._state, entry=entry)
        self._refresh_history_list()

    def _sync_credits_from_entitlements(self, ent: dict) -> None:
        """Pull credit balance fields from an /entitlements or /me response dict."""
        self._state.monthly_credits_usd = float(ent.get("monthly_credits_usd") or 0)
        self._state.monthly_credits_used_usd = float(ent.get("monthly_credits_used_usd") or 0)
        self._state.credit_markup_multiplier = float(ent.get("credit_markup_multiplier") or 1.5)
        self._state.can_use_platform_llm = bool(ent.get("can_use_platform_llm"))
        self._state.credits_billing_interval = str(ent.get("credits_billing_interval") or "").strip().lower()
        self._update_credit_usage_notice()

    def _account_for_run_cost(self, result: AgentRunResult) -> None:
        """Update local credit tally and (best-effort) report cost to the cloud usage API."""
        raw_cost = result.llm_cost_usd
        if raw_cost <= 0:
            return
        markup = self._state.credit_markup_multiplier
        marked_up = raw_cost * markup
        self._state.monthly_credits_used_usd = round(
            self._state.monthly_credits_used_usd + marked_up, 6
        )
        self._state_store.save(self._state)
        self._report_cost_to_cloud(raw_cost)
        self._update_credit_usage_notice()

    def _report_cost_to_cloud(self, raw_cost_usd: float) -> None:
        """Best-effort POST to /v1/usage/events with the raw LLM cost."""
        base, token = self._cloud_base_and_token()
        if not base or not token:
            return
        try:
            import requests as _requests

            did = (self._state.cloud_device_id or "").strip()
            headers: dict[str, str] = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            if did:
                headers["X-SurvyAI-Device-Id"] = did
            payload = {
                "events": [
                    {
                        "kind": "agent_run",
                        "quantity": 1,
                        "cost_usd": round(raw_cost_usd, 6),
                    }
                ]
            }
            resp = _requests.post(
                f"{base.rstrip('/')}/v1/usage/events",
                json=payload,
                headers=headers,
                timeout=8,
            )
            if resp is not None and resp.status_code == 402:
                # Server rejected usage (credits); resync balances from entitlements.
                self._silent_pull_entitlements_from_cloud()
                self._refresh_credits_page()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Slots / actions
    # ------------------------------------------------------------------

    @Slot()
    def _on_fallback_toggled(self, checked: bool) -> None:
        self._state.use_fallback_llm = bool(checked)
        self._state_store.save(self._state)

    @Slot()
    def _on_fast_mode_toggled(self, checked: bool) -> None:
        self._state.fast_mode_non_file_prompts = bool(checked)
        self._state_store.save(self._state)
        self._refresh_fast_mode_indicator()
        # This affects runtime Settings payload sent to the agent subprocess.
        # Avoid rebuilding mid-run.
        if self._thread is not None and self._thread.isRunning():
            return
        self._rebuild_service()
        self.statusBar().showMessage("Fast mode updated.", 2500)

    @Slot()
    def _on_send_clicked(self) -> None:
        text = self._input.toPlainText().strip()
        if not text:
            return
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "Busy", "Please wait for the current task to finish.")
            return
        if not self._workspace_edit.text().strip():
            QMessageBox.warning(self, "Workspace required", "Choose a workspace before running tasks.")
            return
        if self._state.cloud_api_base_url.strip() and self._state.cloud_refresh_token.strip():
            if not self._ensure_cloud_token_valid(silent=False):
                QMessageBox.warning(
                    self,
                    "Session expired",
                    "Your sign-in has expired. Please sign in again from the account menu (top right).",
                )
                return

        blocked, credit_msg = self._platform_credit_wall_should_block()
        if blocked and credit_msg:
            self._pending_plain_query = text
            self._last_query = text
            self._input.clear()
            self._store_conversation_message("user", text)
            self._append_user_message(text)
            self._store_conversation_message("assistant", credit_msg, error=True)
            self._append_assistant_message(credit_msg, error=True)
            self.statusBar().showMessage("Hosted API credits exhausted for this period.", 8000)
            return

        self._pending_plain_query = text
        self._last_query = text
        self._input.clear()
        self._store_conversation_message("user", text)
        self._append_user_message(text)
        enhanced = self._build_continuation_query(text)
        # Store the history-enriched version so _handle_internet_permission can
        # re-submit it (with full context) instead of the bare plain query.
        self._pending_enhanced_query = enhanced
        self._run_agent_thread(enhanced)

    # ------------------------------------------------------------------
    # Conversation-context injection for subprocess agent runs
    # ------------------------------------------------------------------

    _MAX_HISTORY_TURNS = 12       # user + assistant messages to keep
    _MAX_MSG_CHARS     = 3000     # truncation per message (raised: GIS results are verbose)

    def _build_continuation_query(self, raw_query: str) -> str:
        """Prepend recent conversation history to *raw_query* so the spawned
        agent subprocess has enough context for follow-up / continuation
        replies (e.g. "Yes, i want" after the agent offered a next step).

        If the new message is judged unrelated to the last exchange (new topic),
        the raw message is sent alone so the agent does not merge spurious context.

        If the active conversation has no prior messages the raw query is
        returned unchanged — no overhead on the first turn.
        """
        conv = self._active_conversation()
        # The current user message was just appended, so prior history is
        # everything except the last entry.
        prior = conv.messages[:-1] if conv.messages else []
        if not prior:
            return raw_query

        turns = [m for m in prior if m.role in ("user", "assistant")]
        if not turns:
            return raw_query

        # Use a lightweight heuristic to decide whether to include the *full* recent
        # history vs a minimal last-exchange context candidate. The agent itself
        # will run an LLM-based relevance check and can ignore the context if the
        # user clearly switched topics.
        rel_window = turns[-4:]
        rel_parts: list[str] = []
        for t in rel_window:
            c = (t.content or "").strip()
            if not c:
                continue
            rel_parts.append(("User: " if t.role == "user" else "Assistant: ") + c)
        prior_for_relevance = "\n".join(rel_parts)
        include_full_history = _should_inject_conversation_context(raw_query, prior_for_relevance)

        # If not confident it's a follow-up, still include the last exchange only.
        # This gives the agent enough evidence to decide "related vs new topic"
        # using the LLM, without bloating the prompt.
        if include_full_history:
            turns = turns[-self._MAX_HISTORY_TURNS:]
        else:
            turns = turns[-2:]

        parts: list[str] = [
            "=== CONTINUATION OF PREVIOUS WORK ===",
            "Below is recent conversation context. The user may be continuing the last task OR starting a new one.",
            "",
        ]
        exchange = 0
        for t in turns:
            content = t.content
            if len(content) > self._MAX_MSG_CHARS:
                content = content[: self._MAX_MSG_CHARS] + "…[truncated]"
            if t.role == "user":
                exchange += 1
                parts.append(f"--- Exchange {exchange} ---")
                parts.append(f"User: {content}")
            else:
                parts.append(f"Assistant: {content}")
            parts.append("")

        parts.append("--- End of History ---")
        parts.append("")
        parts.append(
            "NOW, the user wants you to continue with this new request:"
        )
        parts.append(raw_query)
        return "\n".join(parts)

    def _run_agent_thread(self, query: str) -> None:
        self._state.workspace_path = self._workspace_edit.text().strip()
        self._settings_workspace.setText(self._state.workspace_path)
        self._state_store.save(self._state)

        self._send_btn.setEnabled(False)
        self._fallback_cb.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._retry_btn.setEnabled(False)
        self._run_started_at = time.monotonic()
        self._run_stage = -1
        self._progress_timer.start()
        self._run_status_label.setText("Running")
        self._session_settings_label.setText(f"{self._session_id}\nStatus: Task in progress")
        self._append_activity("Task submitted.")

        # "Use fallback LLM" is an explicit override (run fallback even when primary is healthy).
        # For Ollama-primary workflows, force this OFF so "offline/local" actually runs locally.
        use_fallback_override = bool(self._state.use_fallback_llm)
        if str(getattr(self._settings, "primary_llm", "") or "").strip() == "ollama" and use_fallback_override:
            use_fallback_override = False
            self._append_activity("Note: 'Use fallback LLM' is disabled while Ollama is Primary.")

        self._thread = AgentRunThread(
            self._service,
            query,
            use_fallback_llm=use_fallback_override,
            session_id=self._session_id,
            interactive=True,
            working_directory=self._state.workspace_path,
            parent=self,
        )
        self._thread.result_ready.connect(self._on_agent_result)
        self._thread.failed.connect(self._on_agent_failed)
        self._thread.progress_text.connect(self._on_worker_progress)
        self._thread.cancelled.connect(self._on_agent_cancelled)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()
        self.statusBar().showMessage("Working…")

    @Slot(object)
    def _on_agent_result(self, result: object) -> None:
        if not isinstance(result, AgentRunResult):
            return
        if result.error == "internet_permission_required":
            self._handle_internet_permission(result)
            return

        self._append_system_line("Done.")
        self._append_activity("Task completed.")
        if result.session_id:
            self._session_id = result.session_id
            conv = self._active_conversation()
            conv.session_id = result.session_id
            self._state_store.save(self._state)
            self._refresh_conversation_list()
            self._session_settings_label.setText(f"{self._session_id}\nStatus: Ready")

        body = result.response or ""
        if not result.success and result.error:
            self._store_conversation_message("assistant", f"{body}\n\n(Error: {result.error})", error=True)
            self._append_assistant_message(f"{body}\n\n(Error: {result.error})", error=True)
        else:
            self._store_conversation_message("assistant", body, error=not result.success)
            self._append_assistant_message(body, error=not result.success)
        self._persist_history_from_result(result)
        self._account_for_run_cost(result)

    def _handle_internet_permission(self, result: AgentRunResult) -> None:
        # CRITICAL: use the history-enriched query so the agent retains full
        # conversation context when it is re-run after the user grants/denies
        # internet permission.  _pending_plain_query is the bare user text
        # (no conversation history) and must NOT be used here.
        # Priority: result.query (set by process_query to the enhanced query)
        #   > _pending_enhanced_query (set just before _run_agent_thread)
        #   > _pending_plain_query (last resort, loses context)
        q = (
            result.query
            or getattr(self, "_pending_enhanced_query", None)
            or self._pending_plain_query
            or ""
        )
        box = QMessageBox(self)
        box.setWindowTitle("Internet search permission")
        box.setIcon(QMessageBox.Question)
        box.setText("This task may need up-to-date information from the internet.")
        box.setInformativeText(
            "Allow SurvyAI to search the web for this request?\n\n"
            "This is a tool permission dialog and will only apply to the current run."
        )
        allow = box.addButton("Allow", QMessageBox.AcceptRole)
        box.addButton("Don't allow", QMessageBox.RejectRole)
        box.setDefaultButton(allow)
        box.exec()

        tagged = (
            f"[INTERNET_PERMISSION_GRANTED]\n{q}"
            if box.clickedButton() == allow
            else f"[INTERNET_PERMISSION_DENIED]\n{q}"
        )
        self._append_activity("Permission dialog answered.")
        self._append_system_line("Running (with your choice)…")
        self._run_agent_thread(tagged)

    @Slot(str)
    def _on_agent_failed(self, tb: str) -> None:
        self._append_system_line("Done.")
        self._append_activity("Unexpected GUI/worker error.")
        msg = "An unexpected error occurred in the agent.\n\n" + tb
        self._store_conversation_message("assistant", msg, error=True)
        self._append_assistant_message(
            msg,
            error=True,
        )

    @Slot(str)
    def _on_worker_progress(self, text: str) -> None:
        self._append_activity(text)

    @Slot(str)
    def _on_agent_cancelled(self, text: str) -> None:
        self._store_conversation_message("system", text)
        self._persist_cancelled_history(self._pending_plain_query or self._last_query, text)
        self._append_system_line(text)
        self._append_activity(text)

    @Slot()
    def _on_thread_finished(self) -> None:
        finished_thread = self.sender()
        if finished_thread is not self._thread:
            return
        self._progress_timer.stop()
        self._send_btn.setEnabled(True)
        self._fallback_cb.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._retry_btn.setEnabled(bool(self._last_query.strip()))
        self._run_status_label.setText("Ready")
        self._session_settings_label.setText(f"{self._session_id}\nStatus: Ready")
        self.statusBar().showMessage("Ready.")

    @Slot()
    def _request_cancel_current_run(self) -> None:
        if self._thread is None or not self._thread.isRunning():
            return
        self._cancel_btn.setEnabled(False)
        self._thread.request_cancel()

    @Slot()
    def _retry_last_query(self) -> None:
        if not self._last_query.strip():
            QMessageBox.information(self, "Retry", "No previous query available yet.")
            return
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "Busy", "Wait for the current task to finish before retrying.")
            return
        self._input.setPlainText(self._last_query)
        self._on_send_clicked()

    @Slot()
    def _new_session(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.warning(self, "Busy", "Wait for the current task to finish before starting a new conversation.")
            return
        conv = self._state_store.new_conversation(self._state)
        self._active_conversation_id = conv.conversation_id
        self._session_id = conv.session_id
        self._refresh_conversation_list()
        self._render_active_conversation()
        self._session_settings_label.setText(f"{self._session_id}\nStatus: Ready")
        self._append_activity("Started new conversation.")

    @Slot()
    def _delete_selected_conversation(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.warning(self, "Busy", "Wait for the current task to finish before deleting a conversation.")
            return
        item = self._conversation_list.currentItem()
        if item is None:
            return
        conv_id = str(item.data(Qt.ItemDataRole.UserRole) or "")
        conv = next((c for c in self._state.conversations if c.conversation_id == conv_id), None)
        if conv is None:
            return
        answer = QMessageBox.question(
            self,
            "Delete conversation",
            f"Delete '{conv.title}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        new_active = self._state_store.delete_conversation(self._state, conv_id)
        self._active_conversation_id = new_active.conversation_id
        self._session_id = new_active.session_id
        self._refresh_conversation_list()
        self._render_active_conversation()
        self._session_settings_label.setText(f"{self._session_id}\nStatus: Ready")
        self._append_activity(f"Deleted conversation '{conv.title}'.")

    @Slot()
    def _on_conversation_changed(self) -> None:
        if self._conversation_list_sync:
            return
        item = self._conversation_list.currentItem()
        if item is None:
            return
        if self._thread is not None and self._thread.isRunning():
            return
        conv_id = str(item.data(Qt.ItemDataRole.UserRole) or "")
        conv = self._state_store.set_active_conversation(
            self._state, conv_id, persist=False
        )
        if conv is None:
            return
        self._active_conversation_id = conv.conversation_id
        self._session_id = conv.session_id
        self._render_active_conversation()
        self._schedule_desktop_state_save()
        self._session_settings_label.setText(f"{self._session_id}\nStatus: Ready")
        self.statusBar().showMessage(f"Switched to conversation: {conv.title}", 3000)

    @Slot()
    def _refresh_capabilities(self) -> None:
        self._caps = scan_machine_capabilities()
        self._refresh_capability_views()
        self._refresh_diagnostics()
        self.statusBar().showMessage("Capabilities refreshed.", 3000)

    @Slot()
    def _on_history_selection_changed(self) -> None:
        entry = self._selected_history_entry()
        if entry is None:
            self._history_detail.setPlainText("Select a run to inspect its output.")
            return
        self._history_detail.setPlainText(
            f"Time: {entry.created_at}\n"
            f"Workspace: {entry.workspace_path}\n"
            f"Session: {entry.session_id}\n"
            f"Success: {entry.success}\n"
            f"LLM: {entry.llm_used} ({entry.model_name})\n"
            f"Error: {entry.error or '—'}\n\n"
            f"Query\n-----\n{entry.query}\n\n"
            f"Response\n--------\n{entry.response}"
        )

    @Slot()
    def _reuse_selected_history_query(self) -> None:
        entry = self._selected_history_entry()
        if entry is None:
            return
        self._input.setPlainText(entry.query)
        self._tabs.setCurrentIndex(0)

    @Slot()
    def _retry_selected_history_item(self) -> None:
        entry = self._selected_history_entry()
        if entry is None:
            return
        self._input.setPlainText(entry.query)
        self._tabs.setCurrentIndex(0)
        self._on_send_clicked()

    @Slot()
    def _choose_workspace(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose workspace folder",
            self._workspace_edit.text().strip() or self._state.workspace_path,
        )
        if not folder:
            return
        self._workspace_edit.setText(folder)
        self._settings_workspace.setText(folder)
        self._state.workspace_path = folder
        self._state_store.save(self._state)
        self._append_activity(f"Workspace set to {folder}")

    @Slot()
    def _open_workspace_folder(self) -> None:
        folder = self._workspace_edit.text().strip()
        if folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    @Slot()
    def _choose_data_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose data folder",
            self._settings_data_folder.text().strip() or self._state.data_folder,
        )
        if folder:
            self._settings_data_folder.setText(folder)

    @Slot(bool)
    def _on_safe_mode_toggled(self, checked: bool) -> None:
        checked = bool(checked)
        self._state.safe_mode = checked

        if self._safe_mode_cb.isChecked() != checked:
            self._safe_mode_cb.blockSignals(True)
            self._safe_mode_cb.setChecked(checked)
            self._safe_mode_cb.blockSignals(False)

        self._state_store.save(self._state)
        self._rebuild_service()
        self._refresh_license_card()
        self._refresh_diagnostics()
        self.statusBar().showMessage("Safe mode updated.", 2500)

    @Slot()
    def _apply_runtime_settings(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.warning(self, "Busy", "Wait for the current task to finish before changing runtime settings.")
            return
        self._state.preferred_primary_llm = self._primary_llm_combo.currentText().strip()
        self._state.preferred_fallback_llm = self._fallback_llm_combo.currentText().strip()
        self._state.fast_mode_non_file_prompts = bool(self._fast_mode_cb.isChecked())
        # If user switches to Ollama from the dropdown, ensure we have a usable local model selected.
        wants_ollama = self._state.preferred_primary_llm == "ollama" or self._state.preferred_fallback_llm == "ollama"
        if wants_ollama:
            if not self._state.ollama_base_url.strip():
                self._state.ollama_base_url = str(getattr(self._settings, "ollama_base_url", "") or "").strip() or "http://localhost:11434"
            if not self._state.ollama_model.strip():
                models = list_local_models()
                if models:
                    self._state.ollama_model = models[0]
        self._state.workspace_path = self._settings_workspace.text().strip() or self._state.workspace_path
        self._state.data_folder = self._settings_data_folder.text().strip() or self._state.data_folder
        self._workspace_edit.setText(self._state.workspace_path)
        self._state_store.save(self._state)
        self._rebuild_service()
        self._refresh_active_llm_status()
        self._refresh_license_card()
        self._refresh_diagnostics()
        self.statusBar().showMessage("Runtime settings applied.", 4000)

        # UX guardrail: if the user chose Ollama as Primary, disable the explicit fallback override.
        if self._state.preferred_primary_llm == "ollama" and self._state.use_fallback_llm:
            self._state.use_fallback_llm = False
            self._state_store.save(self._state)
            if self._fallback_cb.isChecked():
                self._fallback_cb.blockSignals(True)
                self._fallback_cb.setChecked(False)
                self._fallback_cb.blockSignals(False)
            self.statusBar().showMessage("Ollama selected — 'Use fallback LLM' turned off.", 4500)

    def _refresh_active_llm_status(self) -> None:
        """
        Update the Settings page "Active LLMs" card from the current effective settings.
        Called after Apply settings and after any service rebuilds.
        """
        if not hasattr(self, "_active_primary_llm_label") or not hasattr(self, "_active_fallback_llm_label"):
            return

        def _fmt(which: str) -> str:
            prov = str(getattr(self._settings, which, "") or "").strip()
            if not prov:
                return "—"
            if prov == "ollama":
                model = str(getattr(self._settings, "ollama_model", "") or "").strip()
                base = str(getattr(self._settings, "ollama_base_url", "") or "").strip()
                extra = " / ".join([x for x in [model, base] if x])
                return f"ollama ({extra})" if extra else "ollama"
            if prov == "openai":
                m = str(getattr(self._settings, "openai_model", "") or "").strip()
                return f"openai ({m})" if m else "openai"
            if prov == "gemini":
                m = str(getattr(self._settings, "gemini_model", "") or "").strip()
                return f"gemini ({m})" if m else "gemini"
            if prov == "claude":
                m = str(getattr(self._settings, "claude_model", "") or "").strip()
                return f"claude ({m})" if m else "claude"
            return prov

        self._active_primary_llm_label.setText(_fmt("primary_llm"))
        self._active_fallback_llm_label.setText(_fmt("fallback_llm"))

    def _cloud_base_and_token(self) -> tuple[str, str]:
        return self._state.cloud_api_base_url.strip(), self._state.cloud_access_token.strip()

    @Slot()
    def _on_paystack_subscribe(self) -> None:
        base, token = self._cloud_base_and_token()
        if not base or not token:
            QMessageBox.warning(
                self,
                "Sign in required",
                "Sign in from the account menu (top right) first, then start Paystack checkout.",
            )
            return
        if not self._ensure_cloud_token_valid(silent=False):
            return
        base, token = self._cloud_base_and_token()
        try:
            plans_payload = get_billing_plans(base_url=base, access_token=token)
        except CloudApiError as exc:
            QMessageBox.warning(self, "Can't load billing", user_facing_cloud_message(exc))
            return
        plans = plans_payload.get("plans") if isinstance(plans_payload, dict) else None
        if not isinstance(plans, list) or not plans:
            QMessageBox.warning(
                self,
                "Billing not configured",
                "The cloud API has no Paystack plan codes configured. "
                "Set PAYSTACK_PLAN_CODE_PRO_MONTHLY and PAYSTACK_PLAN_CODE_PRO_ANNUAL in .env.cloud "
                "(plan codes from Paystack Dashboard; plans should be ₦15,000/mo and ₦162,000/yr).",
            )
            return
        codes = [str(p.get("plan_code") or "").strip() for p in plans]
        if not all(codes):
            QMessageBox.critical(self, "Billing plans invalid", "Server returned plans without plan_code.")
            return
        picker = PaystackPlanPickerDialog(self, plans)
        if picker.exec() != QDialog.DialogCode.Accepted:
            return
        plan_code = picker.selected_plan_code().strip()
        if not plan_code:
            QMessageBox.critical(self, "Billing plans invalid", "No plan was selected.")
            return
        try:
            init = paystack_initialize(base_url=base, access_token=token, plan_code=plan_code)
        except CloudApiError as exc:
            QMessageBox.warning(
                self,
                "Checkout unavailable",
                _cloud_error_text_for_user(exc),
            )
            return
        url = str(init.get("authorization_url") or "").strip()
        if not url:
            QMessageBox.critical(self, "Paystack checkout failed", "Missing authorization_url from server.")
            return
        QDesktopServices.openUrl(QUrl(url))
        QMessageBox.information(
            self,
            "Complete payment",
            "Finish payment in your browser. Then open Settings from the account menu and click "
            "“Refresh cloud account” (or use the Account menu). If your server has no webhooks yet, use "
            "“Verify payment reference…” with the Paystack reference.",
        )

    @Slot()
    def _on_paystack_manage_subscription(self) -> None:
        base, token = self._cloud_base_and_token()
        if not base or not token:
            QMessageBox.warning(
                self,
                "Sign in required",
                "Sign in from the account menu (top right) first.",
            )
            return
        if not self._ensure_cloud_token_valid(silent=False):
            return
        base, token = self._cloud_base_and_token()
        plan_rows: list[dict] = []
        try:
            plans_payload = get_billing_plans(base_url=base, access_token=token)
            raw = plans_payload.get("plans") if isinstance(plans_payload, dict) else None
            if isinstance(raw, list):
                plan_rows = [x for x in raw if isinstance(x, dict)]
        except CloudApiError:
            pass
        try:
            out = paystack_subscription_manage_url(base_url=base, access_token=token)
        except CloudApiError as exc:
            msg = str(exc)
            low = msg.lower()
            if "no paystack subscription on file" in low:
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Icon.Information)
                box.setWindowTitle("No active subscription yet")
                box.setText(
                    "There’s no Paystack subscription linked to this account yet.\n\n"
                    "If you haven’t paid, start a new checkout.\n"
                    "If you already paid, verify the Paystack reference (or refresh after webhooks)."
                )
                subscribe_btn = box.addButton("Subscribe to Pro…", QMessageBox.ButtonRole.AcceptRole)
                verify_btn = box.addButton(
                    "Verify payment reference…", QMessageBox.ButtonRole.ActionRole
                )
                box.addButton(QMessageBox.StandardButton.Cancel)
                box.exec()
                clicked = box.clickedButton()
                if clicked == subscribe_btn:
                    self._on_paystack_subscribe()
                elif clicked == verify_btn:
                    self._on_paystack_verify_reference()
                return
            QMessageBox.warning(
                self,
                "Can't open subscription portal",
                _cloud_error_text_for_user(exc),
            )
            return
        url = str(out.get("url") or "").strip()
        if not url:
            QMessageBox.warning(
                self,
                "Can't open subscription portal",
                "The subscription link was missing. Please try again or sign in from the account menu.",
            )
            return
        dlg = PaystackManageSubscriptionDialog(self, plan_rows, url)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        QDesktopServices.openUrl(QUrl(dlg.portal_url()))

    @Slot()
    def _on_paystack_verify_reference(self) -> None:
        base, token = self._cloud_base_and_token()
        if not base or not token:
            QMessageBox.warning(
                self,
                "Sign in required",
                "Sign in from the account menu (top right) first.",
            )
            return
        if not self._ensure_cloud_token_valid(silent=False):
            return
        base, token = self._cloud_base_and_token()
        ref, ok = QInputDialog.getText(self, "Verify Paystack payment", "Transaction reference:")
        if not ok or not ref.strip():
            return
        try:
            result = paystack_verify(base_url=base, access_token=token, reference=ref.strip())
        except CloudApiError as exc:
            QMessageBox.warning(self, "Verification failed", user_facing_cloud_message(exc))
            return
        if not result.get("ok"):
            QMessageBox.warning(
                self,
                "Not verified",
                str(result.get("detail") or "Paystack reported a non-success status for this reference."),
            )
            return
        QMessageBox.information(self, "Verified", "Payment verified. Refreshing your cloud session…")
        self._on_refresh_cloud_license()

    @Slot()
    def _on_manage_pcs(self) -> None:
        base, token = self._cloud_base_and_token()
        if not base or not token:
            QMessageBox.warning(
                self,
                "Sign in required",
                "Sign in from the account menu (top right) first.",
            )
            return
        if not self._ensure_cloud_token_valid(silent=False):
            return
        base, token = self._cloud_base_and_token()
        me = self._state.cloud_me if isinstance(self._state.cloud_me, dict) else {}
        raw_max = me.get("max_devices")
        max_d: Optional[int] = None
        try:
            if raw_max is not None:
                max_d = int(raw_max)
        except (TypeError, ValueError):
            max_d = None
        dlg = ManagePcsDialog(
            self,
            base_url=base,
            access_token=token,
            current_device_id=self._state.cloud_device_id.strip(),
            max_devices=max_d,
        )
        dlg.exec()
        if dlg.removed_any:
            if dlg.removed_current_pc:
                self._state.cloud_device_id = ""
                self._state_store.save(self._state)
            self._on_refresh_cloud_license()

    @Slot()
    def _on_refresh_cloud_license(self) -> None:
        base, token = self._cloud_base_and_token()
        if not base or not token:
            QMessageBox.warning(
                self,
                "Sign in required",
                "Sign in from the account menu (top right) first.",
            )
            return
        if not self._ensure_cloud_token_valid(silent=False):
            return
        base, token = self._cloud_base_and_token()
        try:
            me = get_me(base_url=base, access_token=token)
            ent = get_entitlements(base_url=base, access_token=token)
            me_d = me if isinstance(me, dict) else {}
            ent_d = ent if isinstance(ent, dict) else {}
            registered = self._register_cloud_device_for_this_pc(silent=False)
            if not registered and _cloud_entitlements_allow_platform_llm(me_d, ent_d):
                self._state.cloud_bootstrap = {}
            bs: dict = {}
            pro_keys = _cloud_entitlements_allow_platform_llm(me_d, ent_d)
            did = self._state.cloud_device_id.strip()
            try:
                if pro_keys:
                    if not did:
                        self.statusBar().showMessage(
                            "This PC is not registered for hosted Pro keys (device limit or registration error).",
                            9000,
                        )
                    else:
                        bs = get_bootstrap(
                            base_url=base,
                            access_token=token,
                            device_id=did,
                        )
                else:
                    bs = get_bootstrap(base_url=base, access_token=token)
            except CloudApiError:
                bs = {}
                if pro_keys:
                    self.statusBar().showMessage(
                        "Hosted keys unavailable. Confirm this PC is registered and your Pro subscription is active.",
                        8000,
                    )
                else:
                    self.statusBar().showMessage(
                        "Account refreshed. Hosted keys load when your plan includes them.",
                        6000,
                    )
        except CloudApiError as exc:
            QMessageBox.warning(self, "Couldn't refresh account", user_facing_cloud_message(exc))
            return
        self._state.cloud_me = me if isinstance(me, dict) else {}
        self._state.cloud_bootstrap = bs if isinstance(bs, dict) else {}
        self._sync_credits_from_entitlements(ent_d)
        self._state_store.save(self._state)
        self._rebuild_service()
        self._refresh_license_card()
        self._refresh_diagnostics()
        plan = str(ent.get("plan_slug") or me.get("plan_slug") or "")
        st = str(ent.get("subscription_status") or me.get("subscription_status") or "")
        self.statusBar().showMessage(f"Cloud refreshed. Plan={plan} Status={st}", 6000)

    @Slot()
    def _cloud_sign_in(self) -> None:
        """
        Cloud sign-in: optional register -> login -> store tokens -> /v1/me, entitlements, bootstrap
        -> rebuild agent service with injected platform keys + model tiers.
        """
        base_url, ok = QInputDialog.getText(
            self,
            "Cloud API base URL",
            "Enter cloud API base URL (e.g. http://127.0.0.1:8088)",
            text=self._state.cloud_api_base_url.strip() or "http://127.0.0.1:8088",
        )
        if not ok:
            return
        base_url = base_url.strip()
        display_name_for_profile = ""
        company_for_profile = ""
        choice = QMessageBox(self)
        choice.setWindowTitle("SurvyAI cloud")
        choice.setText("Do you already have a cloud account, or do you want to create one?")
        choice.setInformativeText(
            "Create account registers you on the server (password at least 8 characters). "
            "You can subscribe to Pro afterward."
        )
        choice.setIcon(QMessageBox.Question)
        choice.addButton("Sign in", QMessageBox.AcceptRole)
        btn_create = choice.addButton("Create account", QMessageBox.ActionRole)
        btn_cancel = choice.addButton(QMessageBox.Cancel)
        choice.exec()
        clicked = choice.clickedButton()
        if clicked is None or clicked == btn_cancel:
            return
        is_register = clicked == btn_create

        email, ok = QInputDialog.getText(
            self,
            "Cloud account" if is_register else "Cloud sign-in",
            "Email",
            text=self._state.profile.email.strip(),
        )
        if not ok or not email.strip():
            return
        password, ok = QInputDialog.getText(
            self,
            "Cloud account" if is_register else "Cloud sign-in",
            "Password (min. 8 characters)" if is_register else "Password",
            QLineEdit.EchoMode.Password,
        )
        if not ok or not password:
            return
        if not is_register:
            nm, ok_nm = QInputDialog.getText(
                self,
                "Sign in",
                "Name (optional):",
                text=self._state.profile.display_name.strip(),
            )
            display_name_for_profile = (nm or "").strip() if ok_nm else ""
            co, ok_co = QInputDialog.getText(
                self,
                "Sign in",
                "Company (optional):",
                text=self._state.profile.company.strip(),
            )
            company_for_profile = (co or "").strip() if ok_co else ""
        if is_register:
            if len(password) < 8:
                QMessageBox.warning(
                    self,
                    "Create account",
                    "Password must be at least 8 characters (cloud server requirement).",
                )
                return
            confirm, ok = QInputDialog.getText(
                self,
                "Create account",
                "Confirm password",
                QLineEdit.EchoMode.Password,
            )
            if not ok:
                return
            if confirm != password:
                QMessageBox.warning(self, "Create account", "Passwords do not match.")
                return
            display_name, dok = QInputDialog.getText(
                self,
                "Create account",
                "Display name (optional):",
                text=(self._state.profile.display_name or "").strip(),
            )
            if not dok:
                display_name = ""
            display_name_for_profile = (display_name or "").strip()
            comp_in, ok_comp = QInputDialog.getText(
                self,
                "Create account",
                "Company (optional):",
                text=self._state.profile.company.strip(),
            )
            company_for_profile = (comp_in or "").strip() if ok_comp else ""
            try:
                cloud_register(
                    base_url=base_url,
                    email=email.strip(),
                    password=password,
                    display_name=display_name_for_profile or None,
                )
            except CloudApiError as exc:
                QMessageBox.warning(self, "Create account failed", user_facing_cloud_message(exc))
                return
            except Exception as exc:
                QMessageBox.warning(self, "Create account failed", user_facing_cloud_message(exc))
                return
            QMessageBox.information(
                self,
                "Account created",
                "Your cloud account was created. You will be signed in next.",
            )

        try:
            tokens = login(base_url=base_url, email=email.strip(), password=password)
            me = get_me(base_url=base_url, access_token=tokens.access_token)
            ent = get_entitlements(base_url=base_url, access_token=tokens.access_token)
            me_d = me if isinstance(me, dict) else {}
            ent_d = ent if isinstance(ent, dict) else {}
            registered = self._register_cloud_device_with_credentials(
                base_url=base_url,
                access_token=tokens.access_token,
                silent=False,
            )
            if not registered and _cloud_entitlements_allow_platform_llm(me_d, ent_d):
                self._state.cloud_bootstrap = {}
            bs: dict = {}
            pro_keys = _cloud_entitlements_allow_platform_llm(me_d, ent_d)
            did = self._state.cloud_device_id.strip()
            try:
                if pro_keys:
                    if not did:
                        self.statusBar().showMessage(
                            "Signed in. This PC is not registered for hosted Pro keys (device limit or error).",
                            9000,
                        )
                    else:
                        bs = get_bootstrap(
                            base_url=base_url,
                            access_token=tokens.access_token,
                            device_id=did,
                        )
                else:
                    bs = get_bootstrap(base_url=base_url, access_token=tokens.access_token)
            except CloudApiError:
                bs = {}
                if pro_keys:
                    self.statusBar().showMessage(
                        "Signed in. Hosted keys unavailable — confirm Pro subscription and PC registration.",
                        8000,
                    )
                else:
                    self.statusBar().showMessage(
                        "Signed in. Hosted model keys load when your cloud plan includes them.",
                        8000,
                    )
        except CloudApiError as exc:
            QMessageBox.warning(self, "Couldn't sign in", user_facing_cloud_message(exc))
            return
        except Exception as exc:
            QMessageBox.warning(self, "Couldn't sign in", user_facing_cloud_message(exc))
            return

        # Persist cloud auth + bootstrap, then rebuild service.
        self._state.cloud_api_base_url = base_url.strip()
        self._state.cloud_access_token = tokens.access_token
        self._state.cloud_refresh_token = tokens.refresh_token
        self._state.cloud_access_token_expires_at = access_token_expires_at_iso(
            expires_in_seconds=tokens.expires_in
        )
        self._state.cloud_me = me if isinstance(me, dict) else {}
        self._state.cloud_bootstrap = bs if isinstance(bs, dict) else {}
        ent_d = ent if isinstance(ent, dict) else {}
        self._sync_credits_from_entitlements(ent_d)
        from_me = str(me.get("display_name") or "").strip() if isinstance(me, dict) else ""
        local = _email_local_part(email.strip())
        entered = (display_name_for_profile or "").strip()
        self._state.profile.display_name = entered or local or from_me
        self._state.profile.company = (company_for_profile or "").strip()
        self._state.profile.email = email.strip()
        if not self._state.profile.signed_in_at:
            self._state.profile.signed_in_at = datetime.now(timezone.utc).isoformat()
        self._state_store.save(self._state)
        self._rebuild_service()
        self._refresh_license_card()
        self._refresh_account_views()
        self._refresh_diagnostics()
        plan = str(ent.get("plan_slug") or me.get("plan_slug") or "")
        status = str(ent.get("subscription_status") or me.get("subscription_status") or "")
        self.statusBar().showMessage(f"Cloud connected. Plan={plan} Status={status}", 6000)

    @Slot()
    def _sign_out_account(self) -> None:
        if not self._state.profile.is_signed_in and not self._state.cloud_access_token.strip():
            return
        answer = QMessageBox.question(
            self,
            "Sign out",
            "Clear the local desktop account profile and cloud session for this app?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self._state.profile = AccountProfile()
        self._clear_cloud_session()
        self._refresh_account_views()
        self._refresh_diagnostics()

    @Slot()
    def _run_onboarding(self) -> None:
        wizard = OnboardingWizard(
            settings=self._settings,
            capabilities=self._caps,
            initial_profile=self._state.profile,
            initial_data_folder=self._state.data_folder or str(self._state_store.default_data_dir),
            parent=self,
        )
        if wizard.exec() != QDialog.DialogCode.Accepted:
            return
        profile = wizard.profile()
        profile.signed_in_at = self._state.profile.signed_in_at or datetime.now(timezone.utc).isoformat()
        self._state.profile = profile
        self._state.data_folder = wizard.data_folder() or self._state.data_folder
        self._state.onboarding_complete = True
        self._state_store.save(self._state)
        self._settings_data_folder.setText(self._state.data_folder)
        self._rebuild_service()
        self._refresh_all_views()
        self._append_activity("Onboarding completed.")

    @Slot()
    def _export_transcript(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export transcript",
            "",
            "HTML (*.html);;Plain text (*.txt)",
        )
        if not path:
            return
        content = self._transcript.toHtml() if path.lower().endswith(".html") else self._transcript.toPlainText()
        try:
            Path(path).write_text(content, encoding="utf-8")
            self.statusBar().showMessage(f"Saved: {path}", 5000)
        except OSError as e:
            QMessageBox.critical(self, "Export failed", str(e))

    @Slot()
    def _export_diagnostics_bundle(self) -> None:
        answer = QMessageBox.question(
            self,
            "Export diagnostics",
            "Create a support bundle with desktop state, diagnostics, history, transcript, and logs?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export diagnostics bundle",
            "survyai-diagnostics.zip",
            "ZIP (*.zip)",
        )
        if not path:
            return
        log_path = Path(self._settings.log_file)
        try:
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("diagnostics.txt", self._diagnostics_text.toPlainText())
                zf.writestr("transcript.txt", self._transcript.toPlainText())
                zf.writestr("activity_log.txt", self._activity_log.toPlainText())
                zf.writestr(
                    "history.json",
                    json.dumps([entry.__dict__ for entry in self._state.output_history], indent=2, ensure_ascii=True),
                )
                zf.writestr(
                    "desktop_state_snapshot.json",
                    json.dumps(
                        self._state_store.diagnostics_snapshot(self._state),
                        indent=2,
                        ensure_ascii=True,
                    ),
                )
                if self._state_store.state_path.is_file():
                    zf.write(self._state_store.state_path, arcname="desktop_state.json")
                if log_path.is_file():
                    zf.write(log_path, arcname=log_path.name)
            self.statusBar().showMessage(f"Diagnostics exported: {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Diagnostics export failed", str(e))

    @Slot()
    def _open_log_folder(self) -> None:
        folder = Path(self._settings.log_file).parent
        if folder.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    @Slot()
    def _open_readme_docs(self) -> None:
        readme = Path(__file__).resolve().parents[2] / "README.md"
        if readme.is_file():
            self._show_markdown_dialog(readme, "SurvyAI Documentation")
        else:
            QMessageBox.information(self, "Documentation", "README.md was not found.")

    def _show_markdown_dialog(self, markdown_path: Path, title: str) -> None:
        """Show local markdown help inside the app instead of delegating to the OS."""
        try:
            content = markdown_path.read_text(encoding="utf-8")
        except OSError as e:
            QMessageBox.critical(self, title, f"Could not open documentation:\n\n{e}")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(980, 760)

        layout = QVBoxLayout(dlg)
        intro = QLabel(
            f"Showing `{markdown_path.name}` inside SurvyAI so help works consistently without relying on external apps."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        viewer = QTextBrowser()
        viewer.setOpenExternalLinks(True)
        try:
            viewer.setMarkdown(content)
        except Exception:
            viewer.setPlainText(content)
        layout.addWidget(viewer, 1)

        actions = QHBoxLayout()
        open_folder = QPushButton("Open docs folder")
        open_folder.setObjectName("secondaryButton")
        open_folder.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(markdown_path.parent)))
        )
        actions.addWidget(open_folder)
        actions.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        actions.addWidget(close_btn)
        layout.addLayout(actions)

        dlg.exec()

    @Slot()
    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About SurvyAI",
            f"<h3>SurvyAI Desktop</h3>"
            f"<p>Version {html.escape(__version__)}</p>"
            f"<p>Professional Windows GUI for the SurvyAI agent.</p>"
            f"<p>The GUI is the primary product experience; the CLI remains available for support/testing.</p>"
        )

    @Slot()
    def _on_progress_tick(self) -> None:
        if self._thread is None or not self._thread.isRunning():
            self._elapsed_label.setText("Elapsed: 0s")
            return
        elapsed = max(0, int(time.monotonic() - self._run_started_at))
        self._elapsed_label.setText(f"Elapsed: {elapsed}s")
        stage = 0
        if elapsed >= 45:
            stage = 3
        elif elapsed >= 15:
            stage = 2
        elif elapsed >= 5:
            stage = 1
        if stage == self._run_stage:
            return
        self._run_stage = stage
        stage_text = {
            0: "Initializing agent and tools…",
            1: "Running LLM/tool workflow…",
            2: "Still working on a long-running task…",
            3: "Long run in progress. You may cancel now to terminate the active agent run.",
        }[stage]
        self._run_status_label.setText(stage_text)
        self._append_activity(stage_text)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._thread is not None and self._thread.isRunning():
            answer = QMessageBox.question(
                self,
                "Task running",
                "A task is still running. Close anyway and terminate it now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self._thread.request_cancel()
        if self._desktop_state_save_timer.isActive():
            self._desktop_state_save_timer.stop()
            self._flush_desktop_state_save()
        event.accept()
