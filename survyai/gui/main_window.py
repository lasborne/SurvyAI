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
import sys
import time
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QEvent, Qt, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QDesktopServices, QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
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
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from survyai.agent_service import SurvyAIAgentService
from agent.state import looks_like_file_driven_task
from survyai.app_config import merge_settings
from survyai.capabilities import format_capabilities_summary, scan_machine_capabilities
from survyai.credit_gate import credit_limit_enforcement_enabled
from survyai.feature_flags import FeatureFlags
from survyai.llm_routing import (
    AUTO_PRIMARY_LLM,
    normalize_primary_llm_selection,
    resolve_primary_llm_selection,
)
from survyai.gui.branding import SurvyLogoWidget, make_app_icon
from survyai.gui.cad_prompt_defaults import (
    SYSTEM_DEFAULT_CAD_PROMPT,
    is_system_default_text,
    resolve_active_cad_prompt,
)
from survyai.gui.manage_pcs_dialog import ManagePcsDialog
from survyai.gui.theme_toggle import ThemeToggle
from survyai.gui.styles import THEME_DARK, THEME_LIGHT, get_stylesheet
from survyai.gui.onboarding import OnboardingWizard, environment_validation_report
from survyai.gui.help_dialog import MarkdownHelpDialog
from survyai.gui.cursor_affordance import install_clickable_cursor_affordance
from survyai.cloud_user_message import user_facing_cloud_message
from survyai.cloud_api import (
    CloudApiError,
    access_token_expires_at_iso,
    change_password as cloud_change_password,
    cloud_health,
    forgot_password as cloud_forgot_password,
    get_billing_plans,
    get_bootstrap,
    get_entitlements,
    get_me,
    login,
    logout as cloud_logout,
    password_policy_hint,
    paystack_initialize,
    paystack_subscription_manage_url,
    paystack_verify,
    report_usage_batch,
    refresh_tokens,
    register_device,
    register as cloud_register,
    reset_password as cloud_reset_password,
    validate_password_strength,
)
from survyai.device_identity import compute_machine_fingerprint
from survyai.plan_policy import policy_for_plan
from survyai.gui.state import (
    AccountProfile,
    AppStateStore,
    Conversation,
    DEFAULT_CLOUD_API_BASE_URL,
    DesktopState,
    TaskHistoryEntry,
)
from survyai.gui.worker import AgentRunThread
from survyai.gui.agent_process import (
    prewarm_shared_agent_process,
    shutdown_shared_agent_process,
)
from survyai.gui.cloud_sync import (
    CloudAccountSyncPayload,
    CloudAccountSyncResult,
    CloudCreditsSyncPayload,
    CloudCreditsSyncResult,
)
from survyai.gui.cloud_worker import (
    CloudAccountSyncThread,
    CloudCreditsSyncThread,
    UpdateCheckThread,
)
from survyai.ollama_support import (
    OLLAMA_DOWNLOAD_PAGE,
    install_ollama_with_winget,
    is_ollama_installed,
    list_local_models,
    start_pull_model,
    try_connect_ollama,
)
from survyai.types import AgentRunResult
from survyai.updater import (
    UPDATE_CHECK_INTERVAL_HOURS,
    UpdateManager,
    UpdateManifest,
    launch_staged_installer,
    update_check_due,
)
from survyai.version import __version__
from runtime_paths import resource_path
from utils.cost_estimator import summarize_graph_llm_usage

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


def _assistant_asked_internet_permission(assistant_text: str) -> bool:
    body = (assistant_text or "").lower()
    markers = (
        "search the internet",
        "search online",
        "browse the web",
        "may i search",
        "you may search",
        "permission",
        "(yes/no)",
        "latest official",
        "up-to-date",
        "latest confirmed",
    )
    return any(m in body for m in markers)


def _is_permission_affirmation(raw_query: str) -> bool:
    t = (raw_query or "").strip().lower()
    if _fu_short_affirm(raw_query):
        return True
    grants = (
        "you may search",
        "you can search",
        "search the internet",
        "search online",
        "permission granted",
        "go ahead and search",
        "please search",
    )
    return any(g in t for g in grants)


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


def _is_save_session_docx_request(raw_query: str) -> bool:
    """True when the user wants to save a prior answer into a Word document."""
    q = (raw_query or "").lower().strip()
    if not q:
        return False

    # Do not treat GIS/CAD/file automation jobs as essay-save requests.
    if looks_like_file_driven_task(raw_query):
        operational = (
            "arcgis", "arcpy", "cutfill", "cut fill", "tin", "idw", "volume",
            "point feature", "create a copy", "copy each", "calculate", "compute",
            "borrow pit", "exported result", "perform analysis", "geospatial analysis",
        )
        has_docx = ".docx" in q or bool(re.search(r"\bessay[\w\-]*\.docx\b", q, flags=re.I))
        wants_word = has_docx or "word document" in q or "word doc" in q
        explicit_essay = any(
            k in q for k in ("essay", "well-structured", "turn this", "previous topic")
        )
        if any(m in q for m in operational) and not (wants_word and (explicit_essay or has_docx)):
            return False

    has_docx = ".docx" in q or bool(re.search(r"\bessay[\w\-]*\.docx\b", q, flags=re.I))
    wants_word = has_docx or "word document" in q or "word doc" in q
    wants_save = any(k in q for k in ("save", "saved", "write it", "write this"))
    wants_essay = any(
        k in q
        for k in (
            "essay", "well-structured", "turn this", "previous topic", "previous answer",
            "last answer", "above", "report",
        )
    )
    if wants_save and wants_word and (wants_essay or has_docx or "word document" in q):
        return True
    return False


def _is_clearly_new_topic(raw_query: str, last_exchange_messages: list) -> bool:
    """Return True when the new query is unrelated to the most-recent exchange.

    Criteria (ALL must hold):
    - No anaphora (no "it", "this", "the same", etc.)
    - No shared file paths with the last exchange
    - Either a new-topic keyword is detected  OR  the query has ≥12 words and
      shares no significant tokens with the last exchange.
    """
    if _fu_anaphora(raw_query):
        return False
    prev_text = " ".join((m.content or "") for m in last_exchange_messages)
    if _fu_shared_paths(prev_text, raw_query):
        return False
    nt = _fu_topic_score(raw_query, prev_text)
    if nt >= 1:
        return True
    nw = len((raw_query or "").split())
    if nw >= 12:
        sig_new = _fu_sig_tokens(raw_query)
        sig_prev = _fu_sig_tokens(prev_text)
        if not (sig_new & sig_prev):
            return True
    return False


def _is_standalone_knowledge_question(raw_query: str) -> bool:
    """True for self-contained explanatory questions that should not inherit task context."""
    q = (raw_query or "").strip().lower()
    if not q:
        return False
    if re.search(r"[a-z]:\\|/[^ \n]+\.(dwg|dxf|docx|pdf|xlsx?|csv|txt|json)\b", q, flags=re.I):
        return False
    task_markers = (
        "plot", "draw", "modify", "open autocad", "cad", "dwg", "dxf",
        "save", "export", "create a file", "generate a plan", "replot",
        "use this", "template", "summarize this document", "attached",
    )
    if any(marker in q for marker in task_markers):
        return False
    question_markers = (
        "what is", "what are", "explain", "describe", "define", "principle",
        "history of", "brief history", "difference between", "compare",
        "how does", "why does", "as a surveyor", "surveying",
    )
    if any(marker in q for marker in question_markers):
        return True
    return bool(q.endswith("?") and len(q.split()) >= 6)


def _should_inject_conversation_context(raw: str, prior_user_assistant_text: str) -> bool:
    """True when the new message is likely a follow-up to the last exchange, not a brand-new task."""
    raw = (raw or "").strip()
    prev = (prior_user_assistant_text or "").strip()
    if not raw or not prev:
        return False
    nw = len(raw.split())
    nt = _fu_topic_score(raw, prev)
    if _fu_short_affirm(raw):
        return False
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


# Central stack: main tabs (Console + History), then full-page Settings, Diagnostics, Credits, CAD prompt
_PAGE_MAIN = 0
_PAGE_SETTINGS = 1
_PAGE_DIAGNOSTICS = 2
_PAGE_CREDITS = 3
_PAGE_CAD_PROMPT = 4


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
    """Manual access purchase flow: show every plan option at once."""

    def __init__(self, parent: QWidget | None, plans: list[dict]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Buy or extend Pro access")
        self.setMinimumWidth(520)
        self._plan_codes: list[str] = []
        root = QVBoxLayout(self)
        intro = QLabel(
            "Choose daily, weekly, monthly, or annual Pro access. This is a manual one-time "
            "checkout; SurvyAI will not renew or debit your card automatically."
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
    """Legacy recurring subscription management flow."""

    def __init__(self, parent: QWidget | None, plans: list[dict], portal_url: str) -> None:
        super().__init__(parent)
        self._portal_url = portal_url
        self.setWindowTitle("Manage old Paystack subscription")
        self.setMinimumWidth(520)
        root = QVBoxLayout(self)
        intro = QLabel(
            "SurvyAI now uses manual one-time Paystack checkout. Use this portal only for "
            "older recurring subscriptions that need card updates, invoices, or cancellation. "
            "To buy or extend access, use Buy / extend Pro… and complete a new checkout."
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


class _PasswordLineEdit(QWidget):
    """Password field with a compact show/hide eye toggle."""

    textChanged = Signal(str)

    def __init__(self, parent: QWidget | None = None, *, placeholder: str = "") -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self._edit = QLineEdit()
        self._edit.setEchoMode(QLineEdit.EchoMode.Password)
        if placeholder:
            self._edit.setPlaceholderText(placeholder)
        self._toggle = QToolButton()
        self._toggle.setObjectName("secondaryButton")
        # Compact eye-style control (Unicode U+25C9 looks like a pupil/eye on Windows fonts).
        self._toggle.setText("\u25c9")
        self._toggle.setCheckable(True)
        self._toggle.setToolTip("Show password")
        self._toggle.setFixedSize(34, 28)
        self._toggle.setAutoRaise(True)
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle.toggled.connect(self._on_toggled)
        self._edit.textChanged.connect(self.textChanged.emit)
        row.addWidget(self._edit, 1)
        row.addWidget(self._toggle, 0)

    def _on_toggled(self, checked: bool) -> None:
        self._edit.setEchoMode(
            QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        )
        # Filled vs hollow circle reads as visible / hidden without emoji fonts.
        self._toggle.setText("\u25ce" if checked else "\u25c9")
        self._toggle.setToolTip("Hide password" if checked else "Show password")

    def text(self) -> str:
        return self._edit.text()

    def setText(self, value: str) -> None:  # noqa: N802
        self._edit.setText(value)

    def setFocus(self, reason: Qt.FocusReason = Qt.FocusReason.OtherFocusReason) -> None:  # noqa: N802
        self._edit.setFocus(reason)


class _PasswordPromptDialog(QDialog):
    """Single password prompt with eye toggle (sign-in / current password)."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        title: str,
        label: str,
        minimum_width: int = 420,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(minimum_width)
        root = QVBoxLayout(self)
        root.addWidget(QLabel(label))
        self._password = _PasswordLineEdit(self)
        root.addWidget(self._password)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._password.setFocus()

    def password(self) -> str:
        return self._password.text()


class _NewPasswordDialog(QDialog):
    """New password + confirm with live policy hint and disabled OK until valid."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        title: str,
        email: str | None = None,
        minimum_width: int = 460,
    ) -> None:
        super().__init__(parent)
        self._email = (email or "").strip() or None
        self.setWindowTitle(title)
        self.setMinimumWidth(minimum_width)
        root = QVBoxLayout(self)
        hint = QLabel(password_policy_hint())
        hint.setWordWrap(True)
        hint.setObjectName("hintLabel")
        root.addWidget(hint)
        root.addWidget(QLabel("New password"))
        self._password = _PasswordLineEdit(self)
        root.addWidget(self._password)
        root.addWidget(QLabel("Confirm password"))
        self._confirm = _PasswordLineEdit(self)
        root.addWidget(self._confirm)
        self._warn = QLabel("")
        self._warn.setWordWrap(True)
        self._warn.setStyleSheet("color: #b45309; font-size: 12px;")
        root.addWidget(self._warn)
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok.setEnabled(False)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        root.addWidget(self._buttons)
        self._password.textChanged.connect(self._revalidate)
        self._confirm.textChanged.connect(self._revalidate)
        self._password.setFocus()
        self._revalidate()

    def _revalidate(self, *_args: object) -> None:
        password = self._password.text()
        confirm = self._confirm.text()
        if not password:
            self._warn.setText("Enter a password that meets the requirements above.")
            self._ok.setEnabled(False)
            return
        policy_err = validate_password_strength(password, email=self._email)
        if policy_err:
            self._warn.setText(policy_err)
            self._ok.setEnabled(False)
            return
        if confirm != password:
            self._warn.setText("Passwords do not match.")
            self._ok.setEnabled(False)
            return
        self._warn.setText("")
        self._ok.setEnabled(True)

    def password(self) -> str:
        return self._password.text()


class _CloudAuthChoiceDialog(QDialog):
    """Sign-in / create / forgot choice with room for the Forgot password label."""

    CHOICE_SIGN_IN = "sign_in"
    CHOICE_CREATE = "create"
    CHOICE_FORGOT = "forgot"

    def __init__(self, parent: QWidget | None, *, policy_hint: str) -> None:
        super().__init__(parent)
        self._choice = ""
        self.setWindowTitle("SurvyAI cloud")
        self.setMinimumWidth(520)
        root = QVBoxLayout(self)
        title = QLabel("Do you already have a cloud account, or do you want to create one?")
        title.setWordWrap(True)
        root.addWidget(title)
        info = QLabel(
            f"Create account registers you on the server ({policy_hint}). "
            "You can subscribe to Pro afterward. Use Forgot password if you cannot sign in."
        )
        info.setWordWrap(True)
        info.setObjectName("hintLabel")
        root.addWidget(info)
        row = QHBoxLayout()
        row.setSpacing(8)
        btn_sign_in = QPushButton("Sign in")
        btn_create = QPushButton("Create account")
        btn_create.setObjectName("secondaryButton")
        btn_forgot = QPushButton("Forgot password…")
        btn_forgot.setObjectName("secondaryButton")
        btn_forgot.setMinimumWidth(150)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setObjectName("secondaryButton")
        row.addWidget(btn_sign_in)
        row.addWidget(btn_create)
        row.addWidget(btn_forgot)
        row.addStretch(1)
        row.addWidget(btn_cancel)
        root.addLayout(row)

        def _pick(value: str) -> None:
            self._choice = value
            self.accept()

        btn_sign_in.clicked.connect(lambda: _pick(self.CHOICE_SIGN_IN))
        btn_create.clicked.connect(lambda: _pick(self.CHOICE_CREATE))
        btn_forgot.clicked.connect(lambda: _pick(self.CHOICE_FORGOT))
        btn_cancel.clicked.connect(self.reject)

    def choice(self) -> str:
        return self._choice


def _paystack_user_message(server_message: str) -> str:
    """Append setup hint when checkout fails because Paystack is not configured on the server."""
    s = (server_message or "").strip()
    low = s.lower()
    if "not configured" in low and "paystack" in low:
        return (
            f"{s}\n\n"
            "On the machine running the cloud API, set PAYSTACK_SECRET_KEY in backend server environment variables or .env.cloud "
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


class _CadFileConflictDialog(QDialog):
    """Professional confirmation when an existing CAD drawing would be overwritten or modified."""

    def __init__(
        self,
        parent: QWidget | None,
        path: str,
        *,
        mode: str = "overwrite",
        dark: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("cadConflictDialog")
        self.setModal(True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        mode_l = (mode or "overwrite").strip().lower()
        is_modify = mode_l == "modify"
        self.setWindowTitle(
            "Modify existing drawing" if is_modify else "Drawing already exists"
        )
        self.setMinimumWidth(460)
        self.setMaximumWidth(560)

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(14)

        header = QHBoxLayout()
        header.setSpacing(12)
        icon = QLabel()
        icon.setFixedSize(40, 40)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setObjectName("cadConflictIcon")
        # Soft warning glyph (no emoji dependency).
        icon.setText("!")
        header.addWidget(icon, 0, Qt.AlignmentFlag.AlignTop)

        titles = QVBoxLayout()
        titles.setSpacing(4)
        title = QLabel(
            "Modify this drawing?" if is_modify else "This drawing already exists"
        )
        title.setObjectName("cadConflictTitle")
        title.setWordWrap(True)
        subtitle = QLabel(
            "SurvyAI is about to change an existing AutoCAD file at this path. "
            "Confirm to continue, or cancel to leave the file as it is."
            if is_modify
            else "A drawing with this name is already in the folder. "
            "Overwrite it with a new plan from the template, or keep the current file."
        )
        subtitle.setObjectName("cadConflictSubtitle")
        subtitle.setWordWrap(True)
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles, 1)
        root.addLayout(header)

        path_box = QFrame()
        path_box.setObjectName("cadConflictPathBox")
        path_layout = QVBoxLayout(path_box)
        path_layout.setContentsMargins(12, 10, 12, 10)
        path_layout.setSpacing(4)
        path_label = QLabel("File path")
        path_label.setObjectName("cadConflictPathLabel")
        path_value = QLabel(path or "—")
        path_value.setObjectName("cadConflictPathValue")
        path_value.setWordWrap(True)
        path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        path_layout.addWidget(path_label)
        path_layout.addWidget(path_value)
        root.addWidget(path_box)

        hint = QLabel(
            "The survey plan template is never overwritten."
        )
        hint.setObjectName("cadConflictHint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch(1)
        keep_btn = QPushButton("Keep existing" if not is_modify else "Cancel")
        keep_btn.setObjectName("secondaryButton")
        keep_btn.setMinimumWidth(120)
        keep_btn.clicked.connect(self.reject)
        go_btn = QPushButton("Overwrite" if not is_modify else "Apply changes")
        go_btn.setObjectName("sendButton")
        go_btn.setMinimumWidth(130)
        go_btn.setDefault(True)
        go_btn.clicked.connect(self.accept)
        btn_row.addWidget(keep_btn)
        btn_row.addWidget(go_btn)
        root.addLayout(btn_row)

        # Inline theme so the dialog reads clearly even when the parent theme is dense.
        if dark:
            self.setStyleSheet(
                """
                QDialog#cadConflictDialog {
                    background: #18181b;
                    border: 1px solid #3f3f46;
                    border-radius: 14px;
                }
                QLabel#cadConflictIcon {
                    background: #422006;
                    color: #fbbf24;
                    border: 1px solid #78350f;
                    border-radius: 20px;
                    font-size: 18pt;
                    font-weight: 800;
                }
                QLabel#cadConflictTitle {
                    color: #fafafa;
                    font-size: 13.5pt;
                    font-weight: 700;
                }
                QLabel#cadConflictSubtitle, QLabel#cadConflictHint {
                    color: #a1a1aa;
                    font-size: 9.5pt;
                }
                QFrame#cadConflictPathBox {
                    background: #27272a;
                    border: 1px solid #3f3f46;
                    border-radius: 10px;
                }
                QLabel#cadConflictPathLabel {
                    color: #71717a;
                    font-size: 8.5pt;
                    font-weight: 600;
                }
                QLabel#cadConflictPathValue {
                    color: #e4e4e7;
                    font-family: "Cascadia Mono", "Consolas", monospace;
                    font-size: 9pt;
                }
                """
            )
        else:
            self.setStyleSheet(
                """
                QDialog#cadConflictDialog {
                    background: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-radius: 14px;
                }
                QLabel#cadConflictIcon {
                    background: #fff7ed;
                    color: #c2410c;
                    border: 1px solid #fdba74;
                    border-radius: 20px;
                    font-size: 18pt;
                    font-weight: 800;
                }
                QLabel#cadConflictTitle {
                    color: #0f172a;
                    font-size: 13.5pt;
                    font-weight: 700;
                }
                QLabel#cadConflictSubtitle, QLabel#cadConflictHint {
                    color: #64748b;
                    font-size: 9.5pt;
                }
                QFrame#cadConflictPathBox {
                    background: #f8fafc;
                    border: 1px solid #e2e8f0;
                    border-radius: 10px;
                }
                QLabel#cadConflictPathLabel {
                    color: #94a3b8;
                    font-size: 8.5pt;
                    font-weight: 600;
                }
                QLabel#cadConflictPathValue {
                    color: #1e293b;
                    font-family: "Cascadia Mono", "Consolas", monospace;
                    font-size: 9pt;
                }
                """
            )


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        initial_query: Optional[str] = None,
        auto_run_query: bool = False,
    ) -> None:
        super().__init__()
        app = QApplication.instance()
        if app is not None:
            install_clickable_cursor_affordance(app)
        self.setWindowTitle(f"SurvyAI Desktop — {__version__}")
        self.setWindowIcon(make_app_icon())
        self.resize(1420, 900)
        self.setMinimumSize(640, 450)

        self._state_store = AppStateStore()
        self._state: DesktopState = self._state_store.load()
        # Product rule: Primary LLM selection is Auto on every cold start.
        # Users may switch providers during a session; the next restart restores Auto.
        if self._state.preferred_primary_llm != AUTO_PRIMARY_LLM:
            self._state.preferred_primary_llm = AUTO_PRIMARY_LLM
            self._state_store.save(self._state)
        else:
            self._state.preferred_primary_llm = AUTO_PRIMARY_LLM
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
        self._cloud_account_sync_thread: Optional[CloudAccountSyncThread] = None
        self._cloud_credits_sync_thread: Optional[CloudCreditsSyncThread] = None
        self._cloud_busy_depth: int = 0
        self._cloud_busy_saved_texts: dict[int, str] = {}
        self._running_conversation_id: Optional[str] = None
        active_conversation = self._state_store.ensure_conversations(self._state)
        self._session_id = active_conversation.session_id
        self._active_conversation_id = active_conversation.conversation_id
        self._last_query = ""
        self._pending_plain_query: Optional[str] = None
        # CAD prompt page staging: None | "user" | "system"
        self._cad_prompt_pending: Optional[str] = None
        self._run_started_at = 0.0
        self._run_stage = -1
        self._conversation_list_sync = False
        self._startup_initial_query = (initial_query or "").strip()
        self._startup_auto_run = bool(auto_run_query and self._startup_initial_query)

        self._console_content_panel: Optional[QWidget] = None

        self._build_ui()
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

        self._update_check_thread: Optional[UpdateCheckThread] = None
        self._update_check_interactive = False
        self._update_prompt_open = False
        self._update_check_timer = QTimer(self)
        self._update_check_timer.setInterval(int(UPDATE_CHECK_INTERVAL_HOURS * 60 * 60 * 1000))
        self._update_check_timer.timeout.connect(self._maybe_auto_check_updates)

        self._payment_watch_timer = QTimer(self)
        self._payment_watch_timer.setInterval(4000)
        self._payment_watch_timer.timeout.connect(self._on_payment_watch_tick)
        self._payment_watch_baseline: Optional[dict] = None
        self._payment_watch_attempts = 0
        self._payment_watch_max_attempts = 75  # ~5 minutes at 4s
        # Background poll: pick up admin Pro overrides / webhook upgrades without
        # requiring a manual "Refresh cloud account" click.
        self._account_poll_timer = QTimer(self)
        self._account_poll_timer.setInterval(45_000)
        self._account_poll_timer.timeout.connect(self._on_account_status_poll_tick)
        self._account_poll_last_focus_sync_at = 0.0

        QTimer.singleShot(0, self._finish_startup)
        # Credit strip lays out after first frame; refresh once the prompt row has geometry.
        QTimer.singleShot(0, self._update_credit_usage_notice)
        QTimer.singleShot(0, self._apply_prompt_controls_scale)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        QTimer.singleShot(50, self._update_credit_usage_notice)
        QTimer.singleShot(0, self._apply_prompt_controls_scale)
        QTimer.singleShot(0, self._refresh_console_prompt_layout)

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        # Returning to SurvyAI after admin approve / Paystack browser checkout.
        if event is not None and event.type() == QEvent.Type.WindowActivate:
            QTimer.singleShot(250, self._maybe_refresh_account_on_focus)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        QTimer.singleShot(0, self._apply_prompt_controls_scale)
        QTimer.singleShot(0, self._refresh_console_prompt_layout)

    def _refresh_console_prompt_layout(self) -> None:
        """Size the prompt strip to the chat input only; transcript fills the rest of the left column."""
        inp = getattr(self, "_input", None)
        strip = getattr(self, "_input_strip", None)
        if inp is None or strip is None:
            return
        credit_h = (
            self._credit_notice_wrap.sizeHint().height()
            if self._credit_notice_wrap.isVisible()
            else 0
        )
        strip.setFixedHeight(inp.height() + credit_h + 8)

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

        btn_min_h = max(20, int(22 * scale + 4))
        for name in ("_send_btn", "_cancel_btn", "_retry_btn", "_cad_prompt_btn"):
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
        QTimer.singleShot(0, self._refresh_console_prompt_layout)

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
            self._cloud_busy_label = QLabel("")
            self._cloud_busy_label.setObjectName("cloudBusyLabel")
            self._cloud_busy_label.setToolTip("Cloud API activity in progress")
            sb.addPermanentWidget(self._cloud_busy_label)
        except Exception:
            # Cosmetic only; never block app startup.
            pass

    def _refresh_fast_mode_indicator(self) -> None:
        if not hasattr(self, "_fast_mode_indicator"):
            return
        enabled = bool(getattr(self._state, "fast_mode_non_file_prompts", False))
        self._fast_mode_indicator.setText(f"Fast mode: {'ON' if enabled else 'OFF'}")

    def _cloud_network_busy(self) -> bool:
        for attr in ("_cloud_account_sync_thread", "_cloud_credits_sync_thread"):
            thread = getattr(self, attr, None)
            if thread is not None and thread.isRunning():
                return True
        return False

    def _cloud_action_widgets(self) -> list[QPushButton]:
        names = (
            "_cloud_refresh_license_btn",
            "_paystack_subscribe_btn",
            "_paystack_manage_btn",
            "_paystack_verify_btn",
            "_manage_pcs_btn",
        )
        widgets: list[QPushButton] = []
        for name in names:
            w = getattr(self, name, None)
            if isinstance(w, QPushButton):
                widgets.append(w)
        if hasattr(self, "_credits_refresh_btn"):
            w = getattr(self, "_credits_refresh_btn", None)
            if isinstance(w, QPushButton):
                widgets.append(w)
        return widgets

    def _begin_cloud_busy(self, message: str) -> None:
        """Show immediate feedback while cloud HTTP runs on a worker thread."""
        self._cloud_busy_depth += 1
        if self._cloud_busy_depth == 1:
            app = QApplication.instance()
            if app is not None:
                app.setOverrideCursor(Qt.CursorShape.WaitCursor)
            for btn in self._cloud_action_widgets():
                btn_id = id(btn)
                if btn_id not in self._cloud_busy_saved_texts:
                    self._cloud_busy_saved_texts[btn_id] = btn.text()
                btn.setEnabled(False)
            for attr, label in (
                ("_cloud_refresh_license_btn", "Refreshing…"),
                ("_credits_refresh_btn", "Syncing…"),
            ):
                btn = getattr(self, attr, None)
                if isinstance(btn, QPushButton):
                    btn_id = id(btn)
                    if btn_id not in self._cloud_busy_saved_texts:
                        self._cloud_busy_saved_texts[btn_id] = btn.text()
                    btn.setText(label)
            if hasattr(self, "_cloud_busy_label"):
                self._cloud_busy_label.setText("Cloud: working…")
        self.statusBar().showMessage(message)
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def _end_cloud_busy(self) -> None:
        if self._cloud_busy_depth <= 0:
            return
        self._cloud_busy_depth -= 1
        if self._cloud_busy_depth != 0:
            return
        app = QApplication.instance()
        if app is not None:
            app.restoreOverrideCursor()
        for btn in self._cloud_action_widgets():
            saved = self._cloud_busy_saved_texts.pop(id(btn), None)
            if saved is not None:
                btn.setText(saved)
            btn.setEnabled(True)
        self._cloud_busy_saved_texts.clear()
        if hasattr(self, "_cloud_busy_label"):
            self._cloud_busy_label.setText("")

    def _make_cloud_account_sync_payload(self) -> CloudAccountSyncPayload:
        label = (os.environ.get("COMPUTERNAME") or "").strip() or None
        return CloudAccountSyncPayload(
            base_url=self._state.cloud_api_base_url.strip(),
            access_token=self._state.cloud_access_token.strip(),
            refresh_token=self._state.cloud_refresh_token.strip(),
            access_token_expires_at=self._state.cloud_access_token_expires_at.strip(),
            device_id=self._state.cloud_device_id.strip(),
            device_fingerprint=self._state.cloud_device_fingerprint.strip(),
            machine_label=label,
        )

    def _make_cloud_credits_sync_payload(self) -> CloudCreditsSyncPayload:
        return CloudCreditsSyncPayload(
            base_url=self._state.cloud_api_base_url.strip(),
            access_token=self._state.cloud_access_token.strip(),
            refresh_token=self._state.cloud_refresh_token.strip(),
            access_token_expires_at=self._state.cloud_access_token_expires_at.strip(),
        )

    def _apply_cloud_account_sync_result(
        self,
        result: CloudAccountSyncResult,
        *,
        success_status: str,
    ) -> None:
        if result.access_token:
            self._state.cloud_access_token = result.access_token
        if result.refresh_token:
            self._state.cloud_refresh_token = result.refresh_token
        if result.access_token_expires_at:
            self._state.cloud_access_token_expires_at = result.access_token_expires_at
        if result.device_fingerprint:
            self._state.cloud_device_fingerprint = result.device_fingerprint
        if result.device_id:
            self._state.cloud_device_id = result.device_id
        elif not result.registered and result.pro_keys:
            self._state.cloud_bootstrap = {}

        self._state.cloud_me = result.me if isinstance(result.me, dict) else {}
        self._state.cloud_bootstrap = (
            result.bootstrap if isinstance(result.bootstrap, dict) else {}
        )
        ent_d = result.ent if isinstance(result.ent, dict) else {}
        self._sync_credits_from_entitlements(ent_d)
        self._state_store.save(self._state)
        self._rebuild_service(skip_cloud_refresh=True)
        self._refresh_license_card()
        self._refresh_diagnostics()
        self._refresh_account_views()

        if result.bootstrap_status == "skipped_no_device":
            self.statusBar().showMessage(
                "This PC is not registered for hosted Pro keys (device limit or registration error).",
                9000,
            )
        elif result.bootstrap_status == "failed_pro":
            self.statusBar().showMessage(
                "Hosted keys unavailable. Confirm this PC is registered and your Pro subscription is active.",
                8000,
            )
        elif result.bootstrap_status == "failed_free":
            self.statusBar().showMessage(
                "Account refreshed. Hosted keys load when your plan includes them.",
                6000,
            )
        else:
            self.statusBar().showMessage(success_status, 6000)
        self._ensure_account_status_poll()

    def _start_cloud_account_sync(
        self,
        *,
        success_status: str,
        missing_auth_message: str = "Sign in from the account menu (top right) first.",
        silent: bool = False,
        on_success: Optional[object] = None,
    ) -> bool:
        if self._cloud_network_busy():
            if not silent:
                self.statusBar().showMessage("Cloud update already in progress…", 3000)
            return False
        base, token = self._cloud_base_and_token()
        if not base or not token:
            if not silent:
                QMessageBox.warning(self, "Sign in required", missing_auth_message)
            return False

        if not silent:
            self._begin_cloud_busy("Refreshing cloud account…")
        thread = CloudAccountSyncThread(self._make_cloud_account_sync_payload(), parent=self)
        self._cloud_account_sync_thread = thread

        def _done() -> None:
            if not silent:
                self._end_cloud_busy()
            if self._cloud_account_sync_thread is thread:
                self._cloud_account_sync_thread = None

        def _on_ok(result_obj: object) -> None:
            result = result_obj if isinstance(result_obj, CloudAccountSyncResult) else None
            if result is None:
                if not silent:
                    QMessageBox.warning(self, "Couldn't refresh account", "Unexpected sync response.")
                return
            self._apply_cloud_account_sync_result(result, success_status=success_status)
            if callable(on_success):
                try:
                    on_success(result)
                except Exception:
                    pass

        def _on_fail(msg: str) -> None:
            if not silent:
                QMessageBox.warning(self, "Couldn't refresh account", msg)

        thread.succeeded.connect(_on_ok)
        thread.failed.connect(_on_fail)
        thread.finished.connect(_done)
        thread.start()
        return True

    def _payment_watch_snapshot(self) -> dict:
        me = self._state.cloud_me if isinstance(self._state.cloud_me, dict) else {}
        return {
            "plan_slug": str(me.get("plan_slug") or "").strip().lower(),
            "subscription_status": str(me.get("subscription_status") or "").strip().lower(),
            "period_end": str(me.get("subscription_current_period_end") or "").strip(),
            "credits_usd": round(float(self._state.monthly_credits_usd or 0.0), 6),
            "credits_used": round(float(self._state.monthly_credits_used_usd or 0.0), 6),
        }

    def _start_payment_refresh_watch(self) -> None:
        """Poll cloud account until webhook-applied Pro changes appear (or timeout)."""
        self._payment_watch_baseline = self._payment_watch_snapshot()
        self._payment_watch_attempts = 0
        if not self._payment_watch_timer.isActive():
            self._payment_watch_timer.start()
        self.statusBar().showMessage("Waiting for payment confirmation…", 5000)

    def _stop_payment_refresh_watch(self) -> None:
        self._payment_watch_timer.stop()
        self._payment_watch_baseline = None
        self._payment_watch_attempts = 0

    def _payment_watch_detected_upgrade(self, before: dict, after: dict) -> bool:
        if not before or not after:
            return False
        if after.get("plan_slug") == "pro" and before.get("plan_slug") != "pro":
            return True
        # Admin override / webhook: subscription becomes active without plan_slug change yet.
        _active = {"active", "non_renewing", "trialing"}
        before_st = str(before.get("subscription_status") or "").strip().lower()
        after_st = str(after.get("subscription_status") or "").strip().lower()
        if after_st in _active and before_st not in _active:
            return True
        if after.get("period_end") and after.get("period_end") != before.get("period_end"):
            return True
        if float(after.get("credits_usd") or 0) > float(before.get("credits_usd") or 0) + 1e-6:
            return True
        # Exhausted → repurchase: used reset while budget restored.
        if (
            after.get("plan_slug") == "pro"
            and float(after.get("credits_used") or 0) + 1e-6 < float(before.get("credits_used") or 0)
            and float(after.get("credits_usd") or 0) + 1e-6 >= float(before.get("credits_usd") or 0)
        ):
            return True
        return False

    def _ensure_account_status_poll(self) -> None:
        """Start background Pro/status polling while a cloud session is present."""
        base, token = self._cloud_base_and_token()
        if not base or not token:
            self._stop_account_status_poll()
            return
        if not self._account_poll_timer.isActive():
            self._account_poll_timer.start()

    def _stop_account_status_poll(self) -> None:
        if self._account_poll_timer.isActive():
            self._account_poll_timer.stop()

    def _maybe_refresh_account_on_focus(self) -> None:
        """Silent sync when the window regains focus (admin approve / browser payment)."""
        base, token = self._cloud_base_and_token()
        if not base or not token:
            return
        if self._cloud_network_busy():
            return
        # Payment watch already polls aggressively.
        if self._payment_watch_timer.isActive() and self._payment_watch_baseline is not None:
            return
        now = time.monotonic()
        if (now - float(self._account_poll_last_focus_sync_at or 0.0)) < 12.0:
            return
        self._account_poll_last_focus_sync_at = now
        baseline = self._payment_watch_snapshot()

        def _after(_result: object) -> None:
            after = self._payment_watch_snapshot()
            if self._payment_watch_detected_upgrade(baseline, after):
                self.statusBar().showMessage(
                    "Account updated — Pro access and license refreshed.",
                    8000,
                )

        self._start_cloud_account_sync(
            success_status="Account status updated.",
            silent=True,
            on_success=_after,
        )

    @Slot()
    def _on_account_status_poll_tick(self) -> None:
        base, token = self._cloud_base_and_token()
        if not base or not token:
            self._stop_account_status_poll()
            return
        if self._cloud_network_busy():
            return
        if self._payment_watch_timer.isActive() and self._payment_watch_baseline is not None:
            return
        baseline = self._payment_watch_snapshot()

        def _after(_result: object) -> None:
            after = self._payment_watch_snapshot()
            if self._payment_watch_detected_upgrade(baseline, after):
                self.statusBar().showMessage(
                    "Account updated — Pro access and license refreshed.",
                    8000,
                )

        self._start_cloud_account_sync(
            success_status="Account status updated.",
            silent=True,
            on_success=_after,
        )

    @Slot()
    def _on_payment_watch_tick(self) -> None:
        if self._payment_watch_baseline is None:
            self._stop_payment_refresh_watch()
            return
        self._payment_watch_attempts += 1
        if self._payment_watch_attempts > self._payment_watch_max_attempts:
            self._stop_payment_refresh_watch()
            self.statusBar().showMessage(
                "Payment not confirmed yet. Use Refresh cloud account or Verify payment reference…",
                8000,
            )
            return
        if self._cloud_network_busy():
            return
        baseline = dict(self._payment_watch_baseline)

        def _after(result: object) -> None:
            after = self._payment_watch_snapshot()
            if self._payment_watch_detected_upgrade(baseline, after):
                self._stop_payment_refresh_watch()
                self.statusBar().showMessage("Payment confirmed — Pro access updated.", 8000)

        self._start_cloud_account_sync(
            success_status="Checking payment status…",
            silent=True,
            on_success=_after,
        )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _begin_app_scroll_page(self) -> tuple[QWidget, QVBoxLayout]:
        """Stacked full-page shell (Settings, Credits, …) with theme-aware scroll chrome."""
        tab = QWidget()
        tab.setObjectName("appStackPage")
        tab.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("appScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout.addWidget(scroll, 1)

        page = QWidget()
        page.setObjectName("appScrollContent")
        page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        scroll.setWidget(page)
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(14, 14, 14, 14)
        page_layout.setSpacing(12)
        return tab, page_layout

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("centralRoot")
        central.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 6, 12, 8)
        root.setSpacing(6)

        top_bar = QWidget()
        top_bar.setObjectName("topBar")
        top_bar_outer = QVBoxLayout(top_bar)
        top_bar_outer.setContentsMargins(12, 6, 12, 6)
        top_bar_outer.setSpacing(6)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)
        title_wrap = QWidget()
        title_row = QHBoxLayout(title_wrap)
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(6)
        self._logo = SurvyLogoWidget(size=26)
        title_row.addWidget(self._logo)
        title = QLabel('<span class="accent">Survy</span>AI')
        title.setObjectName("wordmarkLabel")
        title.setTextFormat(Qt.TextFormat.RichText)
        title_font = QFont()
        title_font.setPointSize(13)
        title_font.setBold(True)
        title.setFont(title_font)
        sub = QLabel("Desktop")
        sub.setObjectName("wordmarkSub")
        version_badge = QLabel(f"v{__version__}")
        version_badge.setObjectName("versionBadge")
        title_row.addWidget(title)
        title_row.addWidget(sub)
        title_row.addWidget(version_badge)
        title_row.addStretch()
        self._back_workspace_btn = QPushButton("Back to workspace")
        self._back_workspace_btn.setObjectName("secondaryButton")
        self._back_workspace_btn.setVisible(False)
        self._back_workspace_btn.setToolTip("Return to Console and Output History")
        self._back_workspace_btn.clicked.connect(self._back_to_workspace)
        top_row.addWidget(self._back_workspace_btn)
        top_row.addWidget(title_wrap, 1)

        self._theme_toggle = ThemeToggle()
        self._theme_toggle.toggled.connect(self._on_dark_mode_toggled)
        top_row.addWidget(self._theme_toggle, 0, Qt.AlignmentFlag.AlignVCenter)

        self._user_menu_btn = QToolButton()
        self._user_menu_btn.setObjectName("userMenuButtonGuest")
        self._user_menu_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._user_menu_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._user_menu_btn.setAutoRaise(False)
        self._user_menu = QMenu(self._user_menu_btn)
        self._user_menu.setToolTipsVisible(True)
        self._user_menu_btn.setMenu(self._user_menu)
        self._user_menu_btn.setText("Sign in \u25be")
        top_row.addWidget(self._user_menu_btn)
        top_bar_outer.addLayout(top_row)

        divider = QFrame()
        divider.setObjectName("topBarDivider")
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Plain)
        top_bar_outer.addWidget(divider)

        workspace_row = QHBoxLayout()
        workspace_row.setSpacing(8)
        ws_label = QLabel("Workspace")
        ws_label.setMinimumWidth(72)
        ws_label.setObjectName("sectionHeader")
        workspace_row.addWidget(ws_label)
        self._workspace_edit = QLineEdit()
        self._workspace_edit.setPlaceholderText("Project folder for prompts, exports, and CAD context")
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

        root.addWidget(top_bar)

        self._central_stack = QStackedWidget()
        self._central_stack.setObjectName("appStack")
        self._central_stack.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._central_stack.addWidget(self._tabs)
        root.addWidget(self._central_stack, 1)

        self._build_console_tab()
        self._build_history_tab()
        self._settings_page = self._build_settings_page()
        self._diagnostics_page = self._build_diagnostics_page()
        self._credits_page = self._build_credits_page()
        self._cad_prompt_page = self._build_cad_prompt_page()
        self._central_stack.addWidget(self._settings_page)
        self._central_stack.addWidget(self._diagnostics_page)
        self._central_stack.addWidget(self._credits_page)
        self._central_stack.addWidget(self._cad_prompt_page)

    def _build_console_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 4, 8, 6)
        layout.setSpacing(4)

        body_split = QSplitter(Qt.Horizontal)

        conversation_panel = QWidget()
        conversation_panel.setObjectName("sidebarConversations")
        conversation_panel.setMinimumWidth(160)
        conversation_layout = QVBoxLayout(conversation_panel)
        conversation_layout.setContentsMargins(8, 8, 8, 8)
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
        content_layout.setSpacing(0)

        self._transcript = QTextEdit()
        self._transcript.setObjectName("chatTranscript")
        self._transcript.setReadOnly(True)
        self._transcript.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self._transcript.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._transcript.setPlaceholderText("Your conversation with SurvyAI will appear here…")
        self._transcript.setMinimumWidth(200)
        self._transcript.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        activity_panel = QWidget()
        activity_panel.setMinimumWidth(200)
        activity_layout = QVBoxLayout(activity_panel)
        activity_layout.setContentsMargins(0, 0, 0, 0)
        activity_layout.setSpacing(4)
        act_hdr = QLabel("Live activity")
        act_hdr.setObjectName("sectionHeader")
        activity_layout.addWidget(act_hdr)
        self._activity_log = QPlainTextEdit()
        self._activity_log.setObjectName("activityLog")
        self._activity_log.setReadOnly(True)
        self._activity_log.setPlaceholderText("Agent progress, tool calls, and status updates…")
        activity_layout.addWidget(self._activity_log, 1)
        self._run_status_label = QLabel("Ready")
        self._run_status_label.setObjectName("runStatusLabel")
        self._elapsed_label = QLabel("Elapsed: 0s")
        self._elapsed_label.setObjectName("elapsedLabel")
        activity_layout.addWidget(self._run_status_label)
        activity_layout.addWidget(self._elapsed_label)

        self._input = ChatInput()
        self._input.setPlaceholderText("Ask SurvyAI to create CAD drawings, open drawings, run calculations, export reports, perform geospatial analysis, etc.…")
        self._input.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._input.sendRequested.connect(self._on_send_clicked)
        self._input.textChanged.connect(
            lambda: QTimer.singleShot(0, self._refresh_console_prompt_layout)
        )

        controls = QVBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(3)
        controls.addStretch(1)
        self._send_btn = QPushButton("Send")
        self._send_btn.setObjectName("sendButton")
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

        self._cad_prompt_btn = QPushButton("Input CAD plan prompt")
        self._cad_prompt_btn.setObjectName("secondaryButton")
        self._cad_prompt_btn.clicked.connect(self._insert_cad_plan_prompt)
        self._cad_prompt_btn.setToolTip(
            "Insert your current default CAD survey-plan prompt into the input box "
            "(from Account → Edit Default CAD Prompt). Existing text is kept; "
            "the template starts on a new line."
        )
        self._cad_prompt_btn.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed
        )
        controls.addWidget(self._cad_prompt_btn)

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

        self._controls_wrap = QWidget()
        self._controls_wrap.setLayout(controls)
        self._controls_wrap.setMinimumWidth(0)
        self._controls_wrap.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )
        activity_layout.addWidget(self._controls_wrap, 0, Qt.AlignmentFlag.AlignBottom)

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

        self._input_strip = QWidget()
        input_strip_layout = QVBoxLayout(self._input_strip)
        input_strip_layout.setContentsMargins(0, 6, 0, 2)
        input_strip_layout.setSpacing(2)
        input_strip_layout.addWidget(self._input, 0)
        input_strip_layout.addWidget(self._credit_notice_wrap, 0)
        self._input_strip.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        QTimer.singleShot(0, self._refresh_console_prompt_layout)

        # Left: tall transcript + compact input strip. Right: activity + Send/Cancel stack.
        # Transcript extends down to the input row; buttons sit beside it on the right.
        chat_column = QWidget()
        chat_column_layout = QVBoxLayout(chat_column)
        chat_column_layout.setContentsMargins(0, 0, 0, 0)
        chat_column_layout.setSpacing(4)
        chat_column_layout.addWidget(self._transcript, 1)
        chat_column_layout.addWidget(self._input_strip, 0)

        self._console_main_split = QSplitter(Qt.Orientation.Horizontal)
        self._console_main_split.setObjectName("consoleMainSplit")
        self._console_main_split.setChildrenCollapsible(True)
        self._console_main_split.addWidget(chat_column)
        self._console_main_split.addWidget(activity_panel)
        self._console_main_split.setSizes([780, 240])
        self._console_main_split.setStretchFactor(0, 2)
        self._console_main_split.setStretchFactor(1, 1)
        content_layout.addWidget(self._console_main_split, 1)

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
        hist_hint = QLabel(
            "Review past agent runs. Select an entry to inspect the response, or reuse a query in the console."
        )
        hist_hint.setObjectName("consoleHintLabel")
        hist_hint.setWordWrap(True)
        layout.addWidget(hist_hint)
        split = QSplitter(Qt.Horizontal)

        self._history_list = QListWidget()
        self._history_list.setObjectName("historyList")
        self._history_list.setAlternatingRowColors(True)
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
        tab, page_layout = self._begin_app_scroll_page()

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
            "you connect to SurvyAI Cloud. Company may be set during onboarding. "
            f"Passwords: {password_policy_hint()}"
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
        self._change_password_btn = QPushButton("Change password…")
        self._change_password_btn.setObjectName("secondaryButton")
        self._change_password_btn.setToolTip(
            "Update your SurvyAI Cloud password. Requires your current password."
        )
        self._change_password_btn.setMaximumWidth(168)
        self._change_password_btn.clicked.connect(self._on_change_password)
        change_pw_row = QHBoxLayout()
        change_pw_row.addWidget(self._change_password_btn, 0)
        change_pw_row.addStretch(1)
        account_outer.addLayout(change_pw_row)
        page_layout.addWidget(account_group)

        pay_group = QGroupBox("Pro subscription (Paystack)")
        pay_outer = QVBoxLayout(pay_group)
        pay_outer.setSpacing(10)
        pay_hint = QLabel(
            "After you sign in, subscribe or manage billing below. "
            "Successful Paystack payments refresh your plan automatically when webhooks are configured. "
            "Use Refresh cloud account or Verify payment reference… only if the update does not appear. "
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
        self._paystack_subscribe_btn = QPushButton("Buy / extend Pro…")
        self._paystack_subscribe_btn.clicked.connect(self._on_paystack_subscribe)
        self._paystack_subscribe_btn.setToolTip(
            "Choose daily, weekly, monthly, or annual access, then complete a manual checkout in your browser."
        )
        pay_row.addWidget(self._paystack_subscribe_btn)
        self._paystack_manage_btn = QPushButton("Manage old subscription…")
        self._paystack_manage_btn.setObjectName("secondaryButton")
        self._paystack_manage_btn.clicked.connect(self._on_paystack_manage_subscription)
        self._paystack_manage_btn.setToolTip("Open Paystack portal only for older recurring subscriptions.")
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
        # "auto" is the product default: routes to the best paid hosted model for the task.
        self._primary_llm_combo.addItems(
            ["auto", "openai", "gemini", "claude", "deepseek", "ollama"]
        )
        self._primary_llm_combo.setToolTip(
            "Auto selects the best paid hosted model for the task "
            "(currently OpenAI). Choose a specific provider to lock it."
        )
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

        updates_group = QGroupBox("Updates")
        updates_outer = QVBoxLayout(updates_group)
        updates_outer.setSpacing(10)
        updates_hint = QLabel(
            "When enabled, SurvyAI periodically asks the SurvyAI Cloud for a newer Windows "
            "installer (about every 12 hours) and notifies you here. Downloads are verified "
            "before install. You can always use Help → Check for updates…."
        )
        updates_hint.setWordWrap(True)
        updates_hint.setObjectName("hintLabel")
        updates_outer.addWidget(updates_hint)
        self._auto_check_updates_cb = QCheckBox(
            "Automatically check for updates (recommended)"
        )
        self._auto_check_updates_cb.setToolTip(
            "Requires your consent. SurvyAI only contacts the configured cloud API for version metadata."
        )
        self._auto_check_updates_cb.toggled.connect(self._on_auto_check_updates_toggled)
        updates_outer.addWidget(self._auto_check_updates_cb)
        updates_row = QHBoxLayout()
        updates_row.setSpacing(8)
        self._check_updates_now_btn = QPushButton("Check for updates now…")
        self._check_updates_now_btn.setObjectName("secondaryButton")
        self._check_updates_now_btn.clicked.connect(self._check_for_updates)
        self._check_updates_now_btn.setToolTip("Check immediately, even if automatic checks are off.")
        updates_row.addWidget(self._check_updates_now_btn)
        updates_row.addStretch()
        updates_outer.addLayout(updates_row)
        self._updates_status_label = QLabel("")
        self._updates_status_label.setWordWrap(True)
        self._updates_status_label.setObjectName("hintLabel")
        updates_outer.addWidget(self._updates_status_label)
        page_layout.addWidget(updates_group)

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
        tab.setObjectName("appStackPage")
        tab.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("appScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout.addWidget(scroll, 1)

        page = QWidget()
        page.setObjectName("appScrollContent")
        page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
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
        footer.setObjectName("appPageFooter")
        footer.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
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
        tab, page_layout = self._begin_app_scroll_page()

        credits_title = QLabel("Credits & usage")
        credits_title.setObjectName("pageTitle")
        credits_sub = QLabel(
            "Track API spend for runs in this desktop app. Credit pool is your subscription USD "
            "equivalent. Used is the billed cost of hosted model runs inside the active paid window. "
            "Local models (Ollama) are free. Pro hosted plans: sign in and use Refresh from cloud to sync."
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
        summary_form.addRow("Used", self._credits_used_label)
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
        self._credits_refresh_btn = QPushButton("Refresh usage")
        self._credits_refresh_btn.clicked.connect(self._on_refresh_credits_from_cloud)
        btn_row.addWidget(self._credits_refresh_btn)
        btn_row.addStretch()
        page_layout.addLayout(btn_row)

        self._credits_lifetime_label = QLabel("")
        self._credits_lifetime_label.setObjectName("hintLabel")
        self._credits_lifetime_label.setWordWrap(True)
        self._credits_lifetime_label.setAlignment(Qt.AlignRight)
        self._credits_lifetime_label.setStyleSheet("color: #9ca3af; font-size: 11px;")
        page_layout.addWidget(self._credits_lifetime_label)

        page_layout.addStretch()
        return tab

    def _build_cad_prompt_page(self) -> QWidget:
        """Full-page editor for the default CAD survey-plan prompt template."""
        tab, page_layout = self._begin_app_scroll_page()

        title = QLabel("Edit Default CAD Prompt")
        title.setObjectName("pageTitle")
        subtitle = QLabel(
            "Customize the default field layout used when creating new AutoCAD (.dwg) survey "
            "plans — surveyor name and address, plan number, location, state or country, "
            "coordinates, roads, fences, and related metadata. The accepted template is "
            "inserted into the console via the CAD plan prompt button. You can still type "
            "any valid generation prompt manually in the console."
        )
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        page_layout.addWidget(title)
        page_layout.addWidget(subtitle)

        editor_group = QGroupBox("CAD prompt template")
        editor_layout = QVBoxLayout(editor_group)
        self._cad_prompt_editor = QPlainTextEdit()
        self._cad_prompt_editor.setObjectName("chatInput")
        self._cad_prompt_editor.setMinimumHeight(280)
        self._cad_prompt_editor.setPlaceholderText("Enter your default CAD survey-plan prompt…")
        editor_layout.addWidget(self._cad_prompt_editor)

        self._cad_prompt_status = QLabel("")
        self._cad_prompt_status.setObjectName("hintLabel")
        self._cad_prompt_status.setWordWrap(True)
        editor_layout.addWidget(self._cad_prompt_status)
        page_layout.addWidget(editor_group)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._cad_prompt_user_btn = QPushButton("Apply User-defined Prompt")
        self._cad_prompt_user_btn.setObjectName("secondaryButton")
        self._cad_prompt_user_btn.setToolTip(
            "Stage the text in the editor as your custom default CAD prompt. "
            "Click Apply Change to confirm and save."
        )
        self._cad_prompt_user_btn.clicked.connect(self._on_cad_prompt_apply_user_defined)
        btn_row.addWidget(self._cad_prompt_user_btn)

        self._cad_prompt_restore_btn = QPushButton("Restore System-defined Prompt")
        self._cad_prompt_restore_btn.setObjectName("secondaryButton")
        self._cad_prompt_restore_btn.setToolTip(
            "Replace the editor with the packaged SurvyAI system CAD prompt. "
            "Click Apply Change to confirm and save."
        )
        self._cad_prompt_restore_btn.clicked.connect(self._on_cad_prompt_restore_system)
        btn_row.addWidget(self._cad_prompt_restore_btn)

        self._cad_prompt_apply_btn = QPushButton("Apply Change")
        self._cad_prompt_apply_btn.setToolTip(
            "Confirm and save the pending CAD prompt change to this computer."
        )
        self._cad_prompt_apply_btn.clicked.connect(self._on_cad_prompt_apply_change)
        btn_row.addWidget(self._cad_prompt_apply_btn)
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

        act_cad_prompt = QAction("Edit Default CAD Prompt…", self)
        act_cad_prompt.triggered.connect(self._show_cad_prompt_page)
        self._describe_menu_action(
            act_cad_prompt,
            "Edit the default CAD survey-plan prompt template used when inserting via "
            "CAD plan prompt — surveyor details, plan number, location, and related fields.",
        )
        account_menu.addAction(act_cad_prompt)

        act_diag = QAction("Diagnostics…", self)
        act_diag.triggered.connect(self._show_diagnostics_page)
        self._describe_menu_action(
            act_diag,
            "Technical readout for troubleshooting: environment, capabilities, and recent errors — "
            "pair with exporting a diagnostics bundle under File.",
        )
        account_menu.addAction(act_diag)

        account_menu.addSeparator()

        pay_sub = QAction("Paystack: Buy / extend Pro…", self)
        pay_sub.triggered.connect(self._on_paystack_subscribe)
        self._describe_menu_action(
            pay_sub,
            "Opens Paystack checkout in your browser to buy or extend Pro access (hosted models and "
            "periodic API credits). Choose daily, weekly, monthly, or annual billing.",
        )
        account_menu.addAction(pay_sub)

        pay_manage = QAction("Paystack: Manage old subscription…", self)
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
            "Getting started guide, documentation, tutorial, and product version — hover each link for details."
        )
        help_menu.menuAction().setStatusTip("Getting started, README, tutorial wizard, and About SurvyAI.")

        getting_started = QAction("Getting started guide", self)
        getting_started.triggered.connect(self._open_getting_started_guide)
        self._describe_menu_action(
            getting_started,
            "Short playbook with examples for CAD plotting, PDF-to-CAD, ArcGIS, and everyday SurvyAI use.",
        )
        help_menu.addAction(getting_started)

        readme = QAction("Documentation (README)", self)
        readme.triggered.connect(self._open_readme_docs)
        self._describe_menu_action(
            readme,
            "Opens the project README / user guide so you can read how features, tools, and workflows "
            "are meant to be used.",
        )
        help_menu.addAction(readme)

        updates = QAction("Check for updates…", self)
        updates.triggered.connect(self._check_for_updates)
        self._describe_menu_action(
            updates,
            "Compare this build with the cloud manifest, then stage a hash-verified full installer when available.",
        )
        help_menu.addAction(updates)

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
        self._silent_pull_entitlements_from_cloud()
        self._refresh_credits_page()
        self._central_stack.setCurrentWidget(self._credits_page)
        self._central_stack.setCurrentIndex(_PAGE_CREDITS)
        self._back_workspace_btn.setVisible(True)

    @Slot()
    def _show_cad_prompt_page(self) -> None:
        self._load_cad_prompt_editor_from_state()
        self._central_stack.setCurrentIndex(_PAGE_CAD_PROMPT)
        self._back_workspace_btn.setVisible(True)

    def _load_cad_prompt_editor_from_state(self) -> None:
        """Reload the CAD prompt editor from the last saved active template."""
        text = resolve_active_cad_prompt(self._state.default_cad_prompt)
        self._cad_prompt_editor.setPlainText(text)
        self._cad_prompt_pending = None
        if (self._state.default_cad_prompt or "").strip():
            self._cad_prompt_status.setText("Current: your user-defined CAD prompt is active.")
        else:
            self._cad_prompt_status.setText("Current: the packaged system CAD prompt is active.")

    @Slot()
    def _on_cad_prompt_apply_user_defined(self) -> None:
        text = self._cad_prompt_editor.toPlainText().strip()
        if not text:
            QMessageBox.warning(
                self,
                "Empty prompt",
                "Enter a CAD survey-plan prompt before applying a user-defined template.",
            )
            return
        self._cad_prompt_pending = "user"
        self._cad_prompt_status.setText(
            "Ready to apply your custom CAD prompt — click Apply Change to confirm and save."
        )

    @Slot()
    def _on_cad_prompt_restore_system(self) -> None:
        self._cad_prompt_editor.setPlainText(SYSTEM_DEFAULT_CAD_PROMPT)
        self._cad_prompt_pending = "system"
        self._cad_prompt_status.setText(
            "Ready to restore the packaged system CAD prompt — click Apply Change to confirm and save."
        )

    @Slot()
    def _on_cad_prompt_apply_change(self) -> None:
        editor_text = self._cad_prompt_editor.toPlainText()
        saved_active = resolve_active_cad_prompt(self._state.default_cad_prompt)
        has_pending = self._cad_prompt_pending in ("user", "system")
        draft_changed = editor_text.strip() != saved_active.strip()
        if not has_pending and not draft_changed:
            QMessageBox.information(
                self,
                "No changes",
                "There are no pending CAD prompt changes to apply.",
            )
            return

        answer = QMessageBox.question(
            self,
            "Apply Change",
            "Apply this CAD prompt as your default survey-plan template on this computer?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return
        if answer == QMessageBox.StandardButton.No:
            self._load_cad_prompt_editor_from_state()
            return

        # Yes — persist. System restore (or text matching system) stores empty = use packaged default.
        if self._cad_prompt_pending == "system" or is_system_default_text(editor_text):
            self._state.default_cad_prompt = ""
        else:
            self._state.default_cad_prompt = editor_text.strip()
        self._state_store.save(self._state)
        self._cad_prompt_pending = None
        if (self._state.default_cad_prompt or "").strip():
            self._cad_prompt_status.setText("Saved: your user-defined CAD prompt is now active.")
        else:
            self._cad_prompt_status.setText("Saved: the packaged system CAD prompt is now active.")
        self.statusBar().showMessage("Default CAD prompt updated.", 4000)

    @Slot()
    def _insert_cad_plan_prompt(self) -> None:
        """Insert the active default CAD prompt into the console input box."""
        if self._thread is not None and self._thread.isRunning():
            if self._active_conversation_id == self._running_conversation_id:
                QMessageBox.information(
                    self,
                    "Task in progress",
                    "Wait for the current task to finish before inserting a CAD plan prompt.",
                )
                return
        template = resolve_active_cad_prompt(self._state.default_cad_prompt)
        existing = self._input.toPlainText()
        if existing.strip():
            # Keep existing text; start the template on a new line.
            if existing.endswith("\n"):
                self._input.setPlainText(existing + template)
            else:
                self._input.setPlainText(existing + "\n" + template)
        else:
            self._input.setPlainText(template)
        self._input.setFocus(Qt.FocusReason.OtherFocusReason)
        cursor = self._input.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._input.setTextCursor(cursor)

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

            act_cad_prompt = QAction("Edit Default CAD Prompt", self)
            act_cad_prompt.triggered.connect(self._show_cad_prompt_page)
            self._describe_menu_action(
                act_cad_prompt,
                "Edit the default CAD survey-plan prompt template (surveyor details, plan number, "
                "location, and related fields) used by the CAD plan prompt button.",
            )
            self._user_menu.addAction(act_cad_prompt)

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

            act_cad_prompt = QAction("Edit Default CAD Prompt", self)
            act_cad_prompt.triggered.connect(self._show_cad_prompt_page)
            self._describe_menu_action(
                act_cad_prompt,
                "Edit the default CAD survey-plan prompt template used by the CAD plan prompt button.",
            )
            self._user_menu.addAction(act_cad_prompt)

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

        st = self._user_menu_btn.style()
        st.unpolish(self._user_menu_btn)
        st.polish(self._user_menu_btn)
        self._user_menu_btn.update()

    def _agent_process_payloads(self) -> tuple[dict, dict]:
        """Settings + feature-flag payloads matching what `AgentRunThread` sends,
        so the pre-warmed agent is reused (not rebuilt) on the first prompt."""
        ff = self._effective_feature_flags()
        return (
            self._service.settings.model_dump(),
            {
                "license_mode": ff.license_mode,
                "allow_autocad": ff.allow_autocad,
                "allow_arcgis": ff.allow_arcgis,
                "allow_blue_marble": ff.allow_blue_marble,
                "allow_internet_tools": ff.allow_internet_tools,
                "allow_vector_store": ff.allow_vector_store,
            },
        )

    def _prewarm_agent(self) -> None:
        """Build the heavy agent in the background warm process ahead of time so
        the first prompt doesn't pay the cold-start cost."""
        try:
            settings_payload, ff_payload = self._agent_process_payloads()
            prewarm_shared_agent_process(settings_payload, ff_payload)
            self._append_activity("Warming up agent engine…")
        except Exception:
            pass

    def _finish_startup(self) -> None:
        if not self._state.onboarding_complete:
            self._run_onboarding()
        # First install / first open: show the Getting Started playbook once.
        if not self._state.getting_started_seen:
            QTimer.singleShot(120, self._maybe_show_getting_started_first_run)
        # Kick off agent warm-up right away so the engine is ready by the time
        # the user submits their first prompt (eliminates per-prompt cold start).
        QTimer.singleShot(200, self._prewarm_agent)
        # Non-blocking post-start prompts (e.g. local models setup).
        QTimer.singleShot(650, self._maybe_prompt_ollama_install)
        self._sync_update_check_timer()
        # Consent-gated background update check after the UI settles.
        QTimer.singleShot(2800, self._maybe_auto_check_updates)
        # Keep license/Pro status current (admin override + Paystack webhooks).
        QTimer.singleShot(3500, self._ensure_account_status_poll)
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
        data_dir = Path(self._state.data_folder or self._state_store.default_data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        overrides["vector_store_path"] = str(data_dir / "vectordb")
        overrides["log_file"] = str(data_dir / "survyai-desktop.log")
        # Cloud overrides: token + base URL + injected platform keys/models from /v1/bootstrap.
        if self._state.cloud_api_base_url.strip() and self._state.cloud_access_token.strip():
            overrides["survyai_api_base_url"] = self._state.cloud_api_base_url.strip()
            overrides["survyai_access_token"] = self._state.cloud_access_token.strip()
            overrides["survyai_device_id"] = self._state.cloud_device_id.strip()
            bs = self._state.cloud_bootstrap or {}
            if isinstance(bs, dict):
                proxy_enabled = bool(bs.get("llm_proxy_enabled"))
                if proxy_enabled:
                    overrides["survyai_llm_proxy_enabled"] = True
                    overrides["survyai_llm_proxy_path"] = (
                        str(bs.get("llm_proxy_path") or "/v1/llm/chat").strip()
                        or "/v1/llm/chat"
                    )
                else:
                    # Legacy provider key injection path (kept for backward compatibility only).
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
                agent_cfg = bs.get("agent_config")
                if isinstance(agent_cfg, dict):
                    overrides["agent_cloud_config_json"] = json.dumps(agent_cfg)
                    for k in [
                        "primary_llm",
                        "fallback_llm",
                        "openai_model",
                        "openai_model_nano",
                        "openai_model_mini",
                        "openai_model_complex",
                        "enable_tiered_models",
                        "gemini_model",
                        "claude_model",
                        "deepseek_base_url",
                        "agent_temperature",
                        "agent_max_tokens",
                    ]:
                        if k in agent_cfg and agent_cfg.get(k) is not None:
                            overrides[k] = agent_cfg.get(k)
        # User-selected runtime providers win over cloud defaults. The cloud
        # bootstrap supplies hosted model credentials/config, but Apply Settings
        # must still honor the user's selected primary/fallback provider.
        # "auto" is persisted as the preference; Settings/agent only see a concrete provider.
        if self._state.preferred_primary_llm:
            overrides["primary_llm"] = resolve_primary_llm_selection(
                self._state.preferred_primary_llm
            )
        if self._state.preferred_fallback_llm:
            overrides["fallback_llm"] = self._state.preferred_fallback_llm
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
        # Settings/flags changed: re-warm the agent with the new configuration so
        # the next prompt stays fast instead of rebuilding mid-run.
        QTimer.singleShot(0, self._prewarm_agent)

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
        except (CloudApiError, Exception):
            self._clear_cloud_session()
            if not silent:
                return self._prompt_session_expired()
            return False

        self._state.cloud_access_token = tokens.access_token
        self._state.cloud_refresh_token = tokens.refresh_token
        self._state.cloud_access_token_expires_at = access_token_expires_at_iso(
            expires_in_seconds=tokens.expires_in
        )
        self._state_store.save(self._state)
        self._rebuild_service(skip_cloud_refresh=True)
        return True

    def _prompt_session_expired(self) -> bool:
        """
        Show a dialog informing the user their login session has expired and
        offering to continue without login or re-login.

        Returns True if the user chose to re-login and login succeeded, False otherwise.
        """
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Session expired")
        dlg.setIcon(QMessageBox.Warning)
        dlg.setText("Your login session has expired and you have been signed out.")
        dlg.setInformativeText(
            "You can continue using SurvyAI without a cloud login, but you may "
            "experience degraded performance (local/fallback models only, no hosted "
            "API keys, no credit-based features).\n\n"
            "Would you like to sign in again, or continue without login?"
        )
        btn_login = dlg.addButton("Sign in again", QMessageBox.AcceptRole)
        btn_continue = dlg.addButton("Continue without login", QMessageBox.RejectRole)
        dlg.setDefaultButton(btn_login)
        dlg.exec()
        if dlg.clickedButton() == btn_login:
            self._cloud_sign_in()
            return bool(
                self._state.cloud_access_token.strip()
                and self._state.profile.is_signed_in
            )
        return False

    def _clear_cloud_session(self) -> None:
        # Keep the production default so the next Login/Create flow needs no URL prompt.
        self._state.cloud_api_base_url = DEFAULT_CLOUD_API_BASE_URL
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
        self._state.usage_period_anchor = ""
        self._state.subscription_current_period_end = ""
        self._state.credit_banner_anchor_budget_usd = -1.0
        self._state.credit_banner_anchor_used_usd = -1.0
        self._state.credit_banner_dismissed_half = False
        self._state.credit_banner_dismissed_eighty = False
        self._state.credit_banner_dismissed_ninetyfive = False
        self._state.profile = AccountProfile()
        self._state_store.save(self._state)
        self._stop_payment_refresh_watch()
        self._stop_account_status_poll()
        self._rebuild_service(skip_cloud_refresh=True)
        self._refresh_account_views()
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

    def _is_dark_theme(self) -> bool:
        return (getattr(self._state, "theme", THEME_LIGHT) or THEME_LIGHT).strip().lower() == THEME_DARK

    def _apply_theme(self, theme: Optional[str] = None) -> None:
        t = (theme or getattr(self._state, "theme", THEME_LIGHT) or THEME_LIGHT).strip().lower()
        if t not in (THEME_LIGHT, THEME_DARK):
            t = THEME_LIGHT
        self._state.theme = t
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(get_stylesheet(t))
        dark = t == THEME_DARK
        if hasattr(self, "_logo"):
            self._logo.set_dark_ui(dark)
        if hasattr(self, "_theme_toggle"):
            self._theme_toggle.blockSignals(True)
            self._theme_toggle.setChecked(dark, animate=False)
            self._theme_toggle.blockSignals(False)
        if hasattr(self, "_transcript"):
            self._render_active_conversation()

    @Slot(bool)
    def _on_dark_mode_toggled(self, checked: bool) -> None:
        self._apply_theme(THEME_DARK if checked else THEME_LIGHT)
        self._state_store.save(self._state)
        self.statusBar().showMessage(
            f"{'Dark' if checked else 'Light'} mode enabled.",
            3000,
        )

    def _apply_state_to_ui(self) -> None:
        self._workspace_edit.setText(self._state.workspace_path)
        self._settings_workspace.setText(self._state.workspace_path)
        self._settings_data_folder.setText(self._state.data_folder)
        self._fallback_cb.setChecked(self._state.use_fallback_llm)
        self._safe_mode_cb.setChecked(self._state.safe_mode)
        self._apply_theme()
        self._fast_mode_cb.setChecked(bool(getattr(self._state, "fast_mode_non_file_prompts", False)))
        self._refresh_fast_mode_indicator()
        if hasattr(self, "_auto_check_updates_cb"):
            self._auto_check_updates_cb.blockSignals(True)
            self._auto_check_updates_cb.setChecked(bool(getattr(self._state, "auto_check_updates", False)))
            self._auto_check_updates_cb.blockSignals(False)
            self._refresh_updates_status_label()
        self._set_combo_value(
            self._primary_llm_combo,
            normalize_primary_llm_selection(self._state.preferred_primary_llm),
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
        cloud_ok = bool(self._state.cloud_access_token.strip() and self._state.cloud_api_base_url.strip())
        if hasattr(self, "_change_password_btn"):
            self._change_password_btn.setEnabled(cloud_ok)
            self._change_password_btn.setToolTip(
                "Update your SurvyAI Cloud password."
                if cloud_ok
                else "Sign in to SurvyAI Cloud to change your password."
            )
        self._refresh_user_menu()

    def _cloud_pro_still_entitled(self, me: dict | None = None) -> bool:
        """True when cloud plan is Pro with an open period and remaining credits."""
        me = me if isinstance(me, dict) else (
            self._state.cloud_me if isinstance(self._state.cloud_me, dict) else {}
        )
        plan_low = str(me.get("plan_slug") or "").strip().lower()
        st_low = str(me.get("subscription_status") or "").strip().lower()
        if plan_low != "pro" or st_low not in {"active", "trialing", "non_renewing"}:
            return False
        if me.get("can_use_platform_llm") is False:
            return False
        period_end = self._parse_datetime_value(
            me.get("subscription_current_period_end")
            or self._state.subscription_current_period_end
        )
        if period_end is not None and period_end <= datetime.now(timezone.utc):
            return False
        # Cloud /me (and admin overrides) are authoritative for pool/used when present.
        budget = float(self._state.monthly_credits_usd or 0.0)
        used = float(self._state.monthly_credits_used_usd or 0.0)
        if me.get("monthly_credits_usd") is not None:
            try:
                budget = float(me.get("monthly_credits_usd") or 0.0)
            except (TypeError, ValueError):
                pass
        if me.get("monthly_credits_used_usd") is not None:
            try:
                used = float(me.get("monthly_credits_used_usd") or 0.0)
            except (TypeError, ValueError):
                pass
        if budget <= 0 or used + 1e-6 >= budget:
            return False
        return True

    def _refresh_license_card(self) -> None:
        runtime_flags = self._feature_flags
        display_flags = self._display_feature_flags
        plan = "Pro" if display_flags.license_mode == "pro" else "UserMode"
        me = self._state.cloud_me if isinstance(self._state.cloud_me, dict) else {}
        if me.get("plan_slug"):
            plan = str(me.get("plan_slug") or plan)
        # Mirror server reconcile: expired / exhausted Pro should read as Free in Settings.
        cloud_ok = bool(self._state.cloud_api_base_url.strip() and self._state.cloud_access_token.strip())
        st_low = str(me.get("subscription_status") or "").lower()
        plan_low = str(me.get("plan_slug") or "").lower()
        has_pro = self._cloud_pro_still_entitled(me) if cloud_ok and me else False
        if (
            cloud_ok
            and me
            and plan_low == "pro"
            and not has_pro
            and st_low in {"active", "trialing", "non_renewing"}
        ):
            plan = "free"
            plan_low = "free"
            st_low = "none"
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
        sub_status = st_low if st_low else str(me.get("subscription_status") or "").strip()
        if sub_status:
            status = f"{status} | Cloud subscription: {sub_status}"
        period = me.get("subscription_current_period_end")
        if period and has_pro:
            status = f"{status} | Renewal / period end: {period}"
        license_text = (
            f"Plan: {plan}\n"
            f"Status: {status}\n"
            f"Source: {source}\n"
            f"Enabled integrations: {', '.join(enabled) if enabled else 'Core assistant only'}"
        )
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

        if has_pro:
            self._paystack_subscribe_btn.setText("Extend Pro access…")
            self._paystack_subscribe_btn.setToolTip(
                "Add another Paystack access period to your active Pro account."
            )
        else:
            self._paystack_subscribe_btn.setText("Buy Pro access…")
            self._paystack_subscribe_btn.setToolTip(
                "Choose daily, weekly, monthly, or annual access, then complete checkout in your browser."
            )
        self._paystack_subscribe_btn.setEnabled(cloud_ok)
        # Pro users can intentionally buy another period; the label makes that explicit.
        self._paystack_subscribe_btn.setVisible(cloud_ok)
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
        body = html.escape(text).replace("\n", "<br/>")
        dark = self._is_dark_theme()
        # Use <p> tags for label: Qt QTextEdit treats <p> as a block element reliably,
        # avoiding the inline-span issue where label and bubble merge on the same line.
        label_style = (
            "margin:0 0 3px 0;padding:0;font-size:9pt;font-weight:700;"
            "text-transform:uppercase;letter-spacing:0.04em;line-height:1.2;"
        )
        bubble_style = (
            "margin:0;padding:8px 12px;font-size:9.75pt;line-height:1.25;border-radius:9px;"
        )
        if role == "user":
            label_color = "#93c5fd" if dark else "#1d4ed8"
            bg, border, fg = (
                ("#172554", "#3b82f6", "#e4e4e7") if dark else ("#eff6ff", "#bfdbfe", "#0f172a")
            )
            return (
                '<table width="100%" cellpadding="0" cellspacing="0">'
                '<tr><td width="28">&nbsp;</td>'
                '<td>'
                f'<p style="{label_style}color:{label_color};">You</p>'
                f'<p style="{bubble_style}background:{bg};border:1px solid {border};color:{fg};">'
                f'{body}</p>'
                '</td></tr></table>'
            )
        if role == "assistant":
            if error:
                fg, border, bg = (
                    ("#fecaca", "#f87171", "#450a0a") if dark else ("#b91c1c", "#fecaca", "#fef2f2")
                )
            else:
                fg, border, bg = (
                    ("#e4e4e7", "#3f3f46", "#1e1e22") if dark else ("#0f172a", "#e2e8f0", "#ffffff")
                )
            label_color = "#34d399" if dark else "#047857"
            return (
                '<table width="100%" cellpadding="0" cellspacing="0">'
                '<tr><td>'
                f'<p style="{label_style}color:{label_color};">SurvyAI</p>'
                f'<p style="{bubble_style}background:{bg};border:1px solid {border};color:{fg};">'
                f'{body}</p>'
                '</td><td width="28">&nbsp;</td></tr></table>'
            )
        sys_color = "#71717a" if dark else "#64748b"
        return (
            f'<p style="margin:3px 0;padding:0 4px;color:{sys_color};'
            f'font-size:9pt;line-height:1.2;font-style:italic;">{body}</p>'
        )

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
        self._transcript.setHtml("".join(parts))
        self._transcript.moveCursor(QTextCursor.MoveOperation.End)
        self._transcript.ensureCursorVisible()

    def _schedule_desktop_state_save(self) -> None:
        self._desktop_state_save_timer.start()

    @Slot()
    def _flush_desktop_state_save(self) -> None:
        self._state_store.save(self._state)

    def _store_conversation_message(
        self,
        role: str,
        text: str,
        *,
        error: bool = False,
        conversation_id: Optional[str] = None,
    ) -> None:
        """Append a message to a conversation.

        If *conversation_id* is given, the message is written to that specific
        conversation regardless of which tab is currently active.  This prevents
        results from one conversation bleeding into another when the user
        switches tabs while a query is running.
        """
        target_id = conversation_id or self._active_conversation().conversation_id
        self._state_store.append_conversation_message(
            self._state,
            conversation_id=target_id,
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

    def _credit_markup_multiplier(self) -> float:
        m = float(self._state.credit_markup_multiplier or 0.0)
        return m if m > 0 else 2.0

    def _billed_cost_usd(self, raw_cost_usd: float) -> float:
        """SurvyAI user-facing USD = raw LLM API cost × markup (default 2×)."""
        raw = float(raw_cost_usd or 0.0)
        if raw <= 0:
            return 0.0
        return round(raw * self._credit_markup_multiplier(), 6)

    def _parse_datetime_value(self, raw: object) -> Optional[datetime]:
        if raw is None:
            return None
        if isinstance(raw, datetime):
            dt = raw
        else:
            text = str(raw).strip()
            if not text:
                return None
            try:
                if text.endswith("Z"):
                    text = text[:-1] + "+00:00"
                dt = datetime.fromisoformat(text)
            except Exception:
                try:
                    dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                except Exception:
                    return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _format_history_timestamp(self, created_at: str) -> str:
        ts = self._parse_datetime_value(created_at)
        if ts is None:
            return created_at or ""
        return ts.astimezone().strftime("%Y-%m-%d %H:%M:%S")

    def _is_rolling_pro_billing_period(self) -> bool:
        me = self._state.cloud_me if isinstance(self._state.cloud_me, dict) else {}
        interval = (
            self._state.credits_billing_interval
            or me.get("credits_billing_interval")
            or ""
        ).strip().lower()
        return self._cloud_pro_still_entitled(me) and interval in {
            "daily",
            "weekly",
            "monthly",
            "annual",
        }

    def _billing_period_days(self, interval: str) -> int:
        return {"daily": 1, "weekly": 7, "monthly": 30, "annual": 365}.get(
            interval.strip().lower(), 30
        )

    def _is_monthly_pro_billing_period(self) -> bool:
        """Backward-compatible alias for monthly rolling Pro billing."""
        me = self._state.cloud_me if isinstance(self._state.cloud_me, dict) else {}
        interval = (
            self._state.credits_billing_interval
            or me.get("credits_billing_interval")
            or ""
        ).strip().lower()
        return self._cloud_pro_still_entitled(me) and interval == "monthly"

    def _credits_usage_period_bounds(self) -> tuple[datetime, Optional[datetime], str]:
        """
        Return (start inclusive, end exclusive, human label) for the current usage window.

        Prefer the exact cloud paid window (usage_period_anchor → subscription_current_period_end).
        Free / unsigned-in: calendar month local time.
        """
        if self._is_rolling_pro_billing_period():
            me = self._state.cloud_me if isinstance(self._state.cloud_me, dict) else {}
            interval = (
                self._state.credits_billing_interval
                or me.get("credits_billing_interval")
                or "monthly"
            ).strip().lower()
            period_days = self._billing_period_days(interval)
            period_end = self._parse_datetime_value(
                self._state.subscription_current_period_end
                or me.get("subscription_current_period_end")
            )
            period_start = self._parse_datetime_value(
                self._state.usage_period_anchor or me.get("usage_period_anchor")
            )
            if period_start is None and period_end is not None:
                # Legacy fallback when cloud has not yet sent the paid-window anchor.
                period_start = period_end - timedelta(days=period_days)
            if period_start is not None and period_end is not None:
                interval_label = {
                    "daily": "daily",
                    "weekly": "weekly",
                    "monthly": "monthly",
                    "annual": "annual",
                }.get(interval, "billing")
                label = (
                    f"Current {interval_label} paid window "
                    f"({period_start.strftime('%d %b %Y %H:%M')} – "
                    f"{period_end.strftime('%d %b %Y %H:%M')} UTC)"
                )
                return period_start, period_end, label

        now_local = datetime.now().astimezone()
        period_start_local = now_local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if period_start_local.month == 12:
            period_end_local = period_start_local.replace(year=period_start_local.year + 1, month=1)
        else:
            period_end_local = period_start_local.replace(month=period_start_local.month + 1)
        period_start = period_start_local.astimezone(timezone.utc)
        period_end = period_end_local.astimezone(timezone.utc)
        label = (
            f"Current calendar month "
            f"({period_start_local.strftime('%d %b')} – {now_local.strftime('%d %b %Y')}, local time)"
        )
        return period_start, period_end, label

    def _history_entry_in_period(
        self,
        created_at: str,
        *,
        period_start: datetime,
        period_end: Optional[datetime],
    ) -> bool:
        ts = self._parse_datetime_value(created_at)
        if ts is None:
            return False
        if ts < period_start:
            return False
        if period_end is not None and ts >= period_end:
            return False
        return True

    def _billed_usage_total_from_history(
        self,
        *,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> float:
        """Sum of billed run costs; optionally limited to a billing window."""
        try:
            from utils.cost_estimator import is_local_free_model
        except Exception:
            is_local_free_model = lambda _n: False  # type: ignore
        total = 0.0
        for entry in self._state.output_history:
            if period_start is not None:
                if not self._history_entry_in_period(
                    entry.created_at,
                    period_start=period_start,
                    period_end=period_end,
                ):
                    continue
            if is_local_free_model(str(getattr(entry, "model_name", "") or "")):
                continue
            raw = float(getattr(entry, "llm_cost_usd", 0.0) or 0.0)
            if raw > 0:
                total += self._billed_cost_usd(raw)
        return round(total, 6)

    def _cloud_credits_are_authoritative(self) -> bool:
        """True when signed-in cloud counters (incl. admin overrides) own Used/Remaining."""
        return bool(
            self._state.cloud_access_token.strip()
            and self._state.cloud_api_base_url.strip()
        )

    def _period_credits_used_usd(self) -> float:
        """Billed USD consumed in the active billing window (not lifetime)."""
        cloud_used = round(float(self._state.monthly_credits_used_usd or 0.0), 6)
        # Signed-in: cloud /me and /entitlements (and admin overrides) are source of truth.
        # Local output_history is only a recent-activity log — never inflate Used above cloud.
        if self._cloud_credits_are_authoritative():
            return cloud_used
        period_start, period_end, _ = self._credits_usage_period_bounds()
        from_history = self._billed_usage_total_from_history(
            period_start=period_start,
            period_end=period_end,
        )
        if self._state.output_history or cloud_used > 0:
            return round(max(from_history, cloud_used), 6)
        return cloud_used

    def _lifetime_credits_used_usd(self) -> float:
        return self._billed_usage_total_from_history()

    def _reconcile_credits_used_from_history(self) -> None:
        """Align stored period-used with local history only when offline / no cloud session.

        When signed in, never raise Used from the local run log — that would undo admin
        resets (e.g. used=0.1) and show stale totals like $5.09.
        """
        if self._cloud_credits_are_authoritative():
            return
        period_start, period_end, _ = self._credits_usage_period_bounds()
        from_history = self._billed_usage_total_from_history(
            period_start=period_start,
            period_end=period_end,
        )
        stored = float(self._state.monthly_credits_used_usd or 0.0)
        merged = max(from_history, stored)
        if abs(merged - stored) > 1e-6:
            self._state.monthly_credits_used_usd = merged

    def _refresh_credits_page(self) -> None:
        self._reconcile_credits_used_from_history()
        budget = self._state.monthly_credits_usd
        used = self._period_credits_used_usd()
        remaining = max(budget - used, 0.0)
        _, _, period_label = self._credits_usage_period_bounds()
        if self._cloud_credits_are_authoritative():
            self._credits_period_note.setText(
                f"{period_label}. Credit pool and Used come from your SurvyAI account "
                f"(including admin adjustments). Recent usage below is a local activity log "
                f"and does not override the Used counter. Lifetime total is at the bottom."
            )
        else:
            self._credits_period_note.setText(
                f"{period_label}. Credit pool is subscription USD. Used is billed hosted-model "
                f"usage inside this paid window. Lifetime total is at the bottom."
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
        period_start, period_end, _ = self._credits_usage_period_bounds()
        for entry in self._state.output_history:
            if not self._history_entry_in_period(
                entry.created_at,
                period_start=period_start,
                period_end=period_end,
            ):
                continue
            cost_val = float(getattr(entry, "llm_cost_usd", 0.0) or 0.0)
            model = entry.model_name or "?"
            try:
                from utils.cost_estimator import is_local_free_model

                if is_local_free_model(str(model)):
                    continue
            except Exception:
                pass
            if cost_val <= 0:
                continue
            billed_usd = self._billed_cost_usd(cost_val)
            ts = self._format_history_timestamp(entry.created_at)
            q_preview = (entry.query or "")[:60].replace("\n", " ")
            lines.append(f"[{ts}]  ${billed_usd:,.2f}  ({model})  {q_preview}")
            if len(lines) >= 50:
                break
        self._credits_history_text.setPlainText(
            "\n".join(lines) if lines else "No cost data recorded yet for this billing window."
        )
        lifetime = self._lifetime_credits_used_usd()
        self._credits_lifetime_label.setText(
            f"Lifetime hosted-model usage (all time): ${lifetime:,.2f} USD billed"
        )
        self._update_credit_usage_notice()

    def _credit_banner_reset_dismissals_if_period_changed(self) -> None:
        """Clear dismiss flags when the server budget changes or usage rolls back (new period)."""
        b = float(self._state.monthly_credits_usd or 0.0)
        u = self._period_credits_used_usd()
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
        used = self._period_credits_used_usd()
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
                QTimer.singleShot(0, self._refresh_console_prompt_layout)
            else:
                self._credit_notice_current_band = "free"
                self._credit_notice_label.setText(
                    "You are on the free plan with no subscription credit pool ($0). "
                    "Use local models (Ollama), or sign in and upgrade for hosted usage."
                )
                self._credit_notice_dismiss_btn.setVisible(False)
                self._credit_notice_wrap.setMinimumHeight(26)
                self._credit_notice_wrap.setVisible(True)
                QTimer.singleShot(0, self._refresh_console_prompt_layout)
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
            QTimer.singleShot(0, self._refresh_console_prompt_layout)
            return

        self._credit_notice_label.setText(text)
        self._credit_notice_dismiss_btn.setVisible(show_dismiss)
        self._credit_notice_wrap.setMinimumHeight(26)
        self._credit_notice_wrap.setVisible(True)
        QTimer.singleShot(0, self._refresh_console_prompt_layout)

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
        """Average billed USD per run in the current billing window."""
        try:
            from utils.cost_estimator import is_local_free_model
        except Exception:
            is_local_free_model = lambda _n: False  # type: ignore
        period_start, period_end, _ = self._credits_usage_period_bounds()
        costs: list[float] = []
        for entry in self._state.output_history[:20]:
            if not self._history_entry_in_period(
                entry.created_at,
                period_start=period_start,
                period_end=period_end,
            ):
                continue
            if is_local_free_model(str(getattr(entry, "model_name", "") or "")):
                continue
            c = float(getattr(entry, "llm_cost_usd", 0.0) or 0.0)
            if c > 0:
                costs.append(self._billed_cost_usd(c))
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
        if not self._state.cloud_access_token.strip():
            return False, ""
        if self._effective_run_llm_id() == "ollama":
            return False, ""
        budget = float(self._state.monthly_credits_usd or 0.0)
        used = self._period_credits_used_usd()
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
        self._refresh_credits_page()
        base, token = self._cloud_base_and_token()
        if not base or not token:
            self.statusBar().showMessage("Usage refreshed from local run history.", 4000)
            return
        if self._cloud_network_busy():
            self.statusBar().showMessage("Cloud update already in progress…", 3000)
            return

        self._begin_cloud_busy("Syncing credits from cloud…")
        thread = CloudCreditsSyncThread(self._make_cloud_credits_sync_payload(), parent=self)
        self._cloud_credits_sync_thread = thread

        def _done() -> None:
            self._end_cloud_busy()
            if self._cloud_credits_sync_thread is thread:
                self._cloud_credits_sync_thread = None

        def _on_ok(result_obj: object) -> None:
            result = result_obj if isinstance(result_obj, CloudCreditsSyncResult) else None
            if result is None:
                QMessageBox.warning(self, "Credits sync failed", "Unexpected sync response.")
                return
            if result.access_token:
                self._state.cloud_access_token = result.access_token
            if result.refresh_token:
                self._state.cloud_refresh_token = result.refresh_token
            if result.access_token_expires_at:
                self._state.cloud_access_token_expires_at = result.access_token_expires_at
            ent_d = result.ent if isinstance(result.ent, dict) else {}
            self._sync_credits_from_entitlements(ent_d)
            self._state_store.save(self._state)
            self._refresh_credits_page()
            self.statusBar().showMessage("Credits refreshed from cloud.", 4000)

        def _on_fail(msg: str) -> None:
            if self._cloud_sync_message_is_session_expired(msg):
                self._clear_cloud_session()
                self._prompt_session_expired()
                return
            QMessageBox.warning(self, "Credits sync failed", msg)

        thread.succeeded.connect(_on_ok)
        thread.failed.connect(_on_fail)
        thread.finished.connect(_done)
        thread.start()

    # ------------------------------------------------------------------
    # Console / history helpers
    # ------------------------------------------------------------------

    def _append_html(self, html_block: str) -> None:
        self._transcript.moveCursor(QTextCursor.MoveOperation.End)
        self._transcript.insertHtml(html_block)
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
        cost = float(result.llm_cost_usd or 0.0)
        if cost <= 0:
            _, cost = self._usage_event_for_result(result)
        try:
            from utils.cost_estimator import is_local_free_model

            if is_local_free_model(str(result.model_name or "")) or str(
                result.llm_used or ""
            ).strip().lower() in {"ollama", "local"}:
                cost = 0.0
        except Exception:
            pass
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
            llm_cost_usd=float(cost or 0.0),
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
        self._state.credit_markup_multiplier = float(ent.get("credit_markup_multiplier") or 2.0)
        self._state.can_use_platform_llm = bool(ent.get("can_use_platform_llm"))
        self._state.credits_billing_interval = str(ent.get("credits_billing_interval") or "").strip().lower()

        def _as_iso(value: object) -> str:
            if value is None:
                return ""
            if hasattr(value, "isoformat"):
                try:
                    return str(value.isoformat())
                except Exception:
                    return str(value)
            text = str(value).strip()
            return text

        anchor = _as_iso(ent.get("usage_period_anchor"))
        period_end = _as_iso(ent.get("subscription_current_period_end"))
        if anchor:
            self._state.usage_period_anchor = anchor
        if period_end:
            self._state.subscription_current_period_end = period_end

        # Keep cloud_me period fields fresh even on entitlements-only refresh.
        me = self._state.cloud_me if isinstance(self._state.cloud_me, dict) else {}
        me = dict(me)
        if ent.get("plan_slug") is not None:
            me["plan_slug"] = ent.get("plan_slug")
        if ent.get("subscription_status") is not None:
            me["subscription_status"] = ent.get("subscription_status")
        if self._state.credits_billing_interval:
            me["credits_billing_interval"] = self._state.credits_billing_interval
        if self._state.usage_period_anchor:
            me["usage_period_anchor"] = self._state.usage_period_anchor
        if self._state.subscription_current_period_end:
            me["subscription_current_period_end"] = self._state.subscription_current_period_end
        me["monthly_credits_usd"] = self._state.monthly_credits_usd
        me["monthly_credits_used_usd"] = self._state.monthly_credits_used_usd
        me["can_use_platform_llm"] = self._state.can_use_platform_llm
        self._state.cloud_me = me

        self._reconcile_credits_used_from_history()
        self._update_credit_usage_notice()

    def _usage_event_for_result(self, result: AgentRunResult) -> tuple[dict[str, object] | None, float]:
        try:
            from utils.cost_estimator import is_local_free_model
        except Exception:
            is_local_free_model = lambda _n: False  # type: ignore
        model_name = str(result.model_name or "")
        # Local / Ollama runs never bill credits (even if a mis-tagged cost leaked in).
        if is_local_free_model(model_name) or str(result.llm_used or "").strip().lower() in {
            "ollama",
            "local",
        }:
            return None, 0.0
        usage = summarize_graph_llm_usage(
            list((result.raw or {}).get("messages") or []),
            model_name,
            response_text=result.response or "",
            infer_missing_cached=False,
        )
        usage_estimated = bool(usage.get("estimated"))
        raw_cost = (
            float(result.llm_cost_usd or 0.0)
            if usage_estimated
            else float(usage.get("cost_usd") or result.llm_cost_usd or 0.0)
        )
        if raw_cost <= 0:
            return None, 0.0
        if usage_estimated:
            # Local fast paths may carry provider-reported cost without token
            # breakdown. Count it locally, but don't report estimated tokens as
            # billable cloud usage.
            return None, raw_cost
        event: dict[str, object] = {
            "kind": "agent_run",
            "quantity": 1,
            "cost_usd": round(raw_cost, 6),
            "model_name": str(usage.get("model_name") or result.model_name or ""),
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "cached_input_tokens": int(usage.get("cached_input_tokens") or 0),
            "meta": {
                "usage_estimated": bool(usage.get("estimated")),
                "usage_turns": int(usage.get("usage_turns") or 0),
                "llm_used": str(result.llm_used or ""),
            },
        }
        return event, raw_cost

    def _using_cloud_llm_proxy(self) -> bool:
        return bool(
            getattr(self._settings, "survyai_llm_proxy_enabled", False)
            and str(getattr(self._settings, "survyai_api_base_url", "") or "").strip()
            and str(getattr(self._settings, "survyai_access_token", "") or "").strip()
        )

    def _auto_switch_to_ollama_after_credit_exhaustion(self) -> None:
        if self._effective_run_llm_id() == "ollama":
            return
        self._state.preferred_primary_llm = "ollama"
        self._state.preferred_fallback_llm = "ollama"
        self._state.use_fallback_llm = False
        self._state_store.save(self._state)
        self._rebuild_service(skip_cloud_refresh=True)
        self._refresh_account_views()
        self._refresh_credits_page()
        QMessageBox.information(
            self,
            "Hosted credits exhausted",
            "Your hosted SurvyAI credit balance has been exhausted for this billing period.\n\n"
            "SurvyAI has switched your Primary and Fallback models to Ollama so you can keep working locally. "
            "If you have not installed or selected an Ollama model yet, open Account -> Local models (Ollama) to finish setup.",
        )

    def _account_for_run_cost(self, result: AgentRunResult) -> None:
        """Update local credit tally and (best-effort) report cost to the cloud usage API."""
        if self._using_cloud_llm_proxy():
            self._silent_pull_entitlements_from_cloud()
            self._reconcile_credits_used_from_history()
            self._state_store.save(self._state)
            self._refresh_credits_page()
            self._refresh_license_card()
            blocked, _msg = self._platform_credit_wall_should_block()
            if blocked:
                self._auto_switch_to_ollama_after_credit_exhaustion()
            self._update_credit_usage_notice()
            return
        usage_event, raw_cost = self._usage_event_for_result(result)
        if raw_cost <= 0:
            self._refresh_credits_page()
            return
        marked_up = self._billed_cost_usd(raw_cost)
        self._state.monthly_credits_used_usd = round(
            self._state.monthly_credits_used_usd + marked_up, 6
        )
        self._reconcile_credits_used_from_history()
        self._state_store.save(self._state)
        if usage_event is not None:
            self._report_cost_to_cloud(usage_event)
        self._refresh_credits_page()
        self._refresh_license_card()
        self._update_credit_usage_notice()

    def _report_cost_to_cloud(self, usage_event: dict[str, object]) -> None:
        """Best-effort POST to /v1/usage/events with structured run usage."""
        base, token = self._cloud_base_and_token()
        if not base or not token:
            return
        try:
            resp = report_usage_batch(
                base_url=base,
                access_token=token,
                device_id=(self._state.cloud_device_id or "").strip() or None,
                events=[usage_event],
                timeout_s=8,
            )
            self._state.monthly_credits_used_usd = float(resp.get("monthly_credits_used_usd") or 0.0)
            self._state.monthly_credits_usd = float(resp.get("monthly_credits_usd") or self._state.monthly_credits_usd)
            self._reconcile_credits_used_from_history()
            self._state_store.save(self._state)
            self._refresh_credits_page()
            self._refresh_license_card()
        except CloudApiError as exc:
            if "credit balance exhausted" in str(exc).lower():
                self._silent_pull_entitlements_from_cloud()
                self._refresh_credits_page()
                self._refresh_license_card()
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
            if self._active_conversation_id == self._running_conversation_id:
                return
            QMessageBox.information(
                self,
                "Task in progress",
                "Another conversation is currently running. Please wait for it to finish before sending a new prompt.",
            )
            return
        if not self._workspace_edit.text().strip():
            QMessageBox.warning(self, "Workspace required", "Choose a workspace before running tasks.")
            return
        if self._state.cloud_api_base_url.strip() and self._state.cloud_refresh_token.strip():
            if not self._ensure_cloud_token_valid(silent=False):
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
        resolved = self._try_resolve_internet_permission_reply(text)
        if resolved:
            enhanced = resolved
            self._append_activity("Internet permission granted — searching for your original question.")
        else:
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
    _MAX_MSG_CHARS_ESSAY_SAVE = 1_000_000  # preserve full assistant answers for essay export

    def _try_resolve_internet_permission_reply(self, raw_query: str) -> Optional[str]:
        """If the user is answering a prior internet-permission ask, return a clean
        tagged query for the agent (skips history injection that confuses routing)."""
        conv = self._active_conversation()
        prior = conv.messages[:-1] if conv.messages else []
        last_assistant = ""
        for m in reversed(prior):
            if m.role == "assistant":
                last_assistant = m.content or ""
                break
        if not _assistant_asked_internet_permission(last_assistant):
            return None
        if not _is_permission_affirmation(raw_query):
            return None
        users = [(m.content or "").strip() for m in prior if m.role == "user"]
        underlying = ""
        for u in reversed(users):
            if u and len(u) >= 8 and not _is_permission_affirmation(u):
                underlying = u
                break
        if not underlying and users:
            underlying = users[0]
        if not underlying:
            return None
        return f"[INTERNET_PERMISSION_GRANTED]\n{underlying}"

    @staticmethod
    def _clean_question_from_enhanced(q: str) -> str:
        """Extract the substantive user question from a history-enriched agent query."""
        text = (q or "").strip()
        marker = "NOW, the user wants you to continue with this new request:"
        if marker in text:
            text = text.split(marker)[-1].strip()
        for tag in (
            "[INTERNET_PERMISSION_GRANTED]",
            "[INTERNET_PERMISSION_DENIED]",
            "[INTERNET_PERMISSION_REQUEST]",
        ):
            text = text.replace(tag, "").strip()
        return text

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
        if _is_standalone_knowledge_question(raw_query):
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

        # Determine how much context to inject:
        # - Clear follow-up  → full recent history (up to _MAX_HISTORY_TURNS)
        # - Ambiguous        → last exchange only (2 messages)
        # - Clearly new topic → send bare query, no context prefix at all.
        #   This prevents the agent from "continuing" an unrelated prior workflow
        #   (e.g. a knowledge question sent after a CAD task) and is the primary
        #   guard against the agent picking up the wrong task.
        if include_full_history:
            turns = turns[-self._MAX_HISTORY_TURNS:]
        else:
            last_two = turns[-2:]
            if _is_clearly_new_topic(raw_query, last_two):
                return raw_query
            turns = last_two

        parts: list[str] = [
            "=== CONVERSATION CONTEXT (REFERENCE ONLY) ===",
            "The block below is recent conversation history provided ONLY as optional",
            "background. It is NOT a task. Follow these rules strictly:",
            "1. Answer the CURRENT REQUEST at the very bottom — nothing else.",
            "2. Use this history ONLY if the current request clearly refers to it",
            "   (e.g. 'it', 'that plan', 'add another road', 'change the title').",
            "3. If the current request is a new/unrelated topic, IGNORE this history",
            "   completely and do NOT resume or repeat any previous tool/file/CAD",
            "   operation. Do not open files or run tools unless the CURRENT request",
            "   explicitly asks for it.",
            "",
        ]
        preserve_full_assistant = _is_save_session_docx_request(raw_query)
        exchange = 0
        for t in turns:
            content = t.content
            max_chars = (
                self._MAX_MSG_CHARS_ESSAY_SAVE
                if preserve_full_assistant and t.role == "assistant"
                else self._MAX_MSG_CHARS
            )
            if len(content) > max_chars:
                content = content[:max_chars] + "…[truncated]"
            if t.role == "user":
                exchange += 1
                parts.append(f"--- Exchange {exchange} ---")
                parts.append(f"User: {content}")
            else:
                parts.append(f"Assistant: {content}")
            parts.append("")

        parts.append("--- End of History (reference only) ---")
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

        # Host RAM hard-cap before starting a local Ollama run (avoids PC lock/hibernate).
        if self._effective_run_llm_id() == "ollama":
            from survyai.ollama_support import ollama_ram_policy

            model = (
                self._state.ollama_model.strip()
                or str(getattr(self._settings, "ollama_model", "") or "").strip()
                or "llama3.2:1b"
            )
            ok, ram_err, _num_ctx = ollama_ram_policy(model)
            if not ok:
                QMessageBox.warning(self, "Not enough free memory", ram_err)
                self._append_activity(ram_err)
                self.statusBar().showMessage("Blocked: not enough free memory for Ollama")
                return

        self._running_conversation_id = self._active_conversation_id
        self._send_btn.setEnabled(False)
        self._fallback_cb.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._retry_btn.setEnabled(False)
        self._cad_prompt_btn.setEnabled(False)
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
        # QueuedConnection: dialog must run on the GUI thread (never Direct on the worker).
        self._thread.confirm_overwrite.connect(
            self._on_cad_file_conflict,
            Qt.ConnectionType.QueuedConnection,
        )
        self._thread.start()
        self.statusBar().showMessage("Working…")

    @Slot(object)
    def _on_cad_file_conflict(self, payload: object) -> None:
        """Show a foreground SurvyAI dialog for CAD overwrite/modify confirmation."""
        data = payload if isinstance(payload, dict) else {}
        path = str(data.get("path") or "").strip()
        mode = str(data.get("mode") or "overwrite").strip().lower()
        thread = self._thread
        # If a conflict dialog is already up, bring it forward (do not drop the reply path).
        existing = getattr(self, "_cad_conflict_dialog", None)
        if existing is not None:
            try:
                existing.raise_()
                existing.activateWindow()
            except Exception:
                pass
            return

        def _finish(result_code: int) -> None:
            self._cad_conflict_dialog = None
            accepted = int(result_code) == int(QDialog.DialogCode.Accepted)
            if thread is not None:
                thread.provide_confirm_result(accepted)
            if accepted:
                self._append_activity("Overwrite confirmed — continuing CAD generation…")
                self.statusBar().showMessage("Overwrite confirmed — preparing drawing…")
            else:
                self._append_activity("Kept existing drawing; agent was told not to overwrite.")
                self.statusBar().showMessage("Existing drawing kept.")

        try:
            # Bring SurvyAI forward so the dialog is visible (not only a taskbar flash).
            self.showNormal()
            self.raise_()
            self.activateWindow()
            app = QApplication.instance()
            if app is not None:
                app.alert(self, 0)
                app.setActiveWindow(self)
            dark = str(getattr(self._state, "theme", "") or "").lower() == "dark"
            dlg = _CadFileConflictDialog(self, path, mode=mode, dark=dark)
            self._cad_conflict_dialog = dlg
            dlg.finished.connect(_finish)
            # Non-blocking open(): nested exec() can swallow Overwrite clicks while the
            # agent thread waits — leaving the run hung with the dialog still visible.
            dlg.open()
            dlg.raise_()
            dlg.activateWindow()
        except Exception:
            self._cad_conflict_dialog = None
            if thread is not None:
                thread.provide_confirm_result(False)

    @Slot(object)
    def _on_agent_result(self, result: object) -> None:
        if not isinstance(result, AgentRunResult):
            return
        if result.error == "internet_permission_required":
            self._handle_internet_permission(result)
            return

        self._append_system_line("Done.")
        self._append_activity("Task completed.")

        # Use the conversation that *submitted* this query, not the currently
        # selected one.  The user may have switched tabs while the query was
        # running, and writing to the active conversation would bleed the
        # response into the wrong tab.
        target_conv_id = self._running_conversation_id or self._active_conversation_id

        if result.session_id:
            # Update the session_id on the conversation that ran the query.
            target_conv = next(
                (c for c in self._state.conversations if c.conversation_id == target_conv_id),
                None,
            )
            if target_conv is not None:
                target_conv.session_id = result.session_id
            # Only update the UI-level session pointer if we're still on the
            # same conversation that ran the task.
            if target_conv_id == self._active_conversation_id:
                self._session_id = result.session_id
            self._state_store.save(self._state)
            self._refresh_conversation_list()
            self._session_settings_label.setText(f"{self._session_id}\nStatus: Ready")

        body = result.response or ""
        result_is_active = (target_conv_id == self._active_conversation_id)
        if not result.success and result.error:
            self._store_conversation_message(
                "assistant",
                f"{body}\n\n(Error: {result.error})",
                error=True,
                conversation_id=target_conv_id,
            )
            if result_is_active:
                self._append_assistant_message(f"{body}\n\n(Error: {result.error})", error=True)
        else:
            self._store_conversation_message(
                "assistant",
                body,
                error=not result.success,
                conversation_id=target_conv_id,
            )
            if result_is_active:
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
        clean_q = self._clean_question_from_enhanced(q) or self._pending_plain_query or q
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
            f"[INTERNET_PERMISSION_GRANTED]\n{clean_q}"
            if box.clickedButton() == allow
            else f"[INTERNET_PERMISSION_DENIED]\n{clean_q}"
        )
        self._append_activity("Permission dialog answered.")
        self._append_system_line("Running (with your choice)…")
        self._run_agent_thread(tagged)

    @Slot(str)
    def _on_agent_failed(self, tb: str) -> None:
        self._append_system_line("Done.")
        self._append_activity("Unexpected GUI/worker error.")
        msg = "An unexpected error occurred in the agent.\n\n" + tb
        target_conv_id = self._running_conversation_id or self._active_conversation_id
        self._store_conversation_message("assistant", msg, error=True, conversation_id=target_conv_id)
        if target_conv_id == self._active_conversation_id:
            self._append_assistant_message(msg, error=True)

    @Slot(str)
    def _on_worker_progress(self, text: str) -> None:
        self._append_activity(text)

    @Slot(str)
    def _on_agent_cancelled(self, text: str) -> None:
        target_conv_id = self._running_conversation_id or self._active_conversation_id
        self._store_conversation_message("system", text, conversation_id=target_conv_id)
        self._persist_cancelled_history(self._pending_plain_query or self._last_query, text)
        self._append_system_line(text)
        self._append_activity(text)
        # Cancelling terminates the warm worker; re-warm it so the next prompt
        # doesn't pay the cold-start cost again.
        QTimer.singleShot(0, self._prewarm_agent)

    @Slot()
    def _on_thread_finished(self) -> None:
        finished_thread = self.sender()
        if finished_thread is not self._thread:
            return
        self._progress_timer.stop()
        self._running_conversation_id = None
        self._send_btn.setEnabled(True)
        self._fallback_cb.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._retry_btn.setEnabled(bool(self._last_query.strip()))
        self._cad_prompt_btn.setEnabled(True)
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
            QMessageBox.information(
                self,
                "Task in progress",
                "Another conversation is currently running. Please wait for it to finish before retrying.",
            )
            return
        self._input.setPlainText(self._last_query)
        self._on_send_clicked()

    @Slot()
    def _new_session(self) -> None:
        conv = self._state_store.new_conversation(self._state)
        self._active_conversation_id = conv.conversation_id
        self._session_id = conv.session_id
        self._refresh_conversation_list()
        self._render_active_conversation()
        self._refresh_send_btn_state()
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
        self._refresh_send_btn_state()
        running = self._thread is not None and self._thread.isRunning()
        status = "Task in progress" if conv.conversation_id == self._running_conversation_id else "Ready"
        self._session_settings_label.setText(f"{self._session_id}\nStatus: {status}")
        self.statusBar().showMessage(f"Switched to conversation: {conv.title}", 3000)

    def _refresh_send_btn_state(self) -> None:
        """Enable or disable the Send/Cancel buttons based on whether the active conversation has a running task."""
        running = self._thread is not None and self._thread.isRunning()
        is_running_conv = running and (self._active_conversation_id == self._running_conversation_id)
        self._send_btn.setEnabled(not is_running_conv)
        self._cancel_btn.setEnabled(is_running_conv)
        self._fallback_cb.setEnabled(not is_running_conv)
        self._cad_prompt_btn.setEnabled(not is_running_conv)
        if not is_running_conv:
            self._retry_btn.setEnabled(bool(self._last_query.strip()))

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
        self._state.preferred_primary_llm = normalize_primary_llm_selection(
            self._primary_llm_combo.currentText()
        )
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
                tiered = bool(getattr(self._settings, "enable_tiered_models", True))
                if tiered:
                    nano = str(getattr(self._settings, "openai_model_nano", "") or "").strip()
                    mini = str(getattr(self._settings, "openai_model_mini", "") or "").strip()
                    complex_m = str(getattr(self._settings, "openai_model_complex", "") or "").strip()
                    models = [x for x in [nano, mini, complex_m] if x]
                    # De-duplicate while preserving order (in case the user sets the same model for multiple tiers)
                    seen: set[str] = set()
                    uniq: list[str] = []
                    for m in models:
                        if m not in seen:
                            uniq.append(m)
                            seen.add(m)
                    if uniq:
                        return f"openai ({', '.join(uniq)})"
                m = str(getattr(self._settings, "openai_model", "") or "").strip()
                return f"openai ({m})" if m else "openai"
            if prov == "gemini":
                m = str(getattr(self._settings, "gemini_model", "") or "").strip()
                return f"gemini ({m})" if m else "gemini"
            if prov == "claude":
                m = str(getattr(self._settings, "claude_model", "") or "").strip()
                return f"claude ({m})" if m else "claude"
            return prov

        primary_txt = _fmt("primary_llm")
        if normalize_primary_llm_selection(self._state.preferred_primary_llm) == AUTO_PRIMARY_LLM:
            # Show routing explicitly so Auto remains visible while agent runs a concrete provider.
            primary_txt = f"auto → {primary_txt}" if primary_txt != "—" else "auto"
        self._active_primary_llm_label.setText(primary_txt)
        self._active_fallback_llm_label.setText(_fmt("fallback_llm"))

    def _cloud_base_and_token(self) -> tuple[str, str]:
        return self._state.cloud_api_base_url.strip(), self._state.cloud_access_token.strip()

    def _default_cloud_api_base_url(self) -> str:
        """Prefill for sign-in: saved URL, then .env, then production default."""
        saved = self._state.cloud_api_base_url.strip()
        if saved:
            return saved
        from_settings = str(getattr(self._settings, "survyai_api_base_url", "") or "").strip()
        if from_settings:
            return from_settings
        return DEFAULT_CLOUD_API_BASE_URL

    def _preflight_cloud_api(self, base: str, *, require_database: bool = True) -> bool:
        """Reachability (+ optional DB) check before cloud-backed actions."""
        try:
            health = cloud_health(base_url=base)
        except CloudApiError as exc:
            QMessageBox.warning(
                self,
                "Cloud API not reachable",
                (
                    f"Could not reach the cloud API at:\n{base}\n\n"
                    f"{user_facing_cloud_message(exc)}\n\n"
                    "• Confirm you are online and try again\n"
                    "• Local API development: start python -m survyai_cloud and set "
                    f"SURVYAI_API_BASE_URL (default {DEFAULT_CLOUD_API_BASE_URL})"
                ),
            )
            return False
        if require_database and "database_ok" in health and not health.get("database_ok"):
            detail = str(health.get("database_detail") or "").strip()
            QMessageBox.warning(
                self,
                "Database not reachable",
                (
                    f"The cloud API at {base} is running, but it cannot connect to its database.\n\n"
                    f"{detail or 'See the cloud terminal for errors.'}\n\n"
                    "• Local dev: docker compose up -d\n"
                    "  DATABASE_URL=postgresql+asyncpg://survyai:survyai@localhost:5432/survyai\n"
                    "• Supabase: confirm the project is active and the host in DATABASE_URL is correct\n\n"
                    "Restart python -m survyai_cloud after fixing .env, then sign in again."
                ),
            )
            return False
        return True

    def _preflight_cloud_billing(self, base: str) -> bool:
        """
        Confirm the desktop can reach the same API the user signed into and that
        Paystack is configured on that server process.
        """
        try:
            health = cloud_health(base_url=base)
        except CloudApiError as exc:
            QMessageBox.warning(
                self,
                "Cloud API not reachable",
                (
                    f"Could not reach the cloud API at:\n{base}\n\n"
                    f"{user_facing_cloud_message(exc)}\n\n"
                    "• Confirm you are online and try again\n"
                    "• Local API development: start python -m survyai_cloud and set "
                    f"SURVYAI_API_BASE_URL (default {DEFAULT_CLOUD_API_BASE_URL})\n"
                    f"• Open {DEFAULT_CLOUD_API_BASE_URL}/health in a browser — database_ok should be true"
                ),
            )
            return False
        if "database_ok" in health and not health.get("database_ok"):
            detail = str(health.get("database_detail") or "").strip()
            QMessageBox.warning(
                self,
                "Database not reachable",
                (
                    f"The cloud API at {base} is running, but it cannot connect to its database.\n\n"
                    f"{detail or 'See the cloud terminal for errors.'}\n\n"
                    "Fix DATABASE_URL in backend server environment variables or .env, restart the backend server or cloud service, then try again."
                ),
            )
            return False
        if "paystack_plans_configured" in health and not health.get("paystack_plans_configured"):
            QMessageBox.warning(
                self,
                "Billing not configured on server",
                (
                    "This cloud API has no Paystack plan codes.\n\n"
                    "On the machine running the backend server or cloud service, add to environment variables in the backend server or .env or .env.cloud:\n"
                    "  PAYSTACK_PLAN_CODE_PRO_DAILY=PLN_…\n"
                    "  PAYSTACK_PLAN_CODE_PRO_WEEKLY=PLN_…\n"
                    "  PAYSTACK_PLAN_CODE_PRO_MONTHLY=PLN_…\n"
                    "  PAYSTACK_PLAN_CODE_PRO_ANNUAL=PLN_…\n\n"
                    "Copy plan Codes from Paystack Dashboard → Plans (not the Naira price).\n"
                    "Restart the cloud API after saving."
                ),
            )
            return False
        if "paystack_secret_configured" in health and not health.get("paystack_secret_configured"):
            QMessageBox.warning(
                self,
                "Paystack not configured on server",
                (
                    "This cloud API is missing PAYSTACK_SECRET_KEY.\n\n"
                    "Add your Paystack test or live secret key to environment variables in the backend server or .env or .env.cloud, "
                    "then restart the backend server or cloud service and try again."
                ),
            )
            return False
        return True

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
        if not self._preflight_cloud_billing(base):
            return
        if not self._ensure_cloud_token_valid(silent=False):
            return
        base, token = self._cloud_base_and_token()
        try:
            plans_payload = get_billing_plans(base_url=base, access_token=token)
        except CloudApiError as exc:
            QMessageBox.warning(
                self,
                "Can't load billing",
                f"{user_facing_cloud_message(exc)}\n\nCloud API: {base}",
            )
            return
        plans = plans_payload.get("plans") if isinstance(plans_payload, dict) else None
        if not isinstance(plans, list) or not plans:
            QMessageBox.warning(
                self,
                "Billing not configured",
                (
                    "The cloud API returned no billing plans.\n\n"
                    "Set PAYSTACK_PLAN_CODE_PRO_DAILY, PAYSTACK_PLAN_CODE_PRO_WEEKLY, "
                    "PAYSTACK_PLAN_CODE_PRO_MONTHLY, and/or PAYSTACK_PLAN_CODE_PRO_ANNUAL "
                    "in environment variables in the backend server or .env.cloud (plan codes PLN_… from Paystack Dashboard), then restart the backend server or cloud service "
                    "and try again."
                ),
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
        self._start_payment_refresh_watch()
        QMessageBox.information(
            self,
            "Complete payment",
            "Finish payment in your browser. SurvyAI will refresh your plan automatically "
            "once Paystack confirms the payment with the server (usually within a few seconds). "
            "If nothing changes after a few minutes, use “Refresh cloud account” or "
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
                    "If you already paid, verify the Paystack reference (or, click on 'Refresh cloud account')."
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
        # Renewals / plan changes in the Paystack portal should auto-refresh Desktop status.
        self._start_payment_refresh_watch()

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
        self._stop_payment_refresh_watch()
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
        if not self._preflight_cloud_api(base):
            return
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
        if self._cloud_network_busy():
            self.statusBar().showMessage("Cloud update already in progress…", 3000)
            return
        base, token = self._cloud_base_and_token()
        if not base or not token:
            QMessageBox.warning(
                self,
                "Sign in required",
                "Sign in from the account menu (top right) first.",
            )
            return

        self._begin_cloud_busy("Refreshing cloud account…")
        thread = CloudAccountSyncThread(self._make_cloud_account_sync_payload(), parent=self)
        self._cloud_account_sync_thread = thread

        def _done() -> None:
            self._end_cloud_busy()
            if self._cloud_account_sync_thread is thread:
                self._cloud_account_sync_thread = None

        def _on_ok(result_obj: object) -> None:
            result = result_obj if isinstance(result_obj, CloudAccountSyncResult) else None
            if result is None:
                QMessageBox.warning(self, "Couldn't refresh account", "Unexpected sync response.")
                return
            ent = result.ent if isinstance(result.ent, dict) else {}
            me = result.me if isinstance(result.me, dict) else {}
            plan = str(ent.get("plan_slug") or me.get("plan_slug") or "")
            st = str(ent.get("subscription_status") or me.get("subscription_status") or "")
            self._apply_cloud_account_sync_result(
                result,
                success_status=f"Cloud refreshed. Plan={plan} Status={st}",
            )

        def _on_fail(msg: str) -> None:
            if self._cloud_sync_message_is_session_expired(msg):
                self._clear_cloud_session()
                self._prompt_session_expired()
                return
            QMessageBox.warning(self, "Couldn't refresh account", msg)

        thread.succeeded.connect(_on_ok)
        thread.failed.connect(_on_fail)
        thread.finished.connect(_done)
        thread.start()

    @staticmethod
    def _cloud_sync_message_is_session_expired(msg: str) -> bool:
        low = (msg or "").lower()
        return (
            "session expired" in low
            or "refresh token" in low
            or "no refresh token" in low
        )

    @Slot()
    def _cloud_sign_in(self) -> None:
        """
        Cloud sign-in: optional register -> login -> store tokens -> /v1/me, entitlements, bootstrap
        -> rebuild agent service with injected platform keys + model tiers.

        The production cloud API base URL is used by default (no user-facing prompt).
        Developers can still override via SURVYAI_API_BASE_URL / saved state.
        """
        base_url = self._default_cloud_api_base_url().strip() or DEFAULT_CLOUD_API_BASE_URL
        if not self._preflight_cloud_api(base_url):
            return
        display_name_for_profile = ""
        company_for_profile = ""
        choice = _CloudAuthChoiceDialog(self, policy_hint=password_policy_hint())
        if choice.exec() != QDialog.DialogCode.Accepted:
            return
        picked = choice.choice()
        if picked == _CloudAuthChoiceDialog.CHOICE_FORGOT:
            self._on_forgot_password(base_url=base_url)
            return
        if picked not in (
            _CloudAuthChoiceDialog.CHOICE_SIGN_IN,
            _CloudAuthChoiceDialog.CHOICE_CREATE,
        ):
            return
        is_register = picked == _CloudAuthChoiceDialog.CHOICE_CREATE

        email, ok = QInputDialog.getText(
            self,
            "Cloud account" if is_register else "Cloud sign-in",
            "Email",
            text=self._state.profile.email.strip(),
        )
        if not ok or not email.strip():
            return
        if is_register:
            pwd_dlg = _NewPasswordDialog(
                self,
                title="Create account",
                email=email.strip(),
            )
            if pwd_dlg.exec() != QDialog.DialogCode.Accepted:
                return
            password = pwd_dlg.password()
        else:
            pwd_dlg = _PasswordPromptDialog(
                self,
                title="Cloud sign-in",
                label="Password",
            )
            if pwd_dlg.exec() != QDialog.DialogCode.Accepted:
                return
            password = pwd_dlg.password()
            if not password:
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

        self._begin_cloud_busy("Signing in…")
        try:
            tokens = login(base_url=base_url, email=email.strip(), password=password)
        except CloudApiError as exc:
            self._end_cloud_busy()
            QMessageBox.warning(self, "Couldn't sign in", user_facing_cloud_message(exc))
            return
        except Exception as exc:
            self._end_cloud_busy()
            QMessageBox.warning(self, "Couldn't sign in", user_facing_cloud_message(exc))
            return

        self._state.cloud_api_base_url = base_url.strip()
        self._state.cloud_access_token = tokens.access_token
        self._state.cloud_refresh_token = tokens.refresh_token
        self._state.cloud_access_token_expires_at = access_token_expires_at_iso(
            expires_in_seconds=tokens.expires_in
        )
        local = _email_local_part(email.strip())
        entered = (display_name_for_profile or "").strip()
        self._state.profile.display_name = entered or local
        self._state.profile.company = (company_for_profile or "").strip()
        self._state.profile.email = email.strip()
        if not self._state.profile.signed_in_at:
            self._state.profile.signed_in_at = datetime.now(timezone.utc).isoformat()
        self._state_store.save(self._state)

        payload = CloudAccountSyncPayload(
            base_url=base_url.strip(),
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            access_token_expires_at=self._state.cloud_access_token_expires_at,
            device_id="",
            device_fingerprint="",
            machine_label=(os.environ.get("COMPUTERNAME") or "").strip() or None,
        )
        thread = CloudAccountSyncThread(payload, parent=self)
        self._cloud_account_sync_thread = thread

        def _done() -> None:
            self._end_cloud_busy()
            if self._cloud_account_sync_thread is thread:
                self._cloud_account_sync_thread = None

        def _on_ok(result_obj: object) -> None:
            result = result_obj if isinstance(result_obj, CloudAccountSyncResult) else None
            if result is None:
                QMessageBox.warning(self, "Couldn't sign in", "Unexpected sync response.")
                return
            from_me = str((result.me or {}).get("display_name") or "").strip()
            self._state.profile.display_name = entered or local or from_me
            ent = result.ent if isinstance(result.ent, dict) else {}
            me = result.me if isinstance(result.me, dict) else {}
            plan = str(ent.get("plan_slug") or me.get("plan_slug") or "")
            status = str(ent.get("subscription_status") or me.get("subscription_status") or "")
            self._apply_cloud_account_sync_result(
                result,
                success_status=f"Cloud connected. Plan={plan} Status={status}",
            )
            if result.bootstrap_status == "skipped_no_device":
                self.statusBar().showMessage(
                    "Signed in. This PC is not registered for hosted Pro keys (device limit or error).",
                    9000,
                )
            elif result.bootstrap_status == "failed_pro":
                self.statusBar().showMessage(
                    "Signed in. Hosted keys unavailable — confirm Pro subscription and PC registration.",
                    8000,
                )

        def _on_fail(msg: str) -> None:
            if self._cloud_sync_message_is_session_expired(msg):
                self._clear_cloud_session()
                self._prompt_session_expired()
                return
            QMessageBox.warning(self, "Couldn't sign in", msg)

        thread.succeeded.connect(_on_ok)
        thread.failed.connect(_on_fail)
        thread.finished.connect(_done)
        thread.start()

    @Slot()
    def _sign_out_account(self) -> None:
        if not self._state.profile.is_signed_in and not self._state.cloud_access_token.strip():
            return
        answer = QMessageBox.question(
            self,
            "Sign out",
            "Do you wish to sign out?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        base = self._state.cloud_api_base_url.strip()
        refresh = self._state.cloud_refresh_token.strip()
        if base and refresh:
            try:
                cloud_logout(base_url=base, refresh_token=refresh)
            except Exception:
                pass
        self._state.profile = AccountProfile()
        self._clear_cloud_session()
        self._refresh_account_views()
        self._refresh_diagnostics()

    def _on_forgot_password(self, *, base_url: str | None = None) -> None:
        """Request a one-time email code, then set a new password."""
        base = (
            (base_url or "").strip()
            or self._default_cloud_api_base_url().strip()
            or DEFAULT_CLOUD_API_BASE_URL
        )
        if not self._preflight_cloud_api(base):
            return

        email, ok = QInputDialog.getText(
            self,
            "Forgot password",
            "Account email",
            text=self._state.profile.email.strip(),
        )
        if not ok or not email.strip():
            return
        try:
            cloud_forgot_password(base_url=base, email=email.strip())
        except CloudApiError as exc:
            QMessageBox.warning(self, "Forgot password", user_facing_cloud_message(exc))
            return
        except Exception as exc:
            QMessageBox.warning(self, "Forgot password", user_facing_cloud_message(exc))
            return

        QMessageBox.information(
            self,
            "Check your email",
            "If an account exists for that email, a one-time reset code was sent. "
            "Enter the code and choose a new password next.",
        )
        code, ok = QInputDialog.getText(self, "Reset password", "Reset code from email")
        if not ok or not code.strip():
            return
        pwd_dlg = _NewPasswordDialog(
            self,
            title="Reset password",
            email=email.strip(),
        )
        if pwd_dlg.exec() != QDialog.DialogCode.Accepted:
            return
        new_password = pwd_dlg.password()
        try:
            cloud_reset_password(
                base_url=base,
                email=email.strip(),
                code=code.strip(),
                new_password=new_password,
            )
        except CloudApiError as exc:
            QMessageBox.warning(self, "Reset password failed", user_facing_cloud_message(exc))
            return
        except Exception as exc:
            QMessageBox.warning(self, "Reset password failed", user_facing_cloud_message(exc))
            return
        QMessageBox.information(
            self,
            "Password updated",
            "Your password was changed. Sign in with your new password.",
        )

    @Slot()
    def _on_change_password(self) -> None:
        base, token = self._cloud_base_and_token()
        if not base or not token:
            QMessageBox.warning(
                self,
                "Sign in required",
                "Sign in from the account menu (top right) before changing your password.",
            )
            return
        if not self._ensure_cloud_token_valid():
            return
        base, token = self._cloud_base_and_token()
        email = (
            str((self._state.cloud_me or {}).get("email") or "").strip()
            or self._state.profile.email.strip()
        )
        current_dlg = _PasswordPromptDialog(
            self,
            title="Change password",
            label="Current password",
        )
        if current_dlg.exec() != QDialog.DialogCode.Accepted:
            return
        current = current_dlg.password()
        if not current:
            return
        pwd_dlg = _NewPasswordDialog(
            self,
            title="Change password",
            email=email or None,
        )
        if pwd_dlg.exec() != QDialog.DialogCode.Accepted:
            return
        new_password = pwd_dlg.password()
        try:
            tokens = cloud_change_password(
                base_url=base,
                access_token=token,
                current_password=current,
                new_password=new_password,
            )
        except CloudApiError as exc:
            QMessageBox.warning(self, "Change password failed", user_facing_cloud_message(exc))
            return
        except Exception as exc:
            QMessageBox.warning(self, "Change password failed", user_facing_cloud_message(exc))
            return
        if tokens.access_token:
            self._state.cloud_access_token = tokens.access_token
        if tokens.refresh_token:
            self._state.cloud_refresh_token = tokens.refresh_token
        if tokens.expires_in:
            self._state.cloud_access_token_expires_at = access_token_expires_at_iso(
                expires_in_seconds=tokens.expires_in
            )
        self._state_store.save(self._state)
        self.statusBar().showMessage("Password updated.", 5000)
        QMessageBox.information(
            self,
            "Password updated",
            "Your password was changed. Other signed-in devices were signed out.",
        )

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

    def _diagnostic_redaction_secrets(self) -> list[str]:
        secrets: list[str] = []
        for value in (
            self._state.cloud_access_token,
            self._state.cloud_refresh_token,
            self._state.cloud_access_token_expires_at,
        ):
            value = str(value or "").strip()
            if value:
                secrets.append(value)

        def _walk(payload: object, parent_key: str = "") -> None:
            if isinstance(payload, dict):
                for key, value in payload.items():
                    _walk(value, str(key or ""))
                return
            if isinstance(payload, list):
                for item in payload:
                    _walk(item, parent_key)
                return
            if not isinstance(payload, str):
                return
            probe = parent_key.lower()
            if any(marker in probe for marker in ("token", "secret", "key", "password")):
                candidate = payload.strip()
                if candidate:
                    secrets.append(candidate)

        _walk(self._state.cloud_bootstrap or {})
        return sorted(set(secrets), key=len, reverse=True)

    def _redact_text_for_diagnostics(self, text: str) -> str:
        redacted = text or ""
        for secret in self._diagnostic_redaction_secrets():
            redacted = redacted.replace(secret, "[REDACTED]")
        redacted = re.sub(
            r"\beyJ[A-Za-z0-9_\-]+?\.[A-Za-z0-9_\-]+?\.[A-Za-z0-9_\-]+?\b",
            "[REDACTED_JWT]",
            redacted,
        )
        return redacted

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
                zf.writestr("diagnostics.txt", self._redact_text_for_diagnostics(self._diagnostics_text.toPlainText()))
                zf.writestr("transcript.txt", self._redact_text_for_diagnostics(self._transcript.toPlainText()))
                zf.writestr("activity_log.txt", self._redact_text_for_diagnostics(self._activity_log.toPlainText()))
                zf.writestr(
                    "history.json",
                    self._redact_text_for_diagnostics(
                        json.dumps([entry.__dict__ for entry in self._state.output_history], indent=2, ensure_ascii=True)
                    ),
                )
                zf.writestr(
                    "desktop_state_snapshot.json",
                    json.dumps(
                        self._state_store.diagnostics_snapshot(self._state),
                        indent=2,
                        ensure_ascii=True,
                    ),
                )
                zf.writestr(
                    "desktop_state_redacted.json",
                    json.dumps(
                        self._state_store.exportable_state_snapshot(self._state),
                        indent=2,
                        ensure_ascii=True,
                    ),
                )
                if log_path.is_file():
                    zf.writestr(
                        log_path.name,
                        self._redact_text_for_diagnostics(log_path.read_text(encoding="utf-8", errors="replace")),
                    )
            self.statusBar().showMessage(f"Diagnostics exported: {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Diagnostics export failed", str(e))

    @Slot()
    def _open_log_folder(self) -> None:
        folder = Path(self._settings.log_file).parent
        if folder.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _cloud_api_base_for_updates(self) -> str:
        return (
            self._state.cloud_api_base_url
            or getattr(self._settings, "survyai_api_base_url", "")
            or DEFAULT_CLOUD_API_BASE_URL
            or ""
        ).strip()

    def _update_channel(self) -> str:
        return str(getattr(self._state, "update_channel", "") or "stable").strip() or "stable"

    def _refresh_updates_status_label(self) -> None:
        label = getattr(self, "_updates_status_label", None)
        if label is None:
            return
        if not bool(getattr(self._state, "auto_check_updates", False)):
            label.setText("Automatic checks are off. Use “Check for updates now…” anytime.")
            return
        last = str(getattr(self._state, "last_update_check_at", "") or "").strip()
        if not last:
            label.setText("Automatic checks are on. SurvyAI has not checked yet in this session.")
            return
        try:
            dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
            label.setText(f"Automatic checks are on. Last checked: {dt.astimezone().strftime('%Y-%m-%d %H:%M')}.")
        except Exception:
            label.setText("Automatic checks are on.")

    def _sync_update_check_timer(self) -> None:
        if bool(getattr(self._state, "auto_check_updates", False)):
            if not self._update_check_timer.isActive():
                self._update_check_timer.start()
        else:
            self._update_check_timer.stop()

    @Slot(bool)
    def _on_auto_check_updates_toggled(self, checked: bool) -> None:
        self._state.auto_check_updates = bool(checked)
        self._state_store.save(self._state)
        self._sync_update_check_timer()
        self._refresh_updates_status_label()
        self.statusBar().showMessage(
            "Automatic update checks enabled." if checked else "Automatic update checks disabled.",
            3000,
        )
        if checked:
            QTimer.singleShot(400, self._maybe_auto_check_updates)

    def _mark_update_check_completed(self) -> None:
        self._state.last_update_check_at = datetime.now(timezone.utc).isoformat()
        self._state_store.save(self._state)
        self._refresh_updates_status_label()

    @Slot()
    def _maybe_auto_check_updates(self) -> None:
        if not bool(getattr(self._state, "auto_check_updates", False)):
            return
        if not update_check_due(
            str(getattr(self._state, "last_update_check_at", "") or ""),
            interval_hours=UPDATE_CHECK_INTERVAL_HOURS,
        ):
            return
        self._start_update_check(interactive=False)

    @Slot()
    def _check_for_updates(self) -> None:
        self._start_update_check(interactive=True)

    def _start_update_check(self, *, interactive: bool) -> None:
        if self._update_check_thread is not None and self._update_check_thread.isRunning():
            if interactive:
                self.statusBar().showMessage("Update check already in progress…", 3000)
            return
        base = self._cloud_api_base_for_updates()
        if not base:
            if interactive:
                QMessageBox.information(
                    self,
                    "Updates unavailable",
                    "No SurvyAI cloud API base URL is configured for update checks yet.",
                )
            return
        self._update_check_interactive = interactive
        if interactive:
            self.statusBar().showMessage("Checking for updates…", 4000)
        thread = UpdateCheckThread(
            base_url=base,
            channel=self._update_channel(),
            platform="windows-x64",
            current_version=__version__,
            parent=self,
        )
        thread.update_available.connect(self._on_update_check_available)
        thread.up_to_date.connect(self._on_update_check_up_to_date)
        thread.failed.connect(self._on_update_check_failed)
        thread.finished.connect(self._on_update_check_finished)
        self._update_check_thread = thread
        thread.start()

    @Slot()
    def _on_update_check_finished(self) -> None:
        self._update_check_thread = None

    @Slot(object)
    def _on_update_check_up_to_date(self, manifest: object) -> None:
        self._mark_update_check_completed()
        channel = getattr(manifest, "channel", self._update_channel())
        if self._update_check_interactive:
            QMessageBox.information(
                self,
                "Up to date",
                f"SurvyAI {__version__} is the latest available build on the {channel} channel.",
            )
        else:
            self.statusBar().showMessage(f"SurvyAI {__version__} is up to date.", 4000)

    @Slot(str)
    def _on_update_check_failed(self, message: str) -> None:
        # Still record the attempt so a flaky network does not spam every few seconds.
        self._mark_update_check_completed()
        if self._update_check_interactive:
            QMessageBox.warning(self, "Update check failed", message)
        else:
            self.statusBar().showMessage("Update check failed (will retry later).", 5000)

    @Slot(object)
    def _on_update_check_available(self, manifest: object) -> None:
        self._mark_update_check_completed()
        if not isinstance(manifest, UpdateManifest):
            return
        if self._update_prompt_open:
            return
        dismissed = str(getattr(self._state, "dismissed_update_version", "") or "").strip()
        required = manifest.is_required_for(__version__)
        if (
            not self._update_check_interactive
            and not required
            and dismissed == manifest.latest_version
        ):
            self.statusBar().showMessage(
                f"Update {manifest.latest_version} available (reminded later).",
                5000,
            )
            return
        self._prompt_update_available(manifest, interactive=self._update_check_interactive)

    def _prompt_update_available(self, manifest: UpdateManifest, *, interactive: bool) -> None:
        extra_lines = [
            f"Current version: {__version__}",
            f"Available version: {manifest.latest_version}",
            f"Channel: {manifest.channel}",
            f"Package type: {manifest.artifact_kind}",
        ]
        if manifest.release_notes_url:
            extra_lines.append(f"Release notes: {manifest.release_notes_url}")
        if manifest.requires_upgrade_from(__version__):
            extra_lines.append(
                "This update raises the minimum supported version, so a full installer upgrade is required."
            )
        if manifest.mandatory:
            extra_lines.append("This is a mandatory update.")
        if not manifest.download_url or not manifest.sha256:
            QMessageBox.information(
                self,
                "Update available",
                "\n".join(extra_lines + ["No downloadable installer is attached to this manifest yet."]),
            )
            return

        required = manifest.is_required_for(__version__)
        self._update_prompt_open = True
        try:
            if required:
                answer = QMessageBox.warning(
                    self,
                    "Required update",
                    "\n".join(
                        extra_lines
                        + [
                            "",
                            "A required SurvyAI update is available.",
                            "Download and install it now to continue with a supported build?",
                        ]
                    ),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if answer != QMessageBox.Yes:
                    if interactive:
                        return
                    # Soft-block non-interactive mandatory prompts: remind next interval.
                    return
            else:
                answer = QMessageBox.question(
                    self,
                    "Update available",
                    "\n".join(
                        extra_lines
                        + [
                            "",
                            "Download and install this update now?",
                            "Choose No to be reminded later.",
                        ]
                    ),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if answer != QMessageBox.Yes:
                    self._state.dismissed_update_version = manifest.latest_version
                    self._state_store.save(self._state)
                    self.statusBar().showMessage(
                        f"Update {manifest.latest_version} postponed.",
                        4000,
                    )
                    return
        finally:
            self._update_prompt_open = False

        self._download_and_launch_update(manifest)

    def _download_and_launch_update(self, manifest: UpdateManifest) -> None:
        progress = QProgressDialog(
            f"Downloading SurvyAI {manifest.latest_version}…",
            "Cancel",
            0,
            0,
            self,
        )
        progress.setWindowTitle("Downloading update")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        cancelled = {"flag": False}

        def _on_cancel() -> None:
            cancelled["flag"] = True

        progress.canceled.connect(_on_cancel)

        def _progress(downloaded: int, total: Optional[int]) -> None:
            if cancelled["flag"]:
                raise RuntimeError("Update download cancelled.")
            if total and total > 0:
                progress.setMaximum(total)
                progress.setValue(min(downloaded, total))
            else:
                progress.setMaximum(0)
                progress.setValue(0)
            QApplication.processEvents()

        try:
            manager = UpdateManager()
            staged_path = manager.stage_update(
                manifest,
                current_version=__version__,
                current_executable=sys.executable,
                progress_callback=_progress,
            )
        except Exception as exc:
            progress.close()
            if "cancelled" in str(exc).lower():
                self.statusBar().showMessage("Update download cancelled.", 4000)
                return
            QMessageBox.critical(self, "Update download failed", str(exc))
            return
        progress.close()

        self._state.dismissed_update_version = ""
        self._state_store.save(self._state)

        launch = QMessageBox.question(
            self,
            "Installer ready",
            "The verified installer was downloaded successfully.\n\n"
            f"Path: {staged_path}\n\n"
            "SurvyAI will quit so the installer can replace the installed files.\n"
            "Launch the installer now?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if launch != QMessageBox.Yes:
            self.statusBar().showMessage(f"Installer saved: {staged_path}", 8000)
            return
        try:
            launch_staged_installer(Path(staged_path))
        except Exception as exc:
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(staged_path)))
            except Exception:
                QMessageBox.warning(
                    self,
                    "Installer not launched",
                    f"The installer is ready but could not be opened automatically:\n\n{exc}",
                )
                return
        app = QApplication.instance()
        if app is not None:
            QTimer.singleShot(350, app.quit)

    def _getting_started_path(self) -> Path | None:
        for candidate in (
            resource_path("docs", "GETTING_STARTED.md"),
            resource_path("GETTING_STARTED.md"),
        ):
            if candidate.is_file():
                return candidate
        return None

    @Slot()
    def _open_getting_started_guide(self) -> None:
        self._show_getting_started_guide(first_run=False)

    @Slot()
    def _maybe_show_getting_started_first_run(self) -> None:
        if self._state.getting_started_seen:
            return
        self._show_getting_started_guide(first_run=True)

    def _show_getting_started_guide(self, *, first_run: bool) -> None:
        path = self._getting_started_path()
        if path is None:
            if first_run:
                # Do not block forever on missing bundle; mark seen so startup continues.
                self._state.getting_started_seen = True
                self._state_store.save(self._state)
            QMessageBox.information(
                self,
                "Getting started",
                "Getting Started guide was not found in this install. Use Help → Documentation instead.",
            )
            return
        dlg = MarkdownHelpDialog(
            self,
            title="Getting started with SurvyAI",
            subtitle="Learn the basics: workspace, CAD plans, PDF-to-CAD, ArcGIS, and everyday tips.",
            markdown_path=path,
            primary_label="Got it" if first_run else "Close",
            show_dont_show_again=first_run,
        )
        dlg.exec()
        # First-run: always mark seen after the dialog closes so startup is not stuck in a loop.
        # Checkbox defaults to "don't show again"; unchecking still completes this launch's gate.
        if first_run:
            self._state.getting_started_seen = True
            self._state_store.save(self._state)

    @Slot()
    def _open_readme_docs(self) -> None:
        readme = resource_path("README.md")
        if readme.is_file():
            self._show_markdown_dialog(
                readme,
                "SurvyAI Documentation",
                subtitle="Product overview, billing, privacy, and support — shown inside SurvyAI.",
            )
        else:
            QMessageBox.information(self, "Documentation", "README.md was not found.")

    def _show_markdown_dialog(
        self,
        markdown_path: Path,
        title: str,
        *,
        subtitle: str = "",
    ) -> None:
        """Show local markdown help inside the app instead of delegating to the OS."""
        if not markdown_path.is_file():
            QMessageBox.critical(self, title, f"Could not open documentation:\n\nFile not found:\n{markdown_path}")
            return
        dlg = MarkdownHelpDialog(
            self,
            title=title,
            subtitle=subtitle
            or "In-app help so documentation works consistently without relying on external apps.",
            markdown_path=markdown_path,
            primary_label="Close",
            show_dont_show_again=False,
        )
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
        try:
            shutdown_shared_agent_process()
        except Exception:
            pass
        event.accept()
