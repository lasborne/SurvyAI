"""
Onboarding and account dialogs for the desktop GUI.

These are intentionally backend-agnostic for now:
- Users can "sign in" locally by entering identity details.
- Environment validation reflects the current `.env`/env-based configuration.
- A future backend can replace or augment the same screens without changing the
  rest of the main window contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWizard,
    QWizardPage,
    QWidget,
)

from config import Settings
from survyai.capabilities import MachineCapabilities, format_capabilities_summary
from survyai.gui.state import AccountProfile


def environment_validation_report(settings: Settings) -> str:
    """Human-readable validation summary for onboarding/settings/diagnostics."""
    lines = []
    if settings.openai_api_key.strip():
        lines.append("OpenAI API key: configured")
    else:
        lines.append("OpenAI API key: missing")
    if settings.google_api_key.strip():
        lines.append("Google Gemini API key: configured")
    else:
        lines.append("Google Gemini API key: missing")
    if settings.anthropic_api_key.strip():
        lines.append("Anthropic API key: configured")
    else:
        lines.append("Anthropic API key: missing")
    if settings.deepseek_api_key.strip():
        lines.append("DeepSeek API key: configured")
    else:
        lines.append("DeepSeek API key: missing")

    any_key = any(
        [
            settings.openai_api_key.strip(),
            settings.google_api_key.strip(),
            settings.anthropic_api_key.strip(),
            settings.deepseek_api_key.strip(),
        ]
    )
    lines.append("")
    lines.append("Primary LLM: " + str(settings.primary_llm))
    lines.append("Fallback LLM: " + str(settings.fallback_llm))
    lines.append("Vector store: " + ("enabled" if settings.vector_store_enabled else "disabled"))
    lines.append("Overall readiness: " + ("ready" if any_key else "needs at least one API key"))
    return "\n".join(lines)


class AccountDialog(QDialog):
    """Simple local sign-in/profile editor."""

    def __init__(self, parent=None, profile: Optional[AccountProfile] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Account sign-in")
        self.resize(420, 220)
        self._profile = profile or AccountProfile()

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Enter the identity details that should appear in the desktop app.\n"
            "This is local desktop sign-in scaffolding until the cloud backend is added."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self.name_edit = QLineEdit(self._profile.display_name)
        self.email_edit = QLineEdit(self._profile.email)
        self.company_edit = QLineEdit(self._profile.company)
        form.addRow("Name", self.name_edit)
        form.addRow("Email", self.email_edit)
        form.addRow("Company", self.company_edit)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.setObjectName("secondaryButton")
        cancel.clicked.connect(self.reject)
        save = QPushButton("Save")
        save.clicked.connect(self._accept_if_valid)
        buttons.addStretch()
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        layout.addLayout(buttons)

    def _accept_if_valid(self) -> None:
        if not (self.name_edit.text().strip() or self.email_edit.text().strip()):
            QMessageBox.warning(self, "Missing details", "Enter at least a name or email.")
            return
        self.accept()

    def profile(self) -> AccountProfile:
        return AccountProfile(
            display_name=self.name_edit.text().strip(),
            email=self.email_edit.text().strip(),
            company=self.company_edit.text().strip(),
        )


class _SignInPage(QWizardPage):
    def __init__(self, initial_profile: Optional[AccountProfile] = None) -> None:
        super().__init__()
        self.setTitle("Account sign-in")
        self.setSubTitle("Set up the identity shown inside the SurvyAI desktop app.")

        initial_profile = initial_profile or AccountProfile()
        layout = QFormLayout(self)
        self.name_edit = QLineEdit(initial_profile.display_name)
        self.email_edit = QLineEdit(initial_profile.email)
        self.company_edit = QLineEdit(initial_profile.company)
        self.registerField("profile_name*", self.name_edit)
        self.registerField("profile_email", self.email_edit)
        self.registerField("profile_company", self.company_edit)
        layout.addRow("Name", self.name_edit)
        layout.addRow("Email", self.email_edit)
        layout.addRow("Company", self.company_edit)


class _EnvironmentPage(QWizardPage):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.setTitle("Environment validation")
        self.setSubTitle("Check whether the local AI/runtime configuration is ready.")
        layout = QVBoxLayout(self)
        info = QTextEdit()
        info.setReadOnly(True)
        info.setPlainText(environment_validation_report(settings))
        layout.addWidget(info)


class _CapabilityPage(QWizardPage):
    def __init__(self, capabilities: MachineCapabilities) -> None:
        super().__init__()
        self.setTitle("AutoCAD and machine detection")
        self.setSubTitle("SurvyAI checks for CAD/GIS integrations before first use.")
        layout = QVBoxLayout(self)
        info = QTextEdit()
        info.setReadOnly(True)
        info.setPlainText(format_capabilities_summary(capabilities))
        layout.addWidget(info)


class _DataFolderPage(QWizardPage):
    def __init__(self, initial_path: str) -> None:
        super().__init__()
        self.setTitle("Data folder")
        self.setSubTitle("Choose where the desktop app stores logs, vector data, and exports.")

        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.path_edit = QLineEdit(initial_path)
        browse = QPushButton("Browse…")
        browse.setObjectName("secondaryButton")
        browse.clicked.connect(self._browse)
        row.addWidget(self.path_edit, 1)
        row.addWidget(browse)
        layout.addLayout(row)
        hint = QLabel(
            "Recommended: a stable folder under Documents or AppData. "
            "You can change this later from Settings."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.registerField("data_folder*", self.path_edit)

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose data folder", self.path_edit.text().strip())
        if folder:
            self.path_edit.setText(folder)


class _TutorialPage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("First-run tutorial")
        self.setSubTitle("What to expect from the desktop app.")
        layout = QVBoxLayout(self)
        info = QLabel(
            "1. Choose a workspace folder before generating files.\n"
            "2. Use the console tab for normal prompting.\n"
            "3. Output history stores previous runs for reuse.\n"
            "4. Safe mode disables external integrations when troubleshooting.\n"
            "5. Diagnostics export creates a support bundle with logs and environment details."
        )
        info.setWordWrap(True)
        layout.addWidget(info)


class OnboardingWizard(QWizard):
    def __init__(
        self,
        *,
        settings: Settings,
        capabilities: MachineCapabilities,
        initial_profile: Optional[AccountProfile],
        initial_data_folder: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Welcome to SurvyAI")
        self.setWizardStyle(QWizard.ModernStyle)
        self.addPage(_SignInPage(initial_profile))
        self.addPage(_EnvironmentPage(settings))
        self.addPage(_CapabilityPage(capabilities))
        self.addPage(_DataFolderPage(initial_data_folder))
        self.addPage(_TutorialPage())

    def profile(self) -> AccountProfile:
        return AccountProfile(
            display_name=str(self.field("profile_name")).strip(),
            email=str(self.field("profile_email")).strip(),
            company=str(self.field("profile_company")).strip(),
        )

    def data_folder(self) -> str:
        return str(self.field("data_folder")).strip()


__all__ = [
    "AccountDialog",
    "OnboardingWizard",
    "environment_validation_report",
]
