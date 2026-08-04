"""
Persistent desktop state for the packaged Windows GUI.

This is intentionally separate from `config.Settings`:
- `Settings` holds agent/runtime configuration (.env + env vars).
- `DesktopState` holds GUI/product state (onboarding, workspace, account profile,
  safe mode, task history, etc.).

The desktop product needs stable, user-local state even before a backend exists.
That lets us ship a professional GUI now and later swap the login/license source
from "local placeholder" to "real backend" with minimal UI churn.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from runtime_paths import default_documents_folder
from survyai.device_identity import compute_machine_fingerprint
from survyai.gui.secret_store import DesktopSecretStore


DEFAULT_CLOUD_API_BASE_URL = "https://survyai-api.onrender.com"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_app_dir() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "SurvyAI"
    return Path.home() / ".survyai"


@dataclass
class AccountProfile:
    display_name: str = ""
    email: str = ""
    company: str = ""
    signed_in_at: str = ""

    @property
    def is_signed_in(self) -> bool:
        return bool(self.display_name.strip() or self.email.strip())


@dataclass
class TaskHistoryEntry:
    run_id: str
    created_at: str
    workspace_path: str
    session_id: str
    query: str
    response: str
    success: bool
    error: str = ""
    llm_used: str = ""
    model_name: str = ""
    llm_cost_usd: float = 0.0
    cancelled: bool = False


@dataclass
class ConversationMessage:
    role: str
    content: str
    created_at: str
    error: bool = False


@dataclass
class Conversation:
    conversation_id: str
    session_id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[ConversationMessage] = field(default_factory=list)


@dataclass
class DesktopState:
    onboarding_complete: bool = False
    # First-run Getting Started guide (Help → Getting started). Separate from onboarding wizard.
    getting_started_seen: bool = False
    profile: AccountProfile = field(default_factory=AccountProfile)
    cloud_api_base_url: str = DEFAULT_CLOUD_API_BASE_URL
    cloud_access_token: str = ""
    cloud_refresh_token: str = ""
    cloud_access_token_expires_at: str = ""
    cloud_me: Dict[str, Any] = field(default_factory=dict)
    cloud_bootstrap: Dict[str, Any] = field(default_factory=dict)
    # Bound PC for Pro: server device row id + fingerprint used when registering (see POST /v1/devices).
    cloud_device_id: str = ""
    cloud_device_fingerprint: str = ""
    workspace_path: str = ""
    data_folder: str = ""
    safe_mode: bool = False
    use_fallback_llm: bool = False
    # Symbolic primary selection ("auto" = best paid hosted model; persists across restarts).
    preferred_primary_llm: str = "auto"
    preferred_fallback_llm: str = ""
    # Ollama runtime settings for offline/local models.
    # Stored here (not in .env) so end-users don't edit config files.
    ollama_base_url: str = ""
    ollama_model: str = ""
    # Local models (Ollama) UX state
    ollama_last_prompted_at: str = ""
    ollama_prompt_dismissed: bool = False
    # Performance toggles (desktop-only UX; injected into Settings overrides)
    fast_mode_non_file_prompts: bool = True
    # UI theme: "light" (default) or "dark"
    theme: str = "light"
    # App updates (opt-in). When enabled, SurvyAI checks the cloud manifest
    # on a 12-hour cadence and notifies when a newer installer is available.
    auto_check_updates: bool = False
    update_channel: str = "stable"
    last_update_check_at: str = ""
    dismissed_update_version: str = ""
    # Credits accounting (synced from cloud entitlements / usage)
    monthly_credits_usd: float = 0.0
    monthly_credits_used_usd: float = 0.0
    credit_markup_multiplier: float = 2.0
    # Pro platform keys (server decides); used client-side to gate paid LLM when credits hit zero.
    can_use_platform_llm: bool = False
    credits_billing_interval: str = ""  # "daily" | "weekly" | "monthly" | "annual" from cloud
    # Exact paid usage window from cloud (preferred over deriving start from period_end - days).
    usage_period_anchor: str = ""
    subscription_current_period_end: str = ""
    # Soft credit-usage reminders (console strip); reset when budget/period changes (see main_window).
    credit_banner_anchor_budget_usd: float = -1.0
    credit_banner_anchor_used_usd: float = -1.0
    credit_banner_dismissed_half: bool = False
    credit_banner_dismissed_eighty: bool = False
    credit_banner_dismissed_ninetyfive: bool = False
    output_history: List[TaskHistoryEntry] = field(default_factory=list)
    conversations: List[Conversation] = field(default_factory=list)
    active_conversation_id: str = ""
    # User-accepted default CAD survey-plan prompt. Empty string = use packaged system default.
    default_cad_prompt: str = ""

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "DesktopState":
        profile = raw.get("profile") or {}
        hist = raw.get("output_history") or []
        convs = raw.get("conversations") or []
        return cls(
            onboarding_complete=bool(raw.get("onboarding_complete", False)),
            getting_started_seen=bool(raw.get("getting_started_seen", False)),
            profile=AccountProfile(
                display_name=str(profile.get("display_name", "")),
                email=str(profile.get("email", "")),
                company=str(profile.get("company", "")),
                signed_in_at=str(profile.get("signed_in_at", "")),
            ),
            cloud_api_base_url=str(raw.get("cloud_api_base_url", "")),
            cloud_access_token=str(raw.get("cloud_access_token", "")),
            cloud_refresh_token=str(raw.get("cloud_refresh_token", "")),
            cloud_access_token_expires_at=str(raw.get("cloud_access_token_expires_at", "")),
            cloud_me=raw.get("cloud_me") if isinstance(raw.get("cloud_me"), dict) else {},
            cloud_bootstrap=raw.get("cloud_bootstrap") if isinstance(raw.get("cloud_bootstrap"), dict) else {},
            cloud_device_id=str(raw.get("cloud_device_id", "")),
            cloud_device_fingerprint=str(raw.get("cloud_device_fingerprint", "")),
            workspace_path=str(raw.get("workspace_path", "")),
            data_folder=str(raw.get("data_folder", "")),
            safe_mode=bool(raw.get("safe_mode", False)),
            use_fallback_llm=bool(raw.get("use_fallback_llm", False)),
            # Missing or blank → product default "auto" (resolves to best paid provider).
            preferred_primary_llm=(
                str(raw.get("preferred_primary_llm") or "auto").strip().lower() or "auto"
            ),
            preferred_fallback_llm=str(raw.get("preferred_fallback_llm", "")),
            ollama_base_url=str(raw.get("ollama_base_url", "")),
            ollama_model=str(raw.get("ollama_model", "")),
            ollama_last_prompted_at=str(raw.get("ollama_last_prompted_at", "")),
            ollama_prompt_dismissed=bool(raw.get("ollama_prompt_dismissed", False)),
            fast_mode_non_file_prompts=bool(raw.get("fast_mode_non_file_prompts", True)),
            theme=str(raw.get("theme", "light") or "light"),
            auto_check_updates=bool(raw.get("auto_check_updates", False)),
            update_channel=str(raw.get("update_channel", "stable") or "stable"),
            last_update_check_at=str(raw.get("last_update_check_at", "") or ""),
            dismissed_update_version=str(raw.get("dismissed_update_version", "") or ""),
            monthly_credits_usd=float(raw.get("monthly_credits_usd", 0.0)),
            monthly_credits_used_usd=float(raw.get("monthly_credits_used_usd", 0.0)),
            credit_markup_multiplier=float(raw.get("credit_markup_multiplier", 2.0)),
            can_use_platform_llm=bool(raw.get("can_use_platform_llm", False)),
            credits_billing_interval=str(raw.get("credits_billing_interval", "") or ""),
            usage_period_anchor=str(raw.get("usage_period_anchor", "") or ""),
            subscription_current_period_end=str(raw.get("subscription_current_period_end", "") or ""),
            credit_banner_anchor_budget_usd=float(raw.get("credit_banner_anchor_budget_usd", -1.0)),
            credit_banner_anchor_used_usd=float(raw.get("credit_banner_anchor_used_usd", -1.0)),
            credit_banner_dismissed_half=bool(raw.get("credit_banner_dismissed_half", False)),
            credit_banner_dismissed_eighty=bool(raw.get("credit_banner_dismissed_eighty", False)),
            credit_banner_dismissed_ninetyfive=bool(raw.get("credit_banner_dismissed_ninetyfive", False)),
            output_history=[
                TaskHistoryEntry(
                    run_id=str(item.get("run_id", "")),
                    created_at=str(item.get("created_at", "")),
                    workspace_path=str(item.get("workspace_path", "")),
                    session_id=str(item.get("session_id", "")),
                    query=str(item.get("query", "")),
                    response=str(item.get("response", "")),
                    success=bool(item.get("success", False)),
                    error=str(item.get("error", "")),
                    llm_used=str(item.get("llm_used", "")),
                    model_name=str(item.get("model_name", "")),
                    llm_cost_usd=float(item.get("llm_cost_usd", 0.0) or 0.0),
                    cancelled=bool(item.get("cancelled", False)),
                )
                for item in hist
                if isinstance(item, dict)
            ],
            conversations=[
                Conversation(
                    conversation_id=str(item.get("conversation_id", "")),
                    session_id=str(item.get("session_id", "")),
                    title=str(item.get("title", "") or "New conversation"),
                    created_at=str(item.get("created_at", "")),
                    updated_at=str(item.get("updated_at", "")),
                    messages=[
                        ConversationMessage(
                            role=str(msg.get("role", "assistant")),
                            content=str(msg.get("content", "")),
                            created_at=str(msg.get("created_at", "")),
                            error=bool(msg.get("error", False)),
                        )
                        for msg in (item.get("messages") or [])
                        if isinstance(msg, dict)
                    ],
                )
                for item in convs
                if isinstance(item, dict)
            ],
            active_conversation_id=str(raw.get("active_conversation_id", "")),
            default_cad_prompt=str(raw.get("default_cad_prompt", "") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AppStateStore:
    """Read/write desktop state under the user's roaming app-data directory."""

    def __init__(self, app_dir: Optional[Path] = None) -> None:
        self.app_dir = Path(app_dir) if app_dir is not None else _default_app_dir()
        self.app_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.app_dir / "desktop_state.json"
        self.secret_store = DesktopSecretStore(self.app_dir)
        self.default_data_dir = self.app_dir / "data"
        self.default_data_dir.mkdir(parents=True, exist_ok=True)

    def default_workspace_path(self) -> str:
        return str(default_documents_folder())

    def load(self) -> DesktopState:
        raw_data: Dict[str, Any] = {}
        if not self.state_path.is_file():
            state = DesktopState(
                cloud_api_base_url=DEFAULT_CLOUD_API_BASE_URL,
                workspace_path=self.default_workspace_path(),
                data_folder=str(self.default_data_dir),
            )
            self.save(state)
            return state
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            raw_data = raw if isinstance(raw, dict) else {}
            state = DesktopState.from_dict(raw_data)
        except Exception:
            state = DesktopState()
        self._hydrate_secrets(state, raw_data)
        if not state.cloud_api_base_url.strip():
            state.cloud_api_base_url = DEFAULT_CLOUD_API_BASE_URL
        if not state.workspace_path:
            state.workspace_path = self.default_workspace_path()
        if not state.data_folder:
            state.data_folder = str(self.default_data_dir)
        Path(state.data_folder).mkdir(parents=True, exist_ok=True)
        self.ensure_conversations(state)
        if self._legacy_sensitive_fields_present(raw_data):
            self.save(state)
        return state

    def _legacy_sensitive_fields_present(self, raw_state: Dict[str, Any]) -> bool:
        return any(
            [
                str(raw_state.get("cloud_access_token") or "").strip(),
                str(raw_state.get("cloud_refresh_token") or "").strip(),
                str(raw_state.get("cloud_access_token_expires_at") or "").strip(),
                isinstance(raw_state.get("cloud_bootstrap"), dict)
                and bool(raw_state.get("cloud_bootstrap")),
            ]
        )

    def save(self, state: DesktopState) -> None:
        if state.data_folder:
            Path(state.data_folder).mkdir(parents=True, exist_ok=True)
        self.secret_store.save(self._secret_payload(state))
        self.state_path.write_text(
            json.dumps(self._public_state_dict(state), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def _public_state_dict(self, state: DesktopState) -> Dict[str, Any]:
        data = asdict(state)
        data["cloud_access_token"] = ""
        data["cloud_refresh_token"] = ""
        data["cloud_access_token_expires_at"] = ""
        data["cloud_bootstrap"] = {}
        return data

    def _secret_payload(self, state: DesktopState) -> Dict[str, Any]:
        return {
            "cloud_access_token": state.cloud_access_token.strip(),
            "cloud_refresh_token": state.cloud_refresh_token.strip(),
            "cloud_access_token_expires_at": state.cloud_access_token_expires_at.strip(),
            "cloud_bootstrap": state.cloud_bootstrap if isinstance(state.cloud_bootstrap, dict) else {},
        }

    def _hydrate_secrets(self, state: DesktopState, raw_state: Dict[str, Any]) -> None:
        secrets = self.secret_store.load()
        state.cloud_access_token = str(
            secrets.get("cloud_access_token")
            or raw_state.get("cloud_access_token")
            or state.cloud_access_token
            or ""
        )
        state.cloud_refresh_token = str(
            secrets.get("cloud_refresh_token")
            or raw_state.get("cloud_refresh_token")
            or state.cloud_refresh_token
            or ""
        )
        state.cloud_access_token_expires_at = str(
            secrets.get("cloud_access_token_expires_at")
            or raw_state.get("cloud_access_token_expires_at")
            or state.cloud_access_token_expires_at
            or ""
        )
        cloud_bootstrap = secrets.get("cloud_bootstrap")
        if not isinstance(cloud_bootstrap, dict):
            cloud_bootstrap = raw_state.get("cloud_bootstrap")
        state.cloud_bootstrap = cloud_bootstrap if isinstance(cloud_bootstrap, dict) else {}

    def exportable_state_snapshot(self, state: DesktopState) -> Dict[str, Any]:
        data = self._public_state_dict(state)
        data["profile"] = {
            "display_name": state.profile.display_name,
            "email": state.profile.email,
            "company": state.profile.company,
            "signed_in_at": state.profile.signed_in_at,
        }
        return data

    def ensure_conversations(self, state: DesktopState) -> Conversation:
        if state.conversations:
            active = self.get_active_conversation(state)
            if active is not None:
                return active
        conv = Conversation(
            conversation_id=f"conv-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            session_id=f"session-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            title="New conversation",
            created_at=_utc_now(),
            updated_at=_utc_now(),
            messages=[],
        )
        state.conversations = [conv]
        state.active_conversation_id = conv.conversation_id
        self.save(state)
        return conv

    def get_active_conversation(self, state: DesktopState) -> Optional[Conversation]:
        if not state.conversations:
            return None
        target_id = state.active_conversation_id
        if target_id:
            for conv in state.conversations:
                if conv.conversation_id == target_id:
                    return conv
        state.active_conversation_id = state.conversations[0].conversation_id
        return state.conversations[0]

    def auto_title(self, text: str) -> str:
        raw = " ".join((text or "").split()).strip()
        if not raw:
            return "New conversation"
        limit = 48
        return raw if len(raw) <= limit else raw[: limit - 3].rstrip() + "..."

    def new_conversation(self, state: DesktopState, *, title: str = "New conversation") -> Conversation:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        conv = Conversation(
            conversation_id=f"conv-{stamp}",
            session_id=f"session-{stamp}",
            title=title,
            created_at=_utc_now(),
            updated_at=_utc_now(),
            messages=[],
        )
        state.conversations.insert(0, conv)
        state.active_conversation_id = conv.conversation_id
        self.save(state)
        return conv

    def set_active_conversation(
        self,
        state: DesktopState,
        conversation_id: str,
        *,
        persist: bool = True,
    ) -> Optional[Conversation]:
        for conv in state.conversations:
            if conv.conversation_id == conversation_id:
                state.active_conversation_id = conversation_id
                if persist:
                    self.save(state)
                return conv
        return None

    def delete_conversation(self, state: DesktopState, conversation_id: str) -> Conversation:
        remaining = [c for c in state.conversations if c.conversation_id != conversation_id]
        state.conversations = remaining
        active = self.get_active_conversation(state)
        if active is None:
            active = self.ensure_conversations(state)
        else:
            state.active_conversation_id = active.conversation_id
            self.save(state)
        return active

    def append_conversation_message(
        self,
        state: DesktopState,
        *,
        conversation_id: str,
        role: str,
        content: str,
        error: bool = False,
    ) -> Optional[Conversation]:
        for idx, conv in enumerate(state.conversations):
            if conv.conversation_id != conversation_id:
                continue
            conv.messages.append(
                ConversationMessage(
                    role=role,
                    content=content,
                    created_at=_utc_now(),
                    error=error,
                )
            )
            conv.updated_at = _utc_now()
            if conv.title == "New conversation" and role == "user" and content.strip():
                conv.title = self.auto_title(content)
            if idx != 0:
                state.conversations.pop(idx)
                state.conversations.insert(0, conv)
            state.active_conversation_id = conv.conversation_id
            self.save(state)
            return conv
        return None

    def add_history_entry(
        self,
        state: DesktopState,
        *,
        entry: TaskHistoryEntry,
        max_entries: int = 100,
    ) -> DesktopState:
        state.output_history.insert(0, entry)
        state.output_history = state.output_history[:max_entries]
        self.save(state)
        return state

    def diagnostics_snapshot(self, state: DesktopState) -> Dict[str, Any]:
        return {
            "generated_at": _utc_now(),
            "workspace_path": state.workspace_path,
            "data_folder": state.data_folder,
            "safe_mode": state.safe_mode,
            "onboarding_complete": state.onboarding_complete,
            "getting_started_seen": state.getting_started_seen,
            "profile": {
                "display_name": state.profile.display_name,
                "email": state.profile.email,
                "company": state.profile.company,
                "signed_in_at": state.profile.signed_in_at,
            },
            "cloud_connected": bool(state.cloud_api_base_url.strip() and state.cloud_access_token.strip()),
            "cloud_plan": (state.cloud_me or {}).get("plan_slug"),
            "cloud_subscription_status": (state.cloud_me or {}).get("subscription_status"),
            "cloud_device_registered": bool(str(state.cloud_device_id or "").strip()),
            "monthly_credits_usd": state.monthly_credits_usd,
            "monthly_credits_used_usd": state.monthly_credits_used_usd,
            "can_use_platform_llm": state.can_use_platform_llm,
            "credits_billing_interval": state.credits_billing_interval,
            "usage_period_anchor": state.usage_period_anchor,
            "subscription_current_period_end": state.subscription_current_period_end,
            "secret_storage_enabled": bool(self.secret_store.secret_path.exists() or os.name == "nt"),
            "secret_storage_path": str(self.secret_store.secret_path),
            "machine_fingerprint_sha256": compute_machine_fingerprint(),
            "auto_check_updates": bool(state.auto_check_updates),
            "update_channel": state.update_channel,
            "last_update_check_at": state.last_update_check_at,
            "history_count": len(state.output_history),
            "conversation_count": len(state.conversations),
            "active_conversation_id": state.active_conversation_id,
        }


__all__ = [
    "AccountProfile",
    "AppStateStore",
    "Conversation",
    "ConversationMessage",
    "DEFAULT_CLOUD_API_BASE_URL",
    "DesktopState",
    "TaskHistoryEntry",
]
