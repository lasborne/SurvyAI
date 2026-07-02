from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from agent.prompts import SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT
from runtime_paths import prefer_user_data_path, resource_path
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AgentRuntimeConfig:
    version: str = "builtin"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    primary_llm: Optional[str] = None
    fallback_llm: Optional[str] = None
    openai_model: Optional[str] = None
    openai_model_nano: Optional[str] = None
    openai_model_mini: Optional[str] = None
    openai_model_complex: Optional[str] = None
    enable_tiered_models: Optional[bool] = None
    gemini_model: Optional[str] = None
    claude_model: Optional[str] = None
    deepseek_base_url: Optional[str] = None
    agent_temperature: Optional[float] = None
    agent_max_tokens: Optional[int] = None
    source: str = "builtin"

    def to_settings_overrides(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in (
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
        ):
            value = getattr(self, key)
            if isinstance(value, str):
                if value.strip():
                    out[key] = value.strip()
            elif value is not None:
                out[key] = value
        return out

    def to_payload_dict(self) -> dict[str, Any]:
        payload = self.to_settings_overrides()
        payload["version"] = self.version
        payload["system_prompt"] = self.system_prompt
        return payload


def default_agent_config_path() -> Path:
    return prefer_user_data_path("agent", "agent_config.json")


def _resolve_config_path(config_path: str = "") -> Path:
    raw = str(config_path or "").strip()
    if raw:
        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        user_candidate = prefer_user_data_path(*Path(raw).parts)
        if user_candidate.exists():
            return user_candidate.resolve()
        bundled_candidate = resource_path(*Path(raw).parts)
        if bundled_candidate.exists():
            return bundled_candidate.resolve()
        return (Path.cwd() / candidate).resolve()
    return default_agent_config_path()


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logger.warning("Agent runtime config load failed for %s: %s", path, exc)
    return {}


def _parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
    return None


def _parse_float(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except Exception:
        return None


def _parse_int(value: Any) -> Optional[int]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(value)
    except Exception:
        return None


def _resolve_system_prompt(raw: dict[str, Any], *, base_dir: Path) -> str:
    prompt = str(raw.get("system_prompt") or "").strip()
    if prompt:
        return prompt

    prompt_file = str(raw.get("system_prompt_file") or "").strip()
    if prompt_file:
        p = Path(prompt_file)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        try:
            text = p.read_text(encoding="utf-8").strip()
            if text:
                return text
        except Exception as exc:
            logger.warning("Agent system prompt file load failed for %s: %s", p, exc)

    return DEFAULT_SYSTEM_PROMPT


def _normalize_source_label(local_exists: bool, cloud_exists: bool) -> str:
    if cloud_exists and local_exists:
        return "cloud+local-fallback"
    if cloud_exists:
        return "cloud"
    if local_exists:
        return "local-file"
    return "builtin"


def _merged_raw_config(local_raw: dict[str, Any], cloud_raw: dict[str, Any]) -> dict[str, Any]:
    merged = dict(local_raw or {})
    for key, value in (cloud_raw or {}).items():
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                merged[key] = value.strip()
            continue
        merged[key] = value
    return merged


def _cloud_raw_config(cloud_config_json: str = "") -> dict[str, Any]:
    raw = str(cloud_config_json or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception as exc:
        logger.warning("Cloud agent runtime config JSON parse failed: %s", exc)
        return {}
    return data if isinstance(data, dict) else {}


def resolve_agent_runtime_config(
    *,
    local_config_path: str = "",
    cloud_config_json: str = "",
) -> AgentRuntimeConfig:
    local_path = _resolve_config_path(local_config_path)
    local_raw = _load_json_file(local_path)
    cloud_raw = _cloud_raw_config(cloud_config_json)
    merged = _merged_raw_config(local_raw, cloud_raw)
    base_dir = local_path.parent if local_path.parent.exists() else resource_path("agent")

    cfg = AgentRuntimeConfig(
        version=str(merged.get("version") or cloud_raw.get("version") or local_raw.get("version") or "builtin"),
        system_prompt=_resolve_system_prompt(merged, base_dir=base_dir),
        primary_llm=str(merged.get("primary_llm") or "").strip() or None,
        fallback_llm=str(merged.get("fallback_llm") or "").strip() or None,
        openai_model=str(merged.get("openai_model") or "").strip() or None,
        openai_model_nano=str(merged.get("openai_model_nano") or "").strip() or None,
        openai_model_mini=str(merged.get("openai_model_mini") or "").strip() or None,
        openai_model_complex=str(merged.get("openai_model_complex") or "").strip() or None,
        enable_tiered_models=_parse_bool(merged.get("enable_tiered_models")),
        gemini_model=str(merged.get("gemini_model") or "").strip() or None,
        claude_model=str(merged.get("claude_model") or "").strip() or None,
        deepseek_base_url=str(merged.get("deepseek_base_url") or "").strip() or None,
        agent_temperature=_parse_float(merged.get("agent_temperature")),
        agent_max_tokens=_parse_int(merged.get("agent_max_tokens")),
        source=_normalize_source_label(bool(local_raw), bool(cloud_raw)),
    )
    return cfg


__all__ = [
    "AgentRuntimeConfig",
    "default_agent_config_path",
    "resolve_agent_runtime_config",
]
