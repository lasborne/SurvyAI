from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from survyai_cloud.config import get_cloud_settings
from survyai_cloud.schemas import UpdateManifestOut

router = APIRouter(prefix="/updates", tags=["updates"])


@router.get("/manifest", response_model=UpdateManifestOut)
async def update_manifest(
    channel: str | None = None,
    current: str | None = None,
    platform: str | None = None,
) -> UpdateManifestOut:
    settings = get_cloud_settings()
    ch = (channel or settings.updates_default_channel or "stable").strip()
    target_platform = (platform or "windows-x64").strip() or "windows-x64"
    path_str = (settings.updates_manifest_path or "").strip()
    if path_str and Path(path_str).is_file():
        try:
            raw = json.loads(Path(path_str).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            raise HTTPException(status_code=500, detail=f"Invalid manifest file: {e}") from e
        if isinstance(raw, dict) and isinstance(raw.get(ch), dict):
            raw = raw.get(ch)
        if not isinstance(raw, dict):
            raise HTTPException(status_code=500, detail="Invalid manifest structure")
        raw.setdefault("channel", ch)
        raw.setdefault("platform", target_platform)
        raw.setdefault("artifact_kind", "full-installer")
        return UpdateManifestOut.model_validate(raw)

    return UpdateManifestOut(
        channel=ch,
        platform=target_platform,
        latest_version=(current or "0.0.0").strip() or "0.0.0",
        artifact_kind="full-installer",
        mandatory=False,
    )
