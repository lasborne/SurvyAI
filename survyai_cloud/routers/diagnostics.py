from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from survyai_cloud.config import get_cloud_settings
from survyai_cloud.db import get_db
from survyai_cloud.deps import get_current_user
from survyai_cloud.models import DiagnosticsBundle, User
from survyai_cloud.schemas import DiagnosticsOut

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])

_MAX_BYTES = 25 * 1024 * 1024


@router.post("/upload", response_model=DiagnosticsOut)
async def upload_diagnostics(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    file: Annotated[UploadFile, File(description="Support bundle (zip/log)")],
    client_version: Annotated[str | None, Form()] = None,
    notes: Annotated[str | None, Form()] = None,
) -> DiagnosticsBundle:
    settings = get_cloud_settings()
    base = Path(settings.diagnostics_storage_dir)
    base.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename or "upload.bin").name
    bundle_id = uuid.uuid4()
    dest = base / f"{user.id}_{bundle_id}_{safe_name}"

    total = 0
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File too large (max {_MAX_BYTES} bytes)",
                    )
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise

    row = DiagnosticsBundle(
        id=bundle_id,
        user_id=user.id,
        client_version=(client_version or "")[:64] or None,
        filename=safe_name,
        byte_size=total,
        notes=notes,
    )
    db.add(row)
    await db.flush()
    await db.refresh(row)
    return row
