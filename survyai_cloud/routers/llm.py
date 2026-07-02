from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from survyai_cloud.config import get_cloud_settings
from survyai_cloud.db import get_db
from survyai_cloud.deps import device_from_header
from survyai_cloud.models import Device, User
from survyai_cloud.rate_limiting import rate_limit_user_dependency
from survyai_cloud.schemas import LlmProxyChatIn, LlmProxyChatOut
from survyai_cloud.services.llm_proxy import run_proxy_chat

router = APIRouter(prefix="/llm", tags=["llm"])


@router.post("/chat", response_model=LlmProxyChatOut)
async def proxy_chat(
    body: LlmProxyChatIn,
    user: Annotated[
        User,
        Depends(rate_limit_user_dependency("llm_proxy_chat", "rate_limit_usage_events_per_window")),
    ],
    db: Annotated[AsyncSession, Depends(get_db)],
    device: Annotated[Device | None, Depends(device_from_header)],
) -> LlmProxyChatOut:
    return await run_proxy_chat(
        body=body,
        user=user,
        device=device,
        db=db,
        settings=get_cloud_settings(),
    )
