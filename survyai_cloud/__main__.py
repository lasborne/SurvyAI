"""
Run: ``python -m survyai_cloud`` from the SurvyAI repo root (or set PYTHONPATH).
"""

from __future__ import annotations

import os

import uvicorn

if __name__ == "__main__":
    host = os.environ.get("SURVYAI_CLOUD_HOST", "127.0.0.1")
    port = int(os.environ.get("SURVYAI_CLOUD_PORT", "8088"))
    uvicorn.run(
        "survyai_cloud.main:app",
        host=host,
        port=port,
        reload=os.environ.get("SURVYAI_CLOUD_RELOAD", "").lower() in ("1", "true", "yes"),
    )
