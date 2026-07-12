"""Transactional email via Resend (password reset)."""

from __future__ import annotations

import logging
from typing import Optional

import requests

from survyai_cloud.config import CloudSettings, get_cloud_settings

logger = logging.getLogger(__name__)


def send_password_reset_email(
    *,
    to_email: str,
    code: str,
    expires_minutes: int,
    settings: Optional[CloudSettings] = None,
) -> bool:
    """
    Send a one-time password-reset code.

    Returns True if the provider accepted the message (or a non-production
    fallback logged the code). Failures are logged; callers must not leak them.
    """
    settings = settings or get_cloud_settings()
    to_email = (to_email or "").strip().lower()
    code = (code or "").strip()
    ttl = max(1, int(expires_minutes or settings.password_reset_ttl_minutes))
    if not to_email or not code:
        return False

    subject = "SurvyAI password reset code"
    plain = (
        f"Your SurvyAI password reset code is: {code}\n\n"
        f"This code expires in {ttl} minutes.\n\n"
        "Enter the code in the SurvyAI desktop app to choose a new password.\n\n"
        "If you did not request this, you can ignore this email.\n"
    )
    html = (
        "<p>Your SurvyAI password reset code is:</p>"
        f"<p style='font-size:22px;font-weight:700;letter-spacing:2px'>{code}</p>"
        f"<p>This code expires in <strong>{ttl} minutes</strong>.</p>"
        "<p>Enter the code in the SurvyAI desktop app to choose a new password.</p>"
        "<p>If you did not request this, you can ignore this email.</p>"
    )

    api_key = (settings.resend_api_key or "").strip()
    if not api_key:
        if settings.deployment_env == "production":
            logger.error("RESEND_API_KEY is not configured; cannot send password reset email")
            return False
        logger.warning(
            "RESEND_API_KEY unset (%s): password reset code for %s is %s (expires %s min)",
            settings.deployment_env,
            to_email,
            code,
            ttl,
        )
        return True

    from_addr = (settings.email_from or "").strip() or "SurvyAI <noreply@survyai.app>"
    try:
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "from": from_addr,
                "to": [to_email],
                "subject": subject,
                "text": plain,
                "html": html,
            },
            timeout=20,
        )
        if resp.status_code >= 400:
            logger.error(
                "Resend password-reset email failed HTTP %s: %s",
                resp.status_code,
                (resp.text or "")[:400],
            )
            return False
        return True
    except Exception:
        logger.exception("Resend password-reset email request failed")
        return False
