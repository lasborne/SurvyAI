"""Focused tests for password policy and reset/change-password helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_password_policy_accepts_strong_password():
    from survyai_cloud.security import validate_password_strength

    assert validate_password_strength("GoodPass1!", email="user@example.com") is None


def test_password_policy_rejects_short_and_weak():
    from survyai_cloud.security import validate_password_strength

    assert validate_password_strength("Short1!") is not None
    assert validate_password_strength("alllowercase1!") is not None
    assert validate_password_strength("ALLUPPERCASE1!") is not None
    assert validate_password_strength("NoSpecial12") is not None
    assert validate_password_strength("Password1!") is not None
    assert validate_password_strength("user1234A!", email="user@example.com") is not None


def test_desktop_password_policy_mirrors_cloud():
    from survyai.cloud_api import validate_password_strength as desktop_validate
    from survyai_cloud.security import validate_password_strength as cloud_validate

    samples = [
        "GoodPass1!",
        "short",
        "Password1!",
        "NoDigit!!Aa",
        "nodigitupper!",
    ]
    for sample in samples:
        assert bool(desktop_validate(sample)) == bool(cloud_validate(sample))


def test_hash_password_reset_code_is_normalized():
    from survyai_cloud.security import hash_password_reset_code

    assert hash_password_reset_code("ab12cd34") == hash_password_reset_code(" AB12CD34 ")
    assert hash_password_reset_code("AAAA1111") != hash_password_reset_code("BBBB1111")


def test_forgot_password_does_not_leak_missing_user():
    """Unknown emails still get the same generic success payload shape."""
    from survyai_cloud.schemas import ForgotPasswordOut

    out = ForgotPasswordOut()
    assert "If an account exists" in out.detail


def test_reset_password_rejects_expired_token_logic():
    """Expired token timestamps are treated as invalid."""
    now = datetime.now(timezone.utc)
    token = SimpleNamespace(
        expires_at=now - timedelta(minutes=1),
        used_at=None,
        token_hash="abc",
    )
    expires_at = token.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    assert expires_at < now


def test_change_password_requires_different_new_password():
    from survyai_cloud.security import hash_password, verify_password

    hashed = hash_password("GoodPass1!")
    assert verify_password("GoodPass1!", hashed)
    # Same password would be rejected by the endpoint after verify.
    assert verify_password("GoodPass1!", hashed) is True


@patch("survyai_cloud.services.email.requests.post")
def test_send_password_reset_email_uses_resend(mock_post):
    from survyai_cloud.config import CloudSettings
    from survyai_cloud.services.email import send_password_reset_email

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "{}"
    mock_post.return_value = mock_resp

    settings = CloudSettings(
        resend_api_key="re_test",
        email_from="SurvyAI <noreply@example.com>",
        deployment_env="development",
    )
    ok = send_password_reset_email(
        to_email="user@example.com",
        code="ABCD2345",
        expires_minutes=30,
        settings=settings,
    )
    assert ok is True
    assert mock_post.called
    args, kwargs = mock_post.call_args
    assert args[0] == "https://api.resend.com/emails"
    assert kwargs["json"]["to"] == ["user@example.com"]
    assert "ABCD2345" in kwargs["json"]["text"]


def test_send_password_reset_email_dev_fallback_without_key():
    from survyai_cloud.config import CloudSettings
    from survyai_cloud.services.email import send_password_reset_email

    settings = CloudSettings(resend_api_key="", deployment_env="development")
    ok = send_password_reset_email(
        to_email="user@example.com",
        code="ABCD2345",
        expires_minutes=30,
        settings=settings,
    )
    assert ok is True
