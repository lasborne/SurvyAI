"""Session-expiry copy must trigger auto sign-out on the desktop."""

from survyai.cloud_user_message import (
    SESSION_EXPIRED_USER_MESSAGE,
    is_session_expired_cloud_message,
    user_facing_cloud_message,
)


def test_screenshot_session_copy_is_detected():
    msg = (
        "Your session has expired or is invalid. Please sign in again from "
        "Settings (Cloud sign-in)."
    )
    assert msg == SESSION_EXPIRED_USER_MESSAGE
    assert is_session_expired_cloud_message(msg) is True


def test_old_matcher_miss_session_has_expired():
    # The previous GUI check looked for the substring "session expired",
    # which does not appear in "session has expired".
    assert "session expired" not in SESSION_EXPIRED_USER_MESSAGE.lower()
    assert is_session_expired_cloud_message(SESSION_EXPIRED_USER_MESSAGE) is True


def test_unauthorized_maps_to_session_expired_copy():
    mapped = user_facing_cloud_message("401 Unauthorized: invalid token")
    assert mapped == SESSION_EXPIRED_USER_MESSAGE
    assert is_session_expired_cloud_message(mapped) is True


def test_timeout_is_not_session_expiry():
    mapped = user_facing_cloud_message("ReadTimeout")
    assert is_session_expired_cloud_message(mapped) is False


def test_network_cloudapi_error_is_not_session_expiry():
    mapped = user_facing_cloud_message(
        "Network error (ConnectTimeout): unable to reach the cloud server."
    )
    assert is_session_expired_cloud_message(mapped) is False


def test_server_error_is_not_session_expiry():
    mapped = user_facing_cloud_message("HTTP 500 Internal Server Error")
    assert is_session_expired_cloud_message(mapped) is False


def test_forbidden_device_copy_is_not_session_expiry():
    mapped = user_facing_cloud_message("403 Forbidden")
    assert "sign in again" in mapped.lower()
    assert is_session_expired_cloud_message(mapped) is False


def test_raw_invalid_token_phrase_alone_does_not_match_after_unrelated_copy():
    assert is_session_expired_cloud_message("invalid token") is False
    assert is_session_expired_cloud_message("Cloud session expired (no refresh token).") is False
