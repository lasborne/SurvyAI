"""
End-user friendly copy for cloud API failures (desktop GUI).

Keep technical details out of message boxes; users are not operators of the API.
"""

from __future__ import annotations


def user_facing_cloud_message(exc: BaseException | str) -> str:
    raw = str(exc).strip() if isinstance(exc, BaseException) else str(exc).strip()
    low = raw.lower()

    if "invalid email or password" in low:
        return "That email or password is incorrect. Please try again."

    if "email already registered" in low:
        return "An account with this email already exists. Try signing in instead."

    if "missing cloud api base url" in low:
        return "Enter your cloud API address when signing in."

    if any(
        x in low
        for x in (
            "network error",
            "connection refused",
            "failed to establish a new connection",
            "name or service not known",
            "getaddrinfo failed",
        )
    ) or ("timed out" in low or "timeout" in low):
        return (
            "Could not reach the SurvyAI cloud server. Check your internet connection, "
            "confirm the API address, and try again."
        )

    if any(
        x in low
        for x in (
            "unauthorized",
            "invalid refresh token",
            "refresh token expired",
            "not authenticated",
            "invalid token",
            "could not validate credentials",
        )
    ) or ("401" in raw and "token refresh" in low):
        return (
            "Your session has expired or is invalid. Please sign in again from "
            "Settings (Cloud sign-in)."
        )

    if "device limit reached" in low or "active pcs for your plan" in low:
        return (
            "Your Pro plan is already active on the maximum number of PCs. "
            "Open your cloud subscription management page to remove an old device, "
            "or contact support if you replaced a computer."
        )

    if "device not found" in low or "removed already" in low:
        return "That PC is no longer on your account. Reload the list or use Refresh cloud account."

    if "this pc must be registered" in low or "x-survyai-device-id" in low:
        return (
            "This computer must be registered to your account before hosted AI keys load. "
            "Use Refresh cloud account from Settings, or sign out and sign in again."
        )

    if "forbidden" in low or "403" in raw:
        return (
            "Access was denied for this request. Please sign in again from Settings (Cloud sign-in)."
        )

    if any(
        x in low
        for x in (
            "500",
            "502",
            "503",
            "504",
            "internal server error",
            "bad gateway",
            "service unavailable",
        )
    ):
        return (
            "The cloud service is temporarily unavailable. Please try again in a few minutes. "
            "If the problem continues, sign in again from Settings (Cloud sign-in)."
        )

    if "429" in raw or "too many requests" in low:
        return "Too many requests. Please wait a minute and try again."

    if "invalid amount" in low:
        return (
            "Payment could not be started. Check your subscription plan settings, then try again "
            "or contact support."
        )

    if "expected json" in low or "empty response" in low:
        return (
            "Could not reach the cloud service or the server returned an unexpected response. "
            "Check that the cloud API is running and the address is correct, then try signing in again."
        )

    return (
        "Something went wrong with the cloud connection. Please try again, or sign in again "
        "from Settings (Cloud sign-in)."
    )
