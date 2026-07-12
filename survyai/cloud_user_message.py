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

    if "current password is incorrect" in low:
        return "Your current password is incorrect. Please try again."

    if "invalid or expired reset code" in low:
        return (
            "That reset code is invalid or has expired. "
            "Request a new code with Forgot password, then try again."
        )

    if "new password must be different" in low:
        return "Choose a new password that is different from your current password."

    if any(
        x in low
        for x in (
            "password must be at least",
            "password must include",
            "password must not contain",
            "too common",
            "at most 128 characters",
        )
    ):
        return raw if raw else (
            "That password does not meet SurvyAI's requirements. "
            "Use at least 10 characters with upper and lower case, a digit, and a special character."
        )

    if "missing cloud api base url" in low:
        return "Enter your cloud API address when signing in."

    if any(
        x in low
        for x in (
            "getaddrinfo failed",
            "name or service not known",
            "nodename nor servname",
            "temporary failure in name resolution",
        )
    ):
        return (
            "The cloud API is running, but its database hostname could not be resolved. "
            "Check DATABASE_URL in .env (Supabase project active? correct host?). "
            "For local development, use Docker Postgres: docker compose up -d, then set "
            "DATABASE_URL=postgresql+asyncpg://survyai:survyai@localhost:5432/survyai and restart "
            "python -m survyai_cloud."
        )

    if any(
        x in low
        for x in (
            "database unavailable",
            "database connection failed",
            "database_ok",
            "could not connect to server",
            "connection does not exist",
            "password authentication failed",
            "invalidpassworderror",
            "asyncpg",
            "sqlalchemy",
        )
    ):
        return (
            "The cloud API process is up, but it cannot reach the database. "
            "Open http://127.0.0.1:8088/health — if database_ok is false, fix DATABASE_URL in .env "
            "and restart python -m survyai_cloud. Local dev: docker compose up -d then use "
            "postgresql+asyncpg://survyai:survyai@localhost:5432/survyai."
        )

    if any(
        x in low
        for x in (
            "network error",
            "connection refused",
            "failed to establish a new connection",
            "readtimeout",
            "connecttimeout",
        )
    ) or ("timed out" in low or "timeout" in low):
        """return (
            "The cloud API did not respond in time. If http://127.0.0.1:8088/health opens but "
            "database_ok is false, the server is waiting on a database that is down or unreachable. "
            "Fix DATABASE_URL, restart python -m survyai_cloud, then try again."
        )"""
        return (
            "The cloud API did not respond in time. Database_ok is false, the server is waiting on a database that is down or unreachable."
            "Try again."
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
            "undefinedcolumn",
            "does not exist",
            "column users.",
            "schema",
            "programmingerror",
        )
    ):
        return (
            "The cloud database schema is out of date. In the project folder run:\n"
            "  python -m alembic upgrade head\n"
            "then restart python -m survyai_cloud and sign in again."
        )

    if "database unavailable" in low:
        return (
            "The cloud API is running but the database is not ready. "
            "Open http://127.0.0.1:8088/health — if database_ok is false, fix DATABASE_URL in .env "
            "and restart python -m survyai_cloud."
        )

    if any(
        x in low
        for x in (
            "internal server error",
            "bad gateway",
            "service unavailable",
        )
    ) or (
        any(code in raw for code in ("502", "504"))
        or ("500" in raw and "http 500" in low)
        or ("503" in raw and "database" not in low)
    ):
        return (
            "The cloud API returned a server error. Restart python -m survyai_cloud, run "
            "python -m alembic upgrade head if you recently updated the app, then try signing in again."
        )

    if "429" in raw or "too many requests" in low:
        return "Too many requests. Please wait a minute and try again."

    if "invalid amount" in low:
        return (
            "Payment could not be started. Check your subscription plan settings, then try again "
            "or contact support."
        )

    if "no paystack plans configured" in low or "paystack_plan_code" in low:
        return (
            "Billing plans are not set up on the cloud server. Add PAYSTACK_PLAN_CODE_PRO_DAILY, "
            "PAYSTACK_PLAN_CODE_PRO_WEEKLY, PAYSTACK_PLAN_CODE_PRO_MONTHLY, "
            "and/or PAYSTACK_PLAN_CODE_PRO_ANNUAL (PLN_… codes from Paystack) to .env.cloud, "
            "then restart python -m survyai_cloud."
        )

    if "paystack is not configured" in low or "missing paystack_secret_key" in low:
        return (
            "Paystack is not configured on the cloud server. Add PAYSTACK_SECRET_KEY to .env.cloud "
            "and restart python -m survyai_cloud."
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
