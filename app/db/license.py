from datetime import datetime, timezone

from app.exceptions import (
    ExpiredLicenseError,
    InvalidLicenseKeyError,
    LicenseServiceUnavailableError,
    SessionAlreadyActiveError,
)
from app.infra.dependencies import supabase

# Postgres unique_violation. sessions.key is the primary key, so a duplicate insert means
# another connection claimed this licence in the window between validate_license's session
# check and our insert — a race, not an outage.
UNIQUE_VIOLATION = "23505"


def is_duplicate_session(exc: Exception) -> bool:
    """True when a sessions insert failed because the row already exists."""
    return getattr(exc, "code", None) == UNIQUE_VIOLATION


def validate_license(key: str) -> None:
    try:
        response = supabase.table("licenses").select("active, expires_at").eq("key", key).execute()
    except Exception as e:
        raise LicenseServiceUnavailableError() from e

    if not response.data:
        raise InvalidLicenseKeyError()

    license = response.data[0]

    if not license["active"]:
        raise InvalidLicenseKeyError()

    expires_at = license.get("expires_at")
    if expires_at:
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        if expiry < datetime.now(timezone.utc):
            raise ExpiredLicenseError()

    try:
        session = supabase.table("sessions").select("key").eq("key", key).limit(1).execute()
    except Exception as e:
        raise LicenseServiceUnavailableError() from e

    if session.data:
        raise SessionAlreadyActiveError()


def clear_stale_sessions() -> None:
    supabase.table("sessions").delete().neq("key", "").execute()
