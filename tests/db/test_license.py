from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from app.db import license as license_mod
from app.db.license import clear_stale_sessions, validate_license
from app.exceptions import (
    ExpiredLicenseError,
    InvalidLicenseKeyError,
    SessionAlreadyActiveError,
)


class FakeQuery:
    def __init__(self, data):
        self._data = data

    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def execute(self):
        return MagicMock(data=self._data)


def fake_supabase(*, licenses_data, sessions_data):
    queries = {
        "licenses": FakeQuery(licenses_data),
        "sessions": FakeQuery(sessions_data),
    }
    fake = MagicMock()
    fake.table = MagicMock(side_effect=lambda name: queries[name])
    return fake


def test_no_rows_raises_invalid(monkeypatch):
    monkeypatch.setattr(license_mod, "supabase", fake_supabase(licenses_data=[], sessions_data=[]))
    with pytest.raises(InvalidLicenseKeyError):
        validate_license("nope")


def test_inactive_raises_invalid(monkeypatch):
    monkeypatch.setattr(
        license_mod,
        "supabase",
        fake_supabase(
            licenses_data=[{"active": False, "expires_at": None}],
            sessions_data=[],
        ),
    )
    with pytest.raises(InvalidLicenseKeyError):
        validate_license("k")


def test_expired_raises_expired(monkeypatch):
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat().replace("+00:00", "Z")
    monkeypatch.setattr(
        license_mod,
        "supabase",
        fake_supabase(
            licenses_data=[{"active": True, "expires_at": past}],
            sessions_data=[],
        ),
    )
    with pytest.raises(ExpiredLicenseError):
        validate_license("k")


def test_future_expiry_does_not_raise(monkeypatch):
    future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat().replace("+00:00", "Z")
    monkeypatch.setattr(
        license_mod,
        "supabase",
        fake_supabase(
            licenses_data=[{"active": True, "expires_at": future}],
            sessions_data=[],
        ),
    )
    assert validate_license("k") is None


def test_missing_expires_at_skips_expiry(monkeypatch):
    monkeypatch.setattr(
        license_mod,
        "supabase",
        fake_supabase(
            licenses_data=[{"active": True}],
            sessions_data=[],
        ),
    )
    assert validate_license("k") is None


def test_active_session_raises_session_active(monkeypatch):
    monkeypatch.setattr(
        license_mod,
        "supabase",
        fake_supabase(
            licenses_data=[{"active": True, "expires_at": None}],
            sessions_data=[{"key": "k"}],
        ),
    )
    with pytest.raises(SessionAlreadyActiveError):
        validate_license("k")


def test_all_clear_returns_none(monkeypatch):
    monkeypatch.setattr(
        license_mod,
        "supabase",
        fake_supabase(
            licenses_data=[{"active": True, "expires_at": None}],
            sessions_data=[],
        ),
    )
    assert validate_license("k") is None


def test_clear_stale_sessions_deletes_every_row(monkeypatch):
    sb = MagicMock()
    monkeypatch.setattr(license_mod, "supabase", sb)
    clear_stale_sessions()
    sb.table.assert_called_once_with("sessions")
    delete = sb.table.return_value.delete
    delete.assert_called_once_with()
    delete.return_value.neq.assert_called_once_with("key", "")
    delete.return_value.neq.return_value.execute.assert_called_once_with()
