import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from fastapi import WebSocketDisconnect

import app.api.ws as ws_mod
from app.api.ws import authenticate, websocket_endpoint
from app.exceptions import (
    ExpiredLicenseError,
    InvalidLicenseKeyError,
    LicenseServiceUnavailableError,
    SessionAlreadyActiveError,
)
from app.farm.schemas import FarmFilters

VALID_ADDR = "f" * 40
VALID_KEY = "a" * 64


class FakeWebSocket:
    def __init__(self, frames):
        self.accept = AsyncMock()
        self.send_json = AsyncMock()
        self.close = AsyncMock()
        self.receive_json = AsyncMock(side_effect=frames)

    def sent_types(self):
        return [c.args[0]["type"] for c in self.send_json.await_args_list]


class FakeSupabase:
    def __init__(self):
        self.inserts = []
        self.deletes = []

    def table(self, name):
        return Table(self, name)


class Table:
    def __init__(self, sb, name):
        self.sb = sb
        self.name = name

    def insert(self, payload):
        self.sb.inserts.append((self.name, payload))
        return Exec()

    def delete(self):
        return Delete(self.sb, self.name)


class Delete:
    def __init__(self, sb, name):
        self.sb = sb
        self.name = name

    def eq(self, col, val):
        self.sb.deletes.append((self.name, col, val))
        return Exec()


class Exec:
    def execute(self):
        return MagicMock(data=[])


def make_filters_dict() -> dict:
    return FarmFilters(
        vol_min=Decimal(0),
        vol_max=Decimal(100000),
        liq_min=Decimal(0),
        liq_max=Decimal(100000),
        spread_min=Decimal(0),
        spread_max=Decimal(100),
        reward_min=Decimal(0),
        time_remaining="all",
        created_date="all",
        change_24h="all",
    ).model_dump(mode="json")


def auth_frame():
    return {"type": "auth", "license_key": "LIC"}


def install_handlers(monkeypatch):
    fake_sb = FakeSupabase()
    monkeypatch.setattr(ws_mod, "supabase", fake_sb)
    monkeypatch.setattr(ws_mod, "validate_license", MagicMock())
    monkeypatch.setattr(ws_mod, "hydrate_after_auth", AsyncMock())
    monkeypatch.setattr(ws_mod, "handle_register", AsyncMock())
    monkeypatch.setattr(ws_mod, "handle_remove", AsyncMock())
    monkeypatch.setattr(ws_mod, "handle_list_request", AsyncMock())
    monkeypatch.setattr(ws_mod, "handle_farm_cancel", AsyncMock())
    monkeypatch.setattr(ws_mod, "send_error", AsyncMock())
    return fake_sb


async def test_authenticate_rejects_non_auth_first_frame(monkeypatch):
    monkeypatch.setattr(ws_mod, "validate_license", MagicMock())
    ws = FakeWebSocket(frames=[])

    result = await authenticate(ws, {"type": "wallet_list"})

    assert result is None
    ws.close.assert_awaited_once()
    payload = ws.send_json.await_args.args[0]
    assert payload["type"] == "auth_fail"
    assert payload["reason"] == "First message must be auth"


async def test_authenticate_rejects_missing_license_key(monkeypatch):
    monkeypatch.setattr(ws_mod, "validate_license", MagicMock())
    ws = FakeWebSocket(frames=[])

    result = await authenticate(ws, {"type": "auth"})

    assert result is None
    ws.close.assert_awaited_once()


async def test_authenticate_invalid_license(monkeypatch):
    monkeypatch.setattr(ws_mod, "validate_license", MagicMock(side_effect=InvalidLicenseKeyError()))
    ws = FakeWebSocket(frames=[])

    result = await authenticate(ws, auth_frame())

    assert result is None
    payload = ws.send_json.await_args.args[0]
    assert payload["type"] == "auth_fail"
    assert payload["reason"] == InvalidLicenseKeyError.reason
    ws.close.assert_awaited_once()


async def test_authenticate_expired_license(monkeypatch):
    monkeypatch.setattr(ws_mod, "validate_license", MagicMock(side_effect=ExpiredLicenseError()))
    ws = FakeWebSocket(frames=[])

    result = await authenticate(ws, auth_frame())

    assert result is None
    payload = ws.send_json.await_args.args[0]
    assert payload["reason"] == ExpiredLicenseError.reason


async def test_authenticate_session_already_active(monkeypatch):
    monkeypatch.setattr(
        ws_mod, "validate_license", MagicMock(side_effect=SessionAlreadyActiveError())
    )
    ws = FakeWebSocket(frames=[])

    result = await authenticate(ws, auth_frame())

    assert result is None
    payload = ws.send_json.await_args.args[0]
    assert payload["reason"] == SessionAlreadyActiveError.reason


async def test_authenticate_success_returns_key(monkeypatch):
    monkeypatch.setattr(ws_mod, "validate_license", MagicMock())
    ws = FakeWebSocket(frames=[])

    result = await authenticate(ws, auth_frame())

    assert result == "LIC"
    ws.close.assert_not_awaited()
    ws.send_json.assert_not_awaited()


async def test_endpoint_returns_when_auth_fails(monkeypatch):
    fake_sb = install_handlers(monkeypatch)
    monkeypatch.setattr(ws_mod, "validate_license", MagicMock(side_effect=InvalidLicenseKeyError()))
    ws = FakeWebSocket(frames=[auth_frame()])

    await websocket_endpoint(ws)

    ws.accept.assert_awaited_once()
    assert fake_sb.inserts == []
    assert fake_sb.deletes == []
    assert "auth_ok" not in ws.sent_types()


async def test_endpoint_auth_ok_inserts_session_and_hydrates(monkeypatch):
    fake_sb = install_handlers(monkeypatch)
    ws = FakeWebSocket(frames=[auth_frame(), WebSocketDisconnect()])

    await websocket_endpoint(ws)

    assert fake_sb.inserts == [("sessions", {"key": "LIC"})]
    assert "auth_ok" in ws.sent_types()
    ws_mod.hydrate_after_auth.assert_awaited_once_with(ws, "LIC")
    assert fake_sb.deletes == [("sessions", "key", "LIC")]


async def test_endpoint_dispatches_wallet_register(monkeypatch):
    install_handlers(monkeypatch)
    frame = {
        "type": "wallet_register",
        "wallet_id": "wallet1",
        "proxy_address": VALID_ADDR,
        "private_key": VALID_KEY,
    }
    ws = FakeWebSocket(frames=[auth_frame(), frame, WebSocketDisconnect()])

    await websocket_endpoint(ws)

    ws_mod.handle_register.assert_awaited_once()
    args = ws_mod.handle_register.await_args.args
    assert args[0] is ws and args[1] == "LIC"
    assert args[2].type == "wallet_register"


async def test_endpoint_dispatches_wallet_remove(monkeypatch):
    install_handlers(monkeypatch)
    frame = {"type": "wallet_remove", "wallet_id": "wallet2"}
    ws = FakeWebSocket(frames=[auth_frame(), frame, WebSocketDisconnect()])

    await websocket_endpoint(ws)

    ws_mod.handle_remove.assert_awaited_once()
    assert ws_mod.handle_remove.await_args.args[2].wallet_id == "wallet2"


async def test_endpoint_dispatches_wallet_list(monkeypatch):
    install_handlers(monkeypatch)
    frame = {"type": "wallet_list"}
    ws = FakeWebSocket(frames=[auth_frame(), frame, WebSocketDisconnect()])

    await websocket_endpoint(ws)

    ws_mod.handle_list_request.assert_awaited_once_with(ws, "LIC")


async def test_endpoint_dispatches_farm_create(monkeypatch):
    install_handlers(monkeypatch)
    returned_task = asyncio.create_task(asyncio.sleep(60))
    monkeypatch.setattr(ws_mod, "handle_farm_create", AsyncMock(return_value=returned_task))
    frame = {
        "type": "farm_create",
        "filters": make_filters_dict(),
        "bankroll": "100",
        "max_session_loss": "5",
    }
    ws = FakeWebSocket(frames=[auth_frame(), frame, WebSocketDisconnect()])

    await websocket_endpoint(ws)

    ws_mod.handle_farm_create.assert_awaited_once()
    call = ws_mod.handle_farm_create.await_args.args
    assert call[1] == "LIC"
    assert call[2].type == "farm_create"
    assert returned_task.cancelled()


async def test_endpoint_cleanup_survives_farm_cancel_timeout(monkeypatch):
    fake_sb = install_handlers(monkeypatch)
    task = asyncio.create_task(asyncio.sleep(60))
    monkeypatch.setattr(ws_mod, "handle_farm_create", AsyncMock(return_value=task))

    async def timeout_wait(aw, timeout):
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", timeout_wait)
    frame = {
        "type": "farm_create",
        "filters": make_filters_dict(),
        "bankroll": "100",
        "max_session_loss": "5",
    }
    ws = FakeWebSocket(frames=[auth_frame(), frame, WebSocketDisconnect()])

    await websocket_endpoint(ws)

    assert ("sessions", "key", "LIC") in fake_sb.deletes
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def test_endpoint_dispatches_farm_cancel(monkeypatch):
    install_handlers(monkeypatch)
    monkeypatch.setattr(ws_mod, "handle_farm_cancel", AsyncMock(return_value=None))
    frame = {"type": "farm_cancel"}
    ws = FakeWebSocket(frames=[auth_frame(), frame, WebSocketDisconnect()])

    await websocket_endpoint(ws)

    ws_mod.handle_farm_cancel.assert_awaited_once()
    assert ws_mod.handle_farm_cancel.await_args.args[1] is None


async def test_endpoint_farm_cancel_retains_detached_teardown(monkeypatch):
    install_handlers(monkeypatch)
    detached = asyncio.create_task(asyncio.sleep(60))
    monkeypatch.setattr(ws_mod, "handle_farm_cancel", AsyncMock(return_value=detached))
    frame = {"type": "farm_cancel"}
    ws = FakeWebSocket(frames=[auth_frame(), frame, WebSocketDisconnect()])

    await websocket_endpoint(ws)

    assert detached.cancelled()


async def test_endpoint_validation_error_sends_error_and_continues(monkeypatch):
    install_handlers(monkeypatch)
    bad_frame = {"type": "not_a_real_type"}
    good_frame = {"type": "wallet_list"}
    ws = FakeWebSocket(frames=[auth_frame(), bad_frame, good_frame, WebSocketDisconnect()])

    await websocket_endpoint(ws)

    ws_mod.send_error.assert_awaited_once()
    assert ws_mod.send_error.await_args.args[0] is ws
    assert isinstance(ws_mod.send_error.await_args.args[1], str)
    ws_mod.handle_list_request.assert_awaited_once_with(ws, "LIC")


async def test_endpoint_finally_cancels_live_farm_task(monkeypatch):
    install_handlers(monkeypatch)
    live_task = asyncio.create_task(asyncio.sleep(60))
    monkeypatch.setattr(ws_mod, "handle_farm_create", AsyncMock(return_value=live_task))

    farm_frame = {
        "type": "farm_create",
        "filters": make_filters_dict(),
        "bankroll": "100",
        "max_session_loss": "5",
    }
    ws = FakeWebSocket(frames=[auth_frame(), farm_frame, WebSocketDisconnect()])

    await websocket_endpoint(ws)

    assert live_task.done()
    assert live_task.cancelled()
    fake_sb = ws_mod.supabase
    assert ("sessions", "key", "LIC") in fake_sb.deletes


# ── sessions-insert failure: two causes, two messages ────────────────────────
# The insert can fail because another connection claimed the key between
# validate_license's check and this insert (a race — "Key already in use", retry later
# works), or because the licence backend is down ("License server unreachable", retry
# now will not). Reporting the second message for the first cause sends the user
# chasing an outage that isn't happening.


class InsertFailsSupabase(FakeSupabase):
    """Supabase whose sessions insert raises; everything else behaves normally."""

    def __init__(self, exc):
        super().__init__()
        self.exc = exc

    def table(self, name):
        return InsertFailsTable(self, name)


class InsertFailsTable(Table):
    def insert(self, payload):
        raise self.sb.exc


def duplicate_key_error() -> Exception:
    exc = Exception('duplicate key value violates unique constraint "sessions_pkey"')
    exc.code = "23505"  # postgres unique_violation, as postgrest's APIError carries it
    return exc


async def test_endpoint_reports_key_in_use_when_session_row_already_exists(monkeypatch):
    install_handlers(monkeypatch)
    monkeypatch.setattr(ws_mod, "supabase", InsertFailsSupabase(duplicate_key_error()))
    ws = FakeWebSocket(frames=[auth_frame()])

    await websocket_endpoint(ws)

    payload = ws.send_json.await_args.args[0]
    assert payload["type"] == "auth_fail"
    assert payload["reason"] == SessionAlreadyActiveError.reason
    assert "auth_ok" not in ws.sent_types()
    ws_mod.hydrate_after_auth.assert_not_awaited()


async def test_endpoint_reports_unreachable_for_any_other_insert_failure(monkeypatch):
    install_handlers(monkeypatch)
    monkeypatch.setattr(ws_mod, "supabase", InsertFailsSupabase(OSError("getaddrinfo failed")))
    ws = FakeWebSocket(frames=[auth_frame()])

    await websocket_endpoint(ws)

    payload = ws.send_json.await_args.args[0]
    assert payload["type"] == "auth_fail"
    assert payload["reason"] == LicenseServiceUnavailableError.reason
    assert "auth_ok" not in ws.sent_types()
