import asyncio
import logging
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.api.wallet import handlers as handlers_mod
from app.api.wallet.handlers import (
    handle_list_request,
    handle_register,
    handle_remove,
    hydrate_after_auth,
    send_error,
    send_wallet_list,
)
from app.api.wallet.messages import WalletRegisterMessage, WalletRemoveMessage
from app.db.wallets import Wallet
from app.exceptions import WalletPersistenceError, WalletSlotsExhaustedError

VALID_ADDR = "f" * 40
VALID_KEY = "a" * 64


def make_ws() -> MagicMock:
    ws = MagicMock()
    ws.send_json = AsyncMock()
    return ws


def make_wallet(wallet_id="A", proxy="0x" + VALID_ADDR) -> Wallet:
    return Wallet(
        license_key="LIC",
        wallet_id=wallet_id,
        proxy_address=proxy,
        private_key="0x" + VALID_KEY,
    )


def register_msg() -> WalletRegisterMessage:
    return WalletRegisterMessage(
        type="wallet_register",
        wallet_id="wallet1",
        proxy_address=VALID_ADDR,
        private_key=VALID_KEY,
    )


# ── send_wallet_list ─────────────────────────────────────────────────────────


async def test_send_wallet_list_serializes_entries(monkeypatch):
    ws = make_ws()
    monkeypatch.setattr(
        handlers_mod, "list_wallets", AsyncMock(return_value=[make_wallet("A"), make_wallet("B")])
    )

    await send_wallet_list(ws, "LIC")

    ws.send_json.assert_awaited_once()
    payload = ws.send_json.await_args.args[0]
    assert payload["type"] == "wallet_list"
    assert {e["wallet_id"] for e in payload["wallets"]} == {"A", "B"}
    # Each entry carries the live balance (stubbed to 100 by the conftest fixture).
    assert all(e["balance"] == "100" for e in payload["wallets"])


async def test_send_wallet_list_balance_failure_sends_null(monkeypatch):
    # A dead RPC must not block the wallet list — the entry ships with balance=None.
    ws = make_ws()
    monkeypatch.setattr(handlers_mod, "list_wallets", AsyncMock(return_value=[make_wallet("A")]))
    monkeypatch.setattr(
        handlers_mod, "get_balance", AsyncMock(side_effect=RuntimeError("rpc down"))
    )

    await send_wallet_list(ws, "LIC")

    payload = ws.send_json.await_args.args[0]
    assert payload["wallets"][0]["balance"] is None


async def test_send_wallet_list_fetches_balances_concurrently(monkeypatch):
    # N slow wallets must cost one timeout budget total, not N: the per-wallet fetches
    # overlap, so the tracked stub sees both in flight at once.
    active = {"now": 0, "peak": 0}

    async def tracked_balance(wallet):
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        loop.call_later(0.05, lambda: fut.cancelled() or fut.set_result(Decimal("7")))
        try:
            return await fut
        finally:
            active["now"] -= 1

    ws = make_ws()
    monkeypatch.setattr(
        handlers_mod, "list_wallets", AsyncMock(return_value=[make_wallet("A"), make_wallet("B")])
    )
    monkeypatch.setattr(handlers_mod, "get_balance", tracked_balance)

    await send_wallet_list(ws, "LIC")

    assert active["peak"] == 2
    payload = ws.send_json.await_args.args[0]
    assert all(e["balance"] == "7" for e in payload["wallets"])


async def test_send_wallet_list_slow_balance_fetch_times_out(monkeypatch):
    # A slow/hanging RPC must not block the wallet list (register/remove flows wait on
    # it): the fetch is hard-capped and the entry falls back to balance=None.
    async def slow_balance(wallet):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        loop.call_later(0.5, lambda: fut.cancelled() or fut.set_result(Decimal("42")))
        return await fut

    ws = make_ws()
    monkeypatch.setattr(handlers_mod, "list_wallets", AsyncMock(return_value=[make_wallet("A")]))
    monkeypatch.setattr(handlers_mod, "get_balance", slow_balance)
    monkeypatch.setattr(handlers_mod, "BALANCE_FETCH_TIMEOUT_SECONDS", 0.01, raising=False)

    await send_wallet_list(ws, "LIC")

    payload = ws.send_json.await_args.args[0]
    assert payload["wallets"][0]["balance"] is None


# ── send_error ───────────────────────────────────────────────────────────────


async def test_send_error_emits_wallet_error():
    ws = make_ws()
    await send_error(ws, "some reason")
    ws.send_json.assert_awaited_once()
    payload = ws.send_json.await_args.args[0]
    assert payload == {"type": "wallet_error", "reason": "some reason"}


# ── hydrate_after_auth ───────────────────────────────────────────────────────


async def test_hydrate_after_auth_success(monkeypatch):
    ws = make_ws()
    monkeypatch.setattr(handlers_mod, "list_wallets", AsyncMock(return_value=[make_wallet("A")]))

    await hydrate_after_auth(ws, "LIC")

    payload = ws.send_json.await_args.args[0]
    assert payload["type"] == "wallet_list"


async def test_hydrate_after_auth_persistence_error_sends_error(monkeypatch):
    ws = make_ws()
    monkeypatch.setattr(
        handlers_mod, "list_wallets", AsyncMock(side_effect=WalletPersistenceError())
    )

    await hydrate_after_auth(ws, "LIC")

    payload = ws.send_json.await_args.args[0]
    assert payload["type"] == "wallet_error"
    assert payload["reason"] == WalletPersistenceError.reason


# ── handle_register ──────────────────────────────────────────────────────────


async def test_handle_register_success_upserts_and_lists(monkeypatch):
    ws = make_ws()
    upsert = AsyncMock()
    monkeypatch.setattr(handlers_mod, "upsert_wallet", upsert)
    monkeypatch.setattr(handlers_mod, "list_wallets", AsyncMock(return_value=[make_wallet("A")]))

    msg = register_msg()
    await handle_register(ws, "LIC", msg)

    upsert.assert_awaited_once_with("LIC", "wallet1", msg.proxy_address, msg.private_key)
    payload = ws.send_json.await_args.args[0]
    assert payload["type"] == "wallet_list"


async def test_handle_register_persistence_error_sends_error(monkeypatch):
    ws = make_ws()
    monkeypatch.setattr(
        handlers_mod, "upsert_wallet", AsyncMock(side_effect=WalletPersistenceError())
    )
    list_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(handlers_mod, "list_wallets", list_mock)

    await handle_register(ws, "LIC", register_msg())

    payload = ws.send_json.await_args.args[0]
    assert payload["type"] == "wallet_error"
    # Only the allocation read; a failed upsert never reaches the refresh list.
    assert list_mock.await_count == 1


# ── handle_remove ────────────────────────────────────────────────────────────


async def test_handle_remove_success_deletes_and_lists(monkeypatch):
    ws = make_ws()
    delete = AsyncMock()
    monkeypatch.setattr(handlers_mod, "delete_wallet", delete)
    monkeypatch.setattr(handlers_mod, "list_wallets", AsyncMock(return_value=[]))

    msg = WalletRemoveMessage(type="wallet_remove", wallet_id="B")
    await handle_remove(ws, "LIC", msg)

    delete.assert_awaited_once_with("LIC", "B")
    payload = ws.send_json.await_args.args[0]
    assert payload["type"] == "wallet_list"
    assert payload["wallets"] == []


async def test_handle_remove_persistence_error_sends_error(monkeypatch):
    ws = make_ws()
    monkeypatch.setattr(
        handlers_mod, "delete_wallet", AsyncMock(side_effect=WalletPersistenceError())
    )

    await handle_remove(ws, "LIC", WalletRemoveMessage(type="wallet_remove", wallet_id="A"))

    payload = ws.send_json.await_args.args[0]
    assert payload["type"] == "wallet_error"


# ── handle_list_request ──────────────────────────────────────────────────────


async def test_handle_list_request_success(monkeypatch):
    ws = make_ws()
    monkeypatch.setattr(handlers_mod, "list_wallets", AsyncMock(return_value=[make_wallet("A")]))

    await handle_list_request(ws, "LIC")

    payload = ws.send_json.await_args.args[0]
    assert payload["type"] == "wallet_list"


async def test_handle_list_request_persistence_error_sends_error(monkeypatch):
    ws = make_ws()
    monkeypatch.setattr(
        handlers_mod, "list_wallets", AsyncMock(side_effect=WalletPersistenceError())
    )

    await handle_list_request(ws, "LIC")

    payload = ws.send_json.await_args.args[0]
    assert payload["type"] == "wallet_error"


# ── handle_register: allocate vs edit ────────────────────────────────────────


def register_msg_without_id() -> WalletRegisterMessage:
    return WalletRegisterMessage(
        type="wallet_register",
        proxy_address=VALID_ADDR,
        private_key=VALID_KEY,
    )


async def test_handle_register_allocates_next_id_when_omitted(monkeypatch):
    ws = make_ws()
    upsert = AsyncMock()
    monkeypatch.setattr(handlers_mod, "upsert_wallet", upsert)
    monkeypatch.setattr(
        handlers_mod,
        "list_wallets",
        AsyncMock(return_value=[make_wallet("wallet1"), make_wallet("wallet2")]),
    )

    msg = register_msg_without_id()
    await handle_register(ws, "LIC", msg)

    upsert.assert_awaited_once_with("LIC", "wallet3", msg.proxy_address, msg.private_key)


async def test_handle_register_fills_gap_when_omitted(monkeypatch):
    ws = make_ws()
    upsert = AsyncMock()
    monkeypatch.setattr(handlers_mod, "upsert_wallet", upsert)
    monkeypatch.setattr(
        handlers_mod,
        "list_wallets",
        AsyncMock(return_value=[make_wallet("wallet1"), make_wallet("wallet3")]),
    )

    await handle_register(ws, "LIC", register_msg_without_id())

    assert upsert.await_args.args[1] == "wallet2"


async def test_handle_register_allocates_wallet1_for_legacy_only_registry(monkeypatch):
    ws = make_ws()
    upsert = AsyncMock()
    monkeypatch.setattr(handlers_mod, "upsert_wallet", upsert)
    monkeypatch.setattr(
        handlers_mod,
        "list_wallets",
        AsyncMock(return_value=[make_wallet("A"), make_wallet("B")]),
    )

    await handle_register(ws, "LIC", register_msg_without_id())

    assert upsert.await_args.args[1] == "wallet1"


async def test_handle_register_explicit_id_edits_in_place(monkeypatch):
    ws = make_ws()
    upsert = AsyncMock()
    monkeypatch.setattr(handlers_mod, "upsert_wallet", upsert)
    monkeypatch.setattr(
        handlers_mod,
        "list_wallets",
        AsyncMock(return_value=[make_wallet("wallet1"), make_wallet("wallet2")]),
    )

    msg = WalletRegisterMessage(
        type="wallet_register",
        wallet_id="wallet2",
        proxy_address=VALID_ADDR,
        private_key=VALID_KEY,
    )
    await handle_register(ws, "LIC", msg)

    assert upsert.await_args.args[1] == "wallet2"


# ── handle_register: the wallet cap ──────────────────────────────────────────


async def test_handle_register_rejects_new_wallet_at_cap(monkeypatch):
    ws = make_ws()
    upsert = AsyncMock()
    monkeypatch.setattr(handlers_mod, "upsert_wallet", upsert)
    monkeypatch.setattr(handlers_mod, "MAX_WALLETS_PER_LICENSE", 2)
    monkeypatch.setattr(
        handlers_mod,
        "list_wallets",
        AsyncMock(return_value=[make_wallet("wallet1"), make_wallet("wallet2")]),
    )

    await handle_register(ws, "LIC", register_msg_without_id())

    payload = ws.send_json.await_args.args[0]
    assert payload["type"] == "wallet_error"
    # Pin the shape the Protocol Delta commits to, not just the patched number.
    assert payload["reason"] == "Wallet limit reached (2 per license)"
    upsert.assert_not_awaited()


async def test_handle_register_edit_allowed_at_cap(monkeypatch):
    # The cap gates new slots only — replacing credentials in an existing slot must work.
    ws = make_ws()
    upsert = AsyncMock()
    monkeypatch.setattr(handlers_mod, "upsert_wallet", upsert)
    monkeypatch.setattr(handlers_mod, "MAX_WALLETS_PER_LICENSE", 2)
    monkeypatch.setattr(
        handlers_mod,
        "list_wallets",
        AsyncMock(return_value=[make_wallet("wallet1"), make_wallet("wallet2")]),
    )

    msg = WalletRegisterMessage(
        type="wallet_register",
        wallet_id="wallet2",
        proxy_address=VALID_ADDR,
        private_key=VALID_KEY,
    )
    await handle_register(ws, "LIC", msg)

    upsert.assert_awaited_once()
    assert ws.send_json.await_args.args[0]["type"] == "wallet_list"


async def test_handle_register_list_failure_sends_error(monkeypatch):
    ws = make_ws()
    upsert = AsyncMock()
    monkeypatch.setattr(handlers_mod, "upsert_wallet", upsert)
    monkeypatch.setattr(
        handlers_mod, "list_wallets", AsyncMock(side_effect=WalletPersistenceError())
    )

    await handle_register(ws, "LIC", register_msg_without_id())

    payload = ws.send_json.await_args.args[0]
    assert payload["type"] == "wallet_error"
    upsert.assert_not_awaited()


async def test_handle_register_logs_allocated_slot_on_failure(monkeypatch, caplog):
    ws = make_ws()
    monkeypatch.setattr(
        handlers_mod, "list_wallets", AsyncMock(return_value=[make_wallet("wallet1")])
    )
    monkeypatch.setattr(
        handlers_mod, "upsert_wallet", AsyncMock(side_effect=WalletPersistenceError())
    )

    with caplog.at_level(logging.ERROR):
        await handle_register(ws, "LIC", register_msg_without_id())

    # The allocated slot, not "auto": an add that fails mid-upsert must be
    # correlatable to the concrete walletN it tried to write.
    assert "wallet2" in caplog.text
    assert "auto" not in caplog.text


# ── send_wallet_list at scale ────────────────────────────────────────────────


async def test_send_wallet_list_handles_many_wallets(monkeypatch):
    wallets = [make_wallet(f"wallet{n}") for n in range(1, 26)]
    ws = make_ws()
    monkeypatch.setattr(handlers_mod, "list_wallets", AsyncMock(return_value=wallets))

    await send_wallet_list(ws, "LIC")

    payload = ws.send_json.await_args.args[0]
    assert [e["wallet_id"] for e in payload["wallets"]] == [f"wallet{n}" for n in range(1, 26)]


async def test_send_wallet_list_bounds_balance_concurrency(monkeypatch):
    # The semaphore caps in-flight eth_calls; rebound low here since the real bound is
    # read from the constant at import time.
    active = {"now": 0, "peak": 0}

    async def tracked_balance(wallet):
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        try:
            await asyncio.sleep(0)
            return Decimal("7")
        finally:
            active["now"] -= 1

    ws = make_ws()
    monkeypatch.setattr(handlers_mod, "balance_semaphore", asyncio.Semaphore(2))
    monkeypatch.setattr(
        handlers_mod,
        "list_wallets",
        AsyncMock(return_value=[make_wallet(f"wallet{n}") for n in range(1, 26)]),
    )
    monkeypatch.setattr(handlers_mod, "get_balance", tracked_balance)

    await send_wallet_list(ws, "LIC")

    assert active["peak"] <= 2
    payload = ws.send_json.await_args.args[0]
    assert len(payload["wallets"]) == 25


async def test_handle_register_slots_exhausted_sends_error(monkeypatch):
    ws = make_ws()
    upsert = AsyncMock()
    monkeypatch.setattr(handlers_mod, "upsert_wallet", upsert)
    monkeypatch.setattr(handlers_mod, "MAX_WALLETS_PER_LICENSE", 100_000)
    monkeypatch.setattr(
        handlers_mod,
        "list_wallets",
        AsyncMock(return_value=[make_wallet(f"wallet{n}") for n in range(1, 10000)]),
    )

    await handle_register(ws, "LIC", register_msg_without_id())

    payload = ws.send_json.await_args.args[0]
    assert payload["type"] == "wallet_error"
    assert payload["reason"] == WalletSlotsExhaustedError.reason
    upsert.assert_not_awaited()
