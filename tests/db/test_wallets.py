from unittest.mock import MagicMock

import pytest

from app.db import wallets as wallets_mod
from app.db.wallets import (
    Wallet,
    delete_wallet,
    list_wallets,
    next_wallet_id,
    upsert_wallet,
    wallet_sort_key,
)
from app.exceptions import WalletPersistenceError, WalletSlotsExhaustedError

ROW = {
    "license_key": "lic-1",
    "wallet_id": "A",
    "proxy_address": "0xproxy",
    "private_key": "0xpk",
}


def fake_supabase_returning(data):
    """A supabase whose entire query chain returns self and whose .execute()
    yields MagicMock(data=data)."""
    chain = MagicMock()
    chain.table.return_value = chain
    chain.select.return_value = chain
    chain.eq.return_value = chain
    chain.order.return_value = chain
    chain.upsert.return_value = chain
    chain.delete.return_value = chain
    chain.execute.return_value = MagicMock(data=data)
    return chain


def fake_supabase_raising():
    chain = MagicMock()
    chain.table.return_value = chain
    chain.select.return_value = chain
    chain.eq.return_value = chain
    chain.order.return_value = chain
    chain.upsert.return_value = chain
    chain.delete.return_value = chain
    chain.execute.side_effect = RuntimeError("db down")
    return chain


async def test_list_wallets_success(monkeypatch):
    # Row with an EXTRA key to exercise Wallet's extra="ignore".
    row = {**ROW, "wallet_id": "B", "unexpected": "drop-me"}
    monkeypatch.setattr(wallets_mod, "supabase", fake_supabase_returning([row]))

    result = await list_wallets("lic-1")

    assert result == [
        Wallet(
            license_key="lic-1",
            wallet_id="B",
            proxy_address="0xproxy",
            private_key="0xpk",
        )
    ]
    assert not hasattr(result[0], "unexpected")


async def test_list_wallets_empty(monkeypatch):
    monkeypatch.setattr(wallets_mod, "supabase", fake_supabase_returning([]))
    assert await list_wallets("lic-1") == []


async def test_list_wallets_wraps_error(monkeypatch):
    monkeypatch.setattr(wallets_mod, "supabase", fake_supabase_raising())
    with pytest.raises(WalletPersistenceError):
        await list_wallets("lic-1")


async def test_upsert_wallet_success(monkeypatch):
    fake = fake_supabase_returning([ROW])
    monkeypatch.setattr(wallets_mod, "supabase", fake)

    wallet = await upsert_wallet("lic-1", "A", "0xproxy", "0xpk")

    assert wallet == Wallet.model_validate(ROW)
    # Confirm the upsert payload + conflict target were passed through.
    args, kwargs = fake.upsert.call_args
    assert args[0] == {
        "license_key": "lic-1",
        "wallet_id": "A",
        "proxy_address": "0xproxy",
        "private_key": "0xpk",
    }
    assert kwargs["on_conflict"] == "license_key,wallet_id"


async def test_upsert_wallet_wraps_error(monkeypatch):
    monkeypatch.setattr(wallets_mod, "supabase", fake_supabase_raising())
    with pytest.raises(WalletPersistenceError):
        await upsert_wallet("lic-1", "A", "0xproxy", "0xpk")


async def test_delete_wallet_success(monkeypatch):
    fake = fake_supabase_returning([])
    monkeypatch.setattr(wallets_mod, "supabase", fake)

    assert await delete_wallet("lic-1", "B") is None
    fake.delete.assert_called_once()


async def test_delete_wallet_wraps_error(monkeypatch):
    monkeypatch.setattr(wallets_mod, "supabase", fake_supabase_raising())
    with pytest.raises(WalletPersistenceError):
        await delete_wallet("lic-1", "A")


# ── wallet_sort_key ──────────────────────────────────────────────────────────


def test_wallet_sort_key_orders_numerically():
    # The lexicographic trap this key exists to avoid: "wallet10" < "wallet2" as strings.
    assert sorted(["wallet2", "wallet10", "wallet1"], key=wallet_sort_key) == [
        "wallet1",
        "wallet2",
        "wallet10",
    ]


def test_wallet_sort_key_puts_legacy_ids_last():
    assert sorted(["wallet2", "B", "wallet10", "A"], key=wallet_sort_key) == [
        "wallet2",
        "wallet10",
        "A",
        "B",
    ]


# ── next_wallet_id ───────────────────────────────────────────────────────────


def test_next_wallet_id_empty():
    assert next_wallet_id([]) == "wallet1"


def test_next_wallet_id_appends():
    assert next_wallet_id(["wallet1", "wallet2", "wallet3"]) == "wallet4"


def test_next_wallet_id_fills_gap():
    # Deleting wallet2 of {1,2,3} frees that slot for the next registration.
    assert next_wallet_id(["wallet1", "wallet3"]) == "wallet2"


def test_next_wallet_id_ignores_legacy_ids():
    assert next_wallet_id(["A", "B"]) == "wallet1"


def test_next_wallet_id_skips_out_of_sequence_slot():
    # An explicitly-created high slot does not push allocation past the lowest gap.
    assert next_wallet_id(["wallet1", "wallet900"]) == "wallet2"


# ── list_wallets ordering + legacy tolerance ─────────────────────────────────


async def test_list_wallets_sorts_numerically(monkeypatch):
    rows = [{**ROW, "wallet_id": wid} for wid in ("wallet10", "wallet2", "wallet1")]
    monkeypatch.setattr(wallets_mod, "supabase", fake_supabase_returning(rows))

    result = await list_wallets("lic-1")

    assert [w.wallet_id for w in result] == ["wallet1", "wallet2", "wallet10"]


async def test_list_wallets_accepts_legacy_id(monkeypatch):
    # Wallet.model_validate runs OUTSIDE the try/except, so a strict id type would raise a
    # bare ValidationError out of list_wallets and tear down the ws connection.
    rows = [{**ROW, "wallet_id": "A"}, {**ROW, "wallet_id": "wallet1"}]
    monkeypatch.setattr(wallets_mod, "supabase", fake_supabase_returning(rows))

    result = await list_wallets("lic-1")

    assert [w.wallet_id for w in result] == ["wallet1", "A"]


async def test_upsert_wallet_accepts_canonical_id(monkeypatch):
    fake = fake_supabase_returning([{**ROW, "wallet_id": "wallet7"}])
    monkeypatch.setattr(wallets_mod, "supabase", fake)

    wallet = await upsert_wallet("lic-1", "wallet7", "0xproxy", "0xpk")

    assert wallet.wallet_id == "wallet7"
    args, kwargs = fake.upsert.call_args
    assert args[0]["wallet_id"] == "wallet7"
    assert kwargs["on_conflict"] == "license_key,wallet_id"


def test_next_wallet_id_raises_when_slots_exhausted():
    # Past the regex bound there is no canonical id left; handing one out anyway would make
    # every later register overwrite the same non-canonical row instead of adding a wallet.
    with pytest.raises(WalletSlotsExhaustedError):
        next_wallet_id([f"wallet{n}" for n in range(1, 10000)])


async def test_list_wallets_wraps_malformed_row(monkeypatch):
    # model_validate sits inside the guard, so a hand-edited row can't escape as a bare
    # ValidationError and tear down the ws connection.
    monkeypatch.setattr(
        wallets_mod, "supabase", fake_supabase_returning([{**ROW, "wallet_id": ""}])
    )
    with pytest.raises(WalletPersistenceError):
        await list_wallets("lic-1")
