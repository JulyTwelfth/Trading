from unittest.mock import MagicMock

import pytest

from app.db import blacklist as blacklist_mod
from app.db.blacklist import (
    BlacklistedMarket,
    add_blacklist,
    clear_blacklist,
    list_blacklist,
    remove_blacklist,
)
from app.exceptions import BlacklistPersistenceError

ROW = {
    "license_key": "lic-1",
    "condition_id": "0xcond-A",
    "slug": "will-it-rain",
    "question": "Will it rain?",
    "market_url": "https://polymarket.com/event/weather/will-it-rain",
    "created_at": "2025-01-01T00:00:00Z",
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


# ── list_blacklist ───────────────────────────────────────────────────────────


async def test_list_blacklist_success(monkeypatch):
    # Row with an EXTRA key to exercise BlacklistedMarket's extra="ignore".
    row = {**ROW, "condition_id": "0xcond-B", "unexpected": "drop-me"}
    fake = fake_supabase_returning([row])
    monkeypatch.setattr(blacklist_mod, "supabase", fake)

    result = await list_blacklist("lic-1")

    assert result == [
        BlacklistedMarket(
            license_key="lic-1",
            condition_id="0xcond-B",
            slug="will-it-rain",
            question="Will it rain?",
            market_url="https://polymarket.com/event/weather/will-it-rain",
        )
    ]
    # extra column ("created_at" and "unexpected") dropped — not model fields.
    assert not hasattr(result[0], "unexpected")
    assert not hasattr(result[0], "created_at")
    # Scoped + ordered query: select for THIS license, ordered by created_at.
    fake.eq.assert_called_once_with("license_key", "lic-1")
    fake.order.assert_called_once_with("created_at")


async def test_list_blacklist_preserves_db_order(monkeypatch):
    rows = [
        {**ROW, "condition_id": "0xfirst"},
        {**ROW, "condition_id": "0xsecond"},
        {**ROW, "condition_id": "0xthird"},
    ]
    monkeypatch.setattr(blacklist_mod, "supabase", fake_supabase_returning(rows))

    result = await list_blacklist("lic-1")

    assert [m.condition_id for m in result] == ["0xfirst", "0xsecond", "0xthird"]


async def test_list_blacklist_empty(monkeypatch):
    monkeypatch.setattr(blacklist_mod, "supabase", fake_supabase_returning([]))
    assert await list_blacklist("lic-1") == []


async def test_list_blacklist_wraps_error(monkeypatch):
    monkeypatch.setattr(blacklist_mod, "supabase", fake_supabase_raising())
    with pytest.raises(BlacklistPersistenceError):
        await list_blacklist("lic-1")


# ── add_blacklist ────────────────────────────────────────────────────────────


async def test_add_blacklist_single_upsert_with_full_payload(monkeypatch):
    rows = [
        {
            "license_key": "lic-1",
            "condition_id": "0xcond-A",
            "slug": "m-a",
            "question": "A?",
            "market_url": "https://polymarket.com/event/e/m-a",
        },
        {
            "license_key": "lic-1",
            "condition_id": "0xcond-B",
            "slug": "m-b",
            "question": "B?",
            "market_url": "https://polymarket.com/event/e/m-a",
        },
    ]
    returned = [{**ROW, "condition_id": "0xcond-A"}, {**ROW, "condition_id": "0xcond-B"}]
    fake = fake_supabase_returning(returned)
    monkeypatch.setattr(blacklist_mod, "supabase", fake)

    result = await add_blacklist("lic-1", rows)

    # Exactly ONE upsert carrying the whole list (not one call per row).
    fake.upsert.assert_called_once()
    args, kwargs = fake.upsert.call_args
    assert args[0] == rows
    assert kwargs["on_conflict"] == "license_key,condition_id"
    # Parsed models echoed back.
    assert [m.condition_id for m in result] == ["0xcond-A", "0xcond-B"]
    assert all(isinstance(m, BlacklistedMarket) for m in result)


async def test_add_blacklist_empty_rows_still_one_upsert(monkeypatch):
    fake = fake_supabase_returning([])
    monkeypatch.setattr(blacklist_mod, "supabase", fake)

    result = await add_blacklist("lic-1", [])

    assert result == []
    fake.upsert.assert_called_once()
    args, _ = fake.upsert.call_args
    assert args[0] == []


async def test_add_blacklist_forces_passed_license_onto_rows(monkeypatch):
    # Rows carry a stale/wrong license_key (and one omits it entirely); the
    # function must override every row to the license_key argument.
    rows = [
        {"license_key": "other-lic", "condition_id": "0xcond-A"},
        {"condition_id": "0xcond-B"},
    ]
    fake = fake_supabase_returning([])
    monkeypatch.setattr(blacklist_mod, "supabase", fake)

    await add_blacklist("lic-1", rows)

    args, _ = fake.upsert.call_args
    assert [r["license_key"] for r in args[0]] == ["lic-1", "lic-1"]
    # Original input is not mutated.
    assert rows[0]["license_key"] == "other-lic"
    assert "license_key" not in rows[1]


async def test_add_blacklist_wraps_error(monkeypatch):
    monkeypatch.setattr(blacklist_mod, "supabase", fake_supabase_raising())
    with pytest.raises(BlacklistPersistenceError):
        await add_blacklist("lic-1", [{"license_key": "lic-1", "condition_id": "0xc"}])


# ── remove_blacklist ─────────────────────────────────────────────────────────


async def test_remove_blacklist_scopes_delete_to_license_and_condition(monkeypatch):
    fake = fake_supabase_returning([])
    monkeypatch.setattr(blacklist_mod, "supabase", fake)

    assert await remove_blacklist("lic-1", "0xcond-A") is None

    fake.delete.assert_called_once()
    # Both scoping predicates applied (.eq license_key then .eq condition_id).
    eq_calls = {c.args for c in fake.eq.call_args_list}
    assert ("license_key", "lic-1") in eq_calls
    assert ("condition_id", "0xcond-A") in eq_calls


async def test_remove_blacklist_wraps_error(monkeypatch):
    monkeypatch.setattr(blacklist_mod, "supabase", fake_supabase_raising())
    with pytest.raises(BlacklistPersistenceError):
        await remove_blacklist("lic-1", "0xcond-A")


# ── clear_blacklist ──────────────────────────────────────────────────────────


async def test_clear_blacklist_scopes_delete_to_license_only(monkeypatch):
    fake = fake_supabase_returning([])
    monkeypatch.setattr(blacklist_mod, "supabase", fake)

    assert await clear_blacklist("lic-1") is None

    fake.delete.assert_called_once()
    # Scoped to the licence ALONE — a SINGLE eq, unlike remove's two
    # (license_key + condition_id). One eq means the whole licence is wiped.
    fake.eq.assert_called_once_with("license_key", "lic-1")
    eq_calls = {c.args for c in fake.eq.call_args_list}
    assert eq_calls == {("license_key", "lic-1")}
    assert not any(c.args[0] == "condition_id" for c in fake.eq.call_args_list)


async def test_clear_blacklist_wraps_error(monkeypatch):
    monkeypatch.setattr(blacklist_mod, "supabase", fake_supabase_raising())
    with pytest.raises(BlacklistPersistenceError):
        await clear_blacklist("lic-1")
