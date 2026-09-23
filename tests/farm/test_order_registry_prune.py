"""Requote registry-leak fix: requote_leg keeps the superseded oid in order_registry (a
cancel/fill race on it must still resolve — see
tests/fills/test_requote_fill_race.py::test_match_on_rotated_old_oid_is_recognized) but now
marks it retired, and prune_order_registry evicts retired entries once the retention window
elapses so the registry can't grow unbounded as requotes pile up over a long session.
Live orders (retired_at is None) are never pruned."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.schemas import UserTrade, UserTradeMakerOrder
from app.constants import ORDER_REGISTRY_RETENTION_SECONDS
from app.farm import fills as fills_mod
from app.farm import requote as requote_mod
from app.farm import worker as worker_mod
from app.farm.fills import handle_trade
from app.farm.requote import requote_leg
from app.farm.schemas import FarmState, OrderInfo
from app.farm.worker import prune_order_registry, reconcile_tick

NOW = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)
RETENTION = ORDER_REGISTRY_RETENTION_SECONDS


def oi(*, retired_at=None) -> OrderInfo:
    return OrderInfo(
        condition_id="market-A", outcome="YES", token_id="tok-yes", retired_at=retired_at
    )


# ── prune_order_registry unit behavior ───────────────────────────────────────


def test_prune_removes_retired_entries_past_retention(farm_state: FarmState):
    farm_state.order_registry.clear()
    farm_state.order_registry["stale"] = oi(retired_at=NOW - timedelta(seconds=RETENTION + 1))

    prune_order_registry(farm_state, NOW)

    assert "stale" not in farm_state.order_registry


def test_prune_keeps_recently_retired_within_window(farm_state: FarmState):
    farm_state.order_registry.clear()
    # Still inside the race window — a straggler fill on it must still resolve.
    within = NOW - timedelta(seconds=RETENTION - 5)
    farm_state.order_registry["fresh-retired"] = oi(retired_at=within)

    prune_order_registry(farm_state, NOW)

    assert "fresh-retired" in farm_state.order_registry


def test_prune_never_removes_active_entries(farm_state: FarmState):
    farm_state.order_registry.clear()
    farm_state.order_registry["active"] = oi(retired_at=None)

    # Even far in the future, a live order (retired_at None) must survive.
    prune_order_registry(farm_state, NOW + timedelta(days=7))

    assert "active" in farm_state.order_registry


def test_prune_noop_when_all_active(farm_state: FarmState):
    farm_state.order_registry.clear()
    farm_state.order_registry["a"] = oi(retired_at=None)
    farm_state.order_registry["b"] = oi(retired_at=None)

    prune_order_registry(farm_state, NOW)

    assert set(farm_state.order_registry) == {"a", "b"}


def test_prune_only_removes_stale_retired_among_mixed(farm_state: FarmState):
    farm_state.order_registry.clear()
    farm_state.order_registry["active"] = oi(retired_at=None)
    farm_state.order_registry["fresh"] = oi(retired_at=NOW - timedelta(seconds=10))
    farm_state.order_registry["stale1"] = oi(retired_at=NOW - timedelta(seconds=RETENTION + 1))
    farm_state.order_registry["stale2"] = oi(retired_at=NOW - timedelta(hours=1))

    prune_order_registry(farm_state, NOW)

    assert set(farm_state.order_registry) == {"active", "fresh"}


def test_prune_empty_registry_is_safe(farm_state: FarmState):
    farm_state.order_registry.clear()
    prune_order_registry(farm_state, NOW)
    assert farm_state.order_registry == {}


def test_prune_evicts_entry_at_exact_retention_boundary(farm_state: FarmState):
    farm_state.order_registry.clear()
    # Retired exactly RETENTION ago — the window has elapsed, so it must be evicted (<=).
    farm_state.order_registry["boundary"] = oi(retired_at=NOW - timedelta(seconds=RETENTION))

    prune_order_registry(farm_state, NOW)

    assert "boundary" not in farm_state.order_registry


# ── requote_leg retirement ───────────────────────────────────────────────────


@pytest.fixture
def requote_stubs(monkeypatch):
    async def fake_cancel(client, oid):
        return None

    async def fake_place(client, order, post_only=False):
        return "yes-oid-v2"

    monkeypatch.setattr(requote_mod, "cancel_order", fake_cancel)
    monkeypatch.setattr(requote_mod, "place_limit_order", fake_place)


async def test_requote_retires_old_oid_but_keeps_it_registered(
    farm_state: FarmState, requote_stubs
):
    pos = farm_state.positions["market-A"]
    old_oid = pos.yes_order_id  # "yes-oid", active in the registry
    assert farm_state.order_registry[old_oid].retired_at is None

    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.48"))

    assert pos.yes_order_id == "yes-oid-v2"
    # Kept for the cancel/fill race, but now retired so prune can reclaim it later.
    assert old_oid in farm_state.order_registry
    assert farm_state.order_registry[old_oid].retired_at is not None
    # The replacement is a live order — must not be retired.
    assert farm_state.order_registry["yes-oid-v2"].retired_at is None


async def test_requote_then_prune_evicts_superseded_oid_after_window(
    farm_state: FarmState, requote_stubs
):
    pos = farm_state.positions["market-A"]
    old_oid = pos.yes_order_id
    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.48"))
    retired_at = farm_state.order_registry[old_oid].retired_at

    # Within the window the race net stays up.
    prune_order_registry(farm_state, retired_at + timedelta(seconds=RETENTION - 1))
    assert old_oid in farm_state.order_registry

    # Past the window the superseded oid is reclaimed; the live oid survives.
    prune_order_registry(farm_state, retired_at + timedelta(seconds=RETENTION + 1))
    assert old_oid not in farm_state.order_registry
    assert "yes-oid-v2" in farm_state.order_registry


async def test_repeated_requotes_do_not_leak_registry(farm_state: FarmState, monkeypatch):
    """Many requotes on one leg must leave a bounded registry: every superseded oid is
    retired and reclaimed by the prune, leaving only the live current oid (+ the
    untouched NO leg)."""
    pos = farm_state.positions["market-A"]
    seq = iter([f"yes-v{i}" for i in range(1, 6)])

    async def fake_cancel(client, oid):
        return None

    async def fake_place(client, order, post_only=False):
        return next(seq)

    monkeypatch.setattr(requote_mod, "cancel_order", fake_cancel)
    monkeypatch.setattr(requote_mod, "place_limit_order", fake_place)

    for px in ("0.48", "0.47", "0.46", "0.45", "0.44"):
        await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal(px))

    current = pos.yes_order_id
    retired_oids = [
        oid for oid, info in farm_state.order_registry.items() if info.retired_at is not None
    ]
    assert len(retired_oids) >= 4, "each superseded oid must be retained-but-retired"

    prune_order_registry(farm_state, datetime.now(timezone.utc) + timedelta(seconds=RETENTION + 1))
    remaining = set(farm_state.order_registry)
    assert current in remaining, "the live current oid must survive"
    assert "no-oid" in remaining, "the untouched NO leg must survive"
    assert all(o not in remaining for o in retired_oids), "all retired oids reclaimed"


# ── wiring: reconcile_tick runs the prune ────────────────────────────────────


async def test_reconcile_tick_runs_the_registry_prune(farm_state: FarmState, monkeypatch):
    # Clear positions so prune_dead_positions early-returns and no open/close runs;
    # the registry prune still executes.
    farm_state.positions.clear()
    farm_state.order_registry.clear()
    farm_state.order_registry["active"] = oi(retired_at=None)
    farm_state.order_registry["stale"] = oi(
        retired_at=datetime.now(timezone.utc) - timedelta(seconds=RETENTION + 5)
    )

    async def fake_fetch_eligible_markets(http):
        return []

    async def fake_get_balance(addr):
        return Decimal("1000")

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_fetch_eligible_markets)
    monkeypatch.setattr(worker_mod, "get_balance", fake_get_balance, raising=False)

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert "stale" not in farm_state.order_registry, "reconcile_tick must run the registry prune"
    assert "active" in farm_state.order_registry


# ── interaction with the orphan auto-exit (#23) ──────────────────────────────


async def test_retired_oid_still_orphan_rescued_within_window(
    farm_state: FarmState, requote_stubs, monkeypatch
):
    """Retain-but-retire must preserve the #23 orphan safety net: a late fill on a superseded oid
    after its position is dropped still routes to the orphan auto-exit (within the window),
    rather than being silently dropped. This is exactly why we retire instead of evicting."""

    async def fake_cancel(client, oid):
        return None

    monkeypatch.setattr(fills_mod, "cancel_order", fake_cancel)

    pos = farm_state.positions["market-A"]
    old_oid = pos.yes_order_id
    token = pos.market.yes_token_id
    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.48"))
    assert farm_state.order_registry[old_oid].retired_at is not None

    # Position dropped, but the retired oid is still inside its retention window.
    farm_state.positions.pop("market-A")

    late_fill = UserTrade(
        event_type="trade",
        id="late-race",
        asset_id=token,
        market="market-A",
        side="BUY",
        price=Decimal("0.5"),
        size=Decimal("10"),
        outcome="YES",
        status="MATCHED",
        timestamp="2026-06-10T08:00:00Z",
        maker_orders=[
            UserTradeMakerOrder(
                asset_id=token,
                order_id=old_oid,
                matched_amount=Decimal("10"),
                outcome="YES",
                owner="0xowner",
                price=Decimal("0.5"),
            )
        ],
        taker_order_id="some-taker",
    )

    await handle_trade(MagicMock(), late_fill, farm_state, AsyncMock())

    # Orphan auto-exit engaged: shares staged for the MINED→FAK-SELL, market blacklisted.
    from app.farm.volatility import is_blacklisted

    assert "late-race" in farm_state.pending_fok_exits
    assert is_blacklisted(farm_state, "market-A")


# ── booked_exit_fills prune ──────────────────────────────────────────────────
# The set dedupes exit-fill bookings by (trade.id, exit_oid). It is only consulted for an oid
# still in pending_exit_order_ids, so keys whose oid has drained out are unreachable and safe
# to drop — the point being that this is provable, not a size-cap eviction that could discard
# a key still needed to suppress a replayed MINED/CONFIRMED frame.


def test_prune_drops_booked_fills_whose_exit_order_is_gone(farm_state):
    farm_state.pending_exit_order_ids = {"live-oid"}
    farm_state.booked_exit_fills = {("trade-1", "live-oid"), ("trade-2", "retired-oid")}

    prune_order_registry(farm_state, NOW)

    assert farm_state.booked_exit_fills == {("trade-1", "live-oid")}


def test_prune_keeps_every_key_for_a_still_pending_exit(farm_state):
    farm_state.pending_exit_order_ids = {"live-oid"}
    farm_state.booked_exit_fills = {("trade-1", "live-oid"), ("trade-2", "live-oid")}

    prune_order_registry(farm_state, NOW)

    assert farm_state.booked_exit_fills == {("trade-1", "live-oid"), ("trade-2", "live-oid")}


def test_prune_empties_the_set_once_no_exits_are_pending(farm_state):
    farm_state.pending_exit_order_ids = set()
    farm_state.booked_exit_fills = {("trade-1", "oid-a"), ("trade-2", "oid-b")}

    prune_order_registry(farm_state, NOW)

    assert farm_state.booked_exit_fills == set()
