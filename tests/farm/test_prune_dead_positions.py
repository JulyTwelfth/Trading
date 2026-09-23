"""prune_dead_positions: positions whose resting orders vanished from the CLOB
without a fill (server-side cancels are invisible — the user WS only carries
trades) must be closed after two consecutive missing sightings, so
active_markets reflects reality and the market re-quotes. Held shares or
in-flight exits block the prune: those orders were pulled intentionally."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.api.farm.messages import OrderCancelledEvent
from app.constants import GUARD_PULL_RELEASE_SECONDS
from app.farm import worker as worker_mod
from app.farm.schemas import ExitOrder, FarmState, Market
from app.farm.worker import prune_dead_positions, reconcile_tick


def wire_prune(monkeypatch, live_ids: set[str], events: list):
    async def fake_get_open_order_ids(client):
        return live_ids

    async def fake_send_event(websocket, event):
        events.append(event)

    monkeypatch.setattr(worker_mod, "get_open_order_ids", fake_get_open_order_ids)
    monkeypatch.setattr(worker_mod, "send_event", fake_send_event)


def clear_shares(state: FarmState, cid: str = "market-A") -> None:
    pos = state.positions[cid]
    pos.yes_shares = Decimal(0)
    pos.yes_cost_basis = Decimal(0)


async def test_keeps_position_while_any_leg_rests(farm_state: FarmState, monkeypatch):
    clear_shares(farm_state)
    farm_state.positions["market-A"].orders_missing_ticks = 1
    events: list = []
    wire_prune(monkeypatch, live_ids={"no-oid"}, events=events)

    await prune_dead_positions(MagicMock(), farm_state, AsyncMock())

    pos = farm_state.positions["market-A"]
    assert pos.orders_missing_ticks == 0, "a live leg must reset the missing counter"
    assert events == []


async def test_closes_after_two_consecutive_missing_ticks(farm_state: FarmState, monkeypatch):
    clear_shares(farm_state)
    events: list = []
    wire_prune(monkeypatch, live_ids=set(), events=events)

    async def confirmed_cancel(client, oid):
        return True

    monkeypatch.setattr(worker_mod, "cancel_order_with_retry", confirmed_cancel)

    await prune_dead_positions(MagicMock(), farm_state, AsyncMock())
    assert "market-A" in farm_state.positions, "first missing sighting must not close"
    assert farm_state.positions["market-A"].orders_missing_ticks == 1

    await prune_dead_positions(MagicMock(), farm_state, AsyncMock())
    assert "market-A" not in farm_state.positions
    assert "yes-oid" not in farm_state.order_registry
    assert "no-oid" not in farm_state.order_registry
    reasons = [e.reason for e in events if isinstance(e, OrderCancelledEvent)]
    assert reasons == ["server_cancelled", "server_cancelled"], "one cancel event per leg"


async def test_held_shares_block_prune(farm_state: FarmState, monkeypatch):
    # Fixture holds 100 YES shares: orders were pulled by the exit flow, not lost.
    events: list = []
    wire_prune(monkeypatch, live_ids=set(), events=events)

    for _ in range(3):
        await prune_dead_positions(MagicMock(), farm_state, AsyncMock())

    pos = farm_state.positions["market-A"]
    assert pos.orders_missing_ticks == 0
    assert events == []


async def test_inflight_exit_blocks_prune(farm_state: FarmState, monkeypatch):
    clear_shares(farm_state)
    farm_state.positions["market-A"].exit_orders["exit-1"] = ExitOrder(
        outcome="YES", placed_at=datetime.now(timezone.utc)
    )
    events: list = []
    wire_prune(monkeypatch, live_ids=set(), events=events)

    for _ in range(3):
        await prune_dead_positions(MagicMock(), farm_state, AsyncMock())

    assert "market-A" in farm_state.positions
    assert events == []


async def test_poll_failure_skips_prune(farm_state: FarmState, monkeypatch):
    clear_shares(farm_state)
    farm_state.positions["market-A"].orders_missing_ticks = 1

    async def boom(client):
        raise RuntimeError("CLOB down")

    monkeypatch.setattr(worker_mod, "get_open_order_ids", boom)

    await prune_dead_positions(MagicMock(), farm_state, AsyncMock())

    pos = farm_state.positions["market-A"]
    assert pos.orders_missing_ticks == 1, "poll failure must not advance the counter"


# ── Fix C: a guard-pulled position is released on its own clock, not the missing-order strike ──


def make_pulled(state: FarmState, pulled_at, cid: str = "market-A") -> None:
    """Put a position into the clean guard-pulled state: no shares, ids cleared, latch set."""
    clear_shares(state, cid)
    pos = state.positions[cid]
    pos.quotes_pulled = True
    pos.quotes_pulled_at = pulled_at
    pos.yes_order_id = ""
    pos.no_order_id = ""


async def test_pulled_position_takes_no_strikes(farm_state: FarmState, monkeypatch):
    # A freshly guard-pulled position has no resting orders BY DESIGN — the pruner must not
    # misread that as "vanished server-side" and must not accrue missing-order strikes.
    make_pulled(farm_state, datetime.now(timezone.utc))
    events: list = []
    wire_prune(monkeypatch, live_ids=set(), events=events)

    for _ in range(3):
        await prune_dead_positions(MagicMock(), farm_state, AsyncMock())

    pos = farm_state.positions["market-A"]
    assert "market-A" in farm_state.positions, "a fresh guard-pull must not be closed"
    assert pos.orders_missing_ticks == 0, "a guard pull must not accrue missing-order strikes"
    assert events == []


async def test_pulled_position_released_after_window(farm_state: FarmState, monkeypatch):
    # Once nothing re-places within GUARD_PULL_RELEASE_SECONDS, the pruner releases the position
    # with reason="guard_pulled" (NOT server_cancelled), on the FIRST tick past the window — unlike
    # the two-tick missing-order path.
    make_pulled(
        farm_state,
        datetime.now(timezone.utc) - timedelta(seconds=GUARD_PULL_RELEASE_SECONDS + 1),
    )
    closed: list = []

    async def fake_close_position(client, state, websocket, cid, reason):
        closed.append((cid, reason))
        state.positions.pop(cid, None)

    monkeypatch.setattr(worker_mod, "close_position", fake_close_position)
    wire_prune(monkeypatch, live_ids=set(), events=[])

    await prune_dead_positions(MagicMock(), farm_state, AsyncMock())

    assert closed == [("market-A", "guard_pulled")], (
        "a stale guard-pull must be released as guard_pulled on the first tick past the window"
    )


async def test_pulled_without_timestamp_starts_clock(farm_state: FarmState, monkeypatch):
    # A latch set without a timestamp must START the release clock on this tick, not close.
    make_pulled(farm_state, None)
    events: list = []
    wire_prune(monkeypatch, live_ids=set(), events=events)

    await prune_dead_positions(MagicMock(), farm_state, AsyncMock())

    pos = farm_state.positions["market-A"]
    assert "market-A" in farm_state.positions, "the clock-starting tick must not close the position"
    assert pos.quotes_pulled_at is not None, "a latch with no timestamp must start the clock"
    assert events == []


async def test_reconcile_tick_prunes_then_requotes(
    farm_state: FarmState, market: Market, monkeypatch
):
    """Wiring: reconcile_tick runs the prune, and a pruned (still-eligible)
    market is re-opened in the same tick — the whole point of the sweep."""
    clear_shares(farm_state)
    events: list = []
    place_calls: list = []
    wire_prune(monkeypatch, live_ids=set(), events=events)

    async def fake_fetch_eligible_markets(http):
        return [market]

    async def fake_fetch_midpoints(http, token_ids):
        return {tid: Decimal("0.5") for tid in token_ids}

    async def fake_get_balance(addr):
        return Decimal("1000")

    async def fake_place_limit_order(client, order, post_only=False):
        place_calls.append(order.token_id)
        return f"oid-{len(place_calls)}"

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_fetch_eligible_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_fetch_midpoints)
    monkeypatch.setattr(worker_mod, "passes_all", lambda m, f: True)
    monkeypatch.setattr(worker_mod, "get_balance", fake_get_balance, raising=False)
    monkeypatch.setattr(worker_mod, "place_limit_order", fake_place_limit_order)

    # Tick 1: first missing sighting — still tracked, nothing re-placed.
    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())
    assert "market-A" in farm_state.positions
    assert place_calls == []

    # Tick 2: pruned as server_cancelled, then re-opened with fresh orders.
    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())
    assert set(place_calls) == {"tok-yes", "tok-no"}
    pos = farm_state.positions["market-A"]
    assert {pos.yes_order_id, pos.no_order_id} == {"oid-1", "oid-2"}
    cancel_reasons = [e.reason for e in events if isinstance(e, OrderCancelledEvent)]
    assert cancel_reasons == ["server_cancelled", "server_cancelled"]
