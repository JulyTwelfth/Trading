"""Orphan reaper — two-sweep cleanup of untracked resting orders (worker.py).

reap_orphan_orders cancels: (a) close_position's queued orphans (pending_orphan_cancels) that are
still live — every sweep until they are confirmed gone (the queue is NOT drained on a sweep that
still sees them resting); and (b) orders it notices on its own (live but tracked by nothing) — only
after they persist across TWO consecutive sweeps, debouncing a one-tick race against a just-placed
order not yet registered. reconcile_tick hoists a single get_open_order_ids and feeds it to both
prune_dead_positions (optional live_ids param) and the reaper.

"Tracked" is the union of FIVE sources — the order_registry (any entry, retired or not),
pending_exit_order_ids, and per position its yes/no order ids and its exit_orders keys — so no
legitimately-tracked order (a resting BUY leg OR a resting exit SELL) is ever reaped.

A queued orphan that is no longer resting is "resolved": dropped from the queue, and its registry
entry (if any) is stamped retired_at so a late fill still routes for 120s before prune reaps it.

Harness: call reap_orphan_orders directly with crafted live sets; stub worker_mod.cancel_orders and
worker_mod.strat as recorders.
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.farm import worker as worker_mod
from app.farm.schemas import ExitOrder, FarmState, OrderInfo
from app.farm.worker import prune_dead_positions, reap_orphan_orders


def wire_reaper(monkeypatch) -> dict:
    rec = {"cancelled": [], "strats": []}

    async def fake_cancel_orders(client, *ids):
        rec["cancelled"].extend(ids)

    def fake_strat(event, **fields):
        rec["strats"].append((event, fields))

    monkeypatch.setattr(worker_mod, "cancel_orders", fake_cancel_orders)
    monkeypatch.setattr(worker_mod, "strat", fake_strat)
    return rec


def reaped_strats(rec: dict) -> list:
    return [f for (e, f) in rec["strats"] if e == "orphan_reaped"]


def registry_entry(cid: str = "market-A") -> OrderInfo:
    return OrderInfo(condition_id=cid, outcome="YES", token_id="tok")


# ── reaper-discovered unknowns (two-sweep debounce) ─────────────────────────────


async def test_unknown_needs_two_sweeps(farm_state: FarmState, monkeypatch, caplog):
    # A live id tracked by nothing is NOT cancelled on first sight — only after it survives a second
    # sweep still untracked (debounces a just-placed-but-unregistered order).
    rec = wire_reaper(monkeypatch)
    live = {"orphan-1"}

    await reap_orphan_orders(MagicMock(), farm_state, live)
    assert rec["cancelled"] == [], "first sweep must not cancel a freshly-noticed unknown"
    assert farm_state.reaper_unknown_ids == {"orphan-1"}, "it is remembered for the next sweep"

    with caplog.at_level(logging.WARNING, logger="app.farm.worker"):
        await reap_orphan_orders(MagicMock(), farm_state, live)
    assert rec["cancelled"] == ["orphan-1"], "a twice-seen unknown is reaped"
    strats = reaped_strats(rec)
    assert len(strats) == 1 and strats[0]["oid"] == "orphan-1"
    assert strats[0]["path"] == "reaper_sweep"
    assert any("orphan-1" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


async def test_stubborn_unknown_retries_every_sweep(farm_state: FarmState, monkeypatch):
    # A twice-seen unknown whose (swallowed) batch cancel fails stays remembered, so it is
    # re-cancelled on EVERY subsequent sweep — not first-sighted again and stalled for two more.
    rec = wire_reaper(monkeypatch)
    live = {"stubborn-1"}  # never leaves the live set: every cancel silently fails

    await reap_orphan_orders(MagicMock(), farm_state, live)  # sweep 1: noticed
    await reap_orphan_orders(MagicMock(), farm_state, live)  # sweep 2: confirmed, cancelled
    await reap_orphan_orders(MagicMock(), farm_state, live)  # sweep 3: must retry immediately

    assert rec["cancelled"] == ["stubborn-1", "stubborn-1"], (
        "a stubborn unknown is retried on sweep 3, not demoted to a fresh first sighting"
    )


async def test_unknown_registered_between_sweeps_survives(farm_state: FarmState, monkeypatch):
    # An id unknown on sweep 1 that becomes tracked before sweep 2 must NOT be reaped — the debounce
    # exists exactly for this just-placed-but-unregistered race.
    rec = wire_reaper(monkeypatch)
    live = {"late-oid"}

    await reap_orphan_orders(MagicMock(), farm_state, live)
    assert farm_state.reaper_unknown_ids == {"late-oid"}

    farm_state.positions["market-A"].yes_order_id = "late-oid"  # adopted as a real leg

    await reap_orphan_orders(MagicMock(), farm_state, live)
    assert rec["cancelled"] == [], "a now-tracked id must not be reaped"
    assert "late-oid" not in farm_state.reaper_unknown_ids, "and it is dropped from reaper memory"


# ── the five tracking sources are all protected ─────────────────────────────────


async def test_tracked_ids_never_reaped(farm_state: FarmState, monkeypatch):
    # None of the five tracking sources may be reaped, even across repeated sweeps.
    rec = wire_reaper(monkeypatch)
    pos = farm_state.positions["market-A"]
    now = datetime.now(timezone.utc)

    # (a) a position leg id — yes-oid is already pos.yes_order_id (+ a conftest registry entry).
    # (b) a per-position exit_orders key.
    pos.exit_orders["exit-key"] = ExitOrder(outcome="YES", placed_at=now)
    # (c) a pending_exit_order_ids member.
    farm_state.pending_exit_order_ids.add("pending-exit")
    # (d) an UNRETIRED registry-only entry.
    farm_state.order_registry["reg-unretired"] = registry_entry()
    # (e) a RETIRED registry-only entry.
    retired = registry_entry()
    retired.retired_at = now
    farm_state.order_registry["reg-retired"] = retired

    live = {"yes-oid", "exit-key", "pending-exit", "reg-unretired", "reg-retired"}
    for _ in range(2):
        await reap_orphan_orders(MagicMock(), farm_state, live)

    assert rec["cancelled"] == [], "no id from any of the five tracking sources may be reaped"
    assert farm_state.reaper_unknown_ids == set()


async def test_live_exit_order_not_reaped(farm_state: FarmState, monkeypatch):
    # A resting exit SELL is tracked via pending_exit_order_ids + the position's exit_orders (never
    # via yes/no_order_id). It is a legitimate order and must never be reaped.
    rec = wire_reaper(monkeypatch)
    farm_state.pending_exit_order_ids.add("exit-1")
    farm_state.positions["market-A"].exit_orders["exit-1"] = ExitOrder(
        outcome="YES", placed_at=datetime.now(timezone.utc)
    )
    live = {"exit-1"}

    await reap_orphan_orders(MagicMock(), farm_state, live)
    await reap_orphan_orders(MagicMock(), farm_state, live)

    assert rec["cancelled"] == [], "a live exit-SELL order must never be reaped"


# ── close-fail orphans: retried every sweep, dropped only when confirmed gone ───


async def test_close_fail_orphan_reaped_first_sweep(farm_state: FarmState, monkeypatch):
    # A close_position-queued orphan that is still live is cancelled on the FIRST sweep (liveness
    # was already verified during close_position), and STAYS queued (no unconditional drain) so it
    # is retried until confirmed gone. A later sweep where it's gone resolves the queue.
    rec = wire_reaper(monkeypatch)
    farm_state.pending_orphan_cancels.add("cf-1")

    await reap_orphan_orders(MagicMock(), farm_state, {"cf-1"})
    assert rec["cancelled"] == ["cf-1"], "a queued, still-live orphan is reaped immediately"
    assert "cf-1" in farm_state.pending_orphan_cancels, "stays queued — the queue is not drained"
    strats = reaped_strats(rec)
    assert len(strats) == 1 and strats[0]["path"] == "close_fail"

    await reap_orphan_orders(MagicMock(), farm_state, set())
    assert farm_state.pending_orphan_cancels == set(), "confirmed-gone finally resolves the queue"


async def test_still_live_orphan_stays_queued(farm_state: FarmState, monkeypatch):
    # Pins the drain-removal: a persistently-live queued orphan is re-cancelled every sweep and
    # never silently dropped while it is still resting.
    rec = wire_reaper(monkeypatch)
    farm_state.pending_orphan_cancels.add("cf-1")
    live = {"cf-1"}

    await reap_orphan_orders(MagicMock(), farm_state, live)
    await reap_orphan_orders(MagicMock(), farm_state, live)

    assert "cf-1" in farm_state.pending_orphan_cancels, "a still-live orphan is retried, kept"
    assert rec["cancelled"] == ["cf-1", "cf-1"], "re-cancelled every sweep until confirmed gone"


async def test_resolved_orphan_retired_and_dropped(farm_state: FarmState, monkeypatch):
    # A queued orphan that is no longer resting is resolved: dropped from the queue AND its registry
    # entry stamped retired_at (the leak fix) so a late fill still routes for 120s, then prune reaps
    # it. No cancel is issued.
    rec = wire_reaper(monkeypatch)
    farm_state.order_registry["cf-1"] = registry_entry()
    assert farm_state.order_registry["cf-1"].retired_at is None
    farm_state.pending_orphan_cancels.add("cf-1")

    await reap_orphan_orders(MagicMock(), farm_state, set())  # cf-1 no longer live

    assert "cf-1" not in farm_state.pending_orphan_cancels, "a resolved orphan is dropped"
    assert farm_state.order_registry["cf-1"].retired_at is not None, "its registry entry is retired"
    assert rec["cancelled"] == [], "a resolved orphan is not cancelled"


# ── prune accepts the hoisted live set ──────────────────────────────────────────


async def test_prune_accepts_hoisted_live_ids(farm_state: FarmState, monkeypatch):
    # prune_dead_positions accepts a hoisted live_ids set and must NOT self-fetch when it is passed
    # (reconcile_tick fetches once and feeds both the prune and the reaper).
    fetched: list = []

    async def boom(client):
        fetched.append(1)
        raise RuntimeError("must not self-fetch when live_ids is provided")

    monkeypatch.setattr(worker_mod, "get_open_order_ids", boom, raising=False)

    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("0")
    pos.no_shares = Decimal("0")
    pos.orders_missing_ticks = 1

    await prune_dead_positions(MagicMock(), farm_state, AsyncMock(), live_ids={"yes-oid"})

    assert fetched == [], "a provided live_ids must be used, not a fresh self-fetch"
    assert "market-A" in farm_state.positions, "a leg live in the passed set keeps the position"
    assert pos.orders_missing_ticks == 0, "a live leg resets the missing-tick counter"
