"""close_position cancel hardening (worker.py).

close_position now cancels each leg via cancel_order_with_retry and only drops the registry entry
(drop_order) once the cancel CONFIRMS. A leg whose cancel never confirms is checked against a
single get_open_order_ids fetch: confirmed-gone → dropped quietly; still-live or liveness-unknown
→ left in the registry and queued in pending_orphan_cancels for the reaper (with a close_orphan
strat line). The trailing per-leg OrderCancelledEvent emission is unchanged.

Note on drop_order: it POPS the registry entry (removes it), it does not stamp retired_at — so a
"dropped" leg below is one absent from order_registry, and an orphaned leg is one whose entry is
still PRESENT (retired_at still None).

Harness: stub worker_mod.cancel_order_with_retry (programmable per-oid), get_open_order_ids
(a set, or raising), worker_mod.send_event (recorder), worker_mod.strat (recorder). The conftest
farm_state pre-registers yes-oid/no-oid in order_registry.
"""

import logging
from unittest.mock import AsyncMock, MagicMock

from app.api.farm.messages import OrderCancelledEvent
from app.farm import worker as worker_mod
from app.farm.schemas import FarmState
from app.farm.worker import close_position


def wire_close(monkeypatch, cancel_results: dict, live_ids=None, live_raises: bool = False) -> dict:
    rec = {"calls": [], "events": [], "strats": []}

    async def fake_cancel_retry(client, oid):
        rec["calls"].append(oid)
        return cancel_results.get(oid, True)

    async def fake_get_open(client):
        if live_raises:
            raise RuntimeError("open-orders poll down")
        return set() if live_ids is None else set(live_ids)

    async def fake_send_event(websocket, event):
        rec["events"].append(event)

    def fake_strat(event, **fields):
        rec["strats"].append((event, fields))

    monkeypatch.setattr(worker_mod, "cancel_order_with_retry", fake_cancel_retry, raising=False)
    monkeypatch.setattr(worker_mod, "get_open_order_ids", fake_get_open, raising=False)
    monkeypatch.setattr(worker_mod, "send_event", fake_send_event)
    monkeypatch.setattr(worker_mod, "strat", fake_strat)
    return rec


def cancel_events(rec: dict) -> list:
    return [e for e in rec["events"] if isinstance(e, OrderCancelledEvent)]


def close_orphan_strats(rec: dict) -> list:
    return [f for (e, f) in rec["strats"] if e == "close_orphan"]


async def test_success_retires_registry_entries(farm_state: FarmState, monkeypatch):
    # Both legs confirm cancelled → each is retired (popped) only AFTER its confirmed cancel, via
    # cancel_order_with_retry. No leg is queued for the reaper; the liveness fetch is never needed.
    rec = wire_close(monkeypatch, cancel_results={})  # default True for both

    await close_position(MagicMock(), farm_state, AsyncMock(), "market-A", reason="market_dropped")

    assert rec["calls"] == ["yes-oid", "no-oid"], "both legs cancelled via the retry wrapper"
    assert "yes-oid" not in farm_state.order_registry, "a confirmed cancel retires (pops) the entry"
    assert "no-oid" not in farm_state.order_registry
    assert farm_state.pending_orphan_cancels == set(), "no orphan queued when both confirm"


async def test_failed_but_gone_is_clean(farm_state: FarmState, monkeypatch):
    # YES cancel never confirms, but the liveness fetch shows it's gone server-side anyway →
    # retired, NOT queued as an orphan.
    rec = wire_close(monkeypatch, cancel_results={"yes-oid": False}, live_ids=set())

    await close_position(MagicMock(), farm_state, AsyncMock(), "market-A", reason="market_dropped")

    assert farm_state.pending_orphan_cancels == set(), "a confirmed-gone leg is not an orphan"
    assert "yes-oid" not in farm_state.order_registry, "confirmed-gone still retires the entry"
    assert close_orphan_strats(rec) == [], "no close_orphan strat for a gone leg"


async def test_failed_and_live_queues_orphan(farm_state: FarmState, monkeypatch, caplog):
    # YES cancel never confirms AND liveness shows it still resting → queue it for the reaper and
    # keep its registry entry (so a late fill still routes).
    rec = wire_close(monkeypatch, cancel_results={"yes-oid": False}, live_ids={"yes-oid"})

    with caplog.at_level(logging.WARNING, logger="app.farm.worker"):
        await close_position(
            MagicMock(), farm_state, AsyncMock(), "market-A", reason="market_dropped"
        )

    assert "yes-oid" in farm_state.pending_orphan_cancels, "a still-live failed leg is queued"
    assert "no-oid" not in farm_state.pending_orphan_cancels
    assert "yes-oid" in farm_state.order_registry, "an orphan leg's registry entry is retained"
    assert farm_state.order_registry["yes-oid"].retired_at is None, "and left un-retired"
    assert any("yes-oid" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    strats = close_orphan_strats(rec)
    assert len(strats) == 1 and strats[0]["oid"] == "yes-oid" and strats[0]["live"] == "True"


async def test_liveness_fetch_failure_is_conservative(farm_state: FarmState, monkeypatch):
    # YES cancel never confirms and the liveness fetch itself fails → treat as possibly-live and
    # queue the orphan (live=unknown), never assume it's gone.
    rec = wire_close(monkeypatch, cancel_results={"yes-oid": False}, live_raises=True)

    await close_position(MagicMock(), farm_state, AsyncMock(), "market-A", reason="market_dropped")

    assert "yes-oid" in farm_state.pending_orphan_cancels, "an unverifiable leg is queued too"
    assert "yes-oid" in farm_state.order_registry, "retained until the reaper resolves it"
    strats = close_orphan_strats(rec)
    assert len(strats) == 1 and strats[0]["live"] == "unknown"


async def test_per_leg_attribution(farm_state: FarmState, monkeypatch):
    # YES fails (and is live) while NO succeeds → NO retired, only YES orphaned. Attribution must
    # be per-leg, not all-or-nothing.
    rec = wire_close(
        monkeypatch, cancel_results={"yes-oid": False, "no-oid": True}, live_ids={"yes-oid"}
    )

    await close_position(MagicMock(), farm_state, AsyncMock(), "market-A", reason="market_dropped")

    assert "no-oid" not in farm_state.order_registry, "the NO leg confirmed → retired"
    assert "no-oid" not in farm_state.pending_orphan_cancels
    assert farm_state.pending_orphan_cancels == {"yes-oid"}, "only the failed-live YES leg orphaned"
    assert "yes-oid" in farm_state.order_registry, "the orphaned leg is retained"
    assert [s["oid"] for s in close_orphan_strats(rec)] == ["yes-oid"]


async def test_cancel_events_still_emitted(farm_state: FarmState, monkeypatch):
    # The UI cancel events fire for both legs regardless of the orphan outcome (here YES orphans).
    rec = wire_close(monkeypatch, cancel_results={"yes-oid": False}, live_ids={"yes-oid"})

    await close_position(MagicMock(), farm_state, AsyncMock(), "market-A", reason="market_dropped")

    cancels = cancel_events(rec)
    assert len(cancels) == 2, "one cancel event per leg, orphan or not"
    assert all(e.reason == "market_dropped" for e in cancels)
    assert {e.outcome for e in cancels} == {"YES", "NO"}
