"""Fix C — a clean guard pull leaves consistent state.

cancel_position_orders now batch-cancels both resting legs and leaves the position in a coherent
"pulled" state: the stored order ids are cleared to "", both order-registry entries get a
retired_at (starting the retention clock), quotes_pulled + quotes_pulled_at are latched, and a
per-leg OrderCancelledEvent(reason="guard_pulled") is emitted for each TRUTHY id. A later
requote_leg on a pulled leg (old_oid == "") skips the pre-cancel + "requote" cancel event entirely
and just places the replacement, clearing the latch.

Harness: monkeypatch requote_mod.cancel_orders (the batch-cancel) and requote_mod.send_event (the
UI hook — the real send_event no-ops on an AsyncMock websocket, so we replace it with a recorder).
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.farm.messages import OrderCancelledEvent
from app.bot.schemas import BookSnapshot
from app.farm import requote as requote_mod
from app.farm.requote import cancel_position_orders, handle_book_snapshot, requote_leg
from app.farm.schemas import FarmState


@pytest.fixture
def pull_recorders(monkeypatch):
    cancelled: list = []
    events: list = []

    async def fake_cancel_orders(client, *ids):
        cancelled.extend(ids)

    async def fake_send_event(websocket, event):
        events.append(event)

    monkeypatch.setattr(requote_mod, "cancel_orders", fake_cancel_orders)
    monkeypatch.setattr(requote_mod, "send_event", fake_send_event)
    return {"cancelled": cancelled, "events": events}


async def test_pull_clears_ids_and_latches(farm_state: FarmState, pull_recorders):
    pos = farm_state.positions["market-A"]
    yes_oid, no_oid = pos.yes_order_id, pos.no_order_id  # capture BEFORE the pull clears them

    await cancel_position_orders(MagicMock(), farm_state, AsyncMock(), pos)

    assert pos.yes_order_id == "" and pos.no_order_id == "", "stored ids must be cleared"
    assert pos.quotes_pulled is True
    assert pos.quotes_pulled_at is not None, "the pull must latch a timestamp"
    assert farm_state.order_registry[yes_oid].retired_at is not None, "YES registry entry retired"
    assert farm_state.order_registry[no_oid].retired_at is not None, "NO registry entry retired"
    assert set(pull_recorders["cancelled"]) == {yes_oid, no_oid}, "both legs batch-cancelled"


async def test_pull_emits_cancel_events(farm_state: FarmState, pull_recorders):
    pos = farm_state.positions["market-A"]
    yes_oid, no_oid = pos.yes_order_id, pos.no_order_id

    await cancel_position_orders(MagicMock(), farm_state, AsyncMock(), pos)

    cancels = [e for e in pull_recorders["events"] if isinstance(e, OrderCancelledEvent)]
    assert len(cancels) == 2, "one cancel event per truthy leg"
    assert all(e.reason == "guard_pulled" for e in cancels), "reason must be guard_pulled"
    by_outcome = {e.outcome: e.order_id for e in cancels}
    assert by_outcome == {"YES": yes_oid, "NO": no_oid}, "correct oid per outcome"


async def test_pull_skips_events_for_empty_ids(farm_state: FarmState, pull_recorders):
    pos = farm_state.positions["market-A"]
    pos.yes_order_id = ""
    pos.no_order_id = ""

    await cancel_position_orders(MagicMock(), farm_state, AsyncMock(), pos)

    assert pull_recorders["events"] == [], "no truthy id → no cancel event"
    assert pos.quotes_pulled is True, "the latch is still set even with nothing to cancel"
    assert pos.quotes_pulled_at is not None


async def test_requote_leg_skips_cancel_for_pulled_leg(farm_state: FarmState, monkeypatch):
    cancel_calls: list = []
    place_calls: list = []

    async def spy_cancel_order(client, oid):
        cancel_calls.append(oid)

    async def fake_place_limit_order(client, order, post_only=False):
        place_calls.append((order.token_id, float(order.price)))
        return "new-oid"

    async def fake_send_event(websocket, event):
        return None

    monkeypatch.setattr(requote_mod, "cancel_order", spy_cancel_order)
    monkeypatch.setattr(requote_mod, "place_limit_order", fake_place_limit_order)
    monkeypatch.setattr(requote_mod, "send_event", fake_send_event)

    pos = farm_state.positions["market-A"]
    pos.yes_order_id = ""  # this leg was already pulled
    pos.quotes_pulled = True
    pos.quotes_pulled_at = datetime.now(timezone.utc)
    pos.yes_shares = Decimal("0")
    pos.yes_cost_basis = Decimal("0")

    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.31"))

    assert cancel_calls == [], "a pulled leg (old_oid == '') has nothing to pre-cancel"
    assert place_calls == [("tok-yes", 0.31)], "the replacement must still be placed"
    assert pos.yes_order_id == "new-oid"
    assert pos.quotes_pulled is False, "a successful re-place clears the pull latch"
    assert pos.quotes_pulled_at is None


async def test_depth_guard_pull_emits_ui_event(farm_state: FarmState, pull_recorders):
    # End-to-end: a thin live book drives handle_book_snapshot -> depth guard -> clean pull, and the
    # UI sees a guard_pulled cancel per leg.
    farm_state.config.filters.max_fill_loss = Decimal("1")
    pos = farm_state.positions["market-A"]
    yes_oid, no_oid = pos.yes_order_id, pos.no_order_id
    snap = BookSnapshot.model_validate(
        {
            "event_type": "book",
            "asset_id": "tok-yes",
            "market": "market-A",
            "bids": [{"price": "0.50", "size": "10"}],  # only 10 vs our 100 → ~$45 loss >> $1 cap
            "asks": [{"price": "0.52", "size": "5"}],
            "timestamp": "1",
            "hash": "h",
        }
    )

    await handle_book_snapshot(MagicMock(), farm_state, AsyncMock(), pos, snap)

    assert pos.quotes_pulled is True
    assert pos.yes_order_id == "" and pos.no_order_id == "", "the guard pull clears the ids"
    cancels = [e for e in pull_recorders["events"] if isinstance(e, OrderCancelledEvent)]
    assert {(e.outcome, e.order_id, e.reason) for e in cancels} == {
        ("YES", yes_oid, "guard_pulled"),
        ("NO", no_oid, "guard_pulled"),
    }
