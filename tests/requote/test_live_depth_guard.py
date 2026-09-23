"""Tier 1.5 — depth-aware real-time fill-loss guard fed by the live market-WS book.

The market WS streams full depth (`book` snapshot + `price_change` deltas) in real time (measured
~60 frames/s across 120 markets). We now maintain a live in-memory book per token and walk its
ACTUAL depth with immediate_sell_loss on every update — instead of the depth-blind top-of-book
best_bid guard that let the big crash loss through. The guard is gated on max_fill_loss (like the
top-of-book one); the always-on gap-pull + mark-to-market kill cover the unset case.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.schemas import BookSnapshot, PriceChange
from app.farm import requote as requote_mod
from app.farm.requote import (
    apply_book_snapshot,
    apply_price_change,
    depth_fill_loss_guard,
    handle_price_change,
)
from app.farm.schemas import FarmState, LiveBook


@pytest.fixture
def cancels(monkeypatch):
    out: list = []

    async def fake_cancel(client, oid):
        out.append(oid)

    async def fake_cancel_orders(client, *ids):
        # Tier 2: the depth guard pull now batches via cancel_orders; funnel its ids into the SAME
        # recorder so the existing {yes_oid, no_oid} assertions hold.
        out.extend(i for i in ids if i)

    monkeypatch.setattr(requote_mod, "cancel_order", fake_cancel)
    monkeypatch.setattr(requote_mod, "cancel_orders", fake_cancel_orders)
    return out


# ── WS frame parsing (the frames we used to discard) ──────────────────────────


def test_book_frame_parses():
    frame = {
        "event_type": "book",
        "asset_id": "tok-yes",
        "market": "market-A",
        "bids": [{"price": ".48", "size": "30"}, {"price": ".50", "size": "15"}],
        "asks": [{"price": ".52", "size": "25"}],
        "timestamp": "123",
        "hash": "0xabc",
    }
    snap = BookSnapshot.model_validate(frame)
    assert snap.asset_id == "tok-yes"
    assert snap.bids[0].price == Decimal(".48") and snap.bids[0].size == Decimal("30")


def test_price_change_frame_parses_multi_token():
    frame = {
        "event_type": "price_change",
        "market": "market-A",
        "timestamp": "123",
        "price_changes": [
            {
                "asset_id": "tok-yes",
                "price": "0.5",
                "size": "200",
                "side": "BUY",
                "hash": "h",
                "best_bid": "0.5",
                "best_ask": "1",
            },
            {"asset_id": "tok-no", "price": "0.5", "size": "0", "side": "SELL", "hash": "h2"},
        ],
    }
    pc = PriceChange.model_validate(frame)
    assert {c.asset_id for c in pc.price_changes} == {"tok-yes", "tok-no"}
    assert pc.price_changes[1].size == Decimal("0")  # a removal


# ── live-book maintenance ─────────────────────────────────────────────────────


def test_apply_book_snapshot_seeds_book(farm_state: FarmState):
    snap = BookSnapshot.model_validate(
        {
            "event_type": "book",
            "asset_id": "tok-yes",
            "market": "market-A",
            "bids": [{"price": "0.50", "size": "15"}, {"price": "0.49", "size": "20"}],
            "asks": [{"price": "0.52", "size": "25"}],
            "timestamp": "1",
            "hash": "h",
        }
    )
    apply_book_snapshot(farm_state, snap)
    book = farm_state.live_books["tok-yes"]
    assert book.bids == {Decimal("0.50"): Decimal("15"), Decimal("0.49"): Decimal("20")}
    assert book.asks == {Decimal("0.52"): Decimal("25")}


def test_apply_price_change_updates_and_removes_levels(farm_state: FarmState):
    farm_state.live_books["tok-yes"] = LiveBook(bids={Decimal("0.50"): Decimal("15")})

    # Update an existing level.
    chg = PriceChange.model_validate(
        {
            "event_type": "price_change",
            "market": "market-A",
            "timestamp": "1",
            "price_changes": [
                {"asset_id": "tok-yes", "price": "0.50", "size": "40", "side": "BUY"}
            ],
        }
    ).price_changes[0]
    apply_price_change(farm_state, chg)
    assert farm_state.live_books["tok-yes"].bids[Decimal("0.50")] == Decimal("40")

    # size 0 removes the level (the crash signal: bids vanishing).
    chg0 = PriceChange.model_validate(
        {
            "event_type": "price_change",
            "market": "market-A",
            "timestamp": "2",
            "price_changes": [{"asset_id": "tok-yes", "price": "0.50", "size": "0", "side": "BUY"}],
        }
    ).price_changes[0]
    apply_price_change(farm_state, chg0)
    assert Decimal("0.50") not in farm_state.live_books["tok-yes"].bids


# ── the depth guard ───────────────────────────────────────────────────────────


async def test_thin_book_pulls_both_legs(farm_state: FarmState, cancels):
    # size_per_market = 100, our YES BUY @ 0.50. Live bids hold only 10 @ 0.50 → selling 100 dumps
    # 90 into nothing: loss = 100*0.50 - 10*0.50 = $45 >> $1 cap → pull.
    farm_state.config.filters.max_fill_loss = Decimal("1")
    farm_state.live_books["tok-yes"] = LiveBook(bids={Decimal("0.50"): Decimal("10")})
    pos = farm_state.positions["market-A"]
    yes_oid, no_oid = pos.yes_order_id, pos.no_order_id

    pulled = await depth_fill_loss_guard(MagicMock(), farm_state, AsyncMock(), pos, "tok-yes")

    assert pulled is True
    assert set(cancels) == {yes_oid, no_oid}


async def test_deep_book_does_not_pull(farm_state: FarmState, cancels):
    # 200 @ 0.50 absorbs our 100 fully → $0 loss ≤ cap → keep quoting.
    farm_state.config.filters.max_fill_loss = Decimal("1")
    farm_state.live_books["tok-yes"] = LiveBook(bids={Decimal("0.50"): Decimal("200")})
    pos = farm_state.positions["market-A"]

    ws = AsyncMock()
    assert await depth_fill_loss_guard(MagicMock(), farm_state, ws, pos, "tok-yes") is False
    assert cancels == []


async def test_guard_off_when_max_fill_loss_unset(farm_state: FarmState, cancels):
    # Default fixture has max_fill_loss=None → guard inert even on an empty book.
    farm_state.live_books["tok-yes"] = LiveBook(bids={Decimal("0.50"): Decimal("1")})
    pos = farm_state.positions["market-A"]
    ws = AsyncMock()
    assert await depth_fill_loss_guard(MagicMock(), farm_state, ws, pos, "tok-yes") is False
    assert cancels == []


async def test_guard_noop_without_live_book(farm_state: FarmState, cancels):
    farm_state.config.filters.max_fill_loss = Decimal("1")
    pos = farm_state.positions["market-A"]
    # No book seeded yet for the token → nothing to walk → no pull.
    ws = AsyncMock()
    assert await depth_fill_loss_guard(MagicMock(), farm_state, ws, pos, "tok-yes") is False
    assert cancels == []


async def test_crash_via_price_change_stream_triggers_pull(farm_state: FarmState, cancels):
    # End-to-end: a healthy book, then a stream of price_change removals collapse the bids → the
    # depth guard fires off the live feed (no REST fetch, no 60s staleness).
    farm_state.config.filters.max_fill_loss = Decimal("1")
    yes = farm_state.positions["market-A"].market.yes_token_id
    farm_state.live_books[yes] = LiveBook(
        bids={Decimal("0.50"): Decimal("200"), Decimal("0.49"): Decimal("200")}
    )

    # Healthy book first — no pull.
    assert (
        await depth_fill_loss_guard(
            MagicMock(), farm_state, AsyncMock(), farm_state.positions["market-A"], yes
        )
        is False
    )

    # Crash: both deep bid levels get pulled (size 0), leaving a thin 5 @ 0.20.
    pos = farm_state.positions["market-A"]
    yes_oid, no_oid = pos.yes_order_id, pos.no_order_id
    crash = PriceChange.model_validate(
        {
            "event_type": "price_change",
            "market": "market-A",
            "timestamp": "9",
            "price_changes": [
                {"asset_id": yes, "price": "0.50", "size": "0", "side": "BUY"},
                {"asset_id": yes, "price": "0.49", "size": "0", "side": "BUY"},
                {"asset_id": yes, "price": "0.20", "size": "5", "side": "BUY"},
            ],
        }
    )
    await handle_price_change(MagicMock(), farm_state, AsyncMock(), crash)

    assert set(cancels) == {yes_oid, no_oid}, "live depth collapse must pull orders"
