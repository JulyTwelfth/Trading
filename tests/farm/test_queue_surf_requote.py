"""Queue-surf re-quote wiring in app/farm/requote.py — the trade-feed trigger (point 2).

Covers: the `last_trade_price` frame is now parsed (no longer dropped), `handle_last_trade` flags
the leg whose resting price a trade hit (someone ahead of us filled → we moved up), and
`maybe_surf_requote` re-quotes only when the leg is old enough AND moved up AND a deep in-band
level still exists, and stands pat otherwise.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.market_ws import FRAME_MODELS
from app.bot.schemas import BestBidAsk, LastTradePrice
from app.farm import requote as requote_mod
from app.farm.requote import handle_bba, handle_last_trade, maybe_surf_requote
from app.farm.schemas import FarmState, LiveBook

D = Decimal


# ── frame parsing: the feed we used to discard ──────────────────────────────────
def test_frame_models_includes_last_trade_price():
    assert FRAME_MODELS["last_trade_price"] is LastTradePrice


def test_last_trade_price_parses_docs_payload():
    ev = LastTradePrice.model_validate(
        {
            "event_type": "last_trade_price",
            "asset_id": "tok-yes",
            "fee_rate_bps": "0",
            "market": "0xabc",
            "price": "0.456",
            "side": "BUY",
            "size": "219.217767",
            "timestamp": "1750428146322",
        }
    )
    assert ev.price == D("0.456")
    assert ev.size == D("219.217767")
    assert ev.asset_id == "tok-yes"


def test_last_trade_price_tolerates_missing_optional_fields():
    ev = LastTradePrice.model_validate(
        {"event_type": "last_trade_price", "asset_id": "x", "price": "0.5", "size": "10"}
    )
    assert ev.side == "" and ev.market == "" and ev.timestamp == ""


# ── handle_last_trade: flag the leg whose price the trade hit ────────────────────
def trade(asset_id: str, price: str) -> LastTradePrice:
    return LastTradePrice(
        event_type="last_trade_price", asset_id=asset_id, price=D(price), size=D(1)
    )


def test_trade_at_our_yes_price_flags_yes(farm_state: FarmState):
    pos = farm_state.positions["market-A"]  # yes_price 0.5, no_price 0.5
    handle_last_trade(pos, trade("tok-yes", "0.5"))
    assert pos.yes_moved_up is True
    assert pos.no_moved_up is False


def test_trade_at_other_price_does_not_flag(farm_state: FarmState):
    pos = farm_state.positions["market-A"]
    handle_last_trade(pos, trade("tok-yes", "0.49"))  # not our level — no one ahead of us filled
    assert pos.yes_moved_up is False


def test_trade_on_no_leg_flags_no_only(farm_state: FarmState):
    pos = farm_state.positions["market-A"]
    handle_last_trade(pos, trade("tok-no", "0.5"))
    assert pos.no_moved_up is True
    assert pos.yes_moved_up is False


# ── maybe_surf_requote: gating (requote_leg stubbed so no real orders are placed) ──
def aged(state: FarmState, oid: str, seconds: float) -> None:
    state.order_registry[oid].placed_at = datetime.now(timezone.utc) - timedelta(seconds=seconds)


@pytest.fixture
def surf(farm_state: FarmState, monkeypatch):
    """farm_state with a deep in-band YES book + a stubbed requote_leg; returns (state, rq_mock)."""
    farm_state.live_books["tok-yes"] = LiveBook(bids={D("0.50"): D(900)})  # 900 >= 0.2*100 → deep
    rq = AsyncMock()
    monkeypatch.setattr(requote_mod, "requote_leg", rq)
    return farm_state, rq


async def run(state: FarmState):
    pos = state.positions["market-A"]
    return await maybe_surf_requote(MagicMock(), state, AsyncMock(), pos, True, D("0.50"))


async def test_surf_requotes_to_deep_level_when_all_met(surf):
    state, rq = surf
    pos = state.positions["market-A"]
    pos.yes_moved_up = True
    aged(state, "yes-oid", 1000)  # past the 900s (15-min) hold
    assert await run(state) is True
    rq.assert_awaited_once()
    args = rq.await_args.args  # (client, state, ws, pos, outcome, price)
    assert args[-2] == "YES"
    assert args[-1] == D("0.50")


async def test_surf_skips_when_not_moved_up(surf):
    state, rq = surf
    state.positions["market-A"].yes_moved_up = False
    aged(state, "yes-oid", 1000)
    assert await run(state) is False
    rq.assert_not_awaited()


async def test_surf_skips_when_too_young(surf):
    state, rq = surf
    state.positions["market-A"].yes_moved_up = True
    aged(state, "yes-oid", 60)  # < 900s hold
    assert await run(state) is False
    rq.assert_not_awaited()


async def test_surf_skips_when_level_too_thin(surf):
    state, rq = surf
    state.live_books["tok-yes"] = LiveBook(bids={D("0.50"): D(10)})  # 10 < 0.2*100=20 → not deep
    state.positions["market-A"].yes_moved_up = True
    aged(state, "yes-oid", 1000)
    assert await run(state) is False
    rq.assert_not_awaited()


async def test_surf_skips_when_book_missing(surf):
    state, rq = surf
    state.live_books.pop("tok-yes", None)
    state.positions["market-A"].yes_moved_up = True
    aged(state, "yes-oid", 1000)
    assert await run(state) is False
    rq.assert_not_awaited()


# ── never re-quote OUT of the reward band, and never disturb state when inactive ──
async def test_surf_never_requotes_out_of_band(surf):
    state, rq = surf
    # only depth is a huge level OUTSIDE the band (0.46 vs mid 0.50, 3c band) → must not surf there
    state.live_books["tok-yes"] = LiveBook(bids={D("0.46"): D(100000)})
    state.positions["market-A"].yes_moved_up = True
    aged(state, "yes-oid", 1000)
    assert await run(state) is False
    rq.assert_not_awaited()


async def test_surf_picked_price_is_within_band(surf):
    state, rq = surf  # book {0.50: 900}, in-band
    state.positions["market-A"].yes_moved_up = True
    aged(state, "yes-oid", 1000)
    assert await run(state) is True
    price = rq.await_args.args[-1]
    assert D("0.50") - price < D("0.03")  # the re-quote stays inside the reward band


async def test_surf_inactive_leaves_state_untouched(surf):
    state, rq = surf
    pos = state.positions["market-A"]
    pos.yes_moved_up = False  # inactive — nobody ahead filled
    aged(state, "yes-oid", 1000)
    before = (pos.yes_price, pos.yes_order_id, pos.no_price, pos.no_order_id)
    assert await run(state) is False
    rq.assert_not_awaited()
    assert pos.yes_moved_up is False  # no flags flipped
    assert (pos.yes_price, pos.yes_order_id, pos.no_price, pos.no_order_id) == before


# ── handle_bba integration: surf is a parallel trigger that never overrides a protective pull ──
@pytest.fixture
def bba_stub(monkeypatch):
    """Stub both network calls handle_bba can make, recording which fired — so we can tell a
    protective pull (cancel) apart from a surf re-quote without touching the network."""
    rec: dict = {"cancelled": [], "requotes": []}

    async def fake_cancel(client, oid):
        rec["cancelled"].append(oid)

    async def fake_cancel_orders(client, *oids):
        rec["cancelled"].extend(o for o in oids if o)

    async def fake_requote_leg(client, state, ws, pos, outcome, price):
        rec["requotes"].append((outcome, float(price)))

    monkeypatch.setattr(requote_mod, "cancel_order", fake_cancel)
    monkeypatch.setattr(requote_mod, "cancel_orders", fake_cancel_orders)
    monkeypatch.setattr(requote_mod, "requote_leg", fake_requote_leg)
    return rec


def bba(asset_id: str, best_bid: str, best_ask: str) -> BestBidAsk:
    return BestBidAsk(
        event_type="best_bid_ask",
        market="market-A",
        asset_id=asset_id,
        best_bid=D(best_bid),
        best_ask=D(best_ask),
        spread=D(best_ask) - D(best_bid),
        timestamp="2026-06-20T12:00:00Z",
    )


async def test_gap_pull_preempts_surf(farm_state: FarmState, bba_stub):
    pos = farm_state.positions["market-A"]
    pos.yes_shares = D(0)  # isolate the gap-pull from the mark-to-market kill
    pos.yes_cost_basis = D(0)
    yes = pos.market.yes_token_id
    farm_state.live_books[yes] = LiveBook(bids={D("0.49"): D(900)})  # surf-ready deep in-band book
    # frame 1 records the prior bid (moved_up still off → surf can't fire here)
    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba(yes, "0.50", "0.52"))
    bba_stub["cancelled"].clear()
    bba_stub["requotes"].clear()
    # arm surf, then send a crashing bid — the gap-pull must win
    pos.yes_moved_up = True
    aged(farm_state, "yes-oid", 1000)
    yes_oid, no_oid = pos.yes_order_id, pos.no_order_id
    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba(yes, "0.40", "0.52"))
    assert set(bba_stub["cancelled"]) == {yes_oid, no_oid}
    assert bba_stub["requotes"] == [], "gap-pull must preempt the surf re-quote"


async def test_surf_fires_through_handle_bba_when_no_pull(farm_state: FarmState, bba_stub):
    pos = farm_state.positions["market-A"]
    pos.yes_shares = D(0)
    pos.yes_cost_basis = D(0)
    yes = pos.market.yes_token_id
    farm_state.live_books[yes] = LiveBook(bids={D("0.49"): D(900)})
    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba(yes, "0.50", "0.52"))
    bba_stub["requotes"].clear()
    pos.yes_moved_up = True
    aged(farm_state, "yes-oid", 1000)
    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba(yes, "0.50", "0.52"))
    assert bba_stub["requotes"] == [("YES", 0.49)], "surf re-quotes to the in-band level"
