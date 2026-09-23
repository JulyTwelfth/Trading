"""Real-time exit-loss guard in handle_bba: when the live best bid falls far enough below
our resting BUY that a fill + immediate dump would lose more than max_fill_loss, CANCEL both
resting legs so we can't be filled on unfavorable terms. It must NOT pause the market — a
pause would block the instant exit of shares we already hold, and the market should re-enter
on its own once the spread recovers. Off (max_fill_loss=None) = original behaviour unchanged."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.schemas import BestBidAsk
from app.farm import requote as requote_mod
from app.farm.health import is_paused
from app.farm.requote import handle_bba
from app.farm.schemas import FarmState


@pytest.fixture
def stub_network(monkeypatch):
    cancelled: list = []
    requotes: list = []

    async def fake_cancel(client, oid):
        cancelled.append(oid)

    async def fake_cancel_orders(client, *ids):
        # Tier 2: the guard pull now batches via cancel_orders; funnel its ids into the SAME
        # recorder so the existing {yes_oid, no_oid} assertions hold. requote_leg still uses the
        # singular cancel_order above.
        cancelled.extend(i for i in ids if i)

    async def fake_requote_leg(client, state, ws, pos, outcome, price):
        requotes.append((outcome, float(price)))

    monkeypatch.setattr(requote_mod, "cancel_order", fake_cancel)
    monkeypatch.setattr(requote_mod, "cancel_orders", fake_cancel_orders)
    monkeypatch.setattr(requote_mod, "requote_leg", fake_requote_leg)
    return {"cancelled": cancelled, "requotes": requotes}


def make_bba(asset_id: str, best_bid: Decimal, best_ask: Decimal) -> BestBidAsk:
    return BestBidAsk(
        event_type="best_bid_ask",
        market="market-A",
        asset_id=asset_id,
        best_bid=best_bid,
        best_ask=best_ask,
        spread=best_ask - best_bid,
        timestamp="2026-06-01T12:00:00Z",
    )


async def test_wide_gap_on_yes_leg_cancels_without_pausing(farm_state: FarmState, stub_network):
    # size=100, our yes_price=0.50. best_bid 0.44 → est loss 100*(0.50-0.44)=$6 > $1 cap.
    farm_state.config.filters.max_fill_loss = Decimal("1")
    pos = farm_state.positions["market-A"]
    # No held inventory: isolate the exit-loss guard from the mark-to-market kill, which a
    # held, underwater position (the fixture's 100 shares) would trip first.
    pos.yes_shares = Decimal("0")
    pos.yes_cost_basis = Decimal("0")
    yes_oid, no_oid = pos.yes_order_id, pos.no_order_id

    await handle_bba(
        MagicMock(),
        farm_state,
        AsyncMock(),
        pos,
        make_bba(pos.market.yes_token_id, Decimal("0.44"), Decimal("0.56")),
    )

    assert set(stub_network["cancelled"]) == {yes_oid, no_oid}
    assert is_paused(farm_state, "market-A") is False  # must NOT pause
    assert stub_network["requotes"] == []  # cancelled the bad market, did not re-quote it


async def test_wide_gap_on_no_leg_also_cancels(farm_state: FarmState, stub_network):
    farm_state.config.filters.max_fill_loss = Decimal("1")
    pos = farm_state.positions["market-A"]
    yes_oid, no_oid = pos.yes_order_id, pos.no_order_id

    await handle_bba(
        MagicMock(),
        farm_state,
        AsyncMock(),
        pos,
        make_bba(pos.market.no_token_id, Decimal("0.44"), Decimal("0.56")),
    )

    assert set(stub_network["cancelled"]) == {yes_oid, no_oid}
    assert is_paused(farm_state, "market-A") is False


async def test_empty_bid_book_cancels(farm_state: FarmState, stub_network):
    # best_bid 0 (bid side vanished) → est loss = size * our_price = full position → cancel.
    farm_state.config.filters.max_fill_loss = Decimal("1")
    pos = farm_state.positions["market-A"]
    # No held inventory: isolate the exit-loss guard from the mark-to-market kill (a held
    # position marked at best_bid=0 would otherwise trip the session cap first).
    pos.yes_shares = Decimal("0")
    pos.yes_cost_basis = Decimal("0")
    yes_oid, no_oid = pos.yes_order_id, pos.no_order_id

    await handle_bba(
        MagicMock(),
        farm_state,
        AsyncMock(),
        pos,
        make_bba(pos.market.yes_token_id, Decimal("0"), Decimal("0.10")),
    )

    assert set(stub_network["cancelled"]) == {yes_oid, no_oid}
    assert is_paused(farm_state, "market-A") is False


async def test_cancels_before_any_fill_when_no_shares_held(farm_state: FarmState, stub_network):
    # Guard pulls resting BUYs to PREVENT the bad fill — independent of holding shares.
    farm_state.config.filters.max_fill_loss = Decimal("1")
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("0")
    pos.no_shares = Decimal("0")
    yes_oid, no_oid = pos.yes_order_id, pos.no_order_id

    await handle_bba(
        MagicMock(),
        farm_state,
        AsyncMock(),
        pos,
        make_bba(pos.market.yes_token_id, Decimal("0.44"), Decimal("0.56")),
    )

    assert set(stub_network["cancelled"]) == {yes_oid, no_oid}
    assert is_paused(farm_state, "market-A") is False


async def test_held_shares_untouched_on_cancel(farm_state: FarmState, stub_network):
    # The guard pulls resting orders only; held inventory is left for the instant exit flow.
    farm_state.config.filters.max_fill_loss = Decimal("1")
    # Raise the session cap so the mark-to-market kill doesn't preempt the guard here — this
    # test is about the guard leaving held inventory untouched, not the kill switch.
    farm_state.config.max_session_loss = Decimal("1000")
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("100")
    pos.yes_cost_basis = Decimal("50")

    await handle_bba(
        MagicMock(),
        farm_state,
        AsyncMock(),
        pos,
        make_bba(pos.market.yes_token_id, Decimal("0.44"), Decimal("0.56")),
    )

    assert pos.yes_shares == Decimal("100")
    assert pos.yes_cost_basis == Decimal("50")
    assert is_paused(farm_state, "market-A") is False


async def test_small_gap_within_cap_does_not_cancel(farm_state: FarmState, stub_network):
    # best_bid 0.50 == our_price → est loss $0 → no exit-loss cancel, no pause.
    farm_state.config.filters.max_fill_loss = Decimal("1")
    pos = farm_state.positions["market-A"]

    await handle_bba(
        MagicMock(),
        farm_state,
        AsyncMock(),
        pos,
        make_bba(pos.market.yes_token_id, Decimal("0.50"), Decimal("0.52")),
    )

    assert is_paused(farm_state, "market-A") is False


async def test_gap_exactly_at_cap_does_not_cancel(farm_state: FarmState, stub_network):
    # best_bid 0.49 → est loss 100*(0.50-0.49)=$1.00, not > $1 → no exit-loss cancel.
    farm_state.config.filters.max_fill_loss = Decimal("1")
    pos = farm_state.positions["market-A"]

    await handle_bba(
        MagicMock(),
        farm_state,
        AsyncMock(),
        pos,
        make_bba(pos.market.yes_token_id, Decimal("0.49"), Decimal("0.51")),
    )

    assert is_paused(farm_state, "market-A") is False


async def test_filter_off_never_cancels(farm_state: FarmState, stub_network):
    # max_fill_loss is None by default (fixture) → guard inert even on a huge gap.
    pos = farm_state.positions["market-A"]

    await handle_bba(
        MagicMock(),
        farm_state,
        AsyncMock(),
        pos,
        make_bba(pos.market.yes_token_id, Decimal("0.10"), Decimal("0.90")),
    )

    assert is_paused(farm_state, "market-A") is False


# ── pre-place check: a FRESH re-quote is never rested above the cap (the one-tick window) ──


async def test_recovering_market_not_requoted_above_cap(farm_state: FarmState, stub_network):
    # Old resting price (0.50) == best_bid → guard passes (no cancel). But on this wide,
    # recovering book the new quote would be 0.53 (3¢ above the 0.50 bid → $3 loss > $1 cap),
    # so the requote must be SKIPPED rather than rest an unfavorable fresh order.
    farm_state.config.filters.max_fill_loss = Decimal("1")
    pos = farm_state.positions["market-A"]  # yes_price 0.50, size 100, safe depth

    await handle_bba(
        MagicMock(),
        farm_state,
        AsyncMock(),
        pos,
        make_bba(pos.market.yes_token_id, Decimal("0.50"), Decimal("0.60")),
    )

    assert stub_network["requotes"] == []  # fresh order above the cap not placed
    assert stub_network["cancelled"] == []  # guard didn't fire (old price was at the bid)
    assert is_paused(farm_state, "market-A") is False


async def test_aggressive_recovering_market_not_requoted_above_cap(
    farm_state: FarmState, stub_network
):
    # Aggressive depth allows upward re-quotes (no safe-depth hysteresis), so the pre-place
    # cap check is the ONLY thing stopping a fresh order resting above the cap. new quote 0.52
    # vs 0.50 bid = $2 loss > $1 → skipped.
    farm_state.config.quote_depth = "aggressive"
    farm_state.config.filters.max_fill_loss = Decimal("1")
    pos = farm_state.positions["market-A"]

    await handle_bba(
        MagicMock(),
        farm_state,
        AsyncMock(),
        pos,
        make_bba(pos.market.yes_token_id, Decimal("0.50"), Decimal("0.56")),
    )

    assert stub_network["requotes"] == []
    assert is_paused(farm_state, "market-A") is False


async def test_requotes_when_fresh_price_within_cap(farm_state: FarmState, stub_network):
    # New quote 0.49 is at/below the 0.50 bid → $0 loss ≤ cap → the requote proceeds normally.
    farm_state.config.filters.max_fill_loss = Decimal("1")
    pos = farm_state.positions["market-A"]

    await handle_bba(
        MagicMock(),
        farm_state,
        AsyncMock(),
        pos,
        make_bba(pos.market.yes_token_id, Decimal("0.50"), Decimal("0.52")),
    )

    assert stub_network["requotes"] == [("YES", 0.49)]
    assert is_paused(farm_state, "market-A") is False


async def test_pre_place_check_inert_when_filter_off(farm_state: FarmState, stub_network):
    # max_fill_loss None → the pre-place cap check is skipped; the wide-market requote that
    # would be blocked above proceeds (original behaviour preserved when the filter is off).
    pos = farm_state.positions["market-A"]  # max_fill_loss None by default

    await handle_bba(
        MagicMock(),
        farm_state,
        AsyncMock(),
        pos,
        make_bba(pos.market.yes_token_id, Decimal("0.50"), Decimal("0.60")),
    )

    assert stub_network["requotes"] == [("YES", 0.53)]  # requoted despite the wide gap


async def test_no_leg_recovering_market_not_requoted_above_cap(farm_state: FarmState, stub_network):
    # Symmetry: the pre-place cap check applies to the NO leg too. NO resting price 0.50,
    # NO bid 0.50 → guard passes; new NO quote 0.53 (3¢ above bid → $3 > $1) → requote skipped.
    farm_state.config.filters.max_fill_loss = Decimal("1")
    pos = farm_state.positions["market-A"]

    await handle_bba(
        MagicMock(),
        farm_state,
        AsyncMock(),
        pos,
        make_bba(pos.market.no_token_id, Decimal("0.50"), Decimal("0.60")),
    )

    assert stub_network["requotes"] == []
    assert is_paused(farm_state, "market-A") is False


async def test_aggressive_fresh_quote_at_exact_cap_is_allowed(farm_state: FarmState, stub_network):
    # Boundary: aggressive new quote 0.51 vs 0.50 bid = exactly $1.00 loss, NOT > $1 cap → the
    # requote is allowed (inclusive boundary, mirrors the entry filter's <= semantics).
    farm_state.config.quote_depth = "aggressive"
    farm_state.config.filters.max_fill_loss = Decimal("1")
    pos = farm_state.positions["market-A"]

    await handle_bba(
        MagicMock(),
        farm_state,
        AsyncMock(),
        pos,
        make_bba(pos.market.yes_token_id, Decimal("0.50"), Decimal("0.54")),
    )

    assert stub_network["requotes"] == [("YES", 0.51)]  # at-cap fresh quote allowed
    assert is_paused(farm_state, "market-A") is False
