"""Tier-1 crash guards.

#1 Best-bid gap-pull: a single market-WS frame collapsing the bid (the price our resting BUY
   fills against) is treated as a crash — pull both legs and blacklist, instead of the depth-blind
   top-of-book loss guard letting the 'threatened' path chase the bid down into a fill.

#2 Mark-to-market kill (F7): should_kill now adds the unrealized loss on held inventory (marked at
   the live best bid), so a crash whose shares get abandoned/dust-written-off — and never book a
   realized loss — still trips the session cap. handle_bba and reconcile_tick evaluate it.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.schemas import BestBidAsk
from app.farm import requote as requote_mod
from app.farm import worker as worker_mod
from app.farm.kill_switch import should_kill, unrealized_loss
from app.farm.requote import handle_bba
from app.farm.schemas import FarmState
from app.farm.volatility import is_blacklisted


@pytest.fixture
def stub_network(monkeypatch):
    cancelled: list = []
    requotes: list = []

    async def fake_cancel(client, oid):
        cancelled.append(oid)

    async def fake_cancel_orders(client, *ids):
        # Tier 2: the gap-pull now batches via cancel_orders; funnel its ids into the SAME
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
        timestamp="2026-06-17T12:00:00Z",
    )


# ── #1 best-bid gap-pull ──────────────────────────────────────────────────────


async def test_bid_gap_pulls_orders_and_blacklists(farm_state: FarmState, stub_network):
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("0")  # isolate the gap-pull from the mark-to-market kill
    pos.yes_cost_basis = Decimal("0")
    yes = pos.market.yes_token_id

    # Frame 1 establishes the prior bid; clear any requote it triggers so we assay frame 2 alone.
    await handle_bba(
        MagicMock(), farm_state, AsyncMock(), pos, make_bba(yes, Decimal("0.50"), Decimal("0.52"))
    )
    stub_network["cancelled"].clear()
    stub_network["requotes"].clear()
    yes_oid, no_oid = pos.yes_order_id, pos.no_order_id

    # Frame 2: bid collapses 0.50 → 0.40 (20% > 10%, and 0.10 > 2 ticks) → crash → pull + blacklist.
    await handle_bba(
        MagicMock(), farm_state, AsyncMock(), pos, make_bba(yes, Decimal("0.40"), Decimal("0.52"))
    )

    assert set(stub_network["cancelled"]) == {yes_oid, no_oid}
    assert is_blacklisted(farm_state, "market-A") is True
    assert stub_network["requotes"] == [], "must not chase the crashing bid with a requote"


async def test_bid_gap_pulls_on_no_leg_too(farm_state: FarmState, stub_network):
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("0")
    pos.yes_cost_basis = Decimal("0")
    no = pos.market.no_token_id

    await handle_bba(
        MagicMock(), farm_state, AsyncMock(), pos, make_bba(no, Decimal("0.50"), Decimal("0.52"))
    )
    stub_network["cancelled"].clear()
    yes_oid, no_oid = pos.yes_order_id, pos.no_order_id
    await handle_bba(
        MagicMock(), farm_state, AsyncMock(), pos, make_bba(no, Decimal("0.40"), Decimal("0.52"))
    )

    assert set(stub_network["cancelled"]) == {yes_oid, no_oid}
    assert is_blacklisted(farm_state, "market-A") is True


async def test_small_bid_drop_does_not_pull(farm_state: FarmState, stub_network):
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("0")
    pos.yes_cost_basis = Decimal("0")
    yes = pos.market.yes_token_id

    await handle_bba(
        MagicMock(), farm_state, AsyncMock(), pos, make_bba(yes, Decimal("0.50"), Decimal("0.52"))
    )
    stub_network["cancelled"].clear()
    # 0.50 → 0.48 is a 4% drop (< 10%) → normal jitter, no gap-pull.
    await handle_bba(
        MagicMock(), farm_state, AsyncMock(), pos, make_bba(yes, Decimal("0.48"), Decimal("0.52"))
    )

    assert stub_network["cancelled"] == [], "a small drop must not trigger the gap-pull"
    assert is_blacklisted(farm_state, "market-A") is False


async def test_gap_below_tick_floor_does_not_pull(farm_state: FarmState, stub_network):
    # Low-priced market: 0.10 → 0.085 is a 15% drop (> 10%) BUT only 0.015 < 2 ticks (0.02), so the
    # absolute floor suppresses it — guards against over-pulling on 1-tick jitter at low prices.
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("0")
    pos.yes_cost_basis = Decimal("0")
    yes = pos.market.yes_token_id

    await handle_bba(
        MagicMock(), farm_state, AsyncMock(), pos, make_bba(yes, Decimal("0.10"), Decimal("0.12"))
    )
    stub_network["cancelled"].clear()
    await handle_bba(
        MagicMock(), farm_state, AsyncMock(), pos, make_bba(yes, Decimal("0.085"), Decimal("0.12"))
    )

    assert stub_network["cancelled"] == []
    assert is_blacklisted(farm_state, "market-A") is False


async def test_first_frame_has_no_prior_bid_so_no_gap_pull(farm_state: FarmState, stub_network):
    # With no prior bid recorded, the first frame can't compute a gap — never pull on frame 1.
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("0")
    pos.yes_cost_basis = Decimal("0")
    yes = pos.market.yes_token_id

    await handle_bba(
        MagicMock(), farm_state, AsyncMock(), pos, make_bba(yes, Decimal("0.20"), Decimal("0.52"))
    )

    assert stub_network["cancelled"] == []
    assert is_blacklisted(farm_state, "market-A") is False


# ── #2 mark-to-market kill (F7) ───────────────────────────────────────────────


def test_unrealized_loss_marks_held_legs(farm_state: FarmState):
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("100")
    pos.yes_cost_basis = Decimal("50")  # avg 0.50
    pos.yes_best_bid = Decimal("0.44")
    # 50 paid − 100 shares * 0.44 current bid = $6 unrealized loss.
    assert unrealized_loss(farm_state) == Decimal("6.00")


def test_unrealized_loss_skips_legs_without_a_bid(farm_state: FarmState):
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("100")
    pos.yes_cost_basis = Decimal("50")
    pos.yes_best_bid = None  # no frame seen yet — can't mark, so it contributes 0
    assert unrealized_loss(farm_state) == Decimal("0")


def test_unrealized_loss_counts_a_gain_as_negative(farm_state: FarmState):
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("100")
    pos.yes_cost_basis = Decimal("50")
    pos.yes_best_bid = Decimal("0.55")  # marked up → a gain offsets other losses
    assert unrealized_loss(farm_state) == Decimal("-5.00")


def test_should_kill_trips_on_unrealized_drawdown(farm_state: FarmState):
    # cap = 5 (fixture). No realized loss, but held inventory is $6 underwater → must trip.
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("100")
    pos.yes_cost_basis = Decimal("50")
    pos.yes_best_bid = Decimal("0.44")
    assert farm_state.session_loss == Decimal("0")
    assert should_kill(farm_state) is True


def test_should_kill_realized_only_when_no_bid(farm_state: FarmState):
    # No marked bid → unrealized contributes 0 → behaves exactly like the old realized-only cap.
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("100")
    pos.yes_cost_basis = Decimal("50")
    assert should_kill(farm_state) is False


async def test_handle_bba_triggers_kill_on_mtm_drawdown(
    farm_state: FarmState, stub_network, monkeypatch
):
    kills: list = []

    async def fake_trigger_kill(client, state, ws):
        state.killed = True
        kills.append(True)

    monkeypatch.setattr(requote_mod, "trigger_kill", fake_trigger_kill)

    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("100")
    pos.yes_cost_basis = Decimal("50")  # avg 0.50, cap = 5

    # A frame marking the held leg at 0.44 → $6 unrealized > $5 cap → kill, and stop quoting.
    await handle_bba(
        MagicMock(),
        farm_state,
        AsyncMock(),
        pos,
        make_bba(pos.market.yes_token_id, Decimal("0.44"), Decimal("0.56")),
    )

    assert kills == [True], "handle_bba must trigger the kill on a mark-to-market drawdown"
    assert farm_state.killed is True
    assert stub_network["requotes"] == [], "a killed farm must not requote"


async def test_handle_bba_no_kill_within_cap(farm_state: FarmState, stub_network, monkeypatch):
    kills: list = []

    async def fake_trigger_kill(client, state, ws):
        kills.append(True)

    monkeypatch.setattr(requote_mod, "trigger_kill", fake_trigger_kill)

    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("100")
    pos.yes_cost_basis = Decimal("50")  # cap = 5

    # Bid 0.49 → only $1 unrealized < $5 cap → no kill.
    await handle_bba(
        MagicMock(),
        farm_state,
        AsyncMock(),
        pos,
        make_bba(pos.market.yes_token_id, Decimal("0.49"), Decimal("0.51")),
    )

    assert kills == []
    assert farm_state.killed is False


async def test_reconcile_tick_backstops_the_kill(farm_state: FarmState, monkeypatch):
    # If the market WS is quiet, the reconcile tick must still evaluate the mark-to-market cap
    # before doing any discovery work.
    kills: list = []
    fetched: list = []

    async def fake_trigger_kill(client, state, ws, source="realtime"):
        state.killed = True
        kills.append(source)

    async def fake_fetch(http):
        fetched.append(True)
        return []

    monkeypatch.setattr(worker_mod, "trigger_kill", fake_trigger_kill)
    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_fetch)

    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("100")
    pos.yes_cost_basis = Decimal("50")
    pos.yes_best_bid = Decimal("0.40")  # $10 underwater > $5 cap

    await worker_mod.reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert kills == ["reconcile_backstop"], "reconcile backstop must trigger the kill, tagged"
    assert fetched == [], "the kill must short-circuit the tick before discovery runs"
