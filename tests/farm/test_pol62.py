"""POL-62: exclude-on-fill, the held-share exit sweep, and the true two-leg gate.

- A maker fill excludes that market for the rest of the session (and is not re-opened).
- reconcile_tick re-fires an exit for any held inventory, but never while an exit is
  already in flight (no double-sell).
- The per-market gate uses the TRUE two-leg cost, so a negRisk market whose YES+NO
  bids sum past $1 and exceed the bankroll is skipped instead of admitted-then-rejected.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.bot.schemas import UserTrade, UserTradeMakerOrder
from app.constants import OPEN_FAILURE_PAUSE_THRESHOLD, STALE_EXIT_SECONDS
from app.farm import exits as exits_mod
from app.farm import fills as fills_mod
from app.farm import worker as worker_mod
from app.farm.fills import handle_trade
from app.farm.health import is_paused
from app.farm.schemas import ExitCostInfo, ExitOrder, FarmState, Market, MarketPosition
from app.farm.worker import close_position, open_position, reconcile_tick


async def sweep(state):
    await exits_mod.exit_held_legs(MagicMock(), state, cancel_resting=True, skip_in_flight=True)


def wire_open(monkeypatch, markets, midpoints, place_calls, balance=Decimal("1000")):
    async def fake_fetch_eligible_markets(http):
        return markets

    async def fake_fetch_midpoints(http, token_ids):
        return midpoints

    async def fake_get_balance(addr):
        return balance

    async def fake_place_limit_order(client, order, post_only=False):
        place_calls.append(order.token_id)
        return f"oid-{len(place_calls)}"

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_fetch_eligible_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_fetch_midpoints)
    monkeypatch.setattr(worker_mod, "passes_all", lambda m, f: True)
    monkeypatch.setattr(worker_mod, "get_balance", fake_get_balance, raising=False)
    monkeypatch.setattr(worker_mod, "place_limit_order", fake_place_limit_order)


def maker_fill(market: Market, order_id: str, token_id: str, outcome: str) -> UserTrade:
    return UserTrade(
        event_type="trade",
        id="trade-fill-1",
        asset_id=token_id,
        market=market.condition_id,
        side="BUY",
        price=Decimal("0.48"),
        size=Decimal("100"),
        outcome=outcome,
        status="MATCHED",
        timestamp="2026-06-09T01:00:00Z",
        maker_orders=[
            UserTradeMakerOrder(
                asset_id=token_id,
                order_id=order_id,
                matched_amount=Decimal("100"),
                outcome=outcome,
                owner="0xowner",
                price=Decimal("0.48"),
            )
        ],
        taker_order_id="taker-1",
    )


# ── exclude-on-fill ───────────────────────────────────────────────────────────


async def test_fill_blacklists_market(farm_state: FarmState, monkeypatch):
    from app.farm.volatility import is_blacklisted

    async def noop_cancel(_client, _oid):
        return None

    monkeypatch.setattr(fills_mod, "cancel_order", noop_cancel)

    # Our resting NO leg on market-A fills → market is temp-blacklisted (cooldown), not
    # permanently excluded.
    fill = maker_fill(farm_state.positions["market-A"].market, "no-oid", "tok-no", "NO")
    await handle_trade(MagicMock(), fill, farm_state, AsyncMock())

    assert is_blacklisted(farm_state, "market-A")
    assert "market-A" not in farm_state.excluded_markets


async def test_excluded_market_is_not_reopened(farm_state: FarmState, market: Market, monkeypatch):
    # Pretend the position already flattened and was dropped, but the market stays
    # excluded — reconcile_tick must not quote it again.
    farm_state.positions.clear()
    farm_state.excluded_markets.add("market-A")
    place_calls: list = []
    wire_open(
        monkeypatch,
        [market],
        {market.yes_token_id: Decimal("0.5"), market.no_token_id: Decimal("0.5")},
        place_calls,
    )

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert place_calls == [], "an excluded market must never be re-opened"


# ── held-share exit sweep ───────────────────────────────────────────────────────


async def test_sweep_exits_held_inventory(farm_state: FarmState, monkeypatch):
    # Fixture market-A holds 100 YES shares with no exit in flight.
    pos = farm_state.positions["market-A"]
    assert pos.yes_shares > 0 and not pos.exit_orders

    swept: list = []

    async def fake_exit(client, state, token_id, size, cid, slug, outcome, **kwargs):
        swept.append((token_id, outcome))

    async def no_markets(http):
        return []

    async def fake_balance(addr):
        return Decimal("1000")

    monkeypatch.setattr(exits_mod, "exit_position_leg", fake_exit)
    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", no_markets)
    monkeypatch.setattr(worker_mod, "get_balance", fake_balance, raising=False)

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert ("tok-yes", "YES") in swept, "sweep must re-fire an exit for held inventory"


async def test_sweep_skips_leg_with_exit_in_flight(farm_state: FarmState, monkeypatch):
    pos = farm_state.positions["market-A"]
    # A fresh SELL is already in flight for the held (YES) leg.
    pos.exit_orders["exit-already-working"] = ExitOrder(
        outcome="YES", placed_at=datetime.now(timezone.utc)
    )

    swept: list = []

    async def fake_exit(client, state, token_id, size, cid, slug, outcome, **kwargs):
        swept.append((token_id, outcome))

    async def no_markets(http):
        return []

    async def fake_balance(addr):
        return Decimal("1000")

    monkeypatch.setattr(exits_mod, "exit_position_leg", fake_exit)
    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", no_markets)
    monkeypatch.setattr(worker_mod, "get_balance", fake_balance, raising=False)

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert swept == [], "must not double-sell a leg whose exit is already working"


# ── true two-leg gate (negRisk undercount) ──────────────────────────────────────


async def test_gate_rejects_negrisk_oversized(farm_state: FarmState, market: Market, monkeypatch):
    # NBA-futures-style market: YES_bid + NO_bid sum well above $1, so the true
    # two-leg cost (size 100 × (0.82 + 0.82) = $164) exceeds the $100 bankroll even
    # though the old size×$1 gate ($100) would have admitted it.
    farm_state.positions.clear()
    place_calls: list = []
    wire_open(
        monkeypatch,
        [market],
        {market.yes_token_id: Decimal("0.84"), market.no_token_id: Decimal("0.84")},
        place_calls,
    )

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert place_calls == [], "a market whose true two-leg cost exceeds bankroll must be skipped"


async def test_gate_admits_normal_binary_market(farm_state: FarmState, market: Market, monkeypatch):
    # Control: a true binary market (bids ~0.48 each, cost ~$96 < $100) still opens.
    farm_state.positions.clear()
    place_calls: list = []
    wire_open(
        monkeypatch,
        [market],
        {market.yes_token_id: Decimal("0.5"), market.no_token_id: Decimal("0.5")},
        place_calls,
    )

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert set(place_calls) == {market.yes_token_id, market.no_token_id}


# ── BUG 1: maker-filled exit SELL is reconciled ─────────────────────────────────


async def test_maker_filled_exit_is_reconciled(farm_state: FarmState):
    # Our GTC exit SELL rested as a maker and got hit; the fill arrives with our oid
    # in maker_orders, NOT taker_order_id. It must still book PnL + decrement shares.
    pos = farm_state.positions["market-A"]  # holds 100 YES @ cost-basis 50 (fixture)
    farm_state.pending_exit_order_ids.add("exit-oid")
    farm_state.exit_cost_basis["exit-oid"] = ExitCostInfo(
        entry_cost=Decimal("50"), entry_size=Decimal("100"), slug="m1"
    )
    pos.exit_orders["exit-oid"] = ExitOrder(outcome="YES", placed_at=datetime.now(timezone.utc))

    trade = UserTrade(
        event_type="trade",
        id="t-maker-exit",
        asset_id="tok-yes",
        market="market-A",
        side="SELL",
        price=Decimal("0.45"),
        size=Decimal("999"),
        outcome="YES",
        status="MINED",
        timestamp="2026-06-09T01:00:00Z",
        maker_orders=[
            UserTradeMakerOrder(
                asset_id="tok-yes",
                order_id="exit-oid",
                matched_amount=Decimal("100"),
                outcome="YES",
                owner="0xowner",
                price=Decimal("0.45"),
            )
        ],
        taker_order_id="counterparty-oid",  # NOT our oid — we were the maker
    )
    await handle_trade(MagicMock(), trade, farm_state, AsyncMock())

    assert pos.yes_shares == Decimal("0"), "maker-filled exit must decrement held shares"
    assert "exit-oid" not in pos.exit_orders, "in-flight marker must clear"
    assert "exit-oid" not in farm_state.pending_exit_order_ids
    # entry cost 50 − proceeds (100 × 0.45 = 45) = 5 realized loss.
    assert farm_state.session_loss == Decimal("5")


# ── BUG 2: stale-exit reaper ─────────────────────────────────────────────────────


async def test_stale_exit_reaped_and_redriven(farm_state: FarmState, monkeypatch):
    pos = farm_state.positions["market-A"]  # holds 100 YES
    pos.exit_orders["stale-oid"] = ExitOrder(
        outcome="YES",
        placed_at=datetime.now(timezone.utc) - timedelta(seconds=STALE_EXIT_SECONDS + 60),
    )
    farm_state.pending_exit_order_ids.add("stale-oid")

    cancelled: list = []
    swept: list = []

    async def fake_cancel(client, oid):
        cancelled.append(oid)

    async def fake_exit(client, state, token_id, size, cid, slug, outcome, **kw):
        swept.append((token_id, outcome))

    monkeypatch.setattr(exits_mod, "cancel_order", fake_cancel)
    monkeypatch.setattr(exits_mod, "exit_position_leg", fake_exit)

    await sweep(farm_state)

    assert "stale-oid" in cancelled, "stale exit must be cancelled before re-driving"
    assert "stale-oid" not in pos.exit_orders
    assert "stale-oid" not in farm_state.pending_exit_order_ids
    assert ("tok-yes", "YES") in swept, "held leg must be re-driven after reaping the stale exit"


async def test_fresh_exit_not_reaped(farm_state: FarmState, monkeypatch):
    pos = farm_state.positions["market-A"]
    pos.exit_orders["fresh-oid"] = ExitOrder(outcome="YES", placed_at=datetime.now(timezone.utc))

    cancelled: list = []
    swept: list = []

    async def fake_cancel(client, oid):
        cancelled.append(oid)

    async def fake_exit(client, state, token_id, size, cid, slug, outcome, **kw):
        swept.append((token_id, outcome))

    monkeypatch.setattr(exits_mod, "cancel_order", fake_cancel)
    monkeypatch.setattr(exits_mod, "exit_position_leg", fake_exit)

    await sweep(farm_state)

    assert cancelled == [] and swept == [], "a fresh in-flight exit must be left alone"


async def test_stale_exit_failing_cancel_kept_not_redriven(farm_state: FarmState, monkeypatch):
    """If the reaper's cancel of a stale exit FAILS, the order must stay tracked and the leg
    must NOT be re-driven this tick. Re-driving while the original GTC may still rest on-book
    places a second SELL for the same shares — a double-sell. Retry the cancel next tick."""
    pos = farm_state.positions["market-A"]  # holds 100 YES
    pos.exit_orders["stale-oid"] = ExitOrder(
        outcome="YES",
        placed_at=datetime.now(timezone.utc) - timedelta(seconds=STALE_EXIT_SECONDS + 60),
    )
    farm_state.pending_exit_order_ids.add("stale-oid")

    cancelled: list = []
    swept: list = []

    async def boom_cancel(client, oid):
        cancelled.append(oid)
        raise RuntimeError("cancel failed (network blip)")

    async def fake_exit(client, state, token_id, size, cid, slug, outcome, **kw):
        swept.append((token_id, outcome))

    monkeypatch.setattr(exits_mod, "cancel_order", boom_cancel)
    monkeypatch.setattr(exits_mod, "exit_position_leg", fake_exit)

    await sweep(farm_state)

    assert "stale-oid" in cancelled, "the reaper still attempts the cancel"
    # ...but on failure the order stays tracked so the in-flight skip prevents a double-sell.
    assert "stale-oid" in pos.exit_orders
    assert "stale-oid" in farm_state.pending_exit_order_ids
    assert swept == [], "leg must NOT be re-driven while the un-cancelled GTC may still rest"


# ── BUG 3: per-leg in-flight skip (sibling leg still exits) ──────────────────────


async def test_sweep_exits_sibling_leg(farm_state: FarmState, monkeypatch):
    pos = farm_state.positions["market-A"]
    pos.no_shares = Decimal("100")  # now BOTH legs are held
    pos.no_cost_basis = Decimal("50")
    pos.exit_orders["yes-exit"] = ExitOrder(outcome="YES", placed_at=datetime.now(timezone.utc))

    swept: list = []

    async def fake_exit(client, state, token_id, size, cid, slug, outcome, **kw):
        swept.append(outcome)

    monkeypatch.setattr(exits_mod, "exit_position_leg", fake_exit)

    await sweep(farm_state)

    assert "NO" in swept, "the sibling NO leg (no in-flight exit) must still be swept"
    assert "YES" not in swept, "the YES leg whose exit is in flight must be skipped"


# ── BUG 4: open-failure pause after threshold ────────────────────────────────────


async def test_open_failures_pause_market(farm_state: FarmState, market: Market, monkeypatch):
    async def fake_place(client, order, post_only=False):
        if order.token_id == market.no_token_id:
            raise RuntimeError("NO leg POST fails")
        return "yes-oid"

    async def noop_cancel(client, oid):
        return None

    monkeypatch.setattr(worker_mod, "place_limit_order", fake_place)
    monkeypatch.setattr(worker_mod, "cancel_order", noop_cancel)
    midpoints = {market.yes_token_id: Decimal("0.5"), market.no_token_id: Decimal("0.5")}

    for _ in range(OPEN_FAILURE_PAUSE_THRESHOLD - 1):
        await open_position(MagicMock(), farm_state, AsyncMock(), market, midpoints)
    assert not is_paused(farm_state, market.condition_id), "must not pause before the threshold"

    await open_position(MagicMock(), farm_state, AsyncMock(), market, midpoints)
    assert is_paused(farm_state, market.condition_id), "consecutive open failures must pause"


# ── fix-pass #2 ──────────────────────────────────────────────────────────────


async def test_maker_partial_cancels_gtc_then_redrives(farm_state: FarmState, monkeypatch):
    # A resting GTC exit SELL of 100 partially fills (40) as a maker. The remainder
    # is still on-book, so we must CANCEL it before re-driving the residual — else
    # two live SELLs for the same shares (double-sell) and a later dropped fill.
    pos = farm_state.positions["market-A"]  # holds 100 YES @ cost-basis 50
    farm_state.pending_exit_order_ids.add("gtc-exit")
    farm_state.exit_cost_basis["gtc-exit"] = ExitCostInfo(
        entry_cost=Decimal("50"), entry_size=Decimal("100"), slug="m1"
    )
    pos.exit_orders["gtc-exit"] = ExitOrder(outcome="YES", placed_at=datetime.now(timezone.utc))

    calls: list = []

    async def fake_cancel(client, oid):
        calls.append(("cancel", oid))

    async def fake_exit(client, state, token_id, size, cid, slug, outcome, **kw):
        calls.append(("redrive", token_id, float(size)))

    monkeypatch.setattr(fills_mod, "cancel_order", fake_cancel)
    monkeypatch.setattr(fills_mod, "exit_position_leg", fake_exit)

    trade = UserTrade(
        event_type="trade",
        id="t-maker-partial",
        asset_id="tok-yes",
        market="market-A",
        side="SELL",
        price=Decimal("0.45"),
        size=Decimal("40"),
        outcome="YES",
        status="MINED",
        timestamp="2026-06-09T01:00:00Z",
        maker_orders=[
            UserTradeMakerOrder(
                asset_id="tok-yes",
                order_id="gtc-exit",
                matched_amount=Decimal("40"),
                outcome="YES",
                owner="0xowner",
                price=Decimal("0.45"),
            )
        ],
        taker_order_id="counterparty-oid",
    )
    await handle_trade(MagicMock(), trade, farm_state, AsyncMock())

    assert pos.yes_shares == Decimal("60"), "held shares decremented by the matched amount"
    assert ("cancel", "gtc-exit") in calls, "the resting GTC remainder must be cancelled"
    # Cancel must come BEFORE the residual re-drive (no double-sell window).
    assert calls.index(("cancel", "gtc-exit")) < calls.index(("redrive", "tok-yes", 60.0))


async def test_maker_partial_cancel_failure_retracks_not_redrives(
    farm_state: FarmState, monkeypatch
):
    # If the remainder-cancel FAILS, we must NOT re-drive a second SELL (double-sell);
    # the original order is still the working exit, so re-track it for the sweep/reaper.
    pos = farm_state.positions["market-A"]  # holds 100 YES
    farm_state.pending_exit_order_ids.add("gtc-exit")
    farm_state.exit_cost_basis["gtc-exit"] = ExitCostInfo(
        entry_cost=Decimal("50"), entry_size=Decimal("100"), slug="m1"
    )
    pos.exit_orders["gtc-exit"] = ExitOrder(outcome="YES", placed_at=datetime.now(timezone.utc))

    redrives: list = []

    async def failing_cancel(client, oid):
        raise RuntimeError("cancel rejected")

    async def fake_exit(client, state, token_id, size, cid, slug, outcome, **kw):
        redrives.append((token_id, float(size)))

    monkeypatch.setattr(fills_mod, "cancel_order", failing_cancel)
    monkeypatch.setattr(fills_mod, "exit_position_leg", fake_exit)

    trade = UserTrade(
        event_type="trade",
        id="t-cancel-fail",
        asset_id="tok-yes",
        market="market-A",
        side="SELL",
        price=Decimal("0.45"),
        size=Decimal("40"),
        outcome="YES",
        status="MINED",
        timestamp="2026-06-09T01:00:00Z",
        maker_orders=[
            UserTradeMakerOrder(
                asset_id="tok-yes",
                order_id="gtc-exit",
                matched_amount=Decimal("40"),
                outcome="YES",
                owner="0xowner",
                price=Decimal("0.45"),
            )
        ],
        taker_order_id="counterparty-oid",
    )
    await handle_trade(MagicMock(), trade, farm_state, AsyncMock())

    assert redrives == [], "must NOT place a second SELL when the cancel failed"
    assert "gtc-exit" in pos.exit_orders, "the still-resting order must be re-tracked"
    assert "gtc-exit" in farm_state.pending_exit_order_ids
    assert pos.yes_shares == Decimal("60"), "the 40-share fill is still booked"


async def test_failed_exit_frame_clears_tracking(farm_state: FarmState):
    # A FAILED exit-SELL frame must clear its tracking so the sweep re-drives the
    # still-held leg next tick instead of waiting ~120s for the reaper.
    pos = farm_state.positions["market-A"]  # holds 100 YES
    farm_state.pending_exit_order_ids.add("failed-exit")
    pos.exit_orders["failed-exit"] = ExitOrder(outcome="YES", placed_at=datetime.now(timezone.utc))

    trade = UserTrade(
        event_type="trade",
        id="t-failed",
        asset_id="tok-yes",
        market="market-A",
        side="SELL",
        price=Decimal("0.45"),
        size=Decimal("100"),
        outcome="YES",
        status="FAILED",
        timestamp="2026-06-09T01:00:00Z",
        maker_orders=[],
        taker_order_id="failed-exit",
    )
    await handle_trade(MagicMock(), trade, farm_state, AsyncMock())

    assert "failed-exit" not in farm_state.pending_exit_order_ids
    assert "failed-exit" not in pos.exit_orders
    assert pos.yes_shares == Decimal("100"), "SELL failed; shares stay held for re-drive"


async def test_close_prunes_registry_no_phantom_credit_on_reopen(
    farm_state: FarmState, market: Market, monkeypatch
):
    # close_position drops the market's order ids from the registry (fixture pre-
    # registers yes-oid/no-oid), so a late fill on a STALE oid after the market reopens
    # cannot phantom-credit the reopened position (regression guard for R1).
    async def confirmed_cancel(client, oid):
        return True

    monkeypatch.setattr(worker_mod, "cancel_order_with_retry", confirmed_cancel)

    await close_position(MagicMock(), farm_state, AsyncMock(), "market-A", reason="market_dropped")
    assert "yes-oid" not in farm_state.order_registry, "closed market's oids must be pruned"
    assert "no-oid" not in farm_state.order_registry

    # Market reopens with fresh oids; the old yes-oid is now stale.
    farm_state.positions["market-A"] = MarketPosition(
        market=market,
        yes_order_id="yes-oid-v2",
        no_order_id="no-oid-v2",
        yes_price=Decimal("0.5"),
        no_price=Decimal("0.5"),
    )
    reopened = farm_state.positions["market-A"]

    trade = UserTrade(
        event_type="trade",
        id="t-late",
        asset_id="tok-yes",
        market="market-A",
        side="BUY",
        price=Decimal("0.48"),
        size=Decimal("100"),
        outcome="YES",
        status="MATCHED",
        timestamp="2026-06-09T01:00:00Z",
        maker_orders=[
            UserTradeMakerOrder(
                asset_id="tok-yes",
                order_id="yes-oid",
                matched_amount=Decimal("100"),
                outcome="YES",
                owner="0xowner",
                price=Decimal("0.48"),
            )
        ],
        taker_order_id="counterparty-oid",
    )
    await handle_trade(MagicMock(), trade, farm_state, AsyncMock())

    assert reopened.yes_shares == Decimal("0"), "stale-oid fill must not credit reopened position"
