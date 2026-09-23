"""Stale-residual sell fix — partial-balance CLAMP coverage.

A legacy partial-balance SELL rejection means the exchange holds fewer shares than the bot tracks
(an earlier fill is still settling). The old code re-drove the FULL tracked size forever (a
live-observed 6-minute retry loop → kill). The fix CLAMPS the tracked leg down to the
exchange-reported balance and retries once at that size; shares removed from tracking are simply
DROPPED (a bounded, conservative accounting gap in a rare race) — there is no ledger parking a
late fill on the superseded oid.

Two clamp bug fixes are pinned here alongside the base behaviour:
  * orphan (pos is None) and old==0 (tracked leg already flat) re-drives now SCALE the caller's
    entry_cost proportionally (balance/tracked) instead of forwarding the full value — a full
    forward phantom-books a loss on a break-even re-drive.
  * old<=0 with a still-sellable balance retries at the balance instead of no-op'ing forever at
    ``new = min(balance, 0) = 0`` (which the ``size <= 0`` guard silently swallowed every tick).

``FarmState.booked_exit_fills`` + the ``(trade.id, oid)`` dedupe on the ordinary exit_fills path
is an independent, still-live MINED→CONFIRMED double-book fix and is pinned here too.

Fixtures mirror tests/farm/test_exit_balance_guard.py (PolyApiException builder) and
tests/conftest.py (farm_state/market, min_order_size=5).
"""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from py_clob_client_v2.exceptions import PolyApiException

from app.bot.schemas import BookLevel, OrderBook, UserTrade
from app.constants import MAX_CLAMP_RETRIES
from app.farm import exits as exits_mod
from app.farm import fills as fills_mod
from app.farm.exits import exit_position_leg
from app.farm.fills import handle_trade
from app.farm.schemas import ExitCostInfo, FarmState

# ── builders ──────────────────────────────────────────────────────────────────


def partial_exc(balance_micro: int, order_micro: int) -> PolyApiException:
    """A legacy partial-balance 400: exchange holds `balance_micro` (micro-shares) but we tried to
    sell `order_micro`."""
    return PolyApiException(
        error_msg={
            "error": (
                "not enough balance / allowance: the balance is not enough "
                f"-> balance: {balance_micro}, order amount: {order_micro}"
            )
        }
    )


def zero_balance_exc() -> PolyApiException:
    return PolyApiException(
        error_msg={
            "error": (
                "not enough balance / allowance: the balance is not enough "
                "-> balance: 0, order amount: 39400000"
            )
        }
    )


def counts(strat_calls: list, event: str) -> int:
    return sum(1 for e, _ in strat_calls if e == event)


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def strat_calls(monkeypatch):
    """Record every strat(...) emitted by exits.py AND fills.py so we can count exit_clamp,
    dust_writeoff, etc."""
    calls: list = []

    def fake_strat(event, **fields):
        calls.append((event, fields))

    monkeypatch.setattr(exits_mod, "strat", fake_strat)
    monkeypatch.setattr(fills_mod, "strat", fake_strat)
    return calls


@pytest.fixture
def no_cancel(monkeypatch):
    """Neutralise the resting-order cancels the exit path performs (exits.cancel_orders and
    fills.cancel_order) so tests observe only the sell/clamp/book behaviour."""

    async def fake_cancel_orders(client, *oids):
        return None

    async def fake_cancel_order(client, oid):
        return None

    monkeypatch.setattr(exits_mod, "cancel_orders", fake_cancel_orders)
    monkeypatch.setattr(fills_mod, "cancel_order", fake_cancel_order)


def prime_no_leg(farm_state: FarmState, shares: str, cost: str) -> None:
    """Put the divergent inventory on the NO leg and flatten the fixture's YES leg so the
    mark-to-market kill can't interfere."""
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("0")
    pos.yes_cost_basis = Decimal("0")
    pos.no_shares = Decimal(shares)
    pos.no_cost_basis = Decimal(cost)
    farm_state.config.max_session_loss = Decimal("100")


# ── base clamp: shrink the tracked leg to the exchange balance and retry once ───


async def test_clamp_shrinks_leg_and_retries_at_exchange_balance(
    farm_state: FarmState, strat_calls, no_cancel, monkeypatch
):
    prime_no_leg(farm_state, shares="39.4", cost="23.64")  # NO entered @ 0.60

    attempts: list = []

    async def place(client, token_id, side, amount):
        attempts.append(Decimal(str(amount)))
        if len(attempts) == 1:
            # Exchange only holds 27.4 of the tracked 39.4 (an earlier fill still settling).
            raise partial_exc(27400000, 39400000)
        return "resell-oid"

    monkeypatch.setattr(exits_mod, "place_market_order", place)

    pos = farm_state.positions["market-A"]
    await exit_position_leg(
        MagicMock(), farm_state, pos.market.no_token_id, Decimal("39.4"), "market-A", "m1", "NO"
    )

    # Clamp shrank the tracked leg to the exchange balance and retried once at 27.4; the 12 removed
    # shares are simply dropped from tracking (no ledger parking).
    assert pos.no_shares == Decimal("27.4"), "leg clamped to the exchange-reported balance"
    assert counts(strat_calls, "exit_clamp") == 1, "exactly one clamp episode"
    assert len(attempts) == 2, "one failed full-size sell + one clamped retry (no loop)"
    assert "resell-oid" in farm_state.pending_exit_order_ids, "clamped retry registered"
    assert farm_state.session_loss == Decimal("0"), "placing/clamping books nothing"


# ── Musk two-chunk regression: residual re-driven under a NEW oid ───────────────


async def test_two_chunk_residual_books_via_new_oid(
    farm_state: FarmState, no_cancel, monkeypatch
):
    """A residual partial-fill re-driven under a fresh oid still books correctly; the superseded
    old oid is simply forgotten (no ledger)."""
    farm_state.config.max_session_loss = Decimal("100")
    pos = farm_state.positions["market-A"]  # YES: 100 shares / $50 basis (fixture)
    farm_state.pending_exit_order_ids.add("oid-1")
    farm_state.exit_cost_basis["oid-1"] = ExitCostInfo(
        entry_cost=Decimal("50"), entry_size=Decimal("100"), slug="m1"
    )

    async def place(client, token_id, side, amount):
        return "oid-2"  # the residual re-drive gets a fresh oid

    monkeypatch.setattr(exits_mod, "place_market_order", place)

    def chunk(trade_id: str, oid: str, size: str) -> UserTrade:
        return UserTrade(
            event_type="trade",
            id=trade_id,
            asset_id="tok-yes",
            market="market-A",
            side="SELL",
            price=Decimal("0.40"),
            size=Decimal(size),
            outcome="YES",
            status="MINED",
            timestamp="2026-07-03T12:00:00Z",
            maker_orders=[],
            taker_order_id=oid,
        )

    # Chunk 1: 60 of 100 fill on oid-1 → residual 40 re-driven under oid-2.
    await handle_trade(MagicMock(), chunk("chunk-1", "oid-1", "60"), farm_state, MagicMock())
    assert pos.yes_shares == Decimal("40")
    assert "oid-2" in farm_state.pending_exit_order_ids, "residual re-driven under a new oid"

    # Chunk 2: the residual 40 fills on the NEW oid-2 via the normal pending path.
    await handle_trade(MagicMock(), chunk("chunk-2", "oid-2", "40"), farm_state, MagicMock())

    assert pos.yes_shares == Decimal("0"), "position fully flat after both chunks"
    # 50 basis − 40 proceeds = $10 booked, once per chunk (6 + 4), never double.
    assert farm_state.session_loss == Decimal("10")


# ── MINED→CONFIRMED replay dedupe on the ordinary exit_fills path ──────────────


async def test_failed_cancel_same_oid_reregister_then_confirmed_books_once(
    farm_state: FarmState, monkeypatch
):
    """Partial exit fill whose cancel FAILS re-registers the SAME oid in pending; a later
    MINED→CONFIRMED replay of that trade must still book once."""
    farm_state.config.max_session_loss = Decimal("100")
    pos = farm_state.positions["market-A"]  # YES 100 / $50
    farm_state.pending_exit_order_ids.add("oid-1")
    farm_state.exit_cost_basis["oid-1"] = ExitCostInfo(
        entry_cost=Decimal("50"), entry_size=Decimal("100"), slug="m1"
    )

    async def failing_cancel(client, oid):
        raise RuntimeError("cancel raced/failed")

    monkeypatch.setattr(fills_mod, "cancel_order", failing_cancel)

    def chunk(status: str) -> UserTrade:
        return UserTrade(
            event_type="trade",
            id="dd-2",
            asset_id="tok-yes",
            market="market-A",
            side="SELL",
            price=Decimal("0.40"),
            size=Decimal("60"),
            outcome="YES",
            status=status,
            timestamp="2026-07-03T12:00:00Z",
            maker_orders=[],
            taker_order_id="oid-1",
        )

    await handle_trade(MagicMock(), chunk("MINED"), farm_state, MagicMock())
    assert pos.yes_shares == Decimal("40"), "60 booked; residual held"
    assert "oid-1" in farm_state.pending_exit_order_ids, "failed cancel re-registers the same oid"
    assert farm_state.session_loss == Decimal("6")  # 30 basis − 24 proceeds

    await handle_trade(MagicMock(), chunk("CONFIRMED"), farm_state, MagicMock())  # replay

    assert pos.yes_shares == Decimal("40"), "replay on the re-registered oid must not re-book"
    assert farm_state.session_loss == Decimal("6")


# ── balance:0 stays on the phantom path (clamp NOT engaged) ─────────────────────


async def test_zero_balance_rejection_does_not_engage_clamp(
    farm_state: FarmState, strat_calls, no_cancel, monkeypatch
):
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("20")
    pos.yes_cost_basis = Decimal("6")

    attempts: list = []

    async def place(client, token_id, side, amount):
        attempts.append(side)
        raise zero_balance_exc()

    async def fake_refresh(client, token_id):
        return True

    monkeypatch.setattr(exits_mod, "place_market_order", place)
    monkeypatch.setattr(exits_mod, "refresh_conditional_balance", fake_refresh)

    await exit_position_leg(
        MagicMock(), farm_state, pos.market.yes_token_id, Decimal("20"), "market-A", "m1", "YES"
    )

    assert counts(strat_calls, "exit_clamp") == 0, "balance:0 must NOT clamp (shares diverged=0)"
    assert attempts == ["SELL", "SELL"], "zero-balance path refreshes cache + retries the FAK once"
    assert "YES" in pos.zero_balance_since, "first balance:0 starts the give-up clock"
    assert pos.yes_shares == Decimal("20"), "leg left held (entry may be unmined)"


# ── dust: clamp shrinks below min_order_size → dust write-off, no loop ──────────


async def test_clamp_into_dust_writes_off_without_retry_loop(
    farm_state: FarmState, strat_calls, no_cancel, monkeypatch
):
    prime_no_leg(farm_state, shares="20", cost="10")
    pos = farm_state.positions["market-A"]

    attempts: list = []

    async def place(client, token_id, side, amount):
        attempts.append(Decimal(str(amount)))
        # Exchange holds only 3 shares — below min_order_size (5).
        raise partial_exc(3000000, 20000000)

    monkeypatch.setattr(exits_mod, "place_market_order", place)

    await exit_position_leg(
        MagicMock(), farm_state, pos.market.no_token_id, Decimal("20"), "market-A", "m1", "NO"
    )

    assert counts(strat_calls, "exit_clamp") == 1, "clamp shrinks the leg to 3"
    assert counts(strat_calls, "dust_writeoff") == 1, "then the 3-share dust is written off"
    assert pos.no_shares == Decimal("0"), "un-sellable dust leg cleared"
    assert len(attempts) == 1, "the clamped retry hits dust BEFORE placing — no loop"


# ── recursion cap: perpetually-shrinking partial rejection ─────────────────────


async def test_clamp_recursion_cap_stops_and_leaves_leg_held(
    farm_state: FarmState, strat_calls, no_cancel, monkeypatch
):
    prime_no_leg(farm_state, shares="100", cost="50")
    pos = farm_state.positions["market-A"]

    # Every attempt rejects with a strictly-shrinking balance that stays above min_order_size,
    # so the clamp keeps engaging until the recursion guard trips.
    balances = iter([90, 80, 70, 60, 50, 40, 30, 20])
    attempts: list = []

    async def place(client, token_id, side, amount):
        attempts.append(Decimal(str(amount)))
        bal = next(balances)
        raise partial_exc(bal * 1_000_000, 100_000_000)

    monkeypatch.setattr(exits_mod, "place_market_order", place)

    # Must NOT raise — the cap returns quietly, leaving the leg for the sweep/reconcile.
    await exit_position_leg(
        MagicMock(), farm_state, pos.market.no_token_id, Decimal("100"), "market-A", "m1", "NO"
    )

    assert len(attempts) == MAX_CLAMP_RETRIES + 1, "one initial + MAX_CLAMP_RETRIES clamped retries"
    assert counts(strat_calls, "exit_clamp") == MAX_CLAMP_RETRIES, "one clamp per retry below cap"
    assert pos.no_shares == Decimal("50"), "leg left held at the last clamped size (not zeroed)"


# ── clamp wiring on the GTC-fallback except block (not just FAK) ────────────────


async def test_partial_balance_on_gtc_fallback_also_clamps(
    farm_state: FarmState, strat_calls, no_cancel, monkeypatch
):
    # FAK fails with a NON-balance error → falls through to the GTC fallback, whose limit order
    # then raises the partial-balance 400. The GTC except block must clamp too (mirrors the FAK).
    prime_no_leg(farm_state, shares="20", cost="12")
    pos = farm_state.positions["market-A"]

    async def fak_generic_fail(client, token_id, side, amount):
        raise RuntimeError("FAK unavailable — fall through to GTC")

    async def fake_get_order_book(token_id):
        return OrderBook(
            market="market-A",
            asset_id=token_id,
            timestamp="2026-07-03T12:00:00Z",
            bids=[BookLevel(price=Decimal("0.40"), size=Decimal("30"))],
            asks=[BookLevel(price=Decimal("0.55"), size=Decimal("30"))],
            min_order_size=Decimal("1"),
            tick_size=Decimal("0.01"),
            neg_risk=False,
            hash="x",
        )

    gtc_attempts: list = []

    async def gtc_place(client, order, post_only=False):
        gtc_attempts.append(Decimal(str(order.size)))
        if len(gtc_attempts) == 1:
            raise partial_exc(12000000, 20000000)  # exchange holds 12 of the 20 GTC-quoted
        return "gtc-clamp-ok"

    monkeypatch.setattr(exits_mod, "place_market_order", fak_generic_fail)
    monkeypatch.setattr(exits_mod, "get_order_book", fake_get_order_book)
    monkeypatch.setattr(exits_mod, "place_limit_order", gtc_place)

    await exit_position_leg(
        MagicMock(), farm_state, pos.market.no_token_id, Decimal("20"), "market-A", "m1", "NO"
    )

    assert counts(strat_calls, "exit_clamp") == 1, "the GTC except block must clamp as well"
    assert pos.no_shares == Decimal("12"), "leg clamped to the exchange balance on the GTC path"
    assert gtc_attempts == [Decimal("20"), Decimal("12")], "full-size GTC fails, clamped retry"
    assert "gtc-clamp-ok" in farm_state.pending_exit_order_ids


# ── Part-B fixes: orphan/zero-tracked-leg proportional cost + no-op fix ─────────


async def test_orphan_clamp_scales_entry_cost_proportionally(
    farm_state: FarmState, strat_calls, no_cancel, monkeypatch
):
    attempts: list = []

    async def place(client, token_id, side, amount):
        attempts.append(Decimal(str(amount)))
        if len(attempts) == 1:
            raise partial_exc(15_000_000, 20_000_000)
        return "orphan-resell-oid"

    monkeypatch.setattr(exits_mod, "place_market_order", place)

    await exit_position_leg(
        MagicMock(), farm_state, "tok-orphan", Decimal("20"), "market-GONE", "gone-slug", "YES",
        cancel_resting=False, entry_cost=Decimal("10"),
    )

    assert attempts == [Decimal("20"), Decimal("15")]
    cost_info = farm_state.exit_cost_basis["orphan-resell-oid"]
    assert cost_info.entry_cost == Decimal("7.5"), (
        "orphan re-drive must scale entry_cost proportionally, not forward the full value"
    )
    assert cost_info.entry_size == Decimal("15")

    # Fill it and confirm the booked P&L reflects the SCALED basis (break-even), not a phantom
    # loss from booking the full $10 against only 15 shares' worth of proceeds.
    loss_before = farm_state.session_loss
    await handle_trade(
        MagicMock(),
        UserTrade(
            event_type="trade", id="t1", asset_id="tok-orphan", market="market-GONE",
            side="SELL", price=Decimal("0.50"), size=Decimal("15"), outcome="YES",
            status="MINED", timestamp="2026-07-05T00:00:00Z", maker_orders=[],
            taker_order_id="orphan-resell-oid",
        ),
        farm_state, MagicMock(),
    )
    assert farm_state.session_loss - loss_before == Decimal("0"), (
        "scaled basis books break-even; the old bug would book a phantom ~$2.50 loss"
    )


async def test_clamp_retries_at_balance_when_tracked_leg_already_zero(
    farm_state: FarmState, strat_calls, no_cancel, monkeypatch
):
    pos = farm_state.positions["market-A"]
    assert pos.no_shares == Decimal("0"), "fixture default: NO leg untouched/flat"

    attempts: list = []

    async def place(client, token_id, side, amount):
        attempts.append(Decimal(str(amount)))
        if len(attempts) == 1:
            raise partial_exc(15_000_000, 20_000_000)
        return "resell-oid"

    monkeypatch.setattr(exits_mod, "place_market_order", place)

    await exit_position_leg(
        MagicMock(), farm_state, pos.market.no_token_id, Decimal("20"), "market-A", "m1", "NO",
        entry_cost=Decimal("10"),
    )

    assert attempts == [Decimal("20"), Decimal("15")], (
        "must retry at the exchange balance (15), not no-op at min(balance, 0) = 0"
    )
    assert "resell-oid" in farm_state.pending_exit_order_ids
    cost_info = farm_state.exit_cost_basis["resell-oid"]
    assert cost_info.entry_cost == Decimal("7.5"), "scaled proportionally like the orphan branch"
    assert cost_info.entry_size == Decimal("15")
