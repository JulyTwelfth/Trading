from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from py_clob_client_v2.exceptions import PolyApiException

from app.bot.schemas import BookLevel, OrderBook
from app.constants import ZERO_BALANCE_GIVEUP_SECONDS
from app.farm import exits as exits_mod
from app.farm import worker as worker_mod
from app.farm.exits import (
    exit_held_legs,
    exit_position_leg,
    is_zero_share_balance_rejection,
    record_zero_balance_strike,
)


def zero_balance_exc(order_amount: int = 20000000) -> PolyApiException:
    return PolyApiException(
        error_msg={
            "error": (
                "not enough balance / allowance: the balance is not enough "
                f"-> balance: 0, order amount: {order_amount}"
            )
        }
    )


def partial_balance_exc(balance_micro: int, order_micro: int) -> PolyApiException:
    return PolyApiException(
        error_msg={
            "error": (
                "not enough balance / allowance: the balance is not enough "
                f"-> balance: {balance_micro}, order amount: {order_micro}"
            )
        }
    )


@pytest.fixture
def capture_sells(monkeypatch):
    calls: list = []

    async def fail_market(client, token_id, side, amount):
        calls.append(("FAK", token_id, side, amount))
        raise zero_balance_exc()

    async def fail_limit(client, order, post_only=False):
        calls.append(("GTC", order.token_id, order.side, order.size))
        raise zero_balance_exc()

    async def fake_cancel_orders(client, *oids):
        return None

    async def fake_get_order_book(token_id):
        return OrderBook(
            market="market-A",
            asset_id=token_id,
            timestamp="2026-06-09T10:00:00Z",
            bids=[BookLevel(price=Decimal("0.40"), size=Decimal("10"))],
            asks=[BookLevel(price=Decimal("0.55"), size=Decimal("10"))],
            min_order_size=Decimal("1"),
            tick_size=Decimal("0.01"),
            neg_risk=False,
            hash="x",
        )

    async def fake_refresh(client, token_id):
        return True

    monkeypatch.setattr(exits_mod, "place_market_order", fail_market)
    monkeypatch.setattr(exits_mod, "place_limit_order", fail_limit)
    monkeypatch.setattr(exits_mod, "cancel_orders", fake_cancel_orders)
    monkeypatch.setattr(exits_mod, "get_order_book", fake_get_order_book)
    monkeypatch.setattr(exits_mod, "refresh_conditional_balance", fake_refresh)
    return calls


def test_matcher_flags_zero_share_sell_not_buy_overcommit():
    assert is_zero_share_balance_rejection(zero_balance_exc())
    overcommit = PolyApiException(
        error_msg={
            "error": (
                "not enough balance / allowance: the balance is not enough -> "
                "balance: 26120900, sum of active orders: 6200000, sum of matched "
                "orders: 13000000, order amount (inc. fees): 12800000"
            )
        }
    )
    assert not is_zero_share_balance_rejection(overcommit)


async def test_balance_zero_sell_strikes_without_clearing_or_gtc_fallback(
    farm_state, capture_sells
):
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("20")
    pos.yes_cost_basis = Decimal("6")

    await exit_position_leg(
        MagicMock(),
        farm_state,
        pos.market.yes_token_id,
        Decimal("20"),
        "market-A",
        "m1",
        "YES",
    )

    assert pos.yes_shares == Decimal("20"), "one balance:0 must not clear (entry may be unmined)"
    assert "YES" in pos.zero_balance_since, "first balance:0 starts the give-up clock"
    assert [c[0] for c in capture_sells] == ["FAK", "FAK"], (
        "balance:0 refreshes the cache and retries the FAK once — no GTC fallback"
    )


async def test_sweep_clears_phantom_leg_after_giveup_timeout(farm_state, capture_sells):
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("20")
    pos.yes_cost_basis = Decimal("6")

    await exit_held_legs(MagicMock(), farm_state, cancel_resting=True, skip_in_flight=True)
    assert pos.yes_shares == Decimal("20"), "must not clear before the give-up window elapses"
    assert "YES" in pos.zero_balance_since

    pos.zero_balance_since["YES"] = datetime.now(timezone.utc) - timedelta(
        seconds=ZERO_BALANCE_GIVEUP_SECONDS + 1
    )
    await exit_held_legs(MagicMock(), farm_state, cancel_resting=True, skip_in_flight=True)
    assert pos.yes_shares == Decimal("0"), "leg must clear once the give-up window elapses"
    assert pos.yes_cost_basis == Decimal("0")
    assert len(capture_sells) == 4, "2 FAK (refresh+retry) per sweep until clear, no GTC"

    await exit_held_legs(MagicMock(), farm_state, cancel_resting=True, skip_in_flight=True)
    assert len(capture_sells) == 4, "phantom leg already cleared — the sweep must not re-sell it"


async def test_giveup_resets_clock_for_reentered_leg(farm_state, capture_sells):
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("20")
    pos.yes_cost_basis = Decimal("6")

    await exit_held_legs(MagicMock(), farm_state, cancel_resting=True, skip_in_flight=True)
    pos.zero_balance_since["YES"] = datetime.now(timezone.utc) - timedelta(
        seconds=ZERO_BALANCE_GIVEUP_SECONDS + 1
    )
    await exit_held_legs(MagicMock(), farm_state, cancel_resting=True, skip_in_flight=True)
    assert pos.yes_shares == Decimal("0"), "phantom leg cleared after the give-up timeout"
    assert pos.zero_balance_since == {}, "give-up must reset the clock"

    pos.yes_shares = Decimal("20")
    pos.yes_cost_basis = Decimal("13.4")
    await exit_held_legs(MagicMock(), farm_state, cancel_resting=True, skip_in_flight=True)

    assert pos.yes_shares == Decimal("20"), "re-entered leg cleared on its first rejection"
    assert "YES" in pos.zero_balance_since


async def test_balance_zero_on_gtc_fallback_also_strikes(farm_state, monkeypatch):

    async def fail_market_generic(client, token_id, side, amount):
        raise RuntimeError("no match")

    async def fail_limit_balance(client, order, post_only=False):
        raise zero_balance_exc()

    async def fake_cancel_orders(client, *oids):
        return None

    async def fake_get_order_book(token_id):
        return OrderBook(
            market="market-A",
            asset_id=token_id,
            timestamp="2026-06-11T17:30:00Z",
            bids=[BookLevel(price=Decimal("0.40"), size=Decimal("10"))],
            asks=[BookLevel(price=Decimal("0.55"), size=Decimal("10"))],
            min_order_size=Decimal("1"),
            tick_size=Decimal("0.01"),
            neg_risk=False,
            hash="x",
        )

    monkeypatch.setattr(exits_mod, "place_market_order", fail_market_generic)
    monkeypatch.setattr(exits_mod, "place_limit_order", fail_limit_balance)
    monkeypatch.setattr(exits_mod, "cancel_orders", fake_cancel_orders)
    monkeypatch.setattr(exits_mod, "get_order_book", fake_get_order_book)

    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("20")
    pos.yes_cost_basis = Decimal("6")

    await exit_position_leg(
        MagicMock(),
        farm_state,
        pos.market.yes_token_id,
        Decimal("20"),
        "market-A",
        "m1",
        "YES",
    )

    assert pos.yes_shares == Decimal("20")
    assert "YES" in pos.zero_balance_since


async def test_successful_sell_resets_strikes(farm_state, monkeypatch):
    attempts: list = []

    async def market_order_succeeds_third_try(client, token_id, side, amount):
        attempts.append(side)
        if len(attempts) < 3:
            raise zero_balance_exc()
        return "fak-ok"

    async def fake_cancel_orders(client, *oids):
        return None

    async def fake_refresh(client, token_id):
        return True

    monkeypatch.setattr(exits_mod, "place_market_order", market_order_succeeds_third_try)
    monkeypatch.setattr(exits_mod, "cancel_orders", fake_cancel_orders)
    monkeypatch.setattr(exits_mod, "refresh_conditional_balance", fake_refresh)

    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("20")
    pos.yes_cost_basis = Decimal("6")

    for _ in range(3):
        await exit_position_leg(
            MagicMock(),
            farm_state,
            pos.market.yes_token_id,
            Decimal("20"),
            "market-A",
            "m1",
            "YES",
        )

    assert pos.zero_balance_since == {}, "successful placement must reset the give-up clock"
    assert pos.yes_shares == Decimal("20"), "shares stay booked until the exit FILL reconciles"
    assert "fak-ok" in farm_state.pending_exit_order_ids


async def test_balance_zero_without_position_is_terminal(farm_state, capture_sells):
    await exit_position_leg(
        MagicMock(),
        farm_state,
        "tok-orphan",
        Decimal("20"),
        "market-GONE",
        "gone-slug",
        "YES",
        cancel_resting=False,
    )

    assert [c[0] for c in capture_sells] == ["FAK", "FAK"]
    assert farm_state.pending_exit_order_ids == set()


def test_zero_balance_giveup_is_time_based_not_count_based(farm_state):
    t0 = datetime(2026, 6, 18, 12, 0, 0, tzinfo=timezone.utc)
    assert record_zero_balance_strike(farm_state, "market-A", "YES", now=t0) is False
    for _ in range(50):
        assert record_zero_balance_strike(farm_state, "market-A", "YES", now=t0) is False
    near = t0 + timedelta(seconds=ZERO_BALANCE_GIVEUP_SECONDS - 1)
    assert record_zero_balance_strike(farm_state, "market-A", "YES", now=near) is False
    past = t0 + timedelta(seconds=ZERO_BALANCE_GIVEUP_SECONDS + 1)
    assert record_zero_balance_strike(farm_state, "market-A", "YES", now=past) is True


async def test_exit_retry_loop_drives_held_legs_each_tick(farm_state, monkeypatch):
    calls: list = []

    async def fake_exit_held_legs(
        client, state, *, cancel_resting, skip_in_flight, force_dump=False
    ):
        calls.append((cancel_resting, skip_in_flight))

    class Stop(Exception):
        pass

    async def stop(_seconds):
        raise Stop

    monkeypatch.setattr(worker_mod, "exit_held_legs", fake_exit_held_legs)
    monkeypatch.setattr(worker_mod.asyncio, "sleep", stop)

    with pytest.raises(Stop):
        await worker_mod.exit_retry_loop(MagicMock(), farm_state)

    assert calls == [(True, True)], "must sweep held legs with cancel_resting + skip_in_flight"


async def test_balance_zero_refreshes_cache_and_retries_sell(farm_state, monkeypatch):
    attempts: list = []

    async def market_order(client, token_id, side, amount):
        attempts.append(side)
        if len(attempts) == 1:
            raise zero_balance_exc()
        return "fak-ok"

    async def fake_cancel_orders(client, *oids):
        return None

    refreshed: list = []

    async def fake_refresh(client, token_id):
        refreshed.append(token_id)
        return True

    client = MagicMock()
    monkeypatch.setattr(exits_mod, "place_market_order", market_order)
    monkeypatch.setattr(exits_mod, "cancel_orders", fake_cancel_orders)
    monkeypatch.setattr(exits_mod, "refresh_conditional_balance", fake_refresh)

    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("20")
    pos.yes_cost_basis = Decimal("6")

    await exit_position_leg(
        client, farm_state, pos.market.yes_token_id, Decimal("20"), "market-A", "m1", "YES"
    )

    assert len(attempts) == 2, "must retry the SELL once after refreshing the cache"
    assert refreshed == [pos.market.yes_token_id], "must call refresh with the token_id"
    assert "fak-ok" in farm_state.pending_exit_order_ids, "post-refresh SELL must be registered"
    assert pos.zero_balance_since == {}, "a successful sell leaves no give-up clock"


async def test_partial_balance_clamps_and_retries_without_zero_balance_strike(
    farm_state, monkeypatch
):
    # A partial-balance 400 (exchange holds 12 of the tracked 20 shares) is a DIFFERENT failure
    # from balance:0: it must clamp the leg to 12 and retry, and must NOT record a zero-balance
    # strike (the give-up clock is only for a genuine 0-share phantom).
    attempts: list = []

    async def market_order(client, token_id, side, amount):
        attempts.append(Decimal(str(amount)))
        if len(attempts) == 1:
            raise partial_balance_exc(12000000, 20000000)
        return "clamp-ok"

    async def fake_cancel_orders(client, *oids):
        return None

    async def fake_refresh(client, token_id):
        raise AssertionError("refresh_conditional_balance is the zero-balance path — not clamp")

    monkeypatch.setattr(exits_mod, "place_market_order", market_order)
    monkeypatch.setattr(exits_mod, "cancel_orders", fake_cancel_orders)
    monkeypatch.setattr(exits_mod, "refresh_conditional_balance", fake_refresh)

    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("20")
    pos.yes_cost_basis = Decimal("6")

    await exit_position_leg(
        MagicMock(), farm_state, pos.market.yes_token_id, Decimal("20"), "market-A", "m1", "YES"
    )

    assert attempts == [Decimal("20"), Decimal("12")], "full-size SELL fails, retry clamped to 12"
    assert pos.yes_shares == Decimal("12"), "leg clamped to the exchange-reported balance"
    assert pos.yes_cost_basis == Decimal("3.6"), "cost basis scaled proportionally (6 * 12/20)"
    assert "clamp-ok" in farm_state.pending_exit_order_ids, "the clamped retry is registered"
    assert pos.zero_balance_since == {}, "a partial-balance clamp must not touch the give-up clock"
