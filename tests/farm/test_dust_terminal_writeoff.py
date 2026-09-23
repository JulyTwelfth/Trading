"""Fix B — dust-terminal SELL rejections.

Incident: a market-SELL rejected with a CLOB-reported balance BELOW the dust floor
(EXIT_DUST_BALANCE_SHARES = 0.1 shares) is unsellable residue regardless of retry. Before the fix
the exit looped ERROR -> GTC fallback -> identical rejection forever. Now such a rejection (in
EITHER the FAK or the GTC handler, AFTER the zero-balance / overcommit branches) writes the leg
off terminally: the phantom leg is cleared, no GTC fallback is attempted, and it is NOT recorded
as a market failure (the CLOB, not us, is the source of truth).

A partial-balance rejection whose balance is ABOVE the dust floor is a different case: it is NOT
written off and does NOT fall through to the GTC fallback — the elif after the dust guard routes
it into the stale-residual clamp (clamp_leg_to_balance), which shrinks the leg to the exchange
balance and retries within the same path (see tests/fills/test_stale_residual_clamp.py).

Harness mirrors tests/farm/test_exit_balance_guard.py: monkeypatch exits_mod.place_market_order /
place_limit_order / get_order_book. The tests import only exit_position_leg (present on HEAD), so a
revert of the fix reds the terminal cases via ASSERTION, not a collection error.
"""

from decimal import Decimal
from unittest.mock import MagicMock

from py_clob_client_v2.exceptions import PolyApiException

from app.bot.schemas import BookLevel, OrderBook
from app.farm import exits as exits_mod
from app.farm.exits import exit_position_leg

# The production incident string: 3156 µ-shares (0.003156) held vs 12.76 shares ordered.
INCIDENT = "not enough balance -> balance: 3156, order amount: 12760000"
OVERCOMMIT = (
    "not enough balance / allowance: the balance is not enough -> balance: 26120900, "
    "sum of active orders: 6200000, sum of matched orders: 13000000, "
    "order amount (inc. fees): 12800000"
)


def dust_exc(balance: int = 3156, order_amount: int = 12760000) -> PolyApiException:
    return PolyApiException(
        error_msg={
            "error": (
                "not enough balance / allowance: the balance is not enough "
                f"-> balance: {balance}, order amount: {order_amount}"
            )
        }
    )


def a_book(token_id: str) -> OrderBook:
    return OrderBook(
        market="market-A",
        asset_id=token_id,
        timestamp="2026-07-04T10:00:00Z",
        bids=[BookLevel(price=Decimal("0.40"), size=Decimal("10"))],
        asks=[BookLevel(price=Decimal("0.55"), size=Decimal("10"))],
        min_order_size=Decimal("1"),
        tick_size=Decimal("0.01"),
        neg_risk=False,
        hash="x",
    )


def has_failures(state, cid: str = "market-A") -> bool:
    health = state.health.get(cid)
    return bool(health and (health.recent_failures or health.paused_until))


async def test_dust_rejection_is_terminal(farm_state, monkeypatch):
    book_calls: list = []

    async def fail_market(client, token_id, side, amount):
        raise dust_exc()

    async def spy_get_order_book(token_id):
        book_calls.append(token_id)
        raise RuntimeError("GTC fallback must not run on a dust write-off")

    async def fake_cancel_orders(client, *oids):
        return None

    monkeypatch.setattr(exits_mod, "place_market_order", fail_market)
    monkeypatch.setattr(exits_mod, "get_order_book", spy_get_order_book)
    monkeypatch.setattr(exits_mod, "cancel_orders", fake_cancel_orders)

    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("20")
    pos.yes_cost_basis = Decimal("6")

    await exit_position_leg(
        MagicMock(), farm_state, pos.market.yes_token_id, Decimal("20"), "market-A", "m1", "YES"
    )

    assert pos.yes_shares == Decimal("0"), "the dust rejection must clear the phantom leg"
    assert pos.yes_cost_basis == Decimal("0")
    assert book_calls == [], "no GTC fallback: get_order_book must not be called"
    assert not has_failures(farm_state), "a dust write-off must not be recorded as a market failure"


async def test_dust_rejection_orphan_no_crash(farm_state, monkeypatch):
    book_calls: list = []

    async def fail_market(client, token_id, side, amount):
        raise dust_exc()

    async def spy_get_order_book(token_id):
        book_calls.append(token_id)
        raise RuntimeError("GTC fallback must not run on a dust write-off")

    monkeypatch.setattr(exits_mod, "place_market_order", fail_market)
    monkeypatch.setattr(exits_mod, "get_order_book", spy_get_order_book)

    # No tracked position for this condition — a pure orphan sweep.
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

    assert farm_state.pending_exit_order_ids == set()
    assert book_calls == [], "orphan dust rejection must be terminal, no GTC fallback"
    assert not has_failures(farm_state, "market-GONE")


async def test_balance_above_dust_clamps_not_gtc_fallthrough(farm_state, monkeypatch):
    # A partial-balance rejection whose balance is ABOVE the dust floor is NOT written off and does
    # NOT fall through to the GTC fallback within the same call — it is routed into the clamp path
    # (the elif after the dust `if`). Realistic shape: the leg tracks only 20 shares but a STALE
    # driven size of 25 was submitted (e.g. an on-chain reconcile drives a size different from what
    # is locally tracked), so the CLOB's "order amount" truthfully reads 25 (the placed order). The
    # exchange reports a lower-but-real balance of 20 — which is >= the tracked 20 but < the driven
    # 25, a fully legitimate "not enough balance". So `new = min(20, 20) = 20` and `removed = 0`:
    # the "spurious/stale — balance >= tracked" branch, which retries at the TRACKED size via the
    # FAK path (no shrink, no exit_clamp) and never reaches get_order_book / place_limit_order.
    book_calls: list = []
    place_calls: list = []
    market_calls: list = []

    async def fail_then_ok_market(client, token_id, side, amount):
        market_calls.append(Decimal(str(amount)))
        if len(market_calls) == 1:
            raise dust_exc(balance=20_000_000, order_amount=25_000_000)
        return "clamp-ok"

    async def spy_get_order_book(token_id):
        book_calls.append(token_id)
        return a_book(token_id)

    async def spy_limit(client, order, post_only=False):
        place_calls.append(order.token_id)
        return "gtc-oid"

    async def fake_cancel_orders(client, *oids):
        return None

    monkeypatch.setattr(exits_mod, "place_market_order", fail_then_ok_market)
    monkeypatch.setattr(exits_mod, "get_order_book", spy_get_order_book)
    monkeypatch.setattr(exits_mod, "place_limit_order", spy_limit)
    monkeypatch.setattr(exits_mod, "cancel_orders", fake_cancel_orders)

    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("20")
    pos.yes_cost_basis = Decimal("6")

    await exit_position_leg(
        MagicMock(), farm_state, pos.market.yes_token_id, Decimal("25"), "market-A", "m1", "YES"
    )

    assert market_calls == [Decimal("25"), Decimal("20")], (
        "spurious rejection (balance >= tracked) retries at the tracked size, not the stale "
        "driven size"
    )
    assert pos.yes_shares == Decimal("20"), "balance 20 >= tracked 20 — leg not shrunk"
    assert pos.yes_cost_basis == Decimal("6"), "cost basis untouched on a spurious rejection"
    assert book_calls == [], "clamp resolves in the FAK path — no GTC fallback"
    assert place_calls == [], "the GTC fallback (place_limit_order) must not run"
    assert "clamp-ok" in farm_state.pending_exit_order_ids, "the clamped retry is registered"


async def test_zero_balance_still_takes_strike_path(farm_state, monkeypatch):
    async def fail_market_zero(client, token_id, side, amount):
        raise dust_exc(balance=0, order_amount=20000000)  # balance: 0

    async def fake_cancel_orders(client, *oids):
        return None

    async def fake_refresh(client, token_id):
        return True

    monkeypatch.setattr(exits_mod, "place_market_order", fail_market_zero)
    monkeypatch.setattr(exits_mod, "cancel_orders", fake_cancel_orders)
    monkeypatch.setattr(exits_mod, "refresh_conditional_balance", fake_refresh)

    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("20")
    pos.yes_cost_basis = Decimal("6")

    await exit_position_leg(
        MagicMock(), farm_state, pos.market.yes_token_id, Decimal("20"), "market-A", "m1", "YES"
    )

    # balance:0 (parsed as 0 shares) must NOT be written off — the `0 < balance` guard routes it to
    # the existing zero-balance phantom-strike path, which needs multiple strikes before clearing.
    assert "YES" in pos.zero_balance_since, "balance:0 must take the phantom-strike path"
    assert pos.yes_shares == Decimal("20"), "a first balance:0 strike must not clear the leg"


async def test_gtc_dust_rejection_is_terminal(farm_state, monkeypatch):
    async def fail_market_generic(client, token_id, side, amount):
        raise RuntimeError("no match")  # generic — routes past the FAK branches to the GTC fallback

    async def fake_get_order_book(token_id):
        return a_book(token_id)

    async def fail_limit_dust(client, order, post_only=False):
        raise dust_exc()

    async def fake_cancel_orders(client, *oids):
        return None

    monkeypatch.setattr(exits_mod, "place_market_order", fail_market_generic)
    monkeypatch.setattr(exits_mod, "get_order_book", fake_get_order_book)
    monkeypatch.setattr(exits_mod, "place_limit_order", fail_limit_dust)
    monkeypatch.setattr(exits_mod, "cancel_orders", fake_cancel_orders)

    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("20")
    pos.yes_cost_basis = Decimal("6")

    await exit_position_leg(
        MagicMock(), farm_state, pos.market.yes_token_id, Decimal("20"), "market-A", "m1", "YES"
    )

    assert pos.yes_shares == Decimal("0"), "a GTC dust rejection must write off the leg terminally"
    assert farm_state.pending_exit_order_ids == set()
    assert not has_failures(farm_state), "a GTC dust write-off must not be recorded as a failure"


async def test_overcommit_never_written_off(farm_state, monkeypatch):
    book_calls: list = []

    async def fail_market_overcommit(client, token_id, side, amount):
        raise PolyApiException(error_msg={"error": OVERCOMMIT})

    async def spy_get_order_book(token_id):
        book_calls.append(token_id)
        raise RuntimeError("overcommit must defer before the GTC fallback")

    async def fake_cancel_orders(client, *oids):
        return None

    monkeypatch.setattr(exits_mod, "place_market_order", fail_market_overcommit)
    monkeypatch.setattr(exits_mod, "get_order_book", spy_get_order_book)
    monkeypatch.setattr(exits_mod, "cancel_orders", fake_cancel_orders)

    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("20")
    pos.yes_cost_basis = Decimal("6")

    await exit_position_leg(
        MagicMock(), farm_state, pos.market.yes_token_id, Decimal("20"), "market-A", "m1", "YES"
    )

    assert pos.yes_shares == Decimal("20"), "an overcommit rejection defers, never written off"
    assert book_calls == [], "overcommit must defer before the GTC fallback"
    assert farm_state.pending_exit_order_ids == set()
