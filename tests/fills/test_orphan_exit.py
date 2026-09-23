"""#7 — a fill that lands on a registered order whose position is already gone (a stale
requote oid the registry never dropped) must not strand shares for manual exit. The
shares are staged into the same MINED->FAK-SELL machinery as a normal fill and sold
once on-chain, with PnL booked even though no position object exists.

Covers the full flow: stage -> MINED fires the SELL -> the SELL fill books realized
PnL, plus edge cases (zero size, NO leg, dedup replay, remainder cancel, exclusion does
not block the exit)."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.schemas import UserTrade, UserTradeMakerOrder
from app.farm import exits as exits_mod
from app.farm import fills as fills_mod
from app.farm.fills import handle_exit_fill, handle_trade
from app.farm.schemas import ExitCostInfo, FarmState, MarketHealth, OrderInfo
from app.farm.volatility import is_blacklisted

GONE = "market-GONE"  # a condition_id with no live position
TOK_YES = "tok-gone-yes"
TOK_NO = "tok-gone-no"


@pytest.fixture
def stub_io(monkeypatch):
    """Capture the SELL (place_market_order) and cancel_order calls fills/exits make."""
    sells: list[tuple[str, str, float]] = []
    cancels: list[str] = []

    async def fake_place_market_order(client, token_id, side, size):
        sells.append((token_id, side, size))
        return f"exit-oid-{len(sells)}"

    async def fake_cancel_order(client, oid):
        cancels.append(oid)

    monkeypatch.setattr(exits_mod, "place_market_order", fake_place_market_order)
    monkeypatch.setattr(fills_mod, "cancel_order", fake_cancel_order)
    return {"sells": sells, "cancels": cancels}


def register_orphan(state: FarmState, *, oid: str, token: str, outcome: str) -> None:
    """Put an oid in the registry pointing at a market that has no position (the orphan)."""
    state.order_registry[oid] = OrderInfo(condition_id=GONE, outcome=outcome, token_id=token)
    assert GONE not in state.positions


def orphan_fill(
    *,
    trade_id="trade-orphan",
    oid="stale-oid",
    token=TOK_YES,
    outcome="YES",
    size="20",
    price="0.40",
) -> UserTrade:
    return UserTrade(
        event_type="trade",
        id=trade_id,
        asset_id=token,
        market=GONE,
        side="BUY",
        price=Decimal(price),
        size=Decimal(size),
        outcome=outcome,
        status="MATCHED",
        timestamp="2026-06-10T08:00:00Z",
        taker_order_id="someone-elses-taker",  # not ours; our side is the maker
        maker_orders=[
            UserTradeMakerOrder(
                asset_id=token,
                order_id=oid,
                matched_amount=Decimal(size),
                outcome=outcome,
                owner="0xowner",
                price=Decimal(price),
            )
        ],
    )


def mined_frame(trade_id="trade-orphan") -> UserTrade:
    return UserTrade(
        event_type="trade",
        id=trade_id,
        asset_id=TOK_YES,
        market=GONE,
        side="BUY",
        price=Decimal("0.40"),
        size=Decimal("20"),
        outcome="YES",
        status="MINED",
        timestamp="2026-06-10T08:00:01Z",
        maker_orders=[],
        taker_order_id="someone-elses-taker",
    )


# ── core: stage on MATCHED ───────────────────────────────────────────────────


async def test_orphan_fill_stages_exit_books_volume_excludes_no_immediate_sell(
    farm_state: FarmState, stub_io
):
    register_orphan(farm_state, oid="stale-oid", token=TOK_YES, outcome="YES")

    await handle_trade(MagicMock(), orphan_fill(), farm_state, AsyncMock())

    # No FAK SELL yet — shares aren't MINED, so we only stage.
    assert stub_io["sells"] == []
    # Market excluded so we never re-quote it.
    assert is_blacklisted(farm_state, GONE)
    # Volume booked for the acquired shares (20 * 0.40).
    assert farm_state.total_volume == Decimal("8")
    # Staged for the MINED handler.
    staged = farm_state.pending_fok_exits["trade-orphan"]
    assert (staged.token_id, staged.size, staged.outcome, staged.entry_cost) == (
        TOK_YES,
        Decimal("20"),
        "YES",
        Decimal("8"),
    )
    # Unfilled remainder of the stale order pulled.
    assert "stale-oid" in stub_io["cancels"]


async def test_orphan_fill_does_not_touch_live_positions(farm_state: FarmState, stub_io):
    """The existing market-A position must be left entirely untouched by an orphan fill
    on a different, dropped market."""
    before = farm_state.positions["market-A"].model_copy(deep=True)
    register_orphan(farm_state, oid="stale-oid", token=TOK_YES, outcome="YES")

    await handle_trade(MagicMock(), orphan_fill(), farm_state, AsyncMock())

    assert farm_state.positions["market-A"] == before


# ── MINED fires the SELL ─────────────────────────────────────────────────────


async def test_orphan_exit_fires_fak_sell_on_mined(farm_state: FarmState, stub_io):
    register_orphan(farm_state, oid="stale-oid", token=TOK_YES, outcome="YES")
    await handle_trade(MagicMock(), orphan_fill(), farm_state, AsyncMock())
    assert stub_io["sells"] == []  # nothing sold pre-MINED

    await handle_trade(MagicMock(), mined_frame(), farm_state, AsyncMock())

    assert stub_io["sells"] == [(TOK_YES, "SELL", 20.0)]
    # Staging cleared after firing so a replay can't double-sell.
    assert "trade-orphan" not in farm_state.pending_fok_exits


async def test_exclusion_does_not_block_the_orphan_exit(farm_state: FarmState, stub_io):
    """Orphan markets are excluded, but exclusion gates re-quoting, not exits — the MINED
    SELL must still fire on an excluded market."""
    register_orphan(farm_state, oid="stale-oid", token=TOK_YES, outcome="YES")
    await handle_trade(MagicMock(), orphan_fill(), farm_state, AsyncMock())
    assert is_blacklisted(farm_state, GONE)

    await handle_trade(MagicMock(), mined_frame(), farm_state, AsyncMock())

    assert stub_io["sells"] == [(TOK_YES, "SELL", 20.0)]


# ── full round-trip: PnL booked with no position ─────────────────────────────


async def test_orphan_roundtrip_books_realized_pnl_without_position(farm_state: FarmState, stub_io):
    register_orphan(farm_state, oid="stale-oid", token=TOK_YES, outcome="YES")
    # Bought 20 @ 0.40 = cost 8.
    await handle_trade(MagicMock(), orphan_fill(), farm_state, AsyncMock())
    await handle_trade(MagicMock(), mined_frame(), farm_state, AsyncMock())
    # The FAK SELL was placed as "exit-oid-1"; now it fills at 0.35 (proceeds 7).
    exit_fill = UserTrade(
        event_type="trade",
        id="trade-orphan-exit",
        asset_id=TOK_YES,
        market=GONE,
        side="SELL",
        price=Decimal("0.35"),
        size=Decimal("20"),
        outcome="YES",
        status="MINED",
        timestamp="2026-06-10T08:00:05Z",
        maker_orders=[],
        taker_order_id="exit-oid-1",
    )

    await handle_trade(MagicMock(), exit_fill, farm_state, AsyncMock())

    # Loss = entry 8 - proceeds 7 = 1, booked into session_loss with no position object.
    assert farm_state.session_loss == Decimal("1")
    assert GONE not in farm_state.positions
    assert "exit-oid-1" not in farm_state.pending_exit_order_ids


# ── edge cases ───────────────────────────────────────────────────────────────


async def test_zero_size_orphan_fill_is_noop(farm_state: FarmState, stub_io):
    register_orphan(farm_state, oid="stale-oid", token=TOK_YES, outcome="YES")

    await handle_trade(MagicMock(), orphan_fill(size="0"), farm_state, AsyncMock())

    assert stub_io["sells"] == []
    assert farm_state.pending_fok_exits == {}
    assert not is_blacklisted(farm_state, GONE)
    assert farm_state.total_volume == Decimal("0")


async def test_orphan_fill_on_no_leg_routes_to_no_token(farm_state: FarmState, stub_io):
    register_orphan(farm_state, oid="stale-no-oid", token=TOK_NO, outcome="NO")
    fill = orphan_fill(oid="stale-no-oid", token=TOK_NO, outcome="NO", size="15", price="0.60")

    await handle_trade(MagicMock(), fill, farm_state, AsyncMock())
    staged = farm_state.pending_fok_exits["trade-orphan"]
    assert (staged.token_id, staged.outcome, staged.size) == (TOK_NO, "NO", Decimal("15"))
    assert farm_state.total_volume == Decimal("9")  # 15 * 0.60

    await handle_trade(MagicMock(), mined_frame(), farm_state, AsyncMock())
    assert stub_io["sells"] == [(TOK_NO, "SELL", 15.0)]


async def test_orphan_fill_replay_does_not_double_count(farm_state: FarmState, stub_io):
    """A WS reconnect replays the same MATCHED frame — volume and staging must not double."""
    register_orphan(farm_state, oid="stale-oid", token=TOK_YES, outcome="YES")

    await handle_trade(MagicMock(), orphan_fill(), farm_state, AsyncMock())
    await handle_trade(MagicMock(), orphan_fill(), farm_state, AsyncMock())  # replay

    assert farm_state.total_volume == Decimal("8")
    assert len(farm_state.pending_fok_exits) == 1
    # And only one SELL ever fires.
    await handle_trade(MagicMock(), mined_frame(), farm_state, AsyncMock())
    assert stub_io["sells"] == [(TOK_YES, "SELL", 20.0)]


async def test_orphan_exit_fires_even_when_market_paused(farm_state: FarmState, stub_io):
    """A normal fill skips its MINED exit while paused because the held-share sweep backs
    it up — but an orphan has no position for the sweep to find, so it MUST exit while
    paused or the shares strand permanently."""
    register_orphan(farm_state, oid="stale-oid", token=TOK_YES, outcome="YES")
    farm_state.health[GONE] = MarketHealth(
        paused_until=datetime.now(timezone.utc) + timedelta(minutes=5)
    )

    await handle_trade(MagicMock(), orphan_fill(), farm_state, AsyncMock())
    await handle_trade(MagicMock(), mined_frame(), farm_state, AsyncMock())

    assert stub_io["sells"] == [(TOK_YES, "SELL", 20.0)], "paused orphan shares must still exit"


async def test_orphan_staging_survives_remainder_cancel_failure(
    farm_state: FarmState, stub_io, monkeypatch
):
    """If pulling the stale order's remainder raises, staging must still proceed — the
    cancel is best-effort and must never block the exit it precedes."""

    async def boom_cancel(client, oid):
        raise RuntimeError("cancel blew up")

    monkeypatch.setattr(fills_mod, "cancel_order", boom_cancel)
    register_orphan(farm_state, oid="stale-oid", token=TOK_YES, outcome="YES")

    await handle_trade(MagicMock(), orphan_fill(), farm_state, AsyncMock())

    assert "trade-orphan" in farm_state.pending_fok_exits
    assert farm_state.total_volume == Decimal("8")
    assert is_blacklisted(farm_state, GONE)


async def test_two_distinct_orphan_trades_each_recover(farm_state: FarmState, stub_io):
    """Two stale YES oids (the registry-leak scenario) filling in separate trades must
    each be staged and sold — per-trade.id staging keeps them independent."""
    register_orphan(farm_state, oid="stale-oid-1", token=TOK_YES, outcome="YES")
    register_orphan(farm_state, oid="stale-oid-2", token=TOK_YES, outcome="YES")

    await handle_trade(
        MagicMock(), orphan_fill(trade_id="orphan-A", oid="stale-oid-1"), farm_state, AsyncMock()
    )
    await handle_trade(
        MagicMock(), orphan_fill(trade_id="orphan-B", oid="stale-oid-2"), farm_state, AsyncMock()
    )
    assert set(farm_state.pending_fok_exits) == {"orphan-A", "orphan-B"}
    assert farm_state.total_volume == Decimal("16")  # 2 x (20 * 0.40)

    await handle_trade(MagicMock(), mined_frame("orphan-A"), farm_state, AsyncMock())
    await handle_trade(MagicMock(), mined_frame("orphan-B"), farm_state, AsyncMock())

    assert stub_io["sells"] == [(TOK_YES, "SELL", 20.0), (TOK_YES, "SELL", 20.0)]


# ── orphan exit recovery: partial fill & FAILED must re-drive, not strand ─────


def orphan_exit_resolution(*, status: str, sold: str, price: str = "0.35") -> UserTrade:
    """An exit-SELL resolution frame for the orphan's "exit-oid-1" (taker FAK)."""
    return UserTrade(
        event_type="trade",
        id=f"trade-orphan-exit-{status.lower()}",
        asset_id=TOK_YES,
        market=GONE,
        side="SELL",
        price=Decimal(price),
        size=Decimal(sold),
        outcome="YES",
        status=status,
        timestamp="2026-06-10T08:00:05Z",
        maker_orders=[],
        taker_order_id="exit-oid-1",
    )


async def test_orphan_partial_exit_fill_redrives_residual(farm_state: FarmState, stub_io):
    """#1 — an orphan exit SELL that PARTIALLY fills must re-drive the unsold remainder.
    With no position object the per-tick sweep can't recover it (it iterates state.positions),
    so handle_exit_fill must cancel the partly-filled order and place a fresh SELL for the
    residual — otherwise the shares strand forever (the DUST-strand bug class)."""
    register_orphan(farm_state, oid="stale-oid", token=TOK_YES, outcome="YES")
    await handle_trade(MagicMock(), orphan_fill(), farm_state, AsyncMock())
    await handle_trade(MagicMock(), mined_frame(), farm_state, AsyncMock())
    # MINED placed the orphan FAK SELL as "exit-oid-1" for the full 20 shares.
    assert stub_io["sells"] == [(TOK_YES, "SELL", 20.0)]
    assert "exit-oid-1" in farm_state.pending_exit_order_ids

    # Only 8 of 20 fill — 12 remain unsold.
    partial = orphan_exit_resolution(status="MINED", sold="8")
    await handle_trade(MagicMock(), partial, farm_state, AsyncMock())

    # The partly-filled order is cancelled and a fresh SELL re-drives the 12-share residual.
    assert "exit-oid-1" in stub_io["cancels"]
    assert stub_io["sells"] == [(TOK_YES, "SELL", 20.0), (TOK_YES, "SELL", 12.0)]
    # Residual now tracked under the new oid; the partly-filled oid is retired.
    assert "exit-oid-2" in farm_state.pending_exit_order_ids
    assert "exit-oid-1" not in farm_state.pending_exit_order_ids


async def test_orphan_full_exit_fill_does_not_redrive(farm_state: FarmState, stub_io):
    """Guard against over-recovery: a FULLY-filled orphan exit (sold == size) must NOT place
    a second SELL — there's no residual."""
    register_orphan(farm_state, oid="stale-oid", token=TOK_YES, outcome="YES")
    await handle_trade(MagicMock(), orphan_fill(), farm_state, AsyncMock())
    await handle_trade(MagicMock(), mined_frame(), farm_state, AsyncMock())

    full = orphan_exit_resolution(status="MINED", sold="20")
    await handle_trade(MagicMock(), full, farm_state, AsyncMock())

    assert stub_io["sells"] == [(TOK_YES, "SELL", 20.0)], "no residual -> no second SELL"
    assert farm_state.pending_exit_order_ids == set()


async def test_exit_fill_unknown_cost_books_breakeven_not_phantom_gain(farm_state: FarmState):
    """Regression for the Padres +$7.60 phantom: an exit registered with entry_cost=0 (e.g. a
    reconcile/orphan exit re-driven off a stale Data-API avg_price of 0) must NOT count the full
    proceeds as a gain. With no recoverable cost the covered portion books break-even, so realized
    PnL is 0 — not a fictional profit that would also mask session_loss / delay the kill."""
    farm_state.exit_cost_basis["exit-z"] = ExitCostInfo(
        entry_cost=Decimal("0"), entry_size=Decimal("20"), slug="m-z"
    )
    before = farm_state.session_loss
    await handle_exit_fill(
        MagicMock(),
        farm_state,
        AsyncMock(),
        exit_oid="exit-z",
        sold_size=Decimal("20"),
        sell_price=Decimal("0.38"),
        market_id="cid-z",  # no position object -> orphan
        outcome="YES",
        token_id="tok-z",
    )
    # proceeds 7.60 - break-even cost 7.60 = 0 ; session_loss unchanged (NOT reduced by 7.60).
    assert farm_state.session_loss == before


async def test_orphan_exit_failed_redrives_directly(farm_state: FarmState, stub_io):
    """#4 — if an orphan exit SELL comes back FAILED, the shares are still held but there's no
    position for the sweep to re-drive off. handle_trade must re-place the SELL directly so the
    shares aren't abandoned (the old code cleared tracking and relied on a sweep that can't see
    an orphan)."""
    register_orphan(farm_state, oid="stale-oid", token=TOK_YES, outcome="YES")
    await handle_trade(MagicMock(), orphan_fill(), farm_state, AsyncMock())
    await handle_trade(MagicMock(), mined_frame(), farm_state, AsyncMock())
    assert stub_io["sells"] == [(TOK_YES, "SELL", 20.0)]
    assert "exit-oid-1" in farm_state.pending_exit_order_ids

    failed = orphan_exit_resolution(status="FAILED", sold="20")
    await handle_trade(MagicMock(), failed, farm_state, AsyncMock())

    # Re-driven: a fresh SELL for the full 20 shares, tracked under a new oid.
    assert stub_io["sells"] == [(TOK_YES, "SELL", 20.0), (TOK_YES, "SELL", 20.0)]
    assert "exit-oid-1" not in farm_state.pending_exit_order_ids
    assert "exit-oid-2" in farm_state.pending_exit_order_ids
