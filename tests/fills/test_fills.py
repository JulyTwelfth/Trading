from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.schemas import UserTrade, UserTradeMakerOrder
from app.farm import exits as exits_mod
from app.farm import fills as fills_mod
from app.farm.fills import handle_trade
from app.farm.schemas import FarmState, FokExitInfo, MarketHealth


@pytest.fixture
def stub_network(monkeypatch):
    """Stub every outbound call fills.py makes. Returns captured market_sell calls."""
    calls: list[tuple[str, str, float]] = []

    async def fake_place_market_order(client, token_id, side, size):
        calls.append((token_id, side, size))
        return f"exit-oid-{len(calls)}"

    async def fake_cancel_order(client, oid):
        return None

    async def fake_cancel_orders(client, *oids):
        return None

    monkeypatch.setattr(exits_mod, "place_market_order", fake_place_market_order)
    monkeypatch.setattr(fills_mod, "cancel_order", fake_cancel_order)
    monkeypatch.setattr(fills_mod, "cancel_orders", fake_cancel_orders)
    # exit_position_leg (exits.py) also calls cancel_orders on entry; patch there too.
    monkeypatch.setattr(exits_mod, "cancel_orders", fake_cancel_orders)
    return calls


def make_trade(**overrides) -> UserTrade:
    defaults = {
        "event_type": "trade",
        "id": "trade-1",
        "asset_id": "tok-yes",
        "market": "market-A",
        "side": "BUY",
        "price": Decimal("0.5"),
        "size": Decimal("100"),
        "outcome": "YES",
        "status": "MINED",
        "timestamp": "2026-05-17T14:00:00Z",
        "maker_orders": [],
        "taker_order_id": "entry-oid",
    }
    defaults.update(overrides)
    return UserTrade(**defaults)


async def test_entry_replay_after_mined_does_not_double_sell(
    farm_state: FarmState, stub_network: list
):
    # Real bug 1 scenario: WS reconnect replays entry MATCHED after MINED has
    # cleared pending_fok_exits[trade.id]. The pre-existing line-55 guard misses
    # this case; the top-of-handler dedup is what closes it.
    entry = make_trade(
        status="MATCHED",
        taker_order_id="taker-buy-1",
        maker_orders=[
            UserTradeMakerOrder(
                asset_id="tok-yes",
                order_id="yes-oid",
                matched_amount=Decimal("100"),
                outcome="YES",
                owner="0xowner",
                price=Decimal("0.5"),
            )
        ],
    )
    mined = make_trade(status="MINED")
    client, ws = MagicMock(), AsyncMock()

    await handle_trade(client, entry, farm_state, ws)
    await handle_trade(client, mined, farm_state, ws)
    await handle_trade(client, entry, farm_state, ws)  # WS reconnect replay
    await handle_trade(client, mined, farm_state, ws)  # WS reconnect replay

    assert len(stub_network) == 1


async def test_fill_on_paused_market_still_sells_instantly_end_to_end(
    farm_state: FarmState, stub_network: list
):
    # End-to-end tuyo regression: a maker entry fill lands MATCHED then MINED while the market
    # is paused. The MINED exit must STILL fire the instant market-sell, never deferred to the
    # ~60s reconcile sweep.
    farm_state.health["market-A"] = MarketHealth(
        paused_until=datetime.now(timezone.utc) + timedelta(minutes=5)
    )
    entry = make_trade(
        status="MATCHED",
        taker_order_id="taker-buy-1",
        maker_orders=[
            UserTradeMakerOrder(
                asset_id="tok-yes",
                order_id="yes-oid",
                matched_amount=Decimal("100"),
                outcome="YES",
                owner="0xowner",
                price=Decimal("0.5"),
            )
        ],
    )
    mined = make_trade(status="MINED")
    client, ws = MagicMock(), AsyncMock()

    await handle_trade(client, entry, farm_state, ws)
    await handle_trade(client, mined, farm_state, ws)

    assert len(stub_network) == 1, "fill on a paused market must still sell instantly on MINED"
    assert "trade-1" not in farm_state.pending_fok_exits


async def test_exit_matched_decrements_position_shares(farm_state: FarmState, stub_network: list):
    farm_state.pending_exit_order_ids.add("exit-oid-1")
    exit_match = make_trade(
        id="trade-2",
        side="SELL",
        price=Decimal("0.45"),
        status="MINED",
        taker_order_id="exit-oid-1",
    )
    client, ws = MagicMock(), AsyncMock()

    await handle_trade(client, exit_match, farm_state, ws)

    assert farm_state.positions["market-A"].yes_shares == Decimal(0)


async def test_zero_size_pending_exit_does_not_fire_market_sell(
    farm_state: FarmState, stub_network: list
):
    farm_state.pending_fok_exits["trade-1"] = FokExitInfo(
        token_id="tok-yes", size=Decimal(0), outcome="YES", slug="m1"
    )
    mined = make_trade()
    client, ws = MagicMock(), AsyncMock()

    await handle_trade(client, mined, farm_state, ws)

    assert stub_network == []


async def test_partial_sell_decrements_cost_basis_proportionally(
    farm_state: FarmState, stub_network: list
):
    pos = farm_state.positions["market-A"]
    assert pos.yes_shares == Decimal("100")
    assert pos.yes_cost_basis == Decimal("50")
    farm_state.pending_exit_order_ids.add("exit-oid-1")
    partial_exit = make_trade(
        id="trade-partial",
        side="SELL",
        price=Decimal("0.40"),
        size=Decimal("25"),
        status="MINED",
        taker_order_id="exit-oid-1",
    )

    await handle_trade(MagicMock(), partial_exit, farm_state, AsyncMock())

    assert pos.yes_shares == Decimal("75")
    assert pos.yes_cost_basis == Decimal("37.5")


async def test_partial_sell_triggers_residual_retry(farm_state: FarmState, stub_network: list):
    """FAK partial fill must auto-retry exit_position_leg for the residual."""
    farm_state.config.max_session_loss = Decimal("100")  # avoid tripping kill mid-test
    farm_state.pending_exit_order_ids.add("exit-oid-1")
    # pos: yes_shares=100. Partial-fill 60 → residual 40 must retry.
    partial = make_trade(
        id="trade-partial-retry",
        side="SELL",
        price=Decimal("0.40"),
        size=Decimal("60"),
        status="MINED",
        taker_order_id="exit-oid-1",
    )

    await handle_trade(MagicMock(), partial, farm_state, AsyncMock())

    assert len(stub_network) == 1, "exactly one retry expected for the residual"
    _token_id, side, size = stub_network[0]
    assert side == "SELL"
    assert size == 40.0
    assert farm_state.positions["market-A"].yes_shares == Decimal("40")


async def test_full_sell_does_not_trigger_residual_retry(farm_state: FarmState, stub_network: list):
    """Full SELL match (residual=0) must not fire a needless retry."""
    farm_state.config.max_session_loss = Decimal("100")
    farm_state.pending_exit_order_ids.add("exit-oid-1")
    full = make_trade(
        id="trade-full-no-retry",
        side="SELL",
        price=Decimal("0.40"),
        size=Decimal("100"),
        status="MINED",
        taker_order_id="exit-oid-1",
    )

    await handle_trade(MagicMock(), full, farm_state, AsyncMock())

    assert stub_network == [], "no retry expected when shares fully exited"
    assert farm_state.positions["market-A"].yes_shares == Decimal("0")


async def test_residual_retry_skipped_when_kill_switch_trips(farm_state: FarmState, monkeypatch):
    """Verify the if/elif structure: when kill trips, the elif retry must NOT fire.
    Stubs fills.exit_position_leg and kill_switch.exit_position_leg separately so
    we can distinguish which path called exit_position_leg — otherwise the test
    is tautological (both paths land on the same underlying function)."""
    from app.farm import kill_switch as ks_mod

    fills_retry_calls: list = []
    ks_flush_calls: list = []

    async def fake_retry(*args, **kwargs):
        fills_retry_calls.append(args)

    async def fake_flush(*args, **kwargs):
        ks_flush_calls.append(args)

    async def fake_cancel_all(_client):
        pass

    async def fake_send_event(_ws, event):
        pass

    async def fake_cancel_order(_client, _oid):
        pass

    monkeypatch.setattr(fills_mod, "exit_position_leg", fake_retry)
    monkeypatch.setattr(exits_mod, "exit_position_leg", fake_flush)
    monkeypatch.setattr(ks_mod, "cancel_all", fake_cancel_all)
    monkeypatch.setattr(ks_mod, "send_event", fake_send_event)
    monkeypatch.setattr(fills_mod, "cancel_order", fake_cancel_order)

    farm_state.config.max_session_loss = Decimal("1")  # very tight — will trip
    farm_state.pending_exit_order_ids.add("exit-oid-1")
    # 60-share partial @ $0.30 → entry_cost_released=$30, proceeds=$18 → loss=$12 > $1.
    partial = make_trade(
        id="trade-partial-kill",
        side="SELL",
        price=Decimal("0.30"),
        size=Decimal("60"),
        status="MINED",
        taker_order_id="exit-oid-1",
    )

    await handle_trade(MagicMock(), partial, farm_state, AsyncMock())

    assert farm_state.killed is True
    assert len(ks_flush_calls) == 1, "defensive flush must fire on kill"
    assert fills_retry_calls == [], "elif retry must NOT fire when kill trips"


async def test_exit_matched_with_clob_titlecase_outcome_decrements_yes_leg(
    farm_state: FarmState, stub_network: list, caplog
):
    """Overnight-meltdown regression: Polymarket's user-WS emits title-case 'Yes', but the
    bot books shares IN under upper-case 'YES'. Without boundary normalization the exit fill
    misroutes to the empty NO leg, yes_shares never decrements, and the per-tick sweep
    re-sells a phantom forever. The schema validator must uppercase 'Yes' so the YES leg
    is the one that gets decremented."""
    import logging as _logging

    farm_state.pending_exit_order_ids.add("exit-oid-1")
    exit_match = make_trade(
        id="trade-titlecase",
        side="SELL",
        price=Decimal("0.45"),
        status="MINED",
        taker_order_id="exit-oid-1",
        outcome="Yes",  # exactly what the CLOB sends on the wire
    )
    caplog.set_level(_logging.WARNING, logger="app.farm.fills")

    await handle_trade(MagicMock(), exit_match, farm_state, AsyncMock())

    assert farm_state.positions["market-A"].yes_shares == Decimal("0"), (
        "title-case 'Yes' exit fill must decrement the YES leg, not the empty NO leg"
    )
    assert not [r for r in caplog.records if "divergence" in r.getMessage()], (
        "no divergence warning expected — the fill must route to the correct (held) leg"
    )


async def test_exit_matched_with_clob_titlecase_no_outcome_decrements_no_leg(
    farm_state: FarmState, stub_network: list
):
    """Symmetric half of the root cause: a title-case 'No' exit fill must decrement the NO
    leg (guards against a future 'Yes'-only special-case that would re-break NO)."""
    pos = farm_state.positions["market-A"]
    pos.no_shares = Decimal("100")
    pos.no_cost_basis = Decimal("50")
    farm_state.pending_exit_order_ids.add("exit-oid-1")
    exit_match = make_trade(
        id="trade-no-titlecase",
        asset_id="tok-no",
        side="SELL",
        price=Decimal("0.50"),
        status="MINED",
        taker_order_id="exit-oid-1",
        outcome="No",  # title-case, as the CLOB sends
    )

    await handle_trade(MagicMock(), exit_match, farm_state, AsyncMock())

    assert pos.no_shares == Decimal("0"), "title-case 'No' fill must decrement the NO leg"
    assert pos.yes_shares == Decimal("100"), "the YES leg must be left untouched"


async def test_titlecase_exit_isolates_to_correct_leg_when_both_held(
    farm_state: FarmState, stub_network: list
):
    """With both legs held, a title-case 'Yes' exit must touch ONLY the YES leg — direct
    leg-isolation, not inferred from a divergence warning."""
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("30")
    pos.yes_cost_basis = Decimal("15")
    pos.no_shares = Decimal("40")
    pos.no_cost_basis = Decimal("20")
    farm_state.pending_exit_order_ids.add("exit-oid-1")
    exit_match = make_trade(
        id="trade-both-held",
        side="SELL",
        price=Decimal("0.45"),
        size=Decimal("30"),
        status="MINED",
        taker_order_id="exit-oid-1",
        outcome="Yes",
    )

    await handle_trade(MagicMock(), exit_match, farm_state, AsyncMock())

    assert pos.yes_shares == Decimal("0")
    assert pos.yes_cost_basis == Decimal("0")
    assert pos.no_shares == Decimal("40"), "NO leg shares must be untouched"
    assert pos.no_cost_basis == Decimal("20"), "NO leg cost basis must be untouched"


async def test_titlecase_maker_path_exit_fill_decrements_correct_leg(
    farm_state: FarmState, stub_network: list
):
    """A rested GTC fallback fills as a MAKER, so the exit oid + title-case outcome arrive
    in maker_orders, not on the taker fields. The validator on UserTradeMakerOrder.outcome
    must normalize that path too."""
    farm_state.pending_exit_order_ids.add("gtc-exit-oid")
    exit_match = make_trade(
        id="trade-maker-exit",
        side="SELL",
        price=Decimal("0.45"),
        status="MINED",
        taker_order_id="someone-elses-taker",  # not ours — only the maker side is ours
        maker_orders=[
            UserTradeMakerOrder(
                asset_id="tok-yes",
                order_id="gtc-exit-oid",
                matched_amount=Decimal("100"),
                outcome="Yes",  # title-case on the maker leg
                owner="0xowner",
                price=Decimal("0.45"),
            )
        ],
    )

    await handle_trade(MagicMock(), exit_match, farm_state, AsyncMock())

    assert farm_state.positions["market-A"].yes_shares == Decimal("0"), (
        "title-case 'Yes' on the maker-side exit fill must decrement the YES leg"
    )


async def test_sell_size_exceeding_held_shares_is_clamped(
    farm_state: FarmState, stub_network: list, caplog
):
    import logging as _logging

    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("20")
    pos.yes_cost_basis = Decimal("5.80")
    farm_state.pending_exit_order_ids.add("exit-oid-1")
    overlarge_exit = make_trade(
        id="trade-over",
        side="SELL",
        price=Decimal("0.30"),
        size=Decimal("30"),
        status="MINED",
        taker_order_id="exit-oid-1",
    )

    caplog.set_level(_logging.WARNING, logger="app.farm.fills")

    await handle_trade(MagicMock(), overlarge_exit, farm_state, AsyncMock())

    assert pos.yes_shares == Decimal("0"), "shares must not go negative"
    assert pos.yes_cost_basis == Decimal("0")
    drift_records = [r for r in caplog.records if "divergence" in r.getMessage()]
    assert drift_records, "size>held must emit a divergence warning so the drift is visible"
