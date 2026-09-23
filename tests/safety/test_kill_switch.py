from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.schemas import UserTrade
from app.farm import exits as exits_mod
from app.farm import fills as fills_mod
from app.farm import kill_switch as ks_mod
from app.farm.fills import handle_trade
from app.farm.kill_switch import (
    net_session_loss,
    session_reward,
    should_kill,
    trigger_kill,
)
from app.farm.schemas import FarmState, FokExitInfo, Market, MarketPosition


def make_market(cid: str, slug: str, yes_tok: str, no_tok: str) -> Market:
    return Market(
        condition_id=cid,
        slug=slug,
        question="?",
        yes_token_id=yes_tok,
        no_token_id=no_tok,
        rewards_max_spread_cents=Decimal("3"),
        rewards_min_size=Decimal("100"),
        rewards_rate_per_day=Decimal("1"),
        tick_size=Decimal("0.01"),
        min_order_size=Decimal("5"),
        end_date=datetime(2030, 1, 1),
        created_at=datetime(2025, 1, 1),
        volume_24h=Decimal("100"),
        liquidity=Decimal("100"),
        spread_cents=Decimal("1"),
        price_change_24h=Decimal("0"),
    )


def make_exit_trade(
    *,
    taker_oid: str = "exit-oid-1",
    size: str = "100",
    price: str = "0.45",
    outcome: str = "YES",
    market_id: str = "market-A",
    trade_id: str = "t-exit-1",
) -> UserTrade:
    return UserTrade(
        event_type="trade",
        id=trade_id,
        asset_id="tok-yes" if outcome == "YES" else "tok-no",
        market=market_id,
        side="SELL",
        price=Decimal(price),
        size=Decimal(size),
        outcome=outcome,
        status="MINED",
        timestamp="2026-05-24T10:00:00Z",
        maker_orders=[],
        taker_order_id=taker_oid,
    )


@pytest.fixture
def market_b() -> Market:
    return make_market("market-B", "m2", "tok-b-yes", "tok-b-no")


@pytest.fixture
def ks_stubs(monkeypatch):
    """Stub every outbound call trigger_kill makes. Returns call records."""
    order: list[str] = []
    cancel: list = []
    exits: list = []
    events: list = []

    async def fake_cancel_all(client):
        order.append("cancel_all")
        cancel.append(True)

    async def fake_exit(client, state, token_id, size, cid, slug, outcome, **kwargs):
        order.append(f"exit:{token_id}")
        exits.append((token_id, float(size), outcome))

    async def fake_send_event(ws, event):
        order.append("send_event")
        events.append(event)

    monkeypatch.setattr(ks_mod, "cancel_all", fake_cancel_all)
    monkeypatch.setattr(exits_mod, "exit_position_leg", fake_exit)
    monkeypatch.setattr(ks_mod, "send_event", fake_send_event)

    return {"order": order, "cancel": cancel, "exits": exits, "events": events}


@pytest.fixture
def kill_stubs(monkeypatch):
    """Like ks_stubs, but also records the cancel_resting and force_dump flags passed to each leg
    exit so tests can assert the kill path opts out of the per-market cancel and force-dumps (never
    rests a patient smart-exit floor that the post-kill-stopped reaper could never reap)."""
    exits: list = []
    cancel_resting_flags: list = []
    force_dump_flags: list = []
    cancel: list = []
    events: list = []

    async def fake_cancel_all(client):
        cancel.append(True)

    async def fake_exit(
        client,
        state,
        token_id,
        size,
        cid,
        slug,
        outcome,
        *,
        cancel_resting=True,
        entry_cost=Decimal(0),
        force_dump=False,
    ):
        exits.append((token_id, float(size), outcome))
        cancel_resting_flags.append(cancel_resting)
        force_dump_flags.append(force_dump)

    async def fake_send_event(ws, event):
        events.append(event)

    monkeypatch.setattr(ks_mod, "cancel_all", fake_cancel_all)
    monkeypatch.setattr(exits_mod, "exit_position_leg", fake_exit)
    monkeypatch.setattr(ks_mod, "send_event", fake_send_event)

    return {
        "exits": exits,
        "cancel_resting_flags": cancel_resting_flags,
        "force_dump_flags": force_dump_flags,
        "cancel": cancel,
        "events": events,
    }


@pytest.fixture
def net_stubs(monkeypatch):
    """Stub all I/O for integration tests; patches exit_position_leg at both ks_mod and fills_mod
    so tests can distinguish an exit from trigger_kill (ks_exit) vs the fills.py residual-retry
    path (fills_exit)."""
    cancel_all_calls: list = []
    ks_exit_calls: list = []
    fills_exit_calls: list = []
    sent_events: list = []

    async def fake_cancel_all(client):
        cancel_all_calls.append(True)

    async def fake_ks_exit(client, state, token_id, size, cid, slug, outcome, **kwargs):
        ks_exit_calls.append((token_id, float(size), outcome))

    async def fake_fills_exit(client, state, token_id, size, cid, slug, outcome, **kwargs):
        fills_exit_calls.append((token_id, float(size), outcome))

    async def fake_send_event(ws, event):
        sent_events.append(event)

    async def fake_cancel_order(client, oid):
        return None

    async def fake_cancel_orders(client, *oids):
        return None

    monkeypatch.setattr(ks_mod, "cancel_all", fake_cancel_all)
    monkeypatch.setattr(exits_mod, "exit_position_leg", fake_ks_exit)
    monkeypatch.setattr(ks_mod, "send_event", fake_send_event)
    monkeypatch.setattr(fills_mod, "exit_position_leg", fake_fills_exit)
    # M2: cancel_order / cancel_orders are thin shims that await client.cancel_*;
    # patch them directly so tests do not need an adapter-shaped MagicMock client.
    monkeypatch.setattr(fills_mod, "cancel_order", fake_cancel_order)
    monkeypatch.setattr(fills_mod, "cancel_orders", fake_cancel_orders)
    return {
        "cancel_all": cancel_all_calls,
        "ks_exit": ks_exit_calls,
        "fills_exit": fills_exit_calls,
        "events": sent_events,
    }


async def test_trigger_kill_sets_killed_flag(farm_state: FarmState, kill_stubs):
    await trigger_kill(MagicMock(), farm_state, AsyncMock())
    assert farm_state.killed is True


async def test_trigger_kill_skips_per_leg_cancel(farm_state: FarmState, kill_stubs):
    await trigger_kill(MagicMock(), farm_state, AsyncMock())
    assert kill_stubs["exits"], "expected at least one leg exit"
    assert all(flag is False for flag in kill_stubs["cancel_resting_flags"]), (
        "kill switch must pass cancel_resting=False to exit_position_leg"
    )


async def test_trigger_kill_force_dumps_each_leg(farm_state: FarmState, kill_stubs):
    await trigger_kill(MagicMock(), farm_state, AsyncMock())
    assert kill_stubs["force_dump_flags"], "expected at least one leg exit"
    assert all(flag is True for flag in kill_stubs["force_dump_flags"]), (
        "kill switch must pass force_dump=True to exit_position_leg"
    )


async def test_trigger_kill_cancel_all_fires_before_exits(farm_state: FarmState, ks_stubs):
    pos = farm_state.positions["market-A"]
    pos.no_shares = Decimal("50")
    pos.no_cost_basis = Decimal("20")

    await trigger_kill(MagicMock(), farm_state, AsyncMock())

    order = ks_stubs["order"]
    cancel_idx = order.index("cancel_all")
    exit_indices = [i for i, x in enumerate(order) if x.startswith("exit:")]
    assert exit_indices, "at least one exit must have been called"
    assert all(cancel_idx < i for i in exit_indices), (
        f"cancel_all (idx {cancel_idx}) must precede every exit; order={order}"
    )


async def test_trigger_kill_exits_yes_leg(farm_state: FarmState, ks_stubs):
    await trigger_kill(MagicMock(), farm_state, AsyncMock())
    assert ("tok-yes", 100.0, "YES") in ks_stubs["exits"]


async def test_trigger_kill_skips_zero_share_no_leg(farm_state: FarmState, ks_stubs):
    await trigger_kill(MagicMock(), farm_state, AsyncMock())
    assert all(o != "NO" for _, _, o in ks_stubs["exits"])


async def test_trigger_kill_exits_both_legs_when_both_held(farm_state: FarmState, ks_stubs):
    pos = farm_state.positions["market-A"]
    pos.no_shares = Decimal("50")
    pos.no_cost_basis = Decimal("20")

    await trigger_kill(MagicMock(), farm_state, AsyncMock())

    outcomes = {o for _, _, o in ks_stubs["exits"]}
    assert "YES" in outcomes and "NO" in outcomes


async def test_trigger_kill_no_positions_only_cancels_and_emits(farm_state: FarmState, ks_stubs):
    farm_state.positions.clear()
    await trigger_kill(MagicMock(), farm_state, AsyncMock())
    assert len(ks_stubs["cancel"]) == 1
    assert ks_stubs["exits"] == []
    assert len(ks_stubs["events"]) == 1


async def test_trigger_kill_emits_farm_killed_event(farm_state: FarmState, ks_stubs):
    from app.api.farm.messages import FarmKilledEvent

    farm_state.session_loss = Decimal("7.50")
    await trigger_kill(MagicMock(), farm_state, AsyncMock())

    assert len(ks_stubs["events"]) == 1
    evt = ks_stubs["events"][0]
    assert isinstance(evt, FarmKilledEvent)
    assert evt.reason == "max_session_loss"
    assert evt.session_loss == Decimal("7.50")


async def test_trigger_kill_farm_killed_event_fires_after_exits(farm_state: FarmState, ks_stubs):
    await trigger_kill(MagicMock(), farm_state, AsyncMock())
    order = ks_stubs["order"]
    assert order[-1] == "send_event", f"send_event must be last; order={order}"


async def test_trigger_kill_is_idempotent(farm_state: FarmState, ks_stubs):
    await trigger_kill(MagicMock(), farm_state, AsyncMock())
    await trigger_kill(MagicMock(), farm_state, AsyncMock())

    assert len(ks_stubs["cancel"]) == 1, "cancel_all must fire exactly once"
    assert len(ks_stubs["exits"]) == 1, "exit legs must fire exactly once"
    assert len(ks_stubs["events"]) == 1, "FarmKilledEvent must fire exactly once"


async def test_trigger_kill_cancel_all_failure_does_not_abort_exits(
    farm_state: FarmState, monkeypatch
):
    exits: list = []
    events: list = []

    async def exploding_cancel(client):
        raise RuntimeError("network blip during cancel")

    async def fake_exit(client, state, token_id, size, cid, slug, outcome, **kwargs):
        exits.append((token_id, float(size), outcome))

    async def fake_send_event(ws, event):
        events.append(event)

    monkeypatch.setattr(ks_mod, "cancel_all", exploding_cancel)
    monkeypatch.setattr(exits_mod, "exit_position_leg", fake_exit)
    monkeypatch.setattr(ks_mod, "send_event", fake_send_event)

    await trigger_kill(MagicMock(), farm_state, AsyncMock())

    assert ("tok-yes", 100.0, "YES") in exits, "YES leg must still exit despite cancel_all failure"
    assert len(events) == 1, "FarmKilledEvent must still emit"


async def test_trigger_kill_one_leg_failure_does_not_abort_other_legs(
    farm_state: FarmState, market_b: Market, monkeypatch
):
    farm_state.positions["market-B"] = MarketPosition(
        market=market_b,
        yes_order_id="yes-oid-b",
        no_order_id="no-oid-b",
        yes_price=Decimal("0.5"),
        no_price=Decimal("0.5"),
        yes_shares=Decimal("0"),
        yes_cost_basis=Decimal("0"),
        no_shares=Decimal("50"),
        no_cost_basis=Decimal("20"),
    )

    exits: list = []
    events: list = []

    async def selective_exit(client, state, token_id, *args, **kwargs):
        if token_id == "tok-yes":
            raise RuntimeError("simulated failure on market-A YES leg")
        exits.append(token_id)

    async def fake_cancel_all(client):
        pass

    async def fake_send_event(ws, event):
        events.append(event)

    monkeypatch.setattr(ks_mod, "cancel_all", fake_cancel_all)
    monkeypatch.setattr(exits_mod, "exit_position_leg", selective_exit)
    monkeypatch.setattr(ks_mod, "send_event", fake_send_event)

    await trigger_kill(MagicMock(), farm_state, AsyncMock())

    assert "tok-b-no" in exits, "market-B NO leg must exit even when market-A YES leg raises"
    assert len(events) == 1, "FarmKilledEvent must still emit"


async def test_trigger_kill_exits_all_legs_across_two_markets(
    farm_state: FarmState, market_b: Market, ks_stubs
):
    pos_a = farm_state.positions["market-A"]
    pos_a.no_shares = Decimal("30")
    pos_a.no_cost_basis = Decimal("12")

    farm_state.positions["market-B"] = MarketPosition(
        market=market_b,
        yes_order_id="yes-oid-b",
        no_order_id="no-oid-b",
        yes_price=Decimal("0.5"),
        no_price=Decimal("0.5"),
        yes_shares=Decimal("0"),
        yes_cost_basis=Decimal("0"),
        no_shares=Decimal("50"),
        no_cost_basis=Decimal("20"),
    )

    await trigger_kill(MagicMock(), farm_state, AsyncMock())

    exited_tokens = {t for t, _, _ in ks_stubs["exits"]}
    assert exited_tokens == {"tok-yes", "tok-no", "tok-b-no"}, (
        f"expected A.YES, A.NO, B.NO; got {exited_tokens}"
    )
    exited_yes_b = [t for t, _, _ in ks_stubs["exits"] if t == "tok-b-yes"]
    assert exited_yes_b == [], "B.YES is 0 shares — must not be exited"


async def test_trigger_kill_exits_use_shares_at_kill_time(farm_state: FarmState, ks_stubs):
    farm_state.positions["market-A"].yes_shares = Decimal("37")
    farm_state.positions["market-A"].yes_cost_basis = Decimal("18.5")

    await trigger_kill(MagicMock(), farm_state, AsyncMock())

    yes_exit = next(e for e in ks_stubs["exits"] if e[2] == "YES")
    assert yes_exit[1] == 37.0, f"exit size must match current yes_shares; got {yes_exit[1]}"


async def test_kill_not_tripped_below_threshold(farm_state: FarmState, net_stubs):
    farm_state.config.max_session_loss = Decimal("5.01")
    farm_state.pending_exit_order_ids.add("exit-oid-1")

    await handle_trade(
        MagicMock(), make_exit_trade(size="100", price="0.45"), farm_state, AsyncMock()
    )

    assert farm_state.killed is False
    assert net_stubs["cancel_all"] == []


async def test_kill_trips_at_exact_threshold_boundary(farm_state: FarmState, net_stubs):
    farm_state.config.max_session_loss = Decimal("12")
    farm_state.pending_exit_order_ids.add("exit-oid-1")

    await handle_trade(
        MagicMock(), make_exit_trade(size="60", price="0.30"), farm_state, AsyncMock()
    )

    assert farm_state.killed is True
    assert len(net_stubs["cancel_all"]) == 1


async def test_kill_does_not_fire_on_profitable_exit(farm_state: FarmState, net_stubs):
    farm_state.pending_exit_order_ids.add("exit-oid-1")

    await handle_trade(
        MagicMock(), make_exit_trade(size="100", price="0.60"), farm_state, AsyncMock()
    )

    assert farm_state.killed is False
    assert net_stubs["cancel_all"] == []


async def test_kill_fires_cancel_all_once(farm_state: FarmState, net_stubs):
    farm_state.config.max_session_loss = Decimal("1")
    farm_state.pending_exit_order_ids.add("exit-oid-1")

    await handle_trade(
        MagicMock(), make_exit_trade(size="60", price="0.30"), farm_state, AsyncMock()
    )

    assert len(net_stubs["cancel_all"]) == 1


async def test_first_kill_covers_residual_no_double_exit(farm_state: FarmState, net_stubs):
    """When kill fires for the first time from a partial exit fill, trigger_kill reads
    the already-updated pos.yes_shares (= residual) and exits it.  The fills.py retry
    path must NOT also fire (kill_just_fired=True)."""
    farm_state.config.max_session_loss = Decimal("1")
    farm_state.pending_exit_order_ids.add("exit-oid-1")

    await handle_trade(
        MagicMock(), make_exit_trade(size="60", price="0.30"), farm_state, AsyncMock()
    )

    assert farm_state.killed is True
    assert net_stubs["fills_exit"] == [], (
        "fills.py retry path must not fire when kill just triggered"
    )
    yes_exits = [(t, s, o) for t, s, o in net_stubs["ks_exit"] if o == "YES"]
    assert len(yes_exits) == 1, "trigger_kill must exit the residual exactly once"
    assert yes_exits[0][1] == 40.0, (
        f"trigger_kill must exit the 40-share residual; got {yes_exits[0][1]}"
    )


async def test_already_killed_partial_fill_residual_retried_via_fills(
    farm_state: FarmState, net_stubs
):
    """The was_killed fix: when kill was already active before this fill event, trigger_kill
    is a no-op but the fills.py residual-retry path MUST still fire for the leftover shares."""
    farm_state.killed = True
    farm_state.session_loss = Decimal("100")
    farm_state.config.max_session_loss = Decimal("5")

    farm_state.pending_exit_order_ids.add("exit-oid-1")

    await handle_trade(
        MagicMock(), make_exit_trade(size="50", price="0.30"), farm_state, AsyncMock()
    )

    assert net_stubs["ks_exit"] == [], "trigger_kill must not re-exit when already killed"

    assert len(net_stubs["fills_exit"]) == 1, (
        "fills.py residual retry must fire when kill was already active"
    )
    _, size, outcome = net_stubs["fills_exit"][0]
    assert outcome == "YES"
    assert size == 50.0, f"residual must be 50 shares; got {size}"


async def test_ws_replay_of_exit_fill_does_not_re_trigger_kill(farm_state: FarmState, net_stubs):
    """Dedup via processed_events: replaying the same (trade_id, status) after a
    WS reconnect must be a complete no-op — no double kill, no double cancel."""
    farm_state.config.max_session_loss = Decimal("1")
    farm_state.pending_exit_order_ids.add("exit-oid-1")
    trade = make_exit_trade(size="60", price="0.30")

    await handle_trade(MagicMock(), trade, farm_state, AsyncMock())
    await handle_trade(MagicMock(), trade, farm_state, AsyncMock())

    assert len(net_stubs["cancel_all"]) == 1, "cancel_all must fire only once"
    assert len(net_stubs["ks_exit"]) == 1, "exits must fire only once"
    assert len(net_stubs["events"]) == 1, "FarmKilledEvent must emit only once"


async def test_session_loss_accumulates_and_kills_on_third_fill(farm_state: FarmState, net_stubs):
    """Three sequential partial exit fills; kill must fire exactly on the third when
    the accumulated loss crosses the threshold."""
    farm_state.config.max_session_loss = Decimal("9")

    farm_state.pending_exit_order_ids.add("exit-oid-1")
    await handle_trade(
        MagicMock(),
        make_exit_trade(taker_oid="exit-oid-1", size="30", price="0.40", trade_id="t1"),
        farm_state,
        AsyncMock(),
    )
    assert farm_state.killed is False
    assert farm_state.session_loss == Decimal("3")

    farm_state.pending_exit_order_ids.add("exit-oid-2")
    await handle_trade(
        MagicMock(),
        make_exit_trade(taker_oid="exit-oid-2", size="30", price="0.40", trade_id="t2"),
        farm_state,
        AsyncMock(),
    )
    assert farm_state.killed is False
    assert farm_state.session_loss == Decimal("6")

    farm_state.pending_exit_order_ids.add("exit-oid-3")
    await handle_trade(
        MagicMock(),
        make_exit_trade(taker_oid="exit-oid-3", size="40", price="0.40", trade_id="t3"),
        farm_state,
        AsyncMock(),
    )
    assert farm_state.killed is True
    assert farm_state.session_loss == Decimal("10")
    assert len(net_stubs["cancel_all"]) == 1


async def test_in_flight_maker_fill_after_kill_still_fires_exit(farm_state: FarmState, monkeypatch):
    """A maker fill that was already matched on-chain before cancel_all runs arrives
    via a MINED event after kill is set.  The MINED handler checks is_paused, NOT
    state.killed — so exit_position_leg must still be called for those shares."""
    exits: list = []

    async def fake_exit(client, state, token_id, size, cid, slug, outcome, **kwargs):
        exits.append((token_id, float(size), outcome))

    async def fake_cancel_order(client, oid):
        pass

    monkeypatch.setattr(fills_mod, "exit_position_leg", fake_exit)
    monkeypatch.setattr(fills_mod, "cancel_order", fake_cancel_order)

    farm_state.killed = True
    farm_state.pending_fok_exits["trade-inflight"] = FokExitInfo(
        token_id="tok-yes", size=Decimal("30"), outcome="YES", slug="m1"
    )
    mined = UserTrade(
        event_type="trade",
        id="trade-inflight",
        asset_id="tok-yes",
        market="market-A",
        side="BUY",
        price=Decimal("0.5"),
        size=Decimal("30"),
        outcome="YES",
        status="MINED",
        timestamp="2026-05-24T10:00:00Z",
        maker_orders=[],
        taker_order_id="entry-oid",
    )

    await handle_trade(MagicMock(), mined, farm_state, AsyncMock())

    assert exits == [("tok-yes", 30.0, "YES")], (
        "in-flight maker fill must still exit after kill is set"
    )
    assert "trade-inflight" not in farm_state.pending_fok_exits


async def test_in_flight_maker_fill_after_kill_exits_even_when_market_paused(
    farm_state: FarmState, monkeypatch
):
    """Same scenario as above, but the market is also paused. The fill must STILL exit
    instantly: a pause stops opening, never exiting, and after a kill the per-tick sweep has
    stopped — so skipping a paused exit would strand the shares permanently."""
    from datetime import timedelta, timezone

    from app.farm.schemas import MarketHealth

    exits: list = []

    async def fake_exit(client, state, token_id, size, cid, slug, outcome, **kwargs):
        exits.append((token_id, float(size), outcome))

    async def fake_cancel_order(client, oid):
        pass

    monkeypatch.setattr(fills_mod, "exit_position_leg", fake_exit)
    monkeypatch.setattr(fills_mod, "cancel_order", fake_cancel_order)

    farm_state.killed = True
    farm_state.health["market-A"] = MarketHealth(
        paused_until=datetime.now(timezone.utc) + timedelta(minutes=5)
    )
    farm_state.pending_fok_exits["trade-paused"] = FokExitInfo(
        token_id="tok-yes", size=Decimal("20"), outcome="YES", slug="m1"
    )
    mined = UserTrade(
        event_type="trade",
        id="trade-paused",
        asset_id="tok-yes",
        market="market-A",
        side="BUY",
        price=Decimal("0.5"),
        size=Decimal("20"),
        outcome="YES",
        status="MINED",
        timestamp="2026-05-24T10:00:00Z",
        maker_orders=[],
        taker_order_id="entry-oid",
    )

    await handle_trade(MagicMock(), mined, farm_state, AsyncMock())

    assert exits == [("tok-yes", 20.0, "YES")], "paused market must still exit in-flight fill"
    assert "trade-paused" not in farm_state.pending_fok_exits


async def test_second_exit_fill_after_kill_does_not_double_cancel(farm_state: FarmState, net_stubs):
    """Two concurrent exit fills arrive; first trips the kill, second must see
    state.killed=True and call trigger_kill as a no-op — cancel_all fires once total."""
    farm_state.config.max_session_loss = Decimal("1")

    farm_state.pending_exit_order_ids.update({"exit-oid-1", "exit-oid-2"})

    await handle_trade(
        MagicMock(),
        make_exit_trade(taker_oid="exit-oid-1", size="60", price="0.30", trade_id="t1"),
        farm_state,
        AsyncMock(),
    )
    await handle_trade(
        MagicMock(),
        make_exit_trade(taker_oid="exit-oid-2", size="20", price="0.30", trade_id="t2"),
        farm_state,
        AsyncMock(),
    )

    assert len(net_stubs["cancel_all"]) == 1, (
        "cancel_all must fire exactly once regardless of how many exit fills arrive"
    )
    assert len(net_stubs["events"]) == 1, "FarmKilledEvent must emit exactly once"


async def test_kill_from_market_a_also_exits_market_b(
    farm_state: FarmState, market_b: Market, net_stubs
):
    """Kill triggered by an exit fill of market-A must cause trigger_kill to also
    flush market-B's position via ks_exit."""
    farm_state.config.max_session_loss = Decimal("1")

    farm_state.positions["market-B"] = MarketPosition(
        market=market_b,
        yes_order_id="yes-oid-b",
        no_order_id="no-oid-b",
        yes_price=Decimal("0.5"),
        no_price=Decimal("0.5"),
        yes_shares=Decimal("0"),
        yes_cost_basis=Decimal("0"),
        no_shares=Decimal("75"),
        no_cost_basis=Decimal("30"),
    )

    farm_state.pending_exit_order_ids.add("exit-oid-1")
    await handle_trade(
        MagicMock(),
        make_exit_trade(taker_oid="exit-oid-1", size="60", price="0.30", trade_id="t1"),
        farm_state,
        AsyncMock(),
    )

    exited_tokens = {t for t, _, _ in net_stubs["ks_exit"]}
    assert "tok-b-no" in exited_tokens, (
        "market-B NO leg must be exited when kill is triggered from a market-A fill"
    )


async def test_partial_fill_residual_retried_then_second_partial_residual_also_retried(
    farm_state: FarmState, net_stubs
):
    """When kill is already active, every subsequent partial fill must have its
    residual retried via fills_exit.  This verifies the was_killed fix holds
    across multiple consecutive partial fills."""
    farm_state.killed = True
    farm_state.session_loss = Decimal("100")

    farm_state.pending_exit_order_ids.add("exit-oid-1")
    await handle_trade(
        MagicMock(),
        make_exit_trade(taker_oid="exit-oid-1", size="30", price="0.30", trade_id="t1"),
        farm_state,
        AsyncMock(),
    )

    farm_state.pending_exit_order_ids.add("exit-oid-2")
    await handle_trade(
        MagicMock(),
        make_exit_trade(taker_oid="exit-oid-2", size="40", price="0.30", trade_id="t2"),
        farm_state,
        AsyncMock(),
    )

    assert len(net_stubs["fills_exit"]) == 2, (
        "fills.py residual retry must fire for each partial fill when kill is already active"
    )
    assert net_stubs["ks_exit"] == [], "trigger_kill must not re-exit (already killed)"
    sizes = sorted(s for _, s, _ in net_stubs["fills_exit"])
    assert sizes == [30.0, 70.0]


def test_session_reward_sums_earned_minus_baseline(farm_state: FarmState):
    farm_state.market_rewards = {"A": Decimal("5"), "B": Decimal("3")}
    farm_state.market_rewards_baseline = {"A": Decimal("2"), "B": Decimal("0")}
    assert session_reward(farm_state) == Decimal("6")


def test_session_reward_zero_when_none_earned(farm_state: FarmState):
    assert session_reward(farm_state) == Decimal("0")


def test_no_reward_behaves_like_gross_loss(farm_state: FarmState):
    farm_state.positions.clear()
    farm_state.config.max_session_loss = Decimal("5")
    farm_state.session_loss = Decimal("5")
    assert should_kill(farm_state) is True


def test_reward_offsets_loss_no_kill(farm_state: FarmState):
    farm_state.positions.clear()
    farm_state.config.max_session_loss = Decimal("10")
    farm_state.session_loss = Decimal("13")
    farm_state.market_rewards = {"x": Decimal("23")}
    farm_state.market_rewards_baseline = {"x": Decimal("0")}
    assert net_session_loss(farm_state) == Decimal("-10")
    assert should_kill(farm_state) is False


def test_partial_reward_still_kills_when_loss_outruns_it(farm_state: FarmState):
    farm_state.positions.clear()
    farm_state.config.max_session_loss = Decimal("10")
    farm_state.session_loss = Decimal("13")
    farm_state.market_rewards = {"x": Decimal("1")}
    farm_state.market_rewards_baseline = {"x": Decimal("0")}
    assert should_kill(farm_state) is True


def test_net_kill_boundary(farm_state: FarmState):
    farm_state.positions.clear()
    farm_state.config.max_session_loss = Decimal("10")
    farm_state.session_loss = Decimal("20")
    farm_state.market_rewards_baseline = {"x": Decimal("0")}
    farm_state.market_rewards = {"x": Decimal("10")}
    assert should_kill(farm_state) is True
    farm_state.market_rewards = {"x": Decimal("10.01")}
    assert should_kill(farm_state) is False


def test_session_reward_uses_baseline_not_absolute(farm_state: FarmState):
    farm_state.positions.clear()
    farm_state.config.max_session_loss = Decimal("5")
    farm_state.session_loss = Decimal("8")
    farm_state.market_rewards = {"x": Decimal("20")}
    farm_state.market_rewards_baseline = {"x": Decimal("18")}
    assert session_reward(farm_state) == Decimal("2")
    assert should_kill(farm_state) is True


async def test_reward_prevents_kill_on_lossy_exit(farm_state: FarmState, net_stubs):
    farm_state.config.max_session_loss = Decimal("10")
    farm_state.market_rewards = {"x": Decimal("30")}
    farm_state.market_rewards_baseline = {"x": Decimal("0")}
    farm_state.pending_exit_order_ids.add("exit-oid-1")

    await handle_trade(
        MagicMock(), make_exit_trade(size="60", price="0.30"), farm_state, AsyncMock()
    )

    assert farm_state.killed is False
    assert net_stubs["cancel_all"] == []


async def test_killed_event_reports_reward_and_net(farm_state: FarmState, ks_stubs):
    from app.api.farm.messages import FarmKilledEvent

    farm_state.positions.clear()
    farm_state.session_loss = Decimal("15")
    farm_state.market_rewards = {"x": Decimal("3")}
    farm_state.market_rewards_baseline = {"x": Decimal("0")}

    await trigger_kill(MagicMock(), farm_state, AsyncMock())

    evt = ks_stubs["events"][0]
    assert isinstance(evt, FarmKilledEvent)
    assert evt.session_reward == Decimal("3")
    assert evt.net_loss == Decimal("12")
