from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.farm import exits as exits_mod
from app.farm import kill_switch as ks_mod
from app.farm.exits import (
    exit_position_leg,
    reap_stale_exit_orders,
    register_exit,
    smart_exit_floor_price,
)
from app.farm.kill_switch import trigger_kill
from app.farm.schemas import LiveBook

NOW = datetime(2026, 6, 22, 12, 0, tzinfo=timezone.utc)


def set_books(state, yes_bids, yes_asks, no_bids, no_asks):
    state.live_books["tok-yes"] = LiveBook(
        bids={Decimal(str(p)): Decimal(str(s)) for p, s in yes_bids},
        asks={Decimal(str(p)): Decimal(str(s)) for p, s in yes_asks},
    )
    state.live_books["tok-no"] = LiveBook(
        bids={Decimal(str(p)): Decimal(str(s)) for p, s in no_bids},
        asks={Decimal(str(p)): Decimal(str(s)) for p, s in no_asks},
    )


def _floor(state, token="tok-yes", size="100", now=NOW):
    return smart_exit_floor_price(state, state.positions["market-A"], token, Decimal(size), now=now)


def test_fires_on_swept_yes_with_intact_no(farm_state):
    set_books(
        farm_state,
        yes_bids=[(0.04, 50)],
        yes_asks=[(0.30, 50)],
        no_bids=[(0.71, 100)],
        no_asks=[(0.73, 100)],
    )
    assert _floor(farm_state) == Decimal("0.26")


def test_fires_on_swept_no_leg(farm_state):
    pos = farm_state.positions["market-A"]
    pos.no_shares, pos.no_cost_basis = Decimal("100"), Decimal("70")
    set_books(
        farm_state,
        yes_bids=[(0.71, 100)],
        yes_asks=[(0.73, 100)],
        no_bids=[(0.05, 50)],
        no_asks=[(0.30, 50)],
    )
    assert _floor(farm_state, token="tok-no") == Decimal("0.26")


def test_noop_on_normal_healthy_market(farm_state):
    set_books(
        farm_state,
        yes_bids=[(0.58, 100)],
        yes_asks=[(0.62, 100)],
        no_bids=[(0.39, 100)],
        no_asks=[(0.41, 100)],
    )
    assert _floor(farm_state) is None


def test_noop_when_reference_leg_too_wide(farm_state):
    set_books(
        farm_state,
        yes_bids=[(0.04, 50)],
        yes_asks=[(0.30, 50)],
        no_bids=[(0.60, 100)],
        no_asks=[(0.90, 100)],
    )
    assert _floor(farm_state) is None


def test_noop_when_reference_leg_too_thin(farm_state):
    set_books(
        farm_state,
        yes_bids=[(0.04, 50)],
        yes_asks=[(0.30, 50)],
        no_bids=[(0.71, 5)],
        no_asks=[(0.73, 5)],
    )
    assert _floor(farm_state) is None


def test_noop_when_savings_immaterial(farm_state):
    set_books(
        farm_state,
        yes_bids=[(0.04, 50)],
        yes_asks=[(0.30, 50)],
        no_bids=[(0.71, 100)],
        no_asks=[(0.73, 100)],
    )
    assert _floor(farm_state, size="10") is None


def test_noop_when_game_imminent(farm_state):
    pos = farm_state.positions["market-A"]
    pos.market.game_start_time = NOW + timedelta(hours=2)
    set_books(
        farm_state,
        yes_bids=[(0.04, 50)],
        yes_asks=[(0.30, 50)],
        no_bids=[(0.71, 100)],
        no_asks=[(0.73, 100)],
    )
    assert _floor(farm_state) is None


def test_fires_when_game_far_away(farm_state):
    pos = farm_state.positions["market-A"]
    pos.market.game_start_time = NOW + timedelta(hours=48)
    set_books(
        farm_state,
        yes_bids=[(0.04, 50)],
        yes_asks=[(0.30, 50)],
        no_bids=[(0.71, 100)],
        no_asks=[(0.73, 100)],
    )
    assert _floor(farm_state) == Decimal("0.26")


def test_concession_and_threshold_scale_with_market(farm_state):
    pos = farm_state.positions["market-A"]
    pos.market.tick_size = Decimal("0.001")
    pos.market.rewards_max_spread_cents = Decimal("1")
    set_books(
        farm_state,
        yes_bids=[(0.04, 50)],
        yes_asks=[(0.30, 50)],
        no_bids=[(0.71, 100)],
        no_asks=[(0.73, 100)],
    )
    assert _floor(farm_state) == Decimal("0.275")


def test_tighter_market_rejects_a_reference_the_default_would_trust(farm_state):
    pos = farm_state.positions["market-A"]
    pos.market.rewards_max_spread_cents = Decimal("1")
    set_books(
        farm_state,
        yes_bids=[(0.04, 50)],
        yes_asks=[(0.30, 50)],
        no_bids=[(0.70, 100)],
        no_asks=[(0.74, 100)],
    )
    assert _floor(farm_state) is None


def test_noop_when_sports_event_active(farm_state):
    farm_state.positions["market-A"].market.sports_event_active = True
    set_books(
        farm_state,
        yes_bids=[(0.04, 50)],
        yes_asks=[(0.30, 50)],
        no_bids=[(0.71, 100)],
        no_asks=[(0.73, 100)],
    )
    assert _floor(farm_state) is None


def test_noop_when_no_book_for_other_leg(farm_state):
    farm_state.live_books["tok-yes"] = LiveBook(
        bids={Decimal("0.04"): Decimal("50")}, asks={Decimal("0.30"): Decimal("50")}
    )
    assert _floor(farm_state) is None


def test_genuine_resolution_both_legs_moved_does_not_fire(farm_state):
    """THE key distinction. The market really resolved toward NO: YES crashed to 0.04 AND NO rose
    to match (0.95). So fair(YES) = 1 - 0.95 = 0.05 and our 0.04 bid is the TRUE price, not a hole.
    The opposite leg is even tight/trusted here — yet the hole-check correctly keeps us OUT."""
    set_books(
        farm_state,
        yes_bids=[(0.04, 50)],
        yes_asks=[(0.07, 50)],
        no_bids=[(0.94, 100)],
        no_asks=[(0.96, 100)],
    )
    assert _floor(farm_state) is None


@pytest.mark.parametrize("fair_c", range(10, 91, 5))
def test_healthy_market_never_fires_across_price_levels(farm_state, fair_c):
    """A healthy two-sided market at ANY price level (best bid one tick below a tight, matching
    complement) must never fire — the best bid sits at/above the floor."""
    fair = Decimal(fair_c) / 100
    comp = Decimal(1) - fair
    t = Decimal("0.01")
    set_books(
        farm_state,
        yes_bids=[(fair - t, 100)],
        yes_asks=[(fair + t, 100)],
        no_bids=[(comp - t, 100)],
        no_asks=[(comp + t, 100)],
    )
    assert _floor(farm_state) is None


@pytest.mark.parametrize(
    "tick,band,size",
    [
        (Decimal("0.01"), Decimal("3"), Decimal("50")),
        (Decimal("0.001"), Decimal("1"), Decimal("50")),
        (Decimal("0.01"), Decimal("5"), Decimal("200")),
        (Decimal("0.001"), Decimal("2"), Decimal("500")),
    ],
)
def test_healthy_market_never_fires_across_market_shapes(farm_state, tick, band, size):
    """Same healthy market across very different tick sizes, reward bands, and position sizes —
    the dynamic thresholds must still leave it a no-op."""
    pos = farm_state.positions["market-A"]
    pos.market.tick_size = tick
    pos.market.rewards_max_spread_cents = band
    fair = comp = Decimal("0.50")
    set_books(
        farm_state,
        yes_bids=[(fair - tick, 2000)],
        yes_asks=[(fair + tick, 2000)],
        no_bids=[(comp - tick, 2000)],
        no_asks=[(comp + tick, 2000)],
    )
    assert _floor(farm_state, size=str(size)) is None


def test_boundary_fires_only_when_gap_real_and_material(farm_state):
    """Sweep YES best bid from healthy down to a crater (fair=0.60, floor=0.58). It stays a strict
    no-op until the gap below fair is large enough to lose > $3 on 100 shares, then fires at
    0.58."""
    for bid_c in range(59, 40, -1):
        bid = Decimal(bid_c) / 100
        set_books(
            farm_state,
            yes_bids=[(bid, 100)],
            yes_asks=[(0.62, 100)],
            no_bids=[(0.39, 100)],
            no_asks=[(0.41, 100)],
        )
        floor = _floor(farm_state, size="100")
        if bid >= Decimal("0.58"):
            assert floor is None, f"must NOT fire at healthy bid {bid}"
        else:
            assert floor == Decimal("0.58"), f"should fire at dumped bid {bid}"


@pytest.fixture
def stub_orders(monkeypatch):
    placed: list = []

    async def fake_market(client, token_id, side, amount):
        placed.append(("FAK", token_id, side, amount, None))
        return "fak-oid"

    async def fake_limit(client, order, post_only=False):
        placed.append(("GTC", order.token_id, order.side, Decimal(str(order.size)), order.price))
        return "floor-oid"

    async def fake_cancel_orders(client, *oids):
        return None

    async def fake_cancel_order(client, oid):
        placed.append(("CANCEL", oid))
        return None

    monkeypatch.setattr(exits_mod, "place_market_order", fake_market)
    monkeypatch.setattr(exits_mod, "place_limit_order", fake_limit)
    monkeypatch.setattr(exits_mod, "cancel_orders", fake_cancel_orders)
    monkeypatch.setattr(exits_mod, "cancel_order", fake_cancel_order)
    return placed


async def test_hollow_book_places_floor_limit_not_fak(farm_state, stub_orders):
    set_books(
        farm_state,
        yes_bids=[(0.04, 50)],
        yes_asks=[(0.30, 50)],
        no_bids=[(0.71, 100)],
        no_asks=[(0.73, 100)],
    )
    await exit_position_leg(
        MagicMock(),
        farm_state,
        "tok-yes",
        Decimal("100"),
        "market-A",
        "m1",
        "YES",
        entry_cost=Decimal("58"),
    )
    gtc = [p for p in stub_orders if p[0] == "GTC"]
    assert gtc and gtc[0][4] == 0.26, "should rest a LIMIT SELL at the 0.26 floor, not FAK-dump"
    assert not [p for p in stub_orders if p[0] == "FAK"], "must not FAK-dump into the hole"
    eo = next(iter(farm_state.positions["market-A"].exit_orders.values()))
    assert eo.is_floor is True


async def test_force_dump_bypasses_floor_even_on_hollow_book(farm_state, stub_orders):
    set_books(
        farm_state,
        yes_bids=[(0.04, 50)],
        yes_asks=[(0.30, 50)],
        no_bids=[(0.71, 100)],
        no_asks=[(0.73, 100)],
    )
    await exit_position_leg(
        MagicMock(),
        farm_state,
        "tok-yes",
        Decimal("100"),
        "market-A",
        "m1",
        "YES",
        entry_cost=Decimal("58"),
        force_dump=True,
    )
    assert [p for p in stub_orders if p[0] == "FAK"], "force_dump must FAK-dump, skipping the floor"
    assert not [p for p in stub_orders if p[0] == "GTC"]


async def test_reaper_dumps_floor_after_max_hold(farm_state, stub_orders):
    pos = farm_state.positions["market-A"]
    register_exit(farm_state, "market-A", "floor-oid", Decimal("100"), "m1", Decimal("58"), "YES")
    pos.exit_orders["floor-oid"].is_floor = True
    pos.exit_orders["floor-oid"].placed_at = NOW - timedelta(seconds=100)
    await reap_stale_exit_orders(MagicMock(), farm_state, pos, NOW)
    assert ("CANCEL", "floor-oid") in stub_orders, "stale floor order must be cancelled"
    assert [p for p in stub_orders if p[0] == "FAK"], "give-up must dump the remainder (FAK)"


async def test_reaper_keeps_young_floor(farm_state, stub_orders):
    pos = farm_state.positions["market-A"]
    register_exit(farm_state, "market-A", "floor-oid", Decimal("100"), "m1", Decimal("58"), "YES")
    pos.exit_orders["floor-oid"].is_floor = True
    pos.exit_orders["floor-oid"].placed_at = NOW - timedelta(seconds=30)
    await reap_stale_exit_orders(MagicMock(), farm_state, pos, NOW)
    assert "floor-oid" in pos.exit_orders, "a young floor order must keep resting"
    assert not [p for p in stub_orders if p[0] in ("FAK", "CANCEL")]


async def test_kill_on_hollow_book_force_dumps_not_floor(farm_state, stub_orders, monkeypatch):
    """Regression (kill x floor): on a crashing/hollow book the kill must FLATTEN the held leg via
    FAK, NOT rest a patient floor LIMIT — after kill the reaper stops, so a resting floor never
    reaches its 60s give-up and orphans un-sold shares (violates "always exit, never hold")."""

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(ks_mod, "cancel_all", noop)
    monkeypatch.setattr(ks_mod, "send_event", noop)
    set_books(
        farm_state,
        yes_bids=[(0.04, 50)],
        yes_asks=[(0.30, 50)],
        no_bids=[(0.71, 100)],
        no_asks=[(0.73, 100)],
    )
    await trigger_kill(MagicMock(), farm_state, AsyncMock())
    assert [p for p in stub_orders if p[0] == "FAK"], "kill must FAK-dump the held leg"
    assert not [p for p in stub_orders if p[0] == "GTC"], "kill must not rest a patient floor LIMIT"
    assert not any(eo.is_floor for eo in farm_state.positions["market-A"].exit_orders.values()), (
        "kill must not leave a patient is_floor order resting"
    )
