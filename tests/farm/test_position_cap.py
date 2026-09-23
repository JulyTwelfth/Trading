"""Hard ceiling on concurrently-held markets (MAX_CONCURRENT_POSITIONS).

A loose config (low liquidity floor) once opened 122-137 markets at once, which overloads the
single CLOB HTTP/2 connection + asyncio loop: 200+ requote-place failures, WS-1011 keepalive
timeouts, breaker trips, disconnects. The liquidity floor had been *implicitly* capping the count;
this is the explicit safety net so no config can melt the runtime. reconcile_tick must stop opening
once the cap is reached, counting ALL held markets (pre-existing + opened this tick), not just
per-tick opens.

NOTE: the default (MAX_CONCURRENT_POSITIONS) is currently set effectively UNLIMITED for
stress-testing, so these tests exercise the cap MECHANISM via an explicit configured value; one
test asserts the default no longer binds.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.constants import MAX_CONCURRENT_POSITIONS
from app.farm import worker as worker_mod
from app.farm.schemas import FarmState, Market, MarketPosition
from app.farm.worker import reconcile_tick


def pos(market: Market, *, shares: Decimal = Decimal("0")) -> MarketPosition:
    return MarketPosition(
        market=market,
        yes_order_id=f"y-{market.condition_id}",
        no_order_id=f"n-{market.condition_id}",
        yes_price=Decimal("0.5"),
        no_price=Decimal("0.5"),
        yes_shares=shares,
        no_shares=Decimal("0"),
    )


def candidate(market: Market, i: int) -> Market:
    return market.model_copy(
        update={
            "condition_id": f"cid-{i:03d}",
            "slug": f"m-{i:03d}",
            "yes_token_id": f"y-{i:03d}",
            "no_token_id": f"n-{i:03d}",
        }
    )


def wire(monkeypatch, farm_state: FarmState, cands: list[Market]) -> list[str]:
    """Stub every I/O hop in reconcile_tick's open path; return the list opens land in."""
    opened: list[str] = []

    async def fetch_eligible_markets(http):
        return cands

    async def get_balance(addr):
        return Decimal("100000")

    async def annotate_live_metrics(http, markets, **kw):
        return {}

    async def fetch_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    async def get_open_order_ids(client):
        return set()

    async def exit_held_legs(*a, **k):
        return None

    async def fake_open(client, state, websocket, mkt, midpoints):
        opened.append(mkt.condition_id)
        state.positions[mkt.condition_id] = pos(mkt)

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fetch_eligible_markets)
    monkeypatch.setattr(worker_mod, "get_balance", get_balance, raising=False)
    monkeypatch.setattr(worker_mod, "annotate_live_metrics", annotate_live_metrics)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fetch_midpoints)
    monkeypatch.setattr(worker_mod, "get_open_order_ids", get_open_order_ids, raising=False)
    monkeypatch.setattr(worker_mod, "exit_held_legs", exit_held_legs)
    monkeypatch.setattr(worker_mod, "open_position", fake_open)
    monkeypatch.setattr(worker_mod, "passes_all", lambda m, f: True)
    monkeypatch.setattr(worker_mod, "two_leg_cost", lambda m, y, n, d: Decimal("0.01"))
    return opened


async def test_stops_opening_at_cap(farm_state: FarmState, market: Market, monkeypatch):
    # From empty: even with far more eligible markets than the cap, open exactly the cap.
    # Tested via an explicit configured cap (the default is effectively unlimited — see below).
    cap = 25
    farm_state.positions.clear()
    farm_state.config.max_concurrent_positions = cap
    cands = [candidate(market, i) for i in range(cap + 15)]
    opened = wire(monkeypatch, farm_state, cands)

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert len(opened) == cap, "must stop opening once the cap is reached"
    assert len(farm_state.positions) == cap


async def test_default_cap_is_effectively_unlimited(
    farm_state: FarmState, market: Market, monkeypatch
):
    # TESTING default: the cap is effectively unlimited, so with the default config no ceiling
    # binds for any realistic candidate universe — every eligible market opens.
    farm_state.positions.clear()
    n = 150
    assert MAX_CONCURRENT_POSITIONS > n, "default must dwarf any real candidate count"
    cands = [candidate(market, i) for i in range(n)]
    opened = wire(monkeypatch, farm_state, cands)

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert len(opened) == n, "the unlimited default must not cap"
    assert len(farm_state.positions) == n


async def test_cap_is_configurable(farm_state: FarmState, market: Market, monkeypatch):
    # The cap is a tunable config knob (default = MAX_CONCURRENT_POSITIONS) so it can be ratcheted
    # up with monitoring. A lower configured value is honored.
    farm_state.positions.clear()
    farm_state.config.max_concurrent_positions = 8
    cands = [candidate(market, i) for i in range(20)]
    opened = wire(monkeypatch, farm_state, cands)

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert len(opened) == 8, "the configured cap must be honored"
    assert len(farm_state.positions) == 8


async def test_cap_counts_already_held_positions(
    farm_state: FarmState, market: Market, monkeypatch
):
    # The cap is on TOTAL held, not per-tick opens. Pre-seed (cap - 5) held markets (holding
    # shares, so the prune + candidate-miss sweeps both defer them) → only 5 slots remain this
    # tick even though more fresh candidates pass the filters.
    cap = 25
    farm_state.positions.clear()
    farm_state.config.max_concurrent_positions = cap
    held = cap - 5
    for i in range(held):
        held_mkt = market.model_copy(
            update={"condition_id": f"held-{i:03d}", "slug": f"held-{i:03d}"}
        )
        farm_state.positions[held_mkt.condition_id] = pos(held_mkt, shares=Decimal("1"))

    cands = [candidate(market, i) for i in range(cap + 15)]
    opened = wire(monkeypatch, farm_state, cands)

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert len(opened) == 5, "only the remaining headroom may open"
    assert len(farm_state.positions) == cap
