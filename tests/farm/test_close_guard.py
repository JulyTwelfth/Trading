"""POL-55 (part 2): reconcile_tick must NOT close a position that still holds
shares when its market drops out of the candidate set. Closing pops the position,
but a deferred exit can still settle a real on-chain SELL against it — and the
market reopening with shares=0 is what clamps the exit and blinds the kill switch.
A flat position is closed only after POSITION_CANDIDATE_MISS_TICKS consecutive misses
(the sampled discovery endpoint flickers, so one miss must not churn the position).
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.constants import POSITION_CANDIDATE_MISS_TICKS
from app.farm import worker as worker_mod
from app.farm.schemas import FarmState, Market
from app.farm.worker import reconcile_tick


def wire(monkeypatch, closed: list):
    async def fake_fetch_eligible_markets(http):
        return []  # market-A drops out of the candidate set

    async def fake_get_balance(addr):
        return Decimal("100")

    async def fake_fetch_midpoints(http, token_ids):
        return {}

    async def fake_open_order_ids(client):
        # Orders are still live on the book → the dead-position prune keeps the position,
        # isolating the candidate-miss close path under test.
        return {"yes-oid", "no-oid"}

    async def fake_close_position(client, state, websocket, cid, reason):
        closed.append(cid)
        state.positions.pop(cid, None)

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_fetch_eligible_markets)
    monkeypatch.setattr(worker_mod, "get_balance", fake_get_balance, raising=False)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_fetch_midpoints)
    monkeypatch.setattr(worker_mod, "get_open_order_ids", fake_open_order_ids, raising=False)
    monkeypatch.setattr(worker_mod, "passes_all", lambda m, f: True)
    monkeypatch.setattr(worker_mod, "close_position", fake_close_position)


async def test_does_not_close_position_holding_shares(
    farm_state: FarmState, market: Market, monkeypatch
):
    closed: list = []
    wire(monkeypatch, closed)
    # Fixture position on market-A holds 100 YES shares.
    assert farm_state.positions["market-A"].yes_shares > 0

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert "market-A" not in closed, "must not close a position that still holds shares"
    assert "market-A" in farm_state.positions, "position must remain tracked until flat"


async def test_closes_flat_position_after_miss_hysteresis(
    farm_state: FarmState, market: Market, monkeypatch
):
    closed: list = []
    wire(monkeypatch, closed)
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("0")
    pos.no_shares = Decimal("0")

    # First MISS-1 ticks defer (transient sample flicker tolerance); close on the last.
    for _ in range(POSITION_CANDIDATE_MISS_TICKS - 1):
        await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())
        assert closed == [], "must not close a flat position on transient candidacy loss"

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())
    assert closed == ["market-A"], "flat position closes after sustained absence"


async def test_reappearing_market_resets_miss_streak(
    farm_state: FarmState, market: Market, monkeypatch
):
    closed: list = []
    wire(monkeypatch, closed)
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("0")
    pos.no_shares = Decimal("0")

    # Misses accrue (one short of closing)...
    for _ in range(POSITION_CANDIDATE_MISS_TICKS - 1):
        await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())
    assert closed == []
    assert pos.candidate_miss_ticks == POSITION_CANDIDATE_MISS_TICKS - 1

    # ...then the market reappears as a candidate → streak resets, no close.
    async def fetch_with_market(http):
        return [market]

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fetch_with_market)
    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())
    assert farm_state.positions["market-A"].candidate_miss_ticks == 0
    assert closed == []


async def test_held_shares_reset_preserves_hysteresis_after_exit(
    farm_state: FarmState, market: Market, monkeypatch
):
    # A position that accrued misses while flat, then gets filled (holds shares) and stays
    # absent, must NOT carry a frozen counter into the post-exit flat ticks — otherwise it
    # closes on the first flat tick with no flicker tolerance. Holding shares resets it.
    closed: list = []
    wire(monkeypatch, closed)

    async def noop_exit(*args, **kwargs):
        return None

    monkeypatch.setattr(worker_mod, "exit_held_legs", noop_exit)
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("0")
    pos.no_shares = Decimal("0")

    # Accrue misses while flat (one short of the threshold).
    for _ in range(POSITION_CANDIDATE_MISS_TICKS - 1):
        await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())
    assert pos.candidate_miss_ticks == POSITION_CANDIDATE_MISS_TICKS - 1

    # Now it gets filled and holds shares across several absent ticks → counter resets to 0.
    pos.yes_shares = Decimal("100")
    for _ in range(3):
        await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())
        assert pos.candidate_miss_ticks == 0
    assert closed == []

    # Shares sold → flat again; it must get the FULL tolerance again, not close immediately.
    pos.yes_shares = Decimal("0")
    for _ in range(POSITION_CANDIDATE_MISS_TICKS - 1):
        await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())
        assert closed == [], "must not close early on a frozen post-exit counter"
    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())
    assert closed == ["market-A"]


async def test_single_tick_flap_survives(farm_state: FarmState, market: Market, monkeypatch):
    # POSITION_CANDIDATE_MISS_TICKS was raised 1→3 so a single-tick eligibility flap can't churn a
    # flat position: one miss, a candidate tick that resets the streak, then two more misses must
    # STILL leave it open (three CONSECUTIVE misses are required). At K=1 the very first miss below
    # would have closed it — this is the regression guard for the bump.
    closed: list = []
    wire(monkeypatch, closed)  # fetch_eligible_markets returns [] (absent)
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("0")
    pos.no_shares = Decimal("0")

    async def absent(http):
        return []

    async def present(http):
        return [market]

    # Miss 1 (absent): with K>=3 this alone must not close.
    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())
    assert closed == [], "a single miss must not close at K>=3 (would close at K=1)"
    assert pos.candidate_miss_ticks == 1

    # Candidate reappears for one tick → the miss streak resets.
    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", present)
    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())
    assert pos.candidate_miss_ticks == 0, "a candidate tick resets the streak"

    # Absent again for two more ticks → streak 1, then 2 — still below the K=3 threshold.
    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", absent)
    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())
    assert pos.candidate_miss_ticks == 1
    assert closed == []
    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())
    assert pos.candidate_miss_ticks == 2
    assert closed == [], "a flap-interrupted absence must not close within the tolerance window"
