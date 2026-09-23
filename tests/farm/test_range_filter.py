"""24H Range (whipsaw) filter: true high-low swing from prices-history, the metric that net
Change24h misses. fetch_price_ranges parses batch history -> max-min; passes_range_24h
gates on the bucket; reconcile drops whippy candidates as a final live gate."""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.farm import discovery as discovery_mod
from app.farm import worker as worker_mod
from app.farm.discovery import fetch_price_ranges
from app.farm.filters import passes_range_24h
from app.farm.schemas import FarmConfig, FarmFilters, FarmState, Market
from app.farm.worker import reconcile_tick


def market(**o) -> Market:
    base = dict(
        condition_id="c",
        slug="s",
        question="?",
        yes_token_id="y",
        no_token_id="n",
        rewards_max_spread_cents=Decimal("3"),
        rewards_min_size=Decimal("100"),
        rewards_rate_per_day=Decimal("5"),
        tick_size=Decimal("0.01"),
        min_order_size=Decimal("5"),
        end_date=datetime(2030, 1, 1, tzinfo=timezone.utc),
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        volume_24h=Decimal("100"),
        liquidity=Decimal("100"),
        spread_cents=Decimal("1"),
        price_change_24h=Decimal("0"),
    )
    base.update(o)
    return Market(**base)


def _filters(**o) -> FarmFilters:
    base = dict(
        vol_min=Decimal(0),
        vol_max=Decimal(100000),
        liq_min=Decimal(0),
        liq_max=Decimal(100000),
        spread_min=Decimal(0),
        spread_max=Decimal(100),
        reward_min=Decimal(0),
        time_remaining="all",
        created_date="all",
        change_24h="all",
    )
    base.update(o)
    return FarmFilters(**base)


# ── passes_range_24h ─────────────────────────────────────────────────────────


def test_range_off_passes_anything():
    assert passes_range_24h(Decimal("0.50"), "all") is True


def test_range_under_threshold_passes():
    assert passes_range_24h(Decimal("0.03"), "lt5") is True


def test_range_over_threshold_fails():
    assert passes_range_24h(Decimal("0.06"), "lt5") is False
    assert passes_range_24h(Decimal("0.12"), "lt10") is False


def test_range_boundary_is_exclusive():
    assert passes_range_24h(Decimal("0.05"), "lt5") is False  # < 0.05, not <=


def test_unknown_range_excluded_when_active():
    assert passes_range_24h(None, "lt5") is False
    assert passes_range_24h(None, "all") is True  # off → unknown is fine


# ── fetch_price_ranges ───────────────────────────────────────────────────────


async def test_fetch_price_ranges_computes_high_minus_low():
    # NET change here is 0.01 (0.50→0.51) but RANGE is 0.12 — the whole point.
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(
        return_value={
            "history": {
                "tokA": [{"t": 1, "p": "0.50"}, {"t": 2, "p": "0.62"}, {"t": 3, "p": "0.51"}],
                "tokB": [{"t": 1, "p": "0.20"}, {"t": 2, "p": "0.205"}],
            }
        }
    )
    http = MagicMock()
    http.post = AsyncMock(return_value=resp)

    out = await fetch_price_ranges(http, ["tokA", "tokB"])
    assert out["tokA"] == Decimal("0.12")
    assert out["tokB"] == Decimal("0.005")


async def test_fetch_price_ranges_batches_by_20(monkeypatch):
    monkeypatch.setattr(discovery_mod, "BATCH_PRICES_HISTORY_LIMIT", 2)
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"history": {}})
    http = MagicMock()
    http.post = AsyncMock(return_value=resp)
    await fetch_price_ranges(http, ["a", "b", "c"])  # 3 tokens, limit 2 → 2 calls
    assert http.post.await_count == 2


async def test_fetch_price_ranges_empty_skips_http():
    http = MagicMock()
    http.post = AsyncMock()
    assert await fetch_price_ranges(http, []) == {}
    assert http.post.await_count == 0


# ── reconcile integration ────────────────────────────────────────────────────


async def test_reconcile_range_gate_drops_whippy_candidate(monkeypatch, caplog):
    calm = market(condition_id="calm", slug="calm", yes_token_id="cy", no_token_id="cn")
    whippy = market(condition_id="whip", slug="whip", yes_token_id="wy", no_token_id="wn")
    config = FarmConfig(
        filters=_filters(range_24h="lt5"), bankroll=Decimal("1000"), max_session_loss=Decimal("50")
    )
    state = FarmState(config=config, wallet_address="0xabc")

    async def fake_markets(http):
        return [calm, whippy]

    async def fake_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    async def fake_ranges(http, token_ids):
        return {"cy": Decimal("0.02"), "wy": Decimal("0.12")}  # whippy exceeds lt5

    async def fake_balance(addr):
        return Decimal("1000")

    async def fake_exit(*a, **k):
        return None

    async def fake_place(client, order, post_only=False):
        return "oid"

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_price_ranges", fake_ranges)
    monkeypatch.setattr(worker_mod, "get_balance", fake_balance, raising=False)
    monkeypatch.setattr(worker_mod, "exit_held_legs", fake_exit)
    monkeypatch.setattr(worker_mod, "place_limit_order", fake_place)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "filter_funnel range_24h=1" in text  # whippy dropped
    assert "candidates=1" in text  # only calm survives


async def test_reconcile_range_off_skips_history_fetch(monkeypatch):
    m = market(condition_id="x", slug="x", yes_token_id="xy", no_token_id="xn")
    config = FarmConfig(
        filters=_filters(), bankroll=Decimal("1000"), max_session_loss=Decimal("50")
    )
    state = FarmState(config=config, wallet_address="0xabc")
    ranges_spy = AsyncMock(return_value={})

    async def fake_markets(http):
        return [m]

    async def fake_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    async def fake_balance(addr):
        return Decimal("1000")

    async def fake_exit(*a, **k):
        return None

    async def fake_place(client, order, post_only=False):
        return "oid"

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_price_ranges", ranges_spy)
    monkeypatch.setattr(worker_mod, "get_balance", fake_balance, raising=False)
    monkeypatch.setattr(worker_mod, "exit_held_legs", fake_exit)
    monkeypatch.setattr(worker_mod, "place_limit_order", fake_place)

    await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())
    assert ranges_spy.await_count == 0  # range filter off → no prices-history fetch
