"""passes_zone_liquidity + its first_failing_filter wiring, and reconcile_tick behaviour
when the zone filter is active: crowded markets are rejected, uncrowded ones open, the
extra book/midpoint fetches only happen when the filter is on, and a market whose live
book is missing is bucketed as zone_unknown (excluded, not quoted blind)."""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.bot.schemas import BookLevel, OrderBook
from app.constants import POSITION_CANDIDATE_MISS_TICKS
from app.farm import worker as worker_mod
from app.farm.filters import (
    first_failing_filter,
    passes_all,
    passes_price,
    passes_zone_liquidity,
)
from app.farm.schemas import FarmConfig, FarmFilters, FarmState, Market, MarketPosition
from app.farm.worker import reconcile_tick

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
LO, HI = Decimal("0.10"), Decimal("0.90")  # common price band for tests


def _market(**overrides) -> Market:
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
    base.update(overrides)
    return Market(**base)


def _filters(**overrides) -> FarmFilters:
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
    base.update(overrides)
    return FarmFilters(**base)


# ── passes_zone_liquidity (unit) ─────────────────────────────────────────────


def test_none_threshold_passes_everything():
    assert passes_zone_liquidity(_market(zone_liquidity=Decimal("99999")), None) is True


def test_within_threshold_passes():
    assert passes_zone_liquidity(_market(zone_liquidity=Decimal("500")), Decimal("1500")) is True


def test_at_threshold_passes():
    assert passes_zone_liquidity(_market(zone_liquidity=Decimal("1500")), Decimal("1500")) is True


def test_over_threshold_fails():
    assert passes_zone_liquidity(_market(zone_liquidity=Decimal("1501")), Decimal("1500")) is False


def test_unknown_zone_fails_when_filter_active():
    assert passes_zone_liquidity(_market(zone_liquidity=None), Decimal("1500")) is False


# ── first_failing_filter wiring ──────────────────────────────────────────────


def test_first_failing_names_zone_liquidity():
    m = _market(zone_liquidity=Decimal("2000"))
    assert first_failing_filter(m, _filters(zone_liq_max=Decimal("1500")), NOW) == "zone_liquidity"


def test_first_failing_names_zone_unknown():
    m = _market(zone_liquidity=None)
    assert first_failing_filter(m, _filters(zone_liq_max=Decimal("1500")), NOW) == "zone_unknown"


def test_zone_filter_off_does_not_reject():
    m = _market(zone_liquidity=None)
    assert first_failing_filter(m, _filters(), NOW) is None


def test_zero_threshold_is_active_and_only_passes_zero():
    # zone_liq_max=0 is not None → an active filter, not "off".
    assert passes_zone_liquidity(_market(zone_liquidity=Decimal("0")), Decimal("0")) is True
    assert passes_zone_liquidity(_market(zone_liquidity=Decimal("0.01")), Decimal("0")) is False
    f = _filters(zone_liq_max=Decimal("0"))
    assert first_failing_filter(_market(zone_liquidity=Decimal("5")), f, NOW) == "zone_liquidity"
    assert first_failing_filter(_market(zone_liquidity=Decimal("0")), f, NOW) is None


def test_cheaper_filter_named_before_zone():
    # Market fails BOTH volume and zone; volume is earlier in the chain so it wins.
    m = _market(volume_24h=Decimal("5"), zone_liquidity=Decimal("9999"))
    f = _filters(vol_min=Decimal("10"), zone_liq_max=Decimal("100"))
    assert first_failing_filter(m, f, NOW) == "volume"


def test_passes_all_parity_with_first_failing_for_zone():
    # passes_all must agree with (first_failing_filter is None) across zone scenarios.
    cases = [
        (_market(zone_liquidity=Decimal("100")), _filters(zone_liq_max=Decimal("1500"))),
        (_market(zone_liquidity=Decimal("2000")), _filters(zone_liq_max=Decimal("1500"))),
        (_market(zone_liquidity=None), _filters(zone_liq_max=Decimal("1500"))),
        (_market(zone_liquidity=None), _filters()),
    ]
    for m, f in cases:
        assert passes_all(m, f, NOW) == (first_failing_filter(m, f, NOW) is None)


# ── passes_price (unit) ──────────────────────────────────────────────────────


def test_price_both_none_is_off():
    assert passes_price(_market(midpoint=Decimal("0.02")), None, None) is True


def test_price_unknown_midpoint_fails_when_active():
    assert passes_price(_market(midpoint=None), LO, HI) is False


def test_price_within_band_passes():
    assert passes_price(_market(midpoint=Decimal("0.50")), LO, HI) is True


def test_price_below_min_fails():
    assert passes_price(_market(midpoint=Decimal("0.05")), LO, HI) is False


def test_price_above_max_fails():
    assert passes_price(_market(midpoint=Decimal("0.95")), LO, HI) is False


def test_price_boundaries_inclusive():
    assert passes_price(_market(midpoint=Decimal("0.10")), LO, HI) is True
    assert passes_price(_market(midpoint=Decimal("0.90")), LO, HI) is True


def test_price_only_min_bound():
    assert passes_price(_market(midpoint=Decimal("0.05")), LO, None) is False
    assert passes_price(_market(midpoint=Decimal("0.99")), LO, None) is True


def test_price_only_max_bound():
    assert passes_price(_market(midpoint=Decimal("0.95")), None, HI) is False
    assert passes_price(_market(midpoint=Decimal("0.01")), None, HI) is True


def test_first_failing_names_price_and_price_unknown():
    f = _filters(price_min=LO, price_max=HI)
    assert first_failing_filter(_market(midpoint=Decimal("0.05")), f, NOW) == "price"
    assert first_failing_filter(_market(midpoint=None), f, NOW) == "price_unknown"


def test_price_named_before_zone():
    # Fails both price and zone; price is earlier in the chain.
    m = _market(midpoint=Decimal("0.05"), zone_liquidity=Decimal("9999"))
    f = _filters(price_min=LO, zone_liq_max=Decimal("100"))
    assert first_failing_filter(m, f, NOW) == "price"


def test_passes_all_parity_for_price():
    cases = [
        (_market(midpoint=Decimal("0.5")), _filters(price_min=LO, price_max=HI)),
        (_market(midpoint=Decimal("0.05")), _filters(price_min=LO)),
        (_market(midpoint=None), _filters(price_max=HI)),
        (_market(midpoint=None), _filters()),
    ]
    for m, f in cases:
        assert passes_all(m, f, NOW) == (first_failing_filter(m, f, NOW) is None)


# ── reconcile integration ────────────────────────────────────────────────────


def book(bids, asks, asset_id) -> OrderBook:
    return OrderBook(
        market="m",
        asset_id=asset_id,
        timestamp=NOW,
        bids=[BookLevel(price=Decimal(p), size=Decimal(s)) for p, s in bids],
        asks=[BookLevel(price=Decimal(p), size=Decimal(s)) for p, s in asks],
        min_order_size=Decimal("1"),
        tick_size=Decimal("0.01"),
        neg_risk=False,
        hash="h",
    )


def patch_common(monkeypatch):
    async def fake_get_balance(addr):
        return Decimal("1000")

    async def fake_place_limit_order(client, order, post_only=False):
        return "oid"

    async def fake_exit_held_legs(*args, **kwargs):
        return None

    monkeypatch.setattr(worker_mod, "get_balance", fake_get_balance, raising=False)
    monkeypatch.setattr(worker_mod, "place_limit_order", fake_place_limit_order)
    monkeypatch.setattr(worker_mod, "exit_held_legs", fake_exit_held_legs)


async def test_reconcile_rejects_crowded_opens_uncrowded(monkeypatch, caplog):
    crowded = _market(condition_id="crowd", slug="crowd", yes_token_id="cy", no_token_id="cn")
    thin = _market(condition_id="thin", slug="thin", yes_token_id="ty", no_token_id="tn")
    config = FarmConfig(
        filters=_filters(zone_liq_max=Decimal("500")),
        bankroll=Decimal("1000"),
        max_session_loss=Decimal("50"),
    )
    state = FarmState(config=config, wallet_address="0xabc")

    async def fake_markets(http):
        return [crowded, thin]

    async def fake_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    async def fake_books(http, token_ids):
        # crowded legs hold $4900/leg in-zone; thin legs hold $49/leg.
        big = [("0.49", "10000")]
        small = [("0.49", "100")]
        books = {
            "cy": book(big, [], "cy"),
            "cn": book(big, [], "cn"),
            "ty": book(small, [], "ty"),
            "tn": book(small, [], "tn"),
        }
        return {t: books[t] for t in token_ids if t in books}

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", fake_books)
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "filter_funnel zone_liquidity=1" in text
    assert "candidates=1" in text
    assert "opened=1" in text


async def test_reconcile_buckets_zone_unknown_when_book_missing(monkeypatch, caplog):
    m = _market(condition_id="x", slug="x", yes_token_id="xy", no_token_id="xn")
    config = FarmConfig(
        filters=_filters(zone_liq_max=Decimal("500")),
        bankroll=Decimal("1000"),
        max_session_loss=Decimal("50"),
    )
    state = FarmState(config=config, wallet_address="0xabc")

    async def fake_markets(http):
        return [m]

    async def fake_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    async def fake_books(http, token_ids):
        return {}  # book unavailable this tick → zone_liquidity stays None

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", fake_books)
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "filter_funnel zone_unknown=1" in text
    assert "candidates=0" in text


async def test_reconcile_zone_unknown_when_one_leg_book_missing(monkeypatch, caplog):
    # YES book present but NO book missing → can't compute zone → excluded, not quoted blind.
    m = _market(condition_id="x", slug="x", yes_token_id="xy", no_token_id="xn")
    config = FarmConfig(
        filters=_filters(zone_liq_max=Decimal("500")),
        bankroll=Decimal("1000"),
        max_session_loss=Decimal("50"),
    )
    state = FarmState(config=config, wallet_address="0xabc")

    async def fake_markets(http):
        return [m]

    async def fake_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    async def fake_books(http, token_ids):
        return {"xy": book([("0.49", "100")], [], "xy")}  # only YES leg

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", fake_books)
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "filter_funnel zone_unknown=1" in text
    assert "candidates=0" in text


async def test_reconcile_zone_unknown_when_one_midpoint_missing(monkeypatch, caplog):
    # Both books present but a midpoint is missing → zone uncomputable → excluded.
    m = _market(condition_id="x", slug="x", yes_token_id="xy", no_token_id="xn")
    config = FarmConfig(
        filters=_filters(zone_liq_max=Decimal("500")),
        bankroll=Decimal("1000"),
        max_session_loss=Decimal("50"),
    )
    state = FarmState(config=config, wallet_address="0xabc")

    async def fake_markets(http):
        return [m]

    async def fake_midpoints(http, token_ids):
        return {"xy": Decimal("0.5")}  # NO-leg midpoint missing

    async def fake_books(http, token_ids):
        return {"xy": book([("0.49", "100")], [], "xy"), "xn": book([("0.49", "100")], [], "xn")}

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", fake_books)
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "filter_funnel zone_unknown=1" in text
    assert "candidates=0" in text


async def test_reconcile_reuses_universe_midpoints_single_fetch(monkeypatch):
    # When the zone filter is on, midpoints are fetched once for the universe and reused
    # for the affordability gate — not fetched a second time.
    thin = _market(condition_id="thin", slug="thin", yes_token_id="ty", no_token_id="tn")
    config = FarmConfig(
        filters=_filters(zone_liq_max=Decimal("5000")),
        bankroll=Decimal("1000"),
        max_session_loss=Decimal("50"),
    )
    state = FarmState(config=config, wallet_address="0xabc")

    async def fake_markets(http):
        return [thin]

    mid_spy = AsyncMock(return_value={"ty": Decimal("0.5"), "tn": Decimal("0.5")})

    async def fake_books(http, token_ids):
        small = [("0.49", "100")]
        return {"ty": book(small, [], "ty"), "tn": book(small, [], "tn")}

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", mid_spy)
    monkeypatch.setattr(worker_mod, "fetch_books", fake_books)
    patch_common(monkeypatch)

    await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    assert mid_spy.await_count == 1


async def test_reconcile_emits_zone_distribution_log(monkeypatch, caplog):
    crowded = _market(condition_id="c", slug="c", yes_token_id="cy", no_token_id="cn")
    thin = _market(condition_id="t", slug="t", yes_token_id="ty", no_token_id="tn")
    config = FarmConfig(
        filters=_filters(zone_liq_max=Decimal("5000")),
        bankroll=Decimal("1000"),
        max_session_loss=Decimal("50"),
    )
    state = FarmState(config=config, wallet_address="0xabc")

    async def fake_markets(http):
        return [crowded, thin]

    async def fake_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    async def fake_books(http, token_ids):
        big = [("0.49", "10000")]
        small = [("0.49", "100")]
        books = {
            "cy": book(big, [], "cy"),
            "cn": book(big, [], "cn"),
            "ty": book(small, [], "ty"),
            "tn": book(small, [], "tn"),
        }
        return {t: books[t] for t in token_ids if t in books}

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", fake_books)
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    # Both legs summed: thin = 0.49*100 * 2 = 98.00 ; crowded = 0.49*10000 * 2 = 9800.00.
    # Median of two values takes the upper one (typical=highest here).
    assert "zone_liquidity lowest=98.00 typical=9800.00 highest=9800.00" in text


async def test_reconcile_no_zone_log_when_filter_off(monkeypatch, caplog):
    m = _market(condition_id="x", slug="x", yes_token_id="xy", no_token_id="xn")
    config = FarmConfig(
        filters=_filters(),  # off
        bankroll=Decimal("1000"),
        max_session_loss=Decimal("50"),
    )
    state = FarmState(config=config, wallet_address="0xabc")

    async def fake_markets(http):
        return [m]

    async def fake_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", AsyncMock(return_value={}))
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "zone_liquidity lowest=" not in text


async def test_reconcile_no_zone_log_when_only_fill_loss_active(monkeypatch, caplog):
    # Books are fetched for the max-fill-loss filter (zone filter OFF), so zone_liquidity is
    # still computed on every market — but the zone-calibration log must NOT fire, or operators
    # would think zone_liq_max is in play. Regression for the exit-loss-only logging leak.
    m = _market(condition_id="x", slug="x", yes_token_id="xy", no_token_id="xn")
    config = FarmConfig(
        filters=_filters(max_fill_loss=Decimal("100")),  # exit_loss on, zone off
        bankroll=Decimal("1000"),
        max_session_loss=Decimal("50"),
    )
    state = FarmState(config=config, wallet_address="0xabc")

    async def fake_markets(http):
        return [m]

    async def fake_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    async def fake_books(http, token_ids):
        books = {"xy": book([("0.49", "100")], [], "xy"), "xn": book([("0.49", "100")], [], "xn")}
        return {t: books[t] for t in token_ids if t in books}

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", fake_books)
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "zone_liquidity lowest=" not in text


async def test_reconcile_closes_open_position_when_zone_stays_crowded(monkeypatch, caplog):
    # A held market (no shares) whose zone exceeds the cap drops out of candidacy. With the
    # sample-flicker hysteresis it isn't closed on the first miss — only after it stays out
    # for POSITION_CANDIDATE_MISS_TICKS consecutive ticks (here, persistently crowded).
    held = _market(condition_id="h", slug="h", yes_token_id="hy", no_token_id="hn")
    config = FarmConfig(
        filters=_filters(zone_liq_max=Decimal("500")),
        bankroll=Decimal("1000"),
        max_session_loss=Decimal("50"),
    )
    state = FarmState(config=config, wallet_address="0xabc")
    state.positions["h"] = MarketPosition(
        market=held,
        yes_order_id="hy-oid",
        no_order_id="hn-oid",
        yes_price=Decimal("0.49"),
        no_price=Decimal("0.49"),
    )

    async def fake_markets(http):
        return [held]

    async def fake_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    async def fake_books(http, token_ids):
        big = [("0.49", "10000")]  # 4900/leg → 9800 total, over the 500 cap
        return {"hy": book(big, [], "hy"), "hn": book(big, [], "hn")}

    async def fake_open_orders(client):
        return {"hy-oid", "hn-oid"}  # orders still live → survives the dead-position prune

    cancel_spy = AsyncMock()

    async def fake_cancel_with_retry(client, oid):
        # close_position now cancels via cancel_order_with_retry — route it through the same
        # spy so the existing await_count assertion still reflects both legs.
        await cancel_spy(client, oid)
        return True

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", fake_books)
    monkeypatch.setattr(worker_mod, "get_open_order_ids", fake_open_orders, raising=False)
    monkeypatch.setattr(worker_mod, "cancel_order_with_retry", fake_cancel_with_retry)
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        # First MISS-1 ticks defer; close only on the last.
        for _ in range(POSITION_CANDIDATE_MISS_TICKS - 1):
            await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())
            assert "h" in state.positions, "must not close on transient candidacy loss"
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "filter_funnel zone_liquidity=1" in text
    assert "h" not in state.positions  # position removed after sustained crowding
    assert cancel_spy.await_count == 2  # both legs cancelled once


async def test_reconcile_zone_passes_but_unaffordable_still_skips_cost(monkeypatch, caplog):
    # Zone filter and the affordability gate compose: a market can clear the zone cap yet
    # still be skipped as too costly for the bankroll (candidate, but opened=0).
    m = _market(condition_id="x", slug="x", yes_token_id="xy", no_token_id="xn")
    config = FarmConfig(
        filters=_filters(zone_liq_max=Decimal("5000")),
        bankroll=Decimal("50"),  # < two-leg cost (100 * (0.48 + 0.48) = 96)
        max_session_loss=Decimal("50"),
    )
    state = FarmState(config=config, wallet_address="0xabc")

    async def fake_markets(http):
        return [m]

    async def fake_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    async def fake_books(http, token_ids):
        small = [("0.49", "100")]  # zone ~98, well under the 5000 cap
        return {"xy": book(small, [], "xy"), "xn": book(small, [], "xn")}

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", fake_books)
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "candidates=1" in text
    assert "opened=0" in text
    assert "skipped_cost=1" in text


async def test_reconcile_emits_per_market_eval_debug(monkeypatch, caplog):
    # At DEBUG, every market gets a market_eval line with its decision + metrics.
    crowded = _market(condition_id="c", slug="c", yes_token_id="cy", no_token_id="cn")
    thin = _market(condition_id="t", slug="t", yes_token_id="ty", no_token_id="tn")
    config = FarmConfig(
        filters=_filters(zone_liq_max=Decimal("500")),
        bankroll=Decimal("1000"),
        max_session_loss=Decimal("50"),
    )
    state = FarmState(config=config, wallet_address="0xabc")

    async def fake_markets(http):
        return [crowded, thin]

    async def fake_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    async def fake_books(http, token_ids):
        big = [("0.49", "10000")]
        small = [("0.49", "100")]
        books = {
            "cy": book(big, [], "cy"),
            "cn": book(big, [], "cn"),
            "ty": book(small, [], "ty"),
            "tn": book(small, [], "tn"),
        }
        return {t: books[t] for t in token_ids if t in books}

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", fake_books)
    patch_common(monkeypatch)

    with caplog.at_level(logging.DEBUG, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "market_eval slug=c decision=zone_liquidity" in text
    assert "market_eval slug=t decision=candidate" in text
    assert "zone=9800.00" in text  # crowded market's computed zone in its eval line


async def test_reconcile_price_rejects_longshot_opens_mid_without_books(monkeypatch, caplog):
    # Price band on, zone off: longshot (mid 0.05) rejected, mid-priced market opens, and
    # NO /books fetch happens (price needs only midpoints).
    longshot = _market(condition_id="ls", slug="ls", yes_token_id="ly", no_token_id="ln")
    midpriced = _market(condition_id="mp", slug="mp", yes_token_id="my", no_token_id="mn")
    config = FarmConfig(
        filters=_filters(price_min=Decimal("0.10"), price_max=Decimal("0.90")),
        bankroll=Decimal("1000"),
        max_session_loss=Decimal("50"),
    )
    state = FarmState(config=config, wallet_address="0xabc")

    async def fake_markets(http):
        return [longshot, midpriced]

    async def fake_midpoints(http, token_ids):
        mids = {
            "ly": Decimal("0.05"),
            "ln": Decimal("0.95"),
            "my": Decimal("0.5"),
            "mn": Decimal("0.5"),
        }
        return {t: mids[t] for t in token_ids if t in mids}

    books_spy = AsyncMock(return_value={})

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", books_spy)
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "filter_funnel price=1" in text
    assert "candidates=1" in text
    assert "opened=1" in text
    assert books_spy.await_count == 0  # price filter must not fetch order books


async def test_reconcile_price_unknown_when_midpoint_missing(monkeypatch, caplog):
    m = _market(condition_id="x", slug="x", yes_token_id="xy", no_token_id="xn")
    config = FarmConfig(
        filters=_filters(price_min=Decimal("0.10"), price_max=Decimal("0.90")),
        bankroll=Decimal("1000"),
        max_session_loss=Decimal("50"),
    )
    state = FarmState(config=config, wallet_address="0xabc")

    async def fake_markets(http):
        return [m]

    async def fake_midpoints(http, token_ids):
        return {}  # no midpoints → price uncomputable

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", AsyncMock(return_value={}))
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "filter_funnel price_unknown=1" in text
    assert "candidates=0" in text


async def test_reconcile_price_unknown_when_only_one_leg_midpoint(monkeypatch, caplog):
    # Only the YES midpoint returns. midpoint must NOT be set from YES alone (the opener
    # needs both legs), so the market is price_unknown — excluded, not a phantom candidate.
    m = _market(condition_id="x", slug="x", yes_token_id="xy", no_token_id="xn")
    config = FarmConfig(
        filters=_filters(price_min=Decimal("0.10"), price_max=Decimal("0.90")),
        bankroll=Decimal("1000"),
        max_session_loss=Decimal("50"),
    )
    state = FarmState(config=config, wallet_address="0xabc")

    async def fake_markets(http):
        return [m]

    async def fake_midpoints(http, token_ids):
        return {"xy": Decimal("0.5")}  # NO leg (xn) missing

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", AsyncMock(return_value={}))
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "filter_funnel price_unknown=1" in text
    assert "candidates=0" in text


async def test_reconcile_price_and_zone_compose(monkeypatch, caplog):
    # Both filters on: a mid-priced market that clears price can still fail zone, and books
    # ARE fetched (zone needs them).
    m = _market(condition_id="x", slug="x", yes_token_id="xy", no_token_id="xn")
    config = FarmConfig(
        filters=_filters(
            price_min=Decimal("0.10"), price_max=Decimal("0.90"), zone_liq_max=Decimal("500")
        ),
        bankroll=Decimal("1000"),
        max_session_loss=Decimal("50"),
    )
    state = FarmState(config=config, wallet_address="0xabc")

    async def fake_markets(http):
        return [m]

    async def fake_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    big = [("0.49", "10000")]
    books_spy = AsyncMock(return_value={"xy": book(big, [], "xy"), "xn": book(big, [], "xn")})

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", books_spy)
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "filter_funnel zone_liquidity=1" in text  # cleared price (0.5), failed zone
    assert books_spy.await_count >= 1  # zone needs books


async def test_reconcile_skips_book_fetch_when_filter_off(monkeypatch, caplog):
    m = _market(condition_id="x", slug="x", yes_token_id="xy", no_token_id="xn")
    config = FarmConfig(
        filters=_filters(),  # zone_liq_max=None → filter off
        bankroll=Decimal("1000"),
        max_session_loss=Decimal("50"),
    )
    state = FarmState(config=config, wallet_address="0xabc")

    async def fake_markets(http):
        return [m]

    async def fake_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    books_spy = AsyncMock(return_value={})

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", books_spy)
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    assert books_spy.await_count == 0  # no zone filter → no extra fetch
