"""passes_max_fill_loss + its first_failing_filter wiring, and reconcile_tick behaviour when
the max_fill_loss filter is active: markets that would lose more than the cap on an immediate
fill+sell are rejected, cheap-to-exit ones open, books are only fetched when the filter is on,
and a market whose live book is missing is bucketed exit_loss_unknown (excluded, not blind)."""

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.bot.schemas import BookLevel, OrderBook
from app.farm import worker as worker_mod
from app.farm.filters import first_failing_filter, passes_all, passes_max_fill_loss
from app.farm.schemas import FarmConfig, FarmFilters, FarmState, Market, MarketHealth
from app.farm.worker import reconcile_tick

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


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


# ── passes_max_fill_loss (unit) ──────────────────────────────────────────────


def test_none_threshold_passes_everything():
    assert passes_max_fill_loss(_market(exit_loss=Decimal("99")), None) is True


def test_within_cap_passes():
    assert passes_max_fill_loss(_market(exit_loss=Decimal("0.50")), Decimal("1")) is True


def test_at_cap_passes():
    assert passes_max_fill_loss(_market(exit_loss=Decimal("1")), Decimal("1")) is True


def test_over_cap_fails():
    assert passes_max_fill_loss(_market(exit_loss=Decimal("1.01")), Decimal("1")) is False


def test_unknown_exit_loss_fails_when_active():
    assert passes_max_fill_loss(_market(exit_loss=None), Decimal("1")) is False


def test_zero_cap_is_active_and_only_passes_zero():
    assert passes_max_fill_loss(_market(exit_loss=Decimal("0")), Decimal("0")) is True
    assert passes_max_fill_loss(_market(exit_loss=Decimal("0.01")), Decimal("0")) is False


# ── first_failing_filter wiring ──────────────────────────────────────────────


def test_first_failing_names_exit_loss():
    m = _market(exit_loss=Decimal("5"))
    assert first_failing_filter(m, _filters(max_fill_loss=Decimal("1")), NOW) == "exit_loss"


def test_first_failing_names_exit_loss_unknown():
    m = _market(exit_loss=None)
    assert first_failing_filter(m, _filters(max_fill_loss=Decimal("1")), NOW) == "exit_loss_unknown"


def test_exit_loss_filter_off_does_not_reject():
    assert first_failing_filter(_market(exit_loss=None), _filters(), NOW) is None


def test_zone_named_before_exit_loss():
    # Fails BOTH zone and exit_loss; zone is earlier in the chain so it wins.
    m = _market(zone_liquidity=Decimal("9999"), exit_loss=Decimal("9"))
    f = _filters(zone_liq_max=Decimal("100"), max_fill_loss=Decimal("1"))
    assert first_failing_filter(m, f, NOW) == "zone_liquidity"


def test_cheaper_filter_named_before_exit_loss():
    m = _market(volume_24h=Decimal("5"), exit_loss=Decimal("9"))
    f = _filters(vol_min=Decimal("10"), max_fill_loss=Decimal("1"))
    assert first_failing_filter(m, f, NOW) == "volume"


def test_passes_all_parity_with_first_failing():
    cases = [
        (_market(exit_loss=Decimal("0.5")), _filters(max_fill_loss=Decimal("1"))),
        (_market(exit_loss=Decimal("5")), _filters(max_fill_loss=Decimal("1"))),
        (_market(exit_loss=None), _filters(max_fill_loss=Decimal("1"))),
        (_market(exit_loss=None), _filters()),
    ]
    for m, f in cases:
        assert passes_all(m, f, NOW) == (first_failing_filter(m, f, NOW) is None)


# ── reconcile integration ────────────────────────────────────────────────────


def book(bids, asset_id) -> OrderBook:
    return OrderBook(
        market="m",
        asset_id=asset_id,
        timestamp=NOW,
        bids=[BookLevel(price=Decimal(p), size=Decimal(s)) for p, s in bids],
        asks=[],
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


async def test_reconcile_rejects_costly_exit_opens_cheap(monkeypatch, caplog):
    # safe-depth entry = 0.48, size 100. thin book @0.40 → loss $8 (>cap). deep @0.48 → loss 0.
    costly = _market(condition_id="bad", slug="bad", yes_token_id="by", no_token_id="bn")
    cheap = _market(condition_id="ok", slug="ok", yes_token_id="oy", no_token_id="on")
    config = FarmConfig(
        filters=_filters(max_fill_loss=Decimal("1")),
        bankroll=Decimal("1000"),
        max_session_loss=Decimal("50"),
    )
    state = FarmState(config=config, wallet_address="0xabc")

    async def fake_markets(http):
        return [costly, cheap]

    async def fake_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    async def fake_books(http, token_ids):
        thin = [("0.40", "10000")]
        deep = [("0.48", "10000")]
        books = {
            "by": book(thin, "by"),
            "bn": book(thin, "bn"),
            "oy": book(deep, "oy"),
            "on": book(deep, "on"),
        }
        return {t: books[t] for t in token_ids if t in books}

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", fake_books)
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "filter_funnel exit_loss=1" in text
    assert "candidates=1" in text
    assert "opened=1" in text


async def test_reconcile_buckets_exit_loss_unknown_when_book_missing(monkeypatch, caplog):
    m = _market(condition_id="x", slug="x", yes_token_id="xy", no_token_id="xn")
    config = FarmConfig(
        filters=_filters(max_fill_loss=Decimal("1")),
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
    assert "filter_funnel exit_loss_unknown=1" in text
    assert "candidates=0" in text


async def test_reconcile_fetches_books_when_only_fill_loss_on(monkeypatch):
    m = _market(condition_id="x", slug="x", yes_token_id="xy", no_token_id="xn")
    config = FarmConfig(
        filters=_filters(max_fill_loss=Decimal("1")),  # zone off, fill-loss on
        bankroll=Decimal("1000"),
        max_session_loss=Decimal("50"),
    )
    state = FarmState(config=config, wallet_address="0xabc")

    async def fake_markets(http):
        return [m]

    async def fake_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    deep = [("0.48", "10000")]
    books_spy = AsyncMock(return_value={"xy": book(deep, "xy"), "xn": book(deep, "xn")})
    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", books_spy)
    patch_common(monkeypatch)

    await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    assert books_spy.await_count >= 1  # fill-loss filter needs depth


async def test_reconcile_skips_book_fetch_when_filter_off(monkeypatch):
    m = _market(condition_id="x", slug="x", yes_token_id="xy", no_token_id="xn")
    config = FarmConfig(
        filters=_filters(),  # max_fill_loss=None → off
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

    await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    assert books_spy.await_count == 0  # no fill-loss/zone filter → no extra fetch


def cfg(depth="safe", **fextra):
    return FarmConfig(
        filters=_filters(max_fill_loss=Decimal("1"), **fextra),
        bankroll=Decimal("1000"),
        max_session_loss=Decimal("50"),
        quote_depth=depth,
    )


async def test_reconcile_depth_sensitivity_safe_opens_aggressive_rejects(monkeypatch, caplog):
    # Same market + book; quote_depth changes the entry price → changes exit_loss → changes
    # the decision. safe entry 0.48 (loss $0.50, opens) vs aggressive 0.49 (loss $1.50, reject).
    m = _market(condition_id="x", slug="x", yes_token_id="xy", no_token_id="xn")

    async def fake_markets(http):
        return [m]

    async def fake_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    async def fake_books(http, token_ids):
        b = [("0.475", "10000")]
        return {"xy": book(b, "xy"), "xn": book(b, "xn")}

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", fake_books)
    patch_common(monkeypatch)

    safe_state = FarmState(config=cfg("safe"), wallet_address="0xa")
    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), safe_state, AsyncMock())
    safe_text = "\n".join(r.getMessage() for r in caplog.records)
    assert "candidates=1" in safe_text and "opened=1" in safe_text

    caplog.clear()
    aggr_state = FarmState(config=cfg("aggressive"), wallet_address="0xa")
    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), aggr_state, AsyncMock())
    aggr_text = "\n".join(r.getMessage() for r in caplog.records)
    assert "filter_funnel exit_loss=1" in aggr_text and "candidates=0" in aggr_text


async def test_reconcile_exit_loss_boundary_at_cap_opens_just_over_rejects(monkeypatch, caplog):
    # safe entry 0.48, size 100. at-cap (bids 0.47 → loss exactly $1.00) opens; just-over
    # (bids 0.46 → loss $2.00) is rejected. Confirms the <= cap boundary end-to-end.
    at_cap = _market(condition_id="atcap", slug="atcap", yes_token_id="ay", no_token_id="an")
    over = _market(condition_id="over", slug="over", yes_token_id="oy", no_token_id="oxn")
    state = FarmState(config=cfg("safe"), wallet_address="0xabc")

    async def fake_markets(http):
        return [at_cap, over]

    async def fake_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    async def fake_books(http, token_ids):
        books = {
            "ay": book([("0.47", "10000")], "ay"),
            "an": book([("0.47", "10000")], "an"),
            "oy": book([("0.46", "10000")], "oy"),
            "oxn": book([("0.46", "10000")], "oxn"),
        }
        return {t: books[t] for t in token_ids if t in books}

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", fake_books)
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "filter_funnel exit_loss=1" in text  # the over-cap market only
    assert "candidates=1" in text  # the at-cap market
    assert "opened=1" in text


async def test_reconcile_recomputes_exit_loss_each_tick(monkeypatch, caplog):
    # Tick 1 thin book → rejected; tick 2 deep book → opens. exit_loss is recomputed live each
    # tick (the book that strands you can recover), not cached from a prior tick.
    m = _market(condition_id="x", slug="x", yes_token_id="xy", no_token_id="xn")
    state = FarmState(config=cfg("safe"), wallet_address="0xabc")
    book_state = {"bids": [("0.40", "10000")]}  # thin first

    async def fake_markets(http):
        return [m]

    async def fake_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    async def fake_books(http, token_ids):
        b = book_state["bids"]
        return {"xy": book(b, "xy"), "xn": book(b, "xn")}

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", fake_books)
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())
    assert "filter_funnel exit_loss=1" in "\n".join(r.getMessage() for r in caplog.records)

    caplog.clear()
    book_state["bids"] = [("0.48", "10000")]  # deepened → cheap to exit now
    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())
    text2 = "\n".join(r.getMessage() for r in caplog.records)
    assert "opened=1" in text2 and "exit_loss" not in text2


# ── paused-market handling + spread-driven re-entry ──
# The exit-loss guard itself does NOT pause (it cancels resting orders only); these cover the
# general case where a market is paused by the circuit breaker and how it re-enters.


async def test_reconcile_market_stays_out_while_paused(monkeypatch, caplog):
    # A paused market (e.g. circuit breaker) is held out (bucketed "paused") even though its
    # book is now deep enough to pass exit_loss.
    m = _market(condition_id="p", slug="p", yes_token_id="py", no_token_id="pn")
    state = FarmState(config=cfg("safe"), wallet_address="0xabc")
    state.health["p"] = MarketHealth(paused_until=datetime.now(timezone.utc) + timedelta(hours=1))

    async def fake_markets(http):
        return [m]

    async def fake_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    async def fake_books(http, token_ids):
        deep = [("0.48", "10000")]  # would pass exit_loss, but pause wins
        return {"py": book(deep, "py"), "pn": book(deep, "pn")}

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", fake_books)
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "filter_funnel paused=1" in text
    assert "candidates=0" in text


async def test_reconcile_reenters_after_pause_expires_when_spread_recovered(monkeypatch, caplog):
    # Pause expired AND the book recovered (tight/deep) → the market re-enters as a fresh
    # candidate and reopens. This is "go back in when the spread goes down".
    m = _market(condition_id="r", slug="r", yes_token_id="ry", no_token_id="rn")
    state = FarmState(config=cfg("safe"), wallet_address="0xabc")
    state.health["r"] = MarketHealth(paused_until=datetime.now(timezone.utc) - timedelta(seconds=1))

    async def fake_markets(http):
        return [m]

    async def fake_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    async def fake_books(http, token_ids):
        deep = [("0.48", "10000")]  # spread recovered → loss $0
        return {"ry": book(deep, "ry"), "rn": book(deep, "rn")}

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", fake_books)
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "candidates=1" in text
    assert "opened=1" in text


async def test_reconcile_no_reentry_after_pause_if_spread_still_wide(monkeypatch, caplog):
    # Pause expired but the book is still thin → exit_loss keeps the market out. No premature
    # re-entry while the exit cost is still too high.
    m = _market(condition_id="w", slug="w", yes_token_id="wy", no_token_id="wn")
    state = FarmState(config=cfg("safe"), wallet_address="0xabc")
    state.health["w"] = MarketHealth(paused_until=datetime.now(timezone.utc) - timedelta(seconds=1))

    async def fake_markets(http):
        return [m]

    async def fake_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    async def fake_books(http, token_ids):
        thin = [("0.40", "10000")]  # still wide → loss $8
        return {"wy": book(thin, "wy"), "wn": book(thin, "wn")}

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", fake_books)
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "filter_funnel exit_loss=1" in text
    assert "candidates=0" in text


async def test_reconcile_mixed_funnel_across_markets(monkeypatch, caplog):
    # Four markets exercise the chain together: one opens, one fails exit_loss, one fails the
    # (earlier) reward filter, one fails the (earlier) price filter. Funnel counts each once.
    ok = _market(condition_id="ok", slug="ok", yes_token_id="ky", no_token_id="kn")
    thin = _market(condition_id="thin", slug="thin", yes_token_id="ty", no_token_id="tn")
    lowrew = _market(
        condition_id="lr",
        slug="lr",
        yes_token_id="ly",
        no_token_id="ln",
        rewards_rate_per_day=Decimal("1"),  # < reward_min 5
    )
    extreme = _market(condition_id="ex", slug="ex", yes_token_id="ey", no_token_id="en")
    config = FarmConfig(
        filters=_filters(
            max_fill_loss=Decimal("1"),
            reward_min=Decimal("5"),
            price_min=Decimal("0.1"),
            price_max=Decimal("0.9"),
        ),
        bankroll=Decimal("1000"),
        max_session_loss=Decimal("50"),
        quote_depth="safe",
    )
    state = FarmState(config=config, wallet_address="0xabc")

    async def fake_markets(http):
        return [ok, thin, lowrew, extreme]

    async def fake_midpoints(http, token_ids):
        mids = {
            "ky": Decimal("0.5"),
            "kn": Decimal("0.5"),
            "ty": Decimal("0.5"),
            "tn": Decimal("0.5"),
            "ly": Decimal("0.5"),
            "ln": Decimal("0.5"),
            "ey": Decimal("0.97"),
            "en": Decimal("0.03"),  # extreme price
        }
        return {t: mids[t] for t in token_ids if t in mids}

    async def fake_books(http, token_ids):
        deep = [("0.48", "10000")]
        thinb = [("0.40", "10000")]
        books = {
            "ky": book(deep, "ky"),
            "kn": book(deep, "kn"),
            "ty": book(thinb, "ty"),
            "tn": book(thinb, "tn"),
            "ly": book(deep, "ly"),
            "ln": book(deep, "ln"),
            "ey": book(deep, "ey"),
            "en": book(deep, "en"),
        }
        return {t: books[t] for t in token_ids if t in books}

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", fake_books)
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "filter_funnel" in text
    assert "reward=1" in text
    assert "price=1" in text
    assert "exit_loss=1" in text
    assert "candidates=1" in text
    assert "opened=1" in text
