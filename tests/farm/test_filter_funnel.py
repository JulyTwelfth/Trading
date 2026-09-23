import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.farm import worker as worker_mod
from app.farm.filters import (
    first_failing_filter,
    passes_all,
    passes_bid_depth,
    passes_draft_filter,
    passes_election_window,
    passes_ipo_ma_filter,
    passes_wc_aggregate_filter,
)
from app.farm.health import mark_paused
from app.farm.schemas import FarmConfig, FarmFilters, FarmState, Market, MarketPosition
from app.farm.worker import deployed_capital, reconcile_tick

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


def filters(**overrides) -> FarmFilters:
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


def test_passing_market_has_no_failing_filter():
    assert first_failing_filter(_market(), filters(), NOW) is None
    assert passes_all(_market(), filters(), NOW) is True


def test_bid_depth_passes_when_both_legs_deep():
    m = _market(yes_bid_depth=Decimal("250"), no_bid_depth=Decimal("300"))
    assert first_failing_filter(m, filters(min_bid_depth_mult=Decimal("2")), NOW) is None


def test_bid_depth_rejects_thin_weak_leg():
    m = _market(yes_bid_depth=Decimal("9999"), no_bid_depth=Decimal("50"))
    f = filters(min_bid_depth_mult=Decimal("2"))
    assert first_failing_filter(m, f, NOW) == "bid_depth"
    assert passes_all(m, f, NOW) is False


def test_bid_depth_unknown_when_book_missing():
    m = _market(yes_bid_depth=None, no_bid_depth=None)
    f = filters(min_bid_depth_mult=Decimal("2"))
    assert first_failing_filter(m, f, NOW) == "bid_depth_unknown"


def test_bid_depth_off_by_default_even_with_empty_book():
    m = _market(yes_bid_depth=Decimal("0"), no_bid_depth=Decimal("0"))
    assert first_failing_filter(m, filters(), NOW) is None


def test_passes_bid_depth_unit():
    m = _market(yes_bid_depth=Decimal("250"), no_bid_depth=Decimal("150"))
    assert passes_bid_depth(m, None, Decimal("100")) is True
    assert passes_bid_depth(m, Decimal("2"), Decimal("100")) is False
    assert passes_bid_depth(m, Decimal("1"), Decimal("100")) is True
    blind = _market(yes_bid_depth=None, no_bid_depth=None)
    assert passes_bid_depth(blind, Decimal("1"), Decimal("100")) is False


def test_volume_rejection_named():
    m = _market(volume_24h=Decimal("5"))
    f = filters(vol_min=Decimal("10"))
    assert first_failing_filter(m, f, NOW) == "volume"
    assert passes_all(m, f, NOW) is False


def test_spread_rejection_named():
    m = _market(spread_cents=Decimal("50"))
    f = filters(spread_max=Decimal("10"))
    assert first_failing_filter(m, f, NOW) == "spread"


def test_reward_rejection_named():
    m = _market(rewards_rate_per_day=Decimal("0"))
    f = filters(reward_min=Decimal("1"))
    assert first_failing_filter(m, f, NOW) == "reward"


def test_liquidity_rejection_named():
    m = _market(liquidity=Decimal("5"))
    f = filters(liq_min=Decimal("10"))
    assert first_failing_filter(m, f, NOW) == "liquidity"


def test_time_remaining_rejection_named():
    m = _market(end_date=NOW + timedelta(hours=12))
    assert first_failing_filter(m, filters(time_remaining="1d"), NOW) == "time_remaining"


def test_time_remaining_12h_rejection_named():
    m = _market(end_date=NOW + timedelta(hours=6))
    assert first_failing_filter(m, filters(time_remaining="12h"), NOW) == "time_remaining"


def test_created_date_rejection_named():
    m = _market(created_at=NOW - timedelta(hours=12))
    assert first_failing_filter(m, filters(created_date="1d"), NOW) == "created_date"


def test_change_24h_rejection_named():
    m = _market(price_change_24h=Decimal("0.15"))
    assert first_failing_filter(m, filters(change_24h="lt10"), NOW) == "change_24h"


def test_wc_aggregate_stat_rejection_named():
    m = _market(
        slug="will-9-matches-go-to-extra-time-during-the-2026-fifa-world-cup-20260610",
        question="Will 9 matches go to extra time during the 2026 FIFA World Cup?",
    )
    assert first_failing_filter(m, filters(), NOW) == "wc_aggregate_stat"
    assert passes_all(m, filters(), NOW) is False


def test_wc_aggregate_stat_catches_all_families():
    questions = [
        "Will 3 matches be decided by penalty shootout during the 2026 FIFA World Cup?",
        "Will there be 5 missed penalties during the 2026 FIFA World Cup?",
        "Will 6 matches be suspended by weather protocol during the 2026 FIFA World Cup?",
        "Will there be 30 VAR stoppages during the 2026 FIFA World Cup?",
        "Will there be 150 total goals during the 2026 FIFA World Cup?",
    ]
    for q in questions:
        assert first_failing_filter(_market(question=q), filters(), NOW) == "wc_aggregate_stat", q


def test_wc_filters_pass_goalscorer_but_block_team_outcome():
    goalscorer = _market(
        question="Will Erling Haaland score 3 goals during the 2026 FIFA World Cup?"
    )
    assert first_failing_filter(goalscorer, filters(), NOW) is None
    team = _market(question="Will Paraguay reach the quarterfinals at the 2026 FIFA World Cup?")
    assert first_failing_filter(team, filters(), NOW) == "wc_team_outcome"


def test_wc_aggregate_requires_world_cup_context():
    m = _market(question="Will there be 5 missed penalties in the NHL playoffs?", slug="nhl-pens")
    assert passes_wc_aggregate_filter(m) is True


def test_wc_aggregate_catches_highest_scoring_team():
    m = _market(
        slug="will-brazil-be-the-highest-scoring-team-in-group-c-during-the-2026-fifa-world-cup",
        question="Will Brazil be the highest scoring team in Group C at the 2026 FIFA World Cup?",
    )
    assert first_failing_filter(m, filters(), NOW) == "wc_aggregate_stat"


def test_draft_rejection_named():
    m = _market(
        slug="will-aj-dybantsa-be-the-2nd-overall-pick-in-the-2026-nba-draft",
        question="Will AJ Dybantsa be the 2nd overall pick in the 2026 NBA draft?",
    )
    assert first_failing_filter(m, filters(), NOW) == "draft"
    assert passes_all(m, filters(), NOW) is False


def test_draft_catches_variants():
    questions = [
        "Will X be the 1st overall pick in the 2026 NFL draft?",
        "Will Y be a lottery pick in the 2026 NBA draft?",
        "Will Z be a first round pick in the 2026 WNBA draft?",
        "Will Q be the number one overall pick?",
        "Will R be drafted by the Yankees in the 2026 MLB draft?",
    ]
    for q in questions:
        assert first_failing_filter(_market(question=q), filters(), NOW) == "draft", q


def test_draft_no_false_positive_on_benign_pick():
    benign = [
        _market(question="Will the Lakers pick up a win on Tuesday?"),
        _market(question="Will Anthropic be the top model pick of analysts in June?"),
    ]
    for m in benign:
        assert passes_draft_filter(m) is True, m.question


def test_crisis_catches_new_geopolitical_entities():
    slugs = [
        "will-there-be-1-2-north-korea-tests-in-june-2026",
        "trump-meets-with-putin-by-september-30",
        "will-there-be-20-40-daily-transits-of-the-strait-of-hormuz-on-june-30",
        "us-announces-cuba-oil-sanction-relief-by-june-30",
        "will-delcy-rodriguez-be-the-de-facto-leader-of-venezuela-at-the-end-of-2026",
        "mojtaba-khamenei-seen-in-public-by-june-30",
    ]
    for s in slugs:
        m = _market(slug=s, question=s.replace("-", " "))
        assert first_failing_filter(m, filters(), NOW) == "crisis", s


def test_ipo_ma_rejection_named():
    m = _market(
        slug="will-snapchat-be-acquired-before-2027",
        question="Will Snapchat be acquired before 2027?",
    )
    assert first_failing_filter(m, filters(), NOW) == "ipo_ma_earnings"
    assert passes_ipo_ma_filter(m) is False


def test_ipo_ma_catches_variants():
    questions = [
        "Will OpenAI IPO before 2027?",
        "Will the Tesla and SpaceX merger be officially announced by September 30?",
        "Will SpaceX acquire Cursor by September 30 2026?",
        "Will Goldman Sachs Q2 investment banking fees be above 2.1b?",
        "Will Anthropic's market cap be between 1.5t and 1.75t at market close on IPO day?",
    ]
    for q in questions:
        assert first_failing_filter(_market(question=q), filters(), NOW) == "ipo_ma_earnings", q


def test_election_blocked_in_result_window():
    m = _market(
        slug="will-the-democrats-win-the-georgia-governor-race-in-2026",
        question="Will the Democrats win the Georgia governor race in 2026?",
        end_date=NOW + timedelta(hours=24),
    )
    assert first_failing_filter(m, filters(), NOW) == "election_window"


def test_election_farmed_off_window():
    m = _market(
        slug="will-the-democrats-win-the-georgia-governor-race-in-2026",
        question="Will the Democrats win the Georgia governor race in 2026?",
        end_date=NOW + timedelta(days=30),
    )
    assert first_failing_filter(m, filters(), NOW) is None
    assert passes_election_window(m, NOW) is True


def test_non_election_not_window_gated():
    m = _market(question="Will it rain in Seattle tomorrow?", end_date=NOW + timedelta(hours=12))
    assert passes_election_window(m, NOW) is True


def test_plural_primaries_triggers_election_window():
    # ELECTION_RE matched "primary" but not the plural "primaries", so markets like the AIPAC one
    # ("...lose their primaries") slipped the filter entirely. Regression guard for that gap.
    in_window = _market(
        slug="will-these-aipac-endorsees-lose-their-primaries",
        question="Will 0-1 of these AIPAC endorsees lose their primaries?",
        end_date=NOW + timedelta(hours=24),
    )
    assert passes_election_window(in_window, NOW) is False
    off_window = _market(
        slug="will-these-aipac-endorsees-lose-their-primaries",
        question="Will 0-1 of these AIPAC endorsees lose their primaries?",
        end_date=NOW + timedelta(days=30),
    )
    assert passes_election_window(off_window, NOW) is True


def test_ordering_live_event_takes_precedence():
    m = _market(game_start_time=NOW + timedelta(hours=2), volume_24h=Decimal("5"))
    f = filters(vol_min=Decimal("10"))
    assert first_failing_filter(m, f, NOW) == "live_event"


def test_parity_passes_all_equals_no_failing_filter():
    cases = [
        (_market(), filters()),
        (_market(volume_24h=Decimal("5")), filters(vol_min=Decimal("10"))),
        (_market(liquidity=Decimal("0")), filters(liq_min=Decimal("1"))),
    ]
    for m, f in cases:
        assert passes_all(m, f, NOW) == (first_failing_filter(m, f, NOW) is None)


async def test_reconcile_logs_funnel_and_tick_summary(monkeypatch, caplog):
    passing = _market(condition_id="pass", yes_token_id="pass-y", no_token_id="pass-n")
    rejected = _market(
        condition_id="rej", yes_token_id="rej-y", no_token_id="rej-n", volume_24h=Decimal("5")
    )
    config = FarmConfig(
        filters=filters(vol_min=Decimal("10")),
        bankroll=Decimal("1000"),
        max_session_loss=Decimal("50"),
    )
    state = FarmState(config=config, wallet_address="0xabc")

    async def fake_fetch_eligible_markets(http):
        return [passing, rejected]

    async def fake_fetch_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    async def fake_get_balance(addr):
        return Decimal("1000")

    async def fake_place_limit_order(client, order, post_only=False):
        return "oid"

    async def fake_exit_held_legs(*args, **kwargs):
        return None

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_fetch_eligible_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_fetch_midpoints)
    monkeypatch.setattr(worker_mod, "get_balance", fake_get_balance, raising=False)
    monkeypatch.setattr(worker_mod, "place_limit_order", fake_place_limit_order)
    monkeypatch.setattr(worker_mod, "exit_held_legs", fake_exit_held_legs)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "filter_funnel volume=1" in text
    assert "tick fetched=2 candidates=1" in text


async def test_reconcile_counts_unquoteable_separately_from_cost(monkeypatch, caplog):
    unquoteable = _market(
        condition_id="uq",
        yes_token_id="uq-y",
        no_token_id="uq-n",
        rewards_max_spread_cents=Decimal("1"),
        tick_size=Decimal("0.01"),
    )
    config = FarmConfig(filters=filters(), bankroll=Decimal("1000"), max_session_loss=Decimal("50"))
    state = FarmState(config=config, wallet_address="0xabc")

    async def fake_fetch_eligible_markets(http):
        return [unquoteable]

    async def fake_fetch_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    async def fake_get_balance(addr):
        return Decimal("1000")

    async def fake_place_limit_order(client, order, post_only=False):
        return "oid"

    async def fake_exit_held_legs(*args, **kwargs):
        return None

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_fetch_eligible_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_fetch_midpoints)
    monkeypatch.setattr(worker_mod, "get_balance", fake_get_balance, raising=False)
    monkeypatch.setattr(worker_mod, "place_limit_order", fake_place_limit_order)
    monkeypatch.setattr(worker_mod, "exit_held_legs", fake_exit_held_legs)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "skipped_cost=0 skipped_unquoteable=1" in text
    assert "opened=0" in text


def test_deployed_capital_empty_is_zero():
    config = FarmConfig(filters=filters(), bankroll=Decimal("1000"), max_session_loss=Decimal("50"))
    assert deployed_capital(FarmState(config=config)) == Decimal("0")


def test_deployed_capital_sums_resting_notional():
    config = FarmConfig(filters=filters(), bankroll=Decimal("1000"), max_session_loss=Decimal("50"))
    state = FarmState(config=config)
    state.positions["m"] = MarketPosition(
        market=_market(rewards_min_size=Decimal("100"), min_order_size=Decimal("5")),
        yes_order_id="y",
        no_order_id="n",
        yes_price=Decimal("0.4"),
        no_price=Decimal("0.6"),
    )
    assert deployed_capital(state) == Decimal("100")


async def test_reconcile_logs_summary_even_when_balance_fails(monkeypatch, caplog):
    m = _market(condition_id="x", yes_token_id="xy", no_token_id="xn")
    config = FarmConfig(filters=filters(), bankroll=Decimal("1000"), max_session_loss=Decimal("50"))
    state = FarmState(config=config, wallet_address="0xabc")

    async def fake_fetch_eligible_markets(http):
        return [m]

    async def boom_balance(addr):
        raise RuntimeError("rpc down")

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_fetch_eligible_markets)
    monkeypatch.setattr(worker_mod, "get_balance", boom_balance, raising=False)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "tick fetched=1 candidates=1" in text
    assert "opened=0 closed=0" in text


async def test_reconcile_logs_zeros_on_empty_markets(monkeypatch, caplog):
    config = FarmConfig(filters=filters(), bankroll=Decimal("1000"), max_session_loss=Decimal("50"))
    state = FarmState(config=config, wallet_address="0xabc")

    async def fake_empty(http):
        return []

    async def fake_balance(addr):
        return Decimal("1000")

    async def fake_exit_held_legs(*args, **kwargs):
        return None

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_empty)
    monkeypatch.setattr(worker_mod, "get_balance", fake_balance, raising=False)
    monkeypatch.setattr(worker_mod, "exit_held_legs", fake_exit_held_legs)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "tick fetched=0 candidates=0 positions=0 opened=0 closed=0" in text
    assert "skipped_cost=0 skipped_unquoteable=0" in text


async def test_reconcile_funnel_buckets_paused_and_excluded(monkeypatch, caplog):
    paused_m = _market(condition_id="p", yes_token_id="p-y", no_token_id="p-n")
    excluded_m = _market(condition_id="e", yes_token_id="e-y", no_token_id="e-n")
    config = FarmConfig(filters=filters(), bankroll=Decimal("1000"), max_session_loss=Decimal("50"))
    state = FarmState(config=config, wallet_address="0xabc")
    state.excluded_markets.add("e")
    mark_paused(state, "p")

    async def fake_fetch_eligible_markets(http):
        return [paused_m, excluded_m]

    async def fake_balance(addr):
        return Decimal("1000")

    async def fake_exit_held_legs(*args, **kwargs):
        return None

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_fetch_eligible_markets)
    monkeypatch.setattr(worker_mod, "get_balance", fake_balance, raising=False)
    monkeypatch.setattr(worker_mod, "exit_held_legs", fake_exit_held_legs)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "filter_funnel excluded=1 paused=1" in text
    assert "candidates=0" in text
