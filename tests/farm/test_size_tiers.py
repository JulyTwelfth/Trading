from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.farm import worker as worker_mod
from app.farm.filters import first_failing_filter
from app.farm.quoting import compute_quote, resolve_tier, size_per_market, tiers_log_repr
from app.farm.schemas import FarmState, Market, MarketHealth, SizeTier
from app.farm.worker import reconcile_tick

TIERS = [
    SizeTier(
        max_shares=Decimal(20),
        quote_depth="aggressive",
        reward_min=Decimal(5),
        liq_min=Decimal(500),
    ),
    SizeTier(
        max_shares=Decimal(50), quote_depth="normal", reward_min=Decimal(10), liq_min=Decimal(1000)
    ),
    SizeTier(max_shares=None, quote_depth="safe", reward_min=Decimal(25), liq_min=Decimal(2000)),
]


def test_resolve_tier_picks_by_size():
    assert resolve_tier(TIERS, Decimal(10), "safe", Decimal(0), Decimal(0), None, "all") == (
        "aggressive",
        Decimal(5),
        Decimal(500),
        None,
        "all",
    )
    assert resolve_tier(TIERS, Decimal(20), "safe", Decimal(0), Decimal(0), None, "all") == (
        "aggressive",
        Decimal(5),
        Decimal(500),
        None,
        "all",
    )
    assert resolve_tier(TIERS, Decimal(35), "safe", Decimal(0), Decimal(0), None, "all") == (
        "normal",
        Decimal(10),
        Decimal(1000),
        None,
        "all",
    )
    assert resolve_tier(TIERS, Decimal(50), "safe", Decimal(0), Decimal(0), None, "all") == (
        "normal",
        Decimal(10),
        Decimal(1000),
        None,
        "all",
    )
    assert resolve_tier(TIERS, Decimal(100), "safe", Decimal(0), Decimal(0), None, "all") == (
        "safe",
        Decimal(25),
        Decimal(2000),
        None,
        "all",
    )


def test_resolve_tier_order_independent():
    shuffled = [TIERS[2], TIERS[0], TIERS[1]]
    assert resolve_tier(shuffled, Decimal(35), "safe", Decimal(0), Decimal(0), None, "all") == (
        "normal",
        Decimal(10),
        Decimal(1000),
        None,
        "all",
    )


def test_resolve_tier_empty_uses_fallback():
    assert resolve_tier(
        [], Decimal(30), "normal", Decimal(7), Decimal(750), Decimal(3000), "7d"
    ) == (
        "normal",
        Decimal(7),
        Decimal(750),
        Decimal(3000),
        "7d",
    )


def test_resolve_tier_no_catchall_falls_back_for_oversized():
    capped = [
        SizeTier(
            max_shares=Decimal(20),
            quote_depth="aggressive",
            reward_min=Decimal(5),
            liq_min=Decimal(500),
        )
    ]
    assert resolve_tier(
        capped, Decimal(100), "safe", Decimal(3), Decimal(800), Decimal(900), "7d"
    ) == (
        "safe",
        Decimal(3),
        Decimal(800),
        Decimal(900),
        "7d",
    )
    assert resolve_tier(
        capped, Decimal(15), "safe", Decimal(3), Decimal(800), Decimal(900), "7d"
    ) == (
        "aggressive",
        Decimal(5),
        Decimal(500),
        None,
        "all",
    )


def test_resolve_tier_single_catchall_matches_all():
    tiers = [
        SizeTier(max_shares=None, quote_depth="safe", reward_min=Decimal(3), liq_min=Decimal(600))
    ]
    for sz in (Decimal(1), Decimal(50), Decimal(1000)):
        assert resolve_tier(tiers, sz, "normal", Decimal(99), Decimal(0), None, "all") == (
            "safe",
            Decimal(3),
            Decimal(600),
            None,
            "all",
        )


def test_resolve_tier_catchall_first_does_not_shadow_smaller():
    tiers = [
        SizeTier(
            max_shares=None, quote_depth="safe", reward_min=Decimal(25), liq_min=Decimal(2000)
        ),
        SizeTier(
            max_shares=Decimal(20),
            quote_depth="aggressive",
            reward_min=Decimal(5),
            liq_min=Decimal(500),
        ),
    ]
    assert resolve_tier(tiers, Decimal(10), "normal", Decimal(0), Decimal(0), None, "all") == (
        "aggressive",
        Decimal(5),
        Decimal(500),
        None,
        "all",
    )
    assert resolve_tier(tiers, Decimal(99), "normal", Decimal(0), Decimal(0), None, "all") == (
        "safe",
        Decimal(25),
        Decimal(2000),
        None,
        "all",
    )


def test_resolve_tier_returns_per_tier_zone():
    tiers = [
        SizeTier(
            max_shares=Decimal(20),
            quote_depth="aggressive",
            reward_min=Decimal(5),
            liq_min=Decimal(500),
            zone_liq_max=Decimal(1500),
        ),
    ]
    assert resolve_tier(
        tiers, Decimal(10), "safe", Decimal(0), Decimal(0), Decimal(9999), "all"
    ) == (
        "aggressive",
        Decimal(5),
        Decimal(500),
        Decimal(1500),
        "all",
    )
    assert resolve_tier(tiers, Decimal(99), "safe", Decimal(0), Decimal(0), None, "all")[3] is None


def test_resolve_tier_oversized_falls_back_to_global_zone():
    tiers = [
        SizeTier(max_shares=Decimal(20), quote_depth="normal", zone_liq_max=Decimal(1500)),
        SizeTier(max_shares=Decimal(50), quote_depth="normal", zone_liq_max=Decimal(2500)),
    ]
    _, _, _, zone, _ = resolve_tier(
        tiers, Decimal(100), "safe", Decimal(0), Decimal(0), Decimal(2500), "all"
    )
    assert zone == Decimal(2500), "oversized -> derived global fallback (NOT None)"


def test_resolve_tier_returns_per_tier_time_remaining():
    tiers = [
        SizeTier(
            max_shares=Decimal(20),
            quote_depth="aggressive",
            reward_min=Decimal(5),
            liq_min=Decimal(500),
            time_remaining="12h",
        ),
    ]
    assert (
        resolve_tier(tiers, Decimal(10), "safe", Decimal(0), Decimal(0), None, "all")[4] == "12h"
    )
    assert (
        resolve_tier(tiers, Decimal(99), "safe", Decimal(0), Decimal(0), None, "7d")[4] == "7d"
    ), "oversized-no-catchall falls back to the passed global"


def test_resolve_tier_picks_each_tiers_own_time_remaining():
    tiers = [
        SizeTier(max_shares=Decimal(20), quote_depth="aggressive", time_remaining="all"),
        SizeTier(max_shares=None, quote_depth="safe", time_remaining="12h"),
    ]
    # fallback "7d" is deliberately distinct from both tiers so a leak would be visible.
    small = resolve_tier(tiers, Decimal(10), "safe", Decimal(0), Decimal(0), None, "7d")
    large = resolve_tier(tiers, Decimal(100), "safe", Decimal(0), Decimal(0), None, "7d")
    assert small[4] == "all", "small tier keeps its own 'all' window"
    assert large[4] == "12h", "large/catch-all tier applies its own '12h' window"


def test_tiers_log_repr_includes_non_default_time_remaining():
    tiers = [
        SizeTier(max_shares=Decimal(20), quote_depth="aggressive", time_remaining="12h"),
        SizeTier(max_shares=None, quote_depth="safe", time_remaining="7d"),
    ]
    assert tiers_log_repr(tiers) == "20:aggressive:0:0:off:12h|inf:safe:0:0:off:7d"


def test_tiers_log_repr_includes_every_per_tier_knob():
    tiers = [
        SizeTier(
            max_shares=Decimal(20),
            quote_depth="normal",
            reward_min=Decimal(5),
            liq_min=Decimal(500),
            zone_liq_max=Decimal(1500),
        ),
        SizeTier(
            max_shares=None, quote_depth="safe", reward_min=Decimal(25), liq_min=Decimal(2000)
        ),
    ]
    assert tiers_log_repr(tiers) == "20:normal:5:500:1500:all|inf:safe:25:2000:off:all"
    assert tiers_log_repr([]) == "none"


def test_effective_reward_min_gates_the_filter(market: Market, farm_state: FarmState):
    below = market.model_copy(
        update={"rewards_rate_per_day": Decimal(8), "effective_reward_min": Decimal(10)}
    )
    assert first_failing_filter(below, farm_state.config.filters) == "reward", "below tier floor"
    at = market.model_copy(
        update={"rewards_rate_per_day": Decimal(8), "effective_reward_min": Decimal(5)}
    )
    assert first_failing_filter(at, farm_state.config.filters) is None, "above tier floor -> passes"


def test_effective_reward_min_zero_overrides_high_global(market: Market, farm_state: FarmState):
    f = farm_state.config.filters.model_copy(update={"reward_min": Decimal(50)})
    zero = market.model_copy(
        update={"rewards_rate_per_day": Decimal(2), "effective_reward_min": Decimal(0)}
    )
    assert first_failing_filter(zero, f) is None, "tier reward_min=0 overrides global 50 -> passes"
    unset = market.model_copy(
        update={"rewards_rate_per_day": Decimal(2), "effective_reward_min": None}
    )
    assert first_failing_filter(unset, f) == "reward", "unannotated -> global 50 rejects rate 2"


def test_unannotated_market_uses_global_reward_floor(market: Market, farm_state: FarmState):
    m = market.model_copy(update={"rewards_rate_per_day": Decimal(8), "effective_reward_min": None})
    assert first_failing_filter(m, farm_state.config.filters) is None


def test_no_reward_ceiling(market: Market, farm_state: FarmState):
    huge = market.model_copy(
        update={"rewards_rate_per_day": Decimal(999999), "effective_reward_min": Decimal(5)}
    )
    assert first_failing_filter(huge, farm_state.config.filters) is None


def test_effective_liq_min_gates_the_filter(market: Market, farm_state: FarmState):
    below = market.model_copy(
        update={"liquidity": Decimal(800), "effective_liq_min": Decimal(1000)}
    )
    assert first_failing_filter(below, farm_state.config.filters) == "liquidity", "below tier floor"
    at = market.model_copy(update={"liquidity": Decimal(800), "effective_liq_min": Decimal(500)})
    assert first_failing_filter(at, farm_state.config.filters) is None, "above tier floor -> passes"


def test_effective_liq_min_zero_overrides_high_global(market: Market, farm_state: FarmState):
    f = farm_state.config.filters.model_copy(update={"liq_min": Decimal(50000)})
    zero = market.model_copy(update={"liquidity": Decimal(100), "effective_liq_min": Decimal(0)})
    assert first_failing_filter(zero, f) is None, "tier liq_min=0 overrides global 50000 -> passes"
    unset = market.model_copy(update={"liquidity": Decimal(100), "effective_liq_min": None})
    assert first_failing_filter(unset, f) == "liquidity", "unannotated -> global 50000 rejects 100"


def test_effective_zone_liq_max_gates_the_filter(market: Market, farm_state: FarmState):
    over = market.model_copy(
        update={"zone_liquidity": Decimal(3000), "effective_zone_liq_max": Decimal(2500)}
    )
    assert first_failing_filter(over, farm_state.config.filters) == "zone_liquidity", "over cap"
    under = market.model_copy(
        update={"zone_liquidity": Decimal(2000), "effective_zone_liq_max": Decimal(2500)}
    )
    assert first_failing_filter(under, farm_state.config.filters) is None, "under cap -> passes"


def test_tier_zone_none_with_no_global_is_off(market: Market, farm_state: FarmState):
    m = market.model_copy(
        update={"zone_liquidity": Decimal(999999), "effective_zone_liq_max": None}
    )
    assert first_failing_filter(m, farm_state.config.filters) is None


def test_tier_zone_overrides_global(market: Market, farm_state: FarmState):
    f = farm_state.config.filters.model_copy(update={"zone_liq_max": Decimal(1000)})
    used = market.model_copy(
        update={"zone_liquidity": Decimal(2500), "effective_zone_liq_max": Decimal(3000)}
    )
    assert first_failing_filter(used, f) is None, "tier cap 3000 used (not global 1000) -> passes"
    unset = market.model_copy(
        update={"zone_liquidity": Decimal(2500), "effective_zone_liq_max": None}
    )
    assert first_failing_filter(unset, f) == "zone_liquidity", "unannotated -> global 1000 rejects"


def test_tier_zone_unknown_when_book_missing(market: Market, farm_state: FarmState):
    m = market.model_copy(update={"zone_liquidity": None, "effective_zone_liq_max": Decimal(2500)})
    assert first_failing_filter(m, farm_state.config.filters) == "zone_unknown"


def test_tier_zone_zero_is_a_real_cap_not_off(market: Market, farm_state: FarmState):
    f = farm_state.config.filters.model_copy(update={"zone_liq_max": Decimal(5000)})
    crowded = market.model_copy(
        update={"zone_liquidity": Decimal(1), "effective_zone_liq_max": Decimal(0)}
    )
    assert first_failing_filter(crowded, f) == "zone_liquidity", "tier 0-cap rejects any crowding"
    empty = market.model_copy(
        update={"zone_liquidity": Decimal(0), "effective_zone_liq_max": Decimal(0)}
    )
    assert first_failing_filter(empty, f) is None, "zone 0 <= cap 0 -> passes"


def test_tier_zone_boundary_is_inclusive(market: Market, farm_state: FarmState):
    at = market.model_copy(
        update={"zone_liquidity": Decimal(2500), "effective_zone_liq_max": Decimal(2500)}
    )
    assert first_failing_filter(at, farm_state.config.filters) is None, "== cap -> passes"
    over = market.model_copy(
        update={"zone_liquidity": Decimal("2500.01"), "effective_zone_liq_max": Decimal(2500)}
    )
    assert first_failing_filter(over, farm_state.config.filters) == "zone_liquidity", (
        "over -> fails"
    )


def test_effective_time_remaining_gates_the_filter(market: Market, farm_state: FarmState):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    within = market.model_copy(
        update={"end_date": now + timedelta(hours=6), "effective_time_remaining": "12h"}
    )
    assert first_failing_filter(within, farm_state.config.filters, now) == "time_remaining"
    beyond = market.model_copy(
        update={"end_date": now + timedelta(hours=18), "effective_time_remaining": "12h"}
    )
    assert first_failing_filter(beyond, farm_state.config.filters, now) is None


def test_effective_time_remaining_all_overrides_strict_global(
    market: Market, farm_state: FarmState
):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    f = farm_state.config.filters.model_copy(update={"time_remaining": "7d"})
    loosened = market.model_copy(
        update={"end_date": now + timedelta(hours=1), "effective_time_remaining": "all"}
    )
    assert first_failing_filter(loosened, f, now) is None, "tier all overrides global 7d -> passes"
    unset = market.model_copy(
        update={"end_date": now + timedelta(hours=1), "effective_time_remaining": None}
    )
    assert first_failing_filter(unset, f, now) == "time_remaining", (
        "unannotated -> global 7d rejects"
    )


def test_size_per_market_drives_tier(market: Market):
    m = market.model_copy(update={"rewards_min_size": Decimal(50), "min_order_size": Decimal(5)})
    assert size_per_market(m) == Decimal(50)
    assert resolve_tier(TIERS, size_per_market(m), "safe", Decimal(0), Decimal(0), None, "all") == (
        "normal",
        Decimal(10),
        Decimal(1000),
        None,
        "all",
    )


def test_effective_depth_changes_quote_distance(market: Market):
    mid = Decimal("0.5")
    agg = compute_quote(market, mid, "aggressive")
    safe = compute_quote(market, mid, "safe")
    assert agg is not None and safe is not None
    assert agg[0] > safe[0], "aggressive bid is closer to mid (higher) than safe"


def sized(market: Market, cid: str, shares: int, rate: int) -> Market:
    return market.model_copy(
        update={
            "condition_id": cid,
            "slug": cid,
            "yes_token_id": f"y-{cid}",
            "no_token_id": f"n-{cid}",
            "rewards_min_size": Decimal(shares),
            "min_order_size": Decimal(1),
            "rewards_rate_per_day": Decimal(rate),
            "liquidity": Decimal(5000),
        }
    )


def wire_reconcile(monkeypatch, markets: list[Market], opened: list[str]) -> None:
    async def fetch(http):
        return markets

    async def bal(addr):
        return Decimal("100000")

    async def mids(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    async def oids(client):
        return set()

    async def noop(*a, **k):
        return None

    async def rec_open(client, state, websocket, mkt, midpoints):
        opened.append(mkt.condition_id)

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fetch)
    monkeypatch.setattr(worker_mod, "get_balance", bal, raising=False)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", mids)
    monkeypatch.setattr(worker_mod, "get_open_order_ids", oids, raising=False)
    monkeypatch.setattr(worker_mod, "exit_held_legs", noop)
    monkeypatch.setattr(worker_mod, "open_position", rec_open)


async def test_reconcile_tick_annotates_each_market_by_size(
    market: Market, farm_state: FarmState, monkeypatch
):
    farm_state.config.size_tiers = TIERS
    farm_state.positions.clear()
    small = sized(market, "small", 10, 30)
    large = sized(market, "large", 100, 30)
    wire_reconcile(monkeypatch, [small, large], [])

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert (small.effective_depth, small.effective_reward_min) == ("aggressive", Decimal(5))
    assert (large.effective_depth, large.effective_reward_min) == ("safe", Decimal(25))


async def test_reconcile_tick_empty_tiers_annotates_globals(
    market: Market, farm_state: FarmState, monkeypatch
):
    farm_state.config.size_tiers = []
    farm_state.positions.clear()
    m = sized(market, "m", 50, 100)
    wire_reconcile(monkeypatch, [m], [])

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert m.effective_depth == farm_state.config.quote_depth
    assert m.effective_reward_min == farm_state.config.filters.reward_min


async def test_reconcile_tick_annotates_effective_time_remaining_by_size(
    market: Market, farm_state: FarmState, monkeypatch
):
    tiers = [
        SizeTier(
            max_shares=Decimal(20),
            quote_depth="aggressive",
            reward_min=Decimal(5),
            liq_min=Decimal(500),
            time_remaining="12h",
        ),
        SizeTier(
            max_shares=None, quote_depth="safe", reward_min=Decimal(25), liq_min=Decimal(2000)
        ),
    ]
    farm_state.config.size_tiers = tiers
    farm_state.positions.clear()
    small = sized(market, "small", 10, 30).model_copy(
        update={"end_date": datetime.now(timezone.utc) + timedelta(days=365)}
    )
    wire_reconcile(monkeypatch, [small], [])

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert small.effective_time_remaining == "12h"

    farm_state.config.size_tiers = []
    farm_state.positions.clear()
    m = sized(market, "m", 50, 100)
    wire_reconcile(monkeypatch, [m], [])

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert m.effective_time_remaining == farm_state.config.filters.time_remaining


async def test_reconcile_tick_tier_reward_floor_filters_by_size(
    market: Market, farm_state: FarmState, monkeypatch
):
    farm_state.config.size_tiers = TIERS
    farm_state.positions.clear()
    opened: list[str] = []
    small = sized(market, "small", 10, 8)
    large = sized(market, "large", 100, 8)
    wire_reconcile(monkeypatch, [small, large], opened)

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert "small" in opened, "rate 8 >= small tier floor $5 -> farmed"
    assert "large" not in opened, "rate 8 < large tier floor $25 -> rejected"


async def test_reconcile_tick_tier_liq_floor_filters_by_size(
    market: Market, farm_state: FarmState, monkeypatch
):
    farm_state.config.size_tiers = TIERS
    farm_state.positions.clear()
    opened: list[str] = []
    small = sized(market, "small", 10, 30).model_copy(update={"liquidity": Decimal(800)})
    large = sized(market, "large", 100, 30).model_copy(update={"liquidity": Decimal(800)})
    wire_reconcile(monkeypatch, [small, large], opened)

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert "small" in opened, "liq 800 >= small tier floor $500 -> farmed"
    assert "large" not in opened, "liq 800 < large tier floor $2000 -> rejected"


async def test_reconcile_fetches_books_when_a_tier_has_zone_cap(
    market: Market, farm_state: FarmState, monkeypatch
):
    farm_state.config.size_tiers = [
        SizeTier(
            max_shares=None, quote_depth="normal", liq_min=Decimal(0), zone_liq_max=Decimal(2500)
        )
    ]
    farm_state.config.filters = farm_state.config.filters.model_copy(
        update={"zone_liq_max": None, "max_fill_loss": None}
    )
    farm_state.positions.clear()
    captured: dict = {}

    async def fake_annotate(http, markets, need_books, need_exit_loss, quote_depth):
        captured["need_books"] = need_books
        return {}

    monkeypatch.setattr(worker_mod, "annotate_live_metrics", fake_annotate)
    wire_reconcile(monkeypatch, [sized(market, "m", 10, 30)], [])

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert captured.get("need_books") is True, "a per-tier zone cap must trigger book-fetching"


async def test_reconcile_tick_tier_zone_cap_filters_market(
    market: Market, farm_state: FarmState, monkeypatch
):
    farm_state.config.size_tiers = [
        SizeTier(
            max_shares=None,
            quote_depth="normal",
            reward_min=Decimal(0),
            liq_min=Decimal(0),
            zone_liq_max=Decimal(2000),
        )
    ]
    farm_state.config.filters = farm_state.config.filters.model_copy(
        update={"zone_liq_max": None, "max_fill_loss": None}
    )
    farm_state.positions.clear()
    opened: list[str] = []
    crowded = sized(market, "crowded", 10, 30)
    calm = sized(market, "calm", 10, 30)

    async def fake_annotate(http, markets, need_books, need_exit_loss, quote_depth):
        for mk in markets:
            mk.zone_liquidity = Decimal(5000) if mk.condition_id == "crowded" else Decimal(500)
        return {}

    monkeypatch.setattr(worker_mod, "annotate_live_metrics", fake_annotate)
    wire_reconcile(monkeypatch, [crowded, calm], opened)

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert "calm" in opened, "zone 500 <= tier cap 2000 -> farmed"
    assert "crowded" not in opened, "zone 5000 > tier cap 2000 -> rejected"


async def test_reconcile_no_book_fetch_when_no_zone_anywhere(
    market: Market, farm_state: FarmState, monkeypatch
):
    farm_state.config.size_tiers = [
        SizeTier(max_shares=None, quote_depth="normal", liq_min=Decimal(0))
    ]
    farm_state.config.filters = farm_state.config.filters.model_copy(
        update={"zone_liq_max": None, "max_fill_loss": None, "price_min": None, "price_max": None}
    )
    farm_state.positions.clear()
    captured: dict = {}

    async def fake_annotate(http, markets, need_books, need_exit_loss, quote_depth):
        captured["need_books"] = need_books
        return {}

    monkeypatch.setattr(worker_mod, "annotate_live_metrics", fake_annotate)
    wire_reconcile(monkeypatch, [sized(market, "m", 10, 30)], [])

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert captured.get("need_books") is not True, "no zone anywhere -> no zone book-fetch"


async def test_reconcile_mixed_tiers_zone_caps_only_the_tier_that_has_one(
    market: Market, farm_state: FarmState, monkeypatch
):
    farm_state.config.size_tiers = [
        SizeTier(
            max_shares=Decimal(20),
            quote_depth="normal",
            reward_min=Decimal(0),
            liq_min=Decimal(0),
            zone_liq_max=Decimal(2000),
        ),
        SizeTier(max_shares=None, quote_depth="normal", reward_min=Decimal(0), liq_min=Decimal(0)),
    ]
    farm_state.config.filters = farm_state.config.filters.model_copy(
        update={"zone_liq_max": None, "max_fill_loss": None}
    )
    farm_state.positions.clear()
    opened: list[str] = []
    small = sized(market, "small", 10, 30)
    big = sized(market, "big", 100, 30)

    async def fake_annotate(http, markets, need_books, need_exit_loss, quote_depth):
        for mk in markets:
            mk.zone_liquidity = Decimal(5000)
        return {}

    monkeypatch.setattr(worker_mod, "annotate_live_metrics", fake_annotate)
    wire_reconcile(monkeypatch, [small, big], opened)

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert "small" not in opened, "small tier zone cap 2000 < 5000 -> rejected"
    assert "big" in opened, "catch-all has no zone cap -> farmed despite 5000 crowding"


async def test_reconcile_oversized_market_gated_by_derived_global_zone(
    market: Market, farm_state: FarmState, monkeypatch
):
    farm_state.config.size_tiers = [
        SizeTier(
            max_shares=Decimal(20),
            quote_depth="normal",
            reward_min=Decimal(0),
            liq_min=Decimal(0),
            zone_liq_max=Decimal(1500),
        ),
        SizeTier(
            max_shares=Decimal(50),
            quote_depth="normal",
            reward_min=Decimal(0),
            liq_min=Decimal(0),
            zone_liq_max=Decimal(2500),
        ),
    ]
    farm_state.config.filters = farm_state.config.filters.model_copy(
        update={"zone_liq_max": Decimal(2500), "max_fill_loss": None}
    )
    farm_state.positions.clear()
    opened: list[str] = []
    oversized = sized(market, "oversized", 100, 30)

    async def fake_annotate(http, markets, need_books, need_exit_loss, quote_depth):
        for mk in markets:
            mk.zone_liquidity = Decimal(9000)
        return {}

    monkeypatch.setattr(worker_mod, "annotate_live_metrics", fake_annotate)
    wire_reconcile(monkeypatch, [oversized], opened)

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert "oversized" not in opened, "oversized crowded market gated by the derived global zone"


async def test_reconcile_rechecks_event_exclusion_before_each_open(
    market: Market, farm_state: FarmState, monkeypatch
):
    farm_state.config.size_tiers = []
    farm_state.positions.clear()
    m1 = sized(market, "m1", 10, 30).model_copy(update={"event_slug": "evt"})
    m2 = sized(market, "m2", 10, 30).model_copy(update={"event_slug": "evt"})
    opened: list[str] = []

    async def rec_open(client, state, websocket, mkt, midpoints):
        opened.append(mkt.condition_id)
        state.excluded_events["evt"] = datetime.now(timezone.utc) + timedelta(minutes=30)

    wire_reconcile(monkeypatch, [m1, m2], [])
    monkeypatch.setattr(worker_mod, "open_position", rec_open)

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert len(opened) == 1, "only the first sibling opens; the second is re-checked + skipped"


async def test_reconcile_rechecks_blacklist_before_each_open(
    market: Market, farm_state: FarmState, monkeypatch
):
    farm_state.config.size_tiers = []
    farm_state.positions.clear()
    m1 = sized(market, "m1", 10, 30)
    m2 = sized(market, "m2", 10, 30)
    opened: list[str] = []

    async def rec_open(client, state, websocket, mkt, midpoints):
        opened.append(mkt.condition_id)
        if mkt.condition_id == "m1":
            state.health["m2"] = MarketHealth(blacklist_permanent=True)

    wire_reconcile(monkeypatch, [m1, m2], [])
    monkeypatch.setattr(worker_mod, "open_position", rec_open)

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert "m1" in opened
    assert "m2" not in opened, "m2 re-checked: blacklisted by m1's fill mid-batch"


async def test_reconcile_recheck_does_not_overskip(
    market: Market, farm_state: FarmState, monkeypatch
):
    farm_state.config.size_tiers = []
    farm_state.positions.clear()
    m1 = sized(market, "m1", 10, 30)
    m2 = sized(market, "m2", 10, 30)
    opened: list[str] = []
    wire_reconcile(monkeypatch, [m1, m2], opened)

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert set(opened) == {"m1", "m2"}, "no mid-batch exclusion -> both open (re-check is a no-op)"
