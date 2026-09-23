from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.constants import FILL_COOLOFF_TIERS, MAX_TEMP_BLACKLIST_SECONDS
from app.farm.schemas import FarmState, MarketHealth
from app.farm.volatility import (
    blacklist_for_fill,
    clear_ban_if_recovered,
    is_blacklisted,
    market_net,
    record_price_sample,
    record_roundtrip_pnl,
    reevaluate_blacklist,
    window_move,
)

BASE = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
MID = "market-A"


def ts(offset_seconds: int) -> float:
    return (BASE + timedelta(seconds=offset_seconds)).timestamp()


def test_window_move_peak_to_trough():
    samples = [
        (ts(-50), Decimal("0.50")),
        (ts(-30), Decimal("0.54")),
        (ts(-10), Decimal("0.51")),
    ]
    assert window_move(samples, 60, BASE.timestamp()) == Decimal("0.04")


def test_window_move_excludes_samples_outside_window():
    samples = [(ts(-120), Decimal("0.40")), (ts(-10), Decimal("0.50"))]
    assert window_move(samples, 60, BASE.timestamp()) == Decimal("0")


def test_3c_in_60s_blacklists_5min(farm_state: FarmState):
    record_price_sample(farm_state, MID, Decimal("0.50"), now=BASE)
    tier = record_price_sample(farm_state, MID, Decimal("0.53"), now=BASE + timedelta(seconds=30))
    assert tier == "5min"
    assert farm_state.health[MID].blacklist_until == BASE + timedelta(seconds=30) + timedelta(
        minutes=5
    )
    assert farm_state.health[MID].blacklist_permanent is False


def test_7c_in_5min_blacklists_10min(farm_state: FarmState):
    record_price_sample(farm_state, MID, Decimal("0.50"), now=BASE)
    tier = record_price_sample(farm_state, MID, Decimal("0.57"), now=BASE + timedelta(minutes=4))
    assert tier == "10min"
    assert farm_state.health[MID].blacklist_permanent is False


def test_12c_in_15min_is_temp_20min_not_permanent(farm_state: FarmState):
    record_price_sample(farm_state, MID, Decimal("0.40"), now=BASE)
    tier = record_price_sample(farm_state, MID, Decimal("0.52"), now=BASE + timedelta(minutes=10))
    assert tier == "20min"
    assert farm_state.health[MID].blacklist_permanent is False
    assert is_blacklisted(farm_state, MID, now=BASE + timedelta(hours=5)) is False


def test_most_severe_tier_wins_still_temp(farm_state: FarmState):
    record_price_sample(farm_state, MID, Decimal("0.40"), now=BASE)
    tier = record_price_sample(farm_state, MID, Decimal("0.55"), now=BASE + timedelta(seconds=30))
    assert tier == "20min"
    assert farm_state.health[MID].blacklist_permanent is False


def test_does_not_shorten_existing_longer_blacklist(farm_state: FarmState):
    health = farm_state.health.setdefault(MID, MarketHealth())
    far = BASE + timedelta(hours=1)
    health.blacklist_until = far
    record_price_sample(farm_state, MID, Decimal("0.50"), now=BASE)
    tier = record_price_sample(farm_state, MID, Decimal("0.53"), now=BASE + timedelta(seconds=20))
    assert tier is None
    assert health.blacklist_until == far


def test_is_blacklisted_true_then_false_across_expiry(farm_state: FarmState):
    farm_state.health[MID] = MarketHealth(blacklist_until=BASE + timedelta(minutes=15))
    assert is_blacklisted(farm_state, MID, now=BASE + timedelta(minutes=10)) is True
    assert is_blacklisted(farm_state, MID, now=BASE + timedelta(minutes=20)) is False


def test_reevaluate_still_volatile_reapplies_short_cooloff(farm_state: FarmState):
    now = BASE + timedelta(minutes=16)
    farm_state.health[MID] = MarketHealth(
        blacklist_until=BASE + timedelta(minutes=15),
        price_samples=[
            ((now - timedelta(seconds=40)).timestamp(), Decimal("0.50")),
            ((now - timedelta(seconds=10)).timestamp(), Decimal("0.55")),
        ],
    )
    reevaluate_blacklist(farm_state, MID, now=now)
    assert farm_state.health[MID].blacklist_permanent is False
    assert farm_state.health[MID].blacklist_until == now + timedelta(minutes=5)


def test_reevaluate_calm_lifts_blacklist(farm_state: FarmState):
    now = BASE + timedelta(minutes=16)
    farm_state.health[MID] = MarketHealth(
        blacklist_until=BASE + timedelta(minutes=15),
        price_samples=[
            ((now - timedelta(seconds=40)).timestamp(), Decimal("0.50")),
            ((now - timedelta(seconds=10)).timestamp(), Decimal("0.505")),
        ],
    )
    reevaluate_blacklist(farm_state, MID, now=now)
    assert farm_state.health[MID].blacklist_until is None
    assert is_blacklisted(farm_state, MID, now=now) is False


def test_reevaluate_stale_buffer_lifts(farm_state: FarmState):
    farm_state.health[MID] = MarketHealth(
        blacklist_until=BASE + timedelta(minutes=15),
        price_samples=[(BASE.timestamp(), Decimal("0.40")), (ts(30), Decimal("0.55"))],
    )
    reevaluate_blacklist(farm_state, MID, now=BASE + timedelta(minutes=40))
    assert farm_state.health[MID].blacklist_until is None
    assert farm_state.health[MID].blacklist_permanent is False


def test_reevaluate_noop_before_expiry(farm_state: FarmState):
    until = BASE + timedelta(minutes=15)
    farm_state.health[MID] = MarketHealth(blacklist_until=until)
    reevaluate_blacklist(farm_state, MID, now=BASE + timedelta(minutes=5))
    assert farm_state.health[MID].blacklist_until == until


def test_fill_cooloff_first_offense_is_5min(farm_state: FarmState):
    blacklist_for_fill(farm_state, MID, now=BASE)
    h = farm_state.health[MID]
    assert h.blacklist_until == BASE + timedelta(seconds=FILL_COOLOFF_TIERS[0])
    assert h.blacklist_permanent is False
    assert is_blacklisted(farm_state, MID, now=BASE + timedelta(minutes=1)) is True


def test_fill_cooloff_scales_with_loss_history_never_permanent(farm_state: FarmState):
    h = farm_state.health.setdefault(MID, MarketHealth())
    h.loss_roundtrips = 2
    blacklist_for_fill(farm_state, MID, now=BASE)
    assert h.blacklist_until == BASE + timedelta(seconds=FILL_COOLOFF_TIERS[2])
    assert h.blacklist_permanent is False
    for _ in range(10):
        blacklist_for_fill(farm_state, MID, now=BASE)
    assert h.blacklist_permanent is False


def test_fill_cooloff_doubles_when_underwater_capped_30min(farm_state: FarmState):
    h = farm_state.health.setdefault(MID, MarketHealth())
    h.loss_roundtrips = 2
    h.net_fill_pnl = Decimal("-3")
    blacklist_for_fill(farm_state, MID, now=BASE)
    assert h.blacklist_until == BASE + timedelta(seconds=MAX_TEMP_BLACKLIST_SECONDS)
    assert h.blacklist_permanent is False


def test_fill_cooloff_lifts_if_calm(farm_state: FarmState):
    blacklist_for_fill(farm_state, MID, now=BASE)
    after = BASE + timedelta(seconds=FILL_COOLOFF_TIERS[0] + 1)
    reevaluate_blacklist(farm_state, MID, now=after)
    assert is_blacklisted(farm_state, MID, now=after) is False


def test_fill_never_shortens_existing(farm_state: FarmState):
    longer = BASE + timedelta(hours=5)
    farm_state.health[MID] = MarketHealth(blacklist_until=longer)
    blacklist_for_fill(farm_state, MID, now=BASE)
    assert farm_state.health[MID].blacklist_until == longer


def test_fill_skips_permanent(farm_state: FarmState):
    farm_state.health[MID] = MarketHealth(blacklist_permanent=True)
    blacklist_for_fill(farm_state, MID, now=BASE)
    assert farm_state.health[MID].blacklist_until is None
    assert farm_state.health[MID].blacklist_permanent is True


def test_roundtrip_accrues_net_and_counts_losses(farm_state: FarmState):
    record_roundtrip_pnl(farm_state, MID, "YES", Decimal("-1.0"), closed=True, now=BASE)
    h = farm_state.health[MID]
    assert h.net_fill_pnl == Decimal("-1.0")
    assert h.loss_roundtrips == 1
    assert h.blacklist_permanent is False


def test_roundtrip_dust_loss_not_counted(farm_state: FarmState):
    record_roundtrip_pnl(farm_state, MID, "YES", Decimal("-0.20"), closed=True, now=BASE)
    assert farm_state.health[MID].loss_roundtrips == 0


def test_roundtrip_winner_decays_strikes(farm_state: FarmState):
    h = farm_state.health.setdefault(MID, MarketHealth())
    h.loss_roundtrips = 2
    h.net_fill_pnl = Decimal("-1.0")
    record_roundtrip_pnl(farm_state, MID, "YES", Decimal("0.5"), closed=True, now=BASE)
    assert h.loss_roundtrips == 1
    assert h.net_fill_pnl == Decimal("-0.5")
    assert h.blacklist_permanent is False


def test_fragmented_exit_counts_as_one_roundtrip(farm_state: FarmState):
    record_roundtrip_pnl(farm_state, MID, "YES", Decimal("-1.3"), closed=False, now=BASE)
    record_roundtrip_pnl(farm_state, MID, "YES", Decimal("-1.3"), closed=False, now=BASE)
    record_roundtrip_pnl(farm_state, MID, "YES", Decimal("-1.3"), closed=True, now=BASE)
    h = farm_state.health[MID]
    assert h.net_fill_pnl == Decimal("-3.9")
    assert h.loss_roundtrips == 1
    assert h.open_rt_net == {}
    assert h.blacklist_permanent is False


def test_door1_uses_cycle_total_not_fragment(farm_state: FarmState):
    record_roundtrip_pnl(farm_state, MID, "YES", Decimal("-2"), closed=False, now=BASE)
    record_roundtrip_pnl(farm_state, MID, "YES", Decimal("-2"), closed=False, now=BASE)
    assert farm_state.health[MID].blacklist_permanent is False
    record_roundtrip_pnl(farm_state, MID, "YES", Decimal("-2"), closed=True, now=BASE)
    assert farm_state.health[MID].blacklist_permanent is True


def test_door1_catastrophic_single_loss_permanent(farm_state: FarmState):
    record_roundtrip_pnl(farm_state, MID, "YES", Decimal("-9"), closed=True, now=BASE)
    h = farm_state.health[MID]
    assert h.blacklist_permanent is True
    assert is_blacklisted(farm_state, MID, now=BASE + timedelta(hours=10)) is True


def test_door2_persistent_bleeder_permanent(farm_state: FarmState):
    h = farm_state.health.setdefault(MID, MarketHealth())
    h.net_fill_pnl = Decimal("-4.6")
    h.loss_roundtrips = 2
    record_roundtrip_pnl(farm_state, MID, "YES", Decimal("-1.0"), closed=True, now=BASE)
    assert h.net_fill_pnl == Decimal("-5.6")
    assert h.loss_roundtrips == 3
    assert h.blacklist_permanent is True


def test_door3_deep_bleeder_below_floor_still_banned(farm_state: FarmState):
    h = farm_state.health.setdefault(MID, MarketHealth())
    h.net_fill_pnl = Decimal("-6.6")
    record_roundtrip_pnl(farm_state, MID, "YES", Decimal("-0.49"), closed=True, now=BASE)
    assert h.loss_roundtrips == 0
    assert h.blacklist_permanent is True


def test_door2_not_yet_three_losses(farm_state: FarmState):
    h = farm_state.health.setdefault(MID, MarketHealth())
    h.net_fill_pnl = Decimal("-4.6")
    h.loss_roundtrips = 1
    record_roundtrip_pnl(farm_state, MID, "YES", Decimal("-1.0"), closed=True, now=BASE)
    assert h.blacklist_permanent is False


def test_winner_never_permanent_even_on_big_loss(farm_state: FarmState):
    h = farm_state.health.setdefault(MID, MarketHealth())
    h.cum_reward_credit = Decimal("15")
    record_roundtrip_pnl(farm_state, MID, "YES", Decimal("-9"), closed=True, now=BASE)
    assert h.blacklist_permanent is False
    assert market_net(farm_state, MID) == Decimal("6")


def test_reward_from_market_rewards_protects_winner(farm_state: FarmState):
    farm_state.market_rewards[MID] = Decimal("12")
    farm_state.market_rewards_baseline[MID] = Decimal("2")
    record_roundtrip_pnl(farm_state, MID, "YES", Decimal("-9"), closed=True, now=BASE)
    assert farm_state.health[MID].blacklist_permanent is False


def test_clear_ban_if_recovered_lifts_permanent_when_reward_catches_up(farm_state: FarmState):
    farm_state.health[MID] = MarketHealth(blacklist_permanent=True, net_fill_pnl=Decimal("-9"))
    farm_state.market_rewards_baseline[MID] = Decimal("0")
    farm_state.market_rewards[MID] = Decimal("10")
    assert clear_ban_if_recovered(farm_state, MID) is True
    assert farm_state.health[MID].blacklist_permanent is False
    assert is_blacklisted(farm_state, MID, now=BASE) is False


def test_clear_ban_if_recovered_keeps_a_real_loser(farm_state: FarmState):
    farm_state.health[MID] = MarketHealth(blacklist_permanent=True, net_fill_pnl=Decimal("-9"))
    assert clear_ban_if_recovered(farm_state, MID) is False
    assert farm_state.health[MID].blacklist_permanent is True


def test_volatility_cooloff_not_lifted_by_recovery(farm_state: FarmState):
    farm_state.health[MID] = MarketHealth(blacklist_until=BASE + timedelta(minutes=20))
    farm_state.market_rewards[MID] = Decimal("5")
    assert clear_ban_if_recovered(farm_state, MID) is False
    assert farm_state.health[MID].blacklist_until == BASE + timedelta(minutes=20)


def test_economic_temp_cooloff_lifted_by_recovery(farm_state: FarmState):
    farm_state.health[MID] = MarketHealth(
        blacklist_until=BASE + timedelta(minutes=15), net_fill_pnl=Decimal("-2")
    )
    farm_state.market_rewards_baseline[MID] = Decimal("0")
    farm_state.market_rewards[MID] = Decimal("3")
    assert clear_ban_if_recovered(farm_state, MID) is True
    assert farm_state.health[MID].blacklist_until is None


def test_recovery_resets_loss_basis(farm_state: FarmState):
    farm_state.health[MID] = MarketHealth(
        blacklist_permanent=True, net_fill_pnl=Decimal("-11"), loss_roundtrips=2
    )
    farm_state.health[MID].open_rt_net["YES"] = Decimal("-1")
    farm_state.market_rewards_baseline[MID] = Decimal("0")
    farm_state.market_rewards[MID] = Decimal("12")
    assert clear_ban_if_recovered(farm_state, MID) is True
    h = farm_state.health[MID]
    assert h.net_fill_pnl == Decimal("0")
    assert h.loss_roundtrips == 0
    assert h.open_rt_net == {}


def test_two_legs_do_not_commingle_into_a_false_catastrophic(farm_state: FarmState):
    record_roundtrip_pnl(farm_state, MID, "YES", Decimal("-3"), closed=False, now=BASE)
    record_roundtrip_pnl(farm_state, MID, "NO", Decimal("-3"), closed=True, now=BASE)
    h = farm_state.health[MID]
    assert h.net_fill_pnl == Decimal("-6")
    assert h.loss_roundtrips == 1
    assert h.blacklist_permanent is False
    assert h.open_rt_net["YES"] == Decimal("-3")
