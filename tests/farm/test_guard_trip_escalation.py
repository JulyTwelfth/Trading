"""Guard-trip escalation — record_guard_trip.

A depth/exit-loss guard pull whose loss estimate is SEVERE (>= GUARD_TRIP_SEVERE_LOSS) escalates
the market into the temp-blacklist tier via the health.severe_guard_trips counter, instead of only
re-arming the fixed guard_pull_until cooldown. Enforcement rides the existing
is_blacklisted() -> quote_block_reason() path — no new gate.

Noise trips (~94% of pulls, est_loss < $10) are ignored so they don't churn state.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.constants import (
    GUARD_TRIP_COOLOFF_TIERS,
    GUARD_TRIP_SEVERE_LOSS,
)
from app.farm import blacklist_store
from app.farm.blacklist_store import load_blacklist, save_blacklist
from app.farm.gating import quote_block_reason
from app.farm.health import record_guard_trip
from app.farm.schemas import FarmState, MarketHealth
from app.farm.volatility import clear_ban_if_recovered, is_blacklisted

NOW = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)
CID = "market-A"


# ── (a) noise rejection: below-threshold trips do nothing ──────────────────────


def test_noise_trip_below_threshold_is_ignored(farm_state: FarmState):
    # Fresh market: a sub-severe trip must not even create a MarketHealth entry (returns before
    # setdefault), so ~94% of noise pulls leave state untouched.
    record_guard_trip(farm_state, CID, GUARD_TRIP_SEVERE_LOSS - Decimal("0.01"), now=NOW)
    assert CID not in farm_state.health

    # Even with an existing health entry, a below-threshold loss changes nothing.
    farm_state.health[CID] = MarketHealth()
    record_guard_trip(farm_state, CID, Decimal("9.99"), now=NOW)
    h = farm_state.health[CID]
    assert h.severe_guard_trips == 0
    assert h.blacklist_until is None
    assert is_blacklisted(farm_state, CID, now=NOW) is False


# ── (b) first severe trip → 30-min tier ────────────────────────────────────────


def test_first_severe_trip_blacklists_first_tier(farm_state: FarmState):
    record_guard_trip(farm_state, CID, GUARD_TRIP_SEVERE_LOSS, now=NOW)
    h = farm_state.health[CID]
    assert h.severe_guard_trips == 1
    assert GUARD_TRIP_COOLOFF_TIERS[0] == 1800  # first tier is 30 min
    assert h.blacklist_until == NOW + timedelta(seconds=GUARD_TRIP_COOLOFF_TIERS[0])
    assert is_blacklisted(farm_state, CID, now=NOW) is True


# ── (c) second severe trip → escalates to the longer (session-long) tier ───────


def test_second_severe_trip_escalates_to_longer_tier(farm_state: FarmState):
    record_guard_trip(farm_state, CID, Decimal("12"), now=NOW)
    later = NOW + timedelta(seconds=60)
    record_guard_trip(farm_state, CID, Decimal("15"), now=later)

    h = farm_state.health[CID]
    assert h.severe_guard_trips == 2
    second_tier = GUARD_TRIP_COOLOFF_TIERS[1]
    # The escalation must be a genuinely LONGER window, not a same-length refresh — otherwise a
    # repeat offender is re-armed and re-hit within the session (the bug this feature fixes).
    assert second_tier > GUARD_TRIP_COOLOFF_TIERS[0]
    assert h.blacklist_until == later + timedelta(seconds=second_tier)


def test_third_severe_trip_clamps_to_last_tier(farm_state: FarmState):
    t = NOW
    for _ in range(3):
        record_guard_trip(farm_state, CID, Decimal("20"), now=t)
        t += timedelta(seconds=60)
    h = farm_state.health[CID]
    assert h.severe_guard_trips == 3
    # index min(3-1, len-1) clamps to the last tier — never IndexError, stays session-long.
    last_trip = NOW + timedelta(seconds=120)
    assert h.blacklist_until == last_trip + timedelta(seconds=GUARD_TRIP_COOLOFF_TIERS[-1])


# ── (d) permanent blacklist short-circuits the whole function ──────────────────


def test_permanent_blacklist_short_circuits(farm_state: FarmState):
    farm_state.health[CID] = MarketHealth(blacklist_permanent=True)
    record_guard_trip(farm_state, CID, Decimal("50"), now=NOW)
    h = farm_state.health[CID]
    assert h.severe_guard_trips == 0, "a permanently-banned market must not re-count trips"
    assert h.blacklist_until is None


# ── (e) extend never shorten ───────────────────────────────────────────────────


def test_existing_longer_ban_is_not_shortened(farm_state: FarmState):
    far = NOW + timedelta(hours=2)  # already banned well past any guard-trip tier
    farm_state.health[CID] = MarketHealth(blacklist_until=far)

    record_guard_trip(farm_state, CID, Decimal("20"), now=NOW)

    h = farm_state.health[CID]
    assert h.severe_guard_trips == 1, "the counter still increments"
    assert h.blacklist_until == far, "a shorter tier must not pull the ban in"


# ── (f) recovery resets the counter ────────────────────────────────────────────


def test_clear_ban_if_recovered_resets_counter(farm_state: FarmState):
    # Underwater on fills but reward-covered (net >= 0) → clear_ban_if_recovered lifts and resets.
    farm_state.health[CID] = MarketHealth(
        blacklist_until=NOW + timedelta(minutes=30),
        net_fill_pnl=Decimal("-5"),
        cum_reward_credit=Decimal("5"),
        severe_guard_trips=2,
    )
    assert clear_ban_if_recovered(farm_state, CID) is True
    assert farm_state.health[CID].severe_guard_trips == 0


# ── (g) persistence round-trips the counter ────────────────────────────────────


def test_blacklist_store_round_trip_preserves_counter(farm_state, tmp_path, monkeypatch):
    monkeypatch.setattr(blacklist_store, "BLACKLIST_STATE_DIR", str(tmp_path))
    wallet = "0xGUARDtripWALLET"

    src = FarmState(config=farm_state.config, wallet_address=wallet)
    src.health[CID] = MarketHealth(
        blacklist_until=NOW + timedelta(minutes=30), severe_guard_trips=3
    )
    save_blacklist(src)

    dst = FarmState(config=farm_state.config, wallet_address=wallet)
    assert load_blacklist(dst) == 1
    assert dst.health[CID].severe_guard_trips == 3


# ── (h) integration: two severe trips block quoting via the shared gate ─────────


def test_two_severe_trips_block_quoting(farm_state: FarmState):
    # Base off real now so blacklist_until lands in the actual future — quote_block_reason /
    # is_blacklisted use datetime.now() internally (no now= override in the gate).
    base = datetime.now(timezone.utc)
    record_guard_trip(farm_state, CID, Decimal("10"), now=base)
    record_guard_trip(farm_state, CID, Decimal("11"), now=base + timedelta(seconds=30))

    assert farm_state.health[CID].severe_guard_trips == 2
    assert is_blacklisted(farm_state, CID) is True
    assert quote_block_reason(farm_state, CID, "") == "blacklisted"


# ── (i) regression: escalation covers the real re-arm → fill gap (Anduril-125b) ─


def test_escalation_covers_rearm_to_fill_gap(farm_state: FarmState):
    """Replays the real Anduril-125b timing: two severe guard trips ~4 h apart, then the fatal
    fill ~2 h after the 2nd. The 30-min first tier EXPIRES between the trips (the market is
    re-entered), so only the escalated second tier can save it — this is the whole point of the
    fix and the exact case the collapsed (30 min, 30 min) tiers failed to cover."""
    trip1 = datetime(2026, 7, 6, 8, 2, tzinfo=timezone.utc)
    trip2 = datetime(2026, 7, 6, 11, 56, tzinfo=timezone.utc)
    fill = datetime(2026, 7, 6, 13, 54, tzinfo=timezone.utc)

    record_guard_trip(farm_state, CID, Decimal("12.10"), now=trip1)
    # first-tier (30 min) ban has long expired by the second trip — market was re-armed:
    assert is_blacklisted(farm_state, CID, now=trip2 - timedelta(seconds=1)) is False

    record_guard_trip(farm_state, CID, Decimal("11.61"), now=trip2)
    # escalated ban must still be in force at the moment the fatal fill would have landed:
    assert is_blacklisted(farm_state, CID, now=fill) is True
