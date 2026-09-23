from datetime import datetime, timedelta, timezone

import pytest

from app.farm import blacklist_store
from app.farm.blacklist_store import load_blacklist, path, save_blacklist
from app.farm.schemas import FarmState, MarketHealth
from app.farm.volatility import is_blacklisted

WALLET = "0xWALLETaddressAAAA"


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(blacklist_store, "BLACKLIST_STATE_DIR", str(tmp_path))
    return tmp_path


def state(farm_state, wallet=WALLET) -> FarmState:
    return FarmState(config=farm_state.config, wallet_address=wallet)


def test_round_trip_carries_until_permanent_and_strikes(farm_state, store_dir):
    now = datetime.now(timezone.utc)
    src = state(farm_state)
    src.health["m-temp"] = MarketHealth(blacklist_until=now + timedelta(minutes=30), fill_strikes=1)
    src.health["m-perm"] = MarketHealth(blacklist_permanent=True, fill_strikes=3)
    src.health["m-clean"] = MarketHealth()
    save_blacklist(src)

    dst = state(farm_state)
    assert load_blacklist(dst) == 2
    assert "m-clean" not in dst.health
    assert is_blacklisted(dst, "m-temp", now) is True
    assert dst.health["m-temp"].fill_strikes == 1
    assert is_blacklisted(dst, "m-perm", now) is True
    assert dst.health["m-perm"].blacklist_permanent is True
    assert dst.health["m-perm"].fill_strikes == 3
    assert dst.health["m-temp"].blacklist_until.tzinfo is not None


def test_expired_temp_loads_inactive(farm_state, store_dir):
    now = datetime.now(timezone.utc)
    src = state(farm_state)
    src.health["m"] = MarketHealth(blacklist_until=now - timedelta(minutes=5))
    save_blacklist(src)

    dst = state(farm_state)
    load_blacklist(dst)
    assert is_blacklisted(dst, "m", now) is False


def test_remaining_time_is_preserved(farm_state, store_dir):
    now = datetime.now(timezone.utc)
    src = state(farm_state)
    until = now + timedelta(minutes=42)
    src.health["m"] = MarketHealth(blacklist_until=until)
    save_blacklist(src)

    dst = state(farm_state)
    load_blacklist(dst)
    assert is_blacklisted(dst, "m", now) is True
    assert is_blacklisted(dst, "m", until + timedelta(seconds=1)) is False


def test_missing_file_returns_zero(farm_state, store_dir):
    assert load_blacklist(state(farm_state, "0xnever-saved")) == 0


def test_corrupt_file_does_not_crash(farm_state, store_dir):
    file_path = path("0xcorrupt")
    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write("{ this is not json")
    assert load_blacklist(state(farm_state, "0xcorrupt")) == 0


def test_wallets_do_not_cross_contaminate(farm_state, store_dir):
    now = datetime.now(timezone.utc)
    a = state(farm_state, "0xAAAA")
    a.health["m"] = MarketHealth(blacklist_until=now + timedelta(minutes=30))
    save_blacklist(a)

    b = state(farm_state, "0xBBBB")
    assert load_blacklist(b) == 0


def test_empty_wallet_address_is_noop(farm_state, store_dir):
    s = state(farm_state, "")
    s.health["m"] = MarketHealth(blacklist_permanent=True)
    save_blacklist(s)
    assert load_blacklist(s) == 0


def test_save_overwrites_with_latest(farm_state, store_dir):
    now = datetime.now(timezone.utc)
    s = state(farm_state)
    s.health["m"] = MarketHealth(blacklist_until=now + timedelta(minutes=10), fill_strikes=1)
    save_blacklist(s)
    s.health["m"].blacklist_until = now + timedelta(hours=2)
    s.health["m"].fill_strikes = 2
    save_blacklist(s)

    dst = state(farm_state)
    load_blacklist(dst)
    assert dst.health["m"].fill_strikes == 2
    assert is_blacklisted(dst, "m", now + timedelta(minutes=30)) is True
