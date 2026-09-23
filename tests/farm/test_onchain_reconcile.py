"""On-chain reconcile backstop: ask the chain what we hold and re-drive an exit for anything the
bot isn't already exiting — so a sell whose settlement frame was LOST (the 6-stuck-positions
incident) gets re-sold within EXIT_RECONCILE_SECONDS instead of stranding. Guards against
double-selling a position that's merely mid-settlement (recent-exit grace + in-flight check).
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from app.farm import exits as exits_mod
from app.farm.discovery import OnchainPosition
from app.farm.exits import reconcile_onchain_positions
from app.farm.schemas import ExitOrder, FarmState
from app.farm.volatility import blacklist_for_fill, is_blacklisted


@pytest.fixture
def redrives(monkeypatch):
    out: list = []

    async def fake_exit(client, state, token_id, size, cid, slug, outcome, **kw):
        out.append((cid, outcome, size))

    monkeypatch.setattr(exits_mod, "exit_position_leg", fake_exit)
    return out


def onchain(monkeypatch, positions):
    async def fake_fetch(http, addr):
        return positions

    monkeypatch.setattr(exits_mod, "fetch_open_positions", fake_fetch)


HELD = OnchainPosition(
    token_id="tok-x",
    condition_id="cid-x",
    outcome="YES",
    slug="m-x",
    size=Decimal("20"),
    avg_price=Decimal("0.5"),
)


async def test_redrives_untracked_onchain_position(farm_state: FarmState, monkeypatch, redrives):
    # The chain says we hold 20 shares the bot has no idea about (phantom-sold) → re-sell it.
    onchain(monkeypatch, [HELD])
    await reconcile_onchain_positions(MagicMock(), farm_state, MagicMock())
    assert redrives == [("cid-x", "YES", Decimal("20"))]


async def test_noop_when_chain_flat(farm_state: FarmState, monkeypatch, redrives):
    onchain(monkeypatch, [])
    await reconcile_onchain_positions(MagicMock(), farm_state, MagicMock())
    assert redrives == []


async def test_skips_recently_exited_leg(farm_state: FarmState, monkeypatch, redrives):
    # A sell was just placed for this leg → within the grace window, don't re-sell (it's settling).
    onchain(monkeypatch, [HELD])
    farm_state.recent_exits["cid-x:YES"] = datetime.now(timezone.utc)
    await reconcile_onchain_positions(MagicMock(), farm_state, MagicMock())
    assert redrives == [], "must not double-sell a position that's mid-settlement"


async def test_redrives_after_grace_expires(farm_state: FarmState, monkeypatch, redrives):
    # Old exit attempt (past the grace window) that clearly didn't settle → re-drive.
    onchain(monkeypatch, [HELD])
    farm_state.recent_exits["cid-x:YES"] = datetime.now(timezone.utc) - timedelta(seconds=600)
    await reconcile_onchain_positions(MagicMock(), farm_state, MagicMock())
    assert redrives == [("cid-x", "YES", Decimal("20"))]


async def test_skips_leg_with_exit_in_flight(farm_state: FarmState, monkeypatch, redrives):
    # The bot is already actively exiting this leg in its own state → let that flow finish.
    onchain(monkeypatch, [HELD])
    pos = farm_state.positions["market-A"].model_copy(deep=True)
    pos.market.condition_id = "cid-x"
    pos.exit_orders["working-oid"] = ExitOrder(outcome="YES", placed_at=datetime.now(timezone.utc))
    farm_state.positions["cid-x"] = pos
    await reconcile_onchain_positions(MagicMock(), farm_state, MagicMock())
    assert redrives == [], "must not double-drive a leg already being exited"


async def test_never_gives_up_keeps_retrying(farm_state: FarmState, monkeypatch, redrives):
    # Selling the stranded position is the objective, so the reconcile NEVER gives up. A balance:0
    # rejection is usually a transient CLOB cache-lag (the marco-rubio case) that clears in minutes,
    # so quitting early would strand real, sellable shares. Clearing the backoff window each cycle,
    # every cycle re-drives — no give-up cap.
    onchain(monkeypatch, [HELD])
    for _ in range(8):
        farm_state.recent_exits.pop("cid-x:YES", None)  # clear the backoff window so it fires
        await reconcile_onchain_positions(MagicMock(), farm_state, MagicMock())
    assert len(redrives) == 8, "must keep re-driving forever — never gives up"
    assert farm_state.reconcile_attempts["cid-x:YES"] == 8


async def test_backoff_suppresses_repeat_fires(farm_state: FarmState, monkeypatch, redrives):
    # Without clearing the window, the exponential backoff prevents re-firing every loop (so a
    # persistently-rejected leg doesn't churn rejected orders — the milan-32c problem).
    onchain(monkeypatch, [HELD])
    await reconcile_onchain_positions(MagicMock(), farm_state, MagicMock())  # attempt 1 fires
    await reconcile_onchain_positions(MagicMock(), farm_state, MagicMock())  # within backoff: skip
    await reconcile_onchain_positions(MagicMock(), farm_state, MagicMock())  # still backing off
    assert len(redrives) == 1, "backoff must suppress repeat fires within the window"
    assert farm_state.reconcile_attempts["cid-x:YES"] == 1


async def test_recovered_fill_blacklists_market(farm_state: FarmState, monkeypatch, redrives):
    # A held position the bot wasn't tracking = a fill whose user-WS frame we missed, so the normal
    # fill handler never blacklisted it. The reconcile must, else an instant-refilling market churns
    # open->fill->reconcile-sell->reopen forever (the trump-WC-final case), bleeding the spread.
    onchain(monkeypatch, [HELD])
    assert not is_blacklisted(farm_state, "cid-x")
    await reconcile_onchain_positions(MagicMock(), farm_state, MagicMock())
    assert is_blacklisted(farm_state, "cid-x"), "recovered untracked fill must blacklist"
    assert redrives == [("cid-x", "YES", Decimal("20"))], "and still re-drives the exit to sell"


async def test_recovered_fill_blacklists_only_once(farm_state: FarmState, monkeypatch, redrives):
    # Re-driving the same strand every cycle must not re-escalate the fill-strike tier.
    onchain(monkeypatch, [HELD])
    for _ in range(5):
        farm_state.recent_exits.pop("cid-x:YES", None)  # clear backoff so it re-drives each cycle
        await reconcile_onchain_positions(MagicMock(), farm_state, MagicMock())
    assert farm_state.health["cid-x"].fill_strikes == 1, "blacklist once on first detection only"


async def test_already_blacklisted_fill_not_re_escalated(
    farm_state: FarmState, monkeypatch, redrives
):
    # A normally-handled fill (already blacklisted) whose exit MINED frame was merely lost: the
    # reconcile re-drives the lingering exit but must NOT re-apply the blacklist — that would
    # double-count one fill into a longer tier.
    blacklist_for_fill(farm_state, "cid-x")
    assert farm_state.health["cid-x"].fill_strikes == 1
    onchain(monkeypatch, [HELD])
    await reconcile_onchain_positions(MagicMock(), farm_state, MagicMock())
    assert farm_state.health["cid-x"].fill_strikes == 1, "must not re-escalate existing blacklist"
    assert redrives == [("cid-x", "YES", Decimal("20"))], "but still re-drives the exit"


async def test_uses_tracked_cost_basis_not_stale_avgprice(farm_state: FarmState, monkeypatch):
    # Regression for the Padres +$7.60 phantom: the Data API returns avg_price=0 for a position
    # indexed seconds after a fresh fill. The reconcile must pass the bot's OWN tracked cost basis
    # to the exit; size * avg_price(0) = 0 would book full proceeds as a phantom roundtrip gain.
    captured: dict = {}

    async def fake_exit(client, state, token_id, size, cid, slug, outcome, **kw):
        captured["entry_cost"] = kw.get("entry_cost")

    monkeypatch.setattr(exits_mod, "exit_position_leg", fake_exit)
    stale = OnchainPosition(
        token_id="tok-x",
        condition_id="cid-x",
        outcome="Yes",  # Data API title-case (must still match the YES leg)
        slug="m-x",
        size=Decimal("20"),
        avg_price=Decimal("0"),  # stale: not yet indexed
    )
    onchain(monkeypatch, [stale])
    pos = farm_state.positions["market-A"].model_copy(deep=True)
    pos.market.condition_id = "cid-x"
    pos.yes_shares = Decimal("20")
    pos.yes_cost_basis = Decimal("7.80")  # what we actually paid
    pos.exit_orders.clear()
    farm_state.positions["cid-x"] = pos

    await reconcile_onchain_positions(MagicMock(), farm_state, MagicMock())

    assert captured["entry_cost"] == Decimal("7.80"), "must use tracked basis, not size*avg_price=0"


async def test_attempt_counter_resets_when_position_clears(
    farm_state: FarmState, monkeypatch, redrives
):
    # One attempt while held...
    onchain(monkeypatch, [HELD])
    await reconcile_onchain_positions(MagicMock(), farm_state, MagicMock())
    assert farm_state.reconcile_attempts.get("cid-x:YES") == 1
    # ...then milan-x is gone and a different position is held → its counter is pruned, so a fresh
    # occurrence later would start clean (and a resolved give-up wouldn't linger).
    other = OnchainPosition(
        token_id="tok-y",
        condition_id="cid-y",
        outcome="NO",
        slug="m-y",
        size=Decimal("20"),
        avg_price=Decimal("0.5"),
    )
    onchain(monkeypatch, [other])
    await reconcile_onchain_positions(MagicMock(), farm_state, MagicMock())
    assert "cid-x:YES" not in farm_state.reconcile_attempts
