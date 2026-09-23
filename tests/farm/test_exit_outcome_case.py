"""Fix A — outcome-case normalization across the exit-tracking paths.

The Data API reports outcomes title-case ("Yes"/"No"); every runtime path canonicalises to
"YES"/"NO" (see app/bot/schemas.normalize_outcome). Before the fix the recent-exit grace, the
in-flight-exit check, and the reconcile-attempt bookkeeping keyed on the raw string, so a
title-case "Yes" from the chain never matched a "YES" the bot stored — the grace/in-flight guards
were dead code and clear_held_leg("Yes") cleared the WRONG (NO) leg.

Harness mirrors tests/farm/test_onchain_reconcile.py (OnchainPosition fixtures, fake_fetch via
`onchain`, and a `redrives` recorder in place of exit_position_leg). Nothing imports the new
`exit_key` helper on purpose: the reconcile/clear/register/fetch functions all exist on HEAD, so a
revert of the fix reds these via ASSERTION rather than a collection error.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.farm import exits as exits_mod
from app.farm.discovery import OnchainPosition, fetch_open_positions
from app.farm.exits import clear_held_leg, reconcile_onchain_positions, register_exit
from app.farm.schemas import ExitOrder, FarmState


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


def held(outcome: str = "Yes") -> OnchainPosition:
    # Title-case, exactly as the Data API reports it.
    return OnchainPosition(
        token_id="tok-x",
        condition_id="cid-x",
        outcome=outcome,
        slug="m-x",
        size=Decimal("20"),
        avg_price=Decimal("0.5"),
    )


async def test_recent_exit_grace_respected_across_case(
    farm_state: FarmState, monkeypatch, redrives
):
    # A sell was registered runtime-canonically as "YES"; the chain then reports title-case "Yes".
    # The recent-exit grace must still recognise them as the same leg and NOT re-drive a duplicate.
    register_exit(farm_state, "cid-x", "exit-oid", Decimal("20"), "m-x", Decimal("10"), "YES")
    onchain(monkeypatch, [held("Yes")])

    await reconcile_onchain_positions(MagicMock(), farm_state, MagicMock())

    assert redrives == [], "recent-exit grace must survive a Yes/YES case mismatch"


async def test_in_flight_exit_respected_across_case(farm_state: FarmState, monkeypatch, redrives):
    # The bot is already exiting the YES leg (tracked in-state as "YES"); the chain reports "Yes".
    # The in-flight check must match across case and let the running exit finish, not re-drive.
    pos = farm_state.positions["market-A"].model_copy(deep=True)
    pos.market.condition_id = "cid-x"
    pos.exit_orders["working-oid"] = ExitOrder(outcome="YES", placed_at=datetime.now(timezone.utc))
    farm_state.positions["cid-x"] = pos
    onchain(monkeypatch, [held("Yes")])

    await reconcile_onchain_positions(MagicMock(), farm_state, MagicMock())

    assert redrives == [], "in-flight exit must survive a Yes/YES case mismatch"


async def test_clear_held_leg_title_case_clears_correct_leg(farm_state: FarmState):
    # clear_held_leg with a title-case "Yes" must zero the YES leg and leave NO untouched.
    # The pre-fix code (no .upper()) fell to the else branch and cleared the NO leg instead.
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("20")
    pos.yes_cost_basis = Decimal("10")
    pos.no_shares = Decimal("30")
    pos.no_cost_basis = Decimal("15")

    clear_held_leg(farm_state, "market-A", "Yes")

    assert pos.yes_shares == Decimal("0"), "the YES leg must be cleared"
    assert pos.yes_cost_basis == Decimal("0")
    assert pos.no_shares == Decimal("30"), "the NO leg must be left untouched"
    assert pos.no_cost_basis == Decimal("15")


async def test_fetch_open_positions_normalizes_outcome(monkeypatch):
    # The discovery boundary must upper-case the title-case outcome the Data API sends, so the rest
    # of the system only ever sees the canonical "YES"/"NO".
    payload = [
        {
            "asset": "tok-x",
            "conditionId": "cid-x",
            "outcome": "Yes",
            "slug": "m-x",
            "size": "20",
            "avgPrice": "0.5",
        }
    ]
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)
    http = MagicMock()
    http.get = AsyncMock(return_value=resp)

    result = await fetch_open_positions(http, "0xwallet")

    assert len(result) == 1
    assert result[0].outcome == "YES", "data-api title-case must be normalized to canonical YES"


async def test_reconcile_attempts_pruned_across_case(farm_state: FarmState, monkeypatch, redrives):
    # The prune at the top of every reconcile drops any attempt whose key isn't in held_keys. Both
    # the stored key and held_keys must canonicalise to "YES" so a "Yes" chain position's attempt
    # accrues under the canonical key and SURVIVES the prune on the next cycle (rather than being
    # dropped, or re-fragmented under a differently-cased key).
    onchain(monkeypatch, [held("Yes")])

    await reconcile_onchain_positions(MagicMock(), farm_state, MagicMock())
    assert farm_state.reconcile_attempts == {"cid-x:YES": 1}, "stored under the canonical key"

    # Second cycle: the backoff suppresses a re-fire, but the prune still runs — the key must stay.
    await reconcile_onchain_positions(MagicMock(), farm_state, MagicMock())
    assert farm_state.reconcile_attempts == {"cid-x:YES": 1}, (
        "the attempt key must survive the held_keys prune across a Yes/YES case mismatch"
    )
