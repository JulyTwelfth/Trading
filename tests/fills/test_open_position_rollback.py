"""Bug 6 (POL-31): if one leg of the open_position gather fails, the survivor
must be cancelled so we never end up half-hedged."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.farm import worker as worker_mod
from app.farm.schemas import FarmState, Market
from app.farm.worker import open_position


def stub_place_and_cancel(
    monkeypatch,
    *,
    yes_outcome: str,
    no_outcome: str,
    yes_oid: str = "yes-oid",
    no_oid: str = "no-oid",
):
    """Wire stubs for place_limit_order and cancel_order.

    yes_outcome / no_outcome: "success" or "fail" — controls per-leg behavior so
    we can drive each rollback scenario.
    Returns dict with `place_calls` and `cancel_calls` for assertions.
    """
    place_calls: list = []
    cancel_calls: list = []

    async def fake_place(client, order, post_only=False):
        place_calls.append(order.token_id)
        if order.token_id == "tok-yes":
            if yes_outcome == "fail":
                raise RuntimeError("yes leg failed")
            return yes_oid
        else:
            if no_outcome == "fail":
                raise RuntimeError("no leg failed")
            return no_oid

    async def fake_cancel(client, oid):
        cancel_calls.append(oid)

    monkeypatch.setattr(worker_mod, "place_limit_order", fake_place)
    monkeypatch.setattr(worker_mod, "cancel_order", fake_cancel)
    return {"place": place_calls, "cancel": cancel_calls}


async def test_open_position_cancels_yes_when_no_fails(
    farm_state: FarmState, market: Market, monkeypatch
):
    farm_state.positions.clear()
    midpoints = {market.yes_token_id: Decimal("0.5"), market.no_token_id: Decimal("0.5")}
    calls = stub_place_and_cancel(monkeypatch, yes_outcome="success", no_outcome="fail")

    await open_position(MagicMock(), farm_state, AsyncMock(), market, midpoints)

    # Survivor (YES) must be cancelled to maintain hedged invariant.
    assert calls["cancel"] == ["yes-oid"]
    # Position must NOT be saved — we're rolled back.
    assert market.condition_id not in farm_state.positions


async def test_open_position_cancels_no_when_yes_fails(
    farm_state: FarmState, market: Market, monkeypatch
):
    farm_state.positions.clear()
    midpoints = {market.yes_token_id: Decimal("0.5"), market.no_token_id: Decimal("0.5")}
    calls = stub_place_and_cancel(monkeypatch, yes_outcome="fail", no_outcome="success")

    await open_position(MagicMock(), farm_state, AsyncMock(), market, midpoints)

    assert calls["cancel"] == ["no-oid"]
    assert market.condition_id not in farm_state.positions


async def test_open_position_does_not_cancel_when_both_fail(
    farm_state: FarmState, market: Market, monkeypatch
):
    """Both legs fail → no cancel, no position.

    CLASSIFICATION: ⊘ inherently tautological with respect to the gather fix.

    The fix's observable effect is: cancel the surviving leg of a partial failure.
    When BOTH legs fail there is no survivor, so the cancel path is never entered
    in either the pre-fix or post-fix code.  Three rewrite attempts (plain
    assertions, cancel_explodes stub, asyncio.sleep interleaving) all pass on
    both sides of the revert because the end-state is structurally identical.

    The test is retained as a regression guard: if a future change accidentally
    adds cancel logic to the both-fail branch, the cancel_explodes stub will
    catch it immediately.
    """
    import asyncio as _asyncio

    farm_state.positions.clear()
    midpoints = {market.yes_token_id: Decimal("0.5"), market.no_token_id: Decimal("0.5")}

    completed: list = []

    async def fake_place(client, order, post_only=False):
        await _asyncio.sleep(0)
        completed.append(order.token_id)
        raise RuntimeError("simulated EAGAIN")

    async def cancel_explodes(client, oid):
        raise AssertionError(f"cancel_order must NOT be called in both-fail path, got oid={oid!r}")

    monkeypatch.setattr(worker_mod, "place_limit_order", fake_place)
    monkeypatch.setattr(worker_mod, "cancel_order", cancel_explodes)

    await open_position(MagicMock(), farm_state, AsyncMock(), market, midpoints)

    assert market.condition_id not in farm_state.positions
    assert set(completed) == {"tok-yes", "tok-no"}, "both legs must be attempted"


async def test_open_position_saves_position_when_both_succeed(
    farm_state: FarmState, market: Market, monkeypatch
):
    """Happy path: both legs succeed → position saved with correct oids, no cancel.

    CLASSIFICATION: ⊘ inherently tautological with respect to the gather fix.

    The gather refactor adds a `return_exceptions=True` flag and an error-inspection
    block BEFORE the success path.  The success path itself (`yes_oid, no_oid = ...;
    state.positions[...] = MarketPosition(...)`) is structurally unchanged.  Both
    pre-fix and post-fix code produce exactly the same observable state when both
    legs succeed.  Three rewrite attempts all passed on both sides of the revert.

    The test is retained as a regression guard: if the gather refactor accidentally
    breaks happy-path placement (e.g., swaps yes/no oids, adds a spurious cancel
    call, or fails to save the position), this test will catch it.  The
    cancel_explodes stub makes any erroneous rollback call immediately fatal.
    """
    farm_state.positions.clear()
    midpoints = {market.yes_token_id: Decimal("0.5"), market.no_token_id: Decimal("0.5")}

    YES_OID = "yes-sentinel-abc"
    NO_OID = "no-sentinel-xyz"

    async def fake_place(client, order, post_only=False):
        if order.token_id == market.yes_token_id:
            return YES_OID
        return NO_OID

    async def cancel_explodes(client, oid):
        raise AssertionError(
            f"cancel_order must NOT be called in the success path, got oid={oid!r}"
        )

    monkeypatch.setattr(worker_mod, "place_limit_order", fake_place)
    monkeypatch.setattr(worker_mod, "cancel_order", cancel_explodes)

    await open_position(MagicMock(), farm_state, AsyncMock(), market, midpoints)

    pos = farm_state.positions[market.condition_id]
    assert pos.yes_order_id == YES_OID, f"yes leg oid mismatch: got {pos.yes_order_id!r}"
    assert pos.no_order_id == NO_OID, f"no leg oid mismatch: got {pos.no_order_id!r}"
