"""The requote give-up pull is now gated and audited.

When a leg's pre-cancel fails REQUOTE_CANCEL_FAIL_PULL_THRESHOLD (3) times with the old order
reported live, requote_leg gives up and pulls the whole position. That pull now:
  1. marks a LONG re-quote cooldown (REQUOTE_GIVEUP_COOLDOWN_SECONDS = 600, via the new
     mark_guard_pulled(cooldown_s=...) param) BEFORE the batch cancel, so the market can't
     immediately re-enter into the same stuck book; and
  2. re-polls old_order_is_live once AFTER the (swallowed-outcome) batch cancel: confirmed-gone
     logs a quiet INFO, while still-live / unknown logs a WARNING naming the orphan oid and emits
     a `giveup_orphan` strat line (deposit wallets have no heartbeat deadman to reap it).

Driver copied from tests/requote/test_guard_idempotency.py::test_cancel_fail_cap_pulls_position:
preload `pos.yes_requote_cancel_fails = 2` so a single requote_leg call whose pre-cancel RAISES with
the order reported live reaches the cap. `old_order_is_live` is consulted TWICE on the give-up path
(the except-branch ladder, then the re-poll), so stubs return per-call values where needed.
"""

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.schemas import BestBidAsk
from app.constants import (
    CIRCUIT_BREAKER_MAX_FAILURES,
    REQUOTE_CANCEL_FAIL_PULL_THRESHOLD,
    REQUOTE_GIVEUP_COOLDOWN_SECONDS,
)
from app.farm import requote as requote_mod
from app.farm.gating import quote_block_reason
from app.farm.health import is_paused
from app.farm.requote import handle_bba, requote_leg
from app.farm.schemas import FarmState

# ── recorders / stubs (mirror test_guard_idempotency.py conventions) ───────────


@pytest.fixture
def cancels(monkeypatch):
    """Record cancel ids from BOTH the singular cancel_order and the batch cancel_orders."""
    out: list = []

    async def fake_cancel(client, oid):
        out.append(oid)

    async def fake_cancel_orders(client, *ids):
        out.extend(i for i in ids if i)

    monkeypatch.setattr(requote_mod, "cancel_order", fake_cancel)
    monkeypatch.setattr(requote_mod, "cancel_orders", fake_cancel_orders)
    return out


@pytest.fixture
def places(monkeypatch):
    """Record every place_limit_order call and return a fresh, deterministic order id."""
    out: list = []

    async def fake_place(client, order, post_only=True):
        out.append(order)
        return f"new-oid-{len(out)}"

    monkeypatch.setattr(requote_mod, "place_limit_order", fake_place)
    return out


@pytest.fixture
def raising_cancel(monkeypatch):
    """Make the singular pre-cancel (cancel_order) RAISE, exercising the give-up ladder. Applied
    after `cancels`, so cancel_order raises while the batch cancel_orders stays recorded."""

    async def _raise(client, oid):
        raise RuntimeError("cancel failed / unknown")

    monkeypatch.setattr(requote_mod, "cancel_order", _raise)


def stub_live(monkeypatch, result):
    """Pin old_order_is_live to a fixed result and record its calls."""
    calls: list = []

    async def stub(client, old_oid):
        calls.append(old_oid)
        return result

    monkeypatch.setattr(requote_mod, "old_order_is_live", stub)
    return calls


def stub_live_seq(monkeypatch, results):
    """old_order_is_live returns results[i] on the i-th call (last value repeats after exhaustion),
    so a test can drive the failure-ladder verdict then a DIFFERENT re-poll verdict."""
    calls: list = []
    seq = list(results)

    async def stub(client, old_oid):
        idx = len(calls)
        calls.append(old_oid)
        return seq[idx] if idx < len(seq) else seq[-1]

    monkeypatch.setattr(requote_mod, "old_order_is_live", stub)
    return calls


def stub_strat(monkeypatch):
    """Record every strat(...) emission from the requote module."""
    calls: list = []

    def fake_strat(event, **fields):
        calls.append((event, fields))

    monkeypatch.setattr(requote_mod, "strat", fake_strat)
    return calls


def stub_compute_quote(monkeypatch, bid: Decimal, ask: Decimal | None = None) -> None:
    ask = ask if ask is not None else bid + Decimal("0.02")

    def fake(market, midpoint, quote_depth="safe"):
        return (bid, ask)

    monkeypatch.setattr(requote_mod, "compute_quote", fake)


def make_bba(asset_id: str, best_bid: Decimal, best_ask: Decimal) -> BestBidAsk:
    return BestBidAsk(
        event_type="best_bid_ask",
        market="market-A",
        asset_id=asset_id,
        best_bid=best_bid,
        best_ask=best_ask,
        spread=best_ask - best_bid,
        timestamp="2026-07-04T12:00:00Z",
    )


def prime_giveup(pos) -> None:
    """Two prior live-confirmed fails preloaded → the next failed pre-cancel hits the cap."""
    assert REQUOTE_CANCEL_FAIL_PULL_THRESHOLD == 3, "driver assumes a threshold of 3"
    pos.yes_requote_cancel_fails = 2


# ── 1. the give-up pull starts the LONG cooldown ───────────────────────────────


async def test_giveup_pull_starts_long_cooldown(
    farm_state: FarmState, cancels, raising_cancel, monkeypatch
):
    pos = farm_state.positions["market-A"]
    prime_giveup(pos)
    stub_live(monkeypatch, True)
    before = datetime.now(timezone.utc)

    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.49"))

    assert pos.quotes_pulled is True, "the cap must pull the position"
    health = farm_state.health.get(pos.market.condition_id)
    assert health is not None and health.guard_pull_until is not None, "give-up must set a cooldown"
    # Pin the actual 600s window: guard_pull_until = mark_time + cooldown_s and mark_time > before,
    # so a correct give-up (cooldown_s=600) lands at/after before+600s, while a forgotten kwarg
    # (default 180s) lands ~before+180s and fails this — which `> before+180s` could not catch.
    assert health.guard_pull_until >= before + timedelta(
        seconds=REQUOTE_GIVEUP_COOLDOWN_SECONDS
    ), "the give-up pull must use the long REQUOTE_GIVEUP_COOLDOWN_SECONDS window"


# ── 2. the cooldown blocks a self-heal re-quote ────────────────────────────────


async def test_giveup_blocks_requote_self_heal(
    farm_state: FarmState, cancels, places, raising_cancel, monkeypatch
):
    pos = farm_state.positions["market-A"]
    cid = pos.market.condition_id
    prime_giveup(pos)
    stub_live(monkeypatch, True)

    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.49"))
    assert pos.quotes_pulled is True

    assert quote_block_reason(farm_state, cid, pos.market.event_slug) == "guard_cooldown", (
        "the give-up cooldown must block re-quoting"
    )

    # A threatened frame on the pulled leg must NOT self-heal a replacement while cooling.
    pos.yes_shares = Decimal("0")  # neutralize the mark-to-market kill for the driven frame
    pos.yes_cost_basis = Decimal("0")
    stub_compute_quote(monkeypatch, Decimal("0.31"))
    bba = make_bba(pos.market.yes_token_id, Decimal("0.32"), Decimal("0.34"))
    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba)

    assert places == [], "a market in give-up cooldown must not self-heal a replacement order"


# ── 3. still-live re-poll → orphan WARNING + giveup_orphan strat ───────────────


async def test_giveup_orphan_warning_when_still_live(
    farm_state: FarmState, cancels, raising_cancel, monkeypatch, caplog
):
    pos = farm_state.positions["market-A"]
    prime_giveup(pos)
    stub_live(monkeypatch, True)  # ladder verdict AND re-poll both live
    strat_calls = stub_strat(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="app.farm.requote"):
        await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.49"))

    orphan_warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and "STILL REST" in r.getMessage()
        and "yes-oid" in r.getMessage()
    ]
    assert len(orphan_warnings) == 1, "exactly one orphan WARNING naming the oid"
    giveup = [c for c in strat_calls if c[0] == "giveup_orphan"]
    assert len(giveup) == 1, "a still-live orphan must emit one giveup_orphan strat"
    assert giveup[0][1]["oid"] == "yes-oid"
    assert giveup[0][1]["live"] == "True"


# ── 4. confirmed-gone re-poll → quiet INFO, no orphan signal ───────────────────


async def test_giveup_orphan_confirmed_gone_is_quiet(
    farm_state: FarmState, cancels, raising_cancel, monkeypatch, caplog
):
    pos = farm_state.positions["market-A"]
    prime_giveup(pos)
    stub_live_seq(monkeypatch, [True, False])  # ladder live → give up; re-poll gone → quiet
    strat_calls = stub_strat(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.requote"):
        await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.49"))

    assert any(
        r.levelno == logging.INFO and "confirmed gone after batch cancel" in r.getMessage()
        for r in caplog.records
    ), "a confirmed-gone re-poll logs the quiet INFO"
    assert not any("STILL REST" in r.getMessage() for r in caplog.records), "no orphan WARNING"
    assert [c for c in strat_calls if c[0] == "giveup_orphan"] == [], "no giveup_orphan strat"


# ── 5. poll-failure re-poll (None) → orphan WARNING with live=None ─────────────


async def test_giveup_poll_failure_takes_warning_path(
    farm_state: FarmState, cancels, raising_cancel, monkeypatch, caplog
):
    pos = farm_state.positions["market-A"]
    prime_giveup(pos)
    stub_live_seq(monkeypatch, [True, None])  # ladder live → give up; re-poll unknown → warn
    strat_calls = stub_strat(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="app.farm.requote"):
        await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.49"))

    warnings = [
        r for r in caplog.records if r.levelno == logging.WARNING and "STILL REST" in r.getMessage()
    ]
    assert len(warnings) == 1, "an unknown re-poll must take the orphan warning path"
    giveup = [c for c in strat_calls if c[0] == "giveup_orphan"]
    assert len(giveup) == 1 and giveup[0][1]["live"] == "None", "orphan strat records live=None"


# ── 6. characterization (NOT revert-sensitive): the OTHER pull path gates via pause ──


async def test_place_fail_pull_gated_by_pause(farm_state: FarmState, cancels, monkeypatch):
    # This documents the sibling pull path, unchanged by the give-up gate: repeated place failures
    # trip the circuit breaker → mark_paused + cancel_position_orders, and quote_block_reason then
    # reports "paused". (Grep confirmed no existing test asserts this place-fail → paused chain;
    # test_place_failure_keeps_flag_set does a single failure and never trips the breaker.)
    async def ok_cancel(client, oid):
        return None

    async def raising_place(client, order, post_only=True):
        raise RuntimeError("CLOB rejected the place")

    monkeypatch.setattr(requote_mod, "cancel_order", ok_cancel)
    monkeypatch.setattr(requote_mod, "place_limit_order", raising_place)

    pos = farm_state.positions["market-A"]
    cid = pos.market.condition_id
    for _ in range(CIRCUIT_BREAKER_MAX_FAILURES):
        await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.49"))

    assert is_paused(farm_state, cid), "repeated place failures must trip the breaker → pause"
    assert quote_block_reason(farm_state, cid, pos.market.event_slug) == "paused"
