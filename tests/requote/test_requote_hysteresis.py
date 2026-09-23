"""Requote hysteresis (flip-flop damper).

Suppresses a requote when it would bounce our resting price by exactly 1 tick back to the
price we held just before our previous requote, while still inside a short window — damping a
book that is oscillating by a single tick so we do not churn cancel/replace pairs.

Two layers are tested:
  * ``is_flip_flop_requote`` — the pure predicate (1-tick? back to prev? inside window?).
  * ``handle_bba`` — the caller, which ALSO requires the current quote to be in-band before it
    honours the predicate, throttles the ``requote_hysteresis`` strat line to one per episode
    (``REQUOTE_HYSTERESIS_EPISODE_GAP_SECONDS``), and records prev-price on a real requote.

The D2 delta: the in-band check also requires ``not pos.quotes_pulled`` — a guard-pulled leg must
BYPASS the damper and requote to re-establish quotes (test_hysteresis_bypassed_when_quotes_pulled).

Every ``handle_bba`` test zeroes ``yes_shares``/``yes_cost_basis`` first (mirroring
tests/requote/test_crash_guards.py) so the mark-to-market kill switch (which the fixture would
otherwise trip) does not short-circuit the function before the hysteresis check.
"""

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.schemas import BestBidAsk
from app.constants import REQUOTE_HYSTERESIS_EPISODE_GAP_SECONDS, REQUOTE_HYSTERESIS_SECONDS
from app.farm import requote as requote_mod
from app.farm.requote import handle_bba, is_flip_flop_requote
from app.farm.schemas import FarmState

TICK = Decimal("0.01")
# Fixed reference instant for the pure-predicate cases (no event loop / real clock needed).
NOW = datetime(2026, 7, 3, 12, 0, 0, tzinfo=timezone.utc)
AT_EDGE = NOW + timedelta(seconds=REQUOTE_HYSTERESIS_SECONDS)  # exactly at the window boundary
EXPIRED = NOW + timedelta(seconds=REQUOTE_HYSTERESIS_SECONDS + 1)  # just past the window


# ── fixtures / helpers ────────────────────────────────────────────────────────


@pytest.fixture
def stub_network(monkeypatch):
    """Stub the network side of a requote so we can observe WHETHER requote_leg was invoked
    (mirrors tests/requote/test_crash_guards.py): requote_leg is replaced with a recorder, so it
    never actually runs — its own strat/logging side effects stay out of the way."""
    cancelled: list = []
    requotes: list = []

    async def fake_cancel(client, oid):
        cancelled.append(oid)

    async def fake_cancel_orders(client, *ids):
        cancelled.extend(i for i in ids if i)

    async def fake_requote_leg(client, state, ws, pos, outcome, price):
        requotes.append((outcome, float(price)))

    monkeypatch.setattr(requote_mod, "cancel_order", fake_cancel)
    monkeypatch.setattr(requote_mod, "cancel_orders", fake_cancel_orders)
    monkeypatch.setattr(requote_mod, "requote_leg", fake_requote_leg)
    return {"cancelled": cancelled, "requotes": requotes}


@pytest.fixture
def stub_place(monkeypatch):
    """Stub only place_limit_order/cancel_order so requote_leg runs END-TO-END and we can assert on
    the placed price and the recorded prev-price (mirrors tests/requote/test_requote_clamp.py)."""
    place_calls: list = []

    async def fake_place_limit_order(client, order, post_only=False):
        place_calls.append((order.token_id, order.side, order.size, order.price, post_only))
        return f"oid-{len(place_calls)}"

    async def fake_cancel_order(client, oid):
        return None

    monkeypatch.setattr(requote_mod, "place_limit_order", fake_place_limit_order)
    monkeypatch.setattr(requote_mod, "cancel_order", fake_cancel_order)
    return place_calls


@pytest.fixture
def strat_calls(monkeypatch):
    """Record every strat(...) emission so we can count the throttled hysteresis line."""
    calls: list = []

    def fake_strat(event, **fields):
        calls.append((event, fields))

    monkeypatch.setattr(requote_mod, "strat", fake_strat)
    return calls


def hysteresis_calls(calls: list) -> list:
    return [c for c in calls if c[0] == "requote_hysteresis"]


def stub_compute_quote(monkeypatch, bid: Decimal, ask: Decimal | None = None) -> None:
    """Force compute_quote to return a specific (bid, ask) so handle_bba's new target price is
    deterministic regardless of the market's real quote params."""
    ask = ask if ask is not None else bid + Decimal("0.02")

    def fake_compute_quote(market, midpoint, quote_depth="safe"):
        return (bid, ask)

    monkeypatch.setattr(requote_mod, "compute_quote", fake_compute_quote)


def make_bba(asset_id: str, best_bid: Decimal, best_ask: Decimal) -> BestBidAsk:
    return BestBidAsk(
        event_type="best_bid_ask",
        market="market-A",
        asset_id=asset_id,
        best_bid=best_bid,
        best_ask=best_ask,
        spread=best_ask - best_bid,
        timestamp="2026-07-03T12:00:00Z",
    )


def neutralize_kill(pos) -> None:
    """Zero the held YES inventory so the mark-to-market kill switch can't trip and pre-empt the
    hysteresis code path (see tests/requote/test_crash_guards.py lines 66-67)."""
    pos.yes_shares = Decimal("0")
    pos.yes_cost_basis = Decimal("0")


# ── handle_bba: suppression path ──────────────────────────────────────────────


async def test_flip_flop_1tick_suppressed(
    farm_state: FarmState, stub_network, strat_calls, monkeypatch, caplog
):
    # our_px 0.32, target 0.31 == prev_price 0.31, prev_at fresh → 1-tick flip-flop, in window.
    stub_compute_quote(monkeypatch, bid=Decimal("0.31"))
    pos = farm_state.positions["market-A"]
    neutralize_kill(pos)
    pos.yes_price = Decimal("0.32")
    pos.yes_prev_price = Decimal("0.31")
    pos.yes_prev_price_at = datetime.now(timezone.utc)
    # threatened (best_bid == our_px), in-zone (|mid 0.33 - 0.32| = 0.01 <= 0.03) → reaches gate.
    bba = make_bba(pos.market.yes_token_id, Decimal("0.32"), Decimal("0.34"))

    with caplog.at_level(logging.DEBUG, logger="app.farm.requote"):
        await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba)

    assert stub_network["requotes"] == [], "a 1-tick flip-flop must be suppressed, not requoted"
    assert len(hysteresis_calls(strat_calls)) == 1, "one requote_hysteresis strat line expected"
    assert "requote hysteresis skip" in caplog.text, "debug skip line must be emitted"


async def test_two_tick_move_not_suppressed(farm_state: FarmState, stub_network, monkeypatch):
    # our_px 0.33, target 0.31 → 2-tick move: the predicate rejects it regardless of the window.
    stub_compute_quote(monkeypatch, bid=Decimal("0.31"))
    pos = farm_state.positions["market-A"]
    neutralize_kill(pos)
    pos.yes_price = Decimal("0.33")
    pos.yes_prev_price = Decimal("0.31")
    pos.yes_prev_price_at = datetime.now(timezone.utc)
    # best_bid 0.31 <= our_px + 0.02 → threatened; mid 0.335 in-zone.
    bba = make_bba(pos.market.yes_token_id, Decimal("0.31"), Decimal("0.36"))

    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba)

    assert stub_network["requotes"] == [("YES", 0.31)], "a 2-tick move must requote normally"


async def test_out_of_band_current_not_suppressed(farm_state: FarmState, stub_network, monkeypatch):
    # The move IS a 1-tick flip-flop back to prev, but our current quote is OUT of band
    # (mid 0.25, |0.25 - 0.32| = 0.07 > 0.03) → the caller's in-band guard forbids suppression.
    stub_compute_quote(monkeypatch, bid=Decimal("0.31"))
    pos = farm_state.positions["market-A"]
    neutralize_kill(pos)
    pos.yes_price = Decimal("0.32")
    pos.yes_prev_price = Decimal("0.31")
    pos.yes_prev_price_at = datetime.now(timezone.utc)
    bba = make_bba(pos.market.yes_token_id, Decimal("0.10"), Decimal("0.40"))

    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba)

    assert stub_network["requotes"] == [("YES", 0.31)], "out-of-band current quote must requote"


async def test_episode_strat_fires_once(
    farm_state: FarmState, stub_network, strat_calls, monkeypatch
):
    # Two identical suppressed frames back-to-back: the strat line is throttled to one per episode.
    stub_compute_quote(monkeypatch, bid=Decimal("0.31"))
    pos = farm_state.positions["market-A"]
    neutralize_kill(pos)
    pos.yes_price = Decimal("0.32")
    pos.yes_prev_price = Decimal("0.31")
    pos.yes_prev_price_at = datetime.now(timezone.utc)
    bba = make_bba(pos.market.yes_token_id, Decimal("0.32"), Decimal("0.34"))

    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba)
    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba)

    assert stub_network["requotes"] == [], "both frames must be suppressed"
    assert len(hysteresis_calls(strat_calls)) == 1, "strat must fire once for the whole episode"


async def test_episode_reopens_after_gap(
    farm_state: FarmState, stub_network, strat_calls, monkeypatch
):
    # After the episode-gap window elapses, the next suppression logs a fresh strat line.
    stub_compute_quote(monkeypatch, bid=Decimal("0.31"))
    pos = farm_state.positions["market-A"]
    neutralize_kill(pos)
    pos.yes_price = Decimal("0.32")
    pos.yes_prev_price = Decimal("0.31")
    pos.yes_prev_price_at = datetime.now(timezone.utc)
    bba = make_bba(pos.market.yes_token_id, Decimal("0.32"), Decimal("0.34"))

    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba)
    assert len(hysteresis_calls(strat_calls)) == 1

    # Backdate the episode marker past the gap so the next suppression counts as a new episode.
    pos.yes_hysteresis_episode_at = datetime.now(timezone.utc) - timedelta(
        seconds=REQUOTE_HYSTERESIS_EPISODE_GAP_SECONDS + 1
    )
    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba)

    assert stub_network["requotes"] == [], "the second frame is still suppressed"
    assert len(hysteresis_calls(strat_calls)) == 2, "a fresh episode logs a second strat line"


async def test_window_expiry_resumes_requote(farm_state: FarmState, stub_network, monkeypatch):
    # Same 1-tick flip-flop as test 1, but prev_at is older than the hysteresis window → resume.
    stub_compute_quote(monkeypatch, bid=Decimal("0.31"))
    pos = farm_state.positions["market-A"]
    neutralize_kill(pos)
    pos.yes_price = Decimal("0.32")
    pos.yes_prev_price = Decimal("0.31")
    pos.yes_prev_price_at = datetime.now(timezone.utc) - timedelta(
        seconds=REQUOTE_HYSTERESIS_SECONDS + 1
    )
    bba = make_bba(pos.market.yes_token_id, Decimal("0.32"), Decimal("0.34"))

    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba)

    assert stub_network["requotes"] == [("YES", 0.31)], "an expired window must requote normally"


async def test_no_leg_flip_flop_suppressed(
    farm_state: FarmState, stub_network, strat_calls, monkeypatch
):
    # NO-leg mirror of test 1: our_px 0.66, target 0.65 == no_prev_price, in window → suppressed.
    stub_compute_quote(monkeypatch, bid=Decimal("0.65"))
    pos = farm_state.positions["market-A"]
    neutralize_kill(pos)
    pos.no_price = Decimal("0.66")
    pos.no_prev_price = Decimal("0.65")
    pos.no_prev_price_at = datetime.now(timezone.utc)
    bba = make_bba(pos.market.no_token_id, Decimal("0.66"), Decimal("0.68"))

    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba)

    assert stub_network["requotes"] == [], "a 1-tick flip-flop on the NO leg must be suppressed"
    assert len(hysteresis_calls(strat_calls)) == 1


async def test_hysteresis_bypassed_when_quotes_pulled(
    farm_state: FarmState, stub_network, monkeypatch
):
    # The D2 delta: identical to test_flip_flop_1tick_suppressed, but the position was guard-pulled.
    # `current_in_band` is False (the `not pos.quotes_pulled` clause), so the damper is bypassed and
    # the leg requotes to re-establish quotes instead of being suppressed into silence.
    stub_compute_quote(monkeypatch, bid=Decimal("0.31"))
    pos = farm_state.positions["market-A"]
    neutralize_kill(pos)
    pos.yes_price = Decimal("0.32")
    pos.yes_prev_price = Decimal("0.31")
    pos.yes_prev_price_at = datetime.now(timezone.utc)
    pos.quotes_pulled = True
    bba = make_bba(pos.market.yes_token_id, Decimal("0.32"), Decimal("0.34"))

    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba)

    assert stub_network["requotes"] == [("YES", 0.31)], (
        "a guard-pulled leg must bypass hysteresis and requote"
    )


# ── requote_leg: prev-price bookkeeping ───────────────────────────────────────


async def test_prev_price_recorded_on_requote(farm_state: FarmState, stub_place, monkeypatch):
    # A real (non-suppressed) requote must record the OLD price as prev_price with a timestamp,
    # so a later flip-flop back to it can be detected. Use a 2-tick move with no prior prev_price
    # so hysteresis cannot suppress this one.
    stub_compute_quote(monkeypatch, bid=Decimal("0.28"))
    pos = farm_state.positions["market-A"]
    neutralize_kill(pos)
    pos.yes_price = Decimal("0.30")  # yes_prev_price left as its default None → predicate is False
    bba = make_bba(pos.market.yes_token_id, Decimal("0.30"), Decimal("0.34"))

    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba)

    assert stub_place, "the requote must actually place an order"
    _, _, _, placed_price, _ = stub_place[0]
    assert placed_price == 0.28, "placed at the 2-tick-away target"
    assert pos.yes_price == Decimal("0.28"), "resting price advanced to the new target"
    assert pos.yes_prev_price == Decimal("0.30"), "prev_price records the price we just left"
    assert pos.yes_prev_price_at is not None, "prev_price timestamp must be stamped"


# ── is_flip_flop_requote: pure predicate ──────────────────────────────────────


@pytest.mark.parametrize(
    "our_price, new_price, prev_price, prev_at, now, expected",
    [
        # No prior price at all → cannot be a flip-flop.
        (Decimal("0.32"), Decimal("0.31"), None, NOW, NOW, False),
        # Prior price present but no timestamp → cannot judge the window → False.
        (Decimal("0.32"), Decimal("0.31"), Decimal("0.31"), None, NOW, False),
        # Exact 1-tick move back to prev, inside window → True.
        (Decimal("0.32"), Decimal("0.31"), Decimal("0.31"), NOW, NOW, True),
        # Exactly at the window boundary (elapsed == window_seconds) → still True (<=).
        (Decimal("0.32"), Decimal("0.31"), Decimal("0.31"), NOW, AT_EDGE, True),
        # 2-tick move → not a single-tick flip-flop.
        (Decimal("0.33"), Decimal("0.31"), Decimal("0.31"), NOW, NOW, False),
        # Window expired → False even though it is a 1-tick move back to prev.
        (Decimal("0.32"), Decimal("0.31"), Decimal("0.31"), NOW, EXPIRED, False),
        # 1-tick move but the target is not the price we just left → False.
        (Decimal("0.32"), Decimal("0.31"), Decimal("0.30"), NOW, NOW, False),
    ],
    ids=[
        "prev_price_none",
        "prev_at_none",
        "one_tick_in_window",
        "one_tick_at_window_edge",
        "two_tick_move",
        "window_expired",
        "one_tick_but_not_back_to_prev",
    ],
)
def test_is_flip_flop_requote_predicate(our_price, new_price, prev_price, prev_at, now, expected):
    assert (
        is_flip_flop_requote(
            our_price, new_price, prev_price, prev_at, TICK, now, REQUOTE_HYSTERESIS_SECONDS
        )
        is expected
    )
