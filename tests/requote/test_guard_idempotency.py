"""Tier 1 — per-position idempotency flag on the loss guards + a killed-gate on the non-bba
market-event handlers.

The overnight regression: a position whose legs were already pulled by a loss guard kept
re-issuing the same cancel pair on every subsequent bad frame (~60 frames/s), spamming the CLOB
with cancels for orders that no longer exist. `MarketPosition.quotes_pulled` is now the single
latch: `cancel_position_orders` sets it True at the one chokepoint all guard pulls funnel through,
each guard skips its cancel when it's already set, and a successful `requote_leg` clears it
(re-arming the guards). The requote path itself stays reachable when pulled so a pulled position
can still self-heal by placing a fresh order.

Separately, `handle_book_snapshot` / `handle_price_change` / `handle_tick_size_change` now early-
return on `state.killed` (handle_bba already did, and is where the kill is raised).

Mirrors the stub/monkeypatch conventions in test_crash_guards.py and test_live_depth_guard.py.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.schemas import BestBidAsk, BookSnapshot, PriceChange, TickSizeChange
from app.constants import REQUOTE_CANCEL_FAIL_PULL_THRESHOLD
from app.farm import requote as requote_mod
from app.farm.requote import (
    cancel_position_orders,
    depth_fill_loss_guard,
    handle_bba,
    handle_book_snapshot,
    handle_price_change,
    handle_tick_size_change,
    requote_leg,
)
from app.farm.schemas import FarmState, LiveBook

# ── recorders ─────────────────────────────────────────────────────────────────


@pytest.fixture
def cancels(monkeypatch):
    """Record every cancelled order id, from BOTH paths: requote_leg's singular cancel_order and
    cancel_position_orders' Tier-2 batch cancel_orders. Funnelling both into one list keeps the
    flat-id-set assertions ({yes_oid, no_oid}) valid after the batch migration. Tests that need to
    count batch round-trips use the `batch_cancels` fixture instead."""
    out: list = []

    async def fake_cancel(client, oid):
        out.append(oid)

    async def fake_cancel_orders(client, *ids):
        out.extend(i for i in ids if i)

    monkeypatch.setattr(requote_mod, "cancel_order", fake_cancel)
    monkeypatch.setattr(requote_mod, "cancel_orders", fake_cancel_orders)
    return out


@pytest.fixture
def batch_cancels(monkeypatch):
    """Record each cancel_position_orders batch round-trip as the tuple of ids it carried, so the
    idempotency-count tests can assert 'exactly ONE batch call on the first pull, zero on the
    repeat frame'. Does NOT patch the singular cancel_order (requote_leg's path)."""
    calls: list[tuple] = []

    async def fake_cancel_orders(client, *ids):
        calls.append(tuple(ids))

    monkeypatch.setattr(requote_mod, "cancel_orders", fake_cancel_orders)
    return calls


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
def requotes(monkeypatch):
    """Record requote_leg calls without running the real one (for handler-level isolation)."""
    out: list = []

    async def fake_requote_leg(client, state, ws, pos, outcome, price):
        out.append((outcome, Decimal(str(price))))

    monkeypatch.setattr(requote_mod, "requote_leg", fake_requote_leg)
    return out


def make_bba(asset_id: str, best_bid: Decimal, best_ask: Decimal) -> BestBidAsk:
    return BestBidAsk(
        event_type="best_bid_ask",
        market="market-A",
        asset_id=asset_id,
        best_bid=best_bid,
        best_ask=best_ask,
        spread=best_ask - best_bid,
        timestamp="2026-06-20T12:00:00Z",
    )


def make_book(asset_id: str, bids: list[tuple[str, str]]) -> BookSnapshot:
    return BookSnapshot(
        event_type="book",
        market="market-A",
        asset_id=asset_id,
        bids=[{"price": p, "size": s} for p, s in bids],
        asks=[],
        timestamp="1",
        hash="h",
    )


# ── 1. depth guard pulls once, then skips the redundant cancel ─────────────────


async def test_depth_guard_pulls_once_then_skips_cancel(farm_state: FarmState, batch_cancels):
    """The core overnight-spam regression: a thin book that trips immediate_sell_loss must pull
    both legs in ONE batch round-trip on the first bad frame and then issue ZERO further batch
    calls on an identical bad frame, with quotes_pulled latched True.

    (Tier 2: the pull is now a single cancel_orders(yes,no) batch rather than two cancel_order
    calls, so we count batch round-trips via `batch_cancels` instead of flat ids.)"""
    farm_state.config.filters.max_fill_loss = Decimal("1")
    # size_per_market = 100, YES @ 0.50; only 10 resting @ 0.50 → sell 100 dumps 90 into nothing →
    # loss = 100*0.50 - 10*0.50 = $45 >> $1 cap → pull.
    farm_state.live_books["tok-yes"] = LiveBook(bids={Decimal("0.50"): Decimal("10")})
    pos = farm_state.positions["market-A"]

    yes_oid, no_oid = pos.yes_order_id, pos.no_order_id
    pulled_1 = await depth_fill_loss_guard(MagicMock(), farm_state, AsyncMock(), pos, "tok-yes")
    assert pulled_1 is True
    assert len(batch_cancels) == 1, "the first pull is exactly one batch round-trip"
    assert set(batch_cancels[0]) == {yes_oid, no_oid}, "carrying both leg ids"
    assert pos.quotes_pulled is True
    assert pos.yes_order_id == "" and pos.no_order_id == "", "pulled ids are cleared"

    # Second identical bad frame: already pulled → early return, NO new batch call.
    pulled_2 = await depth_fill_loss_guard(MagicMock(), farm_state, AsyncMock(), pos, "tok-yes")
    assert pulled_2 is False
    assert len(batch_cancels) == 1, "an already-pulled position must not re-issue the batch cancel"
    assert pos.quotes_pulled is True


async def test_depth_guard_skips_via_handler_path_too(farm_state: FarmState, batch_cancels):
    """Same idempotency, but driven through handle_book_snapshot (the real entry point) twice on
    the same thin book frame: exactly one batch round-trip total."""
    farm_state.config.filters.max_fill_loss = Decimal("1")
    pos = farm_state.positions["market-A"]
    frame = make_book("tok-yes", [("0.50", "10")])
    yes_oid, no_oid = pos.yes_order_id, pos.no_order_id

    await handle_book_snapshot(MagicMock(), farm_state, AsyncMock(), pos, frame)
    assert len(batch_cancels) == 1
    assert set(batch_cancels[0]) == {yes_oid, no_oid}
    assert pos.quotes_pulled is True

    await handle_book_snapshot(MagicMock(), farm_state, AsyncMock(), pos, frame)
    assert len(batch_cancels) == 1, "second snapshot of the same thin book must not re-cancel"


# ── 2. self-heal: a pulled position requotes, which re-arms the guards ─────────


async def test_self_heal_requotes_and_rearms(farm_state: FarmState, cancels, places, monkeypatch):
    """The critical self-heal property. A pulled position fed a RECOVERED bba frame runs the REAL
    requote_leg → places a fresh order, clears quotes_pulled, updates the leg id. Then a STILL-BAD
    frame must pull AGAIN (the guard was re-armed by the successful requote)."""
    farm_state.config.filters.max_fill_loss = Decimal("1")
    pos = farm_state.positions["market-A"]
    # Use the NO leg so the YES-only volatility pull (record_price_sample) can't interfere.
    no = pos.market.no_token_id
    pos.no_price = Decimal("0.50")
    pos.quotes_pulled = True  # already pulled by an earlier guard

    # Recovered frame: best_bid 0.50, best_ask 0.52 → midpoint 0.51.
    #   gap-pull: no prior no_best_bid → skipped.
    #   exit-loss: top_of_book_fill_loss(our 0.50, bid 0.50) = 0, not > 1 → no pull.
    #   threatened (0.50-0.50=0 <= 2 ticks) → requote; compute_quote(0.51,safe)[0] = 0.49.
    await handle_bba(
        MagicMock(), farm_state, AsyncMock(), pos, make_bba(no, Decimal("0.50"), Decimal("0.52"))
    )

    assert len(places) == 1, "a recovered frame on a pulled leg must place a replacement"
    assert pos.quotes_pulled is False, "a successful requote must re-arm the guards"
    assert pos.no_order_id == "new-oid-1", "the leg id must be updated to the replacement"
    assert pos.no_price == Decimal("0.49")
    assert cancels == ["no-oid"], "requote pre-cancel hits the old leg id exactly once"

    cancels_before_repull = list(cancels)
    # Now a STILL-BAD frame on the re-armed leg. Keep the relative drop small (0.50→0.47 = 6% < the
    # 10% gap-pull floor) so we isolate the EXIT-LOSS guard re-arm, not the gap-pull: our new px is
    # 0.49, so top_of_book_fill_loss = 100*(0.49-0.47) = $2 > $1 cap AND not pulled → pull again.
    await handle_bba(
        MagicMock(), farm_state, AsyncMock(), pos, make_bba(no, Decimal("0.47"), Decimal("0.52"))
    )

    assert pos.quotes_pulled is True, (
        "re-armed exit-loss guard must pull again on a fresh bad frame"
    )
    # The re-pull's cancel_position_orders cancels both live legs: the freshly-placed NO replacement
    # (new-oid-1) and the still-resting YES leg (yes-oid, never requoted in this test).
    assert set(cancels) - set(cancels_before_repull) == {"new-oid-1", "yes-oid"}, (
        "the re-pull must cancel both currently-live legs"
    )
    assert len(places) == 1, "the re-pull must not place anything new"


# ── 3. pulled + still bad → no requote, no cancel spam ─────────────────────────


async def test_pulled_still_bad_no_requote_no_spam(farm_state: FarmState, cancels, places):
    """quotes_pulled=True + a still-in-loss bba frame: ZERO cancels (the exit-loss guard is gated)
    AND zero places (the requote path isn't reached because the loss-cap return fires first).
    The flag stays True."""
    farm_state.config.filters.max_fill_loss = Decimal("1")
    pos = farm_state.positions["market-A"]
    no = pos.market.no_token_id
    pos.no_price = Decimal("0.50")
    pos.quotes_pulled = True

    # best_bid 0.40 → exit-loss est = 100*(0.50-0.40)=$10 > $1, but quotes_pulled gates the pull;
    # the requote path below it then hits the `requote skip (cap)` return → no place either.
    await handle_bba(
        MagicMock(), farm_state, AsyncMock(), pos, make_bba(no, Decimal("0.40"), Decimal("0.60"))
    )

    assert cancels == [], "a pulled position must not re-cancel on a still-bad frame"
    assert places == [], "and must not place into a book still beyond the loss cap"
    assert pos.quotes_pulled is True


# ── 3b/3c. the OTHER two pull clauses are also gated on quotes_pulled ───────────
# Belt-and-suspenders: the spec gated gap-pull (line ~204) and vol-pull (line ~231) with the same
# `and not pos.quotes_pulled` edit as exit-loss. test_bid_gap_pulls_orders_and_blacklists (in
# test_crash_guards.py) proves the same crash frame DOES pull when NOT pulled, so these pin that the
# flag suppresses it.


async def test_pulled_position_skips_gap_pull(
    farm_state: FarmState, cancels, requotes, monkeypatch
):
    """quotes_pulled=True + a crash frame that WOULD trip the best-bid gap-pull (0.50→0.40, a 20%
    drop > 10% and 0.10 > 2 ticks): the gap-pull clause is gated, so no cancel and no blacklist.
    (requote_leg is stubbed so `cancels` reflects ONLY the gap-pull clause, not a later requote.)"""
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("0")  # isolate from the mark-to-market kill
    pos.yes_cost_basis = Decimal("0")
    yes = pos.market.yes_token_id
    pos.yes_best_bid = Decimal("0.50")  # the prior bid the gap is measured against
    pos.quotes_pulled = True

    blacklisted: list = []

    def spy_blacklist(state, cid, now=None):
        blacklisted.append(cid)

    monkeypatch.setattr(requote_mod, "blacklist_for_gap", spy_blacklist)

    await handle_bba(
        MagicMock(), farm_state, AsyncMock(), pos, make_bba(yes, Decimal("0.40"), Decimal("0.52"))
    )

    assert cancels == [], "an already-pulled position must not re-cancel on the gap-pull clause"
    assert blacklisted == [], "the gap-pull blacklist must not fire when already pulled"
    assert pos.quotes_pulled is True


async def test_pulled_position_skips_vol_pull(
    farm_state: FarmState, cancels, requotes, monkeypatch
):
    """quotes_pulled=True + a YES frame whose record_price_sample returns a blacklist tier (would
    trip the volatility pull): the vol-pull clause is gated → no cancel. Stub record_price_sample to
    force the tier deterministically (no need to craft real volatility history). requote_leg is
    stubbed so `cancels` reflects ONLY the vol-pull clause, not the later requote path."""
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("0")  # isolate from the mark-to-market kill
    pos.yes_cost_basis = Decimal("0")
    yes = pos.market.yes_token_id
    pos.quotes_pulled = True

    def fake_record_price_sample(state, cid, price, now=None):
        return "15min"  # a non-None tier → the vol-pull body would run if not gated

    monkeypatch.setattr(requote_mod, "record_price_sample", fake_record_price_sample)

    # Stable bid (no prior yes_best_bid → gap-pull skipped; max_fill_loss unset → exit-loss inert).
    await handle_bba(
        MagicMock(), farm_state, AsyncMock(), pos, make_bba(yes, Decimal("0.50"), Decimal("0.52"))
    )

    assert cancels == [], "an already-pulled position must not re-cancel on the vol-pull clause"
    assert pos.quotes_pulled is True


# ── 4. self-heal of a pulled leg places despite a stale-cancel raise ───────────


async def test_self_heal_pulled_leg_places_despite_stale_cancel_raise(
    farm_state: FarmState, places, monkeypatch
):
    """quotes_pulled=True: the pre-cancel of the (already-gone) old order RAISES — requote_leg must
    log + fall through and still place the replacement, clearing the flag and updating the id.

    (POL-65 follow-up: the pulled-leg self-heal is now liveness-gated and only places when the old
    order is CONFIRMED gone. We stub `old_order_is_live` False explicitly so this confirmed-gone
    self-heal is pinned deterministically rather than relying on the real CLOB poll returning an
    empty set against a MagicMock client. The live/None skip paths are covered in 12.6.)"""
    pos = farm_state.positions["market-A"]
    pos.quotes_pulled = True
    stub_live(monkeypatch, False)

    async def raising_cancel(client, oid):
        raise RuntimeError("order already cancelled")

    monkeypatch.setattr(requote_mod, "cancel_order", raising_cancel)

    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.49"))

    assert len(places) == 1, "self-heal must place even though the stale pre-cancel raised"
    assert pos.yes_order_id == "new-oid-1"
    assert pos.yes_price == Decimal("0.49")
    assert pos.quotes_pulled is False


async def test_self_heal_pulled_leg_places_on_benign_cancel(farm_state: FarmState, cancels, places):
    """Benign-no-raise variant: quotes_pulled=True and the pre-cancel returns normally — the leg
    still re-quotes and re-arms (the fall-through is only for the raise; the happy path
    also places)."""
    pos = farm_state.positions["market-A"]
    pos.quotes_pulled = True

    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.49"))

    assert cancels == ["yes-oid"]
    assert len(places) == 1
    assert pos.yes_order_id == "new-oid-1"
    assert pos.quotes_pulled is False


# ── 5. normal requote whose pre-cancel fails must NOT place (double-rest guard) ─


async def test_normal_requote_failing_cancel_does_not_place(
    farm_state: FarmState, places, monkeypatch
):
    """quotes_pulled=False (a possibly-live order): if the pre-cancel RAISES and the old order is
    confirmed STILL LIVE, requote_leg must abort — leaving the old order id retained and placing NO
    replacement — so we never double-rest two live orders on the same leg.

    (POL-65: the raise + not-pulled path now consults `old_order_is_live`; we stub it LIVE so the
    no-place double-rest guard is the behaviour under test, not the self-heal fall-through. With the
    order live the first failure also bumps the per-leg counter to 1.)"""
    pos = farm_state.positions["market-A"]
    assert pos.quotes_pulled is False  # fixture default

    async def raising_cancel(client, oid):
        raise RuntimeError("cancel failed / unknown")

    async def live_check(client, old_oid):
        return True

    monkeypatch.setattr(requote_mod, "cancel_order", raising_cancel)
    monkeypatch.setattr(requote_mod, "old_order_is_live", live_check)

    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.49"))

    assert places == [], "a normal requote whose cancel failed must not place a second live order"
    assert pos.yes_order_id == "yes-oid", "the old order id must be retained"
    assert pos.yes_price == Decimal("0.5"), "price unchanged when the requote aborts"
    assert pos.quotes_pulled is False
    assert pos.yes_requote_cancel_fails == 1, "a live-confirmed cancel failure bumps the counter"


# ── 6. a failed/quarantined place leaves the pulled flag set ───────────────────


async def test_place_failure_keeps_flag_set(farm_state: FarmState, cancels, monkeypatch):
    """quotes_pulled=True and place_limit_order raises → the position is still pulled (flag stays
    True); we must not silently clear it on a place that never landed."""
    pos = farm_state.positions["market-A"]
    pos.quotes_pulled = True

    async def raising_place(client, order, post_only=True):
        raise RuntimeError("CLOB rejected")

    monkeypatch.setattr(requote_mod, "place_limit_order", raising_place)

    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.49"))

    assert pos.quotes_pulled is True, "a failed place must not clear the pulled latch"
    assert pos.yes_order_id == "yes-oid", "leg id unchanged when the place fails"


async def test_post_place_quarantine_keeps_flag_set(
    farm_state: FarmState, cancels, places, monkeypatch
):
    """quotes_pulled=True; the place succeeds but should_quote flips False just after (market
    quarantined mid-requote) → the new order is retracted and the flag is NOT cleared (the
    re-arm line is below the post-place quarantine return)."""
    pos = farm_state.positions["market-A"]
    pos.quotes_pulled = True

    calls = {"n": 0}

    def flaky_should_quote(state, p):
        # True on the pre-place check, False on the post-place check → triggers retract.
        calls["n"] += 1
        return calls["n"] == 1

    monkeypatch.setattr(requote_mod, "should_quote", flaky_should_quote)

    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.49"))

    assert len(places) == 1, "it placed before discovering the quarantine"
    assert "new-oid-1" in cancels, "the freshly placed order must be retracted"
    assert pos.quotes_pulled is True, "a retracted requote must not clear the pulled latch"
    assert pos.yes_order_id == "yes-oid", "leg id unchanged on a retracted requote"


# ── 7 & 8. cancel_position_orders always latches the flag ──────────────────────


async def test_cancel_position_orders_sets_flag_even_on_error(farm_state: FarmState, monkeypatch):
    """A2 invariant: even when the batch cancel fails at the CLOB layer, cancel_position_orders
    must still latch quotes_pulled.

    M2 note: the swallow moved from cancel_orders (thin shim) into the execution adapter. The
    thin shim propagates exceptions. The adapter (LegacyExecutionClient / SecureExecutionClient)
    catches and logs batch errors. cancel_position_orders itself has no try/except; the invariant
    is guaranteed by the adapter contract. Here we simulate adapter swallow behaviour directly to
    verify the latch is set regardless of underlying cancel outcome."""
    pos = farm_state.positions["market-A"]
    called_with: list = []

    async def fake_cancel_orders_swallows(client, *oids):
        """Simulates the adapter's cancel_orders: accepts oids, logs on error, never raises."""
        called_with.extend(oids)

    monkeypatch.setattr(requote_mod, "cancel_orders", fake_cancel_orders_swallows)
    client = MagicMock()

    await cancel_position_orders(client, farm_state, AsyncMock(), pos)  # must not raise

    assert called_with == ["yes-oid", "no-oid"], "both leg ids forwarded to cancel_orders"
    assert pos.quotes_pulled is True, "the flag must latch even when the batch cancel raises"


async def test_cancel_position_orders_empty_ids_sets_flag(farm_state: FarmState):
    """Both order ids empty → the adapter drops them and makes NO client round-trip, but the
    position is still marked pulled (nothing resting is exactly the pulled state).
    M2: cancel_orders thin shim forwards *args to the adapter; the adapter filters empty ids.
    Here we pass an AsyncMock adapter whose cancel_orders is a no-op for the forwarded ids."""
    pos = farm_state.positions["market-A"]
    pos.yes_order_id = ""
    pos.no_order_id = ""
    client = MagicMock()
    client.cancel_orders = AsyncMock(return_value=None)

    await cancel_position_orders(client, farm_state, AsyncMock(), pos)

    # The thin shim forwards ("", "") to client.cancel_orders; the adapter filters them.
    # We just verify the flag is latched regardless.
    assert pos.quotes_pulled is True


# ── 9. non-bba handlers are no-ops once the farm is killed ─────────────────────


async def test_handlers_noop_after_kill(farm_state: FarmState, cancels, monkeypatch):
    """state.killed=True: handle_book_snapshot / handle_price_change / handle_tick_size_change must
    each early-return before touching the network — zero cancels, zero requotes."""
    farm_state.config.filters.max_fill_loss = Decimal("1")
    farm_state.killed = True
    pos = farm_state.positions["market-A"]
    # A thin book that WOULD pull the depth guard if the handlers ran.
    farm_state.live_books["tok-yes"] = LiveBook(bids={Decimal("0.50"): Decimal("10")})

    requote_calls: list = []

    async def spy_requote(client, state, ws, p, outcome, price):
        requote_calls.append(outcome)

    cpo_calls: list = []

    async def spy_cpo(client, p):
        cpo_calls.append(p.market.condition_id)

    fetch_calls: list = []

    async def spy_fetch(http, token_ids):
        fetch_calls.append(token_ids)
        return {}

    monkeypatch.setattr(requote_mod, "requote_leg", spy_requote)
    monkeypatch.setattr(requote_mod, "cancel_position_orders", spy_cpo)
    monkeypatch.setattr(requote_mod, "fetch_midpoints", spy_fetch)

    await handle_book_snapshot(
        MagicMock(), farm_state, AsyncMock(), pos, make_book("tok-yes", [("0.50", "10")])
    )
    await handle_price_change(
        MagicMock(),
        farm_state,
        AsyncMock(),
        PriceChange(
            event_type="price_change",
            market="market-A",
            timestamp="1",
            price_changes=[{"asset_id": "tok-yes", "price": "0.50", "size": "0", "side": "BUY"}],
        ),
    )
    await handle_tick_size_change(
        MagicMock(),
        MagicMock(),
        farm_state,
        AsyncMock(),
        pos,
        TickSizeChange(
            event_type="tick_size_change",
            market="market-A",
            asset_id="tok-yes",
            old_tick_size=Decimal("0.01"),
            new_tick_size=Decimal("0.001"),
            timestamp="1",
        ),
    )

    assert cancels == [], "no cancel_order may be issued by any handler after a kill"
    assert cpo_calls == [], "no guard pull after a kill"
    assert requote_calls == [], "no requote after a kill"
    assert fetch_calls == [], "tick-size handler must not even fetch midpoints after a kill"


# ── 10. handle_bba still triggers the kill (kill path must survive Tier 1) ──────


async def test_handle_bba_still_triggers_kill(farm_state: FarmState, requotes, monkeypatch):
    """handle_bba was deliberately left as the place the kill is raised. With should_kill→True it
    must still call trigger_kill on the next frame."""
    pos = farm_state.positions["market-A"]
    kills: list = []

    def fake_should_kill(state):
        return True

    async def fake_trigger_kill(client, state, ws):
        state.killed = True
        kills.append(True)

    monkeypatch.setattr(requote_mod, "should_kill", fake_should_kill)
    monkeypatch.setattr(requote_mod, "trigger_kill", fake_trigger_kill)

    await handle_bba(
        MagicMock(),
        farm_state,
        AsyncMock(),
        pos,
        make_bba("tok-yes", Decimal("0.44"), Decimal("0.56")),
    )

    assert kills == [True], "handle_bba must still trigger the kill under Tier 1"
    assert farm_state.killed is True
    assert requotes == [], "a killed farm must not requote"


# ── 11. a pulled position is still exitable (close_position ignores the flag) ───


async def test_pulled_position_still_exitable(farm_state: FarmState, cancels, monkeypatch):
    """Smoke: close_position must cancel the legs and pop the position regardless of quotes_pulled —
    the pulled latch only gates re-quoting, never the exit."""
    from app.farm import worker as worker_mod

    async def fake_cancel_with_retry(client, oid):
        # close_position now cancels via cancel_order_with_retry — route it through the same
        # cancel_order patched by the `cancels` fixture so the recorder still sees both legs.
        await requote_mod.cancel_order(client, oid)
        return True

    monkeypatch.setattr(worker_mod, "cancel_order_with_retry", fake_cancel_with_retry)

    pos = farm_state.positions["market-A"]
    pos.quotes_pulled = True

    await worker_mod.close_position(
        MagicMock(), farm_state, AsyncMock(), "market-A", "market_dropped"
    )

    assert "market-A" not in farm_state.positions, "the position must be popped on close"
    assert set(cancels) == {"yes-oid", "no-oid"}, "both legs must be cancelled on close"


# ══════════════════════════════════════════════════════════════════════════════
# POL-65 — requote cancel-failure loop: a not-pulled leg whose pre-cancel RAISES
# now polls liveness (`old_order_is_live`) instead of blindly retaining forever.
#   live==False  → confirmed gone → reset counter + fall through to place (self-heal)
#   live==True   → still resting   → bump counter, keep leg, no place
#   live==None   → poll failed     → unknown, treated as live: bump counter, no place
#   counter hits REQUOTE_CANCEL_FAIL_PULL_THRESHOLD (3) → pull the whole position
#
# The requote_leg branch tests stub `requote_mod.old_order_is_live` directly to drive the
# live/gone/None outcome; the helper's own four branches are unit-tested separately at the bottom.
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def raising_cancel(monkeypatch):
    """Make requote_leg's singular pre-cancel (cancel_order) RAISE, exercising the except block.
    Returns nothing — its presence in a test's args installs the patch."""

    async def _raise(client, oid):
        raise RuntimeError("cancel failed / unknown")

    monkeypatch.setattr(requote_mod, "cancel_order", _raise)


def stub_live(monkeypatch, result):
    """Pin `old_order_is_live` to a fixed result (True/False/None) and record its calls so a test
    can assert whether the liveness poll was consulted at all."""
    calls: list = []

    async def stub(client, old_oid):
        calls.append(old_oid)
        return result

    monkeypatch.setattr(requote_mod, "old_order_is_live", stub)
    return calls


# ── 12.1 cancel raises + order confirmed GONE → self-heal places ───────────────


async def test_cancel_raise_order_gone_self_heals(
    farm_state: FarmState, places, raising_cancel, monkeypatch
):
    """quotes_pulled=False, the pre-cancel RAISES, but `old_order_is_live` confirms the old order is
    GONE server-side → requote_leg resets the counter and falls through to place the replacement
    (the same shape as the pulled-leg self-heal, just reached via the liveness poll)."""
    pos = farm_state.positions["market-A"]
    pos.yes_requote_cancel_fails = 0
    poll_calls = stub_live(monkeypatch, False)

    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.49"))

    assert poll_calls == ["yes-oid"], "the gone/live decision must consult old_order_is_live once"
    assert len(places) == 1, "a confirmed-gone order must self-heal by placing a replacement"
    assert pos.yes_order_id == "new-oid-1", "the leg id is updated to the replacement"
    assert pos.yes_price == Decimal("0.49")
    assert pos.quotes_pulled is False, "self-heal leaves the position quoting (not pulled)"
    assert pos.yes_requote_cancel_fails == 0, "a confirmed-gone self-heal resets the fail counter"


# ── 12.2 cancel raises + order still LIVE → no place, counter bumps ─────────────


async def test_cancel_raise_order_live_no_place(
    farm_state: FarmState, places, raising_cancel, monkeypatch
):
    """quotes_pulled=False, the pre-cancel RAISES and `old_order_is_live` reports the order is STILL
    LIVE → requote_leg must NOT place (no double-rest), retains the old id, and bumps the per-leg
    fail counter to 1."""
    pos = farm_state.positions["market-A"]
    pos.yes_requote_cancel_fails = 0
    stub_live(monkeypatch, True)

    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.49"))

    assert places == [], "a still-live order must not be replaced (would double-rest)"
    assert pos.yes_order_id == "yes-oid", "the old (still-live) order id is retained"
    assert pos.yes_price == Decimal("0.5"), "price unchanged when the requote aborts"
    assert pos.quotes_pulled is False, "below the cap the position stays un-pulled"
    assert pos.yes_requote_cancel_fails == 1, "a live-confirmed failure bumps the counter to 1"


# ── 12.3 cancel raises + liveness poll FAILS (None) → no place, counter bumps ───


async def test_cancel_raise_poll_fails_unknown_no_place(
    farm_state: FarmState, places, raising_cancel, monkeypatch
):
    """quotes_pulled=False, the pre-cancel RAISES and the liveness poll itself FAILS (`None`,
    unknown) → requote_leg treats unknown as possibly-live: no place, counter bumps to 1. The
    safe/conservative default — never double-rest on an order we can't prove is gone."""
    pos = farm_state.positions["market-A"]
    pos.yes_requote_cancel_fails = 0
    stub_live(monkeypatch, None)

    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.49"))

    assert places == [], "an unknown-liveness order must not be replaced"
    assert pos.yes_order_id == "yes-oid", "the old order id is retained on an unknown poll"
    assert pos.quotes_pulled is False
    assert pos.yes_requote_cancel_fails == 1, "an unknown (None) poll counts as a fail → counter 1"


# ── 12.4 cap reached: 3 consecutive live-confirmed failures pull the position ───


async def test_cancel_fail_cap_pulls_position(
    farm_state: FarmState, cancels, places, raising_cancel, monkeypatch
):
    """Three consecutive requotes whose pre-cancel RAISES with the order reported LIVE each time:
    failures 1 and 2 just bump the counter (no place, no pull); on the 3rd the counter reaches the
    REQUOTE_CANCEL_FAIL_PULL_THRESHOLD (3), so requote_leg gives up and pulls the WHOLE position via
    cancel_position_orders — cancelling BOTH legs, latching quotes_pulled, and resetting BOTH legs'
    counters to 0. No replacement is ever placed across all three frames.

    The give-up pull clears the OTHER leg's counter too: a stale count there (set to 2 below) must
    not survive the pull, or a single later isolated failure on that leg would trip the cap early
    (not truly N consecutive)."""
    assert REQUOTE_CANCEL_FAIL_PULL_THRESHOLD == 3, "test is written for a threshold of 3"
    pos = farm_state.positions["market-A"]
    pos.yes_requote_cancel_fails = 0
    pos.no_requote_cancel_fails = 2  # stale count on the other leg — must be cleared by the pull
    stub_live(monkeypatch, True)

    # Failures 1 and 2: counter climbs, nothing pulled, nothing placed.
    for expected in (1, 2):
        await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.49"))
        assert pos.yes_requote_cancel_fails == expected, f"counter must be {expected} after fail"
        assert pos.quotes_pulled is False, "below the cap the position is not pulled"
        assert cancels == [], "no cancel_position_orders batch below the cap"
        assert places == [], "no replacement is placed on a failed cancel"

    # Failure 3: hits the cap → give up and pull the whole position.
    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.49"))

    assert pos.quotes_pulled is True, "hitting the cap must pull the position (latch set)"
    assert set(cancels) == {"yes-oid", "no-oid"}, "the give-up pull cancels BOTH legs in one batch"
    assert pos.yes_requote_cancel_fails == 0, "the triggering leg's counter resets after the pull"
    assert pos.no_requote_cancel_fails == 0, "the OTHER leg's stale counter is also cleared"
    assert places == [], "the cap path pulls — it never places a replacement"


# ── 12.5 a successful requote resets the per-leg fail counter ──────────────────


async def test_cancel_success_resets_counter(farm_state: FarmState, cancels, places):
    """A leg carrying a non-zero fail counter that then re-quotes SUCCESSFULLY (pre-cancel returns
    normally, replacement places) must have its counter reset to 0 — a clean requote forgets the
    prior cancel failures so a later isolated failure starts fresh."""
    pos = farm_state.positions["market-A"]
    pos.yes_requote_cancel_fails = 2  # two prior failures on this leg

    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.49"))

    assert cancels == ["yes-oid"], "the pre-cancel hits the old leg id once"
    assert len(places) == 1, "a healthy requote places the replacement"
    assert pos.yes_order_id == "new-oid-1", "the leg id is updated"
    assert pos.quotes_pulled is False
    assert pos.yes_requote_cancel_fails == 0, "a successful requote resets the fail counter"


# ── 12.6 the pulled-leg self-heal is now liveness-gated — places only when GONE ──
# POL-65 follow-up: the pulled latch can be set OPTIMISTICALLY by cancel_position_orders even when
# its batch cancel actually failed (the swallow in cancel.py keeps the latch alive on failure), so
# `quotes_pulled` is NOT proof that nothing rests. The pulled-leg self-heal now hoists the same
# `old_order_is_live` poll to the top of the except and places a replacement ONLY when the old order
# is CONFIRMED gone (live is False). A still-live (True) or unknown (None) poll skips the place to
# avoid double-resting a live order. The per-leg fail counter is left untouched in this branch (the
# give-up counter logic lives only in the not-pulled branch).


async def test_pulled_leg_self_heal_live_skips_place(
    farm_state: FarmState, places, raising_cancel, monkeypatch
):
    """quotes_pulled=True, the pre-cancel RAISES, and `old_order_is_live` reports the old order is
    STILL LIVE → the self-heal must NOT place. This is the core double-rest regression guard: a
    stale-but-set latch sitting over a genuinely-live order must not be papered over with a second
    resting order. The poll IS consulted (once), the old id is retained, the leg stays pulled, and
    the per-leg counter is untouched."""
    pos = farm_state.positions["market-A"]
    pos.quotes_pulled = True
    pos.yes_requote_cancel_fails = 0
    poll_calls = stub_live(monkeypatch, True)

    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.49"))

    assert poll_calls == ["yes-oid"], "the pulled-leg self-heal now consults the liveness poll once"
    assert places == [], "a still-live order must not be self-healed (would double-rest)"
    assert pos.yes_order_id == "yes-oid", "the old (still-live) order id is retained"
    assert pos.quotes_pulled is True, "a skipped self-heal leaves the position pulled"
    assert pos.yes_requote_cancel_fails == 0, "the pulled branch leaves the fail counter untouched"


async def test_pulled_leg_self_heal_gone_places(
    farm_state: FarmState, places, raising_cancel, monkeypatch
):
    """quotes_pulled=True, the pre-cancel RAISES, and `old_order_is_live` confirms the old order is
    GONE → the genuine self-heal: place a replacement, clear the latch, update the leg id. Preserves
    the original self-heal-on-raise coverage via the now-explicit confirmed-gone path."""
    pos = farm_state.positions["market-A"]
    pos.quotes_pulled = True
    pos.yes_requote_cancel_fails = 0
    poll_calls = stub_live(monkeypatch, False)

    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.49"))

    assert poll_calls == ["yes-oid"], "the confirmed-gone decision consults the poll once"
    assert len(places) == 1, "a confirmed-gone pulled leg self-heals by placing a replacement"
    assert pos.yes_order_id == "new-oid-1", "the leg id is updated to the replacement"
    assert pos.quotes_pulled is False, "the successful self-heal re-arms the guards"
    assert pos.yes_requote_cancel_fails == 0, "a confirmed-gone self-heal leaves the counter at 0"


async def test_pulled_leg_self_heal_poll_fails_skips_place(
    farm_state: FarmState, places, raising_cancel, monkeypatch
):
    """quotes_pulled=True, the pre-cancel RAISES, and the liveness poll itself FAILS (`None`,
    unknown) → the self-heal treats unknown as possibly-live and SKIPS the place (never double-rest
    on an order we can't prove is gone). The old id is retained, the leg stays pulled, the per-leg
    counter is untouched. Pins the None→skip decision in the pulled branch."""
    pos = farm_state.positions["market-A"]
    pos.quotes_pulled = True
    pos.yes_requote_cancel_fails = 0
    poll_calls = stub_live(monkeypatch, None)

    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.49"))

    assert poll_calls == ["yes-oid"], "the unknown-liveness decision consults the poll once"
    assert places == [], "an unknown-liveness pulled leg must not be self-healed"
    assert pos.yes_order_id == "yes-oid", "the old order id is retained on an unknown poll"
    assert pos.quotes_pulled is True, "a skipped self-heal leaves the position pulled"
    assert pos.yes_requote_cancel_fails == 0, "the pulled branch leaves the fail counter untouched"


# ── 12.7 NO-leg parity: the counter and pull are tracked per-leg ───────────────


async def test_cancel_fail_counter_is_per_leg_no_side(
    farm_state: FarmState, places, raising_cancel, monkeypatch
):
    """The fail counter is per-leg: a live-confirmed cancel failure on the NO leg bumps
    `no_requote_cancel_fails` and leaves the YES counter at 0 (guards against a shared/global
    counter that would conflate the two legs)."""
    pos = farm_state.positions["market-A"]
    pos.no_requote_cancel_fails = 0
    pos.yes_requote_cancel_fails = 0
    stub_live(monkeypatch, True)

    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "NO", Decimal("0.49"))

    assert places == [], "a still-live NO order must not be replaced"
    assert pos.no_requote_cancel_fails == 1, "the NO-leg counter bumps"
    assert pos.yes_requote_cancel_fails == 0, "the YES-leg counter is untouched"
    assert pos.no_order_id == "no-oid", "the NO leg id is retained"


# ══════════════════════════════════════════════════════════════════════════════
# POL-65 — direct unit tests for the `old_order_is_live` helper. These stub
# `requote_mod.get_open_order_ids` (the real CLOB poll) to drive each of the four
# branches: empty-oid short-circuit, present, absent, and poll-raises.
# ══════════════════════════════════════════════════════════════════════════════


async def test_old_order_is_live_empty_oid_returns_false_without_polling(monkeypatch):
    """An empty old_oid → there is nothing to be live → returns False WITHOUT calling the CLOB poll
    (cheap short-circuit; an empty id can never appear in the open set anyway)."""
    polled: list = []

    async def fake_poll(client):
        polled.append(True)
        return {"whatever"}

    monkeypatch.setattr(requote_mod, "get_open_order_ids", fake_poll)

    result = await requote_mod.old_order_is_live(MagicMock(), "")

    assert result is False, "empty oid is treated as not-live"
    assert polled == [], "an empty oid must short-circuit before any CLOB round-trip"


async def test_old_order_is_live_present_returns_true(monkeypatch):
    """The old_oid is in the open-order set the CLOB returns → still live → True."""

    async def fake_poll(client):
        return {"yes-oid", "other-oid"}

    monkeypatch.setattr(requote_mod, "get_open_order_ids", fake_poll)

    assert await requote_mod.old_order_is_live(MagicMock(), "yes-oid") is True


async def test_old_order_is_live_absent_returns_false(monkeypatch):
    """The old_oid is NOT in the open-order set → confirmed gone → False."""

    async def fake_poll(client):
        return {"other-oid"}

    monkeypatch.setattr(requote_mod, "get_open_order_ids", fake_poll)

    assert await requote_mod.old_order_is_live(MagicMock(), "yes-oid") is False


async def test_old_order_is_live_poll_raises_returns_none(monkeypatch):
    """The CLOB poll itself RAISES → liveness is unknown → None (the caller treats None as
    possibly-live and refuses to double-rest)."""

    async def fake_poll(client):
        raise RuntimeError("CLOB 503 while polling open orders")

    monkeypatch.setattr(requote_mod, "get_open_order_ids", fake_poll)

    assert await requote_mod.old_order_is_live(MagicMock(), "yes-oid") is None
