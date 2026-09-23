"""post_fill_sampler_loop — pure post-fill instrumentation (read-only, no trading).

Every POST_FILL_SAMPLE_INTERVAL_SECONDS the loop batch-fetches the books of every filled token in
state.book_samples and logs a "book_sample" strat line (the recovery-curve data gap: today the bot
goes blind the instant it dumps a fill). It prunes samples older than POST_FILL_SAMPLE_SECONDS, is
fully try/except-wrapped so a bad sample can never take down the farm, and stops on state.killed.

Harness mirrors the other worker loop tests (test_summary_loop / test_run_farm): stub the module
function (fetch_books) and the strat recorder, and break the otherwise-infinite loop by raising a
StopLoop sentinel from the patched asyncio.sleep AFTER the first iteration completes.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from app.bot.schemas import BookLevel, OrderBook
from app.constants import POST_FILL_SAMPLE_SECONDS
from app.farm import worker
from app.farm.schemas import BookSample

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


class StopLoop(Exception):
    """Sentinel to break out of the otherwise-infinite sampler loop (raised from sleep)."""


def _book(bids, asks, asset_id="tok-yes") -> OrderBook:
    return OrderBook(
        market="m",
        asset_id=asset_id,
        timestamp=NOW,
        bids=[BookLevel(price=Decimal(p), size=Decimal(s)) for p, s in bids],
        asks=[BookLevel(price=Decimal(p), size=Decimal(s)) for p, s in asks],
        min_order_size=Decimal("1"),
        tick_size=Decimal("0.01"),
        neg_risk=False,
        hash="h",
    )


def _sample(token="tok-yes", *, age_s: float) -> BookSample:
    return BookSample(
        token_id=token,
        slug="m1",
        outcome="YES",
        entry_px=Decimal("0.5"),
        started_at=datetime.now(timezone.utc) - timedelta(seconds=age_s),
    )


def _record_strats(monkeypatch) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(worker, "strat", lambda ev, **f: events.append((ev, f)))
    return events


def _stop_after_first_iteration(monkeypatch) -> None:
    async def stop(seconds):
        raise StopLoop

    monkeypatch.setattr(worker.asyncio, "sleep", stop)


async def test_sampler_logs_book_sample(farm_state, monkeypatch):
    # One iteration over one live token logs a book_sample with the derived book fields.
    token = "tok-yes"
    farm_state.book_samples[token] = _sample(token, age_s=12)

    async def fake_fetch_books(http, token_ids):
        assert token_ids == [token]
        return {token: _book([("0.42", "100")], [("0.59", "100")], token)}

    monkeypatch.setattr(worker, "fetch_books", fake_fetch_books)
    events = _record_strats(monkeypatch)
    _stop_after_first_iteration(monkeypatch)

    with pytest.raises(StopLoop):
        await worker.post_fill_sampler_loop(MagicMock(), farm_state)

    samples = [f for (e, f) in events if e == "book_sample"]
    assert len(samples) == 1
    s = samples[0]
    assert s["token"] == token
    assert s["slug"] == "m1"
    assert s["outcome"] == "YES"
    assert s["entry_px"] == Decimal("0.5")
    assert s["best_bid"] == Decimal("0.42")
    assert s["best_ask"] == Decimal("0.59")
    assert s["mid"] == Decimal("0.505")
    assert s["spread"] == Decimal("0.17")
    assert "bid_depth" in s
    assert s["elapsed_s"] >= 12  # ~12s since started_at, floored to int


async def test_sampler_prunes_stale_sample(farm_state, monkeypatch):
    # A sample older than POST_FILL_SAMPLE_SECONDS is dropped (and never logged) — the window
    # closes so the farm stops fetching a token whose recovery curve is already recorded.
    token = "tok-yes"
    farm_state.book_samples[token] = _sample(token, age_s=POST_FILL_SAMPLE_SECONDS + 5)

    async def fake_fetch_books(http, token_ids):
        return {token: _book([("0.42", "100")], [("0.59", "100")], token)}

    monkeypatch.setattr(worker, "fetch_books", fake_fetch_books)
    events = _record_strats(monkeypatch)
    _stop_after_first_iteration(monkeypatch)

    with pytest.raises(StopLoop):
        await worker.post_fill_sampler_loop(MagicMock(), farm_state)

    assert token not in farm_state.book_samples, "stale sample must be pruned"
    assert not [e for (e, _f) in events if e == "book_sample"], "pruned sample must not be logged"


async def test_sampler_swallows_fetch_error(farm_state, monkeypatch):
    # A raising fetch_books must be swallowed by the loop body (never propagate) so one bad batch
    # can't crash the farm. If it escaped we'd see RuntimeError here instead of StopLoop.
    farm_state.book_samples["tok-yes"] = _sample(age_s=5)

    async def boom(http, token_ids):
        raise RuntimeError("books endpoint down")

    monkeypatch.setattr(worker, "fetch_books", boom)
    _stop_after_first_iteration(monkeypatch)

    with pytest.raises(StopLoop):
        await worker.post_fill_sampler_loop(MagicMock(), farm_state)


async def test_sampler_stops_when_killed(farm_state, monkeypatch):
    # Once killed the loop returns immediately — it must not fetch or sleep.
    farm_state.killed = True
    farm_state.book_samples["tok-yes"] = _sample(age_s=5)
    called = {"n": 0}

    async def fake_fetch_books(http, token_ids):
        called["n"] += 1
        return {}

    async def boom(seconds):
        raise AssertionError("killed loop must not sleep")

    monkeypatch.setattr(worker, "fetch_books", fake_fetch_books)
    monkeypatch.setattr(worker.asyncio, "sleep", boom)

    await worker.post_fill_sampler_loop(MagicMock(), farm_state)
    assert called["n"] == 0


async def test_sampler_skips_fetch_when_no_samples(farm_state, monkeypatch):
    # No filled tokens tracked => the loop must not hit the network (the `if token_ids:` guard).
    called = {"n": 0}

    async def fake_fetch_books(http, token_ids):
        called["n"] += 1
        return {}

    monkeypatch.setattr(worker, "fetch_books", fake_fetch_books)
    _record_strats(monkeypatch)
    _stop_after_first_iteration(monkeypatch)

    with pytest.raises(StopLoop):
        await worker.post_fill_sampler_loop(MagicMock(), farm_state)

    assert called["n"] == 0


async def test_sampler_skips_one_sided_book(farm_state, monkeypatch):
    # A one-sided book (bids but no asks) yields no usable mid/spread — skip it, but keep the
    # sample (its window is still open; the book may fill back in on a later iteration).
    token = "tok-yes"
    farm_state.book_samples[token] = _sample(token, age_s=8)

    async def fake_fetch_books(http, token_ids):
        return {token: _book([("0.42", "100")], [], token)}  # asks empty

    monkeypatch.setattr(worker, "fetch_books", fake_fetch_books)
    events = _record_strats(monkeypatch)
    _stop_after_first_iteration(monkeypatch)

    with pytest.raises(StopLoop):
        await worker.post_fill_sampler_loop(MagicMock(), farm_state)

    assert not [e for (e, _f) in events if e == "book_sample"]
    assert token in farm_state.book_samples, "a one-sided book is skipped, not pruned"


async def test_sampler_skips_when_book_absent(farm_state, monkeypatch):
    # fetch_books may omit a token (no book returned) — that iteration must skip it cleanly and
    # retain the sample for a later retry.
    token = "tok-yes"
    farm_state.book_samples[token] = _sample(token, age_s=8)

    async def fake_fetch_books(http, token_ids):
        return {}  # token missing from the batch response

    monkeypatch.setattr(worker, "fetch_books", fake_fetch_books)
    events = _record_strats(monkeypatch)
    _stop_after_first_iteration(monkeypatch)

    with pytest.raises(StopLoop):
        await worker.post_fill_sampler_loop(MagicMock(), farm_state)

    assert not [e for (e, _f) in events if e == "book_sample"]
    assert token in farm_state.book_samples


async def test_sampler_is_cancellable(farm_state, monkeypatch):
    # A CancelledError (task teardown) must propagate, not be swallowed by the broad except.
    import asyncio

    farm_state.book_samples["tok-yes"] = _sample(age_s=5)

    async def block(http, token_ids):
        await asyncio.Event().wait()

    monkeypatch.setattr(worker, "fetch_books", block)
    task = asyncio.create_task(worker.post_fill_sampler_loop(MagicMock(), farm_state))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
