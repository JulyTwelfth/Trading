"""Fill-hook instrumentation in fills.handle_trade — pure logging on the MATCHED entry-fill path.

A fill (a) registers the filled token in state.book_samples so the sampler loop can watch its book
recover, (b) runs the shadow vacuum classifier and logs a "vacuum_shadow" verdict, and (c) logs
"fill_book" (top of the live book) and "fill_exposure" (usd = size*entry_px). ALL of it is
logging-only and each block is try/except-wrapped: a failure here must never propagate into the
fill path (that would kill the user-WS loop and stop all fill/exit processing).

Uses the conftest `farm_state` (position market-A, yes_token_id tok-yes, yes-oid registered).
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.bot.schemas import UserTrade, UserTradeMakerOrder
from app.farm import exits as exits_mod
from app.farm import fills as fills_mod
from app.farm.fills import handle_trade
from app.farm.schemas import LiveBook


def maker_fill(trade_id: str, price: str = "0.5", size: str = "100") -> UserTrade:
    # A MATCHED maker fill on the resting YES leg (yes-oid) of the conftest position.
    return UserTrade(
        event_type="trade",
        id=trade_id,
        asset_id="tok-yes",
        market="market-A",
        side="BUY",
        price=Decimal(price),
        size=Decimal(size),
        outcome="YES",
        status="MATCHED",
        timestamp="2026-01-01T00:00:00Z",
        taker_order_id="taker-x",
        maker_orders=[
            UserTradeMakerOrder(
                asset_id="tok-yes",
                order_id="yes-oid",
                matched_amount=Decimal(size),
                outcome="YES",
                owner="0xowner",
                price=Decimal(price),
            )
        ],
    )


def stub_fill_network(monkeypatch):
    async def fake_market_order(client, token_id, side, size):
        return "x"

    async def fake_cancel(client, oid):
        return None

    monkeypatch.setattr(exits_mod, "place_market_order", fake_market_order)
    monkeypatch.setattr(fills_mod, "cancel_order", fake_cancel)


def capture_strat(monkeypatch) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(fills_mod, "strat", lambda event, **fields: events.append((event, fields)))
    return events


def one(events: list[tuple[str, dict]], name: str) -> dict:
    matches = [f for (e, f) in events if e == name]
    assert len(matches) == 1, f"expected exactly one {name!r} strat, got {len(matches)}"
    return matches[0]


# ── (B) book_samples registration ─────────────────────────────────────────────────────────


async def test_fill_registers_book_sample(farm_state, monkeypatch):
    stub_fill_network(monkeypatch)
    before = datetime.now(timezone.utc)

    await handle_trade(MagicMock(), maker_fill("t1"), farm_state, AsyncMock())

    after = datetime.now(timezone.utc)
    sample = farm_state.book_samples.get("tok-yes")
    assert sample is not None, "the filled token must be registered for post-fill sampling"
    assert sample.token_id == "tok-yes"
    assert sample.slug == "m1"
    assert sample.outcome == "YES"
    assert sample.entry_px == Decimal("0.5")  # the maker fill price
    assert before <= sample.started_at <= after


# ── (C) vacuum_shadow verdict ───────────────────────────────────────────────────────────────


async def test_fill_vacuum_shadow_no_book_when_live_book_missing(farm_state, monkeypatch):
    # No live book tracked for the filled token → the classifier has nothing to read → "no_book".
    stub_fill_network(monkeypatch)
    strats = capture_strat(monkeypatch)
    assert "tok-yes" not in farm_state.live_books

    await handle_trade(MagicMock(), maker_fill("t1"), farm_state, AsyncMock())

    vac = one(strats, "vacuum_shadow")
    assert vac["verdict"] == "no_book"
    assert vac["entry_px"] == Decimal("0.5")
    assert vac["rest_px"] == "na"


async def test_fill_vacuum_shadow_fires_on_recoverable_book(farm_state, monkeypatch):
    # Live book at fill: bid vacuumed to 0.30, ask held at 0.48 (spread 0.18) — recoverable
    # vacuum → verdict "fire" with rest_px = min(entry 0.5, ask 0.48 - concession 0.05) = 0.43.
    stub_fill_network(monkeypatch)
    strats = capture_strat(monkeypatch)
    farm_state.live_books["tok-yes"] = LiveBook(
        bids={Decimal("0.30"): Decimal("100")},
        asks={Decimal("0.48"): Decimal("100")},
    )

    await handle_trade(MagicMock(), maker_fill("t1"), farm_state, AsyncMock())

    vac = one(strats, "vacuum_shadow")
    assert vac["verdict"] == "fire"
    assert vac["best_bid"] == Decimal("0.30")
    assert vac["best_ask"] == Decimal("0.48")
    assert vac["rest_px"] == Decimal("0.43")


async def test_fill_vacuum_shadow_dumps_on_tight_book(farm_state, monkeypatch):
    # A tight book (bid 0.47 / ask 0.52, spread 0.05 < 0.15) is not a vacuum → verdict "dump".
    stub_fill_network(monkeypatch)
    strats = capture_strat(monkeypatch)
    farm_state.live_books["tok-yes"] = LiveBook(
        bids={Decimal("0.47"): Decimal("100")},
        asks={Decimal("0.52"): Decimal("100")},
    )

    await handle_trade(MagicMock(), maker_fill("t1"), farm_state, AsyncMock())

    vac = one(strats, "vacuum_shadow")
    assert vac["verdict"] == "dump"
    assert vac["rest_px"] == "na"


# ── (C) fill_exposure and fill_book ─────────────────────────────────────────────────────────


async def test_fill_logs_exposure(farm_state, monkeypatch):
    # usd_exposure is size * entry_px — the notional we just took on, for exposure analytics.
    stub_fill_network(monkeypatch)
    strats = capture_strat(monkeypatch)

    trade = maker_fill("t1", price="0.5", size="100")
    await handle_trade(MagicMock(), trade, farm_state, AsyncMock())

    exp = one(strats, "fill_exposure")
    assert exp["outcome"] == "YES"
    assert exp["size"] == Decimal("100")
    assert exp["entry_px"] == Decimal("0.5")
    assert exp["usd_exposure"] == Decimal("50")  # 100 * 0.5


async def test_fill_logs_fill_book_top_of_book(farm_state, monkeypatch):
    # fill_book snapshots the live top-of-book at the moment of the fill.
    stub_fill_network(monkeypatch)
    strats = capture_strat(monkeypatch)
    farm_state.live_books["tok-yes"] = LiveBook(
        bids={Decimal("0.42"): Decimal("100"), Decimal("0.40"): Decimal("50")},
        asks={Decimal("0.59"): Decimal("100")},
    )

    await handle_trade(MagicMock(), maker_fill("t1"), farm_state, AsyncMock())

    fb = one(strats, "fill_book")
    assert fb["best_bid"] == Decimal("0.42")
    assert fb["best_ask"] == Decimal("0.59")
    assert fb["mid"] == Decimal("0.505")


async def test_fill_book_absent_when_no_live_book(farm_state, monkeypatch):
    # fill_book is only emitted when a live book exists for the filled token (it snapshots that
    # book) — with none tracked, the strat is skipped entirely rather than logged empty.
    stub_fill_network(monkeypatch)
    strats = capture_strat(monkeypatch)

    await handle_trade(MagicMock(), maker_fill("t1"), farm_state, AsyncMock())

    assert not [e for (e, _f) in strats if e == "fill_book"]


async def test_fill_book_na_when_one_sided(farm_state, monkeypatch):
    # A one-sided live book (bids only) still logs fill_book, but the ask/mid degrade to "na"
    # rather than raising.
    stub_fill_network(monkeypatch)
    strats = capture_strat(monkeypatch)
    farm_state.live_books["tok-yes"] = LiveBook(
        bids={Decimal("0.42"): Decimal("100")},
        asks={},
    )

    await handle_trade(MagicMock(), maker_fill("t1"), farm_state, AsyncMock())

    fb = one(strats, "fill_book")
    assert fb["best_bid"] == Decimal("0.42")
    assert fb["best_ask"] == "na"
    assert fb["mid"] == "na"


# ── crash-safety: instrumentation must never propagate into the fill path ──────────────────


async def test_fill_survives_vacuum_classifier_crash(farm_state, monkeypatch):
    # If the shadow classifier blows up, the fill must still complete: the sample is registered,
    # fill_exposure is still logged, and handle_trade does not raise (the user-WS loop survives).
    stub_fill_network(monkeypatch)
    strats = capture_strat(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("classifier exploded")

    monkeypatch.setattr(fills_mod, "exit_vacuum_price", boom)
    farm_state.live_books["tok-yes"] = LiveBook(
        bids={Decimal("0.30"): Decimal("100")},
        asks={Decimal("0.48"): Decimal("100")},
    )

    await handle_trade(MagicMock(), maker_fill("t1"), farm_state, AsyncMock())  # must not raise

    assert farm_state.book_samples.get("tok-yes") is not None
    assert one(strats, "fill_exposure")["usd_exposure"] == Decimal("50")
    # the classifier crash means no vacuum_shadow verdict was emitted, but the fill still booked
    assert not [e for (e, _f) in strats if e == "vacuum_shadow"]
    assert farm_state.positions["market-A"].yes_shares == Decimal("200")  # 100 held + 100 filled


async def test_fill_survives_book_sample_registration_crash(farm_state, monkeypatch):
    # A failure constructing/storing the BookSample must not block the rest of the fill.
    stub_fill_network(monkeypatch)
    strats = capture_strat(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("book sample construction failed")

    monkeypatch.setattr(fills_mod, "BookSample", boom)

    await handle_trade(MagicMock(), maker_fill("t1"), farm_state, AsyncMock())  # must not raise

    assert "tok-yes" not in farm_state.book_samples  # registration failed, swallowed
    assert one(strats, "fill_exposure")["usd_exposure"] == Decimal("50")  # fill still processed
    assert farm_state.positions["market-A"].yes_shares == Decimal("200")


async def test_fill_logging_blocks_are_independently_isolated(farm_state, monkeypatch):
    # The fill_book and fill_exposure logs live in separate try/except blocks: a crash logging
    # fill_book must not stop fill_exposure from being logged (nor propagate).
    stub_fill_network(monkeypatch)
    events: list[tuple[str, dict]] = []

    def selective_strat(event, **fields):
        events.append((event, fields))
        if event == "fill_book":
            raise RuntimeError("log sink down mid-fill_book")

    monkeypatch.setattr(fills_mod, "strat", selective_strat)
    farm_state.live_books["tok-yes"] = LiveBook(
        bids={Decimal("0.42"): Decimal("100")},
        asks={Decimal("0.59"): Decimal("100")},
    )

    await handle_trade(MagicMock(), maker_fill("t1"), farm_state, AsyncMock())  # must not raise

    assert [e for (e, _f) in events if e == "fill_book"], "fill_book was attempted"
    assert one(events, "fill_exposure")["usd_exposure"] == Decimal("50"), (
        "fill_exposure still logs after fill_book's block raised"
    )


async def test_fill_survives_exposure_logging_crash(farm_state, monkeypatch):
    # Even the last logging block (fill_exposure) is guarded — a failure there must not propagate.
    stub_fill_network(monkeypatch)

    def selective_strat(event, **fields):
        if event == "fill_exposure":
            raise RuntimeError("log sink down mid-fill_exposure")

    monkeypatch.setattr(fills_mod, "strat", selective_strat)

    await handle_trade(MagicMock(), maker_fill("t1"), farm_state, AsyncMock())  # must not raise

    assert farm_state.positions["market-A"].yes_shares == Decimal("200")  # fill booked regardless
