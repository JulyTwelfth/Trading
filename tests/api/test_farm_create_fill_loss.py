"""End-to-end wiring for max_fill_loss: the value the UI puts in filters.max_fill_loss must
survive client_message_adapter.validate_python (the exact call the WS handler makes) AND then
drive the reconcile-tick exit_loss filter. Covers number/null/omitted/negative/zero/fractional
edge cases, plus a full wire->reconcile path proving the parsed value reaches the decision."""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from app.api.farm.messages import FarmCreateMessage
from app.api.messages import client_message_adapter
from app.bot.schemas import BookLevel, OrderBook
from app.farm import worker as worker_mod
from app.farm.schemas import FarmConfig, FarmState, Market
from app.farm.worker import reconcile_tick

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def payload(filters_extra: dict) -> dict:
    filters = {
        "vol_min": "0",
        "vol_max": "100000",
        "liq_min": "0",
        "liq_max": "100000",
        "spread_min": "0",
        "spread_max": "100",
        "reward_min": "0",
        "time_remaining": "all",
        "created_date": "all",
        "change_24h": "all",
    }
    filters.update(filters_extra)
    return {
        "type": "farm_create",
        "filters": filters,
        "bankroll": "500",
        "max_session_loss": "50",
    }


# ── wire parsing edge cases (the exact WS-handler parse) ──────────────────────


def test_max_fill_loss_number_parsed_as_decimal():
    msg = client_message_adapter.validate_python(payload({"max_fill_loss": "1"}))
    assert isinstance(msg, FarmCreateMessage)
    assert msg.filters.max_fill_loss == Decimal("1")


def test_max_fill_loss_null_parsed_as_none():
    msg = client_message_adapter.validate_python(payload({"max_fill_loss": None}))
    assert msg.filters.max_fill_loss is None


def test_max_fill_loss_omitted_defaults_to_none():
    # Backward compatibility: an older UI that never sends the field still validates (off).
    msg = client_message_adapter.validate_python(payload({}))
    assert msg.filters.max_fill_loss is None


def test_negative_max_fill_loss_rejected():
    # NonNegativeDecimal guard: a negative cap is nonsense and must fail validation.
    with pytest.raises(ValidationError):
        client_message_adapter.validate_python(payload({"max_fill_loss": "-1"}))


def test_zero_max_fill_loss_accepted_and_active():
    # "0" is a valid (non-None) cap — "only farm markets that lose nothing" — not "off".
    msg = client_message_adapter.validate_python(payload({"max_fill_loss": "0"}))
    assert msg.filters.max_fill_loss == Decimal("0")


def test_fractional_and_large_max_fill_loss_parse():
    frac = client_message_adapter.validate_python(payload({"max_fill_loss": "1.50"}))
    assert frac.filters.max_fill_loss == Decimal("1.50")
    big = client_message_adapter.validate_python(payload({"max_fill_loss": "1000"}))
    assert big.filters.max_fill_loss == Decimal("1000")


def test_default_form_value_one_parses_active():
    # The FE FORM_DEFAULTS ships maxFillLoss="1"; confirm that exact value lands active.
    msg = client_message_adapter.validate_python(payload({"max_fill_loss": "1"}))
    assert msg.filters.max_fill_loss == Decimal("1")


# ── full wire -> reconcile path ──────────────────────────────────────────────


def _market(**overrides) -> Market:
    base = dict(
        condition_id="c",
        slug="s",
        question="?",
        yes_token_id="y",
        no_token_id="n",
        rewards_max_spread_cents=Decimal("3"),
        rewards_min_size=Decimal("100"),
        rewards_rate_per_day=Decimal("5"),
        tick_size=Decimal("0.01"),
        min_order_size=Decimal("5"),
        end_date=datetime(2030, 1, 1, tzinfo=timezone.utc),
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        volume_24h=Decimal("100"),
        liquidity=Decimal("100"),
        spread_cents=Decimal("1"),
        price_change_24h=Decimal("0"),
    )
    base.update(overrides)
    return Market(**base)


def book(bids, asset_id) -> OrderBook:
    return OrderBook(
        market="m",
        asset_id=asset_id,
        timestamp=NOW,
        bids=[BookLevel(price=Decimal(p), size=Decimal(s)) for p, s in bids],
        asks=[],
        min_order_size=Decimal("1"),
        tick_size=Decimal("0.01"),
        neg_risk=False,
        hash="h",
    )


def patch_common(monkeypatch):
    async def fake_get_balance(addr):
        return Decimal("1000")

    async def fake_place_limit_order(client, order, post_only=False):
        return "oid"

    async def fake_exit_held_legs(*args, **kwargs):
        return None

    monkeypatch.setattr(worker_mod, "get_balance", fake_get_balance, raising=False)
    monkeypatch.setattr(worker_mod, "place_limit_order", fake_place_limit_order)
    monkeypatch.setattr(worker_mod, "exit_held_legs", fake_exit_held_legs)


async def test_wire_value_drives_reconcile_exit_loss_filter(monkeypatch, caplog):
    # FE sends max_fill_loss="1". After the REAL wire parse, the reconcile tick must reject a
    # market whose immediate fill+sell loses > $1 — proving the value flows wire -> decision.
    msg = client_message_adapter.validate_python(payload({"max_fill_loss": "1", "reward_min": "5"}))
    config = FarmConfig(
        filters=msg.filters,
        bankroll=Decimal("1000"),
        max_session_loss=msg.max_session_loss,
        quote_depth=msg.quote_depth,
    )
    state = FarmState(config=config, wallet_address="0xabc")
    costly = _market(condition_id="bad", slug="bad", yes_token_id="by", no_token_id="bn")
    cheap = _market(condition_id="ok", slug="ok", yes_token_id="oy", no_token_id="on")

    async def fake_markets(http):
        return [costly, cheap]

    async def fake_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    async def fake_books(http, token_ids):
        thin = [("0.40", "10000")]  # safe entry 0.48, size 100 -> loss $8 (>cap)
        deep = [("0.48", "10000")]  # exits at cost -> loss $0
        books = {
            "by": book(thin, "by"),
            "bn": book(thin, "bn"),
            "oy": book(deep, "oy"),
            "on": book(deep, "on"),
        }
        return {t: books[t] for t in token_ids if t in books}

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_midpoints)
    monkeypatch.setattr(worker_mod, "fetch_books", fake_books)
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "filter_funnel exit_loss=1" in text  # costly market rejected by the wired cap
    assert "candidates=1" in text
    assert "opened=1" in text  # cheap market still opens
