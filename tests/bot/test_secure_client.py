import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.bot import secure_client as sc
from app.exceptions import SecureOrderError


def accepted(order_id="oid-1"):
    return SimpleNamespace(ok=True, order_id=order_id, status="live")


def rejected(code="post_only_would_cross", message="would cross"):
    return SimpleNamespace(ok=False, code=code, message=message)


def cancel_resp(not_canceled=None):
    return SimpleNamespace(canceled=("a",), not_canceled=not_canceled or {})


def make_limit_order(**kwargs):
    from app.bot.schemas import LimitOrder

    defaults = dict(token_id="tid-1", side="BUY", size=10.0, price=0.5)
    defaults.update(kwargs)
    return LimitOrder(**defaults)


# ---------------------------------------------------------------------------
# build_secure_client
# ---------------------------------------------------------------------------


def test_build_secure_client_passes_private_key_and_proxy(monkeypatch):
    sentinel = object()
    calls = {}

    class FakeSecureClient:
        @classmethod
        def create(cls, **kwargs):
            calls.update(kwargs)
            return sentinel

    monkeypatch.setattr(sc, "SecureClient", FakeSecureClient)
    result = sc.build_secure_client("0xpk", "0xproxy")

    assert result is sentinel
    assert calls["private_key"] == "0xpk"
    assert calls["wallet"] == "0xproxy"


def test_build_secure_client_wallet_none_when_no_proxy(monkeypatch):
    calls = {}

    class FakeSecureClient:
        @classmethod
        def create(cls, **kwargs):
            calls.update(kwargs)
            return object()

    monkeypatch.setattr(sc, "SecureClient", FakeSecureClient)
    sc.build_secure_client("0xpk")

    assert calls["wallet"] is None


# ---------------------------------------------------------------------------
# wallet_type_for / is_deposit_wallet
# ---------------------------------------------------------------------------


def test_wallet_type_for_and_is_deposit_wallet():
    client = MagicMock()
    client.wallet_type = "DEPOSIT_WALLET"
    assert sc.wallet_type_for(client) == "DEPOSIT_WALLET"
    assert sc.is_deposit_wallet(client) is True

    client.wallet_type = "GNOSIS_SAFE"
    assert sc.is_deposit_wallet(client) is False


# ---------------------------------------------------------------------------
# place_limit_order
# ---------------------------------------------------------------------------


async def test_place_limit_order_forwards_kwargs_and_returns_order_id():
    client = MagicMock()
    client.place_limit_order = MagicMock(return_value=accepted("oid-9"))
    order = make_limit_order(token_id="tid-9", price=0.42, size=5.0, side="SELL")

    result = await sc.place_limit_order(client, order, post_only=True)

    assert result == "oid-9"
    _, kwargs = client.place_limit_order.call_args
    assert kwargs["token_id"] == "tid-9"
    assert kwargs["price"] == 0.42
    assert kwargs["size"] == 5.0
    assert kwargs["side"] == "SELL"
    assert kwargs["post_only"] is True


async def test_place_limit_order_post_only_defaults_false():
    client = MagicMock()
    client.place_limit_order = MagicMock(return_value=accepted())
    order = make_limit_order()

    await sc.place_limit_order(client, order)

    _, kwargs = client.place_limit_order.call_args
    assert kwargs["post_only"] is False


async def test_place_limit_order_raises_on_rejection():
    client = MagicMock()
    client.place_limit_order = MagicMock(
        return_value=rejected(code="not_enough_balance", message="insufficient funds")
    )
    order = make_limit_order()

    with pytest.raises(SecureOrderError) as exc_info:
        await sc.place_limit_order(client, order)

    assert exc_info.value.code == "not_enough_balance"
    assert exc_info.value.message == "insufficient funds"


# ---------------------------------------------------------------------------
# place_market_order
# ---------------------------------------------------------------------------


async def test_place_market_order_forwards_and_returns_id():
    client = MagicMock()
    client.place_market_order = MagicMock(return_value=accepted("oid-m"))

    result = await sc.place_market_order(client, "tid-1", "BUY", 50.0)

    assert result == "oid-m"
    _, kwargs = client.place_market_order.call_args
    assert kwargs["token_id"] == "tid-1"
    assert kwargs["side"] == "BUY"
    assert kwargs["amount"] == 50.0
    assert kwargs["order_type"] == "FAK"


async def test_place_market_order_raises_on_rejection():
    client = MagicMock()
    client.place_market_order = MagicMock(
        return_value=rejected(code="fak_not_filled", message="no fill")
    )

    with pytest.raises(SecureOrderError) as exc_info:
        await sc.place_market_order(client, "tid-1", "BUY", 50.0)

    assert exc_info.value.code == "fak_not_filled"
    assert exc_info.value.message == "no fill"


# ---------------------------------------------------------------------------
# cancel_order / cancel_all
# ---------------------------------------------------------------------------


async def test_cancel_order_forwards_order_id_and_returns_response():
    client = MagicMock()
    resp = cancel_resp()
    client.cancel_order = MagicMock(return_value=resp)

    result = await sc.cancel_order(client, "oid-1")

    assert result is resp
    _, kwargs = client.cancel_order.call_args
    assert kwargs["order_id"] == "oid-1"


async def test_cancel_all_delegates():
    client = MagicMock()
    resp = cancel_resp()
    client.cancel_all = MagicMock(return_value=resp)

    result = await sc.cancel_all(client)

    assert result is resp
    client.cancel_all.assert_called_once()


# ---------------------------------------------------------------------------
# cancel_orders
# ---------------------------------------------------------------------------


async def test_cancel_orders_skips_when_all_ids_empty():
    client = MagicMock()

    result = await sc.cancel_orders(client, "", None)

    assert result is None
    client.cancel_orders.assert_not_called()


async def test_cancel_orders_batches_and_logs_not_canceled(caplog):
    client = MagicMock()
    client.cancel_orders = MagicMock(
        return_value=cancel_resp(not_canceled={"a": "already cancelled"})
    )

    with caplog.at_level(logging.DEBUG, logger="app.bot.secure_client"):
        await sc.cancel_orders(client, "a", "b")

    _, kwargs = client.cancel_orders.call_args
    assert kwargs["order_ids"] == ["a", "b"]
    debug_recs = [
        r for r in caplog.records if r.levelno == logging.DEBUG and "a" in r.message
    ]
    assert debug_recs, "not_canceled entry must be logged at DEBUG"


async def test_cancel_orders_swallows_exception(caplog):
    client = MagicMock()
    client.cancel_orders = MagicMock(side_effect=RuntimeError("network down"))

    with caplog.at_level(logging.WARNING, logger="app.bot.secure_client"):
        result = await sc.cancel_orders(client, "a", "b")

    assert result is None
    warn_recs = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warn_recs, "exception must be logged at WARNING"


# ---------------------------------------------------------------------------
# test_deposit_wallet_derivation_matches_known_wallet
# (offline, no network — imports private polymarket module)
# ---------------------------------------------------------------------------


def test_deposit_wallet_derivation_matches_known_wallet():
    # NOTE: imports a private module (polymarket._internal.wallet); may break across SDK versions.
    from polymarket import PRODUCTION
    from polymarket._internal.wallet import classify_wallet_type, derive_uups_deposit_wallet_address

    eoa = "0xD74C367687fb814653cbbC0F9093305369c5F9cd"
    cfg = PRODUCTION.wallet_derivation
    derived = derive_uups_deposit_wallet_address(eoa, cfg)
    assert derived.lower() == "0x62a1da600b453f60cd76e577b9d352e6d0466aed"
    wtype = classify_wallet_type(signer=eoa, wallet=derived, config=cfg)
    assert wtype == "DEPOSIT_WALLET"


# ===========================================================================
# Tester additions (gap coverage)
# ===========================================================================
#
# Contract distinction being locked in here:
#   place_limit_order / place_market_order  -> a business REJECTION (result.ok is
#       False) is converted to SecureOrderError; a TRANSPORT-style failure
#       (the SDK call itself raising) PROPAGATES UNCHANGED.
#   cancel_orders (batch)                   -> SWALLOWS any raised exception.
#   cancel_order (single) / cancel_all      -> do NOT swallow; exceptions propagate.
# These asymmetries matter for callers, so they get explicit tests.


# ---------------------------------------------------------------------------
# place_* propagate transport-style errors (not converted to SecureOrderError)
# ---------------------------------------------------------------------------


async def test_place_limit_order_propagates_transport_error():
    # A raised SDK error (network/transport) must NOT be caught or rewrapped as
    # SecureOrderError — only result.ok == False is a SecureOrderError.
    from polymarket import TransportError

    client = MagicMock()
    client.place_limit_order = MagicMock(side_effect=TransportError("connection reset"))
    order = make_limit_order()

    with pytest.raises(TransportError):
        await sc.place_limit_order(client, order)


async def test_place_market_order_propagates_transport_error():
    from polymarket import TransportError

    client = MagicMock()
    client.place_market_order = MagicMock(side_effect=TransportError("connection reset"))

    with pytest.raises(TransportError):
        await sc.place_market_order(client, "tid-1", "BUY", 50.0)


# ---------------------------------------------------------------------------
# place_* forward strictly by keyword (real SDK methods are keyword-only;
# a positional call would TypeError). Use keyword-only fakes to prove it.
# ---------------------------------------------------------------------------


async def test_place_limit_order_invokes_sdk_with_keyword_args_only():
    captured = {}

    def fake_place(*, token_id, price, size, side, post_only):
        captured.update(
            token_id=token_id, price=price, size=size, side=side, post_only=post_only
        )
        return accepted("oid-kw")

    client = SimpleNamespace(place_limit_order=fake_place)
    order = make_limit_order(token_id="tid-kw", price=0.33, size=7.0, side="SELL")

    result = await sc.place_limit_order(client, order, post_only=True)

    assert result == "oid-kw"
    assert captured == dict(
        token_id="tid-kw", price=0.33, size=7.0, side="SELL", post_only=True
    )


async def test_place_market_order_invokes_sdk_with_keyword_args_only():
    captured = {}

    def fake_place(*, token_id, side, amount, order_type):
        captured.update(token_id=token_id, side=side, amount=amount, order_type=order_type)
        return accepted("oid-mkw")

    client = SimpleNamespace(place_market_order=fake_place)

    result = await sc.place_market_order(client, "tid-mkw", "BUY", 12.5)

    assert result == "oid-mkw"
    assert captured == dict(token_id="tid-mkw", side="BUY", amount=12.5, order_type="FAK")


# ---------------------------------------------------------------------------
# cancel_orders: id filtering + benign result paths
# ---------------------------------------------------------------------------


async def test_cancel_orders_filters_empty_ids_but_forwards_the_rest():
    # Mixed falsy + valid ids: only the truthy ids reach the client, order preserved.
    client = MagicMock()
    client.cancel_orders = MagicMock(return_value=cancel_resp())

    await sc.cancel_orders(client, "", "a", None, "b")

    client.cancel_orders.assert_called_once()
    _, kwargs = client.cancel_orders.call_args
    assert kwargs["order_ids"] == ["a", "b"]


async def test_cancel_orders_no_failures_logs_nothing_at_debug(caplog):
    # Empty not_canceled dict => nothing to log, no crash.
    client = MagicMock()
    client.cancel_orders = MagicMock(return_value=cancel_resp(not_canceled={}))

    with caplog.at_level(logging.DEBUG, logger="app.bot.secure_client"):
        await sc.cancel_orders(client, "a")

    debug_recs = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert debug_recs == []


async def test_cancel_orders_tolerates_result_without_not_canceled_attr():
    # getattr(..., "not_canceled", None) or {} must not blow up if the attr is absent.
    client = MagicMock()
    client.cancel_orders = MagicMock(return_value=SimpleNamespace(canceled=("a",)))

    result = await sc.cancel_orders(client, "a")

    assert result is None  # no exception raised


async def test_cancel_orders_swallows_exception_returns_none_without_logging_not_canceled(
    caplog,
):
    # When the batch call raises, we warn and return — and never touch result.not_canceled.
    client = MagicMock()
    client.cancel_orders = MagicMock(side_effect=ValueError("boom"))

    with caplog.at_level(logging.DEBUG, logger="app.bot.secure_client"):
        result = await sc.cancel_orders(client, "a", "b")

    assert result is None
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert not any(r.levelno == logging.DEBUG for r in caplog.records)


# ---------------------------------------------------------------------------
# cancel_order / cancel_all: single/global cancels do NOT swallow exceptions
# ---------------------------------------------------------------------------


async def test_cancel_order_propagates_exception():
    client = MagicMock()
    client.cancel_order = MagicMock(side_effect=RuntimeError("cancel failed"))

    with pytest.raises(RuntimeError):
        await sc.cancel_order(client, "oid-1")


async def test_cancel_all_propagates_exception():
    client = MagicMock()
    client.cancel_all = MagicMock(side_effect=RuntimeError("cancel-all failed"))

    with pytest.raises(RuntimeError):
        await sc.cancel_all(client)


# ---------------------------------------------------------------------------
# SecureOrderError contract
# ---------------------------------------------------------------------------


def test_secure_order_error_carries_code_message_and_str():
    err = SecureOrderError("not_enough_balance", "insufficient funds")
    assert err.code == "not_enough_balance"
    assert err.message == "insufficient funds"
    # str is used in logs/propagation; lock the "code: message" format.
    assert str(err) == "not_enough_balance: insufficient funds"
    assert isinstance(err, Exception)
