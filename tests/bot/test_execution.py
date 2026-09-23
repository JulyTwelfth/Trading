"""Tests for app/bot/execution.py — routing, both adapters, delegation shims, cross-cutting.

All tests are offline/mock-only: no network calls, no real wallets, no funds.
The CANARY test (test_private_wallet_module_signature_types) guards the private
polymarket._internal.wallet module against SDK bumps that break the mapping.
"""

import logging
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot import execution as exec_mod
from app.bot.cancel import cancel_all, cancel_order, cancel_orders
from app.bot.execution import (
    LegacyExecutionClient,
    SecureExecutionClient,
    build_execution_client,
    detect_wallet_type,
)
from app.bot.orders import get_open_order_ids
from app.bot.schemas import LimitOrder
from app.bot.trader import place_limit_order, place_market_order
from app.exceptions import SecureOrderError, WalletNotDeployedError
from app.farm.exits import is_zero_share_balance_rejection
from app.farm.rewards import fetch_market_earnings, fetch_reward_percentages, fetch_total_earnings

# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

EOA = "0xD74C367687fb814653cbbC0F9093305369c5F9cd"
DEPOSIT_WALLET = "0x62A1da600B453f60Cd76e577b9D352e6d0466AeD"
# A known Polymarket Proxy / GNOSIS_SAFE proxy used in existing tests
GNOSIS_SAFE_WALLET = "0xfEC13D2a7DFe36f1DEaB4eF94e3ADbD8C3f85eD"
# Private key whose Account.from_key().address == EOA
EOA_PRIVATE_KEY = "0xe4e2d53920fd1e986dd2f8b98fa6c2ee8598e3fae9f7e4406159e315f8ef69a4"


def make_wallet(proxy=GNOSIS_SAFE_WALLET, pk=EOA_PRIVATE_KEY):
    from app.db.wallets import Wallet

    return Wallet(
        license_key="lk",
        wallet_id="A",
        proxy_address=proxy,
        private_key=pk,
    )


def make_limit_order(**kwargs):
    defaults = dict(token_id="tid-1", side="BUY", size=10.0, price=0.5)
    defaults.update(kwargs)
    return LimitOrder(**defaults)


def accepted(order_id="oid-1"):
    return SimpleNamespace(ok=True, order_id=order_id, status="live")


def rejected(code="not_enough_balance", message="no balance"):
    return SimpleNamespace(ok=False, code=code, message=message)


def cancel_resp(not_canceled=None):
    return SimpleNamespace(canceled=("a",), not_canceled=not_canceled or {})


# ---------------------------------------------------------------------------
# CANARY: private module contract
# ---------------------------------------------------------------------------


def test_private_wallet_module_signature_types():
    """CANARY: polymarket._internal.wallet.signature_type_for must map the four
    wallet types to their expected integers. Fails loudly on an SDK bump that
    moves or renames the private module."""
    from polymarket._internal.wallet import signature_type_for

    assert signature_type_for("EOA") == 0
    assert signature_type_for("POLY_PROXY") == 1
    assert signature_type_for("GNOSIS_SAFE") == 2
    assert signature_type_for("DEPOSIT_WALLET") == 3


# ---------------------------------------------------------------------------
# 1. detect_wallet_type
# ---------------------------------------------------------------------------


def test_detect_wallet_type_deposit_wallet():
    # Pure offline check — no network; EOA derives the known deposit wallet.
    result = detect_wallet_type(EOA_PRIVATE_KEY, DEPOSIT_WALLET)
    assert result == "DEPOSIT_WALLET"


# ---------------------------------------------------------------------------
# 2. build_execution_client → SecureExecutionClient for DEPOSIT_WALLET
# ---------------------------------------------------------------------------


def test_build_execution_client_deposit_wallet_returns_secure(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(exec_mod.sc, "build_secure_client", lambda pk, proxy: sentinel)

    class FakeSecure:
        wallet_type = "DEPOSIT_WALLET"

    monkeypatch.setattr(exec_mod.sc, "build_secure_client", lambda pk, proxy: FakeSecure())
    wallet = make_wallet(proxy=DEPOSIT_WALLET, pk=EOA_PRIVATE_KEY)
    client = build_execution_client(wallet)
    assert isinstance(client, SecureExecutionClient)
    assert client.wallet_type == "DEPOSIT_WALLET"


# ---------------------------------------------------------------------------
# 3. build_execution_client → LegacyExecutionClient for GNOSIS_SAFE / POLY_PROXY
# ---------------------------------------------------------------------------


def test_build_execution_client_gnosis_safe_uses_signature_type_2(monkeypatch):
    built = {}

    def fake_build_clob(pk, proxy=None, signature_type=2):
        built["sig"] = signature_type
        built["proxy"] = proxy
        return MagicMock()

    monkeypatch.setattr(exec_mod, "build_clob_client", fake_build_clob)
    monkeypatch.setattr(exec_mod, "detect_wallet_type", lambda pk, proxy: "GNOSIS_SAFE")

    wallet = make_wallet()
    client = build_execution_client(wallet)

    assert isinstance(client, LegacyExecutionClient)
    assert built["sig"] == 2
    assert client.wallet_type == "GNOSIS_SAFE"


def test_build_execution_client_poly_proxy_uses_signature_type_1(monkeypatch):
    built = {}

    def fake_build_clob(pk, proxy=None, signature_type=2):
        built["sig"] = signature_type
        return MagicMock()

    monkeypatch.setattr(exec_mod, "build_clob_client", fake_build_clob)
    monkeypatch.setattr(exec_mod, "detect_wallet_type", lambda pk, proxy: "POLY_PROXY")

    wallet = make_wallet()
    client = build_execution_client(wallet)

    assert isinstance(client, LegacyExecutionClient)
    assert built["sig"] == 1
    assert client.wallet_type == "POLY_PROXY"


# ---------------------------------------------------------------------------
# 4. force_legacy_execution → always Legacy
# ---------------------------------------------------------------------------


def test_build_execution_client_force_legacy_overrides_deposit_wallet(monkeypatch):
    monkeypatch.setattr(exec_mod.settings, "force_legacy_execution", True)
    built = {}

    def fake_build_clob(pk, proxy=None, signature_type=2):
        built["sig"] = signature_type
        return MagicMock()

    monkeypatch.setattr(exec_mod, "build_clob_client", fake_build_clob)
    wallet = make_wallet(proxy=DEPOSIT_WALLET, pk=EOA_PRIVATE_KEY)
    client = build_execution_client(wallet)

    assert isinstance(client, LegacyExecutionClient)
    assert built["sig"] == 2  # hardcoded 2 in force-legacy path


# ---------------------------------------------------------------------------
# 5. classify failure → defaults legacy GNOSIS_SAFE
# ---------------------------------------------------------------------------


def test_build_execution_client_classify_failure_defaults_legacy(monkeypatch):
    def raise_rpc_error(pk, proxy):
        raise RuntimeError("rpc timeout")

    monkeypatch.setattr(exec_mod, "detect_wallet_type", raise_rpc_error)
    built = {}

    def fake_build_clob(pk, proxy=None, signature_type=2):
        built["sig"] = signature_type
        return MagicMock()

    monkeypatch.setattr(exec_mod, "build_clob_client", fake_build_clob)
    wallet = make_wallet()
    client = build_execution_client(wallet)

    assert isinstance(client, LegacyExecutionClient)
    assert client.wallet_type == "GNOSIS_SAFE"


# ---------------------------------------------------------------------------
# 6. undeployed deposit wallet → WalletNotDeployedError
# ---------------------------------------------------------------------------


def test_build_execution_client_undeployed_raises(monkeypatch):
    from polymarket import UserInputError

    monkeypatch.setattr(exec_mod, "detect_wallet_type", lambda pk, proxy: "DEPOSIT_WALLET")
    monkeypatch.setattr(
        exec_mod.sc,
        "build_secure_client",
        lambda pk, proxy: (_ for _ in ()).throw(UserInputError("not deployed")),
    )

    wallet = make_wallet(proxy=DEPOSIT_WALLET, pk=EOA_PRIVATE_KEY)
    with pytest.raises(WalletNotDeployedError):
        build_execution_client(wallet)


# ---------------------------------------------------------------------------
# LegacyExecutionClient tests (fake raw ClobClient)
# ---------------------------------------------------------------------------


def make_legacy(raw_client=None):
    if raw_client is None:
        raw_client = MagicMock()
    return LegacyExecutionClient(raw_client, "GNOSIS_SAFE")


# 7. place_limit_order builds OrderArgsV2(GTC, post_only) → result["orderID"]
async def test_legacy_place_limit_order_builds_args_and_returns_id():
    from py_clob_client_v2.clob_types import OrderArgsV2, OrderType

    raw = MagicMock()
    raw.create_and_post_order = MagicMock(return_value={"orderID": "oid-7"})
    adapter = make_legacy(raw)
    order = make_limit_order(token_id="t7", side="BUY", size=50.0, price=0.4)

    result = await adapter.place_limit_order(order, post_only=True)

    assert result == "oid-7"
    args, _ = raw.create_and_post_order.call_args
    order_args, neg_risk, order_type, post_only = args
    assert isinstance(order_args, OrderArgsV2)
    assert order_args.token_id == "t7"
    assert order_args.side == "BUY"
    assert order_args.size == 50.0
    assert order_args.price == 0.4
    assert neg_risk is None
    assert order_type == OrderType.GTC
    assert post_only is True


# 8. place_market_order builds MarketOrderArgsV2(FAK) → id
async def test_legacy_place_market_order_builds_args_and_returns_id():
    from py_clob_client_v2.clob_types import MarketOrderArgsV2, OrderType

    raw = MagicMock()
    raw.create_and_post_market_order = MagicMock(return_value={"orderID": "mkt-8"})
    adapter = make_legacy(raw)

    result = await adapter.place_market_order("tok-8", "SELL", 20.0)

    assert result == "mkt-8"
    args, _ = raw.create_and_post_market_order.call_args
    order_args, neg_risk, order_type = args
    assert isinstance(order_args, MarketOrderArgsV2)
    assert order_args.token_id == "tok-8"
    assert order_args.side == "SELL"
    assert order_args.amount == 20.0
    assert neg_risk is None
    assert order_type == OrderType.FAK


# 9. cancel_order, cancel_all, cancel_orders
async def test_legacy_cancel_order():
    from py_clob_client_v2.clob_types import OrderPayload

    raw = MagicMock()
    raw.cancel_order = MagicMock(return_value={"canceled": ["oid-1"]})
    adapter = make_legacy(raw)

    result = await adapter.cancel_order("oid-1")

    assert result == {"canceled": ["oid-1"]}
    (payload,), _ = raw.cancel_order.call_args
    assert isinstance(payload, OrderPayload)
    assert payload.orderID == "oid-1"


async def test_legacy_cancel_all():
    raw = MagicMock()
    raw.cancel_all = MagicMock(return_value={"canceled": "all"})
    adapter = make_legacy(raw)

    result = await adapter.cancel_all()

    assert result == {"canceled": "all"}
    raw.cancel_all.assert_called_once()


async def test_legacy_cancel_orders_drops_empty_ids():
    raw = MagicMock()
    adapter = make_legacy(raw)

    await adapter.cancel_orders("", None, "")

    raw.cancel_orders.assert_not_called()


async def test_legacy_cancel_orders_batches_and_logs_not_canceled(caplog):
    raw = MagicMock()
    raw.cancel_orders = MagicMock(
        return_value={"canceled": ["b"], "not_canceled": {"a": "already cancelled"}}
    )
    adapter = make_legacy(raw)

    with caplog.at_level(logging.DEBUG, logger="app.bot.execution"):
        await adapter.cancel_orders("a", "b")

    raw.cancel_orders.assert_called_once_with(["a", "b"])
    debug_recs = [r for r in caplog.records if r.levelno == logging.DEBUG and "a" in r.message]
    assert debug_recs, "not_canceled entry must be logged at DEBUG"


async def test_legacy_cancel_orders_swallows_exception(caplog):
    raw = MagicMock()
    raw.cancel_orders = MagicMock(side_effect=RuntimeError("network down"))
    adapter = make_legacy(raw)

    with caplog.at_level(logging.WARNING, logger="app.bot.execution"):
        result = await adapter.cancel_orders("a", "b")

    assert result is None
    warn_recs = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warn_recs


# 10. get_open_order_ids dicts → set
async def test_legacy_get_open_order_ids():
    raw = MagicMock()
    raw.get_open_orders = MagicMock(
        return_value=[
            {"id": "oid-a", "status": "LIVE"},
            {"id": "oid-b"},
            {"status": "LIVE"},  # no id → skipped
            "junk",  # non-dict → skipped
        ]
    )
    adapter = make_legacy(raw)

    result = await adapter.get_open_order_ids()

    assert result == {"oid-a", "oid-b"}


# 11. send_heartbeat → post_heartbeat; supports_heartbeat True
async def test_legacy_send_heartbeat():
    raw = MagicMock()
    raw.post_heartbeat = MagicMock(return_value={"heartbeat_id": "hb-1"})
    adapter = make_legacy(raw)

    result = await adapter.send_heartbeat("hb-prev")

    assert result == {"heartbeat_id": "hb-1"}
    raw.post_heartbeat.assert_called_once_with("hb-prev")
    assert adapter.supports_heartbeat is True


# 12. ws_auth from .creds.api_*
def test_legacy_ws_auth():
    raw = MagicMock()
    raw.creds.api_key = "ak"
    raw.creds.api_secret = "as"
    raw.creds.api_passphrase = "ap"
    adapter = make_legacy(raw)

    result = adapter.ws_auth()

    assert result == {"apiKey": "ak", "secret": "as", "passphrase": "ap"}


# 13. rewards: total / market / percentages parse legacy dict shapes
async def test_legacy_total_earnings_today_sums_entries():
    raw = MagicMock()
    raw.get_total_earnings_for_user_for_day = MagicMock(
        return_value=[
            {"earnings": "1.50"},
            {"earnings": "0.25"},
            None,  # malformed → skipped
            {"no_earnings": "x"},  # KeyError → skipped
        ]
    )
    adapter = make_legacy(raw)
    result = await adapter.total_earnings_today()
    assert result == Decimal("1.75")


async def test_legacy_market_earnings_today_sums_per_condition_id():
    raw = MagicMock()
    raw.get_earnings_for_user_for_day = MagicMock(
        return_value=[
            {"condition_id": "c1", "earnings": "1.00"},
            {"condition_id": "c1", "earnings": "0.50"},
            {"condition_id": "c2", "earnings": "0.25"},
            "junk",
        ]
    )
    adapter = make_legacy(raw)
    result = await adapter.market_earnings_today()
    assert result == {"c1": Decimal("1.50"), "c2": Decimal("0.25")}


async def test_legacy_reward_percentages():
    raw = MagicMock()
    raw.get_reward_percentages = MagicMock(return_value={"cid-1": 50.0, "cid-2": 12.5})
    adapter = make_legacy(raw)
    result = await adapter.reward_percentages()
    assert result == {"cid-1": Decimal("50.0"), "cid-2": Decimal("12.5")}


# 14. refresh_conditional_balance
async def test_legacy_refresh_conditional_balance_returns_true():
    raw = MagicMock()
    raw.update_balance_allowance = MagicMock(return_value=None)
    adapter = make_legacy(raw)
    assert await adapter.refresh_conditional_balance("tok-1") is True
    raw.update_balance_allowance.assert_called_once()


async def test_legacy_refresh_conditional_balance_exception_returns_false():
    raw = MagicMock()
    raw.update_balance_allowance = MagicMock(side_effect=RuntimeError("network"))
    adapter = make_legacy(raw)
    assert await adapter.refresh_conditional_balance("tok-1") is False


# ---------------------------------------------------------------------------
# SecureExecutionClient tests (fake SecureClient)
# ---------------------------------------------------------------------------


def make_secure_adapter():
    raw = MagicMock()
    raw.wallet_type = "DEPOSIT_WALLET"
    raw.credentials = SimpleNamespace(key="sk", secret="ss", passphrase="sp")
    return SecureExecutionClient(raw), raw


# 15. place_limit_order forwards kwargs; ok → id; reject → SecureOrderError
async def test_secure_place_limit_order_returns_id(monkeypatch):
    adapter, raw = make_secure_adapter()
    raw.place_limit_order = MagicMock(return_value=accepted("oid-15"))
    order = make_limit_order(token_id="t15", side="BUY", size=5.0, price=0.3)

    result = await adapter.place_limit_order(order, post_only=True)

    assert result == "oid-15"
    _, kwargs = raw.place_limit_order.call_args
    assert kwargs["token_id"] == "t15"
    assert kwargs["post_only"] is True


async def test_secure_place_limit_order_raises_on_rejection():
    adapter, raw = make_secure_adapter()
    raw.place_limit_order = MagicMock(return_value=rejected("post_only_would_cross", "cross"))
    order = make_limit_order()

    with pytest.raises(SecureOrderError) as exc_info:
        await adapter.place_limit_order(order)

    assert exc_info.value.code == "post_only_would_cross"


# 16. place_market_order SELL → shares; BUY → amount (CRITICAL correctness test)
async def test_secure_place_market_order_sell_uses_shares_not_amount():
    adapter, raw = make_secure_adapter()
    raw.place_market_order = MagicMock(return_value=accepted("oid-s"))

    result = await adapter.place_market_order("tok-16", "SELL", 42.0)

    assert result == "oid-s"
    _, kwargs = raw.place_market_order.call_args
    assert "shares" in kwargs, "SELL must use shares= kwarg"
    assert kwargs["shares"] == 42.0
    assert "amount" not in kwargs, "SELL must NOT pass amount="


async def test_secure_place_market_order_buy_uses_amount():
    adapter, raw = make_secure_adapter()
    raw.place_market_order = MagicMock(return_value=accepted("oid-b"))

    await adapter.place_market_order("tok-16", "BUY", 100.0)

    _, kwargs = raw.place_market_order.call_args
    assert "amount" in kwargs
    assert kwargs["amount"] == 100.0
    assert "shares" not in kwargs


async def test_secure_place_market_order_raises_on_rejection():
    adapter, raw = make_secure_adapter()
    raw.place_market_order = MagicMock(return_value=rejected("fak_not_filled", "no fill"))

    with pytest.raises(SecureOrderError) as exc_info:
        await adapter.place_market_order("tok-16", "SELL", 10.0)

    assert exc_info.value.code == "fak_not_filled"


# 17. cancel_order, cancel_all, cancel_orders
async def test_secure_cancel_order():
    adapter, raw = make_secure_adapter()
    resp = cancel_resp()
    raw.cancel_order = MagicMock(return_value=resp)

    result = await adapter.cancel_order("oid-1")

    assert result is resp
    _, kwargs = raw.cancel_order.call_args
    assert kwargs["order_id"] == "oid-1"


async def test_secure_cancel_all():
    adapter, raw = make_secure_adapter()
    resp = cancel_resp()
    raw.cancel_all = MagicMock(return_value=resp)

    result = await adapter.cancel_all()

    assert result is resp


async def test_secure_cancel_orders_skips_empty():
    adapter, raw = make_secure_adapter()
    raw.cancel_orders = MagicMock()

    await adapter.cancel_orders("", None)

    raw.cancel_orders.assert_not_called()


# 18. get_open_order_ids iterates iter_items() → set of .id
async def test_secure_get_open_order_ids():
    adapter, raw = make_secure_adapter()
    items = [SimpleNamespace(id="a"), SimpleNamespace(id="b"), SimpleNamespace(id="")]
    paginator = MagicMock()
    paginator.iter_items = MagicMock(return_value=iter(items))
    raw.list_open_orders = MagicMock(return_value=paginator)

    result = await adapter.get_open_order_ids()

    assert result == {"a", "b"}  # empty string id skipped


# 19. send_heartbeat → {}; supports_heartbeat False
async def test_secure_send_heartbeat_returns_empty_dict():
    adapter, _ = make_secure_adapter()
    result = await adapter.send_heartbeat("any-id")
    assert result == {}
    assert adapter.supports_heartbeat is False


# 20. ws_auth from .credentials.key/secret/passphrase
def test_secure_ws_auth():
    adapter, _ = make_secure_adapter()
    result = adapter.ws_auth()
    assert result == {"apiKey": "sk", "secret": "ss", "passphrase": "sp"}


# 21. rewards: total/market/percentages parse secure object shapes
async def test_secure_total_earnings_today():
    adapter, raw = make_secure_adapter()
    entries = [SimpleNamespace(earnings=Decimal("1.00")), SimpleNamespace(earnings=Decimal("0.50"))]
    raw.get_total_earnings_for_user_for_day = MagicMock(return_value=entries)

    result = await adapter.total_earnings_today()

    assert result == Decimal("1.50")


async def test_secure_market_earnings_today():
    adapter, raw = make_secure_adapter()
    items = [
        SimpleNamespace(condition_id="cid-1", earnings=Decimal("0.80")),
        SimpleNamespace(condition_id="cid-1", earnings=Decimal("0.20")),
        SimpleNamespace(condition_id="cid-2", earnings=Decimal("0.30")),
    ]
    paginator = MagicMock()
    paginator.iter_items = MagicMock(return_value=iter(items))
    raw.list_user_earnings_for_day = MagicMock(return_value=paginator)

    result = await adapter.market_earnings_today()

    assert result == {"cid-1": Decimal("1.00"), "cid-2": Decimal("0.30")}


async def test_secure_reward_percentages():
    adapter, raw = make_secure_adapter()
    raw.get_reward_percentages = MagicMock(return_value={"cid-x": 75.0})

    result = await adapter.reward_percentages()

    assert result == {"cid-x": Decimal("75.0")}


# 22. refresh_conditional_balance
async def test_secure_refresh_conditional_balance_returns_true():
    adapter, raw = make_secure_adapter()
    raw.get_balance_allowance = MagicMock(return_value=MagicMock())

    assert await adapter.refresh_conditional_balance("tok-22") is True
    _, kwargs = raw.get_balance_allowance.call_args
    assert kwargs["token_id"] == "tok-22"
    assert kwargs["asset_type"] == "CONDITIONAL"


async def test_secure_refresh_conditional_balance_exception_returns_false():
    adapter, raw = make_secure_adapter()
    raw.get_balance_allowance = MagicMock(side_effect=RuntimeError("http error"))

    assert await adapter.refresh_conditional_balance("tok-22") is False


# ---------------------------------------------------------------------------
# 23. is_zero_share_balance_rejection unified: SecureOrderError + legacy string
# ---------------------------------------------------------------------------


def test_is_zero_share_balance_rejection_secure_order_error():
    exc = SecureOrderError("not_enough_balance", "no balance")
    assert is_zero_share_balance_rejection(exc) is True


def test_is_zero_share_balance_rejection_unrelated_secure_error():
    exc = SecureOrderError("fak_not_filled", "no match")
    assert is_zero_share_balance_rejection(exc) is False


def test_is_zero_share_balance_rejection_legacy_string():
    from py_clob_client_v2.exceptions import PolyApiException

    zero_msg = (
        "not enough balance / allowance: the balance is not enough"
        " -> balance: 0, order amount: 20000000"
    )
    exc = PolyApiException(error_msg={"error": zero_msg})
    assert is_zero_share_balance_rejection(exc) is True


def test_is_zero_share_balance_rejection_non_zero_balance():
    from py_clob_client_v2.exceptions import PolyApiException

    nonzero_msg = "not enough balance / allowance: the balance is not enough -> balance: 26120900"
    exc = PolyApiException(error_msg={"error": nonzero_msg})
    assert is_zero_share_balance_rejection(exc) is False


def test_is_zero_share_balance_rejection_unrelated_exception():
    assert is_zero_share_balance_rejection(RuntimeError("network")) is False


# ---------------------------------------------------------------------------
# 24. heartbeat_loop early-returns when supports_heartbeat is False
# ---------------------------------------------------------------------------


async def test_heartbeat_disabled_for_deposit_wallet():
    """Inline version: ensures heartbeat_loop returns without calling send_heartbeat."""
    from app.bot.heartbeat import heartbeat_loop

    client = MagicMock()
    client.supports_heartbeat = False
    client.wallet_type = "DEPOSIT_WALLET"
    call_log = []

    async def spy_send(heartbeat_id=""):
        call_log.append(heartbeat_id)
        return {}

    client.send_heartbeat = spy_send

    await heartbeat_loop(client)  # must return immediately, not loop forever

    assert call_log == []


# ---------------------------------------------------------------------------
# 25. Delegation shims: trader/cancel/orders/rewards/exits/user_ws
# ---------------------------------------------------------------------------


async def test_delegation_place_limit_order():
    client = AsyncMock()
    client.place_limit_order = AsyncMock(return_value="dlg-1")
    order = make_limit_order()
    result = await place_limit_order(client, order, post_only=True)
    assert result == "dlg-1"
    client.place_limit_order.assert_awaited_once_with(order, post_only=True)


async def test_delegation_place_market_order():
    client = AsyncMock()
    client.place_market_order = AsyncMock(return_value="dlg-2")
    result = await place_market_order(client, "tok", "SELL", 5.0)
    assert result == "dlg-2"
    client.place_market_order.assert_awaited_once_with("tok", "SELL", 5.0)


async def test_delegation_cancel_order():
    client = AsyncMock()
    client.cancel_order = AsyncMock(return_value="cr")
    result = await cancel_order(client, "oid")
    assert result == "cr"
    client.cancel_order.assert_awaited_once_with("oid")


async def test_delegation_cancel_all():
    client = AsyncMock()
    client.cancel_all = AsyncMock(return_value="all")
    result = await cancel_all(client)
    assert result == "all"
    client.cancel_all.assert_awaited_once_with()


async def test_delegation_cancel_orders():
    client = AsyncMock()
    client.cancel_orders = AsyncMock(return_value=None)
    await cancel_orders(client, "a", "b")
    client.cancel_orders.assert_awaited_once_with("a", "b")


async def test_delegation_get_open_order_ids():
    client = AsyncMock()
    client.get_open_order_ids = AsyncMock(return_value={"x", "y"})
    result = await get_open_order_ids(client)
    assert result == {"x", "y"}
    client.get_open_order_ids.assert_awaited_once_with()


async def test_delegation_fetch_total_earnings():
    client = AsyncMock()
    client.total_earnings_today = AsyncMock(return_value=Decimal("2.00"))
    result = await fetch_total_earnings(client)
    assert result == Decimal("2.00")


async def test_delegation_fetch_market_earnings():
    client = AsyncMock()
    client.market_earnings_today = AsyncMock(return_value={"c1": Decimal("1.00")})
    result = await fetch_market_earnings(client)
    assert result == {"c1": Decimal("1.00")}


async def test_delegation_fetch_reward_percentages():
    client = AsyncMock()
    client.reward_percentages = AsyncMock(return_value={"c1": Decimal("50")})
    result = await fetch_reward_percentages(client)
    assert result == {"c1": Decimal("50")}


async def test_delegation_user_ws_calls_ws_auth_on_adapter():
    """stream_user_trades reads auth from client.ws_auth(); any adapter satisfying
    the protocol (LegacyExecutionClient or SecureExecutionClient) works.
    Coverage of the full subscribe frame is in tests/bot/test_user_ws.py."""
    # The function calls client.ws_auth() before entering the reconnect loop.
    # We cancel immediately after the call to avoid a real network attempt.
    import asyncio

    from app.bot import user_ws as user_ws_mod
    from app.bot.user_ws import stream_user_trades

    client = MagicMock()
    client.ws_auth.return_value = {"apiKey": "ka", "secret": "sa", "passphrase": "pa"}

    # Replace websockets.connect with something that raises CancelledError immediately
    user_ws_mod_connect_orig = user_ws_mod.websockets.connect

    def cancel_on_connect(url, ssl=None):
        raise asyncio.CancelledError()

    user_ws_mod.websockets.connect = cancel_on_connect
    try:
        with pytest.raises(asyncio.CancelledError):
            async for _ in stream_user_trades(client):
                pass
    finally:
        user_ws_mod.websockets.connect = user_ws_mod_connect_orig

    client.ws_auth.assert_called_once()


# ===========================================================================
# Tester additions (M2 gap coverage)
# ===========================================================================


# --- Routing: complete the signature_type matrix (EOA → 0) --------------------


def test_build_execution_client_eoa_uses_signature_type_0(monkeypatch):
    # Completes the routing matrix: EOA is a legacy wallet with signature_type 0.
    built = {}

    def fake_build_clob(pk, proxy=None, signature_type=2):
        built["sig"] = signature_type
        return MagicMock()

    monkeypatch.setattr(exec_mod, "build_clob_client", fake_build_clob)
    monkeypatch.setattr(exec_mod, "detect_wallet_type", lambda pk, proxy: "EOA")

    client = build_execution_client(make_wallet())

    assert isinstance(client, LegacyExecutionClient)
    assert built["sig"] == 0
    assert client.wallet_type == "EOA"


# --- Legacy adapter: place_* PROPAGATE (exit sweep + circuit breaker rely on it) ---
# Unlike the secure adapter (result.ok==False → SecureOrderError), the legacy adapter
# surfaces failures as the raw client's raised exception. It must never swallow them.


async def test_legacy_place_limit_order_propagates_exception():
    from py_clob_client_v2.exceptions import PolyApiException

    raw = MagicMock()
    raw.create_and_post_order = MagicMock(
        side_effect=PolyApiException(error_msg={"error": "rejected"})
    )
    adapter = make_legacy(raw)

    with pytest.raises(PolyApiException):
        await adapter.place_limit_order(make_limit_order())


async def test_legacy_place_market_order_propagates_exception():
    from py_clob_client_v2.exceptions import PolyApiException

    raw = MagicMock()
    raw.create_and_post_market_order = MagicMock(
        side_effect=PolyApiException(error_msg={"error": "rejected"})
    )
    adapter = make_legacy(raw)

    with pytest.raises(PolyApiException):
        await adapter.place_market_order("tok", "SELL", 10.0)


# --- Legacy adapter: single cancel_order must NOT swallow ----------------------
# The requote latch depends on cancel_order RAISING so requote_leg's try/except can
# detect a failed pre-cancel. Only the BATCH cancel_orders swallows.


async def test_legacy_cancel_order_propagates_exception():
    from py_clob_client_v2.exceptions import PolyApiException

    raw = MagicMock()
    raw.cancel_order = MagicMock(side_effect=PolyApiException(error_msg={"error": "boom"}))
    adapter = make_legacy(raw)

    with pytest.raises(PolyApiException):
        await adapter.cancel_order("oid-1")


async def test_legacy_cancel_all_propagates_exception():
    from py_clob_client_v2.exceptions import PolyApiException

    raw = MagicMock()
    raw.cancel_all = MagicMock(side_effect=PolyApiException(error_msg={"error": "boom"}))
    adapter = make_legacy(raw)

    with pytest.raises(PolyApiException):
        await adapter.cancel_all()
