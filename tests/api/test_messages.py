from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.api.blacklist.messages import (
    BlacklistAddMessage,
    BlacklistClearMessage,
    BlacklistListRequestMessage,
    BlacklistRemoveMessage,
)
from app.api.farm.messages import (
    FarmCancelledEvent,
    FarmCancelMessage,
    FarmCreateMessage,
    FarmErrorEvent,
    FarmKilledEvent,
    FarmStartedEvent,
    FarmSummaryEvent,
    OrderCancelledEvent,
    OrderFilledEvent,
    OrderPlacedEvent,
)
from app.api.messages import client_message_adapter
from app.api.wallet.messages import (
    WalletErrorResponse,
    WalletListEntry,
    WalletListRequestMessage,
    WalletListResponse,
    WalletRegisterMessage,
    WalletRemoveMessage,
    check_private_key,
    check_proxy_address,
)
from app.farm.schemas import FarmConfig, FarmFilters

VALID_ADDR = "f" * 40
VALID_KEY = "a" * 64


def make_filters() -> FarmFilters:
    return FarmFilters(
        vol_min=Decimal(0),
        vol_max=Decimal(100000),
        liq_min=Decimal(0),
        liq_max=Decimal(100000),
        spread_min=Decimal(0),
        spread_max=Decimal(100),
        reward_min=Decimal(0),
        time_remaining="all",
        created_date="all",
        change_24h="all",
    )


# ── check_proxy_address ──────────────────────────────────────────────────────


def test_check_proxy_address_accepts_without_prefix():
    assert check_proxy_address(VALID_ADDR) == VALID_ADDR


def test_check_proxy_address_accepts_with_0x_prefix():
    val = "0x" + VALID_ADDR
    assert check_proxy_address(val) == val


def test_check_proxy_address_rejects_wrong_length():
    with pytest.raises(Exception) as exc:
        check_proxy_address("f" * 39)
    assert "Not a valid Polymarket address" in str(exc.value)


def test_check_proxy_address_rejects_non_hex():
    with pytest.raises(Exception) as exc:
        check_proxy_address("z" * 40)
    assert "Not a valid Polymarket address" in str(exc.value)


# ── check_private_key ────────────────────────────────────────────────────────


def test_check_private_key_accepts_without_prefix():
    assert check_private_key(VALID_KEY) == VALID_KEY


def test_check_private_key_accepts_with_0x_prefix():
    val = "0x" + VALID_KEY
    assert check_private_key(val) == val


def test_check_private_key_rejects_wrong_length():
    with pytest.raises(Exception) as exc:
        check_private_key("a" * 63)
    assert "Not a valid private key" in str(exc.value)


def test_check_private_key_rejects_non_hex():
    with pytest.raises(Exception) as exc:
        check_private_key("g" * 64)
    assert "Not a valid private key" in str(exc.value)


# ── validators through the model (exercise the Annotated path) ────────────────


def test_wallet_register_rejects_bad_proxy_address():
    with pytest.raises(ValidationError) as exc:
        WalletRegisterMessage(
            type="wallet_register",
            wallet_id="wallet1",
            proxy_address="nope",
            private_key=VALID_KEY,
        )
    assert "Not a valid Polymarket address" in str(exc.value)


def test_wallet_register_rejects_bad_private_key():
    with pytest.raises(ValidationError) as exc:
        WalletRegisterMessage(
            type="wallet_register",
            wallet_id="wallet1",
            proxy_address=VALID_ADDR,
            private_key="nope",
        )
    assert "Not a valid private key" in str(exc.value)


def test_wallet_register_accepts_valid_credentials():
    msg = WalletRegisterMessage(
        type="wallet_register",
        wallet_id="wallet1",
        proxy_address="0x" + VALID_ADDR,
        private_key="0x" + VALID_KEY,
    )
    assert msg.wallet_id == "wallet1"
    assert msg.proxy_address == "0x" + VALID_ADDR


# ── client_message_adapter dispatch ──────────────────────────────────────────


def test_adapter_dispatches_wallet_register():
    msg = client_message_adapter.validate_python(
        {
            "type": "wallet_register",
            "wallet_id": "wallet1",
            "proxy_address": VALID_ADDR,
            "private_key": VALID_KEY,
        }
    )
    assert isinstance(msg, WalletRegisterMessage)


def test_adapter_dispatches_wallet_remove():
    msg = client_message_adapter.validate_python({"type": "wallet_remove", "wallet_id": "B"})
    assert isinstance(msg, WalletRemoveMessage)
    assert msg.wallet_id == "B"


def test_adapter_dispatches_wallet_list():
    msg = client_message_adapter.validate_python({"type": "wallet_list"})
    assert isinstance(msg, WalletListRequestMessage)


def test_adapter_dispatches_farm_create():
    msg = client_message_adapter.validate_python(
        {
            "type": "farm_create",
            "filters": make_filters().model_dump(mode="json"),
            "bankroll": "100",
            "max_session_loss": "5",
        }
    )
    assert isinstance(msg, FarmCreateMessage)
    assert msg.bankroll == Decimal("100")


def test_adapter_dispatches_farm_cancel():
    msg = client_message_adapter.validate_python({"type": "farm_cancel"})
    assert isinstance(msg, FarmCancelMessage)


def test_adapter_dispatches_blacklist_add():
    msg = client_message_adapter.validate_python(
        {"type": "blacklist_add", "market_url": "https://polymarket.com/event/x"}
    )
    assert isinstance(msg, BlacklistAddMessage)
    assert msg.market_url == "https://polymarket.com/event/x"


def test_adapter_dispatches_blacklist_remove():
    msg = client_message_adapter.validate_python(
        {"type": "blacklist_remove", "condition_id": "0xc1"}
    )
    assert isinstance(msg, BlacklistRemoveMessage)
    assert msg.condition_id == "0xc1"


def test_adapter_dispatches_blacklist_list():
    msg = client_message_adapter.validate_python({"type": "blacklist_list"})
    assert isinstance(msg, BlacklistListRequestMessage)


def test_adapter_dispatches_blacklist_clear():
    msg = client_message_adapter.validate_python({"type": "blacklist_clear"})
    assert isinstance(msg, BlacklistClearMessage)


def test_adapter_rejects_unknown_type():
    with pytest.raises(ValidationError):
        client_message_adapter.validate_python({"type": "not_a_real_type"})


# ── client message models constructed directly ───────────────────────────────


def test_farm_create_message_is_farm_config_subclass():
    msg = FarmCreateMessage(
        type="farm_create",
        filters=make_filters(),
        bankroll=Decimal("100"),
        max_session_loss=Decimal("5"),
    )
    assert isinstance(msg, FarmConfig)
    assert msg.type == "farm_create"


def test_wallet_remove_message():
    msg = WalletRemoveMessage(type="wallet_remove", wallet_id="A")
    assert msg.wallet_id == "A"


def test_wallet_list_request_message():
    msg = WalletListRequestMessage(type="wallet_list")
    assert msg.type == "wallet_list"


# ── server → client event models: instantiate + model_dump(mode="json") ──────


def test_wallet_list_response_dump():
    resp = WalletListResponse(wallets=[WalletListEntry(wallet_id="A", proxy_address=VALID_ADDR)])
    dumped = resp.model_dump(mode="json")
    assert dumped["type"] == "wallet_list"
    assert dumped["wallets"][0]["wallet_id"] == "A"
    assert dumped["wallets"][0]["proxy_address"] == VALID_ADDR


def test_wallet_error_response_dump():
    dumped = WalletErrorResponse(reason="boom").model_dump(mode="json")
    assert dumped == {"type": "wallet_error", "reason": "boom"}


def test_farm_started_event_dump():
    assert FarmStartedEvent().model_dump(mode="json") == {"type": "farm_started"}


def test_farm_cancelled_event_dump():
    assert FarmCancelledEvent().model_dump(mode="json") == {"type": "farm_cancelled"}


def test_farm_error_event_dump():
    assert FarmErrorEvent(reason="oops").model_dump(mode="json") == {
        "type": "farm_error",
        "reason": "oops",
    }


def test_farm_killed_event_dump():
    dumped = FarmKilledEvent(reason="max_session_loss", session_loss=Decimal("12.5")).model_dump(
        mode="json"
    )
    assert dumped["type"] == "farm_killed"
    assert dumped["reason"] == "max_session_loss"
    # mode="json" serializes Decimal as a string.
    assert dumped["session_loss"] == "12.5"


def test_order_placed_event_dump():
    dumped = OrderPlacedEvent(
        market_id="m1",
        slug="slug",
        question="q?",
        outcome="YES",
        side="BUY",
        price=Decimal("0.5"),
        size=Decimal("100"),
        capital_locked=Decimal("50"),
        order_id="oid-1",
    ).model_dump(mode="json")
    assert dumped["type"] == "order_placed"
    assert dumped["side"] == "BUY"
    assert dumped["price"] == "0.5"
    assert dumped["order_id"] == "oid-1"


def test_order_cancelled_event_dump():
    dumped = OrderCancelledEvent(
        market_id="m1",
        slug="slug",
        outcome="NO",
        order_id="oid-2",
        reason="requote",
    ).model_dump(mode="json")
    assert dumped["type"] == "order_cancelled"
    assert dumped["reason"] == "requote"


def test_order_filled_event_dump():
    dumped = OrderFilledEvent(
        market_id="m1",
        slug="slug",
        outcome="YES",
        side="SELL",
        price=Decimal("0.6"),
        size=Decimal("10"),
        shares_after=Decimal("90"),
        cost_basis_after=Decimal("45"),
    ).model_dump(mode="json")
    assert dumped["type"] == "order_filled"
    assert dumped["side"] == "SELL"
    assert dumped["shares_after"] == "90"


def test_farm_summary_event_dump_with_defaults():
    dumped = FarmSummaryEvent(
        total_volume=Decimal("1000"),
        total_rewards=Decimal("5"),
        active_markets=3,
        session_loss=Decimal("-2"),
        max_session_loss=Decimal("5"),
    ).model_dump(mode="json")
    assert dumped["type"] == "farm_summary"
    # Defaults executed.
    assert dumped["rewards_per_hour"] == "0"
    assert dumped["rewards_per_day"] == "0"
    assert dumped["elapsed_seconds"] == 0
    assert dumped["active_markets"] == 3
    assert dumped["session_loss"] == "-2"


def test_farm_summary_event_dump_with_overrides():
    dumped = FarmSummaryEvent(
        total_volume=Decimal("1000"),
        total_rewards=Decimal("5"),
        rewards_per_hour=Decimal("0.25"),
        rewards_per_day=Decimal("6"),
        elapsed_seconds=3600,
        active_markets=2,
        session_loss=Decimal("1.5"),
        max_session_loss=Decimal("5"),
    ).model_dump(mode="json")
    assert dumped["rewards_per_hour"] == "0.25"
    assert dumped["rewards_per_day"] == "6"
    assert dumped["elapsed_seconds"] == 3600


# ── wallet id shape (canonical on register, permissive on remove) ────────────


def test_wallet_register_allows_omitted_wallet_id():
    msg = WalletRegisterMessage(
        type="wallet_register",
        proxy_address=VALID_ADDR,
        private_key=VALID_KEY,
    )
    assert msg.wallet_id is None


def test_wallet_register_accepts_canonical_wallet_id():
    msg = WalletRegisterMessage(
        type="wallet_register",
        wallet_id="wallet12",
        proxy_address=VALID_ADDR,
        private_key=VALID_KEY,
    )
    assert msg.wallet_id == "wallet12"


@pytest.mark.parametrize(
    "wallet_id",
    [
        "A",
        "B",
        "Wallet1",
        "WALLET1",
        "wallet0",
        "wallet01",
        "",
        "wallet",
        "wallet10000",
        "1",
        # Python's $ matches before a trailing newline; \Z is what actually anchors.
        "wallet1\n",
        "wallet1\r\n",
    ],
)
def test_wallet_register_rejects_malformed_wallet_ids(wallet_id):
    with pytest.raises(ValidationError) as exc:
        WalletRegisterMessage(
            type="wallet_register",
            wallet_id=wallet_id,
            proxy_address=VALID_ADDR,
            private_key=VALID_KEY,
        )
    assert "wallet_id must look like" in str(exc.value)


@pytest.mark.parametrize("wallet_id", ["A", "wallet3", "wallet9999"])
def test_wallet_remove_accepts_legacy_and_canonical_ids(wallet_id):
    # Remove stays permissive so a pre-migration row can still be deleted.
    msg = WalletRemoveMessage(type="wallet_remove", wallet_id=wallet_id)
    assert msg.wallet_id == wallet_id


def test_adapter_dispatches_wallet_register_without_wallet_id():
    msg = client_message_adapter.validate_python(
        {
            "type": "wallet_register",
            "proxy_address": VALID_ADDR,
            "private_key": VALID_KEY,
        }
    )
    assert isinstance(msg, WalletRegisterMessage)
    assert msg.wallet_id is None
