import re
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field
from pydantic_core import PydanticCustomError

from app.types import CanonicalWalletId, NonNegativeDecimal, WalletId

PROXY_ADDRESS_RE = re.compile(r"^(0x)?[0-9a-fA-F]{40}$")
PRIVATE_KEY_RE = re.compile(r"^(0x)?[0-9a-fA-F]{64}$")


def check_proxy_address(v: str) -> str:
    if not PROXY_ADDRESS_RE.match(v):
        raise PydanticCustomError(
            "invalid_proxy_address",
            "Not a valid Polymarket address",
        )
    return v


def check_private_key(v: str) -> str:
    if not PRIVATE_KEY_RE.match(v):
        raise PydanticCustomError(
            "invalid_private_key",
            "Not a valid private key",
        )
    return v


ProxyAddress = Annotated[str, AfterValidator(check_proxy_address)]
PrivateKey = Annotated[str, AfterValidator(check_private_key)]


# ── Client → server ─────────────────────────────────────────────────────────


class WalletRegisterMessage(BaseModel):
    type: Literal["wallet_register"]
    wallet_id: CanonicalWalletId | None = None  # omit to append a new wallet
    proxy_address: ProxyAddress
    private_key: PrivateKey


class WalletRemoveMessage(BaseModel):
    type: Literal["wallet_remove"]
    wallet_id: WalletId  # permissive: any stored id must stay deletable, incl. legacy "A"/"B"


class WalletListRequestMessage(BaseModel):
    type: Literal["wallet_list"]


ClientMessage = Annotated[
    WalletRegisterMessage | WalletRemoveMessage | WalletListRequestMessage,
    Field(discriminator="type"),
]


# ── Server → client ─────────────────────────────────────────────────────────


class WalletListEntry(BaseModel):
    wallet_id: WalletId
    proxy_address: str
    balance: NonNegativeDecimal | None = None


class WalletListResponse(BaseModel):
    type: Literal["wallet_list"] = "wallet_list"
    wallets: list[WalletListEntry]


class WalletErrorResponse(BaseModel):
    type: Literal["wallet_error"] = "wallet_error"
    reason: str
