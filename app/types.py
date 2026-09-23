import re
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import AfterValidator, Field
from pydantic_core import PydanticCustomError

# \Z, not $: Python's $ also matches before a trailing newline, so "wallet1\n" would
# pass validation and land as a distinct row that renders identically in the UI.
WALLET_ID_RE = re.compile(r"^wallet[1-9][0-9]{0,3}\Z")
MAX_WALLET_SLOT = 9999


def check_wallet_id(v: str) -> str:
    if not WALLET_ID_RE.match(v):
        raise PydanticCustomError(
            "invalid_wallet_id",
            "wallet_id must look like wallet1, wallet2, …",
        )
    return v


# Stored ids stay permissive so pre-migration "A"/"B" rows still load; only new
# registrations are held to the canonical shape.
WalletId = Annotated[str, Field(min_length=1, max_length=64)]
CanonicalWalletId = Annotated[str, AfterValidator(check_wallet_id)]

OrderSide = Literal["BUY", "SELL"]

TimeRemainingFilter = Literal["all", "12h", "1d", "7d", "30d"]
CreatedDateFilter = Literal["all", "1d", "7d", "30d"]
Change24hFilter = Literal["all", "lt10", "gt10", "gt20"]
Range24hFilter = Literal["all", "lt5", "lt10", "lt20"]
QuoteDepth = Literal["safe", "normal", "aggressive"]

PositiveDecimal = Annotated[Decimal, Field(gt=Decimal("0"))]
NonNegativeDecimal = Annotated[Decimal, Field(ge=Decimal("0"))]
ProbabilityDecimal = Annotated[Decimal, Field(ge=Decimal("0"), le=Decimal("1"))]
