from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator

from app.types import OrderSide


def normalize_outcome(value: object) -> object:
    # CLOB WS sends title-case "Yes"/"No"; strip+upper to canonical "YES"/"NO" at the boundary.
    return value.strip().upper() if isinstance(value, str) else value


Outcome = Annotated[str, BeforeValidator(normalize_outcome)]


class LimitOrder(BaseModel):
    token_id: str
    side: OrderSide
    size: float
    price: float


class BookLevel(BaseModel):
    price: Decimal
    size: Decimal


class OrderBook(BaseModel):
    market: str
    asset_id: str
    timestamp: datetime
    bids: list[BookLevel]
    asks: list[BookLevel]
    min_order_size: Decimal
    tick_size: Decimal
    neg_risk: bool
    hash: str


TradeStatus = Literal["MATCHED", "MINED", "CONFIRMED", "RETRYING", "FAILED"]


class UserTradeMakerOrder(BaseModel):
    asset_id: str
    order_id: str
    matched_amount: Decimal
    outcome: Outcome
    owner: str
    price: Decimal


class UserTrade(BaseModel):
    event_type: Literal["trade"]
    id: str
    asset_id: str
    market: str
    side: OrderSide
    price: Decimal
    size: Decimal
    outcome: Outcome
    status: TradeStatus
    timestamp: str
    maker_orders: list[UserTradeMakerOrder]
    taker_order_id: str


class BestBidAsk(BaseModel):
    event_type: Literal["best_bid_ask"]
    market: str
    asset_id: str
    best_bid: Decimal
    best_ask: Decimal
    spread: Decimal
    timestamp: str


class TickSizeChange(BaseModel):
    event_type: Literal["tick_size_change"]
    market: str
    asset_id: str
    old_tick_size: Decimal
    new_tick_size: Decimal
    timestamp: str


class BookSnapshot(BaseModel):
    event_type: Literal["book"]
    market: str
    asset_id: str
    bids: list[BookLevel]
    asks: list[BookLevel]
    timestamp: str
    hash: str = ""


class PriceLevelChange(BaseModel):
    asset_id: str
    price: Decimal
    size: Decimal
    side: OrderSide
    best_bid: Decimal | None = None
    best_ask: Decimal | None = None
    hash: str = ""


class PriceChange(BaseModel):
    event_type: Literal["price_change"]
    market: str
    price_changes: list[PriceLevelChange]
    timestamp: str


class LastTradePrice(BaseModel):
    event_type: Literal["last_trade_price"]
    asset_id: str
    price: Decimal
    size: Decimal
    side: str = ""
    market: str = ""
    timestamp: str = ""


MarketEvent = BestBidAsk | TickSizeChange | BookSnapshot | PriceChange | LastTradePrice
