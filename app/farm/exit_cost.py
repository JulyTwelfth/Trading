from decimal import Decimal

from app.bot.schemas import BookLevel, OrderBook
from app.constants import (
    VACUUM_ASK_HOLD_TOL,
    VACUUM_BID_DROP,
    VACUUM_CONCESSION,
    VACUUM_MIN_SPREAD,
    VACUUM_SPREAD_BAND_FACTOR,
)
from app.farm.fees import taker_fee
from app.farm.quoting import compute_quote, size_per_market
from app.farm.schemas import Market
from app.types import QuoteDepth


def top_of_book_fill_loss(market: Market, our_price: Decimal, best_bid: Decimal) -> Decimal:
    """USDC lost if filled at our_price then immediately sold at the live best_bid — the
    depth-free, top-of-book estimate (vs immediate_sell_loss, which walks the full book).
    Includes the exit taker fee, floored at our_price so a bid above entry can't offset it."""
    size = size_per_market(market)
    spread_loss = size * max(Decimal(0), our_price - best_bid)
    exit_px = min(best_bid, our_price)
    return spread_loss + taker_fee(size, exit_px, market.taker_fee_rate)


def immediate_sell_loss(
    bids: list[BookLevel], entry: Decimal, size: Decimal, fee_rate: Decimal = Decimal(0)
) -> Decimal:
    """USDC lost if we are filled `size` shares at `entry` and immediately market-sell them.
    Sells into bids at/below entry; unabsorbed shares are stranded ($0). `fee_rate` charges
    the taker fee per fill at its execution price; stranded shares pay no fee."""
    if entry <= 0 or size <= 0:
        return Decimal(0)
    remaining = size
    proceeds = Decimal(0)
    for level in sorted((b for b in bids if b.price <= entry), key=lambda x: -x.price):
        take = min(remaining, level.size)
        proceeds += take * level.price - taker_fee(take, level.price, fee_rate)
        remaining -= take
        if remaining <= 0:
            break
    return max(Decimal(0), size * entry - proceeds)


def liquidation_proceeds(bids: list[BookLevel], size: Decimal) -> Decimal:
    """USDC from market-selling `size` shares into bids, best price first (unabsorbed = $0).
    Unlike immediate_sell_loss, doesn't filter by entry — values HELD inventory."""
    if size <= 0:
        return Decimal(0)
    remaining = size
    proceeds = Decimal(0)
    for level in sorted(bids, key=lambda x: -x.price):
        take = min(remaining, level.size)
        proceeds += take * level.price
        remaining -= take
        if remaining <= 0:
            break
    return proceeds


def exit_vacuum_price(
    bids: list[BookLevel],
    asks: list[BookLevel],
    entry_px: Decimal,
    tick: Decimal,
    reward_band: Decimal,
) -> Decimal | None:
    """Shadow-mode vacuum classifier — pure and read-only, takes NO action. Decides whether a
    "wait for the bid to come back, then rest at a concession off the ask" exit would have a
    price to fire at, versus a real crash where the book is genuinely empty and dumping is the
    only option. Gates 1-4 below are the classifier: any one failing means "would-dump"
    (return None). The fifth check — that the computed rest price still sits above the bid — is a
    guard, not a gate: gates 2 and 3 already force rest_px at least
    min(VACUUM_MIN_SPREAD - VACUUM_CONCESSION, VACUUM_BID_DROP) clear of best_bid, so it cannot
    fire while those constants keep that margin positive. It stays because retuning them
    otherwise fails silently by resting an exit at or below the bid, and it is why this function
    reports 97% line coverage rather than 100%. `tick` is accepted for interface symmetry with
    the eventual live exit-price computation but is not needed by the gates themselves.

    Returns the price it would rest the exit at (a value in (best_bid, entry_px]), or None.
    """
    best_bid = max((lvl.price for lvl in bids), default=Decimal(0))
    best_ask = min((lvl.price for lvl in asks), default=None)

    # GATE 1 (empty book): no bids at all — the one clean real-crash signal (Brazil).
    if best_bid <= 0:
        return None

    # GATE 2 (spread floor): a spread this tight isn't a vacuum, just a normal wide-ish book
    # (Canvas-high).
    if best_ask is None:
        return None
    spread = best_ask - best_bid
    if spread < max(VACUUM_MIN_SPREAD, VACUUM_SPREAD_BAND_FACTOR * reward_band):
        return None

    # GATE 3 (bid collapsed): the bid hasn't actually dropped enough to call this a vacuum.
    if best_bid > entry_px - VACUUM_BID_DROP:
        return None

    # GATE 4 (ask held): the ask side must still be resting near entry — otherwise this is a
    # real directional move, not a liquidity vacuum.
    if best_ask < entry_px - VACUUM_ASK_HOLD_TOL:
        return None

    rest_px = min(entry_px, best_ask - VACUUM_CONCESSION)
    if rest_px <= best_bid:
        return None
    return rest_px


def market_exit_loss(
    market: Market,
    yes_book: OrderBook | None,
    yes_mid: Decimal | None,
    no_book: OrderBook | None,
    no_mid: Decimal | None,
    quote_depth: QuoteDepth,
) -> Decimal | None:
    """Worst-leg immediate-sell loss (USDC) at our quote price and size_per_market. Returns
    None when neither leg is quoteable/has a book."""
    size = size_per_market(market)
    losses = []
    for book, mid in ((yes_book, yes_mid), (no_book, no_mid)):
        if book is None or mid is None:
            continue
        quote = compute_quote(market, mid, quote_depth)
        if quote is None:
            continue
        losses.append(immediate_sell_loss(book.bids, quote[0], size, market.taker_fee_rate))
    if not losses:
        return None
    return max(losses)
