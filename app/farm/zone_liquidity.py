from decimal import Decimal

from app.bot.schemas import OrderBook
from app.farm.schemas import Market


def zone_distribution(markets: list[Market]) -> tuple[Decimal, Decimal, Decimal] | None:
    """(lowest, typical, highest) zone liquidity across markets, or None if none computed.

    "typical" is the median. Powers a one-line tick log so an operator can eyeball where
    real markets fall and pick a sensible zone_liq_max threshold without DEBUG noise.
    """
    values = sorted(m.zone_liquidity for m in markets if m.zone_liquidity is not None)
    if not values:
        return None
    median = values[len(values) // 2]
    return values[0], median, values[-1]


def book_zone_liquidity(book: OrderBook, midpoint: Decimal, max_spread_price: Decimal) -> Decimal:
    """USD notional of resting orders (both sides) within max_spread of the midpoint.

    A resting order counts as in-zone when its price is within max_spread_price of the
    midpoint — the same band compute_quote() targets. We sum price*size so the figure is
    in dollars, matching how an operator thinks about "liquidity sitting in the zone".
    """
    total = Decimal(0)
    for level in (*book.bids, *book.asks):
        if abs(level.price - midpoint) <= max_spread_price:
            total += level.price * level.size
    return total


def book_bid_depth(book: OrderBook, midpoint: Decimal, max_spread_price: Decimal) -> Decimal:
    """Shares resting on the BID side within max_spread of the midpoint — the depth a fill on this
    leg could be sold back INTO. The one-sided (bids-only) companion to book_zone_liquidity: a
    market can pass the both-sides zone/aggregate-liquidity checks yet have almost no bids to exit a
    fill into (the one-sided-book strand). Summed as shares so it compares directly to our order
    size. Logged for analysis and, when FarmFilters.min_bid_depth_mult is set, used as the gating
    bid_depth filter in first_failing_filter."""
    return sum(
        (level.size for level in book.bids if abs(level.price - midpoint) <= max_spread_price),
        Decimal(0),
    )


def zone_liquidity_usd(
    market: Market,
    yes_book: OrderBook,
    yes_mid: Decimal,
    no_book: OrderBook,
    no_mid: Decimal,
) -> Decimal:
    """Total USD liquidity resting inside the reward zone across both outcome books.

    Polymarket pays a fixed daily reward pool per market, split by each maker's share of
    qualifying liquidity inside rewards_max_spread of the midpoint. Summing the dollar
    value of every resting order in that band (both bid and ask sides, on both the YES
    and NO books) approximates the pool you are competing for: more zone liquidity means
    a smaller reward share for the same capital. This is the crowding measure the bot
    filters on — distinct from Gamma's whole-book `liquidity`, which counts depth far
    outside the band.
    """
    max_spread_price = market.rewards_max_spread_cents / Decimal(100)
    return book_zone_liquidity(yes_book, yes_mid, max_spread_price) + book_zone_liquidity(
        no_book, no_mid, max_spread_price
    )
