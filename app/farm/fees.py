from decimal import ROUND_HALF_UP, Decimal

FEE_QUANTUM = Decimal("0.00001")


def taker_fee(size: Decimal, price: Decimal, rate: Decimal) -> Decimal:
    """USDC taker fee on a `size`-share trade at `price` for a market whose taker
    feeRate is `rate` (e.g. 0.07 for crypto). Returns 0 for fee-free markets, dust
    trades, or degenerate prices (<=0 or >=1, where the fee is zero anyway)."""
    if rate <= 0 or size <= 0 or price <= 0 or price >= 1:
        return Decimal(0)
    fee = size * rate * price * (Decimal(1) - price)
    return fee.quantize(FEE_QUANTUM, rounding=ROUND_HALF_UP)
