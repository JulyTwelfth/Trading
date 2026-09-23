from decimal import Decimal


def passes_depth_gate(level_size: Decimal, our_size: Decimal, min_ratio: Decimal) -> bool:
    return level_size >= min_ratio * our_size


def pick_surf_level(
    bids: list[tuple[Decimal, Decimal]],
    our_size: Decimal,
    mid: Decimal,
    max_spread_cents: Decimal,
    min_ratio: Decimal,
) -> tuple[Decimal, Decimal] | None:
    """Return (price, depth) for the HIGHEST in-band bid level deep enough to hide behind, or
    None if nothing in-band qualifies — stand down rather than become the front bid."""
    band = max_spread_cents / Decimal(100)
    for price, size in sorted(bids, key=lambda level: level[0], reverse=True):
        if mid - price >= band:
            break
        if passes_depth_gate(size, our_size, min_ratio):
            return price, size
    return None


def should_resurf(
    age_seconds: float,
    moved_up: bool,
    has_surf_level: bool,
    min_hold_seconds: float,
) -> bool:
    """Re-quote to the back of the queue only when all three hold: aged past min_hold (reward
    banked), moved_up (a trade fired at our price), and a surf level still exists."""
    return moved_up and has_surf_level and age_seconds >= min_hold_seconds


def is_sole_qualifier(
    bids: list[tuple[Decimal, Decimal]],
    rewards_min_size: Decimal,
    mid: Decimal,
    max_spread_cents: Decimal,
    our_price: Decimal | None = None,
    our_size: Decimal = Decimal(0),
) -> bool:
    """True when NO in-band bid level (other than our own resting order) holds a full reward-
    qualifying order (>= rewards_min_size), letting us quote at the safe edge for fill buffer.
    our_price/our_size discount our own resting order on the requote path (default None/0)."""
    band = max_spread_cents / Decimal(100)
    for price, size in bids:
        if mid - price >= band:
            continue
        competing = size - our_size if our_price is not None and price == our_price else size
        if competing >= rewards_min_size:
            return False
    return True
