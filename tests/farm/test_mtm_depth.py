"""Depth-aware mark-to-market kill: value held inventory by walking the LIVE book bids, not the
single best bid. In a thin book the best-bid mark over-values the position and trips the $5 session
cap too late (or never); the depth walk values it honestly. The kill still uses max_session_loss —
it's the whole-account stop, distinct from the per-fill max_fill_loss guard."""

from decimal import Decimal

from app.bot.schemas import BookLevel
from app.farm.exit_cost import liquidation_proceeds
from app.farm.kill_switch import should_kill, unrealized_loss
from app.farm.schemas import FarmState, LiveBook

# ── liquidation_proceeds (the depth walk) ─────────────────────────────────────


def test_liquidation_proceeds_walks_levels_best_first():
    bids = [
        BookLevel(price=Decimal("0.40"), size=Decimal("10")),
        BookLevel(price=Decimal("0.50"), size=Decimal("10")),
    ]
    # sell 15: 10 @ 0.50 = 5.0, then 5 @ 0.40 = 2.0 → 7.0
    assert liquidation_proceeds(bids, Decimal("15")) == Decimal("7.0")


def test_liquidation_proceeds_strands_unfillable_size_at_zero():
    bids = [BookLevel(price=Decimal("0.50"), size=Decimal("10"))]
    # sell 100 into only 10 of depth: 10 @ 0.50 = 5.0; the other 90 fetch $0
    assert liquidation_proceeds(bids, Decimal("100")) == Decimal("5.0")


def test_liquidation_proceeds_zero_size():
    assert liquidation_proceeds(
        [BookLevel(price=Decimal("0.5"), size=Decimal("9"))], Decimal("0")
    ) == Decimal("0")


# ── unrealized_loss against the live book ─────────────────────────────────────


def test_unrealized_loss_uses_live_book_depth(farm_state: FarmState):
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("100")
    pos.yes_cost_basis = Decimal("50")  # avg 0.50
    pos.yes_best_bid = Decimal("0.49")  # top of book still looks fine
    # ...but the book under it is hollow: 5 @ 0.49, the rest only at 0.10.
    farm_state.live_books[pos.market.yes_token_id] = LiveBook(
        bids={Decimal("0.49"): Decimal("5"), Decimal("0.10"): Decimal("1000")}
    )
    # proceeds = 5*0.49 + 95*0.10 = 2.45 + 9.50 = 11.95 → loss = 50 - 11.95 = 38.05
    assert unrealized_loss(farm_state) == Decimal("38.05")


def test_live_book_overrides_best_bid_when_both_present(farm_state: FarmState):
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("100")
    pos.yes_cost_basis = Decimal("50")
    pos.yes_best_bid = Decimal("0.49")  # would give a $1 loss on its own
    farm_state.live_books[pos.market.yes_token_id] = LiveBook(
        bids={Decimal("0.49"): Decimal("5"), Decimal("0.10"): Decimal("1000")}
    )
    # The live-book depth ($38.05) is used, not the best-bid mark ($1).
    assert unrealized_loss(farm_state) == Decimal("38.05")


def test_falls_back_to_best_bid_without_live_book(farm_state: FarmState):
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("100")
    pos.yes_cost_basis = Decimal("50")
    pos.yes_best_bid = Decimal("0.49")
    # No live book seeded → top-of-book mark: 50 - 100*0.49 = 1.0
    assert unrealized_loss(farm_state) == Decimal("1.0")


# ── the headline: depth mark trips the $5 kill where best-bid alone would not ──


def test_thin_book_trips_kill_that_best_bid_would_miss(farm_state: FarmState):
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("100")
    pos.yes_cost_basis = Decimal("50")
    pos.yes_best_bid = Decimal("0.49")  # cap is $5 (fixture)

    # Best-bid-only mark = $1 loss → would NOT trip the $5 cap.
    assert should_kill(farm_state) is False

    # Same position, but the live book reveals the depth is gone → true loss $38 → trips.
    farm_state.live_books[pos.market.yes_token_id] = LiveBook(
        bids={Decimal("0.49"): Decimal("5"), Decimal("0.10"): Decimal("1000")}
    )
    assert should_kill(farm_state) is True


def test_deep_book_marks_near_flat_no_kill(farm_state: FarmState):
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("100")
    pos.yes_cost_basis = Decimal("50")
    pos.yes_best_bid = Decimal("0.50")
    # Plenty of depth at 0.50 → sells all 100 at 0.50 → $0 loss → no kill.
    farm_state.live_books[pos.market.yes_token_id] = LiveBook(
        bids={Decimal("0.50"): Decimal("500")}
    )
    assert unrealized_loss(farm_state) == Decimal("0")
    assert should_kill(farm_state) is False
