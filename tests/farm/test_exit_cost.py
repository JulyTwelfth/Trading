"""immediate_sell_loss + market_exit_loss: the size-aware exit-cost model behind the
max_fill_loss filter. Loss = cost - proceeds when we dump `size` shares into bids at/below
our entry; unsellable shares are valued at $0 (stranded), matching the live FAK write-off."""

from datetime import datetime, timezone
from decimal import Decimal

from app.bot.schemas import BookLevel, OrderBook
from app.farm.exit_cost import immediate_sell_loss, market_exit_loss, top_of_book_fill_loss
from app.farm.schemas import Market

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _bids(*levels):
    return [BookLevel(price=Decimal(p), size=Decimal(s)) for p, s in levels]


def _book(bids, asset_id="a"):
    return OrderBook(
        market="m",
        asset_id=asset_id,
        timestamp=NOW,
        bids=bids,
        asks=[],
        min_order_size=Decimal("1"),
        tick_size=Decimal("0.01"),
        neg_risk=False,
        hash="h",
    )


def _market(**ov):
    base = dict(
        condition_id="c",
        slug="s",
        question="?",
        yes_token_id="y",
        no_token_id="n",
        rewards_max_spread_cents=Decimal("3"),
        rewards_min_size=Decimal("100"),
        rewards_rate_per_day=Decimal("5"),
        tick_size=Decimal("0.01"),
        min_order_size=Decimal("5"),
        end_date=datetime(2030, 1, 1, tzinfo=timezone.utc),
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        volume_24h=Decimal("100"),
        liquidity=Decimal("100"),
        spread_cents=Decimal("1"),
        price_change_24h=Decimal("0"),
    )
    base.update(ov)
    return Market(**base)


# ── immediate_sell_loss ──────────────────────────────────────────────────────


def test_deep_book_at_entry_no_loss():
    # 100 shares all clear at the entry price → recover full cost.
    assert immediate_sell_loss(_bids(("0.50", "1000")), Decimal("0.50"), Decimal("100")) == 0


def test_walk_down_loses_the_gap():
    # 40 @ 0.49 + 60 @ 0.45 = 46.60 ; cost 50 ; loss 3.40
    loss = immediate_sell_loss(
        _bids(("0.49", "40"), ("0.45", "1000")), Decimal("0.50"), Decimal("100")
    )
    assert loss == Decimal("3.40")


def test_stranded_shares_valued_at_zero():
    # only 30 sellable @0.49 = 14.70 ; 70 unsold worth $0 ; cost 50 ; loss 35.30
    loss = immediate_sell_loss(_bids(("0.49", "30")), Decimal("0.50"), Decimal("100"))
    assert loss == Decimal("35.30")


def test_empty_book_is_full_loss():
    assert immediate_sell_loss([], Decimal("0.50"), Decimal("100")) == Decimal("50.00")


def test_bids_above_entry_ignored():
    # bids above our entry are gone by the time a seller has swept down to fill us.
    assert immediate_sell_loss(_bids(("0.60", "1000")), Decimal("0.50"), Decimal("100")) == Decimal(
        "50.00"
    )


def test_bid_exactly_at_entry_is_inclusive():
    assert immediate_sell_loss(_bids(("0.50", "1000")), Decimal("0.50"), Decimal("100")) == 0


def test_zero_entry_or_size_is_zero():
    assert immediate_sell_loss(_bids(("0.50", "100")), Decimal("0"), Decimal("100")) == 0
    assert immediate_sell_loss(_bids(("0.50", "100")), Decimal("0.50"), Decimal("0")) == 0


# ── market_exit_loss (worst leg) ─────────────────────────────────────────────


def test_market_exit_loss_takes_worst_leg():
    # aggressive entry = mid-1tick = 0.49 ; size = max(5,100) = 100.
    # YES deep @0.49 → loss 0 ; NO thin @0.40 → 100*0.40=40 vs cost 49 → loss 9. worst = 9.
    m = _market()
    yes_book = _book(_bids(("0.49", "10000")), "y")
    no_book = _book(_bids(("0.40", "10000")), "n")
    loss = market_exit_loss(m, yes_book, Decimal("0.50"), no_book, Decimal("0.50"), "aggressive")
    assert loss == Decimal("9.00")


def test_market_exit_loss_none_when_no_books():
    loss = market_exit_loss(_market(), None, Decimal("0.5"), None, Decimal("0.5"), "aggressive")
    assert loss is None


def test_market_exit_loss_uses_present_leg_when_other_missing():
    no_book = _book(_bids(("0.40", "10000")), "n")
    loss = market_exit_loss(_market(), None, None, no_book, Decimal("0.50"), "aggressive")
    assert loss == Decimal("9.00")


def test_market_exit_loss_safe_vs_aggressive_entry_differ():
    # safe entry = far edge = 0.48 ; aggressive = 0.49. Same thin book → different loss.
    m = _market()
    no_book = _book(_bids(("0.40", "10000")), "n")
    safe = market_exit_loss(m, None, None, no_book, Decimal("0.50"), "safe")
    aggr = market_exit_loss(m, None, None, no_book, Decimal("0.50"), "aggressive")
    assert safe == Decimal("8.00")  # cost 48, proceeds 40
    assert aggr == Decimal("9.00")  # cost 49, proceeds 40


# ── property / invariant tests ───────────────────────────────────────────────


def test_loss_bounded_between_zero_and_full_cost():
    # For any book/entry/size, loss is in [0, cost]: you can't recover more than you paid,
    # and you can't lose more than the whole position (unsold shares are worth $0, not <$0).
    entries = [Decimal("0.10"), Decimal("0.50"), Decimal("0.90")]
    sizes = [Decimal("5"), Decimal("100"), Decimal("250")]
    books = [
        [],
        _bids(("0.05", "10")),
        _bids(("0.49", "30"), ("0.20", "1000")),
        _bids(("0.95", "1000")),  # all above a 0.90 entry → ignored
        _bids(("0.50", "100000")),
    ]
    for entry in entries:
        for size in sizes:
            for bids in books:
                loss = immediate_sell_loss(bids, entry, size)
                assert Decimal(0) <= loss <= size * entry


def test_loss_monotonically_decreases_as_book_deepens():
    # Adding sellable depth (at any price <= entry) can only lower the loss, never raise it.
    entry, size = Decimal("0.50"), Decimal("100")
    shallow = _bids(("0.49", "30"))
    deeper = _bids(("0.49", "30"), ("0.48", "1000"))
    deepest = _bids(("0.49", "30"), ("0.48", "1000"), ("0.50", "100000"))
    a = immediate_sell_loss(shallow, entry, size)
    b = immediate_sell_loss(deeper, entry, size)
    c = immediate_sell_loss(deepest, entry, size)
    assert a >= b >= c
    assert c == 0  # enough depth at/above entry → full recovery


def test_top_of_book_fill_loss_zero_when_bid_at_or_above_entry():
    m = _market()
    assert top_of_book_fill_loss(m, Decimal("0.50"), Decimal("0.50")) == 0
    assert top_of_book_fill_loss(m, Decimal("0.50"), Decimal("0.55")) == 0  # bid above entry


def test_top_of_book_fill_loss_is_gap_times_size():
    m = _market()  # size_per_market = max(5, 100) = 100
    assert top_of_book_fill_loss(m, Decimal("0.53"), Decimal("0.50")) == Decimal("3.00")
    assert top_of_book_fill_loss(m, Decimal("0.50"), Decimal("0.49")) == Decimal("1.00")


# ── taker fee folded into the exit-cost estimates (fee-enabled markets) ───────


def test_top_of_book_fill_loss_adds_taker_fee():
    # crypto market: spread 100*(0.53-0.50)=3.00 + taker fee 100*0.07*0.50*0.50=1.75 = 4.75
    m = _market(taker_fee_rate=Decimal("0.07"))
    assert top_of_book_fill_loss(m, Decimal("0.53"), Decimal("0.50")) == Decimal("4.75")


def test_top_of_book_fill_loss_fee_priced_at_entry_when_bid_above():
    # A best_bid above our entry can't fund the exit (it's consumed by the very fill
    # that hits us): spread clamps to 0 and the fee prices at our 0.50 entry
    # (100*0.07*0.50*0.50 = 1.75), not at the doomed 0.53 bid (1.7437).
    m = _market(taker_fee_rate=Decimal("0.07"))
    assert top_of_book_fill_loss(m, Decimal("0.50"), Decimal("0.53")) == Decimal("1.75")


def test_top_of_book_fill_loss_charges_fee_even_when_spread_zero():
    # The point of the fix: a tight book (bid == entry) has zero spread loss but the exit
    # sell is still a taker, so the guard must see the fee — here 100*0.07*0.5*0.5 = 1.75.
    m = _market(taker_fee_rate=Decimal("0.07"))
    assert top_of_book_fill_loss(m, Decimal("0.50"), Decimal("0.50")) == Decimal("1.75")


def test_market_exit_loss_adds_taker_fee():
    # aggressive quote = mid - 1 tick = 0.49; NO thin @0.40 → spread loss 9.00, plus the
    # taker fee priced at the 0.40 fill (not our quote): 100*0.07*0.40*0.60 = 1.68.
    no_book = _book(_bids(("0.40", "10000")), "n")
    free = market_exit_loss(_market(), None, None, no_book, Decimal("0.50"), "aggressive")
    fee = market_exit_loss(
        _market(taker_fee_rate=Decimal("0.07")), None, None, no_book, Decimal("0.50"), "aggressive"
    )
    assert free == Decimal("9.00")
    assert fee == Decimal("10.68")


def test_immediate_sell_loss_fee_priced_at_fill_not_entry():
    # Entry 0.80 but the book fills at 0.40, near the p*(1-p) fee peak: the fee must be
    # 100*0.07*0.40*0.60 = 1.68, not the 100*0.07*0.80*0.20 = 1.12 an entry-priced fee
    # would give. Spread loss 100*(0.80-0.40) = 40.00 on top.
    loss = immediate_sell_loss(
        _bids(("0.40", "1000")), Decimal("0.80"), Decimal("100"), Decimal("0.07")
    )
    assert loss == Decimal("41.68")


def test_immediate_sell_loss_no_fee_on_stranded_shares():
    # Only 30 of 100 shares find a bid; the stranded 70 never trade, so no fee on them.
    # Loss = 70 stranded * 0.50 + fee on the 30 filled (30*0.07*0.50*0.50 = 0.525).
    loss = immediate_sell_loss(
        _bids(("0.50", "30")), Decimal("0.50"), Decimal("100"), Decimal("0.07")
    )
    assert loss == Decimal("35.525")


def test_loss_grows_with_size_on_a_fixed_thin_book():
    # Bigger position into the same thin book → more shares strand → more loss.
    book = _bids(("0.50", "50"))  # 50 shares of exit liquidity AT our entry price
    entry = Decimal("0.50")
    small = immediate_sell_loss(book, entry, Decimal("40"))
    big = immediate_sell_loss(book, entry, Decimal("200"))
    assert small == 0  # 40 <= 50 available at entry → full recovery
    assert big == Decimal("75.00")  # 50 fill @0.50 = 25 ; 150 strand ; cost 100 ; loss 75
    assert big > small
