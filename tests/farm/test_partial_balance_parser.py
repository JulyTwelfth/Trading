"""``parse_partial_balance`` — the legacy partial-balance SELL-rejection parser that feeds the
dust-terminal write-off (Fix B).

It must recognise ONLY the "balance: N, order amount: M" shape (exchange holds fewer shares than
we tried to sell) and return the two figures as share Decimals (micro→shares, /1e6). Everything
else — the overcommit body ("order amount (inc. fees):" alongside "sum of ..."), a SecureClient
error with no parseable numeric body, and outright garbage — must return None so the caller falls
through to its other rejection branches instead of writing off on a mis-parse.
"""

from decimal import Decimal

import pytest
from py_clob_client_v2.exceptions import PolyApiException

from app.exceptions import SecureOrderError
from app.farm.exits import parse_partial_balance


def poly(msg: str) -> PolyApiException:
    return PolyApiException(error_msg={"error": msg})


PARTIAL = (
    "not enough balance / allowance: the balance is not enough "
    "-> balance: {bal}, order amount: {amt}"
)
OVERCOMMIT = (
    "not enough balance / allowance: the balance is not enough -> balance: 26120900, "
    "sum of active orders: 6200000, sum of matched orders: 13000000, "
    "order amount (inc. fees): 12800000"
)


def test_parses_partial_balance_to_shares():
    parsed = parse_partial_balance(poly(PARTIAL.format(bal=27400000, amt=39400000)))
    assert parsed == (Decimal("27.4"), Decimal("39.4"))


def test_parses_zero_balance_body():
    # balance may be 0 here — the parser reports it faithfully; the caller guards balance > 0
    # before clamping, so a 0 here still routes to the zero-balance phantom path.
    parsed = parse_partial_balance(poly(PARTIAL.format(bal=0, amt=20000000)))
    assert parsed == (Decimal("0"), Decimal("20"))


def test_parses_dust_balance_body():
    # 3 shares held vs 20 ordered — a real partial rejection; still parses (the dust decision is
    # the caller's, made after clamping against min_order_size).
    parsed = parse_partial_balance(poly(PARTIAL.format(bal=3000000, amt=20000000)))
    assert parsed == (Decimal("3"), Decimal("20"))


def test_overcommit_body_is_not_a_partial_balance():
    # The overcommit rejection ("order amount (inc. fees):" + "sum of ...") is a different failure
    # (a resting order reserves balance), NOT an exchange-holds-fewer-shares partial. Must be None
    # so it routes to the overcommit-defer branch, not the clamp.
    assert parse_partial_balance(poly(OVERCOMMIT)) is None


def test_garbage_without_not_enough_balance_is_none():
    assert parse_partial_balance(poly("totally unrelated 400: bad request")) is None


def test_not_enough_balance_but_unparseable_numbers_is_none():
    # Has the "not enough balance" sentinel but no "balance: N, order amount: M" pair to parse.
    exc = poly("not enough balance / allowance: the balance is not enough")
    assert parse_partial_balance(exc) is None


def test_secure_order_error_is_none():
    # SecureClient rejections carry no parseable body and stringify with an underscore
    # ("not_enough_balance"), so the space-form sentinel never matches → None.
    exc = SecureOrderError("not_enough_balance", "insufficient funds")
    assert parse_partial_balance(exc) is None


@pytest.mark.parametrize(
    "msg",
    [
        "not enough balance / allowance: the balance is not enough -> balance:27400000, "
        "order amount:39400000",  # no spaces after the colons
        "not enough balance / allowance -> balance:  27400000 ,  order amount:  39400000",  # extra
        "not enough balance -> BALANCE: 27400000, ORDER AMOUNT: 39400000",  # (case) — see assert
    ],
)
def test_whitespace_variants(msg):
    parsed = parse_partial_balance(poly(msg))
    if "BALANCE" in msg:
        # The regex is case-sensitive (matches the exact CLOB wire form), so an upper-cased body
        # does NOT parse — documents the contract rather than silently accepting a variant.
        assert parsed is None
    else:
        assert parsed == (Decimal("27.4"), Decimal("39.4"))


def test_non_polyapi_exception_type_is_none():
    # A plain exception whose text lacks the sentinel is not a partial-balance rejection.
    assert parse_partial_balance(ValueError("something failed")) is None
