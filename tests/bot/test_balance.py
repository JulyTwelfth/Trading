from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import httpx

from app.bot import balance as balance_mod
from app.bot.balance import get_balance
from app.constants import PUSDC_DECIMALS
from app.farm import worker as worker_mod
from app.farm.schemas import FarmState, Market
from app.farm.worker import reconcile_tick


def patch_tick_stubs(monkeypatch, market: Market, balance: Decimal, place_calls: list):
    async def fake_fetch_eligible_markets(http):
        return [market]

    async def fake_fetch_midpoints(http, token_ids):
        return {tid: Decimal("0.5") for tid in token_ids}

    async def fake_get_balance(addr):
        return balance

    async def fake_place_limit_order(client, order, post_only=False):
        place_calls.append((order.token_id, order.size, order.price))
        return f"oid-{len(place_calls)}"

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_fetch_eligible_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_fetch_midpoints)
    monkeypatch.setattr(worker_mod, "passes_all", lambda m, f: True)
    # raising=False: on a reverted worker that never imported get_balance the
    # setattr is a no-op rather than an AttributeError, so the test reaches its
    # assertion and catches the buggy behavior (placement proceeds unchecked).
    monkeypatch.setattr(worker_mod, "get_balance", fake_get_balance, raising=False)
    monkeypatch.setattr(worker_mod, "place_limit_order", fake_place_limit_order)


async def test_skips_market_exceeding_effective_bankroll(
    farm_state: FarmState, market: Market, monkeypatch
):
    # The fixture market's two-leg cost ≈ size 100 × (0.48 + 0.48) ≈ $96. A wallet
    # holding only $50 caps the effective bankroll below that, so the gate skips it.
    farm_state.positions.clear()
    farm_state.config = farm_state.config.model_copy(update={"bankroll": Decimal("1000")})
    place_calls: list = []
    patch_tick_stubs(monkeypatch, market, balance=Decimal("50"), place_calls=place_calls)

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert place_calls == []


async def test_skips_tick_when_get_balance_fails(
    farm_state: FarmState, market: Market, monkeypatch
):
    farm_state.positions.clear()
    place_calls: list = []
    patch_tick_stubs(monkeypatch, market, balance=Decimal("1000"), place_calls=place_calls)

    async def boom(addr):
        raise httpx.RequestError("rpc down")

    # raising=False: on a reverted worker with no get_balance attribute the
    # setattr is a no-op.  The buggy code then skips the balance check entirely
    # and proceeds to place orders, so place_calls will be non-empty and the
    # assertion below catches the bug rather than dying with AttributeError.
    monkeypatch.setattr(worker_mod, "get_balance", boom, raising=False)

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert place_calls == []


async def test_get_balance_returns_decimal_with_exact_arithmetic(monkeypatch):
    def to_hex(dollars: Decimal) -> str:
        raw = int(dollars * Decimal(10) ** PUSDC_DECIMALS)
        return hex(raw)

    balances = iter([to_hex(Decimal("1000")), to_hex(Decimal("994.20"))])

    class FakeResponse:
        def __init__(self, hex_val: str):
            self._hex = hex_val

        def json(self):
            return {"result": self._hex}

        def raise_for_status(self):
            return None

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json):
            return FakeResponse(next(balances))

    monkeypatch.setattr(balance_mod.httpx, "AsyncClient", FakeClient)

    start = await get_balance("0xabc")
    current = await get_balance("0xabc")

    assert isinstance(start, Decimal)
    assert isinstance(current, Decimal)
    assert start - current == Decimal("5.80")
