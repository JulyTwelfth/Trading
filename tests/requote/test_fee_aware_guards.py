"""The real-time depth guard must fold the taker fee into its exit-cost estimate, so a
book that is deep (zero spread loss) but fee-enabled is still pulled when the fee alone
exceeds max_fill_loss. Without the fee term the bot would round-trip such a market and
bleed the fee every cycle."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.farm import requote as requote_mod
from app.farm.requote import depth_fill_loss_guard
from app.farm.schemas import FarmState, LiveBook


async def test_depth_guard_pulls_on_fee_even_when_spread_is_zero(
    farm_state: FarmState, monkeypatch
):
    pulls: list = []

    async def fake_cancel(client, state, websocket, pos):
        pulls.append(pos.market.condition_id)

    monkeypatch.setattr(requote_mod, "cancel_position_orders", fake_cancel)

    farm_state.config.filters.max_fill_loss = Decimal("1")
    pos = farm_state.positions["market-A"]
    # Deep book AT our 0.50 entry → immediate_sell_loss is 0 (full recovery). size = 100.
    farm_state.live_books["tok-yes"] = LiveBook(bids={Decimal("0.50"): Decimal("1000")})

    # Fee-free market: spread loss 0, fee 0 → nothing to pull.
    assert (
        await depth_fill_loss_guard(MagicMock(), farm_state, AsyncMock(), pos, "tok-yes") is False
    )
    assert pulls == []

    # Crypto market: taker fee = 100 * 0.07 * 0.5 * 0.5 = 1.75 > 1.00 cap → pull.
    pos.market = pos.market.model_copy(update={"taker_fee_rate": Decimal("0.07")})
    pos.quotes_pulled = False
    assert await depth_fill_loss_guard(MagicMock(), farm_state, AsyncMock(), pos, "tok-yes") is True
    assert pulls == ["market-A"]
