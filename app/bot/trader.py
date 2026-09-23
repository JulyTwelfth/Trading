from app.bot.schemas import LimitOrder


async def place_limit_order(client, order: LimitOrder, post_only: bool = False) -> str:
    return await client.place_limit_order(order, post_only=post_only)


async def place_market_order(client, token_id: str, side: str, amount: float) -> str:
    return await client.place_market_order(token_id, side, amount)
