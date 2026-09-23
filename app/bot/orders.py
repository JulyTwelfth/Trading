async def get_open_order_ids(client) -> set[str]:
    """IDs of every order resting on the CLOB for this wallet (all pages)."""
    return await client.get_open_order_ids()
