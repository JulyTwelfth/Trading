from decimal import Decimal

import httpx

from app.constants import BALANCE_OF_SELECTOR, PUSDC_CONTRACT, PUSDC_DECIMALS
from app.infra.config import settings


async def get_balance(wallet_address: str) -> Decimal:
    address_hex = wallet_address.lower().removeprefix("0x")
    calldata = f"0x{BALANCE_OF_SELECTOR}{'0' * 24}{address_hex}"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            settings.polygon_rpc_url,
            json={
                "jsonrpc": "2.0",
                "method": "eth_call",
                "params": [{"to": PUSDC_CONTRACT, "data": calldata}, "latest"],
                "id": 1,
            },
        )
        response.raise_for_status()

    result = response.json()
    raw = int(result["result"], 16)
    return Decimal(raw) / Decimal(10) ** PUSDC_DECIMALS
