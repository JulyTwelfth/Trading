import asyncio

import httpx
from eth_account import Account

from app.constants import (
    ALLOWANCE_SELECTOR,
    APPROVE_SELECTOR,
    MAX_UINT256,
    POLYGON_CHAIN_ID,
    PUSDC_CONTRACT,
    PUSDC_DECIMALS,
    V2_EXCHANGE,
)
from app.infra.config import settings

APPROVAL_THRESHOLD = 1_000 * 10**PUSDC_DECIMALS


async def rpc(method: str, params: list) -> object:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            settings.polygon_rpc_url,
            json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1},
        )
        response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(f"RPC error on {method}: {data['error']}")
    return data["result"]


async def get_allowance(wallet_address: str) -> int:
    owner = wallet_address.lower().removeprefix("0x")
    spender = V2_EXCHANGE.lower().removeprefix("0x")
    calldata = f"0x{ALLOWANCE_SELECTOR}{'0' * 24}{owner}{'0' * 24}{spender}"
    result = await rpc("eth_call", [{"to": PUSDC_CONTRACT, "data": calldata}, "latest"])
    return int(result, 16)


async def ensure_approval(private_key: str) -> None:
    account = Account.from_key(private_key)
    allowance = await get_allowance(account.address)

    if allowance >= APPROVAL_THRESHOLD:
        return

    spender_hex = V2_EXCHANGE.lower().removeprefix("0x")
    calldata = f"0x{APPROVE_SELECTOR}{'0' * 24}{spender_hex}{MAX_UINT256}"

    nonce_hex = await rpc("eth_getTransactionCount", [account.address, "latest"])
    gas_price_hex = await rpc("eth_gasPrice", [])

    signed = account.sign_transaction(
        {
            "nonce": int(nonce_hex, 16),
            "gasPrice": int(gas_price_hex, 16),
            "gas": 100_000,
            "to": PUSDC_CONTRACT,
            "value": 0,
            "data": calldata,
            "chainId": POLYGON_CHAIN_ID,
        }
    )

    tx_hash = await rpc("eth_sendRawTransaction", ["0x" + signed.raw_transaction.hex()])

    for _ in range(30):
        await asyncio.sleep(3)
        receipt = await rpc("eth_getTransactionReceipt", [tx_hash])
        if receipt:
            if int(receipt["status"], 16) != 1:
                raise RuntimeError("Approval transaction failed")
            return

    raise RuntimeError("Approval transaction timed out")
