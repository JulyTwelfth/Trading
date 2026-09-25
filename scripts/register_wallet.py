"""Register a dedicated Polymarket signer through the local WebSocket API.

The private key is entered with hidden input and is never accepted as a command-line
argument. The current backend persists it in the Supabase wallets table.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json

from eth_account import Account
from websockets.asyncio.client import connect


async def register(
    ws_url: str,
    license_key: str,
    wallet_id: str,
    account_wallet: str,
    private_key: str,
) -> None:
    signer = Account.from_key(private_key).address
    print(f"MetaMask signer:          {signer}")
    print(f"Polymarket account wallet: {account_wallet}")

    async with connect(ws_url) as websocket:
        await websocket.send(json.dumps({"type": "auth", "license_key": license_key}))
        registration_sent = False

        while True:
            message = json.loads(await websocket.recv())
            message_type = message.get("type")

            if message_type == "auth_fail":
                raise RuntimeError(f"authentication failed: {message.get('reason')}")

            # wallet_list and blacklist_list are hydrated before the server starts
            # processing client messages. Waiting for the latter makes the next
            # wallet_list unambiguously the result of this registration.
            if message_type in {"blacklist_list", "blacklist_error"} and not registration_sent:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "wallet_register",
                            "wallet_id": wallet_id,
                            "proxy_address": account_wallet,
                            "private_key": private_key,
                        }
                    )
                )
                registration_sent = True
                continue

            if registration_sent and message_type == "wallet_error":
                raise RuntimeError(f"wallet registration failed: {message.get('reason')}")

            if registration_sent and message_type == "wallet_list":
                wallets = message.get("wallets", [])
                match = next((w for w in wallets if w.get("wallet_id") == wallet_id), None)
                if match is None:
                    raise RuntimeError("registration response did not contain the requested slot")
                if match.get("proxy_address", "").lower() != account_wallet.lower():
                    raise RuntimeError(
                        "registered wallet address does not match the requested address"
                    )
                print(f"Registered {wallet_id}; reported USDC balance: {match.get('balance')}")
                return


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Register a MetaMask signer and Polymarket account wallet locally."
    )
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8000/ws")
    parser.add_argument("--wallet-id", default="wallet1")
    args = parser.parse_args()

    license_key = getpass.getpass("License key (hidden): ").strip()
    account_wallet = input("Polymarket account wallet address (from profile menu): ").strip()
    private_key = getpass.getpass("Dedicated MetaMask account private key (hidden): ").strip()

    if not license_key or not account_wallet or not private_key:
        print("All three values are required.")
        return 1

    try:
        asyncio.run(
            register(
                args.ws_url,
                license_key,
                args.wallet_id,
                account_wallet,
                private_key,
            )
        )
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
