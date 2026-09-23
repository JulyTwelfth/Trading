import asyncio
import json
import ssl
from datetime import datetime

import certifi
import websockets
from py_clob_client_v2.client import ClobClient
from py_clob_client_v2.signing.hmac import build_hmac_signature

from app.constants import WS_SIGN_PATH, WS_USER_URL

ssl_ctx = ssl.create_default_context(cafile=certifi.where())


def build_auth(client: ClobClient) -> dict:
    ts = str(int(datetime.now().timestamp()))
    sig = build_hmac_signature(client.creds.api_secret, ts, "GET", WS_SIGN_PATH)
    return {
        "apiKey": client.creds.api_key,
        "secret": client.creds.api_secret,
        "passphrase": client.creds.api_passphrase,
        "timestamp": ts,
        "signature": sig,
    }


def is_fill_event(event: dict, order_id: str) -> bool:
    if event.get("event_type") != "trade" or event.get("status") != "MATCHED":
        return False
    if event.get("taker_order_id") == order_id:
        return True
    return any(mo.get("order_id") == order_id for mo in event.get("maker_orders") or [])


async def wait_for_fill(client: ClobClient, order_id: str) -> None:
    while True:
        # REST fallback: handles the case where the order filled while disconnected
        order = await asyncio.to_thread(client.get_order, order_id)
        if order.get("status") == "MATCHED":
            return

        auth = build_auth(client)
        subscribe = json.dumps({"type": "user", "auth": auth})
        try:
            async with websockets.connect(WS_USER_URL, ssl=ssl_ctx) as ws:
                await ws.send(subscribe)
                async for raw in ws:
                    events = json.loads(raw)
                    if not isinstance(events, list):
                        events = [events]
                    for event in events:
                        if is_fill_event(event, order_id):
                            return
        except websockets.ConnectionClosed:
            continue


async def wait_for_fills(
    client_a: ClobClient,
    client_b: ClobClient,
    order_id_a: str,
    order_id_b: str,
    timeout: float,
) -> None:
    """
    Waits until both order_id_a (wallet A) and order_id_b (wallet B) are MATCHED.
    Raises asyncio.TimeoutError if timeout expires before both fills are confirmed.
    """
    await asyncio.wait_for(
        asyncio.gather(
            wait_for_fill(client_a, order_id_a),
            wait_for_fill(client_b, order_id_b),
        ),
        timeout=timeout,
    )
