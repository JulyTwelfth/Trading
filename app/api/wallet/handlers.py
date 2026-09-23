import asyncio
import logging
from decimal import Decimal

from fastapi import WebSocket

from app.api.wallet.messages import (
    WalletErrorResponse,
    WalletListEntry,
    WalletListResponse,
    WalletRegisterMessage,
    WalletRemoveMessage,
)
from app.bot.balance import get_balance
from app.constants import (
    BALANCE_FETCH_TIMEOUT_SECONDS,
    MAX_WALLETS_PER_LICENSE,
    WALLET_LIST_BALANCE_CONCURRENCY,
)
from app.db.wallets import delete_wallet, list_wallets, next_wallet_id, upsert_wallet
from app.exceptions import WalletPersistenceError, WalletSlotsExhaustedError

logger = logging.getLogger(__name__)

# Bounded so a large registry doesn't fire N simultaneous eth_calls at the RPC provider.
balance_semaphore = asyncio.Semaphore(WALLET_LIST_BALANCE_CONCURRENCY)


async def fetch_balance(proxy_address: str) -> Decimal | None:
    """Best-effort balance decoration for the wallet list: an RPC failure — or a
    slow/hanging RPC, hence the hard timeout — must never block the list
    (registration/removal flows depend on it), so both fall back to None."""
    try:
        # Acquired outside wait_for so the timeout budgets the RPC, not the queue wait.
        async with balance_semaphore:
            return await asyncio.wait_for(
                get_balance(proxy_address), timeout=BALANCE_FETCH_TIMEOUT_SECONDS
            )
    except Exception as exc:
        logger.debug("wallet_list balance fetch failed for %s: %s", proxy_address, exc)
        return None


async def send_wallet_list(websocket: WebSocket, license_key: str) -> None:
    wallets = await list_wallets(license_key)
    balances = await asyncio.gather(*(fetch_balance(w.proxy_address) for w in wallets))
    entries = [
        WalletListEntry(wallet_id=w.wallet_id, proxy_address=w.proxy_address, balance=balance)
        for w, balance in zip(wallets, balances)
    ]
    response = WalletListResponse(wallets=entries)
    await websocket.send_json(response.model_dump(mode="json"))


async def send_error(websocket: WebSocket, reason: str) -> None:
    await websocket.send_json(WalletErrorResponse(reason=reason).model_dump())


async def hydrate_after_auth(websocket: WebSocket, license_key: str) -> None:
    try:
        await send_wallet_list(websocket, license_key)
    except WalletPersistenceError as exc:
        logger.exception("Initial wallet hydrate failed for license %s", license_key)
        await send_error(websocket, exc.reason)


async def handle_register(
    websocket: WebSocket, license_key: str, msg: WalletRegisterMessage
) -> None:
    # Seeded before the try so a failure after allocation logs the slot actually
    # used rather than "auto".
    wallet_id = msg.wallet_id
    try:
        wallets = await list_wallets(license_key)
        existing = {w.wallet_id for w in wallets}
        if wallet_id is None:
            wallet_id = next_wallet_id(existing)
        # The cap gates new slots only; editing an existing wallet must work at the cap.
        if wallet_id not in existing and len(wallets) >= MAX_WALLETS_PER_LICENSE:
            await send_error(
                websocket, f"Wallet limit reached ({MAX_WALLETS_PER_LICENSE} per license)"
            )
            return
        await upsert_wallet(license_key, wallet_id, msg.proxy_address, msg.private_key)
        await send_wallet_list(websocket, license_key)
    except (WalletPersistenceError, WalletSlotsExhaustedError) as exc:
        logger.exception(
            "Wallet register failed for license %s, slot %s",
            license_key,
            wallet_id or "auto",
        )
        await send_error(websocket, exc.reason)


async def handle_remove(websocket: WebSocket, license_key: str, msg: WalletRemoveMessage) -> None:
    try:
        await delete_wallet(license_key, msg.wallet_id)
        await send_wallet_list(websocket, license_key)
    except WalletPersistenceError as exc:
        logger.exception("Wallet remove failed for license %s, slot %s", license_key, msg.wallet_id)
        await send_error(websocket, exc.reason)


async def handle_list_request(websocket: WebSocket, license_key: str) -> None:
    try:
        await send_wallet_list(websocket, license_key)
    except WalletPersistenceError as exc:
        logger.exception("Wallet list request failed for license %s", license_key)
        await send_error(websocket, exc.reason)
