from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from fastapi import WebSocket

from app.api.farm.messages import (
    FarmCancelledEvent,
    FarmCreateMessage,
    FarmErrorEvent,
    FarmStartedEvent,
)
from app.api.messages import send_event
from app.bot.balance import get_balance
from app.constants import FARM_CANCEL_GRACE_SECONDS
from app.db.wallets import list_wallets
from app.farm.worker import run_farm

if TYPE_CHECKING:
    from app.farm.schemas import FarmState

logger = logging.getLogger(__name__)


class FarmSession:
    def __init__(self) -> None:
        self.state: FarmState | None = None


async def send_farm_error(websocket: WebSocket, reason: str) -> None:
    await send_event(websocket, FarmErrorEvent(reason=reason))


async def handle_farm_create(
    websocket: WebSocket,
    license_key: str,
    msg: FarmCreateMessage,
    current_task: asyncio.Task | None,
    farm_session: FarmSession,
) -> asyncio.Task | None:
    if current_task is not None and not current_task.done():
        await send_farm_error(websocket, "farm_already_running")
        return current_task

    wallets = await list_wallets(license_key)
    # Phase 1 still farms a single wallet: the first in list_wallets sort order.
    wallet = wallets[0] if wallets else None
    if wallet is None:
        await send_farm_error(websocket, "no_wallet_registered")
        return current_task

    balance = await get_balance(wallet.proxy_address)
    if msg.bankroll > balance:
        await send_farm_error(websocket, "insufficient_balance")
        return current_task

    task = asyncio.create_task(run_farm(websocket, msg, wallet, license_key, farm_session))
    await send_event(websocket, FarmStartedEvent())
    return task


async def handle_farm_cancel(
    websocket: WebSocket,
    current_task: asyncio.Task | None,
) -> asyncio.Task | None:
    """Stop the farm and ACK the UI. Waits up to FARM_CANCEL_GRACE_SECONDS for the worker's
    shielded teardown, then sends farm_cancelled regardless so the UI can't hang on a slow
    shutdown. Returns the task if teardown outlasts the grace period (caller keeps tracking it
    to block a new farm), else None."""
    still_tearing_down: asyncio.Task | None = None
    if current_task is not None and not current_task.done():
        if not current_task.cancelling():
            current_task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(current_task), timeout=FARM_CANCEL_GRACE_SECONDS)
        except asyncio.TimeoutError:
            still_tearing_down = current_task
        except asyncio.CancelledError:
            if not current_task.done():
                raise
        except Exception:
            logger.exception("farm teardown raised during cancel")
    await send_event(websocket, FarmCancelledEvent())
    return still_tearing_down
