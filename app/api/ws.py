import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.api.blacklist.handlers import (
    handle_blacklist_add,
    handle_blacklist_clear,
    handle_blacklist_list,
    handle_blacklist_remove,
    hydrate_blacklist_after_auth,
)
from app.api.blacklist.messages import (
    BlacklistAddMessage,
    BlacklistClearMessage,
    BlacklistListRequestMessage,
    BlacklistRemoveMessage,
)
from app.api.farm.handlers import FarmSession, handle_farm_cancel, handle_farm_create
from app.api.farm.messages import FarmCancelMessage, FarmCreateMessage
from app.api.messages import client_message_adapter
from app.api.wallet.handlers import (
    handle_list_request,
    handle_register,
    handle_remove,
    hydrate_after_auth,
    send_error,
)
from app.api.wallet.messages import (
    WalletListRequestMessage,
    WalletRegisterMessage,
    WalletRemoveMessage,
)
from app.constants import FARM_CANCEL_TIMEOUT_SECONDS
from app.db.license import is_duplicate_session, validate_license
from app.exceptions import (
    ExpiredLicenseError,
    InvalidLicenseKeyError,
    LicenseServiceUnavailableError,
    SessionAlreadyActiveError,
)
from app.infra.dependencies import supabase

router = APIRouter()
logger = logging.getLogger(__name__)


async def authenticate(websocket: WebSocket, data: dict) -> str | None:
    if data.get("type") != "auth" or not data.get("license_key"):
        await websocket.send_json({"type": "auth_fail", "reason": "First message must be auth"})
        await websocket.close()
        return None

    key = data["license_key"]

    try:
        validate_license(key)
    except (InvalidLicenseKeyError, ExpiredLicenseError, SessionAlreadyActiveError) as e:
        await websocket.send_json({"type": "auth_fail", "reason": e.reason})
        await websocket.close()
        return None
    except LicenseServiceUnavailableError:
        # Without this the transport error escapes the endpoint and the socket dies
        # with no auth_fail frame, so the UI can only report a generic disconnect.
        logger.exception("ws: license check failed — license backend unreachable")
        await websocket.send_json(
            {"type": "auth_fail", "reason": LicenseServiceUnavailableError.reason}
        )
        await websocket.close()
        return None

    return key


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        data = await websocket.receive_json()
    except WebSocketDisconnect:
        return

    key = await authenticate(websocket, data)
    if key is None:
        return

    try:
        supabase.table("sessions").insert({"key": key}).execute()
    except Exception as exc:
        # The session row is what enforces one live farm per key; if we cannot write it we must
        # refuse rather than run unguarded. Two different failures land here and the client needs
        # to tell them apart: a duplicate key means another connection won the race between
        # validate_license's check and this insert (retrying later will work), while anything
        # else means the licence backend is unreachable (retrying now will not).
        if is_duplicate_session(exc):
            logger.info("ws: license %s already claimed by another connection", key)
            reason = SessionAlreadyActiveError.reason
        else:
            logger.exception("ws: could not open session for license %s", key)
            reason = LicenseServiceUnavailableError.reason
        await websocket.send_json({"type": "auth_fail", "reason": reason})
        await websocket.close()
        return

    await websocket.send_json({"type": "auth_ok"})
    logger.info("ws: license %s connected", key)
    await hydrate_after_auth(websocket, key)
    await hydrate_blacklist_after_auth(websocket, key)

    farm_task: asyncio.Task | None = None
    farm_session = FarmSession()

    try:
        while True:
            raw = await websocket.receive_json()

            try:
                msg = client_message_adapter.validate_python(raw)
            except ValidationError as exc:
                await send_error(websocket, exc.errors()[0]["msg"])
                continue

            match msg:
                case WalletRegisterMessage():
                    await handle_register(websocket, key, msg)
                case WalletRemoveMessage():
                    await handle_remove(websocket, key, msg)
                case WalletListRequestMessage():
                    await handle_list_request(websocket, key)
                case FarmCreateMessage():
                    farm_task = await handle_farm_create(
                        websocket, key, msg, farm_task, farm_session
                    )
                case FarmCancelMessage():
                    farm_task = await handle_farm_cancel(websocket, farm_task)
                case BlacklistAddMessage():
                    await handle_blacklist_add(websocket, key, msg, farm_session)
                case BlacklistRemoveMessage():
                    await handle_blacklist_remove(websocket, key, msg, farm_session)
                case BlacklistListRequestMessage():
                    await handle_blacklist_list(websocket, key)
                case BlacklistClearMessage():
                    await handle_blacklist_clear(websocket, key, farm_session)
                case _:
                    logger.warning(
                        "Unhandled message type from license %s: %s",
                        key,
                        type(msg).__name__,
                    )
    except WebSocketDisconnect:
        logger.info("ws: license %s disconnected", key)
    finally:
        farm_session.state = None
        if farm_task is not None and not farm_task.done():
            farm_task.cancel()
            try:
                await asyncio.wait_for(farm_task, timeout=FARM_CANCEL_TIMEOUT_SECONDS)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger.warning(
                    "ws: farm task for %s did not stop within %ss; abandoning it to finish cleanup",
                    key,
                    FARM_CANCEL_TIMEOUT_SECONDS,
                )
        try:
            supabase.table("sessions").delete().eq("key", key).execute()
        except Exception:
            # A failure here strands the session row and locks the key out until the
            # next startup sweep, so it must be visible in the log.
            logger.exception("ws: could not close session for license %s", key)
