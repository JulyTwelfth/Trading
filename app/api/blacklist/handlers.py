import logging

import httpx
from fastapi import WebSocket

from app.api.blacklist.messages import (
    BlacklistAddMessage,
    BlacklistEntry,
    BlacklistErrorResponse,
    BlacklistListResponse,
    BlacklistRemoveMessage,
)
from app.api.farm.handlers import FarmSession
from app.bot.market import resolve_event_markets
from app.db.blacklist import add_blacklist, clear_blacklist, list_blacklist, remove_blacklist
from app.exceptions import BlacklistPersistenceError

logger = logging.getLogger(__name__)


async def send_blacklist_list(websocket: WebSocket, license_key: str) -> None:
    rows = await list_blacklist(license_key)
    response = BlacklistListResponse(
        markets=[
            BlacklistEntry(
                condition_id=r.condition_id,
                slug=r.slug,
                question=r.question,
                market_url=r.market_url,
            )
            for r in rows
        ]
    )
    await websocket.send_json(response.model_dump())


async def send_blacklist_error(websocket: WebSocket, reason: str) -> None:
    await websocket.send_json(BlacklistErrorResponse(reason=reason).model_dump())


async def hydrate_blacklist_after_auth(websocket: WebSocket, license_key: str) -> None:
    try:
        await send_blacklist_list(websocket, license_key)
    except BlacklistPersistenceError as exc:
        logger.exception("Initial blacklist hydrate failed for license %s", license_key)
        await send_blacklist_error(websocket, exc.reason)


async def handle_blacklist_add(
    websocket: WebSocket,
    license_key: str,
    msg: BlacklistAddMessage,
    farm_session: FarmSession,
) -> None:
    try:
        markets = await resolve_event_markets(msg.market_url)
    except ValueError as exc:
        await send_blacklist_error(websocket, str(exc))
        return
    except httpx.HTTPError:
        await send_blacklist_error(websocket, "Could not resolve market URL")
        return

    rows = [
        {
            "license_key": license_key,
            "condition_id": m["condition_id"],
            "slug": m["slug"],
            "question": m["question"],
            "market_url": msg.market_url,
        }
        for m in markets
    ]

    try:
        await add_blacklist(license_key, rows)
    except BlacklistPersistenceError as exc:
        logger.exception("Blacklist add failed for license %s", license_key)
        await send_blacklist_error(websocket, exc.reason)
        return

    if farm_session.state is not None:
        for m in markets:
            farm_session.state.excluded_markets.add(m["condition_id"])

    try:
        await send_blacklist_list(websocket, license_key)
    except BlacklistPersistenceError as exc:
        logger.exception("Blacklist refresh after add failed for license %s", license_key)
        await send_blacklist_error(websocket, exc.reason)


async def handle_blacklist_remove(
    websocket: WebSocket,
    license_key: str,
    msg: BlacklistRemoveMessage,
    farm_session: FarmSession,
) -> None:
    try:
        await remove_blacklist(license_key, msg.condition_id)
    except BlacklistPersistenceError as exc:
        logger.exception("Blacklist remove failed for license %s", license_key)
        await send_blacklist_error(websocket, exc.reason)
        return

    if farm_session.state is not None:
        farm_session.state.excluded_markets.discard(msg.condition_id)

    try:
        await send_blacklist_list(websocket, license_key)
    except BlacklistPersistenceError as exc:
        logger.exception("Blacklist refresh after remove failed for license %s", license_key)
        await send_blacklist_error(websocket, exc.reason)


async def handle_blacklist_list(websocket: WebSocket, license_key: str) -> None:
    try:
        await send_blacklist_list(websocket, license_key)
    except BlacklistPersistenceError as exc:
        logger.exception("Blacklist list request failed for license %s", license_key)
        await send_blacklist_error(websocket, exc.reason)


async def handle_blacklist_clear(
    websocket: WebSocket,
    license_key: str,
    farm_session: FarmSession,
) -> None:
    try:
        rows = await list_blacklist(license_key)
    except BlacklistPersistenceError as exc:
        logger.exception("Blacklist clear pre-fetch failed for license %s", license_key)
        await send_blacklist_error(websocket, exc.reason)
        return

    try:
        await clear_blacklist(license_key)
    except BlacklistPersistenceError as exc:
        logger.exception("Blacklist clear failed for license %s", license_key)
        await send_blacklist_error(websocket, exc.reason)
        return

    if farm_session.state is not None:
        for row in rows:
            farm_session.state.excluded_markets.discard(row.condition_id)

    try:
        await send_blacklist_list(websocket, license_key)
    except BlacklistPersistenceError as exc:
        logger.exception("Blacklist refresh after clear failed for license %s", license_key)
        await send_blacklist_error(websocket, exc.reason)
