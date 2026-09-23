import asyncio
from typing import Annotated

from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect, WebSocketState
from pydantic import BaseModel, Field, TypeAdapter

from app.api.blacklist.messages import (
    BlacklistAddMessage,
    BlacklistClearMessage,
    BlacklistListRequestMessage,
    BlacklistRemoveMessage,
)
from app.api.farm.messages import FarmCancelMessage, FarmCreateMessage
from app.api.wallet.messages import (
    WalletListRequestMessage,
    WalletRegisterMessage,
    WalletRemoveMessage,
)

ClientMessage = Annotated[
    WalletRegisterMessage
    | WalletRemoveMessage
    | WalletListRequestMessage
    | FarmCreateMessage
    | FarmCancelMessage
    | BlacklistAddMessage
    | BlacklistRemoveMessage
    | BlacklistListRequestMessage
    | BlacklistClearMessage,
    Field(discriminator="type"),
]

client_message_adapter = TypeAdapter(ClientMessage)


# Serialize concurrent send_json
send_lock = asyncio.Lock()


async def send_event(websocket: WebSocket, event: BaseModel) -> None:
    """Send any server-side event to the client. mode='json' serializes Decimal as string."""
    if websocket.application_state != WebSocketState.CONNECTED:
        return
    try:
        async with send_lock:
            await websocket.send_json(event.model_dump(mode="json"))
    except (RuntimeError, WebSocketDisconnect):
        pass
