from typing import Literal

from pydantic import BaseModel

# ── Client → server ─────────────────────────────────────────────────────────


class BlacklistAddMessage(BaseModel):
    type: Literal["blacklist_add"]
    market_url: str


class BlacklistRemoveMessage(BaseModel):
    type: Literal["blacklist_remove"]
    condition_id: str


class BlacklistListRequestMessage(BaseModel):
    type: Literal["blacklist_list"]


class BlacklistClearMessage(BaseModel):
    type: Literal["blacklist_clear"]


# ── Server → client ─────────────────────────────────────────────────────────


class BlacklistEntry(BaseModel):
    condition_id: str
    slug: str | None = None
    question: str | None = None
    market_url: str | None = None


class BlacklistListResponse(BaseModel):
    type: Literal["blacklist_list"] = "blacklist_list"
    markets: list[BlacklistEntry]


class BlacklistErrorResponse(BaseModel):
    type: Literal["blacklist_error"] = "blacklist_error"
    reason: str
