import asyncio

from pydantic import BaseModel, ConfigDict

from app.exceptions import BlacklistPersistenceError
from app.infra.dependencies import supabase

TABLE = "blacklisted_markets"
COLUMNS = "license_key,condition_id,slug,question,market_url,created_at"


class BlacklistedMarket(BaseModel):
    model_config = ConfigDict(extra="ignore")

    license_key: str
    condition_id: str
    slug: str | None = None
    question: str | None = None
    market_url: str | None = None


async def list_blacklist(license_key: str) -> list[BlacklistedMarket]:
    def query():
        return (
            supabase.table(TABLE)
            .select(COLUMNS)
            .eq("license_key", license_key)
            .order("created_at")
            .execute()
        )

    try:
        response = await asyncio.to_thread(query)
    except Exception as e:
        raise BlacklistPersistenceError() from e
    return [BlacklistedMarket.model_validate(row) for row in response.data]


async def add_blacklist(license_key: str, rows: list[dict]) -> list[BlacklistedMarket]:
    scoped_rows = [{**row, "license_key": license_key} for row in rows]

    def upsert():
        return (
            supabase.table(TABLE)
            .upsert(
                scoped_rows,
                on_conflict="license_key,condition_id",
            )
            .execute()
        )

    try:
        response = await asyncio.to_thread(upsert)
    except Exception as e:
        raise BlacklistPersistenceError() from e
    return [BlacklistedMarket.model_validate(r) for r in response.data]


async def remove_blacklist(license_key: str, condition_id: str) -> None:
    def delete():
        return (
            supabase.table(TABLE)
            .delete()
            .eq("license_key", license_key)
            .eq("condition_id", condition_id)
            .execute()
        )

    try:
        await asyncio.to_thread(delete)
    except Exception as e:
        raise BlacklistPersistenceError() from e


async def clear_blacklist(license_key: str) -> None:
    def delete():
        return supabase.table(TABLE).delete().eq("license_key", license_key).execute()

    try:
        await asyncio.to_thread(delete)
    except Exception as e:
        raise BlacklistPersistenceError() from e
