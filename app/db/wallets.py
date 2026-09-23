import asyncio
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from app.exceptions import WalletPersistenceError, WalletSlotsExhaustedError
from app.infra.dependencies import supabase
from app.types import MAX_WALLET_SLOT, WALLET_ID_RE, WalletId

TABLE = "wallets"
COLUMNS = "license_key,wallet_id,proxy_address,private_key"


def wallet_sort_key(wallet_id: str) -> tuple[int, int, str]:
    """Canonical walletN ids sort numerically and first; legacy ids trail lexicographically."""
    if WALLET_ID_RE.match(wallet_id):
        return (0, int(wallet_id.removeprefix("wallet")), "")
    return (1, 0, wallet_id)


def next_wallet_id(existing: Iterable[str]) -> str:
    """Lowest free walletN slot, so a delete leaves no gap in the list."""
    taken = {int(w.removeprefix("wallet")) for w in existing if WALLET_ID_RE.match(w)}
    n = 1
    while n in taken:
        n += 1
    # Past the regex bound there is no canonical id left to hand out; returning one anyway
    # would make every later register overwrite the same non-canonical row.
    if n > MAX_WALLET_SLOT:
        raise WalletSlotsExhaustedError()
    return f"wallet{n}"


class Wallet(BaseModel):
    model_config = ConfigDict(extra="ignore")

    license_key: str
    wallet_id: WalletId
    proxy_address: str
    private_key: str


async def list_wallets(license_key: str) -> list[Wallet]:
    def query():
        return supabase.table(TABLE).select(COLUMNS).eq("license_key", license_key).execute()

    try:
        response = await asyncio.to_thread(query)
        wallets = [Wallet.model_validate(row) for row in response.data]
    except Exception as e:
        raise WalletPersistenceError() from e
    return sorted(wallets, key=lambda w: wallet_sort_key(w.wallet_id))


async def upsert_wallet(
    license_key: str,
    wallet_id: WalletId,
    proxy_address: str,
    private_key: str,
) -> Wallet:
    def upsert():
        return (
            supabase.table(TABLE)
            .upsert(
                {
                    "license_key": license_key,
                    "wallet_id": wallet_id,
                    "proxy_address": proxy_address,
                    "private_key": private_key,
                },
                on_conflict="license_key,wallet_id",
            )
            .execute()
        )

    try:
        response = await asyncio.to_thread(upsert)
    except Exception as e:
        raise WalletPersistenceError() from e
    return Wallet.model_validate(response.data[0])


async def delete_wallet(license_key: str, wallet_id: WalletId) -> None:
    def delete():
        return (
            supabase.table(TABLE)
            .delete()
            .eq("license_key", license_key)
            .eq("wallet_id", wallet_id)
            .execute()
        )

    try:
        await asyncio.to_thread(delete)
    except Exception as e:
        raise WalletPersistenceError() from e
