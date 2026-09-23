import json
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal

from app.constants import BLACKLIST_STATE_DIR
from app.farm.schemas import FarmState, MarketHealth

logger = logging.getLogger(__name__)


def path(wallet_address: str) -> str:
    return os.path.join(BLACKLIST_STATE_DIR, f"blacklist-{wallet_address.lower()}.json")


def save_blacklist(state: FarmState) -> None:
    if not state.wallet_address:
        return
    records = {
        cid: {
            "until": h.blacklist_until.isoformat() if h.blacklist_until else None,
            "permanent": h.blacklist_permanent,
            "strikes": h.fill_strikes,
            "severe_guard_trips": h.severe_guard_trips,
            "net_fill_pnl": str(h.net_fill_pnl),
            "loss_roundtrips": h.loss_roundtrips,
            "cum_reward_credit": str(h.cum_reward_credit),
        }
        for cid, h in state.health.items()
        if h.blacklist_permanent
        or h.blacklist_until is not None
        or h.net_fill_pnl < 0
        or h.loss_roundtrips > 0
        or h.severe_guard_trips > 0
    }
    file_path = path(state.wallet_address)
    try:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        tmp = f"{file_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"version": 2, "markets": records}, fh)
        os.replace(tmp, file_path)
    except OSError:
        logger.exception("failed to persist blacklist to %s", file_path)


def load_blacklist(state: FarmState) -> int:
    if not state.wallet_address:
        return 0
    file_path = path(state.wallet_address)
    if not os.path.exists(file_path):
        return 0
    try:
        with open(file_path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        logger.exception("failed to load blacklist from %s; starting clean", file_path)
        return 0
    if isinstance(raw, dict) and "markets" in raw:
        version = int(raw.get("version", 1))
        records = raw["markets"]
    else:
        version = 1
        records = raw
    if not isinstance(records, dict):
        return 0
    drop_permanent = version < 2
    restored = 0
    dropped = 0
    for cid, rec in records.items():
        if not isinstance(rec, dict):
            continue
        try:
            until = rec.get("until")
            parsed = datetime.fromisoformat(until) if until else None
            if parsed is not None and parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            permanent = bool(rec.get("permanent", False))
            if drop_permanent and permanent:
                permanent = False
                dropped += 1
            health = state.health.setdefault(cid, MarketHealth())
            health.blacklist_until = parsed
            health.blacklist_permanent = permanent
            health.fill_strikes = int(rec.get("strikes", 0))
            health.severe_guard_trips = int(rec.get("severe_guard_trips", 0))
            health.net_fill_pnl = Decimal(str(rec.get("net_fill_pnl", "0")))
            health.loss_roundtrips = int(rec.get("loss_roundtrips", 0))
            health.cum_reward_credit = Decimal(str(rec.get("cum_reward_credit", "0")))
            restored += 1
        except (ValueError, TypeError):
            logger.warning("skipping malformed blacklist record for %s", cid)
    if dropped:
        logger.info("blacklist migration (v1->v2): cleared %d stale permanent ban(s)", dropped)
    return restored
