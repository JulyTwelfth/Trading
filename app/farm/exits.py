import logging
import re
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN, Decimal

from app.bot.cancel import cancel_order, cancel_orders
from app.bot.market import get_order_book
from app.bot.schemas import BookLevel, LimitOrder
from app.bot.trader import place_limit_order, place_market_order
from app.constants import (
    EXIT_DUST_BALANCE_SHARES,
    MAX_CLAMP_RETRIES,
    RECENT_EXIT_GRACE_SECONDS,
    RECONCILE_BACKOFF_MAX_SECONDS,
    SMART_EXIT_COMPLEMENT_SPREAD_FACTOR,
    SMART_EXIT_CONCESSION_REWARD_FRACTION,
    SMART_EXIT_CONCESSION_TICKS,
    SMART_EXIT_DEPTH_FACTOR,
    SMART_EXIT_DEPTH_MIN_SHARES,
    SMART_EXIT_EVENT_PROXIMITY_HOURS,
    SMART_EXIT_FLOOR_ENABLED,
    SMART_EXIT_MAX_HOLD_SECONDS,
    SMART_EXIT_MIN_SAVING,
    SMART_EXIT_POSTGAME_HOURS,
    STALE_EXIT_SECONDS,
    STALE_STAGED_EXIT_SECONDS,
    ZERO_BALANCE_GIVEUP_SECONDS,
)
from app.exceptions import SecureOrderError
from app.farm.discovery import fetch_open_positions
from app.farm.exit_cost import liquidation_proceeds
from app.farm.health import mark_paused, record_market_failure
from app.farm.schemas import ExitCostInfo, ExitOrder, FarmState, MarketPosition
from app.farm.volatility import blacklist_for_fill, is_blacklisted
from app.infra.strategy_log import strat

logger = logging.getLogger(__name__)

ZERO_SHARE_BALANCE_RE = re.compile(r"balance:\s*0(?![\d.])")
OVERCOMMIT_RE = re.compile(r"sum of (?:matched|active)")
PARTIAL_BALANCE_RE = re.compile(r"balance:\s*(\d+)\s*,\s*order amount:\s*(\d+)")
MICRO_SHARES = Decimal(10) ** 6


def parse_partial_balance(exc: Exception) -> tuple[Decimal, Decimal] | None:
    """(balance_shares, order_shares) from a legacy partial-balance SELL rejection, else None
    (overcommit/SecureOrderError/malformed don't match). balance may be 0 — caller guards >0."""
    msg = str(exc)
    if "not enough balance" not in msg:
        return None
    m = PARTIAL_BALANCE_RE.search(msg)
    if m is None:
        return None
    return Decimal(m.group(1)) / MICRO_SHARES, Decimal(m.group(2)) / MICRO_SHARES


def exit_key(condition_id: str, outcome: str) -> str:
    return f"{condition_id}:{outcome.upper()}"


def is_zero_share_balance_rejection(exc: Exception) -> bool:
    if isinstance(exc, SecureOrderError) and exc.code == "not_enough_balance":
        return True
    msg = str(exc)
    return "not enough balance" in msg and bool(ZERO_SHARE_BALANCE_RE.search(msg))


def is_invalid_price_rejection(exc: Exception) -> bool:
    return "invalid price" in str(exc).lower()


def is_overcommit_rejection(exc: Exception) -> bool:
    msg = str(exc).lower()
    if "not enough balance" not in msg:
        return False
    if ZERO_SHARE_BALANCE_RE.search(msg):
        return False
    return bool(OVERCOMMIT_RE.search(msg))


def record_zero_balance_strike(
    state: FarmState, condition_id: str, outcome: str, now: datetime | None = None
) -> bool:
    outcome = outcome.upper()
    pos = state.positions.get(condition_id)
    if pos is None:
        return True
    now = now or datetime.now(timezone.utc)
    first = pos.zero_balance_since.get(outcome)
    if first is None:
        pos.zero_balance_since[outcome] = now
        return False
    if (now - first).total_seconds() >= ZERO_BALANCE_GIVEUP_SECONDS:
        pos.zero_balance_since.pop(outcome, None)
        return True
    return False


def clear_held_leg(state: FarmState, condition_id: str, outcome: str) -> None:
    outcome = outcome.upper()
    pos = state.positions.get(condition_id)
    if pos is None:
        return
    if outcome == "YES":
        pos.yes_shares = Decimal(0)
        pos.yes_cost_basis = Decimal(0)
    else:
        pos.no_shares = Decimal(0)
        pos.no_cost_basis = Decimal(0)


def dust_balance_writeoff(
    state: FarmState,
    condition_id: str,
    slug: str,
    outcome: str,
    balance_shares: Decimal,
    size: Decimal,
    entry_cost: Decimal,
    path: str,
) -> None:
    """SELL rejected with CLOB balance at/below the dust floor: unsellable regardless of retry, so
    write off terminally — no GTC loop, no session_loss booking, no failure-mark (CLOB is truth)."""
    pos = state.positions.get(condition_id)
    logger.warning(
        "exit %s/%s: DUST WRITE-OFF — SELL of %s shares rejected, CLOB holds only %s "
        "(<= %s dust floor); already exited on-chain minus residue, abandoning "
        "(terminal: no GTC fallback, not failure-marked)",
        slug,
        outcome,
        size,
        balance_shares,
        EXIT_DUST_BALANCE_SHARES,
    )
    strat(
        "dust_writeoff",
        slug=slug,
        outcome=outcome,
        size=size,
        cost_basis=entry_cost,
        orphan=(1 if pos is None else 0),
        booked=0,
        balance=float(balance_shares),
        path=path,
    )
    if pos is not None:
        clear_held_leg(state, condition_id, outcome)


async def clamp_leg_to_balance(
    client,
    state: FarmState,
    token_id: str,
    tracked_size: Decimal,
    condition_id: str,
    slug: str,
    outcome: str,
    balance_shares: Decimal,
    entry_cost: Decimal,
    force_dump: bool,
    clamp_depth: int,
) -> None:
    """A legacy partial-balance SELL rejection means the tracked shares diverged from what the
    exchange actually holds (e.g. an earlier fill still settling). Clamp the leg down to the
    exchange-reported balance and retry instead of re-driving the stale full size forever — a
    live-observed 6-minute retry loop. The shares removed from tracking are simply dropped (not
    parked for a late fill): a bounded, conservative accounting gap in a rare race."""
    if clamp_depth >= MAX_CLAMP_RETRIES:
        logger.warning(
            "exit %s/%s: PARTIAL-BALANCE CLAMP hit recursion cap (%d) — giving up, leaving %s "
            "shares held for sweep/reconcile",
            slug,
            outcome,
            MAX_CLAMP_RETRIES,
            balance_shares,
        )
        return

    pos = state.positions.get(condition_id)
    if pos is None:
        scaled_cost = (
            entry_cost * (balance_shares / tracked_size) if tracked_size > 0 else Decimal(0)
        )
        logger.warning(
            "exit %s/%s: PARTIAL-BALANCE CLAMP on an orphan (no tracked position) — retrying "
            "at exchange balance %s",
            slug,
            outcome,
            balance_shares,
        )
        strat(
            "exit_clamp",
            slug=slug,
            outcome=outcome,
            tracked=tracked_size,
            balance=balance_shares,
            new=balance_shares,
            removed=Decimal(0),
        )
        await exit_position_leg(
            client,
            state,
            token_id,
            balance_shares,
            condition_id,
            slug,
            outcome,
            cancel_resting=False,
            entry_cost=scaled_cost,
            force_dump=True,
            _clamp_depth=clamp_depth + 1,
        )
        return

    old = pos.yes_shares if outcome == "YES" else pos.no_shares
    old_basis = pos.yes_cost_basis if outcome == "YES" else pos.no_cost_basis

    new = min(balance_shares, old)
    removed = old - new

    if removed <= 0:
        if old <= 0 and balance_shares > EXIT_DUST_BALANCE_SHARES:
            # Tracked leg is already flat but the exchange still reports a sellable balance
            # (e.g. it grew back after the position was cleared) — retry at the actual balance
            # instead of no-op'ing forever at new = min(balance, 0) = 0. old_basis is stale/zero
            # here, so scale the caller's entry_cost like the orphan branch above.
            scaled_cost = (
                entry_cost * (balance_shares / tracked_size) if tracked_size > 0 else Decimal(0)
            )
            logger.warning(
                "exit %s/%s: PARTIAL-BALANCE CLAMP saw tracked leg at 0 with exchange balance "
                "%s above the dust floor — retrying at exchange balance %s",
                slug,
                outcome,
                balance_shares,
                balance_shares,
            )
            strat(
                "exit_clamp",
                slug=slug,
                outcome=outcome,
                tracked=old,
                balance=balance_shares,
                new=balance_shares,
                removed=Decimal(0),
            )
            await exit_position_leg(
                client,
                state,
                token_id,
                balance_shares,
                condition_id,
                slug,
                outcome,
                cancel_resting=False,
                entry_cost=scaled_cost,
                force_dump=True,
                _clamp_depth=clamp_depth + 1,
            )
            return
        logger.warning(
            "exit %s/%s: PARTIAL-BALANCE CLAMP saw exchange balance %s >= tracked %s — "
            "spurious/stale rejection, retrying at %s",
            slug,
            outcome,
            balance_shares,
            old,
            new,
        )
        await exit_position_leg(
            client,
            state,
            token_id,
            new,
            condition_id,
            slug,
            outcome,
            cancel_resting=False,
            entry_cost=old_basis,
            force_dump=force_dump,
            _clamp_depth=clamp_depth + 1,
        )
        return

    new_basis = old_basis * (new / old) if old > 0 else Decimal(0)

    if outcome == "YES":
        pos.yes_shares, pos.yes_cost_basis = new, new_basis
    else:
        pos.no_shares, pos.no_cost_basis = new, new_basis

    logger.warning(
        "exit %s/%s: PARTIAL-BALANCE CLAMP tracked=%s exchange_balance=%s -> selling %s (%s "
        "shares dropped from tracking, not parked for a late fill)",
        slug,
        outcome,
        old,
        balance_shares,
        new,
        removed,
    )
    strat(
        "exit_clamp",
        slug=slug,
        outcome=outcome,
        tracked=old,
        balance=balance_shares,
        new=new,
        removed=removed,
    )

    if new <= 0:
        clear_held_leg(state, condition_id, outcome)
        return

    await exit_position_leg(
        client,
        state,
        token_id,
        new,
        condition_id,
        slug,
        outcome,
        cancel_resting=False,
        entry_cost=new_basis,
        force_dump=True,
        _clamp_depth=clamp_depth + 1,
    )


def register_exit(
    state: FarmState,
    condition_id: str,
    exit_oid: str,
    size: Decimal,
    slug: str,
    entry_cost: Decimal,
    outcome: str,
) -> None:
    outcome = outcome.upper()
    state.pending_exit_order_ids.add(exit_oid)
    state.exit_cost_basis[exit_oid] = ExitCostInfo(
        entry_cost=entry_cost, entry_size=size, slug=slug
    )
    state.recent_exits[exit_key(condition_id, outcome)] = datetime.now(timezone.utc)
    pos = state.positions.get(condition_id)
    if pos is not None:
        pos.exit_orders[exit_oid] = ExitOrder(outcome=outcome, placed_at=datetime.now(timezone.utc))
        pos.zero_balance_since.pop(outcome, None)


async def refresh_conditional_balance(client, token_id: str) -> bool:
    return await client.refresh_conditional_balance(token_id)


def smart_exit_floor_price(
    state: FarmState,
    pos: MarketPosition | None,
    token_id: str,
    size: Decimal,
    now: datetime | None = None,
) -> Decimal | None:
    if not SMART_EXIT_FLOOR_ENABLED or pos is None or size <= 0:
        return None
    now = now or datetime.now(timezone.utc)

    gst = pos.market.game_start_time
    if gst is not None and gst.tzinfo is None:
        gst = gst.replace(tzinfo=timezone.utc)
    if gst is not None:
        secs = (gst - now).total_seconds()
        if -SMART_EXIT_POSTGAME_HOURS * 3600 <= secs <= SMART_EXIT_EVENT_PROXIMITY_HOURS * 3600:
            return None
    if pos.market.sports_event_active:
        return None

    is_yes = token_id == pos.market.yes_token_id
    other_token = pos.market.no_token_id if is_yes else pos.market.yes_token_id
    our_book = state.live_books.get(token_id)
    other_book = state.live_books.get(other_token)
    if our_book is None or not our_book.bids:
        return None
    if other_book is None or not other_book.bids or not other_book.asks:
        return None

    tick = pos.market.tick_size
    reward_band = pos.market.rewards_max_spread_cents / 100
    max_ref_spread = SMART_EXIT_COMPLEMENT_SPREAD_FACTOR * reward_band
    min_ref_depth = max(SMART_EXIT_DEPTH_MIN_SHARES, SMART_EXIT_DEPTH_FACTOR * size)
    concession = max(
        SMART_EXIT_CONCESSION_TICKS * tick, SMART_EXIT_CONCESSION_REWARD_FRACTION * reward_band
    )

    other_bid = max(other_book.bids)
    other_ask = min(other_book.asks)
    if other_ask <= other_bid or other_ask - other_bid > max_ref_spread:
        return None
    if sum(other_book.bids.values()) + sum(other_book.asks.values()) < min_ref_depth:
        return None

    fair_value = Decimal(1) - (other_bid + other_ask) / 2
    floor = (fair_value - concession).quantize(tick, rounding=ROUND_DOWN)
    if floor <= 0 or floor >= 1:
        return None

    if max(our_book.bids) >= floor:
        return None

    bids = [BookLevel(price=p, size=s) for p, s in our_book.bids.items()]
    if fair_value * size - liquidation_proceeds(bids, size) < SMART_EXIT_MIN_SAVING:
        return None

    return floor


async def place_floor_sell(
    client,
    state: FarmState,
    token_id: str,
    size: Decimal,
    condition_id: str,
    slug: str,
    outcome: str,
    floor: Decimal,
    entry_cost: Decimal,
) -> bool:
    order = LimitOrder(token_id=token_id, side="SELL", size=float(size), price=float(floor))
    try:
        exit_oid = await place_limit_order(client, order, post_only=False)
    except Exception as exc:
        if is_zero_share_balance_rejection(exc):
            await refresh_conditional_balance(client, token_id)
            if record_zero_balance_strike(state, condition_id, outcome):
                clear_held_leg(state, condition_id, outcome)
                logger.warning(
                    "exit %s/%s: smart-floor SELL — 0 share balance past give-up, clearing phantom",
                    slug,
                    outcome,
                )
                strat(
                    "phantom_clear", slug=slug, outcome=outcome, cost_basis=entry_cost, path="smart"
                )
            else:
                logger.info(
                    "exit %s/%s: smart-floor SELL bounced balance:0 (buy not mined) — "
                    "sweep retries",
                    slug,
                    outcome,
                )
            return True
        logger.exception(
            "smart-floor SELL failed (non-balance) for %s/%s — falling back to FAK dump",
            slug,
            outcome,
        )
        return False
    register_exit(state, condition_id, exit_oid, size, slug, entry_cost, outcome)
    pos = state.positions.get(condition_id)
    if pos is not None and exit_oid in pos.exit_orders:
        pos.exit_orders[exit_oid].is_floor = True
    logger.warning(
        "exit %s/%s: SMART-EXIT floor SELL %s shares @ %s — book swept hollow, resting at fair "
        "value instead of dumping (%s)",
        slug,
        outcome,
        size,
        floor,
        exit_oid,
    )
    strat(
        "smart_exit_floor",
        slug=slug,
        outcome=outcome,
        size=size,
        floor=floor,
        cost_basis=entry_cost,
    )
    return True


async def exit_position_leg(
    client,
    state: FarmState,
    token_id: str,
    size: Decimal,
    condition_id: str,
    slug: str,
    outcome: str,
    *,
    cancel_resting: bool = True,
    entry_cost: Decimal = Decimal(0),
    force_dump: bool = False,
    _clamp_depth: int = 0,
) -> None:
    outcome = outcome.upper()
    if size <= 0:
        return
    pos = state.positions.get(condition_id)
    if pos is not None and size < pos.market.min_order_size:
        logger.warning(
            "exit %s/%s: DUST WRITE-OFF %s shares (cost_basis=$%s) below min_order_size %s — "
            "un-sellable, abandoning. NOT booked to session_loss; capital lost if this leg "
            "resolves against us",
            slug,
            outcome,
            size,
            entry_cost,
            pos.market.min_order_size,
        )
        strat(
            "dust_writeoff",
            slug=slug,
            outcome=outcome,
            size=size,
            cost_basis=entry_cost,
            orphan=0,
            booked=0,
        )
        clear_held_leg(state, condition_id, outcome)
        return
    if cancel_resting and pos is not None:
        await cancel_orders(client, pos.yes_order_id, pos.no_order_id)

    if not force_dump:
        floor = smart_exit_floor_price(state, pos, token_id, size)
        if floor is not None and await place_floor_sell(
            client, state, token_id, size, condition_id, slug, outcome, floor, entry_cost
        ):
            return

    logger.info("exit %s/%s: selling %s shares (FAK)", slug, outcome, size)
    try:
        exit_oid = await place_market_order(client, token_id, "SELL", float(size))
        register_exit(state, condition_id, exit_oid, size, slug, entry_cost, outcome)
        logger.info("exit %s/%s: FAK SELL placed (%s)", slug, outcome, exit_oid)
        return
    except Exception as exc:
        if is_zero_share_balance_rejection(exc):
            if await refresh_conditional_balance(client, token_id):
                try:
                    exit_oid = await place_market_order(client, token_id, "SELL", float(size))
                    register_exit(state, condition_id, exit_oid, size, slug, entry_cost, outcome)
                    logger.info(
                        "exit %s/%s: FAK SELL placed after balance-cache refresh (%s)",
                        slug,
                        outcome,
                        exit_oid,
                    )
                    return
                except Exception as exc2:
                    if not is_zero_share_balance_rejection(exc2):
                        logger.exception(
                            "exit %s/%s: post-refresh SELL failed (non-balance); reconcile retries",
                            slug,
                            outcome,
                        )
                        return
            if record_zero_balance_strike(state, condition_id, outcome):
                clear_held_leg(state, condition_id, outcome)
                logger.warning(
                    "exit %s/%s: CLOB reports 0 share balance — position already exited "
                    "on-chain, clearing phantom leg (terminal, no retry)",
                    slug,
                    outcome,
                )
                strat(
                    "phantom_clear", slug=slug, outcome=outcome, cost_basis=entry_cost, path="fak"
                )
            else:
                logger.warning(
                    "exit %s/%s: CLOB reports 0 share balance — entry may not be mined "
                    "yet; sweep will retry (strike toward phantom clear)",
                    slug,
                    outcome,
                )
            return
        if is_invalid_price_rejection(exc):
            logger.warning(
                "exit %s/%s: only sub-min-tick bids exist — a marketable SELL of %s shares "
                "(cost_basis=$%s) would price below the exchange minimum and is rejected; leaving "
                "held (NOT booked, no GTC fallback — it would fail identically), sweep retries "
                "when a >=1-tick bid appears",
                slug,
                outcome,
                size,
                entry_cost,
            )
            strat("exit_subtick", slug=slug, outcome=outcome, size=size, cost_basis=entry_cost)
            return
        if is_overcommit_rejection(exc):
            logger.info(
                "exit %s/%s: SELL of %s shares over-commits — a resting/in-flight exit order "
                "already reserves part of the balance; deferring (transient, no GTC fallback — it "
                "would fail identically), sweep retries once that order fills or is reaped",
                slug,
                outcome,
                size,
            )
            strat("exit_overcommit", slug=slug, outcome=outcome, size=size, cost_basis=entry_cost)
            return
        parsed = parse_partial_balance(exc)
        if parsed is not None and Decimal(0) < parsed[0] <= EXIT_DUST_BALANCE_SHARES:
            dust_balance_writeoff(
                state, condition_id, slug, outcome, parsed[0], size, entry_cost, path="fak"
            )
            return
        elif parsed is not None and parsed[0] > EXIT_DUST_BALANCE_SHARES and parsed[0] < parsed[1]:
            await clamp_leg_to_balance(
                client,
                state,
                token_id,
                size,
                condition_id,
                slug,
                outcome,
                parsed[0],
                entry_cost,
                force_dump,
                _clamp_depth,
            )
            return
        logger.exception("FAK SELL failed for %s/%s; falling back to GTC", slug, outcome)

    try:
        book = await get_order_book(token_id)
        best_bid = max((level.price for level in book.bids), default=Decimal("0"))
    except Exception:
        logger.exception("get_order_book failed for GTC fallback %s/%s", slug, outcome)
        if record_market_failure(state, condition_id):
            mark_paused(state, condition_id)
        return

    if size < book.min_order_size:
        logger.warning(
            "exit %s/%s: DUST WRITE-OFF %s shares (cost_basis=$%s) below book min_order_size %s — "
            "un-sellable, abandoning. NOT booked to session_loss; capital lost if this leg "
            "resolves against us",
            slug,
            outcome,
            size,
            entry_cost,
            book.min_order_size,
        )
        strat(
            "dust_writeoff",
            slug=slug,
            outcome=outcome,
            size=size,
            cost_basis=entry_cost,
            orphan=(1 if pos is None else 0),
            booked=0,
        )
        if pos is not None:
            clear_held_leg(state, condition_id, outcome)
        return

    if best_bid <= 0:
        logger.warning(
            "exit %s/%s: no bid (best_bid=0), cannot sell %s shares (cost_basis=$%s) — leaving "
            "held, marked at TOTAL LOSS by the MTM kill; sweep will retry",
            slug,
            outcome,
            size,
            entry_cost,
        )
        strat("exit_no_bid", slug=slug, outcome=outcome, size=size, cost_basis=entry_cost)
        return

    if best_bid < book.tick_size:
        logger.warning(
            "exit %s/%s: best bid %s below min tick %s — a SELL would be rejected as invalid "
            "price, cannot sell %s shares (cost_basis=$%s); leaving held, marked at TOTAL LOSS by "
            "the MTM kill; sweep retries when a >=1-tick bid appears",
            slug,
            outcome,
            best_bid,
            book.tick_size,
            size,
            entry_cost,
        )
        strat("exit_subtick", slug=slug, outcome=outcome, size=size, cost_basis=entry_cost)
        return

    sell_order = LimitOrder(
        token_id=token_id,
        side="SELL",
        size=float(size),
        price=float(best_bid),
    )
    try:
        exit_oid = await place_limit_order(client, sell_order, post_only=False)
        register_exit(state, condition_id, exit_oid, size, slug, entry_cost, outcome)
        logger.info("exit %s/%s: GTC SELL placed at %s (%s)", slug, outcome, best_bid, exit_oid)
    except Exception as exc:
        if is_zero_share_balance_rejection(exc):
            if record_zero_balance_strike(state, condition_id, outcome):
                clear_held_leg(state, condition_id, outcome)
                logger.warning(
                    "exit %s/%s: CLOB reports 0 share balance on GTC fallback — position "
                    "already exited on-chain, clearing phantom leg (terminal, no retry)",
                    slug,
                    outcome,
                )
                strat(
                    "phantom_clear", slug=slug, outcome=outcome, cost_basis=entry_cost, path="gtc"
                )
            else:
                logger.warning(
                    "exit %s/%s: CLOB reports 0 share balance on GTC fallback — entry may "
                    "not be mined yet; sweep will retry (strike toward phantom clear)",
                    slug,
                    outcome,
                )
            return
        if is_overcommit_rejection(exc):
            logger.info(
                "exit %s/%s: GTC SELL over-commits — a resting/in-flight exit order already "
                "reserves part of the balance; deferring (transient), sweep retries once it clears",
                slug,
                outcome,
            )
            strat("exit_overcommit", slug=slug, outcome=outcome, size=size, cost_basis=entry_cost)
            return
        parsed = parse_partial_balance(exc)
        if parsed is not None and Decimal(0) < parsed[0] <= EXIT_DUST_BALANCE_SHARES:
            dust_balance_writeoff(
                state, condition_id, slug, outcome, parsed[0], size, entry_cost, path="gtc"
            )
            return
        elif parsed is not None and parsed[0] > EXIT_DUST_BALANCE_SHARES and parsed[0] < parsed[1]:
            await clamp_leg_to_balance(
                client,
                state,
                token_id,
                size,
                condition_id,
                slug,
                outcome,
                parsed[0],
                entry_cost,
                force_dump,
                _clamp_depth,
            )
            return
        logger.exception(
            "GTC fallback failed for %s/%s — shares still held, sweep will retry", slug, outcome
        )
        if record_market_failure(state, condition_id):
            mark_paused(state, condition_id)


def fresh_staged_tokens(state: FarmState, now: datetime) -> set[str]:
    fresh: set[str] = set()
    cutoff = now - timedelta(seconds=STALE_STAGED_EXIT_SECONDS)
    for trade_id, info in list(state.pending_fok_exits.items()):
        if info.staged_at is None or info.staged_at >= cutoff:
            fresh.add(info.token_id)
        else:
            state.pending_fok_exits.pop(trade_id, None)
            logger.warning(
                "stale staged exit for %s/%s (lost MINED?) — re-driving via sweep",
                info.slug,
                info.outcome,
            )
    return fresh


async def reap_stale_exit_orders(client, state: FarmState, pos, now: datetime) -> None:
    for oid, eo in list(pos.exit_orders.items()):
        age = (now - eo.placed_at).total_seconds()
        floor_giveup = eo.is_floor and age >= SMART_EXIT_MAX_HOLD_SECONDS
        if not floor_giveup and age < STALE_EXIT_SECONDS:
            continue
        try:
            await cancel_order(client, oid)
        except Exception:
            logger.exception("reaper: cancel of stale exit %s failed — keeping tracked", oid)
            continue
        pos.exit_orders.pop(oid, None)
        state.pending_exit_order_ids.discard(oid)
        state.exit_cost_basis.pop(oid, None)
        if floor_giveup:
            is_yes = eo.outcome == "YES"
            shares = pos.yes_shares if is_yes else pos.no_shares
            if shares > 0:
                tok = pos.market.yes_token_id if is_yes else pos.market.no_token_id
                cost = pos.yes_cost_basis if is_yes else pos.no_cost_basis
                logger.warning(
                    "smart-floor give-up after %.0fs on %s/%s — book did not recover, dumping "
                    "%s shares (FAK)",
                    age,
                    pos.market.slug,
                    eo.outcome,
                    shares,
                )
                strat(
                    "smart_exit_giveup",
                    slug=pos.market.slug,
                    outcome=eo.outcome,
                    held_s=round(age),
                    shares=shares,
                )
                try:
                    await exit_position_leg(
                        client,
                        state,
                        tok,
                        shares,
                        pos.market.condition_id,
                        pos.market.slug,
                        eo.outcome,
                        entry_cost=cost,
                        force_dump=True,
                    )
                except Exception:
                    logger.exception(
                        "smart-floor give-up dump failed for %s/%s", pos.market.slug, eo.outcome
                    )
            continue
        logger.warning(
            "reaped stale exit %s for %s/%s — re-driving via sweep",
            oid,
            pos.market.slug,
            eo.outcome,
        )
        strat("exit_reaped", slug=pos.market.slug, outcome=eo.outcome, oid=oid)


async def exit_held_legs(
    client,
    state: FarmState,
    *,
    cancel_resting: bool,
    skip_in_flight: bool,
    force_dump: bool = False,
) -> None:
    now = datetime.now(timezone.utc)
    staged_tokens = fresh_staged_tokens(state, now)
    for cid, pos in list(state.positions.items()):
        if skip_in_flight:
            await reap_stale_exit_orders(client, state, pos, now)
        for token_id, shares, cost, outcome in (
            (pos.market.yes_token_id, pos.yes_shares, pos.yes_cost_basis, "YES"),
            (pos.market.no_token_id, pos.no_shares, pos.no_cost_basis, "NO"),
        ):
            if shares <= 0:
                continue
            if skip_in_flight and any(eo.outcome == outcome for eo in pos.exit_orders.values()):
                continue
            if skip_in_flight and token_id in staged_tokens:
                continue
            try:
                await exit_position_leg(
                    client,
                    state,
                    token_id,
                    shares,
                    cid,
                    pos.market.slug,
                    outcome,
                    cancel_resting=cancel_resting,
                    entry_cost=cost,
                    force_dump=force_dump,
                )
            except Exception:
                logger.exception("exit_held_legs: leg failed for %s/%s", pos.market.slug, outcome)


async def reconcile_onchain_positions(client, state: FarmState, http) -> None:
    positions = await fetch_open_positions(http, state.wallet_address)
    if not positions:
        return
    now = datetime.now(timezone.utc)
    held_keys = {exit_key(p.condition_id, p.outcome) for p in positions if p.size > 0}
    state.reconcile_attempts = {k: v for k, v in state.reconcile_attempts.items() if k in held_keys}
    for p in positions:
        if p.size <= 0:
            continue
        outcome = p.outcome.upper()
        key = exit_key(p.condition_id, outcome)
        attempts = state.reconcile_attempts.get(key, 0)
        backoff = RECENT_EXIT_GRACE_SECONDS * 2 ** min(attempts, 16)
        wait = min(backoff, RECONCILE_BACKOFF_MAX_SECONDS)
        last = state.recent_exits.get(key)
        if last is not None and (now - last) < timedelta(seconds=wait):
            continue
        pos = state.positions.get(p.condition_id)
        if pos is not None and any(eo.outcome == outcome for eo in pos.exit_orders.values()):
            continue
        attempts += 1
        state.reconcile_attempts[key] = attempts
        state.recent_exits[key] = now
        log = logger.warning if attempts <= 5 else logger.debug
        log(
            "on-chain reconcile: wallet holds %s %s shares of %s the bot is NOT exiting — "
            "re-driving exit NOW (attempt %d; never gives up, next retry in <=%ds)",
            p.size,
            outcome,
            p.slug,
            attempts,
            wait,
        )
        strat(
            "onchain_reconcile_exit",
            slug=p.slug,
            cid=p.condition_id,
            outcome=outcome,
            size=p.size,
            attempt=attempts,
        )
        if attempts == 1 and not is_blacklisted(state, p.condition_id, now):
            logger.warning(
                "on-chain reconcile: recovered an untracked fill on %s (missed WS frame) — "
                "blacklisting to stop re-quote churn",
                p.slug,
            )
            blacklist_for_fill(state, p.condition_id, now)
        entry_cost = p.size * p.avg_price
        if pos is not None:
            is_yes = outcome == "YES"
            leg_shares = pos.yes_shares if is_yes else pos.no_shares
            leg_cost = pos.yes_cost_basis if is_yes else pos.no_cost_basis
            if leg_cost > 0 and leg_shares > 0:
                entry_cost = leg_cost * (p.size / leg_shares)
        try:
            await exit_position_leg(
                client,
                state,
                p.token_id,
                p.size,
                p.condition_id,
                p.slug,
                outcome,
                entry_cost=entry_cost,
            )
        except Exception:
            logger.exception("on-chain reconcile exit failed for %s/%s", p.slug, outcome)
