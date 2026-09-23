import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from fastapi import WebSocket

from app.api.farm.messages import (
    OrderCancelledEvent,
    OrderFilledEvent,
)
from app.api.messages import send_event
from app.bot.cancel import cancel_order, cancel_orders
from app.bot.schemas import BookLevel, UserTrade
from app.bot.user_ws import stream_user_trades
from app.constants import EVENT_EXCLUSION_SECONDS
from app.farm.exit_cost import exit_vacuum_price
from app.farm.exits import exit_position_leg, register_exit
from app.farm.fees import taker_fee
from app.farm.health import is_event_excluded
from app.farm.kill_switch import record_realized_pnl, should_kill, trigger_kill
from app.farm.schemas import BookSample, FarmState, FokExitInfo
from app.farm.volatility import blacklist_for_fill, record_roundtrip_pnl
from app.infra.strategy_log import strat

logger = logging.getLogger(__name__)


async def cancel_family_orders(client, state: FarmState, event_slug: str, *, skip_cid: str) -> None:
    for cid, sib in list(state.positions.items()):
        if cid == skip_cid or sib.market.event_slug != event_slug:
            continue
        if sib.yes_shares > 0 or sib.no_shares > 0:
            continue
        oids = [oid for oid in (sib.yes_order_id, sib.no_order_id) if oid]
        if oids:
            await cancel_orders(client, *oids)
        for oid in oids:
            state.order_registry.pop(oid, None)
        state.positions.pop(cid, None)
        logger.warning(
            "family-cancel: pulled resting orders + dropped sibling %s (event %s quarantined)",
            sib.market.slug,
            event_slug,
        )


async def user_ws_loop(client, state: FarmState, websocket: WebSocket) -> None:
    async for trade in stream_user_trades(client):
        await handle_trade(client, trade, state, websocket)


async def handle_exit_fill(
    client,
    state: FarmState,
    websocket: WebSocket,
    *,
    exit_oid: str,
    sold_size: Decimal,
    sell_price: Decimal,
    market_id: str,
    outcome: str,
    token_id: str,
) -> None:
    state.pending_exit_order_ids.discard(exit_oid)
    cost_info = state.exit_cost_basis.pop(exit_oid, None)
    pos = state.positions.get(market_id)
    if pos is not None:
        pos.exit_orders.pop(exit_oid, None)
    if cost_info is not None:
        slug = cost_info.slug
    elif pos is not None:
        slug = pos.market.slug
    else:
        slug = market_id

    known_cost = Decimal(0)
    covered_size = Decimal(0)
    if cost_info is not None and cost_info.entry_size > 0:
        covered = min(sold_size, cost_info.entry_size)
        if cost_info.entry_cost > 0:
            known_cost += cost_info.entry_cost * (covered / cost_info.entry_size)
        else:
            known_cost += covered * sell_price
        covered_size += covered

    residual = Decimal(0)
    residual_token_id: str | None = None
    residual_cost = Decimal(0)
    if pos is not None:
        if outcome == "YES":
            held, cost_basis = pos.yes_shares, pos.yes_cost_basis
        else:
            held, cost_basis = pos.no_shares, pos.no_cost_basis
        if sold_size > held:
            logger.warning(
                "SELL size %s exceeds held %s for %s/%s — divergence "
                "(booking actual fill, not clamped)",
                sold_size,
                held,
                slug,
                outcome,
            )
        booked = min(sold_size, held)
        pos_cost_released = cost_basis * (booked / held) if held > 0 else Decimal(0)
        new_cost_basis = cost_basis - pos_cost_released
        new_shares = held - booked
        if outcome == "YES":
            pos.yes_shares, pos.yes_cost_basis = new_shares, new_cost_basis
            residual_token_id = pos.market.yes_token_id
        else:
            pos.no_shares, pos.no_cost_basis = new_shares, new_cost_basis
            residual_token_id = pos.market.no_token_id
        residual = new_shares
        residual_cost = new_cost_basis
        if cost_info is None and held > 0:
            known_cost += pos_cost_released
            covered_size += booked
    elif cost_info is not None and cost_info.entry_size > 0:
        orphan_residual = cost_info.entry_size - sold_size
        if orphan_residual > 0:
            residual = orphan_residual
            residual_token_id = token_id
            residual_cost = cost_info.entry_cost * (orphan_residual / cost_info.entry_size)

    uncovered = sold_size - covered_size
    if uncovered > 0:
        if covered_size <= 0:
            logger.warning(
                "exit fill %s for %s/%s has no recoverable entry cost — "
                "booking break-even PnL (divergence)",
                exit_oid,
                slug,
                outcome,
            )
        known_cost += uncovered * sell_price

    fee_rate = pos.market.taker_fee_rate if pos is not None else Decimal(0)
    fee = taker_fee(sold_size, sell_price, fee_rate)
    net = sold_size * sell_price - known_cost - fee
    record_realized_pnl(state, known_cost, sold_size, sell_price, slug, outcome, fee=fee)
    record_roundtrip_pnl(state, market_id, outcome, net, closed=residual <= 0)
    strat(
        "roundtrip",
        slug=slug,
        outcome=outcome,
        entry_cost=known_cost,
        proceeds=sold_size * sell_price,
        fee=fee,
        net=net,
        size=sold_size,
        exit_px=sell_price,
    )
    was_killed = state.killed
    if should_kill(state):
        await trigger_kill(client, state, websocket)
    kill_just_fired = not was_killed and state.killed
    if residual > 0 and residual_token_id is not None and not kill_just_fired:
        try:
            await cancel_order(client, exit_oid)
            cancelled = True
        except Exception:
            cancelled = False
            logger.exception("cancel of partial exit %s failed; keeping it tracked", exit_oid)
        if cancelled:
            logger.info(
                "partial exit fill — re-driving residual %s shares of %s/%s",
                residual,
                slug,
                outcome,
            )
            try:
                await exit_position_leg(
                    client,
                    state,
                    residual_token_id,
                    residual,
                    market_id,
                    slug,
                    outcome,
                    entry_cost=residual_cost,
                )
            except Exception:
                logger.exception("residual exit retry failed for %s/%s", slug, outcome)
        else:
            register_exit(state, market_id, exit_oid, residual, slug, residual_cost, outcome)


async def stage_orphan_exit(client, state: FarmState, trade: UserTrade, maker, info) -> None:
    size = maker.matched_amount
    if size <= 0:
        return
    cost = size * maker.price
    slug = info.condition_id

    blacklist_for_fill(state, info.condition_id)
    state.total_volume += cost
    logger.warning(
        "orphan fill: registered maker %s filled for DROPPED position %s — "
        "auto-exiting %s %s shares once MINED",
        maker.order_id,
        info.condition_id,
        size,
        info.outcome,
    )
    strat(
        "fill",
        slug=slug,
        outcome=info.outcome,
        px=maker.price,
        size=size,
        cost=cost,
        mid="na",
        orphaned=True,
    )

    try:
        await cancel_order(client, maker.order_id)
    except Exception:
        logger.exception("orphan fill: cancel of %s failed", maker.order_id)

    state.pending_fok_exits[trade.id] = FokExitInfo(
        token_id=info.token_id,
        size=size,
        outcome=info.outcome,
        slug=slug,
        entry_cost=cost,
        staged_at=datetime.now(timezone.utc),
    )


async def handle_trade(client, trade: UserTrade, state: FarmState, websocket: WebSocket) -> None:
    event_key = (trade.id, trade.status)
    if event_key in state.processed_events:
        return
    state.processed_events.add(event_key)

    exit_fills = (
        [(trade.taker_order_id, trade.size, trade.price, trade.outcome, trade.asset_id)]
        if trade.taker_order_id in state.pending_exit_order_ids
        else []
    )
    exit_fills += [
        (m.order_id, m.matched_amount, m.price, m.outcome, m.asset_id)
        for m in trade.maker_orders
        if m.order_id in state.pending_exit_order_ids
    ]
    if exit_fills:
        if trade.status in ("MINED", "CONFIRMED"):
            for oid, sold, price, oc, tok in exit_fills:
                key = (trade.id, oid)
                if key in state.booked_exit_fills:
                    continue
                # Mark AFTER the booking succeeds, not before: if handle_exit_fill ever raises
                # (today it books synchronously with no I/O first, but that could change), a
                # replayed MINED/CONFIRMED event must be free to re-book rather than be skipped.
                # Trades are processed sequentially (user_ws_loop awaits one at a time), so there
                # is no re-entrancy window that would double-book between the await and the add.
                await handle_exit_fill(
                    client,
                    state,
                    websocket,
                    exit_oid=oid,
                    sold_size=sold,
                    sell_price=price,
                    market_id=trade.market,
                    outcome=oc,
                    token_id=tok,
                )
                state.booked_exit_fills.add(key)
        elif trade.status == "RETRYING":
            for oid, *rest in exit_fills:
                logger.warning(
                    "exit SELL %s RETRYING — operator auto-retrying, awaiting settlement", oid
                )
        elif trade.status == "FAILED":
            pos = state.positions.get(trade.market)
            for oid, _sold, _price, oc, tok in exit_fills:
                cost_info = state.exit_cost_basis.pop(oid, None)
                state.pending_exit_order_ids.discard(oid)
                if pos is not None:
                    pos.exit_orders.pop(oid, None)
                    shares = pos.yes_shares if oc == "YES" else pos.no_shares
                    cost = pos.yes_cost_basis if oc == "YES" else pos.no_cost_basis
                    if shares > 0:
                        logger.warning(
                            "exit SELL %s FAILED — re-firing sell for held %s %s/%s shares NOW",
                            oid,
                            shares,
                            pos.market.slug,
                            oc,
                        )
                        try:
                            await exit_position_leg(
                                client,
                                state,
                                tok,
                                shares,
                                trade.market,
                                pos.market.slug,
                                oc,
                                entry_cost=cost,
                            )
                        except Exception:
                            logger.exception(
                                "immediate FAILED re-drive failed for %s/%s", pos.market.slug, oc
                            )
                    else:
                        logger.warning("exit SELL %s FAILED but %s leg already flat", oid, oc)
                elif cost_info is not None and cost_info.entry_size > 0:
                    logger.warning(
                        "orphan exit SELL %s FAILED — re-driving %s shares NOW",
                        oid,
                        cost_info.entry_size,
                    )
                    try:
                        await exit_position_leg(
                            client,
                            state,
                            tok,
                            cost_info.entry_size,
                            trade.market,
                            cost_info.slug,
                            oc,
                            entry_cost=cost_info.entry_cost,
                        )
                    except Exception:
                        logger.exception("orphan re-drive failed for %s/%s", cost_info.slug, oc)
                else:
                    logger.warning("exit SELL %s FAILED — cleared tracking", oid)
        return

    if trade.status == "MINED":
        info = state.pending_fok_exits.get(trade.id)
        if info is None or info.size <= 0:
            state.pending_fok_exits.pop(trade.id, None)
            return
        try:
            await exit_position_leg(
                client,
                state,
                info.token_id,
                info.size,
                trade.market,
                info.slug,
                info.outcome,
                entry_cost=info.entry_cost,
            )
        finally:
            state.pending_fok_exits.pop(trade.id, None)
        return

    if trade.status == "FAILED":
        state.pending_fok_exits.pop(trade.id, None)
        return

    if trade.status != "MATCHED":
        return

    if trade.id in state.pending_fok_exits:
        return

    for maker in trade.maker_orders:
        info = state.order_registry.get(maker.order_id)
        if info is None:
            continue

        pos = state.positions.get(info.condition_id)
        if pos is None:
            await stage_orphan_exit(client, state, trade, maker, info)
            continue

        outcome = info.outcome
        token_id = info.token_id
        size = maker.matched_amount
        cost = size * maker.price
        old_oid = maker.order_id

        blacklist_for_fill(state, info.condition_id)

        if pos.market.event_slug:
            event_slug = pos.market.event_slug
            newly_excluded = not is_event_excluded(state, event_slug)
            state.excluded_events[event_slug] = datetime.now(timezone.utc) + timedelta(
                seconds=EVENT_EXCLUSION_SECONDS
            )
            strat(
                "event_excluded",
                family=event_slug,
                slug=pos.market.slug,
                outcome=outcome,
                ttl_min=EVENT_EXCLUSION_SECONDS // 60,
                refresh=(0 if newly_excluded else 1),
            )
            if newly_excluded:
                logger.warning(
                    "excluded event family %s after fill on %s/%s for %dmin — pulling resting "
                    "sibling orders, auto re-enters after",
                    event_slug,
                    pos.market.slug,
                    outcome,
                    EVENT_EXCLUSION_SECONDS // 60,
                )
                try:
                    await cancel_family_orders(
                        client, state, event_slug, skip_cid=info.condition_id
                    )
                except Exception:
                    logger.exception("family-cancel failed for event %s", event_slug)
            else:
                logger.info(
                    "event family %s quarantine refreshed after repeat fill on %s/%s",
                    event_slug,
                    pos.market.slug,
                    outcome,
                )

        if outcome == "YES":
            pos.yes_shares += size
            pos.yes_cost_basis += cost
            shares_after = pos.yes_shares
            cost_basis_after = pos.yes_cost_basis
        else:
            pos.no_shares += size
            pos.no_cost_basis += cost
            shares_after = pos.no_shares
            cost_basis_after = pos.no_cost_basis

        state.total_volume += cost

        mid = (
            (pos.last_best_bid + pos.last_best_ask) / 2
            if pos.last_best_bid is not None and pos.last_best_ask is not None
            else None
        )
        hold_s = round((datetime.now(timezone.utc) - info.placed_at).total_seconds(), 1)
        strat(
            "fill",
            slug=pos.market.slug,
            outcome=outcome,
            px=maker.price,
            size=size,
            cost=cost,
            hold_s=hold_s,
            mid=mid if mid is not None else "na",
        )
        filled_token = pos.market.yes_token_id if outcome == "YES" else pos.market.no_token_id
        fbook = state.live_books.get(filled_token)
        if fbook is not None:
            band = pos.market.rewards_max_spread_cents / Decimal(100)
            leg_mid = (max(fbook.bids) + min(fbook.asks)) / 2 if fbook.bids and fbook.asks else None
            fill_bid_depth = (
                sum((s for p, s in fbook.bids.items() if abs(p - leg_mid) <= band), Decimal(0))
                if leg_mid is not None
                else sum(fbook.bids.values(), Decimal(0))
            )
            strat(
                "bid_depth_fill",
                slug=pos.market.slug,
                outcome=outcome,
                bid_depth=float(fill_bid_depth),
                size=float(size),
                ratio=round(float(fill_bid_depth / size), 2) if size else "na",
                bid_levels=len(fbook.bids),
                inband=leg_mid is not None,
            )

        # --- pure instrumentation below: post-fill book sampler registration, the shadow-mode
        # vacuum classifier, and enhanced fill logging. Logging only — never touches order
        # placement/cancel/kill, and any failure here must never propagate into the fill path. ---
        try:
            state.book_samples[filled_token] = BookSample(
                token_id=filled_token,
                slug=pos.market.slug,
                outcome=outcome,
                entry_px=maker.price,
                started_at=datetime.now(timezone.utc),
            )
        except Exception:
            logger.exception(
                "post-fill sampler: failed to register %s/%s for book_samples",
                pos.market.slug,
                outcome,
            )

        try:
            reward_band = pos.market.rewards_max_spread_cents / Decimal(100)
            if fbook is None or (not fbook.bids and not fbook.asks):
                strat(
                    "vacuum_shadow",
                    slug=pos.market.slug,
                    outcome=outcome,
                    verdict="no_book",
                    best_bid="na",
                    best_ask="na",
                    spread="na",
                    entry_px=maker.price,
                    rest_px="na",
                    reward_band=reward_band,
                )
            else:
                vac_bids = [BookLevel(price=p, size=s) for p, s in fbook.bids.items()]
                vac_asks = [BookLevel(price=p, size=s) for p, s in fbook.asks.items()]
                # The no_book branch above only catches a book empty on BOTH sides, so an empty
                # bid side with a live ask still lands here — and that one-sided vacuum is the
                # exact shape this classifier exists to study. Default to None, not 0: a missing
                # bid logged as `best_bid=0` reads as a real 0-priced bid and yields a spread
                # measured off zero, which would silently poison the sampler's own dataset.
                vac_best_bid = max((lvl.price for lvl in vac_bids), default=None)
                vac_best_ask = min((lvl.price for lvl in vac_asks), default=None)
                have_both = vac_best_bid is not None and vac_best_ask is not None
                rest_px = exit_vacuum_price(
                    vac_bids, vac_asks, maker.price, pos.market.tick_size, reward_band
                )
                strat(
                    "vacuum_shadow",
                    slug=pos.market.slug,
                    outcome=outcome,
                    verdict="fire" if rest_px is not None else "dump",
                    best_bid=vac_best_bid if vac_best_bid is not None else "na",
                    best_ask=vac_best_ask if vac_best_ask is not None else "na",
                    spread=(vac_best_ask - vac_best_bid) if have_both else "na",
                    entry_px=maker.price,
                    rest_px=rest_px if rest_px is not None else "na",
                    reward_band=reward_band,
                )
        except Exception:
            logger.exception("vacuum shadow classifier failed for %s/%s", pos.market.slug, outcome)

        try:
            if fbook is not None:
                fb_bid_levels = sorted(fbook.bids.items(), key=lambda kv: -kv[0])
                fb_ask_levels = sorted(fbook.asks.items(), key=lambda kv: kv[0])
                fb_best_bid = fb_bid_levels[0][0] if fb_bid_levels else None
                fb_best_ask = fb_ask_levels[0][0] if fb_ask_levels else None
                fb_mid = (
                    (fb_best_bid + fb_best_ask) / 2
                    if fb_best_bid is not None and fb_best_ask is not None
                    else None
                )
                strat(
                    "fill_book",
                    slug=pos.market.slug,
                    outcome=outcome,
                    best_bid=fb_best_bid if fb_best_bid is not None else "na",
                    best_ask=fb_best_ask if fb_best_ask is not None else "na",
                    mid=fb_mid if fb_mid is not None else "na",
                    top_bids=fb_bid_levels[:5],
                    top_asks=fb_ask_levels[:3],
                )
        except Exception:
            logger.exception("fill_book logging failed for %s/%s", pos.market.slug, outcome)

        try:
            strat(
                "fill_exposure",
                slug=pos.market.slug,
                outcome=outcome,
                size=size,
                entry_px=maker.price,
                usd_exposure=size * maker.price,
            )
        except Exception:
            logger.exception("fill_exposure logging failed for %s/%s", pos.market.slug, outcome)

        await send_event(
            websocket,
            OrderFilledEvent(
                market_id=trade.market,
                slug=pos.market.slug,
                outcome=outcome,
                side=trade.side,
                price=maker.price,
                size=size,
                shares_after=shares_after,
                cost_basis_after=cost_basis_after,
            ),
        )

        cancel_targets: dict[str, str] = {}
        for oid, oc in ((old_oid, outcome), (pos.yes_order_id, "YES"), (pos.no_order_id, "NO")):
            if oid and oid not in cancel_targets:
                cancel_targets[oid] = oc
        logger.warning(
            "fill on %s/%s: pulling ALL %d resting order(s) on market (oids=%s) to prevent "
            "double-fill",
            pos.market.slug,
            outcome,
            len(cancel_targets),
            ",".join(cancel_targets),
        )
        strat(
            "fill_cancel_all",
            slug=pos.market.slug,
            outcome=outcome,
            n_pulled=len(cancel_targets),
            filled_oid=old_oid,
        )
        for cancel_oid, cancel_outcome in cancel_targets.items():
            try:
                await cancel_order(client, cancel_oid)
            except Exception:
                logger.exception(
                    "fill cancel failed for %s/%s oid=%s",
                    pos.market.slug,
                    cancel_outcome,
                    cancel_oid,
                )
            await send_event(
                websocket,
                OrderCancelledEvent(
                    market_id=trade.market,
                    slug=pos.market.slug,
                    outcome=cancel_outcome,
                    order_id=cancel_oid,
                    reason="filled_exit",
                ),
            )

        state.pending_fok_exits[trade.id] = FokExitInfo(
            token_id=token_id,
            size=size,
            outcome=outcome,
            slug=pos.market.slug,
            entry_cost=cost,
            staged_at=datetime.now(timezone.utc),
        )
