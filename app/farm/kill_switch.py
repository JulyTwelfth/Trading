import logging
from decimal import Decimal

from fastapi import WebSocket

from app.api.farm.messages import FarmKilledEvent
from app.api.messages import send_event
from app.bot.cancel import cancel_all
from app.bot.schemas import BookLevel
from app.farm.exit_cost import liquidation_proceeds
from app.farm.exits import exit_held_legs
from app.farm.schemas import FarmState
from app.infra.strategy_log import strat

logger = logging.getLogger(__name__)


def record_realized_pnl(
    state: FarmState,
    entry_cost_released: Decimal,
    sold_size: Decimal,
    sell_price: Decimal,
    slug: str,
    outcome: str,
    fee: Decimal = Decimal(0),
) -> Decimal:
    # `fee` is the taker fee the protocol skims from the exit proceeds. It reduces the
    # USDC we actually receive, so it adds directly to the session loss — without it the
    # kill switch under-counts real losses (fees were a ~$18/session untracked drain).
    proceeds = sold_size * sell_price
    loss_delta = entry_cost_released - proceeds + fee
    state.session_loss += loss_delta
    logger.info(
        "kill_switch pnl: %s/%s entry_cost=%s proceeds=%s fee=%s delta=%s session_loss=%s",
        slug,
        outcome,
        entry_cost_released,
        proceeds,
        fee,
        loss_delta,
        state.session_loss,
    )
    return state.session_loss


def leg_mark_loss(
    state: FarmState,
    token_id: str,
    shares: Decimal,
    cost_basis: Decimal,
    best_bid: Decimal | None,
) -> Decimal:
    """Unrealized loss on one held leg: cost basis minus liquidation proceeds. Walks live book
    depth when available (a thin book is valued honestly), else falls back to best_bid, then 0.
    Positive = loss; a marked-up leg returns negative (offsetting other legs)."""
    if shares <= 0:
        return Decimal(0)
    book = state.live_books.get(token_id)
    if book is not None and book.bids:
        bids = [BookLevel(price=p, size=s) for p, s in book.bids.items()]
        return cost_basis - liquidation_proceeds(bids, shares)
    if best_bid is not None:
        return cost_basis - shares * best_bid
    return Decimal(0)


def unrealized_loss(state: FarmState) -> Decimal:
    """Net mark-to-market loss on held inventory across all positions, marked against live book
    depth. Lets the kill switch see a crash drawdown before (or even if never) an exit fills —
    abandoned/dust shares book no realized loss, so a realized-only cap would never trip."""
    total = Decimal(0)
    for pos in state.positions.values():
        total += leg_mark_loss(
            state, pos.market.yes_token_id, pos.yes_shares, pos.yes_cost_basis, pos.yes_best_bid
        )
        total += leg_mark_loss(
            state, pos.market.no_token_id, pos.no_shares, pos.no_cost_basis, pos.no_best_bid
        )
    return total


def session_reward(state: FarmState) -> Decimal:
    """Reward accrued this session = sum over markets of (today's earned - midnight-UTC-rebased
    baseline). Real payouts that offset fill losses, but they settle at midnight UTC while losses
    hit immediately — so a profitable session can dip cash intraday; size the bankroll for it."""
    total = Decimal(0)
    for cid, earned in state.market_rewards.items():
        total += earned - state.market_rewards_baseline.get(cid, earned)
    return total


def net_session_loss(state: FarmState) -> Decimal:
    return state.session_loss + unrealized_loss(state) - session_reward(state)


def should_kill(state: FarmState) -> bool:
    """Trip only when TRULY losing money: realized + open mark-to-market loss, net of accrued
    reward, reaches the cap (a reward-blind gross cap killed a +$10 run 2026-06-20: $13 vs $23
    reward). Unrealized loss still counts, so a crash that outruns the reward still trips."""
    return net_session_loss(state) >= state.config.max_session_loss


async def trigger_kill(
    client, state: FarmState, websocket: WebSocket, source: str = "realtime"
) -> None:
    if state.killed:
        return
    state.killed = True
    unreal = unrealized_loss(state)
    reward = session_reward(state)
    total = state.session_loss + unreal
    net = total - reward
    logger.warning(
        "kill_switch tripped (%s): net_loss=%s (realized=%s unrealized_mtm=%s reward=%s) "
        "threshold=%s",
        source,
        net,
        state.session_loss,
        unreal,
        reward,
        state.config.max_session_loss,
    )
    strat(
        "kill",
        reason="max_session_loss",
        realized=state.session_loss,
        unrealized=unreal,
        reward=reward,
        net=net,
        total=total,
        threshold=state.config.max_session_loss,
        positions=len(state.positions),
        source=source,
    )

    try:
        await cancel_all(client)
    except Exception:
        logger.exception("kill_switch: cancel_all failed")

    await exit_held_legs(client, state, cancel_resting=False, skip_in_flight=False, force_dump=True)

    try:
        await send_event(
            websocket,
            FarmKilledEvent(
                reason="max_session_loss",
                session_loss=state.session_loss,
                unrealized_loss=unreal,
                total_loss=total,
                session_reward=reward,
                net_loss=net,
            ),
        )
    except Exception:
        logger.exception("kill_switch: failed to emit FarmKilledEvent")
