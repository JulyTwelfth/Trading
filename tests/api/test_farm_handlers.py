import asyncio
import contextlib
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.farm import handlers as handlers_mod
from app.api.farm.handlers import (
    FarmSession,
    handle_farm_cancel,
    handle_farm_create,
    send_farm_error,
)
from app.api.farm.messages import FarmCreateMessage
from app.db.wallets import Wallet
from app.farm.schemas import FarmFilters

VALID_ADDR = "f" * 40
VALID_KEY = "a" * 64


def make_ws() -> MagicMock:
    ws = MagicMock()
    ws.send_json = AsyncMock()
    return ws


def make_filters() -> FarmFilters:
    return FarmFilters(
        vol_min=Decimal(0),
        vol_max=Decimal(100000),
        liq_min=Decimal(0),
        liq_max=Decimal(100000),
        spread_min=Decimal(0),
        spread_max=Decimal(100),
        reward_min=Decimal(0),
        time_remaining="all",
        created_date="all",
        change_24h="all",
    )


def make_msg(bankroll="100") -> FarmCreateMessage:
    return FarmCreateMessage(
        type="farm_create",
        filters=make_filters(),
        bankroll=Decimal(bankroll),
        max_session_loss=Decimal("5"),
    )


def make_wallet(wallet_id="A") -> Wallet:
    return Wallet(
        license_key="LIC",
        wallet_id=wallet_id,
        proxy_address="0x" + VALID_ADDR,
        private_key="0x" + VALID_KEY,
    )


def captured_types(ws: MagicMock) -> list[str]:
    return [call.args[0]["type"] for call in ws.send_json.await_args_list]


# ── send_farm_error ──────────────────────────────────────────────────────────


async def test_send_farm_error_emits_event(monkeypatch):
    ws = make_ws()
    send_event = AsyncMock()
    monkeypatch.setattr(handlers_mod, "send_event", send_event)

    await send_farm_error(ws, "boom")

    send_event.assert_awaited_once()
    websocket_arg, event = send_event.await_args.args
    assert websocket_arg is ws
    assert event.type == "farm_error"
    assert event.reason == "boom"


# ── handle_farm_create ───────────────────────────────────────────────────────


async def test_handle_farm_create_already_running(monkeypatch):
    ws = make_ws()
    send_event = AsyncMock()
    monkeypatch.setattr(handlers_mod, "send_event", send_event)
    list_wallets = AsyncMock()
    monkeypatch.setattr(handlers_mod, "list_wallets", list_wallets)

    running = asyncio.create_task(asyncio.sleep(60))
    try:
        result = await handle_farm_create(ws, "LIC", make_msg(), running, FarmSession())

        # Returns the same (still-running) task untouched.
        assert result is running
        # Did not even query wallets.
        list_wallets.assert_not_awaited()
        event = send_event.await_args.args[1]
        assert event.type == "farm_error"
        assert event.reason == "farm_already_running"
    finally:
        running.cancel()


async def test_handle_farm_create_no_wallet_registered(monkeypatch):
    ws = make_ws()
    send_event = AsyncMock()
    monkeypatch.setattr(handlers_mod, "send_event", send_event)
    # No wallets at all: the farm now runs the first wallet whatever its id.
    monkeypatch.setattr(handlers_mod, "list_wallets", AsyncMock(return_value=[]))
    get_balance = AsyncMock()
    monkeypatch.setattr(handlers_mod, "get_balance", get_balance)

    result = await handle_farm_create(ws, "LIC", make_msg(), None, FarmSession())

    assert result is None
    get_balance.assert_not_awaited()
    event = send_event.await_args.args[1]
    assert event.reason == "no_wallet_registered"


async def test_handle_farm_create_insufficient_balance(monkeypatch):
    ws = make_ws()
    send_event = AsyncMock()
    monkeypatch.setattr(handlers_mod, "send_event", send_event)
    monkeypatch.setattr(handlers_mod, "list_wallets", AsyncMock(return_value=[make_wallet("A")]))
    monkeypatch.setattr(handlers_mod, "get_balance", AsyncMock(return_value=Decimal("50")))
    create_task_spy = MagicMock()
    monkeypatch.setattr(handlers_mod.asyncio, "create_task", create_task_spy)

    # bankroll 100 > balance 50.
    result = await handle_farm_create(ws, "LIC", make_msg("100"), None, FarmSession())

    assert result is None
    create_task_spy.assert_not_called()
    event = send_event.await_args.args[1]
    assert event.reason == "insufficient_balance"


async def test_handle_farm_create_success_starts_task(monkeypatch):
    ws = make_ws()
    send_event = AsyncMock()
    monkeypatch.setattr(handlers_mod, "send_event", send_event)
    wallet_a = make_wallet("A")
    monkeypatch.setattr(handlers_mod, "list_wallets", AsyncMock(return_value=[wallet_a]))
    monkeypatch.setattr(handlers_mod, "get_balance", AsyncMock(return_value=Decimal("100")))

    run_farm_calls = []

    async def fake_run_farm(websocket, msg, wallet, license_key, farm_session):
        run_farm_calls.append((wallet, license_key, farm_session))
        await asyncio.sleep(60)

    monkeypatch.setattr(handlers_mod, "run_farm", fake_run_farm)

    session = FarmSession()
    msg = make_msg("100")  # equal balance is allowed (not >)
    result = await handle_farm_create(ws, "LIC", msg, None, session)

    assert isinstance(result, asyncio.Task)
    assert not result.done()
    # Started event emitted.
    event = send_event.await_args.args[1]
    assert event.type == "farm_started"

    # The task wraps run_farm with the resolved wallet A, the license key, and the session.
    await asyncio.sleep(0)  # let the task start running
    assert run_farm_calls == [(wallet_a, "LIC", session)]

    result.cancel()
    try:
        await result
    except asyncio.CancelledError:
        pass


async def test_handle_farm_create_done_task_treated_as_idle(monkeypatch):
    ws = make_ws()
    send_event = AsyncMock()
    monkeypatch.setattr(handlers_mod, "send_event", send_event)
    monkeypatch.setattr(handlers_mod, "list_wallets", AsyncMock(return_value=[make_wallet("A")]))
    monkeypatch.setattr(handlers_mod, "get_balance", AsyncMock(return_value=Decimal("100")))

    async def fake_run_farm(*a, **k):
        await asyncio.sleep(60)

    monkeypatch.setattr(handlers_mod, "run_farm", fake_run_farm)

    done_task = asyncio.create_task(asyncio.sleep(0))
    await done_task  # complete it
    assert done_task.done()

    result = await handle_farm_create(ws, "LIC", make_msg("100"), done_task, FarmSession())

    # A done task does not short-circuit; a fresh task is created.
    assert isinstance(result, asyncio.Task)
    assert result is not done_task
    event = send_event.await_args.args[1]
    assert event.type == "farm_started"

    result.cancel()
    try:
        await result
    except asyncio.CancelledError:
        pass


# ── handle_farm_cancel ───────────────────────────────────────────────────────


async def test_handle_farm_cancel_cancels_live_task(monkeypatch):
    ws = make_ws()
    send_event = AsyncMock()
    monkeypatch.setattr(handlers_mod, "send_event", send_event)

    live_task = asyncio.create_task(asyncio.sleep(60))

    result = await handle_farm_cancel(ws, live_task)

    assert live_task.cancelled()
    # Fast teardown finished within the grace window -> nothing to keep tracking.
    assert result is None
    event = send_event.await_args.args[1]
    assert event.type == "farm_cancelled"


async def test_handle_farm_cancel_no_task(monkeypatch):
    ws = make_ws()
    send_event = AsyncMock()
    monkeypatch.setattr(handlers_mod, "send_event", send_event)

    result = await handle_farm_cancel(ws, None)

    # Still emits cancelled even with nothing to cancel.
    assert result is None
    event = send_event.await_args.args[1]
    assert event.type == "farm_cancelled"


async def test_handle_farm_cancel_done_task(monkeypatch):
    ws = make_ws()
    send_event = AsyncMock()
    monkeypatch.setattr(handlers_mod, "send_event", send_event)

    done_task = asyncio.create_task(asyncio.sleep(0))
    await done_task
    assert done_task.done()

    result = await handle_farm_cancel(ws, done_task)

    # Already-done task is not re-cancelled; still emits cancelled.
    assert result is None
    assert not done_task.cancelled()
    event = send_event.await_args.args[1]
    assert event.type == "farm_cancelled"


async def test_handle_farm_cancel_slow_teardown_acks_without_blocking(monkeypatch):
    # THE BUG: a teardown that outlives the grace window (e.g. cancel_all on a dead CLOB
    # connection) must NOT trap the UI on "Cancelling…". We ACK after the grace and let the
    # teardown finish detached.
    ws = make_ws()
    send_event = AsyncMock()
    monkeypatch.setattr(handlers_mod, "send_event", send_event)
    monkeypatch.setattr(handlers_mod, "FARM_CANCEL_GRACE_SECONDS", 0.02)

    teardown_started = asyncio.Event()
    teardown_done = asyncio.Event()

    async def slow_worker():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            teardown_started.set()
            await asyncio.sleep(0.3)  # >> grace: simulates a slow deadman cancel_all
            teardown_done.set()
            raise

    worker = asyncio.create_task(slow_worker())
    await asyncio.sleep(0)  # let it reach the outer await

    result = await handle_farm_cancel(ws, worker)

    # ACK sent even though the teardown is still in progress.
    assert send_event.await_args.args[1].type == "farm_cancelled"
    assert teardown_started.is_set()
    assert not teardown_done.is_set()  # we did NOT block on the full teardown
    # The still-running teardown is handed back so the caller keeps it tracked.
    assert result is worker
    assert not worker.done()

    # It finishes on its own in the background.
    await asyncio.wait_for(teardown_done.wait(), timeout=2)
    with contextlib.suppress(asyncio.CancelledError):
        await worker
    assert worker.cancelled()


async def test_handle_farm_cancel_teardown_error_still_acks(monkeypatch):
    # A teardown that raises a non-Cancelled error must not crash the handler or trap the UI.
    ws = make_ws()
    send_event = AsyncMock()
    monkeypatch.setattr(handlers_mod, "send_event", send_event)

    async def erroring_worker():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise ValueError("teardown boom") from None

    worker = asyncio.create_task(erroring_worker())
    await asyncio.sleep(0)

    result = await handle_farm_cancel(ws, worker)

    assert send_event.await_args.args[1].type == "farm_cancelled"
    assert result is None  # task is done (errored), nothing to keep tracking
    # retrieve the exception so it isn't flagged "never retrieved"
    with contextlib.suppress(ValueError):
        await worker


async def test_handle_farm_cancel_propagates_handler_cancellation(monkeypatch):
    # If the HANDLER itself is cancelled mid-wait (not the teardown), that cancellation must
    # propagate — and the shielded teardown must keep running, not be killed.
    ws = make_ws()
    monkeypatch.setattr(handlers_mod, "send_event", AsyncMock())
    monkeypatch.setattr(handlers_mod, "FARM_CANCEL_GRACE_SECONDS", 60)

    async def slow_worker():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.sleep(60)  # teardown that outlives the handler
            raise

    worker = asyncio.create_task(slow_worker())
    await asyncio.sleep(0)

    handler = asyncio.create_task(handle_farm_cancel(ws, worker))
    await asyncio.sleep(0.02)  # let it reach the wait_for
    handler.cancel()

    with pytest.raises(asyncio.CancelledError):
        await handler

    assert not worker.done()  # shielded teardown survived the handler's cancellation

    worker.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await worker


async def test_handle_farm_cancel_task_completes_normally(monkeypatch):
    # Worker finishes normally during the grace (e.g. a graceful kill-switch exit that races the
    # cancel): no exception from the await -> ACK, nothing to keep tracking.
    ws = make_ws()
    send_event = AsyncMock()
    monkeypatch.setattr(handlers_mod, "send_event", send_event)

    async def self_completing():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            return  # graceful exit instead of re-raising

    worker = asyncio.create_task(self_completing())
    await asyncio.sleep(0)

    result = await handle_farm_cancel(ws, worker)

    assert send_event.await_args.args[1].type == "farm_cancelled"
    assert result is None
    assert worker.done() and not worker.cancelled()


async def test_handle_farm_cancel_idempotent_while_detached(monkeypatch):
    # Clicking Stop twice while a slow teardown is still detached: both ACK, task stays tracked.
    ws = make_ws()
    send_event = AsyncMock()
    monkeypatch.setattr(handlers_mod, "send_event", send_event)
    monkeypatch.setattr(handlers_mod, "FARM_CANCEL_GRACE_SECONDS", 0.02)

    release = asyncio.Event()

    async def slow_worker():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await release.wait()  # teardown stays in progress until released
            raise

    worker = asyncio.create_task(slow_worker())
    await asyncio.sleep(0)

    assert await handle_farm_cancel(ws, worker) is worker
    assert await handle_farm_cancel(ws, worker) is worker
    assert send_event.await_count == 2  # every stop re-acks the UI

    release.set()
    with contextlib.suppress(asyncio.CancelledError):
        await worker


async def test_new_farm_blocked_until_detached_teardown_finishes(monkeypatch):
    # The restart race: a detached teardown is handed back as the live task, so a new farm_create
    # is rejected until it finishes — its cancel_all can't clobber the new farm's fresh orders.
    ws = make_ws()
    send_event = AsyncMock()
    monkeypatch.setattr(handlers_mod, "send_event", send_event)
    monkeypatch.setattr(handlers_mod, "FARM_CANCEL_GRACE_SECONDS", 0.02)

    release = asyncio.Event()

    async def slow_worker():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await release.wait()
            raise

    worker = asyncio.create_task(slow_worker())
    await asyncio.sleep(0)

    farm_task = await handle_farm_cancel(ws, worker)
    assert farm_task is worker and not farm_task.done()

    # A new farm while the teardown is still running is rejected (no wallet lookup either).
    result = await handle_farm_create(ws, "LIC", make_msg("100"), farm_task, FarmSession())
    assert result is farm_task
    assert send_event.await_args.args[1].reason == "farm_already_running"

    # Once the teardown finishes, the task is done -> a future create is unblocked.
    release.set()
    with contextlib.suppress(asyncio.CancelledError):
        await worker
    assert farm_task.done()


async def test_handle_farm_create_uses_first_wallet(monkeypatch):
    # Phase 1 farms exactly one wallet: the first in list_wallets sort order.
    ws = make_ws()
    monkeypatch.setattr(handlers_mod, "send_event", AsyncMock())
    first, second = make_wallet("wallet1"), make_wallet("wallet2")
    monkeypatch.setattr(handlers_mod, "list_wallets", AsyncMock(return_value=[first, second]))
    monkeypatch.setattr(handlers_mod, "get_balance", AsyncMock(return_value=Decimal("100")))

    run_farm_calls = []

    async def fake_run_farm(websocket, msg, wallet, license_key, farm_session):
        run_farm_calls.append(wallet)
        await asyncio.sleep(60)

    monkeypatch.setattr(handlers_mod, "run_farm", fake_run_farm)

    result = await handle_farm_create(ws, "LIC", make_msg("100"), None, FarmSession())
    await asyncio.sleep(0)

    assert run_farm_calls == [first]

    result.cancel()
    try:
        await result
    except asyncio.CancelledError:
        pass


async def test_handle_farm_create_starts_on_legacy_only_wallet(monkeypatch):
    # A pre-migration registry holding only "B" now starts a farm (was no_wallet_registered).
    ws = make_ws()
    monkeypatch.setattr(handlers_mod, "send_event", AsyncMock())
    legacy = make_wallet("B")
    monkeypatch.setattr(handlers_mod, "list_wallets", AsyncMock(return_value=[legacy]))
    monkeypatch.setattr(handlers_mod, "get_balance", AsyncMock(return_value=Decimal("100")))

    run_farm_calls = []

    async def fake_run_farm(websocket, msg, wallet, license_key, farm_session):
        run_farm_calls.append(wallet)
        await asyncio.sleep(60)

    monkeypatch.setattr(handlers_mod, "run_farm", fake_run_farm)

    result = await handle_farm_create(ws, "LIC", make_msg("100"), None, FarmSession())
    await asyncio.sleep(0)

    assert run_farm_calls == [legacy]

    result.cancel()
    try:
        await result
    except asyncio.CancelledError:
        pass
