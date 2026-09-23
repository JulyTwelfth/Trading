import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.farm.handlers import FarmSession
from app.db.blacklist import BlacklistedMarket
from app.db.wallets import Wallet
from app.farm import worker as worker_mod

CLOSED = "Cannot send a request, as the client has been closed."


def wallet() -> Wallet:
    return Wallet(
        license_key="lic",
        wallet_id="A",
        proxy_address="0x" + "ab" * 20,
        private_key="0x" + "11" * 32,
    )


def patch_session(monkeypatch, sent):
    mock_client = MagicMock()
    mock_client.wallet_type = "GNOSIS_SAFE"
    monkeypatch.setattr(worker_mod, "build_execution_client", lambda wallet: mock_client)
    monkeypatch.setattr(worker_mod, "ensure_approval", AsyncMock())
    monkeypatch.setattr(worker_mod, "TICK_INTERVAL_SECONDS", 0)

    async def idle(*args, **kwargs):
        await asyncio.Event().wait()

    for name in ("heartbeat_loop", "push_summary_loop", "user_ws_loop", "rewards_poll_loop"):
        monkeypatch.setattr(worker_mod, name, idle)
    monkeypatch.setattr(worker_mod, "respawn_market_ws", AsyncMock(return_value=None))
    cancel_all_mock = AsyncMock()
    monkeypatch.setattr(worker_mod, "cancel_all", cancel_all_mock)

    async def fake_cancel_all_with_retry(client):
        # Shutdown now goes through the retry wrapper — route it through the same recorder so
        # existing `cancel_all.await_count`/`assert_not_awaited` assertions still hold.
        await cancel_all_mock(client)
        return True

    monkeypatch.setattr(worker_mod, "cancel_all_with_retry", fake_cancel_all_with_retry)
    monkeypatch.setattr(worker_mod, "await_inventory_drained", AsyncMock())
    monkeypatch.setattr(worker_mod, "save_blacklist", MagicMock())
    monkeypatch.setattr(worker_mod, "load_blacklist", MagicMock(return_value=0))
    monkeypatch.setattr(worker_mod, "list_blacklist", AsyncMock(return_value=[]))

    async def fake_send_event(websocket, event):
        sent.append(event)

    monkeypatch.setattr(worker_mod, "send_event", fake_send_event)


async def test_run_farm_lifecycle_emits_shutdown_and_cancels(market, farm_state, monkeypatch):
    sent: list = []
    patch_session(monkeypatch, sent)
    monkeypatch.setattr(
        worker_mod,
        "list_blacklist",
        AsyncMock(
            return_value=[
                BlacklistedMarket(license_key="lic", condition_id="0xexcl-1"),
                BlacklistedMarket(license_key="lic", condition_id="0xexcl-2"),
            ]
        ),
    )

    pos = farm_state.positions["market-A"]
    session = FarmSession()
    seen: dict = {}

    async def fake_reconcile(http, client, state, websocket):
        seen["session_state_is_state"] = session.state is state
        seen["excluded"] = set(state.excluded_markets)
        state.positions["market-A"] = pos
        state.killed = True

    monkeypatch.setattr(worker_mod, "reconcile_tick", fake_reconcile)

    await asyncio.wait_for(
        worker_mod.run_farm(MagicMock(), farm_state.config, wallet(), "lic", session),
        timeout=2,
    )

    worker_mod.ensure_approval.assert_awaited_once()
    assert seen["excluded"] == {"0xexcl-1", "0xexcl-2"}
    assert seen["session_state_is_state"] is True
    assert session.state is None
    cancel_events = [e for e in sent if type(e).__name__ == "OrderCancelledEvent"]
    assert len(cancel_events) == 2
    assert all(e.reason == "shutdown" for e in cancel_events)
    assert worker_mod.cancel_all.await_count == 2
    assert worker_mod.save_blacklist.called


async def test_run_farm_persists_blacklist_on_manual_stop(market, farm_state, monkeypatch):
    sent: list = []
    patch_session(monkeypatch, sent)

    in_tick = asyncio.Event()

    async def block(http, client, state, websocket):
        in_tick.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(worker_mod, "reconcile_tick", block)
    task = asyncio.create_task(
        worker_mod.run_farm(MagicMock(), farm_state.config, wallet(), "lic", FarmSession())
    )
    await asyncio.wait_for(in_tick.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert worker_mod.save_blacklist.called, "blacklist must be saved on manual-stop teardown"


async def test_run_farm_survives_blacklist_load_failure(market, farm_state, monkeypatch):
    from app.exceptions import BlacklistPersistenceError

    sent: list = []
    patch_session(monkeypatch, sent)
    monkeypatch.setattr(
        worker_mod, "list_blacklist", AsyncMock(side_effect=BlacklistPersistenceError())
    )

    seen: dict = {}

    async def fake_reconcile(http, client, state, websocket):
        seen["excluded"] = set(state.excluded_markets)
        state.killed = True

    monkeypatch.setattr(worker_mod, "reconcile_tick", fake_reconcile)

    await asyncio.wait_for(
        worker_mod.run_farm(MagicMock(), farm_state.config, wallet(), "lic", FarmSession()),
        timeout=2,
    )

    assert seen["excluded"] == set()
    assert worker_mod.cancel_all.await_count == 2


async def test_run_farm_reconcile_error_is_retried_not_fatal(market, farm_state, monkeypatch):
    sent: list = []
    patch_session(monkeypatch, sent)

    calls = {"n": 0}

    async def flaky_reconcile(http, client, state, websocket):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient tick failure")
        state.killed = True

    monkeypatch.setattr(worker_mod, "reconcile_tick", flaky_reconcile)

    await asyncio.wait_for(
        worker_mod.run_farm(MagicMock(), farm_state.config, wallet(), "lic", FarmSession()),
        timeout=2,
    )

    assert calls["n"] == 2
    assert worker_mod.cancel_all.await_count == 2


async def test_run_farm_undeployed_deposit_wallet_skips_cleanly(market, farm_state, monkeypatch):
    from app.exceptions import WalletNotDeployedError

    sent: list = []
    patch_session(monkeypatch, sent)

    def raise_undeployed(w):
        raise WalletNotDeployedError("proxy not deployed on-chain")

    monkeypatch.setattr(worker_mod, "build_execution_client", raise_undeployed)

    reconcile_calls = {"n": 0}

    async def fake_reconcile(http, client, state, websocket):
        reconcile_calls["n"] += 1
        state.killed = True

    monkeypatch.setattr(worker_mod, "reconcile_tick", fake_reconcile)

    await asyncio.wait_for(
        worker_mod.run_farm(MagicMock(), farm_state.config, wallet(), "lic", FarmSession()),
        timeout=2,
    )

    assert reconcile_calls["n"] == 0
    worker_mod.ensure_approval.assert_not_awaited()
    worker_mod.cancel_all.assert_not_awaited()
    err_events = [e for e in sent if type(e).__name__ == "FarmErrorEvent"]
    assert len(err_events) == 1
    assert wallet().proxy_address in err_events[0].reason


async def test_run_farm_deposit_wallet_skips_ensure_approval(market, farm_state, monkeypatch):
    sent: list = []
    patch_session(monkeypatch, sent)

    deposit_client = MagicMock()
    deposit_client.wallet_type = "DEPOSIT_WALLET"
    monkeypatch.setattr(worker_mod, "build_execution_client", lambda w: deposit_client)

    async def fake_reconcile(http, client, state, websocket):
        state.killed = True

    monkeypatch.setattr(worker_mod, "reconcile_tick", fake_reconcile)

    await asyncio.wait_for(
        worker_mod.run_farm(MagicMock(), farm_state.config, wallet(), "lic", FarmSession()),
        timeout=2,
    )

    worker_mod.ensure_approval.assert_not_awaited()
    assert worker_mod.cancel_all.await_count == 2


async def test_exit_reconcile_loop_stops_on_closed_client(farm_state, monkeypatch):

    async def raise_closed(client, state, http):
        raise RuntimeError(CLOSED)

    monkeypatch.setattr(worker_mod, "reconcile_onchain_positions", raise_closed)
    await asyncio.wait_for(
        worker_mod.exit_reconcile_loop(MagicMock(), farm_state, MagicMock()), timeout=1
    )


async def test_exit_reconcile_loop_retries_other_runtime_errors(farm_state, monkeypatch):
    monkeypatch.setattr(worker_mod, "EXIT_RECONCILE_SECONDS", 0)
    farm_state.positions.clear()
    calls = {"n": 0}

    async def flaky(client, state, http):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("some other transient failure")
        state.killed = True

    monkeypatch.setattr(worker_mod, "reconcile_onchain_positions", flaky)
    await asyncio.wait_for(
        worker_mod.exit_reconcile_loop(MagicMock(), farm_state, MagicMock()), timeout=1
    )
    assert calls["n"] == 2


async def test_exit_reconcile_loop_still_cancellable(farm_state, monkeypatch):

    async def block(client, state, http):
        await asyncio.Event().wait()

    monkeypatch.setattr(worker_mod, "reconcile_onchain_positions", block)
    task = asyncio.create_task(worker_mod.exit_reconcile_loop(MagicMock(), farm_state, MagicMock()))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_exit_reconcile_loop_stops_when_state_killed(farm_state, monkeypatch):
    monkeypatch.setattr(worker_mod, "EXIT_RECONCILE_SECONDS", 0)
    farm_state.positions.clear()

    async def noop(client, state, http):
        return None

    monkeypatch.setattr(worker_mod, "reconcile_onchain_positions", noop)
    farm_state.killed = False
    task = asyncio.create_task(worker_mod.exit_reconcile_loop(MagicMock(), farm_state, MagicMock()))
    await asyncio.sleep(0.01)
    assert not task.done()
    farm_state.killed = True
    await asyncio.wait_for(task, timeout=1)
