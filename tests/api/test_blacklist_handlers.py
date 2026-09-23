from unittest.mock import AsyncMock, MagicMock

import httpx

from app.api.blacklist import handlers as handlers_mod
from app.api.blacklist.handlers import (
    handle_blacklist_add,
    handle_blacklist_clear,
    handle_blacklist_list,
    handle_blacklist_remove,
    hydrate_blacklist_after_auth,
    send_blacklist_error,
    send_blacklist_list,
)
from app.api.blacklist.messages import BlacklistAddMessage, BlacklistRemoveMessage
from app.api.farm.handlers import FarmSession
from app.db.blacklist import BlacklistedMarket
from app.exceptions import BlacklistPersistenceError

MARKET_URL = "https://polymarket.com/event/world-cup-final/"


def make_ws() -> MagicMock:
    ws = MagicMock()
    ws.send_json = AsyncMock()
    return ws


def make_row(condition_id="0xc1", slug="m", question="?") -> BlacklistedMarket:
    return BlacklistedMarket(
        license_key="LIC",
        condition_id=condition_id,
        slug=slug,
        question=question,
        market_url=MARKET_URL,
    )


def resolved(*condition_ids) -> list[dict]:
    return [
        {"condition_id": cid, "slug": f"slug-{cid}", "question": f"q-{cid}"}
        for cid in condition_ids
    ]


def session_with_state(farm_state) -> FarmSession:
    s = FarmSession()
    s.state = farm_state
    return s


def session_no_state() -> FarmSession:
    return FarmSession()  # .state is None


# ── send_blacklist_list ──────────────────────────────────────────────────────


async def test_send_blacklist_list_serializes_entries(monkeypatch):
    ws = make_ws()
    monkeypatch.setattr(
        handlers_mod,
        "list_blacklist",
        AsyncMock(return_value=[make_row("0xA"), make_row("0xB")]),
    )

    await send_blacklist_list(ws, "LIC")

    ws.send_json.assert_awaited_once()
    payload = ws.send_json.await_args.args[0]
    assert payload["type"] == "blacklist_list"
    assert {e["condition_id"] for e in payload["markets"]} == {"0xA", "0xB"}


# ── send_blacklist_error ─────────────────────────────────────────────────────


async def test_send_blacklist_error_emits_blacklist_error():
    ws = make_ws()
    await send_blacklist_error(ws, "some reason")
    ws.send_json.assert_awaited_once()
    assert ws.send_json.await_args.args[0] == {
        "type": "blacklist_error",
        "reason": "some reason",
    }


# ── hydrate_blacklist_after_auth ─────────────────────────────────────────────


async def test_hydrate_after_auth_success(monkeypatch):
    ws = make_ws()
    monkeypatch.setattr(handlers_mod, "list_blacklist", AsyncMock(return_value=[make_row("0xA")]))

    await hydrate_blacklist_after_auth(ws, "LIC")

    assert ws.send_json.await_args.args[0]["type"] == "blacklist_list"


async def test_hydrate_after_auth_persistence_error_sends_error(monkeypatch):
    ws = make_ws()
    monkeypatch.setattr(
        handlers_mod, "list_blacklist", AsyncMock(side_effect=BlacklistPersistenceError())
    )

    await hydrate_blacklist_after_auth(ws, "LIC")

    payload = ws.send_json.await_args.args[0]
    assert payload["type"] == "blacklist_error"
    assert payload["reason"] == BlacklistPersistenceError.reason


# ── handle_blacklist_add ─────────────────────────────────────────────────────


async def test_handle_add_persists_mutates_live_state_and_lists(monkeypatch, farm_state):
    ws = make_ws()
    add = AsyncMock()
    monkeypatch.setattr(
        handlers_mod, "resolve_event_markets", AsyncMock(return_value=resolved("0xc1", "0xc2"))
    )
    monkeypatch.setattr(handlers_mod, "add_blacklist", add)
    monkeypatch.setattr(handlers_mod, "list_blacklist", AsyncMock(return_value=[make_row("0xc1")]))

    session = session_with_state(farm_state)
    msg = BlacklistAddMessage(type="blacklist_add", market_url=MARKET_URL)
    await handle_blacklist_add(ws, "LIC", msg, session)

    # ONE upsert carrying a row per resolved market, with the licence + url attached.
    add.assert_awaited_once()
    call_license, call_rows = add.await_args.args
    assert call_license == "LIC"
    assert [r["condition_id"] for r in call_rows] == ["0xc1", "0xc2"]
    assert {r["market_url"] for r in call_rows} == {MARKET_URL}
    # ALL resolved condition_ids are now live-excluded.
    assert {"0xc1", "0xc2"} <= farm_state.excluded_markets
    # Followed by a fresh list to the client.
    assert ws.send_json.await_args.args[0]["type"] == "blacklist_list"


async def test_handle_add_with_no_running_farm_persists_without_crash(monkeypatch):
    ws = make_ws()
    add = AsyncMock()
    monkeypatch.setattr(
        handlers_mod, "resolve_event_markets", AsyncMock(return_value=resolved("0xc1"))
    )
    monkeypatch.setattr(handlers_mod, "add_blacklist", add)
    monkeypatch.setattr(handlers_mod, "list_blacklist", AsyncMock(return_value=[make_row("0xc1")]))

    session = session_no_state()
    msg = BlacklistAddMessage(type="blacklist_add", market_url=MARKET_URL)
    await handle_blacklist_add(ws, "LIC", msg, session)  # must not raise on state=None

    add.assert_awaited_once()
    assert session.state is None
    assert ws.send_json.await_args.args[0]["type"] == "blacklist_list"


async def test_handle_add_bad_url_sends_error_no_write_no_mutation(monkeypatch, farm_state):
    ws = make_ws()
    add = AsyncMock()
    monkeypatch.setattr(
        handlers_mod,
        "resolve_event_markets",
        AsyncMock(side_effect=ValueError("No event found for slug: bad")),
    )
    monkeypatch.setattr(handlers_mod, "add_blacklist", add)

    session = session_with_state(farm_state)
    before = set(farm_state.excluded_markets)
    msg = BlacklistAddMessage(type="blacklist_add", market_url="https://polymarket.com/event/bad")
    await handle_blacklist_add(ws, "LIC", msg, session)

    payload = ws.send_json.await_args.args[0]
    assert payload["type"] == "blacklist_error"
    assert payload["reason"] == "No event found for slug: bad"
    add.assert_not_awaited()  # no DB write on a resolve failure
    assert farm_state.excluded_markets == before  # live state untouched


async def test_handle_add_resolve_http_error_sends_generic_error(monkeypatch, farm_state):
    ws = make_ws()
    add = AsyncMock()
    request = httpx.Request("GET", "https://gamma/events")
    monkeypatch.setattr(
        handlers_mod,
        "resolve_event_markets",
        AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "boom", request=request, response=httpx.Response(500, request=request)
            )
        ),
    )
    monkeypatch.setattr(handlers_mod, "add_blacklist", add)

    session = session_with_state(farm_state)
    msg = BlacklistAddMessage(type="blacklist_add", market_url=MARKET_URL)
    await handle_blacklist_add(ws, "LIC", msg, session)

    payload = ws.send_json.await_args.args[0]
    assert payload["type"] == "blacklist_error"
    assert payload["reason"] == "Could not resolve market URL"
    add.assert_not_awaited()


async def test_handle_add_resolve_transport_error_sends_generic_error(monkeypatch, farm_state):
    ws = make_ws()
    add = AsyncMock()
    monkeypatch.setattr(
        handlers_mod,
        "resolve_event_markets",
        AsyncMock(side_effect=httpx.ConnectTimeout("boom")),
    )
    monkeypatch.setattr(handlers_mod, "add_blacklist", add)

    session = session_with_state(farm_state)
    msg = BlacklistAddMessage(type="blacklist_add", market_url=MARKET_URL)
    await handle_blacklist_add(ws, "LIC", msg, session)

    payload = ws.send_json.await_args.args[0]
    assert payload["type"] == "blacklist_error"
    assert payload["reason"] == "Could not resolve market URL"
    add.assert_not_awaited()  # resolve failed before any DB write


async def test_handle_add_persistence_error_does_not_mutate_live_state(monkeypatch, farm_state):
    # DB-before-mutation ordering: if the upsert fails we must NOT have already added to the
    # live excluded set (otherwise live state and DB diverge).
    ws = make_ws()
    monkeypatch.setattr(
        handlers_mod, "resolve_event_markets", AsyncMock(return_value=resolved("0xc1", "0xc2"))
    )
    monkeypatch.setattr(
        handlers_mod, "add_blacklist", AsyncMock(side_effect=BlacklistPersistenceError())
    )
    list_mock = AsyncMock()
    monkeypatch.setattr(handlers_mod, "list_blacklist", list_mock)

    session = session_with_state(farm_state)
    before = set(farm_state.excluded_markets)
    msg = BlacklistAddMessage(type="blacklist_add", market_url=MARKET_URL)
    await handle_blacklist_add(ws, "LIC", msg, session)

    payload = ws.send_json.await_args.args[0]
    assert payload["type"] == "blacklist_error"
    assert farm_state.excluded_markets == before  # mutation never happened
    list_mock.assert_not_awaited()  # we bail before the success list


async def test_handle_add_refresh_error_after_write_sends_error_not_raises(monkeypatch, farm_state):
    # add + live mutation succeed, but the trailing list refresh read fails. We must
    # surface a blacklist_error (not let it escape and tear down the ws connection).
    ws = make_ws()
    monkeypatch.setattr(
        handlers_mod, "resolve_event_markets", AsyncMock(return_value=resolved("0xc1", "0xc2"))
    )
    monkeypatch.setattr(handlers_mod, "add_blacklist", AsyncMock())
    monkeypatch.setattr(
        handlers_mod, "list_blacklist", AsyncMock(side_effect=BlacklistPersistenceError())
    )

    session = session_with_state(farm_state)
    msg = BlacklistAddMessage(type="blacklist_add", market_url=MARKET_URL)
    await handle_blacklist_add(ws, "LIC", msg, session)  # must not raise

    assert ws.send_json.await_args.args[0]["type"] == "blacklist_error"
    # The write committed, so the live exclusion stays in place.
    assert {"0xc1", "0xc2"} <= farm_state.excluded_markets


# ── handle_blacklist_remove ──────────────────────────────────────────────────


async def test_handle_remove_deletes_discards_from_live_set_and_lists(monkeypatch, farm_state):
    ws = make_ws()
    delete = AsyncMock()
    monkeypatch.setattr(handlers_mod, "remove_blacklist", delete)
    monkeypatch.setattr(handlers_mod, "list_blacklist", AsyncMock(return_value=[]))

    farm_state.excluded_markets.add("0xc1")
    session = session_with_state(farm_state)
    msg = BlacklistRemoveMessage(type="blacklist_remove", condition_id="0xc1")
    await handle_blacklist_remove(ws, "LIC", msg, session)

    delete.assert_awaited_once_with("LIC", "0xc1")
    assert "0xc1" not in farm_state.excluded_markets
    assert ws.send_json.await_args.args[0]["type"] == "blacklist_list"


async def test_handle_remove_with_no_running_farm(monkeypatch):
    ws = make_ws()
    delete = AsyncMock()
    monkeypatch.setattr(handlers_mod, "remove_blacklist", delete)
    monkeypatch.setattr(handlers_mod, "list_blacklist", AsyncMock(return_value=[]))

    session = session_no_state()
    msg = BlacklistRemoveMessage(type="blacklist_remove", condition_id="0xc1")
    await handle_blacklist_remove(ws, "LIC", msg, session)  # state=None must not crash

    delete.assert_awaited_once_with("LIC", "0xc1")
    assert ws.send_json.await_args.args[0]["type"] == "blacklist_list"


async def test_handle_remove_absent_cid_is_safe(monkeypatch, farm_state):
    # discard() of a cid not in the set is a no-op (would raise with .remove()).
    ws = make_ws()
    monkeypatch.setattr(handlers_mod, "remove_blacklist", AsyncMock())
    monkeypatch.setattr(handlers_mod, "list_blacklist", AsyncMock(return_value=[]))

    session = session_with_state(farm_state)
    assert "0xnever" not in farm_state.excluded_markets
    msg = BlacklistRemoveMessage(type="blacklist_remove", condition_id="0xnever")
    await handle_blacklist_remove(ws, "LIC", msg, session)  # must not raise

    assert ws.send_json.await_args.args[0]["type"] == "blacklist_list"


async def test_handle_remove_persistence_error_sends_error_no_mutation(monkeypatch, farm_state):
    ws = make_ws()
    monkeypatch.setattr(
        handlers_mod, "remove_blacklist", AsyncMock(side_effect=BlacklistPersistenceError())
    )
    list_mock = AsyncMock()
    monkeypatch.setattr(handlers_mod, "list_blacklist", list_mock)

    farm_state.excluded_markets.add("0xc1")
    session = session_with_state(farm_state)
    msg = BlacklistRemoveMessage(type="blacklist_remove", condition_id="0xc1")
    await handle_blacklist_remove(ws, "LIC", msg, session)

    assert ws.send_json.await_args.args[0]["type"] == "blacklist_error"
    # On a failed delete we leave the live set alone and never list.
    assert "0xc1" in farm_state.excluded_markets
    list_mock.assert_not_awaited()


async def test_handle_remove_refresh_error_after_write_sends_error_not_raises(
    monkeypatch, farm_state
):
    # delete + live discard succeed, but the trailing list refresh read fails.
    ws = make_ws()
    monkeypatch.setattr(handlers_mod, "remove_blacklist", AsyncMock())
    monkeypatch.setattr(
        handlers_mod, "list_blacklist", AsyncMock(side_effect=BlacklistPersistenceError())
    )

    farm_state.excluded_markets.add("0xc1")
    session = session_with_state(farm_state)
    msg = BlacklistRemoveMessage(type="blacklist_remove", condition_id="0xc1")
    await handle_blacklist_remove(ws, "LIC", msg, session)  # must not raise

    assert ws.send_json.await_args.args[0]["type"] == "blacklist_error"
    assert "0xc1" not in farm_state.excluded_markets  # delete already committed


# ── handle_blacklist_list ────────────────────────────────────────────────────


async def test_handle_list_success(monkeypatch):
    ws = make_ws()
    monkeypatch.setattr(handlers_mod, "list_blacklist", AsyncMock(return_value=[make_row("0xA")]))

    await handle_blacklist_list(ws, "LIC")

    assert ws.send_json.await_args.args[0]["type"] == "blacklist_list"


async def test_handle_list_persistence_error_sends_error(monkeypatch):
    ws = make_ws()
    monkeypatch.setattr(
        handlers_mod, "list_blacklist", AsyncMock(side_effect=BlacklistPersistenceError())
    )

    await handle_blacklist_list(ws, "LIC")

    assert ws.send_json.await_args.args[0]["type"] == "blacklist_error"


# ── handle_blacklist_clear ───────────────────────────────────────────────────


async def test_handle_clear_drains_all_and_lists_empty(monkeypatch, farm_state):
    # The single list_blacklist mock serves BOTH calls (pre-fetch + send list).
    ws = make_ws()
    clear = AsyncMock()
    monkeypatch.setattr(
        handlers_mod,
        "list_blacklist",
        AsyncMock(return_value=[make_row("0xc1"), make_row("0xc2")]),
    )
    monkeypatch.setattr(handlers_mod, "clear_blacklist", clear)

    farm_state.excluded_markets.update({"0xc1", "0xc2", "0xkeep"})
    session = session_with_state(farm_state)
    await handle_blacklist_clear(ws, "LIC", session)

    clear.assert_awaited_once_with("LIC")
    # Only the blacklisted cids are drained; an unrelated live exclusion stays.
    assert "0xc1" not in farm_state.excluded_markets
    assert "0xc2" not in farm_state.excluded_markets
    assert "0xkeep" in farm_state.excluded_markets
    assert ws.send_json.await_args.args[0]["type"] == "blacklist_list"


async def test_handle_clear_empty_is_noop_and_lists(monkeypatch, farm_state):
    ws = make_ws()
    clear = AsyncMock()
    monkeypatch.setattr(handlers_mod, "list_blacklist", AsyncMock(return_value=[]))
    monkeypatch.setattr(handlers_mod, "clear_blacklist", clear)

    farm_state.excluded_markets.add("0xkeep")
    before = set(farm_state.excluded_markets)
    session = session_with_state(farm_state)
    await handle_blacklist_clear(ws, "LIC", session)

    clear.assert_awaited_once_with("LIC")
    assert farm_state.excluded_markets == before  # nothing to drain
    assert ws.send_json.await_args.args[0]["type"] == "blacklist_list"


async def test_handle_clear_no_running_farm(monkeypatch):
    ws = make_ws()
    clear = AsyncMock()
    monkeypatch.setattr(handlers_mod, "list_blacklist", AsyncMock(return_value=[make_row("0xc1")]))
    monkeypatch.setattr(handlers_mod, "clear_blacklist", clear)

    session = session_no_state()
    await handle_blacklist_clear(ws, "LIC", session)  # state=None must not crash

    clear.assert_awaited_once_with("LIC")
    assert ws.send_json.await_args.args[0]["type"] == "blacklist_list"


async def test_handle_clear_prefetch_error_no_clear(monkeypatch, farm_state):
    # If we can't read the cids first we must NOT clear (we'd lose the cids needed
    # to drain the live set) — bail with an error before touching the DB or state.
    ws = make_ws()
    clear = AsyncMock()
    monkeypatch.setattr(
        handlers_mod, "list_blacklist", AsyncMock(side_effect=BlacklistPersistenceError())
    )
    monkeypatch.setattr(handlers_mod, "clear_blacklist", clear)

    farm_state.excluded_markets.add("0xc1")
    session = session_with_state(farm_state)
    await handle_blacklist_clear(ws, "LIC", session)

    assert ws.send_json.await_args.args[0]["type"] == "blacklist_error"
    clear.assert_not_awaited()  # no delete attempted
    assert "0xc1" in farm_state.excluded_markets  # live state untouched


async def test_handle_clear_delete_error_no_mutation(monkeypatch, farm_state):
    # Pre-fetch succeeds but the delete fails: leave the live set alone and never
    # send the success list — the error is the last thing the client sees.
    ws = make_ws()
    monkeypatch.setattr(handlers_mod, "list_blacklist", AsyncMock(return_value=[make_row("0xc1")]))
    monkeypatch.setattr(
        handlers_mod, "clear_blacklist", AsyncMock(side_effect=BlacklistPersistenceError())
    )

    farm_state.excluded_markets.add("0xc1")
    session = session_with_state(farm_state)
    await handle_blacklist_clear(ws, "LIC", session)

    assert ws.send_json.await_args.args[0]["type"] == "blacklist_error"
    assert "0xc1" in farm_state.excluded_markets  # no mutation on a failed delete
    # The error is the final send — no blacklist_list followed it.
    assert ws.send_json.await_count == 1


async def test_handle_clear_refresh_error_after_write_sends_error_not_raises(
    monkeypatch, farm_state
):
    # clear() drains the live set, but the trailing list refresh read fails. The
    # pre-fetch succeeds (1st call) and the post-clear refresh raises (2nd call).
    ws = make_ws()
    clear = AsyncMock()
    monkeypatch.setattr(
        handlers_mod,
        "list_blacklist",
        AsyncMock(side_effect=[[make_row("0xc1")], BlacklistPersistenceError()]),
    )
    monkeypatch.setattr(handlers_mod, "clear_blacklist", clear)

    farm_state.excluded_markets.add("0xc1")
    session = session_with_state(farm_state)
    await handle_blacklist_clear(ws, "LIC", session)  # must not raise

    clear.assert_awaited_once_with("LIC")
    assert ws.send_json.await_args.args[0]["type"] == "blacklist_error"
    assert "0xc1" not in farm_state.excluded_markets  # drain happened before the refresh
