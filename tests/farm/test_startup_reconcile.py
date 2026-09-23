"""Startup position reconciliation: on farm start, liquidate any outcome-token shares the
wallet already holds (residue from a prior crash/restart that stranded a fill) via the standard
exit path. Closes the 'fill outlived the process -> manual sell' gap."""

import logging
from decimal import Decimal
from unittest.mock import MagicMock

from app.farm import worker as worker_mod
from app.farm.discovery import OnchainPosition, fetch_open_positions
from app.farm.worker import reconcile_startup_positions


class FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeHttp:
    def __init__(self, payload):
        self._payload = payload
        self.calls: list = []

    async def get(self, url, params=None):
        self.calls.append((url, params))
        return FakeResp(self._payload)


# ── fetch_open_positions parsing ─────────────────────────────────────────────


async def test_fetch_open_positions_parses_and_filters_zero_size():
    http = FakeHttp(
        [
            {
                "asset": "tok-1",
                "conditionId": "c1",
                "outcome": "YES",
                "slug": "m-1",
                "size": "20",
                "avgPrice": "0.30",
            },
            # zero size -> dust/closed, skipped
            {"asset": "tok-2", "conditionId": "c2", "outcome": "NO", "size": "0"},
        ]
    )
    out = await fetch_open_positions(http, "0xwallet")

    assert len(out) == 1
    p = out[0]
    assert (p.token_id, p.condition_id, p.outcome, p.slug, p.size, p.avg_price) == (
        "tok-1",
        "c1",
        "YES",
        "m-1",
        Decimal("20"),
        Decimal("0.30"),
    )
    assert http.calls[0][1] == {"user": "0xwallet", "sizeThreshold": "0.1"}


async def test_fetch_open_positions_skips_malformed_entries():
    http = FakeHttp(
        [
            {"asset": "tok-1", "outcome": "YES", "size": "10", "avgPrice": "0.4"},  # ok
            {"size": "5"},  # missing asset -> KeyError -> skipped
            "junk",  # not a dict -> TypeError -> skipped
        ]
    )
    out = await fetch_open_positions(http, "0xw")

    assert len(out) == 1
    assert out[0].token_id == "tok-1"
    assert out[0].condition_id == ""  # missing conditionId tolerated


async def test_fetch_open_positions_returns_empty_on_error():
    class BoomHttp:
        async def get(self, url, params=None):
            raise RuntimeError("network down")

    assert await fetch_open_positions(BoomHttp(), "0xw") == []


async def test_fetch_open_positions_skips_non_numeric_numbers():
    # Decimal("abc") raises InvalidOperation (an ArithmeticError, NOT a ValueError); a malformed
    # size/avgPrice must be skipped, not crash the fetch.
    http = FakeHttp(
        [
            {"asset": "tok-1", "size": "abc", "outcome": "YES"},  # bad size
            {"asset": "tok-2", "size": "10", "avgPrice": "1.2.3", "outcome": "NO"},  # bad avgPrice
            {"asset": "tok-3", "size": "5", "avgPrice": "0.4", "outcome": "YES"},  # ok
        ]
    )
    out = await fetch_open_positions(http, "0xw")
    assert [p.token_id for p in out] == ["tok-3"]


# ── reconcile_startup_positions ──────────────────────────────────────────────

POSITIONS = [
    OnchainPosition(
        token_id="tok-1",
        condition_id="c1",
        outcome="YES",
        slug="m-1",
        size=Decimal("20"),
        avg_price=Decimal("0.3"),
    ),
    OnchainPosition(
        token_id="tok-2",
        condition_id="c2",
        outcome="NO",
        slug="m-2",
        size=Decimal("9"),
        avg_price=Decimal("0.6"),
    ),
]


async def test_reconcile_startup_liquidates_each_position(farm_state, monkeypatch, caplog):
    async def fake_fetch(http, addr):
        return POSITIONS

    exits: list = []

    async def fake_exit(
        client,
        state,
        token_id,
        size,
        cid,
        slug,
        outcome,
        *,
        cancel_resting=True,
        entry_cost=Decimal(0),
    ):
        exits.append((token_id, size, outcome, entry_cost))

    monkeypatch.setattr(worker_mod, "fetch_open_positions", fake_fetch)
    monkeypatch.setattr(worker_mod, "exit_position_leg", fake_exit)

    with caplog.at_level(logging.WARNING, logger="app.farm.worker"):
        await reconcile_startup_positions(MagicMock(), farm_state, MagicMock())

    assert [(e[0], e[1], e[2]) for e in exits] == [
        ("tok-1", Decimal("20"), "YES"),
        ("tok-2", Decimal("9"), "NO"),
    ]
    # entry_cost = size * avg_price → liquidation books the residue's true PnL, not a phantom gain
    assert exits[0][3] == Decimal("6.0")  # 20 * 0.3
    assert "2 untracked position(s)" in "\n".join(r.getMessage() for r in caplog.records)


async def test_reconcile_startup_clean_wallet_is_noop(farm_state, monkeypatch, caplog):
    async def fake_fetch(http, addr):
        return []

    called: list = []

    async def fake_exit(*a, **k):
        called.append(a)

    monkeypatch.setattr(worker_mod, "fetch_open_positions", fake_fetch)
    monkeypatch.setattr(worker_mod, "exit_position_leg", fake_exit)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_startup_positions(MagicMock(), farm_state, MagicMock())

    assert called == []
    assert "no untracked positions" in "\n".join(r.getMessage() for r in caplog.records)


async def test_reconcile_startup_one_failure_does_not_abort_others(farm_state, monkeypatch):
    async def fake_fetch(http, addr):
        return POSITIONS

    done: list = []

    async def fake_exit(
        client,
        state,
        token_id,
        size,
        cid,
        slug,
        outcome,
        *,
        cancel_resting=True,
        entry_cost=Decimal(0),
    ):
        if token_id == "tok-1":
            raise RuntimeError("sell failed")
        done.append(token_id)

    monkeypatch.setattr(worker_mod, "fetch_open_positions", fake_fetch)
    monkeypatch.setattr(worker_mod, "exit_position_leg", fake_exit)

    await reconcile_startup_positions(MagicMock(), farm_state, MagicMock())

    assert done == ["tok-2"], "a failed liquidation must not block the others"
