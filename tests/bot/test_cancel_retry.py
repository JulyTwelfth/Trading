"""One-delayed-retry cancel wrappers (app/bot/cancel.py).

`is_transient_cancel_error` decides whether a cancel failure is worth one retry (network blips,
rate limits, 5xx/429, Cloudflare blocks) vs a permanent/semantic rejection. The `_with_retry`
wrappers wrap the raw cancels: transient → sleep + exactly one more attempt;
non-transient → no retry; NEVER raise, returning True on a confirmed cancel and False otherwise.

The retry delay (CANCEL_RETRY_DELAY_SECONDS) is zeroed via an autouse AsyncMock over
cancel_mod.asyncio.sleep so the tests stay fast and deterministic.
"""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from polymarket import errors as pm_errors
from py_clob_client_v2.exceptions import PolyApiException

from app.bot import cancel as cancel_mod
from app.bot.cancel import (
    cancel_all_with_retry,
    cancel_order_with_retry,
    is_transient_cancel_error,
)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(cancel_mod.asyncio, "sleep", AsyncMock())


def polyapi(status) -> PolyApiException:
    exc = PolyApiException(error_msg={"error": "boom"})
    exc.status_code = status
    return exc


# ── classifier ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "exc, expected",
    [
        (pm_errors.TransportError("net"), True),
        (pm_errors.TimeoutError("slow"), True),
        (pm_errors.RateLimitError("429"), True),
        (httpx.RemoteProtocolError("proto"), True),
        (httpx.ConnectTimeout("timeout"), True),
        (ConnectionError("reset"), True),
        (OSError("broken pipe"), True),
        (pm_errors.RequestRejectedError("blocked by Cloudflare with status 400", status=400), True),
        (pm_errors.RequestRejectedError("not enough balance / allowance", status=400), False),
        (polyapi(500), True),
        (polyapi(503), True),
        (polyapi(429), True),
        (polyapi(400), False),
        (polyapi(None), False),
        (ValueError("nope"), False),
    ],
    ids=[
        "pm_transport",
        "pm_timeout",
        "pm_ratelimit",
        "httpx_remote_protocol",
        "httpx_connect_timeout",
        "connection_error",
        "os_error",
        "cloudflare_reject",
        "semantic_reject",
        "polyapi_500",
        "polyapi_503",
        "polyapi_429",
        "polyapi_400",
        "polyapi_no_status",
        "value_error",
    ],
)
def test_is_transient_cancel_error_classification(exc, expected):
    assert is_transient_cancel_error(exc) is expected


# ── cancel_order_with_retry behavior ────────────────────────────────────────────


async def test_transport_error_retried_then_succeeds(monkeypatch):
    calls: list = []

    async def flaky(client, oid):
        calls.append(oid)
        if len(calls) == 1:
            raise pm_errors.TransportError("blip")

    monkeypatch.setattr(cancel_mod, "cancel_order", flaky)

    assert await cancel_order_with_retry(MagicMock(), "oid-1") is True
    assert calls == ["oid-1", "oid-1"], "a transient failure must be retried exactly once"


async def test_cloudflare_400_is_transient(monkeypatch):
    calls: list = []

    async def flaky(client, oid):
        calls.append(oid)
        if len(calls) == 1:
            raise pm_errors.RequestRejectedError(
                "Request was blocked by Cloudflare with status 400", status=400
            )

    monkeypatch.setattr(cancel_mod, "cancel_order", flaky)

    assert await cancel_order_with_retry(MagicMock(), "oid-1") is True
    assert len(calls) == 2, "a Cloudflare block is transient → retried"


async def test_semantic_rejection_not_retried(monkeypatch):
    calls: list = []

    async def reject(client, oid):
        calls.append(oid)
        raise pm_errors.RequestRejectedError("not enough balance / allowance", status=400)

    monkeypatch.setattr(cancel_mod, "cancel_order", reject)

    assert await cancel_order_with_retry(MagicMock(), "oid-1") is False
    assert len(calls) == 1, "a semantic rejection must NOT be retried"


async def test_transient_both_times_two_calls_false(monkeypatch):
    calls: list = []

    async def always_transient(client, oid):
        calls.append(oid)
        raise pm_errors.TransportError("still down")

    monkeypatch.setattr(cancel_mod, "cancel_order", always_transient)

    assert await cancel_order_with_retry(MagicMock(), "oid-1") is False
    assert len(calls) == 2, "one retry only — a persistent transient failure makes no third attempt"


async def test_non_transient_first_error_no_retry(monkeypatch):
    calls: list = []

    async def value_error(client, oid):
        calls.append(oid)
        raise ValueError("permanent")

    monkeypatch.setattr(cancel_mod, "cancel_order", value_error)

    assert await cancel_order_with_retry(MagicMock(), "oid-1") is False
    assert len(calls) == 1


# ── cancel_all_with_retry mirror ────────────────────────────────────────────────


async def test_cancel_all_with_retry_transient_then_succeeds(monkeypatch):
    calls: list = []

    async def flaky_all(client):
        calls.append(1)
        if len(calls) == 1:
            raise pm_errors.RateLimitError("429")

    monkeypatch.setattr(cancel_mod, "cancel_all", flaky_all)

    assert await cancel_all_with_retry(MagicMock()) is True
    assert len(calls) == 2


async def test_cancel_all_with_retry_non_transient_no_retry(monkeypatch):
    calls: list = []

    async def reject_all(client):
        calls.append(1)
        raise ValueError("permanent")

    monkeypatch.setattr(cancel_mod, "cancel_all", reject_all)

    assert await cancel_all_with_retry(MagicMock()) is False
    assert len(calls) == 1, "cancel_all mirrors: a non-transient failure is not retried"
