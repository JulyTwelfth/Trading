"""Regression for the gamma-api 500s that aborted whole reconcile ticks. A transient
5xx on a /markets batch must be retried (and recover) within the tick; a persistent
failure must still propagate so the tick is skipped rather than acting on partial
discovery."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.constants import GAMMA_MARKETS_MAX_RETRIES
from app.farm import discovery as discovery_mod
from app.farm.discovery import fetch_gamma_markets


def ok_response(payload: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)
    return resp


def err_response(status: int) -> MagicMock:
    request = httpx.Request("GET", "https://gamma-api.polymarket.com/markets")
    response = httpx.Response(status, request=request)
    resp = MagicMock()
    resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(str(status), request=request, response=response)
    )
    return resp


def _500_response() -> MagicMock:
    return err_response(500)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    # Collapse backoff delays so the tests don't actually wait.
    monkeypatch.setattr(discovery_mod.asyncio, "sleep", AsyncMock())


async def test_gamma_batch_recovers_after_transient_500s():
    payload = [{"conditionId": "0xabc", "slug": "m"}]
    http = MagicMock()
    # Fail twice, succeed on the third attempt.
    http.get = AsyncMock(side_effect=[_500_response(), _500_response(), ok_response(payload)])

    result = await fetch_gamma_markets(http, ["0xabc"])

    assert result == {"0xabc": {"conditionId": "0xabc", "slug": "m"}}
    assert http.get.await_count == 3


async def test_gamma_batch_raises_after_exhausting_retries():
    http = MagicMock()
    http.get = AsyncMock(side_effect=[_500_response() for _ in range(GAMMA_MARKETS_MAX_RETRIES)])

    with pytest.raises(httpx.HTTPStatusError):
        await fetch_gamma_markets(http, ["0xabc"])

    assert http.get.await_count == GAMMA_MARKETS_MAX_RETRIES


async def test_gamma_batch_succeeds_first_try_no_retry():
    payload = [{"conditionId": "0xabc", "slug": "m"}]
    http = MagicMock()
    http.get = AsyncMock(return_value=ok_response(payload))

    result = await fetch_gamma_markets(http, ["0xabc"])

    assert result == {"0xabc": {"conditionId": "0xabc", "slug": "m"}}
    assert http.get.await_count == 1


async def test_gamma_batch_fails_fast_on_4xx_without_retrying():
    # A 400 (bad params) is permanent — retrying just burns the budget. Fail on attempt 1.
    http = MagicMock()
    http.get = AsyncMock(side_effect=[err_response(400), ok_response([{"conditionId": "x"}])])

    with pytest.raises(httpx.HTTPStatusError):
        await fetch_gamma_markets(http, ["0xabc"])

    assert http.get.await_count == 1, "4xx must not be retried"


async def test_gamma_batch_retries_on_429_rate_limit():
    # 429 is a 4xx but transient — back off and retry rather than fail fast.
    payload = [{"conditionId": "0xabc", "slug": "m"}]
    http = MagicMock()
    http.get = AsyncMock(side_effect=[err_response(429), ok_response(payload)])

    result = await fetch_gamma_markets(http, ["0xabc"])

    assert result == {"0xabc": {"conditionId": "0xabc", "slug": "m"}}
    assert http.get.await_count == 2


async def test_gamma_empty_condition_ids_skips_http():
    http = MagicMock()
    http.get = AsyncMock()

    result = await fetch_gamma_markets(http, [])

    assert result == {}
    assert http.get.await_count == 0
