"""log_task_exit surfaces a background loop dying mid-session WITH its traceback
(exc_info), and stays silent on the normal cancel/clean-exit paths."""

import asyncio
import logging

from app.farm import worker as worker_mod

WORKER = "app.farm.worker"


def worker_records(caplog):
    return [r for r in caplog.records if r.name == WORKER]


async def test_logs_traceback_on_failure(caplog):
    async def boom():
        raise ValueError("kaboom")

    task = asyncio.create_task(boom())
    try:
        await task
    except ValueError:
        pass

    with caplog.at_level(logging.ERROR, logger=WORKER):
        worker_mod.log_task_exit(task)

    records = worker_records(caplog)
    assert len(records) == 1
    # exc_info is populated and the formatted line carries the real traceback,
    # not just repr(exc) — the whole point of the fix.
    assert records[0].exc_info is not None
    assert records[0].exc_info[0] is ValueError
    assert "ValueError: kaboom" in logging.Formatter().format(records[0])


async def test_silent_on_cancel(caplog):
    async def forever():
        await asyncio.sleep(3600)

    task = asyncio.create_task(forever())
    await asyncio.sleep(0)  # let it start before cancelling
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    with caplog.at_level(logging.ERROR, logger=WORKER):
        worker_mod.log_task_exit(task)

    assert worker_records(caplog) == []


async def test_silent_on_clean_exit(caplog):
    async def ok():
        return 42

    task = asyncio.create_task(ok())
    await task

    with caplog.at_level(logging.ERROR, logger=WORKER):
        worker_mod.log_task_exit(task)

    assert worker_records(caplog) == []
