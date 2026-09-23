import os
import re
import sys

import pytest

import app.cli as cli


@pytest.fixture
def clean_log_env():
    # server() mutates os.environ (LOG_FILE/LOG_LEVEL) directly, which monkeypatch
    # can't track; snapshot and restore so --log tests don't leak into other tests.
    saved = {k: os.environ.pop(k, None) for k in ("LOG_FILE", "LOG_LEVEL")}
    yield
    for k, v in saved.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v


def patch_uvicorn(monkeypatch):
    captured = {}

    def fake_run(app_path, **kwargs):
        captured["app_path"] = app_path
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    return captured


def test_server_runs_app_with_reload_and_no_logging(monkeypatch, clean_log_env):
    captured = patch_uvicorn(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["server"])

    cli.server()

    assert captured["app_path"] == "app.main:app"
    assert captured["kwargs"].get("reload") is True
    # Without --log, the launcher touches neither the file path nor the level.
    assert "LOG_FILE" not in os.environ
    assert "LOG_LEVEL" not in os.environ


def test_server_log_flag_sets_file_and_debug_level(monkeypatch, clean_log_env):
    patch_uvicorn(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["server", "--log"])

    cli.server()

    assert re.fullmatch(r"logs[/\\]session-\d{8}-\d{6}\.log", os.environ["LOG_FILE"])
    assert os.environ["LOG_LEVEL"] == "DEBUG"


def test_server_log_flag_overrides_preexisting_env(monkeypatch, clean_log_env):
    # --log forces the session file + DEBUG even when LOG_FILE/LOG_LEVEL are already
    # set, so the flag's behavior is predictable rather than silently left as-is.
    patch_uvicorn(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["server", "--log"])
    os.environ["LOG_LEVEL"] = "WARNING"
    os.environ["LOG_FILE"] = "preexisting.log"

    cli.server()

    assert os.environ["LOG_LEVEL"] == "DEBUG"
    assert re.fullmatch(r"logs[/\\]session-\d{8}-\d{6}\.log", os.environ["LOG_FILE"])
