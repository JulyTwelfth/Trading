"""Line coverage for app.main.

app.main runs its logging setup at import time and calls logging.basicConfig(
force=True), which removes existing root handlers (including caplog's). To exercise
the unknown-LOG_LEVEL warning branch without (a) corrupting global logging for other
tests and (b) having force=True strip caplog's handler before the warning fires, we
monkeypatch logging.basicConfig to a no-op for the duration of the reload. The
subsequent logging.warning(...) on root then still reaches caplog. A teardown
restores the pre-test LOG_LEVEL and reloads once more so module state is left sane.
"""

import importlib
import logging
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.responses import FileResponse

import app.main as main_mod


def test_app_is_fastapi_with_dashboard_and_ws_routes():
    assert isinstance(main_mod.app, FastAPI)
    paths = {getattr(r, "path", None) for r in main_mod.app.routes}
    assert "/ws" in paths, f"expected /ws route to be included, got {sorted(p for p in paths if p)}"
    assert "/" in paths
    assert "/static" in paths


@pytest.mark.asyncio
async def test_dashboard_returns_bundled_index():
    response = await main_mod.dashboard()

    assert isinstance(response, FileResponse)
    assert Path(response.path) == main_mod.WEB_DIR / "index.html"
    assert Path(response.path).is_file()


def test_dashboard_assets_are_bundled():
    assert (main_mod.WEB_DIR / "styles.css").is_file()
    assert (main_mod.WEB_DIR / "app.js").is_file()
    html = (main_mod.WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert 'id="farm-form"' in html
    assert 'id="start-button"' in html
    assert 'id="stop-button"' in html


@pytest.fixture
def reload_isolation():
    # Restore the pre-test LOG_LEVEL (or remove it if it wasn't set) after a
    # LOG_LEVEL-mutating reload, then reload once more so other tests (and caplog)
    # are unaffected.
    import os

    old = os.environ.get("LOG_LEVEL")
    yield
    if old is None:
        os.environ.pop("LOG_LEVEL", None)
    else:
        os.environ["LOG_LEVEL"] = old
    importlib.reload(main_mod)


def test_unknown_log_level_warns(monkeypatch, caplog, reload_isolation):
    # No-op basicConfig: avoids force=True stripping caplog's handler and avoids
    # mutating global logging during the reload.
    monkeypatch.setattr(logging, "basicConfig", lambda *a, **k: None)
    monkeypatch.setenv("LOG_LEVEL", "NOPE")

    with caplog.at_level(logging.WARNING):
        importlib.reload(main_mod)

    assert any(
        "unknown LOG_LEVEL" in r.getMessage() and "NOPE" in r.getMessage() for r in caplog.records
    ), "expected an 'unknown LOG_LEVEL' warning when LOG_LEVEL is invalid"


def test_valid_log_level_no_warning(monkeypatch, caplog, reload_isolation):
    monkeypatch.setattr(logging, "basicConfig", lambda *a, **k: None)
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    with caplog.at_level(logging.WARNING):
        importlib.reload(main_mod)

    assert not any("unknown LOG_LEVEL" in r.getMessage() for r in caplog.records)


def test_log_file_writes_to_given_path(monkeypatch, tmp_path):
    # With LOG_FILE set, the import attaches a FileHandler that creates and writes to
    # that exact path. basicConfig is no-op'd so force=True doesn't strip global
    # handlers during the reload.
    log_path = tmp_path / "session.log"
    monkeypatch.setattr(logging, "basicConfig", lambda *a, **k: None)
    monkeypatch.setenv("LOG_FILE", str(log_path))

    root = logging.getLogger()
    before = list(root.handlers)
    try:
        importlib.reload(main_mod)

        assert log_path.exists()
    finally:
        # Drop and close the FileHandler the reload attached to root so we don't
        # leak an open file or pollute other tests, then restore sane module state.
        for h in list(root.handlers):
            if h not in before:
                root.removeHandler(h)
                h.close()
        monkeypatch.undo()
        importlib.reload(main_mod)
