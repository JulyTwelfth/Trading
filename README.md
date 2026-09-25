# polymarket-bot — Server

## Setup

This project uses [uv](https://github.com/astral-sh/uv) as the package manager.

Install dependencies:
```
uv sync
```

Add a new package:
```
uv add <package-name>
```

Copy `.env.example` to `.env` and fill in the required values.

## Running the Server

```
uv run server
```

This is shorthand for `uv run uvicorn app.main:app --reload`

Open the local control dashboard after the server starts:

```
http://127.0.0.1:8000
```

The dashboard connects to `/ws` on the same server. It supports license login,
wallet/balance status, farm start and stop controls, positions, order events,
rewards/PnL summaries, and the manual market blacklist. Keep the page open while
the farm is running; closing its WebSocket stops the farm and starts order cleanup.

To also capture the session to a log file, add `--log`.
It writes `logs/session-<timestamp>.log`, one file per launch (reloads keep
appending to that same file):

```
uv run server --log
```

## Testing

Run the test suite with coverage (the `--cov` flags and the 90% gate are baked
into `pyproject.toml`'s `addopts`, so a plain `pytest` already enforces them):
```
uv run pytest
```

To see which lines are uncovered:
```
uv run pytest --cov=app --cov-report=term-missing
```

CI runs the same command and fails the build if overall line coverage drops
below 90%.

## Contributing

Run ruff before every commit:
```
uv run ruff format .
uv run ruff check .
```

## Project Structure

```
polymarket-liq/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── dependencies.py
│   ├── exceptions.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── ws.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── license.py
│   │   └── session.py
│   └── bot/
│       ├── __init__.py
│       ├── auth.py
│       ├── market.py
│       ├── balance.py
│       ├── trader.py
│       ├── monitor.py
│       └── loop.py
├── scripts/
│   └── admin.py
├── tests/
│   ├── __init__.py
│   ├── test_license.py
│   └── test_bot/
├── .env
├── .env.example
├── pyproject.toml
├── uv.lock
└── README.md
```
