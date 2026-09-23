import argparse
import os
from datetime import datetime

import uvicorn


def server():
    parser = argparse.ArgumentParser(prog="server", description="Run the dev server.")
    parser.add_argument(
        "--log",
        action="store_true",
        help="Capture this session at DEBUG to logs/session-<timestamp>.log",
    )
    args = parser.parse_args()

    if args.log:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        os.environ["LOG_FILE"] = os.path.join("logs", f"session-{stamp}.log")
        os.environ["LOG_LEVEL"] = "DEBUG"

    uvicorn.run("app.main:app", reload=True)
