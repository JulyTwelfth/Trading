import asyncio
import logging
import os
from contextlib import asynccontextmanager
from logging.handlers import TimedRotatingFileHandler

from fastapi import FastAPI

from app.api.ws import router
from app.constants import LOG_RETENTION_DAYS
from app.db.license import clear_stale_sessions

levels = logging.getLevelNamesMapping()
level_name = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=levels.get(level_name, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

log_file = os.getenv("LOG_FILE")
if log_file:
    parent = os.path.dirname(log_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    file_handler = TimedRotatingFileHandler(
        log_file, when="midnight", utc=True, backupCount=LOG_RETENTION_DAYS, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)
logging.getLogger("strat").setLevel(logging.INFO)

if level_name not in levels:
    logging.warning("unknown LOG_LEVEL %r — defaulting to INFO", level_name)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        await asyncio.to_thread(clear_stale_sessions)
        logging.info("startup: cleared stale license sessions")
    except Exception:
        logging.exception("startup: could not clear stale license sessions")
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router)
