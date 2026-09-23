import logging

logger = logging.getLogger("strat")


def strat(event: str, **fields: object) -> None:
    if fields:
        payload = " ".join(f"{k}={v}" for k, v in fields.items())
        logger.info("%s %s", event, payload)
    else:
        logger.info("%s", event)
