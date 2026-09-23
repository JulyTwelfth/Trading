"""strat() emits one parseable `event key=value` line on the `strat` logger —
the format the strategy-analytics pipeline greps. Keep it stable."""

import logging
from decimal import Decimal

from app.infra.strategy_log import strat


def test_strat_emits_parseable_keyvalue_line(caplog):
    with caplog.at_level(logging.INFO, logger="strat"):
        strat("fill", slug="trump-putin", outcome="YES", px="0.29", size=20, hold_s=3.2)
    msgs = [r.getMessage() for r in caplog.records if r.name == "strat"]
    assert msgs == ["fill slug=trump-putin outcome=YES px=0.29 size=20 hold_s=3.2"]


def test_strat_event_only_no_fields(caplog):
    with caplog.at_level(logging.INFO, logger="strat"):
        strat("started")
    msgs = [r.getMessage() for r in caplog.records if r.name == "strat"]
    assert msgs == ["started"]


def test_strat_handles_decimal_none_and_negative_values(caplog):
    # The real call sites pass Decimals (prices/PnL), None (missing mid), and
    # negatives (a losing round-trip) — none should raise or mangle the line.
    with caplog.at_level(logging.INFO, logger="strat"):
        strat("roundtrip", slug="m", net=Decimal("-1.89"), mid=None, size=Decimal("20"))
    msgs = [r.getMessage() for r in caplog.records if r.name == "strat"]
    assert msgs == ["roundtrip slug=m net=-1.89 mid=None size=20"]
