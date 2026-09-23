from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from app.api.wallet import handlers as wallet_handlers


@pytest.fixture(autouse=True)
def _stub_wallet_balance(monkeypatch):
    """send_wallet_list now decorates each entry with a live RPC balance read — stub it
    so no API test ever touches the network."""
    monkeypatch.setattr(wallet_handlers, "get_balance", AsyncMock(return_value=Decimal("100")))
