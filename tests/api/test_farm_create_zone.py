from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.api.farm.messages import FarmCreateMessage
from app.api.messages import client_message_adapter
from app.farm.schemas import FarmConfig


def _payload(filters_extra: dict) -> dict:
    filters = {
        "vol_min": "0",
        "vol_max": "100000",
        "liq_min": "0",
        "liq_max": "100000",
        "spread_min": "0",
        "spread_max": "100",
        "reward_min": "0",
        "time_remaining": "all",
        "created_date": "all",
        "change_24h": "all",
    }
    filters.update(filters_extra)
    return {
        "type": "farm_create",
        "filters": filters,
        "bankroll": "500",
        "max_session_loss": "50",
    }


def payload_with(extra: dict) -> dict:
    payload = _payload({})
    payload.update(extra)
    return payload


def test_zone_liq_max_number_parsed_as_decimal():
    msg = client_message_adapter.validate_python(_payload({"zone_liq_max": "1500"}))
    assert isinstance(msg, FarmCreateMessage)
    assert msg.filters.zone_liq_max == Decimal("1500")


def test_zone_liq_max_null_parsed_as_none():
    msg = client_message_adapter.validate_python(_payload({"zone_liq_max": None}))
    assert msg.filters.zone_liq_max is None


def test_zone_liq_max_omitted_defaults_to_none():
    msg = client_message_adapter.validate_python(_payload({}))
    assert msg.filters.zone_liq_max is None


def test_negative_zone_liq_max_rejected():
    with pytest.raises(ValidationError):
        client_message_adapter.validate_python(_payload({"zone_liq_max": "-1"}))


def test_zero_zone_liq_max_is_accepted_and_active():
    msg = client_message_adapter.validate_python(_payload({"zone_liq_max": "0"}))
    assert msg.filters.zone_liq_max == Decimal("0")


def test_fractional_and_large_values_parse():
    frac = client_message_adapter.validate_python(_payload({"zone_liq_max": "1500.50"}))
    assert frac.filters.zone_liq_max == Decimal("1500.50")
    big = client_message_adapter.validate_python(_payload({"zone_liq_max": "100000000"}))
    assert big.filters.zone_liq_max == Decimal("100000000")


def test_price_band_parsed():
    msg = client_message_adapter.validate_python(
        _payload({"price_min": "0.10", "price_max": "0.90"})
    )
    assert msg.filters.price_min == Decimal("0.10")
    assert msg.filters.price_max == Decimal("0.90")


def test_price_band_null_and_omitted_default_to_none():
    null_msg = client_message_adapter.validate_python(
        _payload({"price_min": None, "price_max": None})
    )
    assert null_msg.filters.price_min is None
    assert null_msg.filters.price_max is None
    omitted = client_message_adapter.validate_python(_payload({}))
    assert omitted.filters.price_min is None
    assert omitted.filters.price_max is None


def test_one_sided_price_band_allowed():
    msg = client_message_adapter.validate_python(_payload({"price_min": "0.10"}))
    assert msg.filters.price_min == Decimal("0.10")
    assert msg.filters.price_max is None


def test_price_min_above_max_rejected():
    with pytest.raises(ValidationError):
        client_message_adapter.validate_python(_payload({"price_min": "0.90", "price_max": "0.10"}))


def test_price_above_one_rejected():
    with pytest.raises(ValidationError):
        client_message_adapter.validate_python(_payload({"price_min": "10"}))
    with pytest.raises(ValidationError):
        client_message_adapter.validate_python(_payload({"price_max": "90"}))


def test_quote_depth_parsed():
    msg = client_message_adapter.validate_python(payload_with({"quote_depth": "aggressive"}))
    assert msg.quote_depth == "aggressive"


def test_quote_depth_defaults_to_safe_when_omitted():
    msg = client_message_adapter.validate_python(_payload({}))
    assert msg.quote_depth == "safe"


def test_invalid_quote_depth_rejected():
    with pytest.raises(ValidationError):
        client_message_adapter.validate_python(payload_with({"quote_depth": "yolo"}))


def test_range_24h_now_wired_not_dropped():
    msg = client_message_adapter.validate_python(_payload({"range_24h": "lt5"}))
    assert msg.filters.range_24h == "lt5"
    omitted = client_message_adapter.validate_python(_payload({}))
    assert omitted.filters.range_24h == "all"


def test_invalid_range_24h_rejected():
    with pytest.raises(ValidationError):
        client_message_adapter.validate_python(_payload({"range_24h": "lt99"}))


def test_size_tier_liq_min_parsed_from_payload():
    msg = client_message_adapter.validate_python(
        payload_with(
            {
                "size_tiers": [
                    {
                        "max_shares": "20",
                        "quote_depth": "aggressive",
                        "reward_min": "5",
                        "liq_min": "500",
                    },
                    {
                        "max_shares": "50",
                        "quote_depth": "normal",
                        "reward_min": "10",
                        "liq_min": "1000",
                    },
                    {
                        "max_shares": None,
                        "quote_depth": "safe",
                        "reward_min": "25",
                        "liq_min": "2000",
                    },
                ],
            }
        )
    )
    assert isinstance(msg, FarmCreateMessage)
    assert [t.liq_min for t in msg.size_tiers] == [Decimal("500"), Decimal("1000"), Decimal("2000")]


def test_size_tier_liq_min_omitted_defaults_to_zero():
    msg = client_message_adapter.validate_python(
        payload_with(
            {"size_tiers": [{"max_shares": "20", "quote_depth": "normal", "reward_min": "5"}]}
        )
    )
    assert msg.size_tiers[0].liq_min == Decimal("0")


def test_negative_tier_liq_min_rejected():
    with pytest.raises(ValidationError):
        client_message_adapter.validate_python(
            payload_with(
                {
                    "size_tiers": [
                        {
                            "max_shares": "20",
                            "quote_depth": "normal",
                            "reward_min": "5",
                            "liq_min": "-1",
                        }
                    ]
                }
            )
        )


def test_frontend_liquidity_payload_shape():
    payload = _payload({"liq_min": "2000", "liq_max": "1000000000"})
    payload["size_tiers"] = [
        {"max_shares": "20", "quote_depth": "aggressive", "reward_min": "5", "liq_min": "500"},
        {"max_shares": None, "quote_depth": "safe", "reward_min": "25", "liq_min": "2000"},
    ]
    msg = client_message_adapter.validate_python(payload)
    assert msg.filters.liq_max == Decimal("1000000000")
    assert msg.filters.liq_min == Decimal("2000")
    assert [t.liq_min for t in msg.size_tiers] == [Decimal("500"), Decimal("2000")]


def test_size_tier_zone_liq_max_parsed_from_payload():
    msg = client_message_adapter.validate_python(
        payload_with(
            {
                "size_tiers": [
                    {
                        "max_shares": "20",
                        "quote_depth": "aggressive",
                        "reward_min": "5",
                        "liq_min": "500",
                        "zone_liq_max": "1500",
                    },
                    {
                        "max_shares": None,
                        "quote_depth": "safe",
                        "reward_min": "25",
                        "liq_min": "2000",
                        "zone_liq_max": None,
                    },
                ],
            }
        )
    )
    assert isinstance(msg, FarmCreateMessage)
    assert msg.size_tiers[0].zone_liq_max == Decimal("1500")
    assert msg.size_tiers[1].zone_liq_max is None


def test_size_tier_zone_liq_max_omitted_defaults_to_none():
    msg = client_message_adapter.validate_python(
        payload_with(
            {"size_tiers": [{"max_shares": "20", "quote_depth": "normal", "reward_min": "5"}]}
        )
    )
    assert msg.size_tiers[0].zone_liq_max is None


def test_negative_tier_zone_liq_max_rejected():
    with pytest.raises(ValidationError):
        client_message_adapter.validate_python(
            payload_with(
                {
                    "size_tiers": [
                        {
                            "max_shares": "20",
                            "quote_depth": "normal",
                            "reward_min": "5",
                            "zone_liq_max": "-1",
                        }
                    ]
                }
            )
        )


def test_size_tier_time_remaining_parsed_from_payload():
    msg = client_message_adapter.validate_python(
        payload_with(
            {
                "size_tiers": [
                    {
                        "max_shares": "20",
                        "quote_depth": "aggressive",
                        "reward_min": "5",
                        "liq_min": "500",
                        "time_remaining": "12h",
                    },
                    {
                        "max_shares": None,
                        "quote_depth": "safe",
                        "reward_min": "25",
                        "liq_min": "2000",
                        "time_remaining": "7d",
                    },
                ],
            }
        )
    )
    assert isinstance(msg, FarmCreateMessage)
    assert msg.size_tiers[0].time_remaining == "12h"
    assert msg.size_tiers[1].time_remaining == "7d"


def test_size_tier_time_remaining_omitted_defaults_to_all():
    msg = client_message_adapter.validate_python(
        payload_with(
            {"size_tiers": [{"max_shares": "20", "quote_depth": "normal", "reward_min": "5"}]}
        )
    )
    assert msg.size_tiers[0].time_remaining == "all"


def test_invalid_tier_time_remaining_rejected():
    with pytest.raises(ValidationError):
        client_message_adapter.validate_python(
            payload_with(
                {
                    "size_tiers": [
                        {
                            "max_shares": "20",
                            "quote_depth": "normal",
                            "reward_min": "5",
                            "time_remaining": "13h",
                        }
                    ]
                }
            )
        )


def test_full_farm_create_message_round_trips_three_tiers_with_time_windows():
    # End-to-end: a full farm_create payload with 3 tiers (one omitting time_remaining ->
    # per-tier "all" default alongside explicit "12h"/"7d" siblings) validates into a FarmConfig
    # with every tier's window intact.
    payload = payload_with(
        {
            "size_tiers": [
                {"max_shares": "20", "quote_depth": "aggressive", "reward_min": "5"},
                {
                    "max_shares": "50",
                    "quote_depth": "normal",
                    "reward_min": "10",
                    "time_remaining": "12h",
                },
                {
                    "max_shares": None,
                    "quote_depth": "safe",
                    "reward_min": "25",
                    "time_remaining": "7d",
                },
            ],
        }
    )
    msg = client_message_adapter.validate_python(payload)
    assert isinstance(msg, FarmCreateMessage)
    assert isinstance(msg, FarmConfig)
    assert [t.time_remaining for t in msg.size_tiers] == ["all", "12h", "7d"]
    assert [t.max_shares for t in msg.size_tiers] == [Decimal("20"), Decimal("50"), None]


def test_full_frontend_tier_shape_parses():
    msg = client_message_adapter.validate_python(
        payload_with(
            {
                "size_tiers": [
                    {
                        "max_shares": "100",
                        "quote_depth": "safe",
                        "reward_min": "50",
                        "liq_min": "1000",
                        "zone_liq_max": "4000",
                        "time_remaining": "12h",
                    }
                ],
            }
        )
    )
    tier = msg.size_tiers[0]
    assert tier.max_shares == Decimal("100")
    assert tier.quote_depth == "safe"
    assert tier.reward_min == Decimal("50")
    assert tier.liq_min == Decimal("1000")
    assert tier.zone_liq_max == Decimal("4000")
    assert tier.time_remaining == "12h"
