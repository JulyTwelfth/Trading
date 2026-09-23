from decimal import Decimal

from app.farm.queue_surf import (
    is_sole_qualifier,
    passes_depth_gate,
    pick_surf_level,
    should_resurf,
)

D = Decimal
HOLD = 600.0


def test_gate_deep_passes():
    assert passes_depth_gate(D(900), D(20), D(5)) is True


def test_gate_thin_fails():
    assert passes_depth_gate(D(6), D(20), D(5)) is False


def test_gate_exact_boundary_passes():
    assert passes_depth_gate(D(100), D(20), D(5)) is True


def test_gate_dust_fails():
    assert passes_depth_gate(D("0.02"), D(20), D(5)) is False


def test_pick_best_bid_when_deep():
    bids = [(D("0.49"), D(900)), (D("0.48"), D(50)), (D("0.47"), D(800))]
    assert pick_surf_level(bids, D(20), D("0.50"), D(4), D(5)) == (D("0.49"), D(900))


def test_pick_moves_down_when_best_bid_thin():
    bids = [(D("0.49"), D("0.02")), (D("0.48"), D(6)), (D("0.47"), D(900))]
    assert pick_surf_level(bids, D(20), D("0.50"), D(4), D(5)) == (D("0.47"), D(900))


def test_pick_dust_best_bid_is_skipped():
    bids = [(D("0.49"), D("0.02")), (D("0.48"), D(900))]
    assert pick_surf_level(bids, D(20), D("0.50"), D(4), D(5)) == (D("0.48"), D(900))


def test_pick_none_when_all_thin():
    bids = [(D("0.49"), D(1)), (D("0.48"), D(2))]
    assert pick_surf_level(bids, D(20), D("0.50"), D(4), D(5)) is None


def test_pick_none_when_only_deep_level_is_out_of_band():
    bids = [(D("0.49"), D(1)), (D("0.45"), D(900))]
    assert pick_surf_level(bids, D(20), D("0.50"), D(3), D(5)) is None


def test_pick_handles_unsorted_input():
    bids = [(D("0.47"), D(800)), (D("0.49"), D(900)), (D("0.48"), D(50))]
    assert pick_surf_level(bids, D(20), D("0.50"), D(4), D(5)) == (D("0.49"), D(900))


def test_pick_none_on_empty_book():
    assert pick_surf_level([], D(20), D("0.50"), D(4), D(5)) is None


def test_gate_threshold_scales_with_our_size():
    assert passes_depth_gate(D(250), D(50), D(5)) is True
    assert passes_depth_gate(D(240), D(50), D(5)) is False
    assert passes_depth_gate(D(100), D(20), D(5)) is True


def test_pick_works_for_50_share_min():
    bids = [(D("0.49"), D(300)), (D("0.48"), D(200))]
    assert pick_surf_level(bids, D(50), D("0.50"), D(4), D(5)) == (D("0.49"), D(300))
    bids2 = [(D("0.49"), D(200)), (D("0.48"), D(900))]
    assert pick_surf_level(bids2, D(50), D("0.50"), D(4), D(5)) == (D("0.48"), D(900))


def test_pick_highest_qualifying_not_deepest():
    bids = [(D("0.49"), D(200)), (D("0.48"), D(5000))]
    assert pick_surf_level(bids, D(20), D("0.50"), D(4), D(5)) == (D("0.49"), D(200))


def test_pick_excludes_exact_band_edge():
    bids = [(D("0.49"), D(1)), (D("0.45"), D(900))]
    assert pick_surf_level(bids, D(20), D("0.50"), D(5), D(5)) is None


def test_pick_includes_just_inside_band_edge():
    bids = [(D("0.49"), D(1)), (D("0.46"), D(900))]
    assert pick_surf_level(bids, D(20), D("0.50"), D(5), D(5)) == (D("0.46"), D(900))


def test_pick_best_bid_out_of_band_wide_spread():
    bids = [(D("0.46"), D(900))]
    assert pick_surf_level(bids, D(20), D("0.50"), D(3), D(5)) is None


def test_pick_skips_two_thin_to_third_deep():
    bids = [(D("0.49"), D(1)), (D("0.48"), D(2)), (D("0.47"), D(900))]
    assert pick_surf_level(bids, D(20), D("0.50"), D(4), D(5)) == (D("0.47"), D(900))


def test_pick_fractional_shares():
    bids = [(D("0.49"), D("99.99")), (D("0.48"), D("250.5"))]
    got = pick_surf_level(bids, D(20), D("0.50"), D(4), D(5))
    assert got is not None and got[0] == D("0.48")


def test_pick_single_deep_level():
    got = pick_surf_level([(D("0.49"), D(500))], D(20), D("0.50"), D(4), D(5))
    assert got == (D("0.49"), D(500))


def test_pick_single_thin_level():
    assert pick_surf_level([(D("0.49"), D(10))], D(20), D("0.50"), D(4), D(5)) is None


def test_pick_bid_at_mid_is_in_band():
    got = pick_surf_level([(D("0.50"), D(900))], D(20), D("0.50"), D(4), D(5))
    assert got == (D("0.50"), D(900))


def test_pick_does_not_mutate_input():
    bids = [(D("0.47"), D(800)), (D("0.49"), D(900)), (D("0.48"), D(50))]
    original = list(bids)
    pick_surf_level(bids, D(20), D("0.50"), D(4), D(5))
    assert bids == original


def test_gate_ratio_of_one():
    assert passes_depth_gate(D(20), D(20), D(1)) is True
    assert passes_depth_gate(D(19), D(20), D(1)) is False


def test_gate_fractional_ratio():
    assert passes_depth_gate(D(50), D(20), D("2.5")) is True
    assert passes_depth_gate(D(49), D(20), D("2.5")) is False


def test_gate_fractional_sizes():
    assert passes_depth_gate(D("62.5"), D("12.5"), D(5)) is True
    assert passes_depth_gate(D("62.4"), D("12.5"), D(5)) is False


def test_resurf_all_conditions_met():
    assert should_resurf(700, True, True, HOLD) is True


def test_resurf_exact_hold_boundary_included():
    assert should_resurf(600, True, True, HOLD) is True


def test_resurf_too_young():
    assert should_resurf(599, True, True, HOLD) is False


def test_resurf_not_moved_up():
    assert should_resurf(700, False, True, HOLD) is False


def test_resurf_no_surf_level():
    assert should_resurf(700, True, False, HOLD) is False


def test_resurf_nothing_met():
    assert should_resurf(0, False, False, HOLD) is False


def test_pick_skips_deep_out_of_band_returns_none():
    bids = [(D("0.40"), D(100000))]
    assert pick_surf_level(bids, D(100), D("0.50"), D(3), D("0.2")) is None


def test_pick_prefers_inband_over_deeper_out_of_band():
    bids = [(D("0.49"), D(30)), (D("0.40"), D(100000))]
    assert pick_surf_level(bids, D(100), D("0.50"), D(3), D("0.2")) == (D("0.49"), D(30))


def test_pick_returned_price_strictly_inside_band():
    bids = [(D("0.49"), D(50)), (D("0.46"), D(100000))]
    got = pick_surf_level(bids, D(100), D("0.50"), D(3), D("0.2"))
    assert got is not None
    assert D("0.50") - got[0] < D("0.03")
    assert got[0] == D("0.49")


def test_sole_when_book_empty():
    assert is_sole_qualifier([], D(20), D("0.50"), D(4)) is True


def test_sole_when_all_inband_below_min():
    bids = [(D("0.49"), D(19)), (D("0.48"), D(5))]
    assert is_sole_qualifier(bids, D(20), D("0.50"), D(4)) is True


def test_not_sole_when_inband_qualifier_present():
    bids = [(D("0.49"), D(20))]
    assert is_sole_qualifier(bids, D(20), D("0.50"), D(4)) is False


def test_sole_ignores_out_of_band_qualifier():
    bids = [(D("0.45"), D(900))]
    assert is_sole_qualifier(bids, D(20), D("0.50"), D(4)) is True


def test_sole_band_edge_is_out_of_band():
    bids = [(D("0.46"), D(900))]
    assert is_sole_qualifier(bids, D(20), D("0.50"), D(4)) is True


def test_not_sole_just_inside_band():
    bids = [(D("0.47"), D(900))]
    assert is_sole_qualifier(bids, D(20), D("0.50"), D(4)) is False


def test_sole_discounts_our_own_order():
    bids = [(D("0.49"), D(20))]
    assert (
        is_sole_qualifier(bids, D(20), D("0.50"), D(4), our_price=D("0.49"), our_size=D(20)) is True
    )


def test_not_sole_competitor_stacked_at_our_level():
    bids = [(D("0.49"), D(40))]
    assert (
        is_sole_qualifier(bids, D(20), D("0.50"), D(4), our_price=D("0.49"), our_size=D(20))
        is False
    )


def test_not_sole_other_competitor_despite_our_exclusion():
    bids = [(D("0.49"), D(20)), (D("0.48"), D(50))]
    assert (
        is_sole_qualifier(bids, D(20), D("0.50"), D(4), our_price=D("0.49"), our_size=D(20))
        is False
    )


def test_not_sole_when_our_price_matches_no_level():
    bids = [(D("0.49"), D(50))]
    assert (
        is_sole_qualifier(bids, D(20), D("0.50"), D(4), our_price=D("0.47"), our_size=D(20))
        is False
    )


def test_sole_many_submin_levels():
    bids = [(D("0.49"), D(19)), (D("0.48"), D(19)), (D("0.47"), D(10))]
    assert is_sole_qualifier(bids, D(20), D("0.50"), D(4)) is True


def test_our_oversized_order_cannot_flip_a_real_competitor():
    bids = [(D("0.49"), D(500)), (D("0.48"), D(40))]
    assert (
        is_sole_qualifier(bids, D(20), D("0.50"), D(4), our_price=D("0.49"), our_size=D(500))
        is False
    )
