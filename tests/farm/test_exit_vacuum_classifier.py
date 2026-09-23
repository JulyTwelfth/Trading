"""exit_vacuum_price — the shadow-mode 4-gate vacuum classifier (pure, read-only, no trading).

This pins the classifier against the 5 REAL post-fill incidents it was built from: it must FIRE
(return a rest price it would wait to sell at) on a recoverable liquidity vacuum, and DUMP (return
None) on a genuine crash / directional move. These verdicts ARE the data-quality contract — if a
constant drifts and a verdict flips, the shadow logs become misleading, so they must fail loudly.

The classifier returns a value in (best_bid, entry_px] on FIRE, else None. `tick` is accepted for
interface symmetry with the eventual live exit-price computation and is unused by the gates.
"""

from decimal import Decimal

import pytest

from app.bot.schemas import BookLevel
from app.constants import (
    VACUUM_BID_DROP,
    VACUUM_MIN_SPREAD,
)
from app.farm.exit_cost import exit_vacuum_price

TICK = Decimal("0.01")


def _lvls(*pairs: tuple[str, str]) -> list[BookLevel]:
    return [BookLevel(price=Decimal(p), size=Decimal(s)) for p, s in pairs]


# ── The 5 real incidents: FIRE = recoverable vacuum, DUMP(None) = real crash ──────────────


def test_canva_low_fires():
    # Canva @0.82: bid vacuumed to 0.11 but the ask held at entry (0.82) — a wide, empty-bid
    # book that a "wait, then rest a concession off the ask" exit could have worked. FIRE.
    rest = exit_vacuum_price(
        _lvls(("0.11", "500")), _lvls(("0.82", "500")), Decimal("0.82"), TICK, Decimal("0.03")
    )
    assert rest is not None
    assert Decimal("0.11") < rest <= Decimal("0.82")
    assert rest == Decimal("0.77")  # min(entry, best_ask - VACUUM_CONCESSION) = 0.82 - 0.05


def test_ai_model_fires():
    # AI-model @0.60: bid dropped to 0.42, ask held near entry at 0.59, spread 0.17 clears the
    # floor even against the 2x0.045 reward-band gate. FIRE.
    rest = exit_vacuum_price(
        _lvls(("0.42", "500")), _lvls(("0.59", "500")), Decimal("0.60"), TICK, Decimal("0.045")
    )
    assert rest is not None
    assert Decimal("0.42") < rest <= Decimal("0.60")
    assert rest == Decimal("0.54")  # 0.59 - 0.05


def test_databricks_fires_documented_benign_false_positive():
    # Databricks @0.28: bid collapsed to 0.07, ask at 0.31, spread 0.24 — LOOKS like a recoverable
    # vacuum so the classifier FIRES, but this one did NOT recover. This is the DOCUMENTED benign
    # false positive: the 4-gate classifier structurally cannot catch it (the book is
    # indistinguishable at fill time from Canva/AI-model). It is saved downstream by the
    # self-limiting exit (small size, sub-tick strand write-off), NOT by classification. Pinned so
    # the verdict stays "fire" — flipping it would mean the gates changed, not that this got fixed.
    rest = exit_vacuum_price(
        _lvls(("0.07", "500")), _lvls(("0.31", "500")), Decimal("0.28"), TICK, Decimal("0.03")
    )
    assert rest is not None
    assert Decimal("0.07") < rest <= Decimal("0.28")
    assert rest == Decimal("0.26")  # 0.31 - 0.05


def test_brazil_dumps_via_empty_book_gate():
    # Brazil GDP: the bid side went genuinely empty (no bids at all) — the one clean real-crash
    # signal. GATE 1 (best_bid <= 0) => DUMP. There is no price to wait for; dumping is correct.
    rest = exit_vacuum_price(
        [], _lvls(("0.47", "500")), Decimal("0.50"), TICK, Decimal("0.03")
    )
    assert rest is None


def test_canvas_high_dumps_via_spread_floor_gate():
    # Canvas-high @0.74: bid 0.71 / ask 0.77 — spread 0.06 < VACUUM_MIN_SPREAD (0.15). This is a
    # normal tight-ish book, not a vacuum. GATE 2 (spread floor) => DUMP.
    rest = exit_vacuum_price(
        _lvls(("0.71", "500")), _lvls(("0.77", "500")), Decimal("0.74"), TICK, Decimal("0.03")
    )
    assert rest is None


# ── Individual gate isolation (each failing gate must DUMP) ────────────────────────────────


def test_no_asks_dumps():
    # A book with bids but no asks has no ask to rest a concession off of. GATE 2 => DUMP.
    rest = exit_vacuum_price(
        _lvls(("0.40", "500")), [], Decimal("0.60"), TICK, Decimal("0.03")
    )
    assert rest is None


def test_bid_not_collapsed_dumps_via_gate3():
    # entry 0.60, bid still at 0.50 (only 0.10 below entry < VACUUM_BID_DROP 0.15): the bid has
    # NOT actually collapsed, so this is not a vacuum. GATE 3 => DUMP. (spread 0.25 clears gate 2.)
    rest = exit_vacuum_price(
        _lvls(("0.50", "500")), _lvls(("0.75", "500")), Decimal("0.60"), TICK, Decimal("0.03")
    )
    assert rest is None


def test_ask_dropped_dumps_via_gate4():
    # entry 0.60, ask fell to 0.54 (> VACUUM_ASK_HOLD_TOL 0.05 below entry): the WHOLE book slid
    # down — a real directional move, not a one-sided liquidity vacuum. GATE 4 => DUMP.
    # (bid 0.35 clears gate 3, spread 0.19 clears gate 2, so gate 4 is the one that fires.)
    rest = exit_vacuum_price(
        _lvls(("0.35", "500")), _lvls(("0.54", "500")), Decimal("0.60"), TICK, Decimal("0.03")
    )
    assert rest is None


# ── Reward-band sensitivity: the spread floor scales with 2x the reward band ───────────────


def test_reward_band_widens_spread_floor_to_dump():
    # SAME book as AI-model (bid 0.42 / ask 0.59, spread 0.17) that FIRED at band 0.045. With a
    # wide band 0.10 the floor becomes max(0.15, 2*0.10=0.20) = 0.20; spread 0.17 < 0.20 so GATE 2
    # now DUMPS. Proves the band — not just VACUUM_MIN_SPREAD — governs the spread floor.
    fires = exit_vacuum_price(
        _lvls(("0.42", "500")), _lvls(("0.59", "500")), Decimal("0.60"), TICK, Decimal("0.045")
    )
    dumps = exit_vacuum_price(
        _lvls(("0.42", "500")), _lvls(("0.59", "500")), Decimal("0.60"), TICK, Decimal("0.10")
    )
    assert fires is not None  # 0.17 >= max(0.15, 0.09)
    assert dumps is None      # 0.17 <  max(0.15, 0.20)


# ── The rest_px <= best_bid gate (exit_cost.py:103-104) is UNREACHABLE dead code ───────────


def test_rest_px_gate_is_unreachable_by_construction():
    """The final `if rest_px <= best_bid: return None` (GATE 5) is DEAD CODE given the current
    constants. The coder flagged this; we confirm it and pin WHY, so a future constant change that
    revives the path fails here rather than silently going live untested.

    To reach GATE 5 a book must pass gates 1-4:
      - gate 2: best_ask - best_bid >= VACUUM_MIN_SPREAD (0.15)
      - gate 3: best_bid <= entry_px - VACUUM_BID_DROP  (entry - 0.15)
      - gate 4: best_ask >= entry_px - VACUUM_ASK_HOLD_TOL (entry - 0.05)
    rest_px = min(entry_px, best_ask - VACUUM_CONCESSION):
      * if rest_px == entry_px: entry_px <= best_bid is impossible (gate 3 => best_bid < entry_px).
      * if rest_px == best_ask - 0.05: best_ask - 0.05 <= best_bid => spread <= 0.05, but gate 2
        forces spread >= 0.15. Impossible.
    => whenever gates 1-4 pass, rest_px is strictly > best_bid, so GATE 5 never returns None.

    NOTE (surfaced in the report): this is unreachable DEFENSIVE code, not a bug — every verdict is
    correct; the branch is simply never exercised, so line 104 stays uncovered.
    """
    # The tightest boundary book: best_bid exactly at the gate-3 cap, spread exactly the floor.
    entry = Decimal("0.50")
    best_bid = entry - VACUUM_BID_DROP  # 0.35 — the largest bid gate 3 permits
    best_ask = best_bid + VACUUM_MIN_SPREAD  # 0.50 — the tightest spread gate 2 permits
    rest = exit_vacuum_price(
        _lvls((str(best_bid), "100")),
        _lvls((str(best_ask), "100")),
        entry,
        TICK,
        Decimal("0.03"),
    )
    assert rest is not None and rest > best_bid  # fires; never falls into the dead gate


# ── Contract invariant: FIRE => rest in (best_bid, entry]; else None ───────────────────────


@pytest.mark.parametrize("entry", [Decimal("0.3"), Decimal("0.5"), Decimal("0.7"), Decimal("0.82")])
@pytest.mark.parametrize("best_bid", [Decimal("0.02"), Decimal("0.10"), Decimal("0.30")])
@pytest.mark.parametrize("best_ask", [Decimal("0.30"), Decimal("0.50"), Decimal("0.77")])
@pytest.mark.parametrize("band", [Decimal("0.01"), Decimal("0.03"), Decimal("0.09")])
def test_result_is_none_or_in_open_bid_to_entry_interval(entry, best_bid, best_ask, band):
    # Across the grid the docstring contract holds universally: a returned rest price is always
    # strictly above the current bid (else waiting there would just re-fill us) and never above
    # entry (we never rest at a profit that would sit un-hit), else None.
    rest = exit_vacuum_price(
        _lvls((str(best_bid), "100")), _lvls((str(best_ask), "100")), entry, TICK, band
    )
    assert rest is None or best_bid < rest <= entry
