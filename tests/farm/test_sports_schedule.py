from datetime import datetime, timedelta, timezone

from app.farm.filters import first_failing_filter
from app.farm.schemas import FarmState, Market
from app.farm.sports_roster import NATION_NAME, PLAYER_NATION
from app.farm.sports_schedule import (
    annotate_sports_gate,
    build_fixture_index,
    in_match_window,
    normalize_name,
    wc_scorer_player,
    wc_team_prop,
)

KICK = datetime(2026, 6, 21, 16, 0, tzinfo=timezone.utc)


def _prop(market: Market, question: str) -> Market:
    return market.model_copy(update={"question": question, "slug": "p", "game_start_time": None})


def _fixture(market: Market, slug: str, kickoff: datetime) -> Market:
    return market.model_copy(update={"slug": slug, "game_start_time": kickoff})


def test_normalize_name_accents_and_whitespace():
    assert normalize_name("Sadio Mané") == "sadio mane"
    assert normalize_name("  Lamine   Yamal ") == "lamine yamal"


def test_wc_scorer_player_detects_and_extracts(market: Market):
    m = _prop(market, "Will Michael Olise score a goal at the 2026 FIFA World Cup?")
    assert wc_scorer_player(m) == "Michael Olise"


def test_wc_scorer_player_ignores_single_game(market: Market):
    m = market.model_copy(
        update={"question": "Will Spain win on 2026-06-21?", "game_start_time": KICK}
    )
    assert wc_scorer_player(m) is None


def test_wc_scorer_player_ignores_non_world_cup(market: Market):
    assert wc_scorer_player(_prop(market, "Will Patrick Mahomes score a touchdown?")) is None


def test_build_fixture_index_parses_both_team_codes(market: Market, monkeypatch):
    monkeypatch.setattr("app.farm.sports_schedule.WC_KNOCKOUT_FIXTURES", {})
    games = [
        _fixture(market, "fifwc-esp-ksa-2026-06-21-esp", KICK),
        _fixture(market, "fifwc-ury-esp-2026-06-26", KICK + timedelta(days=5)),
        _prop(market, "Will X score a goal at the 2026 FIFA World Cup?"),
        _fixture(market, "nba-lal-bos-2026-06-21", KICK),
    ]
    idx = build_fixture_index(games)
    assert idx["esp"] == [KICK, KICK + timedelta(days=5)]
    assert idx["ksa"] == [KICK]
    assert idx["ury"] == [KICK + timedelta(days=5)]
    assert "lal" not in idx


def test_in_match_window_boundaries():
    assert in_match_window([KICK], KICK)
    assert in_match_window([KICK], KICK - timedelta(hours=1))
    assert in_match_window([KICK], KICK + timedelta(hours=2))
    assert not in_match_window([KICK], KICK - timedelta(hours=3))
    assert not in_match_window([KICK], KICK + timedelta(hours=4))
    assert not in_match_window([], KICK)


def test_blocks_player_in_match_window_and_names_funnel(market: Market, farm_state: FarmState):
    roster = {"lamine yamal": "esp"}
    prop = _prop(market, "Will Lamine Yamal score a goal at the 2026 FIFA World Cup?")
    markets = [prop, _fixture(market, "fifwc-esp-ksa-2026-06-21-esp", KICK)]
    blocked, unmapped = annotate_sports_gate(markets, KICK, roster=roster)
    assert blocked == 1
    assert unmapped == []
    assert prop.sports_event_active is True
    assert first_failing_filter(prop, farm_state.config.filters) == "sports_event_active"


def test_passes_off_window(market: Market):
    roster = {"lamine yamal": "esp"}
    prop = _prop(market, "Will Lamine Yamal score a goal at the 2026 FIFA World Cup?")
    markets = [prop, _fixture(market, "fifwc-esp-ksa-2026-06-21-esp", KICK)]
    blocked, _ = annotate_sports_gate(markets, KICK + timedelta(days=2), roster=roster)
    assert blocked == 0
    assert prop.sports_event_active is False


def test_unmapped_player_fails_safe_blocks(market: Market):
    prop = _prop(market, "Will Unknown Person score a goal at the 2026 FIFA World Cup?")
    blocked, unmapped = annotate_sports_gate([prop], KICK, roster={})
    assert blocked == 1
    assert unmapped == ["Unknown Person"]
    assert prop.sports_event_active is True


def test_unmapped_player_blocked_even_off_window(market: Market):
    prop = _prop(market, "Will Mystery Player score a goal at the 2026 FIFA World Cup?")
    blocked, unmapped = annotate_sports_gate([prop], KICK + timedelta(days=30), roster={})
    assert blocked == 1
    assert unmapped == ["Mystery Player"]
    assert prop.sports_event_active is True


def test_mixed_mapped_and_unmapped_both_block(market: Market):
    roster = {"lamine yamal": "esp"}
    mapped = _prop(market, "Will Lamine Yamal score a goal at the 2026 FIFA World Cup?")
    unmapped_prop = _prop(market, "Will Nobody Known score a goal at the 2026 FIFA World Cup?")
    fixture = _fixture(market, "fifwc-esp-ksa-2026-06-21-esp", KICK)
    blocked, unmapped = annotate_sports_gate([mapped, unmapped_prop, fixture], KICK, roster=roster)
    assert blocked == 2
    assert unmapped == ["Nobody Known"]
    assert mapped.sports_event_active is True
    assert unmapped_prop.sports_event_active is True


def test_added_star_players_are_mapped():
    expected = {
        "Erling Haaland": "nor",
        "Harry Kane": "eng",
        "Kim Min-jae": "kor",
        "Kylian Mbappe": "fra",
        "Neymar": "bra",
        "Son Heung-min": "kor",
        "Vinicius Jr.": "bra",
    }
    for name, code in expected.items():
        assert PLAYER_NATION[normalize_name(name)] == code


def test_mapped_star_blocks_in_window_via_real_roster(market: Market):
    prop = _prop(market, "Will Kylian Mbappe score a goal at the 2026 FIFA World Cup?")
    fixture = _fixture(market, "fifwc-fra-sen-2026-06-21-fra", KICK)
    blocked, unmapped = annotate_sports_gate([prop, fixture], KICK)
    assert blocked == 1
    assert unmapped == []
    assert prop.sports_event_active is True


def test_mapped_team_with_no_fixtures_passes(market: Market):
    roster = {"lamine yamal": "esp"}
    prop = _prop(market, "Will Lamine Yamal score a goal at the 2026 FIFA World Cup?")
    blocked, unmapped = annotate_sports_gate([prop], KICK, roster=roster)
    assert blocked == 0
    assert unmapped == []
    assert prop.sports_event_active is False


def test_wc_team_prop_detects_and_extracts(market: Market):
    m = _prop(market, "Will Paraguay reach the quarterfinals at the 2026 FIFA World Cup?")
    assert wc_team_prop(m) == "Paraguay"


def test_wc_team_prop_variants(market: Market):
    cases = [
        ("Will Spain win the 2026 FIFA World Cup?", "Spain"),
        ("Will Brazil go unbeaten in the 2026 FIFA World Cup group stage?", "Brazil"),
        (
            "Will Curacao be an advancing group stage third-place team at the 2026 World Cup?",
            "Curacao",
        ),
    ]
    for q, team in cases:
        assert wc_team_prop(_prop(market, q)) == team, q


def test_wc_team_prop_ignores_goalscorer_and_single_game(market: Market):
    gs = _prop(market, "Will Lamine Yamal score a goal at the 2026 FIFA World Cup?")
    assert wc_team_prop(gs) is None
    single = market.model_copy(
        update={"question": "Will Spain win on 2026-06-21?", "game_start_time": KICK}
    )
    assert wc_team_prop(single) is None
    other = _prop(market, "Will there be 100 total goals at the 2026 FIFA World Cup?")
    assert wc_team_prop(other) is None


def test_team_prop_blocked_in_window_via_real_map(market: Market):
    prop = _prop(market, "Will Spain reach the quarterfinals at the 2026 FIFA World Cup?")
    fixture = _fixture(market, "fifwc-esp-ksa-2026-06-21-esp", KICK)
    blocked, unmapped = annotate_sports_gate([prop, fixture], KICK)
    assert blocked == 1
    assert unmapped == []
    assert prop.sports_event_active is True


def test_team_prop_farmed_off_window(market: Market):
    prop = _prop(market, "Will Spain reach the quarterfinals at the 2026 FIFA World Cup?")
    fixture = _fixture(market, "fifwc-esp-ksa-2026-06-21-esp", KICK)
    blocked, _ = annotate_sports_gate([prop, fixture], KICK + timedelta(days=2))
    assert blocked == 0
    assert prop.sports_event_active is False


def test_unmapped_team_fails_open(market: Market):
    prop = _prop(market, "Will Wakanda reach the quarterfinals at the 2026 FIFA World Cup?")
    fixture = _fixture(market, "fifwc-fra-sen-2026-06-21-sen", KICK)
    blocked, unmapped = annotate_sports_gate([prop, fixture], KICK, nations={})
    assert blocked == 0
    assert prop.sports_event_active is False
    assert "Wakanda" in unmapped


def test_team_prop_with_leading_article_blocked_in_window(market: Market):
    prop = _prop(
        market,
        "Will the United States advance to the round of 16 at the 2026 FIFA World Cup?",
    )
    fixture = _fixture(market, "fifwc-usa-mex-2026-06-21-usa", KICK)
    blocked, unmapped = annotate_sports_gate([prop, fixture], KICK)
    assert blocked == 1
    assert unmapped == []
    assert prop.sports_event_active is True


def test_nation_map_has_expected_codes():
    expected = {"France": "fra", "Spain": "esp", "Paraguay": "par", "Croatia": "hrv"}
    for name, code in expected.items():
        assert NATION_NAME[normalize_name(name)] == code
