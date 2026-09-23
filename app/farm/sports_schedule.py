import re
import unicodedata
from datetime import datetime, timedelta, timezone

from app.constants import SPORTS_MATCH_POSTGAME_HOURS, SPORTS_MATCH_PREGAME_HOURS
from app.farm.schemas import Market
from app.farm.sports_roster import NATION_NAME, PLAYER_NATION

SCORER_RE = re.compile(r"^will\s+(.+?)\s+score\b", re.IGNORECASE)
FIXTURE_SLUG_RE = re.compile(r"^fifwc-([a-z]{3})-([a-z]{3})-\d{4}-\d{2}-\d{2}", re.IGNORECASE)

TEAM_OUTCOME_RE = re.compile(
    r"^will\s+(.+?)\s+(?:"
    r"reach the (?:quarter|semi)?finals?"
    r"|reach the round of"
    r"|be an advancing group stage"
    r"|advance (?:to|from|past|out of|beyond)"
    r"|go unbeaten"
    r"|win (?:their |the )?group"
    r"|win(?:s)? the (?:20\d\d )?(?:fifa )?world cup"
    r"|be eliminated"
    r"|finish (?:top|first|second|bottom|last)"
    r")\b",
    re.IGNORECASE,
)


def normalize_name(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_name.casefold().split())


def wc_scorer_player(market: Market) -> str | None:
    if market.game_start_time is not None:
        return None
    if "world cup" not in market.question.casefold():
        return None
    match = SCORER_RE.match(market.question)
    return match.group(1).strip() if match else None


def wc_team_prop(market: Market) -> str | None:
    if market.game_start_time is not None:
        return None
    if "world cup" not in market.question.casefold():
        return None
    if SCORER_RE.match(market.question):
        return None
    match = TEAM_OUTCOME_RE.match(market.question)
    if match is None:
        return None
    team = match.group(1).strip()
    if team[:4].casefold() == "the ":
        team = team[4:].strip()
    return team


WC_KNOCKOUT_FIXTURES: dict[str, list[datetime]] = {
    "civ": [datetime(2026, 6, 30, 17, 0, tzinfo=timezone.utc)],
    "nor": [datetime(2026, 6, 30, 17, 0, tzinfo=timezone.utc)],
    "fra": [datetime(2026, 6, 30, 21, 0, tzinfo=timezone.utc)],
    "swe": [datetime(2026, 6, 30, 21, 0, tzinfo=timezone.utc)],
    "mex": [datetime(2026, 7, 1, 1, 0, tzinfo=timezone.utc)],
    "ecu": [datetime(2026, 7, 1, 1, 0, tzinfo=timezone.utc)],
    "eng": [datetime(2026, 7, 1, 16, 0, tzinfo=timezone.utc)],
    "cdr": [datetime(2026, 7, 1, 16, 0, tzinfo=timezone.utc)],
    "bel": [datetime(2026, 7, 1, 20, 0, tzinfo=timezone.utc)],
    "sen": [datetime(2026, 7, 1, 20, 0, tzinfo=timezone.utc)],
    "usa": [datetime(2026, 7, 2, 0, 0, tzinfo=timezone.utc)],
    "esp": [datetime(2026, 7, 2, 19, 0, tzinfo=timezone.utc)],
    "aut": [datetime(2026, 7, 2, 19, 0, tzinfo=timezone.utc)],
    "prt": [datetime(2026, 7, 2, 23, 0, tzinfo=timezone.utc)],
    "hrv": [datetime(2026, 7, 2, 23, 0, tzinfo=timezone.utc)],
    "che": [datetime(2026, 7, 3, 3, 0, tzinfo=timezone.utc)],
    "egy": [datetime(2026, 7, 3, 18, 0, tzinfo=timezone.utc)],
    "arg": [datetime(2026, 7, 3, 22, 0, tzinfo=timezone.utc)],
    "col": [datetime(2026, 7, 4, 1, 30, tzinfo=timezone.utc)],
    "gha": [datetime(2026, 7, 4, 1, 30, tzinfo=timezone.utc)],
    "can": [datetime(2026, 7, 4, 17, 0, tzinfo=timezone.utc)],
    "mar": [datetime(2026, 7, 4, 17, 0, tzinfo=timezone.utc)],
    "par": [datetime(2026, 7, 4, 21, 0, tzinfo=timezone.utc)],
}


def build_fixture_index(markets: list[Market]) -> dict[str, list[datetime]]:
    index: dict[str, set[datetime]] = {
        code: set(times) for code, times in WC_KNOCKOUT_FIXTURES.items()
    }
    for market in markets:
        if market.game_start_time is None:
            continue
        match = FIXTURE_SLUG_RE.match(market.slug)
        if match is None:
            continue
        for code in match.groups():
            index.setdefault(code.lower(), set()).add(market.game_start_time)
    return {code: sorted(times) for code, times in index.items()}


def in_match_window(kickoffs: list[datetime], now: datetime) -> bool:
    pre = timedelta(hours=SPORTS_MATCH_PREGAME_HOURS)
    post = timedelta(hours=SPORTS_MATCH_POSTGAME_HOURS)
    return any(k - pre <= now <= k + post for k in kickoffs)


def annotate_sports_gate(
    markets: list[Market],
    now: datetime,
    roster: dict[str, str] | None = None,
    nations: dict[str, str] | None = None,
) -> tuple[int, list[str]]:
    roster = roster if roster is not None else PLAYER_NATION
    nations = nations if nations is not None else NATION_NAME
    fixtures = build_fixture_index(markets)
    blocked = 0
    unmapped: list[str] = []
    for market in markets:
        player = wc_scorer_player(market)
        if player is not None:
            code = roster.get(normalize_name(player))
            if code is None:
                unmapped.append(player)
                market.sports_event_active = True
                blocked += 1
            elif in_match_window(fixtures.get(code, []), now):
                market.sports_event_active = True
                blocked += 1
            continue
        team = wc_team_prop(market)
        if team is not None:
            code = nations.get(normalize_name(team))
            if code is None:
                unmapped.append(team)
            elif in_match_window(fixtures.get(code, []), now):
                market.sports_event_active = True
                blocked += 1
    return blocked, unmapped
