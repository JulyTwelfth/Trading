"""Football transfer markets ("Will X join / stay at <club>?") gap ~25c in seconds on a transfer
report — the costliest fill type (~-$6/fill vs ~-$0.30 elsewhere). Block them like crisis/mention,
but precisely: 'join the EU' (politics, the best aggressive category) and 'designated'/'resign'/
'signal' (substring 'sign') must NOT false-positive."""

from datetime import datetime, timezone

from app.farm.filters import first_failing_filter, passes_transfer_filter
from app.farm.schemas import Market

NOW = datetime(2026, 6, 19, tzinfo=timezone.utc)


def mkt(market: Market, question: str = "?", slug: str = "s") -> Market:
    return market.model_copy(update={"question": question, "slug": slug})


def test_blocks_join_and_stay_at_by_slug(market: Market):
    for slug in (
        "will-yan-diomande-join-liverpool-20260612220856166",
        "will-casemiro-join-al-nassr-20260612232503478",
        "will-joao-cancelo-stay-at-al-hilal-20260612225521996",
        "will-bruno-fernandes-stay-at-manchester-united-20260612223235669",
    ):
        assert passes_transfer_filter(mkt(market, "?", slug)) is False, slug


def test_blocks_by_question(market: Market):
    assert passes_transfer_filter(mkt(market, "Will Rodri join Real Madrid?")) is False
    assert passes_transfer_filter(mkt(market, "Will Vinicius stay at Real Madrid?")) is False


def test_blocks_sign_for_and_transfer(market: Market):
    assert passes_transfer_filter(mkt(market, "Will Sandro Tonali sign for Liverpool?")) is False
    assert passes_transfer_filter(mkt(market, "Will the transfer window close early?")) is False


def test_allows_join_the_org_politics(market: Market):
    # "join the EU / bloc" is politics (our best aggressive category) — NOT a transfer.
    for q in ("Will Sweden join the EU by 2027?", "Will Canada join the new trade bloc?"):
        assert passes_transfer_filter(mkt(market, q)) is True, q


def test_no_false_positive_on_sign_substring(market: Market):
    # 'designated' / 'resign' / 'signal' contain 'sign' but are not "sign for/with".
    for q in (
        "Will Kyle Schwarber win the Edgar Martinez Outstanding Designated Hitter award?",
        "Will the president resign before July?",
        "Will the Fed signal a rate cut in July?",
    ):
        assert passes_transfer_filter(mkt(market, q)) is True, q


def test_allows_ordinary_markets(market: Market):
    for q in ("Will the Lakers win the title?", "Aberdeen by-election winner?", "Highest temp?"):
        assert passes_transfer_filter(mkt(market, q)) is True, q


def test_named_in_funnel(market: Market, farm_state):
    m = mkt(market, "Will Yan Diomande join Liverpool?")
    assert first_failing_filter(m, farm_state.config.filters, NOW) == "transfer"
