from decimal import Decimal

CLOB_HOST = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
WS_USER_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
WS_MARKET_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
WS_SIGN_PATH = "/ws/user"

SPREADS_BATCH_LIMIT = 500
GAMMA_MARKETS_BATCH_LIMIT = 50
BATCH_PRICES_HISTORY_LIMIT = 20
PAGINATION_END_CURSOR = "LTE="
GAMMA_MARKETS_MAX_RETRIES = 3
GAMMA_RETRY_BASE_DELAY_SECONDS = 0.5

HEARTBEAT_INTERVAL_SECONDS = 5
HEARTBEAT_ID_ROTATION_SECONDS = 3 * 60 * 60
HEARTBEAT_RETRY_DELAY_SECONDS = 1

TICK_INTERVAL_SECONDS = 60
POSITION_PRUNE_MISSING_TICKS = 2
POSITION_CANDIDATE_MISS_TICKS = 3
MAX_CONCURRENT_POSITIONS = 1000

FARM_CANCEL_GRACE_SECONDS = 5.0

MAX_WALLETS_PER_LICENSE = 50
WALLET_LIST_BALANCE_CONCURRENCY = 10

STALE_EXIT_SECONDS = 120
STALE_STAGED_EXIT_SECONDS = 3
ZERO_BALANCE_GIVEUP_SECONDS = 30
EXIT_RETRY_SECONDS = 0.5
EXIT_RECONCILE_SECONDS = 5
KILL_DRAIN_TIMEOUT_SECONDS = 60
KILL_DRAIN_POLL_SECONDS = 1
RECENT_EXIT_GRACE_SECONDS = 60
RECONCILE_BACKOFF_MAX_SECONDS = 900
ORDER_REGISTRY_RETENTION_SECONDS = 120
EXIT_DUST_BALANCE_SHARES = Decimal("0.1")
GUARD_PULL_RELEASE_SECONDS = 120
# Stale-residual fix: a partial-balance SELL rejection (exchange balance < tracked shares,
# e.g. a still-settling earlier fill) used to re-drive the FULL tracked size forever, since the
# exchange balance never caught up — a live-observed 6-minute retry loop. Now the exit is
# clamped to the exchange-reported balance and retried at that size.
MAX_CLAMP_RETRIES = 5  # recursion guard on clamp-and-retry
SUMMARY_INTERVAL_SECONDS = 5
BALANCE_FETCH_TIMEOUT_SECONDS = 3
REWARDS_POLL_INTERVAL_SECONDS = 60
USER_WS_RECONNECT_DELAY_SECONDS = 1.0
MARKET_WS_RECONNECT_DELAY_SECONDS = 1.0
FARM_CANCEL_TIMEOUT_SECONDS = 10
CANCEL_RETRY_DELAY_SECONDS = 0.5

# Pure instrumentation (crash-instrumentation branch): watches each filled market's book for a
# while after the fill, so we can later see whether a bid-vacuum crash recovers in time to make a
# "wait" exit worth it — today the bot goes blind the instant it dumps a fill.
POST_FILL_SAMPLE_SECONDS = 600
POST_FILL_SAMPLE_INTERVAL_SECONDS = 5

SMART_EXIT_FLOOR_ENABLED = True
SMART_EXIT_MIN_SAVING = Decimal("3")
SMART_EXIT_CONCESSION_TICKS = 2
SMART_EXIT_CONCESSION_REWARD_FRACTION = Decimal("0.5")
SMART_EXIT_COMPLEMENT_SPREAD_FACTOR = Decimal("2")
SMART_EXIT_DEPTH_FACTOR = Decimal("0.2")
SMART_EXIT_DEPTH_MIN_SHARES = Decimal("10")
SMART_EXIT_MAX_HOLD_SECONDS = 60
SMART_EXIT_EVENT_PROXIMITY_HOURS = 6
SMART_EXIT_POSTGAME_HOURS = 3

# Shadow-mode vacuum classifier (instrumentation only — logs a verdict, takes no action).
VACUUM_MIN_SPREAD = Decimal("0.15")
VACUUM_SPREAD_BAND_FACTOR = Decimal("2")
VACUUM_BID_DROP = Decimal("0.15")
VACUUM_ASK_HOLD_TOL = Decimal("0.05")
VACUUM_CONCESSION = Decimal("0.05")

LOG_RETENTION_DAYS = 60

BLACKLIST_STATE_DIR = "state"

DEFAULT_MIN_BID_DEPTH_MULT = Decimal("1.1")

REQUOTE_TICK_BUFFER = 2
REQUOTE_CANCEL_FAIL_PULL_THRESHOLD = 3
REQUOTE_HYSTERESIS_SECONDS = 3.0
REQUOTE_HYSTERESIS_EPISODE_GAP_SECONDS = 5.0
DEPTH_GUARD_PULL_COOLDOWN_SECONDS = 180
REQUOTE_GIVEUP_COOLDOWN_SECONDS = 600
BEST_BID_GAP_PULL_FRACTION = Decimal("0.10")
QUEUE_SURF_MIN_HOLD_SECONDS = 10 * 60

CIRCUIT_BREAKER_MAX_FAILURES = 5
OPEN_FAILURE_PAUSE_THRESHOLD = 3
CIRCUIT_BREAKER_WINDOW_SECONDS = 30
CIRCUIT_BREAKER_COOLDOWN_SECONDS = 300
VOLATILITY_WINDOWS: tuple[tuple[int, Decimal, int], ...] = (
    (60, Decimal("0.03"), 5 * 60),
    (5 * 60, Decimal("0.07"), 10 * 60),
    (15 * 60, Decimal("0.12"), 20 * 60),
)
VOLATILITY_SAMPLE_MAX_AGE_SECONDS = 15 * 60
FILL_COOLOFF_TIERS: tuple[int, ...] = (5 * 60, 10 * 60, 15 * 60)
NET_LOSS_TEMP_MULTIPLIER = 2
MAX_TEMP_BLACKLIST_SECONDS = 30 * 60
# Escalating cool-off for markets that repeatedly trip the depth/exit-loss guard with a severe
# estimated loss: a first severe trip gets a short 30-min cool-off (auto-recovers if it was a
# one-off spike); a second escalates to a session-long 12 h ban, so a market that keeps flashing
# big danger stops being re-armed and re-hit within the session (e.g. Anduril flashed ~$12 twice
# ~4 h apart, then took a -$27.73 fill 2 h after the 2nd flash). Reward cost ~$0 — markets that
# flash >=$10 exit-loss earn cents (all 17 over a 6-day sample earned $5.53 combined).
GUARD_TRIP_SEVERE_LOSS = Decimal("10")
GUARD_TRIP_COOLOFF_TIERS: tuple[int, ...] = (30 * 60, 12 * 60 * 60)
LOSS_NOISE_FLOOR = Decimal("0.50")
CATASTROPHIC_SINGLE_LOSS = Decimal("5")
PERSISTENT_LOSS = Decimal("3")
MIN_LOSS_ROUNDTRIPS = 3
DEEP_NET_LOSS = Decimal("7")
EVENT_EXCLUSION_SECONDS = 30 * 60
LIVE_EVENT_PREGAME_HOURS = 2
SPORTS_MATCH_PREGAME_HOURS = 2
SPORTS_MATCH_POSTGAME_HOURS = 3
ELECTION_RESULT_WINDOW_HOURS = 48
CRISIS_KEYWORDS: tuple[str, ...] = (
    "war",
    "ceasefire",
    "nuclear",
    "missile",
    "missiles",
    "airstrike",
    "invade",
    "invasion",
    "iran",
    "iranian",
    "russia",
    "russian",
    "ukraine",
    "ukrainian",
    "israel",
    "israeli",
    "gaza",
    "hamas",
    "hezbollah",
    "troops",
    "sanctions",
    "hostage",
    "hostages",
    "coup",
    "assassination",
    "assassinate",
    "martial",
    "north korea",
    "dprk",
    "pyongyang",
    "kim jong",
    "khamenei",
    "tehran",
    "venezuela",
    "venezuelan",
    "maduro",
    "putin",
    "kremlin",
    "cuba",
    "cuban",
    "hormuz",
    "sanction",
)
TIME_WINDOW_DAYS: dict[str, float] = {"12h": 0.5, "1d": 1, "7d": 7, "30d": 30}

POLYGON_CHAIN_ID = 137
PUSDC_CONTRACT = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"
PUSDC_DECIMALS = 6
V2_EXCHANGE = "0xE111180000d2663C0091e4f400237545B87B996B"
BALANCE_OF_SELECTOR = "70a08231"
ALLOWANCE_SELECTOR = "dd62ed3e"
APPROVE_SELECTOR = "095ea7b3"
MAX_UINT256 = "f" * 64
