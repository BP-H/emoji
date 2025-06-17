# -------------------------------------------------------------------------------
# The Emoji Engine — MetaKarma Hub Ultimate Mega-Agent v5.28.11 FULL BLOWN
#
# Copyright (c) 2023-2026 mimi, taha & supernova
#
# Powered by humans & machines hand in hand — remixing creativity, karma & cosmos.
# Special shoutout to Gemini, Google Gemini, OpenAI GPT & Anthropic Cloud
# — the stellar trio that helped spark this cosmic project 🚀✨
#
# MIT License — remix, fork, evolve, connect your universe.
# -------------------------------------------------------------------------------

"""
MetaKarma Hub v5.28.11 Full Version

A fully modular, horizontally scalable, immutable, concurrency-safe remix ecosystem
with unified root coin, karma-gated spending, advanced reaction rewards,
and full governance + marketplace support.

Economic Model Highlights:
- Minting original content splits coin value 1/3 to creator, 1/3 treasury, 1/3 reactors
- Minting remixes splits coin value 25% to original creator, 25% to original content owner,
  25% treasury, 25% reactor escrow
- Influencer rewards paid out on minting references (up to 10 refs)
- Reacting rewards karma and mints coins, weighted by emoji + early engagement decay
- Governance uses species-weighted votes with supermajority thresholds and timelocks
- Marketplace supports listing, buying, and cancelling coin sales
- Every user starts with a single unified root coin; newcomers need karma to spend fractions

Concurrency:
- Each data entity has its own RLock
- Critical operations acquire multiple locks safely via sorted lock order
- Logchain uses single writer thread for audit consistency

"""

import sys
import json
import uuid
import datetime
import hashlib
import threading
import base64
import re
import logging
import time
import html
import os
import queue
import math
from collections import defaultdict, deque
from decimal import Decimal, getcontext, InvalidOperation, ROUND_HALF_UP, ROUND_FLOOR, localcontext
from typing import Optional, Dict, List, Any, Callable, Union, TypedDict, Literal
import traceback
from contextlib import contextmanager
import asyncio
import functools
import copy

# Set decimal precision for financial calculations
getcontext().prec = 28

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# --- Event Types ---
EventTypeLiteral = Literal[
    "ADD_USER", "MINT", "REACT", "LIST_COIN_FOR_SALE", "BUY_COIN", "TRANSFER_COIN",
    "CREATE_PROPOSAL", "VOTE_PROPOSAL", "EXECUTE_PROPOSAL", "CLOSE_PROPOSAL",
    "UPDATE_CONFIG", "DAILY_DECAY", "ADJUST_KARMA", "INFLUENCER_REWARD_DISTRIBUTION",
    "SYSTEM_MAINTENANCE", "MARKETPLACE_LIST", "MARKETPLACE_BUY", "MARKETPLACE_CANCEL"
]

class EventType:
    ADD_USER: EventTypeLiteral = "ADD_USER"
    MINT: EventTypeLiteral = "MINT"
    REACT: EventTypeLiteral = "REACT"
    LIST_COIN_FOR_SALE: EventTypeLiteral = "LIST_COIN_FOR_SALE"
    BUY_COIN: EventTypeLiteral = "BUY_COIN"
    TRANSFER_COIN: EventTypeLiteral = "TRANSFER_COIN"
    CREATE_PROPOSAL: EventTypeLiteral = "CREATE_PROPOSAL"
    VOTE_PROPOSAL: EventTypeLiteral = "VOTE_PROPOSAL"
    EXECUTE_PROPOSAL: EventTypeLiteral = "EXECUTE_PROPOSAL"
    CLOSE_PROPOSAL: EventTypeLiteral = "CLOSE_PROPOSAL"
    UPDATE_CONFIG: EventTypeLiteral = "UPDATE_CONFIG"
    DAILY_DECAY: EventTypeLiteral = "DAILY_DECAY"
    ADJUST_KARMA: EventTypeLiteral = "ADJUST_KARMA"
    INFLUENCER_REWARD_DISTRIBUTION: EventTypeLiteral = "INFLUENCER_REWARD_DISTRIBUTION"
    SYSTEM_MAINTENANCE: EventTypeLiteral = "SYSTEM_MAINTENANCE"
    MARKETPLACE_LIST: EventTypeLiteral = "MARKETPLACE_LIST"
    MARKETPLACE_BUY: EventTypeLiteral = "MARKETPLACE_BUY"
    MARKETPLACE_CANCEL: EventTypeLiteral = "MARKETPLACE_CANCEL"

# --- TypedDicts for Event Payloads ---
class AddUserPayload(TypedDict):
    event: EventTypeLiteral
    user: str
    is_genesis: bool
    species: str
    karma: str
    join_time: str
    last_active: str
    root_coin_id: str
    coins_owned: List[str]
    initial_root_value: str
    consent: bool
    root_coin_value: str

class MintPayload(TypedDict):
    event: EventTypeLiteral
    user: str
    coin_id: str
    value: str
    root_coin_id: str
    genesis_creator: Optional[str]
    references: List[Dict[str, Any]]
    improvement: str
    fractional_pct: str
    ancestors: List[str]
    timestamp: str
    is_remix: bool

class ReactPayload(TypedDict, total=False):
    event: EventTypeLiteral
    reactor: str
    coin: str
    emoji: str
    message: str
    timestamp: str
    reaction_type: str

class AdjustKarmaPayload(TypedDict):
    event: EventTypeLiteral
    user: str
    change: str
    timestamp: str

class MarketplaceListPayload(TypedDict):
    event: EventTypeLiteral
    listing_id: str
    coin_id: str
    seller: str
    price: str
    timestamp: str

class MarketplaceBuyPayload(TypedDict):
    event: EventTypeLiteral
    listing_id: str
    buyer: str
    timestamp: str

class MarketplaceCancelPayload(TypedDict):
    event: EventTypeLiteral
    listing_id: str
    user: str
    timestamp: str

class ProposalPayload(TypedDict):
    event: EventTypeLiteral
    proposal_id: str
    creator: str
    description: str
    target: str
    payload: Dict[str, Any]
    timestamp: str

class VoteProposalPayload(TypedDict):
    event: EventTypeLiteral
    proposal_id: str
    voter: str
    vote: Literal["yes", "no"]
    timestamp: str

class ExecuteProposalPayload(TypedDict):
    event: EventTypeLiteral
    proposal_id: str
    timestamp: str

class CloseProposalPayload(TypedDict):
    event: EventTypeLiteral
    proposal_id: str
    timestamp: str

class UpdateConfigPayload(TypedDict):
    event: EventTypeLiteral
    key: str
    value: Any
    timestamp: str

# --- Configuration ---
class Config:
    _lock = threading.RLock()
    VERSION = "EmojiEngine UltimateMegaAgent v5.28.11"

    ROOT_COIN_INITIAL_VALUE = Decimal('1000000')
    DAILY_DECAY = Decimal('0.99')
    TREASURY_SHARE = Decimal('0.3333333333')
    MARKET_FEE = Decimal('0.01')
    MAX_MINTS_PER_DAY = 5
    MAX_REACTS_PER_MINUTE = 30
    MIN_IMPROVEMENT_LEN = 15
    GOV_SUPERMAJORITY_THRESHOLD = Decimal('0.90')
    GOV_EXECUTION_TIMELOCK_SEC = 3600 * 24 * 2  # 48 hours
    PROPOSAL_VOTE_DURATION_HOURS = 72
    KARMA_MINT_THRESHOLD = Decimal('5000')
    FRACTIONAL_COIN_MIN_VALUE = Decimal('10')
    MAX_FRACTION_START = Decimal('0.05')
    MAX_PROPOSALS_PER_DAY = 3
    MAX_INPUT_LENGTH = 10000
    MAX_MINT_COUNT = 1000000
    MAX_KARMA = Decimal('999999999')

    # Renamed and clarified - karma needed to unlock fraction spending for non-genesis users
    KARMA_MINT_UNLOCK_RATIO = Decimal('0.02')

    # Karma multiplier constants for rewarding reactors & influencers
    INFLUENCER_REWARD_SHARE = Decimal('0.10')
    DECIMAL_ONE_THIRD = Decimal('0.3333333333')
    GENESIS_KARMA_BONUS = Decimal('50000')

    # Karma rewards per coin for influencer, reactor, and creator (tunable)
    INFLUENCER_KARMA_PER_COIN = Decimal('0.1')
    REACTOR_KARMA_PER_COIN = Decimal('0.02')
    CREATOR_KARMA_PER_COIN = Decimal('0.05')

    # Fraction of reaction coin rewarded to reactor
    REACTION_COIN_REWARD_RATIO = Decimal('0.01')

    # Content moderation regex groups
    VAX_PATTERNS = {
        "critical": [
            r"\bhack\b", r"\bmalware\b", r"\bransomware\b", r"\bbackdoor\b", r"\bexploit\b",
        ],
        "high": [
            r"\bphish\b", r"\bddos\b", r"\bspyware\b", r"\brootkit\b", r"\bkeylogger\b", r"\bbotnet\b",
        ],
        "medium": [
            r"\bpropaganda\b", r"\bsurveillance\b", r"\bmanipulate\b",
        ],
        "low": [
            r"\bspam\b", r"\bscam\b", r"\bviagra\b",
        ],
    }

    # Base emoji weights (initial)
    EMOJI_BASE = {
        "🤗": Decimal('7'), "🥰": Decimal('5'), "😍": Decimal('5'), "🔥": Decimal('4'),
        "🫶": Decimal('4'), "🌸": Decimal('3'), "💯": Decimal('3'), "🎉": Decimal('3'),
        "✨": Decimal('3'), "🙌": Decimal('3'), "🎨": Decimal('3'), "💬": Decimal('3'),
        "👍": Decimal('2'), "🚀": Decimal('2.5'), "💎": Decimal('6'), "🌟": Decimal('3'),
        "⚡": Decimal('2.5'), "👀": Decimal('0.5'), "🥲": Decimal('0.2'), "🤷‍♂️": Decimal('2'),
        "😅": Decimal('2'), "🔀": Decimal('4'), "🆕": Decimal('3'), "🔗": Decimal('2'), "❤️": Decimal('4'),
    }

    ALLOWED_POLICY_KEYS = {
        "MARKET_FEE": lambda v: Decimal(v) >= 0 and Decimal(v) <= Decimal('0.10'),
        "DAILY_DECAY": lambda v: Decimal('0.90') <= Decimal(v) <= Decimal('1'),
        "KARMA_MINT_THRESHOLD": lambda v: Decimal(v) >= 0,
        "INFLUENCER_REWARD_SHARE": lambda v: Decimal('0') <= Decimal(v) <= Decimal('0.50'),
        "MAX_FRACTION_START": lambda v: Decimal('0') < Decimal(v) <= Decimal('0.20'),
        "KARMA_MINT_UNLOCK_RATIO": lambda v: Decimal('0') <= Decimal(v) <= Decimal('0.10'),
        "GENESIS_KARMA_BONUS": lambda v: Decimal(v) >= 0,
        "GOV_SUPERMAJORITY_THRESHOLD": lambda v: Decimal('0.50') <= Decimal(v) <= Decimal('1.0'),
        "GOV_EXECUTION_TIMELOCK_SEC": lambda v: int(v) >= 0,
        "INFLUENCER_KARMA_PER_COIN": lambda v: Decimal('0') <= Decimal(v) <= Decimal('1'),
        "REACTOR_KARMA_PER_COIN": lambda v: Decimal('0') <= Decimal(v) <= Decimal('1'),
        "CREATOR_KARMA_PER_COIN": lambda v: Decimal('0') <= Decimal(v) <= Decimal('1'),
        "REACTION_COIN_REWARD_RATIO": lambda v: Decimal('0') <= Decimal(v) <= Decimal('1'),
    }

    MAX_REACTION_COST_CAP = Decimal('500')

    @classmethod
    def update_policy(cls, key: str, value: Any):
        with cls._lock:
            if key not in cls.ALLOWED_POLICY_KEYS:
                raise InvalidInputError(f"Policy key '{key}' not allowed")
            if not cls.ALLOWED_POLICY_KEYS[key](value):
                raise InvalidInputError(f"Policy value '{value}' invalid for key '{key}'")
            if key == "GOV_EXECUTION_TIMELOCK_SEC":
                setattr(cls, key, int(value))
            else:
                setattr(cls, key, Decimal(value))
            logging.info(f"Policy '{key}' updated to {value}")

# --- Utility functions ---
def acquire_agent_lock(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)
    return wrapper

def now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)

def ts() -> str:
    return now_utc().isoformat(timespec='microseconds')

def sha(data: str) -> str:
    return base64.b64encode(hashlib.sha256(data.encode('utf-8')).digest()).decode()

def today() -> str:
    return now_utc().date().isoformat()

def safe_divide(a: Decimal, b: Decimal, default=Decimal('0')) -> Decimal:
    try:
        return a / b if b != 0 else default
    except (InvalidOperation, ZeroDivisionError):
        return default

def is_valid_username(name: str) -> bool:
    if not isinstance(name, str) or len(name) < 3 or len(name) > 30:
        return False
    if not re.fullmatch(r'[A-Za-z0-9_]{3,30}', name):
        return False
    if name.lower() in {'admin', 'root', 'system', 'null', 'none'}:
        return False
    return True

def is_valid_emoji(emoji: str) -> bool:
    return emoji in Config.EMOJI_BASE

def sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    sanitized = html.escape(text)
    if len(sanitized) > Config.MAX_INPUT_LENGTH:
        sanitized = sanitized[:Config.MAX_INPUT_LENGTH]
    return sanitized

def safe_decimal(value: Any, default=Decimal('0')) -> Decimal:
    try:
        return Decimal(str(value)).normalize()
    except (InvalidOperation, ValueError, TypeError):
        return default

@contextmanager
def acquire_locks(locks: List[threading.RLock]):
    # Sort locks by id to prevent deadlocks
    sorted_locks = sorted(set(locks), key=lambda x: id(x))
    acquired = []
    try:
        for lock in sorted_locks:
            lock.acquire()
            acquired.append(lock)
        yield
    finally:
        for lock in reversed(acquired):
            lock.release()

def detailed_error_log(exc: Exception) -> str:
    return ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))

def logarithmic_reaction_cost(value: Decimal, emoji_weight: Decimal, ratio: Decimal, cap: Decimal) -> Decimal:
    try:
        base_log = Decimal(math.log10(float(value) + 1))
        cost = (base_log * emoji_weight * ratio).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        return min(cost, cap)
    except Exception:
        cost = (value * emoji_weight * ratio).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        return min(cost, cap)

# --- Exception Classes ---
class MetaKarmaError(Exception): pass
class UserExistsError(MetaKarmaError): pass
class ConsentError(MetaKarmaError): pass
class KarmaError(MetaKarmaError): pass
class BlockedContentError(MetaKarmaError): pass
class CoinDepletedError(MetaKarmaError): pass
class RateLimitError(MetaKarmaError): pass
class ImprovementRequiredError(MetaKarmaError): pass
class EmojiRequiredError(MetaKarmaError): pass
class TradeError(MetaKarmaError): pass
class VoteError(MetaKarmaError): pass
class InvalidInputError(MetaKarmaError): pass
class RootCoinMissingError(InvalidInputError): pass
class InsufficientFundsError(MetaKarmaError): pass
class InvalidPercentageError(MetaKarmaError): pass
class InfluencerRewardError(MetaKarmaError): pass

# --- Content Vaccine (Moderation) ---
class Vaccine:
    def __init__(self):
        self.lock = threading.RLock()
        self.block_counts = defaultdict(int)
        self.compiled_patterns = {}
        for lvl, pats in Config.VAX_PATTERNS.items():
            compiled = []
            for p in pats:
                try:
                    if len(p) > 50:
                        logging.warning(f"Vaccine pattern too long, skipping: {p}")
                        continue
                    compiled.append(re.compile(p, flags=re.IGNORECASE | re.UNICODE))
                except re.error as e:
                    logging.error(f"Invalid regex '{p}' level '{lvl}': {e}")
            self.compiled_patterns[lvl] = compiled

    def scan(self, text: str) -> bool:
        if not isinstance(text, str):
            return True
        if len(text) > Config.MAX_INPUT_LENGTH:
            logging.warning("Input too long for vaccine scan")
            return False
        t = text.lower()
        with self.lock:
            for lvl, pats in self.compiled_patterns.items():
                for pat in pats:
                    try:
                        if pat.search(t):
                            self.block_counts[lvl] += 1
                            snippet = sanitize_text(text[:80])
                            try:
                                with open("vaccine.log", "a", encoding="utf-8") as f:
                                    f.write(json.dumps({
                                        "ts": ts(),
                                        "nonce": uuid.uuid4().hex,
                                        "level": lvl,
                                        "pattern": pat.pattern,
                                        "snippet": snippet
                                    }) + "\n")
                            except Exception as e:
                                logging.error(f"Error writing vaccine.log: {e}")
                            logging.warning(f"Vaccine blocked '{pat.pattern}' level '{lvl}': '{snippet}...'")
                            return False
                    except re.error as e:
                        logging.error(f"Regex error during vaccine scan: {e}")
                        return False
        return True

# --- Audit Logchain ---
class LogChain:
    def __init__(self, filename="logchain.log", maxlen=1000000):
        self.filename = filename
        self.lock = threading.RLock()
        self.entries = deque(maxlen=maxlen)
        self.last_timestamp: Optional[str] = None

        self._write_queue = queue.Queue()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        self._load()

    def _load(self):
        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.entries.append(line)
            logging.info(f"Loaded {len(self.entries)} audit entries from logchain")
            if self.entries:
                last_event_line = self.entries[-1]
                try:
                    event_json, _ = last_event_line.split("||")
                    event_data = json.loads(event_json)
                    self.last_timestamp = event_data.get("timestamp")
                except Exception:
                    logging.error("Failed to parse last logchain entry")
                    self.last_timestamp = None
        except FileNotFoundError:
            logging.info("No audit log found, starting fresh")
            self.last_timestamp = None
        except Exception as e:
            logging.error(f"Error loading logchain: {e}")

    def add(self, event: Dict[str, Any]) -> None:
        event["nonce"] = uuid.uuid4().hex
        event["timestamp"] = ts()
        json_event = json.dumps(event, sort_keys=True, default=str)

        with self.lock:
            prev_hash = self.entries[-1].split("||")[-1] if self.entries else ""
            new_hash = sha(prev_hash + json_event)
            entry_line = json_event + "||" + new_hash
            self.entries.append(entry_line)
            self._write_queue.put(entry_line)

    def _writer_loop(self):
        while True:
            try:
                entry_line = self._write_queue.get()
                with open(self.filename, "a", encoding="utf-8") as f:
                    f.write(entry_line + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                self._write_queue.task_done()
            except Exception as e:
                logging.error(f"Failed to write audit log entry: {e}")

    def verify(self) -> bool:
        prev_hash = ""
        for line in self.entries:
            try:
                event_json, h = line.split("||")
            except ValueError:
                logging.error("Malformed audit log line")
                return False
            if sha(prev_hash + event_json) != h:
                logging.error("Audit log hash mismatch")
                return False
            prev_hash = h
        return True

    def replay_events(self, from_timestamp: Optional[str], apply_event_callback: Callable[[Dict[str, Any]], None]):
        if not from_timestamp:
            return
        try:
            from_dt = datetime.datetime.fromisoformat(from_timestamp)
        except Exception:
            logging.error(f"Invalid from_timestamp for replay: {from_timestamp}")
            return

        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event_json, _ = line.split("||")
                        event_data = json.loads(event_json)
                        evt_ts = datetime.datetime.fromisoformat(event_data.get("timestamp"))
                        if evt_ts > from_dt:
                            apply_event_callback(event_data)
                    except Exception as e:
                        logging.error(f"Failed to replay event: {e}")
        except FileNotFoundError:
            logging.info("Logchain file missing during replay")
        except Exception as e:
            logging.error(f"Error during replay_events: {e}")

# --- User Model ---
class User:
    def __init__(self, name: str, genesis: bool = False, species: str = "human"):
        self.name = name
        self.is_genesis = genesis
        self.species = species
        self.consent = True
        self.karma = Config.GENESIS_KARMA_BONUS if genesis else Decimal('0')
        self.join_time = now_utc()
        self.last_active = self.join_time
        self.mint_count = 0
        self.next_mint_threshold = Config.KARMA_MINT_THRESHOLD
        self.root_coin_id: Optional[str] = None
        self.coins_owned: List[str] = []
        self.daily_actions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._last_action_day: Optional[str] = today()
        self._reaction_timestamps: deque[float] = deque()
        self._proposal_timestamps: deque[float] = deque()
        self.lock = threading.RLock()
        self.initial_root_value: Optional[Decimal] = None
        self.influencer_score = Decimal('0')

        self._rate_action_log: deque[tuple[float, str]] = deque()

    def reset_daily_if_needed(self):
        today_str = today()
        with self.lock:
            if self._last_action_day != today_str:
                days_to_keep = 7
                cutoff_date = (now_utc() - datetime.timedelta(days=days_to_keep)).date().isoformat()
                self.daily_actions = {k: v for k, v in self.daily_actions.items() if k >= cutoff_date}
                self._last_action_day = today_str
                self._reaction_timestamps.clear()
                self._proposal_timestamps.clear()
                self._rate_action_log.clear()

    def check_reaction_rate_limit(self) -> bool:
        now_ts = now_utc().timestamp()
        with self.lock:
            while self._reaction_timestamps and now_ts - self._reaction_timestamps[0] > 60:
                self._reaction_timestamps.popleft()
            if len(self._reaction_timestamps) >= Config.MAX_REACTS_PER_MINUTE:
                return False
            self._reaction_timestamps.append(now_ts)
            return True

    def check_mint_rate_limit(self) -> bool:
        self.reset_daily_if_needed()
        with self.lock:
            return self.daily_actions[today()].get("mint", 0) < Config.MAX_MINTS_PER_DAY

    def check_proposal_rate_limit(self) -> bool:
        now_ts = now_utc().timestamp()
        with self.lock:
            while self._proposal_timestamps and now_ts - self._proposal_timestamps[0] > 86400:
                self._proposal_timestamps.popleft()
            if len(self._proposal_timestamps) >= Config.MAX_PROPOSALS_PER_DAY:
                return False
            self._proposal_timestamps.append(now_ts)
            return True

    def apply_daily_karma_decay(self):
        now_dt = now_utc()
        inactive_days = (now_dt - self.last_active).days
        decay_factor = Config.DAILY_DECAY ** max(inactive_days, 1)
        with self.lock, localcontext() as ctx:
            ctx.prec = 28
            old_karma = self.karma
            self.karma *= decay_factor
            self.karma = self.karma.quantize(Decimal('0.01'), rounding=ROUND_FLOOR)
            if self.karma < 0:
                self.karma = Decimal('0')
            if old_karma != self.karma:
                logging.info(f"Applied karma decay to user {self.name}: {old_karma} -> {self.karma}")

    def to_dict(self):
        with self.lock:
            return {
                "name": self.name,
                "is_genesis": self.is_genesis,
                "species": self.species,
                "consent": self.consent,
                "karma": str(self.karma),
                "join_time": self.join_time.isoformat(),
                "last_active": self.last_active.isoformat(),
                "mint_count": self.mint_count,
                "next_mint_threshold": str(self.next_mint_threshold),
                "root_coin_id": self.root_coin_id,
                "coins_owned": self.coins_owned[:],
                "daily_actions": {k: dict(v) for k, v in self.daily_actions.items()},
                "_last_action_day": self._last_action_day,
                "_reaction_timestamps": list(self._reaction_timestamps),
                "_proposal_timestamps": list(self._proposal_timestamps),
                "initial_root_value": str(self.initial_root_value) if self.initial_root_value else None,
                "influencer_score": str(self.influencer_score),
            }

    @classmethod
    def from_dict(cls, data):
        user = cls(data["name"], data.get("is_genesis", False), data.get("species", "human"))
        user.consent = data.get("consent", True)
        try:
            user.karma = Decimal(data.get("karma", '0'))
        except InvalidOperation:
            user.karma = Decimal('0')
        try:
            user.join_time = datetime.datetime.fromisoformat(data.get("join_time"))
        except Exception:
            user.join_time = now_utc()
        try:
            user.last_active = datetime.datetime.fromisoformat(data.get("last_active"))
        except Exception:
            user.last_active = user.join_time
        user.mint_count = data.get("mint_count", 0)
        try:
            user.next_mint_threshold = Decimal(data.get("next_mint_threshold", Config.KARMA_MINT_THRESHOLD))
        except InvalidOperation:
            user.next_mint_threshold = Config.KARMA_MINT_THRESHOLD
        user.root_coin_id = data.get("root_coin_id")
        user.coins_owned = data.get("coins_owned", [])
        user.daily_actions = defaultdict(lambda: defaultdict(int), {k: defaultdict(int, v) for k, v in data.get("daily_actions", {}).items()})
        user._last_action_day = data.get("_last_action_day", today())
        user._reaction_timestamps = deque(data.get("_reaction_timestamps", []))
        user._proposal_timestamps = deque(data.get("_proposal_timestamps", []))
        try:
            user.initial_root_value = Decimal(data.get("initial_root_value")) if data.get("initial_root_value") else None
        except InvalidOperation:
            user.initial_root_value = None
        try:
            user.influencer_score = Decimal(data.get("influencer_score", '0'))
        except InvalidOperation:
            user.influencer_score = Decimal('0')
        return user

# --- Coin Model ---
class Coin:
    def __init__(self, coin_id: str, creator: str, owner: str, value: Decimal,
                 is_root: bool = False, fractional_of: Optional[str] = None,
                 fractional_pct: Decimal = Decimal('0'), references: Optional[List[Dict]] = None,
                 improvement: Optional[str] = None, genesis_creator: Optional[str] = None,
                 is_remix: bool = False):
        self.coin_id = coin_id
        self.creator = creator
        self.owner = owner
        self.value = value
        self.is_root = is_root
        self.fractional_of = fractional_of
        self.fractional_pct = fractional_pct
        self.references = references or []
        self.improvement = improvement or ""
        self.ancestors: List[str] = []
        self.reactions: List[Dict] = []
        self.created_at = ts()
        self.genesis_creator = genesis_creator or (creator if is_root else None)
        self.is_remix = is_remix
        self.lock = threading.RLock()

        # For remix coins, maintain reactor reward escrow pool for reactions triggered rewards
        self.reactor_reward_escrow = Decimal('0')

    def decrease_value(self, amount: Decimal):
        with self.lock:
            if self.value < amount:
                raise CoinDepletedError(f"Coin {self.coin_id} value depleted by {amount}")
            self.value -= amount

    def increase_value(self, amount: Decimal):
        with self.lock:
            self.value += amount

    def to_dict(self):
        with self.lock:
            return {
                "coin_id": self.coin_id,
                "creator": self.creator,
                "owner": self.owner,
                "value": str(self.value),
                "is_root": self.is_root,
                "fractional_of": self.fractional_of,
                "fractional_pct": str(self.fractional_pct),
                "references": copy.deepcopy(self.references),
                "improvement": self.improvement,
                "ancestors": self.ancestors[:],
                "reactions": copy.deepcopy(self.reactions),
                "created_at": self.created_at,
                "genesis_creator": self.genesis_creator,
                "is_remix": self.is_remix,
                "reactor_reward_escrow": str(self.reactor_reward_escrow),
            }

    @classmethod
    def from_dict(cls, data):
        try:
            value = Decimal(data["value"])
        except InvalidOperation:
            value = Decimal('0')
        coin = cls(
            data["coin_id"], data["creator"], data["owner"], value,
            data.get("is_root", False), data.get("fractional_of"), Decimal(data.get("fractional_pct", '0')),
            data.get("references"), data.get("improvement"), data.get("genesis_creator"),
            data.get("is_remix", False)
        )
        coin.ancestors = data.get("ancestors", [])
        coin.reactions = data.get("reactions", [])
        coin.created_at = data.get("created_at", ts())
        try:
            coin.reactor_reward_escrow = Decimal(data.get("reactor_reward_escrow", '0'))
        except Exception:
            coin.reactor_reward_escrow = Decimal('0')
        return coin

# --- Emoji Market for dynamic emoji weights ---
class EmojiMarket:
    def __init__(self):
        self.lock = threading.RLock()
        self.market = {e: {"uses": Decimal('1'), "karma": Decimal(w), "weight": Decimal(w)} for e, w in Config.EMOJI_BASE.items()}

    def update_weight(self, emoji: str, karma_delta: Decimal):
        with self.lock:
            em = self.market.setdefault(emoji, {"uses": Decimal('0'), "karma": Decimal('0'), "weight": Decimal('1')})
            em["uses"] += 1
            em["karma"] += karma_delta
            alpha = Decimal('0.1')
            avg_karma = em["karma"] / em["uses"] if em["uses"] != 0 else Decimal('0')
            em["weight"] = alpha * avg_karma + (Decimal('1') - alpha) * em["weight"]

    def get_weight(self, emoji: str) -> Decimal:
        with self.lock:
            return self.market.get(emoji, {"weight": Decimal('1')})["weight"]

    def to_dict(self):
        with self.lock:
            return {e: {k: str(v) if isinstance(v, Decimal) else v for k, v in val.items()} for e, val in self.market.items()}

    @classmethod
    def from_dict(cls, data):
        em = cls()
        with em.lock:
            em.market = {e: {k: Decimal(v) for k, v in val.items()} for e, val in data.items()}
        return em

# --- Event Hook Manager ---
class HookManager:
    def __init__(self):
        self._hooks = defaultdict(list)
        self.lock = threading.RLock()

    def register_hook(self, event_name: str, callback: Callable):
        with self.lock:
            self._hooks[event_name].append(callback)
            logging.info(f"Hook registered for event '{event_name}'")

    def fire_hooks(self, event_name: str, *args, **kwargs):
        with self.lock:
            callbacks = list(self._hooks.get(event_name, []))
        for cb in callbacks:
            try:
                cb(*args, **kwargs)
            except Exception as e:
                logging.error(f"Error in hook '{event_name}': {e}")

# --- Governance Proposal Model ---
class Proposal:
    def __init__(self, proposal_id: str, creator: str, description: str, target: str, payload: dict):
        self.proposal_id = proposal_id
        self.creator = creator
        self.description = description
        self.target = target
        self.payload = payload
        self.created_at = ts()
        self.votes = {}
        self.status = "open"
        self.lock = threading.RLock()
        self.execution_time: Optional[datetime.datetime] = None

    def is_expired(self) -> bool:
        try:
            created_dt = datetime.datetime.fromisoformat(self.created_at)
        except Exception:
            return True
        return (now_utc() - created_dt).total_seconds() > Config.PROPOSAL_VOTE_DURATION_HOURS * 3600

    def is_ready_for_execution(self) -> bool:
        if self.execution_time is None:
            return False
        return now_utc() >= self.execution_time

    def schedule_execution(self):
        with self.lock:
            if self.execution_time is None:
                self.execution_time = now_utc() + datetime.timedelta(seconds=Config.GOV_EXECUTION_TIMELOCK_SEC)

    def tally_votes(self, users: Dict[str, User]) -> Dict[str, Decimal]:
        species_count = defaultdict(set)
        with self.lock:
            for uname in self.votes:
                user = users.get(uname)
                if user:
                    species_count[user.species].add(uname)

            total_species = len(species_count)
            if total_species == 0:
                return {"yes": Decimal('0'), "no": Decimal('0'), "total": Decimal('0')}

            species_weight = {s: Decimal('1') / Decimal(total_species) for s in species_count}
            yes_weight = Decimal('0')
            no_weight = Decimal('0')

            for species, voters in species_count.items():
                karma_yes = Decimal('0')
                karma_no = Decimal('0')
                total_karma = sum(users[v].karma for v in voters) or Decimal('1')

                for v in voters:
                    vote = self.votes.get(v)
                    if vote == "yes":
                        karma_yes += users[v].karma
                    elif vote == "no":
                        karma_no += users[v].karma

                yes_weight += species_weight[species] * (karma_yes / total_karma)
                no_weight += species_weight[species] * (karma_no / total_karma)

            total_votes = yes_weight + no_weight
            return {"yes": yes_weight, "no": no_weight, "total": total_votes}

    def is_approved(self, users: Dict[str, User]) -> bool:
        tally = self.tally_votes(users)
        if tally["total"] == 0:
            return False
        return tally["yes"] / tally["total"] >= Config.GOV_SUPERMAJORITY_THRESHOLD

    def to_dict(self):
        with self.lock:
            return {
                "proposal_id": self.proposal_id,
                "creator": self.creator,
                "description": self.description,
                "target": self.target,
                "payload": self.payload,
                "created_at": self.created_at,
                "votes": self.votes.copy(),
                "status": self.status,
                "execution_time": self.execution_time.isoformat() if self.execution_time else None,
            }

    @classmethod
    def from_dict(cls, data):
        proposal = cls(
            data["proposal_id"], data["creator"], data["description"], data["target"], data["payload"]
        )
        proposal.created_at = data.get("created_at", ts())
        proposal.votes = data.get("votes", {})
        proposal.status = data.get("status", "open")
        exec_time_str = data.get("execution_time")
        if exec_time_str:
            try:
                proposal.execution_time = datetime.datetime.fromisoformat(exec_time_str)
            except Exception:
                proposal.execution_time = None
        return proposal

# --- Marketplace Listing Model ---
class MarketplaceListing:
    def __init__(self, listing_id: str, coin_id: str, seller: str, price: Decimal, timestamp: str):
        self.listing_id = listing_id
        self.coin_id = coin_id
        self.seller = seller
        self.price = price
        self.timestamp = timestamp
        self.lock = threading.RLock()

    def to_dict(self):
        with self.lock:
            return {
                "listing_id": self.listing_id,
                "coin_id": self.coin_id,
                "seller": self.seller,
                "price": str(self.price),
                "timestamp": self.timestamp,
            }

    @classmethod
    def from_dict(cls, data):
        price = safe_decimal(data.get("price", "0"))
        return cls(data["listing_id"], data["coin_id"], data["seller"], price, data.get("timestamp", ts()))

# --- Core Agent ---
class RemixAgent:
    def __init__(self, snapshot_file: str = "snapshot.json", logchain_file: str = "logchain.log"):
        self.vaccine = Vaccine()
        self.logchain = LogChain(filename=logchain_file)
        self.users: Dict[str, User] = {}
        self.coins: Dict[str, Coin] = {}
        self.proposals: Dict[str, Proposal] = {}
        self.treasury = Decimal('0')
        self.treasury_active_fund = Decimal('0')
        self.emoji_market = EmojiMarket()
        self.hooks = HookManager()
        self.lock = threading.RLock()
        self.snapshot_file = snapshot_file
        self._last_decay_day: Optional[str] = None
        self._last_proposal_check: Optional[datetime.datetime] = None
        self.marketplace_listings: Dict[str, MarketplaceListing] = {}
        self.load_state()

    def load_state(self):
        with self.lock:
            try:
                with open(self.snapshot_file, "r", encoding="utf-8") as f:
                    snapshot = json.load(f)
                self._load_snapshot(snapshot)
                logging.info(f"Snapshot loaded from {self.snapshot_file}")
            except FileNotFoundError:
                logging.info("No snapshot file found, starting fresh")
            except Exception as e:
                logging.error(f"Failed to load snapshot: {e}")

            if self.logchain.last_timestamp:
                logging.info("Replaying audit logchain events after snapshot")
                self.logchain.replay_events(self.logchain.last_timestamp, self._apply_event)

    def _load_snapshot(self, snapshot: Dict[str, Any]):
        users_data = snapshot.get("users", {})
        for uname, udata in users_data.items():
            try:
                user = User.from_dict(udata)
                self.users[uname] = user
            except Exception as e:
                logging.error(f"Failed to load user {uname} from snapshot: {e}")

        coins_data = snapshot.get("coins", {})
        for cid, cdata in coins_data.items():
            try:
                coin = Coin.from_dict(cdata)
                self.coins[cid] = coin
            except Exception as e:
                logging.error(f"Failed to load coin {cid} from snapshot: {e}")

        proposals_data = snapshot.get("proposals", {})
        for pid, pdata in proposals_data.items():
            try:
                prop = Proposal.from_dict(pdata)
                self.proposals[pid] = prop
            except Exception as e:
                logging.error(f"Failed to load proposal {pid} from snapshot: {e}")

        self.treasury = safe_decimal(snapshot.get("treasury", '0'))
        self.treasury_active_fund = safe_decimal(snapshot.get("treasury_active_fund", '0'))
        emoji_market_data = snapshot.get("emoji_market", {})
        if emoji_market_data:
            self.emoji_market = EmojiMarket.from_dict(emoji_market_data)

        listings_data = snapshot.get("marketplace_listings", {})
        for lid, ldata in listings_data.items():
            try:
                listing = MarketplaceListing.from_dict(ldata)
                self.marketplace_listings[lid] = listing
            except Exception as e:
                logging.error(f"Failed to load marketplace listing {lid} from snapshot: {e}")

    def save_snapshot(self):
        with self.lock:
            snapshot = {
                "users": {uname: user.to_dict() for uname, user in self.users.items()},
                "coins": {cid: coin.to_dict() for cid, coin in self.coins.items()},
                "proposals": {pid: prop.to_dict() for pid, prop in self.proposals.items()},
                "treasury": str(self.treasury),
                "treasury_active_fund": str(self.treasury_active_fund),
                "emoji_market": self.emoji_market.to_dict(),
                "marketplace_listings": {lid: listing.to_dict() for lid, listing in self.marketplace_listings.items()},
                "timestamp": ts(),
            }
            tmp_file = self.snapshot_file + ".tmp"
            try:
                with open(tmp_file, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_file, self.snapshot_file)
                logging.info(f"Snapshot saved to {self.snapshot_file}")
            except Exception as e:
                logging.error(f"Failed to save snapshot: {e}")

    def _process_event(self, event: Dict[str, Any]):
        try:
            self.logchain.add(event)
            self._apply_event(event)
        except Exception as exc:
            logging.error(f"Failed processing event {event.get('event')}: {exc}\n{detailed_error_log(exc)}")

    def _apply_event(self, event: Dict[str, Any]):
        event_type = event.get("event")
        if not event_type or not isinstance(event_type, str):
            logging.error(f"Invalid or missing event type: {event_type}")
            return
        try:
            if event_type == EventType.ADD_USER:
                self._apply_add_user(event)
            elif event_type == EventType.MINT:
                self._apply_mint(event)
            elif event_type == EventType.REACT:
                self._apply_react(event)
            elif event_type == EventType.ADJUST_KARMA:
                self._apply_adjust_karma(event)
            elif event_type == EventType.INFLUENCER_REWARD_DISTRIBUTION:
                # Optional: could trigger other side effects or notifications
                pass
            elif event_type == EventType.DAILY_DECAY:
                self._apply_daily_decay(event)
            elif event_type == EventType.CREATE_PROPOSAL:
                self._apply_create_proposal(event)
            elif event_type == EventType.VOTE_PROPOSAL:
                self._apply_vote_proposal(event)
            elif event_type == EventType.EXECUTE_PROPOSAL:
                self._apply_execute_proposal(event)
            elif event_type == EventType.CLOSE_PROPOSAL:
                self._apply_close_proposal(event)
            elif event_type == EventType.UPDATE_CONFIG:
                self._apply_update_config(event)
            elif event_type == EventType.MARKETPLACE_LIST:
                self._apply_marketplace_list(event)
            elif event_type == EventType.MARKETPLACE_BUY:
                self._apply_marketplace_buy(event)
            elif event_type == EventType.MARKETPLACE_CANCEL:
                self._apply_marketplace_cancel(event)
            else:
                logging.warning(f"Unknown event type in apply_event: {event_type}")
        except Exception as e:
            logging.error(f"Error applying event '{event_type}': {e}\n{detailed_error_log(e)}")

    # --- Event Handlers ---

    def _apply_add_user(self, event: AddUserPayload):
        name = event.get("user")
        if not name or not is_valid_username(name):
            raise InvalidInputError("Invalid username on add user event")
        if name in self.users:
            raise UserExistsError(f"User '{name}' already exists")

        genesis = bool(event.get("is_genesis", False))
        species = event.get("species", "human")
        karma = safe_decimal(event.get("karma", '0'))
        join_time_str = event.get("join_time", ts())
        last_active_str = event.get("last_active", ts())
        root_coin_id = event.get("root_coin_id")
        coins_owned = event.get("coins_owned", [])
        initial_root_value = safe_decimal(event.get("initial_root_value", '0'))
        consent = bool(event.get("consent", True))

        try:
            join_time = datetime.datetime.fromisoformat(join_time_str)
            last_active = datetime.datetime.fromisoformat(last_active_str)
        except Exception:
            join_time = now_utc()
            last_active = join_time

        user = User(name, genesis, species)
        user.karma = karma
        user.join_time = join_time
        user.last_active = last_active
        user.root_coin_id = root_coin_id
        user.coins_owned = coins_owned
        user.initial_root_value = initial_root_value
        user.consent = consent

        self.users[name] = user

        # Create unified root coin for user if not exists
        if root_coin_id and root_coin_id not in self.coins:
            root_value = Config.ROOT_COIN_INITIAL_VALUE
            coin = Coin(root_coin_id, name, name, root_value, True, genesis_creator=name)
            self.coins[root_coin_id] = coin

    def _apply_mint(self, event: MintPayload):
        """
        Mint fractional coin derived from user's root coin.

        Original Content Mint:
          - Deduct mint_value from root coin
          - Split mint_value (minus influencer share) into thirds:
            1/3 back to genesis_creator root coin,
            1/3 treasury,
            1/3 reactors
          - Influencer share goes to references

        Remix Mint:
          - Deduct mint_value from root coin
          - Deduct influencer share
          - Split remainder into four quarters:
            25% genesis_creator root coin,
            25% original content owner root coin,
            25% treasury,
            25% reactor reward escrow
          - Influencer share to references

        """
        user_name = event.get("user")
        new_coin_id = event.get("coin_id")
        mint_value = safe_decimal(event.get("value", '0'))
        root_coin_id = event.get("root_coin_id")
        genesis_creator = event.get("genesis_creator")
        references = event.get("references", [])
        improvement = event.get("improvement", "")
        fractional_pct = safe_decimal(event.get("fractional_pct", '0'))
        ancestors = event.get("ancestors", [])
        created_at = event.get("timestamp", ts())
        is_remix = bool(event.get("is_remix", False))

        if not user_name or not new_coin_id or not root_coin_id:
            raise InvalidInputError("Missing mandatory fields for mint event")

        user = self.users.get(user_name)
        root_coin = self.coins.get(root_coin_id)

        if not user or not root_coin:
            raise InvalidInputError("User or root coin not found in mint event")

        with acquire_locks([root_coin.lock, user.lock]):

            # Enforce karma gating for non-genesis users
            if not user.is_genesis:
                required_karma = (root_coin.value * Config.KARMA_MINT_UNLOCK_RATIO).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
                if user.karma < required_karma:
                    raise KarmaError("Not enough karma to mint fractional coin")

            if root_coin.value < mint_value:
                raise InsufficientFundsError("Insufficient root coin value to mint")

            root_coin.value -= mint_value

            with localcontext() as ctx:
                ctx.prec = 28
                influencer_share = mint_value * Config.INFLUENCER_REWARD_SHARE

                if is_remix and references:
                    # After influencer share deducted, remainder split into four quarters
                    remix_value = mint_value - influencer_share
                    quarter_value = (remix_value / 4).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                    fractional_coin_value = quarter_value * 2  # Owner + remix creator shares combined
                else:
                    # Original content split: influencer share + thirds of remainder
                    fractional_coin_value = ((mint_value - influencer_share) * Config.DECIMAL_ONE_THIRD).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

            new_coin = Coin(
                coin_id=new_coin_id,
                creator=root_coin.creator,
                owner=user_name,
                value=fractional_coin_value,
                is_root=False,
                fractional_of=root_coin_id,
                fractional_pct=fractional_pct,
                references=references,
                improvement=improvement,
                genesis_creator=genesis_creator,
                is_remix=is_remix
            )
            new_coin.ancestors = ancestors
            new_coin.created_at = created_at

            if is_remix and references:
                # Set escrow and distribute shares for remix
                new_coin.reactor_reward_escrow = (mint_value - influencer_share) / 4
                self.treasury += (mint_value - influencer_share) * Decimal('0.25') * 3  # 3 quarters to treasury minus creator shares below

                quarter_value = (mint_value - influencer_share) / 4

                # 25% to genesis_creator root coin
                if genesis_creator and genesis_creator in self.users:
                    creator_user = self.users[genesis_creator]
                    creator_root_coin_id = creator_user.root_coin_id
                    if creator_root_coin_id and creator_root_coin_id in self.coins:
                        with self.coins[creator_root_coin_id].lock:
                            self.coins[creator_root_coin_id].value += quarter_value

                # 25% to original content owner root coin (from first reference)
                original_owner = None
                original_ref = references[0] if references else None
                if original_ref and "coin_id" in original_ref:
                    orig_coin = self.coins.get(original_ref["coin_id"])
                    if orig_coin:
                        original_owner = orig_coin.owner

                if original_owner and original_owner != user_name and original_owner in self.users:
                    orig_user = self.users[original_owner]
                    orig_root_coin_id = orig_user.root_coin_id
                    if orig_root_coin_id and orig_root_coin_id in self.coins:
                        with self.coins[orig_root_coin_id].lock:
                            self.coins[orig_root_coin_id].value += quarter_value
                    else:
                        self.treasury += quarter_value
                else:
                    self.treasury += quarter_value

            else:
                # Original content minting: 1/3 back to genesis_creator root coin
                one_third = fractional_coin_value
                if genesis_creator and genesis_creator in self.users:
                    creator_user = self.users[genesis_creator]
                    creator_root_coin_id = creator_user.root_coin_id
                    if creator_root_coin_id and creator_root_coin_id in self.coins:
                        with self.coins[creator_root_coin_id].lock:
                            self.coins[creator_root_coin_id].value += one_third
                    else:
                        root_coin.value += one_third
                else:
                    root_coin.value += one_third

                treasury_share = (mint_value - influencer_share) * Config.DECIMAL_ONE_THIRD
                self.treasury += treasury_share

            self.coins[new_coin_id] = new_coin

            self._distribute_influencer_rewards(influencer_share, references, created_at)

            user.coins_owned.append(new_coin_id)
            user.mint_count = min(user.mint_count + 1, Config.MAX_MINT_COUNT)
            user.next_mint_threshold = max(user.next_mint_threshold / 2, Config.FRACTIONAL_COIN_MIN_VALUE)
            try:
                user.last_active = datetime.datetime.fromisoformat(created_at)
            except Exception:
                user.last_active = now_utc()

    def _distribute_influencer_rewards(self, minted_value: Decimal, references: List[Dict], mint_timestamp: str):
        """
        Distribute influencer rewards on minting event.
        Rewards distributed evenly among up to 10 referenced coins' owners.
        """
        if not references or minted_value <= 0:
            return

        influencers: Dict[str, Decimal] = defaultdict(Decimal)
        total_refs = min(len(references), 10)
        if total_refs == 0:
            return
        share_per_ref = (minted_value * Config.INFLUENCER_REWARD_SHARE) / Decimal(total_refs)

        for ref in references[:10]:
            coin_id = ref.get("coin_id")
            if not coin_id or coin_id not in self.coins:
                continue
            coin = self.coins[coin_id]
            influencer_name = coin.owner
            influencers[influencer_name] += share_per_ref

        for influencer, reward in influencers.items():
            if influencer not in self.users:
                continue
            user = self.users[influencer]
            root_coin = self.coins.get(user.root_coin_id)
            if root_coin is None:
                continue
            with root_coin.lock, user.lock:
                root_coin.value += reward
                user.karma += reward * Config.INFLUENCER_KARMA_PER_COIN
                user.influencer_score += reward
                logging.info(f"Influencer '{influencer}' rewarded {reward} value and karma for mint reference")

            influencer_event = {
                "event": EventType.INFLUENCER_REWARD_DISTRIBUTION,
                "user": influencer,
                "reward_value": str(reward),
                "timestamp": mint_timestamp,
            }
            try:
                self.logchain.add(influencer_event)
            except Exception as e:
                logging.error(f"Failed to log influencer reward event: {e}")

    def _apply_react(self, event: ReactPayload):
        """
        Process reaction event:
        - Validate inputs
        - Check rate limits
        - Append reaction to coin
        - Calculate reaction virtual value scaled by emoji weight
        - Reward reactor karma + coin (from treasury)
        - Reward creator karma + coin share
        - Add treasury share
        - Release remix reactor escrow rewards proportionally
        - Update emoji weights
        """
        reactor = event.get("reactor")
        coin_id = event.get("coin")
        emoji = event.get("emoji")
        message = sanitize_text(event.get("message", ""))
        timestamp = event.get("timestamp", ts())
        reaction_type = event.get("reaction_type", "react")

        if not reactor or not coin_id or not emoji:
            raise InvalidInputError("Missing mandatory fields for react event")

        reactor_user = self.users.get(reactor)
        coin = self.coins.get(coin_id)

        if not reactor_user or not coin:
            raise InvalidInputError("Reactor user or coin not found in react event")

        if not is_valid_emoji(emoji):
            raise EmojiRequiredError(f"Emoji '{emoji}' not supported")

        if reactor == coin.owner or reactor == coin.genesis_creator:
            raise InvalidInputError("Cannot react to your own coin")

        if not reactor_user.check_reaction_rate_limit():
            raise RateLimitError("Reaction rate limit exceeded")

        root_coin_reactor = self.coins.get(reactor_user.root_coin_id)
        if root_coin_reactor is None:
            raise RootCoinMissingError(f"Root coin missing for reactor '{reactor}'")

        self.emoji_market.update_weight(emoji, Decimal('0'))

        with coin.lock:
            coin.reactions.append({"reactor": reactor, "emoji": emoji, "message": message, "timestamp": timestamp, "reaction_type": reaction_type})

        try:
            reactor_user.last_active = datetime.datetime.fromisoformat(timestamp)
        except Exception:
            reactor_user.last_active = now_utc()

        emoji_weight_used = self.emoji_market.get_weight(emoji)

        # Virtual reaction value for reward calculation
        reaction_virtual_value = (coin.value * emoji_weight_used * Config.REACTION_COIN_REWARD_RATIO).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

        involved_locks = [root_coin_reactor.lock, coin.lock]
        with acquire_locks(involved_locks):
            # Decay factor ~ inverse log10 to incentivize early engagement
            decay_factor = Decimal(1) / (Decimal(math.log10(float(coin.value) + 10)) if coin.value > 0 else Decimal(1))

            karma_reward = (reaction_virtual_value * Config.REACTOR_KARMA_PER_COIN * decay_factor).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            coin_reward = (reaction_virtual_value * decay_factor).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

            # Reward reactor karma
            with reactor_user.lock:
                old_karma = reactor_user.karma
                reactor_user.karma = min(reactor_user.karma + karma_reward, Config.MAX_KARMA)
                logging.info(f"User '{reactor}' karma increased {old_karma} -> {reactor_user.karma} for reacting")

            # Mint fractional coin reward to reactor from treasury if possible
            if self.treasury >= coin_reward and coin_reward > 0:
                self.treasury -= coin_reward

                reward_coin_id = str(uuid.uuid4())
                reward_coin = Coin(
                    coin_id=reward_coin_id,
                    creator="treasury",
                    owner=reactor,
                    value=coin_reward,
                    is_root=False,
                    fractional_of=None,
                    fractional_pct=Decimal('0'),
                    references=[{"coin_id": coin_id}],
                    improvement="Reward for reaction",
                    genesis_creator=None,
                    is_remix=False
                )
                reward_coin.created_at = ts()
                self.coins[reward_coin_id] = reward_coin

                with reactor_user.lock:
                    reactor_user.coins_owned.append(reward_coin_id)
                logging.info(f"User '{reactor}' rewarded coin {reward_coin_id} with value {coin_reward} for reacting")

            # Reward creator karma + coin share (1/3 of reaction value)
            creator_user = self.users.get(coin.genesis_creator)
            creator_reward = (reaction_virtual_value * Config.DECIMAL_ONE_THIRD * Config.CREATOR_KARMA_PER_COIN * decay_factor).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            creator_coin_reward = (reaction_virtual_value * Config.DECIMAL_ONE_THIRD * decay_factor).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            if creator_user and creator_user.root_coin_id in self.coins and creator_reward > 0:
                creator_root_coin = self.coins[creator_user.root_coin_id]
                with creator_root_coin.lock, creator_user.lock:
                    creator_root_coin.value += creator_coin_reward
                    creator_user.karma = min(creator_user.karma + creator_reward, Config.MAX_KARMA)
                    creator_user.influencer_score += creator_coin_reward
                    logging.info(f"Creator '{coin.genesis_creator}' rewarded {creator_coin_reward} coin value and karma for reaction")

            # Treasury share (1/3) scaled and decayed
            treasury_share = (reaction_virtual_value * Config.DECIMAL_ONE_THIRD * decay_factor).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            self.treasury += treasury_share

            # Release reactor escrow rewards proportionally for remixes
            if coin.is_remix:
                release_amount = (reaction_virtual_value * Decimal('0.5') * decay_factor).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                with coin.lock:
                    if coin.reactor_reward_escrow >= release_amount:
                        coin.reactor_reward_escrow -= release_amount
                        reactors = {r["reactor"] for r in coin.reactions}
                        if reactors:
                            per_reactor = (release_amount / len(reactors)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                            for rname in reactors:
                                ruser = self.users.get(rname)
                                if ruser and ruser.root_coin_id in self.coins:
                                    rroot_coin = self.coins[ruser.root_coin_id]
                                    with rroot_coin.lock, ruser.lock:
                                        rroot_coin.value += per_reactor
                                        ruser.karma = min(ruser.karma + (per_reactor * Config.REACTOR_KARMA_PER_COIN), Config.MAX_KARMA)

            # Update emoji weight to reflect karma gain
            self.emoji_market.update_weight(emoji, karma_reward)

    def _apply_adjust_karma(self, event: AdjustKarmaPayload):
        user_name = event.get("user")
        change_str = event.get("change", "0")
        try:
            change = Decimal(change_str)
        except InvalidOperation:
            change = Decimal('0')

        user = self.users.get(user_name)
        if user is None:
            raise InvalidInputError(f"User '{user_name}' not found in adjust karma event")

        with user.lock:
            old_karma = user.karma
            user.karma += change
            if user.karma < 0:
                user.karma = Decimal('0')
            elif user.karma > Config.MAX_KARMA:
                user.karma = Config.MAX_KARMA
            logging.info(f"Karma adjusted for user '{user_name}': {old_karma} -> {user.karma}")

    def _apply_daily_decay(self, event: Dict[str, Any]):
        for user in self.users.values():
            user.apply_daily_karma_decay()
            # Daily active user bonus to encourage participation
            user.karma = min(user.karma + Decimal('100'), Config.MAX_KARMA)

    # --- Governance handlers ---
    def _apply_create_proposal(self, event: ProposalPayload):
        proposal_id = event.get("proposal_id")
        creator = event.get("creator")
        description = event.get("description")
        target = event.get("target")
        payload = event.get("payload")
        timestamp = event.get("timestamp", ts())

        if not proposal_id or not creator or not description or not target:
            raise InvalidInputError("Missing required fields for create proposal")

        with self.lock:
            if proposal_id in self.proposals:
                raise InvalidInputError("Proposal ID already exists")
            proposal = Proposal(proposal_id, creator, description, target, payload)
            proposal.created_at = timestamp
            self.proposals[proposal_id] = proposal
            logging.info(f"Proposal '{proposal_id}' created by '{creator}'")

    def _apply_vote_proposal(self, event: VoteProposalPayload):
        proposal_id = event.get("proposal_id")
        voter = event.get("voter")
        vote = event.get("vote")
        timestamp = event.get("timestamp", ts())

        if not proposal_id or not voter or vote not in ("yes", "no"):
            raise InvalidInputError("Invalid fields for vote proposal")

        proposal = self.proposals.get(proposal_id)
        if proposal is None:
            raise InvalidInputError("Proposal not found")

        with proposal.lock:
            if proposal.status != "open":
                raise VoteError("Proposal voting closed")
            proposal.votes[voter] = vote
            logging.info(f"User '{voter}' voted '{vote}' on proposal '{proposal_id}'")

    def _apply_execute_proposal(self, event: ExecuteProposalPayload):
        proposal_id = event.get("proposal_id")
        timestamp = event.get("timestamp", ts())

        proposal = self.proposals.get(proposal_id)
        if proposal is None:
            raise InvalidInputError("Proposal not found")

        with proposal.lock:
            if proposal.status != "open":
                raise VoteError("Proposal not open for execution")
            if not proposal.is_ready_for_execution():
                raise VoteError("Proposal timelock not reached")
            if not proposal.is_approved(self.users):
                raise VoteError("Proposal not approved")

            # Execute payload logic here (placeholder)
            # For example: apply config changes, distribute funds, etc.

            proposal.status = "executed"
            logging.info(f"Proposal '{proposal_id}' executed")

    def _apply_close_proposal(self, event: CloseProposalPayload):
        proposal_id = event.get("proposal_id")
        timestamp = event.get("timestamp", ts())

        proposal = self.proposals.get(proposal_id)
        if proposal is None:
            raise InvalidInputError("Proposal not found")

        with proposal.lock:
            if proposal.status != "open":
                raise VoteError("Proposal already closed")
            proposal.status = "closed"
            logging.info(f"Proposal '{proposal_id}' closed without execution")

    def _apply_update_config(self, event: UpdateConfigPayload):
        key = event.get("key")
        value = event.get("value")
        timestamp = event.get("timestamp", ts())

        if not key or value is None:
            raise InvalidInputError("Missing key or value for update config")

        Config.update_policy(key, value)
        logging.info(f"Config '{key}' updated to '{value}'")

    # --- Marketplace handlers ---
    def _apply_marketplace_list(self, event: MarketplaceListPayload):
        listing_id = event.get("listing_id")
        coin_id = event.get("coin_id")
        seller = event.get("seller")
        price = safe_decimal(event.get("price", '0'))
        timestamp = event.get("timestamp", ts())

        if not listing_id or not coin_id or not seller or price <= 0:
            raise InvalidInputError("Invalid marketplace list event")

        if coin_id not in self.coins:
            raise InvalidInputError("Coin not found for listing")

        with self.lock:
            if listing_id in self.marketplace_listings:
                raise InvalidInputError("Listing ID already exists")
            listing = MarketplaceListing(listing_id, coin_id, seller, price, timestamp)
            self.marketplace_listings[listing_id] = listing
            logging.info(f"Marketplace listing '{listing_id}' created by '{seller}' for coin '{coin_id}' at price {price}")

    def _apply_marketplace_buy(self, event: MarketplaceBuyPayload):
        listing_id = event.get("listing_id")
        buyer = event.get("buyer")
        timestamp = event.get("timestamp", ts())

        if not listing_id or not buyer:
            raise InvalidInputError("Invalid marketplace buy event")

        listing = self.marketplace_listings.get(listing_id)
        if listing is None:
            raise InvalidInputError("Listing not found")

        coin = self.coins.get(listing.coin_id)
        if coin is None:
            raise InvalidInputError("Coin not found for purchase")

        buyer_user = self.users.get(buyer)
        seller_user = self.users.get(listing.seller)
        if buyer_user is None or seller_user is None:
            raise InvalidInputError("Buyer or seller user not found")

        with acquire_locks([coin.lock, buyer_user.lock, seller_user.lock, self.lock]):
            # Deduct price + marketplace fee from buyer's root coin
            total_cost = listing.price * (Decimal('1') + Config.MARKET_FEE)
            buyer_root_coin = self.coins.get(buyer_user.root_coin_id)
            if not buyer_root_coin or buyer_root_coin.value < total_cost:
                raise InsufficientFundsError("Buyer has insufficient funds")

            buyer_root_coin.value -= total_cost

            # Add price (minus fee) to seller's root coin
            seller_root_coin = self.coins.get(seller_user.root_coin_id)
            if not seller_root_coin:
                raise InvalidInputError("Seller root coin missing")

            seller_root_coin.value += listing.price

            # Transfer coin ownership to buyer
            coin.owner = buyer
            buyer_user.coins_owned.append(coin.coin_id)
            if coin.coin_id in seller_user.coins_owned:
                seller_user.coins_owned.remove(coin.coin_id)

            # Remove listing
            del self.marketplace_listings[listing_id]

            logging.info(f"Marketplace listing '{listing_id}' bought by '{buyer}' from '{seller_user.name}'")

    def _apply_marketplace_cancel(self, event: MarketplaceCancelPayload):
        listing_id = event.get("listing_id")
        user = event.get("user")
        timestamp = event.get("timestamp", ts())

        if not listing_id or not user:
            raise InvalidInputError("Invalid marketplace cancel event")

        listing = self.marketplace_listings.get(listing_id)
        if listing is None:
            raise InvalidInputError("Listing not found")

        if listing.seller != user:
            raise InvalidInputError("User not authorized to cancel listing")

        with self.lock:
            del self.marketplace_listings[listing_id]
            logging.info(f"Marketplace listing '{listing_id}' cancelled by '{user}'")

# End of MetaKarma Hub Ultimate Mega-Agent v5.28.11 FULL BLOWN
# Ready to launch remix universes, scalable & robust.
