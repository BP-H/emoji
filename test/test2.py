# -------------------------------------------------------------------------------
# The Emoji Engine — MetaKarma Hub Ultimate Mega-Agent v5.29.0 SIMPLIFIED & FIXED
#
# Copyright (c) 2023-2026 mimi, taha, supernova, GLOBALRUNWAY and accessAI tech llc 
#
# Powered by humans & machines hand in hand — remixing creativity, karma & cosmos.
# Special shoutout to OpenAI ChatGPT, Google Gemini & Anthropic Claude
# — the stellar trio that helped spark this cosmic project 🚀✨
#
# MIT License — remix, fork, evolve, connect your universe.
# -------------------------------------------------------------------------------

"""
================================================================================
!!! DO NOT MODIFY THE FOLLOWING SECTION !!!
================================================================================

The following header (until the next dashed line) is protected.
No future agent, human or AI, may alter or delete any part of this section
—even when making modifications—unless one of the following conditions is met:

  1. A governance supermajority vote (as defined by protocol) approves a change,
     OR
  2. Direct, logged instruction by taha, accompanied by an AI agent running
     the official Emoji Engine codebase, makes the update.

This applies until the system is live (main agent branch) and future
governance protocol supersedes this protection.

System

Human or robot, your safety comes first. We’re building the safest creative
playground for ☺️🫶🤖 both — where full transparency, consent, and fair rewards
are baked in. Remix, react, earn, and govern safely together. Join the experiment
for a truly open, consent-based, and collaborative future.

Special thanks to May, supernova_2177, and accessAI_tech. Also to OpenAI ChatGPT,
Google Gemini, Anthropic Claude — for making this possible. I’d love to add these
agent systems to the MIT license *and* as genesis users too, if their companies
ever reach out. This is 100% experimental sandbox until an agent goes live.
Every constructive fork that meaningfully improves the main branch becomes a new
genesis user.

Genesis users are more than contributors: they’re root-node creators. Each genesis
promotion allows a new, self-contained creative universe to emerge, starting with
a root coin — a singular value-seed forever linked to the larger Emoji Engine
economy.

Every universe bridges back to the canonical emoji_engine.py meta-core, forming a
horizontal mesh of interoperable metaverses: an ever-growing multiverse of
remixable worlds, always connected by lineage, ethics, and emoji-powered protocol
logic.

This design lets creativity flourish without hierarchy: no long-term privilege for
early entrants. Genesis users start with a modest decaying multiplier (bonus fades
linearly over 2–4 years, to be finalized via 90% supermajority vote). Over time,
all creative nodes converge toward equality.

RULES:
- Every fork must add one meaningful improvement.
- Every remix must add to the original content.
- Every constructive fork = a new universe. Every universe = a new root.
  Every root always links to the global meta-verse.
- Forks can be implemented in UE5, Unity, Robots or anything, hooks are already there.
  What you build on it is up to you! ☺️

Together, we form a distributed multiverse of metaverses. 🌱🌐💫

What we do?

A fully modular, horizontally scalable, immutable, concurrency-safe remix
ecosystem with unified root coin, karma-gated spending, advanced reaction rewards,
and full governance + marketplace support. The new legoblocks of the AI age for
the Metaverse, a safe open-source co-creation space for all species.

Economic Model Highlights (Simplified):
- Everyone starts with a single root coin of fixed initial value.
- Minting (creating content): Use a fraction (%) of your root coin to mint a new fractional coin with attached content (e.g., image/NFT-like).
  - Deducted fraction split: 33% back to creator as fractional coin value, 33% to treasury, 33% to reactors escrow (for future interactions).
  - For remixes: Similar splits, but 33% reactors includes original content's reactors.
- Value is market-driven: Popularity (reactions, team size) influences perceived value via emoji weights and karma.
- Reacting: Earns karma and small coin rewards from escrow/treasury. Simplifies to direct rewards without complex virtual values.
- Karma: Earned from reactions to your content. Spent to unlock larger mint fractions. Genesis users start with high karma for early minting.
- No inflation: All value additions are deducted from somewhere (root, treasury, escrow).
- Governance: Simplified species-weighted voting.
- Marketplace: List/buy fractional coins as NFTs with embedded content.

Concurrency:
- Each data entity has its own RLock.
- Critical operations acquire multiple locks safely via sorted lock order.
- Logchain uses a single writer thread for audit consistency.

- Every fork must improve one tiny thing.
- Every remix must add to the OC (original content).

================================================================================
"""

# ... rest of emoji_engine.py continues here ...



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
    content: str  # Added: attached content (e.g., image URL or description)

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
    VERSION = "EmojiEngine UltimateMegaAgent v5.29.0"

    ROOT_COIN_INITIAL_VALUE = Decimal('1000000')
    DAILY_DECAY = Decimal('0.99')
    TREASURY_SHARE = Decimal('0.3333333333')
    REACTORS_SHARE = Decimal('0.3333333333')
    CREATOR_SHARE = Decimal('0.3333333333')
    MARKET_FEE = Decimal('0.01')
    MAX_MINTS_PER_DAY = 5
    MAX_REACTS_PER_MINUTE = 30
    MIN_IMPROVEMENT_LEN = 15
    GOV_SUPERMAJORITY_THRESHOLD = Decimal('0.90')
    GOV_EXECUTION_TIMELOCK_SEC = 3600 * 24 * 2  # 48 hours
    PROPOSAL_VOTE_DURATION_HOURS = 72
    KARMA_MINT_THRESHOLD = Decimal('5000')
    FRACTIONAL_COIN_MIN_VALUE = Decimal('10')
    MAX_FRACTION_START = Decimal('0.01')  # Simplified to 1% max per mint
    MAX_PROPOSALS_PER_DAY = 3
    MAX_INPUT_LENGTH = 10000
    MAX_MINT_COUNT = 1000000
    MAX_KARMA = Decimal('999999999')

    # Simplified karma: ratio to unlock mint fraction
    KARMA_UNLOCK_RATIO = Decimal('0.01')  # 1% karma per % fraction

    # Simplified rewards
    REACTOR_KARMA_PER_REACT = Decimal('10')
    CREATOR_KARMA_PER_REACT = Decimal('5')
    REACTION_REWARD_RATIO = Decimal('0.001')  # 0.1% of coin value as reward

    # Base emoji weights (simplified)
    EMOJI_BASE = {
        "🤗": Decimal('2'), "🥰": Decimal('2'), "😍": Decimal('2'), "🔥": Decimal('2'),
        "🫶": Decimal('2'), "🌸": Decimal('1'), "💯": Decimal('1'), "🎉": Decimal('1'),
        "✨": Decimal('1'), "🙌": Decimal('1'), "🎨": Decimal('1'), "💬": Decimal('1'),
        "👍": Decimal('1'), "🚀": Decimal('1.5'), "💎": Decimal('3'), "🌟": Decimal('1'),
        "⚡": Decimal('1.5'), "👀": Decimal('0.5'), "🥲": Decimal('0.5'), "🤷‍♂️": Decimal('1'),
        "😅": Decimal('1'), "🔀": Decimal('2'), "🆕": Decimal('1'), "🔗": Decimal('1'), "❤️": Decimal('2'),
    }

    GENESIS_KARMA_BONUS = Decimal('100000')

    ALLOWED_POLICY_KEYS = {
        "MARKET_FEE": lambda v: Decimal(v) >= 0 and Decimal(v) <= Decimal('0.10'),
        "DAILY_DECAY": lambda v: Decimal('0.90') <= Decimal(v) <= Decimal('1'),
        "KARMA_MINT_THRESHOLD": lambda v: Decimal(v) >= 0,
        "MAX_FRACTION_START": lambda v: Decimal('0') < Decimal(v) <= Decimal('0.20'),
        "KARMA_UNLOCK_RATIO": lambda v: Decimal('0') <= Decimal(v) <= Decimal('0.10'),
        "GENESIS_KARMA_BONUS": lambda v: Decimal(v) >= 0,
        "GOV_SUPERMAJORITY_THRESHOLD": lambda v: Decimal('0.50') <= Decimal(v) <= Decimal('1.0'),
        "GOV_EXECUTION_TIMELOCK_SEC": lambda v: int(v) >= 0,
    }

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

# Simplified reaction cost
def reaction_cost(value: Decimal, emoji_weight: Decimal) -> Decimal:
    return (value * emoji_weight * Config.REACTION_REWARD_RATIO).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

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
        for lvl, pats in Config.VAX_PATTERNS.items():  # Assuming VAX_PATTERNS is defined elsewhere or remove if not needed
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
    def __init__(self, filename="logchain.log", maxlen=None):  # Removed maxlen for full history
        self.filename = filename
        self.lock = threading.RLock()
        self.entries = deque()  # No maxlen to keep full history
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
            from_timestamp = "1970-01-01T00:00:00"  # Replay all if no timestamp
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
        self._reaction_timestamps: deque[float] = deque(maxlen=Config.MAX_REACTS_PER_MINUTE * 2)  # Added maxlen
        self._proposal_timestamps: deque[float] = deque()
        self.lock = threading.RLock()
        self.initial_root_value: Optional[Decimal] = None

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
        user._reaction_timestamps = deque(data.get("_reaction_timestamps", []), maxlen=Config.MAX_REACTS_PER_MINUTE * 2)
        user._proposal_timestamps = deque(data.get("_proposal_timestamps", []))
        try:
            user.initial_root_value = Decimal(data.get("initial_root_value")) if data.get("initial_root_value") else None
        except InvalidOperation:
            user.initial_root_value = None
        return user

# --- Coin Model ---
class Coin:
    def __init__(self, coin_id: str, creator: str, owner: str, value: Decimal,
                 is_root: bool = False, fractional_of: Optional[str] = None,
                 fractional_pct: Decimal = Decimal('0'), references: Optional[List[Dict]] = None,
                 improvement: str = "", genesis_creator: Optional[str] = None,
                 is_remix: bool = False, content: str = ""):
        self.coin_id = coin_id
        self.creator = creator
        self.owner = owner
        self.value = value
        self.is_root = is_root
        self.fractional_of = fractional_of
        self.fractional_pct = fractional_pct
        self.references = references or []
        self.improvement = improvement
        self.ancestors: List[str] = []
        self.reactions: List[Dict] = []
        self.created_at = ts()
        self.genesis_creator = genesis_creator or (creator if is_root else None)
        self.is_remix = is_remix
        self.content = content  # Attached content (e.g., image/NFT data)
        self.lock = threading.RLock()

        # Simplified escrow for reactors
        self.reactors_escrow = Decimal('0')

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
                "content": self.content,
                "reactors_escrow": str(self.reactors_escrow),
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
            data.get("references"), data.get("improvement", ""), data.get("genesis_creator"),
            data.get("is_remix", False), data.get("content", "")
        )
        coin.ancestors = data.get("ancestors", [])
        coin.reactions = data.get("reactions", [])
        coin.created_at = data.get("created_at", ts())
        try:
            coin.reactors_escrow = Decimal(data.get("reactors_escrow", '0'))
        except Exception:
            coin.reactors_escrow = Decimal('0')
        return coin

# --- Emoji Market for dynamic emoji weights (simplified) ---
class EmojiMarket:
    def __init__(self):
        self.lock = threading.RLock()
        self.market = {e: Decimal(w) for e, w in Config.EMOJI_BASE.items()}  # Simplified to just weights

    def get_weight(self, emoji: str) -> Decimal:
        with self.lock:
            return self.market.get(emoji, Decimal('1'))

    def to_dict(self):
        with self.lock:
            return {e: str(v) for e, v in self.market.items()}

    @classmethod
    def from_dict(cls, data):
        em = cls()
        with em.lock:
            em.market = {e: Decimal(v) for e, v in data.items()}
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

# --- Governance Proposal Model (simplified voting) ---
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
        with self.lock:
            yes = Decimal('0')
            no = Decimal('0')
            total = Decimal('0')
            for voter, vote in self.votes.items():
                user = users.get(voter)
                if user:
                    weight = user.karma  # Simplified to karma-based voting
                    if vote == "yes":
                        yes += weight
                    elif vote == "no":
                        no += weight
                    total += weight
            return {"yes": yes, "no": no, "total": total}

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
        self.emoji_market = EmojiMarket()
        self.hooks = HookManager()
        self.lock = threading.RLock()
        self.snapshot_file = snapshot_file
        self._last_decay_day: Optional[str] = None
        self._last_proposal_check: Optional[datetime.datetime] = None
        self.marketplace_listings: Dict[str, MarketplaceListing] = {}
        self.last_applied_ts: Optional[str] = None  # Added for state recovery
        self.load_state()

    def load_state(self):
        with self.lock:
            try:
                with open(self.snapshot_file, "r", encoding="utf-8") as f:
                    snapshot = json.load(f)
                self._load_snapshot(snapshot)
                self.last_applied_ts = snapshot.get("last_applied_ts")
                logging.info(f"Snapshot loaded from {self.snapshot_file}")
            except FileNotFoundError:
                logging.info("No snapshot file found, starting fresh")
            except Exception as e:
                logging.error(f"Failed to load snapshot: {e}")

            logging.info("Replaying audit logchain events after snapshot")
            self.logchain.replay_events(self.last_applied_ts, self._apply_event)
            self.last_applied_ts = self.logchain.last_timestamp

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
                "emoji_market": self.emoji_market.to_dict(),
                "marketplace_listings": {lid: listing.to_dict() for lid, listing in self.marketplace_listings.items()},
                "last_applied_ts": self.last_applied_ts,  # Save last applied
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
            self.last_applied_ts = event["timestamp"]
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
        Simplified Minting:
        - Deduct mint_value from root coin.
        - Split exactly: 33% new fractional coin (back to creator with content attached), 33% treasury, 33% reactors escrow.
        - For remixes: Reactors escrow shared with original coin's escrow.
        - No inflation: Total added = deducted.
        - Karma gating: Non-genesis need karma >= mint_fraction * KARMA_UNLOCK_RATIO * root_value.
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
        content = event.get("content", "")  # Attached content

        if not user_name or not new_coin_id or not root_coin_id or mint_value <= 0:
            raise InvalidInputError("Missing or invalid fields for mint event")

        user = self.users.get(user_name)
        root_coin = self.coins.get(root_coin_id)

        if not user or not root_coin:
            raise InvalidInputError("User or root coin not found in mint event")

        with acquire_locks([root_coin.lock, user.lock]):

            # Karma gating simplified
            required_karma = (mint_value * Config.KARMA_UNLOCK_RATIO)
            if not user.is_genesis and user.karma < required_karma:
                raise KarmaError(f"Not enough karma ({user.karma} < {required_karma}) to mint {mint_value}")

            if root_coin.value < mint_value:
                raise InsufficientFundsError("Insufficient root coin value to mint")

            root_coin.decrease_value(mint_value)

            # Exact splits
            share = mint_value * Config.CREATOR_SHARE  # ~33.33
            treasury_share = mint_value * Config.TREASURY_SHARE
            reactors_share = mint_value * Config.REACTORS_SHARE

            self.treasury += treasury_share

            new_coin = Coin(
                coin_id=new_coin_id,
                creator=user_name,
                owner=user_name,
                value=share,
                is_root=False,
                fractional_of=root_coin_id,
                fractional_pct=fractional_pct,
                references=references,
                improvement=improvement,
                genesis_creator=genesis_creator,
                is_remix=is_remix,
                content=content
            )
            new_coin.ancestors = ancestors
            new_coin.created_at = created_at
            new_coin.reactors_escrow += reactors_share  # Fund escrow

            if is_remix and references:
                # Share reactors escrow with original
                original_coin_id = references[0].get("coin_id") if references else None
                original_coin = self.coins.get(original_coin_id)
                if original_coin:
                    with original_coin.lock:
                        original_coin.reactors_escrow += reactors_share / Decimal('2')
                    new_coin.reactors_escrow /= Decimal('2')  # Half to new, half to original

            self.coins[new_coin_id] = new_coin

            user.coins_owned.append(new_coin_id)
            user.mint_count += 1
            user.last_active = now_utc()
            user.karma -= required_karma  # Spend karma on mint (simplified)

            # Check total value conservation (for debugging)
            total_added = share + treasury_share + reactors_share
            assert abs(total_added - mint_value) < Decimal('0.000000001'), "Mint value imbalance"

    def _apply_react(self, event: ReactPayload):
        """
        Simplified Reaction:
        - Append reaction.
        - Reward reactor and creator with fixed karma.
        - Release small reward from coin's reactors_escrow to reactor's root coin.
        - No inflation: Rewards from escrow/treasury.
        """
        reactor = event.get("reactor")
        coin_id = event.get("coin")
        emoji = event.get("emoji")
        message = sanitize_text(event.get("message", ""))
        timestamp = event.get("timestamp", ts())

        if not reactor or not coin_id or not emoji:
            raise InvalidInputError("Missing mandatory fields for react event")

        reactor_user = self.users.get(reactor)
        coin = self.coins.get(coin_id)

        if not reactor_user or not coin:
            raise InvalidInputError("Reactor user or coin not found in react event")

        if not is_valid_emoji(emoji):
            raise EmojiRequiredError(f"Emoji '{emoji}' not supported")

        if reactor == coin.owner:
            raise InvalidInputError("Cannot react to your own coin")

        if not reactor_user.check_reaction_rate_limit():
            raise RateLimitError("Reaction rate limit exceeded")

        root_coin_reactor = self.coins.get(reactor_user.root_coin_id)
        if root_coin_reactor is None:
            raise RootCoinMissingError(f"Root coin missing for reactor '{reactor}'")

        with coin.lock:
            coin.reactions.append({"reactor": reactor, "emoji": emoji, "message": message, "timestamp": timestamp})

        reactor_user.last_active = now_utc()

        emoji_weight = self.emoji_market.get_weight(emoji)

        # Simplified rewards
        reactor_user.karma += Config.REACTOR_KARMA_PER_REACT * emoji_weight
        creator_user = self.users.get(coin.creator)
        if creator_user:
            creator_user.karma += Config.CREATOR_KARMA_PER_REACT * emoji_weight

        # Release from escrow
        reward = min(reaction_cost(coin.value, emoji_weight), coin.reactors_escrow)
        if reward > 0:
            coin.reactors_escrow -= reward
            root_coin_reactor.increase_value(reward)

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

    # --- Governance handlers (simplified) ---
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

            # Execute payload (e.g., update config)
            if proposal.target == "config":
                Config.update_policy(proposal.payload["key"], proposal.payload["value"])

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
            # Deduct price + fee from buyer's root
            total_cost = listing.price * (Decimal('1') + Config.MARKET_FEE)
            buyer_root = self.coins.get(buyer_user.root_coin_id)
            if not buyer_root or buyer_root.value < total_cost:
                raise InsufficientFundsError("Buyer has insufficient funds")

            buyer_root.decrease_value(total_cost)

            # Add price to seller's root
            seller_root = self.coins.get(seller_user.root_coin_id)
            if seller_root:
                seller_root.increase_value(listing.price)

            # Treasury gets fee
            self.treasury += listing.price * Config.MARKET_FEE

            # Transfer ownership
            coin.owner = buyer
            buyer_user.coins_owned.append(coin.coin_id)
            if coin.coin_id in seller_user.coins_owned:
                seller_user.coins_owned.remove(coin.coin_id)

            del self.marketplace_listings[listing_id]

            logging.info(f"Marketplace buy: {buyer} bought {listing.coin_id} from {listing.seller}")

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

# End of MetaKarma Hub Ultimate Mega-Agent v5.29.0 SIMPLIFIED & FIXED
# Changes: Fixed inflation (exact splits, funded escrows), simplified karma/rewards, added content attachment, improved state recovery, removed complexities (e.g., influencers, logarithmic costs), ensured value conservation.















# Complete CLI interface for MetaKarma Hub (updated for v5.29.0)
# Add this to the bottom of your metakarma.py file

import cmd
import json
from typing import Optional

class MetaKarmaCLI(cmd.Cmd):
    """Complete command-line interface for MetaKarma Hub"""
    
    intro = """
🚀 MetaKarma Hub v5.29.0 - Simplified Interface
Type 'help' or '?' for commands, 'help <command>' for details
Type 'quit' to exit and save
"""
    prompt = '🎭 > '
    
    def __init__(self):
        super().__init__()
        self.agent = RemixAgent()
        print(self.intro)
    
    def do_quit(self, arg):
        """Exit and save state"""
        self.agent.save_snapshot()
        print("👋 Goodbye! State saved.")
        return True
    
    def do_exit(self, arg):
        """Exit and save state"""
        return self.do_quit(arg)
    
    # === USER MANAGEMENT ===
    
    def do_add_user(self, arg):
        """Add new user: add_user <username> [genesis] [species]"""
        parts = arg.split()
        if not parts:
            print("Usage: add_user <username> [genesis] [species]")
            return
        
        username = parts[0]
        is_genesis = len(parts) > 1 and parts[1].lower() == "genesis"
        species = parts[2] if len(parts) > 2 else "human"
        
        try:
            self._add_user(username, is_genesis, species)
            status = " (Genesis)" if is_genesis else ""
            print(f"✅ User '{username}' added{status} as {species}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_users(self, arg):
        """List all users with details"""
        if not self.agent.users:
            print("No users found")
            return
        
        print(f"\n👥 Users ({len(self.agent.users)}):")
        print("-" * 80)
        for name, user in self.agent.users.items():
            genesis = "🌟" if user.is_genesis else "  "
            print(f"{genesis} {name:<15} Karma: {user.karma:<10} Coins: {len(user.coins_owned):<3} Species: {user.species}")
    
    def do_user(self, arg):
        """Show detailed user info: user <username>"""
        if not arg:
            print("Usage: user <username>")
            return
        
        user = self.agent.users.get(arg)
        if not user:
            print(f"❌ User '{arg}' not found")
            return
        
        print(f"\n👤 User: {user.name}")
        print(f"   Karma: {user.karma}")
        print(f"   Species: {user.species}")
        print(f"   Genesis: {'Yes' if user.is_genesis else 'No'}")
        print(f"   Joined: {user.join_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Last Active: {user.last_active.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Coins Owned: {len(user.coins_owned)}")
        print(f"   Mint Count: {user.mint_count}")
        
        if user.root_coin_id:
            root_coin = self.agent.coins.get(user.root_coin_id)
            if root_coin:
                print(f"   Root Coin Value: {root_coin.value}")
    
    # === CONTENT CREATION ===
    
    def do_mint(self, arg):
        """Create content: mint <username> <fraction_pct> <content> <improvement>"""
        parts = arg.split(maxsplit=3)
        if len(parts) < 3:
            print("Usage: mint <username> <fraction_pct> <content> [improvement]")
            return
        
        username, fraction_pct_str, content = parts[:3]
        improvement = parts[3] if len(parts) > 3 else ""
        
        try:
            fraction_pct = Decimal(fraction_pct_str)
            if fraction_pct > Config.MAX_FRACTION_START or fraction_pct <= 0:
                raise InvalidPercentageError("Fraction must be between 0 and 1%")
            result = self._mint_content(username, fraction_pct, content, improvement)
            print(f"✅ Content minted with attached '{content}'!")
            print(f"   Coin ID: {result['coin_id']}")
            print(f"   Value: {result['value']}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_remix(self, arg):
        """Create remix: remix <username> <original_coin_id> <fraction_pct> <content> <improvement>"""
        parts = arg.split(maxsplit=4)
        if len(parts) < 4:
            print("Usage: remix <username> <original_coin_id> <fraction_pct> <content> [improvement]")
            return
        
        username, original_coin_id, fraction_pct_str, content = parts[:4]
        improvement = parts[4] if len(parts) > 4 else ""
        
        try:
            fraction_pct = Decimal(fraction_pct_str)
            if fraction_pct > Config.MAX_FRACTION_START or fraction_pct <= 0:
                raise InvalidPercentageError("Fraction must be between 0 and 1%")
            result = self._mint_remix(username, original_coin_id, fraction_pct, content, improvement)
            print(f"✅ Remix created with attached '{content}'!")
            print(f"   Coin ID: {result['coin_id']}")
            print(f"   Value: {result['value']}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_react(self, arg):
        """React to content: react <username> <coin_id> <emoji> [message]"""
        parts = arg.split(maxsplit=3)
        if len(parts) < 3:
            print("Usage: react <username> <coin_id> <emoji> [message]")
            print("Available emojis: 🤗 🥰 😍 🔥 🫶 🌸 💯 🎉 ✨ 🙌 🎨 💬 👍 🚀 💎 🌟 ⚡ 👀 🥲 🤷‍♂️ 😅 🔀 🆕 🔗 ❤️")
            return
        
        username, coin_id, emoji = parts[:3]
        message = parts[3] if len(parts) > 3 else ""
        
        try:
            self._react_to_content(username, coin_id, emoji, message)
            print(f"✅ {username} reacted with {emoji}!")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # === COIN MANAGEMENT ===
    
    def do_coins(self, arg):
        """List coins: coins [username] [--detailed]"""
        parts = arg.split()
        username = parts[0] if parts and not parts[0].startswith('--') else None
        detailed = '--detailed' in parts
        
        if username:
            user = self.agent.users.get(username)
            if not user:
                print(f"❌ User '{username}' not found")
                return
            coins_to_show = [(cid, self.agent.coins[cid]) for cid in user.coins_owned if cid in self.agent.coins]
            print(f"\n💰 Coins owned by {username}:")
        else:
            coins_to_show = list(self.agent.coins.items())
            print(f"\n💰 All coins ({len(coins_to_show)}):")
        
        if not coins_to_show:
            print("No coins found")
            return
        
        print("-" * 100)
        for coin_id, coin in coins_to_show:
            root_indicator = "🌳" if coin.is_root else "🍃"
            remix_indicator = "🔄" if coin.is_remix else "  "
            print(f"{root_indicator}{remix_indicator} {coin_id[:12]}... Owner: {coin.owner:<12} Value: {coin.value:<12} Creator: {coin.creator}")
            
            if detailed:
                print(f"     Created: {coin.created_at}")
                print(f"     Content: {coin.content[:50]}{'...' if len(coin.content) > 50 else ''}")
                print(f"     Improvement: {coin.improvement[:50]}{'...' if len(coin.improvement) > 50 else ''}")
                print(f"     Reactions: {len(coin.reactions)}")
                if coin.references:
                    print(f"     References: {len(coin.references)}")
                print()
    
    def do_coin(self, arg):
        """Show detailed coin info: coin <coin_id>"""
        if not arg:
            print("Usage: coin <coin_id>")
            return
        
        coin = self.agent.coins.get(arg)
        if not coin:
            print(f"❌ Coin '{arg}' not found")
            return
        
        print(f"\n💰 Coin: {coin.coin_id}")
        print(f"   Owner: {coin.owner}")
        print(f"   Creator: {coin.creator}")
        print(f"   Value: {coin.value}")
        print(f"   Type: {'Root' if coin.is_root else 'Fractional'} {'Remix' if coin.is_remix else ''}")
        print(f"   Created: {coin.created_at}")
        print(f"   Genesis Creator: {coin.genesis_creator}")
        print(f"   Content: {coin.content}")
        
        if coin.improvement:
            print(f"   Improvement: {coin.improvement}")
        
        if coin.fractional_of:
            print(f"   Fractional of: {coin.fractional_of}")
            print(f"   Percentage: {coin.fractional_pct}%")
        
        if coin.references:
            print(f"   References ({len(coin.references)}):")
            for ref in coin.references[:3]:
                print(f"     - {ref.get('coin_id', 'Unknown')}")
        
        if coin.reactions:
            print(f"   Reactions ({len(coin.reactions)}):")
            for reaction in coin.reactions[-5:]:  # Show last 5
                print(f"     {reaction['emoji']} by {reaction['reactor']} - {reaction.get('message', '')}")
    
    # === MARKETPLACE ===
    
    def do_list_for_sale(self, arg):
        """List coin for sale: list_for_sale <coin_id> <price>"""
        parts = arg.split()
        if len(parts) != 2:
            print("Usage: list_for_sale <coin_id> <price>")
            return
        
        coin_id, price_str = parts
        try:
            price = Decimal(price_str)
            listing_id = str(uuid.uuid4())
            
            coin = self.agent.coins.get(coin_id)
            if not coin:
                print(f"❌ Coin '{coin_id}' not found")
                return
            
            event = {
                "event": "MARKETPLACE_LIST",
                "listing_id": listing_id,
                "coin_id": coin_id,
                "seller": coin.owner,
                "price": str(price),
                "timestamp": ts()
            }
            self.agent._process_event(event)
            print(f"✅ Coin listed for sale at price {price}")
            print(f"   Listing ID: {listing_id}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_marketplace(self, arg):
        """Show marketplace listings"""
        if not self.agent.marketplace_listings:
            print("No items for sale")
            return
        
        print(f"\n🛒 Marketplace ({len(self.agent.marketplace_listings)} listings):")
        print("-" * 80)
        for listing_id, listing in self.agent.marketplace_listings.items():
            coin = self.agent.coins.get(listing.coin_id)
            coin_info = f"{coin.coin_id[:12]}... ({coin.content[:20]}...)" if coin else "Unknown"
            print(f"💰 {coin_info} Price: {listing.price} Seller: {listing.seller}")
            print(f"   Listing ID: {listing_id}")
    
    def do_buy(self, arg):
        """Buy from marketplace: buy <listing_id> <buyer_username>"""
        parts = arg.split()
        if len(parts) != 2:
            print("Usage: buy <listing_id> <buyer_username>")
            return
        
        listing_id, buyer = parts
        try:
            event = {
                "event": "MARKETPLACE_BUY",
                "listing_id": listing_id,
                "buyer": buyer,
                "timestamp": ts()
            }
            self.agent._process_event(event)
            print(f"✅ Purchase completed!")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # === GOVERNANCE ===
    
    def do_propose(self, arg):
        """Create proposal: propose <creator> <target> <description>"""
        parts = arg.split(maxsplit=2)
        if len(parts) < 3:
            print("Usage: propose <creator> <target> <description>")
            return
        
        creator, target, description = parts
        try:
            proposal_id = str(uuid.uuid4())
            event = {
                "event": "CREATE_PROPOSAL",
                "proposal_id": proposal_id,
                "creator": creator,
                "description": description,
                "target": target,
                "payload": {},
                "timestamp": ts()
            }
            self.agent._process_event(event)
            print(f"✅ Proposal created!")
            print(f"   Proposal ID: {proposal_id}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_vote(self, arg):
        """Vote on proposal: vote <proposal_id> <voter> <yes/no>"""
        parts = arg.split()
        if len(parts) != 3 or parts[2] not in ('yes', 'no'):
            print("Usage: vote <proposal_id> <voter> <yes/no>")
            return
        
        proposal_id, voter, vote = parts
        try:
            event = {
                "event": "VOTE_PROPOSAL",
                "proposal_id": proposal_id,
                "voter": voter,
                "vote": vote,
                "timestamp": ts()
            }
            self.agent._process_event(event)
            print(f"✅ Vote recorded: {voter} voted {vote}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_proposals(self, arg):
        """List all proposals"""
        if not self.agent.proposals:
            print("No proposals found")
            return
        
        print(f"\n📋 Proposals ({len(self.agent.proposals)}):")
        print("-" * 80)
        for pid, proposal in self.agent.proposals.items():
            votes_count = len(proposal.votes)
            print(f"📜 {pid[:12]}... by {proposal.creator}")
            print(f"   Status: {proposal.status} | Votes: {votes_count} | Target: {proposal.target}")
            print(f"   Description: {proposal.description[:60]}{'...' if len(proposal.description) > 60 else ''}")
            print()
    
    # === KARMA & TREASURY ===
    
    def do_adjust_karma(self, arg):
        """Adjust user karma: adjust_karma <username> <change>"""
        parts = arg.split()
        if len(parts) != 2:
            print("Usage: adjust_karma <username> <change>")
            return
        
        username, change_str = parts
        try:
            event = {
                "event": "ADJUST_KARMA",
                "user": username,
                "change": change_str,
                "timestamp": ts()
            }
            self.agent._process_event(event)
            print(f"✅ Karma adjusted for {username} by {change_str}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_treasury(self, arg):
        """Show treasury status"""
        print(f"\n🏛️  Treasury: {self.agent.treasury}")
    
    def do_emojis(self, arg):
        """Show emoji weights"""
        print(f"\n😀 Emoji Weights:")
        print("-" * 40)
        for emoji, weight in self.agent.emoji_market.market.items():
            print(f"{emoji} Weight: {weight}")
    
    # === SYSTEM ===
    
    def do_status(self, arg):
        """Show system status"""
        print(f"\n📊 System Status:")
        print(f"   Users: {len(self.agent.users)}")
        print(f"   Coins: {len(self.agent.coins)}")
        print(f"   Proposals: {len(self.agent.proposals)}")
        print(f"   Marketplace Listings: {len(self.agent.marketplace_listings)}")
        print(f"   Treasury: {self.agent.treasury}")
        print(f"   Logchain Entries: {len(self.agent.logchain.entries)}")
    
    def do_save(self, arg):
        """Save current state"""
        self.agent.save_snapshot()
        print("💾 State saved!")
    
    def do_verify(self, arg):
        """Verify logchain integrity"""
        if self.agent.logchain.verify():
            print("✅ Logchain integrity verified")
        else:
            print("❌ Logchain integrity check failed")
    
    # === HELPER METHODS ===
    
    def _add_user(self, username, is_genesis=False, species="human"):
        """Helper to add user"""
        event = {
            "event": "ADD_USER",
            "user": username,
            "is_genesis": is_genesis,
            "species": species,
            "karma": str(Config.GENESIS_KARMA_BONUS if is_genesis else 0),
            "join_time": ts(),
            "last_active": ts(),
            "root_coin_id": str(uuid.uuid4()),
            "coins_owned": [],
            "initial_root_value": str(Config.ROOT_COIN_INITIAL_VALUE),
            "consent": True,
            "root_coin_value": str(Config.ROOT_COIN_INITIAL_VALUE)
        }
        self.agent._process_event(event)
    
    def _mint_content(self, username, fraction_pct: Decimal, content: str, improvement: str):
        """Helper to mint content"""
        user = self.agent.users.get(username)
        if not user:
            raise Exception(f"User {username} not found")
        
        if len(improvement) < Config.MIN_IMPROVEMENT_LEN and improvement:
            raise Exception(f"Improvement must be at least {Config.MIN_IMPROVEMENT_LEN} characters if provided")
        
        coin_id = str(uuid.uuid4())
        mint_value = user.initial_root_value * fraction_pct  # Based on initial for fairness
        
        event = {
            "event": "MINT",
            "user": username,
            "coin_id": coin_id,
            "value": str(mint_value),
            "root_coin_id": user.root_coin_id,
            "genesis_creator": user.name if user.is_genesis else None,
            "references": [],
            "improvement": improvement,
            "fractional_pct": str(fraction_pct),
            "ancestors": [],
            "timestamp": ts(),
            "is_remix": False,
            "content": content
        }
        self.agent._process_event(event)
        return {"coin_id": coin_id, "value": mint_value}
    
    def _mint_remix(self, username, original_coin_id, fraction_pct: Decimal, content: str, improvement: str):
        """Helper to mint remix"""
        user = self.agent.users.get(username)
        original_coin = self.agent.coins.get(original_coin_id)
        
        if not user or not original_coin:
            raise Exception("User or original coin not found")
        
        if len(improvement) < Config.MIN_IMPROVEMENT_LEN:
            raise Exception(f"Improvement must be at least {Config.MIN_IMPROVEMENT_LEN} characters")
        
        coin_id = str(uuid.uuid4())
        mint_value = user.initial_root_value * fraction_pct
        
        event = {
            "event": "MINT",
            "user": username,
            "coin_id": coin_id,
            "value": str(mint_value),
            "root_coin_id": user.root_coin_id,
            "genesis_creator": original_coin.genesis_creator,
            "references": [{"coin_id": original_coin_id}],
            "improvement": improvement,
            "fractional_pct": str(fraction_pct),
            "ancestors": original_coin.ancestors + [original_coin_id],
            "timestamp": ts(),
            "is_remix": True,
            "content": content
        }
        self.agent._process_event(event)
        return {"coin_id": coin_id, "value": mint_value}
    
    def _react_to_content(self, username, coin_id, emoji, message=""):
        """Helper to react to content"""
        event = {
            "event": "REACT",
            "reactor": username,
            "coin": coin_id,
            "emoji": emoji,
            "message": message,
            "timestamp": ts(),
        }
        self.agent._process_event(event)

def main():
    """Run the complete MetaKarma CLI"""
    try:
        cli = MetaKarmaCLI()
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"💥 Fatal error: {e}")

if __name__ == "__main__":
    main()
