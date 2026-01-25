"""
Unified SofaScore Tennis Scraper

Single scraper for all tennis data needs:
- Historical matches (ATP rankings)
- Upcoming matches
- Player history updates

Usage:
    # Historical ATP data
    python -m src.scraper historical --top 50 --pages 10
    
    # Upcoming matches
    python -m src.scraper upcoming --days 7
    
    # Update specific players
    python -m src.scraper players --ids 12345,67890
"""
import sys
from pathlib import Path
import time
import random
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import argparse
import logging

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from tls_client import Session
    HAS_TLS_CLIENT = True
except ImportError:
    import httpx
    HAS_TLS_CLIENT = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

import polars as pl

from src.schema import SchemaValidator, merge_datasets

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_URL = "https://www.sofascore.com/api/v1"

class RateLimitCircuitBreaker:
    """
    Circuit breaker for API rate limiting.
    """
    def __init__(self, failure_threshold: int = 3, backoff_minutes: int = 10):
        self.failure_threshold = failure_threshold
        self.backoff_minutes = backoff_minutes
        self.failures = 0
        self.state = "closed"
        self.last_failure = None
        self._lock = Lock()

    def is_open(self) -> bool:
        with self._lock:
            if self.state == "open":
                if datetime.now() - self.last_failure > timedelta(minutes=self.backoff_minutes):
                    self.state = "half_open"
                    return False
                return True
            return False

    def record_failure(self, status_code: int):
        if status_code in [403, 429]:
            with self._lock:
                self.failures += 1
                self.last_failure = datetime.now()
                if self.failures >= self.failure_threshold:
                    self.state = "open"
                    logger.warning(f"Circuit breaker OPENED. Backing off for {self.backoff_minutes}m")
        else:
            self.record_success()

    def record_success(self):
        with self._lock:
            self.failures = 0
            self.state = "closed"


RANKING_IDS = {
    "atp_singles": 5,
    "wta_singles": 6,
}

# Rate limiting
MIN_DELAY = 0.3
MAX_DELAY = 0.8

# Data paths
DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "tennis.parquet"
CHECKPOINT_DIR = DATA_DIR / ".checkpoints"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

_thread_local = threading.local()
_request_count = 0
_count_lock = Lock()


def get_session():
    """Get thread-local HTTP session."""
    if not hasattr(_thread_local, "session"):
        if HAS_TLS_CLIENT:
            _thread_local.session = Session(client_identifier="firefox_120")
        else:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
                "Accept": "application/json",
            }
            _thread_local.session = httpx.Client(headers=headers, timeout=30)
    return _thread_local.session


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

# =============================================================================
# CHECKPOINT & STATE MANAGEMENT
# =============================================================================

class CheckpointManager:
    """Unified checkpoint management for resume support."""
    
    def __init__(self, mode: str):
        self.mode = mode
        self.checkpoint_dir = CHECKPOINT_DIR / mode
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.data_file = self.checkpoint_dir / "data.parquet"
        self.state_file = self.checkpoint_dir / "state.json"
    
    def save(self, records: List[Dict], completed: List[int], **kwargs):
        """Save checkpoint."""
        if not records:
            return
        
        df = pl.DataFrame(records)
        df.write_parquet(self.data_file, compression="snappy")
        
        state = {
            "completed": completed,
            "record_count": len(records),
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"[CHECKPOINT] Saved {len(records)} records")
    
    def load(self) -> tuple:
        """Load checkpoint if exists."""
        if not self.state_file.exists() or not self.data_file.exists():
            return [], []
        
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
            
            df = pl.read_parquet(self.data_file)
            records = df.to_dicts()
            
            logger.info(f"[RESUME] Loaded {len(records)} records from checkpoint")
            return records, state.get("completed", [])
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return [], []
    
    def clear(self):
        """Clear checkpoint files."""
        if self.data_file.exists():
            try:
                self.data_file.unlink()
            except: pass
        if self.state_file.exists():
            try:
                self.state_file.unlink()
            except: pass


STATE_FILE = DATA_DIR / "player_update_state.json"

def load_state() -> Dict:
    """Load player update state."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except: pass
    return {"last_update": {}, "player_timestamps": {}}

def save_state(state: Dict):
    """Save player update state."""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save state: {e}")

def should_update_player(player_id: int, state: Dict, force: bool = False) -> bool:
    """Check if player needs updating based on last match timestamp."""
    if force:
        return True
    
    timestamps = state.get("player_timestamps", {})
    player_state = timestamps.get(str(player_id), {})
    
    if not player_state:
        return True
    
    last_scraped = player_state.get("last_scraped")
    if not last_scraped:
        return True
    
    # Check if enough time has passed (24 hours)
    try:
        dt = datetime.fromisoformat(last_scraped)
        if datetime.now() - dt < timedelta(hours=24):
            return False
    except:
        return True
    
    return True


# =============================================================================
# API FUNCTIONS
# =============================================================================

@lru_cache(maxsize=100)
def fetch_json_cached(endpoint: str) -> Optional[str]:
    """Cached fetch for static data (returns JSON string)."""
    result = fetch_json(endpoint)
    return json.dumps(result) if result else None


def fetch_json(endpoint: str, retries: int = 2) -> Optional[Dict]:
    """Fetch JSON from SofaScore API with retries."""
    global _request_count
    
    url = f"{BASE_URL}{endpoint}" if endpoint.startswith('/') else endpoint
    session = get_session()
    
    for attempt in range(retries + 1):
        try:
            delay = random.uniform(MIN_DELAY, MAX_DELAY)
            time.sleep(delay)
            
            response = session.get(url)
            
            with _count_lock:
                _request_count += 1
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                time.sleep(3 * (attempt + 1))
            elif response.status_code == 404:
                return None
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
    
    return None


def convert_fractional(frac_str) -> Optional[float]:
    """Convert fractional odds to decimal."""
    try:
        if '/' in str(frac_str):
            num, den = map(int, str(frac_str).split('/'))
            return round(1 + (num / den), 3)
        return float(frac_str)
    except:
        return None


# =============================================================================
# DATA FETCHING
# =============================================================================

def get_active_player_ids(upcoming_df: pl.DataFrame) -> List[int]:
    """Extract all player IDs from upcoming matches."""
    ids = set()
    if "player_id" in upcoming_df.columns:
        ids.update(upcoming_df["player_id"].drop_nulls().unique().to_list())
    if "opponent_id" in upcoming_df.columns:
        ids.update(upcoming_df["opponent_id"].drop_nulls().unique().to_list())
    return list(ids)

def is_valid_event(event: Dict) -> bool:
    """
    Check if event is a valid ATP or Challenger singles match.
    Strictly filters out ITF, Exhibition, Juniors, etc.
    """
    # 1. Check tournament category
    category = event.get("tournament", {}).get("category", {})
    cat_slug = category.get("slug", "").lower()
    cat_name = category.get("name", "").lower()
    tournament_name = event.get("tournament", {}).get("name", "").lower()
    
    # Valid categories
    # ATP slugs: 'atp', 'atp-singles'
    # Challenger slugs: 'challenger', 'challenger-singles'
    is_atp = "atp" in cat_slug or "atp" in cat_name
    is_challenger = "challenger" in cat_slug or "challenger" in cat_name
    
    # Catch Grand Slams if they have weird categories
    is_slam = any(x in tournament_name for x in ["australian open", "french open", "wimbledon", "us open"])
    
    if not (is_atp or is_challenger or is_slam):
        return False
        
    # 2. Check for Doubles - multiple detection methods
    home_type = event.get("homeTeam", {}).get("type", "")
    away_type = event.get("awayTeam", {}).get("type", "")
    home_name = event.get("homeTeam", {}).get("name", "")
    away_name = event.get("awayTeam", {}).get("name", "")
    
    # Check team type
    if home_type == "doubles" or away_type == "doubles":
        return False
    
    # Check if tournament name contains "doubles"
    if "doubles" in tournament_name:
        return False
    
    # Check player names for "/" pattern (indicates doubles team)
    if "/" in home_name or "/" in away_name:
        return False
        
    # 3. Exclude Exhibitions specifically if detected (often in 'Other' or special cats)
    if "exhibition" in tournament_name:
        return False

    return True

def fetch_rankings(ranking_type: str = "atp_singles", limit: int = 100) -> List[Dict]:
    """Fetch player rankings."""
    ranking_id = RANKING_IDS.get(ranking_type, 5)
    data = fetch_json(f"/rankings/{ranking_id}")
    
    if not data or "rankingRows" not in data:
        return []
    
    players = []
    for row in data["rankingRows"][:limit]:
        team = row.get("team", {})
        players.append({
            "position": row.get("position"),
            "player_id": team.get("id"),
            "name": team.get("name"),
            "country": team.get("country", {}).get("alpha2"),
        })
    
    return players


def fetch_player_matches(player_id: int, max_pages: int = 10, existing_events: Set[int] = None) -> List[Dict]:
    """Fetch match history for a player."""
    all_matches = []
    existing_events = existing_events or set()
    
    for page in range(max_pages):
        data = fetch_json(f"/team/{player_id}/events/singles/last/{page}")
        
        if not data or "events" not in data or not data["events"]:
            break
        
        page_matches = []
        for event in data["events"]:
            event_id = event.get("id")
            if event_id and event_id not in existing_events:
                page_matches.append(event)
        
        all_matches.extend(page_matches)
        
        # Stop if all events on this page are duplicates
        if not page_matches:
            break
        
        if data.get("hasNextPage") is False:
            break
    
    return all_matches


def fetch_match_odds(event_id: int) -> Dict:
    """Fetch match odds."""
    odds_data = {}
    data = fetch_json(f"/event/{event_id}/odds/1/all")
    
    if not data or "markets" not in data:
        return odds_data
    
    for market in data.get("markets", []):
        if market.get("marketId") == 1:
            for choice in market.get("choices", []):
                name = choice.get("name", "")
                frac = choice.get("fractionalValue", "")
                decimal = convert_fractional(frac)
                
                if decimal:
                    if name == "1":
                        odds_data["odds_home"] = decimal
                    elif name == "2":
                        odds_data["odds_away"] = decimal
    
    return odds_data


def fetch_match_stats(event_id: int) -> Optional[Dict]:
    """Fetch match statistics."""
    return fetch_json(f"/event/{event_id}/statistics")


def get_scheduled_events(date_str: str) -> List[Dict]:
    """Get scheduled tennis events for a date."""
    data = fetch_json(f"/sport/tennis/scheduled-events/{date_str}")
    return data.get("events", []) if data else []


# =============================================================================
# DATA PROCESSING
# =============================================================================

def process_match(event: Dict, player_id: int, data_type: str = "historical") -> Dict:
    """Convert raw match to player-centric format."""
    home_id = event.get("homeTeam", {}).get("id")
    is_home = (home_id == player_id)
    
    home = event.get("homeTeam", {})
    away = event.get("awayTeam", {})
    home_score = event.get("homeScore", {})
    away_score = event.get("awayScore", {})
    tournament = event.get("tournament", {}).get("uniqueTournament", {})
    
    winner = event.get("winnerCode")
    player_won = (winner == 1 and is_home) or (winner == 2 and not is_home) if winner else None
    
    record = {
        "_schema_version": "2.0",
        "_data_type": data_type,
        "_scraped_at": datetime.now().isoformat(),
        
        "event_id": event.get("id"),
        "player_id": player_id,
        "opponent_id": away.get("id") if is_home else home.get("id"),
        
        "player_name": home.get("name") if is_home else away.get("name"),
        "opponent_name": away.get("name") if is_home else home.get("name"),
        
        "player_won": player_won,
        "is_home": is_home,
        
        "player_sets": home_score.get("current", 0) if is_home else away_score.get("current", 0),
        "opponent_sets": away_score.get("current", 0) if is_home else home_score.get("current", 0),
        
        "tournament_id": tournament.get("id"),
        "tournament_name": tournament.get("name"),
        "round_name": event.get("roundInfo", {}).get("name"),
        "ground_type": event.get("groundType"),
        
        "start_timestamp": event.get("startTimestamp"),
        "status": event.get("status", {}).get("type"),
    }
    
    # Add date fields
    ts = event.get("startTimestamp")
    if ts:
        dt = datetime.fromtimestamp(ts)
        record["match_date"] = dt.date().isoformat()
        record["match_year"] = dt.year
        record["match_month"] = dt.month
    
    # Add set scores
    for i in range(1, 6):
        record[f"player_set{i}"] = home_score.get(f"period{i}") if is_home else away_score.get(f"period{i}")
        record[f"opponent_set{i}"] = away_score.get(f"period{i}") if is_home else home_score.get(f"period{i}")
    
    return record


def flatten_stats(stats_data: Dict, is_home: bool) -> Dict:
    """Flatten nested statistics."""
    if not stats_data or "statistics" not in stats_data:
        return {}
    
    result = {}
    player_prefix = "home" if is_home else "away"
    opponent_prefix = "away" if is_home else "home"
    
    for period_data in stats_data.get("statistics", []):
        if period_data.get("period", "ALL") != "ALL":
            continue
        
        for group in period_data.get("groups", []):
            group_name = group.get("groupName", "").lower().replace(" ", "_")
            
            for item in group.get("statisticsItems", []):
                key = item.get("key", "").lower()
                stat_name = f"{group_name}_{key}"
                
                if f"{player_prefix}Value" in item:
                    result[f"player_{stat_name}"] = item.get(f"{player_prefix}Value")
                    result[f"opponent_{stat_name}"] = item.get(f"{opponent_prefix}Value")
    
    return result


def add_odds(record: Dict, odds_data: Dict) -> Dict:
    """Add odds and implied probabilities."""
    is_home = record.get("is_home", True)
    
    if is_home:
        record["odds_player"] = odds_data.get("odds_home")
        record["odds_opponent"] = odds_data.get("odds_away")
    else:
        record["odds_player"] = odds_data.get("odds_away")
        record["odds_opponent"] = odds_data.get("odds_home")
    
    if record.get("odds_player"):
        record["implied_prob_player"] = round(1 / record["odds_player"], 4)
    if record.get("odds_opponent"):
        record["implied_prob_opponent"] = round(1 / record["odds_opponent"], 4)
    
    record["has_odds"] = bool(record.get("odds_player"))
    
    return record


# =============================================================================
# SCRAPING MODES
# =============================================================================

def scrape_historical(
    top_players: int = 50,
    max_pages: int = 10,
    ranking_type: str = "atp_singles",
    fetch_details: bool = True,
    resume: bool = False,
    workers: int = 2
) -> pl.DataFrame:
    """
    Scrape historical match data for top-ranked players.
    """
    print(f"=== SCRAPING HISTORICAL ({ranking_type} top {top_players}) ===")
    
    # Setup checkpoint
    checkpoint = CheckpointManager("historical")
    
    # Resume or fresh start
    if resume:
        all_records, completed_players = checkpoint.load()
    else:
        checkpoint.clear()
        all_records, completed_players = [], []
    
    # Fetch rankings
    players = fetch_rankings(ranking_type, top_players)
    if not players:
        print("ERROR: Failed to fetch rankings")
        return pl.DataFrame()
    
    # Filter out completed players
    players_to_fetch = [p for p in players if p["player_id"] not in completed_players]
    print(f"Players to fetch: {len(players_to_fetch)} (skipping {len(completed_players)} completed)")
    
    # Build existing events set
    existing_events = {r["event_id"] for r in all_records}
    
    # Fetch player matches
    pbar = tqdm(players_to_fetch, desc="Scraping players") if HAS_TQDM else players_to_fetch
    
    for i, player in enumerate(pbar):
        player_id = player["player_id"]
        
        matches = fetch_player_matches(player_id, max_pages, existing_events)
        
        for event in matches:
            # STRICT FILTER: Skip non-ATP/Challenger matches immediately
            if not is_valid_event(event):
                continue

            record = process_match(event, player_id, data_type="historical")
            
            # Fetch details if enabled
            if fetch_details:
                event_id = event.get("id")
                
                # Odds
                odds = fetch_match_odds(event_id)
                record = add_odds(record, odds)
                
                # Stats
                stats = fetch_match_stats(event_id)
                if stats:
                    stat_fields = flatten_stats(stats, record.get("is_home", True))
                    record.update(stat_fields)
            
            all_records.append(record)
            existing_events.add(record["event_id"])
        
        completed_players.append(player_id)
        
        # Checkpoint every 5 players
        if (i + 1) % 5 == 0:
            checkpoint.save(all_records, completed_players)
    
    # Final save
    if all_records:
        df = pl.DataFrame(all_records)
        df = df.unique(subset=["event_id", "player_id"], keep="first")
        
        # Validate
        validator = SchemaValidator()
        result = validator.validate_raw_data(df)
        if result.errors:
            print(f"  Validation issues: {result.errors}")
        
        # Save
        df.write_parquet(OUTPUT_FILE)
        checkpoint.clear()
        
        print(f"=== COMPLETE: {len(df)} matches saved to {OUTPUT_FILE} ===")
        return df
    
    return pl.DataFrame()


def scrape_upcoming(days_ahead: int = 7, workers: int = 4) -> pl.DataFrame:
    """
    Scrape upcoming matches for the next N days.
    """
    print(f"=== SCRAPING UPCOMING ({days_ahead} days) ===")
    
    all_records = []
    odds_cache = {}
    
    # Generate dates
    today = datetime.now()
    dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days_ahead)]
    
    # Fetch events for each date
    for date_str in tqdm(dates, desc="Fetching dates") if HAS_TQDM else dates:
        events = get_scheduled_events(date_str)
        
        # Process events
        atp_events = []
        for event in events:
            # STRICT FILTER: Use unified validation logic
            if not is_valid_event(event):
                continue
            
            # Skip doubles types if not caught by filter (extra safety)
            home = event.get("homeTeam", {})
            if home.get("type") == "doubles":
                continue
            
            # Skip finished
            if event.get("status", {}).get("type") == "finished":
                continue
            
            atp_events.append(event)
        
        # Process events
        for event in atp_events:
            home = event.get("homeTeam", {})
            record = process_match(event, home.get("id"), data_type="upcoming")
            all_records.append(record)
    
    if not all_records:
        print("No upcoming matches found")
        return pl.DataFrame()
    
    # Batch fetch odds
    event_ids = [r["event_id"] for r in all_records]
    print(f"Fetching odds for {len(event_ids)} events...")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_match_odds, eid): eid for eid in event_ids}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching odds") if HAS_TQDM else as_completed(futures):
            event_id = futures[future]
            try:
                odds_cache[event_id] = future.result()
            except:
                odds_cache[event_id] = {}
    
    # Add odds to records
    for record in all_records:
        odds = odds_cache.get(record["event_id"], {})
        add_odds(record, odds)
    
    # Create DataFrame
    df = pl.DataFrame(all_records)
    df = df.unique(subset=["event_id", "player_id"], keep="first")
    
    # Save
    output_path = DATA_DIR / "upcoming.parquet"
    df.write_parquet(output_path)
    
    print(f"=== COMPLETE: {len(df)} upcoming matches saved ===")
    return df


def scrape_players(
    player_ids: List[int],
    max_pages: int = 5,
    workers: int = 3,
    smart_update: bool = False
) -> pl.DataFrame:
    """
    Scrape match history for specific players.
    
    Args:
        player_ids: List of player IDs to scrape
        max_pages: Maximum pages of history per player
        workers: Number of parallel threads
        smart_update: If True, skip players updated in last 24h
    """
    print(f"=== SCRAPING {len(player_ids)} PLAYERS (Smart={smart_update}) ===")
    
    # Load state
    state = load_state()
    
    # Filter if smart update
    if smart_update:
        original_count = len(player_ids)
        player_ids = [pid for pid in player_ids if should_update_player(pid, state)]
        skipped = original_count - len(player_ids)
        if skipped > 0:
            print(f"Skipping {skipped} recently updated players")
            
    if not player_ids:
        print("No players to update.")
        return pl.read_parquet(OUTPUT_FILE) if OUTPUT_FILE.exists() else pl.DataFrame()
    
    # Load existing data
    existing_df = None
    existing_events = set()
    
    if OUTPUT_FILE.exists():
        existing_df = pl.read_parquet(OUTPUT_FILE)
        existing_events = set(existing_df["event_id"].unique().to_list())
        print(f"Existing matches: {len(existing_df):,}")
    
    all_records = []
    
    def fetch_single_player(player_id):
        # Update timestamp immediately to allow caching even if no new matches found (prevents loops)
        if "player_timestamps" not in state:
            state["player_timestamps"] = {}
        state["player_timestamps"][str(player_id)] = {
            "last_scraped": datetime.now().isoformat()
        }
            
        records = []
        matches = fetch_player_matches(player_id, max_pages, existing_events)
        
        for event in matches:
            # STRICT FILTER: Skip non-ATP/Challenger matches immediately
            if not is_valid_event(event):
                continue

            record = process_match(event, player_id, data_type="historical")
            
            # Fetch odds
            odds = fetch_match_odds(event.get("id"))
            record = add_odds(record, odds)
            
            records.append(record)
        
        return player_id, records
    
    # Fetch in parallel
    print(f"Fetching {len(player_ids)} players with {workers} workers...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_single_player, pid): pid for pid in player_ids}
        
        pbar = tqdm(total=len(player_ids), desc="Fetching players") if HAS_TQDM else None
        
        for future in as_completed(futures):
            player_id, records = future.result()
            all_records.extend(records)
            
            if pbar:
                pbar.update(1)
                pbar.set_postfix({"new": len(all_records)})
        
        if pbar:
            pbar.close()
    
    # Save state
    save_state(state)
    
    if not all_records:
        print("No new matches found")
        return existing_df if existing_df is not None else pl.DataFrame()
    
    # Create DataFrame
    new_df = pl.DataFrame(all_records)
    
    # Merge with existing
    if existing_df is not None:
        from src.schema import merge_datasets
        df = merge_datasets(new_df, existing_df, prefer_new=True)
    else:
        df = new_df
    
    df = df.unique(subset=["event_id", "player_id"], keep="first")
    
    # Save
    df.write_parquet(OUTPUT_FILE)
    
    print(f"=== COMPLETE: {len(df)} total matches ({len(all_records)} new) ===")
    return df


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Tennis Scraper")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Historical
    hist_parser = subparsers.add_parser("historical", help="Scrape historical ATP data")
    hist_parser.add_argument("--top", type=int, default=50, help="Top N players")
    hist_parser.add_argument("--pages", type=int, default=10, help="Max pages per player")
    hist_parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    hist_parser.add_argument("--no-details", action="store_true", help="Skip odds/stats")
    
    # Upcoming
    upcoming_parser = subparsers.add_parser("upcoming", help="Scrape upcoming matches")
    upcoming_parser.add_argument("--days", type=int, default=7, help="Days ahead")
    
    # Players
    players_parser = subparsers.add_parser("players", help="Scrape specific players")
    players_parser.add_argument("--ids", required=True, help="Comma-separated player IDs")
    players_parser.add_argument("--pages", type=int, default=5, help="Max pages per player")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    if args.command == "historical":
        scrape_historical(
            top_players=args.top,
            max_pages=args.pages,
            fetch_details=not args.no_details,
            resume=args.resume
        )
    elif args.command == "upcoming":
        scrape_upcoming(days_ahead=args.days)
    elif args.command == "players":
        player_ids = [int(x.strip()) for x in args.ids.split(",")]
        scrape_players(player_ids=player_ids, max_pages=args.pages)


if __name__ == "__main__":
    main()
