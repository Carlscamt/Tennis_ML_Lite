"""
Future Matches Scraper (Optimized)

Fetches upcoming ATP matches with odds for prediction.
OPTIMIZED with parallel processing, caching, and progress bars.

Usage:
    python scripts/scrape_future.py
    python scripts/scrape_future.py --days 7
"""
import sys
from pathlib import Path
import time
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import threading

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from tls_client import Session
    HAS_TLS_CLIENT = True
except ImportError:
    import httpx
    HAS_TLS_CLIENT = False

import polars as pl
import argparse

# Optional: tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_URL = "https://www.sofascore.com/api/v1"
TENNIS_SPORT_ID = 5  # Tennis sport ID

# Rate limiting (reduced for parallel)
MIN_DELAY = 0.15
MAX_DELAY = 0.4

# Parallelism settings
MAX_WORKERS = 8  # Concurrent threads
BATCH_SIZE = 20  # Events per batch for odds fetching

# Output paths
DATA_DIR = ROOT / "data"
FUTURE_DIR = DATA_DIR / "future"
FUTURE_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoint path
CHECKPOINT_FILE = FUTURE_DIR / ".scrape_checkpoint.json"

# Thread-local storage for sessions
_thread_local = threading.local()


# =============================================================================
# CHECKPOINT MANAGER (Progressive Saving)
# =============================================================================

class CheckpointManager:
    """Manages checkpointing for resume capability."""
    
    def __init__(self, checkpoint_path: Path = CHECKPOINT_FILE):
        self.path = checkpoint_path
        self.state = self._load()
    
    def _load(self) -> Dict:
        """Load existing checkpoint."""
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    return json.load(f)
            except:
                pass
        return {"completed_dates": [], "records": [], "odds": {}}
    
    def save(self):
        """Save current state."""
        with open(self.path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def mark_date_complete(self, date_str: str, records: List[Dict]):
        """Mark a date as complete and save its records."""
        if date_str not in self.state["completed_dates"]:
            self.state["completed_dates"].append(date_str)
        self.state["records"].extend(records)
        self.save()
    
    def save_odds_batch(self, odds: Dict[int, Dict]):
        """Save a batch of odds."""
        for event_id, event_odds in odds.items():
            self.state["odds"][str(event_id)] = event_odds
        self.save()
    
    def is_date_complete(self, date_str: str) -> bool:
        """Check if a date was already processed."""
        return date_str in self.state["completed_dates"]
    
    def get_saved_records(self) -> List[Dict]:
        """Get all saved records."""
        return self.state.get("records", [])
    
    def get_saved_odds(self) -> Dict[str, Dict]:
        """Get all saved odds."""
        return self.state.get("odds", {})
    
    def clear(self):
        """Clear checkpoint after successful completion."""
        if self.path.exists():
            self.path.unlink()
        self.state = {"completed_dates": [], "records": [], "odds": {}}


# =============================================================================
# SESSION POOL (Thread-safe)
# =============================================================================

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
# LRU CACHE FOR API RESPONSES
# =============================================================================

@lru_cache(maxsize=500)
def fetch_json_cached(endpoint: str, retries: int = 2) -> Optional[str]:
    """Fetch JSON from SofaScore API with caching. Returns JSON string."""
    session = get_session()
    url = endpoint if endpoint.startswith('http') else f"{BASE_URL}{endpoint}"
    
    for attempt in range(retries + 1):
        try:
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            response = session.get(url)
            
            if response.status_code == 200:
                return response.text
            elif response.status_code == 403:
                time.sleep(2 * (attempt + 1))
            elif response.status_code == 404:
                return None
        except Exception as e:
            if attempt < retries:
                time.sleep(0.5)
    
    return None


def fetch_json(endpoint: str, retries: int = 2) -> Optional[Dict]:
    """Fetch JSON and parse it."""
    result = fetch_json_cached(endpoint, retries)
    if result:
        try:
            return json.loads(result)
        except:
            return None
    return None


# =============================================================================
# DATA FETCHING
# =============================================================================

def get_scheduled_events(date_str: str) -> List[Dict]:
    """Get all scheduled tennis events for a specific date."""
    endpoint = f"/sport/tennis/scheduled-events/{date_str}"
    data = fetch_json(endpoint)
    
    if not data or "events" not in data:
        return []
    
    return data["events"]


def get_event_odds(event_id: int) -> Dict:
    """Get pre-match odds for an event."""
    data = fetch_json(f"/event/{event_id}/odds/1/all")
    
    odds = {}
    if not data or "markets" not in data:
        return odds
    
    for market in data.get("markets", []):
        if market.get("marketId") == 1:  # Match winner
            for choice in market.get("choices", []):
                name = choice.get("name", "")
                frac = choice.get("fractionalValue", "")
                
                try:
                    if '/' in str(frac):
                        num, den = map(int, str(frac).split('/'))
                        decimal = round(1 + (num / den), 3)
                    else:
                        decimal = float(frac)
                    
                    if name == "1":
                        odds["odds_home"] = decimal
                    elif name == "2":
                        odds["odds_away"] = decimal
                except:
                    pass
    
    return odds


def get_event_odds_batch(event_ids: List[int]) -> Dict[int, Dict]:
    """Fetch odds for multiple events in parallel."""
    results = {}
    
    def fetch_single(event_id):
        return event_id, get_event_odds(event_id)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_single, eid): eid for eid in event_ids}
        
        for future in as_completed(futures):
            try:
                event_id, odds = future.result()
                results[event_id] = odds
            except:
                pass
    
    return results


def process_future_event(event: Dict) -> Optional[Dict]:
    """Process a future match event."""
    # Skip if already finished
    status = event.get("status", {}).get("type", "")
    if status == "finished":
        return None
    
    # Skip doubles
    home_team = event.get("homeTeam", {})
    away_team = event.get("awayTeam", {})
    
    if home_team.get("type") == "doubles" or away_team.get("type") == "doubles":
        return None
    
    tournament = event.get("tournament", {}).get("uniqueTournament", {})
    tournament_name = tournament.get("name", "") or ""
    
    # Get category for ATP/WTA detection
    category = event.get("tournament", {}).get("category", {})
    category_slug = category.get("slug", "").lower()
    category_name = category.get("name", "").lower()
    
    # Classify tournament type
    tournament_name_lower = tournament_name.lower()
    
    is_doubles = "doubles" in tournament_name_lower or "double" in tournament_name_lower
    is_utr = "utr " in tournament_name_lower or tournament_name_lower.startswith("utr")
    is_itf = "itf " in tournament_name_lower
    is_challenger = "challenger" in tournament_name_lower
    
    if category_slug == "wta" or "wta" in category_name:
        tournament_type = "WTA"
    elif category_slug == "atp" or "atp" in category_name:
        tournament_type = "ATP"
    elif is_doubles:
        tournament_type = "Doubles"
    elif is_utr:
        tournament_type = "UTR"
    elif is_itf:
        tournament_type = "ITF"
    elif is_challenger:
        tournament_type = "Challenger"
    elif any(x in tournament_name_lower for x in ["australian open", "french open", "wimbledon", "us open", "grand slam", "masters"]):
        tournament_type = "ATP"
    else:
        tournament_type = "Other"
    
    record = {
        "event_id": event.get("id"),
        "start_timestamp": event.get("startTimestamp"),
        
        "player_id": home_team.get("id"),
        "player_name": home_team.get("name"),
        
        "opponent_id": away_team.get("id"),
        "opponent_name": away_team.get("name"),
        
        "tournament_id": tournament.get("id"),
        "tournament_name": tournament_name,
        "tournament_type": tournament_type,
        "round_name": event.get("roundInfo", {}).get("name"),
        "ground_type": event.get("groundType"),
        
        "status": status,
    }
    
    # Add date fields
    ts = event.get("startTimestamp")
    if ts:
        dt = datetime.fromtimestamp(ts)
        record["match_date"] = dt.date().isoformat()
        record["match_time"] = dt.strftime("%H:%M")
        record["match_year"] = dt.year
        record["match_month"] = dt.month
    
    return record


def scrape_future_matches(days_ahead: int = 7, resume: bool = True) -> pl.DataFrame:
    """
    Scrape future matches for the next N days.
    OPTIMIZED with parallel odds fetching, progress bars, and checkpointing.
    
    Args:
        days_ahead: Number of days to look ahead
        resume: Whether to resume from checkpoint if available
    """
    print("=" * 60)
    print("FUTURE MATCHES SCRAPER (OPTIMIZED)")
    print("=" * 60)
    print(f"Looking ahead: {days_ahead} days")
    print(f"Max workers: {MAX_WORKERS}")
    
    # Initialize checkpoint manager
    checkpoint = CheckpointManager()
    
    all_records = []
    today = datetime.now().date()
    dates = [today + timedelta(days=i) for i in range(days_ahead)]
    
    # Check for resumed data
    if resume and checkpoint.get_saved_records():
        saved_count = len(checkpoint.get_saved_records())
        print(f"\n📌 Resuming from checkpoint ({saved_count} records found)")
    
    # Step 1: Fetch all scheduled events for all dates
    print("\n[1/3] Fetching scheduled events...")
    
    all_events = []
    date_iter = tqdm(dates, desc="Dates") if HAS_TQDM else dates
    
    for target_date in date_iter:
        date_str = target_date.isoformat()
        
        # Skip if already processed
        if resume and checkpoint.is_date_complete(date_str):
            print(f"  ⏭️  Skipping {date_str} (already processed)")
            continue
        
        events = get_scheduled_events(date_str)
        day_records = []
        
        for event in events:
            record = process_future_event(event)
            if record:
                all_events.append((record, event.get("id")))
                day_records.append(record)
        
        # Progressive save after each date
        if day_records:
            checkpoint.mark_date_complete(date_str, day_records)
    
    # Combine with saved records if resuming
    if resume:
        saved_records = checkpoint.get_saved_records()
        # Don't add duplicates
        existing_ids = {r.get("event_id") for r, _ in all_events}
        for sr in saved_records:
            if sr.get("event_id") not in existing_ids:
                all_events.append((sr, sr.get("event_id")))
    
    if not all_events:
        print("\n[WARN] No future matches found")
        return pl.DataFrame()
    
    print(f"  Found {len(all_events)} valid matches")
    
    # Step 2: Fetch odds in parallel batches
    print("\n[2/3] Fetching odds (parallel)...")
    
    event_ids = [eid for _, eid in all_events]
    
    # Get already-fetched odds from checkpoint
    saved_odds = checkpoint.get_saved_odds()
    already_fetched = set(saved_odds.keys())
    
    # Only fetch odds we don't have
    ids_to_fetch = [eid for eid in event_ids if str(eid) not in already_fetched]
    
    if ids_to_fetch:
        print(f"  Fetching odds for {len(ids_to_fetch)} events ({len(already_fetched)} cached)")
        
        # Fetch in batches with progressive save
        all_odds = dict(saved_odds)  # Start with saved
        batches = [ids_to_fetch[i:i+BATCH_SIZE] for i in range(0, len(ids_to_fetch), BATCH_SIZE)]
        
        batch_iter = tqdm(batches, desc="Odds batches") if HAS_TQDM else batches
        
        for batch in batch_iter:
            batch_odds = get_event_odds_batch(batch)
            all_odds.update({str(k): v for k, v in batch_odds.items()})
            # Progressive save after each batch
            checkpoint.save_odds_batch(batch_odds)
    else:
        print(f"  All {len(already_fetched)} odds loaded from cache")
        all_odds = saved_odds
    
    # Step 3: Combine records with odds
    print("\n[3/3] Combining data...")
    
    for record, event_id in all_events:
        odds = all_odds.get(str(event_id), {})
        if odds:
            record["odds_player"] = odds.get("odds_home")
            record["odds_opponent"] = odds.get("odds_away")
            record["has_odds"] = True
            
            if record.get("odds_player"):
                record["implied_prob_player"] = round(1 / record["odds_player"], 4)
            if record.get("odds_opponent"):
                record["implied_prob_opponent"] = round(1 / record["odds_opponent"], 4)
        else:
            record["has_odds"] = False
        
        all_records.append(record)
    
    df = pl.DataFrame(all_records)
    
    # Remove duplicates and apply schema
    original_count = len(df)
    try:
        from src.schema import deduplicate_matches, enforce_schema, SchemaValidator
        
        # Deduplicate (event_id only for future matches since no player perspective)
        df = df.unique(subset=["event_id"], maintain_order=True)
        
        # Enforce schema with data_type = "upcoming"
        df = enforce_schema(df, data_type="upcoming")
        
        # Validate
        validator = SchemaValidator()
        validator.validate_and_log(df, data_type="upcoming")
        
    except ImportError:
        df = df.unique(subset=["event_id"], maintain_order=True)
    
    if len(df) < original_count:
        print(f"  Removed {original_count - len(df)} duplicates")
    
    # Save
    output_path = FUTURE_DIR / f"upcoming_matches_{today.isoformat()}.parquet"
    df.write_parquet(output_path, compression="snappy")
    
    latest_path = FUTURE_DIR / "upcoming_matches_latest.parquet"
    df.write_parquet(latest_path, compression="snappy")
    
    # Clear checkpoint on success
    checkpoint.clear()
    
    # Summary
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)
    print(f"Total matches: {len(df)}")
    print(f"With odds: {df.filter(pl.col('has_odds')).height}")
    print(f"Date range: {df['match_date'].min()} to {df['match_date'].max()}")
    print(f"Cache hits: {fetch_json_cached.cache_info().hits}")
    print(f"Saved: {output_path}")
    print("=" * 60)
    
    return df


def get_active_player_ids(df: pl.DataFrame) -> List[int]:
    """Get unique player IDs from future matches."""
    player_ids = df["player_id"].unique().to_list()
    opponent_ids = df["opponent_id"].unique().to_list()
    
    all_ids = list(set(player_ids + opponent_ids))
    return [int(i) for i in all_ids if i is not None]


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Scrape Future Tennis Matches (Optimized)")
    parser.add_argument("--days", type=int, default=7, help="Days to look ahead")
    parser.add_argument("--save-players", action="store_true", 
                       help="Save active player IDs for smart updates")
    parser.add_argument("--workers", type=int, default=8,
                       help="Number of parallel workers")
    args = parser.parse_args()
    
    start_time = time.time()
    df = scrape_future_matches(args.days)
    elapsed = time.time() - start_time
    
    print(f"\nTotal time: {elapsed:.1f}s")
    
    if len(df) > 0 and args.save_players:
        player_ids = get_active_player_ids(df)
        
        players_file = FUTURE_DIR / "active_players.json"
        with open(players_file, "w") as f:
            json.dump({
                "player_ids": player_ids,
                "count": len(player_ids),
                "scraped_at": datetime.now().isoformat(),
            }, f, indent=2)
        
        print(f"Active players saved: {len(player_ids)}")
    
    return 0 if len(df) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
