"""
Smart Player History Update

Only updates match history for players who are playing in upcoming matches.
Much faster than scraping all top 100+ players.

Usage:
    python scripts/update_active_players.py
    python scripts/update_active_players.py --future-days 7
"""
import sys
from pathlib import Path
import time
import random
import json
from datetime import datetime
from typing import Dict, List, Optional
from threading import Lock
from functools import lru_cache

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

try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    HAS_PARALLEL = True
except ImportError:
    HAS_PARALLEL = False

import polars as pl
import argparse

from scripts.scrape_future import scrape_future_matches, get_active_player_ids

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_URL = "https://www.sofascore.com/api/v1"
MIN_DELAY = 0.3
MAX_DELAY = 0.8

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FUTURE_DIR = DATA_DIR / "future"
STATE_FILE = DATA_DIR / "player_update_state.json"


# =============================================================================
# SESSION
# =============================================================================

def get_session():
    """Create optimized session with connection pooling."""
    if HAS_TLS_CLIENT:
        return Session(client_identifier="firefox_120")
    else:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
            "Accept": "application/json",
        }
        # Use persistent connections with HTTP/2 support
        return httpx.Client(
            headers=headers,
            timeout=30,
            http2=True,
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20,
                keepalive_expiry=30.0
            )
        )


session = get_session()
request_count = 0
count_lock = Lock()

# Response cache for frequently accessed data (tournaments, player info)
@lru_cache(maxsize=500)
def cached_fetch(endpoint: str) -> Optional[Dict]:
    """Cached version of fetch_json for static data like tournaments."""
    return fetch_json(endpoint)


def fetch_json(endpoint: str, retries: int = 2) -> Optional[Dict]:
    global request_count
    
    url = f"{BASE_URL}{endpoint}" if endpoint.startswith('/') else endpoint
    
    for attempt in range(retries + 1):
        try:
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            response = session.get(url)
            
            with count_lock:
                request_count += 1
            
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


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

class CheckpointManager:
    """Manage checkpoints for resume support."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.data_file = checkpoint_dir / "checkpoint_data.parquet"
        self.state_file = checkpoint_dir / "checkpoint_state.json"
    
    def save(self, records: List[Dict], completed_players: List[int]):
        """Save checkpoint."""
        if not records:
            return
        
        # Save data
        df = pl.DataFrame(records)
        df.write_parquet(self.data_file, compression="snappy")
        
        # Save state
        state = {
            "completed_players": completed_players,
            "record_count": len(records),
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
    
    def load(self) -> tuple:
        """Load checkpoint if exists."""
        if not self.state_file.exists() or not self.data_file.exists():
            return [], []
        
        try:
            # Load state
            with open(self.state_file, "r") as f:
                state = json.load(f)
            
            # Load data
            df = pl.read_parquet(self.data_file)
            records = df.to_dicts()
            
            completed = state.get("completed_players", [])
            print(f"  [CHECKPOINT] Resuming from {len(completed)} completed players")
            return records, completed
        except Exception as e:
            print(f"  [CHECKPOINT] Failed to load: {e}")
            return [], []
    
    def clear(self):
        """Clear checkpoint files."""
        if self.data_file.exists():
            self.data_file.unlink()
        if self.state_file.exists():
            self.state_file.unlink()


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def load_state() -> Dict:
    """Load update state."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"last_update": {}, "player_timestamps": {}}


def save_state(state: Dict):
    """Save update state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def should_update_player(player_id: int, state: Dict, force: bool = False) -> bool:
    """Check if player needs updating based on last match timestamp."""
    if force:
        return True
    
    player_timestamps = state.get("player_timestamps", {})
    player_state = player_timestamps.get(str(player_id), {})
    
    # If never updated, definitely update
    if not player_state:
        return True
    
    last_scraped = player_state.get("last_scraped")
    if not last_scraped:
        return True
    
    # Check if enough time has passed (24 hours)
    from datetime import datetime, timedelta
    last_scraped_dt = datetime.fromisoformat(last_scraped)
    if datetime.now() - last_scraped_dt < timedelta(hours=24):
        # Recently updated, skip
        return False
    
    return True


def get_latest_data() -> Optional[pl.DataFrame]:
    """Get latest raw data file."""
    files = sorted(RAW_DIR.glob("atp_matches_*.parquet"), key=lambda p: p.stat().st_mtime)
    if files:
        return pl.read_parquet(files[-1])
    return None


# =============================================================================
# PLAYER DATA FETCHING
# =============================================================================

def fetch_player_matches(player_id: int, max_pages: int = 3, existing_events: set = None) -> List[Dict]:
    """Fetch recent match history for a player with smart early stopping."""
    all_matches = []
    
    if existing_events is None:
        existing_events = set()
    
    consecutive_existing = 0  # Track consecutive pages with no new matches
    
    for page in range(max_pages):
        data = fetch_json(f"/team/{player_id}/events/singles/last/{page}")
        
        if not data or "events" not in data or not data["events"]:
            break
        
        page_events = data["events"]
        new_on_page = 0
        
        for event in page_events:
            # Check if event is new
            if event.get("id") not in existing_events:
                new_on_page += 1
            all_matches.append(event)
        
        # Smart stopping: if entire page had no new matches, likely no more new data
        if new_on_page == 0:
            consecutive_existing += 1
            if consecutive_existing >= 1:  # Stop after 1 page with no new data
                break
        else:
            consecutive_existing = 0
        
        if data.get("hasNextPage") is False:
            break
    
    return all_matches


def process_match(event: Dict, player_id: int) -> Dict:
    """Process match data."""
    home_id = event.get("homeTeam", {}).get("id")
    is_home = (home_id == player_id)
    
    home = event.get("homeTeam", {})
    away = event.get("awayTeam", {})
    home_score = event.get("homeScore", {})
    away_score = event.get("awayScore", {})
    tournament = event.get("tournament", {}).get("uniqueTournament", {})
    
    winner = event.get("winnerCode")
    player_won = (winner == 1 and is_home) or (winner == 2 and not is_home)
    
    record = {
        "_schema_version": "1.1",
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
    
    ts = event.get("startTimestamp")
    if ts:
        dt = datetime.fromtimestamp(ts)
        record["match_date"] = dt.date().isoformat()
        record["match_year"] = dt.year
        record["match_month"] = dt.month
    
    return record


def fetch_single_player(player_id: int, max_pages: int, existing_events: set, state: Dict) -> tuple:
    """Fetch matches for a single player with timestamp tracking."""
    matches = fetch_player_matches(player_id, max_pages, existing_events)
    
    new_records = []
    latest_timestamp = 0
    
    for event in matches:
        if event.get("status", {}).get("type") != "finished":
            continue
        
        event_id = event.get("id")
        if event_id in existing_events:
            continue
        
        record = process_match(event, player_id)
        new_records.append(record)
        
        # Track latest match timestamp
        ts = event.get("startTimestamp", 0)
        if ts > latest_timestamp:
            latest_timestamp = ts
    
    # Update player state
    if "player_timestamps" not in state:
        state["player_timestamps"] = {}
    
    state["player_timestamps"][str(player_id)] = {
        "last_scraped": datetime.now().isoformat(),
        "last_match_timestamp": latest_timestamp,
        "matches_found": len(new_records)
    }
    
    return player_id, new_records


def update_player_data(
    player_ids: List[int],
    existing_df: Optional[pl.DataFrame],
    max_pages: int = 3,
    parallel_workers: int = 3,
    checkpoint_interval: int = 50,
    smart_update: bool = True,
) -> pl.DataFrame:
    """
    Update data for specific players only.
    
    Args:
        player_ids: List of player IDs to update
        existing_df: Existing data (to merge with)
        max_pages: Max pages of history per player
        parallel_workers: Number of parallel workers (default: 3)
        checkpoint_interval: Save checkpoint every N players
        smart_update: Use timestamp tracking to skip recently updated players
        
    Returns:
        Updated DataFrame
    """
    print("="*60)
    print("SMART PLAYER UPDATE (OPTIMIZED v2)")
    print("="*60)
    
    # Load state for timestamp tracking
    state = load_state()
    
    # Filter by timestamp if smart_update enabled
    if smart_update:
        original_count = len(player_ids)
        player_ids = [pid for pid in player_ids if should_update_player(pid, state)]
        skipped = original_count - len(player_ids)
        
        if skipped > 0:
            print(f"⚡ Smart Update: Skipped {skipped}/{original_count} recently updated players")
    
    # Filter to only NEW players (incremental update)
    if existing_df is not None and len(existing_df) > 0:
        existing_player_ids = set(existing_df["player_id"].unique().to_list())
        new_player_ids = [pid for pid in player_ids if pid not in existing_player_ids]
        
        if len(new_player_ids) < len(player_ids):
            print(f"Players in DB: {len(existing_player_ids)}")
            print(f"NEW players to fetch: {len(new_player_ids)} (skipping {len(player_ids) - len(new_player_ids)} existing)")
            player_ids = new_player_ids
        else:
            print(f"Players to update: {len(player_ids)} (all new)")
    else:
        print(f"Players to update: {len(player_ids)} (first run)")
    
    if not player_ids:
        print("No new players to update!")
        return existing_df if existing_df is not None else pl.DataFrame()
    
    # Setup checkpoint
    checkpoint = CheckpointManager(DATA_DIR / ".checkpoints")
    checkpoint_records, completed_players = checkpoint.load()
    
    # Filter out already completed
    remaining_players = [pid for pid in player_ids if pid not in completed_players]
    
    if checkpoint_records:
        print(f"Resuming: {len(checkpoint_records)} matches from checkpoint")
        print(f"Remaining: {len(remaining_players)}/{len(player_ids)} players")
        new_records = checkpoint_records
        player_ids = remaining_players
    else:
        new_records = []
    
    if not player_ids:
        print("All players already completed!")
    else:
        # Get existing event IDs to avoid duplicates
        existing_events = set()
        if existing_df is not None and len(existing_df) > 0:
            existing_events = set(existing_df["event_id"].unique().to_list())
            print(f"Existing matches: {len(existing_df):,}")
        
        # Add already fetched events
        for rec in new_records:
            existing_events.add(rec["event_id"])
        
        # Parallel or sequential fetch
        if HAS_PARALLEL and parallel_workers > 1:
            print(f"\nFetching with {parallel_workers} parallel workers...")
            
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = {executor.submit(fetch_single_player, pid, max_pages, existing_events, state): pid 
                          for pid in player_ids}
                
                if HAS_TQDM:
                    pbar = tqdm(total=len(player_ids), desc="Updating players", unit="player")
                
                for future in as_completed(futures):
                    player_id, records = future.result()
                    new_records.extend(records)
                    completed_players.append(player_id)
                    
                    # Add to existing events
                    for rec in records:
                        existing_events.add(rec["event_id"])
                    
                    if HAS_TQDM:
                        pbar.update(1)
                        pbar.set_postfix({"new": len(new_records)})
                    
                    # Checkpoint
                    if len(completed_players) % checkpoint_interval == 0:
                        checkpoint.save(new_records, completed_players)
                        save_state(state)  # Save timestamp state too
                
                if HAS_TQDM:
                    pbar.close()
        else:
            # Sequential (original logic)
            print(f"\nFetching sequentially...")
            
            if HAS_TQDM:
                pbar = tqdm(player_ids, desc="Updating players", unit="player")
            else:
                pbar = player_ids
            
            for i, player_id in enumerate(pbar):
                _, records = fetch_single_player(player_id, max_pages, existing_events, state)
                new_records.extend(records)
                completed_players.append(player_id)
                
                for rec in records:
                    existing_events.add(rec["event_id"])
                
                if HAS_TQDM:
                    pbar.set_postfix({"new": len(new_records)})
                
                # Checkpoint
                if (i + 1) % checkpoint_interval == 0:
                    checkpoint.save(new_records, completed_players)
                    save_state(state)  # Save timestamp state too
        
        print(f"\nNew matches found: {len(new_records)}")
    
    # Merge with existing
    if not new_records:
        print("No new data to add.")
        return existing_df if existing_df is not None else pl.DataFrame()
    
    new_df = pl.DataFrame(new_records)
    
    # Use schema utilities for smart merging
    try:
        from src.schema import merge_datasets, enforce_schema, SchemaValidator
        
        if existing_df is not None and len(existing_df) > 0:
            # Smart merge with schema utilities (prefer new data, prefer odds)
            combined = merge_datasets(existing_df, new_df, prefer_new=True)
        else:
            combined = new_df.sort("start_timestamp")
        
        # Enforce schema
        combined = enforce_schema(combined, data_type="historical")
        
        # Validate
        validator = SchemaValidator()
        validator.validate_and_log(combined)
        
    except ImportError:
        # Fallback to basic merge
        if existing_df is not None and len(existing_df) > 0:
            combined = pl.concat([existing_df, new_df], how="diagonal")
            combined = combined.unique(subset=["event_id", "player_id"])
            combined = combined.sort("start_timestamp")
        else:
            combined = new_df.sort("start_timestamp")
    
    # Clear checkpoint after successful completion
    checkpoint.clear()
    
    # Save final state with timestamps
    save_state(state)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RAW_DIR / f"atp_matches_{timestamp}.parquet"
    combined.write_parquet(output_path, compression="snappy")
    
    print(f"\nTotal matches: {len(combined):,}")
    print(f"Saved: {output_path}")
    
    return combined


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Smart Player Update")
    parser.add_argument("--future-days", type=int, default=7,
                       help="Days ahead to check for active players")
    parser.add_argument("--max-pages", type=int, default=3,
                       help="Max history pages per player")
    parser.add_argument("--player-ids", type=str, default=None,
                       help="Comma-separated player IDs (skip future scrape)")
    args = parser.parse_args()
    
    # Get active players
    if args.player_ids:
        player_ids = [int(x) for x in args.player_ids.split(",")]
        print(f"Using provided player IDs: {len(player_ids)}")
    else:
        # Scrape future matches first
        print("Step 1: Fetching future matches to identify active players...")
        future_df = scrape_future_matches(args.future_days)
        
        if len(future_df) == 0:
            print("No future matches found.")
            return 1
        
        player_ids = get_active_player_ids(future_df)
        print(f"\nStep 2: Found {len(player_ids)} active players")
    
    # Load existing data
    existing_df = get_latest_data()
    
    # Update player data
    print(f"\nStep 3: Updating player histories...")
    updated_df = update_player_data(
        player_ids,
        existing_df,
        max_pages=args.max_pages,
        parallel_workers=3,  # Safe rate limit
        checkpoint_interval=50
    )
    
    print("\nSmart update complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
