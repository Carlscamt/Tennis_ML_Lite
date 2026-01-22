"""
Incremental Data Update Script

Updates existing data with new matches since last scrape.
Run weekly or before making predictions.

Usage:
    python scripts/update_data.py              # Update with new matches
    python scripts/update_data.py --full       # Full rescrape
    python scripts/update_data.py --days 7     # Only last 7 days
"""
import sys
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import json

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import polars as pl
from scripts.scrape_atp import (
    fetch_rankings, fetch_player_matches, fetch_match_stats, 
    fetch_match_odds, process_match, flatten_stats, add_odds,
    DATA_DIR, CHECKPOINT_DIR
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default settings for incremental updates
DEFAULT_TOP_PLAYERS = 100   # Cover more players for opponent coverage
DEFAULT_MAX_PAGES = 3       # Fewer pages for incremental (recent matches)
DEFAULT_DAYS_BACK = 14      # Only look at last 2 weeks

# State file to track last update
STATE_FILE = DATA_DIR / "update_state.json"


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def load_state() -> dict:
    """Load last update state."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"last_update": None, "last_match_timestamp": 0}


def save_state(last_timestamp: int):
    """Save update state."""
    state = {
        "last_update": datetime.now().isoformat(),
        "last_match_timestamp": last_timestamp,
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_existing_data() -> pl.DataFrame:
    """Load existing scraped data."""
    parquet_files = list(DATA_DIR.glob("atp_matches_*.parquet"))
    
    if not parquet_files:
        return pl.DataFrame()
    
    # Get most recent file
    latest_file = max(parquet_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading existing data from: {latest_file.name}")
    
    return pl.read_parquet(latest_file)


# =============================================================================
# INCREMENTAL UPDATE
# =============================================================================

def get_new_matches(
    player: dict,
    max_pages: int,
    last_timestamp: int,
    fetch_details: bool = False
) -> list:
    """Get only new matches since last_timestamp."""
    records = []
    player_id = player["player_id"]
    
    raw_matches = fetch_player_matches(player_id, max_pages)
    
    for event in raw_matches:
        if event.get("status", {}).get("type") != "finished":
            continue
        
        match_ts = event.get("startTimestamp", 0)
        
        # Skip old matches
        if match_ts <= last_timestamp:
            continue
        
        record, player_pref, opp_pref = process_match(event, player_id)
        
        if fetch_details:
            event_id = event.get("id")
            
            stats = fetch_match_stats(event_id)
            if stats:
                record.update(flatten_stats(stats, player_pref, opp_pref))
                record["has_stats"] = True
            else:
                record["has_stats"] = False
            
            odds = fetch_match_odds(event_id)
            record = add_odds(record, odds)
        
        records.append(record)
    
    return records


def run_incremental_update(
    top_players: int = DEFAULT_TOP_PLAYERS,
    max_pages: int = DEFAULT_MAX_PAGES,
    days_back: int = DEFAULT_DAYS_BACK,
    fetch_details: bool = False
) -> pl.DataFrame:
    """
    Run incremental update - only fetch new matches.
    """
    print("="*60)
    print("INCREMENTAL DATA UPDATE")
    print("="*60)
    
    # Load existing data
    existing_df = load_existing_data()
    state = load_state()
    
    # Determine cutoff timestamp
    if state["last_match_timestamp"] > 0:
        last_ts = state["last_match_timestamp"]
        print(f"Last update: {state['last_update']}")
    else:
        # First run - use days_back
        cutoff_date = datetime.now() - timedelta(days=days_back)
        last_ts = int(cutoff_date.timestamp())
        print(f"First run - getting matches from last {days_back} days")
    
    # Get rankings
    players = fetch_rankings("atp_singles", top_players)
    
    if not players:
        print("[ERROR] Failed to fetch rankings")
        return existing_df
    
    # Fetch new matches
    print(f"\nChecking {len(players)} players for new matches...")
    
    try:
        from tqdm import tqdm
        pbar = tqdm(players, desc="Updating", unit="player")
    except ImportError:
        pbar = players
    
    all_new_records = []
    new_match_count = 0
    
    for player in pbar:
        new_matches = get_new_matches(player, max_pages, last_ts, fetch_details)
        all_new_records.extend(new_matches)
        new_match_count += len(new_matches)
        
        try:
            pbar.set_postfix({"new_matches": new_match_count})
        except:
            pass
    
    print(f"\nFound {len(all_new_records)} new match records")
    
    if not all_new_records:
        print("No new matches found. Data is up to date.")
        return existing_df
    
    # Create new DataFrame
    new_df = pl.DataFrame(all_new_records)
    
    # Merge with existing
    if len(existing_df) > 0:
        combined = pl.concat([existing_df, new_df], how="diagonal")
        combined = combined.unique(subset=["event_id", "player_id"])
        combined = combined.sort("start_timestamp")
    else:
        combined = new_df.sort("start_timestamp")
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = DATA_DIR / f"atp_matches_{timestamp}.parquet"
    combined.write_parquet(output_path, compression="snappy")
    
    # Update state
    max_ts = combined.select(pl.col("start_timestamp").max()).item()
    save_state(max_ts)
    
    # Summary
    print("\n" + "="*60)
    print("UPDATE COMPLETE")
    print("="*60)
    print(f"New matches added: {len(all_new_records)}")
    print(f"Total matches: {len(combined):,}")
    print(f"Unique players: {combined['player_id'].n_unique()}")
    print(f"Date range: {combined['match_date'].min()} to {combined['match_date'].max()}")
    print(f"Saved: {output_path}")
    print("="*60)
    
    return combined


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Update ATP Tennis Data")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_PLAYERS,
                       help=f"Top N players (default: {DEFAULT_TOP_PLAYERS})")
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES,
                       help=f"Max pages per player (default: {DEFAULT_MAX_PAGES})")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS_BACK,
                       help=f"Days back for first run (default: {DEFAULT_DAYS_BACK})")
    parser.add_argument("--details", action="store_true",
                       help="Fetch stats and odds")
    parser.add_argument("--full", action="store_true",
                       help="Full rescrape (ignore last update)")
    args = parser.parse_args()
    
    if args.full:
        # Full rescrape - reset state
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        print("Full rescrape requested - ignoring previous state")
    
    df = run_incremental_update(
        top_players=args.top,
        max_pages=args.max_pages,
        days_back=args.days,
        fetch_details=args.details,
    )
    
    return 0 if len(df) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
