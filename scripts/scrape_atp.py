"""
Local SofaScore ATP Tennis Scraper

Scrapes ATP match data locally (no Colab required).
Uses TLS fingerprinting to bypass Cloudflare.
Features: Progress bar, checkpoint saves, resume support.

Usage:
    python scripts/scrape_atp.py
    python scripts/scrape_atp.py --top 50 --max-pages 10
    python scripts/scrape_atp.py --resume  # Resume from checkpoint
"""
import sys
from pathlib import Path
import time
import random
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from threading import Lock
import argparse

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from tls_client import Session
    HAS_TLS_CLIENT = True
except ImportError:
    import httpx
    HAS_TLS_CLIENT = False
    print("[WARN] tls_client not installed, using httpx (may get rate limited)")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[WARN] tqdm not installed, using simple progress")

import polars as pl

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_URL = "https://www.sofascore.com/api/v1"

RANKING_IDS = {
    "atp_singles": 5,
    "wta_singles": 6,
}

# Rate limiting
MIN_DELAY = 0.3
MAX_DELAY = 0.8
PARALLEL_WORKERS = 2

# Verbose logging
VERBOSE = True

# Data paths
DATA_DIR = ROOT / "data" / "raw"
CHECKPOINT_DIR = ROOT / "data" / "checkpoints"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoint frequency (save every N players)
CHECKPOINT_FREQUENCY = 5


# =============================================================================
# SESSION POOL
# =============================================================================

class SessionPool:
    """Thread-safe pool of TLS sessions."""
    
    def __init__(self, size: int = 3):
        if HAS_TLS_CLIENT:
            self.sessions = [Session(client_identifier="firefox_120") for _ in range(size)]
        else:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
                "Accept": "application/json",
            }
            self.sessions = [httpx.Client(headers=headers, timeout=30) for _ in range(size)]
        
        self.index = 0
        self.lock = Lock()
    
    def get(self):
        with self.lock:
            session = self.sessions[self.index % len(self.sessions)]
            self.index += 1
            return session


pool = SessionPool(PARALLEL_WORKERS)
request_count = 0
count_lock = Lock()


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

class CheckpointManager:
    """Manage checkpoints for resume support."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.data_file = checkpoint_dir / "checkpoint_data.parquet"
        self.state_file = checkpoint_dir / "checkpoint_state.json"
    
    def save(self, records: List[Dict], completed_players: List[int], player_index: int):
        """Save checkpoint."""
        if not records:
            return
        
        # Save data
        df = pl.DataFrame(records)
        df.write_parquet(self.data_file, compression="snappy")
        
        # Save state
        state = {
            "completed_players": completed_players,
            "last_player_index": player_index,
            "record_count": len(records),
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
        
        print(f"  [CHECKPOINT] Saved {len(records)} records at player {player_index}")
    
    def load(self) -> tuple:
        """Load checkpoint if exists."""
        if not self.state_file.exists() or not self.data_file.exists():
            return [], [], 0
        
        try:
            # Load state
            with open(self.state_file, "r") as f:
                state = json.load(f)
            
            # Load data
            df = pl.read_parquet(self.data_file)
            records = df.to_dicts()
            
            print(f"  [RESUME] Loaded {len(records)} records from checkpoint")
            print(f"  [RESUME] Resuming from player index {state['last_player_index']}")
            
            return records, state["completed_players"], state["last_player_index"]
        except Exception as e:
            print(f"  [WARN] Failed to load checkpoint: {e}")
            return [], [], 0
    
    def clear(self):
        """Clear checkpoint files."""
        if self.data_file.exists():
            self.data_file.unlink()
        if self.state_file.exists():
            self.state_file.unlink()


# =============================================================================
# API FUNCTIONS
# =============================================================================

def fetch_json(endpoint: str, retries: int = 2) -> Optional[Dict]:
    """Fetch JSON from SofaScore API with retries."""
    global request_count
    
    url = f"{BASE_URL}{endpoint}" if endpoint.startswith('/') else endpoint
    session = pool.get()
    
    for attempt in range(retries + 1):
        try:
            delay = random.uniform(MIN_DELAY, MAX_DELAY)
            time.sleep(delay)
            
            if VERBOSE and request_count % 20 == 0:
                print(f"\n  [API] Request #{request_count}: {endpoint}")
            
            response = session.get(url)
            
            with count_lock:
                request_count += 1
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                if VERBOSE:
                    print(f"\n  [WARN] Rate limited (403), backing off...")
                time.sleep(3 * (attempt + 1))
            elif response.status_code == 404:
                return None
            else:
                if VERBOSE:
                    print(f"\n  [WARN] Status {response.status_code}")
        except Exception as e:
            if VERBOSE:
                print(f"\n  [ERROR] Request failed: {e}")
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

def fetch_rankings(ranking_type: str = "atp_singles", limit: int = 100) -> List[Dict]:
    """Fetch player rankings."""
    print(f"Fetching {ranking_type} rankings (top {limit})...")
    
    ranking_id = RANKING_IDS.get(ranking_type, 5)
    data = fetch_json(f"/rankings/{ranking_id}")
    
    if not data or "rankingRows" not in data:
        print("  [ERROR] Failed to fetch rankings")
        return []
    
    players = []
    for row in data["rankingRows"][:limit]:
        team = row.get("team", {})
        players.append({
            "position": row.get("position"),
            "player_id": team.get("id"),
            "name": team.get("name"),
            "country": team.get("country", {}).get("alpha2"),
            "points": row.get("points"),
        })
    
    print(f"  Retrieved {len(players)} players")
    return players


def fetch_player_matches(player_id: int, max_pages: int = 10) -> List[Dict]:
    """Fetch match history for a player."""
    all_matches = []
    
    for page in range(max_pages):
        data = fetch_json(f"/team/{player_id}/events/singles/last/{page}")
        
        if not data or "events" not in data or not data["events"]:
            break
        
        all_matches.extend(data["events"])
        
        if data.get("hasNextPage") is False:
            break
    
    return all_matches


def fetch_match_stats(event_id: int) -> Optional[Dict]:
    """Fetch match statistics."""
    return fetch_json(f"/event/{event_id}/statistics")


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


# =============================================================================
# DATA PROCESSING
# =============================================================================

def process_match(event: Dict, player_id: int) -> tuple:
    """Convert raw match to player-centric format."""
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
    
    for i in range(1, 6):
        record[f"player_set{i}"] = home_score.get(f"period{i}") if is_home else away_score.get(f"period{i}")
        record[f"opponent_set{i}"] = away_score.get(f"period{i}") if is_home else home_score.get(f"period{i}")
    
    return record, "home" if is_home else "away", "away" if is_home else "home"


def flatten_stats(stats_data: Dict, player_prefix: str, opponent_prefix: str) -> Dict:
    """Flatten nested statistics."""
    if not stats_data or "statistics" not in stats_data:
        return {}
    
    result = {}
    for period_data in stats_data.get("statistics", []):
        period = period_data.get("period", "ALL")
        if period != "ALL":
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
# MAIN SCRAPER
# =============================================================================

def scrape_player(player: Dict, max_pages: int = 10, fetch_details: bool = True) -> List[Dict]:
    """Scrape all matches for a player."""
    player_id = player["player_id"]
    
    raw_matches = fetch_player_matches(player_id, max_pages)
    
    if not raw_matches:
        return []
    
    records = []
    
    for event in raw_matches:
        if event.get("status", {}).get("type") != "finished":
            continue
        
        record, player_pref, opp_pref = process_match(event, player_id)
        
        if fetch_details:
            event_id = event.get("id")
            
            stats = fetch_match_stats(event_id)
            if stats:
                stat_cols = flatten_stats(stats, player_pref, opp_pref)
                record.update(stat_cols)
                record["has_stats"] = True
            else:
                record["has_stats"] = False
            
            odds = fetch_match_odds(event_id)
            record = add_odds(record, odds)
        
        records.append(record)
    
    return records


def run_scraper(
    top_players: int = 50,
    max_pages: int = 10,
    fetch_details: bool = True,
    output_path: Path = None,
    resume: bool = False,
) -> pl.DataFrame:
    """
    Run the full scraper with progress bar and checkpoints.
    """
    print("="*60)
    print("ATP TENNIS DATA SCRAPER")
    print("="*60)
    print(f"Top players: {top_players} | Pages: {max_pages} | Details: {fetch_details}")
    print("="*60)
    
    start_time = time.time()
    checkpoint_mgr = CheckpointManager(CHECKPOINT_DIR)
    
    # Get rankings
    players = fetch_rankings("atp_singles", top_players)
    
    if not players:
        print("[ERROR] No players found.")
        return pl.DataFrame()
    
    # Resume from checkpoint?
    all_records = []
    completed_player_ids = []
    start_index = 0
    
    if resume:
        all_records, completed_player_ids, start_index = checkpoint_mgr.load()
    
    # Create progress bar
    remaining_players = players[start_index:]
    
    if HAS_TQDM:
        pbar = tqdm(
            remaining_players,
            desc="Scraping players",
            initial=start_index,
            total=len(players),
            unit="player",
            ncols=80,
        )
    else:
        pbar = remaining_players
    
    try:
        for i, player in enumerate(pbar):
            player_id = player["player_id"]
            actual_index = start_index + i
            
            # Skip if already completed
            if player_id in completed_player_ids:
                continue
            
            # Update progress bar description
            if HAS_TQDM:
                pbar.set_postfix({
                    "player": player["name"][:15],
                    "matches": len(all_records),
                })
            else:
                if (actual_index + 1) % 5 == 0:
                    print(f"  [{actual_index+1}/{len(players)}] {player['name']} - {len(all_records)} matches")
            
            # Scrape player
            try:
                records = scrape_player(player, max_pages, fetch_details)
                all_records.extend(records)
                completed_player_ids.append(player_id)
            except Exception as e:
                print(f"\n  [ERROR] {player['name']}: {e}")
                continue
            
            # Checkpoint
            if (actual_index + 1) % CHECKPOINT_FREQUENCY == 0:
                checkpoint_mgr.save(all_records, completed_player_ids, actual_index + 1)
    
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving checkpoint...")
        checkpoint_mgr.save(all_records, completed_player_ids, actual_index)
        print("  Resume with: python scripts/scrape_atp.py --resume")
        return pl.DataFrame()
    
    finally:
        if HAS_TQDM:
            pbar.close()
    
    # Create DataFrame
    print(f"\nProcessing {len(all_records)} match records...")
    
    if not all_records:
        print("[WARN] No records collected")
        return pl.DataFrame()
    
    df = pl.DataFrame(all_records)
    
    # Use schema utilities for deduplication (prefer rows with odds)
    try:
        from src.schema import deduplicate_matches, enforce_schema, SchemaValidator
        
        # Deduplicate with odds preference
        df = deduplicate_matches(df.lazy(), prefer_with_odds=True).collect()
        
        # Enforce schema
        df = enforce_schema(df, data_type="historical")
        
        # Validate
        validator = SchemaValidator()
        validator.validate_and_log(df, data_type="historical")
        
    except ImportError:
        # Fallback to basic dedup if schema module not available
        df = df.unique(subset=["event_id", "player_id"])
    
    df = df.sort("start_timestamp")
    
    # Save final output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DATA_DIR / f"atp_matches_{timestamp}.parquet"
    
    df.write_parquet(output_path, compression="snappy")
    
    # Clear checkpoint
    checkpoint_mgr.clear()
    
    # Summary
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("SCRAPING COMPLETE")
    print("="*60)
    print(f"Total matches: {len(df):,}")
    print(f"Unique players: {df['player_id'].n_unique()}")
    print(f"Date range: {df['match_date'].min()} to {df['match_date'].max()}")
    
    if "has_odds" in df.columns:
        odds_pct = df.filter(pl.col("has_odds")).height / len(df) * 100
        print(f"Odds coverage: {odds_pct:.1f}%")
    
    print(f"Requests: {request_count} | Time: {elapsed:.1f}s")
    print(f"Saved: {output_path}")
    print("="*60)
    
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Scrape ATP Tennis Data")
    parser.add_argument("--top", type=int, default=50, help="Top N players to scrape")
    parser.add_argument("--max-pages", type=int, default=10, help="Max match pages per player")
    parser.add_argument("--no-details", action="store_true", help="Skip stats/odds fetching")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    
    output_path = Path(args.output) if args.output else None
    
    df = run_scraper(
        top_players=args.top,
        max_pages=args.max_pages,
        fetch_details=not args.no_details,
        output_path=output_path,
        resume=args.resume,
    )
    
    return 0 if len(df) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
