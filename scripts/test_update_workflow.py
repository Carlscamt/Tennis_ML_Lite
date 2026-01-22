"""
Test the Update Player History workflow - simulates dashboard button click
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from datetime import datetime
import polars as pl

DATA_DIR = ROOT / "data"

print("=" * 60)
print("TEST: Update Player History Workflow")
print("=" * 60)

# Step 1: Load upcoming matches
print("\n[Step 1] Loading upcoming matches...")
future_path = DATA_DIR / "future" / "upcoming_matches_latest.parquet"

if not future_path.exists():
    print("[X] No upcoming matches found. Run 'Fetch New Matches' first.")
    sys.exit(1)

future_df = pl.read_parquet(future_path)
print(f"[OK] Loaded {len(future_df)} upcoming matches")

# Step 2: Get active player IDs
print("\n[Step 2] Getting active player IDs...")
from scripts.scrape_future import get_active_player_ids

active_ids = get_active_player_ids(future_df)
print(f"[OK] Found {len(active_ids)} active players")

# Step 3: Limit to first 5 players for test
test_ids = list(active_ids)[:5]
print(f"\n[Step 3] Testing with {len(test_ids)} players (limited for speed)...")

# Step 4: Fetch player data
print("\n[Step 4] Fetching player match history...")
from scripts.update_active_players import update_player_data, get_latest_data

existing_df = get_latest_data()
print(f"[OK] Existing data: {len(existing_df) if existing_df is not None else 0:,} matches")

updated_df = update_player_data(
    player_ids=test_ids,
    existing_df=existing_df,
    max_pages=2,  # Limited for test
    parallel_workers=2,
    smart_update=True
)

if updated_df is not None and len(updated_df) > 0:
    print(f"[OK] Updated data: {len(updated_df):,} matches")
    
    # Step 5: Save to raw
    print("\n[Step 5] Saving raw data...")
    RAW_DIR = DATA_DIR / "raw"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / f"atp_matches_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    updated_df.write_parquet(output_path)
    print(f"[OK] Saved to: {output_path}")
    
    # Step 6: Run data pipeline
    print("\n[Step 6] Running data pipeline...")
    from scripts.run_pipeline import run_data_pipeline
    from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    
    run_data_pipeline(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    print("[OK] Data pipeline complete")
    
    # Step 7: Regenerate predictions
    print("\n[Step 7] Regenerating predictions...")
    from scripts.predict_upcoming import main as predict_main
    predict_main()
    print("[OK] Predictions regenerated")
    
else:
    print("[INFO] No new data to process (all players up to date)")

print("\n" + "=" * 60)
print("[SUCCESS] TEST COMPLETE!")
print("=" * 60)
