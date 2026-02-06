"""Quick script to analyze tournament data coverage."""
import polars as pl
from datetime import datetime
from pathlib import Path

# Load data
data_path = Path("data/tennis.parquet")
df = pl.read_parquet(data_path)

print(f"=" * 70)
print(f"TOURNAMENT DATA ANALYSIS")
print(f"=" * 70)
print(f"\nTotal matches: {len(df):,}")

# Date range
min_ts = df.select(pl.col("start_timestamp").min()).item()
max_ts = df.select(pl.col("start_timestamp").max()).item()
min_dt = datetime.fromtimestamp(min_ts)
max_dt = datetime.fromtimestamp(max_ts)
print(f"Date range: {min_dt.strftime('%Y-%m-%d')} to {max_dt.strftime('%Y-%m-%d')}")

# Add year column
df = df.with_columns(
    pl.from_epoch("start_timestamp").dt.year().alias("year")
)

# Yearly breakdown
print(f"\n{'='*70}")
print("MATCHES BY YEAR")
print(f"{'='*70}")
yearly = df.group_by("year").agg(pl.len().alias("matches")).sort("year")
for row in yearly.iter_rows(named=True):
    print(f"  {row['year']}: {row['matches']:,} matches")

# Top tournaments overall
print(f"\n{'='*70}")
print("TOP 25 TOURNAMENTS (All Years Combined)")
print(f"{'='*70}")
print(f"{'Tournament':<40} {'Matches':<10} {'Years'}")
print("-" * 70)

tournaments = (
    df.group_by("tournament_name")
    .agg([
        pl.len().alias("matches"),
        pl.col("year").n_unique().alias("years"),
        pl.col("year").min().alias("min_year"),
        pl.col("year").max().alias("max_year"),
    ])
    .sort("matches", descending=True)
)

for row in tournaments.head(25).iter_rows(named=True):
    name = row["tournament_name"][:39] if row["tournament_name"] else "Unknown"
    year_range = f"{row['min_year']}-{row['max_year']}"
    print(f"{name:<40} {row['matches']:<10} {year_range}")

# Grand Slams detail
print(f"\n{'='*70}")
print("GRAND SLAM DETAIL")
print(f"{'='*70}")

grand_slams = ["australian open", "french open", "roland garros", "wimbledon", "us open"]
for slam in grand_slams:
    slam_data = df.filter(pl.col("tournament_name").str.to_lowercase().str.contains(slam))
    if len(slam_data) > 0:
        print(f"\n{slam.title()}:")
        by_year = slam_data.group_by("year").agg([
            pl.len().alias("matches"),
            pl.col("round_name").n_unique().alias("rounds")
        ]).sort("year")
        for row in by_year.iter_rows(named=True):
            print(f"  {row['year']}: {row['matches']} matches, {row['rounds']} rounds")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
