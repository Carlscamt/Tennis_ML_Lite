"""
Backtest ROI Analysis by Odds and Confidence Levels
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import polars as pl
from datetime import datetime
from config import PROCESSED_DATA_DIR
from src.model import Predictor, ModelRegistry

def main():
    # Load data
    df = pl.read_parquet(PROCESSED_DATA_DIR / "features_dataset.parquet")
    print(f"Total matches: {len(df):,}")

    # Filter to 2025+ (Strictly out of sample)
    # Training cutoff is 2025-01-01, so we must backtest ONLY on data after this
    start_ts = int(datetime(2025, 1, 1).timestamp())
    df = df.filter(
        (pl.col("start_timestamp") >= start_ts) &
        (pl.col("odds_player").is_not_null()) &
        (pl.col("player_won").is_not_null())
    )
    print(f"Matches with odds (2024+): {len(df):,}")

    # Load model and predict
    registry = ModelRegistry()
    try:
        version, model_path = registry.get_production_model()
    except RuntimeError:
        # Fallback to Staging model if no Production model
        result = registry.get_challenger_model()
        if result is None:
            raise RuntimeError("No Production or Staging model available for backtest")
        version, model_path = result
        print(f"Note: Using Staging model {version} (no Production model available)")
    
    predictor = Predictor(Path(model_path))
    df = predictor.predict_with_value(df, min_edge=0.0)

    # Add bins for odds and confidence
    df = df.with_columns([
        pl.when(pl.col("odds_player") < 1.5).then(pl.lit("1.00-1.50"))
          .when(pl.col("odds_player") < 2.0).then(pl.lit("1.50-2.00"))
          .when(pl.col("odds_player") < 2.5).then(pl.lit("2.00-2.50"))
          .when(pl.col("odds_player") < 3.0).then(pl.lit("2.50-3.00"))
          .otherwise(pl.lit("3.00+")).alias("odds_bin"),
        pl.when(pl.col("model_prob") < 0.50).then(pl.lit("<50%"))
          .when(pl.col("model_prob") < 0.55).then(pl.lit("50-55%"))
          .when(pl.col("model_prob") < 0.60).then(pl.lit("55-60%"))
          .when(pl.col("model_prob") < 0.65).then(pl.lit("60-65%"))
          .when(pl.col("model_prob") < 0.70).then(pl.lit("65-70%"))
          .otherwise(pl.lit("70%+")).alias("conf_bin"),
        # Profit: if won, profit = odds - 1; if lost, profit = -1
        (pl.col("player_won").cast(pl.Int32) * (pl.col("odds_player") - 1) - 
         (1 - pl.col("player_won").cast(pl.Int32))).alias("profit")
    ])

    # ROI by odds range
    print("\n" + "=" * 60)
    print("ROI BY ODDS RANGE")
    print("=" * 60)
    by_odds = df.group_by("odds_bin").agg([
        pl.len().alias("bets"),
        (pl.col("player_won").mean() * 100).alias("win_rate"),
        pl.col("profit").sum().alias("total_profit"),
        (pl.col("profit").sum() / pl.len() * 100).alias("roi")
    ]).sort("odds_bin")
    
    print(f"{'Odds Range':<12} | {'Bets':>5} | {'Win %':>7} | {'Profit':>10} | {'ROI':>8}")
    print("-" * 52)
    for row in by_odds.iter_rows(named=True):
        print(f"{row['odds_bin']:<12} | {row['bets']:>5} | {row['win_rate']:>6.1f}% | {row['total_profit']:>+9.1f} | {row['roi']:>+7.1f}%")

    # ROI by confidence level
    print("\n" + "=" * 60)
    print("ROI BY CONFIDENCE LEVEL")
    print("=" * 60)
    by_conf = df.group_by("conf_bin").agg([
        pl.len().alias("bets"),
        (pl.col("player_won").mean() * 100).alias("win_rate"),
        pl.col("profit").sum().alias("total_profit"),
        (pl.col("profit").sum() / pl.len() * 100).alias("roi")
    ]).sort("conf_bin")
    
    print(f"{'Confidence':<12} | {'Bets':>5} | {'Win %':>7} | {'Profit':>10} | {'ROI':>8}")
    print("-" * 52)
    for row in by_conf.iter_rows(named=True):
        print(f"{row['conf_bin']:<12} | {row['bets']:>5} | {row['win_rate']:>6.1f}% | {row['total_profit']:>+9.1f} | {row['roi']:>+7.1f}%")

    # Best combinations (matrix)
    print("\n" + "=" * 60)
    print("BEST ODDS x CONFIDENCE COMBINATIONS (min 20 bets)")
    print("=" * 60)
    matrix = df.group_by(["odds_bin", "conf_bin"]).agg([
        pl.len().alias("bets"),
        (pl.col("player_won").mean() * 100).alias("win_rate"),
        (pl.col("profit").sum() / pl.len() * 100).alias("roi")
    ])
    best = matrix.filter(pl.col("bets") >= 20).sort("roi", descending=True).head(10)
    
    print(f"{'Odds Range':<12} | {'Confidence':<10} | {'Bets':>5} | {'Win %':>7} | {'ROI':>8}")
    print("-" * 56)
    for row in best.iter_rows(named=True):
        print(f"{row['odds_bin']:<12} | {row['conf_bin']:<10} | {row['bets']:>5} | {row['win_rate']:>6.1f}% | {row['roi']:>+7.1f}%")

    # Worst combinations
    print("\n" + "=" * 60)
    print("WORST ODDS x CONFIDENCE COMBINATIONS (min 20 bets)")
    print("=" * 60)
    worst = matrix.filter(pl.col("bets") >= 20).sort("roi", descending=False).head(10)
    
    print(f"{'Odds Range':<12} | {'Confidence':<10} | {'Bets':>5} | {'Win %':>7} | {'ROI':>8}")
    print("-" * 56)
    for row in worst.iter_rows(named=True):
        print(f"{row['odds_bin']:<12} | {row['conf_bin']:<10} | {row['bets']:>5} | {row['win_rate']:>6.1f}% | {row['roi']:>+7.1f}%")

if __name__ == "__main__":
    main()
