"""
Favorite-Edge Betting Strategy Backtest

Only bets on market FAVORITES (odds < 2.0) when the model identifies positive edge.

Key Features:
- Strict temporal split (no leakage)
- Large sample size for statistical significance
- Binomial test to verify results aren't luck
- Comparison with blind favorite betting baseline

Usage:
    python scripts/backtest_favorite_edge.py
    python scripts/backtest_favorite_edge.py --min-edge 0.03
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import polars as pl
from datetime import datetime, date
from dataclasses import dataclass
from typing import Optional

from config import PROCESSED_DATA_DIR, MODELS_DIR, BACKTEST_DIR
from src.model import Predictor, ModelRegistry
from src.transform.leakage_guard import assert_no_leakage, LeakageError
from src.utils import setup_logging

logger = setup_logging()


# =============================================================================
# CONFIGURATION
# =============================================================================

# Strict temporal cutoff - model was trained on data BEFORE this date
TRAIN_CUTOFF = date(2025, 1, 1)

# Favorite threshold - odds below this = market favorite
FAVORITE_ODDS_THRESHOLD = 2.0


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    strategy_name: str
    total_bets: int
    wins: int
    losses: int
    win_rate: float
    total_profit: float
    roi: float
    avg_odds: float
    avg_edge: float
    p_value: float  # Statistical significance
    is_significant: bool


# =============================================================================
# STATISTICAL TESTING
# =============================================================================

def binomial_test(wins: int, total: int, null_prob: float) -> float:
    """
    Test if win rate is significantly better than expected.
    
    Args:
        wins: Number of wins
        total: Total number of bets
        null_prob: Expected win rate under null hypothesis (implied probability)
        
    Returns:
        p-value (lower = more significant)
    """
    try:
        from scipy import stats
        result = stats.binomtest(wins, total, null_prob, alternative='greater')
        return result.pvalue
    except ImportError:
        # Fallback: normal approximation
        import math
        expected = total * null_prob
        std = math.sqrt(total * null_prob * (1 - null_prob))
        if std == 0:
            return 1.0
        z = (wins - expected) / std
        # One-tailed p-value approximation
        return max(0, 0.5 * (1 - math.erf(z / math.sqrt(2))))


# =============================================================================
# BACKTEST STRATEGIES
# =============================================================================

def run_favorite_edge_backtest(
    df: pl.DataFrame,
    min_edge: float = 0.0,
    min_confidence: float = 0.0,
) -> BacktestResult:
    """
    Backtest: Only bet on favorites with positive model edge.
    
    Favorite = odds_player < 2.0 (implied probability > 50%)
    Edge = model_prob - implied_prob
    
    Uses FLAT BETTING (1 unit per bet) for clearer ROI interpretation.
    """
    # Calculate edge and implied probability
    df = df.with_columns([
        (1 / pl.col("odds_player")).alias("implied_prob"),
        (pl.col("model_prob") - (1 / pl.col("odds_player"))).alias("edge"),
    ])
    
    # Filter to FAVORITES ONLY with positive edge
    favorites = df.filter(
        (pl.col("odds_player") < FAVORITE_ODDS_THRESHOLD) &  # Is favorite
        (pl.col("edge") > min_edge) &                         # Has edge
        (pl.col("model_prob") >= min_confidence)              # Confidence threshold
    )
    
    if len(favorites) == 0:
        return BacktestResult(
            strategy_name="Favorite Edge",
            total_bets=0, wins=0, losses=0, win_rate=0.0,
            total_profit=0.0, roi=0.0, avg_odds=0.0, avg_edge=0.0,
            p_value=1.0, is_significant=False
        )
    
    # Calculate results (flat betting: 1 unit per bet)
    favorites = favorites.with_columns([
        pl.when(pl.col("player_won"))
          .then(pl.col("odds_player") - 1)
          .otherwise(-1.0)
          .alias("profit")
    ])
    
    total_bets = len(favorites)
    wins = favorites.filter(pl.col("player_won")).height
    losses = total_bets - wins
    win_rate = wins / total_bets
    total_profit = favorites.select(pl.col("profit").sum()).item()
    roi = total_profit / total_bets
    avg_odds = favorites.select(pl.col("odds_player").mean()).item()
    avg_edge = favorites.select(pl.col("edge").mean()).item()
    
    # Statistical significance: compare win rate to implied probability
    avg_implied = favorites.select(pl.col("implied_prob").mean()).item()
    p_value = binomial_test(wins, total_bets, avg_implied)
    
    return BacktestResult(
        strategy_name=f"Favorite Edge (min_edge={min_edge:.0%})",
        total_bets=total_bets,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        total_profit=total_profit,
        roi=roi,
        avg_odds=avg_odds,
        avg_edge=avg_edge,
        p_value=p_value,
        is_significant=(p_value < 0.05)
    )


def run_blind_favorite_backtest(df: pl.DataFrame) -> BacktestResult:
    """
    BASELINE: Blindly bet on all favorites (no model, no edge filter).
    This is what we need to beat.
    """
    favorites = df.filter(pl.col("odds_player") < FAVORITE_ODDS_THRESHOLD)
    
    if len(favorites) == 0:
        return BacktestResult(
            strategy_name="Blind Favorites (Baseline)",
            total_bets=0, wins=0, losses=0, win_rate=0.0,
            total_profit=0.0, roi=0.0, avg_odds=0.0, avg_edge=0.0,
            p_value=1.0, is_significant=False
        )
    
    favorites = favorites.with_columns([
        (1 / pl.col("odds_player")).alias("implied_prob"),
        pl.when(pl.col("player_won"))
          .then(pl.col("odds_player") - 1)
          .otherwise(-1.0)
          .alias("profit")
    ])
    
    total_bets = len(favorites)
    wins = favorites.filter(pl.col("player_won")).height
    losses = total_bets - wins
    win_rate = wins / total_bets
    total_profit = favorites.select(pl.col("profit").sum()).item()
    roi = total_profit / total_bets
    avg_odds = favorites.select(pl.col("odds_player").mean()).item()
    avg_implied = favorites.select(pl.col("implied_prob").mean()).item()
    
    p_value = binomial_test(wins, total_bets, avg_implied)
    
    return BacktestResult(
        strategy_name="Blind Favorites (Baseline)",
        total_bets=total_bets,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        total_profit=total_profit,
        roi=roi,
        avg_odds=avg_odds,
        avg_edge=0.0,  # No model edge
        p_value=p_value,
        is_significant=(p_value < 0.05)
    )


def breakdown_by_edge_bucket(df: pl.DataFrame) -> pl.DataFrame:
    """Analyze results by edge bucket for deeper insights."""
    df = df.with_columns([
        (1 / pl.col("odds_player")).alias("implied_prob"),
        (pl.col("model_prob") - (1 / pl.col("odds_player"))).alias("edge"),
        pl.when(pl.col("player_won"))
          .then(pl.col("odds_player") - 1)
          .otherwise(-1.0)
          .alias("profit")
    ])
    
    # Filter to favorites only
    df = df.filter(pl.col("odds_player") < FAVORITE_ODDS_THRESHOLD)
    
    # Create edge buckets
    df = df.with_columns([
        pl.when(pl.col("edge") < 0).then(pl.lit("< 0% (no edge)"))
          .when(pl.col("edge") < 0.03).then(pl.lit("0-3%"))
          .when(pl.col("edge") < 0.05).then(pl.lit("3-5%"))
          .when(pl.col("edge") < 0.10).then(pl.lit("5-10%"))
          .otherwise(pl.lit("10%+"))
          .alias("edge_bucket")
    ])
    
    return df.group_by("edge_bucket").agg([
        pl.len().alias("bets"),
        pl.col("player_won").sum().alias("wins"),
        (pl.col("player_won").mean() * 100).alias("win_rate"),
        pl.col("profit").sum().alias("profit"),
        (pl.col("profit").sum() / pl.len() * 100).alias("roi"),
        pl.col("odds_player").mean().alias("avg_odds"),
        (pl.col("edge") * 100).mean().alias("avg_edge"),
    ]).sort("edge_bucket")


def breakdown_by_month(df: pl.DataFrame, min_edge: float = 0.0) -> pl.DataFrame:
    """
    Analyze results by month to check strategy consistency over time.
    Shows both blind favorites and edge-filtered results side by side.
    """
    # Add month column from timestamp
    df = df.with_columns([
        pl.from_epoch(pl.col("start_timestamp")).dt.strftime("%Y-%m").alias("month"),
        (1 / pl.col("odds_player")).alias("implied_prob"),
        (pl.col("model_prob") - (1 / pl.col("odds_player"))).alias("edge"),
        pl.when(pl.col("player_won"))
          .then(pl.col("odds_player") - 1)
          .otherwise(-1.0)
          .alias("profit")
    ])
    
    # Filter to favorites only
    favorites = df.filter(pl.col("odds_player") < FAVORITE_ODDS_THRESHOLD)
    
    # Blind favorites (all favorites)
    blind_monthly = favorites.group_by("month").agg([
        pl.len().alias("blind_bets"),
        (pl.col("player_won").mean() * 100).alias("blind_win_rate"),
        pl.col("profit").sum().alias("blind_profit"),
        (pl.col("profit").sum() / pl.len() * 100).alias("blind_roi"),
    ])
    
    # Edge-filtered favorites
    edge_filtered = favorites.filter(pl.col("edge") > min_edge)
    edge_monthly = edge_filtered.group_by("month").agg([
        pl.len().alias("edge_bets"),
        (pl.col("player_won").mean() * 100).alias("edge_win_rate"),
        pl.col("profit").sum().alias("edge_profit"),
        (pl.col("profit").sum() / pl.len() * 100).alias("edge_roi"),
    ])
    
    # Join both
    monthly = blind_monthly.join(edge_monthly, on="month", how="left").sort("month")
    
    # Fill nulls with 0 for months with no edge bets
    monthly = monthly.with_columns([
        pl.col("edge_bets").fill_null(0),
        pl.col("edge_win_rate").fill_null(0.0),
        pl.col("edge_profit").fill_null(0.0),
        pl.col("edge_roi").fill_null(0.0),
    ])
    
    return monthly


# =============================================================================
# MAIN
# =============================================================================

def print_result(result: BacktestResult):
    """Pretty print a backtest result."""
    significance = "[SIGNIFICANT]" if result.is_significant else "[Not significant]"
    
    print(f"\n{'='*60}")
    print(f"  {result.strategy_name}")
    print(f"{'='*60}")
    print(f"  Total Bets:     {result.total_bets:,}")
    print(f"  Wins/Losses:    {result.wins:,} / {result.losses:,}")
    print(f"  Win Rate:       {result.win_rate:.1%}")
    print(f"  Total Profit:   {result.total_profit:+.2f} units")
    print(f"  ROI:            {result.roi:+.1%}")
    print(f"  Avg Odds:       {result.avg_odds:.2f}")
    if result.avg_edge > 0:
        print(f"  Avg Edge:       {result.avg_edge:.1%}")
    print(f"  P-value:        {result.p_value:.4f} {significance}")
    print(f"{'='*60}")


def main(args_list=None):
    parser = argparse.ArgumentParser(description="Favorite-Edge Betting Backtest")
    parser.add_argument("--min-edge", type=float, default=0.0, 
                        help="Minimum edge threshold (default: 0.0 = any positive edge)")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Minimum model confidence (default: 0.0)")
    parser.add_argument("--verify-no-leakage", action="store_true",
                        help="Run leakage verification and exit")
    args = parser.parse_args(args_list)
    
    # ==========================================================================
    # LOAD DATA
    # ==========================================================================
    
    data_path = PROCESSED_DATA_DIR / "features_dataset.parquet"
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        logger.error("Run 'python tennis.py train' to generate features first.")
        sys.exit(1)
    
    df = pl.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} total matches")
    
    # ==========================================================================
    # TEMPORAL SPLIT - CRITICAL FOR NO LEAKAGE
    # ==========================================================================
    
    cutoff_ts = int(datetime.combine(TRAIN_CUTOFF, datetime.min.time()).timestamp())
    
    # Training data (for reference only - model was trained on this)
    train_df = df.filter(pl.col("start_timestamp") < cutoff_ts)
    
    # TEST DATA - Strictly out-of-sample
    test_df = df.filter(
        (pl.col("start_timestamp") >= cutoff_ts) &
        (pl.col("odds_player").is_not_null()) &
        (pl.col("player_won").is_not_null())
    )
    
    logger.info(f"Train period: {len(train_df):,} matches (before {TRAIN_CUTOFF})")
    logger.info(f"Test period:  {len(test_df):,} matches (from {TRAIN_CUTOFF} onwards)")
    
    # ==========================================================================
    # LEAKAGE VERIFICATION
    # ==========================================================================
    
    try:
        assert_no_leakage(train_df.lazy(), test_df.lazy())
        logger.info("[OK] No temporal leakage detected")
    except LeakageError as e:
        logger.error(f"[ERROR] LEAKAGE DETECTED: {e}")
        sys.exit(1)
    
    if args.verify_no_leakage:
        logger.info("Leakage verification passed. Exiting.")
        return
    
    # ==========================================================================
    # CHECK SAMPLE SIZE
    # ==========================================================================
    
    if len(test_df) < 500:
        logger.warning(f"[WARNING] Only {len(test_df)} test matches - results may not be statistically reliable")
    else:
        logger.info(f"[OK] {len(test_df):,} test matches - sufficient for statistical analysis")
    
    # ==========================================================================
    # LOAD MODEL AND PREDICT
    # ==========================================================================
    
    registry = ModelRegistry()
    
    # Try production model first, fallback to latest
    try:
        prod_model = registry.get_production_model()
        version, model_path = prod_model
        stage = "Production"
    except RuntimeError:
        # No production model, try latest
        try:
            latest = registry.get_latest_model()
            version, model_path = latest
            stage = "Latest"
        except (RuntimeError, ValueError, TypeError):
            logger.error("No model found. Run 'python tennis.py train' first.")
            sys.exit(1)
    
    logger.info(f"Using model: {version} (stage: {stage})")
    
    predictor = Predictor(Path(model_path))
    test_df = predictor.predict_with_value(test_df, min_edge=0.0)
    
    # ==========================================================================
    # RUN BACKTESTS
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("  FAVORITE-EDGE BETTING STRATEGY BACKTEST")
    print("  Test Period:", TRAIN_CUTOFF, "onwards")
    print("  Favorite Threshold: odds <", FAVORITE_ODDS_THRESHOLD)
    print("=" * 60)
    
    # 1. Baseline: Blind favorite betting
    baseline = run_blind_favorite_backtest(test_df)
    print_result(baseline)
    
    # 2. Main strategy: Favorite with positive edge
    main_result = run_favorite_edge_backtest(
        test_df, 
        min_edge=args.min_edge,
        min_confidence=args.min_confidence
    )
    print_result(main_result)
    
    # 3. Edge buckets breakdown
    print("\n" + "=" * 60)
    print("  BREAKDOWN BY EDGE BUCKET (Favorites Only)")
    print("=" * 60)
    
    edge_breakdown = breakdown_by_edge_bucket(test_df)
    
    print(f"\n{'Edge Bucket':<18} | {'Bets':>6} | {'Win%':>6} | {'Profit':>10} | {'ROI':>8}")
    print("-" * 60)
    for row in edge_breakdown.iter_rows(named=True):
        print(f"{row['edge_bucket']:<18} | {row['bets']:>6} | {row['win_rate']:>5.1f}% | {row['profit']:>+9.1f} | {row['roi']:>+7.1f}%")
    
    # 4. Monthly breakdown for consistency check
    print("\n" + "=" * 80)
    print("  MONTHLY PERFORMANCE BREAKDOWN (Favorites Only)")
    print("=" * 80)
    
    monthly = breakdown_by_month(test_df, min_edge=args.min_edge)
    
    print(f"\n{'Month':<8} | {'Blind Bets':>10} {'Blind ROI':>10} | {'Edge Bets':>10} {'Edge ROI':>10} | {'Diff':>8}")
    print("-" * 80)
    
    cumulative_blind_profit = 0.0
    cumulative_edge_profit = 0.0
    
    for row in monthly.iter_rows(named=True):
        cumulative_blind_profit += row['blind_profit']
        cumulative_edge_profit += row['edge_profit']
        
        diff = row['edge_roi'] - row['blind_roi']
        diff_indicator = "[+]" if diff > 0 else "[-]" if diff < 0 else "[=]"
        
        print(f"{row['month']:<8} | {row['blind_bets']:>10} {row['blind_roi']:>+9.1f}% | {row['edge_bets']:>10} {row['edge_roi']:>+9.1f}% | {diff:>+7.1f}% {diff_indicator}")
    
    # Summary row
    print("-" * 80)
    total_blind_bets = monthly.select(pl.col("blind_bets").sum()).item()
    total_edge_bets = monthly.select(pl.col("edge_bets").sum()).item()
    overall_blind_roi = (cumulative_blind_profit / total_blind_bets * 100) if total_blind_bets > 0 else 0
    overall_edge_roi = (cumulative_edge_profit / total_edge_bets * 100) if total_edge_bets > 0 else 0
    overall_diff = overall_edge_roi - overall_blind_roi
    
    print(f"{'TOTAL':<8} | {total_blind_bets:>10} {overall_blind_roi:>+9.1f}% | {total_edge_bets:>10} {overall_edge_roi:>+9.1f}% | {overall_diff:>+7.1f}%")
    
    # Win rate by month
    winning_months = monthly.filter(pl.col("edge_roi") > 0).height
    total_months = len(monthly)
    print(f"\nWinning Months: {winning_months}/{total_months} ({winning_months/total_months*100:.0f}%)")
    
    # ==========================================================================
    # CONCLUSION
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("  CONCLUSION")
    print("=" * 60)
    
    if main_result.total_bets == 0:
        print("  [X] No bets matched the criteria.")
    elif main_result.roi > baseline.roi:
        improvement = main_result.roi - baseline.roi
        print(f"  [OK] Strategy improves on baseline by {improvement:+.1%} ROI")
        if main_result.is_significant:
            print(f"  [OK] Result is statistically significant (p={main_result.p_value:.4f})")
        else:
            print(f"  [!] Result is NOT statistically significant (p={main_result.p_value:.4f})")
            print("     More data needed to confirm if this isn't just luck.")
    else:
        print(f"  [X] Strategy underperforms baseline by {main_result.roi - baseline.roi:+.1%} ROI")
    
    # Save results
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pl.DataFrame([
        {"strategy": baseline.strategy_name, "bets": baseline.total_bets, 
         "win_rate": baseline.win_rate, "roi": baseline.roi, "p_value": baseline.p_value},
        {"strategy": main_result.strategy_name, "bets": main_result.total_bets,
         "win_rate": main_result.win_rate, "roi": main_result.roi, "p_value": main_result.p_value},
    ])
    output_path = BACKTEST_DIR / "favorite_edge_results.parquet"
    results_df.write_parquet(output_path)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
