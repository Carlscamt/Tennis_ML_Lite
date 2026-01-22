"""
Backtesting Framework

Simulates betting strategy on historical data.

Usage:
    python scripts/backtest.py
    python scripts/backtest.py --start-date 2024-01-01 --end-date 2024-12-31
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import polars as pl
from datetime import date

from config import PROCESSED_DATA_DIR, MODELS_DIR, BACKTEST_DIR, BETTING
from src.model import Predictor, ModelRegistry
from src.betting import BankrollManager, ValueBetFinder
from src.utils import setup_logging

logger = setup_logging()


def run_backtest(
    data_path: Path,
    model_path: Path,
    start_date: date,
    end_date: date = None,
    initial_bankroll: float = 1000.0,
) -> dict:
    """
    Run backtest simulation.
    
    Args:
        data_path: Path to processed dataset with features
        model_path: Path to trained model
        start_date: Backtest start date
        end_date: Backtest end date (defaults to latest data)
        initial_bankroll: Starting bankroll
        
    Returns:
        Dict with backtest results
    """
    logger.info("=" * 60)
    logger.info("BACKTESTING")
    logger.info("=" * 60)
    
    # Load data
    df = pl.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} matches")
    
    # Filter date range using datetime.timestamp()
    from datetime import datetime
    start_dt = datetime.combine(start_date, datetime.min.time())
    start_ts = int(start_dt.timestamp())
    df = df.filter(pl.col("start_timestamp") >= start_ts)
    
    if end_date:
        end_dt = datetime.combine(end_date, datetime.min.time())
        end_ts = int(end_dt.timestamp())
        df = df.filter(pl.col("start_timestamp") <= end_ts)
    
    logger.info(f"Backtest period: {len(df):,} matches")
    
    # Filter to matches with odds
    df = df.filter(pl.col("odds_player").is_not_null())
    logger.info(f"With odds: {len(df):,} matches")
    
    # Load model and predict
    predictor = Predictor(model_path)
    df = predictor.predict_with_value(df, min_edge=BETTING.min_edge)
    
    # Initialize betting components
    bankroll = BankrollManager(
        initial_bankroll=initial_bankroll,
        kelly_fraction=BETTING.kelly_fraction,
        max_stake_pct=BETTING.max_stake_pct,
        min_odds=BETTING.min_odds,
        max_odds=BETTING.max_odds,
    )
    
    finder = ValueBetFinder(
        min_edge=BETTING.min_edge,
        min_confidence=BETTING.min_confidence,
        min_odds=BETTING.min_odds,
        max_odds=BETTING.max_odds,
    )
    
    # Find value bets
    value_bets = finder.find_value_bets(df)
    logger.info(f"Value bets identified: {len(value_bets):,}")
    
    if len(value_bets) == 0:
        logger.warning("No value bets found!")
        return {"total_bets": 0}
    
    # Simulate betting
    for row in value_bets.iter_rows(named=True):
        prob = row["model_prob"]
        odds = row["odds_player"]
        won = row["player_won"]
        
        stake_amount = bankroll.calculate_stake_amount(prob, odds)
        
        if stake_amount > 0:
            bankroll.place_bet(stake_amount, odds, won)
    
    # Get results
    stats = bankroll.get_stats()
    
    # Additional stats from value bets
    avg_edge = value_bets.select(pl.col("edge").mean()).item()
    avg_conf = value_bets.select(pl.col("model_prob").mean()).item()
    avg_odds = value_bets.select(pl.col("odds_player").mean()).item()
    
    results = {
        **stats,
        "period_start": str(start_date),
        "period_end": str(end_date) if end_date else "latest",
        "avg_edge": avg_edge,
        "avg_confidence": avg_conf,
        "avg_odds": avg_odds,
    }
    
    # Log results
    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total bets: {stats['total_bets']}")
    logger.info(f"Win rate: {stats['win_rate']:.1%}")
    logger.info(f"Total profit: ${stats['total_profit']:.2f}")
    logger.info(f"ROI: {stats['roi']:.1%}")
    logger.info(f"Final bankroll: ${stats['current_bankroll']:.2f}")
    logger.info(f"Growth: {stats['growth']:.1%}")
    logger.info("=" * 60)
    
    return results


def compare_strategies(
    data_path: Path,
    model_path: Path,
    start_date: date,
) -> pl.DataFrame:
    """
    Compare different betting strategies.
    """
    strategies = [
        {"name": "Conservative", "kelly": 0.10, "min_edge": 0.07, "min_conf": 0.60},
        {"name": "Moderate", "kelly": 0.25, "min_edge": 0.05, "min_conf": 0.55},
        {"name": "Aggressive", "kelly": 0.50, "min_edge": 0.03, "min_conf": 0.52},
    ]
    
    results = []
    
    for strat in strategies:
        # Temporarily override settings
        original_kelly = BETTING.kelly_fraction
        original_edge = BETTING.min_edge
        original_conf = BETTING.min_confidence
        
        BETTING.kelly_fraction = strat["kelly"]
        BETTING.min_edge = strat["min_edge"]
        BETTING.min_confidence = strat["min_conf"]
        
        try:
            result = run_backtest(data_path, model_path, start_date)
            result["strategy"] = strat["name"]
            results.append(result)
        finally:
            # Restore settings
            BETTING.kelly_fraction = original_kelly
            BETTING.min_edge = original_edge
            BETTING.min_confidence = original_conf
    
    return pl.DataFrame(results)


def main(args_list=None):
    parser = argparse.ArgumentParser(description="Tennis Betting Backtest")
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--compare", action="store_true", help="Compare strategies")
    args = parser.parse_args(args_list)
    
    data_path = PROCESSED_DATA_DIR / "features_dataset.parquet"
    
    # Get active model
    registry = ModelRegistry(MODELS_DIR)
    active = registry.get_active_model()
    
    if not active:
        logger.error("No active model found. Run training first.")
        sys.exit(1)
    
    model_path = Path(active["path"])
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date) if args.end_date else None
    
    if args.compare:
        results = compare_strategies(data_path, model_path, start_date)
        print("\nStrategy Comparison:")
        print(results)
        
        # Save results
        BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
        output_path = BACKTEST_DIR / "strategy_comparison.parquet"
        results.write_parquet(output_path)
        logger.info(f"Saved comparison to {output_path}")
    else:
        run_backtest(
            data_path,
            model_path,
            start_date,
            end_date,
            args.bankroll
        )


if __name__ == "__main__":
    main()
