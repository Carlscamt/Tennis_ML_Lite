"""
Script to optimize model hyperparameters using Optuna Bayesian optimization.

Usage:
    python scripts/optimize_model.py --n-trials 50 --timeout 3600 --objective composite
    python scripts/optimize_model.py --n-trials 10 --objective log_loss  # Quick test
"""
import sys
import argparse
from pathlib import Path

# Add root to pythonpath
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import logging
import polars as pl
from config import PROCESSED_DATA_DIR, MODELS_DIR, settings
from src.utils import setup_logging
from src.transform.features import FeatureEngineer

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize XGBoost hyperparameters with Optuna")
    
    parser.add_argument(
        "--n-trials", type=int,
        default=settings.OPTIMIZATION.n_trials,
        help="Number of optimization trials (default: 50)"
    )
    parser.add_argument(
        "--timeout", type=int,
        default=settings.OPTIMIZATION.timeout_seconds,
        help="Maximum time in seconds (default: 3600)"
    )
    parser.add_argument(
        "--objective", type=str,
        default=settings.OPTIMIZATION.objective,
        choices=["composite", "log_loss", "roi"],
        help="Objective to optimize (default: composite)"
    )
    parser.add_argument(
        "--variance-penalty", type=float,
        default=settings.OPTIMIZATION.variance_penalty,
        help="Penalty for ROI variance in composite objective (default: 0.1)"
    )
    parser.add_argument(
        "--n-jobs", type=int,
        default=settings.OPTIMIZATION.n_jobs,
        help="Number of parallel trials (default: 1)"
    )
    parser.add_argument(
        "--storage", type=str,
        default=settings.OPTIMIZATION.storage or None,
        help="Optuna storage URL (e.g., sqlite:///optuna.db)"
    )
    parser.add_argument(
        "--study-name", type=str,
        default="tennis_xgb",
        help="Name for the Optuna study"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check Optuna availability
    try:
        from src.model.optuna_optimizer import OptunaOptimizer
    except ImportError as e:
        logger.error(f"Failed to import OptunaOptimizer: {e}")
        logger.error("Install Optuna with: pip install optuna")
        sys.exit(1)
    
    logger.info("Loading dataset...")
    df = pl.read_parquet(PROCESSED_DATA_DIR / "features_dataset.parquet")
    
    # Sort by date for time-series CV
    df = df.sort("start_timestamp")
    
    # Rename for compatibility
    if "start_timestamp" in df.columns and "match_date" not in df.columns:
        df = df.rename({"start_timestamp": "match_date"})
    
    # Filter for matches with odds AND outcomes
    df = df.filter(
        pl.col("odds_player").is_not_null() & 
        pl.col("odds_opponent").is_not_null() &
        pl.col("player_won").is_not_null()
    )
    logger.info(f"Dataset ready. {len(df)} matches with odds.")
    
    # Get feature columns
    feature_cols = FeatureEngineer().get_feature_columns(df.lazy())
    
    # Create optimizer
    optimizer = OptunaOptimizer(
        feature_cols=feature_cols,
        target_col="player_won",
        n_splits=5,
        gap_days=7,
        variance_penalty=args.variance_penalty,
        objective_type=args.objective,
    )
    
    # Output path
    save_path = MODELS_DIR / "best_params.json"
    
    # Run optimization
    result = optimizer.optimize(
        df=df,
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=args.study_name,
        storage=args.storage,
        n_jobs=args.n_jobs,
        save_path=save_path,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best Score: {result.best_score:.4f}")
    print(f"Objective: {result.objective_type}")
    print(f"Trials: {result.n_trials}")
    print(f"\nBest Parameters saved to: {save_path}")
    print("-" * 60)
    for k, v in result.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
