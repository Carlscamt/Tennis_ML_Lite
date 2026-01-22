"""
Full ETL + Training Pipeline

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --data-only
    python scripts/run_pipeline.py --train-only
"""
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import polars as pl
from datetime import date

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
    FEATURES, MODEL, BETTING
)
from src.extract import load_all_parquet_files
from src.extract.data_loader import prepare_base_dataset, get_dataset_stats
from src.transform import FeatureEngineer, create_train_test_split, DataValidator
from src.transform.leakage_guard import validate_temporal_order, assert_no_leakage
from src.model import ModelTrainer, ModelRegistry
from src.utils import setup_logging

logger = setup_logging()


def run_data_pipeline(raw_dir: Path, output_dir: Path) -> pl.LazyFrame:
    """
    Run data extraction and feature engineering.
    
    Args:
        raw_dir: Directory with raw parquet files
        output_dir: Directory to save processed data
        
    Returns:
        Processed LazyFrame
    """
    logger.info("=" * 60)
    logger.info("DATA PIPELINE")
    logger.info("=" * 60)
    
    # Load raw data
    logger.info(f"Loading data from {raw_dir}")
    df = load_all_parquet_files(raw_dir)
    
    # Prepare base dataset
    df = prepare_base_dataset(df)
    
    # CRITICAL: Deduplicate to prevent data leakage via rolling windows
    # Each event_id + player_id combination should appear exactly once
    # Prefer rows WITH odds data (more complete)
    df = df.with_columns([
        pl.col("odds_player").is_not_null().cast(pl.Int8).alias("_has_odds")
    ]).sort(["start_timestamp", "_has_odds"], descending=[False, True])  # Sort by time, then odds first
    df = df.unique(subset=["event_id", "player_id"], keep="first", maintain_order=True)
    df = df.drop("_has_odds")
    
    # Validate temporal order
    validate_temporal_order(df)
    
    # Get stats
    stats = get_dataset_stats(df)
    logger.info(f"Loaded {stats['total_matches']:,} matches (after deduplication)")
    logger.info(f"Date range: {stats['earliest_match']} to {stats['latest_match']}")
    if "odds_coverage" in stats:
        logger.info(f"Odds coverage: {stats['odds_coverage']:.1%}")
    
    # Feature engineering
    logger.info("Engineering features...")
    fe = FeatureEngineer(
        rolling_windows=FEATURES.rolling_windows,
        min_matches=FEATURES.min_matches_for_stats,
        elo_k=FEATURES.elo_k_factor,
    )
    df = fe.add_all_features(df)
    
    # Validate data quality
    validator = DataValidator(min_odds_coverage=MODEL.min_odds_coverage)
    if not validator.validate_all(df):
        logger.warning("Data validation failed - check warnings above")
    
    # Save processed data
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "features_dataset.parquet"
    df.collect().write_parquet(output_path)
    logger.info(f"Saved processed data to {output_path}")
    
    return df


def run_training_pipeline(
    data_path: Path,
    models_dir: Path,
    cutoff_date: date = None
) -> None:
    """
    Run model training pipeline.
    
    Args:
        data_path: Path to processed dataset
        models_dir: Directory to save models
        cutoff_date: Train/test split date
    """
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE")
    logger.info("=" * 60)
    
    cutoff_date = cutoff_date or MODEL.train_cutoff_date
    
    # Load processed data
    logger.info(f"Loading data from {data_path}")
    df = pl.scan_parquet(data_path)
    
    # Train/test split
    train_df, test_df = create_train_test_split(df, cutoff_date)
    
    # Assert no leakage
    assert_no_leakage(train_df, test_df)
    
    # Collect to memory
    train_data = train_df.collect()
    test_data = test_df.collect()
    
    logger.info(f"Train: {len(train_data):,} samples")
    logger.info(f"Test: {len(test_data):,} samples")
    
    # Get feature columns
    fe = FeatureEngineer()
    feature_cols = fe.get_feature_columns(train_df)
    logger.info(f"Features: {len(feature_cols)}")
    
    # Filter to features that exist AND are numeric
    existing_cols = [c for c in feature_cols if c in train_data.columns]
    
    # Filter to numeric types only
    numeric_types = [pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                     pl.Float32, pl.Float64, pl.Boolean]
    numeric_cols = [
        c for c in existing_cols 
        if train_data[c].dtype in numeric_types
    ]
    logger.info(f"Using {len(numeric_cols)} numeric features")
    
    # Train model
    trainer = ModelTrainer(params=MODEL.xgb_params, calibrate=True)
    result = trainer.train(
        train_data,
        feature_cols=numeric_cols,
        eval_df=test_data
    )
    
    # Log metrics
    logger.info("Metrics:")
    for key, value in result.metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Top features
    logger.info("Top 10 Features:")
    for i, (feat, imp) in enumerate(list(result.feature_importance.items())[:10]):
        logger.info(f"  {i+1}. {feat}: {imp:.4f}")
    
    # Save model
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "xgboost_model"
    trainer.save(model_path)
    
    # Register model
    registry = ModelRegistry(models_dir)
    version = registry.register(
        model_path,
        metrics=result.metrics,
        description=f"Trained on data up to {cutoff_date}",
        set_active=True
    )
    
    logger.info(f"Model registered: {version}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Tennis Betting ML Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Legacy arguments (for backward compatibility)
    parser.add_argument("--data-only", action="store_true", help="Run data pipeline only")
    parser.add_argument("--train-only", action="store_true", help="Run training only")
    parser.add_argument("--cutoff", type=str, help="Train/test cutoff date (YYYY-MM-DD)")
    
    # New commands
    predict_parser = subparsers.add_parser("predict", help="Get predictions for upcoming matches")
    predict_parser.add_argument("--days", type=int, default=7, help="Days ahead to look")
    predict_parser.add_argument("--min-odds", type=float, default=1.5, help="Minimum odds")
    predict_parser.add_argument("--max-odds", type=float, default=3.0, help="Maximum odds")
    
    daily_parser = subparsers.add_parser("daily", help="Run daily update workflow")
    daily_parser.add_argument("--skip-scrape", action="store_true", help="Skip scraping future matches")
    
    refresh_parser = subparsers.add_parser("refresh", help="Full data refresh and retrain")
    refresh_parser.add_argument("--no-retrain", action="store_true", help="Skip model retraining")
    
    args = parser.parse_args()
    
    # Handle new commands
    if args.command == "predict":
        from src.pipeline import TennisPipeline
        pipeline = TennisPipeline()
        predictions = pipeline.predict_upcoming(
            days=args.days,
            min_odds=args.min_odds,
            max_odds=args.max_odds
        )
        
        # Show summary
        value_bets = predictions.filter(pl.col("edge") > 0.05) if "edge" in predictions.columns else predictions
        logger.info(f"Total predictions: {len(predictions)}")
        logger.info(f"Value bets (>5% edge): {len(value_bets)}")
        
        if len(value_bets) > 0:
            logger.info("\nTop 5 Value Bets:")
            top5 = value_bets.head(5)
            for row in top5.iter_rows(named=True):
                logger.info(f"  {row.get('player_name', 'N/A')} vs {row.get('opponent_name', 'N/A')}")
                logger.info(f"    Odds: {row.get('odds_player', 'N/A'):.2f} | Prob: {row.get('model_prob', 0):.1%} | Edge: {row.get('edge', 0):.1%}")
        
        return
    
    elif args.command == "daily":
        from src.pipeline import TennisPipeline
        pipeline = TennisPipeline()
        results = pipeline.daily_update(scrape_future=not args.skip_scrape)
        logger.info(f"Daily update complete: {results}")
        return
    
    elif args.command == "refresh":
        from src.pipeline import TennisPipeline
        pipeline = TennisPipeline()
        results = pipeline.refresh_all(retrain=not args.no_retrain)
        logger.info(f"Refresh complete: {results}")
        return
    
    # Legacy behavior (backward compatibility)
    cutoff = None
    if args.cutoff:
        cutoff = date.fromisoformat(args.cutoff)
    
    if args.train_only:
        data_path = PROCESSED_DATA_DIR / "features_dataset.parquet"
        if not data_path.exists():
            logger.error(f"Processed data not found: {data_path}")
            logger.error("Run with --data-only first")
            sys.exit(1)
        run_training_pipeline(data_path, MODELS_DIR, cutoff)
    elif args.data_only:
        run_data_pipeline(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    else:
        # Full pipeline (default when no command given)
        run_data_pipeline(RAW_DATA_DIR, PROCESSED_DATA_DIR)
        data_path = PROCESSED_DATA_DIR / "features_dataset.parquet"
        run_training_pipeline(data_path, MODELS_DIR, cutoff)
    
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()

