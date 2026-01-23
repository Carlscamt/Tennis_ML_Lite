"""
Unified Tennis Prediction Pipeline.

Single entry point for all prediction workflows:
- Daily updates
- Live predictions
- Full data refresh

Usage:
    from src.pipeline import TennisPipeline, run_data_pipeline, run_training_pipeline
    
    pipeline = TennisPipeline()
    predictions = pipeline.predict_upcoming(days=7)
"""
import sys
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Dict
import logging
import polars as pl

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
    FEATURES, MODEL, BETTING
)
from src.extract import load_all_parquet_files
from src.extract.data_loader import prepare_base_dataset, get_dataset_stats
from src.transform import FeatureEngineer, create_train_test_split, DataValidator
from src.transform.leakage_guard import validate_temporal_order, assert_no_leakage
from src.model import Predictor, ModelTrainer, ModelRegistry
from src.scraper import scrape_upcoming, scrape_players
from src.schema import merge_datasets, enforce_schema, SchemaValidator

logger = logging.getLogger(__name__)


# =============================================================================
# CORE PIPELINES (Moved from scripts/run_pipeline.py)
# =============================================================================

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
    # Check if odds column exists before sorting
    schema = df.collect_schema().names()
    if "odds_player" in schema:
        df = df.with_columns([
            pl.col("odds_player").is_not_null().cast(pl.Int8).alias("_has_odds")
        ]).sort(["start_timestamp", "_has_odds"], descending=[False, True])
        df = df.unique(subset=["event_id", "player_id"], keep="first", maintain_order=True)
        df = df.drop("_has_odds")
    else:
        df = df.unique(subset=["event_id", "player_id"], keep="first", maintain_order=True)
        df = df.sort("start_timestamp")
    
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


# =============================================================================
# ORCHESTRATOR CLASS 
# =============================================================================

class TennisPipeline:
    """
    Unified pipeline for tennis match predictions.
    
    Provides high-level methods for common workflows:
    - predict_upcoming: Get predictions for upcoming matches
    - daily_update: Full daily update workflow
    - refresh_all: Complete data refresh and model retraining
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        model_path: Optional[Path] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            data_dir: Root data directory (defaults to project data/)
            model_path: Path to trained model (defaults to models/xgboost_model)
        """
        self.root = ROOT
        self.data_dir = data_dir or (ROOT / "data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.future_dir = self.data_dir / "future"
        self.models_dir = ROOT / "models"
        
        self.model_path = model_path or (self.models_dir / "xgboost_model")
        
        self.feature_engineer = FeatureEngineer()
        self._predictor = None
    
    @property
    def predictor(self) -> Predictor:
        """Lazy-load the predictor."""
        if self._predictor is None:
            self._predictor = Predictor(self.model_path)
        return self._predictor
    
    # =========================================================================
    # MAIN WORKFLOWS
    # =========================================================================
    
    def predict_upcoming(
        self,
        days: int = 7,
        min_odds: float = 1.5,
        max_odds: float = 3.0,
        min_confidence: float = 0.55,
        scrape_unknown: bool = True
    ) -> pl.DataFrame:
        """
        Get predictions for upcoming matches.
        """
        logger.info(f"=== PREDICT UPCOMING ({days} days) ===")
        
        # Step 1: Get upcoming matches
        upcoming = self._get_upcoming_matches(days)
        if len(upcoming) == 0:
            logger.warning("No upcoming matches found")
            return pl.DataFrame()
        
        logger.info(f"Found {len(upcoming)} upcoming matches")
        
        # Step 2: Load historical data
        historical = self._load_historical_data()
        if len(historical) == 0:
            logger.error("No historical data found")
            return pl.DataFrame()
        
        logger.info(f"Loaded {len(historical)} historical matches")
        
        # Step 2.5: Check for unknown players and scrape if enabled
        if scrape_unknown:
            unknown_players = self._identify_unknown_players(upcoming, historical)
            if unknown_players:
                logger.info(f"Auto-scraping {len(unknown_players)} unknown players...")
                # Note: scrape_players now has smart_update, but for new players we force it effectively
                # by passing IDs that aren't in history
                self._scrape_unknown_players(unknown_players)
                
                # Reload history
                historical = self._load_historical_data()
                logger.info(f"Historical data now has {len(historical)} matches")
        
        # Step 3: Compute features for prediction
        prediction_ready = self.feature_engineer.compute_features_for_prediction(
            upcoming, historical
        )
        
        # Step 4: Make predictions
        predictions = self.predictor.predict_with_value(prediction_ready)
        
        # Step 5: Filter to recommended bets
        recommended = predictions.filter(
            (pl.col("model_prob") >= min_confidence) &
            (pl.col("odds_player") >= min_odds) &
            (pl.col("odds_player") <= max_odds) &
            (pl.col("edge") > 0.05)
        ).sort("edge", descending=True)
        
        logger.info(f"Found {len(recommended)} value bets from {len(predictions)} matches")
        
        return predictions.with_columns([
            pl.lit(datetime.now().isoformat()).alias("prediction_timestamp")
        ])
    
    def daily_update(self, scrape_future: bool = True) -> Dict:
        """
        Full daily update workflow.
        """
        logger.info("=== DAILY UPDATE ===")
        results = {}
        
        # Step 1: Scrape upcoming matches
        if scrape_future:
            logger.info("Step 1: Scraping upcoming matches...")
            # Use src.scraper direct call
            upcoming_df = scrape_upcoming(days_ahead=7)
            results["upcoming_matches"] = len(upcoming_df)
        else:
            # Load existing
            upcoming_df = pl.read_parquet(self.future_dir / "upcoming_matches.parquet") \
                if (self.future_dir / "upcoming_matches.parquet").exists() else pl.DataFrame()

        # Step 2: Update active players
        if len(upcoming_df) > 0:
            logger.info("Step 2: Updating active player history...")
            active_ids = self._get_active_player_ids(upcoming_df)
            logger.info(f"Found {len(active_ids)} active players")
            
            # Use src.scraper.scrape_players with SMART UPDATE enabled
            scrape_players(
                player_ids=active_ids,
                max_pages=3,
                workers=3,
                smart_update=True
            )
        else:
            logger.info("Skipping player update (no upcoming matches)")
        
        # Step 3: Run feature pipeline (Data Pipeline)
        logger.info("Step 3: Regenerating features...")
        run_data_pipeline(self.raw_dir, self.processed_dir)
        
        # Step 4: Make predictions
        logger.info("Step 4: Making predictions...")
        # Force reload of predictor to ensure it uses any new data/models if applicable?
        # Actually predictor uses saved model, so unless we retrained, model is same.
        predictions = self.predict_upcoming(days=7, scrape_unknown=False) # Already scraped
        results["predictions"] = len(predictions)
        
        # Save predictions
        output_path = self.data_dir / "predictions" / f"daily_{date.today().isoformat()}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.write_parquet(output_path)
        results["saved_to"] = str(output_path)
        
        logger.info(f"Daily update complete: {results}")
        return results
    
    def refresh_all(self, retrain: bool = True) -> Dict:
        """
        Complete data refresh and optional model retraining.
        """
        logger.info("=== FULL REFRESH ===")
        results = {}
        
        # Step 1: Run data pipeline
        logger.info("Step 1: Running data pipeline...")
        df = run_data_pipeline(self.raw_dir, self.processed_dir)
        results["total_matches"] = len(df.collect()) if hasattr(df, 'collect') else len(df)
        
        # Step 2: Retrain model
        if retrain:
            logger.info("Step 2: Retraining model...")
            data_path = self.processed_dir / "features_dataset.parquet"
            run_training_pipeline(data_path, self.models_dir)
            results["model_retrained"] = True
            
            # Reset predictor
            self._predictor = None
        
        logger.info(f"Refresh complete: {results}")
        return results
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _get_upcoming_matches(self, days: int) -> pl.DataFrame:
        """Load or scrape upcoming matches."""
        # Try loading from file first
        latest_path = self.data_dir / "upcoming.parquet"
        
        if latest_path.exists():
            # Check if recent (< 1 hour old)
            import os
            mtime = os.path.getmtime(latest_path)
            age_hours = (datetime.now().timestamp() - mtime) / 3600
            
            if age_hours < 1:
                logger.info(f"Using cached upcoming matches (age: {age_hours:.1f}h)")
                return pl.read_parquet(latest_path)
        
        # Need to scrape
        logger.info("Scraping fresh upcoming matches...")
        return scrape_upcoming(days_ahead=days)
    
    def _load_historical_data(self) -> pl.DataFrame:
        """Load the latest processed historical data."""
        # Try processed data first
        processed_path = self.processed_dir / "features_dataset.parquet"
        
        if processed_path.exists():
            return pl.read_parquet(processed_path)
        
        # Fall back to unified raw data
        # Note: Scraper now saves to DATA_DIR/tennis.parquet by default or similar?
        # Checked src/scraper.py OUTPUT_FILE = DATA_DIR / "tennis.parquet"
        
        # But pipeline was using data/raw/atp_matches_*.parquet
        # Let's support both
        
        unified_path = self.data_dir / "tennis.parquet"
        if unified_path.exists():
            return pl.read_parquet(unified_path)
            
        # Fall back to most recent raw file
        raw_files = sorted(self.raw_dir.glob("atp_matches_*.parquet"))
        if raw_files:
            return pl.read_parquet(raw_files[-1])
        
        return pl.DataFrame()
    
    def _get_active_player_ids(self, upcoming_df: pl.DataFrame) -> List[int]:
        """Extract all player IDs from upcoming matches."""
        ids = set()
        if "player_id" in upcoming_df.columns:
            ids.update(upcoming_df["player_id"].drop_nulls().unique().to_list())
        if "opponent_id" in upcoming_df.columns:
            ids.update(upcoming_df["opponent_id"].drop_nulls().unique().to_list())
        return list(ids)

    def _identify_unknown_players(
        self, 
        upcoming: pl.DataFrame, 
        historical: pl.DataFrame
    ) -> List[int]:
        """Identify players in upcoming matches who don't have historical data."""
        upcoming_players = self._get_active_player_ids(upcoming)
        
        known_players = set()
        if len(historical) > 0 and "player_id" in historical.columns:
            known_players = set(historical["player_id"].unique().to_list())
        
        # Find unknown players
        unknown_players = [pid for pid in upcoming_players if pid not in known_players]
        
        if unknown_players:
            logger.info(f"Found {len(unknown_players)} unknown players in upcoming matches")
        
        return unknown_players
    
    def _scrape_unknown_players(self, player_ids: List[int]) -> None:
        """Scrape match history for unknown players."""
        if not player_ids:
            return
        
        logger.info(f"Scraping history for {len(player_ids)} unknown players...")
        # Use src.scraper direclty
        try:
            scrape_players(
                player_ids=player_ids,
                max_pages=5,
                workers=3,
                smart_update=False # Force update for newbies
            )
        except Exception as e:
            logger.error(f"Error scraping unknown players: {e}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def predict_today(min_odds: float = 1.5, max_odds: float = 3.0) -> pl.DataFrame:
    """Quick function to get today's predictions."""
    pipeline = TennisPipeline()
    return pipeline.predict_upcoming(days=1, min_odds=min_odds, max_odds=max_odds)


def get_value_bets(days: int = 7) -> pl.DataFrame:
    """Get value bets for the next N days."""
    pipeline = TennisPipeline()
    predictions = pipeline.predict_upcoming(days=days)
    
    return predictions.filter(
        pl.col("is_value_bet") == True
    ).sort("edge", descending=True)
