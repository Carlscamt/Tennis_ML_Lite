"""
Unified Tennis Prediction Pipeline.

Single entry point for all prediction workflows:
- Daily updates
- Live predictions
- Full data refresh

Usage:
    from src.pipeline import TennisPipeline
    
    pipeline = TennisPipeline()
    predictions = pipeline.predict_upcoming(days=7)
"""
import sys
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Dict
import logging

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import polars as pl

from src.transform.features import FeatureEngineer
from src.model import Predictor, ModelTrainer

logger = logging.getLogger(__name__)


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
        
        This is the main prediction workflow:
        1. Scrape upcoming matches
        2. Load historical data
        3. Check for unknown players and scrape them (if enabled)
        4. Compute features
        5. Make predictions
        6. Filter to value bets
        
        Args:
            days: Days ahead to look
            min_odds: Minimum odds for recommendations
            max_odds: Maximum odds for recommendations
            min_confidence: Minimum model probability
            scrape_unknown: Whether to auto-scrape data for unknown players
            
        Returns:
            DataFrame with predictions and value bets
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
                historical = self._scrape_unknown_players(
                    player_ids=unknown_players,
                    existing_df=historical,
                    max_pages=5
                )
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
            (pl.col("edge") > 0.05)  # At least 5% edge
        ).sort("edge", descending=True)
        
        logger.info(f"Found {len(recommended)} value bets from {len(predictions)} matches")
        
        return predictions.with_columns([
            pl.lit(datetime.now().isoformat()).alias("prediction_timestamp")
        ])
    
    def daily_update(self, scrape_future: bool = True) -> Dict:
        """
        Full daily update workflow.
        
        1. Scrape upcoming matches (optional)
        2. Update active player history
        3. Regenerate features
        4. Make predictions
        
        Args:
            scrape_future: Whether to scrape future matches first
            
        Returns:
            Dict with summary statistics
        """
        from scripts.update_active_players import main as update_players
        from scripts.scrape_future import scrape_future_matches
        
        logger.info("=== DAILY UPDATE ===")
        results = {}
        
        # Step 1: Scrape upcoming matches
        if scrape_future:
            logger.info("Step 1: Scraping upcoming matches...")
            upcoming_df = scrape_future_matches(days_ahead=7)
            results["upcoming_matches"] = len(upcoming_df)
        
        # Step 2: Update active players
        logger.info("Step 2: Updating active player history...")
        update_players()
        
        # Step 3: Run feature pipeline
        logger.info("Step 3: Regenerating features...")
        self._run_feature_pipeline()
        
        # Step 4: Make predictions
        logger.info("Step 4: Making predictions...")
        predictions = self.predict_upcoming(days=7)
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
        
        1. Scrape all ATP data
        2. Run full feature engineering
        3. Retrain model (optional)
        4. Run audit
        
        Args:
            retrain: Whether to retrain the model
            
        Returns:
            Dict with summary
        """
        from scripts.run_pipeline import run_data_pipeline, run_training_pipeline
        from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
        
        logger.info("=== FULL REFRESH ===")
        results = {}
        
        # Step 1: Run data pipeline
        logger.info("Step 1: Running data pipeline...")
        df = run_data_pipeline(RAW_DATA_DIR, PROCESSED_DATA_DIR)
        results["total_matches"] = len(df.collect()) if hasattr(df, 'collect') else len(df)
        
        # Step 2: Retrain model
        if retrain:
            logger.info("Step 2: Retraining model...")
            data_path = PROCESSED_DATA_DIR / "features_dataset.parquet"
            run_training_pipeline(data_path, MODELS_DIR)
            results["model_retrained"] = True
            
            # Reload predictor
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
        from src.scraper import scrape_upcoming
        return scrape_upcoming(days_ahead=days)
    
    def _load_historical_data(self) -> pl.DataFrame:
        """Load the latest processed historical data."""
        # Try processed data first
        processed_path = self.processed_dir / "features_dataset.parquet"
        
        if processed_path.exists():
            return pl.read_parquet(processed_path)
        
        # Fall back to raw data
        raw_files = sorted(self.raw_dir.glob("atp_matches_*.parquet"))
        if raw_files:
            return pl.read_parquet(raw_files[-1])
        
        return pl.DataFrame()
    
    def _identify_unknown_players(
        self, 
        upcoming: pl.DataFrame, 
        historical: pl.DataFrame
    ) -> List[int]:
        """
        Identify players in upcoming matches who don't have historical data.
        
        Args:
            upcoming: DataFrame of upcoming matches
            historical: DataFrame of historical match data
            
        Returns:
            List of player IDs with no historical data
        """
        # Get all player IDs from upcoming matches
        upcoming_players = set()
        
        if "player_id" in upcoming.columns:
            upcoming_players.update(upcoming["player_id"].unique().to_list())
        
        # Also check for opponent IDs
        if "opponent_id" in upcoming.columns:
            upcoming_players.update(upcoming["opponent_id"].unique().to_list())
        
        # Get players we have historical data for
        known_players = set()
        if len(historical) > 0 and "player_id" in historical.columns:
            known_players = set(historical["player_id"].unique().to_list())
        
        # Find unknown players
        unknown_players = [pid for pid in upcoming_players if pid not in known_players and pid is not None]
        
        if unknown_players:
            logger.info(f"Found {len(unknown_players)} unknown players in upcoming matches")
        
        return unknown_players
    
    def _scrape_unknown_players(
        self, 
        player_ids: List[int], 
        existing_df: pl.DataFrame,
        max_pages: int = 5
    ) -> pl.DataFrame:
        """
        Scrape match history for unknown players and merge with existing data.
        
        Args:
            player_ids: List of player IDs to scrape
            existing_df: Existing historical data
            max_pages: Maximum pages of history to fetch per player
            
        Returns:
            Updated DataFrame with new player data merged in
        """
        if not player_ids:
            return existing_df
        
        logger.info(f"Scraping history for {len(player_ids)} unknown players...")
        
        try:
            from scripts.update_active_players import update_player_data
            from src.schema import merge_datasets, enforce_schema, SchemaValidator
            
            # Scrape the new players
            new_data = update_player_data(
                player_ids=player_ids,
                existing_df=None,  # We handle merging ourselves
                max_pages=max_pages,
                parallel_workers=3,
                smart_update=False  # Force update since these are new players
            )
            
            if new_data is not None and len(new_data) > 0:
                logger.info(f"Scraped {len(new_data)} matches for unknown players")
                
                # Merge with existing data
                merged = merge_datasets(
                    new_df=new_data,
                    existing_df=existing_df,
                    prefer_new=True,
                    prefer_odds=True
                )
                
                # Validate and enforce schema
                merged = enforce_schema(merged, fill_missing=True)
                
                # Run validation
                validator = SchemaValidator(merged)
                issues = validator.validate_all()
                if issues:
                    logger.warning(f"Schema validation issues: {issues}")
                
                # Save the updated raw data
                output_path = self.raw_dir / f"atp_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                merged.write_parquet(output_path)
                logger.info(f"Saved updated data to {output_path}")
                
                return merged
            else:
                logger.warning("No data scraped for unknown players")
                return existing_df
                
        except Exception as e:
            logger.error(f"Error scraping unknown players: {e}")
            return existing_df
    
    def _run_feature_pipeline(self):
        """Regenerate features from raw data."""
        from scripts.run_pipeline import run_data_pipeline
        from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
        
        run_data_pipeline(RAW_DATA_DIR, PROCESSED_DATA_DIR)


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
