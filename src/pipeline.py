"""
Unified Tennis Prediction Pipeline with Observability.
"""
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import polars as pl

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
    FEATURES, MODEL
)
from src.extract import load_all_parquet_files
from src.extract.data_loader import prepare_base_dataset, get_dataset_stats
from src.transform import FeatureEngineer, create_train_test_split, DataValidator
from src.transform.leakage_guard import validate_temporal_order, assert_no_leakage
from src.model import Predictor, ModelTrainer, ModelRegistry
from src.scraper import scrape_upcoming, scrape_players
from src.utils.observability import get_metrics, Logger, CORRELATION_ID

logger = Logger(__name__)
metrics = get_metrics()

class TennisPipeline:
    """
    Unified pipeline for tennis match predictions with production-grade observability.
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        model_path: Optional[Path] = None
    ):
        self.root = ROOT
        self.data_dir = data_dir or (ROOT / "data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.future_dir = self.data_dir / "future"
        self.models_dir = ROOT / "models"
        
        self.model_path = model_path or (self.models_dir / "xgboost_model")
        
        self.feature_engineer = FeatureEngineer()
        self._predictor = None
        self.correlation_id = None

    @contextmanager
    def observability_context(self, operation: str):
        """Context manager for canonical logging and metrics."""
        self.correlation_id = str(uuid.uuid4())
        token = CORRELATION_ID.set(self.correlation_id)
        
        start_time = time.time()
        logger.with_correlation_id(self.correlation_id)
        
        logger.log_event(
            f'{operation}_started',
            operation=operation,
            timestamp=start_time
        )
        
        try:
            yield
            
            duration = time.time() - start_time
            logger.log_event(
                f'{operation}_completed',
                operation=operation,
                duration_seconds=duration,
                status='success',
            )
            
            # Record metric
            metrics.pipeline_duration.labels(
                pipeline_stage=operation
            ).observe(duration)
            
        except Exception as e:
            duration = time.time() - start_time
            
            logger.log_error(
                f'{operation}_failed',
                operation=operation,
                duration_seconds=duration,
                error_type=type(e).__name__,
                error_message=str(e),
                exc_info=True,
            )
            
            # Record failure metric
            metrics.data_pipeline_errors.labels(
                stage=operation,
                error_type=type(e).__name__
            ).inc()
            
            raise
        finally:
            CORRELATION_ID.reset(token)

    @property
    def predictor(self) -> Predictor:
        if self._predictor is None:
            self._predictor = Predictor(self.model_path)
        return self._predictor

    def run_data_pipeline(self) -> Dict[str, Any]:
        """Execute ETL with observability."""
        with self.observability_context('data_pipeline'):
            # Load raw data
            df = load_all_parquet_files(self.raw_dir)
            df = prepare_base_dataset(df)
            
            # Deduplication
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
                
            validate_temporal_order(df)
            
            # Feature Engineering
            fe = FeatureEngineer(
                rolling_windows=FEATURES.rolling_windows,
                min_matches=FEATURES.min_matches_for_stats,
                elo_k=FEATURES.elo_k_factor,
            )
            df = fe.add_all_features(df)
            
            # Save
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.processed_dir / "features_dataset.parquet"
            # Materialize to save
            df_materialized = df.collect()
            df_materialized.write_parquet(output_path)
            
            metrics.training_dataset_size.set(len(df_materialized))
            
            return {'output_path': str(output_path), 'count': len(df_materialized)}

    def run_training_pipeline(self, data_path: Path) -> None:
        """Run training with observability."""
        with self.observability_context('training_pipeline'):
            df = pl.scan_parquet(data_path)
            train_df, test_df = create_train_test_split(df, MODEL.train_cutoff_date)
            assert_no_leakage(train_df, test_df)
            
            train_data = train_df.collect()
            test_data = test_df.collect()
            
            # Prepare features
            fe = FeatureEngineer()
            feature_cols = fe.get_feature_columns(train_df)
            numeric_types = [pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                             pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                             pl.Float32, pl.Float64, pl.Boolean]
            existing_cols = [c for c in feature_cols if c in train_data.columns]
            numeric_cols = [c for c in existing_cols if train_data[c].dtype in numeric_types]
            
            # Train
            trainer = ModelTrainer(params=MODEL.xgb_params, calibrate=True)
            result = trainer.train(train_data, feature_cols=numeric_cols, eval_df=test_data)
            
            # Save
            self.models_dir.mkdir(parents=True, exist_ok=True)
            model_path = self.models_dir / "xgboost_model"
            trainer.save(model_path)
            
            # Metrics
            metrics.model_version.labels(
                version='latest', 
                trained_date=date.today().isoformat()
            ).set(1)
            
            logger.log_event(
                'training_metrics',
                auc=result.metrics.get('auc'),
                accuracy=result.metrics.get('accuracy')
            )
            
            if result.metrics.get('auc', 0) < 0.6:
                metrics.training_pipeline_runs.labels(status='warning').inc()
            else:
                metrics.training_pipeline_runs.labels(status='success').inc()

    def predict_upcoming(
        self,
        days: int = 7,
        min_odds: float = 1.5,
        max_odds: float = 3.0,
        min_confidence: float = 0.55,
        scrape_unknown: bool = True
    ) -> pl.DataFrame:
        """Get predictions with observability."""
        with self.observability_context('predict_upcoming'):
            # Step 1: Get upcoming matches
            upcoming = self._get_upcoming_matches(days)
            if len(upcoming) == 0:
                logger.log_event("no_upcoming_matches_found")
                return pl.DataFrame()
            
            # Step 2: Load history
            historical = self._load_historical_data()
            
            # Step 2.5: Unknown players
            if scrape_unknown:
                unknowns = self._identify_unknown_players(upcoming, historical)
                if unknowns:
                    self._scrape_unknown_players(unknowns)
                    historical = self._load_historical_data()
            
            # Step 3: Features
            prediction_ready = self.feature_engineer.compute_features_for_prediction(
                upcoming, historical
            )
            
            # Step 4: Predict (Batch)
            predictions = self.predictor.predict_with_value(prediction_ready)
            
            # Step 5: Filter
            recommended = predictions.filter(
                (pl.col("model_prob") >= min_confidence) &
                (pl.col("odds_player") >= min_odds) &
                (pl.col("odds_player") <= max_odds) &
                (pl.col("edge") > 0.05)
            ).sort("edge", descending=True)
            
            # Add timestamp
            final_df = recommended.with_columns([
                pl.lit(datetime.now().isoformat()).alias("prediction_timestamp"),
                pl.lit(date.today().isoformat()).alias("match_date") # Ensure column exists
            ])
            
            return final_df

    # Helper methods (reused from original)
    def _get_upcoming_matches(self, days: int) -> pl.DataFrame:
        latest_path = self.data_dir / "upcoming.parquet"
        if latest_path.exists():
            import os
            mtime = os.path.getmtime(latest_path)
            if (datetime.now().timestamp() - mtime) / 3600 < 1:
                logger.log_event("using_cached_upcoming")
                return pl.read_parquet(latest_path)
        
        logger.log_event("scraping_upcoming")
        return scrape_upcoming(days_ahead=days)

    def _load_historical_data(self) -> pl.DataFrame:
        processed_path = self.processed_dir / "features_dataset.parquet"
        if processed_path.exists():
            return pl.read_parquet(processed_path)
        
        unified_path = self.data_dir / "tennis.parquet"
        if unified_path.exists():
            return pl.read_parquet(unified_path)
            
        raw_files = sorted(self.raw_dir.glob("atp_matches_*.parquet"))
        if raw_files:
            return pl.read_parquet(raw_files[-1])
        return pl.DataFrame()

    def _identify_unknown_players(self, upcoming: pl.DataFrame, historical: pl.DataFrame) -> List[int]:
        ids = set()
        if "player_id" in upcoming.columns:
            ids.update(upcoming["player_id"].drop_nulls().unique().to_list())
        if "opponent_id" in upcoming.columns:
            ids.update(upcoming["opponent_id"].drop_nulls().unique().to_list())
            
        known = set()
        if len(historical) > 0 and "player_id" in historical.columns:
            known = set(historical["player_id"].unique().to_list())
            
        return [pid for pid in ids if pid not in known]

    def _scrape_unknown_players(self, player_ids: List[int]) -> None:
        if not player_ids: return
        logger.log_event("scraping_unknown_players", count=len(player_ids))
        try:
            scrape_players(player_ids=player_ids, max_pages=5, workers=3, smart_update=False)
        except Exception as e:
            logger.log_error("unknown_scrape_failed", error=str(e))

# Module-level functions for backward compatibility
def run_data_pipeline(raw_dir: Path, output_dir: Path):
    pipeline = TennisPipeline()
    pipeline.run_data_pipeline()

def run_training_pipeline(data_path: Path, models_dir: Path):
    pipeline = TennisPipeline()
    pipeline.run_training_pipeline(data_path)
