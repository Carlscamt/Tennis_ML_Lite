"""
Unified Tennis Prediction Pipeline with Observability and Data Quality.
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

# Data Quality Imports
from src.schema import SchemaValidator
from src.data_quality.validator import (
    DataQualityMonitor, DriftDetector, AnomalyDetector, StalenessDetector
)

logger = Logger(__name__)
metrics = get_metrics()

class TennisPipeline:
    """
    Unified pipeline for tennis match predictions with production-grade observability and quality gates.
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

        # Data Quality Components
        self.schema_validator = SchemaValidator()
        self.drift_detector = DriftDetector(alpha=0.05)
        self.anomaly_detector = AnomalyDetector(z_threshold=3.0)
        self.staleness_detector = StalenessDetector(max_age_hours=24)
        
        self.quality_monitor = DataQualityMonitor(
            schema_validator=self.schema_validator,
            drift_detector=self.drift_detector,
            anomaly_detector=self.anomaly_detector,
            staleness_detector=self.staleness_detector,
        )

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
            
            metrics.pipeline_duration.labels(pipeline_stage=operation).observe(duration)
            
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
        """Execute ETL with observability and quality gates."""
        with self.observability_context('data_pipeline'):
            # Load raw data
            df = load_all_parquet_files(self.raw_dir)
            
            # --- QUALITY GATE 1: SCHEMA VALIDATION (RAW) ---
            # We collect schema check results but proceed if warnings only? 
            # Or strict fail? User asked for "Gates". Let's log errors but maybe allow proceed if not critical 
            # or strict fail. The plan says "Fails pipeline on schema violation".
            # SchemaValidator checks types.
            schema_res = self.schema_validator.validate_raw_data(df)
            if not schema_res['valid']:
                logger.log_error(
                    'schema_validation_failed', 
                    errors=schema_res['errors'][:5], 
                    count=len(schema_res['errors'])
                )
                # Strict Fail
                raise ValueError(f"Schema Validation Failed: {len(schema_res['errors'])} errors found.")
            
            logger.log_event('schema_validation_passed', num_rows=schema_res['num_rows'])
            
            df = prepare_base_dataset(df)
            
            # Deduplication logic
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
            
            # --- QUALITY GATE 2: FEATURES SCHEMA ---
            # Materialize to validate features strictly
            # Pandera on LazyFrame is supported but eager is safer for strict gate
            df_materialized = df.collect()
            
            feat_res = self.schema_validator.validate_features(df_materialized)
            if not feat_res['valid']:
                raise ValueError(f"Feature Schema Failed: {feat_res['errors'][:3]}")

            # Save
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.processed_dir / "features_dataset.parquet"
            df_materialized.write_parquet(output_path)
            
            metrics.training_dataset_size.set(len(df_materialized))
            
            return {'output_path': str(output_path), 'count': len(df_materialized)}

    def run_training_pipeline(self, data_path: Path) -> None:
        """Run training with observability and drift baseline fitting."""
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
            
            # --- DATA QUALITY: FIT DRIFT & ANOMALY DETECTORS ---
            logger.log_event('fitting_quality_monitors', num_features=len(numeric_cols))
            
            # Select only numeric features for drift/anomaly
            train_features = train_data.select(numeric_cols)
            self.drift_detector.fit_reference(train_features)
            self.anomaly_detector.fit_reference(train_features)
            
            # Train
            trainer = ModelTrainer(params=MODEL.xgb_params, calibrate=True)
            result = trainer.train(train_data, feature_cols=numeric_cols, eval_df=test_data)
            
            # Save
            self.models_dir.mkdir(parents=True, exist_ok=True)
            model_path = self.models_dir / "xgboost_model"
            trainer.save(model_path)
            
            metrics.model_version.labels(version='latest', trained_date=date.today().isoformat()).set(1)
            
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
        """Get predictions with observability and quality gates."""
        with self.observability_context('predict_upcoming'):
            # Step 1: Get upcoming
            upcoming = self._get_upcoming_matches(days)
            if len(upcoming) == 0:
                logger.log_event("no_upcoming_matches_found")
                return pl.DataFrame()
            
            # --- QUALITY GATE 3: INCOMING DATA MONITOR ---
            # Check for staleness, rough schema compliance on raw upcoming data
            # Assuming upcoming has 'start_timestamp'
            quality_rep = self.quality_monitor.check_incoming_data(upcoming, is_live=True)
            if not quality_rep['passed']:
                logger.log_error("incoming_data_quality_failed", errors=quality_rep['errors'])
                # We could raise error here. For now logging error is 
                # good, but user wants strict gates? "Fails pipeline on schema violation"
                if any("Schema" in e for e in quality_rep['errors']):
                     raise ValueError(f"Incoming Data Schema Failed: {quality_rep['errors']}")
            
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
            
            # --- QUALITY GATE 4: FEATURE DRIFT CHECK ---
            # Check for drift in the features we are about to predict on
            # (If detector is fitted)
            drift_rep = self.drift_detector.detect_drift(prediction_ready)
            drifted_cnt = sum(1 for r in drift_rep.values() if r.is_drifted)
            if drifted_cnt > 0:
                logger.log_event("feature_drift_warning", drifted_count=drifted_cnt)
                # We don't block on drift usually, just alert
            
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
                pl.lit(date.today().isoformat()).alias("match_date")
            ])
            
            return final_df

    # Helper methods
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

def run_data_pipeline(raw_dir: Path, output_dir: Path):
    pipeline = TennisPipeline()
    pipeline.run_data_pipeline()

def run_training_pipeline(data_path: Path, models_dir: Path):
    pipeline = TennisPipeline()
    pipeline.run_training_pipeline(data_path)
