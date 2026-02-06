"""
Unified Tennis Prediction Pipeline with Observability, Data Quality, and Advanced Serving.
"""
import sys
import time
import uuid
import os
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import polars as pl
import xgboost as xgb
import json

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
from src.model.trainer import ModelTrainer # Backward compat for params?
from src.scraper import scrape_upcoming, scrape_players
from src.utils.observability import get_metrics, Logger, CORRELATION_ID

# Data Quality Imports
from src.schema import SchemaValidator
from src.data_quality.validator import (
    DataQualityMonitor, DriftDetector, AnomalyDetector, StalenessDetector
)

# Model Serving Imports
from src.model.registry import ModelRegistry
from src.model.serving import get_model_server, ModelServer

logger = Logger(__name__)
metrics = get_metrics()

class TennisPipeline:
    """
    Unified pipeline for tennis match predictions with production-grade observability and quality gates.
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        model_path: Optional[Path] = None,
        root_dir: Optional[Path] = None
    ):
        self.root = root_dir or ROOT
        self.data_dir = data_dir or (self.root / "data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.future_dir = self.data_dir / "future"
        self.models_dir = self.root / "models"
        
        self.model_path = model_path or (self.models_dir / "xgboost_model")
        
        self.feature_engineer = FeatureEngineer()
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
        
        # Model Serving Components
        self.registry = ModelRegistry(root_dir=self.root)
        
        # If root_dir is provided (testing/isolation), create a local server instance
        # Otherwise use the global singleton
        if root_dir:
            from src.model.serving import ServingConfig
            self.model_server = ModelServer(self.registry, ServingConfig.from_env())
        else:
            self.model_server = get_model_server()

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

    def run_data_pipeline(self) -> Dict[str, Any]:
        """Execute ETL with observability and quality gates."""
        with self.observability_context('data_pipeline'):
            # Load raw data (Unified parquet or scattered raw files)
            unified_path = self.data_dir / "tennis.parquet"
            if unified_path.exists():
                logger.log_event("loading_unified_data", path=str(unified_path))
                df = pl.scan_parquet(unified_path)
            else:
                df = load_all_parquet_files(self.raw_dir)
            
            # --- QUALITY GATE 1: SCHEMA VALIDATION (RAW) ---
            schema_res = self.schema_validator.validate_raw_data(df)
            if not schema_res['valid']:
                logger.log_error('schema_validation_failed', errors=schema_res['errors'][:5], count=len(schema_res['errors']))
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
        """Run training with observability, drift baseline fitting, and model registration."""
        with self.observability_context('training_pipeline'):
            df = pl.scan_parquet(data_path).drop_nulls(subset=["player_won"])
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
            
            train_features = train_data.select(numeric_cols)
            self.drift_detector.fit_reference(train_features)
            self.anomaly_detector.fit_reference(train_features)
            
            # Train using XGBoost directly or via simple Trainer wrapper
            # Using XGBoost directly to facilitate file saving for Registry
            import xgboost as xgb
            X_train = train_data.select(numeric_cols).to_numpy()
            y_train = train_data.select(pl.col("player_won").cast(pl.Int8)).to_numpy().flatten()
            
            X_test = test_data.select(numeric_cols).to_numpy()
            y_test = test_data.select(pl.col("player_won").cast(pl.Int8)).to_numpy().flatten()
            
            model = xgb.XGBClassifier(**MODEL.xgb_params)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            
            # Calculate metrics
            from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            
            auc = roc_auc_score(y_test, y_prob)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            
            logger.log_event('training_metrics', auc=auc, accuracy=acc)
            
            # --- REGISTRY Integration ---
            import joblib
            temp_path = "temp_model.joblib"
            joblib.dump(model, temp_path)
            
            # Save metadata for registry to pick up
            meta = {
                "feature_columns": numeric_cols,
                "params": MODEL.xgb_params,
                "calibrated": False, # Pipeline doesn't stick calibration on top yet
            }
            with open("temp_model.meta.json", "w") as f:
                json.dump(meta, f)
            
            try:
                version = self.registry.register_model(
                    model_path=temp_path,
                    auc=auc,
                    precision=prec,
                    recall=rec,
                    feature_schema_version="1.0",
                    training_dataset_size=len(train_data),
                    notes=f"Trained on {len(train_data)} samples",
                    stage="Experimental"
                )
                logger.log_event('model_registered', version=version)
                
                # Auto-promote to Staging if passing threshold?
                if auc > 0.65:
                    self.registry.transition_stage(version, "Staging")
                    
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                if os.path.exists("temp_model.meta.json"):
                    os.remove("temp_model.meta.json")
            
            # --- RELOAD SERVER ---
            # Ensure the model server picks up the new champion immediately
            if self.model_server:
                self.model_server.reload_models()

    async def _predict_with_server(self, prediction_ready_df: pl.DataFrame) -> pl.DataFrame:
        """Internal helper to bridge Polars DataFrame to Model Server (Async)."""
        # Convert to list of dicts for model server
        # Explicit feature selection to ensure order? 
        # ModelServer expects features homogeneous.
        # Ideally we only pass features that were used in training.
        # For now pass all numeric columns present?
        
        # Use FeatureEngineer to identify SAFE feature columns (excluding IDs, timestamps)
        # This matches run_training_pipeline logic
        all_features = self.feature_engineer.get_feature_columns(prediction_ready_df)
        
        # Filter for numeric types only (as done in training)
        # We need to verify which columns are actually present and numeric
        numeric_types = [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt64, pl.UInt32]
        
        features = [
            c for c in all_features 
            if c in prediction_ready_df.columns and prediction_ready_df[c].dtype in numeric_types
        ]
        
        if not features:
            logger.log_warning("no_features_selected_for_prediction", available=prediction_ready_df.columns)
            # Fallback (risky but better than empty) or raise?
            # If no features, model will likely fail anyway.
        
        # Ensure consistent order (lexicographical or schema order? Schema order is safer if consistent)
        # But XGBoost is sensitive to order. 
        # Ideally we should sort them? 
        # Training pipeline used: [c for c in existing_cols ...] where existing_cols came from get_feature_columns
        # get_feature_columns iterates schema.
        # So we should rely on schema order.
        
        records = prediction_ready_df.select(features).to_dicts()
        
        if not self.model_server.champion_model and not self.model_server.challenger_model:
             raise RuntimeError("No models available in Registry. Please run training pipeline first.")

        result = await self.model_server.predict_batch(records)
        
        # Add predictions back to dataframe
        # result['predictions'] is list of 0/1
        # result['confidence_scores'] is list of floats
        
        return prediction_ready_df.with_columns([
            pl.Series("model_prediction", result['predictions']),
            pl.Series("model_prob", result['confidence_scores']),
            pl.lit(result['model_version']).alias("model_version"),
            pl.lit(result['serving_mode']).alias("serving_mode")
        ])

    def predict_upcoming(
        self,
        days: int = 7,
        min_odds: float = 1.5,
        max_odds: float = 3.0,
        min_confidence: float = 0.55,
        scrape_unknown: bool = True
    ) -> pl.DataFrame:
        """Get predictions with observability, data quality, and advanced serving."""
        import asyncio
        
        with self.observability_context('predict_upcoming'):
            # Step 1: Get upcoming
            upcoming = self._get_upcoming_matches(days)
            if len(upcoming) == 0:
                return pl.DataFrame()
            
            # --- QUALITY GATE 3: INCOMING DATA MONITOR ---
            quality_rep = self.quality_monitor.check_incoming_data(upcoming, is_live=True)
            if not quality_rep['passed']:
                 # Downgrade 'Data Stale' to warning, only raise on other errors
                 critical_errors = [e for e in quality_rep['errors'] if "Data Stale" not in e]
                 if critical_errors:
                     raise ValueError(f"Incoming Data Health Check Failed: {critical_errors}")
                 else:
                     logger.log_warning("incoming_data_quality_warning", errors=quality_rep['errors'])
            
            # Step 2: Load history
            historical = self._load_historical_data()
            
            # Step 2.5: Unknown players
            if scrape_unknown:
                unknowns = self._identify_unknown_players(upcoming, historical)
                if unknowns:
                    self._scrape_unknown_players(unknowns)
                    historical = self._load_historical_data()
            
            # Step 3: Create BOTH perspectives (A vs B and B vs A)
            # Original perspective
            original = upcoming.clone()
            
            # Create swapped perspective (B vs A) for opponent value detection
            swap_cols = {}
            if "player_id" in upcoming.columns and "opponent_id" in upcoming.columns:
                swap_cols["player_id"] = pl.col("opponent_id")
                swap_cols["opponent_id"] = pl.col("player_id")
            if "player_name" in upcoming.columns and "opponent_name" in upcoming.columns:
                swap_cols["player_name"] = pl.col("opponent_name")
                swap_cols["opponent_name"] = pl.col("player_name")
            if "odds_player" in upcoming.columns and "odds_opponent" in upcoming.columns:
                swap_cols["odds_player"] = pl.col("odds_opponent")
                swap_cols["odds_opponent"] = pl.col("odds_player")
            
            if swap_cols:
                swapped = upcoming.with_columns([v.alias(k) for k, v in swap_cols.items()])
                # Mark as swapped perspective
                original = original.with_columns(pl.lit("A").alias("_perspective"))
                swapped = swapped.with_columns(pl.lit("B").alias("_perspective"))
                combined_upcoming = pl.concat([original, swapped])
            else:
                combined_upcoming = original.with_columns(pl.lit("A").alias("_perspective"))
            
            # Step 4: Features for BOTH perspectives
            prediction_ready = self.feature_engineer.compute_features_for_prediction(
                combined_upcoming, historical
            )
            
            # --- QUALITY GATE 4: DRIFT ---
            drift_rep = self.drift_detector.detect_drift(prediction_ready)
            
            # Step 5: Predict via Model Server (for both perspectives)
            predictions = asyncio.run(self._predict_with_server(prediction_ready))
            
            # Step 5b: Normalize Probabilities
            # Ensure P(Player A) + P(Player B) = 1.0 per match
            if "model_prob" in predictions.columns:
                # Determine match identifier
                group_col = "event_id" if "event_id" in predictions.columns else "_match_key_temp"
                
                if "event_id" not in predictions.columns:
                     predictions = predictions.with_columns([
                        pl.when(pl.col("player_name") < pl.col("opponent_name"))
                        .then(pl.col("player_name") + "|" + pl.col("opponent_name"))
                        .otherwise(pl.col("opponent_name") + "|" + pl.col("player_name"))
                        .alias("_match_key_temp")
                    ])
                
                # Calculate sum of probs per match
                predictions = predictions.with_columns([
                    pl.col("model_prob").sum().over(group_col).alias("_prob_sum")
                ])
                
                # Normalize (avoid division by zero if something weird happens)
                predictions = predictions.with_columns([
                    (pl.col("model_prob") / pl.col("_prob_sum")).alias("model_prob")
                ])
                
                if "_match_key_temp" in predictions.columns:
                    predictions = predictions.drop("_match_key_temp")
                if "_prob_sum" in predictions.columns:
                    predictions = predictions.drop("_prob_sum")

            # Step 6: Calculate edge
            if "odds_player" in predictions.columns:
                predictions = predictions.with_columns([
                    (1 / pl.col("odds_player")).alias("implied_prob"),
                    (pl.col("model_prob") - (1 / pl.col("odds_player"))).alias("edge"),
                ])
            
            # Step 7: Filter for value bets
            recommended = predictions.filter(
                (pl.col("model_prob") >= min_confidence) &
                (pl.col("odds_player") >= min_odds) &
                (pl.col("odds_player") <= max_odds) &
                (pl.col("edge") > 0.05)
            )
            
            # Step 8: Deduplicate - keep only best bet per match
            # Create a match key that's the same regardless of perspective
            if "event_id" in recommended.columns:
                # Use event_id as match identifier
                recommended = recommended.sort("edge", descending=True)
                recommended = recommended.unique(subset=["event_id"], keep="first")
            else:
                # Fallback: create sorted player pair as match key
                recommended = recommended.with_columns([
                    pl.when(pl.col("player_name") < pl.col("opponent_name"))
                    .then(pl.col("player_name") + " vs " + pl.col("opponent_name"))
                    .otherwise(pl.col("opponent_name") + " vs " + pl.col("player_name"))
                    .alias("_match_key")
                ])
                recommended = recommended.sort("edge", descending=True)
                recommended = recommended.unique(subset=["_match_key"], keep="first")
                recommended = recommended.drop("_match_key")
            
            # Final sort by edge
            recommended = recommended.sort("edge", descending=True)
            
            # Add timestamp
            final_df = recommended.with_columns([
                pl.lit(datetime.now().isoformat()).alias("prediction_timestamp"),
                pl.lit(date.today().isoformat()).alias("match_date")
            ])
            
            # Drop internal columns
            if "_perspective" in final_df.columns:
                final_df = final_df.drop("_perspective")
            
            return final_df

    # Helper methods
    def _get_upcoming_matches(self, days: int) -> pl.DataFrame:
        latest_path = self.data_dir / "upcoming.parquet"
        if latest_path.exists():
            import os
            mtime = os.path.getmtime(latest_path)
            if (datetime.now().timestamp() - mtime) / 3600 < 1:
                return pl.read_parquet(latest_path)
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
        try:
            scrape_players(player_ids=player_ids, max_pages=10, workers=2, smart_update=True)
        except Exception as e:
            logger.log_error("unknown_scrape_failed", error=str(e))

def run_data_pipeline(raw_dir: Path, output_dir: Path):
    pipeline = TennisPipeline()
    pipeline.run_data_pipeline()

def run_training_pipeline(data_path: Path, models_dir: Path):
    pipeline = TennisPipeline()
    pipeline.run_training_pipeline(data_path)
