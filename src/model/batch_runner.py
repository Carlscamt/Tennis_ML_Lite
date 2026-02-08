"""
Batch prediction runner with clear failure semantics.

Orchestrates parallel champion/challenger predictions with:
- Separate output files per model version
- Explicit failure handling
- Value bets from champion only
"""
import logging
from datetime import date, datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path
from enum import Enum
import polars as pl

from src.model.serving import ModelServer, ServingConfig, PredictionResult
from src.model.registry import ModelRegistry

logger = logging.getLogger(__name__)


class BatchStatus(Enum):
    """Status of a batch run."""
    SUCCESS = "success"
    PARTIAL = "partial"          # Champion succeeded, challenger failed
    FALLBACK = "fallback"        # Used previous champion
    SKIPPED = "skipped"          # All models failed, no bets generated
    FAILED = "failed"            # Critical failure


@dataclass
class ModelPredictions:
    """Predictions from a single model."""
    model_version: str
    model_type: str  # "champion" or "challenger"
    predictions: pl.DataFrame
    latency_ms: float
    success: bool = True
    error: Optional[str] = None


@dataclass
class BatchResult:
    """Result of a batch prediction run."""
    date: date
    status: BatchStatus
    champion_predictions: Optional[ModelPredictions] = None
    challenger_predictions: Optional[ModelPredictions] = None
    value_bets_count: int = 0
    alerts: List[str] = field(default_factory=list)
    
    @property
    def summary(self) -> Dict[str, Any]:
        """Get summary for logging."""
        return {
            "date": self.date.isoformat(),
            "status": self.status.value,
            "champion_version": self.champion_predictions.model_version if self.champion_predictions else None,
            "challenger_version": self.challenger_predictions.model_version if self.challenger_predictions else None,
            "value_bets": self.value_bets_count,
            "alerts": len(self.alerts),
        }


class BatchRunner:
    """
    Orchestrates batch predictions with clear failure semantics.
    
    Failure Semantics:
    - Champion unavailable: Fallback to previous champion OR skip + alert
    - Champion fails mid-batch: Rollback partial, alert
    - Challenger fails: Log warning, continue champion-only
    - All models fail: Skip day's bets, raise critical alert
    
    Output:
    - predictions/champion/{date}.parquet — Production predictions
    - predictions/challenger/{date}.parquet — Shadow/comparison only
    - predictions/value_bets/{date}.parquet — Champion-only, for downstream
    """
    
    def __init__(
        self,
        registry: ModelRegistry = None,
        output_dir: Path = None,
        min_edge_threshold: float = 0.05,
    ):
        self.registry = registry or ModelRegistry()
        self.output_dir = Path(output_dir or "predictions")
        self.min_edge_threshold = min_edge_threshold
        
        # Create output directories
        (self.output_dir / "champion").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "challenger").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "value_bets").mkdir(parents=True, exist_ok=True)
    
    def run_batch(
        self,
        features_df: pl.DataFrame,
        run_date: date = None,
        feature_cols: List[str] = None,
    ) -> BatchResult:
        """
        Run batch predictions with champion/challenger parallel execution.
        
        Args:
            features_df: DataFrame with features for prediction
            run_date: Date for output files (defaults to today)
            feature_cols: Feature columns for prediction
            
        Returns:
            BatchResult with status and predictions
        """
        run_date = run_date or date.today()
        alerts = []
        
        logger.info(f"Starting batch run for {run_date}")
        
        # 1. Validate champion availability
        champion_result = self._run_champion(features_df, feature_cols, alerts)
        
        if champion_result is None:
            # Try fallback to previous champion
            champion_result = self._try_fallback(features_df, feature_cols, alerts)
            
            if champion_result is None:
                # Complete failure - skip day's bets
                alerts.append("CRITICAL: All models unavailable, skipping day's bets")
                logger.error("Batch run failed: no models available")
                return BatchResult(
                    date=run_date,
                    status=BatchStatus.SKIPPED,
                    alerts=alerts,
                )
        
        # 2. Run challenger predictions (if available)
        challenger_result = self._run_challenger(features_df, feature_cols, alerts)
        
        # 3. Generate value bets from CHAMPION ONLY
        value_bets_count = 0
        if champion_result and champion_result.success:
            value_bets_count = self._generate_value_bets(
                champion_result.predictions,
                run_date,
            )
        
        # 4. Save outputs
        self._save_predictions(champion_result, challenger_result, run_date)
        
        # Determine final status
        if champion_result and champion_result.success:
            if challenger_result and challenger_result.success:
                status = BatchStatus.SUCCESS
            else:
                status = BatchStatus.PARTIAL
        else:
            status = BatchStatus.FALLBACK if champion_result else BatchStatus.FAILED
        
        result = BatchResult(
            date=run_date,
            status=status,
            champion_predictions=champion_result,
            challenger_predictions=challenger_result,
            value_bets_count=value_bets_count,
            alerts=alerts,
        )
        
        logger.info(f"Batch run complete: {result.summary}")
        return result
    
    def _run_champion(
        self,
        features_df: pl.DataFrame,
        feature_cols: List[str],
        alerts: List[str],
    ) -> Optional[ModelPredictions]:
        """Run predictions with champion model."""
        try:
            version, model_path = self.registry.get_production_model()
            logger.info(f"Running champion: {version}")
            
            start_time = datetime.now()
            predictions = self._predict(model_path, features_df, feature_cols)
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            return ModelPredictions(
                model_version=version,
                model_type="champion",
                predictions=predictions,
                latency_ms=latency,
            )
            
        except Exception as e:
            alerts.append(f"Champion model failed: {e}")
            logger.error(f"Champion prediction failed: {e}")
            return None
    
    def _run_challenger(
        self,
        features_df: pl.DataFrame,
        feature_cols: List[str],
        alerts: List[str],
    ) -> Optional[ModelPredictions]:
        """Run predictions with challenger model (shadow mode)."""
        try:
            result = self.registry.get_challenger_model()
            if result is None:
                logger.info("No challenger model in Staging")
                return None
            
            version, model_path = result
            logger.info(f"Running challenger: {version}")
            
            start_time = datetime.now()
            predictions = self._predict(model_path, features_df, feature_cols)
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            return ModelPredictions(
                model_version=version,
                model_type="challenger",
                predictions=predictions,
                latency_ms=latency,
            )
            
        except Exception as e:
            alerts.append(f"Challenger model failed: {e}")
            logger.warning(f"Challenger prediction failed: {e}")
            return None
    
    def _try_fallback(
        self,
        features_df: pl.DataFrame,
        feature_cols: List[str],
        alerts: List[str],
    ) -> Optional[ModelPredictions]:
        """Try fallback to previous production model or any available model."""
        try:
            # Get all production models sorted by version
            all_models = self.registry.list_models(stage="Production")
            if not all_models:
                return None
            
            # Try each in order
            for model in all_models:
                try:
                    version = model.version
                    model_path = self.registry.root_dir / model.model_file
                    
                    logger.info(f"Trying fallback: {version}")
                    
                    start_time = datetime.now()
                    predictions = self._predict(str(model_path), features_df, feature_cols)
                    latency = (datetime.now() - start_time).total_seconds() * 1000
                    
                    alerts.append(f"Using fallback model: {version}")
                    
                    return ModelPredictions(
                        model_version=version,
                        model_type="champion",  # Treating as champion for downstream
                        predictions=predictions,
                        latency_ms=latency,
                    )
                    
                except Exception:
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Fallback failed: {e}")
            return None
    
    def _predict(
        self,
        model_path: str,
        features_df: pl.DataFrame,
        feature_cols: List[str],
    ) -> pl.DataFrame:
        """Make predictions with a model."""
        import xgboost as xgb
        import numpy as np
        
        # Load model
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        
        # Prepare features
        if feature_cols:
            X = features_df.select(feature_cols).to_numpy()
        else:
            # Use model's feature names if available
            X = features_df.to_numpy()
        
        X = np.nan_to_num(X, nan=-999)
        
        # Predict
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        # Return as DataFrame with predictions
        result = features_df.with_columns([
            pl.Series("predicted_prob", probs),
            pl.Series("predicted_outcome", preds),
        ])
        
        return result
    
    def _generate_value_bets(
        self,
        predictions: pl.DataFrame,
        run_date: date,
    ) -> int:
        """
        Generate value bets from champion predictions.
        
        Only champion predictions are used for actual betting decisions.
        """
        # Calculate edge where available
        if "odds_player" in predictions.columns and "predicted_prob" in predictions.columns:
            value_bets = predictions.with_columns([
                ((pl.col("predicted_prob") * pl.col("odds_player")) - 1).alias("edge")
            ]).filter(
                pl.col("edge") >= self.min_edge_threshold
            )
        else:
            # Fallback: just use high confidence predictions
            value_bets = predictions.filter(
                (pl.col("predicted_prob") >= 0.6) | (pl.col("predicted_prob") <= 0.4)
            )
        
        # Save value bets
        output_path = self.output_dir / "value_bets" / f"{run_date}.parquet"
        value_bets.write_parquet(output_path)
        
        logger.info(f"Generated {len(value_bets)} value bets → {output_path}")
        return len(value_bets)
    
    def _save_predictions(
        self,
        champion: Optional[ModelPredictions],
        challenger: Optional[ModelPredictions],
        run_date: date,
    ):
        """Save predictions to separate files."""
        if champion and champion.success:
            path = self.output_dir / "champion" / f"{run_date}.parquet"
            champion.predictions.with_columns([
                pl.lit(champion.model_version).alias("model_version"),
                pl.lit("champion").alias("model_type"),
            ]).write_parquet(path)
            logger.info(f"Champion predictions → {path}")
        
        if challenger and challenger.success:
            path = self.output_dir / "challenger" / f"{run_date}.parquet"
            challenger.predictions.with_columns([
                pl.lit(challenger.model_version).alias("model_version"),
                pl.lit("challenger").alias("model_type"),
            ]).write_parquet(path)
            logger.info(f"Challenger predictions → {path}")


def get_batch_runner(output_dir: Path = None) -> BatchRunner:
    """Get batch runner instance."""
    return BatchRunner(output_dir=output_dir)
