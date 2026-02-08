"""
Prediction service for live and batch predictions.
"""
import time
from datetime import datetime
from pathlib import Path

import polars as pl

from src.utils.observability import CORRELATION_ID, Logger, get_metrics

from .calibrator import MIN_PROB_THRESHOLD, ProbabilityCalibrator, passes_ev_gate


logger = Logger(__name__)
metrics = get_metrics()

class Predictor:
    """
    Prediction service for tennis match outcomes.
    Handles both live predictions and batch processing.
    """

    def __init__(self, model_path: Path | None = None, fallback_model_path: Path | None = None):
        """
        Args:
            model_path: Path to saved model
            fallback_model_path: Path to fallback model if primary fails
        """
        self.model_path = model_path
        self.fallback_model_path = fallback_model_path
        self.trainer = None
        self.fallback_trainer = None
        self.version = "unknown"
        self.calibrator = None  # Post-hoc isotonic calibrator

        if model_path:
            self._load_primary_model(model_path)
            self._load_calibrator(model_path)

        if fallback_model_path:
            self._load_fallback_model(fallback_model_path)

    def _load_primary_model(self, path: Path) -> None:
        """Load primary model with observability."""
        from .trainer import ModelTrainer
        try:
            self.trainer = ModelTrainer()
            self.trainer.load(path)
            self.version = self._extract_version(path)
            logger.log_event('model_loaded', model_path=str(path), version=self.version)
        except Exception as e:
            logger.log_error('model_load_failed', model_path=str(path), error=str(e))
            raise

    def _load_fallback_model(self, path: Path) -> None:
        """Load fallback model."""
        from .trainer import ModelTrainer
        try:
            self.fallback_trainer = ModelTrainer()
            self.fallback_trainer.load(path)
            logger.log_event('fallback_model_loaded', model_path=str(path))
        except Exception as e:
            logger.log_error('fallback_model_load_failed', model_path=str(path), error=str(e))

    def _load_calibrator(self, model_path: Path) -> None:
        """Load post-hoc calibrator if available."""
        calibrator_path = model_path.parent / "calibrator.joblib"
        if calibrator_path.exists():
            self.calibrator = ProbabilityCalibrator().load(calibrator_path)
            logger.log_event('calibrator_loaded', path=str(calibrator_path))
        else:
            logger.log_event('no_calibrator_found', path=str(calibrator_path))

    def _extract_version(self, path: Path) -> str:
        """Extract version from path or metadata."""
        # Check for adjacent meta file? Or just parse filename
        # Expected: xgboost_model or xgboost_model_v1.0
        name = path.name
        if "v" in name:
            return name.split("v")[-1].split(".")[0]
        return "latest"

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add predictions to dataframe with observability.
        
        Args:
            df: DataFrame with feature columns
            
        Returns:
            DataFrame with prediction columns added
        """
        start_time = time.time()
        correlation_id = CORRELATION_ID.get() or "unknown"

        logger.log_event(
            'prediction_batch_started',
            num_rows=len(df),
            correlation_id=correlation_id
        )

        if self.trainer is None:
            logger.log_error('prediction_failed_no_model')
            raise ValueError("No model loaded")

        try:
            # Primary Inference
            df_result = self._inference(self.trainer, df)

            latency = time.time() - start_time
            metrics.prediction_latency.observe(latency)
            metrics.successful_predictions.inc(len(df))
            metrics.last_prediction_timestamp.set_to_current_time()

            logger.log_event(
                'prediction_batch_completed',
                duration_seconds=latency,
                num_rows=len(df),
                status='success',
                version=self.version
            )
            return df_result

        except Exception as e:
            logger.log_error(
                'primary_inference_failed',
                error=str(e),
                exc_info=True
            )

            # Fallback
            if self.fallback_trainer:
                try:
                    logger.log_event('attempting_fallback_inference')
                    df_result = self._inference(self.fallback_trainer, df)

                    latency = time.time() - start_time
                    metrics.prediction_latency.observe(latency)
                    metrics.successful_predictions.inc(len(df))

                    logger.log_event(
                        'fallback_inference_success',
                        duration_seconds=latency,
                        status='fallback_used'
                    )
                    return df_result
                except Exception as fb_e:
                    logger.log_error('fallback_inference_failed', error=str(fb_e))
                    metrics.failed_predictions.labels(error_type='both_failed').inc()
                    raise

            metrics.failed_predictions.labels(error_type=type(e).__name__).inc()
            raise

    def _inference(self, trainer, df: pl.DataFrame) -> pl.DataFrame:
        """Internal inference logic with optional calibration."""
        raw_probas = trainer.predict_proba(df)

        # Apply post-hoc calibration if available
        if self.calibrator and self.calibrator.is_fitted:
            probas = self.calibrator.calibrate(raw_probas)
            logger.log_event('calibration_applied', num_samples=len(probas))
        else:
            probas = raw_probas

        return df.with_columns([
            pl.Series("model_prob", probas),
            pl.Series("raw_prob", raw_probas),  # Keep raw for comparison
            pl.Series("model_prediction", (probas >= 0.5).astype(int)),
        ])

    def predict_with_value(
        self,
        df: pl.DataFrame,
        min_edge: float = 0.05
    ) -> pl.DataFrame:
        """
        Add predictions with betting value calculations.
        """
        # Call observable predict
        df = self.predict(df)

        # Calculate edge if odds available
        if "odds_player" in df.columns:
            df = df.with_columns([
                (1 / pl.col("odds_player")).alias("implied_prob"),
                (pl.col("model_prob") - (1 / pl.col("odds_player"))).alias("edge"),
            ])

            # Basic edge check
            df = df.with_columns([
                (
                    pl.col("model_prob") * (pl.col("odds_player") - 1) -
                    (1 - pl.col("model_prob"))
                ).alias("expected_value"),
            ])

            # Apply probability-based EV gating
            # Higher required edge for lower probability bets
            df = df.with_columns([
                pl.struct(["model_prob", "edge"])
                .map_elements(
                    lambda x: passes_ev_gate(x["model_prob"], x["edge"]) if x["model_prob"] is not None and x["edge"] is not None else False,
                    return_dtype=pl.Boolean
                )
                .alias("passes_ev_gate"),
                (pl.col("model_prob") >= MIN_PROB_THRESHOLD).alias("above_min_prob"),
            ])

            # is_value_bet now combines edge threshold + EV gate
            df = df.with_columns([
                (
                    (pl.col("edge") >= min_edge) &
                    pl.col("passes_ev_gate")
                ).alias("is_value_bet"),
            ])

        return df

    def get_todays_predictions(
        self,
        matches_df: pl.DataFrame,
        min_confidence: float = 0.55,
        min_edge: float = 0.05
    ) -> pl.DataFrame:
        """Get today's betting recommendations."""
        predictions = self.predict_with_value(matches_df, min_edge)

        value_bets = predictions.filter(
            (pl.col("model_prob") >= min_confidence) &
            (pl.col("is_value_bet") == True)
        )

        value_bets = value_bets.sort("edge", descending=True)

        # Add timestamp
        value_bets = value_bets.with_columns([
            pl.lit(datetime.now().isoformat()).alias("prediction_timestamp")
        ])

        logger.log_event(
            'value_bets_identified',
            count=len(value_bets),
            total_matches=len(predictions)
        )

        return value_bets

    def save_predictions(
        self,
        predictions: pl.DataFrame,
        output_dir: Path,
        prefix: str = "predictions"
    ) -> Path:
        """Save predictions to parquet file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.parquet"
        path = output_dir / filename

        predictions.write_parquet(path)
        logger.log_event('predictions_saved', path=str(path))

        return path
