"""
Model training pipeline with XGBoost and probability calibration.
"""
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


try:
    import xgboost as xgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Result of model training."""
    model: object
    feature_importance: dict[str, float]
    metrics: dict[str, float]
    feature_columns: list[str]


class ModelTrainer:
    """
    XGBoost trainer with probability calibration for betting.
    """

    def __init__(
        self,
        params: dict | None = None,
        calibrate: bool = True,
        n_splits: int = 5
    ):
        """
        Args:
            params: XGBoost parameters (uses defaults if None)
            calibrate: Whether to calibrate probabilities
            n_splits: CV splits for calibration
        """
        if not XGB_AVAILABLE:
            raise ImportError("xgboost and scikit-learn required")

        self.params = params or {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        self.calibrate = calibrate
        self.n_splits = n_splits
        self.model = None
        self.calibrated_model = None
        self.feature_columns = []

    @staticmethod
    def _pin_random_seeds(seed: int = 42):
        """
        Pin all random seeds for reproducibility.
        
        Args:
            seed: Random seed value
        """
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        logger.debug(f"Random seeds pinned to {seed}")

    def train(
        self,
        train_df: pl.DataFrame,
        feature_cols: list[str],
        target_col: str = "player_won",
        eval_df: pl.DataFrame | None = None
    ) -> TrainingResult:
        """
        Train the model.
        
        Args:
            train_df: Training data (Polars DataFrame)
            feature_cols: List of feature column names
            target_col: Target column name
            eval_df: Optional validation set
            
        Returns:
            TrainingResult with model and metrics
        """
        # Pin all random seeds for reproducibility
        seed = self.params.get("random_state", 42)
        self._pin_random_seeds(seed)

        self.feature_columns = feature_cols

        # Convert to numpy
        X_train = train_df.select(feature_cols).to_numpy()
        y_train = train_df[target_col].to_numpy().astype(int)

        # Handle missing values
        X_train = np.nan_to_num(X_train, nan=-999)

        logger.info(f"Training on {len(X_train):,} samples with {len(feature_cols)} features")

        # Build XGBoost model
        self.model = xgb.XGBClassifier(**self.params)

        if eval_df is not None:
            X_eval = eval_df.select(feature_cols).to_numpy()
            X_eval = np.nan_to_num(X_eval, nan=-999)
            y_eval = eval_df[target_col].to_numpy().astype(int)

            self.model.fit(
                X_train, y_train,
                eval_set=[(X_eval, y_eval)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train, verbose=False)

        # Probability calibration
        if self.calibrate:
            logger.info("Calibrating probabilities...")
            # Create a new model without early_stopping for calibration
            calib_params = self.params.copy()
            calib_params.pop("early_stopping_rounds", None)  # Remove if exists
            base_model = xgb.XGBClassifier(**calib_params)

            self.calibrated_model = CalibratedClassifierCV(
                base_model,
                method="isotonic",
                cv=self.n_splits
            )
            self.calibrated_model.fit(X_train, y_train)

        # Feature importance
        importance = dict(zip(
            feature_cols,
            self.model.feature_importances_.tolist()
        ))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        # Metrics
        metrics = self._evaluate(X_train, y_train, "train")

        if eval_df is not None:
            eval_metrics = self._evaluate(X_eval, y_eval, "eval")
            metrics.update(eval_metrics)

        logger.info(f"Training complete. Log-loss: {metrics.get('train_logloss', 0):.4f}")

        return TrainingResult(
            model=self.calibrated_model if self.calibrate else self.model,
            feature_importance=importance,
            metrics=metrics,
            feature_columns=feature_cols
        )

    def _evaluate(self, X: np.ndarray, y: np.ndarray, prefix: str) -> dict[str, float]:
        """Calculate evaluation metrics."""
        from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

        model = self.calibrated_model if self.calibrate and self.calibrated_model else self.model

        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)

        return {
            f"{prefix}_logloss": log_loss(y, proba),
            f"{prefix}_brier": brier_score_loss(y, proba),
            f"{prefix}_auc": roc_auc_score(y, proba),
            f"{prefix}_accuracy": accuracy_score(y, preds),
        }

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        """
        Get calibrated probability predictions.
        
        Args:
            df: Data to predict on
            
        Returns:
            Array of win probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained")

        # Add missing feature columns as null
        existing_cols = set(df.columns)
        missing_cols = [c for c in self.feature_columns if c not in existing_cols]

        if missing_cols:
            logger.warning(f"Adding {len(missing_cols)} missing features as null")
            for col in missing_cols:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

        X = df.select(self.feature_columns).to_numpy()
        X = np.nan_to_num(X, nan=-999)

        model = self.calibrated_model if self.calibrate and self.calibrated_model else self.model
        return model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        """Save model and metadata."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save the actual model (calibrated or not)
        model_path = path.with_suffix(".joblib")
        model_to_save = self.calibrated_model if self.calibrate and self.calibrated_model else self.model
        joblib.dump(model_to_save, model_path)

        # Save metadata
        meta = {
            "feature_columns": self.feature_columns,
            "params": self.params,
            "calibrated": self.calibrate,
        }
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Model saved to {path}")

    def load(self, path: Path) -> None:
        """Load model and metadata."""
        import joblib

        path = Path(path)

        # Determine actual model path
        if path.exists() and path.is_file():
            model_path = path
        else:
            model_path = path.with_suffix(".joblib")

        # Load the model
        loaded_model = joblib.load(model_path)

        # Set both model and calibrated_model
        self.calibrated_model = loaded_model
        self.model = loaded_model  # For predict_proba

        # Load metadata
        # Try different metadata naming conventions
        meta_candidates = [
            path.with_suffix(".meta.json"),
            path.parent / (path.name + ".meta.json"),
            path.parent / "model.meta.json"
        ]

        meta_path = None
        for p in meta_candidates:
            if p.exists():
                meta_path = p
                break

        if meta_path:
            with open(meta_path) as f:
                meta = json.load(f)

            self.feature_columns = meta["feature_columns"]
            self.params = meta["params"]
            self.calibrate = meta.get("calibrated", False)
        else:
            logger.warning(f"No metadata found for {path}, using defaults")

        logger.info(f"Model loaded from {path}")
