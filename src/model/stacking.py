"""
Model Stacking (Meta-Learning) implementation.
Combines multiple base models to create a robust meta-model.
"""
import logging
import numpy as np
import polars as pl
from typing import List, Dict, Optional, Any
from pathlib import Path
import json
import joblib

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb

logger = logging.getLogger(__name__)

class StackedTrainer:
    """
    Trains a Stacking Ensemble:
    1. Base XGBoost (Complex, High Variance)
    2. Base Logistic Regression (Simple, High Bias)
    3. Meta Logistic Regression (Combines 1 & 2)
    """
    
    def __init__(self, xgb_params: Optional[Dict] = None):
        """
        Args:
            xgb_params: Parameters for the XGBoost base model.
        """
        self.xgb_params = xgb_params or {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "random_state": 42
        }
        self.model = None
        self.feature_columns = []
        
    def train(self, train_df: pl.DataFrame, feature_cols: List[str], target_col: str = "player_won"):
        """
        Train the StackingClassifier.
        """
        self.feature_columns = feature_cols
        
        # Prepare Data
        X = train_df.select(feature_cols).to_numpy()
        y = train_df[target_col].to_numpy().astype(int)
        
        # Handle -999 or other specific missing values if necessary, 
        # but SimpleImputer below handles np.nan. 
        # Polars to_numpy() usually converts nulls to nan for floats.
        
        logger.info(f"Training Stacked Model on {len(X)} samples...")
        
        # --- Define Base Estimators ---
        
        # 1. XGBoost (Handles NaNs natively, no scaling needed)
        xgb_clf = xgb.XGBClassifier(**self.xgb_params)
        
        # 2. Logistic Regression (Needs Imputation + Scaling)
        # We use a pipeline: Impute -> Scale -> LogReg
        lr_pipeline = make_pipeline(
            SimpleImputer(strategy="mean"),
            StandardScaler(),
            LogisticRegression(C=0.1, solver='lbfgs', max_iter=1000)
        )
        
        estimators = [
            ('xgb', xgb_clf),
            ('lr', lr_pipeline)
        ]
        
        # --- Define Stacking Classifier ---
        # final_estimator uses predictions from base estimators as features
        # cv=5 means 5-fold cross-validation is used to generate the meta-features (OOF predictions)
        
        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=5,
            n_jobs=-1,
            passthrough=False # If True, adds original features to meta-model. False = Only predictions.
        )
        
        self.model.fit(X, y)
        
        # Calculate Train Score
        acc = self.model.score(X, y)
        logger.info(f"Stacking Training Complete. Accuracy: {acc:.4f}")
        
    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        """
        Get probabilities from the meta-model.
        """
        if self.model is None:
            raise ValueError("Model not trained")
            
        # Add missing columns
        existing_cols = set(df.columns)
        missing_cols = [c for c in self.feature_columns if c not in existing_cols]
        if missing_cols:
            for col in missing_cols:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
                
        X = df.select(self.feature_columns).to_numpy()
        
        # Sklearn predict_proba returns [prob_0, prob_1]
        return self.model.predict_proba(X)[:, 1]
        
    def save(self, path: Path):
        """Save the full stack."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path.with_suffix(".joblib"))
        
        # Save metadata
        meta = {
            "feature_columns": self.feature_columns,
            "xgb_params": self.xgb_params,
            "type": "stacked"
        }
        with open(path.parent / "model.meta.json", "w") as f:
            json.dump(meta, f, indent=2)
            
    def load(self, path: Path):
        """Load the full stack."""
        path = Path(path)
        self.model = joblib.load(path.with_suffix(".joblib"))
        
        # Try load meta
        meta_path = path.parent / "model.meta.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self.feature_columns = meta["feature_columns"]
            self.xgb_params = meta.get("xgb_params", {})
