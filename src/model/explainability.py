"""
Model Explainability module using SHAP (SHapley Additive exPlanations).
"""
import logging
import pandas as pd
import polars as pl
import numpy as np
from typing import Dict, List, Any, Optional, Union
import shap
import xgboost as xgb

logger = logging.getLogger(__name__)

class ModelExplainer:
    """
    Wraps shap.TreeExplainer to provide human-readable explanations 
    for XGBoost predictions.
    """
    
    def __init__(self, model: Union[xgb.XGBClassifier, xgb.Booster], feature_names: List[str] = None):
        """
        Args:
            model: Trained XGBoost model (sklearn wrapper or booster)
            feature_names: List of feature names (optional, auto-detected if sklearn wrapper)
        """
        self.model = model
        
        # Handle sklearn wrapper vs raw booster
        if hasattr(model, "get_booster"):
            self.booster = model.get_booster()
            if feature_names is None:
                self.feature_names = model.feature_names_in_
            else:
                self.feature_names = feature_names
        else:
            self.booster = model
            self.feature_names = feature_names

        # Initialize Explainer
        # TreeExplainer is optimized for fast calculation on trees
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            logger.warning(f"Could not initialize SHAP TreeExplainer: {e}")
            self.explainer = None

    def explain_prediction(self, feature_row: Union[pd.DataFrame, pl.DataFrame, np.ndarray], top_k: int = 5) -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            feature_row: Single row of features
            top_k: Number of top impacting features to return
            
        Returns:
            Dictionary containing base_value, total_shap, and feature_contributions
        """
        if self.explainer is None:
            return {"error": "Explainer not initialized"}

        # Convert to numpy/pandas for SHAP
        if isinstance(feature_row, pl.DataFrame):
            X = feature_row.to_pandas()
        elif isinstance(feature_row, np.ndarray):
            X = pd.DataFrame(feature_row, columns=self.feature_names)
        else:
            X = feature_row

        # Calculate SHAP values
        shap_values = self.explainer(X)
        
        # Determine if binary classification (output dim 0 or 1)
        # XGBoost binary usually returns shape (1,) or (1, 2) depending on config
        # We assume binary target (index 0 usually implies negative class or raw log odds)
        # For classifier, shap_values.values shape is typically (samples, features)
        
        values = shap_values.values[0]
        base_value = shap_values.base_values[0]
        
        # Map values to features
        contributions = dict(zip(self.feature_names, values))
        
        # Sort by absolute impact
        sorted_attribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Get top K
        top_features = {k: float(v) for k, v in sorted_attribs[:top_k]}
        
        # Calculate prob shift (Sigmoid approx for intuition)
        # Note: Sum of SHAP values = LogOdds(Prediction) - LogOdds(Base)
        # We can't strictly sum probabilities, but we can show direction
        
        total_log_odds = base_value + sum(values)
        predicted_prob = 1 / (1 + np.exp(-total_log_odds))
        base_prob = 1 / (1 + np.exp(-base_value))
        
        return {
            "base_probability": float(base_prob),
            "predicted_probability": float(predicted_prob),
            "top_drivers": top_features,
            "all_contributions": {k: float(v) for k, v in sorted_attribs}
        }

    def plot_latest(self, feature_row, matplotlib=True):
        """Util function to just plot if needed (mostly for notebooks)"""
        if self.explainer:
            shap.plots.waterfall(self.explainer(feature_row)[0])
