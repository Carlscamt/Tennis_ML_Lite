"""
Unit tests for SHAP Explainability.
"""
import pytest
import numpy as np
import polars as pl
import xgboost as xgb
from src.model.explainability import ModelExplainer

@pytest.fixture
def dummy_model_and_data():
    """Create a simple trained XGBoost model and data for testing."""
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, 20)
    feature_names = [f"feat_{i}" for i in range(5)]
    
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2, eval_metric="logloss")
    model.fit(X, y)
    
    return model, X, feature_names

def test_explainer_initialization(dummy_model_and_data):
    model, _, feature_names = dummy_model_and_data
    explainer = ModelExplainer(model, feature_names=feature_names)
    
    assert explainer.model is not None
    assert explainer.feature_names == feature_names
    # Explainer might be None if SHAP C++ extension issues, but in most tests it should be OK
    # We allow it to be None in code, but here we expect it to work
    assert explainer.explainer is not None

def test_explain_prediction_structure(dummy_model_and_data):
    model, X, feature_names = dummy_model_and_data
    explainer = ModelExplainer(model, feature_names=feature_names)
    
    # Explain first row
    row = X[0]
    explanation = explainer.explain_prediction(row)
    
    # Check Keys
    assert "base_probability" in explanation
    assert "predicted_probability" in explanation
    assert "top_drivers" in explanation
    assert "all_contributions" in explanation
    
    # Check Types
    assert isinstance(explanation["base_probability"], float)
    assert isinstance(explanation["predicted_probability"], float)
    assert isinstance(explanation["top_drivers"], dict)

def test_explain_prediction_consistency(dummy_model_and_data):
    """Ensure prob shift matches contributions directionally (roughly)"""
    model, X, feature_names = dummy_model_and_data
    explainer = ModelExplainer(model, feature_names=feature_names)
    
    row = X[0]
    explanation = explainer.explain_prediction(row)
    
    base = explanation["base_probability"]
    pred = explanation["predicted_probability"]
    
    # Sum of contributions should roughly explain the Move
    # Note: SHAP sums LogOdds, so we can't sum probs directly.
    # But if Pred > Base, sum(contributions) should be positive? Usually yes.
    
    total_impact = sum(explanation["all_contributions"].values())
    
    if pred > base:
        assert total_impact > -0.01 # allow small float error
    elif pred < base:
        assert total_impact < 0.01
