"""
Script to verify SHAP explainability.
"""
import sys
from pathlib import Path
import logging

# Add root to pythonpath
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import polars as pl
import xgboost as xgb
from src.transform.features import FeatureEngineer
from src.model.explainability import ModelExplainer
from src.utils import setup_logging
from config import PROCESSED_DATA_DIR

setup_logging()
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading data...")
    df = pl.read_parquet(PROCESSED_DATA_DIR / "features_dataset.parquet")
    
    # Needs to be not null
    df = df.drop_nulls(subset=["player_won"])
    
    feature_cols = FeatureEngineer().get_feature_columns(df.lazy())
    target_col = "player_won"
    
    # Train a quick dummy model
    logger.info("Training dummy model for explanation...")
    X = df.select(feature_cols).to_numpy()
    y = df[target_col].to_numpy()
    
    model = xgb.XGBClassifier(n_estimators=10, max_depth=3, eval_metric="logloss")
    model.fit(X, y)
    
    # Initialize Explainer
    explainer = ModelExplainer(model, feature_names=feature_cols)
    
    # Pick a random sample (e.g., Alcaraz match?)
    sample_row = df.filter(pl.col("player_name").str.contains("Alcaraz")).tail(1)
    
    if len(sample_row) == 0:
        sample_row = df.tail(1)
        
    logger.info(f"Explaining prediction for: {sample_row['player_name'][0]} vs {sample_row['opponent_name'][0]}")
    
    explanation = explainer.explain_prediction(sample_row.select(feature_cols))
    
    print("\n" + "="*50)
    print("🔮 PREDICTION EXPLANATION")
    print("="*50)
    print(f"Base Probability:      {explanation['base_probability']:.1%}")
    print(f"Predicted Probability: {explanation['predicted_probability']:.1%}")
    print("\nTOP DRIVERS (+ increases prob, - decreases):")
    
    for feature, impact in explanation['top_drivers'].items():
        direction = "🟩" if impact > 0 else "🟥"
        print(f"{direction} {feature:<30}: {impact:+.4f}")
        
    print("="*50)

if __name__ == "__main__":
    main()
