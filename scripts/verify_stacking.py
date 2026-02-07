"""
Verify Stacking Implementation.
"""
import sys
from pathlib import Path
import logging
import polars as pl
import numpy as np

# Add root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model.stacking import StackedTrainer
from src.transform.features import FeatureEngineer
from src.utils import setup_logging
from config import PROCESSED_DATA_DIR

setup_logging()
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading Data...")
    df = pl.read_parquet(PROCESSED_DATA_DIR / "features_dataset.parquet")
    
    # Filter for valid rows
    df = df.drop_nulls(subset=["player_won"])
    
    # Use subset for speed
    df = df.tail(1000)
    
    feature_cols = FeatureEngineer().get_feature_columns(df.lazy())
    
    logger.info("Initializing Stacked Trainer...")
    trainer = StackedTrainer()
    
    logger.info("Training...")
    trainer.train(df, feature_cols, "player_won")
    
    logger.info("Predicting...")
    # Predict on same data just to check
    probs = trainer.predict_proba(df)
    
    logger.info(f"Predictions Shape: {probs.shape}")
    logger.info(f"Mean Prob: {np.mean(probs):.4f}")
    
    # Check if meta-learner coefficients exist (confirming it trained)
    # The final_estimator is accessible via .final_estimator_
    meta_coef = trainer.model.final_estimator_.coef_
    logger.info(f"Meta-Learner Coefficients: {meta_coef}")
    logger.info("(Values usually [XGB_Weight, LogReg_Weight])")
    
    logger.info("✅ Stacking Verification Complete")

if __name__ == "__main__":
    main()
