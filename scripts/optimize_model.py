"""
Script to optimize model hyperparameters for Profit (ROI).
"""
import sys
from pathlib import Path

# Add root to pythonpath
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import logging
import polars as pl
from config import PROCESSED_DATA_DIR
from src.utils import setup_logging
from src.transform.features import FeatureEngineer
from src.model.optimization import ProfitOptimizer

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading dataset...")
    df = pl.read_parquet(PROCESSED_DATA_DIR / "features_dataset.parquet")
    
    # Sort by date for TimeSeriesSplit
    df = df.sort("start_timestamp")
    
    # Filter for matches with odds AND outcomes
    df = df.filter(
        pl.col("odds_player").is_not_null() & 
        pl.col("odds_opponent").is_not_null() &
        pl.col("player_won").is_not_null()
    )
    logger.info(f"Dataset ready. {len(df)} matches with odds.")
    
    # Get feature columns
    feature_cols = FeatureEngineer().get_feature_columns(df.lazy())
    
    # Define Parameter Grid to Search (Reduced for speed)
    param_grid = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [4, 6],
        "n_estimators": [100, 300],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "objective": ["binary:logistic"],
        "eval_metric": ["logloss"],
        "n_jobs": [-1]
    }
    
    # Run Optimization
    optimizer = ProfitOptimizer(
        feature_cols=feature_cols,
        target_col="player_won",
        n_splits=3  # 3 Fold Walk-Forward
    )
    
    best_params = optimizer.optimize(df, param_grid)
    
    print("\n" + "="*50)
    print(f"BEST PARAMETERS FOR ROI")
    print(f"Avg ROI: {optimizer.best_roi:.2f}%")
    print("="*50)
    for k, v in best_params.items():
        print(f"{k}: {v}")
    print("="*50)

if __name__ == "__main__":
    main()
