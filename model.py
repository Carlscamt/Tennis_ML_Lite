"""
Model - XGBoost training and prediction
"""
import pickle
import numpy as np
import polars as pl
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from config import MODELS_DIR, TEST_SPLIT_DATE, RANDOM_STATE, KELLY_FRACTION
from features import get_feature_columns

MODEL_PATH = MODELS_DIR / "model.pkl"

def train(df: pl.DataFrame) -> None:
    """Train XGBoost model on historical data."""
    print("Training model...")
    
    features = get_feature_columns()
    
    # Split by date
    train_df = df.filter(pl.col("date") < TEST_SPLIT_DATE)
    
    X = train_df.select(features).to_numpy()
    y = train_df["player_won"].to_numpy()
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0)
    
    # Train
    base_model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    )
    
    model = CalibratedClassifierCV(base_model, cv=3, method="isotonic")
    model.fit(X, y)
    
    # Save
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "features": features}, f)
    
    print(f"Model saved to {MODEL_PATH}")

def predict(df: pl.DataFrame) -> pl.DataFrame:
    """Make predictions on dataframe."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model. Run: python main.py train")
    
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    
    model = saved["model"]
    features = saved["features"]
    
    X = df.select(features).to_numpy()
    X = np.nan_to_num(X, nan=0)
    
    probs = model.predict_proba(X)[:, 1]
    
    df = df.with_columns([
        pl.Series("model_prob", probs),
        (pl.Series("model_prob", probs) - (1 / pl.col("odds_player"))).alias("edge"),
    ])
    
    return df

def calculate_kelly(prob: float, odds: float) -> float:
    """Calculate Kelly stake."""
    edge = prob - (1 / odds)
    if edge <= 0:
        return 0
    kelly = edge / (odds - 1)
    return kelly * KELLY_FRACTION
