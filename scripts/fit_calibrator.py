"""
Fit and save the isotonic calibrator for the current model.
Run this after training to create calibrator.joblib.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import polars as pl
import numpy as np
from datetime import datetime, date
from src.model import ModelTrainer, Predictor, ModelRegistry, ProbabilityCalibrator
from config import PROCESSED_DATA_DIR, MODELS_DIR

# Load data
data_path = PROCESSED_DATA_DIR / "features_dataset.parquet"
df = pl.read_parquet(data_path)

# Train/test split (same as training)
TRAIN_CUTOFF = date(2025, 1, 1)
cutoff_ts = int(datetime.combine(TRAIN_CUTOFF, datetime.min.time()).timestamp())

train_df = df.filter(pl.col('start_timestamp') < cutoff_ts)
test_df = df.filter(
    (pl.col('start_timestamp') >= cutoff_ts) &
    (pl.col('player_won').is_not_null())
)

print(f"Train: {len(train_df):,} matches")
print(f"Test: {len(test_df):,} matches")

# Load current model (without calibrator)
registry = ModelRegistry()
version, model_path = registry.get_latest_model()
print(f"Using model: {version}")

# Get raw predictions on test set (for fitting calibrator)
trainer = ModelTrainer()
trainer.load(Path(model_path))
raw_probs = trainer.predict_proba(test_df)

# Get actual outcomes
actual = test_df['player_won'].to_numpy().astype(int)

# Fit calibrator
print("\nFitting isotonic calibrator...")
calibrator = ProbabilityCalibrator()
calibrator.fit(raw_probs, actual)

# Show calibration stats
print(f"\nCalibration Stats:")
print(f"  Samples: {calibrator.calibration_stats['n_samples']:,}")
print(f"  Raw mean prob: {calibrator.calibration_stats['raw_mean']:.3f}")
print(f"  Calibrated mean: {calibrator.calibration_stats['calibrated_mean']:.3f}")
print(f"  Actual win rate: {calibrator.calibration_stats['actual_mean']:.3f}")
print(f"  Max adjustment: {calibrator.calibration_stats['max_adjustment']:.3f}")

# Save calibrator
calibrator_path = Path(model_path).parent / "calibrator.joblib"
calibrator.save(calibrator_path)
print(f"\nCalibrator saved to: {calibrator_path}")

# Verify by running calibration check
print("\n" + "="*60)
print("CALIBRATION BEFORE vs AFTER")
print("="*60)

calibrated_probs = calibrator.calibrate(raw_probs)

# Compare by bin
bins = [(0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.0)]
print(f"\n{'Bin':<12} | {'Raw Pred':>10} | {'Calibrated':>10} | {'Actual':>10}")
print("-"*60)

for low, high in bins:
    mask = (raw_probs >= low) & (raw_probs < high)
    if mask.sum() > 0:
        raw_mean = raw_probs[mask].mean() * 100
        cal_mean = calibrated_probs[mask].mean() * 100
        act_mean = actual[mask].mean() * 100
        print(f"{low:.0%}-{high:.0%}      | {raw_mean:>9.1f}% | {cal_mean:>9.1f}% | {act_mean:>9.1f}%")

print("\n[OK] Calibrator ready for use!")
