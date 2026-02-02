"""Quick calibration analysis script."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import polars as pl
from datetime import datetime, date
from src.model import Predictor, ModelRegistry

# Load data
data_path = Path('data/processed/features_dataset.parquet')
df = pl.read_parquet(data_path)

# Load model
registry = ModelRegistry()
version, model_path = registry.get_latest_model()
predictor = Predictor(Path(model_path))

# Test data only (after 2025-01-01)
TRAIN_CUTOFF = date(2025, 1, 1)
cutoff_ts = int(datetime.combine(TRAIN_CUTOFF, datetime.min.time()).timestamp())
test_df = df.filter(
    (pl.col('start_timestamp') >= cutoff_ts) &
    (pl.col('player_won').is_not_null())
)

# Get predictions
test_df = predictor.predict_with_value(test_df, min_edge=0.0)

# Calibration analysis
print('='*60)
print('  MODEL CALIBRATION ANALYSIS (v1.0.2)')
print('='*60)
print()

# Create probability bins
calibration_df = test_df.with_columns([
    pl.when(pl.col('model_prob') < 0.3).then(pl.lit('<30%'))
    .when(pl.col('model_prob') < 0.4).then(pl.lit('30-40%'))
    .when(pl.col('model_prob') < 0.5).then(pl.lit('40-50%'))
    .when(pl.col('model_prob') < 0.6).then(pl.lit('50-60%'))
    .when(pl.col('model_prob') < 0.7).then(pl.lit('60-70%'))
    .when(pl.col('model_prob') < 0.8).then(pl.lit('70-80%'))
    .otherwise(pl.lit('>80%'))
    .alias('prob_bin')
])

# Group by bin
calibration = calibration_df.group_by('prob_bin').agg([
    pl.len().alias('count'),
    pl.col('model_prob').mean().alias('avg_predicted'),
    pl.col('player_won').mean().alias('actual_win_rate'),
]).sort('avg_predicted')

print(f"{'Prob Bin':<12} | {'Count':>6} | {'Predicted':>10} | {'Actual':>10} | {'Diff':>8}")
print('-'*60)

total_abs_error = 0
total_count = 0
for row in calibration.iter_rows(named=True):
    pred = row['avg_predicted'] * 100
    actual = row['actual_win_rate'] * 100
    diff = actual - pred
    total_abs_error += abs(diff) * row['count']
    total_count += row['count']
    
    indicator = '[OK]' if abs(diff) < 5 else '[!]' if abs(diff) < 10 else '[!!]'
    print(f"{row['prob_bin']:<12} | {row['count']:>6} | {pred:>9.1f}% | {actual:>9.1f}% | {diff:>+7.1f}% {indicator}")

print('-'*60)
mae = total_abs_error / total_count
print(f'Mean Absolute Calibration Error: {mae:.2f}%')
print()

# Brier Score
brier = ((test_df['model_prob'] - test_df['player_won'].cast(pl.Float64))**2).mean()
print(f'Brier Score: {brier:.4f} (lower is better, random=0.25)')

# Additional: Reliability diagram data
print()
print('='*60)
print('  CALIBRATION SUMMARY')
print('='*60)
if mae < 3:
    print('  [EXCELLENT] Model is well-calibrated (MAE < 3%)')
elif mae < 5:
    print('  [GOOD] Model is reasonably calibrated (MAE < 5%)')
elif mae < 10:
    print('  [MODERATE] Model shows some miscalibration (MAE < 10%)')
else:
    print('  [POOR] Model is poorly calibrated (MAE >= 10%)')
