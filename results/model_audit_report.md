# Model Audit Report

**Generated**: 2026-02-08T13:16:17.049192
**Model**: `model.joblib`
**Test Set Size**: 2,439 matches

## Overall Results

| Metric | Value |
|--------|-------|
| **Overall Score** | 53.0/100 |
| **Tests Passed** | 4/8 |
| **Total Warnings** | 7 |

## [PASS] Data Leakage

**Score**: 100.0/100

### Key Metrics
```
leakage_detected: False
leaky_feature_count: 0
leakage_percentage: 0.0
safe_feature_count: 58
```

## [PASS] Statistical Performance

**Score**: 19.9/100

### Key Metrics
```
accuracy: 0.6622
precision: 0.6746
recall: 0.9648
f1_score: 0.794
auc_roc: 0.5996
auc_pr: 0.7646
brier_score: 0.2686
log_loss: 0.7915
```

### Warnings
- [!] Brier Score (0.269) indicates poor probability estimates

## [WARN] Calibration

**Score**: 0.0/100

### Key Metrics
```
expected_calibration_error: 0.2153
maximum_calibration_error: 0.5846
```

### Warnings
- [!] ECE (0.215) exceeds recommended threshold (0.05)
- [!] MCE (0.585) shows severe miscalibration in some bins
- [!] 98.3% of wrong predictions had >70% confidence

## [PASS] Temporal Stability

**Score**: 83.0/100

### Key Metrics
```
n_months: 12
accuracy_mean: 0.6631
accuracy_std: 0.034
```

## [PASS] Robustness

**Score**: 71.1/100

### Key Metrics
```
bootstrap_samples: 100
accuracy_ci_width: 0.038
```

## [WARN] Feature Importance

**Score**: 50.0/100

### Key Metrics
```
n_features: 58
zero_importance_count: 58
```

### Warnings
- [!] 58 features (100.0%) have zero importance

## [WARN] Bias & Fairness

**Score**: 50.0/100

### Key Metrics
```
```

### Warnings
- [!] Large accuracy gap between odds segments: 75.9% vs 29.1%

## [WARN] Adversarial Edge Cases

**Score**: 50.0/100

### Key Metrics
```
```

### Warnings
- [!] 33.4% of predictions are high-confidence errors
