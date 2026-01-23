# Model Audit Report

**Generated**: 2026-01-22T22:08:59.204430
**Model**: `xgboost_model`
**Test Set Size**: 1,970 matches

## Overall Results

| Metric | Value |
|--------|-------|
| **Overall Score** | 65.7/100 |
| **Tests Passed** | 6/8 |
| **Total Warnings** | 3 |

## [PASS] Data Leakage

**Score**: 100.0/100

### Key Metrics
```
leakage_detected: False
leaky_feature_count: 0
leakage_percentage: 0.0
safe_feature_count: 45
```

## [PASS] Statistical Performance

**Score**: 38.3/100

### Key Metrics
```
accuracy: 0.6761
precision: 0.7172
recall: 0.8242
f1_score: 0.767
auc_roc: 0.6915
auc_pr: 0.7974
brier_score: 0.2057
log_loss: 0.5978
```

## [PASS] Calibration

**Score**: 58.5/100

### Key Metrics
```
expected_calibration_error: 0.0415
maximum_calibration_error: 0.113
```

### Warnings
- [!] 29.6% of wrong predictions had >70% confidence

## [PASS] Temporal Stability

**Score**: 78.9/100

### Key Metrics
```
n_months: 12
accuracy_mean: 0.6749
accuracy_std: 0.0422
```

## [PASS] Robustness

**Score**: 80.1/100

### Key Metrics
```
bootstrap_samples: 100
accuracy_ci_width: 0.0369
```

## [PASS] Feature Importance

**Score**: 70.0/100

### Key Metrics
```
n_features: 45
zero_importance_count: 2
```

## [WARN] Bias & Fairness

**Score**: 50.0/100

### Key Metrics
```
```

### Warnings
- [!] Large accuracy gap between odds segments: 77.1% vs 50.5%

## [WARN] Adversarial Edge Cases

**Score**: 50.0/100

### Key Metrics
```
```

### Warnings
- [!] 14.2% of predictions are high-confidence errors
