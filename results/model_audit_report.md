# Model Audit Report

**Generated**: 2026-02-06T14:40:14.933827
**Model**: `model.joblib`
**Test Set Size**: 2,439 matches

## Overall Results

| Metric | Value |
|--------|-------|
| **Overall Score** | 71.3/100 |
| **Tests Passed** | 6/8 |
| **Total Warnings** | 3 |

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

**Score**: 32.2/100

### Key Metrics
```
accuracy: 0.6945
precision: 0.7032
recall: 0.9471
f1_score: 0.8071
auc_roc: 0.6612
auc_pr: 0.7845
brier_score: 0.2017
log_loss: 0.5885
```

## [PASS] Calibration

**Score**: 98.3/100

### Key Metrics
```
expected_calibration_error: 0.0017
maximum_calibration_error: 0.015
```

### Warnings
- [!] 24.8% of wrong predictions had >70% confidence

## [PASS] Temporal Stability

**Score**: 82.0/100

### Key Metrics
```
n_months: 12
accuracy_mean: 0.6996
accuracy_std: 0.0359
```

## [PASS] Robustness

**Score**: 87.7/100

### Key Metrics
```
bootstrap_samples: 100
accuracy_ci_width: 0.0415
```

## [PASS] Feature Importance

**Score**: 70.0/100

### Key Metrics
```
n_features: 58
zero_importance_count: 1
```

## [WARN] Bias & Fairness

**Score**: 50.0/100

### Key Metrics
```
```

### Warnings
- [!] Large accuracy gap between odds segments: 78.2% vs 54.1%

## [WARN] Adversarial Edge Cases

**Score**: 50.0/100

### Key Metrics
```
```

### Warnings
- [!] 11.4% of predictions are high-confidence errors
