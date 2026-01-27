# Model Audit Report

**Generated**: 2026-01-26T18:33:32.004474
**Model**: `model.bin`
**Test Set Size**: 2,439 matches

## Overall Results

| Metric | Value |
|--------|-------|
| **Overall Score** | 68.7/100 |
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

**Score**: 31.1/100

### Key Metrics
```
accuracy: 0.6937
precision: 0.7065
recall: 0.9344
f1_score: 0.8046
auc_roc: 0.6557
auc_pr: 0.7853
brier_score: 0.2041
log_loss: 0.5954
```

## [PASS] Calibration

**Score**: 79.8/100

### Key Metrics
```
expected_calibration_error: 0.0202
maximum_calibration_error: 0.0937
```

### Warnings
- [!] 25.0% of wrong predictions had >70% confidence

## [PASS] Temporal Stability

**Score**: 81.3/100

### Key Metrics
```
n_months: 12
accuracy_mean: 0.6982
accuracy_std: 0.0374
```

## [PASS] Robustness

**Score**: 87.1/100

### Key Metrics
```
bootstrap_samples: 100
accuracy_ci_width: 0.038
```

## [PASS] Feature Importance

**Score**: 70.0/100

### Key Metrics
```
n_features: 45
zero_importance_count: 1
```

## [WARN] Bias & Fairness

**Score**: 50.0/100

### Key Metrics
```
```

### Warnings
- [!] Large accuracy gap between odds segments: 78.0% vs 52.1%

## [WARN] Adversarial Edge Cases

**Score**: 50.0/100

### Key Metrics
```
```

### Warnings
- [!] 12.7% of predictions are high-confidence errors
