# Model Configuration

This document covers hyperparameters, calibration approach, thresholds, and promotion rules.

## Model Architecture

**Algorithm**: XGBoost Classifier

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_depth` | 6 | Maximum tree depth |
| `learning_rate` | 0.05 | Learning rate (eta) |
| `n_estimators` | 500 | Number of boosting rounds |
| `subsample` | 0.8 | Row subsampling ratio |
| `colsample_bytree` | 0.8 | Column subsampling ratio |
| `random_state` | 42 | Reproducibility seed |

### Objective

```python
objective = "binary:logistic"
eval_metric = "logloss"
```

---

## Probability Calibration

**Method**: Isotonic Regression (via `CalibratedClassifierCV`)

### Why Calibrate?

Raw XGBoost probabilities are often overconfident. Calibration ensures:
- Predicted probability ≈ actual win rate
- Kelly staking works correctly with calibrated probabilities

### Process

1. Train base XGBoost model
2. Apply 5-fold calibration using isotonic regression
3. Validate calibration with reliability diagrams

---

## Betting Thresholds

### From `settings.py`

| Threshold | Value | Rationale |
|-----------|-------|-----------|
| `min_edge` | 0.05 (5%) | Minimum expected value to bet |
| `min_confidence` | 0.55 | Minimum predicted probability |
| `min_odds` | 1.3 | Avoid very short prices |
| `max_odds` | 5.0 | Avoid longshots (high variance) |

### Edge Calculation

```python
edge = (predicted_prob * odds) - 1
value_bet = edge >= min_edge
```

### Kelly Staking

```python
kelly_fraction = 0.25  # Quarter Kelly
stake_pct = kelly * kelly_fraction
stake_pct = min(stake_pct, max_stake_pct)  # Cap at 5%
```

---

## Model Lifecycle

### Stages

| Stage | Description |
|-------|-------------|
| Experimental | Newly trained, not validated |
| Staging | Passes validation, ready for A/B test |
| Production | Active model serving predictions |
| Archived | Previous production models |

### Promotion Rules

1. **Experimental → Staging**:
   - AUC ≥ 0.65
   - Log Loss ≤ 0.68
   - All tests pass

2. **Staging → Production**:
   - Shadow testing shows comparable performance
   - ROI > 0% over test period
   - Manual approval required

3. **Production → Archived**:
   - Replaced by newer production model
   - Kept for rollback capability

---

## Serving Modes

| Mode | Description |
|------|-------------|
| CHAMPION_ONLY | Production model only |
| SHADOW | Challenger predictions logged (not used) |
| CANARY | X% traffic to challenger |
| FALLBACK | Use staging if production fails |

### Configuration

```python
canary_percentage: float = 0.0   # 0-100%
shadow_mode: bool = False
enable_fallback: bool = True
```

---

## CI/CD Integration

- **train.yml**: Weekly automated training
- **evaluate.yml**: CV + backtest validation
- **promote.yml**: Manual approval to production

See [CI/CD workflows](../.github/workflows/) for details.
