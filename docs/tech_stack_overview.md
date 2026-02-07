# Tennis Prediction System - Technical Overview for AI Review

## AI Review Prompt

```
You are an expert software architect specializing in ML pipelines and sports betting systems.

Review the following technical overview of a Tennis Prediction application. Provide specific, actionable recommendations for:

1. **Architecture Improvements** - Identify patterns that don't scale or could be simplified
2. **ML Pipeline Enhancements** - Feature engineering, model training, evaluation gaps
3. **Data Pipeline Reliability** - Scraping, storage, validation improvements
4. **Production Readiness** - Deployment, monitoring, alerting gaps
5. **Code Quality** - Testing coverage, documentation, maintainability
6. **Business Logic** - Value betting strategy, bankroll management, risk mitigation

For each recommendation:
- State the current limitation
- Explain why it matters
- Provide a concrete implementation suggestion
- Estimate effort (Low/Medium/High)

Prioritize recommendations by impact. Be specific about libraries, patterns, and approaches.
```

---

## System Overview

**Goal:** Predict ATP/WTA tennis match outcomes to identify value betting opportunities.

**High-Level Flow:**
```
Scrape → Transform → Train → Predict → Output Value Bets
```

---

## Technology Stack

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| **Language** | Python | 3.11+ | Core runtime |
| **ML Framework** | XGBoost | 2.0+ | Gradient boosting classifier |
| **Data Processing** | Polars | 0.19+ | Fast DataFrame operations |
| **Validation** | Pandera | 0.18+ | Schema validation for DataFrames |
| **HTTP Client** | tls-client / httpx | 1.0+ | API requests with TLS fingerprinting |
| **API Framework** | FastAPI | 0.100+ | Optional REST endpoint |
| **Logging** | structlog | 24.1+ | Structured JSON logging |
| **Metrics** | prometheus-client | 0.19+ | Metrics exposition |
| **Visualization** | Plotly | 5.18+ | Charts and graphs |
| **Config** | python-dotenv | 1.0+ | Environment variables |
| **Scheduling** | schedule | 1.2+ | Basic task scheduling |
| **Infrastructure** | Docker | - | Containerization |
| **CI/CD** | GitHub Actions | - | Automated testing |
| **Storage** | Parquet files | - | Data persistence |
| **Database** | SQLite | Built-in | Task queue persistence |

---

## Project Structure

```
Tennis lite/
├── tennis.py                  # CLI entry point (argparse)
├── src/
│   ├── pipeline.py            # Main orchestration (TennisPipeline class)
│   ├── scraper.py             # SofaScore API data collection
│   ├── schema.py              # Pandera schemas
│   ├── transform/
│   │   ├── features.py        # Feature engineering (33KB, ~1000 lines)
│   │   ├── leakage_guard.py   # Temporal leakage prevention
│   │   └── validators.py      # Data validation
│   ├── model/
│   │   ├── trainer.py         # XGBoost training
│   │   ├── predictor.py       # Inference service
│   │   ├── registry.py        # Model versioning (Experimental→Staging→Production)
│   │   ├── serving.py         # Canary/Shadow/Fallback modes
│   │   ├── calibrator.py      # Probability calibration
│   │   └── optimization.py    # Hyperparameter tuning
│   ├── serving/
│   │   └── batch_job.py       # Batch inference scheduler
│   └── utils/
│       ├── observability.py   # Logging + Prometheus metrics
│       ├── task_queue.py      # SQLite-backed task queue
│       └── response_archive.py # Gzip compressed API response storage
├── scripts/
│   ├── backtest_roi_analysis.py  # ROI backtesting
│   └── optimize_model.py         # Hyperparameter optimization
├── tests/
│   ├── unit/           # 16 test files
│   ├── integration/    # 2 test files
│   └── e2e/            # 1 test file
├── data/
│   ├── raw/            # Raw scraped parquet files
│   ├── processed/      # Feature-engineered datasets
│   └── .archive/       # Gzip-compressed API response archive
├── models/
│   ├── registry.json   # Model metadata
│   └── v1.0.x/         # Versioned model artifacts
├── config/
│   └── features.yaml   # Feature configuration
└── docs/
    └── architecture.md # System architecture
```

---

## Data Pipeline

### 1. Data Collection (scraper.py)

**Source:** SofaScore API (unofficial)

**Features:**
- `RateLimitCircuitBreaker`: Opens after 2 consecutive 403/429 errors, backs off 15 minutes
- `ResponseCache`: File-based TTL cache (rankings: 24h, matches: 1h, odds: 15m)
- `CheckpointManager`: Resume scraping from last checkpoint after crash
- `TaskQueue`: SQLite-backed pub/sub queue for resilient task processing
- `ResponseArchive`: Gzip-compressed raw JSON storage for future re-processing

**Data Collected:**
- Player rankings (ATP/WTA singles)
- Historical matches (past 5 years)
- Match details (stats, scores, surfaces)
- Live/upcoming matches
- Betting odds from multiple bookmakers

**Output:** `data/raw/*.parquet`

### 2. Feature Engineering (transform/features.py)

**~50+ Features Generated:**

| Category | Examples |
|----------|----------|
| Player Stats | Win rate (overall, surface, vs ranking tier) |
| Rolling Stats | Last 5/10/20 match performance |
| H2H | Head-to-head record, recent meetings |
| Surface | Hard/Clay/Grass specialization |
| Momentum | Recent form, streak length |
| Fatigue | Days since last match, matches this week |
| Tournament | Slam/Masters/250/500 performance |
| Odds-Derived | Implied probability, market movement |

**Temporal Safety:**
- `LeakageGuard`: Ensures all features use only pre-match data
- Train/test split by date (80% train, sorted chronologically)

**Output:** `data/processed/features_dataset.parquet`

### 3. Schema Validation (schema.py)

Uses Pandera to validate:
- Required columns present
- Data types correct
- Value ranges (probabilities 0-1, odds > 1.0)
- No nulls in critical columns

---

## ML Pipeline

### Model Training (model/trainer.py)

**Algorithm:** XGBoost Binary Classifier

**Current Hyperparameters:**
```python
{
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "early_stopping_rounds": 10
}
```

**Training Process:**
1. Load processed features
2. Split by date (not random)
3. Train XGBoost
4. Evaluate (AUC, log loss, accuracy)
5. Register in model registry

### Model Registry (model/registry.py)

**Stages:**
```
Experimental → Staging → Production → Archived
```

**Promotion Rules:**
- Staging → Production: Requires AUC > previous production
- Metadata tracked: metrics, timestamps, promoter

**Current Models:**
- `v1.0.2` (Staging): AUC 0.662
- `v1.0.3` (Staging): AUC 0.665

### Model Serving (model/serving.py)

**Modes:**
1. **Champion Only**: Production model serves all traffic
2. **Shadow Mode**: Champion + Challenger (logged, not returned)
3. **Canary Mode**: 10% traffic to Challenger
4. **Fallback**: Auto-switch if Champion fails

### Probability Calibration (model/calibrator.py)

- Platt scaling for post-hoc calibration
- Expected Value (EV) gating before outputting bets

---

## Prediction & Betting Logic

### Value Detection

```python
edge = predicted_prob - implied_prob
value_bet = edge > min_edge (default 5%)
```

### EV Gate (calibrator.py)

Only outputs bets where:
- Predicted probability > 0.52 (configurable)
- Kelly criterion stake calculation
- Edge > 5%

### Output

**CLI:** `python tennis.py predict --days 7`

**Output Format:**
- Match details
- Predicted winner
- Confidence %
- Edge vs market
- Suggested stake (Kelly)

---

## Testing

**Current Coverage:**
- 16 unit test files
- 2 integration test files
- 1 e2e test file

**Test Categories:**
- Schema validation
- Feature engineering edge cases
- Model registry operations
- Serving logic (canary, shadow, fallback)
- Data quality checks
- CLI commands

**Command:** `pytest tests/ -v`

---

## Observability

### Logging (structlog)
- Structured JSON logging
- Correlation IDs for request tracing
- Log levels: DEBUG, INFO, WARNING, ERROR

### Metrics (Prometheus)
- Request counts
- Latency histograms
- Model version counters
- Scraper success/failure rates

---

## Automation

### Daily Pipeline (run_daily.bat)
```batch
1. Scrape upcoming matches
2. Generate features
3. Run predictions
4. Output value bets
```

### Scheduling
- Uses `schedule` library for basic timing
- Docker container for isolated execution

---

## Known Limitations

1. **No Real-Time Serving**: Batch-only, no streaming inference
2. **Single Model Type**: Only XGBoost, no ensemble
3. **No A/B Testing Framework**: Canary mode is basic
4. **Limited Backtest**: ROI analysis exists but not comprehensive simulation
5. **No Bankroll Management**: Kelly calculation but no portfolio optimization
6. **Manual Deployment**: No CI/CD for model deployment
7. **No Model Monitoring**: No drift detection or performance decay alerts
8. **Single Data Source**: Only SofaScore, no redundancy
9. **No Feature Store**: Features computed on-demand
10. **Local Storage Only**: No cloud integration

---

## Performance Metrics (Current Model v1.0.3)

| Metric | Value |
|--------|-------|
| AUC | 0.665 |
| Accuracy | ~62% |
| Log Loss | 0.64 |
| Backtest ROI | Under evaluation |

---

## CLI Commands

```bash
# Data collection
python tennis.py scrape historical --top 50 --pages 10
python tennis.py scrape upcoming --days 7

# Training
python tennis.py train

# Predictions
python tennis.py predict --days 7 --output predictions.json

# Model management
python tennis.py list-models
python tennis.py promote v1.0.3 Production
python tennis.py serving-config --shadow true

# Analysis
python tennis.py audit
python tennis.py backtest

# Batch serving
python tennis.py batch-run
python tennis.py show-predictions
```

---

## Configuration

**Environment Variables:**
- `ENVIRONMENT`: development/production
- `LOG_LEVEL`: DEBUG/INFO/WARNING/ERROR
- `SCRAPER_DELAY_MIN`: Minimum delay between requests
- `SCRAPER_DELAY_MAX`: Maximum delay between requests

**Files:**
- `config/features.yaml`: Feature engineering configuration
- `models/registry.json`: Model metadata
- `.env`: Environment variables

---

## Dependencies (requirements.txt)

```
polars>=0.19.0
xgboost>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
tls-client>=1.0.0
httpx>=0.24.0
fastapi>=0.100.0
uvicorn>=0.20.0
structlog>=24.1.0
prometheus-client>=0.19.0
python-dotenv>=1.0.0
pydantic>=2.0.0
pandera[polars]>=0.18.0
schedule>=1.2.0
tqdm>=4.66.0
plotly>=5.18.0
```
