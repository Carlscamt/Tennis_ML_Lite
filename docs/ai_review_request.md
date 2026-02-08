# Tennis ML Lite - AI Architecture Review Request

## Purpose
This document provides a comprehensive overview of the Tennis ML Lite application for AI analysis and improvement recommendations.

---

## Executive Summary

Tennis ML Lite is a CLI-based machine learning pipeline for predicting ATP tennis match outcomes and identifying value betting opportunities. The system scrapes match data, engineers features, trains XGBoost models, and uses probabilistic approaches to determine when to bet.

---

## Technology Stack

### Core Languages & Runtime
| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Primary language |
| Type Hints | Strict via mypy | Code quality |

### Machine Learning
| Library | Purpose |
|---------|---------|
| XGBoost | Binary classification for match winner prediction |
| scikit-learn | Cross-validation, isotonic calibration, TimeSeriesSplit |
| NumPy | Numerical operations |

### Data Processing
| Library | Purpose |
|---------|---------|
| Polars | High-performance DataFrame operations (replaces pandas for most ops) |
| Pandas | Legacy compatibility, some ML integrations |
| Pandera[polars] | Schema validation for data quality |

### API & Web
| Library | Purpose |
|---------|---------|
| FastAPI | REST API for predictions |
| Uvicorn | ASGI server |
| tls-client | TLS fingerprint spoofing for web scraping |
| httpx | Fallback HTTP client |

### Configuration & Validation
| Library | Purpose |
|---------|---------|
| pydantic | Data validation |
| pydantic-settings | Strongly typed configuration from env vars |
| python-dotenv | Environment variable loading |

### Orchestration & Scheduling
| Library | Purpose |
|---------|---------|
| Prefect | Workflow orchestration |
| schedule | Simple task scheduling |

### Observability
| Library | Purpose |
|---------|---------|
| structlog | Structured JSON logging |
| prometheus-client | Metrics collection |

### Infrastructure
| Tool | Purpose |
|------|---------|
| Docker | Containerization (python:3.11-slim base) |
| GitHub Actions | CI/CD workflows |
| pytest | Testing framework |
| ruff | Linting and formatting |

---

## System Architecture

### Module Structure (130+ Python files)

```
Tennis lite/
├── tennis.py                 # CLI entry point (argparse)
├── src/
│   ├── pipeline.py           # Main orchestration
│   ├── scraper.py            # Historical data scraping
│   ├── analysis/
│   │   └── backtester.py     # Enhanced backtesting with latency simulation
│   ├── api/
│   │   ├── main.py           # FastAPI app
│   │   ├── routes.py         # API endpoints
│   │   └── schema.py         # Pydantic models
│   ├── betting/
│   │   ├── signals.py        # Value bet identification
│   │   ├── bankroll.py       # Position sizing
│   │   ├── risk.py           # Risk management
│   │   └── tracker.py        # Bet tracking
│   ├── config/
│   │   └── settings.py       # Typed Pydantic-settings config
│   ├── core/
│   │   ├── container.py      # Dependency injection
│   │   └── protocols.py      # Interface definitions
│   ├── extract/
│   │   ├── data_loader.py    # Parquet loading
│   │   └── sofascore_client.py # Live API client
│   ├── flows/                # Prefect workflows
│   │   ├── daily_pipeline.py
│   │   ├── features.py
│   │   ├── predict.py
│   │   ├── scrape.py
│   │   └── train.py
│   ├── model/
│   │   ├── trainer.py        # XGBoost training
│   │   ├── calibrator.py     # Isotonic regression
│   │   ├── serving.py        # Model server (canary, shadow, fallback)
│   │   ├── registry.py       # Model lifecycle management
│   │   ├── optimization.py   # Profit-based hyperparameter tuning
│   │   ├── cv.py             # Cross-validation
│   │   └── metrics.py        # Evaluation metrics
│   ├── transform/            # Feature engineering
│   └── utils/
│       └── observability.py  # Logging + Prometheus
├── data/
│   ├── raw/*.parquet         # Raw match data
│   └── processed/features.parquet
├── models/registry/
│   ├── production/           # Active model
│   ├── staging/              # Candidate model
│   └── experiments/          # Test models
└── tests/                    # Unit + integration tests
```

---

## Key Processes

### 1. Data Scraping Pipeline

**Source**: SofaScore API (unofficial)

**Implementation** (`src/extract/sofascore_client.py`):
- TLS fingerprint spoofing (Firefox 120) via `tls-client` to bypass Cloudflare
- Thread-safe singleton with request locking
- Rate limiting with configurable delays (1.5-3.0s default)
- Exponential backoff on 403/429 responses (10s, 20s, 40s)
- Circuit breaker pattern (5 failures → 300s cooldown)
- Response caching (300s TTL)

**Data Collected**:
- Player rankings (ATP/WTA)
- Historical matches per player
- Upcoming matches
- Match odds (converted from fractional to decimal)

**Storage**: Raw Parquet files with progressive saves and deduplication

---

### 2. Feature Engineering

**Implementation** (`src/transform/`, `docs/features.md`):

| Feature Category | Features |
|-----------------|----------|
| Rolling Windows (5, 10, 20 matches) | win_rate, set_win_rate, avg_games |
| Elo Ratings | elo_rating, elo_opponent, elo_diff (K=32, Initial=1500) |
| Surface-Specific | surface_win_rate, surface_matches |
| Head-to-Head | h2h_wins, h2h_matches, h2h_win_rate |
| Fatigue/Form | days_since_last, matches_last_14d, recent_form |
| Odds | odds_player, odds_opponent, implied_prob |

**Validation**: Pandera schemas enforce:
- Non-null required fields
- Odds range: 1.01 ≤ odds ≤ 100.0
- Probability range: 0.0 ≤ prob ≤ 1.0
- Elo range: 1000 ≤ elo ≤ 3000

---

### 3. Model Training

**Algorithm**: XGBoost Binary Classifier

**Hyperparameters** (default):
```python
max_depth = 6
learning_rate = 0.05
n_estimators = 500
subsample = 0.8
colsample_bytree = 0.8
objective = "binary:logistic"
eval_metric = "logloss"
```

**Probability Calibration**:
- Isotonic regression via `CalibratedClassifierCV`
- 5-fold calibration post-training
- Purpose: Ensure predicted probability ≈ actual win rate for Kelly staking

**Hyperparameter Optimization** (`src/model/optimization.py`):
- Walk-Forward Validation with `TimeSeriesSplit`
- Objective: Maximize ROI (not accuracy!)
- Grid search over parameter combinations

---

### 4. Model Lifecycle & Serving

**Stages**: Experimental → Staging → Production → Archived

**Promotion Rules**:
| Transition | Requirements |
|------------|--------------|
| Experimental → Staging | AUC ≥ 0.65, Log Loss ≤ 0.68 |
| Staging → Production | Shadow testing OK, ROI > 0%, manual approval |

**Serving Modes** (`src/model/serving.py`):
| Mode | Behavior |
|------|----------|
| CHAMPION_ONLY | Production model only |
| SHADOW | Challenger predictions logged (not returned) |
| CANARY | X% traffic routed to Challenger |
| FALLBACK | Use Staging if Production fails |

---

### 5. Value Betting Logic

**Implementation** (`src/betting/signals.py`):

**Edge Calculation**:
```python
edge = (predicted_prob * odds) - 1
value_bet = edge >= min_edge
```

**Segment-Specific Thresholds**:
| Segment | Min Edge | Rationale |
|---------|----------|-----------|
| Grand Slams | 4.0% | Efficient markets |
| ATP 1000 | 5.0% | Standard |
| Challenger | 7.0% | Noisier data |
| Longshots (odds > 3.0) | 8.0% | High variance |

**Uncertainty Filtering**:
- Margin filter: Reject if |p - 0.5| < 0.10 (near coin-flip)
- Entropy filter: Reject if entropy > 0.65
- Dynamic threshold: `effective_edge = min_edge + k * uncertainty_std`

**Bet Grading**: A+, A, B, C, D based on edge and confidence

---

### 6. Bankroll Management

**Kelly Staking** (`src/config/settings.py`):
```python
kelly_fraction = 0.25           # Quarter Kelly (conservative)
max_bet_fraction = 0.01         # Max 1% per bet
min_bet_fraction = 0.005        # Min 0.5% per bet
max_daily_staked_fraction = 0.10 # Max 10% staked/day
max_bets_per_day = 10
```

**Formula**:
```python
full_kelly = (b * p - q) / b    # where b = odds - 1
stake = full_kelly * kelly_fraction
stake = min(stake, max_bet_fraction)
stake = min(stake, remaining_daily)
```

---

### 7. Enhanced Backtesting

**Implementation** (`src/analysis/backtester.py`):

**Features**:
- Latency simulation (configurable delay in minutes)
- Line movement modeling (odds move against you)
- Book selection strategies: best, average, specific
- Sequential Kelly sizing with bankroll tracking

**Metrics Calculated**:
- Total bets, win rate, ROI %
- Max drawdown, Sharpe ratio
- Daily and monthly aggregations
- Bet log with full audit trail

**Parameter Tuning**:
- Grid search over Kelly fractions, edge thresholds, latencies
- Walk-forward validation

---

### 8. Observability

**Structured Logging** (`structlog`):
- JSON format option for production
- Context managers for tracing

**Prometheus Metrics**:
- Exposed on configurable port (default 8000)
- Custom counters for predictions, bets, errors

---

### 9. Configuration System

**Implementation** (`src/config/settings.py`):

Strongly typed via `pydantic-settings`:
```python
class Settings(BaseSettings):
    scraper: ScraperSettings      # SCRAPER_* env vars
    model: ModelSettings          # MODEL_* env vars
    betting: BettingSettings      # BETTING_* env vars
    observability: ObservabilitySettings  # ENVIRONMENT, LOG_LEVEL
    features: FeatureSettings     # FEATURE_* env vars
```

**Load Order**: defaults → .env file → environment variables

**Field Validation**: Pydantic validators enforce ranges and constraints at startup

---

## Current Known Limitations

1. **Single Data Source**: Only SofaScore (unofficial API, risk of blocks)
2. **ATP Only**: WTA support is partial
3. **No Live Odds API**: No integration with bookmaker APIs for real-time odds
4. **Manual Deployment**: No automated model promotion CI/CD
5. **Single Model**: Only XGBoost, no ensemble or stacking
6. **No Database**: All storage is file-based (Parquet)
7. **Limited Backtesting Realism**: No slippage, order book depth, or market impact

---

## Questions for AI Analysis

1. What architectural patterns could improve maintainability?
2. Are there better ML approaches for this problem domain?
3. How can we improve feature engineering for tennis predictions?
4. What are the risks in the current scraping approach?
5. How could the betting logic be more sophisticated?
6. What observability improvements would add the most value?
7. Are there any obvious performance bottlenecks?
8. What security concerns exist?

---

## AI Prompt for Improvements

Use the following prompt when asking an AI for recommendations:

```
You are a senior ML engineer and sports betting systems architect reviewing a tennis match prediction and value betting application.

### Context
I have a Python application with the following stack:
- XGBoost classifier with isotonic calibration for match outcome prediction
- Polars for data processing (130+ matches scraped via unofficial API)
- Feature engineering: Elo ratings, rolling win rates, H2H, surface-specific stats
- Value betting: Kelly staking with segment-specific edge thresholds
- Model lifecycle: experimental → staging → production with shadow/canary modes
- Enhanced backtesting with latency simulation and line movement modeling
- Pydantic-settings for typed configuration

### Current Architecture Highlights
1. Data comes from a single unofficial API (SofaScore) with TLS fingerprint spoofing
2. Model is trained to maximize ROI via Walk-Forward validation
3. Betting thresholds vary by tournament tier (Grand Slam: 4%, Challenger: 7%)
4. Uncertainty filtering rejects bets with high entropy or near 50% probability
5. Kelly staking capped at 1% per bet, 10% daily exposure

### Request
Please analyze this system and provide specific, actionable recommendations in these areas:

1. **Machine Learning Improvements**
   - Alternative models or ensembles to consider
   - Feature engineering enhancements specific to tennis
   - Calibration alternatives to isotonic regression

2. **Betting Strategy Improvements**
   - Staking optimizations beyond simple Kelly
   - Portfolio-level risk management
   - Market efficiency considerations

3. **Data Pipeline Improvements**
   - Alternative data sources
   - Reducing scraping risk
   - Real-time odds integration

4. **Architecture Improvements**
   - Database vs file storage trade-offs
   - Queue-based processing for predictions
   - Microservices vs monolith considerations

5. **Observability & Reliability**
   - Critical metrics to track
   - Alerting recommendations
   - Disaster recovery

6. **Quick Wins**
   - Low-effort, high-impact changes
   - Common mistakes to avoid

Provide prioritized recommendations with effort estimates (low/medium/high) and expected impact.
```

---

## File References

| Document | Path |
|----------|------|
| README | [README.md](../README.md) |
| Architecture | [docs/architecture.md](architecture.md) |
| Features | [docs/features.md](features.md) |
| Model Config | [docs/model.md](model.md) |
| Settings | [src/config/settings.py](../src/config/settings.py) |
| Backtester | [src/analysis/backtester.py](../src/analysis/backtester.py) |
| Value Betting | [src/betting/signals.py](../src/betting/signals.py) |
| Optimization | [src/model/optimization.py](../src/model/optimization.py) |
