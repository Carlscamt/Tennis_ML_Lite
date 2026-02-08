# Tennis ML Lite - AI Architecture Review Request

## Purpose
This document provides a comprehensive overview of the Tennis ML Lite application for AI analysis and improvement recommendations.

---

## Executive Summary

Tennis ML Lite is a CLI-based machine learning pipeline for predicting ATP tennis match outcomes and identifying value betting opportunities. The system scrapes match data, engineers features, trains XGBoost models, and uses probabilistic approaches with Kelly staking to determine optimal bet sizing.

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
│   │   └── backtester.py     # Enhanced backtesting (latency, line movement, book selection)
│   ├── api/
│   │   ├── main.py           # FastAPI app
│   │   ├── routes.py         # API endpoints
│   │   └── schema.py         # Pydantic models
│   ├── betting/
│   │   ├── signals.py        # Value bet identification (segment thresholds, uncertainty buffer)
│   │   ├── bankroll.py       # Position sizing (fractional Kelly with caps)
│   │   ├── ledger.py         # SQLite-based bankroll tracking + open bet exposure
│   │   ├── risk.py           # Monte Carlo risk-of-ruin analysis
│   │   ├── bookmakers.py     # Multi-bookmaker odds selection (line shopping)
│   │   └── tracker.py        # Bet history tracking
│   ├── config/
│   │   └── settings.py       # Typed Pydantic-settings config (50+ parameters)
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
blended_prob = model_prob * 0.5 + fair_market_prob * 0.5
edge = blended_prob - implied_prob
value_bet = edge >= effective_min_edge
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

**Kelly Staking** (`src/betting/bankroll.py`):
```python
kelly_fraction = 0.25           # Quarter Kelly (conservative)
max_bet_fraction = 0.01         # Max 1% per bet
min_bet_fraction = 0.005        # Min 0.5% per bet
max_daily_staked_fraction = 0.10 # Max 10% staked/day
max_bets_per_day = 10
```

**Formula**:
```python
full_kelly = (b * p - q) / b    # where b = odds - 1, p = prob, q = 1-p
stake = full_kelly * kelly_fraction
stake = min(stake, max_bet_fraction)
stake = min(stake, remaining_daily)
```

**Open Bet Tracking** (`src/betting/ledger.py`):
- SQLite-based persistence
- Effective bankroll = current_bankroll - open_exposure
- Prevents over-betting when bets are pending

---

### 7. Risk Analysis

**Implementation** (`src/betting/risk.py`):

**Analytical Risk-of-Ruin**:
```python
ruin_prob = ruin_probability(edge=0.05, win_prob=0.55, kelly_fraction=0.25, n_bets=500)
```

**Monte Carlo Simulation**:
- 10,000 simulation paths
- Tracks max drawdown distribution
- Returns 95th/99th percentile drawdowns
- Estimates ruin probability empirically

---

### 8. Multi-Bookmaker Line Shopping

**Implementation** (`src/betting/bookmakers.py`):

**Selection Strategies**:
| Strategy | Description |
|----------|-------------|
| max | Best available from vetted books |
| percentile | 75th percentile (avoid outliers) |
| average | Mean across all books |
| single | Specific bookmaker only |

**Vetted Bookmakers**:
- Tier 1: Pinnacle, Bet365, Betfair, William Hill, BWin, Marathonbet
- Tier 2: Betway, 888Sport, Betsson, Interwetten

---

### 9. Enhanced Backtesting

**Implementation** (`src/analysis/backtester.py`):

**Realism Features**:
- Latency simulation (configurable delay in minutes)
- Line movement modeling (odds move against you based on latency)
- Book selection strategies: best, average, specific
- Sequential Kelly sizing with bankroll tracking

**Metrics Calculated**:
- Total bets, win rate, ROI %
- Max drawdown, Sharpe ratio
- Daily and monthly aggregations
- Bet log with full audit trail

**Parameter Tuning**:
- Grid search over Kelly fractions, edge thresholds, latencies
- Compare single-book vs best-available ROI lift

---

### 10. Observability

**Structured Logging** (`structlog`):
- JSON format option for production
- Context managers for tracing

**Prometheus Metrics**:
- Exposed on configurable port (default 8000)
- Custom counters for predictions, bets, errors

---

### 11. Configuration System

**Implementation** (`src/config/settings.py`):

Strongly typed via `pydantic-settings`:
```python
class Settings(BaseSettings):
    scraper: ScraperSettings      # SCRAPER_* env vars
    model: ModelSettings          # MODEL_* env vars
    betting: BettingSettings      # BETTING_* env vars (50+ parameters)
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
6. **No Database**: All storage is file-based (Parquet) except bankroll (SQLite)
7. **Limited Backtesting Realism**: No slippage, order book depth, or market impact
8. **No Feature Store**: Features computed on-demand, not cached

---

## AI Prompt for Improvement Recommendations

Copy and paste the following prompt into an AI assistant:

```
You are a senior ML engineer and sports betting systems architect with expertise in:
- Machine learning for sports predictions
- Quantitative betting strategies
- Production ML systems
- Python best practices

## System Overview

I have a Python CLI application for ATP tennis match prediction and value betting with:

### Technology Stack
- Python 3.11+ with strict type hints
- XGBoost classifier with isotonic calibration
- Polars for data processing
- pydantic-settings for typed configuration
- FastAPI for optional REST endpoints
- SQLite for bankroll tracking
- Parquet for data storage

### ML Pipeline
- Data: SofaScore API (unofficial, with TLS fingerprint spoofing)
- Features: 50+ including Elo, rolling win rates, H2H, surface stats, fatigue
- Training: Walk-Forward Validation optimizing for ROI (not accuracy)
- Calibration: Isotonic regression for probability calibration
- Registry: Experimental → Staging → Production lifecycle

### Betting System
- Value detection: edge = blended_prob - implied_prob
- Segment thresholds: Grand Slams 4%, Challengers 7%, Longshots 8%
- Uncertainty buffer: effective_edge = min_edge + k * uncertainty_std
- Entropy/margin filters reject uncertain predictions
- Kelly staking: 25% Kelly, 1% max per bet, 10% daily cap
- Open bet tracking: Effective bankroll accounts for pending bets

### Risk Management
- Monte Carlo ruin probability simulation
- Max drawdown analysis (95th/99th percentile)
- Analytical risk-of-ruin estimates

### Backtesting
- Latency simulation (line movement based on delay)
- Book selection comparison (single vs best-of-N)
- Parameter grid search (Kelly, edge, latency)
- Daily/monthly ROI aggregation

### Line Shopping
- Multi-bookmaker odds selection: max, percentile, average, single
- Vetted bookmaker lists (Tier 1/2)
- ROI lift analysis vs single book

## Current Limitations
1. Single unofficial data source (SofaScore)
2. Single model type (XGBoost only)
3. No real-time odds integration
4. File-based storage (no proper database for matches)
5. No drift detection or model monitoring
6. Basic A/B testing (canary mode)
7. No ensemble or stacking

## Request

Provide specific, actionable recommendations in these areas:

### 1. Machine Learning
- Alternative models or ensembles worth trying
- Feature engineering specific to tennis
- Better calibration methods
- Uncertainty quantification improvements

### 2. Betting Strategy
- Kelly variations or alternatives
- Portfolio-level optimization
- Correlated bet handling
- Market efficiency adaptation

### 3. Data & Infrastructure
- Alternative data sources
- Database recommendations
- Real-time odds integration options
- Feature store benefits

### 4. Risk Management
- Drawdown control methods
- Dynamic sizing adjustments
- Ruin probability thresholds

### 5. Production Readiness
- Model monitoring and drift detection
- Alerting recommendations
- Deployment automation

### 6. Quick Wins
- Low-effort, high-impact changes
- Common mistakes to avoid

For each recommendation:
- Current limitation
- Why it matters (quantify if possible)
- Specific implementation suggestion
- Effort estimate (Low/Medium/High)
- Expected impact

Prioritize by impact-to-effort ratio. Be specific about libraries, algorithms, and implementation details.
```

---

## File References

| Document | Path |
|----------|------|
| Architecture | [docs/architecture.md](architecture.md) |
| Features | [docs/features.md](features.md) |
| Model Config | [docs/model.md](model.md) |
| Betting Docs | [docs/betting.md](betting.md) |
| Backtesting Docs | [docs/backtesting.md](backtesting.md) |
| Settings | [src/config/settings.py](../src/config/settings.py) |
| Backtester | [src/analysis/backtester.py](../src/analysis/backtester.py) |
| Value Betting | [src/betting/signals.py](../src/betting/signals.py) |
| Bookmakers | [src/betting/bookmakers.py](../src/betting/bookmakers.py) |
| Ledger | [src/betting/ledger.py](../src/betting/ledger.py) |
| Risk | [src/betting/risk.py](../src/betting/risk.py) |
| Optimization | [src/model/optimization.py](../src/model/optimization.py) |
