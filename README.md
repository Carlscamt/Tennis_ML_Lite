---
trigger: always_on
---

# Tennis ML Lite 🎾

A lightweight, CLI-based machine learning pipeline for predicting ATP tennis match outcomes using XGBoost.

## 🚀 Features

*   **Unified CLI**: Single entry point (`tennis.py`) for all operations.
*   **Robust Scraping**: Automated data collection from SofaScore with smart incremental updates.
*   **Machine Learning**: XGBoost model with optimized feature engineering and probability calibration.
*   **Value Betting**: Identifies bets with positive expected value (Edge > 5%).
*   **Leakage-Free**: Strict temporal splitting and validation to prevent data leakage.

## 📋 Prerequisites

*   Python 3.9+
*   Windows/Linux/MacOS

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Carlscamt/Tennis_ML_Lite.git
    cd Tennis_ML_Lite
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Docker Setup (Recommended)**
    Skip Python dependency issues by running in a container.
    ```bash
    # Build image
    docker build -t tennis-cli .
    ```

## 📖 Usage

All interaction is done via the `tennis.py` script.

### 1. Scrape Data
Fetch historical data or upcoming matches. The system handles deduplication and incremental updates.

```bash
# Scrape top 50 players' history
python tennis.py scrape historical --top 50

# Scrape specific players by ID
python tennis.py scrape players --ids 12345,67890

# Scrape upcoming matches for next 7 days
python tennis.py scrape upcoming --days 7
```

### 2. Train Model
Train the XGBoost model on available data. Features are automatically generated.

```bash
python tennis.py train
```

### 3. Get Predictions 🔮
Generate predictions for upcoming matches and find value bets.

```bash
# Predict next 3 days
python tennis.py predict --days 3

# Save predictions to file (CSV/JSON/Parquet)
python tennis.py predict --days 7 --output bets.csv

# Filter for specific odds
python tennis.py predict --days 3 --min-odds 1.5 --max-odds 2.5
```

### 4. Daily Automation 🤖
Use the provided batch script for a one-click daily update.
```bash
# Windows
run_daily.bat
```

### 5. Running with Docker 🐳
All commands work via Docker. Mount the `data` volume to persist scraping results.
```bash
# Scrape
docker run -v %cd%/data:/app/data tennis-cli scrape upcoming

# Predict and save to host machine
docker run -v %cd%/data:/app/data tennis-cli predict --days 3 --output data/bets.csv
```

### 6. Showdown Mode 🏆
Simulate tournament brackets and compare model predictions vs actual results.

```bash
# ASCII bracket in terminal
python tennis.py showdown -t "Wimbledon" -y 2024 --ascii

# HTML visualization (saved to results/)
python tennis.py showdown -t "US Open" -y 2024

# List available tournaments
python tennis.py showdown --list
```

**Output Example (ASCII):**
```
--- Quarterfinals ---
  [+] *Carlos Alcaraz    << vs  Tommy Paul           (80%)
  [+] *Novak Djokovic    << vs  Alex de Minaur       (80%)
  [-]  Jannik Sinner     << vs *Daniil Medvedev      (82%)

Legend: [+] Correct  [-] Wrong  * = Winner  << = Model Pick
```

### 7. Audit & Backtest
Verify model performance.

```bash
# Run model audit
python tennis.py audit

# Run ROI backtest
python tennis.py backtest
```

## 🏗️ Architecture

The system follows a modular architecture:

*   **`src/pipeline.py`**: Orchestrates scraping, ETL, and modeling.
*   **`src/scraper.py`**: Handles data collection.
*   **`src/model/`**: XGBoost training, inference, and **probability calibration**.
*   **`src/transform/`**: Feature engineering with surface-encoded stats.
*   **`data/`**: Stores raw parquet files, processed features, and models.

See [docs/architecture.md](docs/architecture.md) for a detailed diagram.

## 📊 Probability Calibration

The model includes **isotonic calibration** to fix probability biases:

```bash
# Fit calibrator after training
python scripts/fit_calibrator.py

# Check calibration quality
python scripts/check_calibration.py
```

**EV Gating**: Probability-based minimum edge thresholds prevent low-confidence bets:

| Probability | Min Edge Required |
|-------------|-------------------|
| 35-40%      | 4.0%              |
| 40-50%      | 3.5%              |
| 50-60%      | 3.0%              |
| 60-70%      | 2.5%              |
| 70%+        | 2.0%              |

Bets below 35% probability are blocked (insufficient sample size).

## 🤝 Contributing
1.  Fork the repo
2.  Create a feature branch
3.  Commit changes
4.  Push to branch
5.  Create Pull Request
