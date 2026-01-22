# Tennis ML Lite - Ultra Minimal

Stripped-down tennis betting ML in just **5 files**.

## Files

```
Tennis_ML_Lite/
├── main.py        # CLI & workflows
├── scraper.py     # Data collection
├── model.py       # XGBoost training & prediction
├── features.py    # Feature engineering
├── config.py      # All settings
└── README.md
```

## Quick Start

```bash
pip install polars xgboost scikit-learn httpx tqdm

python main.py scrape --days 7
python main.py predict --min-odds 1.5 --max-odds 3.0
```

## Flow

```
scrape → features → model → predict
```
