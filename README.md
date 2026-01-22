# Tennis ML Lite

Simplified tennis betting prediction system - streamlined from 23 files to just 9 core modules.

## Quick Start

```bash
pip install -r requirements.txt

# Scrape upcoming matches
python tennis.py scrape upcoming --days 7

# Get predictions
python tennis.py predict --days 7 --min-odds 1.5 --max-odds 3.0
```

## Architecture

```
scrape → validate → features → predict → bet
```

| # | File | Purpose |
|---|------|---------|
| 1 | `tennis.py` | CLI entry point |
| 2 | `src/scraper.py` | Data scraper |
| 3 | `src/schema.py` | Validation |
| 4 | `src/pipeline.py` | Workflows |
| 5 | `src/transform/features.py` | Features |
| 6 | `scripts/run_pipeline.py` | Training |
| 7 | `scripts/model_audit.py` | Evaluation |
| 8 | `scripts/backtest.py` | Testing |
| 9 | `dashboard/app.py` | UI |

## Commands

```bash
python tennis.py scrape historical --top 50
python tennis.py scrape upcoming --days 7
python tennis.py predict --days 7
python tennis.py train
python tennis.py audit
python tennis.py backtest
```

## License

MIT
