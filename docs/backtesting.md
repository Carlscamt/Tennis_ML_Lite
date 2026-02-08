# Backtesting

Enhanced backtesting with realistic simulation and analytics.

---

## EnhancedBacktester

Located in [src/analysis/backtester.py](file:///c:/Users/Carlos/Documents/Antigravity/Tennis%20lite/src/analysis/backtester.py)

### Features

| Feature | Description |
|---------|-------------|
| **Latency Simulation** | Line movement based on delay (default 5 min) |
| **Book Selection** | best, average, or specific bookmaker |
| **Daily/Monthly Stats** | ROI, volatility, cumulative P&L |
| **Parameter Tuning** | Grid search over Kelly, edge, latency |

---

## Configuration

```python
from src.analysis.backtester import EnhancedBacktester, BacktestConfig

config = BacktestConfig(
    latency_minutes=15,           # Delay between prediction and bet
    line_movement_pct=0.02,       # Expected line movement (2%)
    book_selection="average",      # best | average | specific
    kelly_fraction=0.25,
    min_edge=0.05,
    initial_bankroll=1000.0,
    exclude_tiers=["futures"]     # Skip small tournaments
)

bt = EnhancedBacktester(config)
```

---

## Usage

### Basic Backtest

```python
result = bt.run(predictions_df)
bt.print_report(result)
```

### Parameter Tuning

```python
tuning = bt.tune_parameters(
    predictions_df,
    kelly_fractions=[0.1, 0.25, 0.5],
    edge_thresholds=[0.03, 0.05, 0.07],
    latencies=[0, 5, 30]
)
print(tuning)  # Sorted by ROI
```

### Compare Book Strategies

```python
comparison = bt.compare_book_strategies(predictions_df)
print(comparison)  # Shows ROI lift from line shopping
```

---

## BacktestResult

| Field | Type | Description |
|-------|------|-------------|
| `total_bets` | int | Total bets placed |
| `win_rate` | float | Win percentage |
| `roi_pct` | float | Return on investment |
| `max_drawdown` | float | Maximum drawdown |
| `sharpe_ratio` | float | Risk-adjusted return |
| `daily_stats` | DataFrame | Daily P&L breakdown |
| `monthly_stats` | DataFrame | Monthly aggregation |
| `bet_log` | DataFrame | All individual bets |

---

## Latency Simulation

```
executed_odds = quoted_odds × (1 - movement)
movement = line_movement_pct × (latency_minutes / 5)
```

| Latency | Movement |
|---------|----------|
| 0 min | 0% |
| 5 min | 2% |
| 15 min | 6% |
| 30 min | 12% |
