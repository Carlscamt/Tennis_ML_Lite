# Betting Module

Core betting infrastructure for value detection, bankroll management, and risk analysis.

---

## Components

### [bankroll.py](file:///c:/Users/Carlos/Documents/Antigravity/Tennis%20lite/src/betting/bankroll.py)

**BankrollManager** — Kelly criterion stake sizing with caps.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kelly_fraction` | 0.25 | Quarter Kelly |
| `max_stake_pct` | 0.01 | Max 1% per bet |
| `max_daily_staked_pct` | 0.10 | Max 10% per day |

```python
from src.betting import BankrollManager
manager = BankrollManager(initial_bankroll=1000)
stake = manager.kelly_stake(model_prob=0.6, odds=2.0)
```

---

### [ledger.py](file:///c:/Users/Carlos/Documents/Antigravity/Tennis%20lite/src/betting/ledger.py)

**BankrollLedger** — SQLite-based persistence with open bet tracking.

| Method | Purpose |
|--------|---------|
| `get_effective_bankroll()` | Bankroll minus open exposure |
| `place_bet()` | Record new bet |
| `settle_bet()` | Mark bet won/lost, update bankroll |

```python
from src.betting import BankrollLedger
ledger = BankrollLedger("data/bankroll.db")
manager.attach_ledger(ledger)  # Now uses effective bankroll
```

---

### [signals.py](file:///c:/Users/Carlos/Documents/Antigravity/Tennis%20lite/src/betting/signals.py)

**ValueBetFinder** — Identifies value bets with segment-aware thresholds.

| Feature | Description |
|---------|-------------|
| Segment thresholds | Different min_edge for Slams/Challengers/Longshots |
| Uncertainty buffer | `edge > min_edge + k × uncertainty_std` |
| Blended probability | Model + market probability blend |

---

### [risk.py](file:///c:/Users/Carlos/Documents/Antigravity/Tennis%20lite/src/betting/risk.py)

**RiskAnalyzer** — Risk-of-ruin diagnostics.

| Method | Purpose |
|--------|---------|
| `ruin_probability()` | Analytical estimate |
| `monte_carlo_drawdown()` | Simulated max drawdown distribution |

```python
from src.betting import RiskAnalyzer
analyzer = RiskAnalyzer()
stats = analyzer.monte_carlo_drawdown(n_simulations=10000)
print(f"95th percentile drawdown: {stats.percentile_95:.1%}")
```

---

### [bookmakers.py](file:///c:/Users/Carlos/Documents/Antigravity/Tennis%20lite/src/betting/bookmakers.py)

**BookmakerSelector** — Multi-book odds selection.

| Strategy | Description |
|----------|-------------|
| `max` | Best available from vetted books |
| `percentile` | 75th percentile (avoid outliers) |
| `average` | Mean across all books |
| `single` | Specific bookmaker |

**Vetted Books:**
- **Tier 1:** Pinnacle, Bet365, Betfair, William Hill
- **Tier 2:** Betway, 888Sport, Betsson

```python
from src.betting import BookmakerSelector, BookmakerConfig
selector = BookmakerSelector(BookmakerConfig(strategy="max"))
```

---

### [tracker.py](file:///c:/Users/Carlos/Documents/Antigravity/Tennis%20lite/src/betting/tracker.py)

**BettingTracker** — JSON-based bet history and P&L tracking.

---

## Configuration

All betting parameters configurable via environment:

```bash
export BETTING_KELLY_FRACTION=0.25
export BETTING_MAX_BET_FRACTION=0.01
export BETTING_MIN_EDGE_CHALLENGER=0.07
export BETTING_UNCERTAINTY_MULTIPLIER=1.0
```
