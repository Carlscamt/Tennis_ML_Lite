# Feature Dictionary

This document describes all features used in the tennis prediction model.

## Core Features

### Match Identifiers

| Feature | Type | Description | Source |
|---------|------|-------------|--------|
| `event_id` | int | Unique match identifier | API |
| `player_id` | int | Player identifier | API |
| `opponent_id` | int | Opponent identifier | API |
| `match_date` | datetime | Match start time | API |

### Odds Features

| Feature | Type | Description | Formula |
|---------|------|-------------|---------|
| `odds_player` | float | Bookmaker odds for player | Direct |
| `odds_opponent` | float | Bookmaker odds for opponent | Direct |
| `implied_prob` | float | Implied probability | `1 / odds_player` |

---

## Engineered Features

### Player Performance (Rolling Windows)

Rolling statistics computed over configurable windows (default: 5, 10, 20 matches).

| Feature | Description | Formula |
|---------|-------------|---------|
| `win_rate_{window}d` | Win rate over last N matches | `wins / matches` |
| `set_win_rate_{window}d` | Set win rate | `sets_won / sets_played` |
| `avg_games_{window}d` | Average games won | `mean(games_won)` |

### Elo Ratings

| Feature | Description | Parameters |
|---------|-------------|------------|
| `elo_rating` | Player Elo rating | K=32, Initial=1500 |
| `elo_opponent` | Opponent Elo rating | Same |
| `elo_diff` | Rating difference | `elo_rating - elo_opponent` |

### Surface-Specific

| Feature | Description |
|---------|-------------|
| `surface_win_rate` | Win rate on current surface |
| `surface_matches` | Matches played on surface |

### Head-to-Head

| Feature | Description |
|---------|-------------|
| `h2h_wins` | Wins against opponent |
| `h2h_matches` | Total matches vs opponent |
| `h2h_win_rate` | H2H win rate |

### Fatigue/Form

| Feature | Description |
|---------|-------------|
| `days_since_last` | Days since last match |
| `matches_last_14d` | Match load (fatigue) |
| `recent_form` | Win rate last 5 matches |

---

## Feature Settings

From `src/config/settings.py`:

```python
class FeatureSettings:
    rolling_windows: [5, 10, 20]  # Match windows
    min_matches: 3               # Minimum matches for stats
    elo_k_factor: 32.0           # Elo K-factor
    elo_initial: 1500.0          # Initial Elo rating
    enable_surface_features: True
    enable_h2h_features: True
    enable_fatigue_features: True
```

---

## Feature Validation Rules

1. **Non-null required**: `event_id`, `player_id`, `opponent_id`, `odds_player`
2. **Odds range**: `1.01 ≤ odds ≤ 100.0`
3. **Probability range**: `0.0 ≤ prob ≤ 1.0`
4. **Elo range**: `1000 ≤ elo ≤ 3000`
