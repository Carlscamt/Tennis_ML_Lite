"""
Rolling statistics features - Win rates, service, return, games, points, winners, errors.
"""
import polars as pl
from .registry import FeatureRegistry
from typing import List, Tuple

# Default configuration
DEFAULT_WINDOWS = (5, 10, 20)
MIN_MATCHES = 3


def _add_rolling_stats(
    df: pl.LazyFrame,
    stats: List[Tuple[str, str]],
    windows: tuple = (10, 20),
    min_periods: int = MIN_MATCHES,
    partition_by: str = "player_id"
) -> pl.LazyFrame:
    """Helper to add shifted rolling statistics."""
    schema = df.collect_schema().names()
    
    for window in windows:
        for raw_col, short_name in stats:
            if raw_col in schema:
                df = df.with_columns([
                    pl.col(raw_col)
                    .cast(pl.Float64)
                    .shift(1)  # CRITICAL: Exclude current match
                    .rolling_mean(window_size=window, min_periods=min_periods)
                    .over(partition_by)
                    .alias(f"player_{short_name}_avg_{window}")
                ])
    
    return df


@FeatureRegistry.register("rolling_win_rate", category="rolling", priority=60)
def add_rolling_win_rate(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add rolling win rate features per player.
    Uses shift(1) to exclude current match.
    """
    for window in DEFAULT_WINDOWS:
        df = df.with_columns([
            pl.col("player_won")
            .cast(pl.Float64)
            .shift(1)  # CRITICAL: Exclude current match
            .rolling_mean(window_size=window, min_periods=MIN_MATCHES)
            .over("player_id")
            .alias(f"player_win_rate_{window}")
        ])
    
    return df


@FeatureRegistry.register("rolling_service_stats", category="rolling", priority=61)
def add_rolling_service_stats(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add shifted rolling service statistics."""
    service_stats = [
        ("player_service_aces", "aces"),
        ("player_service_doublefaults", "double_faults"),
        ("player_service_firstserveaccuracy", "first_serve_pct"),
        ("player_service_firstservepointsaccuracy", "first_serve_won_pct"),
        ("player_service_secondservepointsaccuracy", "second_serve_won_pct"),
        ("player_service_breakpointssaved", "bp_saved"),
    ]
    df = _add_rolling_stats(df, service_stats)
    
    # Opponent service stats (smaller window, partitioned by opponent_id)
    schema = df.collect_schema().names()
    opponent_stats = [
        ("opponent_service_aces", "opp_aces"),
        ("opponent_service_firstserveaccuracy", "opp_first_serve_pct"),
    ]
    
    for raw_col, short_name in opponent_stats:
        if raw_col in schema:
            df = df.with_columns([
                pl.col(raw_col)
                .cast(pl.Float64)
                .shift(1)
                .rolling_mean(window_size=10, min_periods=MIN_MATCHES)
                .over("opponent_id")
                .alias(f"{short_name}_avg_10")
            ])
    
    return df


@FeatureRegistry.register("rolling_return_stats", category="rolling", priority=62)
def add_rolling_return_stats(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add shifted rolling return statistics."""
    return_stats = [
        ("player_return_firstreturnpoints", "first_return_won"),
        ("player_return_secondreturnpoints", "second_return_won"),
        ("player_return_breakpointsscored", "bp_converted"),
    ]
    return _add_rolling_stats(df, return_stats)


@FeatureRegistry.register("rolling_games_stats", category="rolling", priority=63)
def add_rolling_games_stats(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add shifted rolling games statistics.
    Uses SERVICE HOLD % instead of raw games (more predictive, less leaky).
    """
    schema = df.collect_schema().names()
    
    # Calculate service hold percentage
    if "player_games_servicegameswon" in schema and "player_service_servicegamestotal" in schema:
        df = df.with_columns([
            (pl.col("player_games_servicegameswon") / 
             pl.col("player_service_servicegamestotal").clip(1))
            .alias("_player_service_hold_pct")
        ])
        
        for window in [10, 20]:
            df = df.with_columns([
                pl.col("_player_service_hold_pct")
                .shift(1)  # Exclude current match
                .rolling_mean(window_size=window, min_periods=MIN_MATCHES)
                .over("player_id")
                .alias(f"player_service_hold_pct_avg_{window}")
            ])
    
    # Max games streak
    if "player_games_maxgamesinrow" in schema:
        df = df.with_columns([
            pl.col("player_games_maxgamesinrow")
            .cast(pl.Float64)
            .shift(1)
            .rolling_mean(window_size=10, min_periods=MIN_MATCHES)
            .over("player_id")
            .alias("player_max_games_streak_avg_10")
        ])
    
    return df


@FeatureRegistry.register("rolling_points_stats", category="rolling", priority=64)
def add_rolling_points_stats(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add shifted rolling points statistics."""
    schema = df.collect_schema().names()
    
    # Max points in row (momentum indicator)
    if "player_points_maxpointsinrow" in schema:
        df = df.with_columns([
            pl.col("player_points_maxpointsinrow")
            .cast(pl.Float64)
            .shift(1)
            .rolling_mean(window_size=10, min_periods=MIN_MATCHES)
            .over("player_id")
            .alias("player_max_points_streak_avg_10")
        ])
    
    return df


@FeatureRegistry.register("rolling_winners_stats", category="rolling", priority=65)
def add_rolling_winners_stats(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add shifted rolling winners statistics."""
    winner_stats = [
        ("player_winners_winnerstotal", "total_winners"),
        ("player_winners_forehandwinners", "fh_winners"),
        ("player_winners_backhandwinners", "bh_winners"),
    ]
    return _add_rolling_stats(df, winner_stats, windows=(10,))


@FeatureRegistry.register("rolling_errors_stats", category="rolling", priority=66)
def add_rolling_errors_stats(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add shifted rolling errors statistics."""
    schema = df.collect_schema().names()
    
    error_stats = [
        ("player_errors_errorstotal", "total_errors"),
        ("player_unforced_errors_unforcederrorstotal", "total_ue"),
    ]
    df = _add_rolling_stats(df, error_stats, windows=(10,))
    
    # Winner-to-error ratio (aggression vs consistency)
    if "player_winners_winnerstotal" in schema and "player_errors_errorstotal" in schema:
        df = df.with_columns([
            (pl.col("player_winners_winnerstotal") / 
             pl.col("player_errors_errorstotal").clip(1))
            .alias("_player_winner_error_ratio")
        ])
        
        df = df.with_columns([
            pl.col("_player_winner_error_ratio")
            .shift(1)
            .rolling_mean(window_size=10, min_periods=MIN_MATCHES)
            .over("player_id")
            .alias("player_winner_error_ratio_avg_10")
        ])
    
    return df
