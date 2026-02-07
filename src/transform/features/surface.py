"""
Surface features - Surface-specific win rates and statistics.
"""
import polars as pl
from .registry import FeatureRegistry
from typing import List, Tuple


@FeatureRegistry.register("surface_normalization", category="pre_match", priority=20)
def add_surface_normalization(df: pl.LazyFrame) -> pl.LazyFrame:
    """Normalize surface names for consistent grouping."""
    schema = df.collect_schema().names()
    
    if "ground_type" not in schema:
        return df
    
    return df.with_columns([
        pl.col("ground_type")
        .str.to_lowercase()
        .str.replace_all(r".*clay.*", "clay")
        .str.replace_all(r".*grass.*", "grass")
        .str.replace_all(r".*hard.*", "hard")
        .alias("surface_normalized")
    ])


@FeatureRegistry.register("surface_win_rate", category="surface", priority=50)
def add_surface_win_rate(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add surface-specific rolling win rate (shifted)."""
    schema = df.collect_schema().names()
    
    if "surface_normalized" not in schema:
        return df
    
    for window in [10, 20]:
        df = df.with_columns([
            pl.col("player_won")
            .cast(pl.Float64)
            .shift(1)  # Exclude current match
            .rolling_mean(window_size=window, min_periods=3)
            .over(["player_id", "surface_normalized"])
            .alias(f"player_surface_win_rate_{window}")
        ])
    
    return df


def _add_surface_rolling_stats(
    df: pl.LazyFrame,
    stats: List[Tuple[str, str]],
    window: int = 10,
    min_periods: int = 3
) -> pl.LazyFrame:
    """Helper to add surface-specific rolling stats."""
    schema = df.collect_schema().names()
    
    if "surface_normalized" not in schema:
        return df
    
    for raw_col, short_name in stats:
        if raw_col in schema:
            df = df.with_columns([
                pl.col(raw_col)
                .cast(pl.Float64)
                .shift(1)  # CRITICAL: Exclude current match
                .rolling_mean(window_size=window, min_periods=min_periods)
                .over(["player_id", "surface_normalized"])
                .alias(f"player_{short_name}_surface_avg_{window}")
            ])
    
    return df


@FeatureRegistry.register("surface_service_stats", category="surface", priority=51)
def add_surface_rolling_service_stats(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add surface-specific rolling service statistics."""
    service_stats = [
        ("player_service_aces", "aces"),
        ("player_service_doublefaults", "double_faults"),
        ("player_service_firstserveaccuracy", "first_serve_pct"),
        ("player_service_firstservepointsaccuracy", "first_serve_won_pct"),
        ("player_service_secondservepointsaccuracy", "second_serve_won_pct"),
    ]
    return _add_surface_rolling_stats(df, service_stats)


@FeatureRegistry.register("surface_return_stats", category="surface", priority=52)
def add_surface_rolling_return_stats(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add surface-specific rolling return statistics."""
    return_stats = [
        ("player_return_firstreturnpoints", "first_return_won"),
        ("player_return_secondreturnpoints", "second_return_won"),
        ("player_return_breakpointsscored", "bp_converted"),
    ]
    return _add_surface_rolling_stats(df, return_stats)


@FeatureRegistry.register("surface_winner_stats", category="surface", priority=53)
def add_surface_rolling_winners_stats(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add surface-specific rolling winners statistics."""
    winner_stats = [
        ("player_winners_winnerstotal", "total_winners"),
        ("player_winners_forehandwinners", "fh_winners"),
        ("player_winners_backhandwinners", "bh_winners"),
    ]
    return _add_surface_rolling_stats(df, winner_stats)


@FeatureRegistry.register("surface_error_stats", category="surface", priority=54)
def add_surface_rolling_errors_stats(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add surface-specific rolling errors statistics."""
    error_stats = [
        ("player_errors_errorstotal", "total_errors"),
        ("player_unforced_errors_unforcederrorstotal", "total_ue"),
    ]
    return _add_surface_rolling_stats(df, error_stats)
