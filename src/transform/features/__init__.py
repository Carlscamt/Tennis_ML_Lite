"""
Features package - Composable, config-driven feature engineering.

Usage:
    from src.transform.features import build_features, FeatureRegistry
    
    df = build_features(df)  # Apply all registered features
    
    # Or selectively by category
    df = FeatureRegistry.build_features(df, categories=["rolling", "surface"])
"""
import polars as pl
from typing import List, Optional

# Import registry first
from .registry import FeatureRegistry, FeatureSpec

# Import all feature modules to register their builders
from . import round
from . import odds
from . import ranking
from . import fatigue
from . import h2h
from . import surface
from . import rolling_stats


def build_features(
    df: pl.LazyFrame,
    config_path: str = "config/features.yaml",
    categories: Optional[List[str]] = None
) -> pl.LazyFrame:
    """
    Apply all registered feature builders to the dataframe.
    
    Args:
        df: Input LazyFrame (should be sorted by start_timestamp)
        config_path: Path to features configuration
        categories: Optional list of categories to include
        
    Returns:
        LazyFrame with all features added
    """
    # Ensure sorted by time
    df = df.sort("start_timestamp")
    
    return FeatureRegistry.build_features(df, config_path, categories)


def get_feature_columns(df: pl.LazyFrame) -> List[str]:
    """
    Return list of safe feature columns for ML training.
    
    Excludes identifiers, targets, and raw post-match statistics.
    """
    schema = df.collect_schema().names()
    
    # Safe feature patterns
    safe_patterns = [
        "odds_player", "odds_opponent", "odds_ratio",
        "implied_prob_", "is_underdog", "round_num",
        "win_rate_", "h2h_", "surface_win_rate",
        "days_since", "_avg_", "_pct_avg_", "_ratio_avg_",
        "_surface_avg_", "ranking_diff",
    ]
    
    # Always exclude
    always_exclude = {
        "event_id", "player_id", "opponent_id", "player_name", "opponent_name",
        "tournament_id", "tournament_name", "matchup_key", "player_won",
        "start_timestamp", "match_date", "status", "ground_type",
        "surface_normalized", "round_name",
    }
    
    # Leaky patterns (raw post-match stats)
    leaky_patterns = [
        "player_service_", "opponent_service_", "player_return_",
        "opponent_return_", "player_games_", "opponent_games_",
        "player_points_", "opponent_points_", "player_winners_",
        "opponent_winners_", "player_errors_", "opponent_errors_",
        "player_unforced_", "opponent_unforced_",
    ]
    
    feature_cols = []
    
    for col in schema:
        if col in always_exclude or col.startswith("_"):
            continue
        
        is_safe = any(p in col for p in safe_patterns)
        is_leaky = any(col.startswith(p) for p in leaky_patterns)
        
        if is_safe and not is_leaky:
            feature_cols.append(col)
        elif not is_leaky and "_avg_" in col:
            feature_cols.append(col)
    
    return feature_cols


__all__ = [
    "FeatureRegistry",
    "FeatureSpec",
    "build_features",
    "get_feature_columns",
]
