"""
Ranking features - Player ranking metrics.
"""
import polars as pl
from .registry import FeatureRegistry


@FeatureRegistry.register("ranking_features", category="pre_match", priority=12)
def add_ranking_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add ranking-based features."""
    schema = df.collect_schema().names()
    
    if "player_rank" not in schema or "opponent_rank" not in schema:
        return df
    
    return df.with_columns([
        (pl.col("player_rank") - pl.col("opponent_rank")).alias("ranking_diff"),
        (pl.col("player_rank").log() - pl.col("opponent_rank").log()).alias("ranking_diff_log"),
    ])
