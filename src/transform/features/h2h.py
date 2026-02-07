"""
Head-to-head features - Historical matchup records.
"""
import polars as pl
from .registry import FeatureRegistry


@FeatureRegistry.register("h2h_features", category="historical", priority=40)
def add_h2h_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add head-to-head record between players.
    Uses shift(1) to exclude current match (temporal safety).
    """
    schema = df.collect_schema().names()
    
    if "player_id" not in schema or "opponent_id" not in schema:
        return df
    
    # Create matchup key (sorted player IDs for consistency)
    df = df.with_columns([
        pl.when(pl.col("player_id") < pl.col("opponent_id"))
        .then(pl.concat_str([pl.col("player_id"), pl.lit("_"), pl.col("opponent_id")]))
        .otherwise(pl.concat_str([pl.col("opponent_id"), pl.lit("_"), pl.col("player_id")]))
        .alias("matchup_key")
    ])
    
    # H2H wins for this player in this matchup (SHIFTED)
    df = df.with_columns([
        pl.col("player_won")
        .cast(pl.Float64)
        .shift(1)  # Exclude current match
        .fill_null(0)
        .cum_sum()
        .over(["player_id", "matchup_key"])
        .alias("h2h_wins"),
        
        pl.col("player_id")
        .is_not_null()
        .cast(pl.Int64)
        .shift(1)  # Exclude current match
        .fill_null(0)
        .cum_sum()
        .over(["player_id", "matchup_key"])
        .alias("h2h_matches")
    ])
    
    df = df.with_columns([
        (pl.col("h2h_wins") / pl.col("h2h_matches").clip(1))
        .alias("h2h_win_rate")
    ])
    
    return df
