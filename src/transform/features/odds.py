"""
Odds features - Pre-match odds-derived features.
"""
import polars as pl
from .registry import FeatureRegistry


@FeatureRegistry.register("odds_features", category="pre_match", priority=11)
def add_odds_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add odds-derived features (pre-match known)."""
    schema = df.collect_schema().names()
    
    if "odds_player" not in schema or "odds_opponent" not in schema:
        return df
    
    return df.with_columns([
        # Implied probabilities
        (1 / pl.col("odds_player")).alias("implied_prob_player"),
        (1 / pl.col("odds_opponent")).alias("implied_prob_opponent"),
        
        # Odds ratio
        (pl.col("odds_opponent") / pl.col("odds_player")).alias("odds_ratio"),
        
        # Is underdog (odds > 2.0)
        (pl.col("odds_player") > 2.0).alias("is_underdog"),
    ])
