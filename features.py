"""
Features - Feature engineering for ML
"""
import polars as pl

def add_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add all features to dataframe."""
    df = df.sort(["player_id", "date"])
    
    # Win rate (rolling 10 matches)
    df = df.with_columns([
        pl.col("player_won")
        .shift(1)
        .rolling_mean(window_size=10, min_periods=3)
        .over("player_id")
        .alias("win_rate_10")
    ])
    
    # Odds-based features
    df = df.with_columns([
        (1 / pl.col("odds_player")).alias("implied_prob"),
        (pl.col("odds_opponent") / pl.col("odds_player")).alias("odds_ratio"),
    ])
    
    return df

def compute_prediction_features(upcoming: pl.DataFrame, historical: pl.DataFrame) -> pl.DataFrame:
    """Compute features for upcoming matches using historical data."""
    
    # Calculate historical win rates per player
    player_stats = (
        historical
        .filter(pl.col("player_won").is_not_null())
        .group_by("player_id")
        .agg([
            pl.col("player_won").mean().alias("historical_win_rate"),
            pl.count().alias("matches_played"),
        ])
    )
    
    # Join with upcoming
    upcoming = upcoming.join(
        player_stats, on="player_id", how="left"
    ).with_columns([
        pl.col("historical_win_rate").fill_null(0.5),
        pl.col("matches_played").fill_null(0),
    ])
    
    # Add odds features
    upcoming = upcoming.with_columns([
        (1 / pl.col("odds_player")).alias("implied_prob"),
        (pl.col("odds_opponent") / pl.col("odds_player")).alias("odds_ratio"),
    ])
    
    return upcoming

def get_feature_columns() -> list:
    """Get list of feature columns for model."""
    return [
        "historical_win_rate",
        "matches_played", 
        "implied_prob",
        "odds_ratio",
        "odds_player",
        "odds_opponent",
    ]
