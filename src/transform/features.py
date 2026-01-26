"""
Feature engineering using Polars with strict temporal ordering.
All rolling features use ONLY past data to prevent leakage.

CRITICAL: All features are computed using shift(1) to ensure we never
use data from the current match when predicting its outcome.
"""
import polars as pl
from typing import List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureEngineer:
    """
    Feature engineering pipeline using Polars.
    All features respect temporal ordering to prevent data leakage.
    
    Features are categorized as:
    - PRE-MATCH: Known before match starts (odds, round, H2H history)
    - HISTORICAL: Shifted rolling averages of past match statistics
    
    POST-MATCH statistics from the current match are NEVER used as features.
    """
    
    rolling_windows: tuple = (5, 10, 20)
    min_matches: int = 3
    elo_k: float = 32.0  # Kept for backward compatibility
    elo_initial: float = 1500.0  # Kept for backward compatibility
    
    def add_all_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Add all features to the dataset.
        
        Args:
            df: LazyFrame sorted by start_timestamp
            
        Returns:
            LazyFrame with all features added
        """
        # Ensure sorted by time
        df = df.sort("start_timestamp")
        
        # Pre-match features (known before match)
        df = self.add_round_features(df)
        df = self.add_odds_features(df)
        df = self.add_ranking_features(df)
        
        # Historical features (shifted rolling stats)
        df = self.add_rolling_win_rate(df)
        df = self.add_days_since_last_match(df)
        df = self.add_h2h_features(df)
        df = self.add_surface_features(df)
        
        # Shifted rolling stats from past matches
        df = self.add_rolling_service_stats(df)
        df = self.add_rolling_return_stats(df)
        df = self.add_rolling_games_stats(df)
        df = self.add_rolling_points_stats(df)
        df = self.add_rolling_winners_stats(df)
        df = self.add_rolling_errors_stats(df)
        
        return df
    
    # =========================================================================
    # PRE-MATCH FEATURES (known before match starts)
    # =========================================================================
    
    def add_round_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Encode tournament round as numeric."""
        round_map = {
            "Final": 7,
            "Semifinal": 6,
            "Quarterfinal": 5,
            "Round of 16": 4,
            "Round of 32": 3,
            "Round of 64": 2,
            "Round of 128": 1,
            "Qualification": 0,
        }
        
        df = df.with_columns([
            pl.col("round_name")
            .replace(round_map, default=2)
            .alias("round_num")
        ])
        
        return df
    
    def add_odds_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Add odds-derived features (pre-match known)."""
        schema = df.collect_schema().names()
        
        if "odds_player" in schema and "odds_opponent" in schema:
            df = df.with_columns([
                # Implied probabilities
                (1 / pl.col("odds_player")).alias("implied_prob_player"),
                (1 / pl.col("odds_opponent")).alias("implied_prob_opponent"),
                
                # Odds ratio
                (pl.col("odds_opponent") / pl.col("odds_player")).alias("odds_ratio"),
                
                # Is underdog (odds > 2.0)
                (pl.col("odds_player") > 2.0).alias("is_underdog"),
            ])
        
        return df
    
    def add_ranking_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Add ranking metrics."""
        schema = df.collect_schema().names()
        
        # Check for standard names (player_rank, opponent_rank) or numbered (player_1_rank)
        # The schema uses player_rank usually after processing.
        
        if "player_rank" in schema and "opponent_rank" in schema:
            df = df.with_columns([
                (pl.col("player_rank") - pl.col("opponent_rank")).alias("ranking_diff"),
                (pl.col("player_rank").log() - pl.col("opponent_rank").log()).alias("ranking_diff_log"),
            ])
        
        return df
    
    # =========================================================================
    # HISTORICAL FEATURES (shifted rolling averages)
    # =========================================================================
    
    def add_rolling_win_rate(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Add rolling win rate features per player.
        Uses shift(1) to exclude current match.
        """
        for window in self.rolling_windows:
            df = df.with_columns([
                pl.col("player_won")
                .cast(pl.Float64)
                .shift(1)  # CRITICAL: Exclude current match
                .rolling_mean(window_size=window, min_periods=self.min_matches)
                .over("player_id")
                .alias(f"player_win_rate_{window}")
            ])
        
        return df
    
    def add_days_since_last_match(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Add days since player's last match (fatigue/rust indicator)."""
        schema = df.collect_schema().names()
        
        if "match_date" in schema:
            return df.with_columns([
                (
                    pl.col("match_date") - 
                    pl.col("match_date").shift(1).over("player_id")
                ).dt.total_days().alias("days_since_last_match")
            ])
        return df
    
    def add_h2h_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Add head-to-head record between players.
        Calculated using only past encounters (shifted).
        """
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
    
    def add_surface_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Add surface-specific win rate (shifted)."""
        schema = df.collect_schema().names()
        
        if "ground_type" not in schema:
            return df
            
        # Normalize surface names
        df = df.with_columns([
            pl.col("ground_type")
            .str.to_lowercase()
            .str.replace_all(r".*clay.*", "clay")
            .str.replace_all(r".*grass.*", "grass")
            .str.replace_all(r".*hard.*", "hard")
            .alias("surface_normalized")
        ])
        
        # Surface-specific rolling win rate (SHIFTED)
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
    
    # =========================================================================
    # SHIFTED ROLLING SERVICE STATISTICS
    # =========================================================================
    
    def add_rolling_service_stats(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Add shifted rolling service statistics.
        All stats are from PAST matches only.
        """
        schema = df.collect_schema().names()
        
        # Service stats to convert to rolling features
        service_stats = [
            ("player_service_aces", "aces"),
            ("player_service_doublefaults", "double_faults"),
            ("player_service_firstserveaccuracy", "first_serve_pct"),
            ("player_service_firstservepointsaccuracy", "first_serve_won_pct"),
            ("player_service_secondservepointsaccuracy", "second_serve_won_pct"),
            ("player_service_breakpointssaved", "bp_saved"),
        ]
        
        for window in [10, 20]:
            for raw_col, short_name in service_stats:
                if raw_col in schema:
                    df = df.with_columns([
                        pl.col(raw_col)
                        .cast(pl.Float64)
                        .shift(1)  # CRITICAL: Exclude current match
                        .rolling_mean(window_size=window, min_periods=self.min_matches)
                        .over("player_id")
                        .alias(f"player_{short_name}_avg_{window}")
                    ])
        
        # Also add opponent's historical service stats
        opponent_service_stats = [
            ("opponent_service_aces", "opp_aces"),
            ("opponent_service_firstserveaccuracy", "opp_first_serve_pct"),
        ]
        
        for window in [10]:
            for raw_col, short_name in opponent_service_stats:
                if raw_col in schema:
                    df = df.with_columns([
                        pl.col(raw_col)
                        .cast(pl.Float64)
                        .shift(1)
                        .rolling_mean(window_size=window, min_periods=self.min_matches)
                        .over("opponent_id")
                        .alias(f"{short_name}_avg_{window}")
                    ])
        
        return df
    
    # =========================================================================
    # SHIFTED ROLLING RETURN STATISTICS
    # =========================================================================
    
    def add_rolling_return_stats(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Add shifted rolling return statistics."""
        schema = df.collect_schema().names()
        
        return_stats = [
            ("player_return_firstreturnpoints", "first_return_won"),
            ("player_return_secondreturnpoints", "second_return_won"),
            ("player_return_breakpointsscored", "bp_converted"),
        ]
        
        for window in [10, 20]:
            for raw_col, short_name in return_stats:
                if raw_col in schema:
                    df = df.with_columns([
                        pl.col(raw_col)
                        .cast(pl.Float64)
                        .shift(1)  # Exclude current match
                        .rolling_mean(window_size=window, min_periods=self.min_matches)
                        .over("player_id")
                        .alias(f"player_{short_name}_avg_{window}")
                    ])
        
        return df
    
    # =========================================================================
    # SHIFTED ROLLING GAMES STATISTICS
    # =========================================================================
    
    def add_rolling_games_stats(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Add shifted rolling games statistics.
        Note: Games won directly correlates with match outcome,
        so we use SERVICE HOLD % instead (more predictive).
        """
        schema = df.collect_schema().names()
        
        # Calculate service hold percentage (service games won / total service games)
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
                    .rolling_mean(window_size=window, min_periods=self.min_matches)
                    .over("player_id")
                    .alias(f"player_service_hold_pct_avg_{window}")
                ])
        
        # Break percentage (break points won / faced)
        if "player_return_breakpointsscored" in schema:
            for window in [10]:
                df = df.with_columns([
                    pl.col("player_games_maxgamesinrow")
                    .cast(pl.Float64)
                    .shift(1)
                    .rolling_mean(window_size=window, min_periods=self.min_matches)
                    .over("player_id")
                    .alias(f"player_max_games_streak_avg_{window}")
                ])
        
        return df
    
    # =========================================================================
    # SHIFTED ROLLING POINTS STATISTICS
    # =========================================================================
    
    def add_rolling_points_stats(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Add shifted rolling points statistics."""
        schema = df.collect_schema().names()
        
        # Calculate point win percentage
        if "player_points_pointstotal" in schema:
            # We need both player and opponent points to calculate %
            # For now, use max points in row as momentum indicator
            if "player_points_maxpointsinrow" in schema:
                for window in [10]:
                    df = df.with_columns([
                        pl.col("player_points_maxpointsinrow")
                        .cast(pl.Float64)
                        .shift(1)
                        .rolling_mean(window_size=window, min_periods=self.min_matches)
                        .over("player_id")
                        .alias(f"player_max_points_streak_avg_{window}")
                    ])
        
        return df
    
    # =========================================================================
    # SHIFTED ROLLING WINNERS STATISTICS
    # =========================================================================
    
    def add_rolling_winners_stats(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Add shifted rolling winners statistics."""
        schema = df.collect_schema().names()
        
        winner_stats = [
            ("player_winners_winnerstotal", "total_winners"),
            ("player_winners_forehandwinners", "fh_winners"),
            ("player_winners_backhandwinners", "bh_winners"),
        ]
        
        for window in [10]:
            for raw_col, short_name in winner_stats:
                if raw_col in schema:
                    df = df.with_columns([
                        pl.col(raw_col)
                        .cast(pl.Float64)
                        .shift(1)  # Exclude current match
                        .rolling_mean(window_size=window, min_periods=self.min_matches)
                        .over("player_id")
                        .alias(f"player_{short_name}_avg_{window}")
                    ])
        
        return df
    
    # =========================================================================
    # SHIFTED ROLLING ERRORS STATISTICS
    # =========================================================================
    
    def add_rolling_errors_stats(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Add shifted rolling errors statistics."""
        schema = df.collect_schema().names()
        
        error_stats = [
            ("player_errors_errorstotal", "total_errors"),
            ("player_unforced_errors_unforcederrorstotal", "total_ue"),
        ]
        
        for window in [10]:
            for raw_col, short_name in error_stats:
                if raw_col in schema:
                    df = df.with_columns([
                        pl.col(raw_col)
                        .cast(pl.Float64)
                        .shift(1)  # Exclude current match
                        .rolling_mean(window_size=window, min_periods=self.min_matches)
                        .over("player_id")
                        .alias(f"player_{short_name}_avg_{window}")
                    ])
        
        # Winner-to-error ratio (aggression vs consistency)
        if "player_winners_winnerstotal" in schema and "player_errors_errorstotal" in schema:
            df = df.with_columns([
                (pl.col("player_winners_winnerstotal") / 
                 pl.col("player_errors_errorstotal").clip(1))
                .alias("_player_winner_error_ratio")
            ])
            
            for window in [10]:
                df = df.with_columns([
                    pl.col("_player_winner_error_ratio")
                    .shift(1)
                    .rolling_mean(window_size=window, min_periods=self.min_matches)
                    .over("player_id")
                    .alias(f"player_winner_error_ratio_avg_{window}")
                ])
        
        return df
    
    # =========================================================================
    # FEATURE SELECTION (SAFE FEATURES ONLY)
    # =========================================================================
    
    def get_feature_columns(self, df: pl.LazyFrame) -> List[str]:
        """
        Return list of SAFE feature columns for ML training.
        
        Only returns:
        - Pre-match known data (odds, round)
        - Properly shifted historical rolling averages
        
        Excludes:
        - Identifiers and metadata
        - Target variable (player_won)
        - RAW post-match statistics (causes leakage!)
        """
        schema = df.collect_schema().names()
        
        # SAFE feature patterns (these are OK to use)
        safe_patterns = [
            # Pre-match known
            "odds_player", "odds_opponent", "odds_ratio",
            "implied_prob_player", "implied_prob_opponent",
            "is_underdog", "round_num",
            
            # Historical shifted features (all end with _avg_N or _rate_N)
            "win_rate_",           # Rolling win rates
            "h2h_",                # Head-to-head (shifted)
            "surface_win_rate",    # Surface-specific (shifted)
            "days_since",          # Pre-match known
            
            # New shifted rolling stats
            "_avg_",               # All rolling averages
            "_pct_avg_",           # Percentage averages
            "_ratio_avg_",         # Ratio averages
        ]
        
        # Columns to ALWAYS exclude (even if they match patterns)
        always_exclude = {
            # Identifiers
            "event_id", "player_id", "opponent_id", "player_name", "opponent_name",
            "tournament_id", "tournament_name", "matchup_key",
            
            # Target
            "player_won",
            
            # Timestamps and dates
            "start_timestamp", "match_date", "match_year", "match_month", "match_day",
            
            # Metadata
            "status", "ground_type", "surface_normalized", "round_name",
            "_schema_version", "_scraped_at", "has_stats", "has_odds", "is_home",
            
            # RAW post-match score data (LEAKAGE!)
            "player_sets", "opponent_sets",
            "player_set1", "player_set2", "player_set3", "player_set4", "player_set5",
            "opponent_set1", "opponent_set2", "opponent_set3", "opponent_set4", "opponent_set5",
        }
        
        # RAW post-match stat patterns to EXCLUDE
        leaky_patterns = [
            "player_service_",      # Raw current-match service stats
            "opponent_service_",
            "player_return_",       # Raw current-match return stats
            "opponent_return_",
            "player_games_",        # Raw current-match games
            "opponent_games_",
            "player_points_",       # Raw current-match points
            "opponent_points_",
            "player_winners_",      # Raw current-match winners
            "opponent_winners_",
            "player_errors_",       # Raw current-match errors
            "opponent_errors_",
            "player_unforced_",     # Raw current-match UE
            "opponent_unforced_",
            "player_miscellaneous_", # Raw current-match misc
            "opponent_miscellaneous_",
        ]
        
        feature_cols = []
        
        for col in schema:
            # Skip always-excluded columns
            if col in always_exclude:
                continue
            
            # Skip columns starting with _ (internal/temp columns)
            if col.startswith("_"):
                continue
            
            # Check if it matches a SAFE pattern
            is_safe = any(pattern in col for pattern in safe_patterns)
            
            # Check if it matches a LEAKY pattern
            is_leaky = any(col.startswith(pattern) for pattern in leaky_patterns)
            
            # Exclude join helper columns/artifacts
            if col.endswith("_right") or col.endswith("_left"):
                continue
            
            # Include if safe and NOT a raw leaky stat
            # Note: shifted features like "player_aces_avg_10" contain "_avg_" so they're safe
            if is_safe and not is_leaky:
                feature_cols.append(col)
            elif not is_leaky and "_avg_" in col:
                # Catch any shifted averages we might have missed
                feature_cols.append(col)
        
        logger.info(f"Selected {len(feature_cols)} safe features for training")
        return feature_cols
    
    # =========================================================================
    # PREDICTION-SPECIFIC FEATURE COMPUTATION
    # =========================================================================
    
    def compute_features_for_prediction(
        self,
        upcoming_df: pl.DataFrame,
        historical_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Compute features for upcoming matches using historical data.
        
        This method is used for live predictions:
        1. Takes historical data (with player_won results)
        2. Computes rolling features from that history
        3. Joins those features to upcoming matches
        
        Args:
            upcoming_df: DataFrame with upcoming matches (no player_won)
            historical_df: DataFrame with completed matches (has player_won)
            
        Returns:
            DataFrame with prediction-ready features
        """
        logger.info(f"Computing features for {len(upcoming_df)} upcoming matches")
        
        # Ensure historical data is sorted
        historical_df = historical_df.sort("start_timestamp")
        
        # Compute features on historical data
        historical_lazy = historical_df.lazy()
        historical_with_features = self.add_all_features(historical_lazy).collect()
        
        # Get the latest feature values for each player
        player_features = self._get_latest_player_features(historical_with_features)
        
        # Add pre-match features to upcoming matches
        result = upcoming_df.clone()
        
        # Add odds features (pre-match known)
        result = self._add_odds_to_upcoming(result)
        
        # Add round features
        result = self._add_round_to_upcoming(result)
        
        # Join player historical features
        result = self._join_player_features(result, player_features)
        
        # Calculate days_since if timestamps available
        if "start_timestamp" in result.columns and "last_match_timestamp" in result.columns:
            result = result.with_columns(
                ((pl.col("start_timestamp") - pl.col("last_match_timestamp")) / 86400).alias("days_since")
            ).drop("last_match_timestamp")
        elif "days_since" not in result.columns:
             # Fallback if no history or timestamp
             result = result.with_columns(pl.lit(0).alias("days_since"))
        
        logger.info(f"Computed {len(result.columns)} columns for prediction")
        return result
    
    def _get_latest_player_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Get the latest feature values for each player (for joining to predictions)."""
        feature_cols = self.get_feature_columns(df.lazy())
        
        # Select only feature columns plus player_id
        select_cols = ["player_id"] + [c for c in feature_cols if c in df.columns]
        
        # Get latest row per player (most recent match)
        latest = (
            df.select(select_cols + ["start_timestamp"])
            .sort("start_timestamp")
            .group_by("player_id")
            .last()
        )
        
        return latest
    
    def _add_odds_to_upcoming(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add odds-derived features to upcoming matches."""
        if "odds_player" in df.columns and "odds_opponent" in df.columns:
            df = df.with_columns([
                (1 / pl.col("odds_player")).alias("implied_prob_player"),
                (1 / pl.col("odds_opponent")).alias("implied_prob_opponent"),
                (pl.col("odds_opponent") / pl.col("odds_player")).alias("odds_ratio"),
                (pl.col("odds_player") > 2.0).alias("is_underdog"),
            ])
        return df
    
    def _add_round_to_upcoming(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add round features to upcoming matches."""
        round_map = {
            "Final": 7, "Semifinal": 6, "Quarterfinal": 5,
            "Round of 16": 4, "Round of 32": 3, "Round of 64": 2,
            "Round of 128": 1, "Qualification": 0,
        }
        
        if "round_name" in df.columns:
            df = df.with_columns([
                pl.col("round_name").replace(round_map, default=2).alias("round_num")
            ])
        else:
            df = df.with_columns([pl.lit(3).alias("round_num")])
        
        return df
    
    def _join_player_features(
        self,
        upcoming: pl.DataFrame,
        player_features: pl.DataFrame
    ) -> pl.DataFrame:
        """Join historical player features to upcoming matches."""
        # Rename feature columns to avoid conflicts
        feature_cols = [c for c in player_features.columns if c != "player_id" and c != "start_timestamp"]
        
        # Select columns including timestamp alias
        cols_to_select = ["player_id"] + feature_cols
        if "start_timestamp" in player_features.columns:
            cols_to_select.append(pl.col("start_timestamp").alias("last_match_timestamp"))
        
        # Join for player (home perspective)
        player_joined = upcoming.join(
            player_features.select(cols_to_select),
            on="player_id",
            how="left"
        )
        
        return player_joined

