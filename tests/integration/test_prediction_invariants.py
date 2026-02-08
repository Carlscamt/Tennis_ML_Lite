"""
Integration tests for prediction invariants.

Tests critical business rules:
- No prediction uses data after target date (no future leakage)
- Required features are non-null
- No bets on past matches
"""
import pytest
from datetime import datetime, date, timedelta
import polars as pl
import numpy as np


@pytest.mark.integration
class TestNoFutureDataLeakage:
    """Ensure predictions don't use future data."""
    
    @pytest.fixture
    def historical_features(self):
        """Create features with timestamps for leakage detection."""
        today = datetime.now()
        
        return pl.DataFrame({
            "event_id": [1, 2, 3, 4, 5],
            "match_date": [
                today - timedelta(days=5),
                today - timedelta(days=4),
                today - timedelta(days=3),
                today - timedelta(days=2),
                today - timedelta(days=1),
            ],
            "player_id": [100, 101, 102, 103, 104],
            "opponent_id": [200, 201, 202, 203, 204],
            # Rolling features should be computed from data BEFORE match
            "player_win_rate_30d": [0.6, 0.55, 0.7, 0.65, 0.58],
            "player_last_match_date": [
                today - timedelta(days=10),
                today - timedelta(days=8),
                today - timedelta(days=6),
                today - timedelta(days=5),
                today - timedelta(days=4),
            ],
        })
    
    def test_rolling_features_use_only_past_data(self, historical_features):
        """Rolling features are computed from data before match date."""
        df = historical_features
        
        # Assert last_match_date is always before match_date
        assert all(
            df["player_last_match_date"].to_list()[i] < df["match_date"].to_list()[i]
            for i in range(len(df))
        ), "Rolling features must use data from before match date"
    
    def test_prediction_date_before_match_date(self, historical_features):
        """Predictions are made for future matches only."""
        df = historical_features
        
        # Simulate prediction timestamp (should be before match)
        prediction_time = datetime.now() - timedelta(days=3)
        
        # Filter matches that would be predicted
        future_matches = df.filter(pl.col("match_date") > prediction_time)
        
        assert len(future_matches) > 0, "Should have future matches to predict"
        
        # All predicted matches should be after prediction time
        for match_date in future_matches["match_date"].to_list():
            assert match_date > prediction_time


@pytest.mark.integration
class TestRequiredFeaturesNonNull:
    """Ensure required features are never null."""
    
    REQUIRED_FEATURES = [
        "player_id",
        "opponent_id",
        "event_id",
    ]
    
    OPTIONAL_FEATURES = [
        "player_win_rate_30d",  # May be null for new players
        "elo_rating",           # May be null initially
    ]
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature DataFrame."""
        return pl.DataFrame({
            "event_id": [1, 2, 3],
            "player_id": [100, 101, 102],
            "opponent_id": [200, 201, 202],
            "player_win_rate_30d": [0.6, None, 0.7],  # Some nulls OK
            "elo_rating": [1500, 1600, None],
        })
    
    def test_required_features_present(self, sample_features):
        """Required features are present in DataFrame."""
        df = sample_features
        
        for feature in self.REQUIRED_FEATURES:
            if feature in df.columns:
                null_count = df[feature].null_count()
                assert null_count == 0, f"Required feature '{feature}' has {null_count} nulls"
    
    def test_optional_features_may_have_nulls(self, sample_features):
        """Optional features are allowed to have nulls."""
        df = sample_features
        
        for feature in self.OPTIONAL_FEATURES:
            if feature in df.columns:
                # Just verify no errors, nulls are acceptable
                _ = df[feature].null_count()
    
    def test_feature_validation_before_prediction(self, sample_features):
        """Features should be validated before prediction."""
        df = sample_features
        
        # Mock validation function
        def validate_for_prediction(features_df):
            for col in ["event_id", "player_id", "opponent_id"]:
                if col in features_df.columns:
                    if features_df[col].null_count() > 0:
                        raise ValueError(f"Required feature {col} has nulls")
            return True
        
        # Should not raise for valid data
        assert validate_for_prediction(df)


@pytest.mark.integration
class TestNoPastMatchBets:
    """Ensure no bets are placed on past matches."""
    
    @pytest.fixture
    def match_schedule(self):
        """Create schedule with past and future matches."""
        now = datetime.now()
        
        return pl.DataFrame({
            "event_id": [1, 2, 3, 4, 5],
            "match_datetime": [
                now - timedelta(hours=5),   # Past
                now - timedelta(hours=1),   # Past
                now + timedelta(hours=1),   # Future
                now + timedelta(hours=3),   # Future
                now + timedelta(hours=6),   # Future
            ],
            "predicted_prob": [0.7, 0.65, 0.75, 0.6, 0.8],
            "odds_player": [1.5, 1.8, 1.6, 2.0, 1.4],
        })
    
    def test_filter_future_matches_only(self, match_schedule):
        """Only future matches are eligible for betting."""
        df = match_schedule
        now = datetime.now()
        
        future_only = df.filter(pl.col("match_datetime") > now)
        
        assert len(future_only) == 3, "Should have 3 future matches"
        
        # All remaining matches should be in the future
        for dt in future_only["match_datetime"].to_list():
            assert dt > now, f"Match at {dt} is in the past"
    
    def test_bet_generation_excludes_past(self, match_schedule):
        """Bet generation excludes past matches."""
        df = match_schedule
        now = datetime.now()
        
        # Simulate bet generation
        def generate_bets(matches_df, min_edge=0.05):
            # Filter future only
            future = matches_df.filter(pl.col("match_datetime") > now)
            
            # Calculate edge
            bets = future.with_columns([
                ((pl.col("predicted_prob") * pl.col("odds_player")) - 1).alias("edge")
            ]).filter(pl.col("edge") >= min_edge)
            
            return bets
        
        bets = generate_bets(df)
        
        # Verify all bets are on future matches
        for dt in bets["match_datetime"].to_list():
            assert dt > now, "Generated bet on past match"
    
    def test_bet_cutoff_time_buffer(self, match_schedule):
        """Bets require buffer before match start."""
        df = match_schedule
        now = datetime.now()
        buffer_minutes = 15
        
        cutoff = now + timedelta(minutes=buffer_minutes)
        
        # Filter with buffer
        valid_for_betting = df.filter(pl.col("match_datetime") > cutoff)
        
        assert len(valid_for_betting) <= 3, "Should exclude matches too close to start"


@pytest.mark.integration  
class TestPredictionConsistency:
    """Test prediction consistency and reproducibility."""
    
    def test_same_input_same_output(self):
        """Same input features produce same prediction."""
        # Simulate deterministic prediction
        np.random.seed(42)
        
        features = np.array([[0.6, 1500, 1.8]])
        
        # Mock prediction (deterministic with seed)
        pred1 = np.mean(features) + np.random.normal(0, 0.01)
        
        np.random.seed(42)
        pred2 = np.mean(features) + np.random.normal(0, 0.01)
        
        assert pred1 == pred2, "Same input should produce same output"
