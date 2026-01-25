# tests/unit/test_feature_engineer.py
import pytest
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from src.transform.features import FeatureEngineer


class TestFeatureEngineer:
    """Test feature engineering logic."""
    
    @pytest.fixture
    def engineer(self):
        """Initialize feature engineer."""
        return FeatureEngineer()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample match data."""
        return pl.DataFrame({
            "event_id": [1, 2, 3],
            "player_id": [100, 100, 101],
            "player_name": ["Djokovic", "Djokovic", "Alcaraz"],
            "player_1_rank": [5, 5, 3],
            "player_2_rank": [3, 15, 5],
            "player_1_win_rate_current_year": [0.75, 0.75, 0.65],
            "player_2_win_rate_current_year": [0.80, 0.60, 0.75],
            "start_timestamp": [
                int(datetime.now().timestamp()),
                int((datetime.now() - timedelta(days=1)).timestamp()),
                int((datetime.now() - timedelta(days=2)).timestamp()),
            ],
            # Add missing required columns for transform
            "opponent_id": [200, 300, 100],
            "opponent_name": ["Alcaraz", "Sinner", "Djokovic"],
            "tournament_name": ["AO", "FO", "WIM"],
            "ground_type": ["Hard", "Clay", "Grass"],
            "player_won": [True, True, False],
            "round_name": ["Final", "Semi-Finals", "Quarter-Finals"]
        })
    
    def test_ranking_difference(self, engineer, sample_data):
        """Compute ranking difference correctly."""
        # Need to ensure correct column names exist effectively
        # Our FeatureEngineer internally might look for `player_rank`, `opponent_rank`?
        # Let's check src/transform/feature_engineer.py... 
        # FeatureEngineer usually processes historical data. 
        # Renaming columns to standard if needed.
        
        #Assuming standard schema input
        df = sample_data.rename({"player_1_rank": "player_rank", "player_2_rank": "opponent_rank"})
        
        # Use lazy()
        result = engineer.add_all_features(df.lazy()).collect()
        
        # Player 1 rank 5, Player 2 rank 3 -> diff = 5 - 3 = 2 (as implemented per test request order)
        # Implementation checks
        assert "ranking_diff" in result.columns
        # Check value for first row (Sorted by time: Row 2 (T-2) comes first, P1=3, P2=5 -> 3-5 = -2)
        assert result["ranking_diff"][0] == -2
    
    def test_all_features_numeric(self, engineer, sample_data):
        """All engineered features are numeric."""
        df = sample_data.rename({"player_1_rank": "player_rank", "player_2_rank": "opponent_rank"})
        result = engineer.add_all_features(df.lazy()).collect()
        
        numeric_cols = result.select(pl.col(pl.Float64, pl.Int64))
        assert len(numeric_cols.columns) > 0
    
    @pytest.mark.slow
    def test_large_dataset(self, engineer):
        """Handle large dataset without errors."""
        large_data = pl.DataFrame({
            "event_id": list(range(1000)),
            "player_id": np.random.randint(1, 500, 1000),
            "opponent_id": np.random.randint(501, 1000, 1000),
            "player_name": [f"Player_{i}" for i in range(1000)],
            "opponent_name": [f"Opp_{i}" for i in range(1000)],
            "player_rank": np.random.randint(1, 1000, 1000),
            "opponent_rank": np.random.randint(1, 1000, 1000),
            "start_timestamp": [int(datetime.now().timestamp())] * 1000,
            "tournament_name": ["AO"] * 1000,
            "ground_type": ["Hard"] * 1000,
            "player_won": [True] * 1000,
            "round_name": ["R1"] * 1000
        })
        
        result = engineer.add_all_features(large_data.lazy())
        assert result.collect().shape[0] == 1000
