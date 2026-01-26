import pytest
import polars as pl
from datetime import datetime, timedelta
from src.transform.features import FeatureEngineer

class TestFeatureEngineerEdgeCases:
    
    
    def test_h2h_win_rate_no_history(self):
        """H2H calculation handles first-time opponents."""
        fe = FeatureEngineer(min_matches=0) # Allow calculation immediately
        
        df = pl.DataFrame({
            "player_id": [1],
            "opponent_id": [2],
            "player_won": [True],
            "start_timestamp": [1000]
        }).lazy()
        
        result = fe.add_h2h_features(df).collect()
        
        assert result["h2h_win_rate"].item(0) == 0.0
        assert result["h2h_matches"].item(0) == 0.0

    def test_h2h_win_rate_with_one_match(self):
        """H2H calculation works for the second match."""
        fe = FeatureEngineer(min_matches=0)
        
        df = pl.DataFrame({
            "player_id": [1, 1],
            "opponent_id": [2, 2],
            "player_won": [True, False],
            "start_timestamp": [1000, 2000]
        }).lazy().sort("start_timestamp")
        
        result = fe.add_h2h_features(df).collect()
        
        # Debugging assertions to isolate failure
        # Check intermediate values
        assert result["h2h_wins"].item(1) == 1.0, f"Wins was {result['h2h_wins'].item(1)}"
        assert result["h2h_matches"].item(1) == 1.0, f"Matches was {result['h2h_matches'].item(1)}"
        
        # Check final values
        assert result["h2h_win_rate"].item(0) == 0.0
        assert result["h2h_win_rate"].item(1) == 1.0
        assert result["h2h_matches"].item(1) == 1

    def test_surface_win_rate_rare_surface(self):
        """Surface win rate handles rare surfaces."""
        fe = FeatureEngineer(min_matches=1, rolling_windows=(5,))
        
        df = pl.DataFrame({
            "player_id": [1, 1],
            "opponent_id": [2, 3],
            "player_won": [True, False],
            "ground_type": ["Carpet", "Carpet"], 
            "start_timestamp": [1000, 2000]
        }).lazy()
        
        result = fe.add_surface_features(df).collect()
        
        assert result["player_surface_win_rate_10"].item(1) is None

    def test_surface_win_rate_normalization(self):
        """Surface variations (Clay (Red), Clay) are normalized."""
        fe = FeatureEngineer()
        
        df = pl.DataFrame({
            "player_id": [1],
            "ground_type": ["Red Clay"],
            "player_won": [True],
            "start_timestamp": [1000]
        }).lazy()
        
        result = fe.add_surface_features(df).collect()
        assert result["surface_normalized"].item(0) == "clay"
