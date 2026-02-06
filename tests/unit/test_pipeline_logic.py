import pytest
import polars as pl
from src.pipeline import TennisPipeline

class TestPipelineLogic:
    
    def test_normalize_probabilities_basic(self):
        """Test simple normalization A+B=1.0"""
        pipe = TennisPipeline()
        
        df = pl.DataFrame({
            "event_id": [1, 1],
            "player_name": ["A", "B"],
            "opponent_name": ["B", "A"],
            "model_prob": [0.6, 0.6]  # Sum = 1.2
        })
        
        result = pipe._normalize_probabilities(df)
        
        normalized = result["model_prob"].to_list()
        assert normalized[0] == pytest.approx(0.5)
        assert normalized[1] == pytest.approx(0.5)
        assert sum(normalized) == pytest.approx(1.0)

    def test_normalize_probabilities_asymmetric(self):
        """Test normalization with asymmetric probabilities"""
        pipe = TennisPipeline()
        
        df = pl.DataFrame({
            "event_id": [1, 1],
            "player_name": ["Fed", "Nadal"],
            "opponent_name": ["Nadal", "Fed"],
            "model_prob": [0.8, 0.8] # Sum 1.6
        })
        
        result = pipe._normalize_probabilities(df)
        
        normalized = result["model_prob"].to_list()
        assert normalized[0] == pytest.approx(0.5)
        assert normalized[1] == pytest.approx(0.5)

    def test_normalize_multiple_matches(self):
        """Test normalization across multiple separate matches"""
        pipe = TennisPipeline()
        
        df = pl.DataFrame({
            "event_id": [1, 1, 2, 2],
            "model_prob": [0.7, 0.7, 0.2, 0.2] 
        })
        
        result = pipe._normalize_probabilities(df)
        
        probs = result["model_prob"].to_list()
        # Match 1: 0.7/(0.7+0.7) = 0.5
        assert probs[0] == pytest.approx(0.5)
        # Match 2: 0.2/(0.2+0.2) = 0.5
        assert probs[2] == pytest.approx(0.5)
