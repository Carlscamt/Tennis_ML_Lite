"""
Unit tests for monitoring module.
"""
import pytest
from datetime import date, datetime
from pathlib import Path

from src.monitoring.prediction_store import PredictionStore, PredictionRecord
from src.monitoring.metrics_job import (
    compute_model_roi,
    compute_calibration,
    compute_feature_drift,
)


class TestPredictionStore:
    """Tests for PredictionStore."""
    
    @pytest.fixture
    def store(self, tmp_path):
        return PredictionStore(db_path=tmp_path / "test_predictions.db")
    
    def test_save_prediction(self, store):
        """Test saving a prediction."""
        result = store.save_prediction(
            event_id=123,
            model_version="v1.0",
            predicted_prob=0.65,
            odds=1.8,
            stake=10.0,
        )
        assert result is True
    
    def test_record_outcome_win(self, store):
        """Test recording a winning outcome."""
        store.save_prediction(
            event_id=123,
            model_version="v1.0",
            predicted_prob=0.65,
            odds=1.8,
            stake=10.0,
        )
        
        updated = store.record_outcome(event_id=123, actual_outcome=True)
        
        assert updated == 1
    
    def test_compute_roi_winning(self, store):
        """Test ROI calculation for winning bets."""
        # Save 2 predictions, both win
        for i in range(2):
            store.save_prediction(
                event_id=100 + i,
                model_version="v1.0",
                predicted_prob=0.6,
                odds=2.0,
                stake=10.0,
            )
            store.record_outcome(event_id=100 + i, actual_outcome=True)
        
        roi = store.compute_roi(model_version="v1.0")
        
        assert roi["total_bets"] == 2
        assert roi["total_stake"] == 20.0
        assert roi["total_pnl"] == 20.0  # Each wins $10
        assert roi["roi"] == 100.0  # 100% ROI
    
    def test_compute_roi_losing(self, store):
        """Test ROI calculation for losing bets."""
        store.save_prediction(
            event_id=123,
            model_version="v1.0",
            predicted_prob=0.6,
            odds=2.0,
            stake=10.0,
        )
        store.record_outcome(event_id=123, actual_outcome=False)
        
        roi = store.compute_roi(model_version="v1.0")
        
        assert roi["total_pnl"] == -10.0
        assert roi["roi"] == -100.0
    
    def test_get_pending_outcomes(self, store):
        """Test getting unresolved predictions."""
        store.save_prediction(event_id=1, model_version="v1.0", predicted_prob=0.6, odds=1.8)
        store.save_prediction(event_id=2, model_version="v1.0", predicted_prob=0.7, odds=1.5)
        store.record_outcome(event_id=1, actual_outcome=True)
        
        pending = store.get_pending_outcomes()
        
        assert 2 in pending
        assert 1 not in pending
    
    def test_odds_band_categorization(self, store):
        """Test odds are categorized into bands."""
        store.save_prediction(event_id=1, model_version="v1.0", predicted_prob=0.9, odds=1.2)
        store.save_prediction(event_id=2, model_version="v1.0", predicted_prob=0.5, odds=2.2)
        store.save_prediction(event_id=3, model_version="v1.0", predicted_prob=0.3, odds=4.0)
        
        # Verify internally
        assert store._get_odds_band(1.2) == "heavy_favorite"
        assert store._get_odds_band(2.2) == "even"
        assert store._get_odds_band(4.0) == "long_shot"


class TestFeatureDrift:
    """Tests for feature drift computation."""
    
    def test_compute_feature_drift(self):
        """Test drift statistics computation."""
        import polars as pl
        
        df = pl.DataFrame({
            "odds_player": [1.5, 2.0, 1.8, None, 2.2],
            "player_win_rate_20": [0.6, 0.7, 0.65, 0.55, 0.8],
        })
        
        result = compute_feature_drift(df)
        
        assert "odds_player" in result
        assert "player_win_rate_20" in result
        assert result["odds_player"]["null_rate"] == 0.2
        assert 0.6 < result["player_win_rate_20"]["mean"] < 0.7


class TestCalibration:
    """Tests for calibration computation."""
    
    @pytest.fixture
    def store_with_data(self, tmp_path):
        store = PredictionStore(db_path=tmp_path / "test_cal.db")
        
        # Add predictions with varying confidence
        for i in range(20):
            prob = 0.5 + (i % 10) * 0.05  # 0.5 to 0.95
            store.save_prediction(
                event_id=i,
                model_version="v1.0",
                predicted_prob=prob,
                odds=1.8,
                stake=1.0,
            )
            # Outcome correlates with probability
            store.record_outcome(event_id=i, actual_outcome=(prob > 0.7))
        
        return store
    
    def test_calibration_returns_bins(self, store_with_data, monkeypatch):
        """Test calibration computation returns bin details."""
        monkeypatch.setattr(
            "src.monitoring.metrics_job.get_prediction_store",
            lambda: store_with_data
        )
        
        result = compute_calibration(model_version="v1.0")
        
        assert "calibration_error" in result
        assert "bins" in result
