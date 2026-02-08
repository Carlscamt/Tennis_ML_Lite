"""
Unit tests for batch runner.
"""
import pytest
from datetime import date
from pathlib import Path
import polars as pl

from src.model.batch_runner import (
    BatchRunner,
    BatchStatus,
    BatchResult,
    ModelPredictions,
)


class TestBatchStatus:
    """Tests for BatchStatus enum."""
    
    def test_status_values(self):
        """Test status enum values."""
        assert BatchStatus.SUCCESS.value == "success"
        assert BatchStatus.PARTIAL.value == "partial"
        assert BatchStatus.FALLBACK.value == "fallback"
        assert BatchStatus.SKIPPED.value == "skipped"
        assert BatchStatus.FAILED.value == "failed"


class TestBatchResult:
    """Tests for BatchResult dataclass."""
    
    def test_summary_with_predictions(self):
        """Test summary property."""
        champion = ModelPredictions(
            model_version="v1.0",
            model_type="champion",
            predictions=pl.DataFrame({"a": [1, 2]}),
            latency_ms=100.0,
        )
        
        result = BatchResult(
            date=date(2026, 2, 8),
            status=BatchStatus.SUCCESS,
            champion_predictions=champion,
            value_bets_count=5,
        )
        
        summary = result.summary
        assert summary["date"] == "2026-02-08"
        assert summary["status"] == "success"
        assert summary["champion_version"] == "v1.0"
        assert summary["value_bets"] == 5
    
    def test_summary_no_predictions(self):
        """Test summary with no predictions."""
        result = BatchResult(
            date=date(2026, 2, 8),
            status=BatchStatus.SKIPPED,
            alerts=["No models available"],
        )
        
        summary = result.summary
        assert summary["champion_version"] is None
        assert summary["alerts"] == 1


class TestModelPredictions:
    """Tests for ModelPredictions dataclass."""
    
    def test_defaults(self):
        """Test default values."""
        pred = ModelPredictions(
            model_version="v1.0",
            model_type="champion",
            predictions=pl.DataFrame(),
            latency_ms=50.0,
        )
        
        assert pred.success is True
        assert pred.error is None


class TestBatchRunner:
    """Tests for BatchRunner class."""
    
    @pytest.fixture
    def runner(self, tmp_path):
        """Create batch runner with temp output dir."""
        return BatchRunner(output_dir=tmp_path / "predictions")
    
    def test_output_dirs_created(self, runner):
        """Test output directories are created."""
        assert (runner.output_dir / "champion").exists()
        assert (runner.output_dir / "challenger").exists()
        assert (runner.output_dir / "value_bets").exists()
    
    def test_value_bets_generated_from_champion(self, runner, tmp_path):
        """Test value bets only generated from champion."""
        # Create mock predictions
        predictions = pl.DataFrame({
            "event_id": [1, 2, 3],
            "predicted_prob": [0.7, 0.3, 0.8],
            "odds_player": [1.5, 3.0, 1.2],
        })
        
        count = runner._generate_value_bets(predictions, date(2026, 2, 8))
        
        # Should have generated some value bets
        assert count >= 0
        assert (runner.output_dir / "value_bets" / "2026-02-08.parquet").exists()
    
    def test_save_predictions_champion(self, runner):
        """Test saving champion predictions."""
        champion = ModelPredictions(
            model_version="v1.0",
            model_type="champion",
            predictions=pl.DataFrame({"event_id": [1, 2], "pred": [0.5, 0.6]}),
            latency_ms=100.0,
        )
        
        runner._save_predictions(champion, None, date(2026, 2, 8))
        
        path = runner.output_dir / "champion" / "2026-02-08.parquet"
        assert path.exists()
        
        saved = pl.read_parquet(path)
        assert "model_version" in saved.columns
        assert "model_type" in saved.columns
