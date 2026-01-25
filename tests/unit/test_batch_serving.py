
"""
Tests for Batch Serving and Caching.
"""
import pytest
import shutil
from pathlib import Path
import polars as pl
from unittest.mock import MagicMock, patch
from src.serving.cache import PredictionCache
from src.serving.batch_job import BatchOrchestrator

@pytest.fixture
def temp_cache_dir(tmp_path):
    return tmp_path / "cache_test"

def test_cache_save_load(temp_cache_dir):
    cache = PredictionCache(cache_dir=str(temp_cache_dir))
    
    df = pl.DataFrame({
        "game_id": [1], 
        "prediction": [1],
        "model_version": ["v1"]
    })
    
    cache.save(df, ttl_hours=1)
    
    # Reload
    loaded = cache.get(max_age_hours=1)
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded["game_id"][0] == 1

def test_cache_ttl(temp_cache_dir):
    cache = PredictionCache(cache_dir=str(temp_cache_dir))
    
    df = pl.DataFrame({"x": [1]})
    
    # Save with 1 hour TTL
    cache.save(df, ttl_hours=1)
    
    # Mock datetime to simulate expiration
    # (metadata file holds timestamp)
    
    # Actually simpler: cache.get checks freshness.
    # Pass max_age_hours=0 to force expire immediately for test
    expired = cache.get(max_age_hours=0)
    assert expired is None

@patch("src.serving.batch_job.TennisPipeline")
def test_batch_orchestrator(mock_pipeline_cls, temp_cache_dir):
    mock_pipeline = mock_pipeline_cls.return_value
    mock_pipeline.predict_upcoming.return_value = pl.DataFrame({"game_id": [1]})
    
    orch = BatchOrchestrator()
    orch.cache = PredictionCache(cache_dir=str(temp_cache_dir))
    
    # 1. First Run
    status = orch.run_daily_batch(force=True)
    assert status == "SUCCESS"
    assert mock_pipeline.predict_upcoming.called
    
    # 2. Check Cache
    cached = orch.get_predictions()
    assert len(cached) == 1
    
    # 3. Second Run (Skipped)
    status_skip = orch.run_daily_batch(days=7) # default not force
    assert status_skip == "SKIPPED"
