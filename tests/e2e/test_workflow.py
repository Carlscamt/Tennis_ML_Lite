# tests/e2e/test_workflow.py
import pytest
from pathlib import Path
from datetime import datetime, timedelta
import polars as pl
import numpy as np
from src.pipeline import TennisPipeline
from src.serving.batch_job import BatchOrchestrator
from unittest.mock import patch
import nest_asyncio
nest_asyncio.apply()

@pytest.mark.e2e
class TestFullWorkflow:
    """Test complete workflow end-to-end."""
    
    @pytest.fixture
    def workspace(self, tmp_path):
        """Setup test workspace."""
        d = tmp_path
        (d / "data" / "raw").mkdir(parents=True)
        (d / "data" / "processed").mkdir(parents=True)
        (d / "data" / "cache").mkdir(parents=True)
        (d / "models").mkdir(parents=True)
        return d
        
    def test_full_workflow_scrape_to_prediction(self, workspace, sample_raw_matches):
        """
        Full workflow:
        1. Ingest Data (Simulated Scrape)
        2. Data Pipeline (Process)
        3. Train Model
        4. Batch Prediction
        5. Verify Cache
        """
        # 1. Ingest Data
        raw_path = workspace / "data" / "raw" / "matches.parquet"
        sample_raw_matches.write_parquet(raw_path)
        
        # Initialize Pipeline with specific paths
        pipeline = TennisPipeline(root_dir=workspace)
        # pipeline.data_dir logic is now handled by root_dir, but we can verify/override if needed
        # Just ensured used dirs exist
        
        # Registry is auto-configured with root_dir in pipeline __init__ 
        
        # 2. Data Pipeline
        res = pipeline.run_data_pipeline()
        assert Path(res['output_path']).exists()
        
        # 3. Train
        pipeline.run_training_pipeline(Path(res['output_path']))
        models = pipeline.registry.list_models()
        assert len(models) > 0
        # Force promote to production for serving (Chain transitions)
        latest_v = models[0].version
        pipeline.registry.transition_stage(latest_v, "Staging")
        pipeline.registry.transition_stage(latest_v, "Production")
        pipeline.model_server.reload_models()
        
        # 4. Batch Prediction job
        # We need to mock 'predict_upcoming' to use our test data / or just run it via Orchestrator
        # Orchestrator uses its own pipeline instance. We should inject ours or patch paths.
        
        with patch("src.serving.batch_job.TennisPipeline") as MockPipelineCls:
            # We want the orchestrator to use the REAL pipeline logic but with our workspace paths.
            # So we shouldn't mock the class, but instance attributes.
            # Easier: Just instantiate BatchOrchestrator and Set attributes.
            
            orch = BatchOrchestrator()
            orch.pipeline = pipeline # Inject our configured pipeline
            orch.cache.cache_dir = workspace / "data" / "cache"
            orch.cache.cache_file = orch.cache.cache_dir / "daily_predictions.parquet"
            orch.cache.metadata_file = orch.cache.cache_dir / "cache_metadata.json"
            
            # Predict upcoming requires 'upcoming' matches.
            # We need to ensure logic finds them.
            # Pipeline.predict_upcoming calls _get_upcoming_matches.
            # We'll mock that specific method to return data that matches model features.
            
            # Create dummy upcoming that looks like processed feature-ready data?
            # Or raw upcoming?
            # predict_upcoming flow: get_upcoming -> feature_engineer -> drift -> server.predict
            
            dummy_upcoming = pl.DataFrame({
                 'event_id': [999], 'player_id': [100], 'opponent_id': [101],
                 'start_timestamp': [int(datetime.now().timestamp() + 86400)],
                 'player_name': ["Djokovic"], 'opponent_name': ["Alcaraz"],
                 'tournament_name': ["AO"], 'ground_type': ["Hard"],
                 'status': ["scheduled"],
                 'odds_player': [1.5], 'odds_opponent': [2.5],
                 'player_won': [None] # Target is unknown for upcoming matches
            }).with_columns(pl.col("player_won").cast(pl.Boolean))
            
            # We also need 'historical' data for features. 
            # Our pipeline.processed_dir has it.
            
            pipeline._get_upcoming_matches = lambda days: dummy_upcoming
            pipeline._identify_unknown_players = lambda u, h: [] # Skip scraping
            
            # Run Batch
            status = orch.run_daily_batch(force=True)
            assert status == "SUCCESS"
            
            # 5. Verify Cache
            cached = orch.get_predictions()
            assert not cached.is_empty()
            assert len(cached) == 1
