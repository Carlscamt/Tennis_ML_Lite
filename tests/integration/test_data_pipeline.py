# tests/integration/test_data_pipeline.py
import pytest
from pathlib import Path
from datetime import datetime
import polars as pl
from src.pipeline import TennisPipeline


@pytest.mark.integration
class TestDataPipeline:
    """Test complete data pipeline."""
    
    @pytest.fixture
    def pipeline(self, tmp_path):
        """Initialize pipeline with temp directory."""
        pipeline = TennisPipeline()
        pipeline.data_dir = tmp_path / "data"
        pipeline.data_dir.mkdir(exist_ok=True)
        pipeline.raw_dir = pipeline.data_dir / "raw"
        pipeline.raw_dir.mkdir(exist_ok=True)
        pipeline.processed_dir = pipeline.data_dir / "processed"
        return pipeline
    
    def test_run_data_pipeline_creates_output(self, pipeline, sample_raw_matches):
        """Data pipeline creates processed features."""
        # Save sample data
        output_path = pipeline.raw_dir / "atp_matches_test.parquet"
        sample_raw_matches.write_parquet(output_path)
        
        # Run pipeline
        result = pipeline.run_data_pipeline()
        
        assert "output_path" in result
        assert result['count'] > 0
        
        # Verify processed file
        processed_file = pipeline.processed_dir / "features_dataset.parquet"
        assert processed_file.exists()
    
    def test_data_validation_catches_schema_errors(self, pipeline):
        """Data pipeline validates schema."""
        # Create invalid raw data (odds < 1.0)
        invalid_data = pl.DataFrame({
             'event_id': [1], 'player_id': [100], 'opponent_id': [200], 
             'start_timestamp': [123456], 'player_name': ["Djokovic"], 'opponent_name': ["Alcaraz"],
             'odds_player': [-0.5], # Invalid (< 0.0)
             'odds_opponent': [2.5],
             'tournament_name': ["Grand Slam"],
             'ground_type': ["Hard"],
             'player_won': [True],
             'status': ["finished"]
        })
        
        output_path = pipeline.raw_dir / "invalid.parquet"
        invalid_data.write_parquet(output_path)
        
        # Should raise error during validation
        with pytest.raises(ValueError, match="Schema Validation Failed"):
            pipeline.run_data_pipeline()
    
    def test_feature_engineering_produces_consistent_output(self, pipeline, sample_raw_matches):
        """Features are consistently engineered."""
        output_path = pipeline.raw_dir / "raw.parquet"
        sample_raw_matches.write_parquet(output_path)
        
        # Process same data twice
        result1 = pipeline.run_data_pipeline()
        df1 = pl.read_parquet(result1['output_path'])
        
        # Run again
        result2 = pipeline.run_data_pipeline()
        df2 = pl.read_parquet(result2['output_path'])
        
        # Results should be identical
        assert df1.shape == df2.shape
        # Compare columns
        assert df1.columns == df2.columns
