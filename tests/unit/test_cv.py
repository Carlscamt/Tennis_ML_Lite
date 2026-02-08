"""
Unit tests for TimeSeriesBettingCV.
"""
import pytest
import numpy as np
import polars as pl
from datetime import date, timedelta

from src.model.cv import TimeSeriesBettingCV, FoldInfo, get_cv_splitter


class TestTimeSeriesBettingCV:
    """Test suite for TimeSeriesBettingCV splitter."""
    
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create a sample DataFrame with dates and outcomes."""
        n_samples = 10000
        base_date = date(2023, 1, 1)
        
        # Generate dates over 2 years
        dates = [base_date + timedelta(days=i // 15) for i in range(n_samples)]
        
        return pl.DataFrame({
            "match_date": dates,
            "player_won": np.random.randint(0, 2, n_samples).tolist(),
            "odds_player": (1.5 + np.random.rand(n_samples)).tolist(),
            "feature_1": np.random.randn(n_samples).tolist(),
            "feature_2": np.random.randn(n_samples).tolist(),
        })
    
    def test_basic_split_generation(self, sample_df):
        """Test that CV generates the expected number of folds."""
        cv = TimeSeriesBettingCV(n_splits=5, gap_days=7, min_train_size=1000)
        
        folds = list(cv.split(sample_df, "match_date"))
        
        # Should generate folds (may be less than n_splits if data insufficient)
        assert len(folds) >= 1
        
        for train_idx, test_idx in folds:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
    
    def test_non_overlapping_folds(self, sample_df):
        """Test that train and test sets don't overlap within each fold."""
        cv = TimeSeriesBettingCV(n_splits=5, gap_days=0, min_train_size=1000)
        
        for train_idx, test_idx in cv.split(sample_df, "match_date"):
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0
    
    def test_temporal_ordering(self, sample_df):
        """Test that all training samples come before test samples."""
        cv = TimeSeriesBettingCV(n_splits=3, gap_days=7, min_train_size=1000)
        
        for train_idx, test_idx in cv.split(sample_df, "match_date"):
            # All train indices should be less than all test indices
            assert max(train_idx) < min(test_idx)
    
    def test_minimum_train_size_respected(self, sample_df):
        """Test that minimum training size is enforced."""
        min_size = 3000
        cv = TimeSeriesBettingCV(n_splits=5, min_train_size=min_size)
        
        for train_idx, test_idx in cv.split(sample_df, "match_date"):
            assert len(train_idx) >= min_size
    
    def test_gap_days_creates_separation(self, sample_df):
        """Test that gap_days creates temporal separation."""
        cv_no_gap = TimeSeriesBettingCV(n_splits=3, gap_days=0, min_train_size=1000)
        cv_with_gap = TimeSeriesBettingCV(n_splits=3, gap_days=30, min_train_size=1000)
        
        folds_no_gap = list(cv_no_gap.split(sample_df, "match_date"))
        folds_with_gap = list(cv_with_gap.split(sample_df, "match_date"))
        
        # With gap, test indices should start later
        if folds_with_gap:
            train_no_gap, test_no_gap = folds_no_gap[0]
            train_gap, test_gap = folds_with_gap[0]
            
            # Test set starts later with gap
            assert min(test_gap) >= min(test_no_gap)
    
    def test_expanding_window_uses_all_prior_data(self, sample_df):
        """Test that expanding window includes all prior data."""
        cv = TimeSeriesBettingCV(n_splits=3, min_train_size=1000, rolling_window_days=None)
        
        prev_train_size = 0
        for train_idx, _ in cv.split(sample_df, "match_date"):
            # Each fold should have at least as much training data as previous
            assert len(train_idx) >= prev_train_size
            prev_train_size = len(train_idx)
    
    def test_get_fold_info(self, sample_df):
        """Test fold info metadata."""
        cv = TimeSeriesBettingCV(n_splits=3, min_train_size=1000)
        
        fold_infos = cv.get_fold_info(sample_df, "match_date")
        
        assert len(fold_infos) >= 1
        for info in fold_infos:
            assert isinstance(info, FoldInfo)
            assert info.train_size > 0
            assert info.test_size > 0
    
    def test_small_dataset_raises_error(self):
        """Test that small dataset raises appropriate error."""
        small_df = pl.DataFrame({
            "match_date": [date(2023, 1, 1) + timedelta(days=i) for i in range(100)],
            "player_won": [1] * 100,
        })
        
        cv = TimeSeriesBettingCV(n_splits=5, min_train_size=5000)
        
        with pytest.raises(ValueError, match="too small"):
            list(cv.split(small_df, "match_date"))
    
    def test_missing_date_column_raises_error(self, sample_df):
        """Test that missing date column raises error."""
        cv = TimeSeriesBettingCV(n_splits=3)
        
        with pytest.raises(ValueError, match="not found"):
            list(cv.split(sample_df, "nonexistent_date_col"))
    
    def test_factory_function(self):
        """Test get_cv_splitter factory function."""
        cv = get_cv_splitter(n_splits=5, gap_days=14, min_train_size=2000)
        
        assert cv.n_splits == 5
        assert cv.gap_days == 14
        assert cv.min_train_size == 2000


class TestTimeSeriesBettingCVEdgeCases:
    """Edge case tests for CV splitter."""
    
    def test_single_split(self):
        """Test with minimum splits."""
        df = pl.DataFrame({
            "match_date": [date(2023, 1, 1) + timedelta(days=i) for i in range(200)],
            "player_won": [1] * 200,
        })
        
        cv = TimeSeriesBettingCV(n_splits=2, min_train_size=50)
        folds = list(cv.split(df, "match_date"))
        
        assert len(folds) >= 1
    
    def test_lazy_dataframe_materialized(self):
        """Test that lazy DataFrames are properly materialized."""
        df = pl.DataFrame({
            "match_date": [date(2023, 1, 1) + timedelta(days=i) for i in range(500)],
            "player_won": [1] * 500,
        }).lazy()
        
        cv = TimeSeriesBettingCV(n_splits=3, min_train_size=100)
        
        # Should work with lazy frame
        folds = list(cv.split(df, "match_date"))
        assert len(folds) >= 1
