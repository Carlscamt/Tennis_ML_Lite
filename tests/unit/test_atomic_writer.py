"""
Unit tests for atomic Parquet writer.
"""
import pytest
import polars as pl
from pathlib import Path

from src.scraper.atomic_writer import (
    AtomicParquetWriter,
    WriteResult,
    get_atomic_writer,
)


class TestWriteResult:
    """Tests for WriteResult dataclass."""
    
    def test_success_result(self):
        """Test successful result."""
        result = WriteResult(
            success=True,
            path=Path("test.parquet"),
            rows_written=100,
        )
        
        assert result.success
        assert result.rows_written == 100
        assert result.error is None


class TestAtomicParquetWriter:
    """Tests for AtomicParquetWriter."""
    
    @pytest.fixture
    def writer(self):
        return AtomicParquetWriter()
    
    @pytest.fixture
    def sample_df(self):
        return pl.DataFrame({
            "event_id": [1, 2, 3],
            "player_id": [100, 200, 300],
            "odds_player": [1.5, 2.0, 2.5],
        })
    
    def test_write_safe_success(self, writer, sample_df, tmp_path):
        """Test successful atomic write."""
        path = tmp_path / "test.parquet"
        
        result = writer.write_safe(sample_df, path)
        
        assert result.success
        assert result.rows_written == 3
        assert path.exists()
        
        # Verify content
        loaded = pl.read_parquet(path)
        assert len(loaded) == 3
    
    def test_write_safe_creates_parent_dirs(self, writer, sample_df, tmp_path):
        """Test parent directories are created."""
        path = tmp_path / "subdir" / "nested" / "test.parquet"
        
        result = writer.write_safe(sample_df, path)
        
        assert result.success
        assert path.exists()
    
    def test_write_safe_no_temp_file_left(self, writer, sample_df, tmp_path):
        """Test no temp file remains after write."""
        path = tmp_path / "test.parquet"
        temp_path = path.with_suffix(path.suffix + writer.temp_suffix)
        
        writer.write_safe(sample_df, path)
        
        assert not temp_path.exists()
    
    def test_write_failure_no_partial_file(self, writer, tmp_path):
        """Test failed write doesn't leave partial file."""
        path = tmp_path / "test.parquet"
        
        # Create writer that will fail
        writer_bad = AtomicParquetWriter()
        
        # Try to write None (should fail)
        result = writer_bad.write_safe(None, path)
        
        assert not result.success
        # Original file shouldn't exist if it didn't before
    
    def test_merge_and_write_new_file(self, writer, sample_df, tmp_path):
        """Test merge with non-existent file."""
        path = tmp_path / "test.parquet"
        
        result = writer.merge_and_write(sample_df, path)
        
        assert result.success
        assert result.rows_written == 3
    
    def test_merge_deduplicates(self, writer, tmp_path):
        """Test merge deduplicates on unique key."""
        path = tmp_path / "test.parquet"
        
        # Write initial data
        df1 = pl.DataFrame({
            "event_id": [1, 2],
            "player_id": [100, 200],
            "value": ["a", "b"],
        })
        writer.write_safe(df1, path)
        
        # Merge with overlapping data
        df2 = pl.DataFrame({
            "event_id": [2, 3],
            "player_id": [200, 300],
            "value": ["b_new", "c"],
        })
        result = writer.merge_and_write(df2, path)
        
        assert result.success
        assert result.rows_deduplicated > 0
        
        # Verify deduplicated
        loaded = pl.read_parquet(path)
        assert len(loaded) == 3  # 1, 2, 3 (not 4)
    
    def test_merge_prefer_new(self, writer, tmp_path):
        """Test new data takes precedence."""
        path = tmp_path / "test.parquet"
        
        # Write initial
        df1 = pl.DataFrame({
            "event_id": [1],
            "player_id": [100],
            "value": ["old"],
        })
        writer.write_safe(df1, path)
        
        # Merge with updated value
        df2 = pl.DataFrame({
            "event_id": [1],
            "player_id": [100],
            "value": ["new"],
        })
        writer.merge_and_write(df2, path, prefer_new=True)
        
        # Check value was updated
        loaded = pl.read_parquet(path)
        assert loaded["value"][0] == "new"
    
    def test_skip_unchanged_rows(self, writer, tmp_path):
        """Test unchanged rows are detected."""
        path = tmp_path / "test.parquet"
        
        # Write initial
        df1 = pl.DataFrame({
            "event_id": [1, 2],
            "player_id": [100, 200],
            "value": ["a", "b"],
        })
        writer.write_safe(df1, path)
        
        # Merge with same data
        result = writer.merge_and_write(df1, path, skip_unchanged=True)
        
        # Should detect unchanged
        assert result.rows_unchanged > 0
    
    def test_exists_with_key(self, writer, tmp_path):
        """Test key existence check."""
        path = tmp_path / "test.parquet"
        
        df = pl.DataFrame({
            "event_id": [1, 2],
            "player_id": [100, 200],
        })
        writer.write_safe(df, path)
        
        # Check existence
        exists = writer.exists_with_key(
            path,
            ["event_id", "player_id"],
            [(1, 100), (3, 300), (2, 200)]
        )
        
        assert exists == [True, False, True]
    
    def test_exists_with_key_no_file(self, writer, tmp_path):
        """Test existence check on missing file."""
        path = tmp_path / "nonexistent.parquet"
        
        exists = writer.exists_with_key(path, ["id"], [(1,), (2,)])
        
        assert exists == [False, False]


class TestGlobalAtomicWriter:
    """Tests for global instance."""
    
    def test_get_atomic_writer_singleton(self):
        """Test global writer is singleton."""
        w1 = get_atomic_writer()
        w2 = get_atomic_writer()
        
        assert w1 is w2
