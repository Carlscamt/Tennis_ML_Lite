"""
Unit tests for quarantine system.
"""
import pytest
import json
from pathlib import Path
from datetime import datetime

from src.scraper.quarantine import (
    QuarantineManager,
    QuarantineRecord,
    get_quarantine_manager,
)


class TestQuarantineRecord:
    """Tests for QuarantineRecord dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        record = QuarantineRecord(
            record={"event_id": 123, "player_id": 456},
            reason="missing odds",
            source="matches",
            timestamp="2024-01-15T12:00:00",
            record_id="123_456",
        )
        
        d = record.to_dict()
        
        assert d["record"]["event_id"] == 123
        assert d["reason"] == "missing odds"
        assert d["source"] == "matches"
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "record": {"event_id": 789},
            "reason": "invalid type",
            "source": "rankings",
            "timestamp": "2024-01-15T12:00:00",
        }
        
        record = QuarantineRecord.from_dict(data)
        
        assert record.record["event_id"] == 789
        assert record.reason == "invalid type"


class TestQuarantineManager:
    """Tests for QuarantineManager."""
    
    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager with temp directory."""
        return QuarantineManager(quarantine_dir=tmp_path / "quarantine")
    
    def test_quarantine_row(self, manager):
        """Test quarantining a single row."""
        record = {"event_id": 123, "player_id": 456}
        
        result = manager.quarantine_row(record, "test reason", "matches")
        
        assert result is True
        assert manager.get_quarantined_count() == 1
    
    def test_quarantine_row_creates_jsonl(self, manager):
        """Test JSONL file is created."""
        record = {"event_id": 123}
        manager.quarantine_row(record, "test", "matches")
        
        # Check file exists
        date_str = datetime.now().strftime("%Y-%m-%d")
        file_path = manager.quarantine_dir / date_str / "matches.jsonl"
        
        assert file_path.exists()
        
        # Verify content
        with open(file_path, "r") as f:
            line = f.readline()
            data = json.loads(line)
            assert data["record"]["event_id"] == 123
            assert data["reason"] == "test"
    
    def test_quarantine_multiple(self, manager):
        """Test quarantining multiple rows."""
        for i in range(5):
            manager.quarantine_row({"event_id": i}, f"error {i}", "matches")
        
        assert manager.get_quarantined_count() == 5
    
    def test_list_quarantined(self, manager):
        """Test listing quarantined records."""
        manager.quarantine_row({"event_id": 1}, "error1", "matches")
        manager.quarantine_row({"event_id": 2}, "error2", "rankings")
        
        records = manager.list_quarantined()
        
        assert len(records) == 2
        assert all(isinstance(r, QuarantineRecord) for r in records)
    
    def test_list_by_source(self, manager):
        """Test filtering by source."""
        manager.quarantine_row({"event_id": 1}, "error1", "matches")
        manager.quarantine_row({"event_id": 2}, "error2", "rankings")
        manager.quarantine_row({"event_id": 3}, "error3", "matches")
        
        records = manager.list_quarantined(source="matches")
        
        assert len(records) == 2
    
    def test_get_summary(self, manager):
        """Test summary statistics."""
        manager.quarantine_row({"event_id": 1}, "error1", "matches")
        manager.quarantine_row({"event_id": 2}, "error2", "matches")
        manager.quarantine_row({"event_id": 3}, "error3", "rankings")
        
        summary = manager.get_summary()
        
        assert summary["total_count"] == 3
        assert summary["by_source"]["matches"] == 2
        assert summary["by_source"]["rankings"] == 1


class TestQuarantineManagerWithDataFrame:
    """Tests for DataFrame quarantine."""
    
    @pytest.fixture
    def manager(self, tmp_path):
        return QuarantineManager(quarantine_dir=tmp_path / "quarantine")
    
    def test_quarantine_dataframe(self, manager):
        """Test quarantining a DataFrame."""
        import polars as pl
        
        df = pl.DataFrame({
            "event_id": [1, 2, 3],
            "player_id": [100, 200, 300],
        })
        
        count = manager.quarantine_dataframe(df, ["missing field"], "matches")
        
        assert count == 3
        assert manager.get_quarantined_count() == 3
    
    def test_quarantine_empty_dataframe(self, manager):
        """Test quarantining empty DataFrame."""
        import polars as pl
        
        df = pl.DataFrame()
        count = manager.quarantine_dataframe(df, ["error"], "matches")
        
        assert count == 0


class TestGlobalQuarantineManager:
    """Tests for global instance."""
    
    def test_get_quarantine_manager(self):
        """Test global manager is singleton."""
        m1 = get_quarantine_manager()
        m2 = get_quarantine_manager()
        
        assert m1 is m2
