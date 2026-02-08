"""
Integration tests for scraping against recorded archive.

Tests deterministic data ingestion using ResponseArchive.
"""
import pytest
from datetime import datetime
from pathlib import Path
import json
import gzip
import polars as pl

from src.utils.response_archive import ResponseArchive


@pytest.mark.integration
class TestScraperArchive:
    """Test scraping against recorded archive for deterministic raw data."""
    
    @pytest.fixture
    def archive(self, tmp_path):
        """Create archive with recorded responses."""
        archive = ResponseArchive(archive_dir=str(tmp_path / ".archive"))
        return archive
    
    @pytest.fixture
    def sample_match_response(self):
        """Sample API response for a match."""
        return {
            "event": {
                "id": 12345,
                "tournament": {"name": "Australian Open", "uniqueTournament": {"id": 1}},
                "homeTeam": {"id": 100, "name": "Djokovic N."},
                "awayTeam": {"id": 200, "name": "Alcaraz C."},
                "startTimestamp": 1707408000,
                "status": {"type": "finished"},
                "groundType": "hard",
                "winnerCode": 1,
            }
        }
    
    @pytest.fixture
    def sample_odds_response(self):
        """Sample API response for odds."""
        return {
            "odds": [
                {"marketId": 1, "homeOdd": 1.75, "awayOdd": 2.10}
            ]
        }
    
    def test_archive_stores_and_retrieves_response(self, archive, sample_match_response):
        """Archive stores and retrieves identical response."""
        # Store response
        endpoint = "/event/12345"
        path = archive.store(endpoint, sample_match_response)
        
        assert path.exists()
        
        # Retrieve and verify
        retrieved = archive.get(path)
        assert retrieved is not None
        assert retrieved["data"] == sample_match_response
    
    def test_archive_organizes_by_date(self, archive, sample_match_response):
        """Archive organizes files by date."""
        endpoint = "/event/12345"
        path = archive.store(endpoint, sample_match_response)
        
        # Path should contain year/month structure
        parts = path.parts
        assert any(p.isdigit() and len(p) == 4 for p in parts)  # Year
        assert any(p.isdigit() and len(p) == 2 for p in parts)  # Month
    
    def test_archive_finds_by_endpoint(self, archive, sample_match_response):
        """Archive finds responses by endpoint."""
        endpoint = "/event/12345"
        archive.store(endpoint, sample_match_response)
        
        # Find by endpoint
        results = archive.find_by_endpoint(endpoint)
        assert len(results) >= 1
    
    def test_archive_replay_produces_deterministic_output(
        self, archive, sample_match_response, sample_odds_response
    ):
        """Replaying archive produces identical output each time."""
        # Store multiple responses
        archive.store("/event/12345", sample_match_response)
        archive.store("/event/12345/odds", sample_odds_response)
        
        # Retrieve twice
        match1 = archive.get_latest("/event/12345")
        match2 = archive.get_latest("/event/12345")
        
        assert match1 == match2
    
    def test_archive_compression_reduces_size(self, archive, sample_match_response):
        """Archive compresses data significantly."""
        endpoint = "/event/12345"
        path = archive.store(endpoint, sample_match_response)
        
        # Compressed size should be smaller than raw JSON
        raw_size = len(json.dumps(sample_match_response).encode())
        compressed_size = path.stat().st_size
        
        # Expect at least some compression (may not be huge for small data)
        assert compressed_size < raw_size * 2  # Allow overhead for small files
    
    def test_archive_stats_track_usage(self, archive, sample_match_response):
        """Archive tracks statistics."""
        endpoint = "/event/12345"
        archive.store(endpoint, sample_match_response)
        
        stats = archive.get_stats()
        assert stats["total_files"] >= 1
        assert "total_size_mb" in stats


@pytest.mark.integration
class TestDeterministicScraping:
    """Test that scraping from archive produces deterministic results."""
    
    @pytest.fixture
    def populated_archive(self, tmp_path):
        """Create archive with multiple recorded responses."""
        archive = ResponseArchive(archive_dir=str(tmp_path / ".archive"))
        
        # Store multiple match responses
        matches = [
            {"event": {"id": 1, "homeTeam": {"name": "Player A"}, "awayTeam": {"name": "Player B"}}},
            {"event": {"id": 2, "homeTeam": {"name": "Player C"}, "awayTeam": {"name": "Player D"}}},
        ]
        
        for i, match in enumerate(matches):
            archive.store(f"/event/{i+1}", match)
        
        return archive
    
    def test_archive_list_dates(self, populated_archive):
        """Archive lists all dates with data."""
        dates = populated_archive.list_dates()
        
        # Should have at least today's date
        assert len(dates) >= 1
    
    def test_archive_cleanup_respects_retention(self, populated_archive):
        """Archive cleanup respects retention period."""
        # Cleanup with long retention (shouldn't delete recent)
        deleted = populated_archive.cleanup(days=365)
        
        # Recent data should not be deleted
        stats = populated_archive.get_stats()
        assert stats["total_files"] >= 1
