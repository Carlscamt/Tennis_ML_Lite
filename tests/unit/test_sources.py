"""
Unit tests for multi-source data adapters and validation.
"""
import pytest
from datetime import date
from unittest.mock import patch, MagicMock
import polars as pl

from src.sources.sackmann import SackmannDataSource
from src.sources.canonical import (
    CanonicalMatch,
    DataSource,
    Surface,
    to_canonical_from_sackmann,
    to_canonical_from_sofascore,
    canonical_to_dataframe,
)
from src.sources.validator import (
    CrossSourceValidator,
    Anomaly,
    ValidationReport,
)


class TestSackmannDataSource:
    """Tests for SackmannDataSource."""
    
    @pytest.fixture
    def source(self, tmp_path):
        return SackmannDataSource(cache_dir=tmp_path / "cache")
    
    def test_surface_mapping(self, source):
        """Test surface mapping is complete."""
        mapping = source.get_surface_mapping()
        assert "hard" in mapping
        assert "clay" in mapping
        assert "grass" in mapping
    
    def test_normalize_matches_renames_columns(self, source):
        """Test column renaming in match normalization."""
        df = pl.DataFrame({
            "tourney_id": ["2023-001"],
            "tourney_name": ["Test Open"],
            "winner_id": [12345],
            "loser_id": [67890],
            "surface": ["Hard"],
        })
        
        normalized = source._normalize_matches(df)
        
        assert "tournament_id" in normalized.columns
        assert "tournament_name" in normalized.columns
        assert "surface" in normalized.columns
    
    def test_normalize_surface_lowercase(self, source):
        """Test surface is lowercased."""
        df = pl.DataFrame({
            "surface": ["HARD", "Clay", "Grass"],
        })
        
        normalized = source._normalize_matches(df)
        
        assert normalized["surface"].to_list() == ["hard", "clay", "grass"]


class TestCanonicalMatch:
    """Tests for CanonicalMatch dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        match = CanonicalMatch(
            source=DataSource.SACKMANN,
            match_id="test_123",
            match_date=date(2023, 6, 15),
            winner_id=12345,
            winner_name="Player A",
            loser_id=67890,
            loser_name="Player B",
            surface=Surface.CLAY,
        )
        
        d = match.to_dict()
        
        assert d["source"] == "sackmann"
        assert d["winner_id"] == 12345
        assert d["surface"] == "clay"
    
    def test_optional_fields_default_none(self):
        """Test optional fields default to None."""
        match = CanonicalMatch(
            source=DataSource.SOFASCORE,
            match_id="test",
            match_date=date.today(),
            winner_id=1,
            winner_name="A",
            loser_id=2,
            loser_name="B",
        )
        
        assert match.winner_odds is None
        assert match.score is None


class TestCanonicalConversion:
    """Tests for source-to-canonical conversion."""
    
    def test_sackmann_to_canonical(self):
        """Test Sackmann DataFrame conversion."""
        df = pl.DataFrame({
            "tournament_id": ["2023-001"],
            "tournament_name": ["Test Open"],
            "match_date": [date(2023, 6, 15)],
            "winner_player_id": [12345],
            "winner_name": ["Player A"],
            "loser_player_id": [67890],
            "loser_name": ["Player B"],
            "surface": ["clay"],
        })
        
        matches = to_canonical_from_sackmann(df)
        
        assert len(matches) == 1
        assert matches[0].winner_id == 12345
        assert matches[0].surface == Surface.CLAY
    
    def test_sofascore_to_canonical_winner(self):
        """Test SofaScore conversion for winner row."""
        df = pl.DataFrame({
            "event_id": [123],
            "player_id": [12345],
            "player_name": ["Player A"],
            "opponent_id": [67890],
            "opponent_name": ["Player B"],
            "player_won": [True],
            "start_timestamp": [1686844800],  # 2023-06-15
            "ground_type": ["clay"],
            "odds_player": [1.5],
            "odds_opponent": [2.5],
        })
        
        matches = to_canonical_from_sofascore(df)
        
        assert len(matches) == 1
        assert matches[0].winner_id == 12345
        assert matches[0].winner_odds == 1.5
    
    def test_canonical_to_dataframe(self):
        """Test conversion back to DataFrame."""
        matches = [
            CanonicalMatch(
                source=DataSource.SACKMANN,
                match_id="1",
                match_date=date(2023, 6, 15),
                winner_id=1,
                winner_name="A",
                loser_id=2,
                loser_name="B",
            ),
            CanonicalMatch(
                source=DataSource.SACKMANN,
                match_id="2",
                match_date=date(2023, 6, 16),
                winner_id=3,
                winner_name="C",
                loser_id=4,
                loser_name="D",
            ),
        ]
        
        df = canonical_to_dataframe(matches)
        
        assert len(df) == 2
        assert "winner_id" in df.columns


class TestCrossSourceValidator:
    """Tests for CrossSourceValidator."""
    
    @pytest.fixture
    def validator(self):
        return CrossSourceValidator(divergence_threshold=0.10)
    
    @pytest.fixture
    def sample_matches(self):
        """Create sample matches for both sources."""
        player_id = 12345
        
        # Player wins 8/10 in Sackmann
        sackmann = []
        for i in range(10):
            if i < 8:
                sackmann.append(CanonicalMatch(
                    source=DataSource.SACKMANN,
                    match_id=f"sack_{i}",
                    match_date=date(2023, 6, i+1),
                    winner_id=player_id,
                    winner_name="Test Player",
                    loser_id=99999,
                    loser_name="Opponent",
                ))
            else:
                sackmann.append(CanonicalMatch(
                    source=DataSource.SACKMANN,
                    match_id=f"sack_{i}",
                    match_date=date(2023, 6, i+1),
                    winner_id=99999,
                    winner_name="Opponent",
                    loser_id=player_id,
                    loser_name="Test Player",
                ))
        
        # Player wins 5/10 in SofaScore (divergence!)
        sofascore = []
        for i in range(10):
            if i < 5:
                sofascore.append(CanonicalMatch(
                    source=DataSource.SOFASCORE,
                    match_id=f"sofa_{i}",
                    match_date=date(2023, 6, i+1),
                    winner_id=player_id,
                    winner_name="Test Player",
                    loser_id=99999,
                    loser_name="Opponent",
                ))
            else:
                sofascore.append(CanonicalMatch(
                    source=DataSource.SOFASCORE,
                    match_id=f"sofa_{i}",
                    match_date=date(2023, 6, i+1),
                    winner_id=99999,
                    winner_name="Opponent",
                    loser_id=player_id,
                    loser_name="Test Player",
                ))
        
        return sackmann, sofascore
    
    def test_compare_win_rates(self, validator, sample_matches):
        """Test win rate comparison."""
        sackmann, sofascore = sample_matches
        validator.add_matches(sackmann)
        validator.add_matches(sofascore)
        
        rates = validator.compare_win_rates(player_id=12345)
        
        assert "sackmann" in rates
        assert "sofascore" in rates
        assert rates["sackmann"]["win_rate"] == 0.8
        assert rates["sofascore"]["win_rate"] == 0.5
    
    def test_detect_anomalies(self, validator, sample_matches):
        """Test anomaly detection."""
        sackmann, sofascore = sample_matches
        validator.add_matches(sackmann)
        validator.add_matches(sofascore)
        
        anomalies = validator.detect_anomalies()
        
        assert len(anomalies) > 0
        assert anomalies[0].anomaly_type == "win_rate_divergence"
    
    def test_validate_returns_report(self, validator, sample_matches):
        """Test full validation returns report."""
        sackmann, sofascore = sample_matches
        validator.add_matches(sackmann)
        validator.add_matches(sofascore)
        
        report = validator.validate()
        
        assert isinstance(report, ValidationReport)
        assert report.matches_compared == 20
        assert report.overall_status in ["healthy", "warning", "alert"]
    
    def test_no_anomalies_when_consistent(self, validator):
        """Test no anomalies when sources agree."""
        # Both sources: player wins 7/10
        for source in [DataSource.SACKMANN, DataSource.SOFASCORE]:
            matches = []
            for i in range(10):
                if i < 7:
                    matches.append(CanonicalMatch(
                        source=source,
                        match_id=f"{source.value}_{i}",
                        match_date=date(2023, 6, i+1),
                        winner_id=12345,
                        winner_name="Test",
                        loser_id=99999,
                        loser_name="Opp",
                    ))
                else:
                    matches.append(CanonicalMatch(
                        source=source,
                        match_id=f"{source.value}_{i}",
                        match_date=date(2023, 6, i+1),
                        winner_id=99999,
                        winner_name="Opp",
                        loser_id=12345,
                        loser_name="Test",
                    ))
            validator.add_matches(matches)
        
        report = validator.validate()
        
        # No high-severity anomalies expected
        high_anomalies = [a for a in report.anomalies if a.severity == "high"]
        assert len(high_anomalies) == 0
