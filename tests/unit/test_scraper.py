# tests/unit/test_scraper.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import polars as pl
from src.scraper import (
    is_valid_event,
    convert_fractional,
    process_match,
    fetch_json,
    RateLimitCircuitBreaker,
)


class TestEventFiltering:
    """Test event filtering logic."""
    
    def test_valid_atp_event(self):
        """ATP singles match is valid."""
        event = {
            "tournament": {
                "category": {"slug": "atp", "name": "ATP"},
                "name": "Australian Open",
            },
            "homeTeam": {"id": 1, "name": "Djokovic", "type": "team"},
            "awayTeam": {"id": 2, "name": "Alcaraz", "type": "team"},
            "status": {"type": "scheduled"},
        }
        assert is_valid_event(event) is True
    
    def test_invalid_doubles_event(self):
        """Doubles match is invalid."""
        event = {
            "tournament": {
                "category": {"slug": "atp", "name": "ATP"},
                "name": "Australian Open Doubles",
            },
            "homeTeam": {"id": 1, "name": "Djokovic/Nedved", "type": "doubles"},
            "awayTeam": {"id": 2, "name": "Murray/Evans", "type": "doubles"},
        }
        assert is_valid_event(event) is False
    
    def test_invalid_itf_event(self):
        """ITF match is invalid."""
        event = {
            "tournament": {
                "category": {"slug": "itf", "name": "ITF"},
                "name": "ITF Women",
            },
            "homeTeam": {"id": 1, "name": "Player1", "type": "team"},
            "awayTeam": {"id": 2, "name": "Player2", "type": "team"},
        }
        assert is_valid_event(event) is False
    
    def test_invalid_exhibition_event(self):
        """Exhibition match is invalid."""
        event = {
            "tournament": {
                "category": {"slug": "atp", "name": "ATP"},
                "name": "ATP Exhibition",
            },
            "homeTeam": {"id": 1, "name": "Federer", "type": "team"},
            "awayTeam": {"id": 2, "name": "Nadal", "type": "team"},
        }
        assert is_valid_event(event) is False


class TestOddsConversion:
    """Test odds conversion logic."""
    
    @pytest.mark.parametrize("frac_str,expected", [
        ("1/2", 1.5),
        ("2/1", 3.0),
        ("1/1", 2.0),
        ("3/5", 1.6),
    ])
    def test_fractional_to_decimal(self, frac_str, expected):
        """Convert fractional odds to decimal."""
        result = convert_fractional(frac_str)
        assert abs(result - expected) < 0.01
    
    def test_decimal_passthrough(self):
        """Decimal odds pass through unchanged."""
        result = convert_fractional("2.5")
        assert result == 2.5
    
    def test_invalid_odds(self):
        """Invalid odds return None."""
        result = convert_fractional("invalid")
        assert result is None


class TestMatchProcessing:
    """Test match data processing."""
    
    def test_process_match_as_home_player(self):
        """Process match from home player perspective."""
        event = {
            "id": 123,
            "homeTeam": {"id": 1, "name": "Djokovic"},
            "awayTeam": {"id": 2, "name": "Alcaraz"},
            "homeScore": {"current": 2, "period1": 6, "period2": 4},
            "awayScore": {"current": 1, "period1": 4, "period2": 6},
            "winnerCode": 1,
            "startTimestamp": 1700000000,
            "tournament": {
                "uniqueTournament": {"id": 100, "name": "Australian Open"},
                "name": "Australian Open",
            },
            "groundType": "Hard",
            "roundInfo": {"name": "Round 1"},
            "status": {"type": "finished"},
        }
        
        record = process_match(event, player_id=1, data_type="historical")
        
        assert record["event_id"] == 123
        assert record["player_id"] == 1
        assert record["opponent_id"] == 2
        assert record["player_name"] == "Djokovic"
        assert record["opponent_name"] == "Alcaraz"
        assert record["player_won"] is True
        assert record["is_home"] is True
        assert record["tournament_name"] == "Australian Open"


class TestCircuitBreaker:
    """Test rate limit circuit breaker."""
    
    def test_circuit_breaker_closed_initially(self, mock_circuit_breaker):
        """Circuit breaker starts in closed state."""
        assert mock_circuit_breaker.is_open() is False
    
    def test_circuit_breaker_opens_on_failures(self, mock_circuit_breaker):
        """Circuit breaker opens after N failures."""
        # Record failures
        for _ in range(3):
            mock_circuit_breaker.record_failure(403)
        
        assert mock_circuit_breaker.is_open() is True
    
    def test_circuit_breaker_recovers(self, mock_circuit_breaker):
        """Circuit breaker recovers from failures."""
        # Record failures
        for _ in range(3):
            mock_circuit_breaker.record_failure(403)
        
        # Manually trigger recovery (in real scenario, time passes)
        mock_circuit_breaker.last_failure = datetime.now() - timedelta(minutes=11)
        
        # Should enter half-open state
        is_open = mock_circuit_breaker.is_open()
        assert mock_circuit_breaker.state == "half_open"


@pytest.mark.parametrize("http_method,expected_call_count", [
    ("get", 1),
    ("post", 1),
])
@patch("src.scraper.get_session")
def test_fetch_json_success(mock_session, http_method, expected_call_count, mock_sofascore_response):
    """Successfully fetch JSON from API."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_sofascore_response
    
    mock_session_instance = Mock()
    mock_session_instance.get.return_value = mock_response
    mock_session.return_value = mock_session_instance
    
    result = fetch_json("/test-endpoint")
    
    assert result == mock_sofascore_response
    assert mock_session_instance.get.call_count == expected_call_count
