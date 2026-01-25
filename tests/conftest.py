# tests/conftest.py
import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
import polars as pl
import numpy as np
from unittest.mock import Mock, MagicMock

# Configure pytest
pytest_plugins = []

# Markers for test categorization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")


@pytest.fixture(scope="session")
def test_data_dir():
    """Temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_sofascore_response():
    """Mock SofaScore API response."""
    return {
        "events": [
            {
                "id": 12345,
                "homeTeam": {
                    "id": 1001,
                    "name": "Djokovic, Novak",
                    "type": "team"
                },
                "awayTeam": {
                    "id": 1002,
                    "name": "Alcaraz, Carlos",
                    "type": "team"
                },
                "homeScore": {"current": 2, "period1": 6, "period2": 4, "period3": 5},
                "awayScore": {"current": 1, "period1": 4, "period2": 6, "period3": 3},
                "status": {"type": "finished"},
                "winnerCode": 1,
                "startTimestamp": int((datetime.now() - timedelta(days=5)).timestamp()),
                "tournament": {
                    "uniqueTournament": {
                        "id": 100,
                        "name": "Australian Open",
                    },
                    "name": "Australian Open",
                    "category": {
                        "slug": "atp",
                        "name": "ATP"
                    }
                },
                "groundType": "Hard",
                "roundInfo": {"name": "Round 1"},
            }
        ]
    }


@pytest.fixture
def mock_circuit_breaker():
    """Mock rate limit circuit breaker."""
    from src.scraper import RateLimitCircuitBreaker
    breaker = RateLimitCircuitBreaker(failure_threshold=3, backoff_minutes=10)
    breaker.state = "closed"
    return breaker


@pytest.fixture
def sample_raw_matches():
    """Create sample raw match data."""
    # Generate 12 matches: 10 historical (2024), 2 future (2026)
    # This ensures we have enough past data for rolling windows (min_periods=3)
    # Generate 24 matches total: 20 historical (2024) to ensure >5 matches per player for stats
    # and 4 future (2026) for testing
    n_past = 20
    n_future = 4
    n_total = n_past + n_future
    
    return pl.DataFrame({
        "event_id": [1000 + i for i in range(n_total)],
        "tournament_name": ["Test Open"] * n_total,
        "tournament_id": [1] * n_total,
        "player_id": [100 + (i % 2) for i in range(n_total)],  # Alternate 100/101
        "opponent_id": [101 - (i % 2) for i in range(n_total)],
        "player_name": ["Djokovic" if i % 2 == 0 else "Alcaraz" for i in range(n_total)],
        "opponent_name": ["Alcaraz" if i % 2 == 0 else "Djokovic" for i in range(n_total)],
        "player_won": [True if i % 2 == 0 else False for i in range(n_total)],
        "is_home": [True] * n_total,
        "round_name": ["Final"] * n_total,
        "match_date": ["2024-01-01"] * n_past + ["2026-01-01"] * n_future,
        "odds_player": [1.5 if i % 2 == 0 else 2.5 for i in range(n_total)],
        "odds_opponent": [2.5 if i % 2 == 0 else 1.5 for i in range(n_total)],
        "surface": ["Hard"] * n_total,
        "ground_type": ["Hard"] * n_total,
        "player_sets": [2] * n_total,
        "opponent_sets": [1] * n_total,
        "start_timestamp": [
            int(datetime(2024, 1, 1).timestamp()) + (i * 86400) for i in range(n_past)
        ] + [
            int(datetime(2026, 1, 1).timestamp()) + (i * 86400) for i in range(n_future)
        ],
        "status": ["finished"] * n_total,
    })


@pytest.fixture
def sample_features():
    """Create sample feature dataframe."""
    return pl.DataFrame({
        "match_id": ["m1", "m2", "m3"],
        "start_timestamp": [
            int(datetime.now().timestamp()),
            int((datetime.now() - timedelta(days=1)).timestamp()),
            int((datetime.now() - timedelta(days=2)).timestamp()),
        ],
        "ranking_diff": [100, -50, 200],
        "h2h_win_rate_p1": [0.6, 0.4, 0.75],
        "surface_win_rate_p1": [0.65, 0.55, 0.80],
        "form_score_p1": [0.70, 0.50, 0.85],
        "target": [1, 0, 1],
    })


@pytest.fixture
def mock_model(tmp_path):
    """Create a mock XGBoost model."""
    import xgboost as xgb
    
    # Create and save a simple model
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    
    model_path = tmp_path / "test_model.json" # Use json for booster
    model.get_booster().save_model(str(model_path))
    
    return model, str(model_path)


@pytest.fixture
def mock_registry(tmp_path, mock_model):
    """Mock model registry."""
    from src.model.registry import ModelRegistry
    
    model, model_path = mock_model
    
    registry = ModelRegistry(model_name="test_model")
    registry.models_dir = tmp_path
    
    # Register a production model
    version = registry.register_model(
        model_path=model_path,
        auc=0.85,
        precision=0.82,
        recall=0.88,
        feature_schema_version="1.0",
        training_dataset_size=1000,
        stage="Production"
    )
    
    return registry


@pytest.fixture
def mock_logger(mocker):
    """Mock structured logger."""
    logger = mocker.MagicMock()
    logger.log_event = mocker.MagicMock()
    logger.log_error = mocker.MagicMock()
    return logger
