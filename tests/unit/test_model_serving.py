
"""
Tests for Model Serving (Canary/Shadow).
"""
import pytest
import numpy as np
from src.model.serving import ModelServer, ServingConfig, ServingMode
from src.model.registry import ModelRegistry
import xgboost as xgb
from unittest.mock import MagicMock, AsyncMock

@pytest.fixture
def mock_registry():
    """Mock registry returning fake model paths."""
    reg = MagicMock(spec=ModelRegistry)
    reg.get_production_model.return_value = ("vProd", "path/to/prod")
    reg.get_challenger_model.return_value = ("vStage", "path/to/stage")
    return reg

class MockXGBClassifier:
    """Mock XGB to avoid loading real files."""
    def __init__(self):
        self.version = "unknown"
        
    def load_model(self, path):
        pass
        
    def predict(self, X):
        # Return generic prediction based on shape
        return np.zeros(X.shape[0])
        
    def predict_proba(self, X):
        return np.zeros((X.shape[0], 2))

@pytest.fixture
def mock_server(mock_registry):
    """Server with mocked models."""
    config = ServingConfig(canary_percentage=0.5)
    server = ModelServer(registry=mock_registry, config=config)
    
    # Mock internal load logic to skip disk I/O
    server._load_xgboost_model = MagicMock(return_value=MockXGBClassifier())
    
    # Re-trigger load to use mock
    server._load_models()
    
    # Set versions explicitly for logic test
    server.champion_model.version = "vProd"
    if server.challenger_model:
        server.challenger_model.version = "vStage"
        
    return server

@pytest.mark.asyncio
async def test_serving_response_structure(mock_server):
    """Ensure response follows schema."""
    features = [{'f1': 1.0, 'f2': 0.5}]
    
    resp = await mock_server.predict_batch(features)
    
    assert 'predictions' in resp
    assert 'model_version' in resp
    assert 'serving_mode' in resp
    assert resp['serving_mode'] in [ServingMode.CHAMPION_ONLY, ServingMode.CANARY]

@pytest.mark.asyncio
async def test_canary_routing(mock_registry):
    """Test 100% canary routing."""
    config = ServingConfig(canary_percentage=1.0) # Force Canary
    server = ModelServer(registry=mock_registry, config=config)
    server._load_xgboost_model = MagicMock(return_value=MockXGBClassifier())
    server._load_models()
    server.champion_model.version = "vProd"
    server.challenger_model.version = "vStage"
    
    features = [{'f1': 1.0}]
    resp = await server.predict_batch(features)
    
    assert resp['model_version'] == "vStage"
    assert resp['serving_mode'] == ServingMode.CANARY

@pytest.mark.asyncio
async def test_shadow_mode(mock_registry):
    """Test Shadow mode (returns Champion, logs Challenger)."""
    config = ServingConfig(shadow_mode=True)
    server = ModelServer(registry=mock_registry, config=config)
    server._load_xgboost_model = MagicMock(return_value=MockXGBClassifier())
    server._load_models()
    
    features = [{'f1': 1.0}]
    resp = await server.predict_batch(features)
    
    # Should always return Champion result
    assert resp['model_version'] == server.champion_model.version
    assert resp['serving_mode'] == ServingMode.SHADOW
    
