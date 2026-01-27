
"""
Tests for Model Serving (Canary/Shadow).
"""
import pytest
import numpy as np
from src.model.serving import ModelServer, ServingConfig, ServingMode
from src.model.registry import ModelRegistry
import xgboost as xgb
from unittest.mock import MagicMock, AsyncMock, patch

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
    
    with patch.object(ModelServer, '_load_model_artifact', side_effect=lambda x: MockXGBClassifier()):
        server = ModelServer(registry=mock_registry, config=config)
        
    # Set versions explicitly for logic test (champion should already be set by _load_models)
    # But _load_models sets .version from registry. So should be fine.
    # Just validation:
    # server.champion_model.version = "vProd"
        
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
    
    with patch.object(ModelServer, '_load_model_artifact', side_effect=lambda x: MockXGBClassifier()):
        server = ModelServer(registry=mock_registry, config=config)
    
    features = [{'f1': 1.0}]
    resp = await server.predict_batch(features)
    
    assert resp['model_version'] == "vStage"
    assert resp['serving_mode'] == ServingMode.CANARY

@pytest.mark.asyncio
async def test_shadow_mode(mock_registry):
    """Test Shadow mode (returns Champion, logs Challenger)."""
    config = ServingConfig(shadow_mode=True)
    
    with patch.object(ModelServer, '_load_model_artifact', side_effect=lambda x: MockXGBClassifier()):
        server = ModelServer(registry=mock_registry, config=config)
    
    features = [{'f1': 1.0}]
    resp = await server.predict_batch(features)
    
    # Should always return Champion result
    assert resp['model_version'] == server.champion_model.version
    assert resp['serving_mode'] == ServingMode.SHADOW
    
