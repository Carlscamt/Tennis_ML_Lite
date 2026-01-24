
"""
Tests for Model Registry.
"""
import pytest
import shutil
import os
from pathlib import Path
from src.model.registry import ModelRegistry

TRAINING_METADATA = {
    "auc": 0.85,
    "precision": 0.82,
    "recall": 0.88,
    "feature_schema_version": "1.0",
    "training_dataset_size": 1000
}

@pytest.fixture
def temp_registry_dir(tmp_path):
    """Fixture for temporary registry persistence."""
    # We pass root_dir to ModelRegistry to use tmp_path
    reg_dir = tmp_path / "model_registry_test"
    reg_dir.mkdir()
    return reg_dir

def test_model_lifecycle(temp_registry_dir, tmp_path):
    registry = ModelRegistry(root_dir=temp_registry_dir)
    
    # Create dummy model file
    model_path = tmp_path / "dummy_model.bin"
    model_path.write_text("dummy content")
    
    # 1. Register
    version = registry.register_model(
        str(model_path), 
        notes="Test model",
        **TRAINING_METADATA
    )
    assert version == "v1.0.0"
    
    meta = registry.get_model_metadata(version)
    assert meta['stage'] == "Experimental"
    
    # 2. Transition
    registry.transition_stage(version, "Staging")
    assert registry.get_model_metadata(version)['stage'] == "Staging"
    
    registry.transition_stage(version, "Production")
    assert registry.get_model_metadata(version)['stage'] == "Production"
    
    # 3. Get Production
    prod_ver, prod_path = registry.get_production_model()
    assert prod_ver == version
    assert Path(prod_path).exists()

def test_semantic_versioning(temp_registry_dir, tmp_path):
    registry = ModelRegistry(root_dir=temp_registry_dir)
    model_path = tmp_path / "dummy.bin"
    model_path.write_text("x")
    
    # v1.0.0
    v1 = registry.register_model(str(model_path), **TRAINING_METADATA)
    assert v1 == "v1.0.0"
    
    # v1.0.1 (Experimental patch)
    v2 = registry.register_model(str(model_path), **TRAINING_METADATA)
    assert v2 == "v1.0.1"
    
    # Promote v1 to Production -> Next should be v2.0.0? 
    # Logic: if stage=Production passed to register, bump major. 
    # But usually we promote AFTER register.
    # The register_model logic bumps based on *passed* stage. 
    # If I register as Experimental (default), it bumps patch.
    
    # Let's try registering straight as Staging
    v3 = registry.register_model(str(model_path), stage="Staging", **TRAINING_METADATA)
    # Latest was v1.0.1. Staging bumps minor -> v1.1.0
    assert v3 == "v1.1.0"
