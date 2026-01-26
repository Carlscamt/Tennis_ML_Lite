import pytest
from unittest.mock import MagicMock, patch
from src.model.registry import ModelRegistry, ModelVersion

class TestModelPromotionRules:

    @pytest.fixture
    def registry(self, tmp_path):
        """Fixture for registry with a temp directory."""
        return ModelRegistry(root_dir=tmp_path)

    def test_promotion_requires_minimum_auc(self, registry):
        """Model must meet minimum AUC threshold to be promoted to Production."""
        # Register a weak model
        weak_version = registry.register_model(
            model_path="dummy.bin", # Won't actually copy in this mock context unless we mock shutil? 
            # We need to create a dummy file for shutil.copy2 to work
            auc=0.60,
            precision=0.7,
            recall=0.7,
            feature_schema_version="1.0",
            training_dataset_size=100
        )
        
        # Try to promote to Production (Should fail if rules enforced)
        # Note: Current code DOES NOT enforce this, so this test is EXPECTED TO FAIL 
        # until we implement the logic.
        with pytest.raises(ValueError, match="Minimum AUC"):
            registry.transition_stage(weak_version, "Production")

    def test_production_demotion_on_new_promotion(self, registry, dummy_model_path):
        """Promoting a new model to Production should demote the old one."""
        # 1. Register and promote V1
        v1 = registry.register_model(dummy_model_path, 0.85, 0.8, 0.8, "1.0", 100, stage="Experimental")
        registry.transition_stage(v1, "Production")
        
        assert registry.get_model_metadata(v1)['stage'] == "Production"
        
        # 2. Register and promote V2
        v2 = registry.register_model(dummy_model_path, 0.90, 0.9, 0.9, "1.0", 100, stage="Experimental")
        registry.transition_stage(v2, "Production")
        
        # 3. Verify V1 is archived and V2 is Production
        assert registry.get_model_metadata(v1)['stage'] == "Archived"
        assert registry.get_model_metadata(v2)['stage'] == "Production"

    @pytest.fixture
    def dummy_model_path(self, tmp_path):
        """Create a dummy model file and return its path."""
        d = tmp_path / "dummy.bin"
        d.write_text("fake model content")
        return str(d)

    def test_promotion_requires_minimum_auc(self, registry, dummy_model_path):
        """Model must meet minimum AUC threshold to be promoted to Production."""
        # Register a weak model
        weak_version = registry.register_model(
            model_path=dummy_model_path,
            auc=0.60,
            precision=0.7,
            recall=0.7,
            feature_schema_version="1.0",
            training_dataset_size=100
        )
