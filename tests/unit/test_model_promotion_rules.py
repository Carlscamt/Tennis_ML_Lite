import pytest
from unittest.mock import MagicMock, patch
from src.model.registry import ModelRegistry, ModelVersion

class TestModelPromotionRules:

    @pytest.fixture
    def registry(self, tmp_path):
        """Fixture for registry with a temp directory."""
        return ModelRegistry(root_dir=tmp_path)

    @pytest.fixture
    def dummy_model_path(self, tmp_path):
        """Create a dummy model file and return its path."""
        d = tmp_path / "dummy.bin"
        d.write_text("fake model content")
        return str(d)

    def test_promotion_requires_minimum_auc(self, registry, dummy_model_path):
        """Model must meet minimum AUC threshold to be promoted to Production."""
        weak_version = registry.register_model(
            model_path=dummy_model_path,
            auc=0.60,
            precision=0.7,
            recall=0.7,
            feature_schema_version="1.0",
            training_dataset_size=100
        )
        
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

    def test_promotion_requires_non_negative_sharpe(self, registry, dummy_model_path):
        """Model with negative Sharpe ratio cannot be promoted to Production."""
        version = registry.register_model(
            model_path=dummy_model_path,
            auc=0.85,
            precision=0.8,
            recall=0.8,
            feature_schema_version="1.0",
            training_dataset_size=100,
            sharpe_ratio=-0.5,  # Negative Sharpe
            roi=-0.02,
        )
        
        with pytest.raises(ValueError, match="Sharpe ratio"):
            registry.transition_stage(version, "Production")

    def test_promotion_allows_positive_sharpe(self, registry, dummy_model_path):
        """Model with positive Sharpe ratio can be promoted to Production."""
        version = registry.register_model(
            model_path=dummy_model_path,
            auc=0.85,
            precision=0.8,
            recall=0.8,
            feature_schema_version="1.0",
            training_dataset_size=100,
            sharpe_ratio=0.5,  # Positive Sharpe
            roi=0.08,
        )
        
        # Should succeed
        registry.transition_stage(version, "Production")
        assert registry.get_model_metadata(version)['stage'] == "Production"

    def test_promotion_backward_compatible_no_sharpe(self, registry, dummy_model_path):
        """Models without Sharpe ratio (legacy) can still be promoted if AUC is met."""
        version = registry.register_model(
            model_path=dummy_model_path,
            auc=0.85,
            precision=0.8,
            recall=0.8,
            feature_schema_version="1.0",
            training_dataset_size=100,
            # No sharpe_ratio or roi provided
        )
        
        # Should succeed (backward compatible)
        registry.transition_stage(version, "Production")
        assert registry.get_model_metadata(version)['stage'] == "Production"

    def test_promotion_allows_zero_sharpe(self, registry, dummy_model_path):
        """Model with zero Sharpe ratio can be promoted (break-even is acceptable)."""
        version = registry.register_model(
            model_path=dummy_model_path,
            auc=0.85,
            precision=0.8,
            recall=0.8,
            feature_schema_version="1.0",
            training_dataset_size=100,
            sharpe_ratio=0.0,
            roi=0.0,
        )
        
        registry.transition_stage(version, "Production")
        assert registry.get_model_metadata(version)['stage'] == "Production"

    def test_registry_stores_betting_metrics(self, registry, dummy_model_path):
        """Verify that betting metrics are stored in registry."""
        version = registry.register_model(
            model_path=dummy_model_path,
            auc=0.85,
            precision=0.8,
            recall=0.8,
            feature_schema_version="1.0",
            training_dataset_size=100,
            log_loss=0.45,
            roi=0.12,
            sharpe_ratio=0.65,
            max_drawdown=0.15,
            cv_folds=5,
        )
        
        meta = registry.get_model_metadata(version)
        
        assert meta['log_loss'] == 0.45
        assert meta['roi'] == 0.12
        assert meta['sharpe_ratio'] == 0.65
        assert meta['max_drawdown'] == 0.15
        assert meta['cv_folds'] == 5
