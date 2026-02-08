"""
Unit tests for Optuna hyperparameter optimizer.
"""
import pytest
import numpy as np
import polars as pl

# Check if optuna is available
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class TestOptunaOptimizerImport:
    """Test import behavior."""
    
    def test_optuna_not_available_graceful(self, monkeypatch):
        """Test graceful handling when optuna not installed."""
        # This test just verifies the module structure
        # Actual optuna availability depends on installation
        from src.model import optuna_optimizer
        
        # Check module has expected components
        assert hasattr(optuna_optimizer, "OptunaOptimizer")
        assert hasattr(optuna_optimizer, "OptimizationResult")
        assert hasattr(optuna_optimizer, "DEFAULT_SEARCH_SPACE")


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        from src.model.optuna_optimizer import OptimizationResult
        
        result = OptimizationResult(
            best_params={"max_depth": 5, "learning_rate": 0.1},
            best_score=0.45,
            n_trials=10,
            study_name="test",
            objective_type="composite",
        )
        
        d = result.to_dict()
        
        assert d["best_params"]["max_depth"] == 5
        assert d["best_score"] == 0.45
        assert d["n_trials"] == 10
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading result."""
        from src.model.optuna_optimizer import OptimizationResult, load_best_params
        
        result = OptimizationResult(
            best_params={"max_depth": 6, "learning_rate": 0.05},
            best_score=0.42,
            n_trials=20,
            study_name="test",
            objective_type="log_loss",
        )
        
        save_path = tmp_path / "best_params.json"
        result.save(save_path)
        
        assert save_path.exists()
        
        loaded = load_best_params(save_path)
        assert loaded["max_depth"] == 6
        assert loaded["learning_rate"] == 0.05


class TestDefaultSearchSpace:
    """Tests for default search space."""
    
    def test_has_all_key_params(self):
        """Test that search space includes all important XGBoost params."""
        from src.model.optuna_optimizer import DEFAULT_SEARCH_SPACE
        
        expected_params = [
            "max_depth",
            "min_child_weight",
            "gamma",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "learning_rate",
            "n_estimators",
        ]
        
        for param in expected_params:
            assert param in DEFAULT_SEARCH_SPACE, f"Missing: {param}"
    
    def test_ranges_are_valid(self):
        """Test that all ranges are (low, high) tuples."""
        from src.model.optuna_optimizer import DEFAULT_SEARCH_SPACE
        
        for param, (low, high) in DEFAULT_SEARCH_SPACE.items():
            assert low < high, f"{param}: low >= high"
            assert low >= 0, f"{param}: negative low"


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestOptunaOptimizerWithOptuna:
    """Tests that require Optuna to be installed."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        n = 500
        
        return pl.DataFrame({
            "feature_1": np.random.randn(n),
            "feature_2": np.random.randn(n),
            "feature_3": np.random.randn(n),
            "player_won": np.random.randint(0, 2, n),
            "odds_player": np.random.uniform(1.5, 3.0, n),
            "odds_opponent": np.random.uniform(1.5, 3.0, n),
            "match_date": pl.date_range(
                pl.date(2022, 1, 1),
                pl.date(2022, 1, 1) + pl.duration(days=n-1),
                eager=True
            ),
        })
    
    def test_optimizer_initialization(self, sample_df):
        """Test optimizer can be initialized."""
        from src.model.optuna_optimizer import OptunaOptimizer
        
        optimizer = OptunaOptimizer(
            feature_cols=["feature_1", "feature_2", "feature_3"],
            target_col="player_won",
        )
        
        assert optimizer.n_splits == 5
        assert optimizer.objective_type == "composite"
    
    def test_sample_params(self, sample_df):
        """Test parameter sampling."""
        import optuna
        from src.model.optuna_optimizer import OptunaOptimizer
        
        optimizer = OptunaOptimizer(
            feature_cols=["feature_1", "feature_2", "feature_3"],
        )
        
        study = optuna.create_study()
        trial = study.ask()
        
        params = optimizer._sample_params(trial)
        
        assert "max_depth" in params
        assert "learning_rate" in params
        assert "objective" in params
        assert params["objective"] == "binary:logistic"
