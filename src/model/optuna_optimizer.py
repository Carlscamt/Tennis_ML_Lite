"""
Bayesian hyperparameter optimization for XGBoost using Optuna.

Optimizes for a composite objective: log_loss + variance_penalty * std(ROI).
Uses time-series CV with gap enforcement to prevent data leakage.
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import log_loss


logger = logging.getLogger(__name__)

# Check for optuna availability
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed. Run: pip install optuna")


# Default search space for XGBoost
DEFAULT_SEARCH_SPACE = {
    "max_depth": (3, 10),
    "min_child_weight": (1, 10),
    "gamma": (0.0, 0.5),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "reg_alpha": (0.0, 10.0),
    "reg_lambda": (0.0, 10.0),
    "learning_rate": (0.01, 0.3),
    "n_estimators": (100, 500),
}


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    best_params: dict[str, Any]
    best_score: float
    n_trials: int
    study_name: str
    objective_type: str
    fold_scores: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": self.n_trials,
            "study_name": self.study_name,
            "objective_type": self.objective_type,
            "fold_scores": self.fold_scores,
        }

    def save(self, path: Path) -> None:
        """Save result to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Optimization result saved to {path}")


@dataclass
class OptunaOptimizer:
    """
    Bayesian hyperparameter optimization for XGBoost using Optuna.
    
    Features:
    - Composite objective: log_loss + variance_penalty * std(ROI)
    - Time-series CV with gap enforcement
    - Early pruning of bad trials
    - Saves best params to JSON
    
    Example:
        optimizer = OptunaOptimizer(feature_cols=features)
        result = optimizer.optimize(df, n_trials=50)
        print(result.best_params)
    """
    feature_cols: list[str]
    target_col: str = "player_won"
    n_splits: int = 5
    gap_days: int = 7
    min_train_size: int = 5000

    # Optimization settings
    variance_penalty: float = 0.1
    objective_type: str = "composite"  # "composite", "log_loss", "roi"

    # Search space (can be customized)
    search_space: dict[str, tuple[float, float]] = field(default_factory=lambda: DEFAULT_SEARCH_SPACE.copy())

    def __post_init__(self):
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for hyperparameter optimization. "
                "Install it with: pip install optuna"
            )

    def optimize(
        self,
        df: pl.DataFrame,
        n_trials: int = 50,
        timeout: int | None = None,
        study_name: str = "tennis_xgb",
        storage: str | None = None,
        n_jobs: int = 1,
        save_path: Path | None = None,
    ) -> OptimizationResult:
        """
        Run Bayesian hyperparameter optimization.
        
        Args:
            df: Training DataFrame with features, target, odds, and date columns
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds (None = no limit)
            study_name: Name for the Optuna study
            storage: Database URL for persistent storage (e.g., "sqlite:///optuna.db")
            n_jobs: Number of parallel trials
            save_path: Path to save best params JSON
            
        Returns:
            OptimizationResult with best params and scores
        """
        from src.model.cv import TimeSeriesBettingCV
        from src.model.metrics import BettingMetrics
        from src.model.trainer import ModelTrainer

        # Ensure sorted by date
        if "match_date" in df.columns:
            df = df.sort("match_date")

        # Create study with TPE sampler and median pruner
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",  # Lower is better for composite score
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        # Create CV splitter
        cv = TimeSeriesBettingCV(
            n_splits=self.n_splits,
            gap_days=self.gap_days,
            min_train_size=self.min_train_size,
        )

        # Metrics calculator for ROI
        metrics_calc = BettingMetrics(
            kelly_fraction=0.25,
            min_edge=0.05,
            min_odds=1.20,
            max_odds=5.00,
        )

        # Define objective function
        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            params = self._sample_params(trial)

            fold_log_losses = []
            fold_rois = []

            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(df, "match_date")):
                # Split data
                train_df = df[train_idx.tolist()]
                test_df = df[test_idx.tolist()]

                # Train model
                trainer = ModelTrainer(params=params, calibrate=False)

                X_train = train_df.select(self.feature_cols).to_numpy()
                y_train = train_df.select(self.target_col).to_numpy().flatten()
                X_test = test_df.select(self.feature_cols).to_numpy()
                y_test = test_df.select(self.target_col).to_numpy().flatten()

                trainer.train(train_df, self.feature_cols, self.target_col)

                # Predict
                y_prob = trainer.predict_proba(test_df)

                # Calculate log loss
                ll = log_loss(y_test, y_prob, labels=[0, 1])
                fold_log_losses.append(ll)

                # Calculate ROI if odds available
                if "odds_player" in test_df.columns:
                    odds = test_df.select("odds_player").to_numpy().flatten()
                    result = metrics_calc.calculate(y_test, y_prob, odds)
                    fold_rois.append(result.roi)

                # Report intermediate value for pruning
                trial.report(np.mean(fold_log_losses), fold_idx)

                if trial.should_prune():
                    raise optuna.TrialPruned()

            # Compute objective
            mean_ll = np.mean(fold_log_losses)

            if self.objective_type == "log_loss":
                score = mean_ll
            elif self.objective_type == "roi":
                score = -np.mean(fold_rois) if fold_rois else mean_ll
            else:  # composite
                if fold_rois:
                    roi_std = np.std(fold_rois)
                    score = mean_ll + self.variance_penalty * roi_std
                else:
                    score = mean_ll

            # Log trial result
            logger.info(
                f"Trial {trial.number}: score={score:.4f}, "
                f"log_loss={mean_ll:.4f}, roi_std={np.std(fold_rois):.4f if fold_rois else 0}"
            )

            return score

        # Run optimization
        logger.info(
            f"Starting Optuna optimization: {n_trials} trials, "
            f"{self.n_splits}-fold CV, objective={self.objective_type}"
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True,
        )

        # Get best result
        best_trial = study.best_trial
        best_params = best_trial.params

        # Add fixed XGBoost params
        best_params["objective"] = "binary:logistic"
        best_params["eval_metric"] = "logloss"
        best_params["use_label_encoder"] = False
        best_params["verbosity"] = 0

        result = OptimizationResult(
            best_params=best_params,
            best_score=best_trial.value,
            n_trials=len(study.trials),
            study_name=study_name,
            objective_type=self.objective_type,
        )

        # Save if path provided
        if save_path:
            result.save(save_path)

        logger.info(f"Optimization complete. Best score: {result.best_score:.4f}")
        logger.info(f"Best params: {json.dumps(best_params, indent=2)}")

        return result

    def _sample_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Sample hyperparameters from search space."""
        params = {}

        for name, (low, high) in self.search_space.items():
            if name == "n_estimators" or name == "max_depth" or name == "min_child_weight":
                params[name] = trial.suggest_int(name, int(low), int(high))
            else:
                # Float parameters
                params[name] = trial.suggest_float(name, low, high)

        # Fixed params
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
        params["use_label_encoder"] = False
        params["verbosity"] = 0

        return params


def load_best_params(path: Path) -> dict[str, Any]:
    """Load best params from a saved optimization result."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No optimization result found at {path}")

    with open(path) as f:
        data = json.load(f)

    return data.get("best_params", {})
