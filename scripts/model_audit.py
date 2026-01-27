"""
Comprehensive Model Audit with Multiple Testing Styles

Performs multi-faceted testing of the Tennis Prediction Model:
1. Statistical Performance Tests
2. Calibration Tests  
3. Temporal Stability Tests
4. Robustness Tests
5. Feature Importance Analysis
6. Bias/Fairness Tests
7. Adversarial Edge Case Tests

Usage:
    python scripts/model_audit.py
    python scripts/model_audit.py --save-plots
    python scripts/model_audit.py --output-dir results/audit
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import json
import polars as pl
import numpy as np
from datetime import datetime, date
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import logging


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# Sklearn imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample

from config import PROCESSED_DATA_DIR, MODELS_DIR, DATA_DIR
from src.model import Predictor, ModelRegistry
from src.utils import setup_logging

logger = setup_logging()

# Define results directory
RESULTS_DIR = ROOT / "results"


@dataclass
class AuditResult:
    """Container for audit section results."""
    section: str
    passed: bool
    score: float
    metrics: Dict
    warnings: List[str]
    details: Optional[Dict] = None


class ModelAuditor:
    """
    Comprehensive model auditor implementing 7 testing styles.
    """
    
    def __init__(
        self,
        model_path: Path,
        data_path: Path,
        test_cutoff: date = date(2025, 1, 1),
        save_plots: bool = False,
        output_dir: Optional[Path] = None
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.test_cutoff = test_cutoff
        self.save_plots = save_plots
        self.output_dir = output_dir or RESULTS_DIR
        
        self.predictor = Predictor(model_path)
        self.results: List[AuditResult] = []
        
        # Load and prepare data
        self._load_data()
    
    def _load_data(self):
        """Load and split data temporally."""
        logger.info(f"Loading data from {self.data_path}")
        self.df = pl.read_parquet(self.data_path)
        
        # Convert cutoff to timestamp
        cutoff_ts = int(datetime.combine(self.test_cutoff, datetime.min.time()).timestamp())
        
        # Split train/test
        self.train_df = self.df.filter(pl.col("start_timestamp") < cutoff_ts)
        self.test_df = self.df.filter(pl.col("start_timestamp") >= cutoff_ts)
        
        # Filter to matches with odds for fair evaluation
        self.test_df = self.test_df.filter(
            pl.col("odds_player").is_not_null() & 
            pl.col("player_won").is_not_null()
        )
        
        logger.info(f"Train: {len(self.train_df):,} | Test: {len(self.test_df):,} matches")
        
        # Generate predictions on test set
        self.test_df = self.predictor.predict_with_value(self.test_df)
        
        # Extract arrays for sklearn
        self.y_true = self.test_df["player_won"].to_numpy().astype(int)
        self.y_prob = self.test_df["model_prob"].to_numpy()
        self.y_pred = (self.y_prob >= 0.5).astype(int)
    
    def run_full_audit(self) -> Dict:
        """Run all audit tests and generate report."""
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE MODEL AUDIT")
        logger.info("=" * 60)
        
        # Run all test suites - LEAKAGE FIRST (most critical)
        self._test_data_leakage()
        self._test_statistical_performance()
        self._test_calibration()
        self._test_temporal_stability()
        self._test_robustness()
        self._test_feature_importance()
        self._test_bias_fairness()
        self._test_adversarial_edge_cases()
        
        # Generate summary
        summary = self._generate_summary()
        
        # Save reports
        self._save_reports(summary)
        
        return summary
    
    # =========================================================================
    # 0. DATA LEAKAGE DETECTION (CRITICAL)
    # =========================================================================
    def _test_data_leakage(self):
        """
        CRITICAL: Detect potential data leakage in features.
        
        Checks for features that may contain current-match outcome data
        instead of properly shifted historical statistics.
        """
        logger.info("\n[0/8] DATA LEAKAGE DETECTION (CRITICAL)")
        logger.info("-" * 40)
        
        warnings = []
        leakage_details = {}
        
        # Get feature columns from model
        trainer = self.predictor.trainer
        feature_cols = trainer.feature_columns
        
        # Features that are SAFE (pre-match or properly shifted)
        safe_patterns = [
            "win_rate_",       # Rolling win rates (shifted)
            "h2h_",            # Head-to-head (shifted)
            "surface_win_rate", # Surface-specific (shifted)
            "days_since",      # Pre-match known
            "odds_",           # Pre-match known
            "implied_prob",    # Derived from pre-match odds
            "odds_ratio",      # Derived from pre-match odds
            "is_underdog",     # Derived from pre-match odds
            "round_num",       # Pre-match known
            "_avg_",           # Shifted rolling averages (properly computed)
            "_pct_avg_",       # Shifted rolling percentages
            "_ratio_avg_",     # Shifted rolling ratios
        ]
        
        # Features that LEAK current match outcome
        leaky_patterns = [
            # Post-match statistics from current match (NOT shifted)
            ("_aces", "Current match aces (post-match data)"),
            ("_doublefaults", "Current match double faults (post-match data)"),
            ("_firstserveaccuracy", "Current match serve accuracy (post-match data)"),
            ("_secondserveaccuracy", "Current match serve accuracy (post-match data)"),
            ("_firstservepointsaccuracy", "Current match serve points (post-match data)"),
            ("_secondservepointsaccuracy", "Current match serve points (post-match data)"),
            ("_servicegamestotal", "Current match service games (post-match data)"),
            ("_breakpointssaved", "Current match break points (post-match data)"),
            ("_pointstotal", "Current match points (post-match data)"),
            ("_servicepointsscored", "Current match service points (post-match data)"),
            ("_receiverpointsscored", "Current match return points (post-match data)"),
            ("_maxpointsinrow", "Current match momentum (post-match data)"),
            ("_gameswon", "CRITICAL: Games won in current match (direct leak!)"),
            ("_servicegameswon", "CRITICAL: Service games won in current match (direct leak!)"),
            ("_maxgamesinrow", "Current match momentum (post-match data)"),
            ("_firstreturnpoints", "Current match return points (post-match data)"),
            ("_secondreturnpoints", "Current match return points (post-match data)"),
            ("_breakpointsscored", "Current match break points (post-match data)"),
            ("_tiebreaks", "Current match tiebreaks (post-match data)"),
            ("_winnerstotal", "Current match winners (post-match data)"),
            ("_forehandwinners", "Current match winners (post-match data)"),
            ("_backhandwinners", "Current match winners (post-match data)"),
            ("_volleywinners", "Current match winners (post-match data)"),
            ("_groundstrokewinners", "Current match winners (post-match data)"),
            ("_lobwinners", "Current match winners (post-match data)"),
            ("_overheadwinners", "Current match winners (post-match data)"),
            ("_dropshotwinners", "Current match winners (post-match data)"),
            ("_returnwinners", "Current match winners (post-match data)"),
            ("_errorstotal", "Current match errors (post-match data)"),
            ("_forehanderrors", "Current match errors (post-match data)"),
            ("_backhanderrors", "Current match errors (post-match data)"),
            ("_groundstrokeerrors", "Current match errors (post-match data)"),
            ("_overheadstrokeerrors", "Current match errors (post-match data)"),
            ("_returnerrors", "Current match errors (post-match data)"),
            ("_unforcederrorstotal", "Current match unforced errors (post-match data)"),
            ("_forehandunforcederrors", "Current match unforced errors (post-match data)"),
            ("_backhandunforcederrors", "Current match unforced errors (post-match data)"),
            ("_volleyunforcederrors", "Current match unforced errors (post-match data)"),
            ("_groundstrokeunforcederrors", "Current match unforced errors (post-match data)"),
            ("_lobunforcederrors", "Current match unforced errors (post-match data)"),
            ("_dropshotunforcederrors", "Current match unforced errors (post-match data)"),
        ]
        
        # Check each feature
        identified_leaks = []
        safe_features = []
        unknown_features = []
        
        for feat in feature_cols:
            feat_lower = feat.lower()
            
            # Check if it's a known safe feature
            is_safe = any(pattern in feat_lower for pattern in safe_patterns)
            if is_safe:
                safe_features.append(feat)
                continue
            
            # Check if it matches leaky patterns
            leaked = False
            for pattern, reason in leaky_patterns:
                if pattern in feat_lower:
                    identified_leaks.append({"feature": feat, "reason": reason})
                    leaked = True
                    break
            
            if not leaked:
                unknown_features.append(feat)
        
        leakage_details = {
            "total_features": len(feature_cols),
            "safe_features": len(safe_features),
            "leaky_features": len(identified_leaks),
            "unknown_features": len(unknown_features),
            "leaky_feature_list": identified_leaks[:20],  # Show first 20
            "safe_feature_list": safe_features,
        }
        
        # Determine severity
        leakage_pct = len(identified_leaks) / len(feature_cols) * 100
        
        if len(identified_leaks) > 0:
            warnings.append(
                f"CRITICAL: {len(identified_leaks)} features ({leakage_pct:.1f}%) contain "
                f"POST-MATCH data (games won, aces, etc.) - THIS IS DATA LEAKAGE!"
            )
            
            # Show top 5 worst offenders
            critical_leaks = [l for l in identified_leaks if "CRITICAL" in l["reason"]]
            if critical_leaks:
                warnings.append(
                    f"Most severe: {', '.join([l['feature'] for l in critical_leaks[:5]])} "
                    f"- these directly reveal match outcome"
                )
        
        # Log results
        if identified_leaks:
            logger.info(f"  [CRITICAL] Found {len(identified_leaks)} features with data leakage!")
            logger.info(f"  Examples of leaky features:")
            for leak in identified_leaks[:5]:
                logger.info(f"    - {leak['feature']}: {leak['reason']}")
        else:
            logger.info(f"  [OK] No obvious data leakage detected")
        
        logger.info(f"  Safe features: {len(safe_features)} | Unknown: {len(unknown_features)}")
        
        metrics = {
            "leakage_detected": len(identified_leaks) > 0,
            "leaky_feature_count": len(identified_leaks),
            "leakage_percentage": round(leakage_pct, 1),
            "safe_feature_count": len(safe_features),
            "details": leakage_details,
        }
        
        # Score: 0 if ANY leakage, 100 if none
        score = 0 if len(identified_leaks) > 0 else 100
        passed = len(identified_leaks) == 0
        
        self.results.append(AuditResult(
            section="Data Leakage",
            passed=passed,
            score=score,
            metrics=metrics,
            warnings=warnings
        ))
    
    # =========================================================================
    # 1. STATISTICAL PERFORMANCE TESTS
    # =========================================================================
    def _test_statistical_performance(self):
        """Test classification and probability metrics."""
        logger.info("\n[1/7] STATISTICAL PERFORMANCE TESTS")
        logger.info("-" * 40)
        
        warnings = []
        
        # Classification metrics
        accuracy = accuracy_score(self.y_true, self.y_pred)
        precision = precision_score(self.y_true, self.y_pred, zero_division=0)
        recall = recall_score(self.y_true, self.y_pred, zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, zero_division=0)
        
        # Probability metrics
        auc_roc = roc_auc_score(self.y_true, self.y_prob)
        auc_pr = average_precision_score(self.y_true, self.y_prob)
        brier = brier_score_loss(self.y_true, self.y_prob)
        logloss = log_loss(self.y_true, self.y_prob)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Performance checks
        if accuracy < 0.52:
            warnings.append(f"Accuracy ({accuracy:.1%}) barely above baseline (50%)")
        if auc_roc < 0.55:
            warnings.append(f"AUC-ROC ({auc_roc:.3f}) indicates weak discrimination")
        if brier > 0.25:
            warnings.append(f"Brier Score ({brier:.3f}) indicates poor probability estimates")
        
        metrics = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "auc_roc": round(auc_roc, 4),
            "auc_pr": round(auc_pr, 4),
            "brier_score": round(brier, 4),
            "log_loss": round(logloss, 4),
            "confusion_matrix": {
                "true_neg": int(tn), "false_pos": int(fp),
                "false_neg": int(fn), "true_pos": int(tp)
            }
        }
        
        # Score based on AUC-ROC (main discriminative metric)
        score = min(100, max(0, (auc_roc - 0.5) * 200))
        passed = auc_roc >= 0.55 and accuracy >= 0.52
        
        logger.info(f"  Accuracy: {accuracy:.1%} | AUC-ROC: {auc_roc:.3f} | Brier: {brier:.3f}")
        
        self.results.append(AuditResult(
            section="Statistical Performance",
            passed=passed,
            score=score,
            metrics=metrics,
            warnings=warnings
        ))
    
    # =========================================================================
    # 2. CALIBRATION TESTS
    # =========================================================================
    def _test_calibration(self):
        """Test probability calibration quality."""
        logger.info("\n[2/7] CALIBRATION TESTS")
        logger.info("-" * 40)
        
        warnings = []
        
        # Calibration curve
        n_bins = 10
        prob_true, prob_pred = calibration_curve(
            self.y_true, self.y_prob, n_bins=n_bins, strategy='uniform'
        )
        
        # Expected Calibration Error (ECE)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0
        mce = 0
        bin_sizes = []
        
        for i in range(n_bins):
            mask = (self.y_prob >= bin_edges[i]) & (self.y_prob < bin_edges[i + 1])
            bin_size = mask.sum()
            bin_sizes.append(int(bin_size))
            
            if bin_size > 0:
                bin_acc = self.y_true[mask].mean()
                bin_conf = self.y_prob[mask].mean()
                bin_error = abs(bin_acc - bin_conf)
                ece += (bin_size / len(self.y_true)) * bin_error
                mce = max(mce, bin_error)
        
        # Calibration checks
        if ece > 0.05:
            warnings.append(f"ECE ({ece:.3f}) exceeds recommended threshold (0.05)")
        if mce > 0.15:
            warnings.append(f"MCE ({mce:.3f}) shows severe miscalibration in some bins")
        
        # Check for overconfidence on wrong predictions
        wrong_preds = self.y_pred != self.y_true
        if wrong_preds.sum() > 0:
            wrong_conf = self.y_prob[wrong_preds]
            high_conf_wrong = (np.maximum(wrong_conf, 1 - wrong_conf) > 0.7).mean()
            if high_conf_wrong > 0.1:
                warnings.append(f"{high_conf_wrong:.1%} of wrong predictions had >70% confidence")
        
        metrics = {
            "expected_calibration_error": round(ece, 4),
            "maximum_calibration_error": round(mce, 4),
            "calibration_curve": {
                "prob_true": [round(p, 4) for p in prob_true],
                "prob_pred": [round(p, 4) for p in prob_pred],
            },
            "bin_sizes": bin_sizes
        }
        
        # Score based on ECE (lower is better)
        score = max(0, 100 - ece * 1000)
        passed = ece <= 0.05
        
        logger.info(f"  ECE: {ece:.4f} | MCE: {mce:.4f}")
        
        self.results.append(AuditResult(
            section="Calibration",
            passed=passed,
            score=score,
            metrics=metrics,
            warnings=warnings
        ))
        
        # Save calibration plot
        if self.save_plots:
            self._plot_calibration(prob_true, prob_pred)
    
    def _plot_calibration(self, prob_true, prob_pred):
        """Generate and save calibration curve plot."""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
            ax.plot(prob_pred, prob_true, 's-', label='Model')
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title('Calibration Curve (Reliability Diagram)')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            
            self.output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.output_dir / 'calibration_curve.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"  Saved calibration_curve.png")
        except ImportError:
            logger.warning("  matplotlib not available for plotting")
    
    # =========================================================================
    # 3. TEMPORAL STABILITY TESTS
    # =========================================================================
    def _test_temporal_stability(self):
        """Test model performance over time."""
        logger.info("\n[3/7] TEMPORAL STABILITY TESTS")
        logger.info("-" * 40)
        
        warnings = []
        
        # Add month column for grouping
        test_with_month = self.test_df.with_columns([
            pl.from_epoch("start_timestamp").dt.strftime("%Y-%m").alias("month")
        ])
        
        monthly_stats = []
        
        for month_data in test_with_month.partition_by("month", maintain_order=True):
            if len(month_data) < 10:
                continue
            
            month = month_data["month"][0]
            y_true = month_data["player_won"].to_numpy().astype(int)
            y_prob = month_data["model_prob"].to_numpy()
            y_pred = (y_prob >= 0.5).astype(int)
            
            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
            
            monthly_stats.append({
                "month": month,
                "n_matches": len(month_data),
                "accuracy": round(acc, 4),
                "auc_roc": round(auc, 4)
            })
            
            logger.info(f"  {month}: n={len(month_data)}, Acc={acc:.1%}, AUC={auc:.3f}")
        
        # Check for degradation
        if len(monthly_stats) >= 3:
            recent_acc = np.mean([s["accuracy"] for s in monthly_stats[-3:]])
            early_acc = np.mean([s["accuracy"] for s in monthly_stats[:3]])
            
            if recent_acc < early_acc - 0.05:
                warnings.append(f"Performance degradation detected: {early_acc:.1%} → {recent_acc:.1%}")
        
        # Variance check
        accs = [s["accuracy"] for s in monthly_stats]
        if len(accs) > 1:
            acc_std = np.std(accs)
            if acc_std > 0.08:
                warnings.append(f"High monthly variance in accuracy (std={acc_std:.3f})")
        
        metrics = {
            "monthly_performance": monthly_stats,
            "n_months": len(monthly_stats),
            "accuracy_mean": round(np.mean(accs), 4) if accs else 0,
            "accuracy_std": round(np.std(accs), 4) if len(accs) > 1 else 0,
        }
        
        # Score based on stability
        score = max(0, 100 - (np.std(accs) * 500 if len(accs) > 1 else 0))
        passed = len(warnings) == 0
        
        self.results.append(AuditResult(
            section="Temporal Stability",
            passed=passed,
            score=score,
            metrics=metrics,
            warnings=warnings
        ))
        
        if self.save_plots:
            self._plot_temporal(monthly_stats)
    
    def _plot_temporal(self, monthly_stats):
        """Generate temporal stability plot."""
        try:
            import matplotlib.pyplot as plt
            
            if not monthly_stats:
                return
            
            months = [s["month"] for s in monthly_stats]
            accs = [s["accuracy"] for s in monthly_stats]
            aucs = [s["auc_roc"] for s in monthly_stats]
            
            fig, ax1 = plt.subplots(figsize=(10, 5))
            
            ax1.plot(months, accs, 'b-o', label='Accuracy')
            ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Accuracy', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.set_ylim(0.4, 0.8)
            
            ax2 = ax1.twinx()
            ax2.plot(months, aucs, 'r-s', label='AUC-ROC')
            ax2.set_ylabel('AUC-ROC', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.set_ylim(0.4, 0.8)
            
            plt.title('Model Performance Over Time')
            plt.xticks(rotation=45)
            fig.tight_layout()
            
            self.output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.output_dir / 'temporal_stability.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"  Saved temporal_stability.png")
        except ImportError:
            logger.warning("  matplotlib not available for plotting")
    
    # =========================================================================
    # 4. ROBUSTNESS TESTS
    # =========================================================================
    def _test_robustness(self):
        """Test model robustness via bootstrap and perturbation."""
        logger.info("\n[4/7] ROBUSTNESS TESTS")
        logger.info("-" * 40)
        
        warnings = []
        
        # Bootstrap confidence intervals
        n_bootstrap = 100
        bootstrap_accs = []
        bootstrap_aucs = []
        
        for i in range(n_bootstrap):
            indices = resample(range(len(self.y_true)), random_state=i)
            y_t = self.y_true[indices]
            y_p = self.y_prob[indices]
            y_pred = (y_p >= 0.5).astype(int)
            
            bootstrap_accs.append(accuracy_score(y_t, y_pred))
            if len(np.unique(y_t)) > 1:
                bootstrap_aucs.append(roc_auc_score(y_t, y_p))
        
        acc_ci = (np.percentile(bootstrap_accs, 2.5), np.percentile(bootstrap_accs, 97.5))
        auc_ci = (np.percentile(bootstrap_aucs, 2.5), np.percentile(bootstrap_aucs, 97.5))
        
        logger.info(f"  Accuracy 95% CI: [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]")
        logger.info(f"  AUC-ROC 95% CI: [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")
        
        # Check CI width
        acc_ci_width = acc_ci[1] - acc_ci[0]
        if acc_ci_width > 0.08:
            warnings.append(f"Wide accuracy CI ({acc_ci_width:.3f}) indicates high variance")
        
        # Check if CI includes baseline
        if acc_ci[0] < 0.52:
            warnings.append("Lower bound of accuracy CI close to or below baseline")
        
        metrics = {
            "bootstrap_samples": n_bootstrap,
            "accuracy_ci_95": [round(acc_ci[0], 4), round(acc_ci[1], 4)],
            "auc_roc_ci_95": [round(auc_ci[0], 4), round(auc_ci[1], 4)],
            "accuracy_ci_width": round(acc_ci_width, 4),
        }
        
        # Score based on CI lower bound
        score = max(0, min(100, (acc_ci[0] - 0.5) * 500))
        passed = acc_ci[0] >= 0.52
        
        self.results.append(AuditResult(
            section="Robustness",
            passed=passed,
            score=score,
            metrics=metrics,
            warnings=warnings
        ))
    
    # =========================================================================
    # 5. FEATURE IMPORTANCE ANALYSIS
    # =========================================================================
    def _test_feature_importance(self):
        """Analyze feature importance and contributions."""
        logger.info("\n[5/7] FEATURE IMPORTANCE ANALYSIS")
        logger.info("-" * 40)
        
        warnings = []
        
        # Get XGBoost importance
        trainer = self.predictor.trainer
        feature_cols = trainer.feature_columns
        
        # Get importance from the underlying XGBoost model
        try:
            if hasattr(trainer.model, 'get_booster'):
                # Direct XGBoost model
                booster = trainer.model.get_booster()
                importance = booster.get_score(importance_type='gain')
            elif hasattr(trainer.model, 'calibrated_classifiers_'):
                # Calibrated model - get base estimator
                base_model = trainer.model.calibrated_classifiers_[0].estimator
                if hasattr(base_model, 'get_booster'):
                    booster = base_model.get_booster()
                    importance = booster.get_score(importance_type='gain')
                else:
                    importance = {}
            else:
                importance = {}
        except Exception as e:
            logger.warning(f"  Could not extract XGBoost importance: {e}")
            importance = {}
        
        # Map importance to feature names
        feature_importance = {}
        for i, col in enumerate(feature_cols):
            key = f"f{i}"
            feature_importance[col] = importance.get(key, 0)
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_20 = sorted_features[:20]
        bottom_10 = sorted_features[-10:]
        
        logger.info("  Top 10 Features:")
        for feat, imp in top_20[:10]:
            logger.info(f"    {feat}: {imp:.2f}")
        
        # Check for dominance
        if sorted_features:
            total_importance = sum(v for _, v in sorted_features if v > 0)
            if total_importance > 0:
                top_3_share = sum(v for _, v in top_20[:3]) / total_importance
                if top_3_share > 0.5:
                    warnings.append(f"Top 3 features account for {top_3_share:.1%} of importance (possible overreliance)")
        
        # Check for zero-importance features
        zero_imp = [f for f, v in sorted_features if v == 0]
        if len(zero_imp) > len(feature_cols) * 0.3:
            warnings.append(f"{len(zero_imp)} features ({len(zero_imp)/len(feature_cols):.1%}) have zero importance")
        
        metrics = {
            "n_features": len(feature_cols),
            "top_20_features": [{"name": n, "importance": round(v, 4)} for n, v in top_20],
            "bottom_10_features": [{"name": n, "importance": round(v, 4)} for n, v in bottom_10],
            "zero_importance_count": len(zero_imp),
        }
        
        score = 70 if len(warnings) == 0 else 50
        passed = len(warnings) == 0
        
        self.results.append(AuditResult(
            section="Feature Importance",
            passed=passed,
            score=score,
            metrics=metrics,
            warnings=warnings
        ))
        
        if self.save_plots:
            self._plot_feature_importance(top_20)
    
    def _plot_feature_importance(self, top_features):
        """Generate feature importance plot."""
        try:
            import matplotlib.pyplot as plt
            
            if not top_features:
                return
            
            names = [f[0][:30] for f in top_features]  # Truncate long names
            values = [f[1] for f in top_features]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            y_pos = range(len(names))
            ax.barh(y_pos, values, color='steelblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names)
            ax.invert_yaxis()
            ax.set_xlabel('Importance (Gain)')
            ax.set_title('Top 20 Feature Importances')
            plt.tight_layout()
            
            self.output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"  Saved feature_importance.png")
        except ImportError:
            logger.warning("  matplotlib not available for plotting")
    
    # =========================================================================
    # 6. BIAS & FAIRNESS TESTS
    # =========================================================================
    def _test_bias_fairness(self):
        """Test for performance bias across segments."""
        logger.info("\n[6/7] BIAS & FAIRNESS TESTS")
        logger.info("-" * 40)
        
        warnings = []
        segment_results = {}
        
        # 1. Performance by surface
        if "ground_type" in self.test_df.columns:
            surface_perf = self._segment_performance("ground_type")
            segment_results["surface"] = surface_perf
            logger.info("  By Surface:")
            for surf, stats in surface_perf.items():
                logger.info(f"    {surf}: n={stats['n']}, Acc={stats['accuracy']:.1%}")
        
        # 2. Performance by odds range
        odds_df = self.test_df.with_columns([
            pl.when(pl.col("odds_player") < 1.5).then(pl.lit("Heavy Favorite (<1.5)"))
            .when(pl.col("odds_player") < 2.0).then(pl.lit("Favorite (1.5-2.0)"))
            .when(pl.col("odds_player") < 3.0).then(pl.lit("Even (2.0-3.0)"))
            .otherwise(pl.lit("Underdog (>3.0)"))
            .alias("odds_segment")
        ])
        
        odds_perf = {}
        for segment_data in odds_df.partition_by("odds_segment"):
            if len(segment_data) < 20:
                continue
            segment = segment_data["odds_segment"][0]
            y_t = segment_data["player_won"].to_numpy().astype(int)
            y_p = (segment_data["model_prob"].to_numpy() >= 0.5).astype(int)
            odds_perf[segment] = {
                "n": len(segment_data),
                "accuracy": round(accuracy_score(y_t, y_p), 4)
            }
        segment_results["odds_range"] = odds_perf
        logger.info("  By Odds Range:")
        for seg, stats in odds_perf.items():
            logger.info(f"    {seg}: n={stats['n']}, Acc={stats['accuracy']:.1%}")
        
        # 3. Performance by tournament round
        if "round_num" in self.test_df.columns:
            round_perf = self._segment_performance("round_num")
            segment_results["round"] = round_perf
        
        # Check for bias
        if odds_perf:
            accs = [v["accuracy"] for v in odds_perf.values()]
            if max(accs) - min(accs) > 0.10:
                warnings.append(f"Large accuracy gap between odds segments: {max(accs):.1%} vs {min(accs):.1%}")
        
        metrics = {
            "segment_performance": segment_results,
        }
        
        score = 70 if len(warnings) == 0 else 50
        passed = len(warnings) == 0
        
        self.results.append(AuditResult(
            section="Bias & Fairness",
            passed=passed,
            score=score,
            metrics=metrics,
            warnings=warnings
        ))
    
    def _segment_performance(self, column: str) -> Dict:
        """Calculate performance by segment."""
        result = {}
        for segment_data in self.test_df.partition_by(column):
            if len(segment_data) < 20:
                continue
            segment = str(segment_data[column][0])
            y_t = segment_data["player_won"].to_numpy().astype(int)
            y_p = (segment_data["model_prob"].to_numpy() >= 0.5).astype(int)
            result[segment] = {
                "n": len(segment_data),
                "accuracy": round(accuracy_score(y_t, y_p), 4)
            }
        return result
    
    # =========================================================================
    # 7. ADVERSARIAL EDGE CASE TESTS
    # =========================================================================
    def _test_adversarial_edge_cases(self):
        """Test model behavior on edge cases."""
        logger.info("\n[7/7] ADVERSARIAL EDGE CASE TESTS")
        logger.info("-" * 40)
        
        warnings = []
        edge_cases = {}
        
        # 1. High confidence wrong predictions
        wrong = self.test_df.filter(
            (pl.col("model_prob") >= 0.5) != pl.col("player_won")
        )
        high_conf_wrong = wrong.filter(
            (pl.col("model_prob") >= 0.65) | (pl.col("model_prob") <= 0.35)
        )
        
        hcw_rate = len(high_conf_wrong) / len(self.test_df) if len(self.test_df) > 0 else 0
        edge_cases["high_confidence_wrong"] = {
            "count": len(high_conf_wrong),
            "rate": round(hcw_rate, 4),
        }
        logger.info(f"  High-confidence wrong predictions: {len(high_conf_wrong)} ({hcw_rate:.1%})")
        
        if hcw_rate > 0.10:
            warnings.append(f"{hcw_rate:.1%} of predictions are high-confidence errors")
        
        # 2. Missing H2H data performance
        if "h2h_matches" in self.test_df.columns:
            no_h2h = self.test_df.filter(pl.col("h2h_matches") == 0)
            if len(no_h2h) >= 20:
                y_t = no_h2h["player_won"].to_numpy().astype(int)
                y_p = (no_h2h["model_prob"].to_numpy() >= 0.5).astype(int)
                no_h2h_acc = accuracy_score(y_t, y_p)
                edge_cases["no_h2h_data"] = {
                    "n": len(no_h2h),
                    "accuracy": round(no_h2h_acc, 4)
                }
                logger.info(f"  No H2H data: n={len(no_h2h)}, Acc={no_h2h_acc:.1%}")
        
        # 3. Extreme odds scenarios
        extreme_high = self.test_df.filter(pl.col("odds_player") > 5.0)
        extreme_low = self.test_df.filter(pl.col("odds_player") < 1.20)
        
        if len(extreme_high) >= 10:
            y_t = extreme_high["player_won"].to_numpy().astype(int)
            y_p = (extreme_high["model_prob"].to_numpy() >= 0.5).astype(int)
            edge_cases["extreme_underdog"] = {
                "n": len(extreme_high),
                "accuracy": round(accuracy_score(y_t, y_p), 4)
            }
            logger.info(f"  Extreme underdog (>5.0): n={len(extreme_high)}")
        
        if len(extreme_low) >= 10:
            y_t = extreme_low["player_won"].to_numpy().astype(int)
            y_p = (extreme_low["model_prob"].to_numpy() >= 0.5).astype(int)
            edge_cases["extreme_favorite"] = {
                "n": len(extreme_low),
                "accuracy": round(accuracy_score(y_t, y_p), 4)
            }
            logger.info(f"  Extreme favorite (<1.20): n={len(extreme_low)}")
        
        # 4. Model disagreement with odds
        model_favors = self.test_df.filter(
            (pl.col("model_prob") >= 0.55) & (pl.col("odds_player") > 2.0)
        )
        if len(model_favors) >= 10:
            y_t = model_favors["player_won"].to_numpy().astype(int)
            y_p = (model_favors["model_prob"].to_numpy() >= 0.5).astype(int)
            edge_cases["model_vs_odds_disagreement"] = {
                "n": len(model_favors),
                "accuracy": round(accuracy_score(y_t, y_p), 4),
                "actual_win_rate": round(y_t.mean(), 4)
            }
            logger.info(f"  Model vs Odds disagreement: n={len(model_favors)}")
        
        metrics = {
            "edge_case_analysis": edge_cases,
        }
        
        score = 70 if len(warnings) == 0 else 50
        passed = len(warnings) == 0
        
        self.results.append(AuditResult(
            section="Adversarial Edge Cases",
            passed=passed,
            score=score,
            metrics=metrics,
            warnings=warnings
        ))
    
    # =========================================================================
    # REPORT GENERATION
    # =========================================================================
    def _generate_summary(self) -> Dict:
        """Generate overall audit summary."""
        total_score = np.mean([r.score for r in self.results])
        all_passed = all(r.passed for r in self.results)
        all_warnings = []
        for r in self.results:
            all_warnings.extend(r.warnings)
        
        summary = {
            "audit_timestamp": datetime.now().isoformat(),
            "model_path": str(self.model_path),
            "test_set_size": len(self.test_df),
            "overall_score": round(total_score, 1),
            "all_tests_passed": all_passed,
            "total_warnings": len(all_warnings),
            "sections": [asdict(r) for r in self.results]
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("AUDIT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Overall Score: {total_score:.1f}/100")
        logger.info(f"Tests Passed: {sum(1 for r in self.results if r.passed)}/{len(self.results)}")
        logger.info(f"Total Warnings: {len(all_warnings)}")
        
        if all_warnings:
            logger.info("\nWarnings:")
            for w in all_warnings:
                logger.info(f"  [!] {w}")
        
        return summary
    
    def _save_reports(self, summary: Dict):
        """Save audit reports to files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON report
        json_path = self.output_dir / "model_audit_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        logger.info(f"\nSaved JSON report: {json_path}")
        
        # Markdown report
        md_path = self.output_dir / "model_audit_report.md"
        self._generate_markdown_report(summary, md_path)
        logger.info(f"Saved Markdown report: {md_path}")
    
    def _generate_markdown_report(self, summary: Dict, path: Path):
        """Generate human-readable markdown report."""
        lines = [
            "# Model Audit Report",
            "",
            f"**Generated**: {summary['audit_timestamp']}",
            f"**Model**: `{Path(summary['model_path']).name}`",
            f"**Test Set Size**: {summary['test_set_size']:,} matches",
            "",
            "## Overall Results",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| **Overall Score** | {summary['overall_score']:.1f}/100 |",
            f"| **Tests Passed** | {sum(1 for s in summary['sections'] if s['passed'])}/{len(summary['sections'])} |",
            f"| **Total Warnings** | {summary['total_warnings']} |",
            "",
        ]
        
        # Section details
        for section in summary['sections']:
            status = "[PASS]" if section['passed'] else "[WARN]"
            lines.append(f"## {status} {section['section']}")
            lines.append("")
            lines.append(f"**Score**: {section['score']:.1f}/100")
            lines.append("")
            
            # Key metrics
            lines.append("### Key Metrics")
            lines.append("```")
            for key, value in section['metrics'].items():
                if not isinstance(value, (dict, list)):
                    lines.append(f"{key}: {value}")
            lines.append("```")
            lines.append("")
            
            # Warnings
            if section['warnings']:
                lines.append("### Warnings")
                for w in section['warnings']:
                    lines.append(f"- [!] {w}")
                lines.append("")
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


def main(args_list=None):
    parser = argparse.ArgumentParser(description="Comprehensive Model Audit")
    parser.add_argument("--save-plots", action="store_true", help="Save visualization plots")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for reports")
    parser.add_argument("--test-cutoff", type=str, default="2025-01-01", help="Test set cutoff date (YYYY-MM-DD)")
    args = parser.parse_args(args_list)
    
    # Get data and model paths
    data_path = PROCESSED_DATA_DIR / "features_dataset.parquet"
    
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)
    
    # Get active model
    # Get active model
    registry = ModelRegistry(root_dir=ROOT)
    
    try:
        version, path = registry.get_production_model()
        model_path = Path(path)
        logger.info(f"Auditing Production model: {version}")
    except Exception:
        # Fallback to challenger or direct file
        try:
            version, path = registry.get_challenger_model()
            model_path = Path(path)
            logger.info(f"Auditing Staging model: {version}")
        except Exception:
            # Fallback to direct model file
            model_path = MODELS_DIR / "xgboost_model.joblib"
            if not model_path.exists():
                model_path = MODELS_DIR / "experimental" / "model.bin"
            
            if not model_path.exists():
                logger.error("No model found (Production, Staging, or Experimental). Run training first.")
                sys.exit(1)
    
    # Run audit
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    test_cutoff = date.fromisoformat(args.test_cutoff)
    
    auditor = ModelAuditor(
        model_path=model_path,
        data_path=data_path,
        test_cutoff=test_cutoff,
        save_plots=args.save_plots,
        output_dir=output_dir
    )
    
    summary = auditor.run_full_audit()
    
    # Exit code based on pass/fail
    sys.exit(0 if summary["all_tests_passed"] else 1)


if __name__ == "__main__":
    main()
