"""
Daily metrics computation job.

Joins predictions to outcomes, computes ROI/calibration,
and exports summary metrics to Prometheus.
"""
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import polars as pl

from .prediction_store import get_prediction_store

logger = logging.getLogger(__name__)


@dataclass
class DailyMetricsResult:
    """Result of daily metrics computation."""
    date: date
    model_version: str
    total_bets: int
    total_stake: float
    total_pnl: float
    roi_pct: float
    win_rate_pct: float
    calibration_error: float
    by_surface: Dict[str, Dict]
    by_odds_band: Dict[str, Dict]


def compute_model_roi(
    model_version: str = None,
    start_date: date = None,
    end_date: date = None,
) -> Dict[str, float]:
    """
    Compute ROI for a model version.
    
    Args:
        model_version: Model version to compute ROI for
        start_date: Start date for computation
        end_date: End date for computation
        
    Returns:
        Dict with ROI metrics
    """
    store = get_prediction_store()
    return store.compute_roi(
        model_version=model_version,
        start_date=start_date,
        end_date=end_date,
    )


def compute_calibration(
    model_version: str = None,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute calibration metrics for a model.
    
    Measures how well predicted probabilities match actual outcomes.
    
    Returns:
        Dict with calibration_error (lower is better) and bin details
    """
    store = get_prediction_store()
    conn = store._get_conn()
    
    try:
        query = """
            SELECT predicted_prob, actual_outcome
            FROM predictions
            WHERE actual_outcome IS NOT NULL
        """
        params = []
        
        if model_version:
            query += " AND model_version = ?"
            params.append(model_version)
        
        rows = conn.execute(query, params).fetchall()
        
        if not rows:
            return {"calibration_error": 0.0, "bins": []}
        
        # Group into bins
        bins = [{"total": 0, "wins": 0, "avg_prob": 0.0} for _ in range(n_bins)]
        
        for row in rows:
            prob = row['predicted_prob']
            bin_idx = min(int(prob * n_bins), n_bins - 1)
            bins[bin_idx]["total"] += 1
            bins[bin_idx]["wins"] += row['actual_outcome']
            bins[bin_idx]["avg_prob"] += prob
        
        # Compute actual rates and calibration error
        calibration_error = 0.0
        bin_details = []
        
        for i, b in enumerate(bins):
            if b["total"] > 0:
                actual_rate = b["wins"] / b["total"]
                expected_rate = b["avg_prob"] / b["total"]
                error = abs(actual_rate - expected_rate)
                calibration_error += error * b["total"]
                
                bin_details.append({
                    "bin": i,
                    "range": f"{i/n_bins:.1f}-{(i+1)/n_bins:.1f}",
                    "count": b["total"],
                    "expected": expected_rate,
                    "actual": actual_rate,
                    "error": error,
                })
        
        total_samples = sum(b["total"] for b in bins)
        if total_samples > 0:
            calibration_error /= total_samples
        
        return {
            "calibration_error": calibration_error,
            "bins": bin_details,
        }
        
    finally:
        conn.close()


def compute_feature_drift(
    df: pl.DataFrame,
    feature_cols: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute feature distribution statistics for drift detection.
    
    Args:
        df: DataFrame with feature values
        feature_cols: Columns to analyze (defaults to common features)
        
    Returns:
        Dict mapping feature name to {mean, stddev, null_rate}
    """
    if feature_cols is None:
        feature_cols = [
            "odds_player", "implied_prob_player", "player_win_rate_20",
            "player_surface_win_rate_20", "h2h_win_rate"
        ]
    
    results = {}
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        series = df[col]
        total = len(series)
        null_count = series.null_count()
        
        # Filter nulls for stats
        non_null = series.drop_nulls()
        
        if len(non_null) > 0:
            results[col] = {
                "mean": float(non_null.mean()),
                "stddev": float(non_null.std()) if len(non_null) > 1 else 0.0,
                "null_rate": null_count / total if total > 0 else 0.0,
            }
    
    return results


def daily_metrics_update(
    export_to_prometheus: bool = True,
) -> Dict[str, DailyMetricsResult]:
    """
    Run daily metrics computation and export to Prometheus.
    
    1. Load resolved predictions from last 24h
    2. Compute ROI by model version
    3. Compute calibration
    4. Export to Prometheus gauges
    
    Returns:
        Dict mapping model_version to DailyMetricsResult
    """
    from src.utils.observability import get_metrics
    
    store = get_prediction_store()
    metrics = get_metrics() if export_to_prometheus else None
    
    results = {}
    today = date.today()
    yesterday = today - timedelta(days=1)
    
    # Get all model versions with predictions
    model_versions = store.get_model_versions()
    
    for version in model_versions:
        # Compute ROI
        roi_data = store.compute_roi(
            model_version=version,
            start_date=yesterday,
            end_date=today,
        )
        
        # Compute by surface
        by_surface = {}
        for surface in ["hard", "clay", "grass"]:
            surface_roi = store.compute_roi(
                model_version=version,
                surface=surface,
            )
            if surface_roi["total_bets"] > 0:
                by_surface[surface] = surface_roi
        
        # Compute by odds band
        by_odds_band = {}
        for band in ["heavy_favorite", "favorite", "slight_favorite", "even", "underdog", "long_shot"]:
            band_roi = store.compute_roi(
                model_version=version,
                odds_band=band,
            )
            if band_roi["total_bets"] > 0:
                by_odds_band[band] = band_roi
        
        # Compute calibration
        calibration = compute_calibration(model_version=version)
        
        result = DailyMetricsResult(
            date=today,
            model_version=version,
            total_bets=roi_data["total_bets"],
            total_stake=roi_data["total_stake"],
            total_pnl=roi_data["total_pnl"],
            roi_pct=roi_data["roi"],
            win_rate_pct=roi_data["win_rate"],
            calibration_error=calibration["calibration_error"],
            by_surface=by_surface,
            by_odds_band=by_odds_band,
        )
        
        results[version] = result
        
        # Export to Prometheus
        if metrics:
            try:
                metrics.model_realized_roi.labels(model_version=version).set(roi_data["roi"])
                
                # Update bankroll metrics from cumulative data
                cumulative = store.compute_roi(model_version=version)
                if cumulative["total_stake"] > 0:
                    metrics.total_stakes.set(cumulative["total_stake"])
                    metrics.total_returns.set(cumulative["total_stake"] + cumulative["total_pnl"])
                    
            except Exception as e:
                logger.warning(f"Failed to export metrics for {version}: {e}")
        
        logger.info(
            f"Daily metrics for {version}: "
            f"ROI={roi_data['roi']:.2f}%, "
            f"WinRate={roi_data['win_rate']:.1f}%, "
            f"Bets={roi_data['total_bets']}"
        )
    
    return results


def update_feature_drift_metrics(df: pl.DataFrame):
    """
    Update Prometheus feature drift metrics from DataFrame.
    
    Args:
        df: DataFrame with feature values
    """
    from src.utils.observability import get_metrics
    
    metrics = get_metrics()
    drift_stats = compute_feature_drift(df)
    
    for feature_name, stats in drift_stats.items():
        try:
            metrics.feature_mean.labels(feature_name=feature_name).set(stats["mean"])
            metrics.feature_stddev.labels(feature_name=feature_name).set(stats["stddev"])
            metrics.feature_null_rate.labels(feature_name=feature_name).set(stats["null_rate"])
        except Exception as e:
            logger.warning(f"Failed to update drift metrics for {feature_name}: {e}")
    
    logger.info(f"Updated drift metrics for {len(drift_stats)} features")


def update_bankroll_metrics(
    current_balance: float,
    peak_balance: float = None,
    total_stakes: float = None,
    total_returns: float = None,
):
    """
    Update bankroll-related Prometheus metrics.
    
    Args:
        current_balance: Current bankroll
        peak_balance: Peak bankroll (for drawdown calculation)
        total_stakes: Total amount staked
        total_returns: Total returns from bets
    """
    from src.utils.observability import get_metrics
    
    metrics = get_metrics()
    
    metrics.bankroll_current.set(current_balance)
    
    if peak_balance is not None:
        metrics.bankroll_peak.set(peak_balance)
        if peak_balance > 0:
            drawdown = (peak_balance - current_balance) / peak_balance * 100
            metrics.bankroll_drawdown_pct.set(max(0, drawdown))
    
    if total_stakes is not None:
        metrics.total_stakes.set(total_stakes)
    
    if total_returns is not None:
        metrics.total_returns.set(total_returns)
    
    logger.debug(f"Updated bankroll metrics: current={current_balance}")
