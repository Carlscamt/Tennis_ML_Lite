
"""
Tests for Data Quality Validation.
"""
import pytest
import polars as pl
import numpy as np
from src.schema import SchemaValidator
from src.data_quality.validator import DriftDetector, AnomalyDetector

def test_schema_validation_valid_data():
    """Valid data passes schema validation."""
    validator = SchemaValidator()
    
    valid_df = pl.DataFrame({
        'event_id': [1, 2],
        'player_id': [101, 102],
        'opponent_id': [201, 202],
        'start_timestamp': [1700000000, 1700000001],
        'player_name': ['Djokovic', 'Nadal'],
        'opponent_name': ['Alcaraz', 'Sinner'],
        'tournament_name': ['Wimbledon', 'Roland Garros'],
        'ground_type': ['Grass', 'Clay'],
        'odds_player': [1.5, 2.0],
        'odds_opponent': [2.5, 1.8],
        'player_won': [True, False]
    })
    
    result = validator.validate_raw_data(valid_df)
    assert result['valid']
    assert result['num_invalid_rows'] == 0

def test_schema_validation_invalid_odds():
    """Invalid odds (must be > 1.0) fail validation."""
    validator = SchemaValidator()
    
    invalid_df = pl.DataFrame({
        'event_id': [1],
        'player_id': [101],
        'opponent_id': [201],
        'start_timestamp': [1700000000],
        'player_name': ['Djokovic'],
        'opponent_name': ['Alcaraz'],
        'tournament_name': ['Wimbledon'],
        'ground_type': ['Grass'],
        'odds_player': [-1.0],  # INVALID: < 0.0
        'odds_opponent': [2.5]
    })
    
    result = validator.validate_raw_data(invalid_df)
    assert not result['valid']
    # Error message should mention odds
    # Note: Pandera might return aggregated errors
    assert any("odds" in e.lower() or "check" in e.lower() for e in result['errors'])

def test_drift_detection_no_drift():
    """Identical distributions should not trigger drift detection."""
    np.random.seed(42)
    detector = DriftDetector()
    
    # Create reference data
    ref_data = pl.DataFrame({
        'feature_1': np.random.normal(100, 15, 1000),
        'feature_2': np.random.normal(0.5, 0.1, 1000),
    })
    detector.fit_reference(ref_data)
    
    # Current data from same distribution
    current_data = pl.DataFrame({
        'feature_1': np.random.normal(100, 15, 1000),
        'feature_2': np.random.normal(0.5, 0.1, 1000),
    })
    
    reports = detector.detect_drift(current_data)
    
    # No significant drift should be detected
    assert not any(r.is_drifted for r in reports.values())

def test_drift_detection_with_drift():
    """Shifted distribution should trigger drift detection."""
    detector = DriftDetector()
    
    ref_data = pl.DataFrame({
        'feature_1': np.random.normal(100, 15, 1000),
    })
    detector.fit_reference(ref_data)
    
    # Current data with mean shift
    current_data = pl.DataFrame({
        'feature_1': np.random.normal(130, 15, 1000),  # Mean shifted by 30
    })
    
    reports = detector.detect_drift(current_data)
    
    # Drift should be detected
    assert any(r.is_drifted for r in reports.values())

def test_anomaly_detection():
    """Outliers should be detected."""
    detector = AnomalyDetector(z_threshold=3.0)
    
    ref_data = pl.DataFrame({
        'feature': np.random.normal(100, 15, 1000),
    })
    detector.fit_reference(ref_data)
    
    # Current data with outliers (e.g. 500 when mean is 100)
    current_data = pl.DataFrame({
        'feature': [100.0] * 90 + [500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0],
    })
    
    anomalies = detector.detect_anomalies(current_data)
    
    assert 'feature' in anomalies
    assert len(anomalies['feature']) >= 5
