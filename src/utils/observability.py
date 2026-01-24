# src/utils/observability.py
import logging
import os
from typing import Optional
import structlog
import contextvars
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from datetime import datetime

# Correlation ID for distributed tracing
CORRELATION_ID = contextvars.ContextVar('correlation_id', default=None)

class ObservabilityConfig:
    """Configuration for observability stack."""
    
    def __init__(self):
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.enable_metrics = os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
        self.metrics_port = int(os.getenv('METRICS_PORT', 8000))
        self.log_format = 'json' if self.environment == 'production' else 'console'

class MetricsRegistry:
    """Centralized metrics management."""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialize all metrics with proper naming conventions."""
        
        # HISTOGRAMS (timing data)
        self.prediction_latency = Histogram(
            'prediction_latency_seconds',
            'Inference latency in seconds',
            buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
            registry=self.registry
        )
        
        self.pipeline_duration = Histogram(
            'pipeline_duration_seconds',
            'Pipeline execution duration in seconds',
            labelnames=['pipeline_stage'],
            buckets=(1, 5, 10, 30, 60, 300, 900),
            registry=self.registry
        )
        
        self.feature_engineering_latency = Histogram(
            'feature_engineering_latency_seconds',
            'Feature engineering latency',
            buckets=(0.1, 0.5, 1.0, 5.0, 10.0),
            registry=self.registry
        )
        
        # COUNTERS (monotonic increases)
        self.failed_predictions = Counter(
            'failed_predictions_total',
            'Total number of prediction failures',
            labelnames=['error_type'],
            registry=self.registry
        )
        
        self.successful_predictions = Counter(
            'successful_predictions_total',
            'Total number of successful predictions',
            registry=self.registry
        )
        
        self.data_pipeline_errors = Counter(
            'data_pipeline_errors_total',
            'Data pipeline errors',
            labelnames=['stage', 'error_type'],
            registry=self.registry
        )
        
        self.training_pipeline_runs = Counter(
            'training_pipeline_runs_total',
            'Training pipeline runs',
            labelnames=['status'],  # 'success' or 'failure'
            registry=self.registry
        )
        
        # GAUGES (point-in-time snapshots)
        self.model_drift = Gauge(
            'model_drift_auc_change',
            'AUC change from baseline (percentage)',
            registry=self.registry
        )
        
        self.last_prediction_timestamp = Gauge(
            'last_prediction_timestamp_unix',
            'Unix timestamp of last prediction',
            registry=self.registry
        )
        
        self.training_dataset_size = Gauge(
            'training_dataset_size_rows',
            'Number of rows in training dataset',
            registry=self.registry
        )
        
        self.model_version = Gauge(
            'model_version_info',
            'Model version metadata',
            labelnames=['version', 'trained_date'],
            registry=self.registry
        )

class StructlogConfig:
    """Structured logging configuration."""
    
    @staticmethod
    def configure(env: str = 'development', log_level: str = 'INFO'):
        """
        Configure structlog with environment-appropriate settings.
        
        Production: JSON output (machine-readable)
        Development: Console output (human-readable)
        """
        
        shared_processors = [
            # Add correlation ID to all logs
            structlog.contextvars.merge_contextvars,
            # Add log level
            structlog.processors.add_log_level,
            # Add timestamp
            structlog.processors.TimeStamper(fmt='iso'),
            # Add exception info
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
        ]
        
        if env == 'production':
            # JSON for log aggregators (ELK, Datadog, etc.)
            processors = shared_processors + [
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(),
            ]
        else:
            # Pretty console for development
            processors = shared_processors + [
                structlog.dev.ConsoleRenderer(),
            ]
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(
                logging.getLevelName(log_level)
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )

class Logger:
    """Wrapper for structured logging with context awareness."""
    
    def __init__(self, module_name: str):
        self.logger = structlog.get_logger(module_name)
        self.module_name = module_name
    
    def with_correlation_id(self, correlation_id: str):
        """Bind correlation ID to all subsequent logs."""
        CORRELATION_ID.set(correlation_id)
        return self.logger.bind(correlation_id=correlation_id)
    
    def log_event(self, event: str, **kwargs):
        """Log structured event with automatic context."""
        corr_id = CORRELATION_ID.get()
        ctx = {'correlation_id': corr_id, 'module': self.module_name}
        ctx.update(kwargs)
        return self.logger.info(event, **ctx)
    
    def log_error(self, event: str, exc_info=None, **kwargs):
        """Log error with exception details."""
        corr_id = CORRELATION_ID.get()
        ctx = {'correlation_id': corr_id, 'module': self.module_name}
        ctx.update(kwargs)
        return self.logger.error(event, exc_info=exc_info, **ctx)

def initialize_observability(environment: str = 'development'):
    """One-stop initialization for all observability components."""
    config = ObservabilityConfig()
    StructlogConfig.configure(env=environment, log_level=config.log_level)
    metrics = MetricsRegistry()
    
    logger = structlog.get_logger(__name__)
    logger.info(
        'observability_initialized',
        environment=environment,
        log_format=config.log_format,
        metrics_enabled=config.enable_metrics,
    )
    
    return metrics, config

# Global metrics instance
METRICS = None
CONFIG = None

def get_metrics() -> MetricsRegistry:
    """Lazy-load metrics singleton."""
    global METRICS
    if METRICS is None:
        METRICS, CONFIG = initialize_observability()
    return METRICS
