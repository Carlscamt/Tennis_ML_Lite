"""
Schema validation and health monitoring for scraper responses.

Provides lightweight validation for API responses without heavy dependencies.
Tracks scraper health metrics and alerts on degradation.
"""
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Callable
from threading import Lock
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ResponseSchema:
    """
    Lightweight schema validator for API responses.
    
    Validates presence of required fields and field types.
    
    Example:
        schema = ResponseSchema(
            name="rankings",
            required_fields=["rankingRows"],
            field_types={"rankingRows": list},
        )
        errors = schema.validate(response)
    """
    name: str
    required_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, type] = field(default_factory=dict)
    nested_fields: Dict[str, "ResponseSchema"] = field(default_factory=dict)
    
    def validate(self, data: Dict) -> List[str]:
        """
        Validate response against schema.
        
        Args:
            data: Response dictionary to validate
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        if data is None:
            errors.append(f"{self.name}: Response is None")
            return errors
        
        if not isinstance(data, dict):
            errors.append(f"{self.name}: Expected dict, got {type(data).__name__}")
            return errors
        
        # Check required fields
        for field_name in self.required_fields:
            if field_name not in data:
                errors.append(f"{self.name}: Missing required field '{field_name}'")
        
        # Check field types
        for field_name, expected_type in self.field_types.items():
            if field_name in data:
                value = data[field_name]
                if value is not None and not isinstance(value, expected_type):
                    errors.append(
                        f"{self.name}: Field '{field_name}' expected {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
        
        # Validate nested schemas
        for field_name, nested_schema in self.nested_fields.items():
            if field_name in data and data[field_name] is not None:
                nested_data = data[field_name]
                if isinstance(nested_data, dict):
                    nested_errors = nested_schema.validate(nested_data)
                    errors.extend(nested_errors)
                elif isinstance(nested_data, list) and nested_data:
                    # Validate first item in list
                    nested_errors = nested_schema.validate(nested_data[0])
                    errors.extend(nested_errors)
        
        return errors


# Pre-defined schemas for SofaScore API endpoints
RANKINGS_SCHEMA = ResponseSchema(
    name="rankings",
    required_fields=["rankingRows"],
    field_types={"rankingRows": list},
)

MATCH_SCHEMA = ResponseSchema(
    name="match",
    required_fields=["id", "homeTeam", "awayTeam"],
    field_types={"id": int},
    nested_fields={
        "homeTeam": ResponseSchema(
            name="team",
            required_fields=["id", "name"],
            field_types={"id": int, "name": str},
        ),
        "awayTeam": ResponseSchema(
            name="team",
            required_fields=["id", "name"],
            field_types={"id": int, "name": str},
        ),
    },
)

EVENTS_SCHEMA = ResponseSchema(
    name="events",
    required_fields=["events"],
    field_types={"events": list},
)

ODDS_SCHEMA = ResponseSchema(
    name="odds",
    required_fields=["markets"],
    field_types={"markets": list},
)


def get_schema_for_endpoint(endpoint: str) -> Optional[ResponseSchema]:
    """Get appropriate schema for an endpoint."""
    if "/rankings/" in endpoint:
        return RANKINGS_SCHEMA
    elif "/scheduled-events/" in endpoint:
        return EVENTS_SCHEMA
    elif "/events/" in endpoint and "/last/" in endpoint:
        return EVENTS_SCHEMA
    elif "/odds/" in endpoint:
        return ODDS_SCHEMA
    elif "/event/" in endpoint:
        return MATCH_SCHEMA
    return None


def validate_response(endpoint: str, data: Dict) -> List[str]:
    """
    Validate response data against appropriate schema.
    
    Args:
        endpoint: API endpoint
        data: Response data
        
    Returns:
        List of validation errors (empty if valid)
    """
    schema = get_schema_for_endpoint(endpoint)
    if schema is None:
        return []  # No schema defined, assume valid
    
    return schema.validate(data)


@dataclass
class ScraperHealthMetrics:
    """
    Tracks scraper health metrics for monitoring and alerting.
    
    Logs anomalies and raises alerts when failure rate exceeds threshold.
    
    Example:
        metrics = ScraperHealthMetrics()
        metrics.record_request("/rankings/5", success=True)
        metrics.record_request("/event/123", success=False, error="schema_failure")
        
        if metrics.should_alert():
            notify_admin(metrics.get_summary())
    """
    alert_threshold: float = 0.9  # Alert if success rate drops below 90%
    alert_schema_failures: int = 10  # Alert after this many schema failures
    window_minutes: int = 60  # Rolling window for metrics
    
    # Metrics
    _total_requests: int = field(default=0)
    _successful_requests: int = field(default=0)
    _rate_limit_hits: int = field(default=0)
    _schema_failures: int = field(default=0)
    _network_errors: int = field(default=0)
    
    # Per-endpoint tracking
    _endpoint_stats: Dict[str, Dict] = field(default_factory=lambda: defaultdict(lambda: {
        "total": 0, "success": 0, "rate_limits": 0, "schema_errors": 0
    }))
    
    # Time-windowed tracking
    _recent_requests: List[Dict] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)
    _alert_callbacks: List[Callable] = field(default_factory=list)
    
    def record_request(
        self,
        endpoint: str,
        success: bool,
        error_type: Optional[str] = None,
        response_time_ms: Optional[float] = None,
    ):
        """
        Record a request result.
        
        Args:
            endpoint: API endpoint
            success: Whether request succeeded
            error_type: Type of error (rate_limit, schema_failure, network_error)
            response_time_ms: Response time in milliseconds
        """
        with self._lock:
            now = datetime.now()
            
            self._total_requests += 1
            self._endpoint_stats[endpoint]["total"] += 1
            
            if success:
                self._successful_requests += 1
                self._endpoint_stats[endpoint]["success"] += 1
            else:
                if error_type == "rate_limit":
                    self._rate_limit_hits += 1
                    self._endpoint_stats[endpoint]["rate_limits"] += 1
                elif error_type == "schema_failure":
                    self._schema_failures += 1
                    self._endpoint_stats[endpoint]["schema_errors"] += 1
                elif error_type == "network_error":
                    self._network_errors += 1
            
            # Add to recent requests
            self._recent_requests.append({
                "timestamp": now,
                "endpoint": endpoint,
                "success": success,
                "error_type": error_type,
                "response_time_ms": response_time_ms,
            })
            
            # Prune old requests
            cutoff = now - timedelta(minutes=self.window_minutes)
            self._recent_requests = [
                r for r in self._recent_requests if r["timestamp"] > cutoff
            ]
            
            # Check for alert condition
            if self.should_alert():
                self._trigger_alert()
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self._total_requests == 0:
            return 1.0
        return self._successful_requests / self._total_requests
    
    @property
    def recent_success_rate(self) -> float:
        """Calculate success rate in recent window."""
        with self._lock:
            if not self._recent_requests:
                return 1.0
            successes = sum(1 for r in self._recent_requests if r["success"])
            return successes / len(self._recent_requests)
    
    def should_alert(self) -> bool:
        """Check if alert condition is met."""
        return (
            self.recent_success_rate < self.alert_threshold or
            self._schema_failures >= self.alert_schema_failures
        )
    
    def _trigger_alert(self):
        """Trigger alert callbacks."""
        summary = self.get_summary()
        logger.warning(f"Scraper health alert: {summary}")
        
        for callback in self._alert_callbacks:
            try:
                callback(summary)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback to be called on alert."""
        self._alert_callbacks.append(callback)
    
    def get_summary(self) -> Dict:
        """Get health metrics summary."""
        with self._lock:
            return {
                "total_requests": self._total_requests,
                "successful_requests": self._successful_requests,
                "success_rate": round(self.success_rate, 4),
                "recent_success_rate": round(self.recent_success_rate, 4),
                "rate_limit_hits": self._rate_limit_hits,
                "schema_failures": self._schema_failures,
                "network_errors": self._network_errors,
                "alert_triggered": self.should_alert(),
            }
    
    def get_endpoint_stats(self) -> Dict[str, Dict]:
        """Get per-endpoint statistics."""
        with self._lock:
            result = {}
            for endpoint, stats in self._endpoint_stats.items():
                total = stats["total"]
                success = stats["success"]
                result[endpoint] = {
                    **stats,
                    "success_rate": round(success / total, 4) if total > 0 else 1.0,
                }
            return result
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._total_requests = 0
            self._successful_requests = 0
            self._rate_limit_hits = 0
            self._schema_failures = 0
            self._network_errors = 0
            self._endpoint_stats.clear()
            self._recent_requests.clear()


# Global health metrics instance
_global_health_metrics = ScraperHealthMetrics()


def get_health_metrics() -> ScraperHealthMetrics:
    """Get global health metrics instance."""
    return _global_health_metrics
