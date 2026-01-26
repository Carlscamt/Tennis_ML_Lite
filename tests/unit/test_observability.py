import pytest
import structlog
from unittest.mock import MagicMock, patch
from src.utils.observability import Logger, MetricsRegistry, ObservabilityConfig

class TestObservability:
    
    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    def test_logger_event_structure(self, mock_logger):
        """Log events include required context fields."""
        # Mock structlog.get_logger to return our mock
        with patch("structlog.get_logger", return_value=mock_logger):
            logger = Logger("test_module")
            logger.log_event("test_event", custom_field=123)
            
            # Check call
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            
            # Message check
            assert call_args[0][0] == "test_event"
            
            # Context check
            kwargs = call_args[1]
            assert kwargs["module"] == "test_module"
            assert kwargs["custom_field"] == 123
            # correlation_id might be None or UUID depending on context, check definition

    def test_logger_error_capture(self, mock_logger):
        """Error logs capture exception info."""
        with patch("structlog.get_logger", return_value=mock_logger):
            logger = Logger("test_module")
            try:
                raise ValueError("Oops")
            except ValueError as e:
                logger.log_error("test_error", exc_info=e)
                
            mock_logger.error.assert_called_once()
            kwargs = mock_logger.error.call_args[1]
            assert kwargs["exc_info"] is not None

    def test_metrics_registry_initialization(self):
        """Metrics registry initializes standard metrics."""
        # This might fail if global registry is dirty, but we are just unit testing the class init
        registry = MetricsRegistry()
        
        # Check if high-level metrics exist
        assert registry.prediction_latency is not None
        assert registry.failed_predictions is not None

    def test_observability_config_defaults(self):
        """Configuration defaults to development mode."""
        config = ObservabilityConfig()
        assert config.environment == "development" # Default unless env var set
        assert config.log_format == "console"
