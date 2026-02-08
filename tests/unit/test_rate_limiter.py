"""
Unit tests for rate limiter components.
"""
import pytest
import time
from unittest.mock import patch

from src.scraper.rate_limiter import (
    ExponentialBackoff,
    SlidingWindowRateLimiter,
    AdaptiveRateLimiter,
    EnhancedCircuitBreaker,
    parse_retry_after,
)


class TestParseRetryAfter:
    """Tests for Retry-After header parsing."""
    
    def test_parse_integer_seconds(self):
        """Test parsing integer seconds."""
        assert parse_retry_after("60") == 60
        assert parse_retry_after("120") == 120
    
    def test_parse_none(self):
        """Test None input."""
        assert parse_retry_after(None) is None
    
    def test_parse_empty(self):
        """Test empty string."""
        assert parse_retry_after("") is None
    
    def test_parse_invalid(self):
        """Test invalid value."""
        assert parse_retry_after("not-a-number") is None


class TestExponentialBackoff:
    """Tests for ExponentialBackoff."""
    
    def test_initial_delay(self):
        """Test delay on first attempt."""
        backoff = ExponentialBackoff(base_delay=2.0, max_delay=300.0, jitter_factor=0.0)
        delay = backoff.get_delay(0)
        assert 1.5 <= delay <= 2.5  # Allow small margin
    
    def test_exponential_growth(self):
        """Test delay grows exponentially."""
        backoff = ExponentialBackoff(base_delay=2.0, max_delay=300.0, jitter_factor=0.0)
        delays = [backoff.get_delay(i) for i in range(5)]
        
        # Each delay should be roughly 2x the previous
        for i in range(1, len(delays)):
            assert delays[i] > delays[i-1], f"Delay {i} should be > delay {i-1}"
    
    def test_max_delay_cap(self):
        """Test delay is capped at max."""
        backoff = ExponentialBackoff(base_delay=2.0, max_delay=10.0, jitter_factor=0.0)
        delay = backoff.get_delay(100)
        assert delay <= 10.5  # Allow small margin for jitter
    
    def test_retry_after_respected(self):
        """Test Retry-After header is respected."""
        backoff = ExponentialBackoff(base_delay=2.0, max_delay=300.0)
        delay = backoff.get_delay(0, retry_after=60)
        
        assert 60 <= delay <= 70  # Should be ~60 + jitter
    
    def test_jitter_adds_variance(self):
        """Test jitter creates variance in delays."""
        backoff = ExponentialBackoff(base_delay=2.0, jitter_factor=0.3)
        delays = [backoff.get_delay(1) for _ in range(20)]
        
        # Should have some variance
        assert len(set(delays)) > 5  # At least some unique values


class TestSlidingWindowRateLimiter:
    """Tests for SlidingWindowRateLimiter."""
    
    def test_initial_no_delay(self):
        """Test first request has minimal delay."""
        limiter = SlidingWindowRateLimiter(max_requests=10, min_delay=0.0)
        delay = limiter.acquire("test")
        assert delay == 0.0
    
    def test_min_delay_enforced(self):
        """Test minimum delay between requests."""
        limiter = SlidingWindowRateLimiter(max_requests=100, min_delay=1.0)
        limiter.record_request("test")
        delay = limiter.acquire("test")
        
        # Should have some delay
        assert delay > 0.5
    
    def test_rate_limit_kicks_in(self):
        """Test rate limit after max requests."""
        limiter = SlidingWindowRateLimiter(
            window_seconds=60, max_requests=5, min_delay=0.0
        )
        
        # Simulate max requests
        for _ in range(5):
            limiter.record_request("test")
        
        delay = limiter.acquire("test")
        assert delay > 0  # Should need to wait
    
    def test_stats_tracking(self):
        """Test statistics are tracked."""
        limiter = SlidingWindowRateLimiter(max_requests=10)
        limiter.record_request("endpoint1")
        limiter.record_request("endpoint1")
        limiter.record_request("endpoint2")
        
        stats = limiter.get_stats("endpoint1")
        assert stats["requests_in_window"] == 2


class TestAdaptiveRateLimiter:
    """Tests for AdaptiveRateLimiter."""
    
    def test_slows_down_on_rate_limit(self):
        """Test delay increases after rate limit."""
        limiter = AdaptiveRateLimiter(base_delay=1.0, increase_factor=2.0)
        
        initial_delay = limiter.get_delay("test")
        limiter.record_rate_limit("test")
        new_delay = limiter.get_delay("test")
        
        assert new_delay > initial_delay
    
    def test_speeds_up_on_success(self):
        """Test delay decreases after successes."""
        limiter = AdaptiveRateLimiter(
            base_delay=2.0, success_threshold=3, decrease_factor=0.5
        )
        
        # Force a slow start
        limiter.record_rate_limit("test")
        slow_delay = limiter.get_delay("test")
        
        # Record enough successes
        for _ in range(5):
            limiter.record_success("test")
        
        fast_delay = limiter.get_delay("test")
        assert fast_delay < slow_delay


class TestEnhancedCircuitBreaker:
    """Tests for EnhancedCircuitBreaker."""
    
    def test_starts_closed(self):
        """Test circuit starts closed."""
        breaker = EnhancedCircuitBreaker()
        assert not breaker.is_open("test")
    
    def test_opens_after_failures(self):
        """Test circuit opens after threshold failures."""
        breaker = EnhancedCircuitBreaker(failure_threshold=2)
        
        breaker.record_failure("test", 429)
        assert not breaker.is_open("test")
        
        breaker.record_failure("test", 429)
        assert breaker.is_open("test")
    
    def test_ignores_non_rate_limit_errors(self):
        """Test non-rate-limit errors don't trip breaker."""
        breaker = EnhancedCircuitBreaker(failure_threshold=2)
        
        breaker.record_failure("test", 500)
        breaker.record_failure("test", 500)
        
        assert not breaker.is_open("test")
    
    def test_success_resets_failures(self):
        """Test success resets failure count."""
        breaker = EnhancedCircuitBreaker(failure_threshold=3)
        
        breaker.record_failure("test", 429)
        breaker.record_failure("test", 429)
        breaker.record_success("test")
        breaker.record_failure("test", 429)
        
        assert not breaker.is_open("test")  # Count was reset
    
    def test_per_endpoint_tracking(self):
        """Test each endpoint tracked separately."""
        breaker = EnhancedCircuitBreaker(failure_threshold=2)
        
        breaker.record_failure("endpoint1", 429)
        breaker.record_failure("endpoint1", 429)
        
        assert breaker.is_open("endpoint1")
        assert not breaker.is_open("endpoint2")
    
    def test_status_reporting(self):
        """Test status dictionary returned."""
        breaker = EnhancedCircuitBreaker()
        breaker.record_failure("test", 429)
        
        status = breaker.get_status("test")
        
        assert "state" in status
        assert "failures" in status
        assert status["failures"] == 1
