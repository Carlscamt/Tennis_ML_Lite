"""
Advanced rate limiting with exponential backoff, jitter, and per-endpoint tracking.

Features:
- Exponential backoff with configurable jitter
- Sliding window rate limiter per endpoint
- Retry-After header parsing
- Adaptive rate limiting based on response patterns
"""
import time
import random
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


def parse_retry_after(header_value: Optional[str]) -> Optional[int]:
    """
    Parse Retry-After header (seconds or HTTP-date).
    
    Args:
        header_value: Value of Retry-After header
        
    Returns:
        Seconds to wait, or None if header not parseable
    """
    if not header_value:
        return None
    
    try:
        # Try parsing as integer (seconds)
        return int(header_value)
    except ValueError:
        pass
    
    # Try parsing as HTTP-date
    try:
        from email.utils import parsedate_to_datetime
        retry_time = parsedate_to_datetime(header_value)
        delta = retry_time - datetime.now(retry_time.tzinfo)
        return max(0, int(delta.total_seconds()))
    except Exception:
        return None


@dataclass
class ExponentialBackoff:
    """
    Exponential backoff with jitter for rate limit recovery.
    
    Implements decorrelated jitter for more stable performance:
    delay = min(max_delay, random_between(base, previous_delay * 3))
    
    Example:
        backoff = ExponentialBackoff()
        for attempt in range(5):
            delay = backoff.get_delay(attempt)
            time.sleep(delay)
    """
    base_delay: float = 2.0
    max_delay: float = 300.0  # 5 minutes
    jitter_factor: float = 0.3
    multiplier: float = 2.0
    
    def get_delay(
        self,
        attempt: int,
        retry_after: Optional[int] = None,
    ) -> float:
        """
        Calculate delay with exponential backoff and jitter.
        
        Args:
            attempt: Attempt number (0-indexed)
            retry_after: Optional Retry-After header value (seconds)
            
        Returns:
            Delay in seconds
        """
        # If Retry-After is provided, respect it with small jitter
        if retry_after is not None and retry_after > 0:
            jitter = random.uniform(0, 5)
            return min(retry_after + jitter, self.max_delay)
        
        # Exponential backoff: base * multiplier^attempt
        delay = self.base_delay * (self.multiplier ** attempt)
        delay = min(delay, self.max_delay)
        
        # Add jitter: ±jitter_factor * delay
        jitter = delay * random.uniform(-self.jitter_factor, self.jitter_factor)
        delay = max(self.base_delay, delay + jitter)
        
        return delay
    
    def get_delay_decorrelated(self, previous_delay: float = None) -> float:
        """
        Decorrelated jitter variant (AWS best practice).
        
        delay = random_between(base, prev_delay * 3)
        """
        if previous_delay is None:
            previous_delay = self.base_delay
        
        delay = random.uniform(self.base_delay, previous_delay * 3)
        return min(delay, self.max_delay)


@dataclass
class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter for per-endpoint request throttling.
    
    Tracks requests in a sliding time window and enforces limits.
    
    Example:
        limiter = SlidingWindowRateLimiter(window_seconds=60, max_requests=40)
        delay = limiter.acquire("/rankings/5")
        time.sleep(delay)
        # Make request
    """
    window_seconds: int = 60
    max_requests: int = 40
    min_delay: float = 1.0  # Minimum delay between any requests
    
    # Internal state
    _requests: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    _lock: Lock = field(default_factory=Lock)
    _last_request_time: float = field(default=0.0)
    
    def acquire(self, endpoint: str = "global") -> float:
        """
        Calculate delay before request can proceed.
        
        Args:
            endpoint: Endpoint identifier for per-endpoint limiting
            
        Returns:
            Delay in seconds (0 if can proceed immediately)
        """
        with self._lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            # Remove old requests outside window
            self._requests[endpoint] = [
                t for t in self._requests[endpoint] if t > window_start
            ]
            
            # Check global minimum delay
            time_since_last = now - self._last_request_time
            min_delay_remaining = max(0, self.min_delay - time_since_last)
            
            # Check window limit
            if len(self._requests[endpoint]) >= self.max_requests:
                # Wait until oldest request falls out of window
                oldest = self._requests[endpoint][0]
                window_delay = (oldest + self.window_seconds) - now
                delay = max(min_delay_remaining, window_delay)
            else:
                delay = min_delay_remaining
            
            return delay
    
    def record_request(self, endpoint: str = "global"):
        """Record that a request was made."""
        with self._lock:
            now = time.time()
            self._requests[endpoint].append(now)
            self._last_request_time = now
    
    def get_stats(self, endpoint: str = "global") -> Dict:
        """Get current rate limiting stats."""
        with self._lock:
            now = time.time()
            window_start = now - self.window_seconds
            recent = [t for t in self._requests[endpoint] if t > window_start]
            
            return {
                "endpoint": endpoint,
                "requests_in_window": len(recent),
                "max_requests": self.max_requests,
                "utilization": len(recent) / self.max_requests,
            }


@dataclass
class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts based on API response patterns.
    
    - Decreases rate after 429/503 responses
    - Gradually increases rate on successful responses
    - Tracks per-endpoint health
    """
    base_delay: float = 1.5
    min_delay: float = 0.5
    max_delay: float = 10.0
    
    # Adaptation params
    increase_factor: float = 1.5  # Slow down by 50% on rate limit
    decrease_factor: float = 0.95  # Speed up by 5% on success
    success_threshold: int = 10  # Successes before speed up
    
    # State
    _current_delay: Dict[str, float] = field(default_factory=dict)
    _success_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _lock: Lock = field(default_factory=Lock)
    
    def get_delay(self, endpoint: str = "global") -> float:
        """Get current delay for endpoint."""
        with self._lock:
            return self._current_delay.get(endpoint, self.base_delay)
    
    def record_success(self, endpoint: str = "global"):
        """Record successful request, potentially speed up."""
        with self._lock:
            self._success_count[endpoint] += 1
            
            if self._success_count[endpoint] >= self.success_threshold:
                current = self._current_delay.get(endpoint, self.base_delay)
                new_delay = max(self.min_delay, current * self.decrease_factor)
                self._current_delay[endpoint] = new_delay
                self._success_count[endpoint] = 0
                
                if new_delay < current:
                    logger.debug(f"Adaptive rate limiter: decreased delay for {endpoint} to {new_delay:.2f}s")
    
    def record_rate_limit(self, endpoint: str = "global", retry_after: int = None):
        """Record rate limit hit, slow down."""
        with self._lock:
            current = self._current_delay.get(endpoint, self.base_delay)
            
            if retry_after:
                # Use retry_after as new baseline
                new_delay = min(retry_after / 10, self.max_delay)
            else:
                new_delay = min(current * self.increase_factor, self.max_delay)
            
            self._current_delay[endpoint] = new_delay
            self._success_count[endpoint] = 0
            
            logger.warning(f"Adaptive rate limiter: increased delay for {endpoint} to {new_delay:.2f}s")
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """Get stats for all endpoints."""
        with self._lock:
            return {
                endpoint: {
                    "delay": delay,
                    "success_count": self._success_count.get(endpoint, 0),
                }
                for endpoint, delay in self._current_delay.items()
            }


@dataclass
class EnhancedCircuitBreaker:
    """
    Enhanced circuit breaker with per-endpoint tracking and exponential backoff.
    
    States:
    - CLOSED: Normal operation
    - OPEN: All requests blocked, waiting for backoff
    - HALF_OPEN: Allowing probe requests to test recovery
    """
    failure_threshold: int = 3
    success_threshold: int = 2  # Successes needed in half-open to close
    base_backoff_seconds: float = 30.0
    max_backoff_seconds: float = 600.0  # 10 minutes
    
    # Per-endpoint state
    _state: Dict[str, str] = field(default_factory=lambda: defaultdict(lambda: "closed"))
    _failures: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _successes: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _last_failure: Dict[str, datetime] = field(default_factory=dict)
    _backoff_multiplier: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _lock: Lock = field(default_factory=Lock)
    
    def is_open(self, endpoint: str = "global") -> bool:
        """Check if circuit is open (blocking requests)."""
        with self._lock:
            state = self._state[endpoint]
            
            if state == "closed":
                return False
            
            if state == "open":
                last_failure = self._last_failure.get(endpoint)
                if last_failure:
                    multiplier = self._backoff_multiplier[endpoint]
                    backoff = min(
                        self.base_backoff_seconds * (2 ** multiplier),
                        self.max_backoff_seconds
                    )
                    
                    if datetime.now() - last_failure > timedelta(seconds=backoff):
                        self._state[endpoint] = "half_open"
                        self._successes[endpoint] = 0
                        logger.info(f"Circuit breaker for {endpoint} entering half-open state")
                        return False
                
                return True
            
            # Half-open allows requests
            return False
    
    def record_failure(self, endpoint: str = "global", status_code: int = 0):
        """Record a failure."""
        with self._lock:
            if status_code not in [403, 429, 503]:
                return
            
            self._failures[endpoint] += 1
            self._last_failure[endpoint] = datetime.now()
            
            if self._state[endpoint] == "half_open":
                # Any failure in half-open reopens circuit
                self._state[endpoint] = "open"
                self._backoff_multiplier[endpoint] += 1
                logger.warning(f"Circuit breaker for {endpoint} reopened (backoff x{self._backoff_multiplier[endpoint]})")
            elif self._failures[endpoint] >= self.failure_threshold:
                self._state[endpoint] = "open"
                logger.warning(f"Circuit breaker for {endpoint} opened after {self._failures[endpoint]} failures")
    
    def record_success(self, endpoint: str = "global"):
        """Record a success."""
        with self._lock:
            if self._state[endpoint] == "half_open":
                self._successes[endpoint] += 1
                
                if self._successes[endpoint] >= self.success_threshold:
                    self._state[endpoint] = "closed"
                    self._failures[endpoint] = 0
                    self._backoff_multiplier[endpoint] = max(0, self._backoff_multiplier[endpoint] - 1)
                    logger.info(f"Circuit breaker for {endpoint} closed")
            elif self._state[endpoint] == "closed":
                # Reset failure count on success
                self._failures[endpoint] = 0
    
    def get_status(self, endpoint: str = "global") -> Dict:
        """Get circuit breaker status for endpoint."""
        with self._lock:
            return {
                "state": self._state[endpoint],
                "failures": self._failures[endpoint],
                "backoff_multiplier": self._backoff_multiplier[endpoint],
            }
