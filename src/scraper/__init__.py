"""
Rate limiter module for scraper hardening.
"""
from .rate_limiter import (
    ExponentialBackoff,
    SlidingWindowRateLimiter,
    AdaptiveRateLimiter,
    parse_retry_after,
)
from .schema_monitor import (
    ScraperHealthMetrics,
    ResponseSchema,
    validate_response,
)
from .quarantine import (
    QuarantineManager,
    QuarantineRecord,
    get_quarantine_manager,
)
from .atomic_writer import (
    AtomicParquetWriter,
    WriteResult,
    get_atomic_writer,
)

# Import from the scraper.py module (sibling to this package)
# This resolves the naming conflict between src/scraper/ and src/scraper.py
import importlib.util
import sys
from pathlib import Path

_scraper_module_path = Path(__file__).parent.parent / "scraper.py"
_spec = importlib.util.spec_from_file_location("_scraper_module", _scraper_module_path)
_scraper_module = importlib.util.module_from_spec(_spec)
sys.modules["_scraper_module"] = _scraper_module
_spec.loader.exec_module(_scraper_module)

scrape_upcoming = _scraper_module.scrape_upcoming
scrape_players = _scraper_module.scrape_players

__all__ = [
    "ExponentialBackoff",
    "SlidingWindowRateLimiter",
    "AdaptiveRateLimiter",
    "parse_retry_after",
    "ScraperHealthMetrics",
    "ResponseSchema",
    "validate_response",
    "QuarantineManager",
    "QuarantineRecord",
    "get_quarantine_manager",
    "AtomicParquetWriter",
    "WriteResult",
    "get_atomic_writer",
    "scrape_upcoming",
    "scrape_players",
]
