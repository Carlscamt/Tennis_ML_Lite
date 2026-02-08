"""
Multi-source data adapters for tennis data.
"""
from .sackmann import SackmannDataSource
from .canonical import CanonicalMatch, to_canonical
from .validator import CrossSourceValidator

__all__ = [
    "SackmannDataSource",
    "CanonicalMatch",
    "to_canonical",
    "CrossSourceValidator",
]
