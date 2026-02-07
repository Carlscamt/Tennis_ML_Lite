"""
Configuration module with strongly typed settings.

Usage:
    from src.config import settings
    
    print(settings.scraper.delay_min)
    print(settings.model.min_edge)
"""
from .settings import (
    Settings,
    ScraperSettings,
    ModelSettings,
    ObservabilitySettings,
    FeatureSettings,
)

# Singleton instance - validates on import
settings = Settings()

__all__ = [
    "settings",
    "Settings",
    "ScraperSettings",
    "ModelSettings",
    "ObservabilitySettings",
    "FeatureSettings",
]
