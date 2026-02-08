"""
Strongly typed configuration using pydantic-settings.

All settings are validated at startup and loaded from:
1. Default values defined here
2. .env file (if present)
3. Environment variables (highest priority)

Environment variable naming:
- ScraperSettings: SCRAPER_DELAY_MIN, SCRAPER_RATE_LIMIT_MAX_REQUESTS, etc.
- ModelSettings: MODEL_CANARY_PERCENTAGE, MODEL_MIN_EDGE, etc.
- ObservabilitySettings: ENVIRONMENT, LOG_LEVEL (no prefix)
"""
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
from pathlib import Path


class ScraperSettings(BaseSettings):
    """Scraper and API settings."""
    
    model_config = SettingsConfigDict(env_prefix="SCRAPER_")
    
    # Request timing
    delay_min: float = Field(default=0.5, ge=0.1, description="Minimum delay between requests")
    delay_max: float = Field(default=2.0, ge=0.5, description="Maximum delay between requests")
    
    # Rate limiting
    rate_limit_window_s: int = Field(default=60, ge=10, description="Rate limit window in seconds")
    rate_limit_max_requests: int = Field(default=100, ge=10, description="Max requests per window")
    
    # Caching
    cache_ttl_seconds: int = Field(default=300, ge=0, description="Response cache TTL")
    cache_enabled: bool = Field(default=True)
    
    # Retries
    max_retries: int = Field(default=3, ge=1, le=10)
    
    # Circuit breaker
    circuit_breaker_threshold: int = Field(default=5, ge=1)
    circuit_breaker_timeout_s: int = Field(default=300, ge=60)
    
    # Archive
    archive_enabled: bool = Field(default=True)
    archive_retention_days: int = Field(default=30, ge=1)
    
    @field_validator('delay_max')
    @classmethod
    def delay_max_gte_min(cls, v, info):
        if 'delay_min' in info.data and v < info.data['delay_min']:
            raise ValueError('delay_max must be >= delay_min')
        return v


class ModelSettings(BaseSettings):
    """Model serving and prediction settings."""
    
    model_config = SettingsConfigDict(env_prefix="MODEL_")
    
    # Serving modes
    canary_percentage: float = Field(default=0.0, ge=0.0, le=1.0, description="Canary traffic split")
    shadow_mode: bool = Field(default=False, description="Enable shadow predictions")
    enable_fallback: bool = Field(default=True, description="Enable fallback to staging model")
    
    # Prediction thresholds
    min_confidence: float = Field(default=0.55, ge=0.5, le=1.0, description="Minimum prediction confidence")
    min_edge: float = Field(default=0.05, ge=0.0, le=0.5, description="Minimum edge for value bets")
    max_odds: float = Field(default=5.0, ge=1.0, description="Maximum odds to consider")
    min_odds: float = Field(default=1.3, ge=1.0, description="Minimum odds to consider")
    
    # Training
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_state: int = Field(default=42)


class ObservabilitySettings(BaseSettings):
    """Logging and metrics settings."""
    
    model_config = SettingsConfigDict(env_prefix="")  # Direct: ENVIRONMENT, LOG_LEVEL
    
    environment: str = Field(default="development", pattern="^(development|staging|production)$")
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_format: str = Field(default="console", pattern="^(console|json)$")
    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=8000, ge=1024, le=65535)


class BettingSettings(BaseSettings):
    """Bankroll and stake sizing settings."""
    
    model_config = SettingsConfigDict(env_prefix="BETTING_")
    
    # Kelly sizing
    kelly_fraction: float = Field(
        default=0.25, ge=0.0, le=1.0,
        description="Fractional Kelly (0.25 = quarter Kelly)"
    )
    
    # Per-bet caps
    max_bet_fraction: float = Field(
        default=0.01, ge=0.001, le=0.10,
        description="Max stake as fraction of bankroll (1%)"
    )
    min_bet_fraction: float = Field(
        default=0.005, ge=0.001, le=0.05,
        description="Min stake as fraction of bankroll (0.5%)"
    )
    
    # Daily caps
    max_daily_staked_fraction: float = Field(
        default=0.10, ge=0.01, le=0.50,
        description="Max total staked per day as fraction of bankroll (10%)"
    )
    max_bets_per_day: int = Field(
        default=10, ge=1, le=50,
        description="Maximum number of bets per day"
    )
    
    # Odds filters
    min_odds: float = Field(default=1.3, ge=1.01, description="Minimum odds")
    max_odds: float = Field(default=5.0, ge=1.5, description="Maximum odds")
    
    # Edge threshold
    min_edge: float = Field(default=0.05, ge=0.0, le=0.5, description="Minimum edge")


class FeatureSettings(BaseSettings):
    """Feature engineering settings."""
    
    model_config = SettingsConfigDict(env_prefix="FEATURE_")
    
    # Rolling windows
    rolling_windows: List[int] = Field(default=[5, 10, 20])
    min_matches: int = Field(default=3, ge=1)
    
    # Elo settings
    elo_k_factor: float = Field(default=32.0, ge=1.0, le=100.0)
    elo_initial: float = Field(default=1500.0, ge=1000.0, le=2000.0)
    
    # Feature flags
    enable_surface_features: bool = Field(default=True)
    enable_h2h_features: bool = Field(default=True)
    enable_fatigue_features: bool = Field(default=True)


class Settings(BaseSettings):
    """
    Root settings aggregating all subsections.
    
    Usage:
        from src.config import settings
        
        settings.scraper.delay_min
        settings.model.min_edge
        settings.betting.kelly_fraction
        settings.observability.log_level
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    scraper: ScraperSettings = Field(default_factory=ScraperSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    betting: BettingSettings = Field(default_factory=BettingSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    
    # Paths
    data_dir: Path = Field(default=Path("data"))
    models_dir: Path = Field(default=Path("models"))
    results_dir: Path = Field(default=Path("results"))

