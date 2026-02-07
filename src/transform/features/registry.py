"""
Feature Registry - Config-driven feature builder registration.

Provides a decorator-based pattern for registering feature builders
and a central function to apply enabled features in sequence.
"""
import polars as pl
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)

# Type alias for feature builder function
FeatureBuilder = Callable[[pl.LazyFrame], pl.LazyFrame]


@dataclass
class FeatureSpec:
    """Specification for a registered feature builder."""
    name: str
    builder: FeatureBuilder
    category: str
    description: str = ""
    enabled: bool = True
    priority: int = 100  # Lower = runs first
    dependencies: List[str] = field(default_factory=list)


class FeatureRegistry:
    """
    Central registry for feature builders.
    
    Usage:
        @FeatureRegistry.register("rolling_win_rate", category="rolling")
        def add_rolling_win_rate(df: pl.LazyFrame) -> pl.LazyFrame:
            ...
        
        # Apply all registered features
        df = FeatureRegistry.build_features(df)
    """
    _features: Dict[str, FeatureSpec] = {}
    _config: Optional[Dict] = None
    
    @classmethod
    def register(
        cls,
        name: str,
        category: str,
        priority: int = 100,
        dependencies: Optional[List[str]] = None,
        description: str = ""
    ):
        """
        Decorator to register a feature builder.
        
        Args:
            name: Unique feature name
            category: Feature category (rolling, surface, h2h, etc.)
            priority: Execution order (lower = earlier)
            dependencies: List of feature names that must run first
            description: Human-readable description
        """
        def decorator(func: FeatureBuilder) -> FeatureBuilder:
            spec = FeatureSpec(
                name=name,
                builder=func,
                category=category,
                description=description,
                priority=priority,
                dependencies=dependencies or []
            )
            cls._features[name] = spec
            logger.debug(f"Registered feature: {name} (category={category})")
            return func
        return decorator
    
    @classmethod
    def load_config(cls, config_path: str = "config/features.yaml") -> Dict:
        """Load feature configuration from YAML."""
        if cls._config is not None:
            return cls._config
        
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config not found: {config_path}, using defaults")
            cls._config = {}
            return cls._config
        
        with open(path, 'r') as f:
            cls._config = yaml.safe_load(f) or {}
        
        logger.info(f"Loaded feature config from {config_path}")
        return cls._config
    
    @classmethod
    def get_enabled_features(cls, config_path: str = "config/features.yaml") -> List[FeatureSpec]:
        """Get list of enabled features sorted by priority."""
        config = cls.load_config(config_path)
        
        # Check if feature is disabled in config
        disabled = set(config.get("disabled_features", []))
        
        enabled = [
            spec for name, spec in cls._features.items()
            if name not in disabled and spec.enabled
        ]
        
        # Sort by priority (lower first), then by name for stability
        enabled.sort(key=lambda s: (s.priority, s.name))
        
        return enabled
    
    @classmethod
    def build_features(
        cls,
        df: pl.LazyFrame,
        config_path: str = "config/features.yaml",
        categories: Optional[List[str]] = None
    ) -> pl.LazyFrame:
        """
        Apply all registered feature builders to the dataframe.
        
        Args:
            df: Input LazyFrame
            config_path: Path to features config YAML
            categories: Optional list of categories to include (None = all)
            
        Returns:
            LazyFrame with all features added
        """
        features = cls.get_enabled_features(config_path)
        
        if categories:
            features = [f for f in features if f.category in categories]
        
        logger.info(f"Building {len(features)} features")
        
        for spec in features:
            try:
                df = spec.builder(df)
                logger.debug(f"Applied feature: {spec.name}")
            except Exception as e:
                logger.error(f"Feature {spec.name} failed: {e}")
                raise
        
        return df
    
    @classmethod
    def list_features(cls) -> List[Dict[str, Any]]:
        """List all registered features."""
        return [
            {
                "name": spec.name,
                "category": spec.category,
                "description": spec.description,
                "enabled": spec.enabled,
                "priority": spec.priority
            }
            for spec in cls._features.values()
        ]
    
    @classmethod
    def clear(cls):
        """Clear all registered features (for testing)."""
        cls._features = {}
        cls._config = None
