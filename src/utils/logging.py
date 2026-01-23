"""
Improved logging configuration for Tennis Betting ML Pipeline.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
import os


def setup_logging(
    name: str = "tennis_betting",
    level: Optional[str] = None,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Set up structured logging with environment-aware configuration.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to env var LOG_LEVEL
        log_file: Optional file path for log output
    
    Returns:
        Configured logger instance
    """
    # Get log level from environment or parameter
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Determine if we're in production
    env = os.getenv("ENVIRONMENT", "development").lower()
    is_production = env == "production"
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    if is_production:
        # JSON logging for production (structured logs)
        try:
            from pythonjsonlogger import jsonlogger
            formatter = jsonlogger.JsonFormatter(
                fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S"
            )
        except ImportError:
            # Fallback if pythonjsonlogger not available
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
    else:
        # Human-readable for development
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
