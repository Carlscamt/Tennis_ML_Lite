"""
Service Container - Simple dependency injection for swappable implementations.

Usage:
    from src.core import ServiceContainer
    
    # Get default implementations
    storage = ServiceContainer.get_storage()
    queue = ServiceContainer.get_queue()
    
    # Register custom implementations
    ServiceContainer.register_storage(S3Repository(bucket="my-bucket"))
    ServiceContainer.register_queue(RedisQueue(host="localhost"))
"""
from typing import Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .protocols import DataRepository, TaskQueueProtocol

logger = logging.getLogger(__name__)


class ServiceContainer:
    """
    Simple dependency injection container.
    
    Provides lazy initialization of default implementations
    and allows swapping to alternative implementations.
    """
    
    _storage: Optional["DataRepository"] = None
    _queue: Optional["TaskQueueProtocol"] = None
    
    @classmethod
    def get_storage(cls) -> "DataRepository":
        """Get the configured storage repository."""
        if cls._storage is None:
            from .storage import LocalParquetRepository
            cls._storage = LocalParquetRepository()
            logger.debug("Initialized default LocalParquetRepository")
        return cls._storage
    
    @classmethod
    def get_queue(cls) -> "TaskQueueProtocol":
        """Get the configured task queue."""
        if cls._queue is None:
            from src.utils import TaskQueue
            cls._queue = TaskQueue()
            logger.debug("Initialized default SQLite TaskQueue")
        return cls._queue
    
    @classmethod
    def register_storage(cls, storage: "DataRepository") -> None:
        """Register a custom storage implementation."""
        cls._storage = storage
        logger.info(f"Registered storage: {type(storage).__name__}")
    
    @classmethod
    def register_queue(cls, queue: "TaskQueueProtocol") -> None:
        """Register a custom queue implementation."""
        cls._queue = queue
        logger.info(f"Registered queue: {type(queue).__name__}")
    
    @classmethod
    def reset(cls) -> None:
        """Reset to defaults (for testing)."""
        cls._storage = None
        cls._queue = None
