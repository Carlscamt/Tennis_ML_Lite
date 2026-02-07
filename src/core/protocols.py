"""
Protocol definitions for storage and queue abstractions.

These protocols define the interfaces that concrete implementations must follow.
Using Protocol allows duck typing while still providing type checking support.
"""
from typing import Protocol, Optional, Dict, List, Any, runtime_checkable
import polars as pl


@runtime_checkable
class DataRepository(Protocol):
    """
    Storage abstraction for data persistence.
    
    Implementations:
    - LocalParquetRepository (default): Parquet files on local disk
    - Future: S3Repository, GCSRepository, PostgresRepository
    """
    
    def load_raw(self, partition: str) -> pl.DataFrame:
        """
        Load raw data from a partition.
        
        Args:
            partition: Partition identifier (e.g., "historical_atp_20260101")
            
        Returns:
            DataFrame with raw data
        """
        ...
    
    def save_raw(self, df: pl.DataFrame, partition: str) -> None:
        """
        Save raw data to a partition.
        
        Args:
            df: DataFrame to save
            partition: Partition identifier
        """
        ...
    
    def load_features(self, name: str = "features_dataset") -> pl.DataFrame:
        """
        Load processed features.
        
        Args:
            name: Feature set name
            
        Returns:
            DataFrame with features
        """
        ...
    
    def save_features(self, df: pl.DataFrame, name: str = "features_dataset") -> None:
        """
        Save processed features.
        
        Args:
            df: DataFrame with features
            name: Feature set name
        """
        ...
    
    def list_partitions(self, prefix: str = "") -> List[str]:
        """
        List available partitions.
        
        Args:
            prefix: Optional prefix filter
            
        Returns:
            List of partition identifiers
        """
        ...
    
    def exists(self, partition: str) -> bool:
        """Check if a partition exists."""
        ...


@runtime_checkable  
class TaskQueueProtocol(Protocol):
    """
    Queue abstraction for task management.
    
    Implementations:
    - TaskQueue (default): SQLite-backed queue
    - Future: RedisQueue, SQSQueue
    """
    
    def enqueue(
        self, 
        task_type: str, 
        payload: Dict[str, Any],
        task_id: Optional[str] = None
    ) -> str:
        """
        Add a task to the queue.
        
        Args:
            task_type: Category of task
            payload: Task data
            task_id: Optional custom ID
            
        Returns:
            Task ID
        """
        ...
    
    def dequeue(self, task_type: Optional[str] = None) -> Optional[Any]:
        """
        Get the next pending task.
        
        Args:
            task_type: Optional filter by type
            
        Returns:
            Task object or None
        """
        ...
    
    def ack(self, task_id: str, result: Optional[Dict] = None) -> None:
        """
        Acknowledge successful task completion.
        
        Args:
            task_id: Task to acknowledge
            result: Optional result data
        """
        ...
    
    def nack(self, task_id: str, error: str) -> None:
        """
        Negative acknowledge - task failed.
        
        Args:
            task_id: Task that failed
            error: Error message
        """
        ...
    
    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics (pending, processing, completed, failed)."""
        ...
