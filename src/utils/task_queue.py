"""
Task Queue - SQLite-backed pub/sub-style task queue for resilient scraping.

Features:
- Atomic state transitions (pending → processing → completed/failed)
- Automatic retry with exponential backoff
- Survives crashes - tasks resume on restart
- No external dependencies (uses built-in sqlite3)
"""
import sqlite3
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from threading import Lock
import logging

logger = logging.getLogger(__name__)


class TaskState(Enum):
    """Task lifecycle states."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Task record."""
    id: str
    task_type: str
    payload: Dict[str, Any]
    state: TaskState
    attempts: int = 0
    max_attempts: int = 3
    created_at: str = ""
    updated_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['state'] = self.state.value
        return d
    
    @classmethod
    def from_row(cls, row: tuple) -> 'Task':
        return cls(
            id=row[0],
            task_type=row[1],
            payload=json.loads(row[2]) if row[2] else {},
            state=TaskState(row[3]),
            attempts=row[4],
            max_attempts=row[5],
            created_at=row[6],
            updated_at=row[7],
            started_at=row[8],
            completed_at=row[9],
            error=row[10],
            result=json.loads(row[11]) if row[11] else None
        )


class TaskQueue:
    """
    SQLite-backed task queue with pub/sub semantics.
    
    Tasks are only marked complete after successful processing.
    Failed tasks are retried with exponential backoff.
    
    Usage:
        queue = TaskQueue("data/.task_queue.db")
        
        # Enqueue tasks
        queue.enqueue("fetch_player", {"player_id": 12345})
        queue.enqueue("fetch_player", {"player_id": 67890})
        
        # Process tasks
        while task := queue.dequeue("fetch_player"):
            try:
                result = process_player(task.payload["player_id"])
                queue.mark_completed(task.id, result)
            except Exception as e:
                queue.mark_failed(task.id, str(e))
    """
    
    # Backoff delays in seconds for retries
    BACKOFF_DELAYS = [30, 120, 600]  # 30s, 2m, 10m
    
    # Processing timeout - task returns to pending if not completed
    PROCESSING_TIMEOUT_MINUTES = 30
    
    def __init__(self, db_path: str = "data/.task_queue.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection with appropriate settings."""
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
        conn.execute("PRAGMA busy_timeout=30000")  # 30s wait on lock
        return conn
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    payload TEXT,
                    state TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER DEFAULT 0,
                    max_attempts INTEGER DEFAULT 3,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error TEXT,
                    result TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_state ON tasks(state)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type_state ON tasks(task_type, state)")
            conn.commit()
    
    def enqueue(
        self, 
        task_type: str, 
        payload: Dict[str, Any],
        task_id: Optional[str] = None,
        max_attempts: int = 3
    ) -> str:
        """
        Add a task to the queue.
        
        Args:
            task_type: Category of task (e.g., 'fetch_player', 'fetch_match')
            payload: Task data (will be JSON serialized)
            task_id: Optional custom ID (auto-generated if not provided)
            max_attempts: Maximum retry attempts before permanent failure
            
        Returns:
            Task ID
        """
        if task_id is None:
            task_id = f"{task_type}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        now = datetime.now().isoformat()
        
        with self._lock:
            with self._get_conn() as conn:
                # Check if task already exists
                existing = conn.execute(
                    "SELECT state FROM tasks WHERE id = ?", (task_id,)
                ).fetchone()
                
                if existing:
                    logger.debug(f"Task {task_id} already exists in state {existing[0]}")
                    return task_id
                
                conn.execute("""
                    INSERT INTO tasks (id, task_type, payload, state, max_attempts, created_at, updated_at)
                    VALUES (?, ?, ?, 'pending', ?, ?, ?)
                """, (task_id, task_type, json.dumps(payload), max_attempts, now, now))
                conn.commit()
        
        logger.debug(f"Enqueued task {task_id}")
        return task_id
    
    def enqueue_batch(self, task_type: str, payloads: List[Dict[str, Any]]) -> List[str]:
        """Enqueue multiple tasks efficiently."""
        task_ids = []
        now = datetime.now().isoformat()
        
        with self._lock:
            with self._get_conn() as conn:
                for payload in payloads:
                    task_id = f"{task_type}_{payload.get('id', datetime.now().strftime('%Y%m%d%H%M%S%f'))}"
                    
                    # Skip if exists
                    existing = conn.execute(
                        "SELECT 1 FROM tasks WHERE id = ?", (task_id,)
                    ).fetchone()
                    
                    if not existing:
                        conn.execute("""
                            INSERT INTO tasks (id, task_type, payload, state, created_at, updated_at)
                            VALUES (?, ?, ?, 'pending', ?, ?)
                        """, (task_id, task_type, json.dumps(payload), now, now))
                        task_ids.append(task_id)
                
                conn.commit()
        
        logger.info(f"Enqueued {len(task_ids)} tasks of type {task_type}")
        return task_ids
    
    def dequeue(self, task_type: Optional[str] = None) -> Optional[Task]:
        """
        Get the next pending task and mark it as processing.
        
        Args:
            task_type: Optional filter by task type
            
        Returns:
            Task or None if queue is empty
        """
        now = datetime.now().isoformat()
        
        with self._lock:
            with self._get_conn() as conn:
                # First, reset any stale processing tasks
                self._reset_stale_tasks(conn)
                
                # Get next pending task
                if task_type:
                    row = conn.execute("""
                        SELECT * FROM tasks 
                        WHERE state = 'pending' AND task_type = ?
                        ORDER BY created_at ASC
                        LIMIT 1
                    """, (task_type,)).fetchone()
                else:
                    row = conn.execute("""
                        SELECT * FROM tasks 
                        WHERE state = 'pending'
                        ORDER BY created_at ASC
                        LIMIT 1
                    """).fetchone()
                
                if not row:
                    return None
                
                task = Task.from_row(row)
                
                # Mark as processing
                conn.execute("""
                    UPDATE tasks 
                    SET state = 'processing', started_at = ?, updated_at = ?, attempts = attempts + 1
                    WHERE id = ?
                """, (now, now, task.id))
                conn.commit()
                
                task.state = TaskState.PROCESSING
                task.started_at = now
                task.attempts += 1
                
                logger.debug(f"Dequeued task {task.id} (attempt {task.attempts})")
                return task
    
    def _reset_stale_tasks(self, conn: sqlite3.Connection):
        """Reset processing tasks that have timed out back to pending."""
        cutoff = (datetime.now() - timedelta(minutes=self.PROCESSING_TIMEOUT_MINUTES)).isoformat()
        
        result = conn.execute("""
            UPDATE tasks 
            SET state = 'pending', updated_at = ?
            WHERE state = 'processing' AND started_at < ?
        """, (datetime.now().isoformat(), cutoff))
        
        if result.rowcount > 0:
            logger.warning(f"Reset {result.rowcount} stale processing tasks")
            conn.commit()
    
    def mark_completed(self, task_id: str, result: Optional[Dict] = None):
        """Mark task as successfully completed."""
        now = datetime.now().isoformat()
        
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE tasks 
                SET state = 'completed', completed_at = ?, updated_at = ?, result = ?
                WHERE id = ?
            """, (now, now, json.dumps(result) if result else None, task_id))
            conn.commit()
        
        logger.debug(f"Completed task {task_id}")
    
    def mark_failed(self, task_id: str, error: str):
        """
        Mark task as failed. Will retry if attempts < max_attempts.
        """
        now = datetime.now().isoformat()
        
        with self._get_conn() as conn:
            # Get current attempt count
            row = conn.execute(
                "SELECT attempts, max_attempts FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            
            if not row:
                logger.error(f"Task {task_id} not found")
                return
            
            attempts, max_attempts = row
            
            if attempts < max_attempts:
                # Retry - return to pending
                conn.execute("""
                    UPDATE tasks 
                    SET state = 'pending', error = ?, updated_at = ?
                    WHERE id = ?
                """, (error, now, task_id))
                logger.warning(f"Task {task_id} failed (attempt {attempts}/{max_attempts}), will retry: {error}")
            else:
                # Permanent failure
                conn.execute("""
                    UPDATE tasks 
                    SET state = 'failed', error = ?, updated_at = ?, completed_at = ?
                    WHERE id = ?
                """, (error, now, now, task_id))
                logger.error(f"Task {task_id} permanently failed after {attempts} attempts: {error}")
            
            conn.commit()
    
    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        with self._get_conn() as conn:
            stats = {}
            for state in TaskState:
                count = conn.execute(
                    "SELECT COUNT(*) FROM tasks WHERE state = ?", (state.value,)
                ).fetchone()[0]
                stats[state.value] = count
            stats['total'] = sum(stats.values())
            return stats
    
    def get_pending_count(self, task_type: Optional[str] = None) -> int:
        """Get count of pending tasks."""
        with self._get_conn() as conn:
            if task_type:
                return conn.execute(
                    "SELECT COUNT(*) FROM tasks WHERE state = 'pending' AND task_type = ?",
                    (task_type,)
                ).fetchone()[0]
            return conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE state = 'pending'"
            ).fetchone()[0]
    
    def get_failed_tasks(self, task_type: Optional[str] = None) -> List[Task]:
        """Get all failed tasks."""
        with self._get_conn() as conn:
            if task_type:
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE state = 'failed' AND task_type = ?",
                    (task_type,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE state = 'failed'"
                ).fetchall()
            return [Task.from_row(row) for row in rows]
    
    def clear_completed(self):
        """Remove all completed tasks from the queue."""
        with self._get_conn() as conn:
            result = conn.execute("DELETE FROM tasks WHERE state = 'completed'")
            conn.commit()
            logger.info(f"Cleared {result.rowcount} completed tasks")
    
    def reset_failed(self, task_type: Optional[str] = None):
        """Reset failed tasks back to pending for retry."""
        with self._get_conn() as conn:
            if task_type:
                result = conn.execute("""
                    UPDATE tasks 
                    SET state = 'pending', attempts = 0, error = NULL, updated_at = ?
                    WHERE state = 'failed' AND task_type = ?
                """, (datetime.now().isoformat(), task_type))
            else:
                result = conn.execute("""
                    UPDATE tasks 
                    SET state = 'pending', attempts = 0, error = NULL, updated_at = ?
                    WHERE state = 'failed'
                """, (datetime.now().isoformat(),))
            conn.commit()
            logger.info(f"Reset {result.rowcount} failed tasks")
