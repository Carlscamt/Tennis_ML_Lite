# tests/unit/test_resilience.py
"""
Tests for resilience infrastructure: TaskQueue and ResponseArchive.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from src.utils.task_queue import TaskQueue, Task, TaskState
from src.utils.response_archive import ResponseArchive


class TestTaskQueue:
    """Test task queue functionality."""
    
    @pytest.fixture
    def queue(self, tmp_path):
        """Create a temporary task queue."""
        db_path = tmp_path / "test_queue.db"
        return TaskQueue(str(db_path))
    
    def test_enqueue_creates_task(self, queue):
        """Enqueue adds task to queue."""
        task_id = queue.enqueue("fetch_player", {"player_id": 12345})
        
        assert task_id is not None
        stats = queue.get_stats()
        assert stats["pending"] == 1
    
    def test_dequeue_returns_task(self, queue):
        """Dequeue returns pending task and marks it processing."""
        queue.enqueue("fetch_player", {"player_id": 12345})
        
        task = queue.dequeue("fetch_player")
        
        assert task is not None
        assert task.payload["player_id"] == 12345
        assert task.state == TaskState.PROCESSING
    
    def test_dequeue_empty_returns_none(self, queue):
        """Dequeue on empty queue returns None."""
        result = queue.dequeue("fetch_player")
        assert result is None
    
    def test_mark_completed_updates_state(self, queue):
        """Marking task completed moves it to completed state."""
        queue.enqueue("fetch_player", {"player_id": 12345})
        task = queue.dequeue("fetch_player")
        
        queue.mark_completed(task.id, {"matches": 10})
        
        stats = queue.get_stats()
        assert stats["completed"] == 1
        assert stats["processing"] == 0
    
    def test_mark_failed_retries(self, queue):
        """Failed task returns to pending for retry."""
        task_id = queue.enqueue("fetch_player", {"player_id": 12345}, max_attempts=3)
        task = queue.dequeue("fetch_player")
        
        queue.mark_failed(task.id, "Connection error")
        
        stats = queue.get_stats()
        # Should be back to pending, not failed (still has retries)
        assert stats["pending"] == 1
        assert stats["failed"] == 0
    
    def test_max_retries_leads_to_failure(self, queue):
        """Task permanently fails after max attempts."""
        task_id = queue.enqueue("fetch_player", {"player_id": 12345}, max_attempts=2)
        
        # First attempt
        task = queue.dequeue("fetch_player")
        queue.mark_failed(task.id, "Error 1")
        
        # Second attempt (final)
        task = queue.dequeue("fetch_player")
        queue.mark_failed(task.id, "Error 2")
        
        stats = queue.get_stats()
        assert stats["failed"] == 1
        assert stats["pending"] == 0
    
    def test_enqueue_batch(self, queue):
        """Batch enqueue adds multiple tasks."""
        payloads = [
            {"id": "player_1", "name": "Player 1"},
            {"id": "player_2", "name": "Player 2"},
            {"id": "player_3", "name": "Player 3"},
        ]
        
        task_ids = queue.enqueue_batch("fetch_player", payloads)
        
        assert len(task_ids) == 3
        assert queue.get_pending_count("fetch_player") == 3
    
    def test_duplicate_task_not_added(self, queue):
        """Same task ID is not added twice."""
        queue.enqueue("fetch_player", {"player_id": 12345}, task_id="player_12345")
        queue.enqueue("fetch_player", {"player_id": 12345}, task_id="player_12345")
        
        stats = queue.get_stats()
        assert stats["total"] == 1
    
    def test_get_failed_tasks(self, queue):
        """Can retrieve all failed tasks."""
        queue.enqueue("fetch_player", {"player_id": 1}, task_id="t1", max_attempts=1)
        queue.enqueue("fetch_player", {"player_id": 2}, task_id="t2", max_attempts=1)
        
        task1 = queue.dequeue("fetch_player")
        queue.mark_failed(task1.id, "Error 1")
        task2 = queue.dequeue("fetch_player")
        queue.mark_failed(task2.id, "Error 2")
        
        failed = queue.get_failed_tasks()
        assert len(failed) == 2
    
    def test_reset_failed(self, queue):
        """Can reset failed tasks for retry."""
        queue.enqueue("fetch_player", {"player_id": 1}, task_id="t1", max_attempts=1)
        task = queue.dequeue("fetch_player")
        queue.mark_failed(task.id, "Error")
        
        queue.reset_failed()
        
        stats = queue.get_stats()
        assert stats["pending"] == 1
        assert stats["failed"] == 0


class TestResponseArchive:
    """Test response archive functionality."""
    
    @pytest.fixture
    def archive(self, tmp_path):
        """Create a temporary archive."""
        return ResponseArchive(str(tmp_path / "archive"))
    
    def test_store_creates_file(self, archive):
        """Storing response creates compressed file."""
        data = {"rankingRows": [{"position": 1, "name": "Djokovic"}]}
        
        path = archive.store("/rankings/5", data)
        
        assert path.exists()
        assert path.suffix == ".gz"
    
    def test_store_and_retrieve(self, archive):
        """Can retrieve stored data."""
        original = {"events": [{"id": 123, "name": "Match"}]}
        path = archive.store("/events/123", original)
        
        retrieved = archive.get(path)
        
        assert retrieved["data"] == original
        assert "_archived_at" in retrieved
        assert "_endpoint" in retrieved
    
    def test_compression_reduces_size(self, archive):
        """Compression significantly reduces file size."""
        # Create large data
        large_data = {"items": [{"id": i, "value": "x" * 100} for i in range(100)]}
        
        path = archive.store("/test/large", large_data)
        
        import json
        original_size = len(json.dumps(large_data).encode())
        compressed_size = path.stat().st_size
        
        # Should achieve at least 50% compression
        assert compressed_size < original_size * 0.5
    
    def test_find_by_endpoint(self, archive):
        """Can find archived files by endpoint."""
        import time
        archive.store("/rankings/5", {"data": 1})
        time.sleep(0.01)  # Ensure different timestamps
        archive.store("/rankings/5", {"data": 2})
        time.sleep(0.01)
        archive.store("/events/123", {"data": 3})
        
        found = archive.find_by_endpoint("/rankings/5")
        
        assert len(found) == 2
    
    def test_get_latest(self, archive):
        """Get most recent archive for endpoint."""
        archive.store("/rankings/5", {"version": "old"})
        import time
        time.sleep(0.01)  # Ensure different timestamps
        archive.store("/rankings/5", {"version": "new"})
        
        latest = archive.get_latest("/rankings/5")
        
        assert latest["data"]["version"] == "new"
    
    def test_get_stats(self, archive):
        """Can get archive statistics."""
        archive.store("/test/1", {"a": 1})
        archive.store("/test/2", {"b": 2})
        
        stats = archive.get_stats()
        
        assert stats["total_files"] == 2
        assert stats["total_size_mb"] >= 0
    
    def test_cleanup_removes_old_files(self, archive, tmp_path):
        """Cleanup removes files older than retention period."""
        # Create old directory structure manually
        old_date = datetime.now() - timedelta(days=35)
        old_dir = Path(archive.archive_dir) / old_date.strftime("%Y/%m/%d")
        old_dir.mkdir(parents=True)
        old_file = old_dir / "old_data.json.gz"
        old_file.write_bytes(b"test")
        
        # Create recent file
        archive.store("/test/recent", {"recent": True})
        
        # Cleanup with 30 day retention
        deleted = archive.cleanup(days=30)
        
        assert deleted == 1
        assert not old_file.exists()
    
    def test_metadata_preserved(self, archive):
        """Custom metadata is preserved."""
        metadata = {"source": "test", "version": "1.0"}
        path = archive.store("/test/meta", {"data": 1}, metadata=metadata)
        
        retrieved = archive.get(path)
        
        assert retrieved["_metadata"] == metadata
