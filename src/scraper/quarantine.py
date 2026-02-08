"""
Quarantine system for failed data validations.

Stores records that fail schema validation for manual inspection,
preventing bad data from polluting the training set.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from threading import Lock
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class QuarantineRecord:
    """A single quarantined record with metadata."""
    record: Dict[str, Any]
    reason: str
    source: str
    timestamp: str
    record_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "record": self.record,
            "reason": self.reason,
            "source": self.source,
            "timestamp": self.timestamp,
            "record_id": self.record_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "QuarantineRecord":
        return cls(
            record=data.get("record", {}),
            reason=data.get("reason", "unknown"),
            source=data.get("source", "unknown"),
            timestamp=data.get("timestamp", ""),
            record_id=data.get("record_id"),
        )


@dataclass
class QuarantineManager:
    """
    Manages quarantine area for failed validations.
    
    Stores records as JSONL (JSON Lines) files organized by date and source.
    Each line is a complete JSON object with record data and failure metadata.
    
    Directory structure:
        quarantine_dir/
        ├── 2024-01-15/
        │   ├── rankings.jsonl
        │   ├── matches.jsonl
        │   └── odds.jsonl
        └── 2024-01-16/
            └── matches.jsonl
    
    Example:
        manager = QuarantineManager()
        manager.quarantine_row({"event_id": 123}, "missing odds", "matches")
        
        # Later inspection
        records = manager.list_quarantined("matches", days=7)
    """
    quarantine_dir: Path = field(default_factory=lambda: Path("data/raw/quarantine"))
    max_records_per_file: int = 10000
    _lock: Lock = field(default_factory=Lock)
    
    def __post_init__(self):
        self.quarantine_dir = Path(self.quarantine_dir)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_file_path(self, source: str, date_str: str = None) -> Path:
        """Get path for quarantine file."""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        date_dir = self.quarantine_dir / date_str
        date_dir.mkdir(parents=True, exist_ok=True)
        
        return date_dir / f"{source}.jsonl"
    
    def quarantine_row(
        self,
        record: Dict[str, Any],
        reason: str,
        source: str,
        record_id: str = None,
    ) -> bool:
        """
        Quarantine a single failed record.
        
        Args:
            record: The data record that failed validation
            reason: Human-readable failure reason
            source: Data source (e.g., "rankings", "matches", "odds")
            record_id: Optional unique ID for the record
            
        Returns:
            True if successfully quarantined
        """
        try:
            qr = QuarantineRecord(
                record=record,
                reason=reason,
                source=source,
                timestamp=datetime.now().isoformat(),
                record_id=record_id or str(record.get("event_id", "unknown")),
            )
            
            file_path = self._get_file_path(source)
            
            with self._lock:
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(qr.to_dict()) + "\n")
            
            logger.debug(f"Quarantined record from {source}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to quarantine record: {e}")
            return False
    
    def quarantine_dataframe(
        self,
        df: pl.DataFrame,
        errors: List[str],
        source: str,
    ) -> int:
        """
        Quarantine multiple rows from a DataFrame.
        
        Args:
            df: DataFrame with failed rows
            errors: List of error messages for context
            source: Data source identifier
            
        Returns:
            Number of rows quarantined
        """
        if df.is_empty():
            return 0
        
        count = 0
        reason = "; ".join(errors[:3])  # Limit reason length
        
        for record in df.to_dicts():
            record_id = f"{record.get('event_id', 'unknown')}_{record.get('player_id', 'unknown')}"
            if self.quarantine_row(record, reason, source, record_id):
                count += 1
        
        logger.info(f"Quarantined {count} rows from {source}")
        return count
    
    def get_quarantined_count(self, source: str = None, days: int = 7) -> int:
        """
        Count records in quarantine.
        
        Args:
            source: Filter by source (None = all sources)
            days: Look back this many days
            
        Returns:
            Total count of quarantined records
        """
        count = 0
        
        for date_dir in self._get_recent_dirs(days):
            if source:
                files = [date_dir / f"{source}.jsonl"]
            else:
                files = list(date_dir.glob("*.jsonl"))
            
            for file_path in files:
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        count += sum(1 for _ in f)
        
        return count
    
    def list_quarantined(
        self,
        source: str = None,
        days: int = 7,
        limit: int = 100,
    ) -> List[QuarantineRecord]:
        """
        List quarantined records.
        
        Args:
            source: Filter by source (None = all sources)
            days: Look back this many days
            limit: Maximum records to return
            
        Returns:
            List of QuarantineRecord objects
        """
        records = []
        
        for date_dir in self._get_recent_dirs(days):
            if source:
                files = [date_dir / f"{source}.jsonl"]
            else:
                files = list(date_dir.glob("*.jsonl"))
            
            for file_path in files:
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if len(records) >= limit:
                                return records
                            try:
                                data = json.loads(line.strip())
                                records.append(QuarantineRecord.from_dict(data))
                            except json.JSONDecodeError:
                                continue
        
        return records
    
    def _get_recent_dirs(self, days: int) -> List[Path]:
        """Get date directories for recent days."""
        dirs = []
        for i in range(days):
            date = datetime.now() - __import__("datetime").timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            date_dir = self.quarantine_dir / date_str
            if date_dir.exists():
                dirs.append(date_dir)
        return dirs
    
    def get_summary(self, days: int = 7) -> Dict:
        """Get quarantine summary statistics."""
        summary = {
            "total_count": 0,
            "by_source": {},
            "by_date": {},
        }
        
        for date_dir in self._get_recent_dirs(days):
            date_str = date_dir.name
            summary["by_date"][date_str] = 0
            
            for file_path in date_dir.glob("*.jsonl"):
                source = file_path.stem
                count = sum(1 for _ in open(file_path, "r", encoding="utf-8"))
                
                summary["total_count"] += count
                summary["by_date"][date_str] += count
                
                if source not in summary["by_source"]:
                    summary["by_source"][source] = 0
                summary["by_source"][source] += count
        
        return summary
    
    def clear_old(self, days_to_keep: int = 30) -> int:
        """
        Remove quarantine files older than specified days.
        
        Args:
            days_to_keep: Keep files from this many recent days
            
        Returns:
            Number of files removed
        """
        import shutil
        
        cutoff = datetime.now() - __import__("datetime").timedelta(days=days_to_keep)
        removed = 0
        
        for date_dir in self.quarantine_dir.iterdir():
            if not date_dir.is_dir():
                continue
            
            try:
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                if dir_date < cutoff:
                    shutil.rmtree(date_dir)
                    removed += 1
                    logger.info(f"Removed old quarantine dir: {date_dir.name}")
            except ValueError:
                continue
        
        return removed


# Global instance
_quarantine_manager = None


def get_quarantine_manager() -> QuarantineManager:
    """Get global quarantine manager instance."""
    global _quarantine_manager
    if _quarantine_manager is None:
        _quarantine_manager = QuarantineManager()
    return _quarantine_manager
