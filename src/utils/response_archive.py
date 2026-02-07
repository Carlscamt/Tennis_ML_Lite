"""
Response Archive - Compressed storage for raw API responses.

Features:
- Gzip compression (70-80% size reduction)
- Date-organized directory structure
- Configurable retention period
- Replay/re-process historical responses
"""
import gzip
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import shutil

logger = logging.getLogger(__name__)


class ResponseArchive:
    """
    Compressed JSON archive for API responses.
    
    Preserves raw API data for future feature re-extraction without re-scraping.
    
    Structure:
        data/.archive/
            2026/
                02/
                    06/
                        rankings_5_1707235200.json.gz
                        event_12345_1707235201.json.gz
    
    Usage:
        archive = ResponseArchive()
        
        # Store response
        archive.store("rankings/5", {"rankingRows": [...]})
        
        # Retrieve
        data = archive.get("rankings/5")
        
        # Cleanup old data
        archive.cleanup(days=30)
    """
    
    DEFAULT_RETENTION_DAYS = 30
    
    def __init__(self, archive_dir: str = "data/.archive"):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_archive_path(self, endpoint: str, timestamp: Optional[datetime] = None) -> Path:
        """Generate archive path for an endpoint."""
        ts = timestamp or datetime.now()
        
        # Create date-based directory structure
        date_dir = self.archive_dir / ts.strftime("%Y/%m/%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        
        # Create safe filename from endpoint
        safe_name = endpoint.replace("/", "_").replace("?", "_").strip("_")
        # Add hash for uniqueness if endpoint is too long
        if len(safe_name) > 100:
            hash_suffix = hashlib.md5(endpoint.encode()).hexdigest()[:8]
            safe_name = f"{safe_name[:90]}_{hash_suffix}"
        
        filename = f"{safe_name}_{int(ts.timestamp() * 1000000)}.json.gz"
        return date_dir / filename
    
    def store(
        self, 
        endpoint: str, 
        data: Dict[str, Any],
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        Store API response in compressed archive.
        
        Args:
            endpoint: API endpoint (e.g., "/rankings/5")
            data: Response data to archive
            metadata: Optional extra metadata
            
        Returns:
            Path to archived file
        """
        now = datetime.now()
        archive_path = self._get_archive_path(endpoint, now)
        
        # Wrap data with metadata
        record = {
            "_archived_at": now.isoformat(),
            "_endpoint": endpoint,
            "_metadata": metadata or {},
            "data": data
        }
        
        try:
            # Compress and write
            json_bytes = json.dumps(record, ensure_ascii=False).encode('utf-8')
            with gzip.open(archive_path, 'wb') as f:
                f.write(json_bytes)
            
            # Log compression stats
            original_size = len(json_bytes)
            compressed_size = archive_path.stat().st_size
            ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            
            logger.debug(
                f"Archived {endpoint}: {original_size:,}B → {compressed_size:,}B ({ratio:.1f}% reduction)"
            )
            
            return archive_path
            
        except Exception as e:
            logger.error(f"Failed to archive {endpoint}: {e}")
            raise
    
    def get(self, path: Path) -> Optional[Dict]:
        """
        Retrieve archived response.
        
        Args:
            path: Path to archived file
            
        Returns:
            Archived data or None if not found
        """
        if not path.exists():
            return None
        
        try:
            with gzip.open(path, 'rb') as f:
                data = json.loads(f.read().decode('utf-8'))
            return data
        except Exception as e:
            logger.error(f"Failed to read archive {path}: {e}")
            return None
    
    def find_by_endpoint(
        self, 
        endpoint: str, 
        date: Optional[datetime] = None,
        limit: int = 10
    ) -> List[Path]:
        """
        Find archived responses for an endpoint.
        
        Args:
            endpoint: API endpoint to search for
            date: Optional date to filter by
            limit: Maximum results to return
            
        Returns:
            List of archive file paths (newest first)
        """
        safe_name = endpoint.replace("/", "_").replace("?", "_").strip("_")
        pattern = f"*{safe_name}*.json.gz"
        
        if date:
            search_dir = self.archive_dir / date.strftime("%Y/%m/%d")
            if not search_dir.exists():
                return []
            files = list(search_dir.glob(pattern))
        else:
            files = list(self.archive_dir.rglob(pattern))
        
        # Sort by modification time, newest first
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files[:limit]
    
    def get_latest(self, endpoint: str) -> Optional[Dict]:
        """Get the most recent archived response for an endpoint."""
        files = self.find_by_endpoint(endpoint, limit=1)
        if files:
            return self.get(files[0])
        return None
    
    def list_dates(self) -> List[str]:
        """List all dates with archived data."""
        dates = set()
        for year_dir in self.archive_dir.iterdir():
            if year_dir.is_dir() and year_dir.name.isdigit():
                for month_dir in year_dir.iterdir():
                    if month_dir.is_dir():
                        for day_dir in month_dir.iterdir():
                            if day_dir.is_dir():
                                dates.add(f"{year_dir.name}/{month_dir.name}/{day_dir.name}")
        return sorted(dates, reverse=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get archive statistics."""
        total_files = 0
        total_size = 0
        oldest_file = None
        newest_file = None
        
        for gzfile in self.archive_dir.rglob("*.json.gz"):
            total_files += 1
            total_size += gzfile.stat().st_size
            mtime = gzfile.stat().st_mtime
            
            if oldest_file is None or mtime < oldest_file[1]:
                oldest_file = (gzfile, mtime)
            if newest_file is None or mtime > newest_file[1]:
                newest_file = (gzfile, mtime)
        
        return {
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "oldest": datetime.fromtimestamp(oldest_file[1]).isoformat() if oldest_file else None,
            "newest": datetime.fromtimestamp(newest_file[1]).isoformat() if newest_file else None,
            "dates": len(self.list_dates())
        }
    
    def cleanup(self, days: int = None) -> int:
        """
        Remove archived data older than specified days.
        
        Args:
            days: Retention period in days (default: DEFAULT_RETENTION_DAYS)
            
        Returns:
            Number of files deleted
        """
        days = days or self.DEFAULT_RETENTION_DAYS
        cutoff = datetime.now() - timedelta(days=days)
        deleted = 0
        
        for year_dir in list(self.archive_dir.iterdir()):
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
                
            for month_dir in list(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                    
                for day_dir in list(month_dir.iterdir()):
                    if not day_dir.is_dir():
                        continue
                    
                    try:
                        dir_date = datetime.strptime(
                            f"{year_dir.name}/{month_dir.name}/{day_dir.name}",
                            "%Y/%m/%d"
                        )
                        
                        if dir_date < cutoff:
                            file_count = len(list(day_dir.glob("*.json.gz")))
                            shutil.rmtree(day_dir)
                            deleted += file_count
                            logger.info(f"Cleaned up {day_dir}: {file_count} files")
                    except ValueError:
                        continue
                
                # Remove empty month directories
                if month_dir.exists() and not any(month_dir.iterdir()):
                    month_dir.rmdir()
            
            # Remove empty year directories
            if year_dir.exists() and not any(year_dir.iterdir()):
                year_dir.rmdir()
        
        if deleted > 0:
            logger.info(f"Cleanup complete: {deleted} files deleted (retention: {days} days)")
        
        return deleted
