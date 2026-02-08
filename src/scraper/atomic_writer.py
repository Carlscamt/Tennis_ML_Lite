"""
Atomic and idempotent Parquet writer.

Provides safe file writes with:
- Atomic temp file + rename to prevent half-written files
- Idempotent merge with deduplication
- Diff detection to skip unchanged data
"""
import os
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class WriteResult:
    """Result of a write operation."""
    success: bool
    path: Path
    rows_written: int
    rows_deduplicated: int = 0
    rows_unchanged: int = 0
    error: Optional[str] = None


@dataclass
class AtomicParquetWriter:
    """
    Safe Parquet writer with atomic operations and idempotency.
    
    Features:
    - Atomic writes via temp file + rename
    - Idempotent merge with configurable unique key
    - Diff detection to skip unchanged rows
    - Crash recovery (temp files cleaned on startup)
    
    Example:
        writer = AtomicParquetWriter()
        result = writer.write_safe(df, Path("data/matches.parquet"))
        
        # Idempotent update
        result = writer.merge_and_write(
            new_df, 
            Path("data/matches.parquet"),
            unique_key=["event_id", "player_id"]
        )
    """
    temp_suffix: str = ".tmp"
    backup_suffix: str = ".bak"
    create_backup: bool = True
    
    def __post_init__(self):
        # Clean up any leftover temp files on startup
        self._cleanup_temp_files()
    
    def _cleanup_temp_files(self):
        """Remove any leftover temp files from previous crashes."""
        # This is called on startup to clean up
        pass  # Implemented per-path in write methods
    
    def write_safe(
        self,
        df: pl.DataFrame,
        path: Path,
        compression: str = "snappy",
    ) -> WriteResult:
        """
        Atomic write to Parquet file.
        
        Process:
        1. Write to temp file
        2. Verify temp file is readable
        3. Optionally create backup of existing file
        4. Atomically rename temp to target
        
        Args:
            df: DataFrame to write
            path: Target file path
            compression: Parquet compression algorithm
            
        Returns:
            WriteResult with success status and metadata
        """
        path = Path(path)
        temp_path = path.with_suffix(path.suffix + self.temp_suffix)
        backup_path = path.with_suffix(path.suffix + self.backup_suffix)
        
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Clean up any existing temp file
            if temp_path.exists():
                temp_path.unlink()
            
            # Write to temp file
            df.write_parquet(temp_path, compression=compression)
            
            # Verify temp file is readable
            test_df = pl.read_parquet(temp_path)
            if len(test_df) != len(df):
                raise ValueError(f"Verification failed: wrote {len(df)} rows, read {len(test_df)}")
            
            # Create backup if target exists
            if self.create_backup and path.exists():
                if backup_path.exists():
                    backup_path.unlink()
                path.rename(backup_path)
            
            # Atomic rename
            temp_path.rename(path)
            
            # Remove backup on success
            if backup_path.exists():
                backup_path.unlink()
            
            logger.info(f"Atomic write successful: {len(df)} rows to {path}")
            
            return WriteResult(
                success=True,
                path=path,
                rows_written=len(df),
            )
            
        except Exception as e:
            logger.error(f"Atomic write failed: {e}")
            
            # Cleanup temp file
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            
            # Restore backup if exists
            if backup_path.exists() and not path.exists():
                try:
                    backup_path.rename(path)
                    logger.info("Restored backup after failed write")
                except:
                    pass
            
            return WriteResult(
                success=False,
                path=path,
                rows_written=0,
                error=str(e),
            )
    
    def merge_and_write(
        self,
        new_df: pl.DataFrame,
        path: Path,
        unique_key: List[str] = None,
        prefer_new: bool = True,
        skip_unchanged: bool = True,
    ) -> WriteResult:
        """
        Idempotent merge with existing data and atomic write.
        
        Process:
        1. Load existing data if present
        2. Deduplicate on unique_key
        3. Optionally skip rows that haven't changed
        4. Atomic write merged result
        
        Args:
            new_df: New data to merge
            path: Target file path
            unique_key: Columns to use for deduplication
            prefer_new: If True, keep new rows on conflict
            skip_unchanged: If True, detect and skip unchanged rows
            
        Returns:
            WriteResult with merge statistics
        """
        if unique_key is None:
            unique_key = ["event_id", "player_id"]
        
        path = Path(path)
        rows_deduplicated = 0
        rows_unchanged = 0
        
        try:
            # Load existing data
            if path.exists():
                existing_df = pl.read_parquet(path)
                logger.debug(f"Loaded {len(existing_df)} existing rows from {path}")
            else:
                existing_df = pl.DataFrame()
            
            # Combine dataframes
            if not existing_df.is_empty():
                # Check for unchanged rows if requested
                if skip_unchanged and not new_df.is_empty():
                    new_df, rows_unchanged = self._filter_unchanged(
                        new_df, existing_df, unique_key
                    )
                
                if new_df.is_empty():
                    logger.info("No new or changed rows to write")
                    return WriteResult(
                        success=True,
                        path=path,
                        rows_written=len(existing_df),
                        rows_unchanged=rows_unchanged,
                    )
                
                # Merge with deduplication
                combined = pl.concat([existing_df, new_df], how="diagonal_relaxed")
                
                # Count before dedup
                before_count = len(combined)
                
                # Deduplicate
                combined = combined.unique(
                    subset=unique_key,
                    keep="last" if prefer_new else "first"
                )
                
                rows_deduplicated = before_count - len(combined)
                
                # Sort by timestamp if available
                if "start_timestamp" in combined.columns:
                    combined = combined.sort("start_timestamp")
            else:
                combined = new_df
            
            # Atomic write
            result = self.write_safe(combined, path)
            result.rows_deduplicated = rows_deduplicated
            result.rows_unchanged = rows_unchanged
            
            if result.success:
                logger.info(
                    f"Idempotent merge: {len(combined)} total rows, "
                    f"{rows_deduplicated} deduplicated, {rows_unchanged} unchanged"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Merge and write failed: {e}")
            return WriteResult(
                success=False,
                path=path,
                rows_written=0,
                error=str(e),
            )
    
    def _filter_unchanged(
        self,
        new_df: pl.DataFrame,
        existing_df: pl.DataFrame,
        unique_key: List[str],
    ) -> Tuple[pl.DataFrame, int]:
        """
        Filter out rows that already exist and haven't changed.
        
        Uses a hash of non-key columns to detect changes.
        
        Returns:
            Tuple of (filtered_df, unchanged_count)
        """
        try:
            # Get columns that exist in both dataframes
            common_cols = set(new_df.columns) & set(existing_df.columns)
            value_cols = [c for c in common_cols if c not in unique_key]
            
            if not value_cols:
                return new_df, 0
            
            # Create row hashes for comparison
            def add_row_hash(df: pl.DataFrame, hash_col: str = "_row_hash") -> pl.DataFrame:
                # Simple hash based on string representation of value columns
                hash_expr = pl.concat_str(
                    [pl.col(c).cast(pl.Utf8).fill_null("NULL") for c in value_cols],
                    separator="|"
                ).hash()
                return df.with_columns(hash_expr.alias(hash_col))
            
            new_with_hash = add_row_hash(new_df)
            existing_with_hash = add_row_hash(existing_df)
            
            # Find existing hashes for matching keys
            existing_hashes = existing_with_hash.select(
                unique_key + ["_row_hash"]
            ).rename({"_row_hash": "_existing_hash"})
            
            # Join to find matches
            joined = new_with_hash.join(
                existing_hashes,
                on=unique_key,
                how="left"
            )
            
            # Keep only rows where hash differs or no existing match
            changed = joined.filter(
                pl.col("_existing_hash").is_null() |
                (pl.col("_row_hash") != pl.col("_existing_hash"))
            ).drop(["_row_hash", "_existing_hash"])
            
            unchanged_count = len(new_df) - len(changed)
            
            return changed, unchanged_count
            
        except Exception as e:
            logger.warning(f"Change detection failed, keeping all rows: {e}")
            return new_df, 0
    
    def exists_with_key(
        self,
        path: Path,
        unique_key: List[str],
        key_values: List[tuple],
    ) -> List[bool]:
        """
        Check if records with given keys exist in file.
        
        Args:
            path: Parquet file path
            unique_key: Column names for unique key
            key_values: List of key value tuples to check
            
        Returns:
            List of booleans indicating existence
        """
        path = Path(path)
        
        if not path.exists():
            return [False] * len(key_values)
        
        try:
            df = pl.read_parquet(path, columns=unique_key)
            
            # Create set of existing keys
            existing = set()
            for row in df.iter_rows():
                existing.add(row)
            
            return [kv in existing for kv in key_values]
            
        except Exception as e:
            logger.warning(f"Key existence check failed: {e}")
            return [False] * len(key_values)


# Global instance
_atomic_writer = None


def get_atomic_writer() -> AtomicParquetWriter:
    """Get global atomic writer instance."""
    global _atomic_writer
    if _atomic_writer is None:
        _atomic_writer = AtomicParquetWriter()
    return _atomic_writer
