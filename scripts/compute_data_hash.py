#!/usr/bin/env python
"""
Compute SHA256 hash of data files for reproducibility tracking.

Usage:
    python scripts/compute_data_hash.py data/processed/features.parquet
    python scripts/compute_data_hash.py --all
"""
import hashlib
import argparse
import json
from pathlib import Path
from datetime import datetime


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    
    with open(filepath, 'rb') as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    
    return sha256.hexdigest()


def compute_data_snapshot(data_dir: Path = None) -> dict:
    """
    Compute hashes for all data files in a directory.
    
    Returns:
        Dict with file paths and their hashes
    """
    if data_dir is None:
        data_dir = Path("data/processed")
    
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        return {"error": f"Directory not found: {data_dir}"}
    
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "directory": str(data_dir),
        "files": {},
    }
    
    # Key files to hash
    key_patterns = ["*.parquet", "*.csv"]
    
    for pattern in key_patterns:
        for filepath in data_dir.glob(pattern):
            rel_path = str(filepath.relative_to(data_dir))
            snapshot["files"][rel_path] = {
                "hash": compute_file_hash(filepath),
                "size_bytes": filepath.stat().st_size,
                "modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
            }
    
    # Compute combined hash
    if snapshot["files"]:
        combined = hashlib.sha256()
        for name in sorted(snapshot["files"].keys()):
            combined.update(snapshot["files"][name]["hash"].encode())
        snapshot["combined_hash"] = combined.hexdigest()[:16]  # Short hash
    else:
        snapshot["combined_hash"] = "no-files"
    
    return snapshot


def main():
    parser = argparse.ArgumentParser(description="Compute data file hashes")
    parser.add_argument(
        "file",
        nargs="?",
        help="Specific file to hash"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Hash all data files in data/processed/"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for snapshot"
    )
    
    args = parser.parse_args()
    
    if args.all:
        snapshot = compute_data_snapshot()
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(snapshot, f, indent=2)
            print(f"Snapshot saved to {args.output}")
        else:
            print(json.dumps(snapshot, indent=2))
        
        print(f"\nCombined hash: {snapshot.get('combined_hash', 'N/A')}")
        
    elif args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return 1
        
        file_hash = compute_file_hash(filepath)
        print(f"{file_hash}  {filepath}")
        
    else:
        # Default: hash features.parquet
        default_file = Path("data/processed/features.parquet")
        if default_file.exists():
            file_hash = compute_file_hash(default_file)
            print(f"{file_hash}  {default_file}")
        else:
            print("No file specified. Use --all or provide a file path.")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
