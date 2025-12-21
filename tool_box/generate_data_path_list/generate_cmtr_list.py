"""
Generate a CSV file listing all .cmtr files larger than 10 GB
in drives O:, P:, Q:, R:, L:, M:
"""

import os
import csv
from pathlib import Path
from datetime import datetime


def get_file_size_gb(file_path: Path) -> float:
    """Get file size in GB."""
    try:
        size_bytes = file_path.stat().st_size
        return size_bytes / (1024 ** 3)
    except (OSError, PermissionError):
        return 0.0


def find_large_cmtr_files(drives: list[str], min_size_gb: float = 10.0) -> list[dict]:
    """
    Find all .cmtr files larger than min_size_gb in the specified drives.
    
    Args:
        drives: List of drive letters (e.g., ['O:', 'P:'])
        min_size_gb: Minimum file size in GB
        
    Returns:
        List of dicts with file info
    """
    large_files = []
    
    for drive in drives:
        drive_path = Path(f"{drive}/")
        if not drive_path.exists():
            print(f"Drive {drive} not accessible, skipping...")
            continue
            
        print(f"Scanning {drive}...")
        
        try:
            for file_path in drive_path.rglob("*.cmtr"):
                try:
                    size_gb = get_file_size_gb(file_path)
                    if size_gb >= min_size_gb:
                        stat_info = file_path.stat()
                        large_files.append({
                            "file_path": str(file_path),
                            "file_name": file_path.name,
                            "size_gb": round(size_gb, 2),
                            "size_bytes": stat_info.st_size,
                            "modified_time": datetime.fromtimestamp(stat_info.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                            "drive": drive
                        })
                        print(f"  Found: {file_path.name} ({size_gb:.2f} GB)")
                except (OSError, PermissionError) as e:
                    continue
        except (OSError, PermissionError) as e:
            print(f"Error accessing {drive}: {e}")
            continue
    
    return large_files


def save_to_csv(files: list[dict], output_path: Path) -> None:
    """Save file list to CSV."""
    if not files:
        print("No files found matching criteria.")
        return
    
    fieldnames = ["file_path", "file_name", "size_gb", "size_bytes", "modified_time", "drive"]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(files)
    
    print(f"\nSaved {len(files)} files to: {output_path}")


def main():
    # Drives to scan
    drives = ["O:", "P:", "Q:", "R:", "L:", "M:"]
    
    # Minimum file size in GB
    min_size_gb = 0.0
    
    # Output CSV path (same folder as this script)
    script_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = script_dir / f"large_cmtr_files_{timestamp}.csv"
    
    print(f"Searching for .cmtr files >= {min_size_gb} GB in drives: {', '.join(drives)}")
    print("-" * 60)
    
    # Find files
    large_files = find_large_cmtr_files(drives, min_size_gb)
    
    # Sort by size (largest first)
    large_files.sort(key=lambda x: x["size_bytes"], reverse=True)
    
    # Save to CSV
    save_to_csv(large_files, output_path)
    
    # Print summary
    if large_files:
        #total_size_gb = sum(f["size_gb"] for f in large_files)
        print(f"\nSummary:")
        print(f"  Total files found: {len(large_files)}")
        #print(f"  Total size: {total_size_gb:.2f} GB")


if __name__ == "__main__":
    main()

