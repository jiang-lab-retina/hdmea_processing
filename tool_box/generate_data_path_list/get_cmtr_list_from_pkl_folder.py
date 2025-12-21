"""
Match PKL files from a folder to CMTR files based on date-time labels in filenames.
"""

import os
import re
import csv
from pathlib import Path
from datetime import datetime


# Regex pattern to extract date-time from filenames like "2024.09.18-12.17.43-Rec.pkl"
DATETIME_PATTERN = re.compile(r"(\d{4}\.\d{2}\.\d{2}-\d{2}\.\d{2}\.\d{2})")


def extract_datetime_label(filename: str) -> str | None:
    """Extract the date-time label from a filename."""
    match = DATETIME_PATTERN.search(filename)
    return match.group(1) if match else None


def search_files(
    folder_path: Path,
    include_keywords: list[str],
    exclude_keywords: list[str] | str | None = None,
    recursive: bool = False
) -> list[Path]:
    """
    Search for files matching include keywords but excluding files with exclude keywords.
    
    Args:
        folder_path: Directory to search
        include_keywords: List of keywords that ALL must be present in filename
        exclude_keywords: Keywords that will exclude the file if ANY are present
        recursive: If True, search subdirectories
        
    Returns:
        List of matching file paths
    """
    if not folder_path.exists():
        return []
    
    # Normalize exclude_keywords to list
    if exclude_keywords is None:
        exclude_keywords = []
    elif isinstance(exclude_keywords, str):
        exclude_keywords = [exclude_keywords]
    
    results = []
    
    if recursive:
        file_iterator = folder_path.rglob("*")
    else:
        file_iterator = folder_path.iterdir()
    
    for file_path in file_iterator:
        if not file_path.is_file():
            continue
        
        filename = file_path.name
        
        # Check all include keywords are present
        if not all(kw in filename for kw in include_keywords):
            continue
        
        # Check no exclude keywords are present
        if any(kw in filename for kw in exclude_keywords):
            continue
        
        results.append(file_path)
    
    return results


def get_pkl_files(folder_path: Path) -> list[dict]:
    """
    Get all .pkl files from the specified folder (no subfolders).
    
    Returns:
        List of dicts with pkl file info
    """
    pkl_files = []
    
    if not folder_path.exists():
        print(f"Folder not found: {folder_path}")
        return pkl_files
    
    for file_path in folder_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == ".pkl":
            datetime_label = extract_datetime_label(file_path.name)
            # Only include files with valid datetime label in filename
            if datetime_label:
                pkl_files.append({
                    "pkl_file": file_path.name,
                    "pkl_path": str(file_path),
                    "datetime_label": datetime_label
                })
    
    return pkl_files


def get_pkl_files_with_keywords(
    folder_path: Path,
    search_specs: list[tuple[list[str], str | None]]
) -> list[dict]:
    """
    Get PKL files matching keyword specifications.
    
    Args:
        folder_path: Directory to search
        search_specs: List of (include_keywords, exclude_keyword) tuples
                     e.g., [([".pkl", "2024.02.26"], "15.20.32"), ([".pkl", "2024.03"], None)]
    
    Returns:
        List of dicts with pkl file info (deduplicated)
    """
    seen_paths = set()
    pkl_files = []
    
    for include_keywords, exclude_keyword in search_specs:
        matches = search_files(folder_path, include_keywords, exclude_keyword, recursive=False)
        
        for file_path in matches:
            if str(file_path) in seen_paths:
                continue
            seen_paths.add(str(file_path))
            
            datetime_label = extract_datetime_label(file_path.name)
            if datetime_label:
                pkl_files.append({
                    "pkl_file": file_path.name,
                    "pkl_path": str(file_path),
                    "datetime_label": datetime_label
                })
    
    return pkl_files


def load_cmtr_csv(csv_path: Path) -> dict[str, dict]:
    """
    Load CMTR files CSV and create a lookup dict by datetime label.
    
    Returns:
        Dict mapping datetime_label -> cmtr file info
    """
    cmtr_lookup = {}
    
    if not csv_path.exists():
        print(f"CMTR CSV not found: {csv_path}")
        return cmtr_lookup
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row.get("file_name", "")
            datetime_label = extract_datetime_label(file_name)
            if datetime_label:
                # If multiple cmtr files have same datetime, keep the first one
                if datetime_label not in cmtr_lookup:
                    cmtr_lookup[datetime_label] = {
                        "cmtr_file": file_name,
                        "cmtr_path": row.get("file_path", "")
                    }
    
    return cmtr_lookup


def get_cmcr_path(cmtr_path: str) -> str:
    """
    Derive the cmcr file path from the cmtr file path.
    
    cmtr: 2025.01.29-13.12.28-Rec-.cmtr
    cmcr: 2025.01.29-13.12.28-Rec.cmcr (no "-" at the end, .cmcr extension)
    
    Returns:
        Path to cmcr file if it exists, empty string otherwise
    """
    if not cmtr_path:
        return ""
    
    cmtr_file = Path(cmtr_path)
    cmtr_name = cmtr_file.stem  # e.g., "2025.01.29-13.12.28-Rec-"
    
    # Remove trailing "-" if present
    if cmtr_name.endswith("-"):
        cmcr_name = cmtr_name[:-1]
    else:
        cmcr_name = cmtr_name
    
    # Build cmcr path (same folder, .cmcr extension)
    cmcr_path = cmtr_file.parent / f"{cmcr_name}.cmcr"
    
    if cmcr_path.exists():
        return str(cmcr_path)
    return ""


def match_pkl_to_cmtr(pkl_files: list[dict], cmtr_lookup: dict[str, dict]) -> list[dict]:
    """
    Match PKL files to CMTR files by datetime label.
    
    Returns:
        List of matched records
    """
    results = []
    
    for pkl in pkl_files:
        datetime_label = pkl["datetime_label"]
        
        if datetime_label and datetime_label in cmtr_lookup:
            cmtr_info = cmtr_lookup[datetime_label]
            cmtr_path = cmtr_info["cmtr_path"]
            cmcr_path = get_cmcr_path(cmtr_path)
            results.append({
                "pkl_file": pkl["pkl_file"],
                "pkl_path": pkl["pkl_path"],
                "datetime_label": datetime_label,
                "cmtr_file": cmtr_info["cmtr_file"],
                "cmtr_path": cmtr_path,
                "cmcr_path": cmcr_path,
                "matched": bool(cmcr_path)  # Only matched if cmcr file exists
            })
        else:
            results.append({
                "pkl_file": pkl["pkl_file"],
                "pkl_path": pkl["pkl_path"],
                "datetime_label": datetime_label or "",
                "cmtr_file": "",
                "cmtr_path": "",
                "cmcr_path": "",
                "matched": False
            })
    
    return results


def save_results(results: list[dict], output_path: Path) -> None:
    """Save matching results to CSV."""
    fieldnames = ["pkl_file", "pkl_path", "datetime_label", "cmtr_file", "cmtr_path", "cmcr_path", "matched"]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Saved results to: {output_path}")


def main():
    # PKL folder path (direct files only, no subfolders)
    pkl_folder = Path(r"M:\Python_Project\Data_Processing_2024\Data")
    
    # CMTR CSV path (same folder as this script)
    script_dir = Path(__file__).parent
    cmtr_csv_path = script_dir / "cmtr_files.csv"
    
    # Output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = script_dir / f"pkl_to_cmtr_mapping_{timestamp}.csv"
    
    # Define search specifications: (include_keywords, exclude_keyword)
    # Format: ([".pkl", "date_pattern"], "exclude_pattern" or None)
    search_specs = [
        ([".pkl", "2024.02.26"], "15.20.32"),  # 2024.02.26 but exclude 15.20.32
        ([".pkl", "2024.02.28"], None),
        ([".pkl", "2024.03"], None),
        ([".pkl", "2024.04"], None),
        ([".pkl", "2024.05"], None),
        ([".pkl", "2024.06"], None),
        ([".pkl", "2024.07"], None),
        ([".pkl", "2024.08"], None),
        ([".pkl", "2024.09"], None),
        ([".pkl", "2024.1"], None),  # 2024.10, 2024.11, 2024.12
        ([".pkl", "2025."], None),   # All 2025 files
    ]
    
    print(f"Loading PKL files from: {pkl_folder}")
    pkl_files = get_pkl_files_with_keywords(pkl_folder, search_specs)
    pkl_files.sort(key=lambda x: x["pkl_file"])
    print(f"Found {len(pkl_files)} PKL files")
    
    print(f"\nLoading CMTR files from: {cmtr_csv_path}")
    cmtr_lookup = load_cmtr_csv(cmtr_csv_path)
    print(f"Found {len(cmtr_lookup)} unique CMTR datetime labels")
    
    print("\nMatching PKL to CMTR files...")
    results = match_pkl_to_cmtr(pkl_files, cmtr_lookup)
    
    # Count matches
    matched_count = sum(1 for r in results if r["matched"])
    unmatched_count = len(results) - matched_count
    cmcr_count = sum(1 for r in results if r.get("cmcr_path"))
    
    print(f"\nResults:")
    print(f"  Total PKL files: {len(results)}")
    print(f"  Matched to CMTR: {matched_count}")
    print(f"  With CMCR file: {cmcr_count}")
    print(f"  Unmatched: {unmatched_count}")
    
    # Save results
    save_results(results, output_path)


if __name__ == "__main__":
    main()

