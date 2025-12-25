"""
Batch Soma Geometry Extraction

Processes all HDF5 files in an input folder, extracts soma geometry
using the session workflow, and saves results to an output folder.

Usage:
    python batch_geometry_extraction.py --input_dir /path/to/h5_files --output_dir /path/to/output
    
    # Or with default paths (edit INPUT_DIR and OUTPUT_DIR below):
    python batch_geometry_extraction.py

Author: Generated for experimental analysis
Date: 2024-12
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Optional
import h5py
import numpy as np

# Import session utilities
from hdmea.pipeline import create_session, PipelineSession

# Import geometry extraction functions
from ap_sta import extract_eimage_sta_geometry


# =============================================================================
# Default Configuration
# =============================================================================

# Default input/output directories (modify as needed)
INPUT_DIR = Path(__file__).parent.parent / "pipeline_test" / "output"
OUTPUT_DIR = Path(__file__).parent / "eimage_sta_output_20251225"

# Geometry extraction parameters
DEFAULT_FRAME_RANGE = (10, 14)
DEFAULT_THRESHOLD_FRACTION = 0.5

# Processing options
COPY_ORIGINAL = True  # If True, copy original HDF5 to output folder before processing


# =============================================================================
# Helper Functions
# =============================================================================

def find_h5_files(input_dir: Path) -> List[Path]:
    """
    Find all HDF5 files in the input directory.
    
    Args:
        input_dir: Directory to search
        
    Returns:
        List of paths to HDF5 files
    """
    h5_files = []
    
    # Search for common HDF5 extensions
    for pattern in ["*.h5", "*.hdf5", "*.hdf"]:
        h5_files.extend(input_dir.glob(pattern))
    
    # Sort by filename for consistent ordering
    h5_files = sorted(h5_files, key=lambda p: p.name)
    
    return h5_files


def load_hdf5_to_session(hdf5_path: Path, dataset_id: str = None) -> PipelineSession:
    """
    Load an existing HDF5 file into a PipelineSession.
    
    Args:
        hdf5_path: Path to existing HDF5 file
        dataset_id: Optional dataset ID (uses filename stem if not provided)
        
    Returns:
        PipelineSession with data loaded
    """
    if dataset_id is None:
        dataset_id = hdf5_path.stem
    
    session = create_session(dataset_id=dataset_id)
    session.hdf5_path = hdf5_path
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'units' not in f:
            print(f"  Warning: No 'units' group in {hdf5_path.name}")
            return session
        
        for unit_id in f['units'].keys():
            unit_group = f[f'units/{unit_id}']
            unit_data = {}
            
            # Load eimage_sta if available
            eimage_sta_path = 'features/eimage_sta/data'
            if eimage_sta_path in unit_group:
                if 'features' not in unit_data:
                    unit_data['features'] = {}
                if 'eimage_sta' not in unit_data['features']:
                    unit_data['features']['eimage_sta'] = {}
                unit_data['features']['eimage_sta']['data'] = unit_group[eimage_sta_path][:]
            
            if unit_data:
                session.units[unit_id] = unit_data
    
    session.completed_steps.add('load_hdf5')
    return session


def save_session_to_hdf5(session: PipelineSession, hdf5_path: Path) -> None:
    """
    Save session geometry data to HDF5 file.
    
    Args:
        session: PipelineSession with geometry data
        hdf5_path: Path to save to
    """
    with h5py.File(hdf5_path, 'r+') as f:
        for unit_id, unit_data in session.units.items():
            geom_data = unit_data.get('features', {}).get('eimage_sta', {}).get('geometry')
            
            if geom_data is None:
                continue
            
            unit_path = f'units/{unit_id}'
            if unit_path not in f:
                continue
            
            geometry_path = f'{unit_path}/features/eimage_sta/geometry'
            if geometry_path in f:
                del f[geometry_path]
            
            geom_group = f.create_group(geometry_path)
            
            for key, value in geom_data.items():
                if value is None:
                    continue
                if isinstance(value, np.ndarray):
                    geom_group.create_dataset(key, data=value, compression='gzip')
                else:
                    geom_group.create_dataset(key, data=value)


def process_single_file(
    input_path: Path,
    output_path: Path,
    frame_range: tuple = DEFAULT_FRAME_RANGE,
    threshold_fraction: float = DEFAULT_THRESHOLD_FRACTION,
    copy_original: bool = COPY_ORIGINAL,
) -> dict:
    """
    Process a single HDF5 file with geometry extraction.
    
    Args:
        input_path: Path to input HDF5 file
        output_path: Path for output HDF5 file
        frame_range: Frames for size estimation
        threshold_fraction: Threshold for soma mask
        copy_original: Whether to copy original file first
        
    Returns:
        Dictionary with processing results
    """
    result = {
        'input_path': input_path,
        'output_path': output_path,
        'success': False,
        'units_processed': 0,
        'error': None,
    }
    
    try:
        # Copy original file to output location if requested
        if copy_original:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, output_path)
            working_path = output_path
        else:
            working_path = input_path
        
        # Load into session
        session = load_hdf5_to_session(working_path)
        
        units_with_eimage_sta = sum(
            1 for u in session.units.values()
            if 'features' in u and 'eimage_sta' in u.get('features', {})
            and 'data' in u['features']['eimage_sta']
        )
        
        if units_with_eimage_sta == 0:
            result['error'] = "No units with eimage_sta data"
            return result
        
        # Extract geometry
        session = extract_eimage_sta_geometry(
            frame_range=frame_range,
            threshold_fraction=threshold_fraction,
            session=session,
        )
        
        # Save to output file
        save_session_to_hdf5(session, working_path)
        
        # Count processed units
        units_processed = sum(
            1 for u in session.units.values()
            if 'features' in u and 'eimage_sta' in u.get('features', {})
            and 'geometry' in u['features']['eimage_sta']
        )
        
        result['units_processed'] = units_processed
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        import traceback
        result['traceback'] = traceback.format_exc()
    
    return result


# =============================================================================
# Batch Processing
# =============================================================================

def batch_process(
    input_dir: Path,
    output_dir: Path,
    frame_range: tuple = DEFAULT_FRAME_RANGE,
    threshold_fraction: float = DEFAULT_THRESHOLD_FRACTION,
    copy_original: bool = COPY_ORIGINAL,
    file_pattern: str = None,
) -> List[dict]:
    """
    Process all HDF5 files in a directory.
    
    Args:
        input_dir: Directory containing HDF5 files
        output_dir: Directory for output files
        frame_range: Frames for size estimation
        threshold_fraction: Threshold for soma mask
        copy_original: Whether to copy original files
        file_pattern: Optional glob pattern to filter files
        
    Returns:
        List of result dictionaries for each file
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Find HDF5 files
    h5_files = find_h5_files(input_dir)
    
    if file_pattern:
        h5_files = [f for f in h5_files if f.match(file_pattern)]
    
    if not h5_files:
        print(f"No HDF5 files found in {input_dir}")
        return []
    
    print("=" * 70)
    print("Batch Soma Geometry Extraction")
    print("=" * 70)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Files to process: {len(h5_files)}")
    print(f"Frame range:      {frame_range}")
    print(f"Threshold:        {threshold_fraction}")
    print(f"Copy original:    {copy_original}")
    print("=" * 70)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    results = []
    
    for i, h5_file in enumerate(h5_files, 1):
        print(f"\n[{i}/{len(h5_files)}] Processing: {h5_file.name}")
        print("-" * 50)
        
        output_path = output_dir / h5_file.name
        
        result = process_single_file(
            input_path=h5_file,
            output_path=output_path,
            frame_range=frame_range,
            threshold_fraction=threshold_fraction,
            copy_original=copy_original,
        )
        
        results.append(result)
        
        if result['success']:
            print(f"  ✓ Success: {result['units_processed']} units processed")
        else:
            print(f"  ✗ Failed: {result['error']}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 70)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Total files:    {len(results)}")
    print(f"Successful:     {len(successful)}")
    print(f"Failed:         {len(failed)}")
    
    total_units = sum(r['units_processed'] for r in successful)
    print(f"Total units:    {total_units}")
    
    if failed:
        print("\nFailed files:")
        for r in failed:
            print(f"  - {r['input_path'].name}: {r['error']}")
    
    print(f"\nOutput saved to: {output_dir}")
    
    # Save processing log
    log_path = output_dir / "processing_log.txt"
    with open(log_path, 'w') as f:
        f.write("Batch Geometry Extraction Log\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input directory: {input_dir}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Frame range: {frame_range}\n")
        f.write(f"Threshold: {threshold_fraction}\n\n")
        
        f.write("Results:\n")
        f.write("-" * 50 + "\n")
        for r in results:
            status = "SUCCESS" if r['success'] else "FAILED"
            f.write(f"{r['input_path'].name}: {status}")
            if r['success']:
                f.write(f" ({r['units_processed']} units)\n")
            else:
                f.write(f" - {r['error']}\n")
        
        f.write("\n" + "-" * 50 + "\n")
        f.write(f"Total: {len(successful)}/{len(results)} successful\n")
    
    print(f"Log saved to: {log_path}")
    
    return results


# =============================================================================
# Command Line Interface
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch process HDF5 files for soma geometry extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all HDF5 files in a folder
    python batch_geometry_extraction.py --input_dir /data/recordings --output_dir /data/processed
    
    # Custom frame range and threshold
    python batch_geometry_extraction.py --input_dir /data/recordings --output_dir /data/processed \\
        --frame_range 10 14 --threshold 0.4
    
    # Process in-place (don't copy files)
    python batch_geometry_extraction.py --input_dir /data/recordings --output_dir /data/recordings \\
        --no_copy
        """,
    )
    
    parser.add_argument(
        "--input_dir", "-i",
        type=Path,
        default=INPUT_DIR,
        help=f"Input directory containing HDF5 files (default: {INPUT_DIR})",
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for processed files (default: {OUTPUT_DIR})",
    )
    
    parser.add_argument(
        "--frame_range", "-f",
        type=int,
        nargs=2,
        default=list(DEFAULT_FRAME_RANGE),
        metavar=("START", "END"),
        help=f"Frame range for size estimation (default: {DEFAULT_FRAME_RANGE})",
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=DEFAULT_THRESHOLD_FRACTION,
        help=f"Threshold fraction for soma mask (default: {DEFAULT_THRESHOLD_FRACTION})",
    )
    
    parser.add_argument(
        "--no_copy",
        action="store_true",
        help="Don't copy original files (process in-place if input=output)",
    )
    
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        default=None,
        help="Glob pattern to filter files (e.g., '*2024*')",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    batch_process(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        frame_range=tuple(args.frame_range),
        threshold_fraction=args.threshold,
        copy_original=not args.no_copy,
        file_pattern=args.pattern,
    )


if __name__ == "__main__":
    main()

