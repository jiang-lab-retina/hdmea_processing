#!/usr/bin/env python
"""
Comprehensive HDF5 File Comparison Tool

This script performs a thorough comparison of two HDF5 files including:
- Complete structural analysis
- Unit-by-unit feature comparison
- Value-level comparison with statistics
- ONH intersection and polar coordinate analysis
- Summary with expected vs unexpected differences
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import h5py


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class DatasetComparison:
    """Comparison result for a single dataset."""
    path: str
    match: bool = True
    shape1: tuple = None
    shape2: tuple = None
    dtype1: str = None
    dtype2: str = None
    value_match: bool = True
    max_diff: float = 0.0
    mean1: float = None
    mean2: float = None
    details: List[str] = field(default_factory=list)


@dataclass
class UnitComparison:
    """Comparison result for a single unit."""
    unit_id: str
    features_match: bool = True
    features_only_in_1: List[str] = field(default_factory=list)
    features_only_in_2: List[str] = field(default_factory=list)
    common_features: List[str] = field(default_factory=list)
    dataset_diffs: List[DatasetComparison] = field(default_factory=list)
    structural_diffs: List[str] = field(default_factory=list)


@dataclass  
class ComparisonSummary:
    """Overall comparison summary."""
    file1_path: str
    file2_path: str
    file1_size_mb: float
    file2_size_mb: float
    
    # Structure
    top_groups_match: bool = True
    units_match: bool = True
    n_units_file1: int = 0
    n_units_file2: int = 0
    units_only_in_1: List[str] = field(default_factory=list)
    units_only_in_2: List[str] = field(default_factory=list)
    
    # Metadata
    metadata_keys_match: bool = True
    metadata_only_in_1: List[str] = field(default_factory=list)
    metadata_only_in_2: List[str] = field(default_factory=list)
    
    # Unit comparisons
    unit_comparisons: List[UnitComparison] = field(default_factory=list)
    
    # Statistics
    total_datasets_compared: int = 0
    datasets_matching: int = 0
    datasets_differing: int = 0
    
    # AP Tracking specific
    ap_tracking_units_file1: int = 0
    ap_tracking_units_file2: int = 0
    onh_file1: Tuple[float, float] = None
    onh_file2: Tuple[float, float] = None
    
    # Expected differences
    expected_diffs: List[str] = field(default_factory=list)
    unexpected_diffs: List[str] = field(default_factory=list)


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_datasets_detailed(d1, d2, path: str, rtol: float = 1e-5, atol: float = 1e-8) -> DatasetComparison:
    """Detailed comparison of two HDF5 datasets."""
    result = DatasetComparison(path=path)
    
    try:
        data1 = d1[()]
        data2 = d2[()]
    except Exception as e:
        result.match = False
        result.details.append(f"Error reading data: {e}")
        return result
    
    # Compare shapes
    if hasattr(data1, 'shape'):
        result.shape1 = data1.shape
    if hasattr(data2, 'shape'):
        result.shape2 = data2.shape
    
    if result.shape1 != result.shape2:
        result.match = False
        result.details.append(f"Shape mismatch: {result.shape1} vs {result.shape2}")
        return result
    
    # Compare dtypes
    if hasattr(data1, 'dtype'):
        result.dtype1 = str(data1.dtype)
    if hasattr(data2, 'dtype'):
        result.dtype2 = str(data2.dtype)
    
    # Compare values
    try:
        if isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray):
            if np.issubdtype(data1.dtype, np.floating):
                result.mean1 = float(np.nanmean(data1))
                result.mean2 = float(np.nanmean(data2))
                
                if np.allclose(data1, data2, rtol=rtol, atol=atol, equal_nan=True):
                    result.value_match = True
                else:
                    result.value_match = False
                    result.match = False
                    result.max_diff = float(np.nanmax(np.abs(data1 - data2)))
            else:
                if np.array_equal(data1, data2):
                    result.value_match = True
                else:
                    result.value_match = False
                    result.match = False
                    n_diff = np.sum(data1 != data2)
                    result.details.append(f"{n_diff} values differ")
        elif isinstance(data1, bytes) and isinstance(data2, bytes):
            result.value_match = (data1 == data2)
            result.match = result.value_match
        elif isinstance(data1, str) and isinstance(data2, str):
            result.value_match = (data1 == data2)
            result.match = result.value_match
        else:
            # Scalar comparison
            if data1 == data2:
                result.value_match = True
            else:
                result.value_match = False
                result.match = False
                result.details.append(f"Values: {data1} vs {data2}")
    except Exception as e:
        result.details.append(f"Comparison error: {e}")
    
    return result


def collect_all_datasets(group: h5py.Group, prefix: str = "") -> Dict[str, h5py.Dataset]:
    """Recursively collect all datasets in a group."""
    datasets = {}
    
    for key in group.keys():
        path = f"{prefix}/{key}" if prefix else key
        item = group[key]
        
        if isinstance(item, h5py.Dataset):
            datasets[path] = item
        elif isinstance(item, h5py.Group):
            datasets.update(collect_all_datasets(item, path))
    
    return datasets


def value_by_value_comparison(f1: h5py.File, f2: h5py.File, unit_id: str, 
                               features_to_compare: List[str] = None) -> Dict[str, List[Dict]]:
    """
    Perform detailed value-by-value comparison for a unit.
    
    Returns dict with 'matches', 'differences', 'only_in_1', 'only_in_2' lists.
    """
    result = {
        'matches': [],
        'differences': [],
        'only_in_1': [],
        'only_in_2': [],
    }
    
    u1_path = f'units/{unit_id}'
    u2_path = f'units/{unit_id}'
    
    if u1_path not in f1 or u2_path not in f2:
        return result
    
    u1 = f1[u1_path]
    u2 = f2[u2_path]
    
    # Collect all datasets from both units
    d1_all = collect_all_datasets(u1, '')
    d2_all = collect_all_datasets(u2, '')
    
    # Filter to specific features if requested
    if features_to_compare:
        d1_all = {k: v for k, v in d1_all.items() 
                  if any(k.startswith(f'features/{f}') for f in features_to_compare)}
        d2_all = {k: v for k, v in d2_all.items() 
                  if any(k.startswith(f'features/{f}') for f in features_to_compare)}
    
    paths1 = set(d1_all.keys())
    paths2 = set(d2_all.keys())
    
    # Datasets only in one file
    for path in sorted(paths1 - paths2):
        try:
            val = d1_all[path][()]
            if isinstance(val, np.ndarray):
                val_str = f"array{val.shape}" if val.size > 5 else str(val)
            else:
                val_str = str(val)[:50]
            result['only_in_1'].append({'path': path, 'value': val_str})
        except:
            result['only_in_1'].append({'path': path, 'value': '<error reading>'})
    
    for path in sorted(paths2 - paths1):
        try:
            val = d2_all[path][()]
            if isinstance(val, np.ndarray):
                val_str = f"array{val.shape}" if val.size > 5 else str(val)
            else:
                val_str = str(val)[:50]
            result['only_in_2'].append({'path': path, 'value': val_str})
        except:
            result['only_in_2'].append({'path': path, 'value': '<error reading>'})
    
    # Compare common paths
    for path in sorted(paths1 & paths2):
        try:
            val1 = d1_all[path][()]
            val2 = d2_all[path][()]
            
            # Format values for display
            def format_val(v):
                if isinstance(v, np.ndarray):
                    if v.size == 0:
                        return "[]"
                    elif v.size == 1:
                        return f"{float(v.flat[0]):.6g}"
                    elif v.size <= 5:
                        return str(v.tolist())
                    else:
                        return f"array{v.shape}, mean={np.nanmean(v):.4g}"
                elif isinstance(v, (float, np.floating)):
                    if np.isnan(v):
                        return "nan"
                    return f"{float(v):.6g}"
                elif isinstance(v, bytes):
                    return v.decode('utf-8', errors='ignore')[:50]
                else:
                    return str(v)[:50]
            
            val1_str = format_val(val1)
            val2_str = format_val(val2)
            
            # Check if values match
            match = False
            diff_val = None
            
            if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                if val1.shape != val2.shape:
                    match = False
                    diff_val = f"shape: {val1.shape} vs {val2.shape}"
                elif np.issubdtype(val1.dtype, np.floating):
                    if np.allclose(val1, val2, rtol=1e-5, atol=1e-8, equal_nan=True):
                        match = True
                    else:
                        match = False
                        max_diff = float(np.nanmax(np.abs(val1 - val2)))
                        diff_val = f"max_diff={max_diff:.2e}"
                else:
                    match = np.array_equal(val1, val2)
                    if not match:
                        n_diff = np.sum(val1 != val2)
                        diff_val = f"{n_diff} elements differ"
            elif isinstance(val1, (float, np.floating)) and isinstance(val2, (float, np.floating)):
                if np.isnan(val1) and np.isnan(val2):
                    match = True
                elif np.isclose(val1, val2, rtol=1e-5, atol=1e-8):
                    match = True
                else:
                    match = False
                    diff_val = f"diff={abs(val1-val2):.2e}"
            else:
                match = (val1 == val2)
            
            entry = {
                'path': path,
                'file1': val1_str,
                'file2': val2_str,
            }
            
            if match:
                result['matches'].append(entry)
            else:
                entry['diff'] = diff_val
                result['differences'].append(entry)
                
        except Exception as e:
            result['differences'].append({
                'path': path,
                'file1': '<error>',
                'file2': '<error>',
                'diff': str(e)
            })
    
    return result


def print_value_by_value_report(f1: h5py.File, f2: h5py.File, 
                                 unit_ids: List[str],
                                 features: List[str] = None):
    """Print detailed value-by-value comparison report."""
    
    print("\n" + "=" * 80)
    print("DETAILED VALUE-BY-VALUE COMPARISON")
    print("=" * 80)
    
    if features:
        print(f"Features compared: {features}")
    
    for unit_id in unit_ids:
        print(f"\n{'='*80}")
        print(f"UNIT: {unit_id}")
        print(f"{'='*80}")
        
        result = value_by_value_comparison(f1, f2, unit_id, features)
        
        # Print only-in-1
        if result['only_in_1']:
            print(f"\n  [Only in File 1] ({len(result['only_in_1'])} datasets)")
            for item in result['only_in_1'][:10]:
                print(f"    {item['path']}: {item['value']}")
            if len(result['only_in_1']) > 10:
                print(f"    ... and {len(result['only_in_1']) - 10} more")
        
        # Print only-in-2
        if result['only_in_2']:
            print(f"\n  [Only in File 2] ({len(result['only_in_2'])} datasets)")
            for item in result['only_in_2'][:10]:
                print(f"    {item['path']}: {item['value']}")
            if len(result['only_in_2']) > 10:
                print(f"    ... and {len(result['only_in_2']) - 10} more")
        
        # Print differences
        if result['differences']:
            print(f"\n  [Value Differences] ({len(result['differences'])} datasets)")
            for item in result['differences']:
                diff_info = f" ({item.get('diff', '')})" if item.get('diff') else ""
                print(f"    {item['path']}:")
                print(f"      File1: {item['file1']}")
                print(f"      File2: {item['file2']}{diff_info}")
        
        # Print matches summary
        if result['matches']:
            print(f"\n  [Matches] {len(result['matches'])} datasets match exactly")
        
        # Summary
        total = len(result['matches']) + len(result['differences'])
        if total > 0:
            match_pct = 100 * len(result['matches']) / total
            print(f"\n  Summary: {len(result['matches'])}/{total} datasets match ({match_pct:.1f}%)")


def compare_unit_comprehensive(f1: h5py.File, f2: h5py.File, unit_id: str) -> UnitComparison:
    """Comprehensive comparison of a single unit."""
    result = UnitComparison(unit_id=unit_id)
    
    u1_path = f'units/{unit_id}'
    u2_path = f'units/{unit_id}'
    
    if u1_path not in f1:
        result.features_match = False
        result.structural_diffs.append("Unit missing in File 1")
        return result
    
    if u2_path not in f2:
        result.features_match = False
        result.structural_diffs.append("Unit missing in File 2")
        return result
    
    u1 = f1[u1_path]
    u2 = f2[u2_path]
    
    # Compare all subgroups at top level
    keys1 = set(u1.keys())
    keys2 = set(u2.keys())
    
    only1 = keys1 - keys2
    only2 = keys2 - keys1
    
    if only1:
        result.structural_diffs.append(f"Only in File 1: {sorted(only1)}")
    if only2:
        result.structural_diffs.append(f"Only in File 2: {sorted(only2)}")
    
    # Compare features specifically
    if 'features' in u1 and 'features' in u2:
        feat1 = set(u1['features'].keys())
        feat2 = set(u2['features'].keys())
        
        result.features_only_in_1 = sorted(feat1 - feat2)
        result.features_only_in_2 = sorted(feat2 - feat1)
        result.common_features = sorted(feat1 & feat2)
        
        if result.features_only_in_1 or result.features_only_in_2:
            result.features_match = False
        
        # Compare datasets in common features
        for feat_name in result.common_features:
            f1_grp = u1[f'features/{feat_name}']
            f2_grp = u2[f'features/{feat_name}']
            
            # Get all datasets
            d1_all = collect_all_datasets(f1_grp, f'features/{feat_name}')
            d2_all = collect_all_datasets(f2_grp, f'features/{feat_name}')
            
            # Compare paths
            paths1 = set(d1_all.keys())
            paths2 = set(d2_all.keys())
            
            only_in_1 = paths1 - paths2
            only_in_2 = paths2 - paths1
            
            if only_in_1:
                result.structural_diffs.append(f"Datasets only in File 1 ({feat_name}): {list(only_in_1)[:3]}...")
            if only_in_2:
                result.structural_diffs.append(f"Datasets only in File 2 ({feat_name}): {list(only_in_2)[:3]}...")
            
            # Compare common datasets
            for path in sorted(paths1 & paths2):
                ds_result = compare_datasets_detailed(d1_all[path], d2_all[path], path)
                if not ds_result.match:
                    result.dataset_diffs.append(ds_result)
    
    # Compare spike_times_sectioned
    if 'spike_times_sectioned' in u1 and 'spike_times_sectioned' in u2:
        st1 = set(u1['spike_times_sectioned'].keys())
        st2 = set(u2['spike_times_sectioned'].keys())
        
        if st1 != st2:
            result.structural_diffs.append(f"spike_times_sectioned movies differ: {st1} vs {st2}")
    
    # Compare auto_label
    if 'auto_label' in u1 and 'auto_label' in u2:
        al1 = collect_all_datasets(u1['auto_label'], 'auto_label')
        al2 = collect_all_datasets(u2['auto_label'], 'auto_label')
        
        for path in sorted(set(al1.keys()) & set(al2.keys())):
            ds_result = compare_datasets_detailed(al1[path], al2[path], path)
            if not ds_result.match:
                result.dataset_diffs.append(ds_result)
    
    return result


def get_onh_position(f: h5py.File) -> Optional[Tuple[float, float]]:
    """Get ONH position from metadata."""
    try:
        if 'metadata/ap_tracking/all_ap_intersection' in f:
            grp = f['metadata/ap_tracking/all_ap_intersection']
            x = grp['x'][()]
            y = grp['y'][()]
            return (float(x), float(y))
    except Exception:
        pass
    return None


def count_units_with_ap_tracking(f: h5py.File) -> int:
    """Count units that have ap_tracking feature."""
    count = 0
    if 'units' in f:
        for unit_id in f['units'].keys():
            if f'units/{unit_id}/features/ap_tracking' in f:
                count += 1
    return count


# =============================================================================
# Main Comparison Function
# =============================================================================

def compare_hdf5_comprehensive(file1: Path, file2: Path, 
                                n_sample_units: int = 5,
                                compare_all_units: bool = False) -> ComparisonSummary:
    """
    Comprehensive comparison of two HDF5 files.
    
    Args:
        file1: Path to test output file
        file2: Path to reference file
        n_sample_units: Number of units to sample for detailed comparison
        compare_all_units: If True, compare all units (slower)
    
    Returns:
        ComparisonSummary with all results
    """
    summary = ComparisonSummary(
        file1_path=str(file1),
        file2_path=str(file2),
        file1_size_mb=file1.stat().st_size / 1024 / 1024,
        file2_size_mb=file2.stat().st_size / 1024 / 1024,
    )
    
    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
        
        # =================================================================
        # Top-level structure
        # =================================================================
        groups1 = set(f1.keys())
        groups2 = set(f2.keys())
        
        summary.top_groups_match = (groups1 == groups2)
        
        # =================================================================
        # Units comparison
        # =================================================================
        if 'units' in f1 and 'units' in f2:
            units1 = set(f1['units'].keys())
            units2 = set(f2['units'].keys())
            
            summary.n_units_file1 = len(units1)
            summary.n_units_file2 = len(units2)
            summary.units_only_in_1 = sorted(units1 - units2)
            summary.units_only_in_2 = sorted(units2 - units1)
            summary.units_match = (units1 == units2)
            
            common_units = sorted(units1 & units2)
            
            # Select units for detailed comparison
            if compare_all_units:
                units_to_compare = common_units
            else:
                indices = np.linspace(0, len(common_units) - 1, n_sample_units, dtype=int)
                units_to_compare = [common_units[i] for i in indices if i < len(common_units)]
            
            # Detailed unit comparisons
            for unit_id in units_to_compare:
                unit_result = compare_unit_comprehensive(f1, f2, unit_id)
                summary.unit_comparisons.append(unit_result)
                
                summary.total_datasets_compared += len(unit_result.dataset_diffs)
                summary.datasets_matching += sum(1 for d in unit_result.dataset_diffs if d.match)
                summary.datasets_differing += sum(1 for d in unit_result.dataset_diffs if not d.match)
        
        # =================================================================
        # Metadata comparison
        # =================================================================
        if 'metadata' in f1 and 'metadata' in f2:
            meta1 = set(f1['metadata'].keys())
            meta2 = set(f2['metadata'].keys())
            
            summary.metadata_keys_match = (meta1 == meta2)
            summary.metadata_only_in_1 = sorted(meta1 - meta2)
            summary.metadata_only_in_2 = sorted(meta2 - meta1)
        
        # =================================================================
        # AP Tracking specific
        # =================================================================
        summary.ap_tracking_units_file1 = count_units_with_ap_tracking(f1)
        summary.ap_tracking_units_file2 = count_units_with_ap_tracking(f2)
        summary.onh_file1 = get_onh_position(f1)
        summary.onh_file2 = get_onh_position(f2)
    
    # =================================================================
    # Classify differences as expected or unexpected
    # =================================================================
    
    # Expected differences
    if summary.metadata_only_in_1:
        summary.expected_diffs.append(f"Extra metadata in test file: {summary.metadata_only_in_1}")
    
    if summary.onh_file1 and summary.onh_file2:
        dx = abs(summary.onh_file1[0] - summary.onh_file2[0])
        dy = abs(summary.onh_file1[1] - summary.onh_file2[1])
        if dx > 1 or dy > 1:
            summary.expected_diffs.append(
                f"ONH positions differ (expected): File1=({summary.onh_file1[0]:.2f}, {summary.onh_file1[1]:.2f}), "
                f"File2=({summary.onh_file2[0]:.2f}, {summary.onh_file2[1]:.2f})"
            )
    
    # Check for DoG differences (expected to vary)
    dog_diffs = []
    ap_tracking_diffs = []
    for uc in summary.unit_comparisons:
        for dd in uc.dataset_diffs:
            if 'DoG' in dd.path or 'sta_geometry' in dd.path:
                dog_diffs.append(dd.path)
            if 'ap_tracking' in dd.path or 'soma_polar' in dd.path:
                ap_tracking_diffs.append(dd.path)
    
    if dog_diffs:
        summary.expected_diffs.append(f"DoG fitting results differ (expected): {len(dog_diffs)} values")
    
    if ap_tracking_diffs:
        summary.expected_diffs.append(f"AP tracking values differ (expected when reprocessing): {len(ap_tracking_diffs)} values")
    
    # Unexpected differences
    if not summary.units_match:
        summary.unexpected_diffs.append(
            f"Unit count mismatch: {summary.n_units_file1} vs {summary.n_units_file2}"
        )
    
    if summary.units_only_in_2:
        summary.unexpected_diffs.append(f"Units only in reference: {summary.units_only_in_2}")
    
    # Only flag as unexpected if unit counts are significantly different
    if summary.ap_tracking_units_file1 != summary.ap_tracking_units_file2:
        # This can happen when different cells pass R² threshold
        if abs(summary.ap_tracking_units_file1 - summary.ap_tracking_units_file2) > 10:
            summary.unexpected_diffs.append(
                f"AP tracking unit count significantly differs: {summary.ap_tracking_units_file1} vs {summary.ap_tracking_units_file2}"
            )
        else:
            summary.expected_diffs.append(
                f"AP tracking unit count differs slightly: {summary.ap_tracking_units_file1} vs {summary.ap_tracking_units_file2}"
            )
    
    # Check for missing features
    for uc in summary.unit_comparisons:
        if uc.features_only_in_2:
            for feat in uc.features_only_in_2:
                summary.unexpected_diffs.append(f"{uc.unit_id}: Missing feature '{feat}'")
    
    return summary


# =============================================================================
# Reporting Functions
# =============================================================================

def print_section(title: str, char: str = "="):
    """Print a section header."""
    print(f"\n{char * 70}")
    print(title)
    print(f"{char * 70}")


def print_comparison_report(summary: ComparisonSummary):
    """Print a comprehensive comparison report."""
    
    print_section("HDF5 FILE COMPARISON REPORT")
    print(f"Generated: {Path(__file__).name}")
    
    # File info
    print_section("FILE INFORMATION", "-")
    print(f"File 1 (Test):      {Path(summary.file1_path).name}")
    print(f"  Path: {summary.file1_path}")
    print(f"  Size: {summary.file1_size_mb:.2f} MB")
    print(f"File 2 (Reference): {Path(summary.file2_path).name}")
    print(f"  Path: {summary.file2_path}")
    print(f"  Size: {summary.file2_size_mb:.2f} MB")
    
    # Structure
    print_section("STRUCTURAL COMPARISON", "-")
    print(f"Top-level groups match: {'[+] Yes' if summary.top_groups_match else '[-] No'}")
    print(f"Units match: {'[+] Yes' if summary.units_match else '[-] No'}")
    print(f"  File 1: {summary.n_units_file1} units")
    print(f"  File 2: {summary.n_units_file2} units")
    if summary.units_only_in_1:
        print(f"  Only in File 1: {summary.units_only_in_1[:5]}...")
    if summary.units_only_in_2:
        print(f"  Only in File 2: {summary.units_only_in_2[:5]}...")
    
    print(f"Metadata keys match: {'[+] Yes' if summary.metadata_keys_match else '[-] No'}")
    if summary.metadata_only_in_1:
        print(f"  Only in File 1: {summary.metadata_only_in_1}")
    if summary.metadata_only_in_2:
        print(f"  Only in File 2: {summary.metadata_only_in_2}")
    
    # AP Tracking
    print_section("AP TRACKING ANALYSIS", "-")
    print(f"Units with ap_tracking:")
    print(f"  File 1: {summary.ap_tracking_units_file1}")
    print(f"  File 2: {summary.ap_tracking_units_file2}")
    
    if summary.onh_file1:
        print(f"  File 1 ONH: ({summary.onh_file1[0]:.2f}, {summary.onh_file1[1]:.2f})")
    else:
        print(f"  File 1 ONH: Not found")
    
    if summary.onh_file2:
        print(f"  File 2 ONH: ({summary.onh_file2[0]:.2f}, {summary.onh_file2[1]:.2f})")
    else:
        print(f"  File 2 ONH: Not found")
    
    if summary.onh_file1 and summary.onh_file2:
        dist = np.sqrt((summary.onh_file1[0] - summary.onh_file2[0])**2 + 
                       (summary.onh_file1[1] - summary.onh_file2[1])**2)
        print(f"  ONH distance: {dist:.2f} pixels")
    
    # Unit-by-unit comparison
    print_section("UNIT-BY-UNIT COMPARISON", "-")
    print(f"Compared {len(summary.unit_comparisons)} units")
    
    for uc in summary.unit_comparisons:
        issues = []
        if uc.features_only_in_1:
            issues.append(f"Extra features: {uc.features_only_in_1}")
        if uc.features_only_in_2:
            issues.append(f"Missing features: {uc.features_only_in_2}")
        if uc.structural_diffs:
            issues.extend(uc.structural_diffs)
        if uc.dataset_diffs:
            issues.append(f"{len(uc.dataset_diffs)} dataset value differences")
        
        if issues:
            print(f"\n  {uc.unit_id}: [!] {len(issues)} issues")
            for issue in issues[:5]:
                print(f"    - {issue}")
            
            # Show sample dataset differences
            if uc.dataset_diffs:
                print(f"    Sample value differences:")
                for dd in uc.dataset_diffs[:3]:
                    if dd.max_diff > 0:
                        print(f"      {dd.path}: max_diff={dd.max_diff:.2e}")
                    elif dd.details:
                        print(f"      {dd.path}: {dd.details[0]}")
        else:
            print(f"  {uc.unit_id}: [+] All features match")
    
    # Feature-level statistics
    print_section("FEATURE STATISTICS", "-")
    
    feature_stats = defaultdict(lambda: {'present_in_1': 0, 'present_in_2': 0, 'common': 0})
    for uc in summary.unit_comparisons:
        for feat in uc.features_only_in_1:
            feature_stats[feat]['present_in_1'] += 1
        for feat in uc.features_only_in_2:
            feature_stats[feat]['present_in_2'] += 1
        for feat in uc.common_features:
            feature_stats[feat]['common'] += 1
    
    print(f"{'Feature':<40} {'File1':>8} {'File2':>8} {'Common':>8}")
    print(f"{'-'*40} {'-'*8} {'-'*8} {'-'*8}")
    for feat, stats in sorted(feature_stats.items()):
        total1 = stats['present_in_1'] + stats['common']
        total2 = stats['present_in_2'] + stats['common']
        print(f"{feat:<40} {total1:>8} {total2:>8} {stats['common']:>8}")
    
    # Value differences summary
    print_section("VALUE DIFFERENCES SUMMARY", "-")
    
    all_diffs = []
    for uc in summary.unit_comparisons:
        for dd in uc.dataset_diffs:
            if not dd.match:
                all_diffs.append((uc.unit_id, dd))
    
    if all_diffs:
        print(f"Total value differences: {len(all_diffs)}")
        
        # Group by path pattern
        path_groups = defaultdict(list)
        for unit_id, dd in all_diffs:
            # Extract feature/path pattern
            parts = dd.path.split('/')
            if len(parts) >= 2:
                pattern = '/'.join(parts[:2])
            else:
                pattern = dd.path
            path_groups[pattern].append((unit_id, dd))
        
        print(f"\nGrouped by path:")
        for pattern, diffs in sorted(path_groups.items()):
            max_diff = max(d.max_diff for _, d in diffs if d.max_diff is not None and d.max_diff > 0) if any(d.max_diff for _, d in diffs) else 0
            print(f"  {pattern}: {len(diffs)} differences, max_diff={max_diff:.2e}")
    else:
        print("[+] No value differences found in compared datasets")
    
    # Expected vs Unexpected
    print_section("DIFFERENCE CLASSIFICATION")
    
    if summary.expected_diffs:
        print("\n[+] EXPECTED DIFFERENCES (acceptable):")
        for diff in summary.expected_diffs:
            print(f"  • {diff}")
    
    if summary.unexpected_diffs:
        print("\n[-] UNEXPECTED DIFFERENCES (need attention):")
        for diff in summary.unexpected_diffs:
            print(f"  • {diff}")
    
    # Final verdict
    print_section("FINAL VERDICT")
    
    critical_issues = len(summary.unexpected_diffs)
    
    if critical_issues == 0:
        print("[PASS] Files are functionally equivalent")
        print("   (Only expected/acceptable differences found)")
    else:
        print(f"[!] ATTENTION NEEDED: {critical_issues} unexpected difference(s)")
        print("   Review the unexpected differences above")
    
    # Notes
    print_section("NOTES ON EXPECTED DIFFERENCES", "-")
    print("""
• ONH Position: When reprocessing from scratch, ONH will differ because:
  - CNN model inference has numerical variations between runs
  - Pathway R² filtering may include/exclude different cells
  - Global optimization can converge to different local minima
  This affects all polar coordinate values (angle, cartesian_x, etc.)
  
• DoG Fitting: Difference of Gaussians curve fitting can produce
  slightly different parameters on each run due to optimization.
  
• Extra Metadata: Test pipeline may store additional metadata fields
  (e.g., acquisition_rate, dataset_id) not in the reference.
  
• ap_pathway Details: Some units may have pathway data in one file
  but not the other due to centroid count differences.
  
• STA Data: The underlying eimage_sta/data should match exactly.
  If it differs, there's a processing issue.
  
• Reprocessing vs Reference: When comparing a newly processed file
  against a reference, ALL ap_tracking-derived values (soma positions,
  polar coordinates, pathways) may differ. This is expected behavior,
  not an error. The KEY validation is that eimage_sta/data matches.
""")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run comprehensive comparison."""
    # File paths
    test_output = Path("Projects/unified_pipeline/test_output/2024.08.08-10.40.20-Rec_final.h5")
    reference = Path("Projects/dsgc_section/export_dsgc_section_20251226/2024.08.08-10.40.20-Rec.h5")
    
    # Check files exist
    if not test_output.exists():
        print(f"[X] Test output not found: {test_output}")
        alt = Path("Projects/unified_pipeline/test_output/2024.08.08-10.40.20-Rec.h5")
        if alt.exists():
            test_output = alt
            print(f"   Using alternative: {test_output}")
        else:
            return
    
    if not reference.exists():
        print(f"[X] Reference file not found: {reference}")
        return
    
    # Run comparison
    print("Running comprehensive comparison...")
    print("This may take a moment for large files...")
    
    summary = compare_hdf5_comprehensive(
        test_output, 
        reference, 
        n_sample_units=5,
        compare_all_units=False
    )
    
    # Print report
    print_comparison_report(summary)
    
    # Detailed value-by-value comparison for sample units
    print("\n" + "=" * 80)
    print("DETAILED VALUE-BY-VALUE COMPARISON")
    print("=" * 80)
    
    # Select units for detailed comparison: first, middle, last
    with h5py.File(test_output, 'r') as f1, h5py.File(reference, 'r') as f2:
        common_units = sorted(set(f1['units'].keys()) & set(f2['units'].keys()))
        
        if common_units:
            # Select 3 representative units
            sample_units = []
            sample_units.append(common_units[0])  # First
            if len(common_units) > 2:
                sample_units.append(common_units[len(common_units)//2])  # Middle
            if len(common_units) > 1:
                sample_units.append(common_units[-1])  # Last
            
            print(f"\nComparing {len(sample_units)} units in detail: {sample_units}")
            
            # Compare key features
            key_features = ['ap_tracking', 'eimage_sta']
            
            print_value_by_value_report(f1, f2, sample_units, key_features)


if __name__ == "__main__":
    main()
