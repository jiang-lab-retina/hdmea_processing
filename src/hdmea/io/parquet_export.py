"""
Parquet export for HD-MEA pipeline.

Exports features from Zarr archives to Parquet tables for cross-recording analysis.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from hdmea.io.zarr_store import open_recording_zarr, list_units, list_features


logger = logging.getLogger(__name__)


def export_features_to_parquet(
    zarr_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    *,
    features: Optional[List[str]] = None,
    include_metadata: bool = True,
) -> Path:
    """
    Export features from Zarr archive(s) to Parquet table.
    
    Flattens per-unit features to tabular format for cross-recording analysis.
    
    Args:
        zarr_paths: List of Zarr archive paths to export from.
        output_path: Path for output Parquet file.
        features: List of feature names to include. None = all features.
        include_metadata: Include dataset_id, unit_id columns. Default: True.
    
    Returns:
        Path to created Parquet file.
    
    Raises:
        FileNotFoundError: If any Zarr path does not exist.
        ValueError: If features specified but not found in any Zarr.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_rows = []
    
    for zarr_path in zarr_paths:
        zarr_path = Path(zarr_path)
        
        if not zarr_path.exists():
            raise FileNotFoundError(f"Zarr not found: {zarr_path}")
        
        root = open_recording_zarr(zarr_path, mode="r")
        dataset_id = root.attrs.get("dataset_id", zarr_path.stem)
        
        unit_ids = list_units(root)
        logger.info(f"Exporting {len(unit_ids)} units from {dataset_id}")
        
        for unit_id in unit_ids:
            row = {}
            
            if include_metadata:
                row["dataset_id"] = dataset_id
                row["unit_id"] = unit_id
                
                # Add unit metadata
                unit_group = root["units"][unit_id]
                row["row"] = unit_group.attrs.get("row", -1)
                row["col"] = unit_group.attrs.get("col", -1)
                row["spike_count"] = unit_group.attrs.get("spike_count", 0)
            
            # Get unit features
            unit_features = list_features(root, unit_id)
            
            if features is not None:
                # Filter to requested features
                unit_features = [f for f in unit_features if f in features]
            
            for feature_name in unit_features:
                feature_group = root["units"][unit_id]["features"][feature_name]
                
                # Flatten feature data to columns
                row.update(_flatten_feature_group(feature_group, feature_name))
            
            all_rows.append(row)
    
    if not all_rows:
        logger.warning("No data to export")
        # Create empty DataFrame with minimal columns
        df = pd.DataFrame(columns=["dataset_id", "unit_id"])
    else:
        df = pd.DataFrame(all_rows)
    
    # Add export metadata
    df["_export_version"] = "1.0.0"
    df["_export_timestamp"] = datetime.now(timezone.utc).isoformat()
    
    # Write to Parquet
    table = pa.Table.from_pandas(df)
    
    # Add file-level metadata
    metadata = {
        b"hdmea_export_version": b"1.0.0",
        b"created_at": datetime.now(timezone.utc).isoformat().encode(),
        b"source_zarrs": ",".join(str(p) for p in zarr_paths).encode(),
    }
    table = table.replace_schema_metadata(metadata)
    
    pq.write_table(table, output_path, compression="snappy")
    
    logger.info(f"Exported {len(df)} rows to {output_path}")
    
    return output_path


def _flatten_feature_group(
    feature_group,
    feature_name: str,
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Flatten a feature group to dictionary of column values.
    
    Args:
        feature_group: Zarr group containing feature data
        feature_name: Name of the feature
        prefix: Prefix for nested group names
    
    Returns:
        Dictionary of column_name -> value
    """
    result = {}
    full_prefix = f"{feature_name}__{prefix}" if prefix else f"{feature_name}__"
    
    # Process arrays (datasets)
    for key in feature_group.keys():
        item = feature_group[key]
        
        if hasattr(item, "keys"):
            # Nested group - recurse
            nested = _flatten_feature_group(item, feature_name, f"{prefix}{key}__")
            result.update(nested)
        else:
            # Dataset (array)
            data = item[:]
            col_name = f"{full_prefix}{key}"
            
            if data.ndim == 0:
                # Scalar
                result[col_name] = data.item()
            elif data.ndim == 1 and len(data) <= 10:
                # Small array - store each element
                for i, val in enumerate(data):
                    result[f"{col_name}_{i}"] = val
            else:
                # Large array - store summary statistics
                result[f"{col_name}_mean"] = float(np.mean(data))
                result[f"{col_name}_std"] = float(np.std(data))
                result[f"{col_name}_min"] = float(np.min(data))
                result[f"{col_name}_max"] = float(np.max(data))
    
    # Process scalar attributes
    for key, value in feature_group.attrs.items():
        if key.startswith("_"):
            continue  # Skip internal attributes
        col_name = f"{full_prefix}{key}"
        result[col_name] = value
    
    return result


def read_exported_features(parquet_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read exported features from Parquet file.
    
    Args:
        parquet_path: Path to Parquet file
    
    Returns:
        DataFrame with exported features
    """
    return pd.read_parquet(parquet_path)


def get_export_metadata(parquet_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get metadata from exported Parquet file.
    
    Args:
        parquet_path: Path to Parquet file
    
    Returns:
        Dictionary of metadata
    """
    parquet_file = pq.ParquetFile(parquet_path)
    metadata = parquet_file.schema_arrow.pandas_metadata or {}
    file_metadata = parquet_file.metadata.metadata or {}
    
    return {
        "pandas_metadata": metadata,
        "file_metadata": {
            k.decode(): v.decode() for k, v in file_metadata.items()
        },
        "num_rows": parquet_file.metadata.num_rows,
        "num_columns": parquet_file.metadata.num_columns,
    }

