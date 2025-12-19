"""
Input validation utilities for HD-MEA pipeline.
"""

import re
from pathlib import Path
from typing import Optional, Union

from hdmea.utils.exceptions import ConfigurationError


# Dataset ID pattern: any string without path-unsafe characters
DATASET_ID_PATTERN = re.compile(r'^[^/\\:*?"<>|]+$')


def validate_dataset_id(dataset_id: str) -> str:
    """
    Validate and normalize a dataset ID.
    
    Dataset IDs can be any string without path-unsafe characters.
    Trailing hyphens are stripped for consistency.
    Examples: 2025.04.10-11.12.57-Rec, JIANG009_2024-01-15
    
    Args:
        dataset_id: Dataset identifier to validate
    
    Returns:
        Validated dataset ID (trailing hyphens stripped)
    
    Raises:
        ConfigurationError: If dataset_id format is invalid
    """
    if not dataset_id:
        raise ConfigurationError("dataset_id cannot be empty")
    
    # Strip trailing hyphens for consistency
    dataset_id = dataset_id.rstrip("-")
    
    if not dataset_id:
        raise ConfigurationError("dataset_id cannot be empty after stripping trailing hyphens")
    
    if not DATASET_ID_PATTERN.match(dataset_id):
        raise ConfigurationError(
            f"Invalid dataset_id: '{dataset_id}'. "
            f"Dataset ID cannot contain path-unsafe characters (/, \\, :, *, ?, \", <, >, |)."
        )
    
    return dataset_id


def validate_path_exists(path: Union[str, Path], file_type: str = "file") -> Path:
    """
    Validate that a path exists.
    
    Args:
        path: Path to validate (string or Path object)
        file_type: Description of what the path should be (for error messages)
    
    Returns:
        Path object
    
    Raises:
        FileNotFoundError: If path does not exist
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"{file_type} not found: {path}")
    
    return path


def validate_cmcr_path(path: Union[str, Path, None]) -> Optional[Path]:
    """
    Validate a CMCR file path.
    
    Args:
        path: Path to .cmcr file (or None)
    
    Returns:
        Validated Path object, or None if input is None
    
    Raises:
        ConfigurationError: If path is invalid
        FileNotFoundError: If file does not exist
    """
    if path is None:
        return None
    
    path = Path(path)
    
    if path.suffix.lower() != ".cmcr":
        raise ConfigurationError(f"Expected .cmcr file, got: {path}")
    
    return validate_path_exists(path, "CMCR file")


def validate_cmtr_path(path: Union[str, Path, None]) -> Optional[Path]:
    """
    Validate a CMTR file path.
    
    Args:
        path: Path to .cmtr file (or None)
    
    Returns:
        Validated Path object, or None if input is None
    
    Raises:
        ConfigurationError: If path is invalid
        FileNotFoundError: If file does not exist
    """
    if path is None:
        return None
    
    path = Path(path)
    
    if path.suffix.lower() != ".cmtr":
        raise ConfigurationError(f"Expected .cmtr file, got: {path}")
    
    return validate_path_exists(path, "CMTR file")


def validate_input_files(
    cmcr_path: Union[str, Path, None],
    cmtr_path: Union[str, Path, None]
) -> tuple[Optional[Path], Optional[Path]]:
    """
    Validate that at least one input file is provided.
    
    Args:
        cmcr_path: Path to .cmcr file (or None)
        cmtr_path: Path to .cmtr file (or None)
    
    Returns:
        Tuple of (validated_cmcr_path, validated_cmtr_path)
    
    Raises:
        ConfigurationError: If neither file is provided
    """
    if cmcr_path is None and cmtr_path is None:
        raise ConfigurationError(
            "At least one of cmcr_path or cmtr_path must be provided"
        )
    
    validated_cmcr = validate_cmcr_path(cmcr_path)
    validated_cmtr = validate_cmtr_path(cmtr_path)
    
    return validated_cmcr, validated_cmtr


def derive_dataset_id(cmcr_path: Optional[Path], cmtr_path: Optional[Path]) -> str:
    """
    Derive dataset_id from file path if not provided.
    
    Uses the stem of the first available file path, with trailing hyphens removed.
    
    Args:
        cmcr_path: Path to .cmcr file (or None)
        cmtr_path: Path to .cmtr file (or None)
    
    Returns:
        Derived dataset ID (original case, trailing hyphens stripped)
    
    Raises:
        ConfigurationError: If no path is available
    """
    path = cmtr_path or cmcr_path
    
    if path is None:
        raise ConfigurationError("Cannot derive dataset_id: no file path provided")
    
    # Use file stem, remove trailing hyphens
    return path.stem.rstrip("-")

