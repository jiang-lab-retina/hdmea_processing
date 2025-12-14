"""
Configuration hashing utilities for HD-MEA pipeline.

Used for cache invalidation and artifact versioning.
"""

import hashlib
import json
from typing import Any, Dict


def hash_config(config: Dict[str, Any], prefix: str = "sha256") -> str:
    """
    Produce deterministic hash of a configuration dictionary.
    
    Args:
        config: Configuration dictionary to hash
        prefix: Hash prefix (default: "sha256")
    
    Returns:
        Hash string in format "prefix:hash_value"
    
    Example:
        >>> config = {"param1": 10, "param2": "value"}
        >>> hash_config(config)
        'sha256:a1b2c3d4e5f6...'
    """
    # Sort keys for determinism
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    hash_value = hashlib.sha256(serialized.encode()).hexdigest()[:16]
    return f"{prefix}:{hash_value}"


def hash_file(file_path: str) -> str:
    """
    Compute SHA256 hash of a file's contents.
    
    Args:
        file_path: Path to file
    
    Returns:
        Hash string in format "sha256:hash_value"
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return f"sha256:{sha256_hash.hexdigest()[:16]}"


def verify_hash(config: Dict[str, Any], expected_hash: str) -> bool:
    """
    Verify that a config matches an expected hash.
    
    Args:
        config: Configuration dictionary
        expected_hash: Expected hash string
    
    Returns:
        True if hashes match, False otherwise
    """
    actual_hash = hash_config(config)
    return actual_hash == expected_hash

