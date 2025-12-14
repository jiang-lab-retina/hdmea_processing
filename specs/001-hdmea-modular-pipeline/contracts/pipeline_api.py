"""
Pipeline API Contracts

This module defines the public API interfaces for the HD-MEA pipeline.
Implementation MUST conform to these signatures and behaviors.

Date: 2025-12-14
Plan: ../plan.md
"""

from __future__ import annotations
from pathlib import Path
from typing import Protocol, Literal, Any
from dataclasses import dataclass


# =============================================================================
# Type Definitions
# =============================================================================

@dataclass
class LoadResult:
    """Result of Stage 1 data loading."""
    zarr_path: Path
    dataset_id: str
    num_units: int
    stage1_completed: bool
    warnings: list[str]


@dataclass
class ExtractionResult:
    """Result of Stage 2 feature extraction."""
    zarr_path: Path
    features_extracted: list[str]
    features_skipped: list[str]  # Cache hits
    features_failed: list[str]
    warnings: list[str]


@dataclass
class FlowResult:
    """Result of running a complete flow."""
    zarr_path: Path
    load_result: LoadResult | None
    extraction_result: ExtractionResult | None
    success: bool


# =============================================================================
# Stage 1: Data Loading API
# =============================================================================

def load_recording(
    cmcr_path: str | Path | None = None,
    cmtr_path: str | Path | None = None,
    dataset_id: str | None = None,
    *,
    output_dir: str | Path = "artifacts",
    force: bool = False,
) -> LoadResult:
    """
    Load recording from external .cmcr/.cmtr files to Zarr artifact.
    
    This is Stage 1 of the pipeline. Produces exactly ONE Zarr archive
    per recording containing all data needed for feature extraction.
    
    Args:
        cmcr_path: External path to .cmcr file (raw sensor data).
                   May be UNC path (e.g., "//server/share/file.cmcr").
                   At least one of cmcr_path/cmtr_path must be provided.
        cmtr_path: External path to .cmtr file (spike-sorted data).
                   May be UNC path. At least one of cmcr_path/cmtr_path required.
        dataset_id: Unique identifier for the recording. If None, derived
                    from file name using documented rules.
        output_dir: Directory for Zarr output. Default: "artifacts".
        force: If True, overwrite existing Zarr. Default: False (skip if exists).
    
    Returns:
        LoadResult with zarr_path, dataset_id, unit count, and any warnings.
    
    Raises:
        ConfigurationError: If neither cmcr_path nor cmtr_path provided.
        FileNotFoundError: If specified file(s) do not exist.
        DataLoadError: If files cannot be read or are corrupt.
        IOError: If Zarr cannot be written.
    
    Example:
        >>> result = load_recording(
        ...     cmcr_path="//server/data/JIANG009.cmcr",
        ...     cmtr_path="//server/data/JIANG009.cmtr",
        ...     dataset_id="JIANG009_2024-01-15"
        ... )
        >>> print(result.zarr_path)
        artifacts/JIANG009_2024-01-15.zarr
    """
    ...


# =============================================================================
# Stage 2: Feature Extraction API
# =============================================================================

def extract_features(
    zarr_path: str | Path,
    features: list[str],
    *,
    force: bool = False,
    config_overrides: dict[str, Any] | None = None,
) -> ExtractionResult:
    """
    Extract features from loaded Zarr and write back to same archive.
    
    This is Stage 2 of the pipeline. Reads from Zarr, computes features
    for each unit, and writes results to units/{unit_id}/features/{feature_name}/.
    
    Args:
        zarr_path: Path to Zarr archive from Stage 1.
        features: List of feature names to extract (must be registered).
        force: If True, overwrite existing features. Default: False (skip cache hits).
        config_overrides: Optional parameter overrides for extractors.
    
    Returns:
        ExtractionResult with lists of extracted, skipped, and failed features.
    
    Raises:
        FileNotFoundError: If zarr_path does not exist.
        KeyError: If any feature name is not registered.
        MissingInputError: If required inputs for a feature are missing.
        FeatureExtractionError: If extraction fails (includes details).
    
    Example:
        >>> result = extract_features(
        ...     zarr_path="artifacts/JIANG009_2024-01-15.zarr",
        ...     features=["step_up_5s_5i_3x", "moving_h_bar_s5_d8_3x"]
        ... )
        >>> print(result.features_extracted)
        ['step_up_5s_5i_3x', 'moving_h_bar_s5_d8_3x']
    """
    ...


# =============================================================================
# Flow Orchestration API
# =============================================================================

def run_flow(
    flow_name: str,
    cmcr_path: str | Path | None = None,
    cmtr_path: str | Path | None = None,
    dataset_id: str | None = None,
    *,
    output_dir: str | Path = "artifacts",
    force_load: bool = False,
    force_extract: bool = False,
) -> FlowResult:
    """
    Run a named pipeline flow (Stage 1 + Stage 2).
    
    Flows are defined in config/flows/{flow_name}.json and specify
    which features to extract.
    
    Args:
        flow_name: Name of flow configuration (e.g., "set6a_full").
        cmcr_path: External path to .cmcr file.
        cmtr_path: External path to .cmtr file.
        dataset_id: Unique identifier for the recording.
        output_dir: Directory for Zarr output.
        force_load: Force re-run of Stage 1 even if Zarr exists.
        force_extract: Force re-extraction of features even if cached.
    
    Returns:
        FlowResult with combined results from both stages.
    
    Raises:
        ConfigurationError: If flow config not found or invalid.
        (Plus all exceptions from load_recording and extract_features)
    
    Example:
        >>> result = run_flow(
        ...     flow_name="set6a_full",
        ...     cmcr_path="//server/data/JIANG009.cmcr",
        ...     cmtr_path="//server/data/JIANG009.cmtr",
        ...     dataset_id="JIANG009_2024-01-15"
        ... )
        >>> print(result.success)
        True
    """
    ...


# =============================================================================
# Feature Registry API
# =============================================================================

class FeatureExtractorProtocol(Protocol):
    """Protocol defining the feature extractor interface."""
    
    name: str
    version: str
    required_inputs: list[str]
    output_schema: dict[str, dict]
    runtime_class: Literal["fast", "slow"]
    
    def extract(self, unit_data: Any, stimulus_data: Any) -> dict:
        """
        Extract features for a single unit.
        
        Args:
            unit_data: Zarr group for the unit (spike_times, waveform, etc.)
            stimulus_data: Zarr group with stimulus information
        
        Returns:
            Dict of feature name -> value (scalar or array)
        """
        ...


class FeatureRegistryProtocol(Protocol):
    """Protocol defining the feature registry interface."""
    
    @classmethod
    def register(cls, name: str) -> Any:
        """Decorator to register a feature extractor."""
        ...
    
    @classmethod
    def get(cls, name: str) -> type[FeatureExtractorProtocol]:
        """Get a registered feature extractor by name."""
        ...
    
    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered feature names."""
        ...
    
    @classmethod
    def get_metadata(cls, name: str) -> dict:
        """Get metadata for a registered feature."""
        ...


# =============================================================================
# Export API
# =============================================================================

def export_features_to_parquet(
    zarr_paths: list[str | Path],
    output_path: str | Path,
    *,
    features: list[str] | None = None,
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
    
    Example:
        >>> path = export_features_to_parquet(
        ...     zarr_paths=["artifacts/JIANG009.zarr", "artifacts/JIANG010.zarr"],
        ...     output_path="exports/combined_features.parquet",
        ...     features=["step_up_5s_5i_3x"]
        ... )
    """
    ...

