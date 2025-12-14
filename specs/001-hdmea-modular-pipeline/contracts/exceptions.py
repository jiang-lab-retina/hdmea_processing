"""
Exception Contracts

This module defines the exception hierarchy for the HD-MEA pipeline.
Implementation MUST use these exceptions for error handling.

Date: 2025-12-14
Plan: ../plan.md
"""


class HDMEAError(Exception):
    """
    Base exception for all HD-MEA pipeline errors.
    
    All custom exceptions MUST inherit from this class.
    """
    pass


class ConfigurationError(HDMEAError):
    """
    Invalid configuration or parameters.
    
    Raised when:
    - Neither cmcr_path nor cmtr_path provided
    - dataset_id format invalid
    - Flow config not found or malformed
    - Stimulus config invalid
    
    Example:
        raise ConfigurationError(
            "At least one of cmcr_path or cmtr_path must be provided"
        )
    """
    pass


class DataLoadError(HDMEAError):
    """
    Error loading raw data files.
    
    Raised when:
    - .cmcr/.cmtr file cannot be read
    - File format is corrupt or unsupported
    - McsPy raises an error
    
    Example:
        raise DataLoadError(
            f"Cannot read CMTR file: {path}",
            file_path=path,
            original_error=e
        )
    """
    def __init__(
        self, 
        message: str, 
        file_path: str | None = None,
        original_error: Exception | None = None
    ):
        super().__init__(message)
        self.file_path = file_path
        self.original_error = original_error


class FeatureExtractionError(HDMEAError):
    """
    Error during feature extraction.
    
    Raised when:
    - Feature computation fails
    - Numerical errors occur
    - Unexpected data shape
    
    Example:
        raise FeatureExtractionError(
            f"Failed to compute DSI for unit {unit_id}: {reason}",
            feature_name="moving_h_bar_s5_d8_3x",
            unit_id=unit_id
        )
    """
    def __init__(
        self, 
        message: str, 
        feature_name: str | None = None,
        unit_id: str | None = None,
        original_error: Exception | None = None
    ):
        super().__init__(message)
        self.feature_name = feature_name
        self.unit_id = unit_id
        self.original_error = original_error


class MissingInputError(HDMEAError):
    """
    Required input not found in Zarr.
    
    Raised when:
    - Feature extractor requires data not present in Zarr
    - Stimulus data missing for stimulus-aligned features
    - Unit data incomplete
    
    Example:
        raise MissingInputError(
            f"Feature '{feature}' requires 'stimulus/light_reference' which is not present",
            feature_name=feature,
            missing_input="stimulus/light_reference"
        )
    """
    def __init__(
        self, 
        message: str, 
        feature_name: str | None = None,
        missing_input: str | None = None
    ):
        super().__init__(message)
        self.feature_name = feature_name
        self.missing_input = missing_input


class CacheConflictError(HDMEAError):
    """
    Feature already exists and force=False.
    
    Raised when:
    - Attempting to extract a feature that already exists
    - force=False (default behavior)
    
    Example:
        raise CacheConflictError(
            f"Feature '{feature}' already exists for unit '{unit_id}'. "
            f"Use force=True to overwrite.",
            feature_name=feature,
            unit_id=unit_id,
            existing_version=existing_version
        )
    """
    def __init__(
        self, 
        message: str, 
        feature_name: str | None = None,
        unit_id: str | None = None,
        existing_version: str | None = None
    ):
        super().__init__(message)
        self.feature_name = feature_name
        self.unit_id = unit_id
        self.existing_version = existing_version


class ValidationError(HDMEAError):
    """
    Data validation failed.
    
    Raised when:
    - Spike times not monotonically increasing
    - Waveform shape incorrect
    - Zarr structure invalid
    
    Example:
        raise ValidationError(
            f"Spike times for unit {unit_id} are not monotonically increasing",
            entity="unit",
            entity_id=unit_id,
            field="spike_times"
        )
    """
    def __init__(
        self, 
        message: str, 
        entity: str | None = None,
        entity_id: str | None = None,
        field: str | None = None
    ):
        super().__init__(message)
        self.entity = entity
        self.entity_id = entity_id
        self.field = field


class MismatchError(HDMEAError):
    """
    Input files do not match.
    
    Raised when:
    - .cmcr and .cmtr metadata indicates different recordings
    - Timestamp mismatch between files
    
    Example:
        raise MismatchError(
            f"CMCR and CMTR files do not match: {reason}",
            cmcr_path=cmcr_path,
            cmtr_path=cmtr_path,
            reason=reason
        )
    """
    def __init__(
        self, 
        message: str, 
        cmcr_path: str | None = None,
        cmtr_path: str | None = None,
        reason: str | None = None
    ):
        super().__init__(message)
        self.cmcr_path = cmcr_path
        self.cmtr_path = cmtr_path
        self.reason = reason

