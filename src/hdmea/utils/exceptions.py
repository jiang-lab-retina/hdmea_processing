"""
Exception hierarchy for HD-MEA pipeline.

All custom exceptions inherit from HDMEAError.
"""

from typing import Optional


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
    """
    pass


class DataLoadError(HDMEAError):
    """
    Error loading raw data files.
    
    Raised when:
    - .cmcr/.cmtr file cannot be read
    - File format is corrupt or unsupported
    - McsPy raises an error
    """
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        original_error: Optional[Exception] = None,
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
    """
    
    def __init__(
        self,
        message: str,
        feature_name: Optional[str] = None,
        unit_id: Optional[str] = None,
        original_error: Optional[Exception] = None,
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
    """
    
    def __init__(
        self,
        message: str,
        feature_name: Optional[str] = None,
        missing_input: Optional[str] = None,
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
    """
    
    def __init__(
        self,
        message: str,
        feature_name: Optional[str] = None,
        unit_id: Optional[str] = None,
        existing_version: Optional[str] = None,
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
    """
    
    def __init__(
        self,
        message: str,
        entity: Optional[str] = None,
        entity_id: Optional[str] = None,
        field: Optional[str] = None,
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
    """
    
    def __init__(
        self,
        message: str,
        cmcr_path: Optional[str] = None,
        cmtr_path: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        super().__init__(message)
        self.cmcr_path = cmcr_path
        self.cmtr_path = cmtr_path
        self.reason = reason


class SessionError(HDMEAError):
    """
    Error with PipelineSession operations.
    
    Raised when:
    - Session is in invalid state for requested operation
    - Session data is corrupted or incomplete
    - Required data not found in session
    """
    
    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        super().__init__(message)
        self.session_id = session_id
        self.operation = operation


class CheckpointError(HDMEAError):
    """
    Error with checkpoint operations.
    
    Raised when:
    - Checkpoint file cannot be written
    - Checkpoint file is corrupted or invalid
    - Resume from checkpoint fails
    """
    
    def __init__(
        self,
        message: str,
        checkpoint_path: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.checkpoint_path = checkpoint_path
        self.original_error = original_error
