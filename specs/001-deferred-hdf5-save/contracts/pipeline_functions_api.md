# API Contract: Modified Pipeline Functions

**Module**: `hdmea.pipeline.runner`  
**Version**: 2.0.0 (backwards compatible)

## Overview

All existing pipeline functions gain an optional `session` parameter. When provided, data accumulates in the session instead of writing to HDF5 immediately.

---

## load_recording

```python
def load_recording(
    cmcr_path: Optional[Union[str, Path]] = None,
    cmtr_path: Optional[Union[str, Path]] = None,
    dataset_id: Optional[str] = None,
    *,
    output_dir: Union[str, Path] = "artifacts",
    force: bool = False,
    allow_overwrite: bool = False,
    config: Optional[Dict[str, Any]] = None,
    # NEW PARAMETER
    session: Optional[PipelineSession] = None,
) -> Union[LoadResult, PipelineSession]:
    """
    Load recording from external .cmcr/.cmtr files.
    
    Behavior:
        - session=None (default): Immediate save to HDF5, returns LoadResult
        - session provided: Data added to session, returns session
    
    Args:
        cmcr_path: Path to .cmcr file
        cmtr_path: Path to .cmtr file
        dataset_id: Unique recording identifier
        output_dir: Directory for HDF5 output (ignored if session provided)
        force: Overwrite existing HDF5
        allow_overwrite: Allow overwrite even if params differ
        config: Optional configuration
        session: Optional PipelineSession for deferred save
    
    Returns:
        LoadResult if session=None, otherwise the updated PipelineSession
    """
```

---

## load_recording_with_eimage_sta

```python
def load_recording_with_eimage_sta(
    cmcr_path: Union[str, Path],
    cmtr_path: Union[str, Path],
    dataset_id: Optional[str] = None,
    *,
    output_dir: Union[str, Path] = "artifacts",
    force: bool = False,
    # eimage_sta parameters
    duration_s: float = 120.0,
    spike_limit: int = 10000,
    unit_ids: Optional[List[str]] = None,
    window_range: Tuple[int, int] = (-10, 40),
    cutoff_hz: float = 100.0,
    filter_order: int = 2,
    skip_highpass: bool = False,
    chunk_duration_s: float = 30.0,
    # NEW PARAMETER
    session: Optional[PipelineSession] = None,
) -> Union[LoadWithEImageSTAResult, PipelineSession]:
    """
    Load recording and compute eimage_sta in one pass.
    
    Behavior:
        - session=None (default): Immediate save to HDF5, returns LoadWithEImageSTAResult
        - session provided: Data added to session, returns session
    
    Returns:
        LoadWithEImageSTAResult if session=None, otherwise the updated PipelineSession
    """
```

---

## extract_features

```python
def extract_features(
    hdf5_path: Optional[Union[str, Path]] = None,
    features: List[str] = [],
    *,
    force: bool = False,
    config_overrides: Optional[Dict[str, Any]] = None,
    # NEW PARAMETER
    session: Optional[PipelineSession] = None,
) -> Union[ExtractionResult, PipelineSession]:
    """
    Extract features from recording data.
    
    Behavior:
        - session=None: Requires hdf5_path, writes features to HDF5
        - session provided (DEFERRED): Uses session data, stores features in session
        - session provided (SAVED): Auto-reads from session.hdf5_path
    
    Args:
        hdf5_path: Path to HDF5 file (ignored if session provided in DEFERRED mode)
        features: List of feature names to extract
        force: Overwrite existing features
        config_overrides: Feature configuration overrides
        session: Optional PipelineSession
    
    Returns:
        ExtractionResult if session=None, otherwise the updated PipelineSession
    """
```

---

## add_section_time

```python
def add_section_time(
    hdf5_path: Optional[Union[str, Path]] = None,
    playlist_name: str = "",
    *,
    force: bool = False,
    # NEW PARAMETER
    session: Optional[PipelineSession] = None,
) -> Union[bool, PipelineSession]:
    """
    Add section timing information from playlist.
    
    Behavior:
        - session=None: Requires hdf5_path, writes to HDF5
        - session provided: Adds section_time to session.stimulus
    
    Returns:
        True/False if session=None, otherwise the updated PipelineSession
    """
```

---

## section_spike_times

```python
def section_spike_times(
    hdf5_path: Optional[Union[str, Path]] = None,
    trial_repeats: int = 3,
    pad_margin: Tuple[float, float] = (0.0, 0.0),
    *,
    force: bool = False,
    # NEW PARAMETER
    session: Optional[PipelineSession] = None,
) -> Union[SectionResult, PipelineSession]:
    """
    Section spike times by trial boundaries.
    
    Behavior:
        - session=None: Requires hdf5_path, reads/writes HDF5
        - session provided: Operates on session data in memory
    
    Returns:
        SectionResult if session=None, otherwise the updated PipelineSession
    """
```

---

## compute_sta

```python
def compute_sta(
    hdf5_path: Optional[Union[str, Path]] = None,
    cover_range: Tuple[int, int] = (-60, 0),
    *,
    use_multiprocessing: bool = True,
    force: bool = False,
    # NEW PARAMETER
    session: Optional[PipelineSession] = None,
) -> Union[STAResult, PipelineSession]:
    """
    Compute spike-triggered average.
    
    Behavior:
        - session=None: Requires hdf5_path, reads/writes HDF5
        - session provided: Operates on session data in memory
    
    Returns:
        STAResult if session=None, otherwise the updated PipelineSession
    """
```

---

## Helper: create_session

```python
def create_session(
    dataset_id: Optional[str] = None,
    cmcr_path: Optional[Union[str, Path]] = None,
    cmtr_path: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "artifacts",
) -> PipelineSession:
    """
    Convenience function to create a new PipelineSession.
    
    If dataset_id is not provided, derives it from file paths.
    
    Args:
        dataset_id: Unique recording identifier
        cmcr_path: Optional path to derive dataset_id from
        cmtr_path: Optional path to derive dataset_id from
        output_dir: Default output directory
    
    Returns:
        New PipelineSession in DEFERRED state
    """
```

