"""
Core AP tracking functionality for HDF5 pipeline.

This module provides the main entry points for AP tracking analysis:
- compute_ap_tracking: Process a single HDF5 file
- compute_ap_tracking_batch: Process multiple HDF5 files

All results are written as explicit HDF5 datasets (not attributes).
"""

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np

from .ais_refiner import AxonInitialSegment, RefinedSoma, ais_refiner, soma_refiner
from .dvnt_parser import DVNTPosition, parse_dvnt_from_center_xy
from .model_inference import load_cnn_model, run_model_inference, select_device
from .pathway_analysis import (
    APIntersection,
    APPathway,
    ONHResult,
    SomaPolarCoordinates,
    _calculate_angle_correction,
    calculate_direction_from_centroids,
    calculate_enhanced_intersection,
    calculate_optimal_intersection,
    calculate_soma_polar_coordinates,
    fit_line_to_centroids,
    # Default parameters
    DEFAULT_R2_THRESHOLD,
    DEFAULT_DIRECTION_TOLERANCE,
    DEFAULT_MAX_DISTANCE_FROM_CENTER,
    DEFAULT_CENTER_POINT,
    DEFAULT_CLUSTER_EPS,
    DEFAULT_CLUSTER_MIN_SAMPLES,
)
from .postprocess import process_predictions
from .soma_detector import find_soma_from_3d_sta

logger = logging.getLogger(__name__)

# =============================================================================
# HDF5 Reading Helpers
# =============================================================================


def read_sta_data(root: h5py.File, unit_id: str) -> Optional[np.ndarray]:
    """
    Read STA data from HDF5 for a unit.

    Args:
        root: Open HDF5 file handle
        unit_id: Unit identifier

    Returns:
        STA data array or None if not found
    """
    try:
        sta_path = f"units/{unit_id}/features/eimage_sta/data"
        if sta_path.split("/")[-1] in root["/".join(sta_path.split("/")[:-1])]:
            sta_data = root[sta_path][:]
            return sta_data
    except KeyError:
        pass

    logger.warning(f"Unit {unit_id}: eimage_sta data not found")
    return None


def read_cell_geometry(
    root: h5py.File, unit_id: str
) -> Tuple[Optional[int], Optional[int]]:
    """
    Read cell geometry (center_row, center_col) from HDF5.

    Args:
        root: Open HDF5 file handle
        unit_id: Unit identifier

    Returns:
        Tuple of (center_row, center_col) or (None, None) if not found
    """
    try:
        geom_path = f"units/{unit_id}/features/eimage_sta/geometry"
        if "geometry" in root[f"units/{unit_id}/features/eimage_sta"]:
            geom = root[geom_path]
            center_row = int(geom["center_row"][()])
            center_col = int(geom["center_col"][()])
            return center_row, center_col
    except (KeyError, TypeError):
        pass

    return None, None


def read_dvnt_metadata(root: h5py.File) -> DVNTPosition:
    """
    Read DVNT position from metadata/gsheet_row/Center_xy.

    Args:
        root: Open HDF5 file handle

    Returns:
        DVNTPosition parsed from metadata
    """
    try:
        if "metadata" in root and "gsheet_row" in root["metadata"]:
            gsheet = root["metadata/gsheet_row"]
            if "Center_xy" in gsheet:
                center_xy = gsheet["Center_xy"][()]

                # Handle numpy array (take first element if array)
                if isinstance(center_xy, np.ndarray):
                    if center_xy.size > 0:
                        center_xy = center_xy.flat[0]
                    else:
                        center_xy = None

                # Handle bytes vs string
                if isinstance(center_xy, bytes):
                    center_xy = center_xy.decode("utf-8")

                if center_xy:
                    return parse_dvnt_from_center_xy(center_xy)
    except (KeyError, TypeError) as e:
        logger.debug(f"Could not read Center_xy: {e}")

    return DVNTPosition(dv_position=None, nt_position=None, lr_position=None)


# =============================================================================
# HDF5 Writing Helpers
# =============================================================================


def _write_scalar_dataset(group: h5py.Group, name: str, value: Any) -> None:
    """Write a scalar value as an HDF5 dataset."""
    if name in group:
        del group[name]

    if value is None:
        # Store NaN for missing numeric values
        group.create_dataset(name, data=np.nan)
    elif isinstance(value, str):
        dt = h5py.special_dtype(vlen=str)
        group.create_dataset(name, data=value, dtype=dt)
    else:
        group.create_dataset(name, data=value)


def _write_array_dataset(group: h5py.Group, name: str, value: np.ndarray) -> None:
    """Write an array as an HDF5 dataset."""
    if name in group:
        del group[name]
    group.create_dataset(name, data=value, dtype=value.dtype)


def write_ap_tracking_to_hdf5(
    root: h5py.File,
    unit_id: str,
    dvnt: DVNTPosition,
    refined_soma: Optional[RefinedSoma],
    ais: Optional[AxonInitialSegment],
    prediction_data: Optional[np.ndarray],
    post_processed: Optional[Dict],
    ap_pathway: Optional[APPathway],
    intersection: Optional[APIntersection],
    polar_coords: Optional[SomaPolarCoordinates],
    onh_result: Optional[ONHResult] = None,
) -> None:
    """
    Write AP tracking results to HDF5 as explicit datasets.

    All values are stored as datasets, not attributes, per design decision D6.

    Args:
        root: Open HDF5 file handle
        unit_id: Unit identifier
        dvnt: DVNT position data
        refined_soma: Refined soma position
        ais: Axon initial segment position
        prediction_data: CNN prediction array
        post_processed: Post-processing results dict
        ap_pathway: Fitted AP pathway
        intersection: Optimal intersection point
        polar_coords: Soma polar coordinates
    """
    features_path = f"units/{unit_id}/features"

    # Ensure features group exists
    if features_path not in root:
        root.create_group(features_path)

    features_group = root[features_path]

    # Delete existing ap_tracking group if present (always overwrite per FR-014)
    if "ap_tracking" in features_group:
        del features_group["ap_tracking"]

    # Create ap_tracking group
    ap_group = features_group.create_group("ap_tracking")

    # Write DVNT positions
    _write_scalar_dataset(ap_group, "DV_position", dvnt.dv_position)
    _write_scalar_dataset(ap_group, "NT_position", dvnt.nt_position)
    _write_scalar_dataset(ap_group, "LR_position", dvnt.lr_position)

    # Write refined soma
    soma_group = ap_group.create_group("refined_soma")
    if refined_soma:
        _write_scalar_dataset(soma_group, "t", refined_soma.t)
        _write_scalar_dataset(soma_group, "x", refined_soma.x)
        _write_scalar_dataset(soma_group, "y", refined_soma.y)
    else:
        _write_scalar_dataset(soma_group, "t", None)
        _write_scalar_dataset(soma_group, "x", None)
        _write_scalar_dataset(soma_group, "y", None)

    # Write axon initial segment
    ais_group = ap_group.create_group("axon_initial_segment")
    if ais:
        _write_scalar_dataset(ais_group, "t", ais.t)
        _write_scalar_dataset(ais_group, "x", ais.x)
        _write_scalar_dataset(ais_group, "y", ais.y)
    else:
        _write_scalar_dataset(ais_group, "t", None)
        _write_scalar_dataset(ais_group, "x", None)
        _write_scalar_dataset(ais_group, "y", None)

    # Write prediction data
    if prediction_data is not None:
        _write_array_dataset(ap_group, "prediction_sta_data", prediction_data.astype(np.float32))

    # Write post-processed data
    if post_processed:
        pp_group = ap_group.create_group("post_processed_data")
        if "filtered_prediction" in post_processed:
            _write_array_dataset(
                pp_group, "filtered_prediction",
                post_processed["filtered_prediction"].astype(np.float32)
            )
        if "axon_centroids" in post_processed:
            _write_array_dataset(
                pp_group, "axon_centroids",
                post_processed["axon_centroids"].astype(np.float32)
            )

    # Write AP pathway
    pathway_group = ap_group.create_group("ap_pathway")
    if ap_pathway:
        _write_scalar_dataset(pathway_group, "slope", ap_pathway.slope)
        _write_scalar_dataset(pathway_group, "intercept", ap_pathway.intercept)
        _write_scalar_dataset(pathway_group, "r_value", ap_pathway.r_value)
        _write_scalar_dataset(pathway_group, "p_value", ap_pathway.p_value)
        _write_scalar_dataset(pathway_group, "std_err", ap_pathway.std_err)
    else:
        for key in ["slope", "intercept", "r_value", "p_value", "std_err"]:
            _write_scalar_dataset(pathway_group, key, None)

    # Write intersection (enhanced ONH result takes precedence)
    int_group = ap_group.create_group("all_ap_intersection")
    if onh_result:
        # Enhanced intersection with clustering
        _write_scalar_dataset(int_group, "x", onh_result.x)
        _write_scalar_dataset(int_group, "y", onh_result.y)
        _write_scalar_dataset(int_group, "mse", onh_result.mse)
        _write_scalar_dataset(int_group, "rmse", onh_result.rmse)
        _write_scalar_dataset(int_group, "n_cluster_points", onh_result.n_cluster_points)
        _write_scalar_dataset(int_group, "n_total_intersections", onh_result.n_total_intersections)
        _write_scalar_dataset(int_group, "n_valid_after_direction", onh_result.n_valid_after_direction)
        _write_scalar_dataset(int_group, "consensus_direction", onh_result.consensus_direction)
        _write_scalar_dataset(int_group, "r2_threshold", onh_result.r2_threshold)
        _write_scalar_dataset(int_group, "direction_tolerance", onh_result.direction_tolerance)
        _write_scalar_dataset(int_group, "cluster_eps", onh_result.cluster_eps)
        _write_scalar_dataset(int_group, "cluster_min_samples", onh_result.cluster_min_samples)
        _write_scalar_dataset(int_group, "method", onh_result.method)
        # Save cluster points array
        if onh_result.cluster_points is not None and len(onh_result.cluster_points) > 0:
            _write_array_dataset(int_group, "cluster_points", onh_result.cluster_points)
    elif intersection:
        # Legacy intersection (fallback)
        _write_scalar_dataset(int_group, "x", intersection.x)
        _write_scalar_dataset(int_group, "y", intersection.y)
        _write_scalar_dataset(int_group, "mse", intersection.mse)
        _write_scalar_dataset(int_group, "method", "legacy_weighted_mean")
        _write_scalar_dataset(int_group, "r2_threshold", 0.0)  # Legacy doesn't filter by R²
    else:
        _write_scalar_dataset(int_group, "x", None)
        _write_scalar_dataset(int_group, "y", None)
        _write_scalar_dataset(int_group, "method", None)
        _write_scalar_dataset(int_group, "r2_threshold", None)

    # Write polar coordinates (legacy method with angle correction)
    polar_group = ap_group.create_group("soma_polar_coordinates")
    if polar_coords:
        # Basic polar coordinates
        _write_scalar_dataset(polar_group, "radius", polar_coords.radius)
        _write_scalar_dataset(polar_group, "angle", polar_coords.angle)
        _write_scalar_dataset(polar_group, "cartesian_x", polar_coords.cartesian_x)
        _write_scalar_dataset(polar_group, "cartesian_y", polar_coords.cartesian_y)
        _write_scalar_dataset(polar_group, "quadrant", polar_coords.quadrant)
        _write_scalar_dataset(polar_group, "anatomical_quadrant", polar_coords.anatomical_quadrant)

        # Legacy angle fields (degrees)
        _write_scalar_dataset(polar_group, "theta_deg", polar_coords.theta_deg)
        _write_scalar_dataset(polar_group, "theta_deg_raw", polar_coords.theta_deg_raw)
        _write_scalar_dataset(polar_group, "theta_deg_corrected", polar_coords.theta_deg_corrected)

        # Transformed coordinates (after angle correction)
        _write_scalar_dataset(polar_group, "transformed_x", polar_coords.transformed_x)
        _write_scalar_dataset(polar_group, "transformed_y", polar_coords.transformed_y)

        # Original soma position
        _write_scalar_dataset(polar_group, "original_x", polar_coords.original_x)
        _write_scalar_dataset(polar_group, "original_y", polar_coords.original_y)

        # Angle correction info
        _write_scalar_dataset(polar_group, "angle_correction_applied", polar_coords.angle_correction_applied)
    else:
        # Write None for all fields
        for key in [
            "radius", "angle", "cartesian_x", "cartesian_y", "quadrant", "anatomical_quadrant",
            "theta_deg", "theta_deg_raw", "theta_deg_corrected",
            "transformed_x", "transformed_y", "original_x", "original_y",
            "angle_correction_applied",
        ]:
            _write_scalar_dataset(polar_group, key, None)

    # Add metadata
    now = datetime.now(timezone.utc).isoformat()
    _write_scalar_dataset(ap_group, "_processed_at", now)
    _write_scalar_dataset(ap_group, "_version", "1.0.0")

    logger.debug(f"Wrote ap_tracking for unit {unit_id}")


# =============================================================================
# Unit Processing
# =============================================================================


def process_single_unit(
    root: h5py.File,
    unit_id: str,
    model: Any,
    device: str,
    dvnt: DVNTPosition,
    # Soma/AIS parameters
    soma_std_threshold: float = 3.0,
    soma_temporal_range: Tuple[int, int] = (5, 27),
    soma_refine_radius: int = 5,
    ais_search_xy_radius: int = 5,
    ais_search_t_radius: int = 5,
    # Post-processing parameters
    temporal_window_size: int = 5,
    exclude_radius: int = 5,
    centroid_threshold: float = 0.05,
    max_displacement: int = 5,
    centroid_start_frame: int = 0,
    max_displacement_post: float = 5.0,
    # Pathway fitting parameters
    min_points_for_fit: int = 10,
) -> Dict[str, Any]:
    """
    Process a single unit for AP tracking.

    Args:
        root: Open HDF5 file handle
        unit_id: Unit identifier
        model: Loaded CNN model
        device: Device string
        dvnt: DVNT position from metadata
        [other parameters]: Processing parameters

    Returns:
        Dictionary with processing results for this unit
    """
    result = {
        "unit_id": unit_id,
        "status": "skipped",
        "refined_soma": None,
        "ais": None,
        "prediction": None,
        "post_processed": None,
        "ap_pathway": None,
    }

    # Read STA data
    sta_data = read_sta_data(root, unit_id)
    if sta_data is None:
        result["skip_reason"] = "no_eimage_sta"
        logger.warning(f"Unit {unit_id}: Skipped (no eimage_sta data)")
        return result

    # Detect soma
    try:
        soma_row, soma_col = find_soma_from_3d_sta(
            sta_data,
            std_threshold=soma_std_threshold,
            sta_temporal_range=soma_temporal_range,
        )
    except Exception as e:
        logger.error(f"Unit {unit_id}: Soma detection failed: {e}")
        result["skip_reason"] = "soma_detection_failed"
        return result

    # Refine soma
    refined_soma = soma_refiner(sta_data, (soma_row, soma_col), refine_radius=soma_refine_radius)
    result["refined_soma"] = refined_soma

    # Detect AIS
    ais = None
    if refined_soma:
        ais = ais_refiner(
            sta_data, refined_soma,
            search_xy_radius=ais_search_xy_radius,
            search_t_radius=ais_search_t_radius,
        )
    result["ais"] = ais

    # Run model inference
    prediction = run_model_inference(sta_data, model, device)
    result["prediction"] = prediction

    # Post-process predictions
    post_processed = None
    if prediction is not None:
        soma_xy = (refined_soma.x, refined_soma.y) if refined_soma else None
        post_processed = process_predictions(
            prediction,
            soma_xy=soma_xy,
            temporal_window_size=temporal_window_size,
            exclude_radius=exclude_radius,
            centroid_threshold=centroid_threshold,
            max_displacement=max_displacement,
            start_frame=centroid_start_frame,
            max_displacement_post=max_displacement_post,
        )
    result["post_processed"] = post_processed

    # Fit AP pathway using axon centroids
    ap_pathway = None
    if post_processed:
        axon_centroids = post_processed.get("axon_centroids")
        ap_pathway = fit_line_to_centroids(axon_centroids, min_points=min_points_for_fit)
    result["ap_pathway"] = ap_pathway

    result["status"] = "complete"
    return result


# =============================================================================
# Main Entry Points
# =============================================================================


def compute_ap_tracking(
    hdf5_path: Path,
    model_path: Path,
    *,
    session: Optional[Any] = None,
    force_cpu: bool = False,
    max_units: Optional[int] = None,
    # Cell type filtering
    filter_by_cell_type: bool = True,
    cell_type_filter: str = "rgc",
    # Soma/AIS detection parameters
    soma_std_threshold: float = 3.0,
    soma_temporal_range: Tuple[int, int] = (5, 27),
    soma_refine_radius: int = 5,
    ais_search_xy_radius: int = 5,
    ais_search_t_radius: int = 5,
    # Model inference parameters
    batch_size: Optional[int] = None,
    # Post-processing parameters
    temporal_window_size: int = 5,
    exclude_radius: int = 5,
    centroid_threshold: float = 0.05,
    max_displacement: int = 5,
    centroid_start_frame: int = 0,
    max_displacement_post: float = 5.0,
    # Pathway fitting parameters
    min_points_for_fit: int = 10,
    r2_threshold: float = DEFAULT_R2_THRESHOLD,
    # Enhanced ONH detection parameters
    direction_tolerance: float = DEFAULT_DIRECTION_TOLERANCE,
    max_distance_from_center: float = DEFAULT_MAX_DISTANCE_FROM_CENTER,
    center_point: Tuple[float, float] = DEFAULT_CENTER_POINT,
    cluster_eps: float = DEFAULT_CLUSTER_EPS,
    cluster_min_samples: int = DEFAULT_CLUSTER_MIN_SAMPLES,
) -> Optional[Any]:
    """
    Compute AP tracking features for all units in an HDF5 file.

    This is the main entry point for AP tracking analysis. It:
    1. Filters units by cell type (RGC only by default)
    2. Reads eimage_sta data from each unit
    3. Detects soma and axon initial segment
    4. Applies CNN model to predict axon signal probability
    5. Post-processes predictions to extract axon centroids
    6. Fits AP pathway lines and calculates intersection
    7. Computes soma polar coordinates
    8. Parses DVNT positions from metadata
    9. Writes all results to units/{unit_id}/features/ap_tracking/

    Args:
        hdf5_path: Path to input HDF5 file with eimage_sta computed
        model_path: Path to trained CNN model (.pth file)
        session: Optional PipelineSession for deferred save mode
        force_cpu: Force CPU inference even if GPU available
        max_units: Maximum number of units to process (None = all)
        filter_by_cell_type: If True, only process units matching cell_type_filter.
            Cell type is read from units/{unit_id}/auto_label/axon_type.
            Units without auto_label are skipped when filtering is enabled.
        cell_type_filter: Cell type to filter for (default: "rgc").
            Only used when filter_by_cell_type=True.
        min_points_for_fit: Minimum tracked axon points required for line fitting
        r2_threshold: Minimum R² value for line fit to be considered valid

    Returns:
        If session provided: Updated PipelineSession with ap_tracking results
        If no session: None (results written directly to HDF5)

    Raises:
        FileNotFoundError: If HDF5 or model file not found
        ValueError: If HDF5 has no units with eimage_sta data
    """
    hdf5_path = Path(hdf5_path)
    model_path = Path(model_path)

    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Starting AP tracking for {hdf5_path}")

    # Select device and load model
    device = select_device(force_cpu=force_cpu)
    model = load_cnn_model(model_path, device=device)

    # Process units
    with h5py.File(str(hdf5_path), "r+") as root:
        # Read DVNT from metadata (shared across all units)
        dvnt = read_dvnt_metadata(root)
        logger.info(f"DVNT position: DV={dvnt.dv_position}, NT={dvnt.nt_position}, LR={dvnt.lr_position}")

        # Get unit list
        if "units" not in root:
            raise ValueError(f"No units group in HDF5 file: {hdf5_path}")

        all_unit_ids = list(root["units"].keys())
        
        # Filter by cell type if enabled
        if filter_by_cell_type:
            unit_ids = []
            skipped_no_label = 0
            skipped_wrong_type = 0
            
            for uid in all_unit_ids:
                auto_label_path = f"units/{uid}/auto_label/axon_type"
                if auto_label_path not in root:
                    skipped_no_label += 1
                    continue
                
                cell_type = root[auto_label_path][()]
                if isinstance(cell_type, bytes):
                    cell_type = cell_type.decode('utf-8')
                
                if cell_type.lower() == cell_type_filter.lower():
                    unit_ids.append(uid)
                else:
                    skipped_wrong_type += 1
            
            logger.info(
                f"Cell type filter '{cell_type_filter}': {len(unit_ids)} units selected, "
                f"{skipped_wrong_type} wrong type, {skipped_no_label} no label"
            )
        else:
            unit_ids = all_unit_ids
        
        if max_units:
            unit_ids = unit_ids[:max_units]

        logger.info(f"Processing {len(unit_ids)} units")

        # Process each unit and collect results
        all_results = {}
        all_pathways = {}

        for unit_id in unit_ids:
            logger.info(f"Processing unit {unit_id}")

            result = process_single_unit(
                root, unit_id, model, device, dvnt,
                soma_std_threshold=soma_std_threshold,
                soma_temporal_range=soma_temporal_range,
                soma_refine_radius=soma_refine_radius,
                ais_search_xy_radius=ais_search_xy_radius,
                ais_search_t_radius=ais_search_t_radius,
                temporal_window_size=temporal_window_size,
                exclude_radius=exclude_radius,
                centroid_threshold=centroid_threshold,
                max_displacement=max_displacement,
                centroid_start_frame=centroid_start_frame,
                max_displacement_post=max_displacement_post,
                min_points_for_fit=min_points_for_fit,
            )

            all_results[unit_id] = result

            # Collect pathways and add direction info
            if result["ap_pathway"] is not None:
                ap_pathway = result["ap_pathway"]
                
                # Calculate direction from centroids
                if result["post_processed"]:
                    axon_centroids = result["post_processed"].get("axon_centroids")
                    direction_angle, start_point = calculate_direction_from_centroids(
                        axon_centroids
                    )
                    ap_pathway.direction_angle = direction_angle
                    ap_pathway.start_point = start_point
                
                all_pathways[unit_id] = ap_pathway

        # Calculate enhanced ONH intersection using clustering
        onh_result = None
        intersection = None
        used_method = None
        actual_r2_threshold = r2_threshold
        
        if len(all_pathways) >= 2:
            logger.info(f"Calculating ONH from {len(all_pathways)} pathways...")
            
            # Try enhanced algorithm with progressively lower R² thresholds
            current_r2 = r2_threshold
            while current_r2 >= 0.4 and onh_result is None:
                logger.info(f"Trying enhanced ONH detection with R²≥{current_r2:.1f}...")
                onh_result = calculate_enhanced_intersection(
                    all_pathways,
                    r2_threshold=current_r2,
                    direction_tolerance=direction_tolerance,
                    max_distance_from_center=max_distance_from_center,
                    center_point=center_point,
                    cluster_eps=cluster_eps,
                    cluster_min_samples=cluster_min_samples,
                )
                
                if onh_result:
                    actual_r2_threshold = current_r2
                    used_method = "enhanced"
                    logger.info(
                        f"ONH detected at ({onh_result.x:.2f}, {onh_result.y:.2f}), "
                        f"RMSE={onh_result.rmse:.2f}, R²≥{current_r2:.1f}, "
                        f"cluster={onh_result.n_cluster_points}/{onh_result.n_total_intersections} points"
                    )
                    # Create APIntersection for backward compatibility
                    intersection = APIntersection(
                        x=onh_result.x, y=onh_result.y, mse=onh_result.mse
                    )
                else:
                    logger.warning(f"Enhanced ONH failed with R²≥{current_r2:.1f}")
                    current_r2 -= 0.2
            
            # Fall back to legacy method if enhanced failed at all thresholds
            if onh_result is None:
                logger.warning("Enhanced ONH detection failed at all R² thresholds, trying legacy method")
                intersection = calculate_optimal_intersection(all_pathways)
                if intersection:
                    used_method = "legacy"
                    actual_r2_threshold = 0.0  # Legacy doesn't filter by R²
                    logger.info(
                        f"Legacy intersection at ({intersection.x:.2f}, {intersection.y:.2f})"
                    )
        else:
            logger.warning(
                f"Insufficient pathways ({len(all_pathways)}) for intersection calculation"
            )

        # Calculate angle correction once (shared by all units)
        angle_correction = None
        if intersection:
            angle_correction = _calculate_angle_correction(
                intersection,
                dv_position=dvnt.dv_position,
                nt_position=dvnt.nt_position,
            )
            if angle_correction is not None:
                logger.info(f"Angle correction: {angle_correction:.1f}°")

        # Write results to HDF5
        for unit_id, result in all_results.items():
            # Calculate polar coordinates if we have intersection and soma
            polar_coords = None
            if intersection and result["refined_soma"]:
                soma_xy = (result["refined_soma"].x, result["refined_soma"].y)
                polar_coords = calculate_soma_polar_coordinates(
                    soma_xy, intersection,
                    dv_position=dvnt.dv_position,
                    nt_position=dvnt.nt_position,
                    angle_correction=angle_correction,
                )

            write_ap_tracking_to_hdf5(
                root, unit_id,
                dvnt=dvnt,
                refined_soma=result["refined_soma"],
                ais=result["ais"],
                prediction_data=result["prediction"],
                post_processed=result["post_processed"],
                ap_pathway=result["ap_pathway"],
                intersection=intersection,
                polar_coords=polar_coords,
                onh_result=onh_result,
            )

    # Count successes
    complete_count = sum(1 for r in all_results.values() if r["status"] == "complete")
    logger.info(f"AP tracking complete: {complete_count}/{len(unit_ids)} units processed")

    # TODO: Session support (T029-T032) - for now return None
    if session is not None:
        logger.warning("Session-based processing not yet implemented, wrote directly to HDF5")

    return None


def compute_ap_tracking_batch(
    hdf5_paths: List[Path],
    model_path: Path,
    *,
    force_cpu: bool = False,
    skip_existing: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    **kwargs,
) -> Dict[str, str]:
    """
    Process multiple HDF5 files for AP tracking.

    Args:
        hdf5_paths: List of HDF5 file paths to process
        model_path: Path to trained CNN model
        force_cpu: Force CPU inference
        skip_existing: Skip files that already have ap_tracking features
        progress_callback: Optional callback(current, total, filename)
        **kwargs: Additional parameters passed to compute_ap_tracking

    Returns:
        Dictionary mapping file paths to status strings:
        - "complete": Successfully processed
        - "skipped": Already has ap_tracking
        - "error: {message}": Processing failed
    """
    results = {}
    total = len(hdf5_paths)

    # Import torch for GPU cleanup
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    for i, hdf5_path in enumerate(hdf5_paths):
        hdf5_path = Path(hdf5_path)
        filename = hdf5_path.name

        if progress_callback:
            progress_callback(i + 1, total, filename)

        # Check if already processed
        if skip_existing:
            try:
                with h5py.File(str(hdf5_path), "r") as f:
                    if "units" in f:
                        first_unit = next(iter(f["units"].keys()), None)
                        if first_unit and "ap_tracking" in f[f"units/{first_unit}/features"]:
                            logger.info(f"Skipping {filename}: already processed")
                            results[str(hdf5_path)] = "skipped"
                            continue
            except Exception:
                pass  # If we can't check, proceed with processing

        # Process file
        try:
            compute_ap_tracking(
                hdf5_path, model_path,
                force_cpu=force_cpu,
                **kwargs,
            )
            results[str(hdf5_path)] = "complete"
            logger.info(f"Completed {filename}")
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            results[str(hdf5_path)] = f"error: {e}"

        # GPU memory cleanup
        if has_torch and not force_cpu and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results

