"""
Step Wrapper: Compute AP Tracking (Deferred Mode)

Runs action potential tracking analysis on RGC units.
Works entirely in session/deferred mode - no HDF5 file required.
Faithfully replicates the algorithm from hdmea.features.ap_tracking.core.compute_ap_tracking
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

from hdmea.pipeline import PipelineSession
from Projects.unified_pipeline.config import (
    AP_TRACKING_MODEL_PATH,
    APTrackingConfig,
    red_warning,
)

logger = logging.getLogger(__name__)

STEP_NAME = "compute_ap_tracking"


# =============================================================================
# Bad Lanes Preprocessing (session-based)
# =============================================================================

def parse_bad_lanes(bad_lanes_str: str) -> List[int]:
    """Parse bad lanes string to list of 0-indexed column indices."""
    if not bad_lanes_str or bad_lanes_str.strip() == "":
        return []
    
    lanes = []
    for part in bad_lanes_str.split(","):
        part = part.strip()
        if part:
            try:
                lane_1indexed = int(part)
                lane_0indexed = lane_1indexed - 1
                lanes.append(lane_0indexed)
            except ValueError:
                pass
    
    return sorted(lanes)


def read_bad_lanes_from_session(session: PipelineSession) -> List[int]:
    """Read Bad_lanes from session metadata/gsheet_row/Bad_lanes."""
    gsheet_row = session.metadata.get('gsheet_row', {})
    bad_lanes_data = gsheet_row.get('Bad_lanes', '')
    
    if isinstance(bad_lanes_data, bytes):
        bad_lanes_str = bad_lanes_data.decode("utf-8")
    elif isinstance(bad_lanes_data, np.ndarray):
        if bad_lanes_data.size > 0:
            val = bad_lanes_data.flat[0]
            if isinstance(val, bytes):
                bad_lanes_str = val.decode("utf-8")
            else:
                bad_lanes_str = str(val)
        else:
            bad_lanes_str = ""
    else:
        bad_lanes_str = str(bad_lanes_data)
    
    return parse_bad_lanes(bad_lanes_str)


def fix_bad_lanes_in_sta(sta_data: np.ndarray, bad_lanes: List[int]) -> np.ndarray:
    """Replace bad lane columns with the mean of the entire STA data."""
    if len(bad_lanes) == 0:
        return sta_data
    
    result = sta_data.copy()
    sta_mean = np.mean(sta_data)
    n_cols = sta_data.shape[-1]
    
    for lane_idx in bad_lanes:
        if 0 <= lane_idx < n_cols:
            result[:, :, lane_idx] = sta_mean
    
    return result


def preprocess_bad_lanes_session(session: PipelineSession) -> int:
    """
    Preprocess all STA data in the session to fix bad lanes.
    
    Returns:
        Number of units processed
    """
    bad_lanes = read_bad_lanes_from_session(session)
    
    if not bad_lanes:
        return 0
    
    processed = 0
    for unit_id in session.units:
        features = session.units[unit_id].get('features', {})
        eimage_sta = features.get('eimage_sta', {})
        sta_data = eimage_sta.get('data') if isinstance(eimage_sta, dict) else None
        
        if sta_data is None or not isinstance(sta_data, np.ndarray):
            continue
        
        fixed_data = fix_bad_lanes_in_sta(sta_data, bad_lanes)
        session.units[unit_id]['features']['eimage_sta']['data'] = fixed_data
        processed += 1
    
    return processed


# =============================================================================
# Session-based AP Tracking (faithfully follows core.py algorithm)
# =============================================================================

def compute_ap_tracking_step(
    session: PipelineSession,
    model_path: Optional[Path] = None,
    config: Optional[APTrackingConfig] = None,
) -> PipelineSession:
    """
    Compute AP tracking features for all units in session.
    
    Faithfully replicates the algorithm from:
    hdmea.features.ap_tracking.core.compute_ap_tracking
    
    Args:
        session: PipelineSession with eimage_sta computed
        model_path: Path to CNN model (default from config)
        config: APTrackingConfig with processing parameters
        
    Returns:
        Updated session with ap_tracking features
    """
    if config is None:
        config = APTrackingConfig()
    
    if model_path is None:
        model_path = config.model_path
    
    model_path = Path(model_path)
    fix_bad_lanes = config.fix_bad_lanes
    
    logger.info("Step 10: Computing AP tracking (deferred)...")
    
    try:
        # Import all required modules (same as core.py)
        from hdmea.features.ap_tracking.model_inference import (
            load_cnn_model, run_model_inference, select_device
        )
        from hdmea.features.ap_tracking.ais_refiner import ais_refiner, soma_refiner
        from hdmea.features.ap_tracking.soma_detector import find_soma_from_3d_sta
        from hdmea.features.ap_tracking.postprocess import process_predictions
        from hdmea.features.ap_tracking.pathway_analysis import (
            fit_line_to_centroids, 
            calculate_soma_polar_coordinates,
            calculate_direction_from_centroids,
            calculate_enhanced_intersection,
            calculate_optimal_intersection,
            _calculate_angle_correction,
            APIntersection,
            DEFAULT_CLUSTER_EPS,
            DEFAULT_CLUSTER_MIN_SAMPLES,
        )
        from hdmea.features.ap_tracking.dvnt_parser import parse_dvnt_from_center_xy
        from tqdm import tqdm
        
        # Preprocess: fix bad lanes in STA data
        if fix_bad_lanes:
            n_fixed = preprocess_bad_lanes_session(session)
            if n_fixed > 0:
                logger.info(f"  Fixed bad lanes in {n_fixed} units")
        
        # Filter units by cell type (same as core.py)
        all_unit_ids = list(session.units.keys())
        
        if config.filter_by_cell_type:
            unit_ids = []
            skipped_no_label = 0
            skipped_wrong_type = 0
            
            for uid in all_unit_ids:
                unit_data = session.units.get(uid, {})
                auto_label = unit_data.get('auto_label', {})
                cell_type = auto_label.get('axon_type')
                
                if cell_type is None:
                    skipped_no_label += 1
                    continue
                
                # Handle different storage formats (same as core.py lines 714-726)
                if isinstance(cell_type, bytes):
                    cell_type = cell_type.decode('utf-8')
                elif isinstance(cell_type, np.ndarray):
                    if cell_type.size > 0:
                        val = cell_type.flat[0]
                        if isinstance(val, bytes):
                            cell_type = val.decode('utf-8')
                        else:
                            cell_type = str(val)
                    else:
                        skipped_no_label += 1
                        continue
                
                if str(cell_type).lower() == config.cell_type_filter.lower():
                    unit_ids.append(uid)
                else:
                    skipped_wrong_type += 1
            
            logger.info(
                f"  Cell type filter '{config.cell_type_filter}': {len(unit_ids)} units selected, "
                f"{skipped_wrong_type} wrong type, {skipped_no_label} no label"
            )
        else:
            unit_ids = all_unit_ids
        
        if not unit_ids:
            logger.warning("  No units to process")
            session.mark_step_complete(STEP_NAME)
            return session
        
        # Select device and load model (same as core.py lines 685-686)
        logger.info(f"  Loading AP tracking model...")
        device = select_device(force_cpu=config.force_cpu)
        model = load_cnn_model(model_path, device=device)
        
        # Read DVNT from metadata (same as core.py line 691)
        gsheet_row = session.metadata.get('gsheet_row', {})
        center_xy_str = gsheet_row.get('Center_XY') or gsheet_row.get('Center_xy')
        if isinstance(center_xy_str, np.ndarray) and center_xy_str.size > 0:
            center_xy_str = str(center_xy_str.flat[0])
        elif isinstance(center_xy_str, bytes):
            center_xy_str = center_xy_str.decode('utf-8')
        dvnt = parse_dvnt_from_center_xy(str(center_xy_str) if center_xy_str else None)
        logger.info(f"  DVNT position: DV={dvnt.dv_position}, NT={dvnt.nt_position}, LR={dvnt.lr_position}")
        
        # Process each unit and collect results (same as core.py lines 746-784)
        all_results = {}
        all_pathways = {}
        all_centroids = {}
        
        logger.info(f"  Processing {len(unit_ids)} units...")
        
        for unit_id in tqdm(unit_ids, desc="AP tracking"):
            result = _process_single_unit_session(
                session, unit_id, model, device, dvnt,
                # Soma/AIS parameters (same defaults as core.py)
                soma_std_threshold=3.0,
                soma_temporal_range=(5, 27),
                soma_refine_radius=5,
                ais_search_xy_radius=5,
                ais_search_t_radius=5,
                # Post-processing parameters
                temporal_window_size=5,
                exclude_radius=5,
                centroid_threshold=0.05,
                max_displacement=config.max_displacement,
                centroid_start_frame=config.centroid_start_frame,
                max_displacement_post=config.max_displacement_post,
                min_points_for_fit=config.min_points_for_fit,
            )
            
            all_results[unit_id] = result
            
            # Collect pathways and add direction info (same as core.py lines 771-784)
            if result["ap_pathway"] is not None:
                ap_pathway = result["ap_pathway"]
                
                # Calculate direction from centroids
                if result["post_processed"]:
                    axon_centroids = result["post_processed"].get("axon_centroids")
                    if axon_centroids is not None and len(axon_centroids) > 0:
                        direction_angle, start_point = calculate_direction_from_centroids(axon_centroids)
                        ap_pathway.direction_angle = direction_angle
                        ap_pathway.start_point = start_point
                
                all_pathways[unit_id] = ap_pathway
            
            # Collect centroids (same as core.py lines 786-792)
            if result["post_processed"] and "axon_centroids" in result["post_processed"]:
                centroids = result["post_processed"]["axon_centroids"]
                if centroids is not None and len(centroids) > 0:
                    all_centroids[unit_id] = centroids
        
        # Calculate enhanced ONH intersection (same as core.py lines 794-856)
        onh_result = None
        intersection = None
        used_method = None
        actual_r2_threshold = config.r2_threshold
        
        if len(all_pathways) >= 2:
            logger.info(f"  Calculating ONH from {len(all_pathways)} pathways, {len(all_centroids)} cells with centroids...")
            
            # Try enhanced algorithm with progressively lower R² thresholds
            current_r2 = config.r2_threshold
            while current_r2 >= 0.4 and onh_result is None:
                logger.debug(f"  Trying ONH detection with R²≥{current_r2:.1f}...")
                try:
                    onh_result = calculate_enhanced_intersection(
                        all_pathways,
                        all_centroids=all_centroids,
                        r2_threshold=current_r2,
                        direction_tolerance=config.direction_tolerance,
                        max_distance_from_center=config.max_distance_from_center,
                        center_point=config.center_point,
                        cluster_eps=DEFAULT_CLUSTER_EPS,
                        cluster_min_samples=DEFAULT_CLUSTER_MIN_SAMPLES,
                        centroid_exclude_fraction=config.centroid_exclude_fraction,
                        min_remaining_fraction=config.min_remaining_fraction,
                    )
                except Exception as e:
                    logger.debug(f"  Enhanced ONH failed: {e}")
                    onh_result = None
                
                if onh_result:
                    actual_r2_threshold = current_r2
                    used_method = onh_result.method
                    logger.info(
                        f"  ONH detected at ({onh_result.x:.2f}, {onh_result.y:.2f}), "
                        f"RMSE={onh_result.rmse:.2f}, R²≥{current_r2:.1f}"
                    )
                    intersection = APIntersection(
                        x=onh_result.x, y=onh_result.y, mse=onh_result.mse
                    )
                else:
                    current_r2 -= 0.2
            
            # Fall back to legacy method if enhanced failed (same as core.py lines 844-852)
            if onh_result is None:
                logger.warning("  Enhanced ONH detection failed, trying legacy method")
                intersection = calculate_optimal_intersection(all_pathways)
                if intersection:
                    used_method = "legacy"
                    actual_r2_threshold = 0.0
                    logger.info(f"  Legacy intersection at ({intersection.x:.2f}, {intersection.y:.2f})")
        else:
            logger.warning(f"  Insufficient pathways ({len(all_pathways)}) for intersection calculation")
        
        # Calculate angle correction (same as core.py lines 858-867)
        angle_correction = None
        if intersection:
            angle_correction = _calculate_angle_correction(
                intersection,
                dv_position=dvnt.dv_position,
                nt_position=dvnt.nt_position,
            )
            if angle_correction is not None:
                logger.info(f"  Angle correction: {angle_correction:.1f}°")
        
        # Write results to session (same as core.py lines 869-890)
        ap_count = 0
        for unit_id, result in all_results.items():
            if result["status"] != "complete":
                continue
            
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
            
            # Store in session
            if 'features' not in session.units[unit_id]:
                session.units[unit_id]['features'] = {}
            
            ap_tracking_data = {
                'refined_soma': {
                    't': result["refined_soma"].t if result["refined_soma"] else None,
                    'x': result["refined_soma"].x if result["refined_soma"] else None,
                    'y': result["refined_soma"].y if result["refined_soma"] else None,
                },
                'axon_initial_segment': {
                    't': result["ais"].t if result["ais"] else None,
                    'x': result["ais"].x if result["ais"] else None,
                    'y': result["ais"].y if result["ais"] else None,
                },
            }
            
            # Write prediction_sta_data at root level (matches reference core.py line 230)
            if result["prediction"] is not None:
                ap_tracking_data['prediction_sta_data'] = result["prediction"].astype(np.float32)
            
            # Write post_processed_data group (matches reference core.py lines 233-244)
            post_processed = result["post_processed"]
            if post_processed:
                pp_data = {}
                if "filtered_prediction" in post_processed and post_processed["filtered_prediction"] is not None:
                    pp_data['filtered_prediction'] = post_processed["filtered_prediction"].astype(np.float32)
                if "axon_centroids" in post_processed and post_processed["axon_centroids"] is not None:
                    centroids = post_processed["axon_centroids"]
                    if isinstance(centroids, (list, np.ndarray)) and len(centroids) > 0:
                        pp_data['axon_centroids'] = np.array(centroids).astype(np.float32)
                if pp_data:
                    ap_tracking_data['post_processed_data'] = pp_data
            
            if polar_coords:
                ap_tracking_data['soma_polar_coordinates'] = {
                    'radius': polar_coords.radius,
                    'angle': polar_coords.angle,
                    'cartesian_x': polar_coords.cartesian_x,
                    'cartesian_y': polar_coords.cartesian_y,
                    'quadrant': polar_coords.quadrant,
                    'anatomical_quadrant': polar_coords.anatomical_quadrant,
                    'theta_deg': polar_coords.theta_deg,
                    'theta_deg_raw': polar_coords.theta_deg_raw,
                    'theta_deg_corrected': polar_coords.theta_deg_corrected,
                    'transformed_x': polar_coords.transformed_x,
                    'transformed_y': polar_coords.transformed_y,
                    'original_x': polar_coords.original_x,
                    'original_y': polar_coords.original_y,
                    'angle_correction_applied': polar_coords.angle_correction_applied,
                }
            
            # Always write ap_pathway group (reference writes None values if no pathway)
            # Reference only writes: slope, intercept, r_value, p_value, std_err (core.py lines 248-256)
            ap_pathway = result["ap_pathway"]
            if ap_pathway:
                ap_tracking_data['ap_pathway'] = {
                    'slope': ap_pathway.slope,
                    'intercept': ap_pathway.intercept,
                    'r_value': ap_pathway.r_value,
                    'p_value': ap_pathway.p_value,
                    'std_err': ap_pathway.std_err,
                }
            else:
                # Write NaN values for all fields (same as reference core.py lines 146-148)
                ap_tracking_data['ap_pathway'] = {
                    'slope': np.nan,
                    'intercept': np.nan,
                    'r_value': np.nan,
                    'p_value': np.nan,
                    'std_err': np.nan,
                }
            
            session.units[unit_id]['features']['ap_tracking'] = ap_tracking_data
            ap_count += 1
        
        # Process non-RGC cells for soma/polar coordinates (same as core.py lines 901-945)
        non_rgc_count = 0
        if config.filter_by_cell_type and intersection:
            non_rgc_unit_ids = [uid for uid in all_unit_ids if uid not in unit_ids]
            
            if non_rgc_unit_ids:
                logger.info(f"  Processing {len(non_rgc_unit_ids)} non-RGC cells for soma/polar coordinates...")
                
                for unit_id in non_rgc_unit_ids:
                    result = _process_single_unit_minimal_session(
                        session, unit_id,
                        soma_std_threshold=3.0,
                        soma_temporal_range=(5, 27),
                        soma_refine_radius=5,
                        ais_search_xy_radius=5,
                        ais_search_t_radius=5,
                    )
                    
                    if result["status"] != "complete":
                        continue
                    
                    # Calculate polar coordinates if we have soma
                    polar_coords = None
                    if result["refined_soma"]:
                        soma_xy = (result["refined_soma"].x, result["refined_soma"].y)
                        polar_coords = calculate_soma_polar_coordinates(
                            soma_xy, intersection,
                            dv_position=dvnt.dv_position,
                            nt_position=dvnt.nt_position,
                            angle_correction=angle_correction,
                        )
                    
                    # Store in session (minimal data)
                    if result["refined_soma"]:
                        if 'features' not in session.units[unit_id]:
                            session.units[unit_id]['features'] = {}
                        
                        ap_tracking_data = {
                            'refined_soma': {
                                't': result["refined_soma"].t,
                                'x': result["refined_soma"].x,
                                'y': result["refined_soma"].y,
                            },
                            'axon_initial_segment': {
                                't': result["ais"].t if result["ais"] else None,
                                'x': result["ais"].x if result["ais"] else None,
                                'y': result["ais"].y if result["ais"] else None,
                            },
                        }
                        
                        if polar_coords:
                            ap_tracking_data['soma_polar_coordinates'] = {
                                'radius': polar_coords.radius,
                                'angle': polar_coords.angle,
                                'cartesian_x': polar_coords.cartesian_x,
                                'cartesian_y': polar_coords.cartesian_y,
                                'quadrant': polar_coords.quadrant,
                                'anatomical_quadrant': polar_coords.anatomical_quadrant,
                                'theta_deg': polar_coords.theta_deg,
                                'theta_deg_raw': polar_coords.theta_deg_raw,
                                'theta_deg_corrected': polar_coords.theta_deg_corrected,
                                'transformed_x': polar_coords.transformed_x,
                                'transformed_y': polar_coords.transformed_y,
                                'original_x': polar_coords.original_x,
                                'original_y': polar_coords.original_y,
                                'angle_correction_applied': polar_coords.angle_correction_applied,
                            }
                        
                        # Always write ap_pathway group with NaN values (matches reference core.py lines 146-148)
                        ap_tracking_data['ap_pathway'] = {
                            'slope': np.nan,
                            'intercept': np.nan,
                            'r_value': np.nan,
                            'p_value': np.nan,
                            'std_err': np.nan,
                        }
                        
                        session.units[unit_id]['features']['ap_tracking'] = ap_tracking_data
                        non_rgc_count += 1
        
        # Store metadata about AP tracking (same as core.py write_ap_tracking_metadata_to_hdf5)
        ap_tracking_meta = {
            'model_path': str(model_path),
            'units_processed': ap_count,
            'non_rgc_processed': non_rgc_count,
            'filter_by_cell_type': config.filter_by_cell_type,
            'cell_type_filter': config.cell_type_filter,
            'dvnt_dv': dvnt.dv_position,
            'dvnt_nt': dvnt.nt_position,
            'dvnt_lr': dvnt.lr_position,
        }
        
        if intersection:
            ap_tracking_meta['all_ap_intersection'] = {
                'x': intersection.x,
                'y': intersection.y,
                'method': used_method or 'unknown',
                'rmse': onh_result.rmse if onh_result else 0.0,
                'n_cells_used': onh_result.n_cells_used if onh_result and hasattr(onh_result, 'n_cells_used') else len(all_pathways),
            }
        
        session.metadata['ap_tracking'] = ap_tracking_meta
        
        logger.info(f"  AP tracking complete: {ap_count} RGCs + {non_rgc_count} non-RGCs processed")
        session.mark_step_complete(STEP_NAME)
        
    except ImportError as e:
        logger.warning(red_warning(f"  Cannot import AP tracking: {e}"))
        session.warnings.append(f"{STEP_NAME}: Import error - {e}")
        session.completed_steps.add(f"{STEP_NAME}:skipped")
    
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(red_warning(f"  Error in AP tracking: {e}"))
        logger.error(f"  Traceback:\n{tb}")
        session.warnings.append(f"{STEP_NAME}: {e}")
        session.completed_steps.add(f"{STEP_NAME}:failed")
    
    return session


def _process_single_unit_session(
    session: PipelineSession,
    unit_id: str,
    model: Any,
    device: str,
    dvnt: Any,
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
    Process a single unit for AP tracking (session-based).
    
    Faithfully replicates core.py process_single_unit but reads from session.
    """
    from hdmea.features.ap_tracking.model_inference import run_model_inference
    from hdmea.features.ap_tracking.ais_refiner import ais_refiner, soma_refiner
    from hdmea.features.ap_tracking.soma_detector import find_soma_from_3d_sta
    from hdmea.features.ap_tracking.postprocess import process_predictions
    from hdmea.features.ap_tracking.pathway_analysis import fit_line_to_centroids
    
    result = {
        "unit_id": unit_id,
        "status": "skipped",
        "refined_soma": None,
        "ais": None,
        "prediction": None,
        "post_processed": None,
        "ap_pathway": None,
    }
    
    # Read STA data from session
    unit_data = session.units.get(unit_id, {})
    features = unit_data.get('features', {})
    eimage_sta = features.get('eimage_sta', {})
    sta_data = eimage_sta.get('data') if isinstance(eimage_sta, dict) else None
    
    if sta_data is None or not isinstance(sta_data, np.ndarray):
        result["skip_reason"] = "no_eimage_sta"
        return result
    
    # Detect soma
    try:
        soma_row, soma_col = find_soma_from_3d_sta(
            sta_data,
            std_threshold=soma_std_threshold,
            sta_temporal_range=soma_temporal_range,
        )
    except Exception as e:
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


def _process_single_unit_minimal_session(
    session: PipelineSession,
    unit_id: str,
    # Soma/AIS parameters
    soma_std_threshold: float = 3.0,
    soma_temporal_range: Tuple[int, int] = (5, 27),
    soma_refine_radius: int = 5,
    ais_search_xy_radius: int = 5,
    ais_search_t_radius: int = 5,
) -> Dict[str, Any]:
    """
    Process a single unit for soma/AIS only (no model inference).
    
    Faithfully replicates core.py process_single_unit_minimal but reads from session.
    """
    from hdmea.features.ap_tracking.ais_refiner import ais_refiner, soma_refiner
    from hdmea.features.ap_tracking.soma_detector import find_soma_from_3d_sta
    
    result = {
        "unit_id": unit_id,
        "status": "skipped",
        "refined_soma": None,
        "ais": None,
    }
    
    # Read STA data from session
    unit_data = session.units.get(unit_id, {})
    features = unit_data.get('features', {})
    eimage_sta = features.get('eimage_sta', {})
    sta_data = eimage_sta.get('data') if isinstance(eimage_sta, dict) else None
    
    if sta_data is None or not isinstance(sta_data, np.ndarray):
        result["skip_reason"] = "no_eimage_sta"
        return result
    
    # Detect soma
    try:
        soma_row, soma_col = find_soma_from_3d_sta(
            sta_data,
            std_threshold=soma_std_threshold,
            sta_temporal_range=soma_temporal_range,
        )
    except Exception as e:
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
    
    result["status"] = "complete"
    return result
