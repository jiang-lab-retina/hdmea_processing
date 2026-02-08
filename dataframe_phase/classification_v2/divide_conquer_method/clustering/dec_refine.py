"""
DEC refinement training loop.

Implements the IDEC-style training with:
- Periodic target distribution updates
- Convergence detection based on assignment changes
- Optional reconstruction loss
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .. import config
from ..models.autoencoder import MultiSegmentAutoencoder
from ..models.dec import (
    IDEC, DECLayer, dec_loss, reconstruction_loss,
    cluster_balance_loss, iprgc_enrichment_loss, compute_iprgc_target_clusters,
)
from ..train import MultiSegmentDataset

logger = logging.getLogger(__name__)


def refine_with_dec(
    model: MultiSegmentAutoencoder,
    segments: dict[str, np.ndarray],
    initial_centers: np.ndarray,
    k: int,
    scaler_params: dict | None = None,
    max_iterations: int | None = None,
    update_interval: int | None = None,
    convergence_threshold: float | None = None,
    reconstruction_weight: float | None = None,
    balance_weight: float | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    checkpoint_dir: Path | None = None,
    iprgc_labels: np.ndarray | None = None,
    iprgc_enrichment_weight: float | None = None,
    n_iprgc_target_clusters: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Refine clusters using DEC/IDEC.
    
    Args:
        model: Pre-trained autoencoder.
        segments: Dict of segment arrays.
        initial_centers: (k, embedding_dim) GMM cluster centers (in standardized space).
        k: Number of clusters.
        scaler_params: Dict with 'mean' and 'std' for embedding standardization.
        max_iterations: Maximum DEC iterations.
        update_interval: Update target P every N iterations.
        convergence_threshold: Stop when assignment change < threshold.
        reconstruction_weight: Weight for reconstruction loss (gamma).
        balance_weight: Weight for cluster balance regularization (prevents collapse).
        lr: Learning rate.
        batch_size: Training batch size.
        device: "cuda" or "cpu".
        checkpoint_dir: Directory to save final model.
        iprgc_labels: (n_cells,) boolean array marking ipRGC cells.
            When provided, adds ipRGC enrichment loss to concentrate ipRGC cells
            into a few dedicated clusters.
        iprgc_enrichment_weight: Weight for ipRGC enrichment loss.
            Defaults to config.IPRGC_ENRICHMENT_WEIGHT.
        n_iprgc_target_clusters: Number of target ipRGC-enriched clusters.
            Defaults to config.IPRGC_N_TARGET_CLUSTERS.
    
    Returns:
        Tuple of (cluster_labels, refined_embeddings, training_history).
    """
    # Apply defaults
    max_iterations = max_iterations if max_iterations is not None else config.DEC_MAX_ITERATIONS
    update_interval = update_interval if update_interval is not None else config.DEC_UPDATE_INTERVAL
    convergence_threshold = convergence_threshold if convergence_threshold is not None else config.DEC_CONVERGENCE_THRESHOLD
    reconstruction_weight = reconstruction_weight if reconstruction_weight is not None else config.DEC_RECONSTRUCTION_WEIGHT
    balance_weight = balance_weight if balance_weight is not None else getattr(config, 'DEC_BALANCE_WEIGHT', 0.0)
    iprgc_enrichment_weight = iprgc_enrichment_weight if iprgc_enrichment_weight is not None else getattr(config, 'IPRGC_ENRICHMENT_WEIGHT', 0.0)
    n_iprgc_target_clusters = n_iprgc_target_clusters if n_iprgc_target_clusters is not None else getattr(config, 'IPRGC_N_TARGET_CLUSTERS', 6)
    lr = lr if lr is not None else config.DEC_LEARNING_RATE
    batch_size = batch_size if batch_size is not None else config.AE_BATCH_SIZE
    device = device if device is not None else config.DEVICE
    
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU for DEC")
        device = "cpu"
    
    # Create IDEC model with standardization parameters
    embedding_mean = scaler_params['mean'] if scaler_params else None
    embedding_std = scaler_params['std'] if scaler_params else None
    
    idec = IDEC(
        autoencoder=model,
        n_clusters=k,
        alpha=config.DEC_ALPHA,
        initial_centers=initial_centers,
        embedding_mean=embedding_mean,
        embedding_std=embedding_std,
    )
    idec = idec.to(device)
    
    # Prepare data
    n_cells = next(iter(segments.values())).shape[0]
    segment_tensors = {
        name: torch.tensor(arr, dtype=torch.float32)
        for name, arr in segments.items()
    }
    
    # Prepare ipRGC enrichment data
    use_iprgc_enrichment = (
        iprgc_labels is not None
        and iprgc_enrichment_weight > 0
        and np.any(iprgc_labels)
    )
    
    if use_iprgc_enrichment:
        n_iprgc = int(np.sum(iprgc_labels))
        logger.info(f"ipRGC enrichment enabled: {n_iprgc} ipRGC cells, "
                     f"weight={iprgc_enrichment_weight}, "
                     f"target_clusters={n_iprgc_target_clusters}")
        # Add ipRGC mask as pseudo-segment so it shuffles with the data
        loader_segments = {**segment_tensors}
        loader_segments['_iprgc_mask'] = torch.tensor(
            iprgc_labels.astype(np.float32), dtype=torch.float32
        ).unsqueeze(1)  # (n_cells, 1)
    else:
        loader_segments = segment_tensors
        if iprgc_labels is not None and not np.any(iprgc_labels):
            logger.info("ipRGC enrichment: no ipRGC cells found, skipping")
    
    dataset = MultiSegmentDataset(loader_segments)
    # shuffle=False: P-target is computed in sequential order and indexed by
    # p_idx.  Shuffling would misalign P with the actual batch samples,
    # corrupting the DEC self-training signal.  Standard DEC uses sequential
    # sampling because it refines an already-initialised model.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer for DEC layer and autoencoder
    optimizer = optim.Adam(idec.parameters(), lr=lr)
    
    # Training history
    history = {
        'dec_loss': [],
        'rec_loss': [],
        'total_loss': [],
        'assignment_change': [],
        'converged': False,
        'final_iteration': 0,
    }
    
    # Initial cluster assignments
    prev_labels = _get_all_labels(idec, segment_tensors, device, batch_size)
    
    logger.info(f"Starting DEC refinement: k={k}, max_iter={max_iterations}, "
                f"rec_weight={reconstruction_weight}, balance_weight={balance_weight}")
    
    # Check initial cluster distribution BEFORE computing target
    initial_labels = _get_all_labels(idec, segment_tensors, device, batch_size)
    unique, counts = np.unique(initial_labels, return_counts=True)
    all_clusters = sorted(zip(counts, unique), reverse=True)
    all_str = ", ".join([f"c{c}:{n}" for n, c in all_clusters])
    logger.info(f"Initial DEC cluster distribution (BEFORE training):")
    logger.info(f"  [{all_str}]")
    
    # Initial target distribution
    p_target = _compute_target_distribution(idec, segment_tensors, device, batch_size)
    logger.debug("Computed initial target distribution")
    
    # Precompute global ipRGC target clusters (stable within each phase)
    iprgc_target_indices = None
    if use_iprgc_enrichment:
        q_all = _compute_all_q(idec, segment_tensors, device, batch_size)
        iprgc_target_indices = compute_iprgc_target_clusters(
            q_all, iprgc_labels, n_iprgc_target_clusters
        )
        logger.info(f"Initial ipRGC target clusters: {iprgc_target_indices.tolist()}")
    
    # Two-phase DEC training when ipRGC enrichment is active:
    #   Phase 1 (first half):  Standard DEC builds functional clusters
    #   Phase 2 (second half): Reconstruction anchor is slashed (100x) so
    #                          the enrichment loss can freely reorganise cells.
    # For groups without enrichment, weights are constant throughout.
    phase2_start = max_iterations // 2
    if use_iprgc_enrichment:
        logger.info(f"Two-phase training: Phase 1 iter 0-{phase2_start-1}, "
                     f"Phase 2 iter {phase2_start}-{max_iterations-1} "
                     f"(rec_weight x0.01)")
    
    for iteration in range(max_iterations):
        # Update target distribution periodically (this sharpens clusters)
        if iteration > 0 and iteration % update_interval == 0:
            p_target = _compute_target_distribution(idec, segment_tensors, device, batch_size)
            # Also update global ipRGC target clusters
            if use_iprgc_enrichment:
                q_all = _compute_all_q(idec, segment_tensors, device, batch_size)
                iprgc_target_indices = compute_iprgc_target_clusters(
                    q_all, iprgc_labels, n_iprgc_target_clusters
                )
            logger.debug(f"Iteration {iteration}: Updated target distribution")
        
        # Phase-dependent weights
        in_phase2 = use_iprgc_enrichment and iteration >= phase2_start
        current_rec_weight = reconstruction_weight * 0.01 if in_phase2 else reconstruction_weight
        
        if iteration == phase2_start and use_iprgc_enrichment:
            logger.info(f"=== Phase 2 started at iteration {iteration} ===")
            logger.info(f"  rec_weight: {reconstruction_weight} -> {current_rec_weight}")
        
        # Training epoch
        epoch_dec_loss = []
        epoch_rec_loss = []
        epoch_bal_loss = []
        epoch_iprgc_loss = []
        
        idec.train()
        p_idx = 0  # Index into p_target (aligned because shuffle=False)
        
        for batch in loader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            
            # Extract ipRGC mask before model forward pass (not a real segment)
            iprgc_batch = batch.pop('_iprgc_mask', None)
            
            batch_size_actual = next(iter(batch.values())).shape[0]
            
            optimizer.zero_grad()
            
            # Forward pass
            output = idec(batch)
            q = output['q']
            
            # Get corresponding target distribution slice
            p_batch = p_target[p_idx:p_idx + batch_size_actual].to(device)
            p_idx = (p_idx + batch_size_actual) % n_cells
            
            # DEC loss
            loss_dec = dec_loss(q, p_batch)
            
            # Reconstruction loss
            loss_rec = reconstruction_loss(batch, output['reconstructions'])
            
            # Cluster balance loss (prevents collapse into single cluster)
            loss_bal = cluster_balance_loss(q) if balance_weight > 0 else torch.tensor(0.0)
            
            # ipRGC enrichment loss (Other group only, uses precomputed target indices)
            if use_iprgc_enrichment and iprgc_batch is not None and iprgc_target_indices is not None:
                loss_iprgc = iprgc_enrichment_loss(
                    q, iprgc_batch.squeeze(1).bool(),
                    iprgc_target_indices.to(device),
                )
            else:
                loss_iprgc = torch.tensor(0.0, device=q.device)
            
            # Total loss (uses phase-dependent rec weight)
            loss = (loss_dec
                    + current_rec_weight * loss_rec
                    + balance_weight * loss_bal
                    + iprgc_enrichment_weight * loss_iprgc)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_dec_loss.append(loss_dec.item())
            epoch_rec_loss.append(loss_rec.item())
            epoch_bal_loss.append(loss_bal.item())
            epoch_iprgc_loss.append(loss_iprgc.item())
        
        # Record epoch stats
        mean_dec_loss = np.mean(epoch_dec_loss)
        mean_rec_loss = np.mean(epoch_rec_loss)
        mean_bal_loss = np.mean(epoch_bal_loss)
        mean_iprgc_loss = np.mean(epoch_iprgc_loss)
        history['dec_loss'].append(mean_dec_loss)
        history['rec_loss'].append(mean_rec_loss)
        history['total_loss'].append(
            mean_dec_loss
            + current_rec_weight * mean_rec_loss
            + balance_weight * mean_bal_loss
            + iprgc_enrichment_weight * mean_iprgc_loss
        )
        
        # Check convergence
        curr_labels = _get_all_labels(idec, segment_tensors, device, batch_size)
        assignment_change = np.mean(curr_labels != prev_labels)
        history['assignment_change'].append(assignment_change)
        
        if (iteration + 1) % 10 == 0:
            # Log all loss components
            unique, counts = np.unique(curr_labels, return_counts=True)
            all_clusters = sorted(zip(counts, unique), reverse=True)
            all_str = ", ".join([f"c{c}:{n}" for n, c in all_clusters])
            logger.info(
                f"Iteration {iteration+1}/{max_iterations}: "
                f"dec={mean_dec_loss:.4f}, rec={mean_rec_loss:.1f}, "
                f"bal={mean_bal_loss:.4f}, iprgc={mean_iprgc_loss:.4f}, "
                f"change={assignment_change:.4f}"
            )
            logger.info(f"  Clusters: [{all_str}]")
            if use_iprgc_enrichment and iprgc_target_indices is not None:
                logger.info(f"  ipRGC targets: {iprgc_target_indices.tolist()}")
        
        # Only check convergence after minimum iterations
        min_iterations = getattr(config, 'DEC_MIN_ITERATIONS', 20)
        if iteration >= min_iterations and assignment_change < convergence_threshold:
            logger.info(f"DEC converged at iteration {iteration+1} "
                       f"(assignment_change={assignment_change:.4f} < {convergence_threshold})")
            history['converged'] = True
            history['final_iteration'] = iteration + 1
            break
        
        prev_labels = curr_labels
    
    if not history['converged']:
        history['final_iteration'] = max_iterations
        logger.warning(f"DEC did not converge after {max_iterations} iterations")
    
    # Get final labels and embeddings
    idec.eval()
    final_labels = _get_all_labels(idec, segment_tensors, device, batch_size)
    final_embeddings = _get_all_embeddings(idec, segment_tensors, device, batch_size)
    
    # Save checkpoint
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(idec.state_dict(), checkpoint_dir / "dec_refined.pt")
        logger.info(f"Saved DEC model to {checkpoint_dir / 'dec_refined.pt'}")
    
    return final_labels, final_embeddings, history


def _get_all_labels(
    model: IDEC,
    segments: dict[str, torch.Tensor],
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Get cluster labels for all samples."""
    model.eval()
    n_samples = next(iter(segments.values())).shape[0]
    all_labels = []
    
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = {
                name: tensor[start_idx:end_idx].to(device)
                for name, tensor in segments.items()
            }
            output = model(batch)
            labels = output['q'].argmax(dim=1).cpu().numpy()
            all_labels.append(labels)
    
    return np.concatenate(all_labels)


def _get_all_embeddings(
    model: IDEC,
    segments: dict[str, torch.Tensor],
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Get embeddings for all samples."""
    model.eval()
    n_samples = next(iter(segments.values())).shape[0]
    all_embeddings = []
    
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = {
                name: tensor[start_idx:end_idx].to(device)
                for name, tensor in segments.items()
            }
            output = model(batch)
            embeddings = output['embedding'].cpu().numpy()
            all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings)


def _compute_target_distribution(
    model: IDEC,
    segments: dict[str, torch.Tensor],
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """Compute target distribution P for all samples."""
    model.eval()
    n_samples = next(iter(segments.values())).shape[0]
    all_q = []
    
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = {
                name: tensor[start_idx:end_idx].to(device)
                for name, tensor in segments.items()
            }
            output = model(batch)
            all_q.append(output['q'].cpu())
    
    q = torch.cat(all_q, dim=0)
    p = DECLayer.target_distribution(q)
    
    return p


def _compute_all_q(
    model: IDEC,
    segments: dict[str, torch.Tensor],
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """Compute soft assignments Q for all samples (used for global ipRGC target selection)."""
    model.eval()
    n_samples = next(iter(segments.values())).shape[0]
    all_q = []
    
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = {
                name: tensor[start_idx:end_idx].to(device)
                for name, tensor in segments.items()
            }
            output = model(batch)
            all_q.append(output['q'].cpu())
    
    return torch.cat(all_q, dim=0)


def load_dec_results(
    checkpoint_path: Path,
    model: MultiSegmentAutoencoder,
    k: int,
    segments: dict[str, np.ndarray],
    device: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load cached DEC results.
    
    Args:
        checkpoint_path: Path to dec_refined.pt.
        model: Autoencoder model.
        k: Number of clusters.
        segments: Segment data.
        device: Device to load to.
    
    Returns:
        Tuple of (cluster_labels, embeddings).
    """
    device = device if device is not None else config.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    # Create IDEC model
    idec = IDEC(
        autoencoder=model,
        n_clusters=k,
        alpha=config.DEC_ALPHA,
    )
    
    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    idec.load_state_dict(state_dict)
    idec = idec.to(device)
    idec.eval()
    
    # Convert segments
    segment_tensors = {
        name: torch.tensor(arr, dtype=torch.float32)
        for name, arr in segments.items()
    }
    
    # Get labels and embeddings
    labels = _get_all_labels(idec, segment_tensors, device, config.AE_BATCH_SIZE)
    embeddings = _get_all_embeddings(idec, segment_tensors, device, config.AE_BATCH_SIZE)
    
    logger.info(f"Loaded DEC results from {checkpoint_path}")
    
    return labels, embeddings
