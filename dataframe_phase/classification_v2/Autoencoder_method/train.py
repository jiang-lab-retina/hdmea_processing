"""
Autoencoder training loop with checkpointing.

Handles:
- Dataset creation from preprocessed segments
- Training loop with early stopping
- Model checkpointing
- Training history logging
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from . import config
from .models.autoencoder import MultiSegmentAutoencoder
from .models.losses import CombinedAELoss, CombinedAELossWithPurity
from .preprocessing import get_segment_lengths

logger = logging.getLogger(__name__)


def train_autoencoder(
    segments: dict[str, np.ndarray],
    group_labels: np.ndarray,
    purity_labels: np.ndarray | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    beta: float | None = None,
    alpha: float | None = None,
    temperature: float | None = None,
    device: str | None = None,
    checkpoint_dir: Path | None = None,
    excluded_labels: list[int] | None = None,
    hidden_dims: list[int] | None = None,
    dropout: float | None = None,
    use_purity_loss: bool | None = None,
    purity_n_clusters: int | None = None,
    purity_temperature: float | None = None,
) -> Tuple[MultiSegmentAutoencoder, dict]:
    """
    Train the multi-segment autoencoder.
    
    Args:
        segments: Preprocessed segment arrays per cell. Dict mapping segment_name
            to (n_cells, segment_length) arrays.
        group_labels: Coarse group label per cell (integer encoded).
        purity_labels: Binary labels for purity loss (n_cells, n_labels). 
            If None and use_purity_loss=True, purity loss is skipped.
        epochs: Number of training epochs. Defaults to config.AE_EPOCHS.
        batch_size: Training batch size. Defaults to config.AE_BATCH_SIZE.
        lr: Learning rate. Defaults to config.AE_LEARNING_RATE.
        weight_decay: L2 regularization. Defaults to config.AE_WEIGHT_DECAY.
        beta: Supervised contrastive loss weight. Defaults to config.SUPCON_WEIGHT.
        alpha: Purity loss weight. Defaults to config.PURITY_LOSS_WEIGHT.
        temperature: SupCon temperature. Defaults to config.SUPCON_TEMPERATURE.
        device: "cuda" or "cpu". Defaults to config.DEVICE.
        checkpoint_dir: Directory to save checkpoints. Defaults to config.MODELS_DIR.
        excluded_labels: Labels to exclude from supervision (for CV turns).
        hidden_dims: Hidden layer dimensions. Defaults to config.AE_HIDDEN_DIMS.
        dropout: Dropout probability. Defaults to config.AE_DROPOUT.
        use_purity_loss: Whether to use purity loss. Defaults to config.USE_PURITY_LOSS.
        purity_n_clusters: Number of clusters for purity. Defaults to config.PURITY_N_CLUSTERS.
        purity_temperature: Temperature for purity clustering. Defaults to config.PURITY_TEMPERATURE.
    
    Returns:
        (trained_model, training_history)
        history contains: loss, rec_loss, sup_loss, purity_loss per epoch
    
    Notes:
        - Saves best model based on total loss
        - Uses early stopping if loss plateaus
    """
    # Apply defaults
    epochs = epochs if epochs is not None else config.AE_EPOCHS
    batch_size = batch_size if batch_size is not None else config.AE_BATCH_SIZE
    lr = lr if lr is not None else config.AE_LEARNING_RATE
    weight_decay = weight_decay if weight_decay is not None else config.AE_WEIGHT_DECAY
    beta = beta if beta is not None else config.SUPCON_WEIGHT
    alpha = alpha if alpha is not None else getattr(config, 'PURITY_LOSS_WEIGHT', 0.0)
    temperature = temperature if temperature is not None else config.SUPCON_TEMPERATURE
    device = device if device is not None else config.DEVICE
    checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else config.MODELS_DIR
    hidden_dims = hidden_dims if hidden_dims is not None else config.AE_HIDDEN_DIMS
    dropout = dropout if dropout is not None else config.AE_DROPOUT
    use_purity_loss = use_purity_loss if use_purity_loss is not None else getattr(config, 'USE_PURITY_LOSS', False)
    purity_n_clusters = purity_n_clusters if purity_n_clusters is not None else getattr(config, 'PURITY_N_CLUSTERS', 100)
    purity_temperature = purity_temperature if purity_temperature is not None else getattr(config, 'PURITY_TEMPERATURE', 1.0)
    
    # Disable purity loss if no purity labels provided
    if use_purity_loss and purity_labels is None:
        logger.warning("use_purity_loss=True but purity_labels not provided, disabling purity loss")
        use_purity_loss = False
    
    # Check CUDA availability
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    logger.info(f"Training autoencoder on {device}")
    logger.info(f"  epochs={epochs}, batch_size={batch_size}, lr={lr}")
    logger.info(f"  beta(supcon)={beta}, alpha(purity)={alpha}, use_purity={use_purity_loss}")
    
    # Get segment lengths and create model
    segment_lengths = get_segment_lengths(segments)
    model = MultiSegmentAutoencoder.from_segment_lengths(
        segment_lengths=segment_lengths,
        segment_latent_dims=config.SEGMENT_LATENT_DIMS,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    model = model.to(device)
    
    logger.info(f"Model created: {model.total_latent_dim}D latent space")
    
    # Create loss function
    if use_purity_loss:
        loss_fn = CombinedAELossWithPurity(
            segment_lengths=segment_lengths,
            beta=beta,
            alpha=alpha,
            temperature_supcon=temperature,
            n_clusters=purity_n_clusters,
            embedding_dim=model.total_latent_dim,
            temperature_purity=purity_temperature,
        )
        loss_fn = loss_fn.to(device)
        logger.info(f"  Using purity loss with {purity_n_clusters} clusters")
    else:
        loss_fn = CombinedAELoss(
            segment_lengths=segment_lengths,
            beta=beta,
            temperature=temperature,
        )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Create dataset - normalize each segment to zero mean, unit variance
    n_cells = len(group_labels)
    segment_tensors = {}
    for name, arr in segments.items():
        # Normalize: (x - mean) / (std + eps) per segment
        arr = arr.astype(np.float32)
        mean = np.mean(arr)
        std = np.std(arr) + 1e-8
        arr_normalized = (arr - mean) / std
        segment_tensors[name] = torch.tensor(arr_normalized, dtype=torch.float32)
        logger.debug(f"  Normalized {name}: mean={mean:.2f}, std={std:.2f}")
    
    label_tensor = torch.tensor(group_labels, dtype=torch.long)
    
    # Purity labels tensor (if using purity loss)
    if use_purity_loss and purity_labels is not None:
        purity_tensor = torch.tensor(purity_labels, dtype=torch.float32)
    else:
        purity_tensor = None
    
    # Create indices for DataLoader
    indices = torch.arange(n_cells)
    dataset = TensorDataset(indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training history
    history = {
        'loss': [],
        'rec_loss': [],
        'sup_loss': [],
        'purity_loss': [],
        'lr': [],
    }
    
    # Early stopping
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_losses = {'total': 0.0, 'rec': 0.0, 'sup': 0.0, 'purity': 0.0}
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for (batch_indices,) in pbar:
            # Get batch data
            batch_segments = {
                name: tensor[batch_indices].to(device)
                for name, tensor in segment_tensors.items()
            }
            batch_labels = label_tensor[batch_indices].to(device)
            
            # Forward pass
            output = model(batch_segments)
            
            # Compute loss
            if use_purity_loss and purity_tensor is not None:
                batch_purity_labels = purity_tensor[batch_indices].to(device)
                losses = loss_fn(
                    originals=batch_segments,
                    reconstructions=output['reconstructions'],
                    embeddings=output['full_embedding'],
                    group_labels=batch_labels,
                    purity_labels=batch_purity_labels,
                    excluded_labels=excluded_labels,
                )
            else:
                losses = loss_fn(
                    originals=batch_segments,
                    reconstructions=output['reconstructions'],
                    embeddings=output['full_embedding'],
                    labels=batch_labels,
                    excluded_labels=excluded_labels,
                )
                losses['purity'] = torch.tensor(0.0)
            
            # Backward pass
            optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            epoch_losses['total'] += losses['total'].item()
            epoch_losses['rec'] += losses['reconstruction'].item()
            epoch_losses['sup'] += losses['supervision'].item()
            epoch_losses['purity'] += losses['purity'].item()
            n_batches += 1
            
            # Update progress bar
            if use_purity_loss:
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'rec': f"{losses['reconstruction'].item():.4f}",
                    'purity': f"{losses['purity'].item():.4f}",
                })
            else:
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'rec': f"{losses['reconstruction'].item():.4f}",
                    'sup': f"{losses['supervision'].item():.4f}",
                })
        
        # Average losses
        avg_loss = epoch_losses['total'] / n_batches
        avg_rec = epoch_losses['rec'] / n_batches
        avg_sup = epoch_losses['sup'] / n_batches
        avg_purity = epoch_losses['purity'] / n_batches
        
        # Update scheduler
        scheduler.step(avg_loss)
        
        # Record history
        history['loss'].append(avg_loss)
        history['rec_loss'].append(avg_rec)
        history['sup_loss'].append(avg_sup)
        history['purity_loss'].append(avg_purity)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        if use_purity_loss:
            logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}, rec={avg_rec:.4f}, purity={avg_purity:.4f}")
        else:
            logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}, rec={avg_rec:.4f}, sup={avg_sup:.4f}")
        
        # Check for best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Save checkpoint
            if checkpoint_dir:
                checkpoint_dir = Path(checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / "autoencoder_best.pt"
                torch.save({
                    'model_state_dict': best_model_state,
                    'epoch': epoch,
                    'loss': best_loss,
                    'segment_configs': model.segment_configs,
                    'hidden_dims': hidden_dims,
                    'dropout': dropout,
                }, checkpoint_path)
                logger.debug(f"Saved checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    logger.info(f"Training complete. Best loss: {best_loss:.4f}")
    
    return model, history


def load_model(
    checkpoint_path: Path | str,
    device: str | None = None,
) -> MultiSegmentAutoencoder:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model to.
    
    Returns:
        Loaded model.
    """
    device = device if device is not None else config.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = MultiSegmentAutoencoder(
        segment_configs=checkpoint['segment_configs'],
        hidden_dims=checkpoint.get('hidden_dims'),
        dropout=checkpoint.get('dropout', 0.1),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model
