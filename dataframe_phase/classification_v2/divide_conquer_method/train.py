"""
Training loop for the reconstruction-only autoencoder.

Simplified from Autoencoder_method (no SupCon, no purity loss).
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from . import config
from .models.autoencoder import MultiSegmentAutoencoder
from .preprocessing import get_segment_lengths

logger = logging.getLogger(__name__)


def train_autoencoder(
    segments: dict[str, np.ndarray],
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    device: str | None = None,
    checkpoint_dir: Path | None = None,
    validation_split: float | None = None,
    early_stopping_patience: int | None = None,
) -> Tuple[MultiSegmentAutoencoder, dict]:
    """
    Train the multi-segment autoencoder with reconstruction-only loss.
    
    Args:
        segments: Dict mapping segment_name to (n_cells, segment_length) arrays.
        epochs: Number of training epochs. Defaults to config.AE_EPOCHS.
        batch_size: Training batch size. Defaults to config.AE_BATCH_SIZE.
        lr: Learning rate. Defaults to config.AE_LEARNING_RATE.
        device: "cuda" or "cpu". Defaults to config.DEVICE.
        checkpoint_dir: Directory to save model checkpoints.
        validation_split: Fraction for validation. Defaults to config.VALIDATION_SPLIT.
        early_stopping_patience: Epochs to wait before early stop.
    
    Returns:
        Tuple of (trained_model, training_history)
        training_history contains per-epoch loss values.
    """
    # Apply defaults
    epochs = epochs if epochs is not None else config.AE_EPOCHS
    batch_size = batch_size if batch_size is not None else config.AE_BATCH_SIZE
    lr = lr if lr is not None else config.AE_LEARNING_RATE
    device = device if device is not None else config.DEVICE
    validation_split = validation_split if validation_split is not None else config.VALIDATION_SPLIT
    early_stopping_patience = early_stopping_patience if early_stopping_patience is not None else config.EARLY_STOPPING_PATIENCE
    
    # Check CUDA availability
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
    
    # Get segment lengths and create model
    segment_lengths = get_segment_lengths(segments)
    
    model = MultiSegmentAutoencoder.from_segment_lengths(
        segment_lengths=segment_lengths,
        segment_latent_dims=config.SEGMENT_LATENT_DIMS,
        hidden_dims=config.AE_HIDDEN_DIMS,
        dropout=config.AE_DROPOUT,
    )
    
    model = model.to(device)
    logger.info(f"Model created: {model.total_latent_dim}D latent space")
    logger.info(f"Training on device: {device}")
    
    # Convert segments to tensors
    n_cells = next(iter(segments.values())).shape[0]
    segment_tensors = {
        name: torch.tensor(arr, dtype=torch.float32)
        for name, arr in segments.items()
    }
    
    # Create train/val split
    n_val = int(n_cells * validation_split)
    n_train = n_cells - n_val
    
    indices = np.random.permutation(n_cells)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    logger.info(f"Train/val split: {n_train}/{n_val} cells")
    
    # Create data loaders
    train_segments = {name: tensor[train_idx] for name, tensor in segment_tensors.items()}
    val_segments = {name: tensor[val_idx] for name, tensor in segment_tensors.items()}
    
    train_dataset = MultiSegmentDataset(train_segments)
    val_dataset = MultiSegmentDataset(val_segments)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0,
        'best_val_loss': float('inf'),
    }
    
    # Early stopping state
    patience_counter = 0
    best_state_dict = None
    
    # Create checkpoint directory
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting training for {epochs} epochs")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch in train_loader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch)
            
            # Reconstruction loss (MSE, normalized by segment length)
            loss = compute_reconstruction_loss(batch, output['reconstructions'])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {name: tensor.to(device) for name, tensor in batch.items()}
                output = model(batch)
                loss = compute_reconstruction_loss(batch, output['reconstructions'])
                val_losses.append(loss.item())
        
        # Compute epoch statistics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Early stopping check
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch
            best_state_dict = model.state_dict().copy()
            patience_counter = 0
            
            # Save best checkpoint
            if checkpoint_dir is not None:
                torch.save(best_state_dict, checkpoint_dir / "autoencoder_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1} "
                           f"(no improvement for {early_stopping_patience} epochs)")
                break
    
    # Load best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    
    logger.info(f"Training complete. Best val_loss={history['best_val_loss']:.4f} "
               f"at epoch {history['best_epoch']+1}")
    
    return model, history


def compute_reconstruction_loss(
    inputs: dict[str, torch.Tensor],
    reconstructions: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute reconstruction loss normalized by segment length.
    
    Args:
        inputs: Dict of input segment tensors.
        reconstructions: Dict of reconstructed segment tensors.
    
    Returns:
        Total reconstruction loss.
    """
    total_loss = 0.0
    
    for name in inputs:
        if name in reconstructions:
            x = inputs[name]
            x_hat = reconstructions[name]
            
            # MSE normalized by segment length
            segment_loss = torch.mean((x - x_hat) ** 2)
            total_loss = total_loss + segment_loss
    
    return total_loss


def load_model(
    checkpoint_path: Path | str,
    segment_lengths: dict[str, int],
    device: str | None = None,
) -> MultiSegmentAutoencoder:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file.
        segment_lengths: Dict mapping segment_name to length.
        device: Device to load model to.
    
    Returns:
        Loaded model.
    """
    device = device if device is not None else config.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    # Create model with same architecture
    model = MultiSegmentAutoencoder.from_segment_lengths(
        segment_lengths=segment_lengths,
        segment_latent_dims=config.SEGMENT_LATENT_DIMS,
        hidden_dims=config.AE_HIDDEN_DIMS,
        dropout=config.AE_DROPOUT,
    )
    
    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    
    return model


class MultiSegmentDataset(torch.utils.data.Dataset):
    """
    Dataset for multi-segment autoencoder training.
    
    Returns a dict of segment tensors for each sample.
    """
    
    def __init__(self, segments: dict[str, torch.Tensor]):
        self.segments = segments
        self.n_samples = next(iter(segments.values())).shape[0]
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {name: tensor[idx] for name, tensor in self.segments.items()}
