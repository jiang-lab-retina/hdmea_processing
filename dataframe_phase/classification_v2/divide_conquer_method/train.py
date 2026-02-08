"""
Training loop for the autoencoder.

Supports optional semi-supervised training: when ipRGC labels are provided,
a classification head is added and trained jointly with reconstruction.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    iprgc_labels: np.ndarray | None = None,
    classification_weight: float | None = None,
) -> Tuple[MultiSegmentAutoencoder, dict]:
    """
    Train the multi-segment autoencoder.
    
    When iprgc_labels is provided, adds a classification head and trains
    jointly with reconstruction + classification (semi-supervised).
    
    Args:
        segments: Dict mapping segment_name to (n_cells, segment_length) arrays.
        epochs: Number of training epochs. Defaults to config.AE_EPOCHS.
        batch_size: Training batch size. Defaults to config.AE_BATCH_SIZE.
        lr: Learning rate. Defaults to config.AE_LEARNING_RATE.
        device: "cuda" or "cpu". Defaults to config.DEVICE.
        checkpoint_dir: Directory to save model checkpoints.
        validation_split: Fraction for validation. Defaults to config.VALIDATION_SPLIT.
        early_stopping_patience: Epochs to wait before early stop.
        iprgc_labels: Optional (n_cells,) boolean array for semi-supervised training.
        classification_weight: Weight for classification loss. Defaults to config value.
    
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
    
    # Semi-supervised setup
    use_classification = iprgc_labels is not None
    n_classes = 1 if use_classification else 0
    if use_classification:
        classification_weight = (
            classification_weight if classification_weight is not None
            else getattr(config, 'IPRGC_CLASSIFICATION_WEIGHT', 1.0)
        )
        n_pos = int(iprgc_labels.sum())
        n_neg = len(iprgc_labels) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        logger.info(f"Semi-supervised mode: {n_pos} ipRGC / {n_neg} non-ipRGC, "
                     f"pos_weight={pos_weight.item():.2f}, cls_weight={classification_weight}")
    
    # Get segment lengths and create model
    segment_lengths = get_segment_lengths(segments)
    
    # Get encoder type from config
    encoder_type = getattr(config, 'ENCODER_TYPE', 'tcn')
    classifier_hidden = getattr(config, 'IPRGC_CLASSIFIER_HIDDEN', 32)
    
    model = MultiSegmentAutoencoder.from_segment_lengths(
        segment_lengths=segment_lengths,
        segment_latent_dims=config.SEGMENT_LATENT_DIMS,
        encoder_type=encoder_type,
        hidden_dims=config.AE_HIDDEN_DIMS,
        dropout=config.AE_DROPOUT,
        use_mlp_threshold=getattr(config, 'USE_MLP_THRESHOLD', 30),
        tcn_channels=getattr(config, 'TCN_CHANNELS', None),
        tcn_kernel_size=getattr(config, 'TCN_KERNEL_SIZE', 3),
        multiscale_kernel_sizes=getattr(config, 'MULTISCALE_KERNEL_SIZES', None),
        multiscale_channels=getattr(config, 'MULTISCALE_CHANNELS', 32),
        n_classes=n_classes,
        classifier_hidden=classifier_hidden,
    )
    
    model = model.to(device)
    mode_str = "semi-supervised" if use_classification else "reconstruction-only"
    logger.info(f"Model created: {encoder_type.upper()} encoder, {model.total_latent_dim}D latent space ({mode_str})")
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
    
    # Add ipRGC labels as pseudo-segment (extracted before model forward)
    if use_classification:
        label_tensor = torch.tensor(iprgc_labels.astype(np.float32), dtype=torch.float32)
        segment_tensors['_iprgc_label'] = label_tensor
    
    # Create data loaders
    train_segments = {name: tensor[train_idx] for name, tensor in segment_tensors.items()}
    val_segments = {name: tensor[val_idx] for name, tensor in segment_tensors.items()}
    
    train_dataset = MultiSegmentDataset(train_segments)
    val_dataset = MultiSegmentDataset(val_segments)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Move pos_weight to device if semi-supervised
    if use_classification:
        pos_weight = pos_weight.to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_cls_loss': [],
        'train_cls_acc': [],
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
        epoch_cls_losses = []
        epoch_cls_correct = 0
        epoch_cls_total = 0
        
        for batch in train_loader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            
            # Extract ipRGC label before model forward (not a real segment)
            iprgc_label_batch = batch.pop('_iprgc_label', None)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch)
            
            # Reconstruction loss (MSE, normalized by segment length)
            loss = compute_reconstruction_loss(batch, output['reconstructions'])
            
            # Classification loss (semi-supervised)
            if use_classification and iprgc_label_batch is not None and 'class_logits' in output:
                logits = output['class_logits'].squeeze(-1)  # (batch,)
                cls_loss = F.binary_cross_entropy_with_logits(
                    logits, iprgc_label_batch, pos_weight=pos_weight
                )
                loss = loss + classification_weight * cls_loss
                epoch_cls_losses.append(cls_loss.item())
                # Track accuracy
                preds = (logits > 0).float()
                epoch_cls_correct += (preds == iprgc_label_batch).sum().item()
                epoch_cls_total += len(iprgc_label_batch)
            
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
                # Remove pseudo-label for validation too
                batch.pop('_iprgc_label', None)
                output = model(batch)
                loss = compute_reconstruction_loss(batch, output['reconstructions'])
                val_losses.append(loss.item())
        
        # Compute epoch statistics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if use_classification and epoch_cls_losses:
            cls_loss_mean = np.mean(epoch_cls_losses)
            cls_acc = epoch_cls_correct / max(epoch_cls_total, 1)
            history['train_cls_loss'].append(cls_loss_mean)
            history['train_cls_acc'].append(cls_acc)
        
        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            msg = (f"Epoch {epoch+1}/{epochs}: "
                   f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            if use_classification and epoch_cls_losses:
                msg += f", cls_loss={cls_loss_mean:.4f}, cls_acc={cls_acc:.3f}"
            logger.info(msg)
        
        # Early stopping check
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch
            best_state_dict = model.state_dict().copy()
            patience_counter = 0
            
            # Save best checkpoint with metadata
            if checkpoint_dir is not None:
                checkpoint = {
                    'state_dict': best_state_dict,
                    'encoder_type': encoder_type,
                    'segment_lengths': segment_lengths,
                    'n_classes': n_classes,
                    'classifier_hidden': classifier_hidden,
                    'epoch': epoch,
                    'val_loss': val_loss,
                }
                torch.save(checkpoint, checkpoint_dir / "autoencoder_best.pt")
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
    segment_lengths: dict[str, int] | None = None,
    device: str | None = None,
    encoder_type: str | None = None,
) -> MultiSegmentAutoencoder:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file.
        segment_lengths: Dict mapping segment_name to length. If None, loaded from checkpoint.
        device: Device to load model to.
        encoder_type: Encoder type to use. If None, loaded from checkpoint or config.
    
    Returns:
        Loaded model.
    """
    device = device if device is not None else config.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    # Load checkpoint (may be dict with metadata or just state_dict)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle both old format (just state_dict) and new format (dict with metadata)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # New format with metadata
        state_dict = checkpoint['state_dict']
        saved_encoder_type = checkpoint.get('encoder_type', 'tcn')
        saved_segment_lengths = checkpoint.get('segment_lengths', None)
        saved_n_classes = checkpoint.get('n_classes', 0)
        saved_classifier_hidden = checkpoint.get('classifier_hidden', 32)
        logger.info(f"Checkpoint contains metadata: encoder_type={saved_encoder_type}, "
                     f"n_classes={saved_n_classes}")
    else:
        # Old format (just state_dict)
        state_dict = checkpoint
        saved_encoder_type = None
        saved_segment_lengths = None
        saved_n_classes = 0
        saved_classifier_hidden = 32
        logger.warning("Checkpoint is in old format (no metadata). Using config defaults.")
    
    # Determine encoder type (priority: argument > checkpoint > config)
    if encoder_type is None:
        encoder_type = saved_encoder_type or getattr(config, 'ENCODER_TYPE', 'tcn')
    
    # Determine segment lengths (priority: argument > checkpoint)
    if segment_lengths is None:
        if saved_segment_lengths is not None:
            segment_lengths = saved_segment_lengths
        else:
            raise ValueError("segment_lengths must be provided for old checkpoint format")
    
    # Create model with same architecture (including classifier if saved)
    model = MultiSegmentAutoencoder.from_segment_lengths(
        segment_lengths=segment_lengths,
        segment_latent_dims=config.SEGMENT_LATENT_DIMS,
        encoder_type=encoder_type,
        hidden_dims=config.AE_HIDDEN_DIMS,
        dropout=config.AE_DROPOUT,
        use_mlp_threshold=getattr(config, 'USE_MLP_THRESHOLD', 30),
        tcn_channels=getattr(config, 'TCN_CHANNELS', None),
        tcn_kernel_size=getattr(config, 'TCN_KERNEL_SIZE', 3),
        multiscale_kernel_sizes=getattr(config, 'MULTISCALE_KERNEL_SIZES', None),
        multiscale_channels=getattr(config, 'MULTISCALE_CHANNELS', 32),
        n_classes=saved_n_classes,
        classifier_hidden=saved_classifier_hidden,
    )
    
    # Load state dict with strict=False for backward compatibility
    # (old checkpoints without classifier keys will load fine)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys when loading checkpoint: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading checkpoint: {unexpected}")
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded {encoder_type.upper()} model from {checkpoint_path}")
    
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
