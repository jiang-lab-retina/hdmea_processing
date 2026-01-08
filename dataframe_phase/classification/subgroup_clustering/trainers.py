"""
Training Loops for Subgroup Clustering Models.

Implements training for:
1. Standard Autoencoder
2. Variational Autoencoder (VAE)
3. Deep Embedded Clustering (DEC)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm

from .models import (
    ConvAutoencoder, ConvVAE, DECModel, 
    ContrastiveAutoencoder, SupervisedContrastiveLoss,
    get_device,
)
from .config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    TRAINING_PATIENCE,
    BATCH_SIZE,
    RANDOM_SEED,
    VAE_BETA,
    DEC_PRETRAIN_EPOCHS,
    DEC_FINETUNE_EPOCHS,
    DEC_UPDATE_INTERVAL,
    DEC_TOL,
    CONTRASTIVE_WEIGHT,
    CONTRASTIVE_TEMPERATURE,
    PROJECTION_DIM,
)


def set_seed(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# AUTOENCODER TRAINER
# =============================================================================

def train_autoencoder(
    X: np.ndarray,
    latent_dim: int = 64,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    patience: int = TRAINING_PATIENCE,
    verbose: bool = True,
    save_path: Optional[Path] = None,
) -> Tuple[ConvAutoencoder, np.ndarray]:
    """
    Train standard autoencoder.
    
    Parameters
    ----------
    X : np.ndarray
        Input traces [N, input_length]
    latent_dim : int
        Latent space dimension
    num_epochs : int
        Maximum training epochs
    batch_size : int
        Batch size
    learning_rate : float
        Learning rate
    patience : int
        Early stopping patience
    verbose : bool
        Print training progress
    save_path : Path, optional
        Path to save best model
        
    Returns
    -------
    tuple
        (trained_model, latent_codes)
    """
    set_seed()
    device = get_device()
    input_length = X.shape[1]
    
    # Prepare data
    X_tensor = torch.from_numpy(X).float().unsqueeze(1)
    dataset = TensorDataset(X_tensor, X_tensor)
    
    n = len(dataset)
    n_val = int(0.2 * n)
    n_train = n - n_val
    
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    # Build model
    model = ConvAutoencoder(input_length, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    trials = 0
    
    if verbose:
        print(f"    Training AE: {n_train} train, {n_val} val samples")
    
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xb, _ in train_loader:
            xb = xb.to(device)
            recon, _ = model(xb)
            loss = criterion(recon, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= n_train
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                recon, _ = model(xb)
                val_loss += criterion(recon, xb).item() * xb.size(0)
        val_loss /= n_val
        
        if verbose and (epoch % 50 == 0 or epoch == num_epochs):
            print(f"      Epoch {epoch:03d}: Train={train_loss:.4f}, Val={val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            trials = 0
        else:
            trials += 1
            if trials >= patience:
                if verbose:
                    print(f"      Early stopping at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(best_state)
    model.eval()
    
    if save_path:
        torch.save(best_state, save_path)
    
    # Extract latent codes
    with torch.no_grad():
        latents = model.encode(X_tensor.to(device)).cpu().numpy()
    
    return model, latents


# =============================================================================
# VAE TRAINER
# =============================================================================

def train_vae(
    X: np.ndarray,
    latent_dim: int = 64,
    beta: float = VAE_BETA,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    patience: int = TRAINING_PATIENCE,
    verbose: bool = True,
    save_path: Optional[Path] = None,
) -> Tuple[ConvVAE, np.ndarray]:
    """
    Train Variational Autoencoder.
    
    Parameters
    ----------
    X : np.ndarray
        Input traces [N, input_length]
    latent_dim : int
        Latent space dimension
    beta : float
        KL divergence weight (beta-VAE)
    num_epochs : int
        Maximum training epochs
    batch_size : int
        Batch size
    learning_rate : float
        Learning rate
    patience : int
        Early stopping patience
    verbose : bool
        Print training progress
    save_path : Path, optional
        Path to save best model
        
    Returns
    -------
    tuple
        (trained_model, latent_codes)
    """
    set_seed()
    device = get_device()
    input_length = X.shape[1]
    
    # Prepare data
    X_tensor = torch.from_numpy(X).float().unsqueeze(1)
    dataset = TensorDataset(X_tensor, X_tensor)
    
    n = len(dataset)
    n_val = int(0.2 * n)
    n_train = n - n_val
    
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    # Build model
    model = ConvVAE(input_length, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    
    best_val_loss = float('inf')
    best_state = None
    trials = 0
    
    if verbose:
        print(f"    Training VAE (β={beta}): {n_train} train, {n_val} val samples")
    
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_recon = 0.0
        train_kl = 0.0
        for xb, _ in train_loader:
            xb = xb.to(device)
            recon, z, mu, logvar = model(xb)
            
            recon_loss = nn.functional.mse_loss(recon, xb)
            kl_loss = ConvVAE.kl_divergence(mu, logvar)
            loss = recon_loss + beta * kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_recon += recon_loss.item() * xb.size(0)
            train_kl += kl_loss.item() * xb.size(0)
        
        train_recon /= n_train
        train_kl /= n_train
        
        # Validate
        model.eval()
        val_recon = 0.0
        val_kl = 0.0
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                recon, z, mu, logvar = model(xb)
                recon_loss = nn.functional.mse_loss(recon, xb)
                kl_loss = ConvVAE.kl_divergence(mu, logvar)
                val_recon += recon_loss.item() * xb.size(0)
                val_kl += kl_loss.item() * xb.size(0)
        
        val_recon /= n_val
        val_kl /= n_val
        val_loss = val_recon + beta * val_kl
        
        if verbose and (epoch % 50 == 0 or epoch == num_epochs):
            print(f"      Epoch {epoch:03d}: Train R={train_recon:.4f} KL={train_kl:.4f}, "
                  f"Val R={val_recon:.4f} KL={val_kl:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            trials = 0
        else:
            trials += 1
            if trials >= patience:
                if verbose:
                    print(f"      Early stopping at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(best_state)
    model.eval()
    
    if save_path:
        torch.save(best_state, save_path)
    
    # Extract latent codes (use mu for clustering)
    with torch.no_grad():
        latents = model.get_latent(X_tensor.to(device)).cpu().numpy()
    
    return model, latents


# =============================================================================
# DEC TRAINER
# =============================================================================

def train_dec(
    X: np.ndarray,
    n_clusters: int,
    latent_dim: int = 64,
    pretrain_epochs: int = DEC_PRETRAIN_EPOCHS,
    finetune_epochs: int = DEC_FINETUNE_EPOCHS,
    update_interval: int = DEC_UPDATE_INTERVAL,
    tol: float = DEC_TOL,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    verbose: bool = True,
    save_path: Optional[Path] = None,
) -> Tuple[DECModel, np.ndarray, np.ndarray]:
    """
    Train Deep Embedded Clustering model.
    
    Two-phase training:
    1. Pre-train autoencoder with reconstruction loss
    2. Fine-tune with clustering loss (KL divergence to target distribution)
    
    Parameters
    ----------
    X : np.ndarray
        Input traces [N, input_length]
    n_clusters : int
        Number of clusters
    latent_dim : int
        Latent space dimension
    pretrain_epochs : int
        Pre-training epochs for autoencoder
    finetune_epochs : int
        Fine-tuning epochs with clustering loss
    update_interval : int
        Epochs between target distribution updates
    tol : float
        Convergence tolerance (stop if assignment change < tol)
    batch_size : int
        Batch size
    learning_rate : float
        Learning rate
    verbose : bool
        Print training progress
    save_path : Path, optional
        Path to save best model
        
    Returns
    -------
    tuple
        (trained_model, latent_codes, cluster_assignments)
    """
    set_seed()
    device = get_device()
    input_length = X.shape[1]
    
    # Prepare data
    X_tensor = torch.from_numpy(X).float().unsqueeze(1)
    dataset = TensorDataset(X_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Build model
    model = DECModel(input_length, latent_dim, n_clusters).to(device)
    
    # =========================================================================
    # Phase 1: Pre-train autoencoder
    # =========================================================================
    if verbose:
        print(f"    DEC Phase 1: Pre-training autoencoder ({pretrain_epochs} epochs)")
    
    optimizer = torch.optim.Adam(model.autoencoder.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    
    for epoch in range(1, pretrain_epochs + 1):
        model.train()
        total_loss = 0.0
        for (xb,) in data_loader:
            xb = xb.to(device)
            recon, z = model.autoencoder(xb)
            loss = criterion(recon, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        
        if verbose and (epoch % 50 == 0 or epoch == pretrain_epochs):
            print(f"      Epoch {epoch:03d}: Loss={total_loss/len(X):.4f}")
    
    # =========================================================================
    # Initialize cluster centers with k-means
    # =========================================================================
    if verbose:
        print(f"    DEC: Initializing {n_clusters} cluster centers with k-means")
    
    model.eval()
    with torch.no_grad():
        z_all = model.encode(X_tensor.to(device))
    model.initialize_clusters(z_all, method="kmeans")
    
    # =========================================================================
    # Phase 2: Fine-tune with clustering loss
    # =========================================================================
    if verbose:
        print(f"    DEC Phase 2: Fine-tuning with clustering loss ({finetune_epochs} epochs)")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate * 0.1, weight_decay=WEIGHT_DECAY)
    
    # Initial assignments
    with torch.no_grad():
        _, _, q_all = model(X_tensor.to(device))
        prev_assignments = q_all.argmax(dim=1).cpu().numpy()
    
    for epoch in range(1, finetune_epochs + 1):
        # Update target distribution every update_interval epochs
        if epoch % update_interval == 1 or epoch == 1:
            model.eval()
            with torch.no_grad():
                _, _, q_all = model(X_tensor.to(device))
                p_all = model.clustering.target_distribution(q_all)
        
        model.train()
        total_loss = 0.0
        idx = 0
        
        for (xb,) in data_loader:
            xb = xb.to(device)
            bs = xb.size(0)
            
            recon, z, q = model(xb)
            p = p_all[idx:idx+bs]
            
            # Combined loss: reconstruction + clustering
            recon_loss = criterion(recon, xb)
            cluster_loss = model.clustering_loss(q, p)
            loss = recon_loss + cluster_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * bs
            idx += bs
        
        # Check convergence
        if epoch % update_interval == 0:
            model.eval()
            with torch.no_grad():
                _, _, q_all = model(X_tensor.to(device))
                curr_assignments = q_all.argmax(dim=1).cpu().numpy()
            
            delta = np.sum(curr_assignments != prev_assignments) / len(X)
            prev_assignments = curr_assignments
            
            if verbose:
                print(f"      Epoch {epoch:03d}: Loss={total_loss/len(X):.4f}, Δ assignments={delta:.4f}")
            
            if delta < tol:
                if verbose:
                    print(f"      Converged at epoch {epoch} (Δ < {tol})")
                break
    
    # Final results
    model.eval()
    if save_path:
        torch.save(model.state_dict(), save_path)
    
    with torch.no_grad():
        latents = model.encode(X_tensor.to(device)).cpu().numpy()
        _, _, q = model(X_tensor.to(device))
        assignments = q.argmax(dim=1).cpu().numpy()
    
    return model, latents, assignments


# =============================================================================
# CONTRASTIVE AUTOENCODER TRAINER
# =============================================================================

def train_contrastive_ae(
    X: np.ndarray,
    labels: np.ndarray,
    latent_dim: int = 64,
    projection_dim: int = PROJECTION_DIM,
    contrastive_weight: float = CONTRASTIVE_WEIGHT,
    temperature: float = CONTRASTIVE_TEMPERATURE,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    patience: int = TRAINING_PATIENCE,
    n_conv_layers: int = 3,
    base_channels: int = 32,
    dropout: float = 0.1,
    verbose: bool = True,
    save_path: Optional[Path] = None,
) -> Tuple[ContrastiveAutoencoder, np.ndarray]:
    """
    Train Contrastive Autoencoder with supervised contrastive loss.
    
    The model jointly optimizes:
    1. Reconstruction loss (MSE): Preserve information
    2. Supervised contrastive loss: Pull same-label samples together,
       push different-label samples apart
    
    L_total = L_reconstruction + λ * L_contrastive
    
    Parameters
    ----------
    X : np.ndarray
        Input traces [N, input_length]
    labels : np.ndarray
        Cluster labels for contrastive learning [N]
        These should be initial cluster assignments (e.g., from K-Means on standard AE)
    latent_dim : int
        Latent space dimension
    projection_dim : int
        Projection head output dimension (for contrastive loss)
    contrastive_weight : float
        Weight λ for contrastive loss (higher = more emphasis on separation)
    temperature : float
        Temperature τ for contrastive loss (lower = sharper separation)
    num_epochs : int
        Maximum training epochs
    batch_size : int
        Batch size (larger is better for contrastive learning)
    learning_rate : float
        Learning rate
    patience : int
        Early stopping patience
    n_conv_layers : int
        Number of convolutional layers
    base_channels : int
        Base channel count
    dropout : float
        Dropout rate
    verbose : bool
        Print training progress
    save_path : Path, optional
        Path to save best model
        
    Returns
    -------
    tuple
        (trained_model, latent_codes)
    """
    set_seed()
    device = get_device()
    input_length = X.shape[1]
    
    # Convert labels to integers for contrastive loss
    unique_labels = np.unique(labels)
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    labels_int = np.array([label_to_idx[lbl] for lbl in labels])
    
    # Prepare data
    X_tensor = torch.from_numpy(X).float().unsqueeze(1)
    labels_tensor = torch.from_numpy(labels_int).long()
    dataset = TensorDataset(X_tensor, labels_tensor)
    
    n = len(dataset)
    n_val = int(0.2 * n)
    n_train = n - n_val
    
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    
    # Use larger batch size for better contrastive learning (more negatives per batch)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    # Build model
    model = ContrastiveAutoencoder(
        input_length, latent_dim, projection_dim,
        n_conv_layers, base_channels, dropout
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    recon_criterion = nn.MSELoss()
    contrastive_criterion = SupervisedContrastiveLoss(temperature=temperature)
    
    best_val_loss = float('inf')
    best_state = None
    trials = 0
    
    if verbose:
        print(f"    Training Contrastive AE (λ={contrastive_weight}, τ={temperature}): {n_train} train, {n_val} val")
    
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_recon = 0.0
        train_contrast = 0.0
        n_batches = 0
        
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            # Forward pass
            recon, z, proj = model(xb)
            
            # Losses
            recon_loss = recon_criterion(recon, xb)
            contrast_loss = contrastive_criterion(proj, yb)
            total_loss = recon_loss + contrastive_weight * contrast_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_recon += recon_loss.item()
            train_contrast += contrast_loss.item()
            n_batches += 1
        
        scheduler.step()
        
        train_recon /= max(n_batches, 1)
        train_contrast /= max(n_batches, 1)
        
        # Validate
        model.eval()
        val_recon = 0.0
        val_contrast = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                
                recon, z, proj = model(xb)
                recon_loss = recon_criterion(recon, xb)
                contrast_loss = contrastive_criterion(proj, yb)
                
                val_recon += recon_loss.item()
                val_contrast += contrast_loss.item()
                val_batches += 1
        
        val_recon /= max(val_batches, 1)
        val_contrast /= max(val_batches, 1)
        val_loss = val_recon + contrastive_weight * val_contrast
        
        if verbose and (epoch % 50 == 0 or epoch == num_epochs):
            print(f"      Epoch {epoch:03d}: Train R={train_recon:.4f} C={train_contrast:.4f}, "
                  f"Val R={val_recon:.4f} C={val_contrast:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            trials = 0
        else:
            trials += 1
            if trials >= patience:
                if verbose:
                    print(f"      Early stopping at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(best_state)
    model.eval()
    
    if save_path:
        torch.save(best_state, save_path)
    
    # Extract latent codes
    with torch.no_grad():
        latents = model.encode(X_tensor.to(device)).cpu().numpy()
    
    return model, latents

