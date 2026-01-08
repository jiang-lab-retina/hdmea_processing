"""
RGC Classification with Supervised Contrastive Autoencoder.

This script:
1. Loads firing rate data and filters for RGC units only
2. Classifies RGCs into 4 subgroups (ipRGC, DSGC, OSGC, Other)
3. Concatenates mean traces from all movies into one trace per unit
4. Uses a Supervised Contrastive Autoencoder to learn subtype-discriminative 
   latent representations with dimensionality 100
5. Saves the classified dataframe with latent codes

The model jointly optimizes:
    L_total = L_reconstruction + λ * L_contrastive

Where L_contrastive pulls same-subtype samples together and pushes 
different subtypes apart in the latent space.

Usage:
    python classification.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
INPUT_PARQUET = PROJECT_ROOT / "dataframe_phase/load_feature/firing_rate_with_dsgc_features_typed20251230.parquet"
OUTPUT_PARQUET = Path(__file__).parent / "rgc_classified_with_ae20251230.parquet"
MODEL_SAVE_PATH = Path(__file__).parent / "contrastive_autoencoder_best.pth"

# Classification thresholds
DS_P_THRESHOLD = 0.05
OS_P_THRESHOLD = 0.05
IPRGC_QI_THRESHOLD = 0.8

# Subtype label mapping
SUBTYPE_TO_LABEL = {"ipRGC": 0, "DSGC": 1, "OSGC": 2, "Other": 3}
LABEL_TO_SUBTYPE = {v: k for k, v in SUBTYPE_TO_LABEL.items()}

# Movie columns to concatenate (excluding baseline_127)
MOVIE_COLUMNS = [
    "freq_step_5st_3x",
    "green_blue_3s_3i_3x",
    "step_up_5s_5i_b0_3x",
    "step_up_5s_5i_b0_30x",
    "moving_h_bar_s5_d8_3x_0",
    "moving_h_bar_s5_d8_3x_45",
    "moving_h_bar_s5_d8_3x_90",
    "moving_h_bar_s5_d8_3x_135",
    "moving_h_bar_s5_d8_3x_180",
    "moving_h_bar_s5_d8_3x_225",
    "moving_h_bar_s5_d8_3x_270",
    "moving_h_bar_s5_d8_3x_315",
    "iprgc_test",
]

# Autoencoder hyperparameters
LATENT_DIM = 100
PROJECTION_DIM = 128  # Larger projection head for better contrastive learning
LEARNING_RATE = 5e-5  # Lower LR for stability
WEIGHT_DECAY = 1e-4  # Stronger regularization
NUM_EPOCHS = 300
TRAINING_PATIENCE = 40
BATCH_SIZE = 256  # Larger batch improves contrastive learning
RANDOM_SEED = 42

# Contrastive loss hyperparameters
CONTRASTIVE_WEIGHT = 0.3  # Lower weight to balance with reconstruction
TEMPERATURE = 0.07  # Lower temperature for sharper similarity distribution


# =============================================================================
# SUPERVISED CONTRASTIVE LOSS
# =============================================================================

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss from Khosla et al. (2020).
    
    For each anchor, pulls together samples with the same label
    and pushes apart samples with different labels.
    
    L = -sum_i (1/|P(i)|) * sum_{p in P(i)} log(
        exp(z_i · z_p / τ) / sum_{a in A(i)} exp(z_i · z_a / τ)
    )
    
    where:
        P(i) = set of positive samples (same class as anchor i)
        A(i) = set of all samples except anchor i
        τ = temperature parameter
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Parameters
        ----------
        features : torch.Tensor
            L2-normalized feature vectors of shape [batch_size, feature_dim]
        labels : torch.Tensor
            Class labels of shape [batch_size]
            
        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size <= 1:
            return torch.tensor(0.0, device=device)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix: [B, B]
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same class, excluding self)
        labels = labels.view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float()  # [B, B]
        mask_self = torch.eye(batch_size, device=device)
        mask_positive = mask_positive - mask_self  # Exclude self
        
        # Number of positives per anchor
        num_positives = mask_positive.sum(dim=1)  # [B]
        
        # For numerical stability, subtract max from similarity
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()
        
        # Compute log_softmax over all samples except self
        # exp_logits: [B, B]
        exp_logits = torch.exp(logits) * (1 - mask_self)  # Exclude self
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positive pairs
        # Only consider anchors that have at least one positive
        mask_valid = num_positives > 0
        
        if not mask_valid.any():
            return torch.tensor(0.0, device=device)
        
        # Mean log-prob for positive pairs
        mean_log_prob_pos = (mask_positive * log_prob).sum(dim=1) / (num_positives + 1e-8)
        
        # Loss: negative mean over valid anchors
        loss = -mean_log_prob_pos[mask_valid].mean()
        
        return loss


# =============================================================================
# CONTRASTIVE AUTOENCODER MODEL
# =============================================================================

class ContrastiveAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder with Contrastive Projection Head.
    
    Architecture:
        Encoder: Conv1d (1->16->32->64) with stride 2
        Bottleneck: FC layers to latent_dim
        Projection Head: MLP for contrastive learning
        Decoder: ConvTranspose1d (64->32->16->1) with stride 2
    """
    
    def __init__(
        self, 
        input_length: int, 
        latent_dim: int = 100,
        projection_dim: int = 64,
    ):
        super().__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim
        act = nn.LeakyReLU(0.2, inplace=True)
        
        # Encoder conv stack
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16), act,
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32), act,
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), act,
        )
        
        # Figure out flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            conv_out = self.encoder_conv(dummy)
            C, L = conv_out.shape[1], conv_out.shape[2]
            self._flattened_size = C * L
            self._conv_channels = C
            self._conv_length = L
        
        # Bottleneck layers
        self.fc_enc = nn.Linear(self._flattened_size, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self._flattened_size)
        
        # Projection head for contrastive learning (MLP with dropout)
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, projection_dim),
        )
        
        # Decoder conv-transpose stack
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            nn.BatchNorm1d(32), act,
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            nn.BatchNorm1d(16), act,
            nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        x = self.encoder_conv(x)
        B = x.size(0)
        x = x.view(B, self._flattened_size)
        z = self.fc_enc(x)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to reconstruction."""
        x = self.fc_dec(z).view(-1, self._conv_channels, self._conv_length)
        x = self.decoder_conv(x)
        x = x[..., :self.input_length]
        return x
    
    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Project latent code for contrastive learning."""
        return self.projection_head(z)
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass returning reconstruction, latent code, and projection.
        
        Returns
        -------
        tuple
            (reconstruction, latent_code, projection)
        """
        z = self.encode(x)
        recon = self.decode(z)
        proj = self.project(z)
        return recon, z, proj


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_contrastive_autoencoder(
    X: np.ndarray,
    labels: np.ndarray,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    num_epochs: int = 500,
    latent_dim: int = 100,
    projection_dim: int = 64,
    contrastive_weight: float = 0.5,
    temperature: float = 0.1,
    training_patience: int = 50,
    random_seed: int = 42,
    batch_size: int = 128,
    save_path: str = "contrastive_autoencoder_best.pth",
):
    """
    Train supervised contrastive autoencoder.
    
    Parameters
    ----------
    X : np.ndarray
        2D array of traces, shape [N, input_length]
    labels : np.ndarray
        1D array of subtype labels (integers), shape [N]
    learning_rate : float
        Learning rate for Adam optimizer
    weight_decay : float
        L2 regularization weight
    num_epochs : int
        Maximum number of training epochs
    latent_dim : int
        Dimension of latent space
    projection_dim : int
        Dimension of contrastive projection
    contrastive_weight : float
        Weight λ for contrastive loss (L_total = L_recon + λ * L_contrastive)
    temperature : float
        Temperature for contrastive loss
    training_patience : int
        Early stopping patience
    random_seed : int
        Random seed for reproducibility
    batch_size : int
        Batch size (larger is better for contrastive learning)
    save_path : str
        Path to save best model
        
    Returns
    -------
    tuple
        (model, latents, X_tensor, device)
    """
    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    input_length = X.shape[1]
    X_tensor = torch.from_numpy(X).float().unsqueeze(1)  # [N, 1, input_length]
    labels_tensor = torch.from_numpy(labels).long()
    
    full_dataset = TensorDataset(X_tensor, labels_tensor)
    
    # Split into train/val (80/20) with stratification approximation
    n = len(full_dataset)
    n_val = int(0.2 * n)
    n_train = n - n_val
    
    # Use generator for reproducibility
    generator = torch.Generator().manual_seed(random_seed)
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=generator)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, drop_last=False)
    
    # Build model
    model = ContrastiveAutoencoder(input_length, latent_dim, projection_dim).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Losses
    recon_criterion = nn.MSELoss()
    contrastive_criterion = SupervisedContrastiveLoss(temperature=temperature)
    
    # Early stopping params
    best_val_loss = float('inf')
    trials = 0
    
    print(f"\nTraining Supervised Contrastive Autoencoder...")
    print(f"  Latent dim: {latent_dim}, Projection dim: {projection_dim}")
    print(f"  Contrastive weight (λ): {contrastive_weight}")
    print(f"  Temperature (τ): {temperature}")
    print(f"  Training samples: {n_train}")
    print(f"  Validation samples: {n_val}")
    print(f"  Input length: {input_length}")
    print(f"  Batch size: {batch_size}")
    
    for epoch in range(1, num_epochs + 1):
        # Training step
        model.train()
        train_recon_loss = 0.0
        train_contrastive_loss = 0.0
        train_total_loss = 0.0
        n_batches = 0
        
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            # Forward pass
            recon, z, proj = model(xb)
            
            # Compute losses
            loss_recon = recon_criterion(recon, xb)
            loss_contrastive = contrastive_criterion(proj, yb)
            loss_total = loss_recon + contrastive_weight * loss_contrastive
            
            # Backward pass
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            # Accumulate losses
            train_recon_loss += loss_recon.item()
            train_contrastive_loss += loss_contrastive.item()
            train_total_loss += loss_total.item()
            n_batches += 1
        
        train_recon_loss /= n_batches
        train_contrastive_loss /= n_batches
        train_total_loss /= n_batches
        
        # Validation step
        model.eval()
        val_recon_loss = 0.0
        val_contrastive_loss = 0.0
        val_total_loss = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                
                recon, z, proj = model(xb)
                
                loss_recon = recon_criterion(recon, xb)
                loss_contrastive = contrastive_criterion(proj, yb)
                loss_total = loss_recon + contrastive_weight * loss_contrastive
                
                val_recon_loss += loss_recon.item()
                val_contrastive_loss += loss_contrastive.item()
                val_total_loss += loss_total.item()
                n_val_batches += 1
        
        val_recon_loss /= n_val_batches
        val_contrastive_loss /= n_val_batches
        val_total_loss /= n_val_batches
        
        # Update learning rate
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == num_epochs:
            print(f"  Epoch {epoch:03d} – "
                  f"Train: {train_total_loss:.4f} (R:{train_recon_loss:.4f}, C:{train_contrastive_loss:.4f}) – "
                  f"Val: {val_total_loss:.4f} (R:{val_recon_loss:.4f}, C:{val_contrastive_loss:.4f})")
        
        # Early stopping check on total validation loss
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            torch.save(model.state_dict(), save_path)
            trials = 0
        else:
            trials += 1
            if trials >= training_patience:
                print(f"  Early stopping at epoch {epoch}: no improvement for {training_patience} epochs.")
                break
    
    # Load best model
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()
    
    # Extract latent codes for all data
    with torch.no_grad():
        latents = model.encode(X_tensor.to(device)).cpu()
    
    print(f"\n  Best validation loss: {best_val_loss:.4f}")
    print(f"  Latent codes shape: {latents.shape}")
    
    return model, latents, X_tensor, device


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def load_and_filter_rgc(parquet_path: Path) -> pd.DataFrame:
    """
    Load parquet and filter for RGC units only.
    
    Parameters
    ----------
    parquet_path : Path
        Path to input parquet file
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing only RGC units
    """
    print(f"Loading data from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"  Total units: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
    
    # Filter for RGC only (case-insensitive)
    rgc_mask = df["axon_type"].str.lower() == "rgc"
    df_rgc = df[rgc_mask].copy()
    print(f"  RGC units: {len(df_rgc)}")
    
    return df_rgc


def classify_rgc_subtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify RGCs into 4 subgroups with priority: ipRGC > DSGC > OSGC > Other.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with RGC units
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'rgc_subtype' and 'rgc_label' columns
    """
    print("\nClassifying RGC subtypes...")
    print(f"  Thresholds: DS p < {DS_P_THRESHOLD}, OS p < {OS_P_THRESHOLD}, ipRGC QI > {IPRGC_QI_THRESHOLD}")
    
    # Initialize subtype column
    df["rgc_subtype"] = "Other"
    
    # Priority order: ipRGC > DSGC > OSGC > Other
    # Apply in reverse order so higher priority overwrites
    
    # OSGC: os_p_value < 0.05
    osgc_mask = df["os_p_value"] < OS_P_THRESHOLD
    df.loc[osgc_mask, "rgc_subtype"] = "OSGC"
    
    # DSGC: ds_p_value < 0.05 (overwrites OSGC if both)
    dsgc_mask = df["ds_p_value"] < DS_P_THRESHOLD
    df.loc[dsgc_mask, "rgc_subtype"] = "DSGC"
    
    # ipRGC: iprgc_2hz_QI > 0.8 (highest priority)
    iprgc_mask = df["iprgc_2hz_QI"] > IPRGC_QI_THRESHOLD
    df.loc[iprgc_mask, "rgc_subtype"] = "ipRGC"
    
    # Add numeric label for training
    df["rgc_label"] = df["rgc_subtype"].map(SUBTYPE_TO_LABEL)
    
    # Print summary
    subtype_counts = df["rgc_subtype"].value_counts()
    print("\n  Subtype distribution:")
    for subtype, count in subtype_counts.items():
        pct = 100 * count / len(df)
        print(f"    {subtype}: {count} ({pct:.1f}%)")
    
    return df


def compute_mean_trace(trial_data) -> np.ndarray:
    """
    Compute mean trace from trial data.
    
    Parameters
    ----------
    trial_data : list or np.ndarray
        Trial data (list of traces)
        
    Returns
    -------
    np.ndarray
        Mean trace across trials, or empty array if invalid
    """
    if trial_data is None:
        return np.array([])
    
    if isinstance(trial_data, float) and np.isnan(trial_data):
        return np.array([])
    
    try:
        # Filter out None trials
        valid_trials = [np.array(trial) for trial in trial_data if trial is not None]
        if len(valid_trials) == 0:
            return np.array([])
        
        # Stack and compute mean
        trials_array = np.vstack(valid_trials)
        mean_trace = np.mean(trials_array, axis=0)
        return mean_trace
    except Exception:
        return np.array([])


def concatenate_mean_traces(df: pd.DataFrame, movie_columns: list) -> tuple:
    """
    Compute mean trace for each movie and concatenate.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with movie columns
    movie_columns : list
        List of movie column names to concatenate
        
    Returns
    -------
    tuple
        (concatenated_traces: np.ndarray, valid_mask: np.ndarray)
    """
    print("\nConcatenating mean traces...")
    print(f"  Movies to concatenate: {len(movie_columns)}")
    
    # Check which columns exist
    existing_cols = [col for col in movie_columns if col in df.columns]
    missing_cols = [col for col in movie_columns if col not in df.columns]
    
    if missing_cols:
        print(f"  Warning: Missing columns: {missing_cols}")
    print(f"  Using columns: {len(existing_cols)}")
    
    concatenated_traces = []
    valid_indices = []
    
    for idx in tqdm(df.index, desc="Processing traces"):
        row = df.loc[idx]
        traces_to_concat = []
        valid = True
        
        for col in existing_cols:
            mean_trace = compute_mean_trace(row.get(col))
            if len(mean_trace) == 0:
                valid = False
                break
            traces_to_concat.append(mean_trace)
        
        if valid and len(traces_to_concat) > 0:
            concatenated = np.concatenate(traces_to_concat)
            concatenated_traces.append(concatenated)
            valid_indices.append(idx)
    
    print(f"  Valid units with all traces: {len(valid_indices)} / {len(df)}")
    
    # Convert to numpy array
    if len(concatenated_traces) > 0:
        # Ensure all traces have same length
        trace_lengths = [len(t) for t in concatenated_traces]
        min_len = min(trace_lengths)
        max_len = max(trace_lengths)
        
        if min_len != max_len:
            print(f"  Warning: Trace lengths vary ({min_len} - {max_len}), truncating to min")
            concatenated_traces = [t[:min_len] for t in concatenated_traces]
        
        X = np.vstack(concatenated_traces)
        print(f"  Concatenated trace shape: {X.shape}")
    else:
        X = np.array([])
    
    return X, valid_indices


def replace_nan_in_traces(X: np.ndarray, replace_value: float = 0.0) -> np.ndarray:
    """
    Replace NaN values in trace array.
    
    Parameters
    ----------
    X : np.ndarray
        Trace array
    replace_value : float
        Value to replace NaNs with
        
    Returns
    -------
    np.ndarray
        Trace array with NaNs replaced
    """
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"  Replacing {nan_count} NaN values with {replace_value}")
        X = np.nan_to_num(X, nan=replace_value)
    return X


def normalize_traces(X: np.ndarray) -> np.ndarray:
    """
    Normalize traces (z-score per trace).
    
    Parameters
    ----------
    X : np.ndarray
        Trace array of shape [N, L]
        
    Returns
    -------
    np.ndarray
        Normalized trace array
    """
    # Per-trace normalization
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    stds[stds == 0] = 1  # Avoid division by zero
    X_normalized = (X - means) / stds
    return X_normalized


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main pipeline: load, classify, train contrastive autoencoder, save results."""
    print("=" * 80)
    print("RGC Classification with Supervised Contrastive Autoencoder")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 1. Load and filter data
    # -------------------------------------------------------------------------
    df = load_and_filter_rgc(INPUT_PARQUET)
    
    # -------------------------------------------------------------------------
    # 2. Classify RGC subtypes
    # -------------------------------------------------------------------------
    df = classify_rgc_subtypes(df)
    
    # -------------------------------------------------------------------------
    # 3. Concatenate mean traces
    # -------------------------------------------------------------------------
    X, valid_indices = concatenate_mean_traces(df, MOVIE_COLUMNS)
    
    if len(X) == 0:
        print("\nERROR: No valid traces found. Exiting.")
        return
    
    # Filter DataFrame to valid indices only
    df_valid = df.loc[valid_indices].copy()
    print(f"\n  Filtered DataFrame: {len(df_valid)} units with valid traces")
    
    # Get labels for contrastive learning
    labels = df_valid["rgc_label"].values
    
    # -------------------------------------------------------------------------
    # 4. Preprocess traces
    # -------------------------------------------------------------------------
    print("\nPreprocessing traces...")
    X = replace_nan_in_traces(X)
    X = normalize_traces(X)
    print(f"  Final trace shape: {X.shape}")
    
    # -------------------------------------------------------------------------
    # 5. Train supervised contrastive autoencoder
    # -------------------------------------------------------------------------
    model, latents, X_tensor, device = train_contrastive_autoencoder(
        X,
        labels,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_epochs=NUM_EPOCHS,
        latent_dim=LATENT_DIM,
        projection_dim=PROJECTION_DIM,
        contrastive_weight=CONTRASTIVE_WEIGHT,
        temperature=TEMPERATURE,
        training_patience=TRAINING_PATIENCE,
        random_seed=RANDOM_SEED,
        batch_size=BATCH_SIZE,
        save_path=str(MODEL_SAVE_PATH),
    )
    
    # -------------------------------------------------------------------------
    # 6. Add latent codes to DataFrame
    # -------------------------------------------------------------------------
    print("\nAdding latent codes to DataFrame...")
    latents_np = latents.numpy()
    
    # Create latent columns DataFrame and concat (avoid fragmentation warning)
    latent_cols = {f"AE_latent_{i}": latents_np[:, i] for i in range(LATENT_DIM)}
    latent_df = pd.DataFrame(latent_cols, index=df_valid.index)
    df_valid = pd.concat([df_valid, latent_df], axis=1)
    
    # Add concatenated trace length info
    df_valid["concatenated_trace_length"] = X.shape[1]
    
    print(f"  Added {LATENT_DIM} latent columns (AE_latent_0 to AE_latent_{LATENT_DIM-1})")
    
    # -------------------------------------------------------------------------
    # 7. Save output
    # -------------------------------------------------------------------------
    print(f"\nSaving to: {OUTPUT_PARQUET}")
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_valid.to_parquet(OUTPUT_PARQUET)
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n  Input units: {len(df)}")
    print(f"  Output units (with valid traces): {len(df_valid)}")
    print(f"  Latent dimension: {LATENT_DIM}")
    print(f"  Concatenated trace length: {X.shape[1]}")
    print(f"  Contrastive weight (λ): {CONTRASTIVE_WEIGHT}")
    print(f"  Temperature (τ): {TEMPERATURE}")
    
    print("\n  Subtype distribution in output:")
    subtype_counts = df_valid["rgc_subtype"].value_counts()
    for subtype, count in subtype_counts.items():
        pct = 100 * count / len(df_valid)
        print(f"    {subtype}: {count} ({pct:.1f}%)")
    
    print(f"\n  Model saved to: {MODEL_SAVE_PATH}")
    print(f"  DataFrame saved to: {OUTPUT_PARQUET}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
