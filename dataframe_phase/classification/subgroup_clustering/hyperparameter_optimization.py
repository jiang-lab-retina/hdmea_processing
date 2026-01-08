"""
Hyperparameter Optimization using Optuna.

Fast Bayesian optimization to find best hyperparameters for each subgroup.
Objective: maximize silhouette score within expected cluster range.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import warnings
warnings.filterwarnings("ignore")

from .config import (
    EXPECTED_K_RANGES,
    HP_SEARCH_SPACE,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT,
    OPTUNA_N_STARTUP_TRIALS,
    OPTUNA_DIR,
    RANDOM_SEED,
    NUM_EPOCHS,
    TRAINING_PATIENCE,
)


def get_device():
    """Get the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlexibleConvAE(nn.Module):
    """
    Flexible Convolutional Autoencoder with configurable architecture.
    
    Parameters
    ----------
    input_length : int
        Length of input traces
    latent_dim : int
        Dimension of latent space
    n_conv_layers : int
        Number of convolutional layers (2-4)
    base_channels : int
        Base number of channels (doubles each layer)
    dropout : float
        Dropout rate
    """
    
    def __init__(
        self,
        input_length: int,
        latent_dim: int = 64,
        n_conv_layers: int = 3,
        base_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim
        
        # Build encoder
        encoder_layers = []
        in_ch = 1
        current_length = input_length
        
        for i in range(n_conv_layers):
            out_ch = base_channels * (2 ** i)
            encoder_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            if dropout > 0 and i < n_conv_layers - 1:
                encoder_layers.append(nn.Dropout(dropout))
            in_ch = out_ch
            current_length = (current_length + 1) // 2
        
        self.encoder_conv = nn.Sequential(*encoder_layers)
        self.flat_dim = in_ch * current_length
        self.final_channels = in_ch
        self.final_length = current_length
        
        # Bottleneck
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.flat_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(latent_dim * 2, self.flat_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Build decoder (mirror of encoder)
        decoder_layers = []
        channels = [base_channels * (2 ** i) for i in range(n_conv_layers)]
        channels = channels[::-1]  # Reverse
        
        for i in range(n_conv_layers):
            in_ch = channels[i] if i == 0 else channels[i-1]
            out_ch = channels[i+1] if i < n_conv_layers - 1 else 1
            
            if i < n_conv_layers - 1:
                decoder_layers.extend([
                    nn.ConvTranspose1d(channels[i], channels[i+1], kernel_size=5, stride=2, padding=2, output_padding=1),
                    nn.BatchNorm1d(channels[i+1]),
                    nn.LeakyReLU(0.2, inplace=True),
                ])
            else:
                decoder_layers.append(
                    nn.ConvTranspose1d(channels[i], 1, kernel_size=5, stride=2, padding=2, output_padding=1)
                )
        
        self.decoder_conv = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode input to latent space."""
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        z = self.encoder_fc(h)
        return z
    
    def decode(self, z):
        """Decode latent to reconstruction."""
        h = self.decoder_fc(z)
        h = h.view(h.size(0), self.final_channels, self.final_length)
        x_recon = self.decoder_conv(h)
        # Adjust length if needed
        if x_recon.size(2) != self.input_length:
            x_recon = torch.nn.functional.interpolate(
                x_recon, size=self.input_length, mode='linear', align_corners=False
            )
        return x_recon
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def train_model_for_trial(
    model: nn.Module,
    X: np.ndarray,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    num_epochs: int = 100,
    patience: int = 15,
    trial: Optional[optuna.Trial] = None,
) -> Tuple[np.ndarray, float]:
    """
    Train model for a single Optuna trial.
    
    Returns latents and final validation loss.
    """
    device = get_device()
    model = model.to(device)
    
    # Prepare data
    X_tensor = torch.from_numpy(X).float().unsqueeze(1)  # [N, 1, L]
    dataset = TensorDataset(X_tensor)
    
    # Train/val split
    n_val = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            x_recon, z = model(xb)
            loss = criterion(x_recon, xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                x_recon, z = model(xb)
                val_loss += criterion(x_recon, xb).item() * xb.size(0)
        val_loss /= len(val_ds)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        
        # Optuna pruning
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    # Get latents
    model.eval()
    with torch.no_grad():
        latents = model.encode(X_tensor.to(device)).cpu().numpy()
    
    return latents, best_val_loss


def evaluate_clustering(
    latents: np.ndarray,
    k_min: int,
    k_max: int,
) -> Tuple[float, int, np.ndarray]:
    """
    Evaluate clustering quality within expected k range.
    
    Returns best silhouette score, optimal k, and labels.
    """
    best_sil = -1
    best_k = k_min
    best_labels = None
    
    for k in range(k_min, k_max + 1):
        if k >= len(latents):
            continue
        
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(latents)
        
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(latents, labels)
            if sil > best_sil:
                best_sil = sil
                best_k = k
                best_labels = labels
    
    if best_labels is None:
        kmeans = KMeans(n_clusters=k_min, random_state=RANDOM_SEED, n_init=10)
        best_labels = kmeans.fit_predict(latents)
        best_sil = 0.0
    
    return best_sil, best_k, best_labels


def create_objective(X: np.ndarray, subgroup: str):
    """
    Create Optuna objective function for a subgroup.
    
    Objective: maximize silhouette score within expected k range.
    """
    k_min, k_max = EXPECTED_K_RANGES.get(subgroup, (5, 15))
    input_length = X.shape[1]
    
    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        latent_dim = trial.suggest_int(
            "latent_dim", 
            HP_SEARCH_SPACE["latent_dim"][0], 
            HP_SEARCH_SPACE["latent_dim"][1],
            step=16
        )
        learning_rate = trial.suggest_float(
            "learning_rate",
            HP_SEARCH_SPACE["learning_rate"][0],
            HP_SEARCH_SPACE["learning_rate"][1],
            log=True
        )
        weight_decay = trial.suggest_float(
            "weight_decay",
            HP_SEARCH_SPACE["weight_decay"][0],
            HP_SEARCH_SPACE["weight_decay"][1],
            log=True
        )
        batch_size = trial.suggest_categorical(
            "batch_size",
            HP_SEARCH_SPACE["batch_size"]
        )
        n_conv_layers = trial.suggest_int(
            "n_conv_layers",
            HP_SEARCH_SPACE["n_conv_layers"][0],
            HP_SEARCH_SPACE["n_conv_layers"][1]
        )
        base_channels = trial.suggest_categorical(
            "base_channels",
            HP_SEARCH_SPACE["base_channels"]
        )
        dropout = trial.suggest_float(
            "dropout",
            HP_SEARCH_SPACE["dropout"][0],
            HP_SEARCH_SPACE["dropout"][1]
        )
        
        # Create model
        model = FlexibleConvAE(
            input_length=input_length,
            latent_dim=latent_dim,
            n_conv_layers=n_conv_layers,
            base_channels=base_channels,
            dropout=dropout,
        )
        
        # Train
        try:
            latents, val_loss = train_model_for_trial(
                model, X,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                num_epochs=100,  # Reduced for optimization
                patience=10,
                trial=trial,
            )
        except Exception as e:
            return -1.0  # Failed trial
        
        # Evaluate clustering
        sil_score, optimal_k, labels = evaluate_clustering(latents, k_min, k_max)
        
        # Store additional info
        trial.set_user_attr("optimal_k", optimal_k)
        trial.set_user_attr("val_loss", val_loss)
        trial.set_user_attr("n_samples", len(X))
        
        return sil_score
    
    return objective


def optimize_subgroup(
    X: np.ndarray,
    subgroup: str,
    n_trials: int = OPTUNA_N_TRIALS,
    timeout: int = OPTUNA_TIMEOUT,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run hyperparameter optimization for a single subgroup.
    
    Parameters
    ----------
    X : np.ndarray
        Input traces [N, L]
    subgroup : str
        Subgroup name
    n_trials : int
        Number of optimization trials
    timeout : int
        Maximum time in seconds
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Best hyperparameters and study results
    """
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check device
    device = get_device()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Optimizing hyperparameters for: {subgroup}")
        print(f"  Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
        print(f"  Samples: {len(X)}, Expected k: {EXPECTED_K_RANGES.get(subgroup)}")
        print(f"  Trials: {n_trials}, Timeout: {timeout//60} min")
        print(f"{'='*60}")
    
    # Create study
    sampler = TPESampler(
        seed=RANDOM_SEED,
        n_startup_trials=OPTUNA_N_STARTUP_TRIALS,
    )
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    study = optuna.create_study(
        direction="maximize",  # Maximize silhouette
        sampler=sampler,
        pruner=pruner,
        study_name=f"{subgroup}_optimization",
    )
    
    # Optimize
    objective = create_objective(X, subgroup)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=verbose,
        gc_after_trial=True,
    )
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial
    
    if verbose:
        print(f"\nBest silhouette: {best_value:.4f}")
        print(f"Best k: {best_trial.user_attrs.get('optimal_k', 'N/A')}")
        print(f"Best params: {best_params}")
    
    # Save study
    study_path = OPTUNA_DIR / f"{subgroup}_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    
    # Save best params
    params_path = OPTUNA_DIR / f"{subgroup}_best_params.json"
    with open(params_path, "w") as f:
        json.dump({
            "params": best_params,
            "silhouette": best_value,
            "optimal_k": best_trial.user_attrs.get("optimal_k"),
            "n_trials": len(study.trials),
        }, f, indent=2)
    
    return {
        "params": best_params,
        "silhouette": best_value,
        "optimal_k": best_trial.user_attrs.get("optimal_k"),
        "study": study,
    }


def load_best_params(subgroup: str) -> Optional[Dict[str, Any]]:
    """Load previously optimized parameters for a subgroup."""
    params_path = OPTUNA_DIR / f"{subgroup}_best_params.json"
    if params_path.exists():
        with open(params_path, "r") as f:
            return json.load(f)
    return None


def optimize_all_subgroups(
    subgroup_data: Dict[str, Tuple[np.ndarray, Any, Any]],
    n_trials: int = OPTUNA_N_TRIALS,
    timeout_per_subgroup: int = OPTUNA_TIMEOUT,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Optimize hyperparameters for all subgroups.
    
    Parameters
    ----------
    subgroup_data : dict
        Dict mapping subgroup name to (X, indices, df_subset)
    n_trials : int
        Trials per subgroup
    timeout_per_subgroup : int
        Timeout per subgroup in seconds
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Best params for each subgroup
    """
    all_results = {}
    
    for subgroup, (X, indices, df_subset) in subgroup_data.items():
        result = optimize_subgroup(
            X, subgroup,
            n_trials=n_trials,
            timeout=timeout_per_subgroup,
            verbose=verbose,
        )
        all_results[subgroup] = result
    
    # Save summary
    summary_path = OPTUNA_DIR / "optimization_summary.json"
    summary = {
        subgroup: {
            "params": res["params"],
            "silhouette": res["silhouette"],
            "optimal_k": res["optimal_k"],
        }
        for subgroup, res in all_results.items()
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        for subgroup, res in all_results.items():
            print(f"\n{subgroup}:")
            print(f"  Best silhouette: {res['silhouette']:.4f}")
            print(f"  Optimal k: {res['optimal_k']}")
    
    return all_results


if __name__ == "__main__":
    from .data_loader import load_subgroup_data
    
    # Load data
    print("Loading data...")
    subgroup_data = load_subgroup_data()
    
    # Run optimization
    results = optimize_all_subgroups(
        subgroup_data,
        n_trials=OPTUNA_N_TRIALS,
        verbose=True,
    )

