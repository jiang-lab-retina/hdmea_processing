"""
Run Clustering with Pre-Optimized Parameters.

This script uses the saved OPTIMIZED_PARAMS from config.py directly,
skipping the hyperparameter optimization step.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pickle
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .config import (
    OUTPUT_DIR,
    PLOTS_DIR,
    SUBGROUPS,
    OPTIMIZED_PARAMS,
    EXPECTED_K_RANGES,
    NUM_EPOCHS,
    TRAINING_PATIENCE,
    RANDOM_SEED,
)
from .data_loader import load_subgroup_data


def get_device():
    """Get the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlexibleConvAE(nn.Module):
    """Flexible Convolutional Autoencoder with configurable architecture."""
    
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
        in_channels = 1
        current_length = input_length
        
        for i in range(n_conv_layers):
            out_channels = base_channels * (2 ** i)
            encoder_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels
            current_length = (current_length + 2 * 2 - 5) // 2 + 1
        
        self.encoder_conv = nn.Sequential(*encoder_layers)
        self.flat_size = in_channels * current_length
        
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_size, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2),
        )
        
        # Build decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, self.flat_size),
            nn.BatchNorm1d(self.flat_size),
            nn.LeakyReLU(0.2),
        )
        
        decoder_layers = []
        for i in range(n_conv_layers - 1, -1, -1):
            out_channels = base_channels * (2 ** (i - 1)) if i > 0 else 1
            in_ch = base_channels * (2 ** i)
            
            decoder_layers.append(
                nn.ConvTranspose1d(in_ch, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
            )
            if i > 0:
                decoder_layers.extend([
                    nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                ])
        
        self.decoder_conv = nn.Sequential(*decoder_layers)
        self.final_channels = in_channels
        self.compressed_length = current_length
    
    def encode(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.encoder_conv(x)
        return self.encoder_fc(h)
    
    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, self.final_channels, self.compressed_length)
        out = self.decoder_conv(h)
        # Adjust output size
        if out.size(2) > self.input_length:
            out = out[:, :, :self.input_length]
        elif out.size(2) < self.input_length:
            out = nn.functional.pad(out, (0, self.input_length - out.size(2)))
        return out.squeeze(1)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def train_model(model, X, params, device, verbose=True):
    """Train the autoencoder model."""
    model = model.to(device)
    
    # Prepare data
    X_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(X_tensor)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    batch_size = params.get("batch_size", 64)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params.get("weight_decay", 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon, _ = model(batch)
                loss = criterion(recon, batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= TRAINING_PATIENCE:
                if verbose:
                    print(f"      Early stopping at epoch {epoch + 1}")
                break
        
        if verbose and (epoch + 1) % 50 == 0:
            print(f"      Epoch {epoch + 1}/{NUM_EPOCHS}: train={train_loss:.6f}, val={val_loss:.6f}")
    
    if verbose:
        print(f"    [OK] Final val loss: {best_val_loss:.6f}")
    
    return model, best_val_loss


def cluster_latents(latents, k_range, subgroup, verbose=True):
    """Cluster latent representations using K-Means."""
    k_min, k_max = k_range
    
    if verbose:
        print(f"    Fitting K-Means with k in [{k_min}, {k_max}]")
    
    best_k = k_min
    best_score = -1
    best_labels = None
    
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(latents)
        
        if len(np.unique(labels)) < 2:
            continue
        
        score = silhouette_score(latents, labels)
        
        if verbose:
            print(f"      k={k}: silhouette={score:.3f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
    
    if verbose:
        print(f"    [OK] Optimal k={best_k} (silhouette={best_score:.3f})")
    
    return best_labels, best_k, best_score


def compute_metrics(latents, labels):
    """Compute clustering metrics."""
    return {
        "silhouette": silhouette_score(latents, labels),
        "calinski_harabasz": calinski_harabasz_score(latents, labels),
        "davies_bouldin": davies_bouldin_score(latents, labels),
    }


def process_subgroup(subgroup, X, params, device, verbose=True):
    """Process a single subgroup with saved parameters."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {subgroup} ({len(X)} units)")
        print(f"  Using saved params: latent={params['latent_dim']}, "
              f"lr={params['learning_rate']:.2e}, layers={params['n_conv_layers']}")
        print(f"{'='*60}")
    
    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Build model
    model = FlexibleConvAE(
        input_length=X.shape[1],
        latent_dim=params["latent_dim"],
        n_conv_layers=params["n_conv_layers"],
        base_channels=params["base_channels"],
        dropout=params["dropout"],
    )
    
    # Train
    model, val_loss = train_model(model, X_scaled, params, device, verbose)
    
    # Extract latents
    model.eval()
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    with torch.no_grad():
        latents = model.encode(X_tensor).cpu().numpy()
    
    # Cluster
    k_range = EXPECTED_K_RANGES.get(subgroup, (5, 15))
    labels, optimal_k, silhouette = cluster_latents(latents, k_range, subgroup, verbose)
    
    # Compute metrics
    metrics = compute_metrics(latents, labels)
    metrics["n_samples"] = len(X)
    metrics["n_clusters"] = optimal_k
    
    if verbose:
        # Cluster distribution
        unique, counts = np.unique(labels, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        print(f"    Clusters: {dist}")
        print(f"    Silhouette: {metrics['silhouette']:.3f}")
    
    return {
        "latents": latents,
        "labels": labels,
        "optimal_k": optimal_k,
        "metrics": metrics,
        "params": params,
    }


def main():
    """Main function."""
    print("=" * 70)
    print("Clustering with Saved Optimized Parameters")
    print("=" * 70)
    
    device = get_device()
    print(f"\nDevice: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    
    # Load data
    print("\n[1] Loading data...")
    subgroup_data = load_subgroup_data()
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each subgroup
    print("\n[2] Processing subgroups with saved parameters...")
    
    all_results = {}
    all_metrics = {}
    
    for subgroup in SUBGROUPS:
        if subgroup not in subgroup_data:
            print(f"  Skipping {subgroup}: no data")
            continue
        
        # subgroup_data returns (X, valid_indices, df_subset)
        X, valid_indices, df_subset = subgroup_data[subgroup]
        params = OPTIMIZED_PARAMS.get(subgroup)
        
        if params is None:
            print(f"  Skipping {subgroup}: no saved parameters")
            continue
        
        result = process_subgroup(subgroup, X, params, device)
        all_results[subgroup] = {"optimized_ae": result}
        all_metrics[subgroup] = {"optimized_ae": result["metrics"]}
    
    # Save results (merge with existing)
    print("\n[3] Saving results...")
    
    results_file = OUTPUT_DIR / "optimized_results.pkl"
    metrics_file = OUTPUT_DIR / "optimized_metrics_summary.json"
    
    # Load existing results and merge
    existing_results = {}
    existing_metrics = {}
    
    if results_file.exists():
        try:
            with open(results_file, "rb") as f:
                existing_results = pickle.load(f)
            print(f"    Loaded existing results for: {list(existing_results.keys())}")
        except Exception:
            existing_results = {}
    
    if metrics_file.exists():
        try:
            with open(metrics_file, "r") as f:
                existing_metrics = json.load(f)
        except Exception:
            existing_metrics = {}
    
    # Merge new results with existing
    merged_results = existing_results.copy()
    merged_results.update(all_results)
    
    merged_metrics = existing_metrics.copy()
    merged_metrics.update(all_metrics)
    
    with open(results_file, "wb") as f:
        pickle.dump(merged_results, f)
    
    with open(metrics_file, "w") as f:
        json.dump(merged_metrics, f, indent=2)
    
    print(f"    [OK] Results now include: {list(merged_results.keys())}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Subgroup':<10} {'k (expected)':<14} {'k (found)':<12} {'Silhouette':<12} {'CH Index':<10}")
    print("-" * 60)
    
    for subgroup in SUBGROUPS:
        if subgroup in all_metrics:
            m = all_metrics[subgroup]["optimized_ae"]
            k_exp = EXPECTED_K_RANGES.get(subgroup, (0, 0))
            k_found = m.get("n_clusters", 0)
            sil = m.get("silhouette", 0)
            ch = m.get("calinski_harabasz", 0)
            in_range = "OK" if k_exp[0] <= k_found <= k_exp[1] else "X"
            print(f"{subgroup:<10} {k_exp[0]}-{k_exp[1]:<11} {k_found:<4} {in_range:<6} {sil:<12.3f} {ch:<10.1f}")
    
    print("\n" + "=" * 70)
    print("Done! Run validation/visualize_optimized.py to generate plots.")
    print("=" * 70)


if __name__ == "__main__":
    main()

