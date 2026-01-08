"""
Run Clustering with Optimized Hyperparameters.

This script:
1. Runs hyperparameter optimization (or loads existing results)
2. Trains models with optimized hyperparameters
3. Clusters using expected k ranges per subgroup
4. Saves results and generates visualizations

Usage:
    python -m dataframe_phase.classification.subgroup_clustering.run_optimized
    python -m dataframe_phase.classification.subgroup_clustering.run_optimized --quick
    python -m dataframe_phase.classification.subgroup_clustering.run_optimized --skip-optimization
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Ensure proper encoding for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from .config import (
    SUBGROUPS,
    OUTPUT_DIR,
    MODELS_DIR,
    OPTUNA_DIR,
    PLOTS_DIR,
    APPROACH_NAMES,
    EXPECTED_K_RANGES,
    RANDOM_SEED,
    NUM_EPOCHS,
    TRAINING_PATIENCE,
    HP_QUICK_TRIALS,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT,
    CONTRASTIVE_WEIGHT,
    CONTRASTIVE_TEMPERATURE,
    PROJECTION_DIM,
)
from .data_loader import load_subgroup_data
from .models import FlexibleConvAE, ConvVAE, DECModel, ContrastiveAutoencoder, get_device
from .trainers import train_contrastive_ae
from .clustering import (
    fit_clustering_with_expected_k,
    compute_cluster_metrics,
    get_cluster_summary,
)
from .hyperparameter_optimization import (
    optimize_subgroup,
    load_best_params,
    optimize_all_subgroups,
)


def train_with_params(
    X: np.ndarray,
    params: Dict[str, Any],
    subgroup: str,
    num_epochs: int = NUM_EPOCHS,
    patience: int = TRAINING_PATIENCE,
    verbose: bool = True,
) -> Tuple[nn.Module, np.ndarray]:
    """
    Train autoencoder with specific hyperparameters.
    
    Parameters
    ----------
    X : np.ndarray
        Input data [N, L]
    params : dict
        Hyperparameters from optimization
    subgroup : str
        Subgroup name
    num_epochs : int
        Number of training epochs
    patience : int
        Early stopping patience
    verbose : bool
        Print progress
        
    Returns
    -------
    tuple
        (trained_model, latent_codes)
    """
    device = get_device()
    input_length = X.shape[1]
    
    # Extract params with defaults
    latent_dim = params.get("latent_dim", 64)
    learning_rate = params.get("learning_rate", 1e-4)
    weight_decay = params.get("weight_decay", 1e-5)
    batch_size = params.get("batch_size", 64)
    n_conv_layers = params.get("n_conv_layers", 3)
    base_channels = params.get("base_channels", 32)
    dropout = params.get("dropout", 0.1)
    
    if verbose:
        print(f"    Training with: latent={latent_dim}, lr={learning_rate:.2e}, "
              f"layers={n_conv_layers}, channels={base_channels}")
    
    # Create model
    model = FlexibleConvAE(
        input_length=input_length,
        latent_dim=latent_dim,
        n_conv_layers=n_conv_layers,
        base_channels=base_channels,
        dropout=dropout,
    ).to(device)
    
    # Prepare data
    X_tensor = torch.from_numpy(X).float().unsqueeze(1)
    dataset = TensorDataset(X_tensor)
    
    n_val = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) > batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            x_recon, z = model(xb)
            loss = criterion(x_recon, xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= n_train
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                x_recon, z = model(xb)
                val_loss += criterion(x_recon, xb).item() * xb.size(0)
        val_loss /= n_val
        
        if verbose and (epoch + 1) % 50 == 0:
            print(f"      Epoch {epoch+1}/{num_epochs}: train={train_loss:.6f}, val={val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"      Early stopping at epoch {epoch+1}")
                break
    
    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Get latents
    model.eval()
    with torch.no_grad():
        latents = model.encode(X_tensor.to(device)).cpu().numpy()
    
    # Save model
    model_path = MODELS_DIR / f"{subgroup}_optimized_ae.pth"
    torch.save({
        "state_dict": model.state_dict(),
        "params": params,
        "input_length": input_length,
    }, model_path)
    
    if verbose:
        print(f"    ✓ Final val loss: {best_val_loss:.6f}")
    
    return model, latents


def run_optimized_approach(
    X: np.ndarray,
    subgroup: str,
    params: Dict[str, Any],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run optimized AE + clustering for a subgroup.
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    subgroup : str
        Subgroup name
    params : dict
        Optimized hyperparameters
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Results including latents, labels, metrics
    """
    if verbose:
        print(f"\n  [Optimized AE + K-Means]")
    
    # Train with optimized params
    model, latents = train_with_params(X, params, subgroup, verbose=verbose)
    
    # Cluster within expected k range
    labels, optimal_k, _, cluster_results = fit_clustering_with_expected_k(
        latents,
        subgroup=subgroup,
        method="kmeans",
        verbose=verbose,
    )
    
    # Compute metrics
    metrics = compute_cluster_metrics(latents, labels)
    metrics["optimal_k"] = optimal_k
    
    if verbose:
        k_min, k_max = EXPECTED_K_RANGES.get(subgroup, (2, 15))
        print(f"    Clusters (expected {k_min}-{k_max}): {get_cluster_summary(labels)}")
        print(f"    Silhouette: {metrics['silhouette']:.3f}")
    
    return {
        "approach": "optimized_ae",
        "latents": latents,
        "labels": labels,
        "optimal_k": optimal_k,
        "metrics": metrics,
        "params": params,
        "cluster_results": cluster_results,
    }


def run_contrastive_approach(
    X: np.ndarray,
    subgroup: str,
    params: Dict[str, Any],
    initial_labels: np.ndarray,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run Contrastive AE + clustering for a subgroup.
    
    This approach:
    1. Uses initial cluster labels (from standard AE) to train contrastive AE
    2. The contrastive loss pulls same-cluster samples together 
       and pushes different-cluster samples apart
    3. Re-clusters the improved latent space
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    subgroup : str
        Subgroup name
    params : dict
        Hyperparameters (from optimization)
    initial_labels : np.ndarray
        Initial cluster labels for contrastive learning
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Results including latents, labels, metrics
    """
    if verbose:
        print(f"\n  [Contrastive AE + K-Means]")
    
    # Extract params
    latent_dim = params.get("latent_dim", 64)
    learning_rate = params.get("learning_rate", 1e-4)
    batch_size = params.get("batch_size", 64)
    n_conv_layers = params.get("n_conv_layers", 3)
    base_channels = params.get("base_channels", 32)
    dropout = params.get("dropout", 0.1)
    
    # Use larger batch size for contrastive learning (more negatives)
    contrastive_batch_size = min(batch_size * 4, 256)
    
    if verbose:
        print(f"    Using initial labels from {len(np.unique(initial_labels))} clusters")
        print(f"    Contrastive weight: {CONTRASTIVE_WEIGHT}, Temperature: {CONTRASTIVE_TEMPERATURE}")
    
    # Train contrastive autoencoder
    model, latents = train_contrastive_ae(
        X=X,
        labels=initial_labels,
        latent_dim=latent_dim,
        projection_dim=PROJECTION_DIM,
        contrastive_weight=CONTRASTIVE_WEIGHT,
        temperature=CONTRASTIVE_TEMPERATURE,
        num_epochs=NUM_EPOCHS,
        batch_size=contrastive_batch_size,
        learning_rate=learning_rate,
        patience=TRAINING_PATIENCE,
        n_conv_layers=n_conv_layers,
        base_channels=base_channels,
        dropout=dropout,
        verbose=verbose,
        save_path=MODELS_DIR / f"{subgroup}_contrastive_ae.pth",
    )
    
    # Re-cluster on the improved latent space
    labels, optimal_k, _, cluster_results = fit_clustering_with_expected_k(
        latents,
        subgroup=subgroup,
        method="kmeans",
        verbose=verbose,
    )
    
    # Compute metrics
    metrics = compute_cluster_metrics(latents, labels)
    metrics["optimal_k"] = optimal_k
    
    if verbose:
        k_min, k_max = EXPECTED_K_RANGES.get(subgroup, (2, 15))
        print(f"    Clusters (expected {k_min}-{k_max}): {get_cluster_summary(labels)}")
        print(f"    Silhouette: {metrics['silhouette']:.3f}")
    
    return {
        "approach": "contrastive_ae",
        "latents": latents,
        "labels": labels,
        "optimal_k": optimal_k,
        "metrics": metrics,
        "params": {**params, "contrastive_weight": CONTRASTIVE_WEIGHT, "temperature": CONTRASTIVE_TEMPERATURE},
        "cluster_results": cluster_results,
        "initial_labels": initial_labels,
    }


def run_all_for_subgroup(
    X: np.ndarray,
    subgroup: str,
    params: Dict[str, Any],
    verbose: bool = True,
    include_contrastive: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Run all approaches for a single subgroup with optimized parameters.
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    subgroup : str
        Subgroup name
    params : dict
        Hyperparameters
    verbose : bool
        Print progress
    include_contrastive : bool
        Whether to also run contrastive approach
    """
    results = {}
    
    # Primary: Optimized AE
    results["optimized_ae"] = run_optimized_approach(X, subgroup, params, verbose)
    
    # Secondary: Contrastive AE (using initial labels from optimized AE)
    if include_contrastive:
        initial_labels = results["optimized_ae"]["labels"]
        results["contrastive_ae"] = run_contrastive_approach(
            X, subgroup, params, initial_labels, verbose
        )
    
    return results


def save_results(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    subgroup_data: Dict,
    output_dir: Path,
):
    """Save all results to disk, merging with existing results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "optimized_results.pkl"
    metrics_file = output_dir / "optimized_metrics_summary.json"
    
    # Load existing results and merge (for single-subgroup runs)
    existing_results = {}
    existing_metrics = {}
    
    if results_file.exists():
        try:
            with open(results_file, "rb") as f:
                existing_results = pickle.load(f)
            print(f"  Loaded existing results for: {list(existing_results.keys())}")
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
    
    # Build metrics summary
    metrics_summary = existing_metrics.copy()
    for subgroup, approaches in all_results.items():
        metrics_summary[subgroup] = {}
        for approach, results in approaches.items():
            metrics_summary[subgroup][approach] = {
                "optimal_k": int(results["optimal_k"]),
                "n_samples": len(results["labels"]),
                "silhouette": float(results["metrics"]["silhouette"]) if not np.isnan(results["metrics"]["silhouette"]) else None,
                "calinski_harabasz": float(results["metrics"]["calinski_harabasz"]) if not np.isnan(results["metrics"]["calinski_harabasz"]) else None,
                "davies_bouldin": float(results["metrics"]["davies_bouldin"]) if not np.isnan(results["metrics"]["davies_bouldin"]) else None,
            }
    
    with open(metrics_file, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    
    # Save merged results
    with open(results_file, "wb") as f:
        pickle.dump(merged_results, f)
    
    print(f"  Results now include: {list(merged_results.keys())}")
    
    # Save per-subgroup parquet
    for subgroup, approaches in all_results.items():
        _, indices, df_subset = subgroup_data[subgroup]
        
        for approach, results in approaches.items():
            df_out = df_subset.copy()
            df_out["cluster"] = results["labels"]
            
            latents = results["latents"]
            latent_cols = {f"latent_{i}": latents[:, i] for i in range(latents.shape[1])}
            df_out = pd.concat([df_out, pd.DataFrame(latent_cols, index=df_out.index)], axis=1)
            
            out_path = output_dir / f"{subgroup}_optimized_results.parquet"
            df_out.to_parquet(out_path)
    
    print(f"\n✓ Results saved to: {output_dir}")


def generate_plots(output_dir: Path, plots_dir: Path):
    """Generate comparison plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Import visualization tools
    try:
        from .validation.compare_results import main as compare_main
        compare_main()
    except ImportError:
        print("  (Skipping visualization - run compare_results.py separately)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run optimized subgroup clustering")
    parser.add_argument("--quick", action="store_true", help="Use fewer optimization trials")
    parser.add_argument("--skip-optimization", action="store_true", help="Use existing optimized params")
    parser.add_argument("--subgroup", type=str, help="Run for specific subgroup only")
    parser.add_argument("--no-contrastive", action="store_true", help="Skip contrastive AE approach")
    parser.add_argument("--contrastive-only", action="store_true", help="Run only contrastive AE (requires existing optimized results)")
    args = parser.parse_args()
    
    print("=" * 80)
    print("Subgroup Clustering - Optimized Pipeline")
    print("=" * 80)
    
    # Setup directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1] Loading data...")
    subgroup_data = load_subgroup_data()
    
    for sg, (X, _, _) in subgroup_data.items():
        k_range = EXPECTED_K_RANGES.get(sg, (2, 15))
        print(f"    {sg}: {len(X)} samples, expected k: {k_range[0]}-{k_range[1]}")
    
    # Filter to specific subgroup if requested
    if args.subgroup:
        if args.subgroup not in subgroup_data:
            print(f"Error: Unknown subgroup '{args.subgroup}'")
            return
        subgroup_data = {args.subgroup: subgroup_data[args.subgroup]}
    
    # Hyperparameter optimization
    optimized_params = {}
    
    if args.skip_optimization:
        print("\n[2] Loading existing optimized parameters...")
        for subgroup in subgroup_data.keys():
            params = load_best_params(subgroup)
            if params:
                optimized_params[subgroup] = params["params"]
                print(f"    {subgroup}: loaded (silhouette={params.get('silhouette', 'N/A'):.3f})")
            else:
                print(f"    {subgroup}: not found, will use defaults")
                optimized_params[subgroup] = {}
    else:
        print("\n[2] Running hyperparameter optimization...")
        n_trials = HP_QUICK_TRIALS if args.quick else OPTUNA_N_TRIALS
        timeout = 1800 if args.quick else OPTUNA_TIMEOUT  # 30 min for quick, 2h for full
        
        for subgroup, (X, _, _) in subgroup_data.items():
            result = optimize_subgroup(
                X, subgroup,
                n_trials=n_trials,
                timeout=timeout,
                verbose=True,
            )
            optimized_params[subgroup] = result["params"]
    
    # Run optimized clustering
    print("\n[3] Training with optimized parameters...")
    all_results = {}
    
    include_contrastive = not args.no_contrastive
    
    # Handle contrastive-only mode
    if args.contrastive_only:
        print("\n[3] Running contrastive approach only (loading existing results)...")
        # Load existing results
        results_file = OUTPUT_DIR / "optimized_results.pkl"
        if results_file.exists():
            with open(results_file, "rb") as f:
                existing_results = pickle.load(f)
        else:
            print("ERROR: No existing results found. Run without --contrastive-only first.")
            return
        
        for subgroup, (X, indices, df_subset) in subgroup_data.items():
            print("\n" + "=" * 60)
            print(f"Processing: {subgroup} ({len(X)} units) - Contrastive Only")
            print("=" * 60)
            
            params = optimized_params.get(subgroup, {})
            
            # Get initial labels from existing results
            if subgroup in existing_results and "optimized_ae" in existing_results[subgroup]:
                initial_labels = existing_results[subgroup]["optimized_ae"]["labels"]
            else:
                print(f"  WARNING: No existing optimized_ae results for {subgroup}, running full pipeline")
                results = run_all_for_subgroup(X, subgroup, params, verbose=True, include_contrastive=True)
                all_results[subgroup] = results
                continue
            
            # Run contrastive approach only
            contrastive_result = run_contrastive_approach(X, subgroup, params, initial_labels, verbose=True)
            
            # Merge with existing results
            all_results[subgroup] = existing_results.get(subgroup, {}).copy()
            all_results[subgroup]["contrastive_ae"] = contrastive_result
    else:
        print("\n[3] Training with optimized parameters...")
        
        for subgroup, (X, indices, df_subset) in subgroup_data.items():
            print("\n" + "=" * 60)
            print(f"Processing: {subgroup} ({len(X)} units)")
            print("=" * 60)
            
            params = optimized_params.get(subgroup, {})
            results = run_all_for_subgroup(X, subgroup, params, verbose=True, include_contrastive=include_contrastive)
            all_results[subgroup] = results
    
    # Save results
    print("\n[4] Saving results...")
    save_results(all_results, subgroup_data, OUTPUT_DIR)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: Optimized Clustering Results")
    print("=" * 80)
    
    print(f"\n{'Subgroup':<10} {'Approach':<18} {'k (expected)':<14} {'k':<5} {'Silhouette':<12} {'CH Index':<12}")
    print("-" * 75)
    
    for subgroup, approaches in all_results.items():
        for approach, results in approaches.items():
            m = results["metrics"]
            k_min, k_max = EXPECTED_K_RANGES.get(subgroup, (2, 15))
            expected = f"{k_min}-{k_max}"
            found = results["optimal_k"]
            sil = f"{m['silhouette']:.3f}" if not np.isnan(m['silhouette']) else "N/A"
            ch = f"{m['calinski_harabasz']:.1f}" if not np.isnan(m['calinski_harabasz']) else "N/A"
            
            # Check if in expected range
            in_range = "[OK]" if k_min <= found <= k_max else "[X]"
            
            approach_short = "OptimizedAE" if approach == "optimized_ae" else "ContrastiveAE"
            print(f"{subgroup:<10} {approach_short:<18} {expected:<14} {found:<5} {sil:<12} {ch:<12}")
    
    print("\n" + "=" * 80)
    print("Done! Run validation/compare_results.py to visualize clusters.")
    print("=" * 80)


if __name__ == "__main__":
    main()

