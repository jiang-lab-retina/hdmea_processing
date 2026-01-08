"""
Run All Approaches for Subgroup Clustering.

This script:
1. Loads data and splits by subgroup
2. Runs 3 approaches on each subgroup:
   - Approach 1: Standard AE + GMM
   - Approach 2: VAE + GMM
   - Approach 3: Deep Embedded Clustering (DEC)
3. Saves results for comparison

Usage:
    python run_all_approaches.py
"""

import os
# Disable parallelization to avoid psutil issues
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import pickle
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

from .config import (
    SUBGROUPS,
    LATENT_DIM,
    OUTPUT_DIR,
    MODELS_DIR,
    APPROACH_NAMES,
    GMM_K_MIN,
    GMM_K_MAX,
)
from .data_loader import load_subgroup_data
from .trainers import train_autoencoder, train_vae, train_dec
from .clustering import fit_gmm_auto_k, compute_cluster_metrics, get_cluster_summary


def run_approach_ae_gmm(
    X: np.ndarray,
    subgroup: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Approach 1: Standard Autoencoder + GMM clustering.
    
    Parameters
    ----------
    X : np.ndarray
        Input traces [N, input_length]
    subgroup : str
        Subgroup name
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Results including latents, labels, metrics
    """
    if verbose:
        print(f"\n  [Approach 1] Standard AE + GMM")
    
    # Train autoencoder
    model_path = MODELS_DIR / f"{subgroup}_ae.pth"
    model, latents = train_autoencoder(
        X, 
        latent_dim=LATENT_DIM, 
        verbose=verbose,
        save_path=model_path,
    )
    
    # Cluster with GMM
    labels, optimal_k, gmm, gmm_results = fit_gmm_auto_k(
        latents,
        k_min=GMM_K_MIN,
        k_max=min(GMM_K_MAX, len(X) // 10),  # Ensure enough samples per cluster
        verbose=verbose,
    )
    
    # Compute metrics
    metrics = compute_cluster_metrics(latents, labels)
    metrics["optimal_k"] = optimal_k
    
    if verbose:
        print(f"    Clusters: {get_cluster_summary(labels)}")
        print(f"    Silhouette: {metrics['silhouette']:.3f}")
    
    return {
        "approach": "ae_gmm",
        "latents": latents,
        "labels": labels,
        "optimal_k": optimal_k,
        "metrics": metrics,
        "gmm_results": gmm_results,
    }


def run_approach_vae_gmm(
    X: np.ndarray,
    subgroup: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Approach 2: Variational Autoencoder + GMM clustering.
    
    Parameters
    ----------
    X : np.ndarray
        Input traces [N, input_length]
    subgroup : str
        Subgroup name
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Results including latents, labels, metrics
    """
    if verbose:
        print(f"\n  [Approach 2] VAE + GMM")
    
    # Train VAE
    model_path = MODELS_DIR / f"{subgroup}_vae.pth"
    model, latents = train_vae(
        X, 
        latent_dim=LATENT_DIM, 
        verbose=verbose,
        save_path=model_path,
    )
    
    # Cluster with GMM
    labels, optimal_k, gmm, gmm_results = fit_gmm_auto_k(
        latents,
        k_min=GMM_K_MIN,
        k_max=min(GMM_K_MAX, len(X) // 10),
        verbose=verbose,
    )
    
    # Compute metrics
    metrics = compute_cluster_metrics(latents, labels)
    metrics["optimal_k"] = optimal_k
    
    if verbose:
        print(f"    Clusters: {get_cluster_summary(labels)}")
        print(f"    Silhouette: {metrics['silhouette']:.3f}")
    
    return {
        "approach": "vae_gmm",
        "latents": latents,
        "labels": labels,
        "optimal_k": optimal_k,
        "metrics": metrics,
        "gmm_results": gmm_results,
    }


def run_approach_dec(
    X: np.ndarray,
    subgroup: str,
    n_clusters: int = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Approach 3: Deep Embedded Clustering.
    
    Parameters
    ----------
    X : np.ndarray
        Input traces [N, input_length]
    subgroup : str
        Subgroup name
    n_clusters : int, optional
        Number of clusters. If None, estimate using GMM on pretrained AE.
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Results including latents, labels, metrics
    """
    if verbose:
        print(f"\n  [Approach 3] Deep Embedded Clustering (DEC)")
    
    # If n_clusters not specified, estimate it first
    if n_clusters is None:
        if verbose:
            print("    Estimating n_clusters from AE + GMM...")
        # Quick AE training to estimate k
        _, latents_init = train_autoencoder(X, latent_dim=LATENT_DIM, num_epochs=50, verbose=False)
        _, n_clusters, _, _ = fit_gmm_auto_k(latents_init, verbose=False)
        if verbose:
            print(f"    Estimated n_clusters = {n_clusters}")
    
    # Train DEC
    model_path = MODELS_DIR / f"{subgroup}_dec.pth"
    model, latents, labels = train_dec(
        X,
        n_clusters=n_clusters,
        latent_dim=LATENT_DIM,
        verbose=verbose,
        save_path=model_path,
    )
    
    # Compute metrics
    metrics = compute_cluster_metrics(latents, labels)
    metrics["optimal_k"] = n_clusters
    
    if verbose:
        print(f"    Clusters: {get_cluster_summary(labels)}")
        print(f"    Silhouette: {metrics['silhouette']:.3f}")
    
    return {
        "approach": "dec",
        "latents": latents,
        "labels": labels,
        "optimal_k": n_clusters,
        "metrics": metrics,
    }


def run_all_approaches_for_subgroup(
    X: np.ndarray,
    subgroup: str,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Run all 3 approaches for a single subgroup.
    
    Parameters
    ----------
    X : np.ndarray
        Input traces
    subgroup : str
        Subgroup name
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Results for all approaches
    """
    results = {}
    
    # Approach 1: AE + GMM
    results["ae_gmm"] = run_approach_ae_gmm(X, subgroup, verbose)
    
    # Approach 2: VAE + GMM
    results["vae_gmm"] = run_approach_vae_gmm(X, subgroup, verbose)
    
    # Approach 3: DEC (use k from AE+GMM as initialization)
    estimated_k = results["ae_gmm"]["optimal_k"]
    results["dec"] = run_approach_dec(X, subgroup, n_clusters=estimated_k, verbose=verbose)
    
    return results


def save_results(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    subgroup_data: Dict,
    output_dir: Path,
):
    """
    Save all results to disk.
    
    Parameters
    ----------
    all_results : dict
        Nested dict: subgroup -> approach -> results
    subgroup_data : dict
        Original subgroup data with indices
    output_dir : Path
        Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics summary as JSON
    metrics_summary = {}
    for subgroup, approaches in all_results.items():
        metrics_summary[subgroup] = {}
        for approach, results in approaches.items():
            metrics_summary[subgroup][approach] = {
                "optimal_k": int(results["optimal_k"]),
                "n_samples": len(results["labels"]),
                **{k: float(v) if not np.isnan(v) else None 
                   for k, v in results["metrics"].items() 
                   if k != "n_clusters"},
            }
    
    with open(output_dir / "metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    
    # Save full results as pickle
    with open(output_dir / "all_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    
    # Save cluster assignments as parquet for each subgroup/approach
    for subgroup, approaches in all_results.items():
        _, indices, df_subset = subgroup_data[subgroup]
        
        for approach, results in approaches.items():
            # Add cluster labels to dataframe
            df_out = df_subset.copy()
            df_out[f"cluster_{approach}"] = results["labels"]
            
            # Add latent codes
            latents = results["latents"]
            for i in range(latents.shape[1]):
                df_out[f"latent_{approach}_{i}"] = latents[:, i]
            
            # Save
            out_path = output_dir / f"{subgroup}_{approach}_results.parquet"
            df_out.to_parquet(out_path)
    
    print(f"\nResults saved to: {output_dir}")


def main():
    """Main function to run all approaches on all subgroups."""
    print("=" * 80)
    print("Subgroup Clustering Pipeline - Running All Approaches")
    print("=" * 80)
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    subgroup_data = load_subgroup_data()
    
    # Run all approaches for each subgroup
    all_results = {}
    
    for subgroup in SUBGROUPS:
        if subgroup not in subgroup_data:
            print(f"\nSkipping {subgroup} (no data)")
            continue
        
        X, indices, df_subset = subgroup_data[subgroup]
        
        print("\n" + "=" * 80)
        print(f"Processing: {subgroup} ({len(X)} units)")
        print("=" * 80)
        
        results = run_all_approaches_for_subgroup(X, subgroup, verbose=True)
        all_results[subgroup] = results
    
    # Save results
    save_results(all_results, subgroup_data, OUTPUT_DIR)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Clustering Results")
    print("=" * 80)
    
    print(f"\n{'Subgroup':<10} {'Approach':<20} {'k':<5} {'Silhouette':<12} {'CH Index':<12} {'DB Index':<12}")
    print("-" * 75)
    
    for subgroup, approaches in all_results.items():
        for approach, results in approaches.items():
            m = results["metrics"]
            sil = f"{m['silhouette']:.3f}" if not np.isnan(m['silhouette']) else "N/A"
            ch = f"{m['calinski_harabasz']:.1f}" if not np.isnan(m['calinski_harabasz']) else "N/A"
            db = f"{m['davies_bouldin']:.3f}" if not np.isnan(m['davies_bouldin']) else "N/A"
            
            print(f"{subgroup:<10} {APPROACH_NAMES[approach]:<20} {results['optimal_k']:<5} {sil:<12} {ch:<12} {db:<12}")
    
    print("\n" + "=" * 80)
    print("Done! Run validation/compare_results.py to generate visualizations.")
    print("=" * 80)


if __name__ == "__main__":
    main()

