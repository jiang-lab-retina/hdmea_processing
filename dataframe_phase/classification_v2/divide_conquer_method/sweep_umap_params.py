"""
Sweep UMAP parameters for the Other group umap_supervised_final plot.

Tests all combinations of specified parameter grids and saves each plot
with the parameter values noted in the filename and figure title.

Usage:
    python -m dataframe_phase.classification_v2.divide_conquer_method.sweep_umap_params
"""

import itertools
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import umap

# Support direct execution
_this_dir = Path(__file__).resolve().parent
_parent_dir = _this_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from divide_conquer_method import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Parameter grid ──────────────────────────────────────────────────────
PARAM_GRID = {
    'n_neighbors':        [15, 20, 25],
    'min_dist':           [0.1, 0.2, 0.4, 0.6],
    'spread':             [1, 2, 4],
    'repulsion_strength': [5, 10],
    'local_connectivity': [1, 5, 10],
    'target_weight':      [0.1, 0.2, 0.4, 0.8],
}

GROUP = "Other"


def load_embeddings_and_labels():
    """Load saved DEC embeddings and labels for the Other group."""
    results_dir = config.RESULTS_DIR / GROUP
    emb_dec_path = results_dir / "embeddings_dec_refined.parquet"
    cluster_path = results_dir / "cluster_assignments.parquet"

    if not emb_dec_path.exists():
        raise FileNotFoundError(f"No saved embeddings at {emb_dec_path}")

    emb_dec_df = pd.read_parquet(emb_dec_path)
    cluster_df = pd.read_parquet(cluster_path)

    z_cols = sorted([c for c in emb_dec_df.columns if c.startswith('z_')])
    embeddings = emb_dec_df[z_cols].values
    labels = cluster_df['dec_cluster'].values

    logger.info(f"Loaded {len(labels)} cells, {len(np.unique(labels))} clusters, "
                f"{embeddings.shape[1]}D embeddings")
    return embeddings, labels


def make_plot(embeddings, labels, params, output_path):
    """Run supervised UMAP with given params and save the plot."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(
            n_neighbors=params['n_neighbors'],
            min_dist=params['min_dist'],
            spread=params['spread'],
            metric="euclidean",
            random_state=42,
            target_weight=params['target_weight'],
            repulsion_strength=params['repulsion_strength'],
            local_connectivity=params['local_connectivity'],
        )
        coords = reducer.fit_transform(embeddings, y=labels)

    unique_labels = np.sort(np.unique(labels))
    n_clusters = len(unique_labels)
    cmap = plt.cm.get_cmap('tab20', max(n_clusters, 1))

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        n = mask.sum()
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(i)],
            label=f'C{lab} (n={n})',
            s=12,
            alpha=0.7,
        )

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)

    # Title with all parameter values
    title = (
        f"{GROUP}: Supervised Final (k={n_clusters})\n"
        f"nn={params['n_neighbors']}  md={params['min_dist']}  "
        f"sp={params['spread']}  rep={params['repulsion_strength']}  "
        f"lc={params['local_connectivity']}  tw={params['target_weight']}"
    )
    ax.set_title(title, fontsize=11)
    ax.legend(
        fontsize=7,
        markerscale=2,
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        frameon=True,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)


def main():
    embeddings, labels = load_embeddings_and_labels()

    output_dir = config.PLOTS_DIR / GROUP / "umap_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build all combinations
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))
    total = len(combos)

    logger.info(f"Starting sweep: {total} combinations → {output_dir}")

    for idx, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))

        # Build filename from params
        fname = (
            f"nn{params['n_neighbors']}_"
            f"md{params['min_dist']}_"
            f"sp{params['spread']}_"
            f"rep{params['repulsion_strength']}_"
            f"lc{params['local_connectivity']}_"
            f"tw{params['target_weight']}.png"
        )
        out_path = output_dir / fname

        # Skip if already generated (allows resuming)
        if out_path.exists():
            logger.info(f"[{idx}/{total}] SKIP (exists): {fname}")
            continue

        logger.info(f"[{idx}/{total}] {fname}")
        try:
            make_plot(embeddings, labels, params, out_path)
        except Exception as e:
            logger.error(f"  FAILED: {e}")

    logger.info(f"Sweep complete. {total} plots in {output_dir}")


if __name__ == "__main__":
    main()
