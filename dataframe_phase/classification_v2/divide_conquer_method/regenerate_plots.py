"""
Regenerate plots from saved results without rerunning the full pipeline.

Loads embeddings, labels, and metrics from the results directory,
then calls the visualization functions.

Usage:
    python -m dataframe_phase.classification_v2.divide_conquer_method.regenerate_plots
    python -m dataframe_phase.classification_v2.divide_conquer_method.regenerate_plots --group ipRGC
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Support direct execution
_this_dir = Path(__file__).resolve().parent
_parent_dir = _this_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from divide_conquer_method import config
from divide_conquer_method.visualization import generate_all_plots
from divide_conquer_method.preprocessing import preprocess_all_segments, extract_iprgc_last_trial
from divide_conquer_method.grouping import assign_groups, filter_group
from divide_conquer_method.validation.iprgc_metrics import get_iprgc_labels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_artifacts_for_group(group: str) -> dict:
    """Load saved results and reconstruct the artifacts dict for plotting."""
    results_dir = config.RESULTS_DIR / group

    # Load embeddings
    emb_init_path = results_dir / "embeddings_initial.parquet"
    emb_dec_path = results_dir / "embeddings_dec_refined.parquet"
    cluster_path = results_dir / "cluster_assignments.parquet"
    k_sel_path = results_dir / "k_selection.json"
    iprgc_val_path = results_dir / "iprgc_validation.json"

    if not emb_dec_path.exists():
        raise FileNotFoundError(f"No saved embeddings at {emb_dec_path}")

    emb_init_df = pd.read_parquet(emb_init_path)
    emb_dec_df = pd.read_parquet(emb_dec_path)
    cluster_df = pd.read_parquet(cluster_path)

    # Extract embedding arrays
    z_cols = sorted([c for c in emb_init_df.columns if c.startswith('z_')])
    emb_init = emb_init_df[z_cols].values
    emb_dec = emb_dec_df[z_cols].values

    # Extract labels
    dec_labels = cluster_df['dec_cluster'].values
    gmm_labels = cluster_df['gmm_cluster'].values

    # Load k-selection
    with open(k_sel_path) as f:
        k_sel = json.load(f)
    k_range = k_sel.get('k_range', list(range(1, 21)))
    bic_values = np.array(k_sel.get('bic_values', []))
    k_selected = k_sel.get('k_selected', len(np.unique(dec_labels)))

    # Load ipRGC validation metrics
    with open(iprgc_val_path) as f:
        dec_metrics = json.load(f)

    artifacts = {
        'embeddings_initial': emb_init,
        'embeddings_dec': emb_dec,
        'gmm_labels': gmm_labels,
        'dec_labels': dec_labels,
        'bic_values': bic_values,
        'k_range': k_range,
        'k_selected': k_selected,
        'dec_metrics': dec_metrics,
    }

    return artifacts


def add_segments_and_iprgc(artifacts: dict, group: str, df: pd.DataFrame) -> dict:
    """Add segments and ipRGC labels by loading and preprocessing the group data."""
    group_df = filter_group(df, group)

    logger.info(f"Preprocessing traces for {group} ({len(group_df)} cells)...")
    segments, full_segments = preprocess_all_segments(group_df)
    iprgc_labels = get_iprgc_labels(group_df)

    artifacts['segments'] = segments
    artifacts['full_segments'] = full_segments
    artifacts['iprgc_labels'] = iprgc_labels
    return artifacts


def main():
    parser = argparse.ArgumentParser(description="Regenerate plots from saved results")
    parser.add_argument("--group", type=str, choices=["ipRGC", "DSGC", "OSGC", "Other"],
                        default=None, help="Single group to regenerate (default: all)")
    parser.add_argument("--skip-prototypes", action="store_true",
                        help="Skip prototype plots (faster, no data loading needed)")
    args = parser.parse_args()

    groups = [args.group] if args.group else config.GROUP_NAMES

    # Only load full data if we need segments (for prototypes)
    df = None
    if not args.skip_prototypes:
        logger.info(f"Loading data from {config.INPUT_PATH}")
        from divide_conquer_method.run_pipeline import load_and_filter_data
        df, _ = load_and_filter_data()
        df = assign_groups(df)

    for group in groups:
        logger.info(f"{'='*60}")
        logger.info(f"Regenerating plots for: {group}")
        logger.info(f"{'='*60}")

        try:
            artifacts = load_artifacts_for_group(group)

            if df is not None:
                artifacts = add_segments_and_iprgc(artifacts, group, df)

            plots_dir = config.PLOTS_DIR / group
            generate_all_plots(artifacts, group, plots_dir)
            plt.close('all')

            logger.info(f"Done: {group}")
        except Exception as e:
            logger.error(f"Failed for {group}: {e}")
            import traceback
            traceback.print_exc()

    logger.info("All plot regeneration complete.")


if __name__ == "__main__":
    main()
