"""
Classify blocker experiment cells into known RGC subtypes.

Uses the trained autoencoder + DEC models from
dataframe_phase/classification_v2/divide_conquer_method/
to assign each RGC pair in compared_dataframe_v2.parquet to a known
cluster/subtype.

Column mapping strategy:
- Most data uses before_ columns (control condition)
- ipRGC-related data uses after_ columns (only available in after recordings)
- Green-blue trace uses before_green_blue_3s_3i_3x (high-intensity, 3 trials)
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Path setup -- add project root so classification pipeline can be imported
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataframe_phase.classification_v2.divide_conquer_method import config
from dataframe_phase.classification_v2.divide_conquer_method.grouping import (
    assign_groups,
)
from dataframe_phase.classification_v2.divide_conquer_method.preprocessing import (
    preprocess_all_segments,
    get_segment_lengths,
)
from dataframe_phase.classification_v2.divide_conquer_method.models.autoencoder import (
    MultiSegmentAutoencoder,
)
from dataframe_phase.classification_v2.divide_conquer_method.models.dec import IDEC

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
from compare_config import OUTPUT_DIR as _OUTPUT_DIR

DEFAULT_INPUT = _OUTPUT_DIR / "compared_dataframe_v2.parquet"
DEFAULT_OUTPUT = _OUTPUT_DIR / "compared_dataframe_v2_labeled.parquet"
DEFAULT_MODELS_DIR = (
    PROJECT_ROOT
    / "dataframe_phase"
    / "classification_v2"
    / "divide_conquer_method"
    / "models_saved"
)
DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT
    / "dataframe_phase"
    / "classification_v2"
    / "divide_conquer_method"
    / "results"
)

# Columns that should be sourced from the AFTER recording (ipRGC-related)
IPRGC_SOURCE_COLUMNS = {"iprgc_test", "iprgc_2hz_QI", "iprgc_20hz_QI"}

# Trace columns required by the classification pipeline
BAR_DIRECTIONS = ["000", "045", "090", "135", "180", "225", "270", "315"]

TRACE_COLUMNS = [
    "freq_section_0p5hz",
    "freq_section_1hz",
    "freq_section_2hz",
    "freq_section_4hz",
    "freq_section_10hz",
    "green_blue_3s_3i_3x",
    "sta_time_course",
    "iprgc_test",
    "step_up_5s_5i_b0_3x",
] + [f"corrected_moving_h_bar_s5_d8_3x_{d}" for d in BAR_DIRECTIONS]

# Metadata columns needed for filtering and group assignment
METADATA_COLUMNS = [
    "axon_type",
    "ds_p_value",
    "os_p_value",
    "step_up_QI",
    "iprgc_2hz_QI",
]

GROUP_NAMES = config.GROUP_NAMES  # ["ipRGC", "DSGC", "OSGC", "Other"]


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------
def build_standard_dataframe(df_blocker: pd.DataFrame) -> pd.DataFrame:
    """
    Build a standard-format DataFrame by mapping prefixed blocker columns
    to the un-prefixed column names expected by the classification pipeline.

    - Most columns sourced from before_ (control condition)
    - ipRGC-related columns sourced from after_
    """
    std_df = pd.DataFrame(index=df_blocker.index)

    all_needed = set(TRACE_COLUMNS + METADATA_COLUMNS)

    for col in sorted(all_needed):
        if col in IPRGC_SOURCE_COLUMNS:
            src = f"after_{col}"
        else:
            src = f"before_{col}"

        if src in df_blocker.columns:
            std_df[col] = df_blocker[src]
        else:
            logger.warning(f"Source column '{src}' not found -> '{col}' will be NaN")
            std_df[col] = np.nan

    logger.info(
        f"Built standard DataFrame: {len(std_df)} rows, "
        f"{len(std_df.columns)} columns"
    )
    return std_df


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def filter_rgc_cells(
    df: pd.DataFrame, qi_threshold: float = 0.5
) -> pd.DataFrame:
    """Filter to RGC cells passing quality thresholds."""
    initial = len(df)

    # RGC only
    rgc_mask = df["axon_type"].astype(str).str.lower() == "rgc"
    df = df[rgc_mask].copy()
    logger.info(f"  RGC filter: {initial} -> {len(df)}")

    # QI threshold
    if "step_up_QI" in df.columns:
        qi_mask = df["step_up_QI"] >= qi_threshold
        removed = (~qi_mask).sum()
        df = df[qi_mask].copy()
        if removed:
            logger.info(f"  QI >= {qi_threshold}: removed {removed}, {len(df)} left")

    # Drop NaN in required metadata
    for col in ["ds_p_value", "os_p_value"]:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count:
                df = df[df[col].notna()].copy()
                logger.info(f"  NaN in {col}: removed {nan_count}")

    # Ensure iprgc_2hz_QI exists (NaN is fine -- assign_groups handles it)
    if "iprgc_2hz_QI" not in df.columns:
        df["iprgc_2hz_QI"] = np.nan

    # Drop rows with None in any trace column
    for col in TRACE_COLUMNS:
        if col in df.columns:
            none_mask = df[col].apply(lambda x: x is None)
            none_count = none_mask.sum()
            if none_count:
                df = df[~none_mask].copy()
                logger.info(f"  None in '{col}': removed {none_count}")

    logger.info(f"  Final filtered RGC cells: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Model inference helpers
# ---------------------------------------------------------------------------
def get_k_for_group(results_dir: Path, group: str) -> int:
    """Read number of clusters from saved cluster assignments."""
    ca_path = results_dir / group / "cluster_assignments.parquet"
    if not ca_path.exists():
        raise FileNotFoundError(f"Cluster assignments not found: {ca_path}")
    ca = pd.read_parquet(ca_path)
    k = int(ca["dec_cluster"].nunique())
    return k


def adjust_segment_lengths(
    segments: dict[str, np.ndarray],
    expected_lengths: dict[str, int],
) -> dict[str, np.ndarray]:
    """
    Truncate or pad segments to match expected lengths from the trained model.

    This ensures compatibility when the blocker data produces slightly
    different trace lengths than the original training data.
    """
    adjusted = {}
    for name, arr in segments.items():
        if name not in expected_lengths:
            adjusted[name] = arr
            continue
        exp_len = expected_lengths[name]
        cur_len = arr.shape[1]
        if cur_len == exp_len:
            adjusted[name] = arr
        elif cur_len > exp_len:
            adjusted[name] = arr[:, :exp_len]
            logger.debug(
                f"  {name}: truncated {cur_len} -> {exp_len}"
            )
        else:
            pad_width = exp_len - cur_len
            adjusted[name] = np.pad(
                arr, ((0, 0), (0, pad_width)), mode="constant", constant_values=0
            )
            logger.debug(
                f"  {name}: padded {cur_len} -> {exp_len}"
            )
    return adjusted


def get_raw_embeddings(
    autoencoder: torch.nn.Module,
    segments: dict[str, np.ndarray],
    device: str,
    batch_size: int = 128,
) -> np.ndarray:
    """Extract raw (un-standardized) embeddings from the autoencoder."""
    autoencoder.eval()
    segment_tensors = {
        name: torch.tensor(arr, dtype=torch.float32)
        for name, arr in segments.items()
    }
    n_samples = next(iter(segment_tensors.values())).shape[0]
    all_emb = []

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = {
                name: tensor[start:end].to(device)
                for name, tensor in segment_tensors.items()
            }
            ae_out = autoencoder(batch)
            all_emb.append(ae_out["full_embedding"].cpu().numpy())

    return np.vstack(all_emb)


def recompute_standardization(
    idec: IDEC,
    segments: dict[str, np.ndarray],
    device: str,
) -> None:
    """
    Re-compute embedding_mean / embedding_std from the blocker data
    and update the IDEC buffers in-place.

    This corrects for the global domain shift between reference and
    blocker datasets so that the blocker embeddings are centered and
    scaled the same way the reference embeddings were during training.
    """
    raw_emb = get_raw_embeddings(idec.autoencoder, segments, device)
    new_mean = raw_emb.mean(axis=0)
    new_std = raw_emb.std(axis=0)

    old_mean = idec.embedding_mean.cpu().numpy()
    old_std = idec.embedding_std.cpu().numpy()

    logger.info(
        f"    Re-standardizing embeddings (domain adaptation):"
    )
    logger.info(
        f"      ref  mean range [{old_mean.min():.2f}, {old_mean.max():.2f}], "
        f"std range [{old_std.min():.2f}, {old_std.max():.2f}]"
    )
    logger.info(
        f"      blk  mean range [{new_mean.min():.2f}, {new_mean.max():.2f}], "
        f"std range [{new_std.min():.2f}, {new_std.max():.2f}]"
    )

    idec.embedding_mean.copy_(torch.tensor(new_mean, dtype=torch.float32))
    idec.embedding_std.copy_(torch.tensor(new_std, dtype=torch.float32))


def get_all_labels(
    idec: IDEC,
    segments: dict[str, np.ndarray],
    device: str,
    batch_size: int = 128,
) -> np.ndarray:
    """Run batched inference through IDEC to get hard cluster labels."""
    idec.eval()
    segment_tensors = {
        name: torch.tensor(arr, dtype=torch.float32)
        for name, arr in segments.items()
    }
    n_samples = next(iter(segment_tensors.values())).shape[0]
    all_labels = []

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = {
                name: tensor[start:end].to(device)
                for name, tensor in segment_tensors.items()
            }
            output = idec(batch)
            labels = output["q"].argmax(dim=1).cpu().numpy()
            all_labels.append(labels)

    return np.concatenate(all_labels)


def infer_segment_lengths_from_dec(state_dict: dict) -> dict[str, int]:
    """
    Infer the expected segment lengths from a DEC state dict by
    reading the decoder adjust.bias shapes.
    """
    lengths = {}
    for key, val in state_dict.items():
        if key.startswith("autoencoder.decoders.") and key.endswith(".adjust.bias"):
            seg_name = key.split(".")[2]
            lengths[seg_name] = val.shape[0]
    return lengths


def preprocess_no_downsample(group_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Preprocess segments at the original 60 Hz sampling rate (no downsampling).

    The trained DEC models expect 60 Hz data.  We temporarily override
    the config target rates to match the sampling rate, which effectively
    disables downsampling while keeping the low-pass filter.
    """
    saved_default = config.TARGET_RATE_DEFAULT
    saved_iprgc = config.TARGET_RATE_IPRGC
    try:
        config.TARGET_RATE_DEFAULT = config.SAMPLING_RATE  # 60 Hz -> no downsample
        config.TARGET_RATE_IPRGC = config.SAMPLING_RATE
        segments, _ = preprocess_all_segments(group_df)
    finally:
        config.TARGET_RATE_DEFAULT = saved_default
        config.TARGET_RATE_IPRGC = saved_iprgc
    return segments


def classify_group(
    group_df: pd.DataFrame,
    group_name: str,
    models_dir: Path,
    results_dir: Path,
    device: str,
) -> np.ndarray:
    """
    Classify cells in a single group using the trained DEC model.

    Returns an array of integer cluster labels (one per cell in group_df).
    """
    n_cells = len(group_df)
    logger.info(f"  Classifying {group_name} ({n_cells} cells)...")

    # Number of clusters for this group
    k = get_k_for_group(results_dir, group_name)
    logger.info(f"    k = {k} clusters")

    # Load DEC state dict to infer the correct architecture
    dec_path = models_dir / group_name / "dec_refined.pt"
    if not dec_path.exists():
        raise FileNotFoundError(f"DEC checkpoint not found: {dec_path}")
    dec_state = torch.load(dec_path, map_location=device, weights_only=False)

    # Infer segment lengths the model actually expects
    expected_lengths = infer_segment_lengths_from_dec(dec_state)
    logger.info(f"    DEC expected segment lengths: {expected_lengths}")

    # Preprocess traces at 60 Hz (no downsampling) to match training
    segments = preprocess_no_downsample(group_df)
    actual_lengths = get_segment_lengths(segments)
    logger.info(f"    Preprocessed (60 Hz) segments:  {actual_lengths}")

    # Adjust to match the exact lengths expected by the model
    segments = adjust_segment_lengths(segments, expected_lengths)

    # Read encoder metadata from the autoencoder checkpoint
    ae_path = models_dir / group_name / "autoencoder_best.pt"
    ae_ckpt = torch.load(ae_path, map_location="cpu", weights_only=False)
    if isinstance(ae_ckpt, dict):
        encoder_type = ae_ckpt.get("encoder_type", "tcn")
        n_classes = ae_ckpt.get("n_classes", 0)
        classifier_hidden = ae_ckpt.get("classifier_hidden", 32)
    else:
        encoder_type = "tcn"
        n_classes = 0
        classifier_hidden = 32

    # Build fresh autoencoder with the DEC's expected architecture
    model = MultiSegmentAutoencoder.from_segment_lengths(
        segment_lengths=expected_lengths,
        segment_latent_dims=config.SEGMENT_LATENT_DIMS,
        encoder_type=encoder_type,
        hidden_dims=config.AE_HIDDEN_DIMS,
        dropout=config.AE_DROPOUT,
        use_mlp_threshold=getattr(config, "USE_MLP_THRESHOLD", 30),
        tcn_channels=getattr(config, "TCN_CHANNELS", None),
        tcn_kernel_size=getattr(config, "TCN_KERNEL_SIZE", 3),
        multiscale_kernel_sizes=getattr(config, "MULTISCALE_KERNEL_SIZES", None),
        multiscale_channels=getattr(config, "MULTISCALE_CHANNELS", 32),
        n_classes=n_classes,
        classifier_hidden=classifier_hidden,
    )

    # Build IDEC wrapper and load DEC-refined state dict (all weights)
    idec = IDEC(autoencoder=model, n_clusters=k, alpha=config.DEC_ALPHA)
    idec.load_state_dict(dec_state)
    idec = idec.to(device)
    idec.eval()
    logger.info(f"    Loaded IDEC model ({encoder_type}, {k} clusters)")

    # Re-compute standardization from blocker data to correct domain shift
    recompute_standardization(idec, segments, device)

    # Run inference
    labels = get_all_labels(idec, segments, device)

    # Log cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in sorted(zip(unique, counts)):
        logger.info(f"      {group_name}_{u}: {c} cells")

    return labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Classify blocker cells into known RGC subtypes"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input parquet (compared_dataframe_v2.parquet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output parquet with labels",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory containing trained models per group",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing saved results per group",
    )
    parser.add_argument("--device", default="cuda", help="PyTorch device")
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Header
    logger.info("=" * 70)
    logger.info("Blocker Cell Classification Pipeline")
    logger.info("=" * 70)
    logger.info(f"Input:      {args.input}")
    logger.info(f"Output:     {args.output}")
    logger.info(f"Models:     {args.models_dir}")
    logger.info(f"Results:    {args.results_dir}")
    logger.info(f"Device:     {args.device}")

    # ------------------------------------------------------------------
    # Step 1: Load input
    # ------------------------------------------------------------------
    logger.info("\n--- Step 1: Load input ---")
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return
    df_blocker = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df_blocker)} pairs, {len(df_blocker.columns)} columns")

    # ------------------------------------------------------------------
    # Step 2: Build standard-format DataFrame (column mapping)
    # ------------------------------------------------------------------
    logger.info("\n--- Step 2: Column mapping ---")
    std_df = build_standard_dataframe(df_blocker)

    # ------------------------------------------------------------------
    # Step 3: Filter to RGC + quality
    # ------------------------------------------------------------------
    logger.info("\n--- Step 3: Filter RGC cells ---")
    rgc_df = filter_rgc_cells(std_df)

    if len(rgc_df) == 0:
        logger.error("No RGC cells after filtering. Aborting.")
        return

    # ------------------------------------------------------------------
    # Step 4: Assign functional groups
    # ------------------------------------------------------------------
    logger.info("\n--- Step 4: Group assignment ---")
    rgc_df = assign_groups(rgc_df)

    # ------------------------------------------------------------------
    # Step 5: Classify each group
    # ------------------------------------------------------------------
    logger.info("\n--- Step 5: Per-group classification ---")

    # Initialize label columns in the full blocker DataFrame
    df_blocker["group"] = ""
    df_blocker["subtype"] = ""
    df_blocker["cluster_id"] = -1

    total_classified = 0

    for group_name in GROUP_NAMES:
        group_mask = rgc_df["group"] == group_name
        group_df = rgc_df[group_mask].copy()

        if len(group_df) == 0:
            logger.info(f"  {group_name}: 0 cells, skipping")
            continue

        if len(group_df) < 5:
            logger.warning(
                f"  {group_name}: only {len(group_df)} cells, "
                f"too few for reliable classification, skipping"
            )
            continue

        try:
            labels = classify_group(
                group_df, group_name, args.models_dir, args.results_dir, args.device
            )

            # Map labels back to the original blocker DataFrame
            for idx, label in zip(group_df.index, labels):
                df_blocker.at[idx, "group"] = group_name
                df_blocker.at[idx, "subtype"] = f"{group_name}_{label}"
                df_blocker.at[idx, "cluster_id"] = int(label)

            total_classified += len(labels)

        except Exception as e:
            logger.error(
                f"  Error classifying {group_name}: {e}", exc_info=True
            )

    # ------------------------------------------------------------------
    # Step 6: Summary and save
    # ------------------------------------------------------------------
    logger.info("\n--- Step 6: Save results ---")

    labeled_mask = df_blocker["group"] != ""
    n_labeled = labeled_mask.sum()
    n_unlabeled = len(df_blocker) - n_labeled

    logger.info(f"Classification complete:")
    logger.info(f"  Total pairs:  {len(df_blocker)}")
    logger.info(f"  Labeled:      {n_labeled}")
    logger.info(f"  Unlabeled:    {n_unlabeled}")

    if labeled_mask.any():
        logger.info("\nGroup distribution:")
        for grp, cnt in df_blocker.loc[labeled_mask, "group"].value_counts().items():
            logger.info(f"  {grp}: {cnt}")

        logger.info("\nSubtype distribution:")
        for sub, cnt in (
            df_blocker.loc[labeled_mask, "subtype"].value_counts().head(20).items()
        ):
            logger.info(f"  {sub}: {cnt}")
        remaining = df_blocker.loc[labeled_mask, "subtype"].nunique() - 20
        if remaining > 0:
            logger.info(f"  ... and {remaining} more subtypes")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_blocker.to_parquet(args.output)
    logger.info(f"\nSaved to {args.output}")
    logger.info(f"  {len(df_blocker)} rows, {len(df_blocker.columns)} columns")


if __name__ == "__main__":
    main()
