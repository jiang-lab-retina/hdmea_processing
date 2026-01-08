"""
Combined Validation Pipeline.

Runs all validation plots from a single parquet file containing all extracted features.
Imports and uses high-level plotting functions from individual validation modules.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# IMPORTS FROM VALIDATION MODULES
# =============================================================================

# DSGC validation plots
from validation_dsgc import (
    plot_selectivity_examples,
    plot_top_index_examples,
    plot_dsgc_histograms,
    DIRECTION_COLUMNS,
)

# Frequency step feature validation
from validation_freq_features import (
    plot_freq_step_sine_fits,
    plot_freq_step_full_trace_with_fits,
    plot_freq_feature_distributions,
    plot_r_squared_analysis,
    plot_feature_statistics_table as plot_freq_statistics_table,
    plot_valid_invalid_comparison,
)

# Quality index validation
from validation import (
    plot_qi_example_traces,
    plot_qi_threshold_curve,
    plot_quality_index_histogram,
)

# Green-blue feature validation
from validation_gb_features import (
    plot_gb_features_validation,
)

# Step feature validation
from validation_step_features import (
    plot_step_features_validation,
)

# Cell type classification validation
from validation_cell_types import (
    classify_cells,
    plot_cell_type_pie,
    plot_venn_diagram,
    plot_overlap_matrix,
    plot_upset_style,
)

# Import step_config for trace column names
from step_config import (
    STEP_TRACE_COLUMN,
    GB_TRACE_COLUMN,
    FREQ_TRACE_COLUMN,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_PARQUET = Path(__file__).parent.parent / "firing_rate_with_all_features_loaded_extracted20260102.parquet"
OUTPUT_DIR = Path(__file__).parent / "plots_combined"


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def main():
    """Run all validation plots from combined feature file."""
    print("=" * 80)
    print("COMBINED VALIDATION PIPELINE")
    print("=" * 80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from: {INPUT_PARQUET}")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded DataFrame: {df.shape} (units x columns)")
    
    # Print column summary
    print(f"\nTotal columns: {len(df.columns)}")
    print(f"Total units: {len(df)}")
    
    # =========================================================================
    # 1. QUALITY INDEX VALIDATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("SECTION 1: Quality Index Validation")
    print("=" * 60)
    
    # step_up_QI histogram
    print("\n[1.1] Generating step_up_QI histogram...")
    plot_quality_index_histogram(
        df,
        qi_column="step_up_QI",
        output_path=OUTPUT_DIR / "qi_step_up_histogram.png",
        bins=50,
    )
    
    # step_up_QI threshold curve
    print("[1.2] Generating step_up_QI threshold curve...")
    plot_qi_threshold_curve(
        df,
        qi_column="step_up_QI",
        output_path=OUTPUT_DIR / "qi_step_up_threshold_curve.png",
    )
    
    # iprgc_2hz_QI histogram
    print("[1.3] Generating iprgc_2hz_QI histogram...")
    plot_quality_index_histogram(
        df,
        qi_column="iprgc_2hz_QI",
        output_path=OUTPUT_DIR / "qi_iprgc_2hz_histogram.png",
        bins=50,
    )
    
    # iprgc_2hz_QI threshold curve
    print("[1.4] Generating iprgc_2hz_QI threshold curve...")
    plot_qi_threshold_curve(
        df,
        qi_column="iprgc_2hz_QI",
        output_path=OUTPUT_DIR / "qi_iprgc_2hz_threshold_curve.png",
    )
    
    # iprgc_20hz_QI histogram
    print("[1.5] Generating iprgc_20hz_QI histogram...")
    plot_quality_index_histogram(
        df,
        qi_column="iprgc_20hz_QI",
        output_path=OUTPUT_DIR / "qi_iprgc_20hz_histogram.png",
        bins=50,
    )
    
    # =========================================================================
    # 2. DSGC VALIDATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("SECTION 2: DSGC (Direction/Orientation Selectivity) Validation")
    print("=" * 60)
    
    # Check if DSGC columns exist
    if "dsi" in df.columns and "ds_p_value" in df.columns:
        # DSGC histograms
        print("\n[2.1] Generating DSI/OSI histograms...")
        plot_dsgc_histograms(
            df,
            output_path=OUTPUT_DIR / "dsgc_histograms.png",
        )
        
        # Significant DSGC examples
        print("[2.2] Generating significant DSGC examples...")
        plot_selectivity_examples(
            df,
            filter_column="ds_p_value",
            significant=True,
            index_column="dsi",
            output_path=OUTPUT_DIR / "dsgc_significant_examples.png",
            n_units=10,
        )
        
        # Non-significant DSGC examples
        print("[2.3] Generating non-significant DSGC examples...")
        plot_selectivity_examples(
            df,
            filter_column="ds_p_value",
            significant=False,
            index_column="dsi",
            output_path=OUTPUT_DIR / "dsgc_nonsignificant_examples.png",
            n_units=10,
        )
        
        # Top DSI examples
        print("[2.4] Generating top DSI examples...")
        plot_top_index_examples(
            df,
            index_column="dsi",
            output_path=OUTPUT_DIR / "dsgc_top_dsi_examples.png",
            n_units=10,
        )
        
        # Top OSI examples
        print("[2.5] Generating top OSI examples...")
        plot_top_index_examples(
            df,
            index_column="osi",
            output_path=OUTPUT_DIR / "osgc_top_osi_examples.png",
            n_units=10,
        )
    else:
        print("  SKIPPED: DSGC columns (dsi, ds_p_value) not found")
    
    # =========================================================================
    # 3. STEP RESPONSE VALIDATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("SECTION 3: Step Response Feature Validation")
    print("=" * 60)
    
    # Check if step columns exist
    if STEP_TRACE_COLUMN in df.columns and "on_peak_extreme" in df.columns:
        print("\n[3.1] Generating step feature validation traces...")
        plot_step_features_validation(
            df,
            trace_column=STEP_TRACE_COLUMN,
            qi_column="step_up_QI",
            output_path=OUTPUT_DIR / "step_features_validation.png",
            n_units=10,
            top_percentile=0.20,
            figsize=(24, 12),
        )
        
        # Step QI example traces
        print("[3.2] Generating step_up_QI example traces...")
        plot_qi_example_traces(
            df,
            trace_column=STEP_TRACE_COLUMN,
            qi_column="step_up_QI",
            output_path=OUTPUT_DIR / "step_qi_example_traces.png",
            n_examples=5,
            n_ranges=10,
            filter_cutoff=10.0,
            sampling_rate=60.0,
        )
    else:
        print(f"  SKIPPED: Step columns ({STEP_TRACE_COLUMN}, on_peak_extreme) not found")
    
    # =========================================================================
    # 4. GREEN-BLUE VALIDATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("SECTION 4: Green-Blue Response Feature Validation")
    print("=" * 60)
    
    # Check if GB columns exist
    if GB_TRACE_COLUMN in df.columns and "green_on_peak_extreme" in df.columns:
        print("\n[4.1] Generating green-blue feature validation traces...")
        plot_gb_features_validation(
            df,
            trace_column=GB_TRACE_COLUMN,
            qi_column="step_up_QI",
            output_path=OUTPUT_DIR / "gb_features_validation.png",
            n_units=10,
            top_percentile=0.20,
            figsize=(24, 12),
        )
    else:
        print(f"  SKIPPED: Green-blue columns ({GB_TRACE_COLUMN}, green_on_peak_extreme) not found")
    
    # =========================================================================
    # 5. FREQUENCY STEP VALIDATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("SECTION 5: Frequency Step Feature Validation")
    print("=" * 60)
    
    # Check if frequency columns exist
    if FREQ_TRACE_COLUMN in df.columns and "freq_step_1hz_amp" in df.columns:
        print("\n[5.1] Generating frequency sine fit validation...")
        plot_freq_step_sine_fits(
            df,
            trace_column=FREQ_TRACE_COLUMN,
            qi_column="step_up_QI",
            output_path=OUTPUT_DIR / "freq_features_sine_fits.png",
            n_units=6,
            top_percentile=0.20,
        )
        
        print("[5.2] Generating frequency full trace with fits...")
        plot_freq_step_full_trace_with_fits(
            df,
            trace_column=FREQ_TRACE_COLUMN,
            qi_column="step_up_QI",
            output_path=OUTPUT_DIR / "freq_features_full_traces.png",
            n_units=4,
            top_percentile=0.20,
        )
        
        print("[5.3] Generating frequency feature distributions...")
        plot_freq_feature_distributions(
            df,
            output_path=OUTPUT_DIR / "freq_features_distributions.png",
        )
        
        print("[5.4] Generating R-squared analysis...")
        plot_r_squared_analysis(
            df,
            output_path=OUTPUT_DIR / "freq_features_r_squared_analysis.png",
        )
        
        print("[5.5] Generating frequency statistics table...")
        plot_freq_statistics_table(
            df,
            output_path=OUTPUT_DIR / "freq_features_statistics.png",
        )
        
        print("[5.6] Generating valid/invalid comparison...")
        plot_valid_invalid_comparison(
            df,
            trace_column=FREQ_TRACE_COLUMN,
            output_path=OUTPUT_DIR / "freq_features_valid_invalid_comparison.png",
            n_valid=5,
            n_invalid=5,
        )
    else:
        print(f"  SKIPPED: Frequency columns ({FREQ_TRACE_COLUMN}, freq_step_1hz_amp) not found")
    
    # =========================================================================
    # 6. CELL TYPE CLASSIFICATION VALIDATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("SECTION 6: Cell Type Classification Validation")
    print("=" * 60)
    
    # Check if classification columns exist
    if all(col in df.columns for col in ["ds_p_value", "os_p_value", "iprgc_2hz_QI"]):
        # Classify cells
        print("\n[6.1] Classifying cells...")
        df_classified = classify_cells(df)
        
        # Print summary
        total = len(df_classified)
        print(f"  DSGC (ds_p < 0.05): {df_classified['is_DSGC'].sum():,} ({100*df_classified['is_DSGC'].sum()/total:.1f}%)")
        print(f"  OSGC (os_p < 0.05): {df_classified['is_OSGC'].sum():,} ({100*df_classified['is_OSGC'].sum()/total:.1f}%)")
        print(f"  ipRGC (QI > 0.8): {df_classified['is_ipRGC'].sum():,} ({100*df_classified['is_ipRGC'].sum()/total:.1f}%)")
        print(f"  Other: {df_classified['is_Other'].sum():,} ({100*df_classified['is_Other'].sum()/total:.1f}%)")
        
        print("[6.2] Generating cell type pie charts...")
        plot_cell_type_pie(
            df_classified,
            output_path=OUTPUT_DIR / "cell_type_pie_chart.png",
        )
        
        print("[6.3] Generating Venn diagram...")
        plot_venn_diagram(
            df_classified,
            output_path=OUTPUT_DIR / "cell_type_venn.png",
        )
        
        print("[6.4] Generating overlap matrix...")
        plot_overlap_matrix(
            df_classified,
            output_path=OUTPUT_DIR / "cell_type_overlap_matrix.png",
        )
        
        print("[6.5] Generating UpSet-style plot...")
        plot_upset_style(
            df_classified,
            output_path=OUTPUT_DIR / "cell_type_combinations.png",
        )
    else:
        print("  SKIPPED: Classification columns (ds_p_value, os_p_value, iprgc_2hz_QI) not found")
    
    # =========================================================================
    # 7. IPRGC EXAMPLE TRACES
    # =========================================================================
    print("\n" + "=" * 60)
    print("SECTION 7: ipRGC Example Traces")
    print("=" * 60)
    
    if "iprgc_test" in df.columns and "iprgc_2hz_QI" in df.columns:
        print("\n[7.1] Generating iprgc_2hz_QI example traces...")
        plot_qi_example_traces(
            df,
            trace_column="iprgc_test",
            qi_column="iprgc_2hz_QI",
            output_path=OUTPUT_DIR / "iprgc_2hz_example_traces.png",
            n_examples=5,
            n_ranges=10,
            figsize=(16, 20),
            filter_cutoff=2.0,
            sampling_rate=60.0,
        )
        
        print("[7.2] Generating iprgc_20hz_QI example traces...")
        plot_qi_example_traces(
            df,
            trace_column="iprgc_test",
            qi_column="iprgc_20hz_QI",
            output_path=OUTPUT_DIR / "iprgc_20hz_example_traces.png",
            n_examples=5,
            n_ranges=10,
            figsize=(16, 20),
            max_samples=600,  # First 10 seconds
            filter_cutoff=20.0,
            sampling_rate=60.0,
        )
    else:
        print("  SKIPPED: ipRGC columns (iprgc_test, iprgc_2hz_QI) not found")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    
    # List generated files
    plot_files = list(OUTPUT_DIR.glob("*.png"))
    print(f"\nGenerated {len(plot_files)} plots:")
    for f in sorted(plot_files):
        print(f"  - {f.name}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

