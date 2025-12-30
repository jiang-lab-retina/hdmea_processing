"""
Analyze Trial Length Variation

Shows min, max, mean, and frequency tables for each movie.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import h5py
import numpy as np
import pandas as pd

# Configuration
project_root = Path(__file__).parent.parent.parent
HDF5_DIR = project_root / "Projects/unified_pipeline/export_all_steps"


def collect_trial_lengths():
    """Collect frame/bin counts for all trials across all movies."""
    h5_files = sorted(HDF5_DIR.glob("*.h5"))
    data = []
    
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as f:
            dataset_id = h5_path.stem
            
            if "metadata/frame_timestamps" not in f:
                continue
            all_frames = f["metadata/frame_timestamps"][:]
            acq_rate = float(f["metadata/acquisition_rate"][()])
            
            unit_ids = list(f["units"].keys())
            if not unit_ids:
                continue
            sample_unit = unit_ids[0]
            
            # Get all movies
            sectioned_path = f"units/{sample_unit}/spike_times_sectioned"
            if sectioned_path not in f:
                continue
            
            for movie_name in f[sectioned_path].keys():
                trials_path = f"{sectioned_path}/{movie_name}/trials_start_end"
                if trials_path not in f:
                    continue
                
                trials_start_end = f[trials_path][:]
                
                # Check for direction sections
                dir_section_path = f"{sectioned_path}/{movie_name}/direction_section"
                has_dir_section = dir_section_path in f
                
                if has_dir_section and movie_name == "moving_h_bar_s5_d8_3x":
                    # Process direction sections
                    dir_group = f[dir_section_path]
                    for direction in dir_group.keys():
                        if direction == "_attrs":
                            continue
                        dir_data = dir_group[direction]
                        if "section_bounds" not in dir_data:
                            continue
                        bounds = dir_data["section_bounds"][:]
                        for rep_idx, (start_rel, end_rel) in enumerate(bounds):
                            n_frames = end_rel - start_rel
                            data.append({
                                "movie": f"{movie_name}_dir{direction}",
                                "movie_base": movie_name,
                                "dataset": dataset_id,
                                "trial": f"{direction}_{rep_idx}",
                                "n_frames": int(n_frames),
                                "n_bins": int(n_frames - 1) if n_frames > 1 else 0,
                            })
                else:
                    # Regular movies
                    for trial_idx, (start, end) in enumerate(trials_start_end):
                        # Only analyze first two trials for iprgc_test
                        if movie_name == "iprgc_test" and trial_idx >= 2:
                            continue
                        
                        mask = (all_frames >= start) & (all_frames < end)
                        n_frames = np.sum(mask)
                        
                        # For iprgc, use sample-based calculation
                        if movie_name == "iprgc_test":
                            duration = end - start
                            n_bins = int(np.ceil(duration / (acq_rate / 60.0)))
                        else:
                            n_bins = n_frames - 1 if n_frames > 1 else 0
                        
                        data.append({
                            "movie": movie_name,
                            "movie_base": movie_name,
                            "dataset": dataset_id,
                            "trial": trial_idx,
                            "n_frames": int(n_frames),
                            "n_bins": int(n_bins),
                        })
    
    return pd.DataFrame(data)


def main():
    print("=" * 80)
    print("TRIAL LENGTH VARIATION ANALYSIS")
    print("=" * 80)
    
    df = collect_trial_lengths()
    
    # Group by movie_base for summary
    movies = sorted(df["movie_base"].unique())
    
    for movie in movies:
        movie_df = df[df["movie_base"] == movie]
        
        print(f"\n{'-' * 80}")
        print(f"MOVIE: {movie}")
        print(f"{'-' * 80}")
        
        # Statistics
        n_bins = movie_df["n_bins"]
        print(f"\nStatistics (bins):")
        print(f"  Min:    {n_bins.min()}")
        print(f"  Max:    {n_bins.max()}")
        print(f"  Mean:   {n_bins.mean():.2f}")
        print(f"  Std:    {n_bins.std():.2f}")
        print(f"  Range:  {n_bins.max() - n_bins.min()}")
        
        # Unique trials
        print(f"\nTrials: {movie_df['trial'].nunique()} unique trial types")
        print(f"Datasets: {movie_df['dataset'].nunique()} recordings")
        
        # Frequency table
        print(f"\nFrequency Table (n_bins):")
        freq = n_bins.value_counts().sort_index()
        if len(freq) <= 20:
            for val, count in freq.items():
                pct = count / len(n_bins) * 100
                bar = "#" * int(pct / 2)
                print(f"  {val:>6}: {count:>4} ({pct:>5.1f}%) {bar}")
        else:
            # Show histogram for many unique values
            print(f"  (Too many unique values: {len(freq)}, showing summary)")
            print(f"  Most common: {freq.idxmax()} ({freq.max()} occurrences)")
            print(f"  Least common: {freq.idxmin()} ({freq.min()} occurrences)")
            # Show binned histogram
            bins = np.linspace(n_bins.min(), n_bins.max(), 11)
            hist, edges = np.histogram(n_bins, bins=bins)
            print(f"\n  Histogram (10 bins):")
            for i in range(len(hist)):
                pct = hist[i] / len(n_bins) * 100
                bar = "#" * int(pct / 2)
                print(f"  {edges[i]:>7.0f}-{edges[i+1]:<7.0f}: {hist[i]:>4} ({pct:>5.1f}%) {bar}")
    
    # Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY TABLE")
    print("=" * 80)
    
    summary = df.groupby("movie_base").agg({
        "n_bins": ["min", "max", "mean", "std", lambda x: x.max() - x.min()],
        "dataset": "nunique",
        "trial": "nunique"
    }).round(2)
    summary.columns = ["min", "max", "mean", "std", "range", "n_datasets", "n_trial_types"]
    print("\n" + summary.to_string())
    
    # Consistency check
    print(f"\n{'=' * 80}")
    print("CONSISTENCY CHECK (per trial type across datasets)")
    print("=" * 80)
    
    for movie in movies:
        movie_df = df[df["movie_base"] == movie]
        
        # Check variation per trial type
        trial_variation = movie_df.groupby("trial")["n_bins"].agg(["min", "max"])
        trial_variation["range"] = trial_variation["max"] - trial_variation["min"]
        
        has_variation = trial_variation["range"].max() > 0
        
        if has_variation:
            print(f"\n{movie}: VARIES across datasets")
            varying_trials = trial_variation[trial_variation["range"] > 0]
            for trial, row in varying_trials.iterrows():
                print(f"  Trial {trial}: {int(row['min'])}-{int(row['max'])} bins (range: {int(row['range'])})")
        else:
            print(f"\n{movie}: CONSISTENT (all trials same length across datasets)")
    
    print(f"\n{'=' * 80}")
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

