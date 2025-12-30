"""
Plot Firing Rates by Movie

Creates one plot per dataset with:
- Rows: units
- Columns: movies
- Each subplot: all trials as transparent traces + mean on top
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
OUTPUT_DIR = Path(__file__).parent / "output"
INPUT_FILE = OUTPUT_DIR / "firing_rate_by_movie.parquet"
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Plot settings
TRIAL_ALPHA = 0.4
TRIAL_COLOR = "steelblue"
MEAN_COLOR = "darkred"
MEAN_LINEWIDTH = 1.5
FIGSIZE_PER_SUBPLOT = (2.5, 1.5)  # Width, height per subplot
MAX_UNITS_PER_PLOT = 20  # Limit units per plot for readability


def plot_dataset(df: pd.DataFrame, dataset_id: str, movies: list):
    """
    Create a plot for a single dataset.
    
    Args:
        df: DataFrame filtered for this dataset
        dataset_id: Dataset identifier
        movies: List of movie columns to plot
    """
    n_units = len(df)
    n_movies = len(movies)
    
    # Limit units for readability
    if n_units > MAX_UNITS_PER_PLOT:
        df = df.iloc[:MAX_UNITS_PER_PLOT]
        n_units = MAX_UNITS_PER_PLOT
        truncated = True
    else:
        truncated = False
    
    # Calculate figure size
    fig_width = FIGSIZE_PER_SUBPLOT[0] * n_movies
    fig_height = FIGSIZE_PER_SUBPLOT[1] * n_units
    
    # Create figure
    fig, axes = plt.subplots(
        n_units, n_movies,
        figsize=(fig_width, fig_height),
        squeeze=False
    )
    
    # Plot each unit and movie
    for row_idx, (unit_idx, unit_row) in enumerate(df.iterrows()):
        unit_id = unit_idx.split("_")[-1]  # Extract unit_XXX part
        
        for col_idx, movie in enumerate(movies):
            ax = axes[row_idx, col_idx]
            
            data = unit_row[movie]
            
            if data is None or (isinstance(data, float) and np.isnan(data)):
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # Convert to numpy array if needed
                if isinstance(data, list):
                    # Handle nested lists from parquet
                    data = np.array([np.array(row) for row in data])
                elif isinstance(data, np.ndarray):
                    # Handle nested numpy arrays from parquet
                    if data.dtype == object:
                        data = np.array([np.array(row) for row in data])
                
                # Ensure 2D shape (n_trials, n_bins)
                if data.ndim == 1:
                    # If 1D, assume it's a single trial
                    data = data.reshape(1, -1)
                
                n_trials, n_bins = data.shape
                x = np.arange(n_bins)
                
                # Plot individual trials (transparent)
                for trial_idx in range(n_trials):
                    ax.plot(x, data[trial_idx], color=TRIAL_COLOR, alpha=TRIAL_ALPHA, linewidth=0.5)
                
                # Plot mean (opaque, on top)
                mean_trace = np.mean(data, axis=0)
                ax.plot(x, mean_trace, color=MEAN_COLOR, linewidth=MEAN_LINEWIDTH)
                
                # Set axis limits
                ax.set_xlim(0, n_bins)
                y_max = np.nanmax(data) * 1.1 if np.nanmax(data) > 0 else 1
                ax.set_ylim(0, y_max)
            
            # Labels
            if row_idx == 0:
                # Shorten movie name for title
                short_name = movie.replace("moving_h_bar_s5_d8_3x_", "mbar_")
                short_name = short_name.replace("step_up_5s_5i_b0_", "step_")
                ax.set_title(short_name, fontsize=8)
            
            if col_idx == 0:
                ax.set_ylabel(unit_id, fontsize=7)
            
            # Remove tick labels for cleaner look
            ax.tick_params(axis="both", labelsize=5)
            if row_idx < n_units - 1:
                ax.set_xticklabels([])
    
    # Suptitle
    title = f"Dataset: {dataset_id}"
    if truncated:
        title += f" (showing {MAX_UNITS_PER_PLOT} of {len(df)} units)"
    fig.suptitle(title, fontsize=12, y=1.02)
    
    plt.tight_layout()
    
    return fig


def main():
    print("=" * 80)
    print("Plot Firing Rates by Movie")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df)} units, {len(df.columns)} movies")
    
    # Get movies (columns)
    movies = df.columns.tolist()
    print(f"\nMovies: {movies}")
    
    # Extract dataset_id from index
    df["dataset_id"] = df.index.map(lambda x: "_".join(x.split("_")[:-1]))
    datasets = df["dataset_id"].unique()
    print(f"\nDatasets: {len(datasets)}")
    
    # Create plots for each dataset
    print("\nGenerating plots...")
    for dataset_id in tqdm(datasets, desc="Plotting datasets"):
        dataset_df = df[df["dataset_id"] == dataset_id].drop(columns=["dataset_id"])
        
        fig = plot_dataset(dataset_df, dataset_id, movies)
        
        # Save
        output_path = PLOTS_DIR / f"firing_rate_{dataset_id}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    print(f"\nSaved {len(datasets)} plots to: {PLOTS_DIR}")
    
    # Create an index HTML file
    html_path = PLOTS_DIR / "view_plots.html"
    with open(html_path, "w") as f:
        f.write("<!DOCTYPE html>\n<html>\n<head>\n")
        f.write("<title>Firing Rate Plots</title>\n")
        f.write("<style>body{font-family:sans-serif;} img{max-width:100%; margin:10px 0;}</style>\n")
        f.write("</head>\n<body>\n")
        f.write("<h1>Firing Rate by Movie - All Datasets</h1>\n")
        
        for dataset_id in sorted(datasets):
            f.write(f"<h2>{dataset_id}</h2>\n")
            f.write(f'<img src="firing_rate_{dataset_id}.png" alt="{dataset_id}">\n')
        
        f.write("</body>\n</html>")
    
    print(f"Created HTML viewer: {html_path}")
    
    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

