"""
Visualize Trial Length Distributions

Creates plots showing:
1. Trial duration distribution by stimulus type
2. Per-trial length variation across recordings
3. Trial timing overview
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configuration
OUTPUT_DIR = Path(__file__).parent / "output"
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10


def load_data():
    """Load all CSV files."""
    summary_df = pd.read_csv(OUTPUT_DIR / "trial_start_end_summary.csv")
    full_df = pd.read_csv(OUTPUT_DIR / "trial_start_end_full.csv")
    
    # Also try to load the older summary if it exists
    try:
        length_summary_df = pd.read_csv(OUTPUT_DIR / "trial_length_summary.csv")
    except FileNotFoundError:
        length_summary_df = None
    
    return summary_df, full_df, length_summary_df


def plot_trials_per_stimulus(full_df):
    """Plot number of trials per stimulus type."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Count unique trial indices per movie
    trials_per_movie = full_df.groupby('movie_name')['n_trials_total'].first().sort_values(ascending=True)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(trials_per_movie)))
    bars = ax.barh(trials_per_movie.index, trials_per_movie.values, color=colors)
    
    ax.set_xlabel('Number of Trials per Recording', fontsize=12)
    ax.set_title('Trials per Stimulus Type', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, trials_per_movie.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, str(int(val)), 
                va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlim(0, max(trials_per_movie.values) * 1.15)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "01_trials_per_stimulus.png", dpi=150, bbox_inches='tight')
    print(f"Saved: 01_trials_per_stimulus.png")
    plt.close()


def plot_duration_distribution(summary_df):
    """Plot trial duration distribution with error bars."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Get unique movies and sort by mean duration
    movie_stats = summary_df.groupby('movie_name').agg({
        'dur_mean_s': 'mean',
        'dur_min_s': 'min',
        'dur_max_s': 'max',
        'samples_range': 'mean',
    }).sort_values('dur_mean_s')
    
    x = np.arange(len(movie_stats))
    
    # Plot bars with error bars showing min-max range
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(movie_stats)))
    bars = ax.bar(x, movie_stats['dur_mean_s'], color=colors, edgecolor='black', linewidth=0.5)
    
    # Add error bars for min-max range
    yerr_low = movie_stats['dur_mean_s'] - movie_stats['dur_min_s']
    yerr_high = movie_stats['dur_max_s'] - movie_stats['dur_mean_s']
    ax.errorbar(x, movie_stats['dur_mean_s'], yerr=[yerr_low, yerr_high], 
                fmt='none', color='black', capsize=5, capthick=2, linewidth=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(movie_stats.index, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Duration (seconds)', fontsize=12)
    ax.set_title('Trial Duration by Stimulus Type\n(Error bars show min-max range across recordings)', 
                 fontsize=14, fontweight='bold')
    
    # Add duration labels on bars
    for i, (bar, dur) in enumerate(zip(bars, movie_stats['dur_mean_s'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{dur:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylim(0, movie_stats['dur_max_s'].max() * 1.15)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "02_duration_distribution.png", dpi=150, bbox_inches='tight')
    print(f"Saved: 02_duration_distribution.png")
    plt.close()


def plot_sample_variation(summary_df):
    """Plot sample variation across recordings for each trial."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Filter out iprgc_test (no variation) for clarity
    df = summary_df[summary_df['samples_range'] > 0].copy()
    
    # Color by stimulus type
    movies = df['movie_name'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(movies)))
    color_map = dict(zip(movies, colors))
    
    # Create grouped bar positions
    movie_groups = df.groupby('movie_name')
    
    y_pos = 0
    y_labels = []
    y_positions = []
    
    for movie, group in movie_groups:
        for _, row in group.iterrows():
            y_positions.append(y_pos)
            y_labels.append(f"{movie}\nTrial {int(row['trial_idx'])}")
            ax.barh(y_pos, row['samples_range'], color=color_map[movie], 
                    edgecolor='black', linewidth=0.5, height=0.7)
            
            # Add label with ms
            ms = row['samples_range'] / 20  # 20kHz
            ax.text(row['samples_range'] + 500, y_pos, f"{ms:.0f}ms", 
                    va='center', fontsize=8)
            y_pos += 1
        y_pos += 0.5  # Gap between movies
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel('Sample Variation (samples @ 20kHz)', fontsize=12)
    ax.set_title('Trial Length Variation Across Recordings\n(excluding iprgc_test with 0 variation)', 
                 fontsize=14, fontweight='bold')
    
    # Add secondary x-axis in milliseconds
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()[0] / 20, ax.get_xlim()[1] / 20)
    ax2.set_xlabel('Variation (milliseconds)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "03_sample_variation.png", dpi=150, bbox_inches='tight')
    print(f"Saved: 03_sample_variation.png")
    plt.close()


def plot_trial_timeline(full_df):
    """Plot trial start/end times as a timeline for each recording."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    datasets = full_df['dataset_id'].unique()
    
    # Color map for movies
    movies = full_df['movie_name'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(movies)))
    color_map = dict(zip(movies, colors))
    
    for idx, dataset in enumerate(datasets[:6]):
        ax = axes[idx]
        df = full_df[full_df['dataset_id'] == dataset]
        
        y_pos = 0
        movie_positions = {}
        
        for movie in sorted(df['movie_name'].unique()):
            movie_df = df[df['movie_name'] == movie]
            movie_positions[movie] = y_pos
            
            for _, row in movie_df.iterrows():
                start_s = row['start_sample'] / 20000
                end_s = row['end_sample'] / 20000
                duration = end_s - start_s
                
                rect = plt.Rectangle((start_s, y_pos - 0.4), duration, 0.8,
                                      facecolor=color_map[movie], edgecolor='black',
                                      linewidth=0.5, alpha=0.8)
                ax.add_patch(rect)
            
            y_pos += 1
        
        ax.set_xlim(0, df['end_sample'].max() / 20000 * 1.02)
        ax.set_ylim(-0.5, len(movie_positions) - 0.5)
        ax.set_yticks(range(len(movie_positions)))
        ax.set_yticklabels(list(movie_positions.keys()), fontsize=8)
        ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.set_title(dataset[:20], fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    # Add legend
    legend_patches = [mpatches.Patch(color=color_map[m], label=m) for m in movies]
    fig.legend(handles=legend_patches, loc='upper center', ncol=4, 
               bbox_to_anchor=(0.5, 1.02), fontsize=9)
    
    plt.suptitle('Trial Timeline by Recording', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "04_trial_timeline.png", dpi=150, bbox_inches='tight')
    print(f"Saved: 04_trial_timeline.png")
    plt.close()


def plot_consistency_heatmap(summary_df):
    """Plot heatmap showing trial length consistency across trials and stimuli."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Pivot data for heatmap
    pivot_df = summary_df.pivot_table(
        index='movie_name', 
        columns='trial_idx', 
        values='samples_range',
        aggfunc='first'
    ).fillna(0)
    
    # Create heatmap
    im = ax.imshow(pivot_df.values, cmap='RdYlGn_r', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Sample Variation')
    
    # Labels
    ax.set_xticks(range(pivot_df.shape[1]))
    ax.set_xticklabels([f'T{int(c)}' for c in pivot_df.columns], fontsize=8)
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index, fontsize=9)
    ax.set_xlabel('Trial Index', fontsize=12)
    ax.set_ylabel('Stimulus Type', fontsize=12)
    ax.set_title('Trial Length Variation Heatmap\n(Green = consistent, Red = variable)', 
                 fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(pivot_df.index)):
        for j in range(pivot_df.shape[1]):
            val = pivot_df.iloc[i, j]
            if val > 0:
                text_color = 'white' if val > pivot_df.values.max() * 0.5 else 'black'
                ax.text(j, i, f'{val/20:.0f}', ha='center', va='center', 
                        fontsize=7, color=text_color)
            else:
                ax.text(j, i, '0', ha='center', va='center', fontsize=7, color='green')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "05_consistency_heatmap.png", dpi=150, bbox_inches='tight')
    print(f"Saved: 05_consistency_heatmap.png")
    plt.close()


def plot_duration_by_recording(full_df):
    """Plot trial durations grouped by recording to show cross-recording variation."""
    # Select a few representative stimuli
    stimuli = ['step_up_5s_5i_b0_30x', 'freq_step_5st_3x', 'iprgc_test', 'moving_h_bar_s5_d8_3x']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, stim in enumerate(stimuli):
        ax = axes[idx]
        df = full_df[full_df['movie_name'] == stim]
        
        if df.empty:
            continue
        
        # Group by dataset
        datasets = df['dataset_id'].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(datasets)))
        
        for d_idx, dataset in enumerate(datasets):
            ds_df = df[df['dataset_id'] == dataset]
            trials = ds_df['trial_idx'].values
            durations = ds_df['duration_s'].values
            
            ax.scatter(trials, durations, color=colors[d_idx], s=50, alpha=0.7,
                       label=dataset[:15], edgecolors='black', linewidth=0.5)
            ax.plot(trials, durations, color=colors[d_idx], alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Trial Index', fontsize=10)
        ax.set_ylabel('Duration (seconds)', fontsize=10)
        ax.set_title(f'{stim}\n({len(df["trial_idx"].unique())} trials/recording)', 
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.3)
    
    plt.suptitle('Trial Duration Variation Across Recordings', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "06_duration_by_recording.png", dpi=150, bbox_inches='tight')
    print(f"Saved: 06_duration_by_recording.png")
    plt.close()


def plot_summary_dashboard(summary_df, full_df):
    """Create a summary dashboard with key metrics."""
    fig = plt.figure(figsize=(16, 12))
    
    # Grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Number of trials per stimulus (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    trials_per_movie = full_df.groupby('movie_name')['n_trials_total'].first().sort_values()
    bars = ax1.barh(range(len(trials_per_movie)), trials_per_movie.values, 
                    color=plt.cm.viridis(np.linspace(0.2, 0.8, len(trials_per_movie))))
    ax1.set_yticks(range(len(trials_per_movie)))
    ax1.set_yticklabels(trials_per_movie.index, fontsize=8)
    ax1.set_xlabel('Trials/Recording')
    ax1.set_title('Trials per Stimulus', fontweight='bold')
    for bar, val in zip(bars, trials_per_movie.values):
        ax1.text(val + 0.3, bar.get_y() + bar.get_height()/2, str(int(val)), va='center', fontsize=8)
    
    # 2. Mean duration (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    mean_duration = summary_df.groupby('movie_name')['dur_mean_s'].mean().sort_values()
    bars = ax2.barh(range(len(mean_duration)), mean_duration.values,
                    color=plt.cm.plasma(np.linspace(0.2, 0.8, len(mean_duration))))
    ax2.set_yticks(range(len(mean_duration)))
    ax2.set_yticklabels(mean_duration.index, fontsize=8)
    ax2.set_xlabel('Duration (s)')
    ax2.set_title('Mean Trial Duration', fontweight='bold')
    
    # 3. Consistency indicator (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    consistency = summary_df.groupby('movie_name')['all_same_length'].all()
    colors = ['green' if c else 'red' for c in consistency.values]
    bars = ax3.barh(range(len(consistency)), [1]*len(consistency), color=colors)
    ax3.set_yticks(range(len(consistency)))
    ax3.set_yticklabels(consistency.index, fontsize=8)
    ax3.set_xlim(0, 1.2)
    ax3.set_xticks([])
    ax3.set_title('Length Consistency\n(Green=Consistent)', fontweight='bold')
    for i, (bar, c) in enumerate(zip(bars, consistency.values)):
        label = 'YES' if c else 'NO'
        ax3.text(0.5, i, label, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # 4. Variation boxplot (middle, spanning 2 columns)
    ax4 = fig.add_subplot(gs[1, :2])
    df_var = summary_df[summary_df['samples_range'] > 0]
    movies = df_var['movie_name'].unique()
    data = [df_var[df_var['movie_name'] == m]['samples_range'].values / 20 for m in movies]  # Convert to ms
    bp = ax4.boxplot(data, labels=movies, patch_artist=True)
    for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(movies)))):
        patch.set_facecolor(color)
    ax4.set_ylabel('Variation (ms)')
    ax4.set_title('Sample Variation Distribution by Stimulus', fontweight='bold')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    
    # 5. Recording count (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    recordings = full_df['dataset_id'].nunique()
    stimuli_count = full_df['movie_name'].nunique()
    total_trials = len(full_df)
    
    ax5.text(0.5, 0.7, f'{recordings}', ha='center', va='center', fontsize=48, fontweight='bold', color='#2E86AB')
    ax5.text(0.5, 0.45, 'Recordings', ha='center', va='center', fontsize=14)
    ax5.text(0.5, 0.25, f'{stimuli_count} stimuli | {total_trials} total trials', 
             ha='center', va='center', fontsize=10, color='gray')
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis('off')
    ax5.set_title('Dataset Overview', fontweight='bold')
    
    # 6. Timeline preview (bottom, full width)
    ax6 = fig.add_subplot(gs[2, :])
    sample_dataset = full_df['dataset_id'].iloc[0]
    df_sample = full_df[full_df['dataset_id'] == sample_dataset]
    
    movies = df_sample['movie_name'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(movies)))
    color_map = dict(zip(movies, colors))
    
    y_pos = 0
    for movie in sorted(movies):
        movie_df = df_sample[df_sample['movie_name'] == movie]
        for _, row in movie_df.iterrows():
            start_s = row['start_sample'] / 20000
            duration = row['duration_s']
            rect = plt.Rectangle((start_s, y_pos - 0.35), duration, 0.7,
                                  facecolor=color_map[movie], edgecolor='black',
                                  linewidth=0.5, alpha=0.8)
            ax6.add_patch(rect)
        y_pos += 1
    
    ax6.set_xlim(0, df_sample['end_sample'].max() / 20000 * 1.02)
    ax6.set_ylim(-0.5, len(movies) - 0.5)
    ax6.set_yticks(range(len(movies)))
    ax6.set_yticklabels(sorted(movies), fontsize=8)
    ax6.set_xlabel('Time (seconds)', fontsize=10)
    ax6.set_title(f'Trial Timeline: {sample_dataset}', fontweight='bold')
    ax6.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Trial Length Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(PLOT_DIR / "00_summary_dashboard.png", dpi=150, bbox_inches='tight')
    print(f"Saved: 00_summary_dashboard.png")
    plt.close()


def main():
    print("=" * 60)
    print("Generating Trial Visualization Plots")
    print("=" * 60)
    
    # Load data
    summary_df, full_df, length_summary_df = load_data()
    print(f"\nLoaded data:")
    print(f"  - Summary: {len(summary_df)} rows")
    print(f"  - Full: {len(full_df)} rows")
    
    # Generate plots
    print(f"\nGenerating plots in: {PLOT_DIR}")
    
    plot_summary_dashboard(summary_df, full_df)
    plot_trials_per_stimulus(full_df)
    plot_duration_distribution(summary_df)
    plot_sample_variation(summary_df)
    plot_trial_timeline(full_df)
    plot_consistency_heatmap(summary_df)
    plot_duration_by_recording(full_df)
    
    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print(f"Output directory: {PLOT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

