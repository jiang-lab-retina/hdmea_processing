"""
Run visualization on a quick subset of data to test it works.
This avoids re-running the full expensive pipeline.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Force reimport
for mod_name in list(sys.modules.keys()):
    if 'Baden_method' in mod_name:
        del sys.modules[mod_name]

import numpy as np
import pandas as pd

print("=" * 70)
print("VISUALIZATION-ONLY TEST")
print("=" * 70)

# Import after path setup
from dataframe_phase.classification_v2.Baden_method import config
from dataframe_phase.classification_v2.Baden_method import preprocessing
from dataframe_phase.classification_v2.Baden_method import features
from dataframe_phase.classification_v2.Baden_method import clustering
from dataframe_phase.classification_v2.Baden_method import visualization

# Load and filter data (quick)
print("\n[1/5] Loading data...")
input_path = project_root / config.INPUT_PATH
df = preprocessing.load_data(input_path)

print("\n[2/5] Filtering...")
df = preprocessing.filter_rows(df)

print("\n[3/5] Splitting and preprocessing (using sample for speed)...")
df_ds, df_nds = preprocessing.split_ds_nds(df)

# Use a smaller sample for quick testing
SAMPLE_SIZE = 500
if len(df_nds) > SAMPLE_SIZE:
    df_nds_sample = df_nds.sample(n=SAMPLE_SIZE, random_state=42)
else:
    df_nds_sample = df_nds

print(f"Using {len(df_nds_sample)} non-DS cells (sampled from {len(df_nds)})")

# Preprocess
df_nds_sample = preprocessing.preprocess_traces(df_nds_sample)

print("\n[4/5] Extracting features...")
X, feature_names, models = features.build_feature_matrix(df_nds_sample, return_models=True)
X_std, scaler = features.standardize_features(X)

print("\n[5/5] Quick clustering (k=5 only for speed)...")
from sklearn.mixture import GaussianMixture

# Just fit a simple GMM
gmm = GaussianMixture(n_components=5, covariance_type='diag', n_init=5, random_state=42)
gmm.fit(X_std)
labels = gmm.predict(X_std)
posteriors = gmm.predict_proba(X_std)

# Create mock BIC table
bic_table = pd.DataFrame({
    'k': range(1, 11),
    'bic': [1e6 - i * 50000 for i in range(10)],
    'log_bf': [np.nan] + [3.0] * 9,
})

print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

output_dir = project_root / config.OUTPUT_DIR / "plots"
output_dir.mkdir(parents=True, exist_ok=True)

# 1. BIC curve
print("\n[1/3] Generating BIC curve...")
try:
    fig = visualization.plot_bic_curve(bic_table, "test-nDS", output_dir / "test_bic.png")
    print(f"✓ Saved: {output_dir / 'test_bic.png'}")
except Exception as e:
    print(f"✗ Failed: {e}")

# 2. Posterior curves  
print("\n[2/3] Generating posterior curves...")
try:
    from dataframe_phase.classification_v2.Baden_method import evaluation
    curves = evaluation.compute_posterior_curves(labels, posteriors)
    fig = visualization.plot_posterior_curves(curves, "test-nDS", output_dir / "test_posterior.png")
    print(f"✓ Saved: {output_dir / 'test_posterior.png'}")
except Exception as e:
    print(f"✗ Failed: {e}")

# 3. UMAP/PCA projection
print("\n[3/3] Generating cluster projection...")
try:
    fig = visualization.plot_umap_clusters(X_std, labels, "test-nDS", output_dir / "test_umap.png")
    print(f"✓ Saved: {output_dir / 'test_umap.png'}")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n" + "=" * 70)
print("VISUALIZATION TEST COMPLETE")
print(f"Check output directory: {output_dir}")
print("=" * 70)

