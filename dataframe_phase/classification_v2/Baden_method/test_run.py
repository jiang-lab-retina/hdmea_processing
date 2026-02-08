"""Test script for Baden-method pipeline with validation."""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Force reimport of the module (clear any cached versions)
for mod_name in list(sys.modules.keys()):
    if 'Baden_method' in mod_name or 'dataframe_phase' in mod_name:
        del sys.modules[mod_name]

print(f"Project root: {project_root}")
print(f"Python path includes: {sys.path[0]}")
print()
1000.
if __name__ == "__main__":
    print("=" * 70)
    print("BADEN-METHOD PIPELINE VALIDATION TEST")
    print("=" * 70)
    
    # Step 1: Test imports
    print("\n[1/5] Testing imports...")
    try:
        from dataframe_phase.classification_v2.Baden_method import run_baden_pipeline
        from dataframe_phase.classification_v2.Baden_method import config
        print("[OK] Imports successful")
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        sys.exit(1)
    
    # Step 2: Verify input file exists
    print("\n[2/5] Checking input file...")
    # config.INPUT_PATH is already absolute
    input_path = config.INPUT_PATH
    if input_path.exists():
        print(f"[OK] Input file exists: {input_path}")
    else:
        print(f"[FAIL] Input file not found: {input_path}")
        sys.exit(1)
    
    # Step 3: Test data loading
    print("\n[3/5] Testing data loading...")
    try:
        from dataframe_phase.classification_v2.Baden_method import preprocessing
        df = preprocessing.load_data(input_path)
        print(f"[OK] Loaded {len(df)} cells")
        print(f"  Columns: {len(df.columns)}")
    except Exception as e:
        print(f"[FAIL] Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Test preprocessing (often where errors occur)
    print("\n[4/5] Testing preprocessing filter...")
    try:
        df_filtered = preprocessing.filter_rows(df)
        print(f"[OK] Filtered to {len(df_filtered)} cells")
    except Exception as e:
        print(f"[FAIL] Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 5: Run full pipeline
    print("\n[5/5] Running full pipeline (with bootstrap stability)...")
    print("-" * 70)
    try:
        results = run_baden_pipeline(
            input_path=input_path,
            # run_bootstrap uses config.RUN_BOOTSTRAP by default
        )
        print("-" * 70)
        print("\n[OK] PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"\nResults:")
        print(f"  DS clusters: {results['ds'].optimal_k}")
        print(f"  non-DS clusters: {results['nds'].optimal_k}")
        
        # Bootstrap stability results
        ds_stability = results['ds'].evaluation_metrics.get('bootstrap_median_correlation')
        nds_stability = results['nds'].evaluation_metrics.get('bootstrap_median_correlation')
        print(f"  DS bootstrap stability: {ds_stability:.3f}" if ds_stability else "  DS bootstrap stability: N/A")
        print(f"  non-DS bootstrap stability: {nds_stability:.3f}" if nds_stability else "  non-DS bootstrap stability: N/A")
        
        print(f"  Output directory: {results['run_info']['output_dir']}")
    except Exception as e:
        print(f"\n[FAIL] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
