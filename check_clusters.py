import json
import pandas as pd
import numpy as np
from pathlib import Path

results_dir = Path("m:/Python_Project/Data_Processing_2027/dataframe_phase/classification_v2/divide_conquer_method/results")

for group in ["ipRGC", "DSGC", "OSGC", "Other"]:
    k_path = results_dir / group / "k_selection.json"
    if k_path.exists():
        with open(k_path) as f:
            k_data = json.load(f)
        print("\n=== " + group + " k_selection ===")
        print(json.dumps(k_data, indent=2)[:500])
    ca_path = results_dir / group / "cluster_assignments.parquet"
    if ca_path.exists():
        ca = pd.read_parquet(ca_path)
        print("  dec_cluster unique values:", sorted(ca["dec_cluster"].unique()))
        print("  k (from dec_cluster) =", ca["dec_cluster"].nunique())
    emb_path = results_dir / group / "embeddings_dec_refined.parquet"
    if emb_path.exists():
        emb = pd.read_parquet(emb_path)
        print("  embeddings shape:", emb.shape)
        print("  embedding columns:", list(emb.columns)[:5], "...")

print("\n\n=== DEC Model State Dict Keys ===")
import torch
for group in ["DSGC"]:
    pt_path = Path("m:/Python_Project/Data_Processing_2027/dataframe_phase/classification_v2/divide_conquer_method/models_saved/" + group + "/dec_refined.pt")
    if pt_path.exists():
        state = torch.load(pt_path, map_location="cpu")
        print("\n" + group + " dec_refined.pt keys:")
        for k, v in state.items():
            if hasattr(v, "shape"):
                print("  " + k + ":", v.shape)
            else:
                print("  " + k + ":", type(v))

ae_path = Path("m:/Python_Project/Data_Processing_2027/dataframe_phase/classification_v2/divide_conquer_method/models_saved/DSGC/autoencoder_best.pt")
if ae_path.exists():
    ae_state = torch.load(ae_path, map_location="cpu")
    if isinstance(ae_state, dict) and "state_dict" in ae_state:
        print("\nAutoencoder checkpoint is a dict with keys:", list(ae_state.keys()))
        if "segment_lengths" in ae_state:
            print("  segment_lengths:", ae_state["segment_lengths"])
    else:
        print("\nAutoencoder checkpoint type:", type(ae_state))
        if isinstance(ae_state, dict):
            print("  Keys (first 10):", list(ae_state.keys())[:10])