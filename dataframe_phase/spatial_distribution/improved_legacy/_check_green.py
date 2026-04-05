import pandas as pd, numpy as np

df = pd.read_parquet(r'm:\Python_Project\Data_Processing_2027\dataframe_phase\spatial_distribution\results\labeled_dataframe_with_legacy_coords_freq.parquet')
g = df['green_on_peak_extreme'].dropna()

print(f"n = {len(g)}")
print(f"min = {g.min():.4f}")
print(f"max = {g.max():.4f}")
print(f"mean = {g.mean():.4f}")
print(f"median = {g.median():.4f}")
print(f"std = {g.std():.4f}")
print()
print("Percentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th: {np.percentile(g, p):.4f}")
print()
print(f"> 0:  {(g > 0).sum()} ({100*(g > 0).mean():.1f}%)")
print(f"== 0: {(g == 0).sum()} ({100*(g == 0).mean():.1f}%)")
print(f"< 0:  {(g < 0).sum()} ({100*(g < 0).mean():.1f}%)")
