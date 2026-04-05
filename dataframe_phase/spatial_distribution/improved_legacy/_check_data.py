import pandas as pd, numpy as np

enriched = pd.read_parquet(r'm:\Python_Project\Data_Processing_2027\dataframe_phase\spatial_distribution\results\labeled_dataframe_enriched.parquet')
print('Enriched columns:', [c for c in enriched.columns if any(k in c for k in ['ap_','soma_','center_xy','angle_corr'])])
print('Shape:', enriched.shape)

# Check pathway data
print(f'\nap_slope non-null: {enriched["ap_slope"].notna().sum()}')
print(f'soma_row non-null: {enriched["soma_row"].notna().sum()}')
print(f'center_xy sample:')
for v in enriched['center_xy'].head(5):
    print(f'  "{v}"')

enriched['recording'] = enriched.index.map(lambda x: x.rsplit('_unit_', 1)[0])
pw = enriched.groupby('recording')['ap_slope'].apply(lambda x: x.notna().sum())
print(f'\nPathways per recording: mean={pw.mean():.1f}, min={pw.min()}, max={pw.max()}')
r2 = enriched['ap_r_value'].dropna()
print(f'\nR values: mean={r2.mean():.3f}, median={r2.median():.3f}')
print(f'  R^2 > 0.3: {((r2**2) > 0.3).sum()}')
print(f'  R^2 > 0.5: {((r2**2) > 0.5).sum()}')
print(f'  R^2 > 0.7: {((r2**2) > 0.7).sum()}')
