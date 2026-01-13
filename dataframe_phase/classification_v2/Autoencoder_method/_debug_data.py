"""Debug script to check data structure."""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0].rsplit('/', 2)[0])

from Autoencoder_method import config
from Autoencoder_method.data_loader import extract_trace_arrays

df = pd.read_parquet(config.INPUT_PATH)
print('Total cells:', len(df))
print()

col = config.BASELINE_TRACE_COL
print(f'Column: {col}')
first_val = df[col].iloc[0]
print(f'Type of first value: {type(first_val)}')
arr = np.array(first_val)
print(f'As array shape: {arr.shape}')
print(f'As array dtype: {arr.dtype}')

# Check if it's nested
if arr.dtype == object:
    print('Array is object dtype - checking inner elements')
    print(f'First element type: {type(arr[0])}')
    if hasattr(arr[0], '__len__'):
        print(f'First element length: {len(arr[0])}')

# Test extract_trace_arrays with subset
print()
print('Testing extract_trace_arrays on first 10 cells...')
df_small = df.head(10)
result = extract_trace_arrays(df_small, [col])
print(f'Result shape: {result[col].shape}')
print(f'Result dtype: {result[col].dtype}')
print('Success!')
