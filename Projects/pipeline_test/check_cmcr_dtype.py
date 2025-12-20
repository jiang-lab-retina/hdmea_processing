"""
Temporary script to examine CMCR sensor data dtype and units.
"""

from pathlib import Path
import numpy as np

# CMCR file path
cmcr_path = Path("O:/20250410/set6/2025.04.10-11.12.57-Rec.cmcr")

print(f"Checking CMCR file: {cmcr_path}")
print(f"File exists: {cmcr_path.exists()}")
print()

# Method 1: Using McsPy (legacy approach)
print("=" * 60)
print("Method 1: McsPy (legacy approach)")
print("=" * 60)
try:
    from McsPy import McsCMOSMEA
    
    cmcr_data = McsCMOSMEA.McsData(str(cmcr_path))
    
    # Access sensor data
    sensor_stream = cmcr_data['Acquisition']['Sensor Data']["SensorData 1 1"]
    
    print(f"Type of sensor_stream: {type(sensor_stream)}")
    print(f"sensor_stream.dtype: {sensor_stream.dtype if hasattr(sensor_stream, 'dtype') else 'N/A'}")
    print(f"sensor_stream.shape: {sensor_stream.shape if hasattr(sensor_stream, 'shape') else 'N/A'}")
    
    # Check for unit/scale information
    print("\n--- Unit/Scale Information (McsPy) ---")
    attrs_to_check = ['Unit', 'unit', 'Units', 'units', 'Label', 'label', 
                      'ConversionFactor', 'conversion_factor', 'ADZero', 'Exponent',
                      'Info', 'info', 'Tick', 'tick']
    for attr in attrs_to_check:
        if hasattr(sensor_stream, attr):
            print(f"  {attr}: {getattr(sensor_stream, attr)}")
    
    # Check parent stream for metadata
    sensor_data_group = cmcr_data['Acquisition']['Sensor Data']
    print(f"\nType of sensor_data_group: {type(sensor_data_group)}")
    if hasattr(sensor_data_group, 'keys'):
        print(f"Keys in Sensor Data: {list(sensor_data_group.keys())}")
    
    # Try to get info dict
    if hasattr(sensor_stream, 'info'):
        print(f"\nSensor stream info: {sensor_stream.info}")
    if hasattr(sensor_stream, 'Info'):
        print(f"\nSensor stream Info: {sensor_stream.Info}")
        
    # Check for any attributes on the stream object
    print(f"\nAll attributes on sensor_stream:")
    for attr in dir(sensor_stream):
        if not attr.startswith('_'):
            try:
                val = getattr(sensor_stream, attr)
                if not callable(val):
                    print(f"  {attr}: {val}")
            except:
                pass
    
    # Load a small sample
    sample = sensor_stream[:1000, :, :]
    sample_np = np.array(sample)
    
    print(f"\n--- Data Values ---")
    print(f"After np.array():")
    print(f"  sample_np.dtype: {sample_np.dtype}")
    print(f"  sample_np.shape: {sample_np.shape}")
    print(f"  sample_np.min(): {sample_np.min()}")
    print(f"  sample_np.max(): {sample_np.max()}")
    print(f"  sample_np.mean(): {sample_np.mean():.2f}")
    print(f"  sample_np.std(): {sample_np.std():.2f}")
    
    # Check what astype(np.int16) does
    sample_int16 = sample_np.astype(np.int16)
    print(f"\nAfter .astype(np.int16):")
    print(f"  sample_int16.dtype: {sample_int16.dtype}")
    print(f"  sample_int16.min(): {sample_int16.min()}")
    print(f"  sample_int16.max(): {sample_int16.max()}")
    
    # Check if values are preserved
    print(f"\nValues preserved after int16 conversion: {np.allclose(sample_np.astype(np.float64), sample_int16.astype(np.float64))}")
    
except Exception as e:
    print(f"McsPy method failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Method 2: Using h5py directly
print("=" * 60)
print("Method 2: h5py direct access")
print("=" * 60)
try:
    import h5py
    
    with h5py.File(cmcr_path, 'r') as f:
        # List top-level groups
        print(f"Top-level keys: {list(f.keys())}")
        
        # Check root attributes for metadata
        print(f"\nRoot attributes:")
        for key in f.attrs.keys():
            try:
                val = f.attrs[key]
                if isinstance(val, bytes):
                    val = val.decode('utf-8', errors='ignore')
                print(f"  {key}: {val}")
            except:
                print(f"  {key}: <error reading>")
        
        # Navigate to sensor data
        paths_to_try = [
            'Acquisition/Sensor Data/SensorData 1 1',
            'Acquisition/Sensor Data/SensorData_1_1',
            'Acquisition/Sensor Data/SensorData1',
        ]
        
        for path in paths_to_try:
            if path in f:
                dataset = f[path]
                print(f"\nFound dataset at: {path}")
                print(f"  dataset.dtype: {dataset.dtype}")
                print(f"  dataset.shape: {dataset.shape}")
                print(f"  dataset.chunks: {dataset.chunks}")
                
                # Check dataset attributes for unit info
                print(f"\n  Dataset attributes:")
                for key in dataset.attrs.keys():
                    try:
                        val = dataset.attrs[key]
                        if isinstance(val, bytes):
                            val = val.decode('utf-8', errors='ignore')
                        print(f"    {key}: {val}")
                    except:
                        print(f"    {key}: <error reading>")
                
                # Check parent group attributes
                parent_path = '/'.join(path.split('/')[:-1])
                if parent_path in f:
                    parent = f[parent_path]
                    print(f"\n  Parent group ({parent_path}) attributes:")
                    for key in parent.attrs.keys():
                        try:
                            val = parent.attrs[key]
                            if isinstance(val, bytes):
                                val = val.decode('utf-8', errors='ignore')
                            print(f"    {key}: {val}")
                        except:
                            print(f"    {key}: <error reading>")
                
                # Check for InfoChannel or similar metadata groups
                acq = f['Acquisition']
                print(f"\n  All groups in /Acquisition:")
                for key in acq.keys():
                    print(f"    {key}: {type(acq[key])}")
                
                # Look for unit info in common locations
                unit_paths = [
                    'Acquisition/InfoChannel',
                    'Acquisition/Sensor Data/InfoChannel',
                    'Data/InfoChannel',
                ]
                for unit_path in unit_paths:
                    if unit_path in f:
                        print(f"\n  Found InfoChannel at: {unit_path}")
                        info_group = f[unit_path]
                        if hasattr(info_group, 'attrs'):
                            for key in info_group.attrs.keys():
                                print(f"    {key}: {info_group.attrs[key]}")
                        if hasattr(info_group, 'keys'):
                            for key in info_group.keys():
                                print(f"    Dataset: {key}")
                
                # Load small sample
                sample = dataset[:1000, :, :]
                print(f"\n  Sample loaded:")
                print(f"    sample.dtype: {sample.dtype}")
                print(f"    sample.min(): {sample.min()}")
                print(f"    sample.max(): {sample.max()}")
                print(f"    sample.mean(): {sample.mean():.2f}")
                break
        else:
            print("No sensor data found in expected paths")
            
            # List what's in Acquisition
            if 'Acquisition' in f:
                print(f"\nContents of /Acquisition: {list(f['Acquisition'].keys())}")
                if 'Sensor Data' in f['Acquisition']:
                    print(f"Contents of /Acquisition/Sensor Data: {list(f['Acquisition']['Sensor Data'].keys())}")
                    
except Exception as e:
    print(f"h5py method failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("Done!")
print("=" * 60)

