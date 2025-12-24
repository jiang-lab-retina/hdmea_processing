"""
Explore CMTR file structure and unit information using McsPy.
"""
from McsPy.McsCMOSMEA import McsCMOSMEAData
import numpy as np

cmtr_path = "O:/20250410/set6/2025.04.10-11.12.57-Rec-.cmtr"
print(f"Loading CMTR: {cmtr_path}")
print("=" * 70)

cmtr_data = McsCMOSMEAData(cmtr_path)

# =============================================================================
# 1. File-level attributes
# =============================================================================
print("\n1. FILE-LEVEL ATTRIBUTES")
print("-" * 70)
if hasattr(cmtr_data, "attrs"):
    for key in sorted(cmtr_data.attrs.keys()):
        val = cmtr_data.attrs[key]
        # Truncate long values
        val_str = str(val)
        if len(val_str) > 80:
            val_str = val_str[:80] + "..."
        print(f"  {key}: {val_str}")
else:
    print("  No attrs found on cmtr_data")

# =============================================================================
# 2. Spike Sorter structure
# =============================================================================
print("\n2. SPIKE SORTER STRUCTURE")
print("-" * 70)
spike_sorter = cmtr_data.Spike_Sorter

# List all attributes on spike_sorter
ss_attrs = [a for a in dir(spike_sorter) if not a.startswith('_')]
print(f"  Spike_Sorter attributes: {ss_attrs}")

# Check Unit_Info
print("\n3. UNIT_INFO (if available)")
print("-" * 70)
if hasattr(spike_sorter, "Unit_Info"):
    unit_info = spike_sorter.Unit_Info
    print(f"  Type: {type(unit_info)}")
    ui_attrs = [a for a in dir(unit_info) if not a.startswith('_')]
    print(f"  Attributes: {ui_attrs}")
    
    # Try to read content
    for attr in ui_attrs:
        try:
            val = getattr(unit_info, attr)
            if callable(val):
                continue
            val_str = str(val)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            print(f"    {attr}: {val_str}")
        except Exception as e:
            print(f"    {attr}: <error: {e}>")
else:
    print("  No Unit_Info found")

# =============================================================================
# 4. Explore first unit in detail
# =============================================================================
print("\n4. FIRST UNIT DETAILED EXPLORATION")
print("-" * 70)

# Find all units
unit_attrs = [a for a in dir(spike_sorter) if a.startswith("Unit_") and a != "Unit_Info"]
print(f"  Found {len(unit_attrs)} units: {unit_attrs[:5]}{'...' if len(unit_attrs) > 5 else ''}")

if unit_attrs:
    first_unit_name = sorted(unit_attrs)[0]
    unit = getattr(spike_sorter, first_unit_name)
    print(f"\n  Exploring: {first_unit_name}")
    print(f"  Type: {type(unit)}")
    
    # All public attributes
    public_attrs = [a for a in dir(unit) if not a.startswith('_')]
    print(f"  Public attributes: {public_attrs}")
    
    # Try to get each attribute's value
    print("\n  Attribute values:")
    for attr in public_attrs:
        try:
            val = getattr(unit, attr)
            if callable(val):
                # Try calling methods with no args
                if attr in ['get_peaks_timestamps', 'get_peaks_cutouts']:
                    result = val()
                    if result is not None:
                        if hasattr(result, 'shape'):
                            print(f"    {attr}(): shape={result.shape}, dtype={result.dtype}")
                        elif hasattr(result, '__len__'):
                            print(f"    {attr}(): len={len(result)}, type={type(result)}")
                        else:
                            print(f"    {attr}(): {type(result)}")
                else:
                    print(f"    {attr}: <method>")
            elif hasattr(val, 'shape'):
                # numpy array or similar
                print(f"    {attr}: shape={val.shape}, dtype={val.dtype}")
                if val.size <= 10:
                    print(f"      values: {val}")
            elif hasattr(val, '__len__') and not isinstance(val, str):
                print(f"    {attr}: len={len(val)}, type={type(val).__name__}")
                if len(val) <= 5:
                    print(f"      values: {val}")
            else:
                val_str = str(val)
                if len(val_str) > 80:
                    val_str = val_str[:80] + "..."
                print(f"    {attr}: {val_str}")
        except Exception as e:
            print(f"    {attr}: <error: {e}>")
    
    # Check for attrs dict
    print("\n  Checking for .attrs dict:")
    if hasattr(unit, 'attrs'):
        attrs_dict = unit.attrs
        if hasattr(attrs_dict, 'keys'):
            for key in attrs_dict.keys():
                val = attrs_dict[key]
                val_str = str(val)
                if len(val_str) > 80:
                    val_str = val_str[:80] + "..."
                print(f"    {key}: {val_str}")
        else:
            print(f"    attrs = {attrs_dict}")
    else:
        print("    No .attrs dict found")

# =============================================================================
# 5. Summary statistics
# =============================================================================
print("\n5. SUMMARY STATISTICS")
print("-" * 70)
spike_counts = []
for unit_name in unit_attrs:
    try:
        u = getattr(spike_sorter, unit_name)
        if hasattr(u, 'get_peaks_timestamps'):
            ts = u.get_peaks_timestamps()
            if ts is not None:
                spike_counts.append(len(ts))
    except Exception:
        pass

if spike_counts:
    print(f"  Total units: {len(spike_counts)}")
    print(f"  Total spikes: {sum(spike_counts):,}")
    print(f"  Spikes per unit: min={min(spike_counts)}, max={max(spike_counts)}, mean={np.mean(spike_counts):.1f}")

print("\n" + "=" * 70)
print("Done!")
