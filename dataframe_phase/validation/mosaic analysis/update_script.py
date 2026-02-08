"""Temporary script to update mosaic_validation.py"""
import re

path = 'm:/Python_Project/Data_Processing_2027/dataframe_phase/classification_v2/divide_conquer_method/validation/mosaic_validation.py'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern to find and replace the RF area calculation
old_pattern = r"(        cells_df = df\.loc\[original_indices\]\n        rf_areas = calculate_rf_areas_for_cells\(cells_df\)\n        \n)        total_rf_area_mm2 = rf_areas\.sum\(\)\n        n_cells = len\(cells_df\)"

new_text = r"\1        n_cells = len(cells_df)\n        median_rf_area_mm2 = rf_areas.median()\n        # Calculate total RF area as median x cell count (robust to outliers)\n        total_rf_area_mm2 = median_rf_area_mm2 * n_cells"

content = re.sub(old_pattern, new_text, content)

# Also add median_rf_area_mm2 to the results dict
old_results = "'n_cells': n_cells,\n            'total_rf_area_mm2': total_rf_area_mm2,"
new_results = "'n_cells': n_cells,\n            'median_rf_area_mm2': median_rf_area_mm2,\n            'total_rf_area_mm2': total_rf_area_mm2,"

content = content.replace(old_results, new_results)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Updated mosaic_validation.py to use median x cell_count')
