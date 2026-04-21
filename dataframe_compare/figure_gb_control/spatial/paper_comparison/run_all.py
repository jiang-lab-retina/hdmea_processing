"""
Single entry point: run all fig1..fig7 paper-comparison scripts in order.

Usage
-----
    python run_all.py

Each script imports `_common` and is otherwise independent, so they can
also be run individually.
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path
import sys

SCRIPTS = [
    "fig1_opsin_gradient",
    "fig2_spatial_maps",
    "fig3_dv_gradient",
    "fig4_opponency_map",
    "fig5_group_specific",
    "fig5b_subtype_specific",
    "fig6_contrast_breakdown",
    "fig7_peak_response_maps",
    "fig8_sustained_opponency",
    "fig9_threshold_on_ratio",
    "fig10_fullfield_opponency",
]


def main() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    total_t0 = time.time()
    for name in SCRIPTS:
        print(f"\n========== {name} ==========")
        t0 = time.time()
        mod = importlib.import_module(name)
        mod.main()
        print(f"[{name}] elapsed {time.time() - t0:.1f}s")
    print(f"\nAll scripts done in {time.time() - total_t0:.1f}s total")


if __name__ == "__main__":
    main()
