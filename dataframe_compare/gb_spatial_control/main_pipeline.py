"""
Master Pipeline Runner for GB Control Spatial Analysis.

Orchestrates all 7 steps in the correct order:

  1. prepare_data            -- combine 3 experiments into one parquet
  2. spatial_plots           -- hexbin + GAM heatmaps (all-cells + per-group)
  3. spatial_plots_cluster   -- hexbin + GAM + step-up trace (per-cluster)
  4. spatial_quant           -- gradient, Moran, radial quantification
  5. radial_center           -- optimal radial center search
  6. viz_quant               -- quantification figures
  7. viz_radial              -- radial center figures

Usage:
    python main_pipeline.py                # run all steps
    python main_pipeline.py --steps 1 2 3  # specific steps
    python main_pipeline.py --steps 2-7    # range
    python main_pipeline.py --dry-run      # show plan only
"""

import argparse
import importlib
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import List


@dataclass
class StepInfo:
    number: int
    name: str
    module: str
    description: str


STEPS = OrderedDict([
    (1, StepInfo(1, "prepare",              "prepare_data",                "Combine 3 experiments, strip prefix, filter coords")),
    (2, StepInfo(2, "spatial_plots",        "spatial_plots",               "Hexbin + GAM heatmaps (all-cells and per-group)")),
    (3, StepInfo(3, "spatial_plots_cluster", "spatial_plots_per_cluster",  "Hexbin + GAM + step-up trace (per-cluster)")),
    (4, StepInfo(4, "spatial_quant",        "spatial_quantification",      "Spatial quantification (gradient, Moran, radial, FDR)")),
    (5, StepInfo(5, "radial_center",        "spatial_radial_center",       "Optimal radial center search on hexbin data")),
    (6, StepInfo(6, "viz_quant",            "visualize_quant",             "Visualize spatial quantification results")),
    (7, StepInfo(7, "viz_radial",           "visualize_radial",            "Visualize radial center analysis results")),
])


def parse_step_tokens(tokens: List[str]) -> List[int]:
    result = []
    for tok in tokens:
        if "-" in tok and not tok.startswith("-"):
            parts = tok.split("-", 1)
            try:
                lo, hi = int(parts[0]), int(parts[1])
                result.extend(range(lo, hi + 1))
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid step range: {tok}")
        else:
            try:
                result.append(int(tok))
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid step: {tok}")
    valid = set(STEPS.keys())
    for s in result:
        if s not in valid:
            raise argparse.ArgumentTypeError(
                f"Step {s} not in valid range {min(valid)}-{max(valid)}"
            )
    return sorted(set(result))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GB Control Spatial Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--steps", nargs="+", default=None,
        help="Steps to run (e.g. 1 2 3, 2-6, 1-3 5). Default: all.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show execution plan only, do not run.",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.steps:
        steps_to_run = parse_step_tokens(args.steps)
    else:
        steps_to_run = list(STEPS.keys())

    # Banner
    print("\n" + "=" * 70)
    print("  GB Control Spatial Analysis Pipeline")
    print("=" * 70)
    print(f"\n  Steps to run: {steps_to_run}")
    for sn in steps_to_run:
        si = STEPS[sn]
        print(f"    Step {si.number}: {si.name:20s}  {si.description}")
    print()

    if args.dry_run:
        print("  (dry-run mode -- exiting)")
        return

    overall_t0 = time.time()
    for sn in steps_to_run:
        si = STEPS[sn]
        print(f"\n{'#' * 70}")
        print(f"#  Step {si.number}/{len(STEPS)}: {si.name}")
        print(f"#  {si.description}")
        print(f"{'#' * 70}\n", flush=True)

        step_t0 = time.time()
        try:
            mod = importlib.import_module(si.module)
            mod.main()
        except Exception as e:
            print(f"\n  !! Step {si.number} ({si.name}) FAILED: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        elapsed = time.time() - step_t0
        print(f"\n  >> Step {si.number} ({si.name}): OK  ({elapsed / 60:.1f} min)")

    total = time.time() - overall_t0
    print(f"\n{'=' * 70}")
    print(f"  ALL STEPS COMPLETE: {total / 60:.1f} min total")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
