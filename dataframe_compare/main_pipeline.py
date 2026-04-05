"""
Master Pipeline Runner for Blocker Before/After Comparison.

Orchestrates all 9 pipeline steps in the correct order, with CLI control
over which steps to run, experiment selection, and per-step arguments.

See PIPELINE.md for full documentation.

Usage:
    python main_pipeline.py                          # run all steps
    python main_pipeline.py --steps 1 2 3            # specific steps
    python main_pipeline.py --steps 4-9              # range of steps
    python main_pipeline.py --experiment _ptx         # override experiment
    python main_pipeline.py --dry-run                 # show plan only
    python main_pipeline.py --steps 1 --s1 "--end 2"  # pass args to step 1
"""

import argparse
import importlib
import os
import shlex
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

@dataclass
class StepInfo:
    number: int
    name: str
    module: str
    description: str
    has_argparse: bool


STEPS = OrderedDict([
    (1, StepInfo(1, "compare",        "pipeline_compare",                 "Build pair index, firing rates, features, merge, intensity GB", True)),
    (2, StepInfo(2, "classify",       "classify_blocker",                 "Classify cells into RGC subtypes via AE+DEC models",           True)),
    (3, StepInfo(3, "improve_onh",    "spatial_improve_onh",              "Robust ONH detection and improved coordinates",                False)),
    (4, StepInfo(4, "spatial_plots",  "spatial_plots_compare",            "Hexbin + GAM triptych plots (all-cells and per-group)",        False)),
    (5, StepInfo(5, "spatial_quant",  "spatial_quantification_compare",   "Spatial quantification (gradient, Moran, radial, FDR)",        False)),
    (6, StepInfo(6, "radial_center",  "spatial_radial_center",            "Optimal radial center search on hexbin data",                  False)),
    (7, StepInfo(7, "viz_quant",      "spatial_visualize_quant",          "Visualize spatial quantification results",                     False)),
    (8, StepInfo(8, "viz_radial",     "spatial_visualize_radial",         "Visualize radial center analysis results",                     False)),
    (9, StepInfo(9, "validation",     "plot_step_up_validation",          "Step-up response validation plots (Ref vs Before vs After)",   False)),
])


def parse_step_tokens(tokens: List[str]) -> List[int]:
    """Parse step tokens that may be integers or ranges like '4-9'."""
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
    parser = argparse.ArgumentParser(
        description="Master pipeline runner for blocker comparison analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_build_epilog(),
    )
    parser.add_argument(
        "--steps", nargs="+", default=None, metavar="N",
        help="Steps to run (integers or ranges like 4-9). Default: all.",
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Experiment profile name (e.g. _ptx_str, _ptx, _str). "
             "Overrides BLOCKER_EXPERIMENT env var and specific_config default.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print execution plan without running any steps.",
    )
    for num, info in STEPS.items():
        if info.has_argparse:
            parser.add_argument(
                f"--s{num}", type=str, default="", metavar="ARGS",
                help=f"Extra CLI args for step {num} ({info.module}), "
                     f'e.g. --s{num} "--end 2"',
            )
    return parser


def _build_epilog() -> str:
    lines = ["Steps:", ""]
    for num, info in STEPS.items():
        lines.append(f"  {num}  {info.name:<16s} {info.description}")
    lines.append("")
    lines.append("Examples:")
    lines.append("  python main_pipeline.py --steps 1-3")
    lines.append("  python main_pipeline.py --steps 4 5 6 7 8")
    lines.append('  python main_pipeline.py --steps 1 --s1 "--end 2 --start-step 3"')
    lines.append("  python main_pipeline.py --experiment _ptx --dry-run")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step execution
# ---------------------------------------------------------------------------

def run_step(info: StepInfo, extra_args: str = "") -> None:
    """Import a step module and call its main(), with sys.argv patching."""
    saved_argv = sys.argv[:]
    try:
        if extra_args:
            sys.argv = [info.module + ".py"] + shlex.split(extra_args)
        else:
            sys.argv = [info.module + ".py"]

        mod = importlib.import_module(info.module)
        importlib.reload(mod)
        mod.main()
    finally:
        sys.argv = saved_argv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    # --- Resolve experiment early (before any config import) ---
    if args.experiment:
        os.environ["BLOCKER_EXPERIMENT"] = args.experiment

    # --- Resolve steps ---
    if args.steps is None:
        steps_to_run = list(STEPS.keys())
    else:
        steps_to_run = parse_step_tokens(args.steps)

    # --- Print banner ---
    experiment_label = (
        args.experiment
        or os.environ.get("BLOCKER_EXPERIMENT")
        or "(default from specific_config.py)"
    )
    print("=" * 80)
    print("  BLOCKER COMPARISON -- MASTER PIPELINE")
    print("=" * 80)
    print(f"  Experiment : {experiment_label}")
    print(f"  Steps      : {steps_to_run}")
    print(f"  Dry run    : {args.dry_run}")
    print()

    for num in steps_to_run:
        info = STEPS[num]
        extra = getattr(args, f"s{num}", "") if info.has_argparse else ""
        tag = f"[Step {num}/{max(steps_to_run)}]"
        print(f"  {tag}  {info.name:<16s}  {info.description}")
        if extra:
            print(f"          args: {extra}")

    print("=" * 80)

    if args.dry_run:
        print("\n  Dry run -- nothing executed.\n")
        return

    # --- Execute ---
    t_total = time.time()
    failed = []

    for num in steps_to_run:
        info = STEPS[num]
        extra = getattr(args, f"s{num}", "") if info.has_argparse else ""

        header = f"Step {num}/{max(steps_to_run)}: {info.name}"
        print("\n" + "#" * 80)
        print(f"#  {header}")
        print("#" * 80 + "\n")

        t0 = time.time()
        try:
            run_step(info, extra)
            elapsed = time.time() - t0
            print(f"\n  >> Step {num} completed in {elapsed:.1f}s")
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"\n  >> Step {num} FAILED after {elapsed:.1f}s: {exc}")
            failed.append((num, str(exc)))
            # Continue with remaining steps that don't depend on this one
            continue

    # --- Summary ---
    total_elapsed = time.time() - t_total
    print("\n" + "=" * 80)
    print("  PIPELINE SUMMARY")
    print("=" * 80)
    print(f"  Total time : {total_elapsed:.1f}s")
    print(f"  Steps run  : {len(steps_to_run)}")
    if failed:
        print(f"  FAILED     : {len(failed)}")
        for num, msg in failed:
            print(f"    Step {num}: {msg}")
    else:
        print("  Status     : ALL PASSED")
    print("=" * 80 + "\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
