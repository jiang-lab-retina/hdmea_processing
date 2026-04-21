"""
Run the full pipeline chain for a single experiment:
  1. alignment.py        (upstream - skip existing)
  2. batch_update.py     (upstream - skip existing)
  3. main_pipeline.py    (all 9 dataframe_compare steps)

Usage:
    python docs/_run_full_chain.py --experiment _ptx_str
    python docs/_run_full_chain.py --experiment _ptx
    python docs/_run_full_chain.py --experiment _str
"""
import subprocess
import sys
import time
from pathlib import Path

PROJ_ROOT = Path(r"m:\Python_Project\Data_Processing_2027")
UPSTREAM_DIR = PROJ_ROOT / "Projects" / "unified_special_pipeline" / "blocker_alignment_analysis"
COMPARE_DIR = PROJ_ROOT / "dataframe_compare"


def run_step(label, cmd, cwd):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  cmd: {' '.join(str(c) for c in cmd)}")
    print(f"  cwd: {cwd}")
    print(f"{'='*70}\n", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(cwd))
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n  >> {label}: {status}  ({elapsed/60:.1f} min)\n", flush=True)
    return result.returncode


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--skip-upstream", action="store_true",
                        help="Skip alignment + batch_update, run only dataframe_compare")
    args = parser.parse_args()

    exp = args.experiment
    print(f"\n{'#'*70}")
    print(f"  FULL CHAIN for experiment: {exp}")
    print(f"{'#'*70}\n", flush=True)

    overall_t0 = time.time()

    if not args.skip_upstream:
        # Step A: alignment.py
        rc = run_step(
            f"[{exp}] alignment.py",
            [sys.executable, "alignment.py", "--experiment", exp],
            UPSTREAM_DIR,
        )
        if rc != 0:
            print(f"  !! alignment failed for {exp}, stopping chain")
            sys.exit(1)

        # Step B: batch_update.py
        rc = run_step(
            f"[{exp}] batch_update.py",
            [sys.executable, "batch_update.py", "--experiment", exp],
            UPSTREAM_DIR,
        )
        if rc != 0:
            print(f"  !! batch_update failed for {exp}, stopping chain")
            sys.exit(1)

    # Step C: main_pipeline.py (all 9 steps)
    rc = run_step(
        f"[{exp}] main_pipeline.py (all 9 steps)",
        [sys.executable, "main_pipeline.py", "--experiment", exp],
        COMPARE_DIR,
    )
    if rc != 0:
        print(f"  !! main_pipeline failed for {exp}")
        sys.exit(1)

    elapsed = time.time() - overall_t0
    print(f"\n{'#'*70}")
    print(f"  FULL CHAIN COMPLETE for {exp}: {elapsed/60:.1f} min total")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
