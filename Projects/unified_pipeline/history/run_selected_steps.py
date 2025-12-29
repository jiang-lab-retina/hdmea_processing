#!/usr/bin/env python
"""
Unified Pipeline: Run Selected Steps

Loads an existing HDF5 file and runs selected pipeline steps sequentially,
then saves the result to a new HDF5 file.

This script demonstrates the deferred-save pattern where:
1. Session is loaded from HDF5 into memory (PipelineSession)
2. Selected steps are executed sequentially, modifying the session in memory
3. Session is saved once to a new HDF5 file

Available steps:
    - section_time: Add section timing from playlist CSV
    - section_time_analog: Add section timing from analog signal (raw_ch1)
    - section_spike_times: Section spike times based on stimulus timing
    - section_spike_times_analog: Section spike times using sample indices (no frame conversion)

Usage:
    python run_selected_steps.py
    python run_selected_steps.py --steps section_time_analog section_spike_times_analog
    python run_selected_steps.py --input path/to/file.h5 --output path/to/output.h5
    python run_selected_steps.py --force-steps section_time_analog section_spike_times_analog
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from hdmea.pipeline import load_session_from_hdf5, PipelineSession

from Projects.unified_pipeline.steps import (
    add_section_time_step,
    add_section_time_analog_step,
    section_spike_times_step,
    section_spike_times_analog_step,
)
from Projects.unified_pipeline.config import (
    setup_logging,
    SectionTimeConfig,
    SectionTimeAnalogConfig,
    DEFAULT_OUTPUT_DIR,
    green_success,
    red_warning,
)


# =============================================================================
# Default Paths
# =============================================================================

# Default input file for testing
DEFAULT_INPUT_FILE = (
    project_root / "Projects/unified_pipeline/export_all_steps/2024.02.26-10.53.19-Rec.h5"
)

# Default output directory
OUTPUT_DIR = DEFAULT_OUTPUT_DIR

# Available steps
AVAILABLE_STEPS = [
    "section_time",
    "section_time_analog",
    "section_spike_times",
    "section_spike_times_analog",
]


# =============================================================================
# Pipeline Execution
# =============================================================================

def run_selected_steps(
    input_path: Path,
    output_path: Path,
    steps: List[str],
    section_time_config: SectionTimeConfig,
    section_time_analog_config: SectionTimeAnalogConfig,
    pad_margin: tuple,
    force_steps: List[str] = None,
) -> Path:
    """
    Load HDF5, run selected steps sequentially, and save result.
    
    Args:
        input_path: Path to input HDF5 file
        output_path: Path to save output HDF5 file
        steps: List of step names to run
        section_time_config: Config for section_time step
        section_time_analog_config: Config for section_time_analog step
        pad_margin: Padding margins for section_spike_times step
        force_steps: List of step names to force re-run (removes from completed_steps)
    
    Returns:
        Path to output HDF5 file
    """
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    
    print("=" * 70)
    print("Unified Pipeline: Run Selected Steps")
    print("=" * 70)
    print(f"Input:    {input_path}")
    print(f"Output:   {output_path}")
    print(f"Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Steps:    {', '.join(steps)}")
    if force_steps:
        print(f"Force:    {', '.join(force_steps)}")
    print("=" * 70)
    
    # =========================================================================
    # Load session from HDF5
    # =========================================================================
    logger.info("Loading session from HDF5...")
    session = load_session_from_hdf5(input_path)
    logger.info(f"  Loaded {session.unit_count} units, dataset_id={session.dataset_id}")
    
    # =========================================================================
    # Force re-run: remove steps from completed_steps
    # =========================================================================
    if force_steps:
        # Map step names to their internal step marker names
        step_markers = {
            "section_time": "add_section_time",
            "section_time_analog": "add_section_time_analog",
            "section_spike_times": "section_spike_times",
            "section_spike_times_analog": "section_spike_times_analog",
        }
        for step_name in force_steps:
            marker = step_markers.get(step_name, step_name)
            # Remove exact match
            if marker in session.completed_steps:
                session.completed_steps.discard(marker)
                logger.info(f"  Forcing re-run of {step_name} (removed '{marker}' from completed_steps)")
            # Also remove any movie-specific markers (e.g., "section_spike_times_analog:iprgc_test")
            to_remove = [s for s in session.completed_steps if s.startswith(f"{marker}:")]
            for s in to_remove:
                session.completed_steps.discard(s)
                logger.info(f"  Forcing re-run of {step_name} (removed '{s}' from completed_steps)")
    
    # =========================================================================
    # Run selected steps sequentially
    # =========================================================================
    for step_name in steps:
        print(f"\n--- Running step: {step_name} ---")
        
        if step_name == "section_time":
            logger.info(f"Config: playlist_name={section_time_config.playlist_name}")
            session = add_section_time_step(
                playlist_name=section_time_config.playlist_name,
                session=session,
            )
        
        elif step_name == "section_time_analog":
            logger.info(f"Config: threshold={section_time_analog_config.threshold_value}, "
                       f"movie={section_time_analog_config.movie_name}")
            session = add_section_time_analog_step(
                config=section_time_analog_config,
                session=session,
            )
        
        elif step_name == "section_spike_times":
            logger.info(f"Config: pad_margin={pad_margin}")
            session = section_spike_times_step(
                pad_margin=pad_margin,
                session=session,
            )
        
        elif step_name == "section_spike_times_analog":
            logger.info(f"Config: movie_name={section_time_analog_config.movie_name}, pad_margin={section_time_analog_config.pad_margin}")
            session = section_spike_times_analog_step(
                movie_name=section_time_analog_config.movie_name,
                pad_margin=section_time_analog_config.pad_margin,
                session=session,
            )
        
        else:
            logger.warning(f"Unknown step: {step_name}, skipping")
    
    # =========================================================================
    # Save Result
    # =========================================================================
    logger.info("Saving result to HDF5...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    session.save(output_path=output_path, overwrite=True)
    
    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time
    
    _print_summary(session, output_path, elapsed, steps)
    
    return output_path


def _print_summary(session: PipelineSession, output_path: Path, elapsed: float, steps: List[str]) -> None:
    """Print execution summary."""
    print("\n" + "=" * 70)
    print(green_success("PIPELINE COMPLETE"))
    print("=" * 70)
    print(f"Output:   {output_path}")
    print(f"Units:    {session.unit_count}")
    print(f"Steps run: {len(steps)} ({', '.join(steps)})")
    print(f"Total steps: {len(session.completed_steps)}")
    print(f"Time:     {elapsed:.1f}s")
    
    if session.warnings:
        print(f"\nWarnings ({len(session.warnings)}):")
        for warning in session.warnings[:5]:
            print(f"  - {warning}")
        if len(session.warnings) > 5:
            print(f"  ... and {len(session.warnings) - 5} more")
    
    print("\nCompleted steps:")
    for step in sorted(session.completed_steps):
        status = "[!]" if ":skipped" in step or ":failed" in step else "[+]"
        print(f"  {status} {step}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run selected pipeline steps sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run analog steps (default)
  python run_selected_steps.py
  
  # Run specific steps
  python run_selected_steps.py --steps section_time_analog section_spike_times_analog
  
  # Force re-run specific steps
  python run_selected_steps.py --force-steps section_time_analog section_spike_times_analog
  
  # Run with custom threshold
  python run_selected_steps.py --threshold 1e5 --movie-name iprgc_test
        """
    )
    
    # Common arguments
    parser.add_argument(
        "--steps", "-s", type=str, nargs="+", choices=AVAILABLE_STEPS,
        default=["section_time_analog", "section_spike_times_analog"],
        help=f"Steps to run (default: section_time_analog section_spike_times_analog)"
    )
    parser.add_argument(
        "--input", "-i", type=Path, default=DEFAULT_INPUT_FILE,
        help=f"Path to input HDF5 file (default: {DEFAULT_INPUT_FILE.name})"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output path (default: test_output/<dataset_id>_selected.h5)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--force-steps", type=str, nargs="+", choices=AVAILABLE_STEPS, default=None,
        help="Force re-run these steps even if already completed"
    )
    
    # section_time arguments
    parser.add_argument(
        "--playlist", "-p", type=str, default="play_optimization_set6_ipRGC_manual",
        help="Playlist name for section_time step (default: play_optimization_set6_ipRGC_manual)"
    )
    
    # section_spike_times arguments
    parser.add_argument(
        "--pad-before", type=float, default=0.0,
        help="Padding before section in seconds for section_spike_times (default: 0.0)"
    )
    parser.add_argument(
        "--pad-after", type=float, default=0.0,
        help="Padding after section in seconds for section_spike_times (default: 0.0)"
    )
    
    # section_time_analog arguments
    parser.add_argument(
        "--threshold", "-t", type=float, default=SectionTimeAnalogConfig.threshold_value,
        help=f"Peak detection threshold for section_time_analog (default: {SectionTimeAnalogConfig.threshold_value})"
    )
    parser.add_argument(
        "--movie-name", "-m", type=str, default="iprgc_test",
        help="Movie name identifier for section_time_analog (default: iprgc_test)"
    )
    parser.add_argument(
        "--plot-duration", "-d", type=float, default=120.0,
        help="Section duration in seconds for section_time_analog (default: 120.0)"
    )
    parser.add_argument(
        "--repeat", "-r", type=int, default=None,
        help="Limit to first N trials for section_time_analog (default: all)"
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Overwrite existing section_time if exists"
    )
    
    args = parser.parse_args()
    
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    
    # Validate input
    if not args.input.exists():
        print(red_warning(f"Input file not found: {args.input}"))
        sys.exit(1)
    
    # Determine output path
    if args.output is None:
        stem = args.input.stem
        output_path = OUTPUT_DIR / f"{stem}_selected.h5"
    else:
        output_path = args.output
    
    # Create configs
    section_time_config = SectionTimeConfig(
        playlist_name=args.playlist,
    )
    
    section_time_analog_config = SectionTimeAnalogConfig(
        threshold_value=args.threshold,
        movie_name=args.movie_name,
        plot_duration=args.plot_duration,
        repeat=args.repeat,
        force=args.force,
    )
    
    pad_margin = (args.pad_before, args.pad_after)
    
    try:
        result_path = run_selected_steps(
            input_path=args.input,
            output_path=output_path,
            steps=args.steps,
            section_time_config=section_time_config,
            section_time_analog_config=section_time_analog_config,
            pad_margin=pad_margin,
            force_steps=args.force_steps,
        )
        print(f"\nSuccess! Output saved to: {result_path}")
    except Exception as e:
        print(red_warning(f"\nPipeline failed: {e}"))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

