"""
CLI entry point for zarr_viz module (supports HDF5 and Zarr).

Allows launching the visualization tool from command line:
    python -m hdmea.viz.zarr_viz [path]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="zarr_viz",
        description="Interactive HDF5/Zarr Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m hdmea.viz.zarr_viz                     # Launch with file picker
  python -m hdmea.viz.zarr_viz /path/to/data.h5    # Open HDF5 file
  python -m hdmea.viz.zarr_viz /path/to/data.zarr  # Open Zarr archive (legacy)
        """,
    )

    parser.add_argument(
        "file_path",
        nargs="?",
        default=None,
        help="Path to HDF5 file (.h5) or zarr archive (.zarr) (optional)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for Streamlit server (default: 8501)",
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )

    args = parser.parse_args()

    # Validate path if provided
    if args.file_path:
        path = Path(args.file_path)
        if not path.exists():
            print(f"Error: Path does not exist: {path}", file=sys.stderr)
            sys.exit(1)
        # HDF5 files are files, Zarr archives are directories
        if path.suffix.lower() in ('.h5', '.hdf5', '.hdf'):
            if not path.is_file():
                print(f"Error: HDF5 path is not a file: {path}", file=sys.stderr)
                sys.exit(1)
        else:
            if not path.is_dir():
                print(f"Error: Zarr path is not a directory: {path}", file=sys.stderr)
                sys.exit(1)

    # Print startup message
    print("=" * 60)
    print("  🔬 HDF5/Zarr Visualization Tool")
    print("=" * 60)
    if args.file_path:
        print(f"  📁 Opening: {args.file_path}")
    else:
        print("  📁 No path provided - file picker will be shown")
    print(f"  🌐 Starting on port {args.port}")
    print("=" * 60)
    print()

    # Launch streamlit
    try:
        import streamlit.web.cli as stcli
    except ImportError:
        print(
            "Error: Streamlit is not installed.\n"
            "Install with: pip install hdmea[viz]",
            file=sys.stderr,
        )
        sys.exit(1)

    # Get path to app.py
    app_path = Path(__file__).parent / "app.py"

    # Build streamlit arguments
    st_args = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port", str(args.port),
        "--server.headless", str(args.no_browser).lower(),
        "--browser.gatherUsageStats", "false",
    ]

    # Add file path as script argument if provided
    if args.file_path:
        st_args.extend(["--", args.file_path])

    sys.argv = st_args
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
