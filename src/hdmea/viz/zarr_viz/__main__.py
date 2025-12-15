"""
CLI entry point for zarr_viz module.

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
        description="Interactive Zarr Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m hdmea.viz.zarr_viz                    # Launch with file picker
  python -m hdmea.viz.zarr_viz /path/to/data.zarr # Open specific archive
        """,
    )

    parser.add_argument(
        "zarr_path",
        nargs="?",
        default=None,
        help="Path to zarr archive (optional - shows file picker if not provided)",
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
    if args.zarr_path:
        path = Path(args.zarr_path)
        if not path.exists():
            print(f"Error: Path does not exist: {path}", file=sys.stderr)
            sys.exit(1)
        if not path.is_dir():
            print(f"Error: Path is not a directory: {path}", file=sys.stderr)
            sys.exit(1)

    # Print startup message
    print("=" * 60)
    print("  🔬 Zarr Visualization Tool")
    print("=" * 60)
    if args.zarr_path:
        print(f"  📁 Opening: {args.zarr_path}")
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

    # Add zarr path as script argument if provided
    if args.zarr_path:
        st_args.extend(["--", args.zarr_path])

    sys.argv = st_args
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
