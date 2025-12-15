"""
Zarr Visualization GUI - Interactive explorer for Zarr archives.

This module provides a Streamlit-based visualization tool for exploring
and plotting data stored in Zarr archives.

Public API:
    launch: Launch the visualization tool

Example:
    >>> from hdmea.viz.zarr_viz import launch
    >>> launch("/path/to/archive.zarr")

Command Line:
    python -m hdmea.viz.zarr_viz [path]
"""

from pathlib import Path
from typing import Union

__all__ = ["launch"]


def launch(
    zarr_path: Union[str, Path, None] = None,
    port: int = 8501,
    open_browser: bool = True,
) -> None:
    """Launch the Zarr visualization tool.

    Args:
        zarr_path: Optional path to a zarr archive. If not provided,
                   the tool will show a file picker interface.
        port: Port for the Streamlit server (default: 8501).
        open_browser: Whether to open browser automatically (default: True).

    Raises:
        ImportError: If streamlit is not installed.
        FileNotFoundError: If zarr_path is provided but doesn't exist.
    """
    try:
        import streamlit.web.cli as stcli
    except ImportError as e:
        raise ImportError(
            "Streamlit is required for the visualization tool. "
            "Install with: pip install hdmea[viz]"
        ) from e

    import sys

    # Validate path if provided
    if zarr_path is not None:
        path = Path(zarr_path)
        if not path.exists():
            raise FileNotFoundError(f"Zarr path does not exist: {zarr_path}")

    # Get the path to app.py
    app_path = Path(__file__).parent / "app.py"

    # Build streamlit arguments
    args = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port", str(port),
        "--server.headless", str(not open_browser).lower(),
        "--browser.gatherUsageStats", "false",
    ]

    if zarr_path is not None:
        args.extend(["--", str(zarr_path)])

    # Set up sys.argv for streamlit
    sys.argv = args

    # Run streamlit
    sys.exit(stcli.main())
