"""
Streamlit application entry point for Zarr Visualization GUI.

This module provides the main Streamlit application for exploring
and visualizing Zarr archives.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, TYPE_CHECKING

# Add project src to path for Streamlit execution (runs file directly, not as module)
# When Streamlit runs app.py directly, relative imports fail because there's no parent package.
# We need to add the 'src' directory to sys.path so absolute imports work.
_APP_DIR = Path(__file__).resolve().parent  # src/hdmea/viz/zarr_viz
_SRC_DIR = _APP_DIR.parent.parent.parent  # -> src
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Also ensure the project root is in path (for editable installs)
_PROJECT_ROOT = _SRC_DIR.parent  # -> Data_Processing_2027
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import streamlit as st
import zarr

if TYPE_CHECKING:
    pass  # For future type hints if needed

from hdmea.viz.zarr_viz.tree import TreeNode, parse_zarr_tree, get_node_by_path
from hdmea.viz.zarr_viz.plots import create_plot, get_plot_bytes
from hdmea.viz.zarr_viz.metadata import format_array_info, format_group_info, format_attributes
from hdmea.viz.zarr_viz.utils import InvalidZarrPathError, UnsupportedArrayError, should_warn_large

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Zarr Visualization Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Session State Initialization
# =============================================================================


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "zarr_path" not in st.session_state:
        st.session_state.zarr_path = None
    if "tree" not in st.session_state:
        st.session_state.tree = None
    if "selected_path" not in st.session_state:
        st.session_state.selected_path = None
    if "expanded_nodes" not in st.session_state:
        st.session_state.expanded_nodes = {}
    if "slice_config" not in st.session_state:
        st.session_state.slice_config = {}
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "recent_files" not in st.session_state:
        st.session_state.recent_files = []
    if "show_recent" not in st.session_state:
        st.session_state.show_recent = False
    if "show_browser" not in st.session_state:
        st.session_state.show_browser = False
    if "browser_path" not in st.session_state:
        # Start in home directory or current working directory
        st.session_state.browser_path = str(Path.home())


def add_to_recent_files(path: str, max_recent: int = 10) -> None:
    """Add a path to the recent files list.
    
    Args:
        path: Path to add.
        max_recent: Maximum number of recent files to keep.
    """
    if path in st.session_state.recent_files:
        st.session_state.recent_files.remove(path)
    st.session_state.recent_files.insert(0, path)
    st.session_state.recent_files = st.session_state.recent_files[:max_recent]


def get_directory_contents(path: str) -> tuple[list[str], list[str]]:
    """Get directories and zarr files in a path.
    
    Args:
        path: Directory path to list.
        
    Returns:
        Tuple of (directories, zarr_folders).
    """
    try:
        p = Path(path)
        if not p.exists() or not p.is_dir():
            return [], []
        
        dirs = []
        zarr_folders = []
        
        for item in sorted(p.iterdir()):
            if item.is_dir():
                # Check if it's a zarr archive
                if item.suffix == '.zarr' or (item / '.zarray').exists() or (item / '.zgroup').exists():
                    zarr_folders.append(item.name)
                else:
                    dirs.append(item.name)
        
        return dirs, zarr_folders
    except PermissionError:
        return [], []
    except Exception:
        return [], []


# =============================================================================
# Cached Functions
# =============================================================================


@st.cache_data(show_spinner="Loading zarr structure...")
def load_zarr_tree(zarr_path: str) -> TreeNode:
    """Load and cache zarr tree structure.

    Args:
        zarr_path: Path to zarr archive.

    Returns:
        Parsed TreeNode structure.
    """
    return parse_zarr_tree(zarr_path)


# =============================================================================
# UI Components
# =============================================================================


def render_sidebar() -> None:
    """Render the sidebar with path input and tree navigation."""
    with st.sidebar:
        st.header("📁 Zarr Explorer")
        
        # Inject custom CSS for compact styling
        st.markdown("""
        <style>
        /* Compact browser styling */
        section[data-testid="stSidebar"] .browser-item {
            font-size: 12px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # Path input with browse button
        zarr_path = st.text_input(
            "Zarr Archive Path",
            value=st.session_state.zarr_path or "",
            placeholder="Enter path to .zarr directory",
            help="Enter the full path to a zarr archive directory",
        )

        # Buttons row: Browse and Load
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📂 Browse", use_container_width=True):
                st.session_state.show_browser = not st.session_state.show_browser
                st.rerun()
        
        with col2:
            if st.button("▶ Load", type="primary", use_container_width=True):
                if zarr_path:
                    try:
                        st.session_state.zarr_path = zarr_path
                        st.session_state.tree = load_zarr_tree(zarr_path)
                        st.session_state.selected_path = None
                        st.session_state.last_error = None
                        st.session_state.show_browser = False
                        add_to_recent_files(zarr_path)
                        st.rerun()
                    except InvalidZarrPathError as e:
                        st.session_state.last_error = str(e)
                        st.session_state.tree = None
                    except Exception as e:
                        st.session_state.last_error = f"Error loading archive: {e}"
                        st.session_state.tree = None
                else:
                    st.warning("Please enter a zarr path")

        # Web-based file browser (toggle display)
        if st.session_state.show_browser:
            render_file_browser()

        # Recent files dropdown
        if st.session_state.recent_files:
            triangle = "▼" if st.session_state.show_recent else "▶"
            if st.button(
                f"{triangle} Recent ({len(st.session_state.recent_files)})",
                key="toggle_recent",
                use_container_width=True,
            ):
                st.session_state.show_recent = not st.session_state.show_recent
                st.rerun()
            
            if st.session_state.show_recent:
                for i, recent_path in enumerate(st.session_state.recent_files):
                    display_name = Path(recent_path).name
                    if st.button(
                        f"  📄 {display_name}",
                        key=f"recent_{i}",
                        use_container_width=True,
                        help=recent_path,
                    ):
                        st.session_state.zarr_path = recent_path
                        try:
                            st.session_state.tree = load_zarr_tree(recent_path)
                            st.session_state.selected_path = None
                            st.session_state.last_error = None
                            add_to_recent_files(recent_path)
                            st.rerun()
                        except Exception as e:
                            st.session_state.last_error = f"Error loading: {e}"
                            st.session_state.tree = None
                            st.rerun()

        # Show error if any
        if st.session_state.last_error:
            st.error(st.session_state.last_error)

        st.divider()

        # Tree navigation
        if st.session_state.tree is not None:
            st.subheader("📂 Structure")
            render_tree_sidebar(st.session_state.tree)
        else:
            st.info("Enter a zarr path above to explore")


def render_file_browser() -> None:
    """Render a web-based file browser for selecting zarr folders."""
    st.markdown("---")
    st.markdown("**📂 File Browser**")
    
    current_path = Path(st.session_state.browser_path)
    
    # Show current path (editable)
    new_path = st.text_input(
        "Current Path",
        value=str(current_path),
        key="browser_path_input",
        label_visibility="collapsed",
    )
    
    # Update path if changed
    if new_path != str(current_path) and Path(new_path).exists():
        st.session_state.browser_path = new_path
        st.rerun()
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬆ Parent", key="browser_parent", use_container_width=True):
            parent = current_path.parent
            if parent != current_path:
                st.session_state.browser_path = str(parent)
                st.rerun()
    with col2:
        if st.button("🏠 Home", key="browser_home", use_container_width=True):
            st.session_state.browser_path = str(Path.home())
            st.rerun()
    
    # Get directory contents
    dirs, zarr_folders = get_directory_contents(str(current_path))
    
    # Show zarr folders first (selectable)
    if zarr_folders:
        st.markdown("**Zarr Archives:**")
        for zf in zarr_folders[:20]:  # Limit for performance
            full_path = str(current_path / zf)
            if st.button(
                f"📦 {zf}",
                key=f"zarr_{zf}",
                use_container_width=True,
                type="primary",
            ):
                st.session_state.zarr_path = full_path
                st.session_state.show_browser = False
                try:
                    st.session_state.tree = load_zarr_tree(full_path)
                    st.session_state.selected_path = None
                    st.session_state.last_error = None
                    add_to_recent_files(full_path)
                except Exception as e:
                    st.session_state.last_error = f"Error: {e}"
                    st.session_state.tree = None
                st.rerun()
    
    # Show regular directories (navigable)
    if dirs:
        st.markdown("**Folders:**")
        # Limit display for performance
        display_dirs = dirs[:30]
        for d in display_dirs:
            if st.button(
                f"📁 {d}",
                key=f"dir_{d}",
                use_container_width=True,
            ):
                st.session_state.browser_path = str(current_path / d)
                st.rerun()
        
        if len(dirs) > 30:
            st.caption(f"... and {len(dirs) - 30} more folders")
    
    if not dirs and not zarr_folders:
        st.caption("No folders found")
    
    st.markdown("---")


def render_tree_sidebar(root: TreeNode, level: int = 0) -> None:
    """Render tree structure in sidebar as a compact file browser style.

    Args:
        root: TreeNode to render.
        level: Current indentation level.
    """
    # Inject compact file browser style CSS once at root level
    if level == 0:
        st.markdown("""
        <style>
        /* ===== COMPACT FILE BROWSER TREE ===== */
        
        /* Remove gaps between elements */
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0 !important;
        }
        section[data-testid="stSidebar"] .element-container {
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Tree item buttons - compact and left-aligned */
        section[data-testid="stSidebar"] .stButton {
            margin: 0 !important;
            padding: 0 !important;
        }
        section[data-testid="stSidebar"] .stButton > div {
            width: 100% !important;
        }
        section[data-testid="stSidebar"] .stButton button {
            padding: 4px 8px !important;
            margin: 0 !important;
            font-size: 13px !important;
            min-height: 28px !important;
            height: 28px !important;
            line-height: 20px !important;
            display: flex !important;
            justify-content: flex-start !important;
            align-items: center !important;
            text-align: left !important;
            font-family: 'Consolas', 'SF Mono', 'Monaco', monospace !important;
            border: none !important;
            border-radius: 4px !important;
            background: transparent !important;
            color: #d4d4d4 !important;
            width: 100% !important;
            box-shadow: none !important;
            white-space: nowrap !important;
            overflow: hidden !important;
        }
        section[data-testid="stSidebar"] .stButton button:hover {
            background: rgba(255, 255, 255, 0.06) !important;
        }
        section[data-testid="stSidebar"] .stButton button p {
            margin: 0 !important;
            padding: 0 !important;
            white-space: nowrap !important;
        }
        
        /* Selected item highlight */
        section[data-testid="stSidebar"] .stButton button[kind="primary"] {
            background: rgba(55, 65, 81, 0.8) !important;
            color: #ffffff !important;
        }
        section[data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
            background: rgba(75, 85, 99, 0.9) !important;
        }
        
        /* Disabled state for empty folders */
        section[data-testid="stSidebar"] .stButton button:disabled {
            opacity: 0.5 !important;
            color: #6b7280 !important;
        }
        
        /* Folder card containers */
        .fb-card {
            background: rgba(38, 40, 48, 0.9);
            border-radius: 8px;
            padding: 6px;
            margin: 4px 0;
            border: 1px solid rgba(255, 255, 255, 0.06);
        }
        .fb-card-inner {
            background: rgba(50, 52, 62, 0.85);
            border-radius: 6px;
            padding: 4px;
            margin: 3px 0 3px 16px;
            border: 1px solid rgba(255, 255, 255, 0.04);
        }
        </style>
        """, unsafe_allow_html=True)
    
    _render_file_browser_node(root, level)


def _render_file_browser_node(node: TreeNode, level: int = 0) -> None:
    """Render a tree node in compact file browser style.
    
    Args:
        node: TreeNode to render.
        level: Current indentation level (0 = root).
    """
    # Build indentation string (narrow indent for compactness)
    indent_str = "  " * level
    
    if node.is_array:
        # === ARRAY (FILE) NODE ===
        is_selected = st.session_state.selected_path == node.path
        shape_info = f" {node.shape}" if node.shape else ""
        
        # Format: indent + spacer + icon + name + shape
        label = f"{indent_str}    📄 {node.name}{shape_info}"
        
        if st.button(
            label,
            key=f"arr_{node.path}",
            use_container_width=True,
            type="primary" if is_selected else "secondary",
        ):
            st.session_state.selected_path = node.path
            st.rerun()
    
    else:
        # === GROUP (FOLDER) NODE ===
        is_expanded = st.session_state.expanded_nodes.get(node.path, level == 0)
        child_count = len(node.children) if node.children else 0
        
        # Toggle indicator
        toggle = "v" if is_expanded else ">"
        folder_icon = "📁"
        
        # Count display
        count_str = f"({child_count})" if child_count > 0 else "(empty)"
        
        # Wrap in card for levels 0 and 1
        use_card = (level == 0)
        use_inner_card = (level == 1)
        
        if use_card:
            st.markdown('<div class="fb-card">', unsafe_allow_html=True)
        elif use_inner_card:
            st.markdown('<div class="fb-card-inner">', unsafe_allow_html=True)
        
        # Format: indent + toggle + folder + name + count
        label = f"{indent_str}{toggle} {folder_icon} {node.name}  {count_str}"
        
        if child_count > 0:
            if st.button(
                label,
                key=f"grp_{node.path}",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state.expanded_nodes[node.path] = not is_expanded
                st.rerun()
            
            # Render children when expanded
            if is_expanded:
                for child in node.children:
                    _render_file_browser_node(child, level + 1)
        else:
            # Empty folder (disabled)
            st.button(
                label,
                key=f"grp_{node.path}",
                use_container_width=True,
                type="secondary",
                disabled=True,
            )
        
        if use_card or use_inner_card:
            st.markdown('</div>', unsafe_allow_html=True)


def render_main_area() -> None:
    """Render the main content area with plot display."""
    st.title("🔬 Zarr Visualization Tool")

    if st.session_state.tree is None:
        render_welcome()
        return

    if st.session_state.selected_path is None:
        st.info("👈 Select an array from the sidebar to visualize")
        return

    # Get selected node
    node = get_node_by_path(st.session_state.tree, st.session_state.selected_path)

    if node is None:
        st.error(f"Node not found: {st.session_state.selected_path}")
        return

    if not node.is_array:
        st.info(f"Selected: {node.display_name} (group - select an array to plot)")
        return

    # Display array info and plot
    render_array_view(node)


def render_welcome() -> None:
    """Render welcome screen when no archive is loaded."""
    st.markdown("""
    ## Welcome to the Zarr Visualization Tool

    This tool allows you to explore and visualize data stored in Zarr archives.

    ### Getting Started

    1. **Enter a path** to a zarr archive in the sidebar
    2. **Click "Load Archive"** to parse the structure
    3. **Navigate** the tree to find arrays
    4. **Click an array** to visualize its contents

    ### Features

    - 📁 **Tree View**: Navigate the hierarchical structure
    - 📊 **Interactive Plots**: Zoom, pan, and hover for details
    - 💾 **Export**: Save plots as PNG or SVG
    - 📋 **Metadata**: View array properties and attributes
    """)

    # File picker section
    st.divider()
    st.subheader("📂 Quick Open")
    st.markdown("Enter a path to a zarr archive directory:")

    col1, col2 = st.columns([3, 1])
    with col1:
        quick_path = st.text_input(
            "Path",
            placeholder="e.g., C:/data/experiment.zarr or /home/user/data.zarr",
            label_visibility="collapsed",
        )
    with col2:
        if st.button("Open", type="primary", use_container_width=True):
            if quick_path:
                try:
                    st.session_state.zarr_path = quick_path
                    st.session_state.tree = load_zarr_tree(quick_path)
                    st.session_state.last_error = None
                    st.rerun()
                except InvalidZarrPathError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter a path")


def render_array_view(node: TreeNode) -> None:
    """Render array visualization and metadata.

    Args:
        node: TreeNode representing an array.
    """
    st.subheader(f"📊 {node.path}")

    # Display basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Shape", str(node.shape))
    with col2:
        st.metric("Type", node.dtype)
    with col3:
        if node.nbytes:
            size_mb = node.nbytes / (1024 * 1024)
            st.metric("Size", f"{size_mb:.2f} MB")

    st.divider()

    # Load the actual array from zarr
    try:
        root = zarr.open(st.session_state.zarr_path, mode="r")
        # Navigate to the array
        array_path = node.path.strip("/")
        if array_path:
            array = root[array_path]
        else:
            array = root
    except Exception as e:
        st.error(f"Failed to load array: {e}")
        return

    # Check if array is too large
    if should_warn_large(array):
        size_mb = node.nbytes / (1024 * 1024) if node.nbytes else 0
        st.warning(
            f"⚠️ This array is large ({size_mb:.1f} MB). "
            "Data will be sampled for visualization."
        )

    # Handle ND arrays - show dimension sliders
    slice_indices = {}
    if node.shape and len(node.shape) > 2:
        st.markdown("**Dimension Slicing** (for dimensions > 2)")
        cols = st.columns(min(len(node.shape) - 2, 4))
        for i, dim in enumerate(range(2, len(node.shape))):
            with cols[i % len(cols)]:
                max_val = node.shape[dim] - 1
                slice_indices[dim] = st.slider(
                    f"Dim {dim}",
                    min_value=0,
                    max_value=max_val,
                    value=st.session_state.slice_config.get(f"{node.path}_dim{dim}", 0),
                    key=f"slice_{node.path}_{dim}",
                )
                st.session_state.slice_config[f"{node.path}_dim{dim}"] = slice_indices[dim]

    # Check if array is numeric
    if not np.issubdtype(array.dtype, np.number):
        st.info(f"📋 This array contains non-numeric data ({array.dtype}). Cannot visualize.")
        # Show first few values as text
        try:
            sample = array[:min(10, array.shape[0])]
            # Convert to Python list/string for safe display
            sample_list = list(sample) if hasattr(sample, '__iter__') else [sample]
            sample_str = ", ".join(str(v) for v in sample_list[:10])
            st.text(f"Sample values: [{sample_str}]")
        except Exception as e:
            st.text(f"Could not load sample: {e}")
        return

    # Generate plot
    try:
        title = f"{node.path}\nShape: {node.shape}, Type: {node.dtype}"
        fig = create_plot(array, title=title, slice_indices=slice_indices)

        # Display interactive plot
        st.plotly_chart(fig, use_container_width=True, config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "displaylogo": False,
        })

        # Export buttons
        st.markdown("### 💾 Export")
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            try:
                png_bytes = get_plot_bytes(fig, format="png")
                st.download_button(
                    label="📥 Save PNG",
                    data=png_bytes,
                    file_name=f"{node.name}.png",
                    mime="image/png",
                    use_container_width=True,
                )
            except Exception as e:
                st.button("📥 Save PNG", disabled=True, help=f"PNG export unavailable: {e}")

        with col2:
            try:
                svg_bytes = get_plot_bytes(fig, format="svg")
                st.download_button(
                    label="📥 Save SVG",
                    data=svg_bytes,
                    file_name=f"{node.name}.svg",
                    mime="image/svg+xml",
                    use_container_width=True,
                )
            except Exception as e:
                st.button("📥 Save SVG", disabled=True, help=f"SVG export unavailable: {e}")

    except UnsupportedArrayError as e:
        st.info(f"📋 {e}")
    except Exception as e:
        st.error(f"Failed to create plot: {e}")

    # Metadata panel
    render_metadata_panel(array, node)


def render_metadata_panel(array: "zarr.Array", node: TreeNode) -> None:
    """Render metadata panel for array.

    Args:
        array: Zarr array object.
        node: TreeNode representing the array.
    """
    st.divider()
    st.markdown("### 📋 Metadata")

    # Array properties
    with st.expander("Array Properties", expanded=True):
        info = format_array_info(array)
        for key, value in info.items():
            st.markdown(f"**{key}:** {value}")

    # Zarr attributes
    try:
        attrs = format_attributes(array.attrs)
        if attrs:
            with st.expander(f"Zarr Attributes ({len(attrs)} items)", expanded=False):
                for key, value in attrs.items():
                    if "\n" in value:
                        st.markdown(f"**{key}:**")
                        st.code(value, language="json")
                    else:
                        st.markdown(f"**{key}:** `{value}`")
        else:
            st.info("No zarr attributes found for this array.")
    except Exception as e:
        st.warning(f"Could not load attributes: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main application entry point."""
    init_session_state()

    # Check for command line arguments
    if len(sys.argv) > 1:
        # First arg after script is the zarr path (passed via --)
        potential_path = sys.argv[-1]
        if (
            st.session_state.zarr_path is None
            and not potential_path.startswith("-")
            and Path(potential_path).exists()
        ):
            st.session_state.zarr_path = potential_path
            try:
                st.session_state.tree = load_zarr_tree(potential_path)
            except Exception as e:
                st.session_state.last_error = str(e)

    # Render UI
    render_sidebar()
    render_main_area()


if __name__ == "__main__":
    main()
