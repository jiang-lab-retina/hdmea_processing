# Feature Specification: Zarr Visualization GUI

**Feature Branch**: `002-zarr-viz-gui`  
**Created**: 2025-01-14  
**Status**: Draft  
**Input**: User description: "Create a Streamlit-based zarr visualization GUI in src/hdmea/viz/zarr_viz. Make it an independent sub-module. It shall allow me to see the tree structure of zarr dictionary data. When clicked on the leaves of the tree, it will allow me to plot the data."

## Overview

A standalone visualization tool for exploring and plotting data stored in Zarr archives. The tool provides an interactive tree view of the zarr hierarchy and allows users to visualize array data by clicking on leaf nodes. This tool is designed as an independent sub-module that can be launched separately from the main pipeline.

## Clarifications

### Session 2025-01-14

- Q: Should users be able to save/export the generated plots as image files? → A: View-only with save button for PNG/SVG export.
- Q: Should the plots support interactive features like zoom, pan, and hover-to-see-values? → A: Full interactivity (zoom, pan, hover tooltips).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Browse Zarr Structure (Priority: P1)

As a researcher, I want to open a zarr archive and see its complete hierarchical structure displayed as an interactive tree, so I can understand what data is available and navigate to specific datasets.

**Why this priority**: This is the foundational functionality - without the ability to see the zarr structure, users cannot discover or select data to visualize.

**Independent Test**: Can be fully tested by opening any valid zarr file and verifying the tree displays all groups and arrays with correct nesting.

**Acceptance Scenarios**:

1. **Given** the visualization tool is launched, **When** the user provides a path to a valid zarr archive, **Then** the tool displays a collapsible tree view showing all groups and arrays in the archive.
2. **Given** a zarr tree is displayed, **When** the user clicks on a group node (folder icon), **Then** the node expands or collapses to show/hide its children.
3. **Given** a zarr tree is displayed, **When** the user views the tree, **Then** groups are visually distinguished from arrays (leaf nodes) using different icons or styling.

---

### User Story 2 - Plot Array Data (Priority: P1)

As a researcher, I want to click on any array (leaf node) in the tree and immediately see a visualization of that data, so I can quickly inspect the contents without writing code.

**Why this priority**: Data visualization is the core value proposition - users need to see their data, not just the structure.

**Independent Test**: Can be tested by clicking on any array node and verifying an appropriate plot appears with the data.

**Acceptance Scenarios**:

1. **Given** a zarr tree is displayed, **When** the user clicks on a 1D array leaf node, **Then** the tool displays a line plot of the array values.
2. **Given** a zarr tree is displayed, **When** the user clicks on a 2D array leaf node, **Then** the tool displays an appropriate 2D visualization (heatmap or image).
3. **Given** a zarr tree is displayed, **When** the user clicks on an array, **Then** the plot title shows the array path and basic metadata (shape, dtype).
4. **Given** a plot is displayed, **When** the user interacts with the plot, **Then** the user can zoom, pan, and see data values on hover.
5. **Given** a plot is displayed, **When** the user clicks the save button, **Then** the plot is exported as PNG or SVG format.

---

### User Story 3 - View Array Metadata (Priority: P2)

As a researcher, I want to see the metadata and attributes associated with any group or array, so I can understand the context and parameters of the stored data.

**Why this priority**: Metadata provides essential context for interpreting the data correctly, but the tool is still useful without it.

**Independent Test**: Can be tested by selecting any node and verifying its attributes are displayed in a dedicated panel.

**Acceptance Scenarios**:

1. **Given** a zarr tree is displayed, **When** the user selects any node (group or array), **Then** the tool displays all zarr attributes associated with that node.
2. **Given** an array is selected, **When** viewing metadata, **Then** the display includes shape, dtype, chunk configuration, and any custom attributes.

---

### User Story 4 - Launch as Standalone Tool (Priority: P2)

As a researcher, I want to launch the visualization tool independently from the command line, so I can quickly inspect zarr files without integrating into my analysis scripts.

**Why this priority**: Independence and ease of launch makes the tool more accessible for ad-hoc data exploration.

**Independent Test**: Can be tested by running a single command from the terminal and verifying the tool launches in a browser.

**Acceptance Scenarios**:

1. **Given** the tool is installed, **When** the user runs the launch command with a zarr path, **Then** a browser window opens with the visualization interface.
2. **Given** the tool is launched without a path argument, **When** the interface loads, **Then** the user is presented with a file/folder picker to select a zarr archive.

---

### Edge Cases

- What happens when the user opens an empty zarr archive? → Display a message indicating no data found.
- What happens when the user clicks on a very large array (>100MB)? → Display a warning and offer to plot a subset/sample of the data.
- How does the system handle corrupted or inaccessible zarr files? → Display a clear error message indicating the file cannot be read.
- What happens when an array has more than 2 dimensions? → Display the first 2 dimensions with controls to select slices of other dimensions.
- What happens when zarr attributes contain non-serializable objects? → Display as string representation or indicate "complex object".

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST display zarr archives as an interactive, collapsible tree structure.
- **FR-002**: System MUST visually distinguish between groups (containers) and arrays (data) in the tree view.
- **FR-003**: System MUST generate appropriate plots when users click on array leaf nodes.
- **FR-004**: System MUST support visualization of 1D arrays as line plots.
- **FR-005**: System MUST support visualization of 2D arrays as heatmaps or images.
- **FR-006**: System MUST display array metadata (shape, dtype, chunks) alongside visualizations.
- **FR-007**: System MUST display zarr attributes for any selected node.
- **FR-008**: System MUST provide a file picker interface when no zarr path is provided.
- **FR-009**: System MUST be launchable as a standalone application from the command line.
- **FR-010**: System MUST handle arrays larger than available memory by loading only visible data or samples.
- **FR-011**: System MUST provide clear error messages for invalid or inaccessible zarr paths.
- **FR-012**: System MUST support multi-dimensional arrays (>2D) with dimension slicing controls.
- **FR-013**: System MUST provide interactive plots with zoom, pan, and hover-to-see-values functionality.
- **FR-014**: System MUST provide a save button to export plots as PNG or SVG image files.

### Key Entities

- **Zarr Archive**: A directory-based data store containing groups and arrays. Identified by path.
- **Group**: A container node in the zarr hierarchy that can contain other groups or arrays. Has attributes.
- **Array**: A leaf node containing numerical data. Has shape, dtype, chunks, and attributes.
- **Tree Node**: UI representation of a group or array, with expand/collapse state and selection state.
- **Visualization**: A plot or display generated from array data, with appropriate type based on data dimensions. Supports interactivity (zoom, pan, hover) and export (PNG/SVG).

## Assumptions

- Users have valid zarr v2 or v3 archives to visualize.
- The tool will be used primarily for HD-MEA pipeline outputs but should work with any valid zarr archive.
- Arrays contain numerical data suitable for plotting (not string or object arrays).
- The primary use case is single-user local exploration, not multi-user or remote access.
- Browser-based interface (Streamlit default) is acceptable for the user experience.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can open a zarr archive and see its complete tree structure within 3 seconds for archives with up to 1000 nodes.
- **SC-002**: Users can click on any array and see a plot within 2 seconds for arrays up to 10 million elements.
- **SC-003**: 95% of users can successfully navigate to and plot a specific array on their first attempt without documentation.
- **SC-004**: The tool launches and is ready for use within 5 seconds of running the launch command.
- **SC-005**: Users can explore zarr archives without writing any code, reducing data inspection time by at least 50% compared to manual scripting.
- **SC-006**: Users can export any displayed plot as PNG or SVG within 1 second of clicking save.
