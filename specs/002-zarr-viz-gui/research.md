# Research: Zarr Visualization GUI

**Feature**: 002-zarr-viz-gui  
**Date**: 2025-01-14

## Technology Decisions

### 1. GUI Framework: Streamlit

**Decision**: Use Streamlit as the GUI framework

**Rationale**:
- User explicitly specified Streamlit in feature request
- Rapid development with minimal boilerplate
- Native Python - no JavaScript required
- Built-in session state management
- Easy deployment for local use
- Excellent integration with data science libraries

**Alternatives Considered**:

| Alternative | Rejected Because |
|-------------|------------------|
| Dash (Plotly) | More complex setup, better for production dashboards |
| Panel (HoloViews) | Steeper learning curve |
| Gradio | More focused on ML demos than data exploration |
| PyQt/Tkinter | Desktop-only, more development effort |

### 2. Plotting Library: Plotly

**Decision**: Use Plotly for all visualizations

**Rationale**:
- Native interactive features (zoom, pan, hover) without extra code
- Built-in PNG/SVG export functionality
- Seamless Streamlit integration via st.plotly_chart()
- Consistent API for 1D, 2D, and heatmap plots
- WebGL rendering for large datasets

**Alternatives Considered**:

| Alternative | Rejected Because |
|-------------|------------------|
| Matplotlib | Static by default, interactivity requires mpld3 or similar |
| Altair | Good for declarative viz but less control over interactivity |
| Bokeh | More complex, better for standalone dashboards |

### 3. Tree Component: Custom with Streamlit Expanders

**Decision**: Build tree view using Streamlit native expander and session state

**Rationale**:
- No external dependencies
- Consistent with Streamlit design patterns
- Simple implementation with recursion
- Session state handles expand/collapse naturally

### 4. Large Array Handling: Uniform Sampling

**Decision**: For arrays exceeding 10 million elements, uniformly sample to fit memory

**Rationale**:
- Simple and predictable behavior
- Preserves overall data distribution
- Users can see warning and choose to proceed or slice
- 10M threshold balances responsiveness and data fidelity

### 5. Multi-dimensional Array Handling: Slice Selection

**Decision**: For >2D arrays, display first 2 dimensions with slider controls for other dimensions

**Rationale**:
- Matches user expectations from tools like napari
- Avoids overwhelming the interface
- Allows exploration of all dimensions sequentially
- Streamlit sliders provide natural dimension navigation

## Best Practices Applied

### Streamlit Performance

1. Use st.cache_data for expensive operations (parsing zarr tree, loading metadata)
2. Lazy Loading - only load array data when selected
3. Session State - store selected node path, not the data itself

### Zarr Access Patterns

1. Read-only mode - open zarr with mode r to prevent modifications
2. Chunk-aware slicing when possible
3. Use attrs.asdict() for safe attribute serialization

### Error Handling

1. Graceful degradation - show error but do not crash
2. User feedback with st.error(), st.warning(), st.info()
3. Validate path exists before attempting to open

## Open Questions Resolved

| Question | Resolution |
|----------|------------|
| Zarr v2 vs v3 support | Both - zarr library handles transparently |
| Remote zarr (S3, HTTP) | Not in scope for MVP - local paths only |
| Authentication | Not needed - local single-user tool |
| Persistence | Nice-to-have using Streamlit session state |
