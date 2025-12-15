# Tasks: Zarr Visualization GUI

**Input**: Design documents from `/specs/002-zarr-viz-gui/`  
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: Not explicitly requested - skipped

**Organization**: Tasks grouped by user story for independent implementation and testing

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story (US1, US2, US3, US4)
- All paths relative to repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and module structure

- [x] T001 Create zarr_viz sub-module directory at src/hdmea/viz/zarr_viz/
- [x] T002 [P] Add viz optional dependencies (streamlit, plotly) to pyproject.toml
- [x] T003 [P] Create module __init__.py with public API exports in src/hdmea/viz/zarr_viz/__init__.py
- [x] T004 [P] Create utils.py with sample_array and should_warn_large in src/hdmea/viz/zarr_viz/utils.py
- [x] T005 [P] Create custom exceptions (ZarrVizError, InvalidZarrPathError, UnsupportedArrayError) in src/hdmea/viz/zarr_viz/utils.py

**Checkpoint**: Module structure ready, dependencies configured

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: TreeNode data structure and zarr parsing - required by ALL user stories

**⚠️ CRITICAL**: All user stories depend on tree parsing functionality

- [x] T006 Implement TreeNode dataclass in src/hdmea/viz/zarr_viz/tree.py
- [x] T007 Implement parse_zarr_tree function in src/hdmea/viz/zarr_viz/tree.py
- [x] T008 Implement get_node_by_path function in src/hdmea/viz/zarr_viz/tree.py
- [x] T009 Add path validation (exists, is zarr directory) in src/hdmea/viz/zarr_viz/tree.py
- [x] T010 Create base Streamlit app skeleton in src/hdmea/viz/zarr_viz/app.py

**Checkpoint**: Tree parsing works, basic app shell exists - user story work can begin

---

## Phase 3: User Story 1 - Browse Zarr Structure (Priority: P1) 🎯 MVP

**Goal**: Display zarr archive as interactive, collapsible tree with groups/arrays distinguished

**Independent Test**: Launch app, provide zarr path, verify tree displays all nodes with correct hierarchy

### Implementation for User Story 1

- [x] T011 [US1] Implement render_tree function with st.expander for groups in src/hdmea/viz/zarr_viz/tree.py
- [x] T012 [US1] Add visual icons/styling to distinguish groups (📁) from arrays (📊) in src/hdmea/viz/zarr_viz/tree.py
- [x] T013 [US1] Implement sidebar with path input and tree display in src/hdmea/viz/zarr_viz/app.py
- [x] T014 [US1] Add session state management for expanded_nodes and selected_path in src/hdmea/viz/zarr_viz/app.py
- [x] T015 [US1] Add error handling for invalid/corrupted zarr paths with st.error in src/hdmea/viz/zarr_viz/app.py
- [x] T016 [US1] Implement empty archive handling with info message in src/hdmea/viz/zarr_viz/app.py
- [x] T017 [US1] Add caching with @st.cache_data for parse_zarr_tree in src/hdmea/viz/zarr_viz/app.py

**Checkpoint**: Tree browsing fully functional - can open zarr and navigate structure

---

## Phase 4: User Story 2 - Plot Array Data (Priority: P1)

**Goal**: Click array node to see interactive plot with zoom, pan, hover; export to PNG/SVG

**Independent Test**: Click any array in tree, verify appropriate plot appears with interactivity and save buttons

### Implementation for User Story 2

- [x] T018 [P] [US2] Implement plot_1d function with Plotly line chart in src/hdmea/viz/zarr_viz/plots.py
- [x] T019 [P] [US2] Implement plot_2d function with Plotly heatmap in src/hdmea/viz/zarr_viz/plots.py
- [x] T020 [US2] Implement create_plot dispatcher (1D/2D/ND) in src/hdmea/viz/zarr_viz/plots.py
- [x] T021 [US2] Implement plot_nd with dimension slicing controls in src/hdmea/viz/zarr_viz/plots.py
- [x] T022 [US2] Add sampling logic for large arrays (>10M elements) in src/hdmea/viz/zarr_viz/plots.py
- [x] T023 [US2] Add large array warning dialog (>100MB) in src/hdmea/viz/zarr_viz/app.py
- [x] T024 [US2] Implement export_figure function (PNG/SVG) in src/hdmea/viz/zarr_viz/plots.py
- [x] T025 [US2] Add main area with plot display and st.plotly_chart in src/hdmea/viz/zarr_viz/app.py
- [x] T026 [US2] Add Save PNG and Save SVG buttons with download in src/hdmea/viz/zarr_viz/app.py
- [x] T027 [US2] Add plot title with array path and shape/dtype info in src/hdmea/viz/zarr_viz/plots.py
- [x] T028 [US2] Add dimension sliders for ND arrays in main area in src/hdmea/viz/zarr_viz/app.py
- [x] T029 [US2] Handle non-numeric arrays with info message in src/hdmea/viz/zarr_viz/app.py

**Checkpoint**: Full plotting functionality - click to plot, interactive, exportable

---

## Phase 5: User Story 3 - View Array Metadata (Priority: P2)

**Goal**: Display zarr attributes and array properties for selected node

**Independent Test**: Select any node, verify metadata panel shows shape, dtype, chunks, and attributes

### Implementation for User Story 3

- [x] T030 [P] [US3] Implement format_array_info function in src/hdmea/viz/zarr_viz/metadata.py
- [x] T031 [P] [US3] Implement format_group_info function in src/hdmea/viz/zarr_viz/metadata.py
- [x] T032 [US3] Implement format_attributes with non-serializable handling in src/hdmea/viz/zarr_viz/metadata.py
- [x] T033 [US3] Add right panel/expander for metadata display in src/hdmea/viz/zarr_viz/app.py
- [x] T034 [US3] Display shape, dtype, chunks for arrays in metadata panel in src/hdmea/viz/zarr_viz/app.py
- [x] T035 [US3] Display all zarr attributes in formatted table in src/hdmea/viz/zarr_viz/app.py

**Checkpoint**: Metadata viewing complete - all node info visible

---

## Phase 6: User Story 4 - Launch as Standalone Tool (Priority: P2)

**Goal**: Launch tool from command line with optional zarr path

**Independent Test**: Run `python -m hdmea.viz.zarr_viz [path]` and verify browser opens with app

### Implementation for User Story 4

- [x] T036 [US4] Create __main__.py with CLI argument parsing in src/hdmea/viz/zarr_viz/__main__.py
- [x] T037 [US4] Implement launch() function in __init__.py in src/hdmea/viz/zarr_viz/__init__.py
- [x] T038 [US4] Add file picker interface when no path provided in src/hdmea/viz/zarr_viz/app.py
- [x] T039 [US4] Configure streamlit run with appropriate options in src/hdmea/viz/zarr_viz/__main__.py
- [x] T040 [US4] Add startup message and browser launch in src/hdmea/viz/zarr_viz/__main__.py

**Checkpoint**: Tool launches from CLI - full standalone functionality

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final refinements across all user stories

- [x] T041 [P] Add synthetic zarr fixture generator in tests/fixtures/synthetic_zarr.py
- [x] T042 [P] Create unit tests for tree.py in tests/unit/test_zarr_viz.py
- [x] T043 [P] Create unit tests for plots.py in tests/unit/test_zarr_viz.py
- [x] T044 Add module docstrings to all zarr_viz files
- [x] T045 Validate quickstart.md scenarios work end-to-end
- [x] T046 Update src/hdmea/viz/__init__.py to expose zarr_viz if needed

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup) ──────────────────┐
                                  │
                                  ▼
Phase 2 (Foundational) ───────────┤ BLOCKS ALL USER STORIES
                                  │
         ┌────────────────────────┴────────────────────────┐
         │                                                 │
         ▼                                                 ▼
Phase 3 (US1: Tree)                              Can start in parallel
         │                                       if team capacity allows
         ▼
Phase 4 (US2: Plot) ─── depends on tree selection
         │
         ▼
Phase 5 (US3: Metadata) ─── depends on node selection
         │
         ▼
Phase 6 (US4: Launch) ─── depends on working app
         │
         ▼
Phase 7 (Polish)
```

### User Story Dependencies

| Story | Depends On | Can Parallelize With |
|-------|------------|---------------------|
| US1 (Tree) | Foundational only | - |
| US2 (Plot) | US1 (needs node selection) | US3 (partially) |
| US3 (Metadata) | US1 (needs node selection) | US2 (partially) |
| US4 (Launch) | US1 + working app | - |

### Parallel Opportunities

**Within Phase 1 (Setup)**:
- T002, T003, T004, T005 can all run in parallel

**Within Phase 4 (US2: Plot)**:
- T018, T019 can run in parallel (different plot types)

**Within Phase 5 (US3: Metadata)**:
- T030, T031 can run in parallel (different info formatters)

**Within Phase 7 (Polish)**:
- T041, T042, T043 can all run in parallel

---

## Parallel Example: Phase 1 Setup

```bash
# Launch all parallel setup tasks together:
Task T002: "Add viz optional dependencies to pyproject.toml"
Task T003: "Create module __init__.py in src/hdmea/viz/zarr_viz/__init__.py"
Task T004: "Create utils.py in src/hdmea/viz/zarr_viz/utils.py"
Task T005: "Create custom exceptions in src/hdmea/viz/zarr_viz/utils.py"
```

## Parallel Example: Phase 4 Plot Functions

```bash
# Launch parallel plot implementations:
Task T018: "Implement plot_1d function in src/hdmea/viz/zarr_viz/plots.py"
Task T019: "Implement plot_2d function in src/hdmea/viz/zarr_viz/plots.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 + 2 Only)

1. Complete Phase 1: Setup (T001-T005)
2. Complete Phase 2: Foundational (T006-T010)
3. Complete Phase 3: US1 - Tree Browsing (T011-T017)
4. **VALIDATE**: Can open zarr and see tree structure
5. Complete Phase 4: US2 - Plotting (T018-T029)
6. **VALIDATE**: Can click array and see interactive plot
7. **MVP COMPLETE** - Core functionality delivered

### Full Delivery

1. MVP above +
2. Complete Phase 5: US3 - Metadata (T030-T035)
3. Complete Phase 6: US4 - CLI Launch (T036-T040)
4. Complete Phase 7: Polish (T041-T046)
5. **FULL FEATURE COMPLETE**

---

## Notes

- [P] marks tasks that can run in parallel (different files)
- [USx] maps task to specific user story for traceability
- US1 and US2 are both P1 priority but US2 depends on US1 tree selection
- Commit after each logical task group
- Validate at each checkpoint before proceeding
- Total: 46 tasks across 7 phases
