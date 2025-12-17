# Feature Specification: Load Section Time Metadata

**Feature Branch**: `004-load-section-time`  
**Created**: 2025-12-16  
**Status**: Draft  
**Input**: User description: "Add section_time loading to metadata from playlist and movie_length CSVs"

## Overview

This feature enables automatic computation and storage of section time boundaries for visual stimulation experiments. It parses playlist and movie length configuration files to determine when each movie stimulus begins and ends within a recording, storing this timing information in the Zarr metadata for downstream feature extraction.

### Algorithm Summary

The legacy algorithm works as follows:

1. **Load Configuration Files**: Read `playlist.csv` (maps playlist names to movie sequences) and `movie_length.csv` (contains frame counts for each movie)

2. **Parse Movie Sequence**: For a given playlist name, extract the ordered list of movies and apply repeat multiplier

3. **Compute Frame Boundaries**: For each movie in sequence:
   - $\text{start\_frame} = \text{cumulative\_count} + \text{pad} - \text{pre\_margin}$
   - $\text{end\_frame} = \text{cumulative\_count} + \text{pad} + \text{post\_margin} + \text{movie\_length} + 1$
   - Update cumulative count: $\text{count} \mathrel{+}= 2 \times \text{pad} + \text{movie\_length} + 1$

4. **Extract Light Templates**: Using frame-to-time conversion, extract the light reference signal segment for each movie section

5. **Average Repeated Movies**: For movies that repeat, average the light templates using `zip_longest` to handle variable lengths

6. **Store Results**: Add `section_time_auto` (frame boundaries) and `template_auto` (averaged light templates) to metadata

## Clarifications

### Session 2025-12-16

- Q: What should happen when `add_section_time()` is called on a Zarr that already contains section_time data? → A: Require `force=True` parameter to overwrite; error by default

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Load Section Times from Playlist (Priority: P1)

A researcher loads a recording and wants to automatically segment it by visual stimulation movies using a predefined playlist configuration.

**Why this priority**: This is the core functionality - without it, downstream feature extraction cannot identify which frames correspond to which stimuli.

**Independent Test**: Can be fully tested by loading a recording with known playlist and verifying frame boundaries match expected values.

**Acceptance Scenarios**:

1. **Given** a valid .cmcr/.cmtr recording and matching playlist.csv/movie_length.csv, **When** the user calls `add_section_time()` with a valid playlist name, **Then** section_time metadata is added to the Zarr under `stimulus/section_time/` with correct frame boundaries for each movie
2. **Given** a playlist with 3 movies repeated 2 times, **When** section times are computed, **Then** each movie has an array of 2 frame pairs (start, end) representing each repeat
3. **Given** valid input files, **When** section times are added, **Then** light templates are extracted and averaged for each movie section

---

### User Story 2 - Handle Missing Configuration Files (Priority: P2)

A researcher attempts to load section times but the playlist or movie_length file is missing or inaccessible.

**Why this priority**: Graceful error handling prevents pipeline crashes and provides actionable feedback.

**Independent Test**: Can be tested by calling the function with non-existent file paths and verifying appropriate warnings are logged.

**Acceptance Scenarios**:

1. **Given** a missing playlist.csv file, **When** `add_section_time()` is called, **Then** a warning is logged and the function returns without modifying metadata
2. **Given** a missing movie_length.csv file, **When** `add_section_time()` is called, **Then** a warning is logged and the function returns without modifying metadata
3. **Given** a playlist name not found in playlist.csv, **When** `add_section_time()` is called, **Then** a warning is logged with the available playlist names

---

### User Story 3 - Custom Configuration Paths (Priority: P3)

A researcher uses configuration files stored in a project-specific location rather than the default network path.

**Why this priority**: Flexibility for different lab setups and local testing.

**Independent Test**: Can be tested by providing custom paths and verifying they are used correctly.

**Acceptance Scenarios**:

1. **Given** custom playlist_file_path and movie_length_file_path, **When** `add_section_time()` is called, **Then** those paths are used instead of defaults
2. **Given** a relative path, **When** `add_section_time()` is called, **Then** the path is resolved correctly relative to the working directory

---

### Edge Cases

- What happens when a movie in the playlist is not found in movie_length.csv?
  - Log warning with movie name, skip that movie, continue processing others
- What happens when the light reference data is shorter than expected?
  - Truncate the template extraction to available data
- What happens when repeats parameter is 0 or negative?
  - Treat as 1 repeat (default behavior)
- What happens when frame boundaries exceed recording length?
  - Clip to recording length and log warning
- What happens when section_time data already exists in the Zarr?
  - Raise an error by default; overwrite only when `force=True` is provided

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST read playlist configuration from CSV file with columns: `playlist_name`, `movie_names`
- **FR-002**: System MUST read movie length configuration from CSV file with columns: `movie_name`, `movie_length`
- **FR-003**: System MUST compute frame boundaries for each movie using the formula: start = cumulative_count + pad - pre_margin, end = cumulative_count + pad + post_margin + length + 1
- **FR-004**: System MUST store section_time data under `stimulus/section_time/{movie_name}` in Zarr format
- **FR-005**: System MUST extract light template for each movie section from the light reference signal
- **FR-006**: System MUST average light templates across repeats of the same movie
- **FR-007**: System MUST store averaged light templates under `stimulus/light_template/{movie_name}` in Zarr
- **FR-008**: System MUST use default configuration file paths when not explicitly provided:
  - playlist: `//Jiangfs1/fs_1_2_data/Python_Project/Design_Stimulation_Pattern/Data/playlist.csv`
  - movie_length: `//Jiangfs1/fs_1_2_data/Python_Project/Design_Stimulation_Pattern/Data/movie_length.csv`
- **FR-009**: System MUST support configurable parameters: pre_margin_frame_num (default: 60), post_margin_frame_num (default: 120), pad_frame (default: 180)
- **FR-010**: System MUST log warnings for missing files, unknown movies, or playlist names not found
- **FR-011**: System MUST raise an error if section_time data already exists in the Zarr, unless `force=True` parameter is provided; when `force=True`, existing section_time and light_template data MUST be overwritten

### Key Entities

- **Playlist**: Maps a playlist name to an ordered list of movie file names (stored as Python list string in CSV)
- **Movie Length**: Maps movie names to their duration in frames
- **Section Time**: Array of (start_frame, end_frame) pairs for each occurrence of a movie
- **Light Template**: Averaged light reference signal segment for a movie section

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Section times can be added to any existing Zarr archive within 5 seconds for typical playlists (< 20 movies)
- **SC-002**: Researchers can access section boundaries for any movie using standard Zarr navigation
- **SC-003**: 100% of movies in a valid playlist have corresponding section_time entries after successful execution
- **SC-004**: Feature extractors that depend on section timing can retrieve boundaries without additional file parsing
- **SC-005**: Light templates are correctly aligned with frame boundaries (verified by visual inspection of template peaks)

## Assumptions

- Playlist CSV uses `playlist_name` as index column after loading
- Movie names in playlist include file extension (e.g., "movie1.mov") which gets stripped
- Movie length CSV uses `movie_name` as index column after loading
- Light reference data is available in the Zarr under `stimulus/light_reference/`
- Frame time array is available in metadata for frame-to-time conversion
- All movies in a playlist have entries in movie_length.csv for successful processing
