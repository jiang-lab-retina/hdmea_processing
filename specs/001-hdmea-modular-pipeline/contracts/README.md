# API Contracts

This directory contains the public API contracts for the HD-MEA Data Analysis Pipeline.

## Files

| File | Description |
|------|-------------|
| `pipeline_api.py` | Main pipeline API (load, extract, run_flow, export) |
| `exceptions.py` | Exception hierarchy for error handling |

## Purpose

These contracts define:

1. **Function signatures** - Parameters, return types, and behavior
2. **Data structures** - Result objects and their fields
3. **Protocols** - Interfaces for extensible components (FeatureExtractor)
4. **Exceptions** - Error types and when they are raised

## Usage

Implementation modules MUST:

- Conform to the signatures defined here
- Raise the exceptions defined in `exceptions.py`
- Return the data structures defined in `pipeline_api.py`

## Updating Contracts

1. Propose changes in a PR
2. Update this directory with new signatures
3. Update `data-model.md` if entities change
4. Bump version in constitution if breaking change

