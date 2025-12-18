# API Contracts: HDF5 Store Module

**Feature**: 007-hdf5-replace-zarr  
**Module**: `hdmea.io.hdf5_store`

---

## Overview

This directory contains API contracts for the new HDF5 storage module. The contracts define function signatures, parameters, return types, and behaviors that must be implemented.

## Files

| File | Description |
|------|-------------|
| `api.md` | Function signatures and contracts |
| `README.md` | This overview document |

## Implementation Notes

- All functions mirror `zarr_store.py` signatures for compatibility
- Context managers (`with` statements) must be used for file handles
- Single-writer access is enforced with explicit error messages

