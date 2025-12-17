# Contracts: Analog Section Time Detection

This directory contains API contracts for the analog section time feature.

## Files

- [api.md](./api.md) - Function signatures, parameters, return types, and error codes

## Overview

This feature modifies two existing functions in `src/hdmea/io/section_time.py`:

1. **`add_section_time_analog()`** - Detects stimulus onsets from raw_ch1 signal
2. **`add_section_time()`** - Computes section times from playlist metadata

Both functions now output section_time arrays in **acquisition sample indices** (unified unit).

## Key Changes

| Function | Before | After |
|----------|--------|-------|
| `add_section_time_analog()` | Display frame indices | Acquisition sample indices |
| `add_section_time()` | Display frame indices | Acquisition sample indices |

## Usage

See [quickstart.md](../quickstart.md) for usage examples.
