# ThermoScope

Smart India Hackathon 2026 — Problem Statement SIH26162 (NTRO):
"AI-Based Detection and Classification of Industrial Fires and Persistent
Thermal Sources Using NASA FIRMS, OSM & Satellite Data."

ThermoScope uses NASA FIRMS thermal anomaly detections as a primary data
source and adds contextual analysis (spatial behaviour, temporal behaviour,
industrial/geospatial context, and eventually satellite data) to identify
persistent thermal sources, classify likely source types, and prioritize
unusual or significant sources for investigation. It does not attempt to
recreate FIRMS itself.

See [PROJECT_BRIEF.md](PROJECT_BRIEF.md) for the full problem/solution
description, [ARCHITECTURE.md](ARCHITECTURE.md) for the current (draft)
pipeline design, [DECISIONS.md](DECISIONS.md) for decisions made so far, and
[PROGRESS.md](PROGRESS.md) for current milestone status.

## Current Scope

Prototype stage, focused on Gujarat using historical VIIRS 375m FIRMS data
(current working assumption — see `DECISIONS.md`). First milestone: inspect,
clean, validate, and explore FIRMS historical data, and produce an initial
geospatial visualization.

## Repository Layout

```
data/
  raw/        # original, unmodified source data
  processed/  # cleaned/validated intermediate data
  gold/       # final, analysis-ready data
src/          # source code (not yet started)
tests/        # tests (not yet started)
notebooks/    # exploratory notebooks
results/
  maps/       # geospatial visualization outputs
  figures/    # charts/figures
```

## Status

Documentation and repository structure only. No code, dependencies, or data
have been added yet. See [PROGRESS.md](PROGRESS.md) for details.
