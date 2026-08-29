# Project Brief — ThermoScope

## Problem Statement

**SIH Problem Statement ID:** SIH26162
**Title:** AI-Based Detection and Classification of Industrial Fires and
Persistent Thermal Sources Using NASA FIRMS, OSM & Satellite Data
**Organization:** National Technical Research Organisation (NTRO)
**Category:** Software
**Theme:** Miscellaneous

## Proposed Solution

NASA FIRMS already detects satellite-based thermal anomalies (fire/thermal
hotspots) using VIIRS and MODIS sensors. ThermoScope does not attempt to
recreate this detection capability.

Instead, ThermoScope proposes to use FIRMS hotspot data as a **primary input**
and add a layer of contextual analysis on top of it, combining:

- **Spatial behaviour** — clustering and location patterns of repeated
  detections.
- **Temporal behaviour** — persistence, frequency, and recurrence of thermal
  activity at a location over time.
- **Industrial/geospatial context** — proximity to known infrastructure
  (e.g. from OpenStreetMap) that could explain or fail to explain a thermal
  source.
- **Satellite data** (future/eventual) — additional imagery-based evidence
  for higher-priority sources.

The eventual goal is to:

1. Identify **persistent thermal sources** (as opposed to one-off or
   naturally occurring fire events).
2. Classify likely **source types** (e.g. industrial activity vs. other
   causes), to the extent the available data supports this.
3. **Prioritize** unusual or potentially significant sources for human
   investigation.

## Why FIRMS Alone Is Insufficient for This Use Case

FIRMS provides raw thermal anomaly detections (location, timestamp,
confidence, brightness/FRP, etc.). It does not, by itself:

- Distinguish between a single transient event and a recurring/persistent
  source at the same or nearby location.
- Provide context about what is actually present at a detection location
  (e.g. a factory, a landfill, agricultural land, etc.).
- Classify or rank detections by likely cause or by investigative priority.

These gaps are the specific space ThermoScope is intended to address. This is
a stated rationale for the project's existence, not a validated technical
finding — it should be checked against the actual FIRMS data as part of the
first milestone's exploratory analysis.

## Current Scope

- Prototype only — not an India-wide deployment.
- Candidate region: **Gujarat**, using **historical VIIRS 375m FIRMS data**.
- This region/dataset choice is a **current working assumption**, not a final
  technical decision. It may change based on data availability and quality
  found during exploratory analysis.

## Target Users

Not yet defined beyond the problem statement's framing: analysts/investigators
at or working with NTRO who would use prioritized thermal-source output to
direct further investigation. This will be refined as the project progresses.

## Intended Outcome

A working, defensible hackathon prototype that demonstrates:

- Ingestion and validation of historical FIRMS data for the candidate region.
- Exploratory analysis of spatial/temporal patterns in that data.
- Initial geospatial visualization of thermal detections.

Later milestones (contextual scoring, classification, satellite data
integration, results interface) are intentionally out of scope until the
first milestone is complete and reviewed.
