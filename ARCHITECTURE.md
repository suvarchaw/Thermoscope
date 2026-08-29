# Architecture — Initial Draft (Provisional)

**Status: DRAFT.** Nothing in this document is a final technical decision
unless explicitly stated otherwise. Anything not marked "confirmed" should be
treated as TBD/provisional and subject to change once real data has been
inspected.

## High-Level Pipeline (Expected Shape Only)

```
Data Sources → Ingestion → Processing → Analysis → ML/Classification → Results/Interface
```

### 1. Data Sources

- NASA FIRMS (VIIRS 375m historical hotspot data) — **provisional primary
  source**, candidate region Gujarat.
- OpenStreetMap (OSM), via the public Overpass API — used for contextual
  investigation (not classification) of recurring FIRMS groups/clusters.
  Reusable lookup module: [src/osm_lookup.py](src/osm_lookup.py). Applied
  so far to the top-20 recurring grid groups
  ([src/osm_context.py](src/osm_context.py)) and to all 60 final DBSCAN
  clusters ([src/build_cluster_osm_context.py](src/build_cluster_osm_context.py),
  output `data/processed/gujarat_cluster_osm_context.csv`). Records nearby
  features (nearest overall, nearest named, counts by broad category:
  industrial/power/waste/agricultural/transport/other) purely as observed
  context — no source-type label or risk score is derived from it.
- Additional satellite data — **eventual/future** source per the problem
  statement. Not part of the first milestone.

### 2. Ingestion

- TBD. Expected to involve reading FIRMS historical export(s) for the
  candidate region/time range into `data/raw/` without modification.

### 3. Processing

- **Partially implemented (provisional).** No cleaning/validation of raw
  field values (e.g. confidence, duplicates) has been done yet — raw FIRMS
  values are currently used as-is.
- What exists: [src/spatial_recurrence.py](src/spatial_recurrence.py) reads
  the raw CSV, filters to the current Gujarat bounding box (lat 20.0–24.7,
  lon 68.0–74.5 — a working assumption, not a validated boundary), and
  groups detections into a fixed-size lat/lon grid sized to the VIIRS 375m
  product's nominal nadir pixel resolution. This is a simple equirectangular
  grid (not equal-area) and does not account for pixel growth away from
  nadir (see `scan`/`track` fields) — acceptable for exploratory grouping,
  not for precise geolocation.
- Output: `data/processed/gujarat_spatial_groups.csv`, one row per occupied
  grid cell.

### 4. Analysis

- **Partially implemented.** The fixed 375m grid
  (`src/spatial_recurrence.py`) was used for initial exploration, but a
  follow-up investigation (Hazira-area analysis) showed it fragments a
  single contiguous ~1.5–2.5km persistent thermal zone into ~20 separate
  grid cells — a fixed grid was found to not be an adequate final
  representation of a physical thermal source.
- **Superseded for spatial grouping purposes by DBSCAN**
  ([src/thermal_clustering.py](src/thermal_clustering.py),
  [src/build_thermal_clusters.py](src/build_thermal_clusters.py)): spatial-only
  density-based clustering with a haversine (geodesic) distance metric,
  parameters selected via a documented sensitivity analysis (see
  DECISIONS.md). This allows irregularly-shaped, variable-density zones to
  be represented as single clusters, and explicitly separates isolated
  ("noise") detections from clustered ones rather than forcing every point
  into a group. The original grid output/scripts are kept as-is (not
  deleted) since they informed this decision.
- Per-cluster stats computed: detection count, unique dates, first/last
  date, active span, mean/max FRP, day/night counts, bounding box, and an
  approximate extent radius.
- No persistence threshold has been chosen — that is explicitly a separate,
  later step from spatial grouping (see DECISIONS.md for why). No claim is
  made that any cluster represents an industrial or otherwise specific
  source type. The FIRMS `type` field is not used anywhere in this
  pipeline (see prior investigation: it is not a reliable label for this
  product version).
- Broader exploratory analysis beyond spatial grouping (e.g. OSM context for
  candidate zones) has been done separately for the top 20 recurring grid
  groups and, later, for all 60 final DBSCAN clusters, but is not part of
  the automated pipeline output.
- **OSM context caveats worth carrying forward**: for large polygon
  features (e.g. a port), Overpass's reported "center" can sit further
  from the query point than the search radius itself, since only part of
  the polygon needs to be within range — so a reported distance can
  slightly exceed the nominal search radius. Geometry classification
  (point facility vs. land-use polygon) is a heuristic based on OSM element
  type and tag key, not a verified geometric analysis. OSM tagging in this
  region is itself inconsistent (e.g. a named power plant tagged only as
  generic `landuse=industrial`, not `power=plant`) — absence of an
  expected tag does not mean absence of the real-world feature.
- **Descriptive persistence-related analysis** has been run over the 60
  clusters ([src/cluster_persistence_analysis.py](src/cluster_persistence_analysis.py),
  [src/run_persistence_analysis.py](src/run_persistence_analysis.py)) to
  understand how detection count, recurrence, timing, and FRP are
  distributed before any persistence rule is designed. This step is
  descriptive only — see PROGRESS.md for findings. It surfaced one
  methodologically relevant fact worth carrying forward: `active_span_days`
  alone is a weak proxy for recurrence (many clusters span nearly the full
  year while being active on only a small fraction of those days), whereas
  `unique_dates` and a derived `occurrence_rate` (unique_dates /
  (active_span_days+1)) appear more discriminating. This observation should
  inform, but does not itself decide, how persistence is eventually
  defined.

### 5. ML / Classification

- **Not started. No model, library, or approach has been chosen.**
- Whether ML is even the right tool for source classification is itself an
  open question to be assessed once exploratory analysis is done, not an
  assumed requirement.

### 6. Results / Interface

- TBD. First milestone only calls for "initial geospatial visualization" —
  no interface, dashboard, or delivery mechanism has been decided.

## Explicit Non-Decisions

The following have **not** been chosen and should not be assumed:

- Database or storage technology.
- Cloud platform or hosting.
- ML model or modeling approach.
- A mapping/geospatial visualization library (matplotlib is in use only for
  a plain statistical histogram so far, not a map).
- Final geographic scope (Gujarat is a current working assumption only).
- Any data cleaning/validation rules for raw FIRMS field values.
- Any persistence threshold or source-classification logic.

**Confirmed so far:** Python (standard library) for data reading/grouping
logic, plus `matplotlib` (visualization) and `scikit-learn` (DBSCAN
clustering) as added dependencies — both single-purpose, standard,
well-justified additions rather than custom reimplementations. See
`requirements.txt`.

## Data Directory Convention (Confirmed)

- `data/raw/` — original, unmodified source data.
- `data/processed/` — cleaned/validated intermediate data.
- `data/gold/` — final, analysis-ready data.
