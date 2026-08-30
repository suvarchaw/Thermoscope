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

- **Implemented for multi-year FIRMS acquisition, blocked on credentials
  for actually running it beyond 2023.** The original 2023 dataset
  (`data/raw/fire_archive_SV-C2_794895.csv`) was manually placed and is
  read-only, unchanged by anything below.
- [src/firms_ingestion.py](src/firms_ingestion.py): reusable module for
  NASA's official FIRMS Area API
  (`https://firms.modaps.eosdis.nasa.gov/api/area/csv/...`). Reads
  `FIRMS_MAP_KEY` from the environment — via a local, gitignored `.env`
  file (loaded automatically with `python-dotenv`) or a real environment
  variable — never hardcodes a key, never commits one. Chunks any date
  range into non-overlapping ≤5-day windows (the API's documented
  per-request limit), fetches from the `VIIRS_SNPP_SP` (standard
  processing / historical) source scoped to the Gujarat bounding box
  (reusing `LAT_MIN`/`LAT_MAX`/`LON_MIN`/`LON_MAX` from
  `spatial_recurrence.py`, not redefined), normalizes the API's
  `bright_ti4`/`bright_ti5` column names to this project's existing
  `brightness`/`bright_t31` names, validates required columns, deduplicates,
  and skips re-ingestion if a year is already present on disk.
- [src/ingest_multi_year_gujarat.py](src/ingest_multi_year_gujarat.py):
  orchestrates fetching 2019–2022 into
  `data/raw/firms_gujarat_{year}.csv` (2023 is deliberately not
  re-fetched — see Decisions). **Confirmed working**: a MAP_KEY was
  provided and all four years were successfully ingested and verified
  (correct bounds, full-year date coverage, single satellite, zero
  duplicates, all required columns present) — see PROGRESS.md for exact
  row counts.
- [src/multi_year_gujarat_processing.py](src/multi_year_gujarat_processing.py):
  combines all years actually present (now 2019–2023) into
  `data/processed/gujarat_multi_year_detections.csv` (93,018 rows),
  tagging every row with `year` and re-applying Gujarat bounds defensively
  regardless of source.

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
- **Per-cluster evidence/review table**
  ([src/build_cluster_evidence_review.py](src/build_cluster_evidence_review.py)):
  joins the cluster stats, derived recurrence metrics, and OSM context
  into one row-per-cluster table
  (`data/processed/gujarat_cluster_evidence_review.csv`, 60 rows, 46
  columns) for human review ahead of any persistence/labeling decision.
  This is the intended handoff artifact between "spatial + contextual
  analysis" and whatever persistence-definition discussion comes next — it
  contains no classification, label, or score column of any kind.
- **Recurrence Profile (temporal-recurrence annotation, confirmed).**
  [src/recurrence_profile.py](src/recurrence_profile.py) computes three
  fields — `recurrence_strength` (Strong/Moderate/Limited, from
  `unique_dates`, thresholds anchored to natural gaps in the 60-cluster
  dataset), `short_window_recurrence` (bool, `active_span_days <= 80`),
  and `burst_concentrated` (bool, `top3_days_share > 0.5`, a caution flag
  only, never a tier) — each a pure function of exactly one existing
  metric. [src/build_recurrence_profiles.py](src/build_recurrence_profiles.py)
  applies this to all 60 clusters, preserving every existing evidence
  column unchanged, and writes
  `data/processed/gujarat_cluster_recurrence_profiles.csv`.
  **A Recurrence Profile is explicitly not a source-type classification
  and not a final persistent-source label — it describes only temporal
  recurrence behavior.** OSM context and FIRMS `type` are structurally
  excluded (the functions don't accept them as inputs at all), verified by
  dedicated tests. See DECISIONS.md for the threshold rationale.
- **Cross-year recurrence (descriptive only, spatial baseline preserved).**
  [src/cross_year_recurrence_analysis.py](src/cross_year_recurrence_analysis.py)
  answers "did the same spatial zone recur across years" without
  re-running or altering the 2023 DBSCAN clustering. 2023 detections keep
  their real, already-computed cluster assignments unchanged; historical
  (non-2023) detections are matched to the nearest 2023 cluster centroid
  only if within that cluster's own `extent_radius_m` (a documented,
  deliberately conservative circular-buffer approximation of an often
  irregular true DBSCAN shape — see the module docstring and DECISIONS.md
  for why this was chosen over re-clustering each year independently).
  Unmatched historical detections are retained separately, not discarded.
  Output: `data/processed/gujarat_cluster_cross_year_recurrence.csv`
  (years_detected, unique_years, first/last_year, per-year counts,
  recurs_across_multiple_years) — purely descriptive, does not redefine
  `recurrence_strength` or the other locked Recurrence Profile fields.
  **Run against real 2019–2023 data**: all 60 clusters show
  `recurs_across_multiple_years=True` (44 matched in all 5 years). This is
  the project's first independent, non-circular corroboration of
  persistence — it does not by itself justify any label, but it is
  genuine multi-year evidence, not a single-year artifact.
- **Cluster-level longitudinal feature table (descriptive feature
  engineering, additive).**
  [src/cluster_longitudinal_features.py](src/cluster_longitudinal_features.py)
  combines every existing per-cluster metric (spatial, Recurrence Profile,
  OSM context, cross-year summary — all read unchanged) with new
  descriptive features computed from the same already-validated per-
  detection cross-year assignments: zero-filled per-year detections/active
  days, total/mean/std/CV of annual detections, a simple 5-point OLS trend
  slope with a stated threshold-based direction label, and a monthly
  seasonal distribution with a top-3-months concentration indicator.
  Output: `data/processed/gujarat_cluster_longitudinal_features.csv` (60
  rows, 73 columns) — a new, additive artifact; no existing dataset was
  modified. [src/analyze_cluster_longitudinal_features.py](src/analyze_cluster_longitudinal_features.py)
  answers persistence/trend/seasonality questions descriptively — no
  model, label, or risk score is produced.
- **Integrated cluster evidence layer (interpretability, additive).**
  [src/cluster_integrated_evidence.py](src/cluster_integrated_evidence.py)
  reorganizes all of the above (unchanged) into seven documented evidence
  groups for human review — A. Temporal persistence, B. Activity/
  intensity, C. Trend behavior, D. Seasonality, E. Spatial characteristics,
  F. OSM/context, G. Recurrence-Profile evidence — plus six small,
  single-rule derived indicators (persistence/activity/seasonality
  categories, a notable-OSM-context flag, and an `evidence_notes`
  cross-check field). None of these is a composite/weighted score.
  Output: `data/processed/gujarat_cluster_integrated_evidence.csv` (60
  rows, 68 columns). This is the current best single artifact for human
  review of a cluster's full evidence base ahead of any future
  labeling/ML decision — it is explicitly not a classification and
  contains no source-type or risk field.

### 5. ML / Classification

- **Not started. No model, library, or approach has been chosen.**
- Whether ML is even the right tool for source classification is itself an
  open question to be assessed once exploratory analysis is done, not an
  assumed requirement.
- The Recurrence Profile fields and all underlying continuous metrics are
  intended as candidate future ML features, not as labels — no training
  data or ground truth exists yet.
- A dedicated ground-truth/labeling-strategy investigation (analysis-only,
  no files changed) concluded: the 60 Gujarat-2023 clusters are not
  sufficient for a trained ML classifier (small N, single year, single
  state, uneven class balance, incomplete OSM coverage), but are
  sufficient as a hand-reviewable pilot; OSM proximity must never become
  the training label (circularity risk — a model would just relearn a
  distance rule); multi-year cross-referencing (this milestone) was
  recommended as the first, lowest-cost, non-circular step before any
  manual labeling begins.

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
