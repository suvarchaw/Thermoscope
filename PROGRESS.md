# Progress

## Current Milestone

**Milestone 1 (complete):** FIRMS historical data → inspect dataset →
clean and validate → exploratory analysis → initial geospatial
visualization.

**Milestone 2 (in progress): Persistent Thermal Source Detection Engine.**
Spatial grouping (DBSCAN) is complete; persistence-related descriptive
analysis and OSM contextual investigation are complete for all 60 clusters.
No persistence threshold, source-type label, or risk score has been
defined yet — that is the next open decision, not yet made.

## End of Day — 2026-08-30

Today's work, in order: fixed-grid recurrence analysis → top-20 OSM
context → Hazira-area investigation (showed the grid fragments a real
zone) → DBSCAN methodology proposal and approval → parameter sensitivity
analysis → final 60-cluster spatial grouping → descriptive persistence
analysis of the 60 clusters → nighttime-dominance investigation → OSM
context for all 60 clusters. Full detail for each step is in the dated
entries below. Nothing beyond spatial grouping + descriptive/contextual
analysis has been decided — no persistence threshold, no source-type
label, no ML, no dashboard.

## Status

- [x] Project documentation created (this file and its siblings).
- [x] Repository directory structure created.
- [x] FIRMS historical data obtained: `data/raw/fire_archive_SV-C2_794895.csv`
      (579,733 rows, VIIRS 375m, S-NPP, calendar year 2023, pan-India extent).
- [x] Dataset inspected (schema, dtypes, date range, geographic extent,
      missing values, duplicates, categorical field breakdowns).
- [x] FIRMS field semantics investigated against NASA documentation,
      including a known bug affecting the `type` field for this product
      version — `type` is confirmed unsuitable as a ground-truth label.
- [x] Gujarat-bounds feasibility check performed (17,596 detections, 3.04%
      of the full file; enough spatio-temporal recurrence to continue).
- [x] Spatial-temporal recurrence analysis implemented (fixed 375m grid,
      [src/spatial_recurrence.py](src/spatial_recurrence.py)): per-group
      detection count, unique dates, first/last date, active span, mean/max
      FRP, day/night counts. Output: `data/processed/gujarat_spatial_groups.csv`
      (11,490 groups). Histogram at
      `results/figures/gujarat_unique_dates_per_group_hist.png`.
- [x] Top-20 recurring-group exploration and interactive map
      ([src/gujarat_recurrence_explore.py](src/gujarat_recurrence_explore.py)):
      `results/maps/gujarat_recurrence_map.html`.
- [x] OSM contextual investigation for the top 20 recurring groups
      ([src/osm_context.py](src/osm_context.py), Overpass API, 2km radius,
      no labels assigned): `data/processed/gujarat_top20_osm_context.csv`,
      `results/maps/gujarat_top20_osm_context_map.html`.
- [x] Hazira-area spatial-scale investigation
      ([src/hazira_cluster_analysis.py](src/hazira_cluster_analysis.py)):
      showed the 375m grid fragments one contiguous ~1.5–2.5km zone into
      ~20 separate cells. Output: `results/maps/hazira_cluster_raw_detections_map.html`,
      `results/figures/hazira_cluster_scale_comparison.png`.
- [x] **Persistent Thermal Source Detection Engine — spatial grouping stage
      (DBSCAN) implemented.** Documented parameter sensitivity analysis run
      across eps ∈ {250,300,375,400,500}m × min_samples ∈ {3,5,8}
      ([src/dbscan_sensitivity_analysis.py](src/dbscan_sensitivity_analysis.py),
      figure at `results/figures/dbscan_parameter_sensitivity.png`).
      Selected eps=375m, min_samples=8 (reasoning in DECISIONS.md) and built
      the clustering output
      ([src/build_thermal_clusters.py](src/build_thermal_clusters.py),
      core logic in [src/thermal_clustering.py](src/thermal_clustering.py)):
      60 clusters, 5,741 clustered detections, 11,855 noise detections
      (67.37%). Outputs: `data/processed/gujarat_thermal_clusters.csv`,
      `data/processed/gujarat_clustered_detections.csv`,
      `results/maps/gujarat_thermal_clusters.html`. The Hazira zone is now
      represented as a single cluster (2,386 detections) instead of ~20 grid
      cells. Unit tests in `tests/test_thermal_clustering.py` (10 tests, all
      passing).
- [x] Descriptive persistence-related analysis of the 60 DBSCAN clusters
      ([src/cluster_persistence_analysis.py](src/cluster_persistence_analysis.py),
      [src/run_persistence_analysis.py](src/run_persistence_analysis.py)):
      distributions, correlations, and gap analysis for detection_count,
      unique_dates, active_span_days, occurrence_rate, night/day split,
      FRP, and single-day dominance. Figure at
      `results/figures/cluster_persistence_distributions.png`. Key findings
      recorded below and in ARCHITECTURE.md. **No persistence threshold was
      defined** — this was explicitly descriptive only. Unit tests in
      `tests/test_cluster_persistence_analysis.py` (8 tests, all passing).
- [ ] Data cleaning/validation of the raw records themselves (e.g. handling
      confidence flags) not yet done — all analysis so far uses raw values
      as-is.
- [ ] Persistence classification (deciding which spatial clusters count as
      "persistent" vs. merely spatially dense) — explicitly NOT done. The
      descriptive analysis above is preparatory input for this decision,
      not the decision itself.
- [ ] Source type labeling/classification — not done, and no field in the
      current data (including `type`) is considered suitable as a
      ground-truth label.

### Key findings from the persistence-related descriptive analysis (2026-08-30)

- **unique_dates is heavily right-skewed with a clear 3-cluster "top tier"**:
  clusters 0 (295 unique dates, the Hazira zone), 7 (196), and 8 (143) sit
  well above the rest (next-highest is 121, then a further gap down to 110).
  The other 57 clusters range from 2 to 121 unique dates with no comparably
  sharp break.
- **active_span_days is a poor stand-alone indicator of recurrence**: median
  active_span is 335 days (most clusters span nearly the full year) even
  though median unique_dates is only 22 — i.e. many clusters are only
  active on a small fraction of the days between their first and last
  detection. Correlation between unique_dates and active_span_days is only
  moderate (r=0.405).
- **occurrence_rate (unique_dates / (active_span+1)) varies far more
  informatively**: median 0.079, max 0.808 (cluster 0). This looks more
  discriminating than active_span alone.
- **detection_count and unique_dates are fairly strongly correlated**
  (r=0.741), as expected, but detections_per_active_date is usually close
  to 1 (median 1.2) — most clusters typically produce about one detection
  per active day, not repeated same-day bursts.
- **Almost all clusters are dominated by nighttime detections**: median
  night_fraction is 0.984; 52 of 60 clusters are >70% nighttime, only 1
  is mostly daytime. This was not expected going in and has no explanation
  attempted here — it is reported as an observation only.
- **Single-day domination is rare**: only 1 of 60 clusters (cluster 58, the
  smallest, right at the min_samples=8 floor) has more than half its
  detections on one date. Most clusters are not artifacts of one busy day.
- **FRP shows no relationship to recurrence**: Pearson r(mean_frp,
  unique_dates) = -0.024 — intensity and how often a cluster recurs appear
  independent in this dataset.

### Follow-up: nighttime dominance investigation (2026-08-30)

- [x] Investigated the nighttime dominance found above
      ([src/nighttime_dominance_investigation.py](src/nighttime_dominance_investigation.py)),
      using only existing Gujarat FIRMS/cluster data. Figure at
      `results/figures/nighttime_dominance_investigation.png`. Unit tests
      in `tests/test_nighttime_dominance_investigation.py` (6 tests, all
      passing).
- Confirmed a normal two-overpass VIIRS/SNPP sampling structure (day pass
  ~7.2–9.5h UTC, night pass ~19.75–22.0h UTC, both tight, single-mode
  windows) — not a timing anomaly.
- Found a sharp divergence: all Gujarat detections are 61.8% day / 38.2%
  night, but detections that end up **inside** a DBSCAN cluster are 16.8%
  day / 83.2% night, while **noise** detections are 83.6% day / 16.4% night
  — day and night trade places almost exactly between the two groups.
- Found that **100% of nighttime detections in this dataset carry
  `confidence` = nominal or high; zero are "low"**, whereas 32.7% of
  daytime detections are low-confidence. This is consistent with NASA's
  own documented VIIRS confidence algorithm (low-confidence daytime
  detections are typically sun-glint related — glint cannot occur at
  night), previously verified in the FIRMS field-semantics investigation.
  This is a plausible, data-consistent contributing factor, not a proven
  causal explanation.
- Cluster 0 (Hazira, the largest/most recurring cluster) has a
  meaningfully lower night_fraction (0.713) than the remaining 57 clusters
  (median 0.989) — the most prominent persistent zone does not follow the
  extreme pattern seen elsewhere.
- Conclusion: night_fraction looks like a real, structured signal (not an
  artifact of overpass timing), but is entangled with a known sensor-side
  confidence effect — it should not yet be treated as a clean physical
  signal without accounting for that.

### OSM contextual analysis of all 60 DBSCAN clusters (2026-08-30)

- [x] Built a reusable OSM lookup module
      ([src/osm_lookup.py](src/osm_lookup.py): Overpass query construction,
      dual-endpoint fallback, tag/geometry classification, adaptive search
      radius) and a cluster-context builder
      ([src/build_cluster_osm_context.py](src/build_cluster_osm_context.py)).
      Output: `data/processed/gujarat_cluster_osm_context.csv` (60 rows,
      one per cluster), map at
      `results/maps/gujarat_cluster_osm_context_map.html`. Unit tests in
      `tests/test_osm_lookup.py` (12 tests, all passing, no network calls).
- All 60 clusters resolved successfully (1 of 60 needed a retry after a
  transient Overpass server error).
- 55 of 60 clusters had at least one matching OSM feature within their
  adaptive search radius; 5 had none.
- 19 of 60 clusters had a **named** nearest feature (e.g. ArcelorMittal
  Nippon Steel India, TATA Chemicals, Vadinar Power Plant, Mundra Port,
  Gujarat Narmada Valley Fertilizers & Chemicals Limited) spread across
  multiple, geographically distinct clusters — not confined to the Hazira
  area alone.
- This is observed nearby OSM context only. No cluster has been labeled
  "industrial" or any other source type, and no risk score was computed.

## Not Started

ML classification, training-label creation, persistence-threshold
definition, satellite imagery, additional data sources, dashboard
development, and LLM integration have not been started and are out of
scope until explicitly requested.
