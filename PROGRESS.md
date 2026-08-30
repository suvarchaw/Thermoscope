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

### Per-cluster evidence/review table for human inspection (2026-08-31)

- [x] Built [src/build_cluster_evidence_review.py](src/build_cluster_evidence_review.py),
      joining `gujarat_thermal_clusters.csv`, `gujarat_clustered_detections.csv`
      (via the existing, unmodified `derived_metrics`/`per_cluster_date_counts`
      functions from `src/cluster_persistence_analysis.py`), and
      `gujarat_cluster_osm_context.csv` into one row-per-cluster table:
      `data/processed/gujarat_cluster_evidence_review.csv` (60 rows, 46
      columns). Contains spatial, temporal, thermal, day/night, and OSM
      evidence side by side, for human review only — **no classification,
      label, or risk score column of any kind**. Overlapping fields between
      the two source tables (centroid, extent, detection_count,
      unique_dates) are verified to match before being deduplicated into a
      single copy, rather than assumed equal.
- Data-quality checks (enforced in code, not just eyeballed): exactly 60
  clusters in, exactly 60 rows out; no duplicate cluster_id at input or
  output; OSM context cluster_id set matches the clusters table exactly
  (no missing, no orphaned rows); missing OSM matches remain blank in the
  output rather than defaulted to any value. Unit tests in
  `tests/test_build_cluster_evidence_review.py` (8 tests, all passing).
- Follow-up descriptive-only inspection
  ([src/analyze_cluster_evidence.py](src/analyze_cluster_evidence.py)):
  figure at `results/figures/cluster_evidence_osm_context_comparison.png`
  comparing `unique_dates` and `occurrence_rate` across a transient,
  plotting-only bucketing of "nearest observed OSM tag category" (not
  persisted anywhere, not a classification). Findings:
  - OSM context breakdown: unnamed industrial-only=29, power=14, named
    industrial=8, no OSM context=5, waste=3, other=1.
  - Clusters with **no OSM context at all** have the lowest median
    recurrence of any group (median unique_dates=9, occurrence_rate=0.034).
  - Clusters whose nearest feature is **named industrial** have the
    highest median recurrence (median unique_dates=35.5,
    occurrence_rate=0.182), but with substantial overlap/spread across all
    groups in the box plots — this is a descriptive tendency, not a clean
    separation, and is not treated as evidence of source type.

### Detailed cluster-level descriptive deep-dive (2026-08-31)

- [x] Built [src/cluster_evidence_deep_dive.py](src/cluster_evidence_deep_dive.py):
      IQR-based outlier detection across 8 metrics, natural-gap analysis
      per metric, day/night and burst-concentration pattern review, and a
      definition-sensitivity probe using 5 illustrative (not adopted)
      persistence rules. Output:
      `data/processed/gujarat_cluster_descriptive_flags.csv` (60 rows,
      descriptive statistical flags only — no classification columns),
      figure at `results/figures/cluster_definition_sensitivity.png`. Unit
      tests in `tests/test_cluster_evidence_deep_dive.py` (9 tests).
- Key findings: 5 compound outliers (clusters 0, 9, 7, 38, 43); two
  distinct temporal *shapes* identified — long-duration/moderate-frequency
  (cluster 0) vs. short-duration/concentrated (cluster 42, 31 unique dates
  concentrated in 31 days); only 1 cluster (17) sits at a genuine
  definition-sensitive boundary once the crude 5-probe count is examined
  properly — the earlier "45 sensitive clusters" reading was refined down
  to this single real boundary case.
- This directly fed the Recurrence Profile methodology proposal (below).

### Recurrence Profile methodology — proposed, reviewed, and implemented (2026-08-31)

- [x] Proposed a persistence/recurrence methodology grounded in the deep
  dive above; reviewed and approved with specific terminology and
  thresholds (see DECISIONS.md for the full rationale).
- [x] **Implemented** as
  [src/recurrence_profile.py](src/recurrence_profile.py) (pure functions,
  each taking only the one numeric input it needs — structurally
  incapable of using OSM or FIRMS `type` data) and
  [src/build_recurrence_profiles.py](src/build_recurrence_profiles.py)
  (runner). Output:
  `data/processed/gujarat_cluster_recurrence_profiles.csv` — every
  existing evidence column preserved unchanged, plus three new columns:
  `recurrence_strength` (Strong/Moderate/Limited), `short_window_recurrence`
  (bool), `burst_concentrated` (bool). Figure at
  `results/figures/cluster_recurrence_profiles.png`.
- Tier counts: **Strong=3, Moderate=12, Limited=45**.
  `short_window_recurrence=True`: 3 clusters (31, 42, 58).
  `burst_concentrated=True`: 6 clusters (38, 43, 48, 54, 58, 59).
  Only cluster 58 has both flags — the single most marginal cluster in the
  dataset (2 unique dates, 8 total detections, right at the DBSCAN
  min_samples=8 floor).
- Unit tests in `tests/test_recurrence_profile.py` (20 tests, covering
  every stated boundary: unique_dates=72/73/142/143,
  active_span_days=80/81, top3_days_share=0.5/just above) and
  `tests/test_build_recurrence_profiles.py` (5 tests, verifying all
  existing evidence columns survive unchanged and that differing OSM
  context never changes the computed Recurrence Profile).
- **A Recurrence Profile is explicitly not a source-type classification
  and not a final persistent-source label** — it describes only the
  temporal recurrence behavior of a spatial cluster. OSM context and
  FIRMS `type` were not used anywhere in this computation (verified by
  test and by the functions' signatures).

### Ground-truth/labeling strategy investigation (2026-08-31)

- [x] Analysis-only milestone (no files changed): evaluated candidate
  ground-truth sources, the OSM-proximity circularity risk, whether the 60
  Gujarat-2023 clusters are sufficient for ML (concluded: no — sufficient
  only as a hand-reviewable pilot), and recommended multi-year Gujarat
  FIRMS data as the first, lowest-cost, non-circular next step before any
  labeling begins.

### Multi-year Gujarat FIRMS ingestion (2026-08-31)

- [x] Investigated NASA's official FIRMS Area API (verified against NASA's
  own documentation, not assumed): endpoint
  `https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{west},{south},{east},{north}/{day_range}/{date}`,
  max `day_range`=5 days/request, `VIIRS_SNPP_SP` = historical
  standard-processing source, coordinate order confirmed as
  west,south,east,north. Confirmed (via live request) that the API
  requires a registered MAP_KEY — a `DEMO_KEY` test request was correctly
  rejected by NASA's server, confirming the mechanism works and that no
  key is currently configured.
- [x] Built [src/firms_ingestion.py](src/firms_ingestion.py): reusable
  ingestion module — reads `FIRMS_MAP_KEY` from the environment only
  (never hardcoded), chunks any date range into non-overlapping ≤5-day
  windows, fetches via the Area API, normalizes the API's
  `bright_ti4`/`bright_ti5` columns to this project's existing
  `brightness`/`bright_t31` names, validates required columns, deduplicates,
  and skips re-ingestion if a year's file already exists. Raises a clear,
  actionable `FirmsConfigError` (with setup instructions) when the key is
  missing — never fabricates data or fails silently.
- [x] [src/ingest_multi_year_gujarat.py](src/ingest_multi_year_gujarat.py):
  orchestrator targeting 2019–2022 (2023 intentionally reused from the
  existing, unmodified raw archive rather than re-fetched). **Actually
  run**: all 4 years correctly reported `config_error` — **no
  FIRMS_MAP_KEY is available in this environment, and one cannot be
  obtained autonomously (requires a human NASA Earthdata registration)**.
  This is the genuine, honestly-reported data-access blocker for this
  milestone. No placeholder/fake historical raw files were created.
- [x] [src/multi_year_gujarat_processing.py](src/multi_year_gujarat_processing.py):
  combines 2023 (via the existing, unchanged
  `spatial_recurrence.read_gujarat_detections`) with any ingested
  historical years into `data/processed/gujarat_multi_year_detections.csv`,
  re-applying Gujarat bounds defensively and tagging every row with `year`.
  **Actually run**: produced a combined file containing only 2023's 17,596
  detections (the only year currently available), with the 4 missing years
  explicitly reported, not silently omitted.
- [x] [src/cross_year_recurrence_analysis.py](src/cross_year_recurrence_analysis.py):
  documented spatial reconciliation between independently-clustered years
  and the fixed 2023 DBSCAN baseline (see DECISIONS.md for the full
  rationale) plus descriptive-only cross-year metrics (years_detected,
  unique_years, first/last_year, per-year detection counts/unique dates,
  recurs_across_multiple_years). Does not redefine recurrence tiers, does
  not use OSM, does not use FIRMS `type`. **Actually run**: output
  `data/processed/gujarat_cluster_cross_year_recurrence.csv` (60 rows), all
  currently showing `unique_years=1` (2023 only) since no historical years
  exist yet — figure at `results/figures/cluster_cross_year_recurrence.png`
  honestly labeled as single-year, not a fabricated multi-year comparison.
- Unit tests: `tests/test_firms_ingestion.py` (24, fully mocked — no
  network/credentials needed), `tests/test_multi_year_gujarat_processing.py`
  (10), `tests/test_cross_year_recurrence_analysis.py` (12, including a
  synthetic multi-year case proving the heatmap code path works correctly
  even though real multi-year data isn't available yet). All 136 project
  tests pass.
- **Ready the moment a MAP_KEY is provided**: set `FIRMS_MAP_KEY` and
  re-run `src/ingest_multi_year_gujarat.py` — already-ingested years are
  skipped automatically, and the processing/analysis scripts require no
  changes to pick up new years.

### Blocker resolved — 2019–2022 ingested and cross-year analysis run (2026-08-31)

- [x] User provided a FIRMS_MAP_KEY. Added `.env` support to
  [src/firms_ingestion.py](src/firms_ingestion.py) via `python-dotenv`
  (already a project dependency) so the key loads automatically from a
  local, gitignored `.env` file — no methodology change, purely a
  credential-loading convenience. Verified the key against a live
  single-day request before committing to the full run.
- [x] **Ran the existing, unmodified pipeline for real**:
  `data/raw/firms_gujarat_{2019,2020,2021,2022}.csv` ingested
  successfully (17,009 / 19,332 / 20,881 / 18,200 rows respectively).
  Each file verified: correct Gujarat bounds, full Jan 1–Dec 31 date
  coverage, single satellite (N/SNPP), zero duplicate keys, all required
  columns present, confidence distributions consistent in shape with the
  existing 2023 file. Original 2023 raw file checksum unchanged.
- [x] Two ingestion attempts were killed by the harness's background
  command timeout partway through a year (2022 took longer than expected
  under this run's network conditions); re-running was safe and lossless
  because `fetch_year_gujarat` skips years already written to disk and
  only writes a year's file after that year's data is fully validated —
  no partial/corrupt files were ever produced.
- [x] Ran `src/multi_year_gujarat_processing.py`:
  `data/processed/gujarat_multi_year_detections.csv` now contains all
  93,018 Gujarat detections across 2019–2023.
  Ran `src/cross_year_recurrence_analysis.py` (spatial reconciliation
  rule unchanged from proposal): **all 60 of 60 2023 clusters now show
  `recurs_across_multiple_years=True`** — 44 clusters matched in all 5
  years, 7 in 4 years, 4 in 3 years, 5 in 2 years (the weakest,
  predominantly matching only 2022–2023). 53,189 of 75,422 historical
  (non-2023) detections were unmatched to any cluster (~70.5%), broadly
  consistent with 2023's own 67.37% DBSCAN noise rate.
  Cluster 0 (Hazira) shows near-identical year-over-year behavior:
  2,266/1,961/2,203/2,111/2,386 detections and 298/293/300/293/295 unique
  active days across 2019–2023 — the first genuinely independent,
  non-circular corroboration of a persistent zone this project has
  produced. Figure at `results/figures/cluster_cross_year_recurrence.png`
  regenerated as a real 5-year heatmap.
- All 136 tests still pass; all pre-existing outputs (DBSCAN, OSM,
  recurrence profiles) verified unchanged (identical row counts).

### Cluster-level longitudinal feature table, 2019–2023 (2026-08-31)

- [x] Built [src/cluster_longitudinal_features.py](src/cluster_longitudinal_features.py):
  a new, additive per-cluster feature table combining every existing
  spatial/Recurrence-Profile/OSM metric (read unchanged from
  `gujarat_cluster_recurrence_profiles.csv`) and the existing cross-year
  summary (`gujarat_cluster_cross_year_recurrence.csv`) with new
  descriptive longitudinal features: zero-filled per-year
  detections/active-days (2019–2023), total/mean/std/coefficient-of-
  variation of annual detections, a simple OLS trend slope + threshold-
  based direction label (increasing/stable/decreasing — a stated
  descriptive heuristic, not a significance test), and a monthly/seasonal
  distribution (aggregated across all 5 years) with a top3-months
  concentration indicator, reusing the exact per-detection cross-year
  assignments from the previous milestone (not recomputed). Output:
  `data/processed/gujarat_cluster_longitudinal_features.csv` (60 rows, 73
  columns). Does not modify any existing dataset; does not change DBSCAN,
  Recurrence Profile thresholds, or the spatial reconciliation rule.
- [x] [src/analyze_cluster_longitudinal_features.py](src/analyze_cluster_longitudinal_features.py):
  descriptive-only analysis answering the six posed questions, with two
  figures (`results/figures/cluster_longitudinal_overview.png`,
  `results/figures/cluster_seasonal_patterns.png`).
- Unit tests: `tests/test_cluster_longitudinal_features.py` (21 tests).
  **All 157 project tests pass.**

**Key findings (descriptive only — no causal or classification claims):**
- 44 of 60 clusters (73%) were detected in all 5 years; the remaining 16
  span 2–4 years — none was detected in only 1 year (consistent with the
  prior milestone's cross-year matching result).
- Trend direction split: 34 increasing, 14 decreasing, 12 stable (by the
  stated ±10%-of-mean threshold on a 5-point OLS slope — explicitly not a
  statistical significance claim).
- Highest total 2019–2023 activity: cluster 0 (10,927), far above cluster
  1 (1,419) and cluster 17 (1,377) — consistent with cluster 0's outlier
  status in every prior milestone.
- Highest interannual variability (coefficient of variation, restricted to
  clusters with mean annual detections ≥5 to avoid noise): clusters 52,
  40, 18, 34, 12 — several show one dominant year against otherwise
  near-zero activity (e.g. cluster 18: 0/0/0/23/63).
- Most seasonally concentrated: cluster 58 (100% of its detections in its
  top 3 months) and cluster 42 (98%) — both already flagged in the
  Recurrence Profile milestone as `short_window_recurrence` cases: the two
  independent measures agree. Cluster 25 shows a sharp, unexplained
  May–July peak unlike most other high-activity clusters' broader dip
  across the monsoon months (Jun–Sep) — reported as an observation only.
- Cross-tabulating with `recurrence_strength`: all 3 Strong and all 12
  Moderate clusters were detected in all 5 years; only 29 of 45 Limited
  clusters were — i.e. a 2023-based "Limited" tier does not necessarily
  mean a cluster is not multi-year persistent, just that its 2023 volume
  was low. This nuance was not visible before this milestone.

### Integrated, interpretable cluster evidence layer (2026-08-31)

- [x] Built [src/cluster_integrated_evidence.py](src/cluster_integrated_evidence.py):
  reorganizes every existing per-cluster metric (spatial, Recurrence
  Profile, OSM context, cross-year summary, longitudinal features — all
  read unchanged from `gujarat_cluster_longitudinal_features.csv`) into
  seven documented evidence groups (A. Temporal persistence,
  B. Activity/intensity, C. Trend behavior, D. Seasonality,
  E. Spatial characteristics, F. OSM/context, G. Recurrence-Profile
  evidence) plus six small, single-rule derived indicators — no composite
  or weighted score. Output:
  `data/processed/gujarat_cluster_integrated_evidence.csv` (60 rows, 68
  columns) — new and additive; no existing dataset modified.
- Derived indicators (each a direct function of 1–2 existing columns,
  thresholds computed fresh from the sample, not invented):
  `persistence_category` (Persistent = unique_years==5, the natural
  ceiling), `activity_category` (High/Low split at the sample median
  total_detections_5yr), `persistence_activity_quadrant` (combination of
  the two), `seasonality_category` (Strongly seasonal = top3_months_share
  ≥ sample p75), `has_notable_osm_context` (any OSM feature found nearby),
  and `evidence_notes` — a semicolon-separated list of specific,
  individually-documented cross-checks between independent evidence
  sources (e.g. "2023 recurrence_strength=Limited but persistent across
  all 5 years"; "named industrial OSM feature nearby despite below-median
  activity"). `evidence_notes` surfaces agreement/disagreement for human
  review — it does not resolve it and is not a score.
- Two figures:
  `results/figures/cluster_integrated_evidence_overview.png` (persistence
  vs. activity quadrants, seasonality distribution, trend-by-persistence,
  OSM context breakdown) and
  `results/figures/cluster_evidence_agreement.png` (a small contingency
  matrix of 2023 `recurrence_strength` vs. 5-year `unique_years`).
- Unit tests: `tests/test_cluster_integrated_evidence.py` (17 tests,
  including an explicit check that no risk-score or source-type column
  exists in the output). **All 174 project tests pass.**

**Key findings (descriptive only):**
- Quadrants: Persistent/High-activity=29, Persistent/Low-activity=15,
  Intermittent/Low-activity=15, Intermittent/High-activity=1.
- 15 of 60 clusters flagged strongly seasonal; clusters 54 and 58 are
  100% concentrated in their top 3 months.
- **All 14 "decreasing"-trend clusters are also `Persistent`** (detected
  in all 5 years) — no intermittent cluster showed a decreasing trend.
  "Increasing" trend is roughly evenly split between persistent (18) and
  intermittent (16) clusters.
- The agreement matrix shows **every Strong and every Moderate
  recurrence_strength cluster is also 5-year persistent** (no
  off-diagonal cases in those two rows); the Limited tier spans the full
  2–5 year range, with 29 of 45 still fully persistent — the clearest
  visual confirmation yet of the "2023 tier undersells persistence" nuance
  found in the previous milestone.
- Only 3 clusters (31, 42, 55) combine a *named* industrial OSM feature
  with below-median 5-year activity — the specific, individually
  interesting "enticing proximity, weak evidence" cases, clearly
  distinguished from the much larger (41-cluster) "2023 tier understates
  persistence" pattern, which is a systematic, non-anomalous fact about
  most clusters rather than a rare exception.
- 43 of 60 clusters have at least one evidence-note flag; 17 show no
  checked inconsistency (explicitly reported as "no flagged disagreement
  found," not as "confirmed").

## Not Started

ML classification, training-label creation, source-type labeling,
risk scoring, satellite imagery, additional data sources, dashboard
development, live FIRMS ingestion, and LLM integration have not been
started and are out of scope until explicitly requested.
