# Decisions Log

Only decisions actually made are recorded here. This is not a list of
anticipated or planned decisions.

## 2026-08-29

- **Project name:** ThermoScope.
- **Problem statement targeted:** SIH26162 (NTRO, "AI-Based Detection and
  Classification of Industrial Fires and Persistent Thermal Sources Using
  NASA FIRMS, OSM & Satellite Data").
- **Overall approach:** Use NASA FIRMS as a primary data source and add
  contextual analysis (spatial, temporal, industrial/geospatial context, and
  eventually satellite data) rather than attempting to recreate FIRMS itself.
- **First milestone scope:** FIRMS historical data → inspect → clean/validate
  → exploratory analysis → initial geospatial visualization. Later milestones
  are explicitly deferred.
- **Repository structure:** `data/{raw,processed,gold}/`, `src/`, `tests/`,
  `notebooks/`, `results/{maps,figures}/`.

## 2026-08-30

- **Spatial grouping methodology: DBSCAN, not a fixed grid.** The
  Hazira-area investigation showed a fixed 375m grid fragments one
  contiguous ~1.5–2.5km persistent thermal zone into ~20 separate cells,
  with cross-date detection spread (median ~140–200m, p90 up to ~330m,
  max ~500m) exceeding a single grid cell's width. Three candidate
  methodologies were considered (DBSCAN; grid + connected-component
  merging; agglomerative/hierarchical clustering with a distance cut).
  DBSCAN was selected because it is a standard, citable, well-documented
  algorithm (not a custom heuristic); it naturally separates isolated/noise
  detections from real clusters, matching the fact that ~94% of fixed-grid
  cells contained only one detection; it handles irregular/elongated
  cluster shapes without a rigid grid boundary; and it is computationally
  trivial at this data size (sub-second for 17,596 points).

- **DBSCAN parameters: eps=375m, min_samples=8.** Chosen after running a
  documented sensitivity grid (eps ∈ {250, 300, 375, 400, 500}m ×
  min_samples ∈ {3, 5, 8}, 15 combinations total) over all 17,596 Gujarat
  detections. Reasoning, grounded in the actual grid results:
  - At every tested combination, the Hazira zone collapsed into exactly
    **one** cluster (never fragmented) — this criterion did not
    discriminate between candidates, so it was not the deciding factor.
  - `min_samples=8` was the most stable choice across the eps range tested:
    cluster count moved only 54→77 (1.4×) across eps 250–500m, versus
    70→168 (2.4×) at min_samples=5 and 194→711 (3.7×) at min_samples=3.
    Low min_samples values produced results highly sensitive to the exact
    eps chosen — a sign of fragile, low-confidence clusters rather than
    real structure.
  - At min_samples=8 and eps=375m, **all 60** resulting clusters had
    repeated detection dates (unique_dates > 1) — i.e. none were a
    same-day spatial coincidence masquerading as a "recurring" cluster. At
    min_samples=3, ~23% of clusters (95 of 407 at eps=375m) had only a
    single detection date, which is not meaningful recurrence.
  - eps=375m keeps the same physical justification already used for the
    grid (VIIRS nominal pixel resolution), sits within the empirically
    observed p90 cross-date spread (~300–330m) with margin below the
    observed max (~500m), and is the middle value of the tested range
    rather than an untested extreme.
  - The largest cluster's size was stable (2384–2391 detections) across
    all 15 combinations, and was independently confirmed to be exactly the
    Hazira zone in every case — evidence against runaway chaining within
    the tested parameter range.
  - This was not chosen to maximize cluster count or minimize noise:
    eps=500m/min_samples=3 gives more clusters (711) and less noise
    (49.96%) but was rejected for being the least stable configuration
    tested.

- **Persistence classification is explicitly deferred, and is a different
  kind of decision from the clustering parameters above.** `eps` and
  `min_samples` determine which raw detections spatially belong together;
  a persistence threshold (e.g. minimum unique dates or active span) would
  determine which of the resulting 60 clusters count as "persistent"
  versus merely spatially dense. No such threshold has been chosen. It
  requires reviewing the actual distribution of unique_dates/active_span
  across these 60 clusters first, which has not yet been done as a
  distinct step.

- **New dependency: scikit-learn**, added specifically for its DBSCAN
  implementation with haversine-metric support — reimplementing this by
  hand was judged less reliable and harder to defend than using the
  standard library for the job.

- **OSM context lookup radius per cluster: adaptive, not fixed.** For the
  60-cluster OSM investigation, the earlier top-20 approach's flat 2000m
  search radius was replaced with `extent_radius_m + 500m`, floored at
  750m and capped at 3000m. This was necessary because cluster spatial
  extents now range from 169m to 1,748m (the earlier top-20 pass only
  covered grid cells of near-uniform size) — a flat radius would either
  waste Overpass load on tiny clusters or risk missing relevant context for
  large ones. This is a search-parameter decision only; it does not affect
  spatial grouping or any label.

## 2026-08-31

- **Recurrence Profile methodology adopted for temporal-recurrence
  annotation of the 60 DBSCAN clusters.** This is explicitly *not* a
  source-type classification and *not* a final persistent-source label —
  it describes only the temporal recurrence behavior of a spatial cluster,
  as a distinct stage from spatial grouping (DBSCAN, unchanged), OSM
  context (kept fully separate), and any future ML classification (not
  started).

  **Primary axis — `recurrence_strength`, from `unique_dates` only:**
  - Strong: `unique_dates >= 143`
  - Moderate: `73 <= unique_dates < 143`
  - Limited: `unique_dates < 73`

  Empirical basis: these two boundaries are anchored to the two largest,
  unambiguous natural gaps found in the 60-cluster `unique_dates`
  distribution — 143→196 (gap of 53) and 56→73 (gap of 17) — separating
  clusters {0, 7, 8} as Strong and clusters with unique_dates in [73, 143)
  as Moderate. A third, larger gap (196→295) sits inside the Strong tier
  and does not add a further boundary. The thresholds were fixed *before*
  seeing what tier-count distribution they would produce, and were not
  adjusted to hit a target number of clusters per tier.

  Result: **Strong=3, Moderate=12, Limited=45.** The Limited tier is
  intentionally left undivided — no further gap evidence in the current
  60-cluster dataset supports subdividing it, and inventing a sub-cut
  would not be grounded in the data.

  **Secondary flag — `short_window_recurrence` (bool): `active_span_days
  <= 80`.** Independent of the tier above; does not override it. Basis: a
  70-day gap (80→150) cleanly isolates exactly 3 clusters (31, 42, 58).
  This exists specifically so that short-duration-but-internally-recurring
  activity (cluster 42: 17 unique dates packed into a 31-day window) is
  distinguishable from clusters that are simply sparse throughout the
  year — `recurrence_strength` alone would otherwise place both in
  "Limited" with no way to tell them apart.

  **Tertiary caution flag — `burst_concentrated` (bool): `top3_days_share
  > 0.5`.** Explicitly a quality/caution annotation, not a tier and not
  combined into `recurrence_strength`. Basis: unlike the two thresholds
  above, this is a round, interpretable cut ("more than half of a
  cluster's evidence came from at most 3 days") rather than a gap-derived
  one — flagged here as the one threshold in this methodology not
  anchored to an observed distributional gap. Empirically, all 6 clusters
  it flags (38, 43, 48, 54, 58, 59) already fall in the Limited tier, so
  it adds a caution signal within that tier rather than cutting across
  tiers.

  **Explicit separations preserved:** OSM context (industrial proximity,
  named facilities, distance to features, OSM category) was not used in
  computing any of the three fields above, and cannot be — the computing
  functions in `src/recurrence_profile.py` structurally accept only
  `unique_dates`, `active_span_days`, and `top3_days_share` as arguments.
  The FIRMS `type` field was not used. No ML model, risk score, or
  industrial/wildfire/agricultural/other label was produced.

## 2026-08-31 (continued) — Multi-year FIRMS ingestion

- **Data source: NASA FIRMS Area API, `VIIRS_SNPP_SP` source.** Chosen
  over re-requesting a bulk archive-download-tool export (the mechanism
  that produced the original 2023 file) because the Area API supports
  direct geographic (bounding-box) and date filtering per request, so
  Gujarat-only, year-scoped pulls avoid downloading unnecessary pan-India
  volume. Verified directly against NASA's own Area API documentation
  (not assumed): endpoint format, west/south/east/north coordinate order,
  a 5-day maximum `day_range` per request, and that `VIIRS_SNPP_SP` is the
  standard-processing/historical source (`VIIRS_SNPP_NRT` is the
  near-real-time source, noted for the later live-ingestion milestone).

- **Column normalization: `bright_ti4`→`brightness`, `bright_ti5`→`bright_t31`.**
  The Area API returns different VIIRS column names than the existing
  archive-download-tool file this project was built on. Rather than
  changing every downstream module to handle two schemas, the ingestion
  module renames these two columns on read so `brightness`/`bright_t31`
  keep meaning what they already mean everywhere else in the codebase.

- **2023 is not re-fetched via the new API.** The existing raw file
  (`fire_archive_SV-C2_794895.csv`) remains the sole source for 2023,
  reused as-is through the unchanged `spatial_recurrence.read_gujarat_detections`.
  Re-fetching 2023 from a different endpoint risked subtle numerical
  differences from the file every prior milestone (DBSCAN, recurrence
  profiles, OSM context) was built and validated against — not worth the
  risk for a year we already have.

- **Genuine blocker, not a code defect**: no `FIRMS_MAP_KEY` is available
  in this environment. Obtaining one requires a human NASA Earthdata
  account registration (https://firms.modaps.eosdis.nasa.gov/api/map_key/),
  which cannot be done autonomously. Confirmed the API mechanism itself
  works by sending a live request with an invalid test key and receiving
  NASA's own rejection message — ruling out a connectivity or
  implementation problem. Per project rules, no key was invented or
  hardcoded, and no placeholder/fake historical data was written. 2019–2022
  ingestion is fully implemented and tested (with mocked responses) and
  will run without any code changes once a key is set.

  **Resolved 2026-08-31**: the user provided a MAP_KEY. It was written to
  a local, gitignored `.env` file (never committed) and loaded via
  `python-dotenv`, added to `src/firms_ingestion.py` as a config-loading
  convenience — not a change to the ingestion methodology or the
  requirement that credentials stay out of the repository. All four years
  (2019–2022) were ingested and verified against the same schema/bounds
  checks used for the mocked tests. Two background-command timeouts
  occurred mid-run (2022 took longer than the harness's per-command
  timeout under this run's network conditions); both were resolved simply
  by re-running the same script, since already-ingested years are skipped
  and a year's file is only written after that year's full validation
  succeeds — no partial or corrupt data resulted.

- **Cross-year spatial reconciliation rule: nearest-2023-centroid-within-
  extent_radius_m, not independent re-clustering.** Independently running
  DBSCAN on each year would produce cluster IDs with no relationship to
  2023's IDs (arbitrary per-run numbering, no guaranteed same cluster
  count, no guaranteed same boundaries) — comparing such IDs across years
  would answer "did DBSCAN number a cluster the same way twice," not "did
  the same zone recur." Instead, 2023's clustering (eps=375m,
  min_samples=8, unchanged) is treated as the fixed baseline: 2023
  detections keep their real DBSCAN assignments unchanged, and only
  historical (non-2023) detections are matched to the nearest 2023 cluster
  centroid, and only if within that cluster's own `extent_radius_m`
  (already computed as the maximum observed 2023 member distance from
  centroid — not a newly invented radius). This is a known approximation
  — a circular buffer cannot exactly reproduce an irregular DBSCAN shape
  (e.g. cluster 0's elongated Hazira footprint, established in an earlier
  milestone) — accepted as reasonable for a first descriptive pass at
  "did activity recur nearby," and documented as such rather than silently
  assumed correct. Historical detections that don't fall within any
  cluster's radius are retained separately as `unmatched_historical`, not
  discarded and not forced into the nearest cluster regardless of distance.

## 2026-08-31 (continued) — Longitudinal feature engineering

- **Trend direction threshold: ±10% of mean annual detections.** With
  only 5 yearly data points per cluster, a formal statistical significance
  test for trend would overstate precision the data doesn't support.
  Instead, a cluster's 5-point OLS slope is labeled "increasing" or
  "decreasing" only if its magnitude exceeds 10% of that cluster's own
  mean annual detection count; otherwise "stable". This is a simple,
  explicitly documented descriptive rule (`TREND_STABLE_THRESHOLD_FRACTION`
  in `src/cluster_longitudinal_features.py`), not a claim of statistical
  or causal significance, and is reported as such everywhere it's used.

## 2026-08-31 (continued) — Integrated evidence layer derived indicators

- **Persistence ceiling, not an invented cutoff.** `persistence_category`
  = "Persistent" only at `unique_years == 5` — the maximum possible given
  5 years of data currently exist, not an arbitrary number chosen to hit
  a target count.
- **Activity/seasonality splits use sample statistics, not fixed numbers.**
  `activity_category` splits at the sample median of `total_detections_5yr`
  across the current 60 clusters; `seasonality_category` splits at the
  sample 75th percentile of `top3_months_share`. Both are recomputed from
  whatever data is present each run (and printed to the log), rather than
  hardcoded — if the cluster population changes in a future run, these
  splits will shift accordingly rather than silently becoming stale.
- **No composite score.** Every derived indicator in this milestone is a
  direct function of one or two existing columns with an explicitly
  stated rule (see `src/cluster_integrated_evidence.py` module docstring).
  None combines more than two inputs and none is weighted — this was a
  deliberate constraint from the milestone brief, not a limitation
  discovered after the fact.
- **`evidence_notes` reports disagreement; it does not resolve it.** The
  five cross-check rules implemented were chosen because each is
  independently interpretable (e.g. "2023 tier vs. 5-year persistence",
  "named industrial context vs. activity level") — the absence of a note
  is reported as "no checked inconsistency found," explicitly not as
  "confirmed" or "verified", to avoid overstating what an empty
  evidence_notes field means.

## Working Assumptions (Not Final Decisions)

- Candidate region: Gujarat.
- Candidate dataset: historical VIIRS 375m FIRMS data.

These are recorded as current working assumptions per the team's stated
intent, and are explicitly not yet validated technical decisions.
