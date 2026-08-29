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

## Working Assumptions (Not Final Decisions)

- Candidate region: Gujarat.
- Candidate dataset: historical VIIRS 375m FIRMS data.

These are recorded as current working assumptions per the team's stated
intent, and are explicitly not yet validated technical decisions.
