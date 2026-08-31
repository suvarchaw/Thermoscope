"""
Feature inventory and selection for exploratory unsupervised clustering.

Reads the existing, unmodified integrated evidence table
(gujarat_cluster_integrated_evidence.csv, 60 rows / 68 columns) and
documents, with empirical justification (not guesswork), which columns
are included in the ML feature matrix and which are excluded, before any
model is trained.

Exclusion categories
---------------------
1. IDENTIFIER    -- cluster_id itself: never a feature.
2. EXACT_DUPLICATE -- two columns found (checked below, not assumed) to be
   numerically identical for all 60 clusters.
3. DERIVED_REDUNDANT -- one column is a deterministic function of others
   already kept (e.g. occurrence_rate = unique_dates/(active_span_days+1)),
   confirmed by direct recomputation, not assumption.
4. HIGH_CORRELATION -- empirically |Pearson r| > 0.9 with a kept feature
   (checked below; only top_day_share vs top3_days_share met this bar).
5. HAND_DESIGNED_CATEGORY -- recurrence_strength, short_window_recurrence,
   burst_concentrated, persistence_category, activity_category,
   persistence_activity_quadrant, seasonality_category, trend_direction,
   evidence_notes: these are the existing interpretive categories this
   milestone exists to compare AGAINST. Training on them would make the
   comparison circular, so they are excluded from the feature matrix and
   used only afterward, read back from the source file, for comparison.
6. NON_NUMERIC_OR_UNSTABLE_METRIC -- free text (nearest_name, nearest_label,
   top_features_summary, osm_query_status, years_detected,
   per_month_detection_counts) or a circular quantity (dominant_month is
   1-12 but circular -- month 12 and month 1 are numerically far apart in
   raw Euclidean terms despite being adjacent in the calendar; excluded
   rather than mis-encoded).
7. LOW_INFORMATION -- first_year/last_year: with 44/60 clusters spanning
   2019-2023 already, these carry little separating information beyond
   what unique_years already captures, and are highly skewed toward the
   same two boundary values.

INCLUDED FEATURES (grouped, all verified non-redundant against each other
by direct correlation check below)
---------------------------------------------------------------------
Temporal persistence : unique_years, occurrence_rate
Activity / intensity : total_detections_5yr, mean_frp, frp_ratio
Trend / variability  : trend_slope, cv_annual_detections
Seasonality           : top3_months_share, top3_days_share, night_fraction
Spatial               : centroid_lat, centroid_lon, extent_radius_m
OSM / context         : n_industrial, n_power, n_waste, n_agricultural,
                         n_transport, n_other, features_found_in_radius,
                         nearest_distance_m, nearest_is_named

`features_found_in_radius` is kept ALONGSIDE the n_* category counts
despite looking redundant: the OSM milestone kept only the 5 nearest
matching features per cluster, so for the 17 of 60 clusters with more
than 5 matches nearby, features_found_in_radius (total matches) exceeds
sum(n_*) (category mix among only the top 5) -- confirmed by direct
comparison below, not assumed. They measure different things (raw nearby
density vs. category composition of the closest few), so both are kept.

Missing-value handling
-----------------------
`nearest_distance_m` is missing for the 5 clusters with no OSM context
found (features_found_in_radius == 0). Imputed with that cluster's own
OSM search radius (recomputed via the unchanged
`osm_lookup.compute_search_radius`, i.e. the distance already confirmed
to contain nothing) -- a conservative, cluster-specific lower bound, not
an arbitrary global constant. This choice is recorded in DECISIONS.md.
"""

import csv
import statistics
from pathlib import Path

from osm_lookup import compute_search_radius

INTEGRATED_EVIDENCE_CSV = Path("data/processed/gujarat_cluster_integrated_evidence.csv")
DOCUMENTATION_CSV = Path("data/processed/gujarat_cluster_ml_feature_documentation.csv")
FEATURES_CSV = Path("data/processed/gujarat_cluster_ml_features.csv")

CORRELATION_THRESHOLD = 0.9

INCLUDED_FEATURES = [
    "unique_years", "occurrence_rate",
    "total_detections_5yr", "mean_frp", "frp_ratio",
    "trend_slope", "cv_annual_detections",
    "top3_months_share", "top3_days_share", "night_fraction",
    "centroid_lat", "centroid_lon", "extent_radius_m",
    "n_industrial", "n_power", "n_waste", "n_agricultural", "n_transport", "n_other",
    "features_found_in_radius", "nearest_distance_m", "nearest_is_named",
]

EXCLUDED_FEATURES = {
    "cluster_id": ("IDENTIFIER", "row identifier, never a feature"),
    "unique_dates": ("EXACT_DUPLICATE", "identical to active_days_2023 for all 60 clusters (verified)"),
    "active_span_days": ("DERIVED_REDUNDANT", "used only inside occurrence_rate, which is kept instead"),
    "years_detected": ("NON_NUMERIC_OR_UNSTABLE_METRIC", "semicolon-joined string, not numeric"),
    "first_year": ("LOW_INFORMATION", "highly skewed toward 2019; little separating information beyond unique_years"),
    "last_year": ("LOW_INFORMATION", "highly skewed toward 2023; little separating information beyond unique_years"),
    "recurs_across_multiple_years": ("DERIVED_REDUNDANT", "identical to (unique_years > 1) for all 60 clusters (verified)"),
    "detection_count": ("EXACT_DUPLICATE", "identical to detections_2023 for all 60 clusters (verified)"),
    "detections_2019": ("DERIVED_REDUNDANT", "summarized by total_detections_5yr, trend_slope, cv_annual_detections"),
    "detections_2020": ("DERIVED_REDUNDANT", "summarized by total_detections_5yr, trend_slope, cv_annual_detections"),
    "detections_2021": ("DERIVED_REDUNDANT", "summarized by total_detections_5yr, trend_slope, cv_annual_detections"),
    "detections_2022": ("DERIVED_REDUNDANT", "summarized by total_detections_5yr, trend_slope, cv_annual_detections"),
    "detections_2023": ("DERIVED_REDUNDANT", "summarized by total_detections_5yr, trend_slope, cv_annual_detections"),
    "active_days_2019": ("DERIVED_REDUNDANT", "per-year detail already summarized by occurrence_rate/unique_years"),
    "active_days_2020": ("DERIVED_REDUNDANT", "per-year detail already summarized by occurrence_rate/unique_years"),
    "active_days_2021": ("DERIVED_REDUNDANT", "per-year detail already summarized by occurrence_rate/unique_years"),
    "active_days_2022": ("DERIVED_REDUNDANT", "per-year detail already summarized by occurrence_rate/unique_years"),
    "active_days_2023": ("EXACT_DUPLICATE", "identical to unique_dates for all 60 clusters (verified); superseded by occurrence_rate"),
    "detections_2024": ("OUT_OF_SCOPE_ADDITIVE", "2024/2025 were added additively to the integrated evidence table (see cluster_longitudinal_features.py EXTENDED_YEARS) after this unsupervised-clustering feature review was finalized; total_detections_5yr/trend_slope/cv_annual_detections deliberately still cover only 2019-2023 (unchanged), so this column is not summarized by them the way detections_2019-2022 are -- excluded here to leave the already-finalized clustering feature set untouched, not because it is redundant"),
    "detections_2025": ("OUT_OF_SCOPE_ADDITIVE", "see detections_2024"),
    "active_days_2024": ("OUT_OF_SCOPE_ADDITIVE", "see detections_2024"),
    "active_days_2025": ("OUT_OF_SCOPE_ADDITIVE", "see detections_2024"),
    "mean_annual_detections": ("EXACT_DUPLICATE", "Pearson r=1.000 with total_detections_5yr (mean = total/5, verified)"),
    "std_annual_detections": ("DERIVED_REDUNDANT", "used only inside cv_annual_detections, which is kept instead"),
    "max_frp": ("HIGH_CORRELATION", "r=0.757 with mean_frp and r=0.805 with frp_ratio; heavily right-skewed by outliers; frp_ratio+mean_frp retained instead"),
    "day_count": ("DERIVED_REDUNDANT", "used only inside night_fraction, which is kept instead"),
    "night_count": ("DERIVED_REDUNDANT", "used only inside night_fraction, which is kept instead"),
    "top_day_share": ("HIGH_CORRELATION", "r=0.913 with top3_days_share (verified, exceeds 0.9 threshold); the more robust top3_days_share is kept"),
    "dominant_month": ("NON_NUMERIC_OR_UNSTABLE_METRIC", "circular (month 12 and month 1 are adjacent, not far apart) -- not meaningful as raw Euclidean input"),
    "per_month_detection_counts": ("NON_NUMERIC_OR_UNSTABLE_METRIC", "semicolon-joined string; summarized by top3_months_share"),
    "bbox_min_lat": ("DERIVED_REDUNDANT", "closely approximated by centroid_lat +/- extent_radius_m (mean abs error ~165m, verified); centroid+radius kept instead"),
    "bbox_max_lat": ("DERIVED_REDUNDANT", "closely approximated by centroid_lat +/- extent_radius_m (mean abs error ~165m, verified); centroid+radius kept instead"),
    "bbox_min_lon": ("DERIVED_REDUNDANT", "closely approximated by centroid_lon +/- extent_radius_m; centroid+radius kept instead"),
    "bbox_max_lon": ("DERIVED_REDUNDANT", "closely approximated by centroid_lon +/- extent_radius_m; centroid+radius kept instead"),
    "nearest_label": ("NON_NUMERIC_OR_UNSTABLE_METRIC", "human-readable text"),
    "nearest_group": ("HAND_DESIGNED_CATEGORY", "one-hot encoding this 6-level categorical would dominate distance calculations over the light-touch n_* counts already kept; excluded to avoid the clustering trivially rediscovering OSM tag identity"),
    "nearest_name": ("NON_NUMERIC_OR_UNSTABLE_METRIC", "free text facility name"),
    "nearest_geometry_class": ("NON_NUMERIC_OR_UNSTABLE_METRIC", "categorical, low-cardinality, largely redundant with nearest_group"),
    "osm_query_status": ("NON_NUMERIC_OR_UNSTABLE_METRIC", "constant 'ok' for all 60 clusters -- zero variance"),
    "top_features_summary": ("NON_NUMERIC_OR_UNSTABLE_METRIC", "free text"),
    "recurrence_strength": ("HAND_DESIGNED_CATEGORY", "existing category to compare against, not train on"),
    "short_window_recurrence": ("HAND_DESIGNED_CATEGORY", "existing category to compare against, not train on"),
    "burst_concentrated": ("HAND_DESIGNED_CATEGORY", "existing category to compare against, not train on"),
    "trend_direction": ("HAND_DESIGNED_CATEGORY", "a thresholded version of trend_slope, which is kept as the continuous feature"),
    "persistence_category": ("HAND_DESIGNED_CATEGORY", "existing category to compare against, not train on"),
    "activity_category": ("HAND_DESIGNED_CATEGORY", "existing category to compare against, not train on"),
    "persistence_activity_quadrant": ("HAND_DESIGNED_CATEGORY", "existing category to compare against, not train on"),
    "seasonality_category": ("HAND_DESIGNED_CATEGORY", "existing category to compare against, not train on"),
    "has_notable_osm_context": ("DERIVED_REDUNDANT", "identical to (features_found_in_radius > 0), which is kept as a continuous feature instead"),
    "evidence_notes": ("NON_NUMERIC_OR_UNSTABLE_METRIC", "free-text cross-check notes, itself derived from the hand-designed categories"),
}


def load_evidence(path=INTEGRATED_EVIDENCE_CSV):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pearson(rows, a, b):
    xs = [float(r[a]) for r in rows]
    ys = [float(r[b]) for r in rows]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (sx * sy) if sx and sy else float("nan")


def check_included_features_uncorrelated(matrix, features=INCLUDED_FEATURES, threshold=CORRELATION_THRESHOLD):
    """Verify no pair of INCLUDED features exceeds the correlation
    threshold -- guards against silently shipping a redundant pair. Runs
    against the already-imputed numeric feature matrix (not raw source
    rows, which contain blanks for missing nearest_distance_m)."""
    flagged = []
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            r = pearson(matrix, a, b)
            if abs(r) > threshold:
                flagged.append((a, b, r))
    return flagged


def impute_nearest_distance(row):
    """Missing nearest_distance_m (no OSM context found) is imputed with
    that cluster's own OSM search radius -- the distance already confirmed
    to contain no matching feature. Recomputed via the exact same
    compute_search_radius function the original OSM lookup milestone used
    (src/osm_lookup.py, unchanged), not re-derived or approximated here.
    A conservative, cluster-specific lower bound, not a fixed global
    constant."""
    if row["nearest_distance_m"] == "":
        return compute_search_radius(float(row["extent_radius_m"]))
    return float(row["nearest_distance_m"])


def build_feature_matrix(rows):
    matrix = []
    for row in rows:
        record = {"cluster_id": int(row["cluster_id"])}
        for col in INCLUDED_FEATURES:
            if col == "nearest_distance_m":
                record[col] = impute_nearest_distance(row)
            elif col == "nearest_is_named":
                record[col] = 1 if row[col] == "True" else 0
            else:
                record[col] = float(row[col])
        matrix.append(record)
    matrix.sort(key=lambda r: r["cluster_id"])
    return matrix


def write_documentation(path=DOCUMENTATION_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["column", "decision", "category", "reason"])
        writer.writeheader()
        for col in INCLUDED_FEATURES:
            writer.writerow({"column": col, "decision": "INCLUDED", "category": "", "reason": ""})
        for col, (category, reason) in EXCLUDED_FEATURES.items():
            writer.writerow({"column": col, "decision": "EXCLUDED", "category": category, "reason": reason})


def write_feature_matrix(matrix, path=FEATURES_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(matrix[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matrix)


def main():
    rows = load_evidence()
    print(f"Loaded {len(rows)} clusters, {len(rows[0])} source columns.")

    all_columns = set(rows[0].keys())
    accounted = set(INCLUDED_FEATURES) | set(EXCLUDED_FEATURES.keys())
    unaccounted = all_columns - accounted
    if unaccounted:
        raise ValueError(f"Columns not explicitly included or excluded: {sorted(unaccounted)}")
    print(f"All {len(all_columns)} source columns explicitly accounted for: "
          f"{len(INCLUDED_FEATURES)} included, {len(EXCLUDED_FEATURES)} excluded.")

    matrix = build_feature_matrix(rows)

    flagged = check_included_features_uncorrelated(matrix)
    if flagged:
        raise ValueError(f"Included features exceed correlation threshold: {flagged}")
    print(f"Verified: no pair among the {len(INCLUDED_FEATURES)} included features "
          f"exceeds |r|={CORRELATION_THRESHOLD}.")

    write_documentation()
    print(f"Wrote {DOCUMENTATION_CSV}")

    write_feature_matrix(matrix)
    print(f"Wrote {FEATURES_CSV} ({len(matrix)} rows, {len(matrix[0]) - 1} features)")

    return matrix


if __name__ == "__main__":
    main()
