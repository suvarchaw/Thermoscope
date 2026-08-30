"""
Build a single per-cluster evidence/review table for the 60 final DBSCAN
thermal clusters, combining spatial, temporal, thermal, day/night, and OSM
context evidence side-by-side for HUMAN REVIEW.

This is explicitly NOT a classification dataset. It contains no
industrial/source-type/persistent/risk labels of any kind. It reuses
metric definitions already computed in prior milestones rather than
inventing new ones:
  - cluster geometry/temporal/FRP/day-night stats: gujarat_thermal_clusters.csv
    (src/thermal_clustering.py / src/build_thermal_clusters.py)
  - derived recurrence metrics (occurrence_rate, night_fraction,
    detections_per_active_date, frp_ratio, top_day_share, top3_days_share):
    src/cluster_persistence_analysis.py (unchanged, imported directly)
  - OSM context: gujarat_cluster_osm_context.csv
    (src/osm_lookup.py / src/build_cluster_osm_context.py)

No FIRMS `type` field is used. No persistence threshold is applied. No
proximity is treated as evidence of source type.
"""

import csv
from pathlib import Path

from cluster_persistence_analysis import (
    load_clusters,
    load_clustered_detections,
    per_cluster_date_counts,
    derived_metrics,
)
from spatial_recurrence import RAW_CSV

OSM_CONTEXT_CSV = Path("data/processed/gujarat_cluster_osm_context.csv")
OUTPUT_CSV = Path("data/processed/gujarat_cluster_evidence_review.csv")

# Fields present in both the clusters table and the OSM context table
# (both were derived from the same 60 clusters). Kept from the clusters
# table only, after verifying equality, to avoid duplicate/ambiguous
# columns in the combined evidence table.
OVERLAPPING_FIELDS = ["centroid_lat", "centroid_lon", "detection_count", "unique_dates", "extent_radius_m"]

# OSM context columns carried into the evidence table as-is (verbatim
# names/values from gujarat_cluster_osm_context.csv - observed context only).
OSM_EVIDENCE_FIELDS = [
    "search_radius_m", "osm_query_status", "features_found_in_radius",
    "nearest_tag", "nearest_label", "nearest_group", "nearest_name",
    "nearest_is_named", "nearest_geometry_class", "nearest_distance_m",
    "nearest_lat", "nearest_lon",
    "nearest_named_tag", "nearest_named_label", "nearest_named_name",
    "nearest_named_distance_m",
    "n_industrial", "n_power", "n_waste", "n_agricultural", "n_transport", "n_other",
    "top_features_summary",
]

EVIDENCE_COLUMNS = [
    "cluster_id",
    "centroid_lat", "centroid_lon",
    "bbox_min_lat", "bbox_max_lat", "bbox_min_lon", "bbox_max_lon", "extent_radius_m",
    "detection_count", "unique_dates", "first_date", "last_date", "active_span_days",
    "occurrence_rate", "detections_per_active_date",
    "mean_frp", "max_frp", "frp_ratio",
    "day_count", "night_count", "night_fraction",
    "top_day_share", "top3_days_share",
] + OSM_EVIDENCE_FIELDS


def load_osm_context(path=OSM_CONTEXT_CSV):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    by_id = {}
    for r in rows:
        cid = int(r["cluster_id"])
        if cid in by_id:
            raise ValueError(f"Duplicate cluster_id {cid} in OSM context file {path}")
        by_id[cid] = r
    return by_id


def check_overlap_consistency(cluster, osm_row, tolerance=1e-6):
    """Verify fields present in both source tables agree, since both were
    derived from the same clustering output. Raises on mismatch rather than
    silently picking one value."""
    mismatches = []
    for field in OVERLAPPING_FIELDS:
        cluster_val = cluster[field]
        osm_val = osm_row[field]
        try:
            if abs(float(cluster_val) - float(osm_val)) > tolerance:
                mismatches.append((field, cluster_val, osm_val))
        except (TypeError, ValueError):
            if str(cluster_val) != str(osm_val):
                mismatches.append((field, cluster_val, osm_val))
    if mismatches:
        raise ValueError(
            f"cluster_id {cluster['cluster_id']}: mismatched overlapping fields "
            f"between clusters table and OSM context table: {mismatches}"
        )


def build_evidence_rows(clusters, date_counts_by_cluster, osm_by_id):
    if len(clusters) != 60:
        raise ValueError(f"Expected exactly 60 clusters, found {len(clusters)}")

    cluster_ids = [c["cluster_id"] for c in clusters]
    if len(set(cluster_ids)) != len(cluster_ids):
        raise ValueError("Duplicate cluster_id found in gujarat_thermal_clusters.csv")

    osm_ids = set(osm_by_id.keys())
    cluster_id_set = set(cluster_ids)
    if osm_ids != cluster_id_set:
        missing_osm = cluster_id_set - osm_ids
        extra_osm = osm_ids - cluster_id_set
        raise ValueError(
            f"OSM context cluster_id set does not match clusters table. "
            f"Missing OSM rows for: {sorted(missing_osm)}; "
            f"extra OSM rows for unknown clusters: {sorted(extra_osm)}"
        )

    rows = []
    for cluster in clusters:
        cid = cluster["cluster_id"]
        osm_row = osm_by_id[cid]
        check_overlap_consistency(cluster, osm_row)

        metrics = derived_metrics(cluster, date_counts_by_cluster[cid])

        row = {
            "cluster_id": cid,
            "centroid_lat": cluster["centroid_lat"],
            "centroid_lon": cluster["centroid_lon"],
            "bbox_min_lat": cluster["bbox_min_lat"],
            "bbox_max_lat": cluster["bbox_max_lat"],
            "bbox_min_lon": cluster["bbox_min_lon"],
            "bbox_max_lon": cluster["bbox_max_lon"],
            "extent_radius_m": cluster["extent_radius_m"],
            "detection_count": cluster["detection_count"],
            "unique_dates": cluster["unique_dates"],
            "first_date": cluster["first_date"],
            "last_date": cluster["last_date"],
            "active_span_days": cluster["active_span_days"],
            "occurrence_rate": metrics["occurrence_rate"],
            "detections_per_active_date": metrics["detections_per_active_date"],
            "mean_frp": cluster["mean_frp"],
            "max_frp": cluster["max_frp"],
            "frp_ratio": metrics["frp_ratio"],
            "day_count": cluster["day_count"],
            "night_count": cluster["night_count"],
            "night_fraction": metrics["night_fraction"],
            "top_day_share": metrics["top_day_share"],
            "top3_days_share": metrics["top3_days_share"],
        }
        for field in OSM_EVIDENCE_FIELDS:
            row[field] = osm_row[field]

        rows.append(row)

    rows.sort(key=lambda r: r["cluster_id"])
    return rows


def write_csv(rows, path=OUTPUT_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVIDENCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    clusters = load_clusters()
    detections = load_clustered_detections()
    date_counts_by_cluster = per_cluster_date_counts(detections)
    osm_by_id = load_osm_context()

    rows = build_evidence_rows(clusters, date_counts_by_cluster, osm_by_id)

    output_ids = [r["cluster_id"] for r in rows]
    assert len(output_ids) == 60, f"Expected 60 output rows, got {len(output_ids)}"
    assert len(set(output_ids)) == 60, "Duplicate cluster_id in output evidence table"

    write_csv(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
    print(f"Columns ({len(EVIDENCE_COLUMNS)}): {', '.join(EVIDENCE_COLUMNS)}")

    return rows


if __name__ == "__main__":
    main()
