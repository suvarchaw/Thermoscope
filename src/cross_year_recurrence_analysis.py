"""
Cross-year spatial reconciliation and descriptive recurrence analysis.

THE SPATIAL PROBLEM (read before touching this file)
------------------------------------------------------
The existing 60 clusters were produced by running DBSCAN once on 2023
Gujarat detections (eps=375m, min_samples=8 -- locked, unchanged here).
DBSCAN cluster IDs are arbitrary integers assigned during that single run;
they have no meaning outside it. If DBSCAN were simply re-run independently
on a different year's detections, the resulting "cluster 7" for that year
would NOT correspond to 2023's "cluster 7" -- IDs are not stable across
independent runs, cluster boundaries could shift or split/merge slightly
with a different point cloud, and there is no guarantee that two
independent runs even produce the same NUMBER of clusters. Naively
comparing cluster IDs across independently-run years would answer "did
DBSCAN happen to number a cluster the same way twice" -- not the actual
question, "did the same spatial zone recur."

CHOSEN APPROACH (least invasive)
------------------------------------------------------
The 2023 clustering is treated as the fixed baseline geometry, exactly as
approved (eps=375m, min_samples=8) and NOT recomputed here:
  - Every 2023 detection keeps its already-computed DBSCAN cluster_id from
    data/processed/gujarat_clustered_detections.csv, unchanged.
  - Every historical (non-2023) detection is assigned to the nearest 2023
    cluster centroid IF that distance is within the cluster's own
    extent_radius_m (already computed in gujarat_thermal_clusters.csv --
    the maximum distance from centroid to any 2023 member of that
    cluster). This reuses an existing, already-justified quantity rather
    than inventing a new radius.
  - Historical detections whose nearest 2023 centroid is still farther
    than that cluster's extent_radius_m are NOT forced into a cluster.
    They are retained separately as "unmatched_historical" -- evidence
    that something was detected there in a prior year, but not within the
    footprint of any zone 2023 was dense enough to define.

KNOWN LIMITATION, STATED EXPLICITLY
------------------------------------------------------
A circular buffer around a centroid is only an approximation of a DBSCAN
cluster's true (often irregular/elongated) shape -- e.g. cluster 0's
Hazira-area footprint is known from an earlier milestone to be an
elongated ~1.5-2.5km zone, not a circle. Using extent_radius_m as a
circular buffer radius is a deliberately generous, conservative choice
(it is the maximum observed 2023 extent in any direction), so it will
sometimes include area outside the true 2023 shape rather than exclude
area inside it. This is judged acceptable for a first descriptive pass at
"did activity recur near this zone," not for precise re-clustering. It is
NOT used to redefine, move, or resize any 2023 cluster -- 2023's own
detections keep their real DBSCAN assignment regardless of this rule.

This module produces descriptive cross-year evidence only. It does not
redefine recurrence_strength/short_window_recurrence/burst_concentrated,
does not use FIRMS `type`, does not use OSM context, and does not change
DBSCAN parameters.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

from multi_year_gujarat_processing import (
    discover_historical_raw_files, load_historical_year,
)

CLUSTERS_CSV = Path("data/processed/gujarat_thermal_clusters.csv")
CLUSTERED_DETECTIONS_2023_CSV = Path("data/processed/gujarat_clustered_detections.csv")
OUTPUT_CSV = Path("data/processed/gujarat_cluster_cross_year_recurrence.csv")
OUTPUT_FIGURE = Path("results/figures/cluster_cross_year_recurrence.png")

EARTH_RADIUS_M = 6_371_000.0

BASELINE_YEAR = 2023


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def load_cluster_definitions(path=CLUSTERS_CSV):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["cluster_id"] = int(r["cluster_id"])
        r["centroid_lat"] = float(r["centroid_lat"])
        r["centroid_lon"] = float(r["centroid_lon"])
        r["extent_radius_m"] = float(r["extent_radius_m"])
    return rows


def load_2023_assignments(path=CLUSTERED_DETECTIONS_2023_CSV):
    """Reuses the existing, unchanged 2023 DBSCAN per-detection cluster
    assignments as-is -- does not recompute them."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["cluster_id"] = int(r["cluster_id"])
        r["year"] = int(r["acq_date"][:4])
        r["match_method"] = "dbscan_2023"
    return rows


def match_historical_to_baseline(rows, cluster_defs):
    """Assign each historical (non-2023) detection to the nearest 2023
    cluster centroid if within that cluster's own extent_radius_m;
    otherwise mark as unmatched. Documented rule -- see module docstring."""
    matched = []
    for row in rows:
        lat = float(row["latitude"])
        lon = float(row["longitude"])

        best_cluster = None
        best_dist = None
        for c in cluster_defs:
            d = haversine_m(lat, lon, c["centroid_lat"], c["centroid_lon"])
            if best_dist is None or d < best_dist:
                best_dist = d
                best_cluster = c

        out_row = dict(row)
        if best_cluster is not None and best_dist <= best_cluster["extent_radius_m"]:
            out_row["cluster_id"] = best_cluster["cluster_id"]
            out_row["match_method"] = "nearest_centroid_within_radius"
        else:
            out_row["cluster_id"] = -1
            out_row["match_method"] = "unmatched_historical"
        matched.append(out_row)
    return matched


def build_all_year_assignments():
    """Combine 2023's real DBSCAN assignments with historical detections
    matched via the documented spatial rule. Returns the full list of
    per-detection rows (each with cluster_id, year, match_method) plus a
    per-year row count of what was actually available."""
    cluster_defs = load_cluster_definitions()
    assignments = load_2023_assignments()
    years_available = {BASELINE_YEAR: len(assignments)}

    historical_files = discover_historical_raw_files()
    for year, path in sorted(historical_files.items()):
        rows = load_historical_year(path)
        matched = match_historical_to_baseline(rows, cluster_defs)
        assignments.extend(matched)
        years_available[year] = len(rows)

    return assignments, years_available


def compute_cross_year_metrics(assignments, cluster_defs):
    """Purely descriptive per-cluster cross-year evidence. Does NOT define
    or redefine any recurrence tier."""
    by_cluster = defaultdict(list)
    for row in assignments:
        if row["cluster_id"] == -1:
            continue
        by_cluster[row["cluster_id"]].append(row)

    results = []
    for c in cluster_defs:
        cid = c["cluster_id"]
        rows = by_cluster.get(cid, [])

        years_detected = sorted({row["year"] for row in rows})
        per_year_counts = defaultdict(int)
        per_year_unique_dates = defaultdict(set)
        for row in rows:
            per_year_counts[row["year"]] += 1
            per_year_unique_dates[row["year"]].add(row["acq_date"])

        results.append({
            "cluster_id": cid,
            "years_detected": ";".join(str(y) for y in years_detected),
            "unique_years": len(years_detected),
            "first_year": years_detected[0] if years_detected else "",
            "last_year": years_detected[-1] if years_detected else "",
            "recurs_across_multiple_years": len(years_detected) > 1,
            "per_year_detection_counts": ";".join(
                f"{y}:{per_year_counts[y]}" for y in years_detected
            ),
            "per_year_unique_dates": ";".join(
                f"{y}:{len(per_year_unique_dates[y])}" for y in years_detected
            ),
        })

    results.sort(key=lambda r: r["cluster_id"])
    return results


def write_csv(rows, path=OUTPUT_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_cross_year_recurrence(cross_year_rows, years_available, output_path=OUTPUT_FIGURE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_years = sorted(years_available.keys())

    fig, ax = plt.subplots(figsize=(10, 7))

    if len(all_years) <= 1:
        # Only one year of data exists -- an honest single-year figure,
        # not a fabricated multi-year heatmap.
        unique_years_counts = [r["unique_years"] for r in cross_year_rows]
        ax.bar(["Only 2023 available"], [len(cross_year_rows)], color="#2b6cb0")
        ax.set_ylabel("Number of clusters")
        ax.set_title(
            "Cross-Year Recurrence: only 1 year of data currently available\n"
            "(FIRMS_MAP_KEY not configured -- see DECISIONS.md). "
            "Multi-year comparison pending additional ingested years."
        )
        ax.text(0, len(cross_year_rows) / 2,
                f"{len(cross_year_rows)} clusters,\nall with unique_years=1",
                ha="center", va="center", fontsize=10, color="white")
    else:
        import numpy as np
        matrix = []
        for r in cross_year_rows:
            year_counts = dict(
                item.split(":") for item in r["per_year_detection_counts"].split(";") if item
            )
            matrix.append([int(year_counts.get(str(y), 0)) for y in all_years])
        matrix = np.array(matrix)

        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(all_years)))
        ax.set_xticklabels(all_years)
        ax.set_xlabel("Year")
        ax.set_ylabel("Cluster ID (2023 baseline)")
        ax.set_title("Cross-Year Detection Counts per 2023 Cluster\n"
                      "(historical years matched via documented spatial rule, not re-clustered)")
        fig.colorbar(im, ax=ax, label="Detections")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    cluster_defs = load_cluster_definitions()
    assignments, years_available = build_all_year_assignments()

    n_unmatched_historical = sum(
        1 for r in assignments
        if r.get("match_method") == "unmatched_historical"
    )

    print("Years available for cross-year analysis:")
    for year in sorted(years_available):
        print(f"  {year}: {years_available[year]} detections")
    print(f"\nUnmatched historical detections (outside all cluster extents): "
          f"{n_unmatched_historical}")

    cross_year_rows = compute_cross_year_metrics(assignments, cluster_defs)
    write_csv(cross_year_rows)
    print(f"\nWrote {OUTPUT_CSV}")

    n_recurring = sum(1 for r in cross_year_rows if r["recurs_across_multiple_years"])
    print(f"Clusters recurring across multiple years: {n_recurring} of {len(cross_year_rows)}")

    plot_cross_year_recurrence(cross_year_rows, years_available)
    print(f"Saved figure to {OUTPUT_FIGURE}")

    return cross_year_rows, years_available


if __name__ == "__main__":
    main()
