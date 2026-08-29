"""
Descriptive analysis of the 60 DBSCAN thermal clusters
(data/processed/gujarat_thermal_clusters.csv, eps=375m, min_samples=8) to
understand how detection-count, recurrence, timing, and FRP quantities are
distributed BEFORE any persistence threshold is defined.

This module computes descriptive metrics only. It does not define a
persistence threshold, does not label any cluster by source type, and does
not use the FIRMS `type` field. All functions operate on already-computed
cluster/detection data and are pure descriptive statistics.
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

CLUSTERS_CSV = Path("data/processed/gujarat_thermal_clusters.csv")
DETECTIONS_CSV = Path("data/processed/gujarat_clustered_detections.csv")


def load_clusters(path=CLUSTERS_CSV):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["cluster_id"] = int(r["cluster_id"])
        r["detection_count"] = int(r["detection_count"])
        r["unique_dates"] = int(r["unique_dates"])
        r["active_span_days"] = int(r["active_span_days"])
        r["mean_frp"] = float(r["mean_frp"])
        r["max_frp"] = float(r["max_frp"])
        r["day_count"] = int(r["day_count"])
        r["night_count"] = int(r["night_count"])
        r["extent_radius_m"] = float(r["extent_radius_m"])
    return rows


def load_clustered_detections(path=DETECTIONS_CSV):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if int(r["cluster_id"]) != -1]


def per_cluster_date_counts(detection_rows):
    """Map cluster_id -> Counter of acq_date -> detection count on that date."""
    result = defaultdict(lambda: defaultdict(int))
    for r in detection_rows:
        cid = int(r["cluster_id"])
        result[cid][r["acq_date"]] += 1
    return result


def derived_metrics(cluster, date_counts):
    """Compute derived, purely descriptive metrics for one cluster row.
    date_counts: dict of acq_date -> count for this cluster."""
    detection_count = cluster["detection_count"]
    unique_dates = cluster["unique_dates"]
    active_span_days = cluster["active_span_days"]

    detections_per_active_date = detection_count / unique_dates
    occurrence_rate = unique_dates / (active_span_days + 1)
    night_fraction = cluster["night_count"] / detection_count
    frp_ratio = (cluster["max_frp"] / cluster["mean_frp"]) if cluster["mean_frp"] else None

    day_counts_sorted = sorted(date_counts.values(), reverse=True)
    top_day_share = day_counts_sorted[0] / detection_count
    top3_days_share = sum(day_counts_sorted[:3]) / detection_count

    return {
        "detections_per_active_date": detections_per_active_date,
        "occurrence_rate": occurrence_rate,
        "night_fraction": night_fraction,
        "frp_ratio": frp_ratio,
        "top_day_share": top_day_share,
        "top3_days_share": top3_days_share,
    }


def summarize(values, label):
    values = sorted(values)
    n = len(values)
    q = statistics.quantiles(values, n=4, method="inclusive") if n >= 2 else [values[0]] * 3
    print(f"\n{label} (n={n})")
    print(f"  min={values[0]:.3g}  p25={q[0]:.3g}  median={q[1]:.3g}  "
          f"p75={q[2]:.3g}  max={values[-1]:.3g}  mean={statistics.mean(values):.3g}")


def pearson_r(xs, ys):
    n = len(xs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math_sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math_sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def math_sqrt(x):
    return x ** 0.5


def find_largest_gaps(values, top_n=5):
    """Return the top_n largest gaps between consecutive sorted values,
    as (gap_size, value_before, value_after)."""
    values = sorted(values)
    gaps = [(values[i + 1] - values[i], values[i], values[i + 1]) for i in range(len(values) - 1)]
    gaps.sort(reverse=True)
    return gaps[:top_n]
