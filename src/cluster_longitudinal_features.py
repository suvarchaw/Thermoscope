"""
Cluster-level longitudinal feature table (2019-2023, plus additive
2024/2025 per-year detection counts -- see EXTENDED_YEARS below).

Builds ONE new, additive artifact
(data/processed/gujarat_cluster_longitudinal_features.csv) by combining:
  - every existing per-cluster metric already computed and validated in
    prior milestones (spatial/extent, Recurrence Profile, OSM context --
    all read as-is from gujarat_cluster_recurrence_profiles.csv, which
    already inherits the OSM evidence table),
  - the existing cross-year summary fields
    (gujarat_cluster_cross_year_recurrence.csv: years_detected,
    unique_years, first_year, last_year, recurs_across_multiple_years),
  - NEW descriptive longitudinal features computed here for the first
    time: zero-filled per-year detection/active-day counts, total/mean/
    std/coefficient-of-variation of annual detections, a simple linear
    trend slope + descriptive direction, and a monthly/seasonal
    distribution (aggregated across all 5 years) with a burstiness-style
    seasonality indicator.

This module does NOT change DBSCAN parameters, cluster assignments, the
approved Recurrence Profile thresholds, or the cross-year spatial
reconciliation rule -- it only reads their already-computed outputs and
the existing per-detection cross-year assignments (reused unchanged from
src/cross_year_recurrence_analysis.build_all_year_assignments). It does
not train a model, invent a label, or claim causality -- every feature
here is a descriptive summary statistic.
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

from cross_year_recurrence_analysis import (
    build_all_year_assignments, load_cluster_definitions, SUMMARY_YEARS,
)

RECURRENCE_PROFILES_CSV = Path("data/processed/gujarat_cluster_recurrence_profiles.csv")
CROSS_YEAR_CSV = Path("data/processed/gujarat_cluster_cross_year_recurrence.csv")
OUTPUT_CSV = Path("data/processed/gujarat_cluster_longitudinal_features.csv")

YEARS = [2019, 2020, 2021, 2022, 2023]
MONTHS = list(range(1, 13))

# 2024 and 2025 were confirmed fully available via VIIRS_SNPP_SP (read-only
# investigation, see PROGRESS.md/DECISIONS.md) and ingested/matched via the
# existing, unchanged pipeline (firms_ingestion.py, cross_year_recurrence_
# analysis.py). Their per-year detections_{year}/active_days_{year} columns
# are added here ADDITIVELY, via the same existing zero-fill logic used for
# YEARS above -- but YEARS itself is deliberately left unchanged, so every
# existing aggregate statistic (total_detections_5yr, mean/std/cv_annual_
# detections, trend_slope/direction, per_month_detection_counts,
# top3_months_share, dominant_month) keeps its original 2019-2023 meaning
# unchanged. Redefining YEARS to span 7 years would silently change what
# "5yr" statistics mean without being asked to -- not done here.
EXTENDED_YEARS = [2024, 2025]

# A trend is only called "increasing"/"decreasing" if the slope magnitude
# exceeds this fraction of the cluster's own mean annual detection count --
# otherwise it's reported as "stable". This is a simple, explicitly stated
# descriptive rule, not a statistical significance test: with only 5 yearly
# points per cluster, a rigorous trend test is not meaningful, and this
# code does not claim one.
TREND_STABLE_THRESHOLD_FRACTION = 0.10


def parse_year_count_string(s):
    """Parse 'YYYY:count;YYYY:count' into {year: count}. Empty string -> {}."""
    result = {}
    for part in s.split(";"):
        if not part:
            continue
        year_str, count_str = part.split(":")
        result[int(year_str)] = int(count_str)
    return result


def zero_fill(counts_by_key, keys):
    return [counts_by_key.get(k, 0) for k in keys]


def compute_trend(yearly_counts):
    """Simple ordinary-least-squares slope of yearly_counts against year
    index (0..n-1). Returns (slope, direction). direction is a descriptive
    label only -- see TREND_STABLE_THRESHOLD_FRACTION."""
    n = len(yearly_counts)
    xs = list(range(n))
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(yearly_counts)

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, yearly_counts))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = numerator / denominator if denominator else 0.0

    if y_mean == 0:
        direction = "stable"
    elif abs(slope) < TREND_STABLE_THRESHOLD_FRACTION * y_mean:
        direction = "stable"
    elif slope > 0:
        direction = "increasing"
    else:
        direction = "decreasing"

    return slope, direction


def compute_annual_stats(yearly_counts):
    total = sum(yearly_counts)
    mean = statistics.mean(yearly_counts)
    std = statistics.stdev(yearly_counts) if len(yearly_counts) > 1 else 0.0
    cv = (std / mean) if mean else 0.0
    return total, mean, std, cv


def compute_monthly_distribution(detection_rows):
    """detection_rows: list of dicts with 'acq_date' (YYYY-MM-DD).
    Returns {month(1-12): count}, aggregated across all years present."""
    counts = defaultdict(int)
    for row in detection_rows:
        month = int(row["acq_date"][5:7])
        counts[month] += 1
    return dict(counts)


def compute_seasonality(month_counts_zero_filled):
    total = sum(month_counts_zero_filled)
    if total == 0:
        return 0.0, None
    sorted_counts = sorted(month_counts_zero_filled, reverse=True)
    top3_share = sum(sorted_counts[:3]) / total
    dominant_month = month_counts_zero_filled.index(max(month_counts_zero_filled)) + 1
    return top3_share, dominant_month


def load_recurrence_profiles(path=RECURRENCE_PROFILES_CSV):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {int(r["cluster_id"]): r for r in rows}


def load_cross_year_summary(path=CROSS_YEAR_CSV):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {int(r["cluster_id"]): r for r in rows}


def group_detections_by_cluster(assignments):
    """Only rows from SUMMARY_YEARS (2019-2023) are grouped here -- the
    resulting detection_rows feed compute_monthly_distribution, whose
    output (per_month_detection_counts, top3_months_share, dominant_month)
    is documented as "aggregated across all 5 years" and read downstream
    by seasonality_category. Without this filter, 2024/2025 detections
    would silently be absorbed into what is supposed to be a fixed 5-year
    (2019-2023) seasonality summary -- the same class of issue fixed for
    unique_years in cross_year_recurrence_analysis.py, here for the
    monthly/seasonality distribution instead."""
    by_cluster = defaultdict(list)
    for row in assignments:
        if row["cluster_id"] == -1:
            continue
        if row["year"] not in SUMMARY_YEARS:
            continue
        by_cluster[row["cluster_id"]].append(row)
    return by_cluster


def build_feature_row(cluster_id, profile_row, cross_year_row, detection_rows):
    per_year = parse_year_count_string(cross_year_row["per_year_detection_counts"])
    per_year_dates = parse_year_count_string(cross_year_row["per_year_unique_dates"])

    yearly_detections = zero_fill(per_year, YEARS)
    yearly_active_days = zero_fill(per_year_dates, YEARS)

    total, mean, std, cv = compute_annual_stats(yearly_detections)
    slope, direction = compute_trend(yearly_detections)

    month_counts = compute_monthly_distribution(detection_rows)
    month_counts_zero_filled = zero_fill(month_counts, MONTHS)
    top3_months_share, dominant_month = compute_seasonality(month_counts_zero_filled)

    row = dict(profile_row)  # every existing spatial/recurrence/OSM metric, unchanged
    row["cluster_id"] = cluster_id
    for year, count in zip(YEARS, yearly_detections):
        row[f"detections_{year}"] = count
    for year, count in zip(YEARS, yearly_active_days):
        row[f"active_days_{year}"] = count
    row["total_detections_5yr"] = total
    row["mean_annual_detections"] = mean
    row["std_annual_detections"] = std
    row["cv_annual_detections"] = cv
    row["trend_slope"] = slope
    row["trend_direction"] = direction
    row["per_month_detection_counts"] = ";".join(
        f"{m}:{c}" for m, c in zip(MONTHS, month_counts_zero_filled)
    )
    row["top3_months_share"] = top3_months_share
    row["dominant_month"] = dominant_month if dominant_month is not None else ""
    row["years_detected"] = cross_year_row["years_detected"]
    row["unique_years"] = int(cross_year_row["unique_years"])
    row["first_year"] = cross_year_row["first_year"]
    row["last_year"] = cross_year_row["last_year"]
    row["recurs_across_multiple_years"] = cross_year_row["recurs_across_multiple_years"]

    # Additive 2024/2025 per-year counts -- same zero-fill logic as YEARS
    # above, kept separate so it never alters the 2019-2023 "5yr" aggregates.
    yearly_detections_extended = zero_fill(per_year, EXTENDED_YEARS)
    yearly_active_days_extended = zero_fill(per_year_dates, EXTENDED_YEARS)
    for year, count in zip(EXTENDED_YEARS, yearly_detections_extended):
        row[f"detections_{year}"] = count
    for year, count in zip(EXTENDED_YEARS, yearly_active_days_extended):
        row[f"active_days_{year}"] = count

    return row


def build_feature_table():
    profiles = load_recurrence_profiles()
    cross_year = load_cross_year_summary()
    cluster_defs = load_cluster_definitions()
    assignments, _years_available = build_all_year_assignments()
    detections_by_cluster = group_detections_by_cluster(assignments)

    if set(profiles.keys()) != {c["cluster_id"] for c in cluster_defs}:
        raise ValueError("recurrence profiles cluster set does not match cluster definitions")
    if set(cross_year.keys()) != set(profiles.keys()):
        raise ValueError("cross-year summary cluster set does not match recurrence profiles")

    rows = []
    for cluster_id in sorted(profiles.keys()):
        row = build_feature_row(
            cluster_id, profiles[cluster_id], cross_year[cluster_id],
            detections_by_cluster.get(cluster_id, []),
        )
        rows.append(row)
    return rows


def write_csv(rows, path=OUTPUT_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = build_feature_table()
    assert len(rows) == 60, f"expected 60 clusters, got {len(rows)}"
    assert len({r["cluster_id"] for r in rows}) == 60, "duplicate cluster_id in output"

    write_csv(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
    print(f"Columns: {len(rows[0])}")
    return rows


if __name__ == "__main__":
    main()
