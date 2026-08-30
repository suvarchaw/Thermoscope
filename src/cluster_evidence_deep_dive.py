"""
Detailed cluster-level descriptive analysis of the 60 DBSCAN clusters,
built on top of data/processed/gujarat_cluster_evidence_review.csv.

Purpose: surface natural groupings, gaps, statistical outliers,
internally-contradictory cases, and clusters whose classification would
flip depending on which of several *illustrative* persistence definitions
is applied — to give the next design discussion concrete evidence, not to
make the decision itself.

This module does NOT:
  - choose or adopt a persistence threshold (the "illustrative definitions"
    below are explicitly probes for sensitivity analysis, not proposals),
  - assign a source-type or persistence label,
  - compute a risk score,
  - train a model,
  - change the DBSCAN spatial clustering methodology.

Reuses existing definitions rather than inventing new ones:
  - occurrence_rate, night_fraction, top_day_share, etc. are read directly
    from gujarat_cluster_evidence_review.csv (already computed in the prior
    milestone via src/cluster_persistence_analysis.py).
  - the day/night bucket cutoffs (<0.3 / 0.3-0.7 / >0.7) reuse the exact
    thresholds already used for descriptive grouping in
    src/nighttime_dominance_investigation.py.
  - the OSM context bucketing reuses osm_context_bucket() from
    src/analyze_cluster_evidence.py unchanged.
"""

import csv
import statistics
from pathlib import Path

from analyze_cluster_evidence import osm_context_bucket
from cluster_persistence_analysis import find_largest_gaps

EVIDENCE_CSV = Path("data/processed/gujarat_cluster_evidence_review.csv")
OUTPUT_CSV = Path("data/processed/gujarat_cluster_descriptive_flags.csv")
OUTPUT_FIGURE = Path("results/figures/cluster_definition_sensitivity.png")

NUMERIC_FIELDS = [
    "unique_dates", "occurrence_rate", "detection_count", "active_span_days",
    "mean_frp", "max_frp", "frp_ratio", "night_fraction", "top_day_share",
    "top3_days_share", "detections_per_active_date",
]

# Metrics checked for IQR-based statistical outliers in this analysis.
OUTLIER_METRICS = [
    "unique_dates", "occurrence_rate", "detection_count", "active_span_days",
    "mean_frp", "max_frp", "frp_ratio", "top_day_share",
]

# ---------------------------------------------------------------------------
# Illustrative persistence definitions, for SENSITIVITY ANALYSIS ONLY.
# None of these is adopted or recommended. They exist only to reveal which
# clusters' classification would flip depending on which reasonable-looking
# rule is used -- that instability is itself the finding, not the rules.
# ---------------------------------------------------------------------------
ILLUSTRATIVE_DEFINITIONS = [
    ("unique_dates>=20", lambda r: r["unique_dates"] >= 20),
    ("occurrence_rate>=0.10", lambda r: r["occurrence_rate"] >= 0.10),
    ("unique_dates>=20_and_occurrence>=0.10", lambda r: r["unique_dates"] >= 20 and r["occurrence_rate"] >= 0.10),
    ("active_span>=300", lambda r: r["active_span_days"] >= 300),
    ("gap_based_top_tier", lambda r: r["unique_dates"] >= 143),  # matches the natural gap found previously
]


def load_evidence(path=EVIDENCE_CSV):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["cluster_id"] = int(r["cluster_id"])
        for field in NUMERIC_FIELDS:
            r[field] = float(r[field])
        r["features_found_in_radius"] = int(r["features_found_in_radius"])
        r["nearest_is_named"] = r["nearest_is_named"] == "True"
    return rows


def iqr_bounds(values):
    values = sorted(values)
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def compute_outlier_flags(rows):
    """Return {cluster_id: {metric: bool}} using Tukey's IQR rule per metric."""
    bounds = {m: iqr_bounds([r[m] for r in rows]) for m in OUTLIER_METRICS}
    flags = {}
    for r in rows:
        cid = r["cluster_id"]
        flags[cid] = {}
        for m in OUTLIER_METRICS:
            lo, hi = bounds[m]
            flags[cid][m] = r[m] < lo or r[m] > hi
    return flags, bounds


def day_night_pattern(night_fraction):
    """Reuses the exact <0.3 / 0.3-0.7 / >0.7 cutoffs already used in
    src/nighttime_dominance_investigation.py for descriptive grouping."""
    if night_fraction > 0.7:
        return "mostly_night"
    if night_fraction < 0.3:
        return "mostly_day"
    return "mixed"


def compute_definition_satisfaction(rows):
    result = {}
    for r in rows:
        cid = r["cluster_id"]
        satisfied = {name: bool(pred(r)) for name, pred in ILLUSTRATIVE_DEFINITIONS}
        result[cid] = satisfied
    return result


def compute_contradiction_notes(row, outlier_flags_for_row, def_satisfaction_for_row, medians):
    notes = []

    if day_night_pattern(row["night_fraction"]) != "mostly_night":
        notes.append(f"day_night_pattern={day_night_pattern(row['night_fraction'])}")

    if row["top_day_share"] > 0.5:
        notes.append("single_day_dominated(>50%)")
    elif row["top3_days_share"] > 0.5:
        notes.append("top3_days_dominated(>50%,but_no_single_day>50%)")

    if outlier_flags_for_row.get("frp_ratio"):
        notes.append("extreme_frp_ratio_outlier")

    if row["active_span_days"] >= medians["active_span_days_p75"] and row["occurrence_rate"] <= medians["occurrence_rate_p25"]:
        notes.append("long_span_but_low_occurrence")

    if row["features_found_in_radius"] == 0 and row["unique_dates"] > medians["unique_dates_median"]:
        notes.append("above_median_recurrence_but_no_osm_context")

    if row["nearest_is_named"] and row["nearest_group"] == "industrial" and row["unique_dates"] < medians["unique_dates_median"]:
        notes.append("named_industrial_nearby_but_below_median_recurrence")

    n_outliers = sum(1 for v in outlier_flags_for_row.values() if v)
    if n_outliers >= 3:
        notes.append(f"compound_outlier(n_metrics={n_outliers})")

    n_defs = sum(1 for v in def_satisfaction_for_row.values() if v)
    if 1 <= n_defs <= len(ILLUSTRATIVE_DEFINITIONS) - 1:
        notes.append(f"definition_sensitive({n_defs}/{len(ILLUSTRATIVE_DEFINITIONS)}_illustrative_defs)")

    return "; ".join(notes)


def build_flags_table(rows):
    outlier_flags, outlier_bounds = compute_outlier_flags(rows)
    def_satisfaction = compute_definition_satisfaction(rows)

    medians = {
        "unique_dates_median": statistics.median(r["unique_dates"] for r in rows),
        "occurrence_rate_p25": statistics.quantiles([r["occurrence_rate"] for r in rows], n=4, method="inclusive")[0],
        "active_span_days_p75": statistics.quantiles([r["active_span_days"] for r in rows], n=4, method="inclusive")[2],
    }

    table = []
    for r in rows:
        cid = r["cluster_id"]
        row_out = {
            "cluster_id": cid,
            "unique_dates": r["unique_dates"],
            "occurrence_rate": r["occurrence_rate"],
            "detection_count": r["detection_count"],
            "active_span_days": r["active_span_days"],
            "mean_frp": r["mean_frp"],
            "max_frp": r["max_frp"],
            "frp_ratio": r["frp_ratio"],
            "night_fraction": r["night_fraction"],
            "top_day_share": r["top_day_share"],
            "top3_days_share": r["top3_days_share"],
            "day_night_pattern": day_night_pattern(r["night_fraction"]),
            "osm_context_bucket": osm_context_bucket({
                "features_found_in_radius": str(r["features_found_in_radius"]),
                "nearest_group": r["nearest_group"],
                "nearest_is_named": str(r["nearest_is_named"]),
            }),
            "nearest_group": r["nearest_group"],
            "nearest_is_named": r["nearest_is_named"],
            "features_found_in_radius": r["features_found_in_radius"],
        }
        for m in OUTLIER_METRICS:
            row_out[f"outlier_{m}"] = outlier_flags[cid][m]
        row_out["n_metrics_outlier"] = sum(1 for m in OUTLIER_METRICS if outlier_flags[cid][m])

        for name, _ in ILLUSTRATIVE_DEFINITIONS:
            row_out[f"meets_illustrative__{name}"] = def_satisfaction[cid][name]
        row_out["n_illustrative_defs_satisfied"] = sum(1 for v in def_satisfaction[cid].values() if v)

        row_out["contradiction_notes"] = compute_contradiction_notes(
            r, outlier_flags[cid], def_satisfaction[cid], medians
        )
        table.append(row_out)

    table.sort(key=lambda x: x["cluster_id"])
    return table, outlier_bounds


def write_csv(table, path=OUTPUT_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(table[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)


def print_report(rows, table, outlier_bounds):
    print(f"Loaded {len(rows)} clusters.\n")

    print("=" * 70)
    print("NATURAL GAPS PER METRIC (top 3 largest consecutive gaps)")
    print("=" * 70)
    for m in ["unique_dates", "occurrence_rate", "detection_count", "active_span_days", "mean_frp", "max_frp"]:
        print(f"\n  {m}:")
        for gap, before, after in find_largest_gaps([r[m] for r in rows], top_n=3):
            print(f"    gap={gap:.3g} between {before:.3g} and {after:.3g}")

    print("\n" + "=" * 70)
    print("IQR OUTLIER BOUNDS AND COUNTS")
    print("=" * 70)
    for m in OUTLIER_METRICS:
        lo, hi = outlier_bounds[m]
        n_out = sum(1 for row in table if row[f"outlier_{m}"])
        print(f"  {m}: bounds=({lo:.3g}, {hi:.3g}), outliers={n_out}")

    print("\n" + "=" * 70)
    print("COMPOUND OUTLIERS (outlier on >=3 of 8 checked metrics)")
    print("=" * 70)
    compound = sorted([row for row in table if row["n_metrics_outlier"] >= 3],
                       key=lambda r: -r["n_metrics_outlier"])
    for row in compound:
        print(f"  cluster {row['cluster_id']}: n_metrics_outlier={row['n_metrics_outlier']}, "
              f"unique_dates={row['unique_dates']:.0f}, occurrence_rate={row['occurrence_rate']:.3f}, "
              f"detection_count={row['detection_count']:.0f}")

    print("\n" + "=" * 70)
    print("DAY/NIGHT PATTERN BREAKDOWN")
    print("=" * 70)
    from collections import Counter
    pattern_counts = Counter(row["day_night_pattern"] for row in table)
    for k, v in pattern_counts.items():
        print(f"  {k}: {v}")
    print("  Non-mostly-night clusters:")
    for row in table:
        if row["day_night_pattern"] != "mostly_night":
            print(f"    cluster {row['cluster_id']}: {row['day_night_pattern']}, "
                  f"night_fraction={row['night_fraction']:.2f}, unique_dates={row['unique_dates']:.0f}")

    print("\n" + "=" * 70)
    print("SINGLE-DAY / TOP-3-DAY DOMINATED CLUSTERS")
    print("=" * 70)
    for row in table:
        if row["top_day_share"] > 0.5 or row["top3_days_share"] > 0.5:
            print(f"  cluster {row['cluster_id']}: top_day_share={row['top_day_share']:.2f}, "
                  f"top3_days_share={row['top3_days_share']:.2f}, unique_dates={row['unique_dates']:.0f}")

    print("\n" + "=" * 70)
    print("CONTRADICTORY / NOTEWORTHY CASES (from contradiction_notes)")
    print("=" * 70)
    flagged = [row for row in table if row["contradiction_notes"]]
    print(f"  {len(flagged)} of {len(table)} clusters have at least one noted pattern.\n")
    for row in flagged:
        print(f"  cluster {row['cluster_id']}: {row['contradiction_notes']}")

    print("\n" + "=" * 70)
    print("DEFINITION SENSITIVITY (illustrative definitions only, none adopted)")
    print("=" * 70)
    for name, _ in ILLUSTRATIVE_DEFINITIONS:
        n = sum(1 for row in table if row[f"meets_illustrative__{name}"])
        print(f"  {name}: {n} of 60 clusters satisfy this")
    from collections import Counter
    sat_dist = Counter(row["n_illustrative_defs_satisfied"] for row in table)
    print("\n  Distribution of how many of the 5 illustrative definitions each cluster satisfies:")
    for k in sorted(sat_dist):
        print(f"    {k}/5: {sat_dist[k]} clusters")
    boundary = [row for row in table if 1 <= row["n_illustrative_defs_satisfied"] <= 4]
    print(f"\n  {len(boundary)} of 60 clusters are 'definition-sensitive' "
          f"(satisfy some but not all illustrative definitions):")
    for row in sorted(boundary, key=lambda r: r["n_illustrative_defs_satisfied"]):
        print(f"    cluster {row['cluster_id']}: {row['n_illustrative_defs_satisfied']}/5, "
              f"unique_dates={row['unique_dates']:.0f}, occurrence_rate={row['occurrence_rate']:.3f}, "
              f"active_span_days={row['active_span_days']:.0f}")


def plot_sensitivity(table, output_path=OUTPUT_FIGURE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unique_dates = [row["unique_dates"] for row in table]
    occurrence_rate = [row["occurrence_rate"] for row in table]
    n_defs = [row["n_illustrative_defs_satisfied"] for row in table]

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(unique_dates, occurrence_rate, c=n_defs, cmap="viridis",
                          s=60, edgecolor="black", vmin=0, vmax=len(ILLUSTRATIVE_DEFINITIONS))
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("# of 5 illustrative definitions satisfied (none adopted)")
    ax.set_xlabel("unique_dates")
    ax.set_ylabel("occurrence_rate")
    ax.set_title("Definition Sensitivity Across 60 Clusters\n(illustrative probes only, not a proposed rule)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    rows = load_evidence()
    table, outlier_bounds = build_flags_table(rows)
    print_report(rows, table, outlier_bounds)
    write_csv(table)
    print(f"\nWrote {OUTPUT_CSV}")
    plot_sensitivity(table)
    print(f"Saved figure to {OUTPUT_FIGURE}")
    return table


if __name__ == "__main__":
    main()
