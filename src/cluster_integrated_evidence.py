"""
Integrated, interpretable cluster evidence layer (additive).

This module does NOT compute anything new from raw detections. It reads
the already-validated 60-cluster longitudinal feature table
(gujarat_cluster_longitudinal_features.csv -- itself an unmodified merge
of DBSCAN cluster geometry, the Recurrence Profile, OSM context, and the
cross-year recurrence result) and reorganizes those existing columns into
seven clearly labeled evidence groups for human interpretation, plus a
small number of transparent, single-rule derived indicators.

EVIDENCE GROUPS (column order in the output CSV follows this grouping)
------------------------------------------------------------------------
A. Temporal persistence   -- unique_years, years_detected, first/last_year,
                              recurs_across_multiple_years, unique_dates
                              (2023), active_span_days (2023),
                              occurrence_rate (2023)
B. Activity / intensity   -- total_detections_5yr, mean_annual_detections,
                              per-year detections/active-days (2019-2023),
                              detection_count (2023), mean/max FRP, frp_ratio
C. Trend behavior         -- trend_slope, trend_direction,
                              std_annual_detections, cv_annual_detections
D. Seasonality            -- per_month_detection_counts, top3_months_share,
                              dominant_month, day/night counts and
                              night_fraction, top_day_share/top3_days_share
                              (2023 within-year burstiness)
E. Spatial characteristics-- centroid, extent_radius_m, bounding box
F. OSM / context          -- nearest feature (group/label/name/distance/
                              geometry), named-feature fields, category
                              counts, features_found_in_radius,
                              osm_query_status. OBSERVED CONTEXT ONLY --
                              never a source-type determination.
G. Recurrence-Profile evidence -- recurrence_strength,
                              short_window_recurrence, burst_concentrated
                              (2023-based, locked thresholds, unchanged)

DERIVED INDICATORS (new, transparent, single-rule -- NOT composite scores)
------------------------------------------------------------------------
Each indicator below is a direct function of one or two existing columns,
with its rule stated in the code and in DECISIONS.md. None combines more
than two inputs, none is weighted, and none is a score:
  - persistence_category: "Persistent" if unique_years == 5 (the maximum
    possible -- detected in every year of the available record), else
    "Intermittent". Not an invented cutoff -- 5 is the natural ceiling.
  - activity_category: "High-activity" if total_detections_5yr is at or
    above the sample median across these 60 clusters, else "Low-activity".
    The median is a data-driven split, computed fresh each run and
    reported, not a fixed invented number.
  - persistence_activity_quadrant: the combination of the two categories
    above (e.g. "Persistent / High-activity"), directly answering
    "persistent+high-activity" vs "persistent but low-activity" etc.
  - seasonality_category: "Strongly seasonal" if top3_months_share is at
    or above the sample 75th percentile across these 60 clusters, else
    "Not strongly seasonal". Percentile computed fresh each run.
  - has_notable_osm_context: True if features_found_in_radius > 0 (i.e.
    any OSM-tagged feature exists within the adaptive search radius --
    context existing, not a claim about what it means).
  - evidence_notes: a semicolon-separated list of specific, individually
    documented cross-checks between independently-computed evidence
    sources (e.g. "2023 recurrence_strength=Limited but persistent across
    all 5 years"). This surfaces agreement/disagreement between sources
    for a human reviewer -- it is not a score and does not resolve the
    disagreement.

This module does NOT: assign a source-type label, infer that an OSM
feature proves what a cluster is, claim causality between land use and
thermal activity, change DBSCAN/Recurrence-Profile/cross-year matching
methodology, or compute any weighted/composite score.
"""

import csv
import statistics
from pathlib import Path

LONGITUDINAL_FEATURES_CSV = Path("data/processed/gujarat_cluster_longitudinal_features.csv")
OUTPUT_CSV = Path("data/processed/gujarat_cluster_integrated_evidence.csv")
OUTPUT_FIGURE_1 = Path("results/figures/cluster_integrated_evidence_overview.png")
OUTPUT_FIGURE_2 = Path("results/figures/cluster_evidence_agreement.png")

GROUP_A = ["unique_years", "years_detected", "first_year", "last_year",
           "recurs_across_multiple_years", "unique_dates", "active_span_days",
           "occurrence_rate"]
GROUP_B = ["total_detections_5yr", "mean_annual_detections",
           "detections_2019", "detections_2020", "detections_2021",
           "detections_2022", "detections_2023",
           # 2024/2025 added additively (see cluster_longitudinal_features.py
           # EXTENDED_YEARS) -- NOT included in total_detections_5yr/
           # mean_annual_detections above, which keep their original
           # 2019-2023 meaning unchanged.
           "detections_2024", "detections_2025",
           "active_days_2019", "active_days_2020", "active_days_2021",
           "active_days_2022", "active_days_2023",
           "active_days_2024", "active_days_2025",
           "detection_count", "mean_frp", "max_frp", "frp_ratio"]
GROUP_C = ["trend_slope", "trend_direction", "std_annual_detections",
           "cv_annual_detections"]
GROUP_D = ["per_month_detection_counts", "top3_months_share", "dominant_month",
           "day_count", "night_count", "night_fraction",
           "top_day_share", "top3_days_share"]
GROUP_E = ["centroid_lat", "centroid_lon", "extent_radius_m",
           "bbox_min_lat", "bbox_max_lat", "bbox_min_lon", "bbox_max_lon"]
GROUP_F = ["nearest_group", "nearest_label", "nearest_name", "nearest_is_named",
           "nearest_distance_m", "nearest_geometry_class",
           "n_industrial", "n_power", "n_waste", "n_agricultural",
           "n_transport", "n_other", "features_found_in_radius",
           "osm_query_status", "top_features_summary"]
GROUP_G = ["recurrence_strength", "short_window_recurrence", "burst_concentrated"]

DERIVED = ["persistence_category", "activity_category",
           "persistence_activity_quadrant", "seasonality_category",
           "has_notable_osm_context", "evidence_notes"]

ALL_GROUPS = {"A_temporal_persistence": GROUP_A, "B_activity_intensity": GROUP_B,
              "C_trend_behavior": GROUP_C, "D_seasonality": GROUP_D,
              "E_spatial_characteristics": GROUP_E, "F_osm_context": GROUP_F,
              "G_recurrence_profile": GROUP_G}


def load_longitudinal_features(path=LONGITUDINAL_FEATURES_CSV):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def persistence_category(row):
    return "Persistent" if int(row["unique_years"]) == 5 else "Intermittent"


def activity_category(row, median_total):
    return "High-activity" if float(row["total_detections_5yr"]) >= median_total else "Low-activity"


def seasonality_category(row, p75_top3_months_share):
    return ("Strongly seasonal" if float(row["top3_months_share"]) >= p75_top3_months_share
            else "Not strongly seasonal")


def has_notable_osm_context(row):
    return int(row["features_found_in_radius"]) > 0


def compute_evidence_notes(row, median_total):
    """Simple, individually-documented cross-checks between independently
    computed evidence sources. Each check is a single boolean rule -- no
    weighting or aggregation. Returns a semicolon-joined string."""
    notes = []
    unique_years = int(row["unique_years"])
    rec_strength = row["recurrence_strength"]
    total_5yr = float(row["total_detections_5yr"])
    top3_months_share = float(row["top3_months_share"])
    short_window = row["short_window_recurrence"] == "True"
    features_found = int(row["features_found_in_radius"])
    nearest_is_named = row["nearest_is_named"] == "True"
    nearest_group = row["nearest_group"]

    if rec_strength in ("Moderate", "Limited") and unique_years == 5:
        notes.append(
            f"2023 recurrence_strength={rec_strength} but persistent across all 5 years "
            f"(2023 volume alone understates multi-year persistence)"
        )
    if rec_strength == "Strong" and unique_years < 5:
        notes.append(
            f"2023 recurrence_strength=Strong but only detected in {unique_years}/5 years "
            f"(2023 was evidently an unusually active year for this zone)"
        )
    if short_window and top3_months_share < 0.5:
        notes.append(
            "short_window_recurrence=True (2023) but 5-year seasonality is not concentrated "
            "(top3_months_share < 0.5) -- the short 2023 window may not reflect a broader seasonal pattern"
        )
    if features_found == 0 and unique_years == 5:
        notes.append(
            "no OSM context found nearby despite persisting across all 5 years "
            "(context evidence incomplete, not evidence of absence)"
        )
    if nearest_is_named and nearest_group == "industrial" and total_5yr < median_total:
        notes.append(
            f"named industrial OSM feature nearby ({row['nearest_name']}) despite "
            f"below-median 5-year activity -- proximity is not evidence this feature "
            f"is the thermal source"
        )
    return "; ".join(notes)


def build_integrated_evidence(rows):
    totals = [float(r["total_detections_5yr"]) for r in rows]
    median_total = statistics.median(totals)

    top3_shares = sorted(float(r["top3_months_share"]) for r in rows)
    p75_top3_months_share = statistics.quantiles(top3_shares, n=4, method="inclusive")[2]

    ordered_columns = (
        ["cluster_id"] + GROUP_A + GROUP_B + GROUP_C + GROUP_D + GROUP_E + GROUP_F + GROUP_G + DERIVED
    )

    integrated = []
    for row in rows:
        out = {col: row[col] for col in ordered_columns if col in row}
        out["cluster_id"] = int(row["cluster_id"])

        pcat = persistence_category(row)
        acat = activity_category(row, median_total)
        out["persistence_category"] = pcat
        out["activity_category"] = acat
        out["persistence_activity_quadrant"] = f"{pcat} / {acat}"
        out["seasonality_category"] = seasonality_category(row, p75_top3_months_share)
        out["has_notable_osm_context"] = has_notable_osm_context(row)
        out["evidence_notes"] = compute_evidence_notes(row, median_total)

        integrated.append(out)

    integrated.sort(key=lambda r: r["cluster_id"])
    return integrated, {"median_total_detections_5yr": median_total,
                         "p75_top3_months_share": p75_top3_months_share}


def write_csv(rows, path=OUTPUT_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_descriptive_report(integrated):
    from collections import Counter

    print("\n" + "=" * 70)
    print("PERSISTENCE x ACTIVITY QUADRANTS")
    print("=" * 70)
    quadrant_counts = Counter(r["persistence_activity_quadrant"] for r in integrated)
    for q, n in sorted(quadrant_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {q}: {n}")

    print("\n  Persistent / High-activity (top 5 by total activity):")
    ph = sorted([r for r in integrated if r["persistence_activity_quadrant"] == "Persistent / High-activity"],
                key=lambda r: -float(r["total_detections_5yr"]))[:5]
    for r in ph:
        print(f"    cluster {r['cluster_id']}: total_5yr={r['total_detections_5yr']}, "
              f"recurrence_strength={r['recurrence_strength']}")

    print("\n  Persistent / Low-activity (persistent but modest volume):")
    pl = [r for r in integrated if r["persistence_activity_quadrant"] == "Persistent / Low-activity"]
    for r in sorted(pl, key=lambda r: float(r["total_detections_5yr"]))[:5]:
        print(f"    cluster {r['cluster_id']}: total_5yr={r['total_detections_5yr']}, "
              f"recurrence_strength={r['recurrence_strength']}")

    print("\n" + "=" * 70)
    print("SEASONALITY")
    print("=" * 70)
    strongly_seasonal = [r for r in integrated if r["seasonality_category"] == "Strongly seasonal"]
    print(f"  {len(strongly_seasonal)} of {len(integrated)} clusters flagged strongly seasonal "
          f"(top3_months_share >= sample p75)")
    for r in sorted(strongly_seasonal, key=lambda r: -float(r["top3_months_share"]))[:8]:
        print(f"    cluster {r['cluster_id']}: top3_months_share={float(r['top3_months_share']):.2f}, "
              f"dominant_month={r['dominant_month']}")

    print("\n" + "=" * 70)
    print("TREND")
    print("=" * 70)
    trend_counts = Counter(r["trend_direction"] for r in integrated)
    print(f"  {dict(trend_counts)}")

    print("\n" + "=" * 70)
    print("VARIABILITY / INTERMITTENCY")
    print("=" * 70)
    meaningful = [r for r in integrated if float(r["mean_annual_detections"]) >= 5]
    top_cv = sorted(meaningful, key=lambda r: -float(r["cv_annual_detections"]))[:5]
    for r in top_cv:
        print(f"    cluster {r['cluster_id']}: cv_annual_detections={float(r['cv_annual_detections']):.2f}")

    print("\n" + "=" * 70)
    print("NOTABLE OSM CONTEXT")
    print("=" * 70)
    osm_group_counts = Counter(r["nearest_group"] for r in integrated if r["has_notable_osm_context"] == "True" or r["has_notable_osm_context"] is True)
    print(f"  Nearest-OSM-group breakdown (context observed, not a source-type claim): {dict(osm_group_counts)}")

    print("\n" + "=" * 70)
    print("EVIDENCE AGREEMENT / DISAGREEMENT")
    print("=" * 70)
    with_notes = [r for r in integrated if r["evidence_notes"]]
    print(f"  {len(with_notes)} of {len(integrated)} clusters have at least one cross-check note.")
    without_notes = [r for r in integrated if not r["evidence_notes"]]
    print(f"  {len(without_notes)} clusters show no flagged disagreement across the checked evidence sources"
          f" (this does not mean 'confirmed', only 'no checked inconsistency found').")
    # The most selective, individually-interesting flag: named industrial context despite low activity.
    named_low_activity = [r for r in integrated if "not evidence this feature" in r["evidence_notes"]]
    print(f"  Named industrial OSM context despite below-median activity: {len(named_low_activity)} clusters "
          f"({[r['cluster_id'] for r in named_low_activity]})")


def plot_overview(integrated, thresholds, output_path=OUTPUT_FIGURE_1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    quadrant_colors = {
        "Persistent / High-activity": "#c53030",
        "Persistent / Low-activity": "#d69e2e",
        "Intermittent / High-activity": "#3182ce",
        "Intermittent / Low-activity": "#a0aec0",
    }
    ax = axes[0, 0]
    for q, color in quadrant_colors.items():
        pts = [r for r in integrated if r["persistence_activity_quadrant"] == q]
        ax.scatter([int(r["unique_years"]) for r in pts],
                   [float(r["total_detections_5yr"]) for r in pts],
                   color=color, label=q, edgecolor="black", s=60, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("unique_years (of 5)")
    ax.set_ylabel("total_detections_5yr (log scale)")
    ax.set_title("Persistence vs. Activity")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[0, 1]
    shares = [float(r["top3_months_share"]) for r in integrated]
    ax.hist(shares, bins=15, color="#2b6cb0", edgecolor="black")
    ax.axvline(thresholds["p75_top3_months_share"], color="#c53030", linestyle="--",
               label=f"sample p75 = {thresholds['p75_top3_months_share']:.2f}")
    ax.set_xlabel("top3_months_share (5-year seasonality concentration)")
    ax.set_ylabel("Number of clusters")
    ax.set_title("Seasonality distribution")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    trend_by_persistence = Counter(
        (r["persistence_category"], r["trend_direction"]) for r in integrated
    )
    directions = ["increasing", "stable", "decreasing"]
    persistent_counts = [trend_by_persistence.get(("Persistent", d), 0) for d in directions]
    intermittent_counts = [trend_by_persistence.get(("Intermittent", d), 0) for d in directions]
    x = range(len(directions))
    ax.bar([i - 0.2 for i in x], persistent_counts, width=0.4, label="Persistent", color="#38a169")
    ax.bar([i + 0.2 for i in x], intermittent_counts, width=0.4, label="Intermittent", color="#a0aec0")
    ax.set_xticks(list(x))
    ax.set_xticklabels(directions)
    ax.set_ylabel("Number of clusters")
    ax.set_title("Trend direction by persistence category")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    osm_counts = Counter(r["nearest_group"] for r in integrated if r["nearest_group"])
    groups = sorted(osm_counts, key=lambda g: -osm_counts[g])
    ax.bar(groups, [osm_counts[g] for g in groups], color="#805ad5", edgecolor="black")
    ax.set_ylabel("Number of clusters")
    ax.set_title("Nearest OSM context group\n(observed context only, not a source-type claim)")
    ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Integrated Cluster Evidence Overview (60 clusters, descriptive only)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_agreement_matrix(integrated, output_path=OUTPUT_FIGURE_2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    tiers = ["Strong", "Moderate", "Limited"]
    year_values = sorted({int(r["unique_years"]) for r in integrated})

    matrix = np.zeros((len(tiers), len(year_values)))
    for r in integrated:
        i = tiers.index(r["recurrence_strength"])
        j = year_values.index(int(r["unique_years"]))
        matrix[i, j] += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="YlOrRd")
    ax.set_xticks(range(len(year_values)))
    ax.set_xticklabels(year_values)
    ax.set_yticks(range(len(tiers)))
    ax.set_yticklabels(tiers)
    ax.set_xlabel("unique_years (5-year cross-year evidence)")
    ax.set_ylabel("recurrence_strength (2023-only Recurrence Profile)")
    ax.set_title("Agreement Between 2023 Recurrence Tier and 5-Year Persistence\n"
                  "(diagonal-ish = independent sources agree; off-diagonal = worth reviewing)")
    for i in range(len(tiers)):
        for j in range(len(year_values)):
            if matrix[i, j] > 0:
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                        color="black" if matrix[i, j] < matrix.max() / 2 else "white")
    fig.colorbar(im, ax=ax, label="Number of clusters")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    rows = load_longitudinal_features()
    integrated, thresholds = build_integrated_evidence(rows)

    assert len(integrated) == 60, f"expected 60 clusters, got {len(integrated)}"
    assert len({r["cluster_id"] for r in integrated}) == 60, "duplicate cluster_id"

    write_csv(integrated)
    print(f"Wrote {len(integrated)} rows, {len(integrated[0])} columns to {OUTPUT_CSV}")
    print(f"Derived-indicator thresholds used this run: {thresholds}")

    print_descriptive_report(integrated)

    plot_overview(integrated, thresholds)
    print(f"\nSaved figure to {OUTPUT_FIGURE_1}")
    plot_agreement_matrix(integrated)
    print(f"Saved figure to {OUTPUT_FIGURE_2}")

    return integrated, thresholds


if __name__ == "__main__":
    main()
