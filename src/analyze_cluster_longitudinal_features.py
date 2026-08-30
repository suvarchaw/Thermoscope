"""
Descriptive analysis of the 60-cluster longitudinal feature table
(2019-2023). Answers, with concrete numbers, the questions posed for this
milestone. Purely descriptive: no model is trained, no label is invented,
no risk score is computed, and no causal claim is made anywhere below.
"""

import csv
import statistics
from pathlib import Path

FEATURES_CSV = Path("data/processed/gujarat_cluster_longitudinal_features.csv")
OUTPUT_FIGURE_1 = Path("results/figures/cluster_longitudinal_overview.png")
OUTPUT_FIGURE_2 = Path("results/figures/cluster_seasonal_patterns.png")


def load_rows(path=FEATURES_CSV):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["cluster_id"] = int(r["cluster_id"])
        for f_ in ["total_detections_5yr", "unique_years", "dominant_month"]:
            r[f_] = int(r[f_]) if r[f_] not in ("", None) else None
        for f_ in ["mean_annual_detections", "std_annual_detections",
                   "cv_annual_detections", "trend_slope", "top3_months_share"]:
            r[f_] = float(r[f_])
        for year in [2019, 2020, 2021, 2022, 2023]:
            r[f"detections_{year}"] = int(r[f"detections_{year}"])
    return rows


def main():
    rows = load_rows()
    print(f"Loaded {len(rows)} clusters.\n")

    print("=" * 70)
    print("1. PERSISTENT ACROSS ALL FIVE YEARS (unique_years == 5)")
    print("=" * 70)
    persistent_5yr = [r for r in rows if r["unique_years"] == 5]
    print(f"  {len(persistent_5yr)} of {len(rows)} clusters detected in all 5 years.")
    dist = {}
    for r in rows:
        dist[r["unique_years"]] = dist.get(r["unique_years"], 0) + 1
    print("  Full distribution of unique_years:", dict(sorted(dist.items())))

    print("\n" + "=" * 70)
    print("2. TREND DIRECTION (descriptive; see TREND_STABLE_THRESHOLD_FRACTION)")
    print("=" * 70)
    increasing = [r for r in rows if r["trend_direction"] == "increasing"]
    decreasing = [r for r in rows if r["trend_direction"] == "decreasing"]
    stable = [r for r in rows if r["trend_direction"] == "stable"]
    print(f"  increasing: {len(increasing)}, decreasing: {len(decreasing)}, stable: {len(stable)}")
    print("  Top 5 increasing (by slope):")
    for r in sorted(increasing, key=lambda r: -r["trend_slope"])[:5]:
        print(f"    cluster {r['cluster_id']}: slope={r['trend_slope']:.1f}, "
              f"detections 2019->2023 = {r['detections_2019']}->{r['detections_2023']}")
    print("  Top 5 decreasing (by slope):")
    for r in sorted(decreasing, key=lambda r: r["trend_slope"])[:5]:
        print(f"    cluster {r['cluster_id']}: slope={r['trend_slope']:.1f}, "
              f"detections 2019->2023 = {r['detections_2019']}->{r['detections_2023']}")

    print("\n" + "=" * 70)
    print("3. HIGHEST TOTAL ACTIVITY (total_detections_5yr)")
    print("=" * 70)
    top_total = sorted(rows, key=lambda r: -r["total_detections_5yr"])[:10]
    for r in top_total:
        print(f"  cluster {r['cluster_id']}: total_5yr={r['total_detections_5yr']}, "
              f"recurrence_strength={r['recurrence_strength']}")

    print("\n" + "=" * 70)
    print("4. GREATEST INTERANNUAL VARIABILITY (cv_annual_detections)")
    print("=" * 70)
    # Restrict to clusters with a meaningful mean to avoid CV noise on near-empty clusters
    meaningful = [r for r in rows if r["mean_annual_detections"] >= 5]
    top_cv = sorted(meaningful, key=lambda r: -r["cv_annual_detections"])[:10]
    for r in top_cv:
        print(f"  cluster {r['cluster_id']}: cv={r['cv_annual_detections']:.2f}, "
              f"mean_annual={r['mean_annual_detections']:.1f}, "
              f"years={r['detections_2019']}/{r['detections_2020']}/"
              f"{r['detections_2021']}/{r['detections_2022']}/{r['detections_2023']}")

    print("\n" + "=" * 70)
    print("5. STRONGLY SEASONAL (top3_months_share)")
    print("=" * 70)
    seasonal = [r for r in rows if r["total_detections_5yr"] >= 20]  # exclude near-empty clusters
    top_seasonal = sorted(seasonal, key=lambda r: -r["top3_months_share"])[:10]
    for r in top_seasonal:
        print(f"  cluster {r['cluster_id']}: top3_months_share={r['top3_months_share']:.2f}, "
              f"dominant_month={r['dominant_month']}, total_5yr={r['total_detections_5yr']}")

    print("\n" + "=" * 70)
    print("6. PERSISTENT vs INTERMITTENT — cross-tab with recurrence_strength")
    print("=" * 70)
    for tier in ("Strong", "Moderate", "Limited"):
        tier_rows = [r for r in rows if r["recurrence_strength"] == tier]
        n5 = sum(1 for r in tier_rows if r["unique_years"] == 5)
        n1 = sum(1 for r in tier_rows if r["unique_years"] == 1)
        print(f"  {tier} (n={len(tier_rows)}): unique_years==5 -> {n5}, "
              f"unique_years==1 -> {n1}")

    plot_overview(rows)
    print(f"\nSaved figure to {OUTPUT_FIGURE_1}")
    plot_seasonal(rows)
    print(f"Saved figure to {OUTPUT_FIGURE_2}")

    return rows


def plot_overview(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) unique_years histogram
    unique_years_vals = [r["unique_years"] for r in rows]
    axes[0, 0].hist(unique_years_vals, bins=[1, 2, 3, 4, 5, 6], align="left",
                     rwidth=0.8, color="#2b6cb0", edgecolor="black")
    axes[0, 0].set_xlabel("Number of years detected (of 5)")
    axes[0, 0].set_ylabel("Number of clusters")
    axes[0, 0].set_title("Persistence across years (unique_years)")

    # (b) trend direction counts
    from collections import Counter
    trend_counts = Counter(r["trend_direction"] for r in rows)
    order = ["increasing", "stable", "decreasing"]
    axes[0, 1].bar(order, [trend_counts.get(t, 0) for t in order],
                    color=["#38a169", "#a0aec0", "#c53030"], edgecolor="black")
    axes[0, 1].set_ylabel("Number of clusters")
    axes[0, 1].set_title("Trend direction (descriptive, threshold-based)")

    # (c) total activity vs interannual CV (log x)
    totals = [r["total_detections_5yr"] for r in rows]
    cvs = [r["cv_annual_detections"] for r in rows]
    axes[1, 0].scatter(totals, cvs, color="#805ad5", edgecolor="black", alpha=0.8)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("Total detections (2019-2023, log scale)")
    axes[1, 0].set_ylabel("Coefficient of variation (annual detections)")
    axes[1, 0].set_title("Scale vs. interannual variability")

    # (d) sorted total_detections_5yr (rank plot)
    sorted_totals = sorted(totals)
    axes[1, 1].plot(range(1, len(sorted_totals) + 1), sorted_totals, marker="o",
                     color="#1a202c", markersize=3)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Cluster rank")
    axes[1, 1].set_ylabel("Total detections 2019-2023 (log scale)")
    axes[1, 1].set_title("Sorted total activity across 60 clusters")

    fig.suptitle("Longitudinal Overview — 60 Clusters, 2019-2023 (descriptive only)")
    fig.tight_layout()
    OUTPUT_FIGURE_1.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE_1, dpi=150)
    plt.close(fig)


def plot_seasonal(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    top15 = sorted(rows, key=lambda r: -r["total_detections_5yr"])[:15]
    matrix = []
    labels = []
    for r in top15:
        month_counts = dict(
            item.split(":") for item in r["per_month_detection_counts"].split(";") if item
        )
        total = sum(int(v) for v in month_counts.values())
        row_frac = [int(month_counts.get(str(m), 0)) / total if total else 0 for m in range(1, 13)]
        matrix.append(row_frac)
        labels.append(f"cluster {r['cluster_id']}")
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Monthly Detection Share — Top 15 Clusters by Total Activity\n"
                  "(aggregated across 2019-2023, descriptive only)")
    fig.colorbar(im, ax=ax, label="Fraction of cluster's detections in that month")
    fig.tight_layout()
    OUTPUT_FIGURE_2.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE_2, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
