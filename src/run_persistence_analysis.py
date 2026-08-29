"""
Run the descriptive persistence-related analysis over the 60 DBSCAN
thermal clusters and produce plots where they add real interpretive value.

No persistence threshold is defined here. No cluster is labeled by source
type. The FIRMS `type` field is not used. This is purely descriptive.
"""

from pathlib import Path

from cluster_persistence_analysis import (
    load_clusters,
    load_clustered_detections,
    per_cluster_date_counts,
    derived_metrics,
    summarize,
    pearson_r,
    find_largest_gaps,
)

OUTPUT_FIGURE = Path("results/figures/cluster_persistence_distributions.png")


def main():
    clusters = load_clusters()
    detections = load_clustered_detections()
    date_counts_by_cluster = per_cluster_date_counts(detections)

    for c in clusters:
        c.update(derived_metrics(c, date_counts_by_cluster[c["cluster_id"]]))

    print(f"Loaded {len(clusters)} clusters, {len(detections)} clustered detections.\n")
    print("=" * 70)
    print("1-2. CORE DISTRIBUTIONS ACROSS 60 CLUSTERS")
    print("=" * 70)
    summarize([c["detection_count"] for c in clusters], "detection_count")
    summarize([c["unique_dates"] for c in clusters], "unique_dates")
    summarize([c["active_span_days"] for c in clusters], "active_span_days")
    summarize([c["extent_radius_m"] for c in clusters], "extent_radius_m")

    print("\n" + "=" * 70)
    print("3. DETECTION FREQUENCY OVER TIME")
    print("=" * 70)
    summarize([c["detections_per_active_date"] for c in clusters],
               "detections_per_active_date (detection_count / unique_dates)")
    summarize([c["occurrence_rate"] for c in clusters],
               "occurrence_rate (unique_dates / (active_span_days+1))")

    print("\n" + "=" * 70)
    print("4. DAY/NIGHT RECURRENCE PATTERNS")
    print("=" * 70)
    summarize([c["night_fraction"] for c in clusters], "night_fraction (night_count/detection_count)")
    mostly_day = sum(1 for c in clusters if c["night_fraction"] < 0.3)
    mostly_night = sum(1 for c in clusters if c["night_fraction"] > 0.7)
    mixed = len(clusters) - mostly_day - mostly_night
    print(f"  clusters with night_fraction < 0.3 (mostly day): {mostly_day}")
    print(f"  clusters with night_fraction > 0.7 (mostly night): {mostly_night}")
    print(f"  mixed (0.3-0.7): {mixed}")

    print("\n" + "=" * 70)
    print("5. FRP BEHAVIOUR")
    print("=" * 70)
    summarize([c["mean_frp"] for c in clusters], "mean_frp")
    summarize([c["max_frp"] for c in clusters], "max_frp")
    frp_ratios = [c["frp_ratio"] for c in clusters if c["frp_ratio"] is not None]
    summarize(frp_ratios, "frp_ratio (max_frp / mean_frp)")
    r_frp_unique = pearson_r([c["mean_frp"] for c in clusters], [c["unique_dates"] for c in clusters])
    print(f"  Pearson r(mean_frp, unique_dates) = {r_frp_unique:.3f}")

    print("\n" + "=" * 70)
    print("6. DETECTION COUNT vs UNIQUE DATES")
    print("=" * 70)
    r_count_dates = pearson_r([c["detection_count"] for c in clusters], [c["unique_dates"] for c in clusters])
    print(f"  Pearson r(detection_count, unique_dates) = {r_count_dates:.3f}")
    r_dates_span = pearson_r([c["unique_dates"] for c in clusters], [c["active_span_days"] for c in clusters])
    print(f"  Pearson r(unique_dates, active_span_days) = {r_dates_span:.3f}")

    print("\n" + "=" * 70)
    print("7. NATURAL GROUPS / GAPS IN DISTRIBUTIONS (descriptive only)")
    print("=" * 70)
    print("  Largest gaps in sorted unique_dates values:")
    for gap, before, after in find_largest_gaps([c["unique_dates"] for c in clusters]):
        print(f"    gap of {gap} between {before} and {after} unique dates")
    print("  Largest gaps in sorted active_span_days values:")
    for gap, before, after in find_largest_gaps([c["active_span_days"] for c in clusters]):
        print(f"    gap of {gap} between {before} and {after} days")

    print("\n" + "=" * 70)
    print("8. CLUSTERS DOMINATED BY A SMALL NUMBER OF ACTIVE DAYS")
    print("=" * 70)
    summarize([c["top_day_share"] for c in clusters], "top_day_share (busiest single day / total detections)")
    summarize([c["top3_days_share"] for c in clusters], "top3_days_share (busiest 3 days / total detections)")
    dominated = [c for c in clusters if c["top_day_share"] > 0.5]
    print(f"\n  Clusters where >50% of detections occurred on a single date: {len(dominated)} of {len(clusters)}")
    for c in sorted(dominated, key=lambda c: -c["top_day_share"]):
        print(f"    cluster {c['cluster_id']}: top_day_share={c['top_day_share']:.2f}, "
              f"unique_dates={c['unique_dates']}, detection_count={c['detection_count']}, "
              f"active_span_days={c['active_span_days']}")

    plot_distributions(clusters)
    print(f"\nSaved figure to {OUTPUT_FIGURE}")

    return clusters


def plot_distributions(clusters):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unique_dates = [c["unique_dates"] for c in clusters]
    active_span = [c["active_span_days"] for c in clusters]
    detection_count = [c["detection_count"] for c in clusters]
    occurrence_rate = [c["occurrence_rate"] for c in clusters]
    top_day_share = [c["top_day_share"] for c in clusters]
    mean_frp = [c["mean_frp"] for c in clusters]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes[0, 0].hist(unique_dates, bins=15, color="#2b6cb0", edgecolor="black")
    axes[0, 0].set_title("Unique detection dates per cluster")
    axes[0, 0].set_xlabel("Unique dates")
    axes[0, 0].set_ylabel("Number of clusters")

    axes[0, 1].hist(active_span, bins=15, color="#2f855a", edgecolor="black")
    axes[0, 1].set_title("Active span (days) per cluster")
    axes[0, 1].set_xlabel("Active span (days)")
    axes[0, 1].set_ylabel("Number of clusters")

    axes[0, 2].scatter(unique_dates, detection_count, color="#805ad5", edgecolor="black")
    axes[0, 2].set_title("Detection count vs. unique dates")
    axes[0, 2].set_xlabel("Unique dates")
    axes[0, 2].set_ylabel("Detection count")
    axes[0, 2].set_yscale("log")

    axes[1, 0].scatter(occurrence_rate, top_day_share, color="#d69e2e", edgecolor="black")
    axes[1, 0].set_title("Occurrence rate vs. single-day dominance")
    axes[1, 0].set_xlabel("Occurrence rate (unique_dates / (span+1))")
    axes[1, 0].set_ylabel("Top-day share of detections")

    axes[1, 1].scatter(unique_dates, mean_frp, color="#c53030", edgecolor="black")
    axes[1, 1].set_title("Mean FRP vs. unique dates")
    axes[1, 1].set_xlabel("Unique dates")
    axes[1, 1].set_ylabel("Mean FRP (MW)")

    sizes_sorted = sorted(unique_dates)
    axes[1, 2].plot(range(1, len(sizes_sorted) + 1), sizes_sorted, marker="o", color="#1a202c")
    axes[1, 2].set_title("Sorted unique-dates values (60 clusters)")
    axes[1, 2].set_xlabel("Cluster rank")
    axes[1, 2].set_ylabel("Unique dates")

    fig.suptitle("Cluster-Level Persistence-Related Distributions (60 DBSCAN clusters, descriptive only)")
    fig.tight_layout()
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
