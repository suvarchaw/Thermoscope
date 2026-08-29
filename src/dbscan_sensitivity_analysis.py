"""
DBSCAN parameter sensitivity analysis for the ThermoScope spatial grouping
stage, run over the 17,596 Gujarat FIRMS detections.

Tests a small documented grid of eps (spatial radius) and min_samples
values and reports, for each combination: cluster count, noise count/pct,
largest/median cluster size, cluster-size distribution, number of clusters
with repeated detection dates, and how many distinct clusters the
previously-identified Hazira contiguous zone gets split into.

No parameter combination is assumed correct in advance; this script only
reports numbers. Parameter selection and reasoning are recorded separately
(DECISIONS.md) after reviewing this output.
"""

import statistics
from collections import Counter
from pathlib import Path

from spatial_recurrence import read_gujarat_detections, RAW_CSV
from thermal_clustering import run_dbscan, hazira_cluster_ids

OUTPUT_FIGURE = Path("results/figures/dbscan_parameter_sensitivity.png")

EPS_VALUES_M = [250, 300, 375, 400, 500]
MIN_SAMPLES_VALUES = [3, 5, 8]


def summarize_combination(rows, labels):
    cluster_sizes = Counter(l for l in labels if l != -1)
    n_total = len(labels)
    n_noise = int((labels == -1).sum())
    n_clusters = len(cluster_sizes)
    sizes = sorted(cluster_sizes.values())

    hazira_ids = hazira_cluster_ids(rows, labels)

    # clusters with repeated detection dates (unique_dates > 1)
    dates_by_cluster = {}
    for row, label in zip(rows, labels):
        if label == -1:
            continue
        dates_by_cluster.setdefault(label, set()).add(row["acq_date"])
    n_repeated_date_clusters = sum(1 for d in dates_by_cluster.values() if len(d) > 1)

    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": 100 * n_noise / n_total,
        "largest_cluster": sizes[-1] if sizes else 0,
        "median_cluster": statistics.median(sizes) if sizes else 0,
        "size_p90": sizes[int(0.9 * (len(sizes) - 1))] if sizes else 0,
        "n_repeated_date_clusters": n_repeated_date_clusters,
        "n_hazira_clusters": len(hazira_ids),
        "sizes": sizes,
    }


def main():
    rows = list(read_gujarat_detections(RAW_CSV))
    print(f"Loaded {len(rows)} Gujarat detections.\n")

    header = (
        f"{'eps(m)':>7}{'min_s':>7}{'clusters':>10}{'noise':>8}{'noise%':>8}"
        f"{'largest':>9}{'median':>8}{'p90':>6}{'repdate':>9}{'hazira#':>9}"
    )
    print(header)
    print("-" * len(header))

    grid_results = {}
    for eps_m in EPS_VALUES_M:
        for min_samples in MIN_SAMPLES_VALUES:
            labels = run_dbscan(rows, eps_m, min_samples)
            summary = summarize_combination(rows, labels)
            grid_results[(eps_m, min_samples)] = summary
            print(
                f"{eps_m:>7}{min_samples:>7}{summary['n_clusters']:>10}"
                f"{summary['n_noise']:>8}{summary['noise_pct']:>7.2f}%"
                f"{summary['largest_cluster']:>9}{summary['median_cluster']:>8.1f}"
                f"{summary['size_p90']:>6}{summary['n_repeated_date_clusters']:>9}"
                f"{summary['n_hazira_clusters']:>9}"
            )

    plot_sensitivity(grid_results)
    print(f"\nSaved sensitivity figure to {OUTPUT_FIGURE}")
    return grid_results


def plot_sensitivity(grid_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    colors = {3: "#2b6cb0", 5: "#d69e2e", 8: "#c53030"}

    for min_samples in MIN_SAMPLES_VALUES:
        n_clusters = [grid_results[(eps, min_samples)]["n_clusters"] for eps in EPS_VALUES_M]
        noise_pct = [grid_results[(eps, min_samples)]["noise_pct"] for eps in EPS_VALUES_M]
        largest = [grid_results[(eps, min_samples)]["largest_cluster"] for eps in EPS_VALUES_M]
        hazira = [grid_results[(eps, min_samples)]["n_hazira_clusters"] for eps in EPS_VALUES_M]

        c = colors[min_samples]
        axes[0, 0].plot(EPS_VALUES_M, n_clusters, marker="o", color=c, label=f"min_samples={min_samples}")
        axes[0, 1].plot(EPS_VALUES_M, noise_pct, marker="o", color=c, label=f"min_samples={min_samples}")
        axes[1, 0].plot(EPS_VALUES_M, largest, marker="o", color=c, label=f"min_samples={min_samples}")
        axes[1, 1].plot(EPS_VALUES_M, hazira, marker="o", color=c, label=f"min_samples={min_samples}")

    axes[0, 0].set_title("Number of clusters vs. eps")
    axes[0, 0].set_xlabel("eps (m)")
    axes[0, 0].set_ylabel("Cluster count")
    axes[0, 0].legend()

    axes[0, 1].set_title("Noise percentage vs. eps")
    axes[0, 1].set_xlabel("eps (m)")
    axes[0, 1].set_ylabel("Noise (%)")
    axes[0, 1].legend()

    axes[1, 0].set_title("Largest cluster size vs. eps")
    axes[1, 0].set_xlabel("eps (m)")
    axes[1, 0].set_ylabel("Detections in largest cluster")
    axes[1, 0].set_ylim(0, 2600)
    axes[1, 0].legend()

    axes[1, 1].set_title("Hazira zone: # of clusters it splits into")
    axes[1, 1].set_xlabel("eps (m)")
    axes[1, 1].set_ylabel("Cluster count intersecting Hazira box")
    axes[1, 1].set_yticks(range(0, max(g["n_hazira_clusters"] for g in grid_results.values()) + 2))
    axes[1, 1].legend()

    fig.suptitle("DBSCAN Parameter Sensitivity — Gujarat FIRMS 2023 (17,596 detections)")
    fig.tight_layout()
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
