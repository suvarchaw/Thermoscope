"""
Investigate the unexpected nighttime dominance found in the 60 DBSCAN
thermal clusters (median night_fraction 0.984), using only the existing
2023 Gujarat FIRMS data and the existing clustered-detections output.

This script does not assume an explanation in advance. It checks:
  1. acq_time distribution for day vs night detections (satellite overpass
     structure).
  2. Whether detections concentrate around particular overpass times.
  3. Day/night counts across all 60 clusters.
  4. Whether nighttime dominance is broad (present in noise/unclustered
     detections too) or specific to clustered detections.
  5. Whether the three highly-recurring clusters (0, 7, 8) behave like the
     rest or differently.
  6. Any obvious sampling characteristic (e.g. single-satellite, two fixed
     daily overpasses) that could account for the imbalance.

Does not define a persistence threshold, assign source labels, or change
the DBSCAN methodology. Purely descriptive.
"""

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from spatial_recurrence import read_gujarat_detections, RAW_CSV

DETECTIONS_CSV = Path("data/processed/gujarat_clustered_detections.csv")
CLUSTERS_CSV = Path("data/processed/gujarat_thermal_clusters.csv")
OUTPUT_FIGURE = Path("results/figures/nighttime_dominance_investigation.png")

TOP_TIER_CLUSTER_IDS = {0, 7, 8}


def acq_time_to_hours(acq_time_str):
    """Convert HHMM (possibly unpadded, e.g. '745') to decimal hours (UTC)."""
    t = int(acq_time_str)
    hh = t // 100
    mm = t % 100
    return hh + mm / 60.0


def load_all_gujarat():
    return list(read_gujarat_detections(RAW_CSV))


def load_clustered_detections():
    with open(DETECTIONS_CSV, newline="") as f:
        return list(csv.DictReader(f))


def daynight_breakdown(rows, label):
    d = sum(1 for r in rows if r.get("daynight") == "D")
    n = sum(1 for r in rows if r.get("daynight") == "N")
    total = len(rows)
    print(f"  {label}: D={d} ({100*d/total:.1f}%)  N={n} ({100*n/total:.1f}%)  total={total}")
    return d, n, total


def time_concentration(rows, label):
    """Report how tightly acq_time values concentrate, per daynight flag."""
    for flag in ("D", "N"):
        times = [acq_time_to_hours(r["acq_time"]) for r in rows if r.get("daynight") == flag]
        if not times:
            continue
        times_sorted = sorted(times)
        n = len(times_sorted)
        mode_counter = Counter(round(t, 2) for t in times)
        most_common_time, most_common_count = mode_counter.most_common(1)[0]
        print(f"  {label} [{flag}] n={n}: median={statistics.median(times_sorted):.2f}h "
              f"stdev={statistics.pstdev(times_sorted):.3f}h "
              f"range=[{times_sorted[0]:.2f},{times_sorted[-1]:.2f}]h "
              f"most_common_time~{most_common_time}h ({100*most_common_count/n:.1f}% of rows)")


def main():
    all_rows = load_all_gujarat()
    clustered_rows = load_clustered_detections()
    non_noise = [r for r in clustered_rows if int(r["cluster_id"]) != -1]
    noise = [r for r in clustered_rows if int(r["cluster_id"]) == -1]

    print(f"Loaded {len(all_rows)} total Gujarat detections; "
          f"{len(non_noise)} clustered, {len(noise)} noise.\n")

    print("=" * 70)
    print("1-2. ACQUISITION TIME STRUCTURE (day vs night, all Gujarat detections)")
    print("=" * 70)
    time_concentration(all_rows, "ALL")

    print("\n" + "=" * 70)
    print("4. DAY/NIGHT SPLIT: ALL vs. CLUSTERED vs. NOISE")
    print("=" * 70)
    daynight_breakdown(all_rows, "All Gujarat detections (17,596)")
    daynight_breakdown(non_noise, "Clustered detections (60 clusters)")
    daynight_breakdown(noise, "Noise (unclustered) detections")

    print("\n  Acquisition-time structure, clustered vs noise:")
    time_concentration(non_noise, "CLUSTERED")
    time_concentration(noise, "NOISE")

    print("\n" + "=" * 70)
    print("3. DAY/NIGHT COUNTS ACROSS ALL 60 CLUSTERS")
    print("=" * 70)
    with open(CLUSTERS_CSV, newline="") as f:
        clusters = list(csv.DictReader(f))
    night_fractions = []
    for c in clusters:
        day = int(c["day_count"])
        night = int(c["night_count"])
        total = day + night
        night_fractions.append(night / total)
    print(f"  night_fraction across 60 clusters: min={min(night_fractions):.3f} "
          f"median={statistics.median(night_fractions):.3f} max={max(night_fractions):.3f}")
    below_50 = sum(1 for f in night_fractions if f < 0.5)
    above_90 = sum(1 for f in night_fractions if f > 0.9)
    print(f"  clusters with night_fraction < 0.50: {below_50}")
    print(f"  clusters with night_fraction > 0.90: {above_90}")

    print("\n" + "=" * 70)
    print("5. TOP-TIER CLUSTERS (0, 7, 8) vs. THE REST")
    print("=" * 70)
    top_tier = [c for c in clusters if int(c["cluster_id"]) in TOP_TIER_CLUSTER_IDS]
    rest = [c for c in clusters if int(c["cluster_id"]) not in TOP_TIER_CLUSTER_IDS]
    for c in sorted(top_tier, key=lambda c: int(c["cluster_id"])):
        day, night = int(c["day_count"]), int(c["night_count"])
        total = day + night
        print(f"  cluster {c['cluster_id']}: day={day} night={night} "
              f"night_fraction={night/total:.3f} (detection_count={c['detection_count']})")
    rest_fractions = [int(c["night_count"]) / (int(c["day_count"]) + int(c["night_count"])) for c in rest]
    print(f"  Remaining 57 clusters: median night_fraction={statistics.median(rest_fractions):.3f}, "
          f"min={min(rest_fractions):.3f}, max={max(rest_fractions):.3f}")

    print("\n" + "=" * 70)
    print("6. TOP-TIER CLUSTER ACQUISITION-TIME STRUCTURE")
    print("=" * 70)
    non_noise_by_cluster = defaultdict(list)
    for r in non_noise:
        non_noise_by_cluster[int(r["cluster_id"])].append(r)
    for cid in sorted(TOP_TIER_CLUSTER_IDS):
        time_concentration(non_noise_by_cluster[cid], f"cluster {cid}")

    plot_investigation(all_rows, non_noise, noise, night_fractions)
    print(f"\nSaved figure to {OUTPUT_FIGURE}")


def plot_investigation(all_rows, non_noise, noise, night_fractions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel 1: acq_time histogram, all Gujarat detections, split by D/N
    day_times = [acq_time_to_hours(r["acq_time"]) for r in all_rows if r.get("daynight") == "D"]
    night_times = [acq_time_to_hours(r["acq_time"]) for r in all_rows if r.get("daynight") == "N"]
    axes[0, 0].hist(day_times, bins=48, range=(0, 24), color="#d69e2e", alpha=0.7, label="Day-flagged")
    axes[0, 0].hist(night_times, bins=48, range=(0, 24), color="#2b6cb0", alpha=0.7, label="Night-flagged")
    axes[0, 0].set_title("acq_time (UTC) distribution — all Gujarat detections")
    axes[0, 0].set_xlabel("Acquisition time (UTC, hours)")
    axes[0, 0].set_ylabel("Detections")
    axes[0, 0].legend()

    # Panel 2: day/night % comparison across groups
    groups = ["All\n(17,596)", "Clustered\n(5,741)", "Noise\n(11,855)"]
    day_pct = []
    night_pct = []
    for rows in (all_rows, non_noise, noise):
        d = sum(1 for r in rows if r.get("daynight") == "D")
        n = sum(1 for r in rows if r.get("daynight") == "N")
        t = d + n
        day_pct.append(100 * d / t)
        night_pct.append(100 * n / t)
    x = range(len(groups))
    axes[0, 1].bar(x, day_pct, color="#d69e2e", label="Day %")
    axes[0, 1].bar(x, night_pct, bottom=day_pct, color="#2b6cb0", label="Night %")
    axes[0, 1].set_xticks(list(x))
    axes[0, 1].set_xticklabels(groups)
    axes[0, 1].set_title("Day/Night split: all vs. clustered vs. noise")
    axes[0, 1].set_ylabel("% of detections")
    axes[0, 1].legend()

    # Panel 3: histogram of night_fraction across the 60 clusters
    axes[1, 0].hist(night_fractions, bins=20, range=(0, 1), color="#805ad5", edgecolor="black")
    axes[1, 0].set_title("night_fraction distribution across 60 clusters")
    axes[1, 0].set_xlabel("night_fraction")
    axes[1, 0].set_ylabel("Number of clusters")

    # Panel 4: acq_time histogram for clustered vs noise (night only, since that's the dominant flag)
    clustered_night_times = [acq_time_to_hours(r["acq_time"]) for r in non_noise if r.get("daynight") == "N"]
    noise_night_times = [acq_time_to_hours(r["acq_time"]) for r in noise if r.get("daynight") == "N"]
    axes[1, 1].hist(noise_night_times, bins=48, range=(0, 24), color="#a0aec0", alpha=0.7,
                     density=True, label="Noise (night-flagged)")
    axes[1, 1].hist(clustered_night_times, bins=48, range=(0, 24), color="#2b6cb0", alpha=0.7,
                     density=True, label="Clustered (night-flagged)")
    axes[1, 1].set_title("Night-flagged acq_time: clustered vs. noise (normalized)")
    axes[1, 1].set_xlabel("Acquisition time (UTC, hours)")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend()

    fig.suptitle("Nighttime Dominance Investigation — Gujarat FIRMS 2023 (descriptive only)")
    fig.tight_layout()
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
