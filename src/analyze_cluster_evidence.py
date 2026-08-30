"""
Descriptive inspection of the 60-cluster evidence/review table
(data/processed/gujarat_cluster_evidence_review.csv).

Purely descriptive: distributions, OSM-context breakdown, and the
relationship between recurrence metrics and observed OSM context. Produces
one cross-cutting figure that was not covered by prior milestones (how
recurrence metrics vary across observed "nearest OSM tag category" groups).

Does NOT define a persistence threshold, does NOT assign a source-type
label, and does NOT compute a risk score. The "OSM context group" used
below for grouping is a transient, plotting-only bucketing of the nearest
observed OSM tag category — it is not persisted to any dataset column and
is not a classification of the cluster itself.
"""

import csv
import statistics
from pathlib import Path

EVIDENCE_CSV = Path("data/processed/gujarat_cluster_evidence_review.csv")
OUTPUT_FIGURE = Path("results/figures/cluster_evidence_osm_context_comparison.png")


def load_evidence(path=EVIDENCE_CSV):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarize(values, label):
    values = sorted(values)
    n = len(values)
    q = statistics.quantiles(values, n=4, method="inclusive") if n >= 2 else [values[0]] * 3
    print(f"  {label}: min={values[0]:.3g} p25={q[0]:.3g} median={q[1]:.3g} "
          f"p75={q[2]:.3g} max={values[-1]:.3g} mean={statistics.mean(values):.3g}")


def osm_context_bucket(row):
    """Transient, descriptive-only bucketing of the nearest observed OSM
    tag category for plotting/grouping purposes. Not a classification of
    the cluster and not persisted anywhere."""
    if int(row["features_found_in_radius"]) == 0:
        return "No OSM context"
    group = row["nearest_group"]
    is_named = row["nearest_is_named"] == "True"
    if group == "industrial":
        return "Named industrial" if is_named else "Unnamed industrial-only"
    return {
        "power": "Power",
        "waste": "Waste",
        "agricultural": "Agricultural",
        "transport": "Transport",
    }.get(group, "Other")


def main():
    rows = load_evidence()
    print(f"Loaded {len(rows)} rows from {EVIDENCE_CSV}\n")

    print("=" * 70)
    print("DISTRIBUTIONS ACROSS 60 CLUSTERS")
    print("=" * 70)
    summarize([int(r["unique_dates"]) for r in rows], "unique_dates")
    summarize([float(r["occurrence_rate"]) for r in rows], "occurrence_rate")
    summarize([int(r["detection_count"]) for r in rows], "detection_count")
    summarize([float(r["mean_frp"]) for r in rows], "mean_frp")
    summarize([float(r["max_frp"]) for r in rows], "max_frp")
    summarize([float(r["night_fraction"]) for r in rows], "night_fraction")

    print("\n" + "=" * 70)
    print("OSM CONTEXT BREAKDOWN (observed nearest tag category)")
    print("=" * 70)
    buckets = {}
    for r in rows:
        b = osm_context_bucket(r)
        buckets.setdefault(b, []).append(r)
    for b, group_rows in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        print(f"  {b}: {len(group_rows)} clusters")

    named_industrial = sum(1 for r in rows if r["nearest_group"] == "industrial" and r["nearest_is_named"] == "True")
    unnamed_industrial = sum(1 for r in rows if r["nearest_group"] == "industrial" and r["nearest_is_named"] != "True")
    no_context = sum(1 for r in rows if int(r["features_found_in_radius"]) == 0)
    print(f"\n  Named industrial context: {named_industrial}")
    print(f"  Unnamed (industrial land-use only) context: {unnamed_industrial}")
    print(f"  No OSM context at all: {no_context}")

    print("\n" + "=" * 70)
    print("RECURRENCE METRICS BY OSM CONTEXT BUCKET (descriptive only)")
    print("=" * 70)
    for b, group_rows in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        unique_dates = [int(r["unique_dates"]) for r in group_rows]
        occurrence = [float(r["occurrence_rate"]) for r in group_rows]
        print(f"  {b} (n={len(group_rows)}): "
              f"unique_dates median={statistics.median(unique_dates):.1f}, "
              f"occurrence_rate median={statistics.median(occurrence):.3f}")

    plot_comparison(rows, buckets)
    print(f"\nSaved figure to {OUTPUT_FIGURE}")


def plot_comparison(rows, buckets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(buckets.keys(), key=lambda b: -len(buckets[b]))
    data_unique_dates = [[int(r["unique_dates"]) for r in buckets[b]] for b in order]
    data_occurrence = [[float(r["occurrence_rate"]) for r in buckets[b]] for b in order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].boxplot(data_unique_dates, tick_labels=order, showmeans=True)
    axes[0].set_title("unique_dates by nearest OSM tag category\n(observed context, not a classification)")
    axes[0].set_ylabel("unique_dates")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].boxplot(data_occurrence, tick_labels=order, showmeans=True)
    axes[1].set_title("occurrence_rate by nearest OSM tag category\n(observed context, not a classification)")
    axes[1].set_ylabel("occurrence_rate")
    axes[1].tick_params(axis="x", rotation=30)

    for ax in axes:
        counts = [len(buckets[b]) for b in order]
        for i, c in enumerate(counts):
            ax.annotate(f"n={c}", (i + 1, ax.get_ylim()[1] * 0.95), ha="center", fontsize=8)

    fig.suptitle("Recurrence Metrics vs. Observed Nearest OSM Context (60 clusters, descriptive only)")
    fig.tight_layout()
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
