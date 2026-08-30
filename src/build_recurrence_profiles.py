"""
Build Recurrence Profiles for all 60 DBSCAN thermal clusters.

Reads the existing per-cluster evidence table
(data/processed/gujarat_cluster_evidence_review.csv) and adds three new
columns computed ONLY from unique_dates, active_span_days, and
top3_days_share (see src/recurrence_profile.py):

  - recurrence_strength
  - short_window_recurrence
  - burst_concentrated

Every existing evidence column (spatial, temporal, thermal, day/night, and
OSM context) is carried through unchanged. This is a temporal-recurrence
annotation only — it is not a source-type classification, not a final
persistent-source label, does not use FIRMS `type`, and does not use any
OSM field.
"""

import csv
from pathlib import Path

from recurrence_profile import compute_recurrence_profile

INPUT_CSV = Path("data/processed/gujarat_cluster_evidence_review.csv")
OUTPUT_CSV = Path("data/processed/gujarat_cluster_recurrence_profiles.csv")
OUTPUT_FIGURE = Path("results/figures/cluster_recurrence_profiles.png")

NEW_FIELDS = ["recurrence_strength", "short_window_recurrence", "burst_concentrated"]


def load_evidence(path=INPUT_CSV):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_profiles(rows):
    profiled = []
    for row in rows:
        unique_dates = int(row["unique_dates"])
        active_span_days = int(row["active_span_days"])
        top3_days_share = float(row["top3_days_share"])

        profile = compute_recurrence_profile(unique_dates, active_span_days, top3_days_share)

        out_row = dict(row)  # preserve every existing evidence column unchanged
        out_row.update(profile)
        profiled.append(out_row)
    return profiled


def write_csv(rows, path=OUTPUT_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_profiles(rows, output_path=OUTPUT_FIGURE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tier_colors = {"Strong": "#c53030", "Moderate": "#d69e2e", "Limited": "#2b6cb0"}
    tier_order = ["Limited", "Moderate", "Strong"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # Panel 1: unique_dates vs occurrence_rate, colored by tier, marker shape
    # by short_window_recurrence, ringed if burst_concentrated.
    ax = axes[0]
    for tier in tier_order:
        tier_rows = [r for r in rows if r["recurrence_strength"] == tier]
        for r in tier_rows:
            marker = "^" if r["short_window_recurrence"] else "o"
            edgecolor = "black"
            linewidth = 2.5 if r["burst_concentrated"] else 0.8
            ax.scatter(
                float(r["unique_dates"]), float(r["occurrence_rate"]),
                c=tier_colors[tier], marker=marker, s=90,
                edgecolor=edgecolor, linewidth=linewidth, alpha=0.85,
            )
    ax.set_xlabel("unique_dates")
    ax.set_ylabel("occurrence_rate")
    ax.set_title("Recurrence Profiles: strength tier (color), short-window (triangle),\nburst-concentrated (thick outline)")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=tier_colors[t], markersize=10, label=t)
        for t in tier_order
    ] + [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=10, label="short_window_recurrence"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black",
               markeredgewidth=2.5, markersize=10, label="burst_concentrated"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    # Panel 2: counts bar chart
    ax2 = axes[1]
    tier_counts = [sum(1 for r in rows if r["recurrence_strength"] == t) for t in tier_order]
    bars = ax2.bar(tier_order, tier_counts, color=[tier_colors[t] for t in tier_order], edgecolor="black")
    for bar, count in zip(bars, tier_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, count, str(count), ha="center", va="bottom")
    ax2.set_ylabel("Number of clusters")
    ax2.set_title("Cluster counts per recurrence_strength tier\n(60 clusters total)")

    n_short = sum(1 for r in rows if r["short_window_recurrence"])
    n_burst = sum(1 for r in rows if r["burst_concentrated"])
    ax2.text(0.5, -0.18, f"short_window_recurrence=True: {n_short}/60    burst_concentrated=True: {n_burst}/60",
              transform=ax2.transAxes, ha="center", fontsize=9)

    fig.suptitle("ThermoScope Recurrence Profiles — 60 DBSCAN Clusters (temporal recurrence only, not a source-type label)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    rows = load_evidence()
    print(f"Loaded {len(rows)} clusters from {INPUT_CSV}")

    profiled = build_profiles(rows)
    write_csv(profiled)
    print(f"Wrote {OUTPUT_CSV}")

    from collections import Counter
    tier_counts = Counter(r["recurrence_strength"] for r in profiled)
    n_short = sum(1 for r in profiled if r["short_window_recurrence"])
    n_burst = sum(1 for r in profiled if r["burst_concentrated"])

    print("\nRecurrence strength tier counts:")
    for tier in ("Strong", "Moderate", "Limited"):
        print(f"  {tier}: {tier_counts.get(tier, 0)}")
    print(f"\nshort_window_recurrence=True: {n_short}")
    print(f"burst_concentrated=True: {n_burst}")

    plot_profiles(profiled)
    print(f"\nSaved figure to {OUTPUT_FIGURE}")

    return profiled


if __name__ == "__main__":
    main()
