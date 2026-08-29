"""
Spatial-temporal recurrence analysis for FIRMS VIIRS detections.

Reads the raw FIRMS CSV (unmodified), filters to the current Gujarat
working-assumption bounding box, groups detections into fixed-size spatial
cells sized to match the VIIRS 375m product's nominal nadir pixel
resolution, and computes per-cell recurrence statistics.

Grid method
-----------
Each detection is assigned to a cell in a fixed lat/lon grid anchored at
the bounding box's lower-left corner. Cell size is derived from the VIIRS
375m nominal pixel resolution:

  lat_step_deg = 375 / METERS_PER_DEGREE_LAT
  lon_step_deg = 375 / (METERS_PER_DEGREE_LAT * cos(mean_latitude))

This is a simple equirectangular grid, not an equal-area projection, so
actual cell width drifts slightly across a several-degree latitude span,
and it does not account for the pixel footprint growing away from nadir
(reflected in the FIRMS `scan`/`track` columns). That's an accepted
simplification for this exploratory step, not a claim of precise
geolocation.

This script does not train a model, does not use the FIRMS `type` field,
and does not decide a persistence threshold. Repeated detections at a cell
are reported as-is; no claim is made that they represent industrial
sources.
"""

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

RAW_CSV = Path("data/raw/fire_archive_SV-C2_794895.csv")
OUTPUT_CSV = Path("data/processed/gujarat_spatial_groups.csv")
OUTPUT_FIGURE = Path("results/figures/gujarat_unique_dates_per_group_hist.png")

LAT_MIN, LAT_MAX = 20.0, 24.7
LON_MIN, LON_MAX = 68.0, 74.5

VIIRS_NOMINAL_PIXEL_M = 375.0
METERS_PER_DEGREE_LAT = 111_320.0


def grid_steps_deg(lat_min, lat_max, pixel_m=VIIRS_NOMINAL_PIXEL_M):
    """Return (lat_step_deg, lon_step_deg) for a grid sized to pixel_m,
    with the longitude step evaluated at the bounding box's mean latitude."""
    mean_lat_rad = math.radians((lat_min + lat_max) / 2)
    lat_step = pixel_m / METERS_PER_DEGREE_LAT
    lon_step = pixel_m / (METERS_PER_DEGREE_LAT * math.cos(mean_lat_rad))
    return lat_step, lon_step


def assign_cell(lat, lon, lat_min, lon_min, lat_step, lon_step):
    """Assign a (lat, lon) point to a grid cell id, anchored at (lat_min, lon_min)."""
    row = math.floor((lat - lat_min) / lat_step)
    col = math.floor((lon - lon_min) / lon_step)
    return f"{row}_{col}"


def read_gujarat_detections(raw_csv_path):
    """Read the raw FIRMS CSV and yield rows within the Gujarat bounding box.
    Does not modify the source file."""
    with open(raw_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
            except (ValueError, KeyError):
                continue
            if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
                yield row


def group_detections(rows, lat_min=LAT_MIN, lon_min=LON_MIN,
                      lat_max=LAT_MAX, lon_max=LON_MAX):
    """Group detection rows into spatial cells and compute per-cell stats.

    Returns a list of dicts, one per occupied cell, sorted by cell_id.
    """
    lat_step, lon_step = grid_steps_deg(lat_min, lat_max)

    cells = defaultdict(lambda: {
        "dates": set(),
        "frp_values": [],
        "day_count": 0,
        "night_count": 0,
        "lat_sum": 0.0,
        "lon_sum": 0.0,
        "count": 0,
    })

    for row in rows:
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        cell_id = assign_cell(lat, lon, lat_min, lon_min, lat_step, lon_step)

        c = cells[cell_id]
        c["count"] += 1
        c["dates"].add(row["acq_date"])
        c["lat_sum"] += lat
        c["lon_sum"] += lon

        try:
            c["frp_values"].append(float(row.get("frp", "")))
        except ValueError:
            pass

        daynight = row.get("daynight", "")
        if daynight == "D":
            c["day_count"] += 1
        elif daynight == "N":
            c["night_count"] += 1

    results = []
    for cell_id, c in cells.items():
        dates_sorted = sorted(c["dates"])
        first_date = dates_sorted[0]
        last_date = dates_sorted[-1]
        active_span_days = (
            _date_to_ordinal(last_date) - _date_to_ordinal(first_date)
        )
        frp_values = c["frp_values"]
        results.append({
            "cell_id": cell_id,
            "cell_lat_center": c["lat_sum"] / c["count"],
            "cell_lon_center": c["lon_sum"] / c["count"],
            "detection_count": c["count"],
            "unique_dates": len(c["dates"]),
            "first_date": first_date,
            "last_date": last_date,
            "active_span_days": active_span_days,
            "mean_frp": statistics.mean(frp_values) if frp_values else None,
            "max_frp": max(frp_values) if frp_values else None,
            "day_count": c["day_count"],
            "night_count": c["night_count"],
        })

    results.sort(key=lambda r: r["cell_id"])
    return results


def _date_to_ordinal(date_str):
    """Convert an acq_date string (YYYY-MM-DD) to a day ordinal, stdlib-only."""
    import datetime
    y, m, d = (int(p) for p in date_str.split("-"))
    return datetime.date(y, m, d).toordinal()


def write_groups_csv(groups, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cell_id", "cell_lat_center", "cell_lon_center",
        "detection_count", "unique_dates", "first_date", "last_date",
        "active_span_days", "mean_frp", "max_frp",
        "day_count", "night_count",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for g in groups:
            writer.writerow(g)


def summarize(values, label):
    values_sorted = sorted(values)
    n = len(values_sorted)
    print(f"\n{label} (n={n})")
    print(f"  min:    {values_sorted[0]}")
    print(f"  median: {statistics.median(values_sorted)}")
    print(f"  mean:   {statistics.mean(values_sorted):.3f}")
    print(f"  p90:    {values_sorted[int(0.9 * (n - 1))]}")
    print(f"  max:    {values_sorted[-1]}")


def plot_unique_dates_histogram(groups, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unique_dates = [g["unique_dates"] for g in groups]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    max_val = max(unique_dates)
    ax.hist(unique_dates, bins=range(1, max_val + 2), align="left", edgecolor="black")
    ax.set_yscale("log")
    ax.set_xlabel("Unique detection dates per spatial group")
    ax.set_ylabel("Number of spatial groups (log scale)")
    ax.set_title("Gujarat FIRMS 2023: Unique Detection Dates per ~375m Spatial Group")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    rows = list(read_gujarat_detections(RAW_CSV))
    print(f"Gujarat detections read: {len(rows)}")

    groups = group_detections(rows)
    print(f"Spatial groups (~375m cells) occupied: {len(groups)}")

    write_groups_csv(groups, OUTPUT_CSV)
    print(f"Wrote group-level stats to {OUTPUT_CSV}")

    summarize([g["detection_count"] for g in groups], "Detections per group")
    summarize([g["unique_dates"] for g in groups], "Unique dates per group")
    summarize([g["active_span_days"] for g in groups], "Active span (days) per group")

    plot_unique_dates_histogram(groups, OUTPUT_FIGURE)
    print(f"\nSaved figure to {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
