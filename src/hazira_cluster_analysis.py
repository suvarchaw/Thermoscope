"""
Focused analysis of the Hazira-area high-recurrence cluster
(~21.10-21.11N, 72.63-72.65E) to check whether the existing fixed 375m grid
is an appropriate spatial grouping scale, or whether it fragments/merges a
broader pattern.

This script does NOT change the existing grouping method in
spatial_recurrence.py, does NOT assign source-type labels, does NOT use
OSM, and does NOT train a model. It only re-uses the existing grid math at
several scales, purely for comparison, on raw detections pulled fresh from
the (unmodified) raw CSV.
"""

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from spatial_recurrence import (
    read_gujarat_detections,
    grid_steps_deg,
    assign_cell,
    RAW_CSV,
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
)

GROUPS_CSV = Path("data/processed/gujarat_spatial_groups.csv")
OUTPUT_MAP = Path("results/maps/hazira_cluster_raw_detections_map.html")
OUTPUT_SCALE_FIGURE = Path("results/figures/hazira_cluster_scale_comparison.png")

CLUSTER_BOX_LAT = (21.10, 21.11)
CLUSTER_BOX_LON = (72.63, 72.65)
RADIUS_M = 3000
SCALES_M = [100, 250, 375, 500, 1000]
MIN_DATES_FOR_SPREAD_ANALYSIS = 10


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_groups(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def compute_cluster_center(groups):
    """Detection-count-weighted centroid of existing 375m groups inside the
    observed high-recurrence box."""
    in_box = [
        g for g in groups
        if CLUSTER_BOX_LAT[0] <= float(g["cell_lat_center"]) <= CLUSTER_BOX_LAT[1]
        and CLUSTER_BOX_LON[0] <= float(g["cell_lon_center"]) <= CLUSTER_BOX_LON[1]
    ]
    total_count = sum(int(g["detection_count"]) for g in in_box)
    lat = sum(float(g["cell_lat_center"]) * int(g["detection_count"]) for g in in_box) / total_count
    lon = sum(float(g["cell_lon_center"]) * int(g["detection_count"]) for g in in_box) / total_count
    return lat, lon, in_box


def extract_local_detections(center_lat, center_lon, radius_m):
    local = []
    for row in read_gujarat_detections(RAW_CSV):
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        if haversine_m(center_lat, center_lon, lat, lon) <= radius_m:
            local.append(row)
    return local


def occupied_cells_at_scale(rows, pixel_m):
    lat_step, lon_step = grid_steps_deg(LAT_MIN, LAT_MAX, pixel_m=pixel_m)
    cell_counts = defaultdict(int)
    for row in rows:
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        cell_id = assign_cell(lat, lon, LAT_MIN, LON_MIN, lat_step, lon_step)
        cell_counts[cell_id] += 1
    return cell_counts, lat_step, lon_step


def cell_bounds(cell_id, lat_step, lon_step):
    row, col = (int(x) for x in cell_id.split("_"))
    lat0 = LAT_MIN + row * lat_step
    lat1 = lat0 + lat_step
    lon0 = LON_MIN + col * lon_step
    lon1 = lon0 + lon_step
    return lat0, lon0, lat1, lon1


def build_map(local_rows, cell_counts_375, lat_step_375, lon_step_375, center_lat, center_lon, output_path):
    points = [[float(r["latitude"]), float(r["longitude"])] for r in local_rows]
    rects = []
    for cell_id, count in cell_counts_375.items():
        lat0, lon0, lat1, lon1 = cell_bounds(cell_id, lat_step_375, lon_step_375)
        rects.append({"bounds": [[lat0, lon0], [lat1, lon1]], "count": count, "cell_id": cell_id})

    points_json = json.dumps(points)
    rects_json = json.dumps(rects)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Hazira Cluster: Raw Detections &amp; 375m Grid</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body {{ margin: 0; height: 100%; }}
  #map {{ height: 100%; }}
  .legend {{
    background: white; padding: 8px 10px; font: 13px sans-serif;
    line-height: 1.4; border-radius: 4px; box-shadow: 0 0 4px rgba(0,0,0,0.3);
  }}
</style>
</head>
<body>
<div id="map"></div>
<script>
  var map = L.map('map').setView([{center_lat}, {center_lon}], 15);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    attribution: '&copy; OpenStreetMap contributors'
  }}).addTo(map);

  var points = {points_json};
  points.forEach(function(p) {{
    L.circleMarker([p[0], p[1]], {{
      radius: 2, color: '#2b6cb0', weight: 0, fillOpacity: 0.5
    }}).addTo(map);
  }});

  var rects = {rects_json};
  rects.forEach(function(r) {{
    var rect = L.rectangle(r.bounds, {{
      color: '#c53030', weight: 1, fillOpacity: 0.05
    }});
    rect.bindPopup('375m cell ' + r.cell_id + '<br>Detections: ' + r.count);
    rect.addTo(map);
  }});

  var legend = L.control({{position: 'bottomright'}});
  legend.onAdd = function() {{
    var div = L.DomUtil.create('div', 'legend');
    div.innerHTML =
      'Blue dots: raw FIRMS detections<br>Red boxes: existing 375m grid cells (unchanged method)';
    return div;
  }};
  legend.addTo(map);
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)


def plot_scale_comparison(scale_results, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scales = sorted(scale_results.keys())
    occupied = [len(scale_results[s]) for s in scales]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([str(s) for s in scales], occupied, color="#2b6cb0", edgecolor="black")
    ax.set_xlabel("Grid cell size (m)")
    ax.set_ylabel("Occupied cells within 3km of Hazira cluster center")
    ax.set_title("Occupied Spatial Groups vs. Grid Scale (Hazira Cluster)")
    for i, v in enumerate(occupied):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def cross_date_distance_stats(rows_by_date_position):
    """Given a list of (lat, lon, date) for one recurring area, compute the
    distribution of distances between detections on different dates."""
    distances = []
    n = len(rows_by_date_position)
    for i in range(n):
        lat1, lon1, d1 = rows_by_date_position[i]
        for j in range(i + 1, n):
            lat2, lon2, d2 = rows_by_date_position[j]
            if d1 != d2:
                distances.append(haversine_m(lat1, lon1, lat2, lon2))
    return distances


def main():
    all_groups = load_groups(GROUPS_CSV)
    center_lat, center_lon, box_groups = compute_cluster_center(all_groups)
    print(f"Hazira cluster center (detection-weighted, from {len(box_groups)} "
          f"existing 375m groups in the observed box): "
          f"lat={center_lat:.5f}, lon={center_lon:.5f}\n")

    local_rows = extract_local_detections(center_lat, center_lon, RADIUS_M)
    print(f"Raw detections within {RADIUS_M}m of cluster center: {len(local_rows)}")

    unique_dates = {r["acq_date"] for r in local_rows}
    print(f"Unique detection dates: {len(unique_dates)}")

    lats = [float(r["latitude"]) for r in local_rows]
    lons = [float(r["longitude"]) for r in local_rows]
    lat_extent_m = haversine_m(min(lats), center_lon, max(lats), center_lon)
    lon_extent_m = haversine_m(center_lat, min(lons), center_lat, max(lons))
    print(f"Spatial extent: lat [{min(lats):.5f}, {max(lats):.5f}] "
          f"({lat_extent_m:.0f}m), lon [{min(lons):.5f}, {max(lons):.5f}] "
          f"({lon_extent_m:.0f}m)")

    print(f"\nOccupied cell counts at different scales (within {RADIUS_M}m radius):")
    scale_results = {}
    scale_375_data = None
    for pixel_m in SCALES_M:
        cell_counts, lat_step, lon_step = occupied_cells_at_scale(local_rows, pixel_m)
        scale_results[pixel_m] = cell_counts
        counts = list(cell_counts.values())
        print(f"  {pixel_m:>5}m: {len(cell_counts):>4} occupied cells | "
              f"detections/cell min={min(counts)} median={statistics.median(counts):.1f} "
              f"mean={statistics.mean(counts):.2f} max={max(counts)}")
        if pixel_m == 375:
            scale_375_data = (cell_counts, lat_step, lon_step)

    plot_scale_comparison(scale_results, OUTPUT_SCALE_FIGURE)
    print(f"\nSaved scale-comparison figure to {OUTPUT_SCALE_FIGURE}")

    build_map(local_rows, scale_375_data[0], scale_375_data[1], scale_375_data[2],
               center_lat, center_lon, OUTPUT_MAP)
    print(f"Saved map to {OUTPUT_MAP}")

    print(f"\nCross-date distance distribution for 375m groups with "
          f"unique_dates >= {MIN_DATES_FOR_SPREAD_ANALYSIS} inside this cluster area:")
    major_groups = [
        g for g in box_groups
        if int(g["unique_dates"]) >= MIN_DATES_FOR_SPREAD_ANALYSIS
    ]
    # Assign each local raw detection to its exact 375m cell id, using the
    # same grid function/anchor as the existing (unchanged) grouping method,
    # so membership exactly matches gujarat_spatial_groups.csv.
    cell_counts_375, lat_step_375, lon_step_375 = scale_375_data[0], scale_375_data[1], scale_375_data[2]
    members_by_cell = defaultdict(list)
    for r in local_rows:
        lat, lon = float(r["latitude"]), float(r["longitude"])
        cid = assign_cell(lat, lon, LAT_MIN, LON_MIN, lat_step_375, lon_step_375)
        members_by_cell[cid].append((lat, lon, r["acq_date"]))

    for g in sorted(major_groups, key=lambda g: -int(g["unique_dates"])):
        member_rows = members_by_cell.get(g["cell_id"], [])
        if len(member_rows) < 2:
            continue
        distances = cross_date_distance_stats(member_rows)
        if not distances:
            continue
        distances.sort()
        n = len(distances)
        print(f"  cell {g['cell_id']} (unique_dates={g['unique_dates']}, "
              f"members~{len(member_rows)}, cross-date pairs={n}): "
              f"min={distances[0]:.1f}m median={statistics.median(distances):.1f}m "
              f"mean={statistics.mean(distances):.1f}m "
              f"p90={distances[int(0.9*(n-1))]:.1f}m max={distances[-1]:.1f}m")


if __name__ == "__main__":
    main()
