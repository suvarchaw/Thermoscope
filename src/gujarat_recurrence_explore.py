"""
Exploratory analysis of where spatial-temporal recurrence is occurring in
the Gujarat FIRMS subset, ahead of any decision to change the grouping
method.

This script:
  1. Loads the existing group-level stats produced by spatial_recurrence.py
     (does not recompute or change the grouping methodology).
  2. Reports the top 20 spatial groups by unique detection dates.
  3. Loads the raw FIRMS detections within the Gujarat bounds (raw file is
     read-only, never modified) for map plotting context.
  4. Writes a self-contained interactive HTML map (Leaflet, loaded from a
     CDN at view time) showing all Gujarat detections plus the top 20
     recurring groups highlighted separately.

No location is labeled by source type (industrial, agricultural, wildfire,
etc.) anywhere in this script or its output. No model is trained.
"""

import csv
import json
from pathlib import Path

from spatial_recurrence import read_gujarat_detections, RAW_CSV

GROUPS_CSV = Path("data/processed/gujarat_spatial_groups.csv")
OUTPUT_MAP = Path("results/maps/gujarat_recurrence_map.html")

TOP_N = 20


def load_groups(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def top_groups_by_unique_dates(groups, n=TOP_N):
    return sorted(groups, key=lambda g: -int(g["unique_dates"]))[:n]


def print_report(top_groups):
    header = (
        f"{'Rank':<5}{'Lat':>10}{'Lon':>11}{'Count':>8}{'UniqDates':>11}"
        f"{'Span(d)':>9}{'MeanFRP':>10}{'MaxFRP':>9}{'Day':>6}{'Night':>7}"
    )
    print(header)
    print("-" * len(header))
    for i, g in enumerate(top_groups, start=1):
        print(
            f"{i:<5}"
            f"{float(g['cell_lat_center']):>10.5f}"
            f"{float(g['cell_lon_center']):>11.5f}"
            f"{int(g['detection_count']):>8}"
            f"{int(g['unique_dates']):>11}"
            f"{int(g['active_span_days']):>9}"
            f"{float(g['mean_frp']):>10.2f}"
            f"{float(g['max_frp']):>9.2f}"
            f"{int(g['day_count']):>6}"
            f"{int(g['night_count']):>7}"
        )


def build_map_html(all_points, top_groups, output_path):
    all_points_json = json.dumps(all_points)
    top_groups_json = json.dumps(top_groups)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Gujarat FIRMS Recurrence Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body {{ margin: 0; height: 100%; }}
  #map {{ height: 100%; }}
  .legend {{
    background: white; padding: 8px 10px; font: 13px sans-serif;
    line-height: 1.4; border-radius: 4px; box-shadow: 0 0 4px rgba(0,0,0,0.3);
  }}
  .legend span.dot {{
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    margin-right: 6px;
  }}
</style>
</head>
<body>
<div id="map"></div>
<script>
  var map = L.map('map').setView([22.35, 71.2], 7);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    attribution: '&copy; OpenStreetMap contributors'
  }}).addTo(map);

  var allPoints = {all_points_json};
  var topGroups = {top_groups_json};

  var detectionLayer = L.layerGroup();
  allPoints.forEach(function(p) {{
    L.circleMarker([p[0], p[1]], {{
      radius: 2, color: '#2b6cb0', weight: 0, fillOpacity: 0.35
    }}).addTo(detectionLayer);
  }});
  detectionLayer.addTo(map);

  var topLayer = L.layerGroup();
  topGroups.forEach(function(g, i) {{
    var marker = L.circleMarker([g.lat, g.lon], {{
      radius: 9, color: '#c53030', weight: 2, fillColor: '#feb2b2',
      fillOpacity: 0.85
    }});
    marker.bindPopup(
      '<b>Recurring group #' + (i + 1) + '</b><br>' +
      'Lat/Lon: ' + g.lat.toFixed(5) + ', ' + g.lon.toFixed(5) + '<br>' +
      'Detections: ' + g.detection_count + '<br>' +
      'Unique dates: ' + g.unique_dates + '<br>' +
      'Active span: ' + g.active_span_days + ' days<br>' +
      'Mean FRP: ' + g.mean_frp.toFixed(2) + ' MW<br>' +
      'Max FRP: ' + g.max_frp.toFixed(2) + ' MW<br>' +
      'Day/Night: ' + g.day_count + ' / ' + g.night_count +
      '<br><i>No source type is implied by this marker.</i>'
    );
    marker.addTo(topLayer);
  }});
  topLayer.addTo(map);

  var legend = L.control({{position: 'bottomright'}});
  legend.onAdd = function() {{
    var div = L.DomUtil.create('div', 'legend');
    div.innerHTML =
      '<span class="dot" style="background:#2b6cb0"></span>All Gujarat detections (2023)<br>' +
      '<span class="dot" style="background:#c53030"></span>Top 20 groups by unique detection dates';
    return div;
  }};
  legend.addTo(map);
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)


def main():
    groups = load_groups(GROUPS_CSV)
    print(f"Loaded {len(groups)} spatial groups from {GROUPS_CSV}")

    top20 = top_groups_by_unique_dates(groups, TOP_N)
    print(f"\nTop {TOP_N} spatial groups by unique detection dates:\n")
    print_report(top20)

    raw_rows = list(read_gujarat_detections(RAW_CSV))
    all_points = [[float(r["latitude"]), float(r["longitude"])] for r in raw_rows]
    print(f"\nLoaded {len(all_points)} raw Gujarat detections for map context.")

    top_groups_for_js = [
        {
            "lat": float(g["cell_lat_center"]),
            "lon": float(g["cell_lon_center"]),
            "detection_count": int(g["detection_count"]),
            "unique_dates": int(g["unique_dates"]),
            "active_span_days": int(g["active_span_days"]),
            "mean_frp": float(g["mean_frp"]),
            "max_frp": float(g["max_frp"]),
            "day_count": int(g["day_count"]),
            "night_count": int(g["night_count"]),
        }
        for g in top20
    ]

    build_map_html(all_points, top_groups_for_js, OUTPUT_MAP)
    print(f"\nSaved interactive map to {OUTPUT_MAP}")


if __name__ == "__main__":
    main()
