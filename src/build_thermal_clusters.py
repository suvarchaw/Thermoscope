"""
Build the first version of the ThermoScope Persistent Thermal Source
Detection Engine's spatial grouping output.

Runs DBSCAN with the parameters selected after the documented sensitivity
analysis (see DECISIONS.md): eps=375m, min_samples=8, haversine metric.

Produces:
  - data/processed/gujarat_thermal_clusters.csv   (per-cluster stats)
  - data/processed/gujarat_clustered_detections.csv (every detection + its
    cluster assignment, including noise as cluster_id = -1)
  - results/maps/gujarat_thermal_clusters.html    (interactive map)

This is a spatial grouping step only. It does not classify clusters by
persistence, does not use the FIRMS `type` field, and does not assign any
source-type label.
"""

import csv
import json
from pathlib import Path

from spatial_recurrence import read_gujarat_detections, RAW_CSV
from thermal_clustering import run_dbscan, compute_cluster_stats

EPS_M = 375
MIN_SAMPLES = 8

CLUSTERS_CSV = Path("data/processed/gujarat_thermal_clusters.csv")
DETECTIONS_CSV = Path("data/processed/gujarat_clustered_detections.csv")
OUTPUT_MAP = Path("results/maps/gujarat_thermal_clusters.html")

CLUSTER_FIELDNAMES = [
    "cluster_id", "centroid_lat", "centroid_lon", "detection_count",
    "unique_dates", "first_date", "last_date", "active_span_days",
    "mean_frp", "max_frp", "day_count", "night_count",
    "bbox_min_lat", "bbox_max_lat", "bbox_min_lon", "bbox_max_lon",
    "extent_radius_m",
]


def write_clusters_csv(cluster_stats, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CLUSTER_FIELDNAMES)
        writer.writeheader()
        writer.writerows(cluster_stats)


def write_detections_csv(rows, labels, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) + ["cluster_id"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, label in zip(rows, labels):
            out = dict(row)
            out["cluster_id"] = int(label)
            writer.writerow(out)


def build_map(rows, labels, cluster_stats, output_path):
    noise_points = [
        [float(r["latitude"]), float(r["longitude"])]
        for r, label in zip(rows, labels) if label == -1
    ]
    cluster_points = [
        {"lat": float(r["latitude"]), "lon": float(r["longitude"]), "cluster_id": int(label)}
        for r, label in zip(rows, labels) if label != -1
    ]
    cluster_summaries = [
        {
            "cluster_id": c["cluster_id"],
            "lat": c["centroid_lat"],
            "lon": c["centroid_lon"],
            "detection_count": c["detection_count"],
            "unique_dates": c["unique_dates"],
            "active_span_days": c["active_span_days"],
            "mean_frp": c["mean_frp"],
            "max_frp": c["max_frp"],
            "day_count": c["day_count"],
            "night_count": c["night_count"],
            "extent_radius_m": c["extent_radius_m"],
        }
        for c in cluster_stats
    ]

    noise_json = json.dumps(noise_points)
    cluster_points_json = json.dumps(cluster_points)
    cluster_summaries_json = json.dumps(cluster_summaries)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Gujarat FIRMS Thermal Clusters (DBSCAN)</title>
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

  var noisePoints = {noise_json};
  var clusterPoints = {cluster_points_json};
  var clusterSummaries = {cluster_summaries_json};

  // color noise as light gray, clustered detections colored by cluster id (cycled palette)
  var palette = ['#c53030','#2b6cb0','#2f855a','#d69e2e','#805ad5','#dd6b20','#319795','#b83280'];
  function colorFor(id) {{ return palette[id % palette.length]; }}

  noisePoints.forEach(function(p) {{
    L.circleMarker([p[0], p[1]], {{
      radius: 2, color: '#a0aec0', weight: 0, fillOpacity: 0.5
    }}).addTo(map);
  }});

  clusterPoints.forEach(function(p) {{
    L.circleMarker([p.lat, p.lon], {{
      radius: 3, color: colorFor(p.cluster_id), weight: 0, fillOpacity: 0.7
    }}).addTo(map);
  }});

  clusterSummaries.forEach(function(c) {{
    var marker = L.circleMarker([c.lat, c.lon], {{
      radius: 10, color: '#1a202c', weight: 2, fillColor: colorFor(c.cluster_id), fillOpacity: 0.9
    }});
    marker.bindPopup(
      '<b>Cluster ' + c.cluster_id + '</b><br>' +
      'Detections: ' + c.detection_count + '<br>' +
      'Unique dates: ' + c.unique_dates + '<br>' +
      'Active span: ' + c.active_span_days + ' days<br>' +
      'Mean FRP: ' + (c.mean_frp ? c.mean_frp.toFixed(2) : 'n/a') + ' MW<br>' +
      'Max FRP: ' + (c.max_frp ? c.max_frp.toFixed(2) : 'n/a') + ' MW<br>' +
      'Day/Night: ' + c.day_count + ' / ' + c.night_count + '<br>' +
      'Extent radius: ' + c.extent_radius_m.toFixed(0) + ' m' +
      '<br><i>No source type is implied by this marker.</i>'
    );
    marker.addTo(map);
  }});

  var legend = L.control({{position: 'bottomright'}});
  legend.onAdd = function() {{
    var div = L.DomUtil.create('div', 'legend');
    div.innerHTML =
      '<span class="dot" style="background:#a0aec0"></span>Noise (isolated detections)<br>' +
      '<span class="dot" style="background:#c53030"></span>Clustered detections (color = cluster id)<br>' +
      '<span class="dot" style="background:#1a202c"></span>Cluster centroid (click for stats)';
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
    rows = list(read_gujarat_detections(RAW_CSV))
    print(f"Loaded {len(rows)} Gujarat detections.")
    print(f"Running DBSCAN with eps={EPS_M}m, min_samples={MIN_SAMPLES}...")

    labels = run_dbscan(rows, EPS_M, MIN_SAMPLES)
    n_noise = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Clusters found: {n_clusters}")
    print(f"Noise detections: {n_noise} ({100 * n_noise / len(rows):.2f}%)")

    cluster_stats = compute_cluster_stats(rows, labels)
    write_clusters_csv(cluster_stats, CLUSTERS_CSV)
    print(f"Wrote {CLUSTERS_CSV}")

    write_detections_csv(rows, labels, DETECTIONS_CSV)
    print(f"Wrote {DETECTIONS_CSV}")

    build_map(rows, labels, cluster_stats, OUTPUT_MAP)
    print(f"Saved map to {OUTPUT_MAP}")

    sizes = sorted(c["detection_count"] for c in cluster_stats)
    import statistics
    print(f"\nCluster size stats: min={sizes[0]} median={statistics.median(sizes)} "
          f"mean={statistics.mean(sizes):.2f} max={sizes[-1]}")


if __name__ == "__main__":
    main()
