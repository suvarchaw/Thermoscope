"""
OSM contextual analysis for all 60 DBSCAN thermal clusters
(data/processed/gujarat_thermal_clusters.csv).

For each cluster, queries Overpass for nearby OSM-tagged features (using an
adaptive radius derived from the cluster's own spatial extent) and records
observed context: nearest feature, nearest *named* feature, and counts of
nearby features by broad category. Up to 5 nearest matches per cluster are
kept for inspection.

This script does NOT assign a source-type label, does NOT compute a risk
score, does NOT use the FIRMS `type` field, and does NOT treat proximity as
causation. It only records what is mapped nearby.
"""

import csv
import json
import time
from collections import Counter
from pathlib import Path

from osm_lookup import compute_search_radius, lookup_context

CLUSTERS_CSV = Path("data/processed/gujarat_thermal_clusters.csv")
OUTPUT_CSV = Path("data/processed/gujarat_cluster_osm_context.csv")
OUTPUT_MAP = Path("results/maps/gujarat_cluster_osm_context_map.html")

TOP_N_FEATURES = 5
REQUEST_DELAY_S = 1.1


def load_clusters(path=CLUSTERS_CSV):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def process_cluster(cluster):
    cluster_id = int(cluster["cluster_id"])
    lat = float(cluster["centroid_lat"])
    lon = float(cluster["centroid_lon"])
    extent_radius_m = float(cluster["extent_radius_m"])
    search_radius_m = compute_search_radius(extent_radius_m)

    status, n_found, features = lookup_context(lat, lon, search_radius_m, top_n=TOP_N_FEATURES)

    result = {
        "cluster_id": cluster_id,
        "centroid_lat": lat,
        "centroid_lon": lon,
        "extent_radius_m": extent_radius_m,
        "search_radius_m": search_radius_m,
        "detection_count": cluster["detection_count"],
        "unique_dates": cluster["unique_dates"],
        "osm_query_status": status,
        "features_found_in_radius": n_found,
        "nearest_tag": "", "nearest_label": "", "nearest_group": "",
        "nearest_name": "", "nearest_is_named": "", "nearest_geometry_class": "",
        "nearest_distance_m": "", "nearest_lat": "", "nearest_lon": "",
        "nearest_named_tag": "", "nearest_named_label": "", "nearest_named_name": "",
        "nearest_named_distance_m": "",
        "n_industrial": 0, "n_power": 0, "n_waste": 0,
        "n_agricultural": 0, "n_transport": 0, "n_other": 0,
        "top_features_summary": "",
    }

    if features:
        group_counts = Counter(f["group"] for f in features)
        for group in ("industrial", "power", "waste", "agricultural", "transport", "other"):
            result[f"n_{group}"] = group_counts.get(group, 0)

        nearest = features[0]
        result.update({
            "nearest_tag": nearest["tag"],
            "nearest_label": nearest["label"],
            "nearest_group": nearest["group"],
            "nearest_name": nearest["name"],
            "nearest_is_named": nearest["is_named"],
            "nearest_geometry_class": nearest["geometry_class"],
            "nearest_distance_m": round(nearest["distance_m"], 1),
            "nearest_lat": nearest["lat"],
            "nearest_lon": nearest["lon"],
        })

        named_features = [f for f in features if f["is_named"]]
        if named_features:
            nn = named_features[0]
            result.update({
                "nearest_named_tag": nn["tag"],
                "nearest_named_label": nn["label"],
                "nearest_named_name": nn["name"],
                "nearest_named_distance_m": round(nn["distance_m"], 1),
            })

        summary_parts = [
            f"{f['tag']}@{f['distance_m']:.0f}m"
            f"({f['name'] if f['name'] else 'unnamed'}, {f['geometry_class']})"
            for f in features
        ]
        result["top_features_summary"] = "; ".join(summary_parts)

    return result, features


def write_csv(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def build_map_html(results, all_features_by_cluster, output_path):
    cluster_points = [
        {
            "cluster_id": r["cluster_id"], "lat": r["centroid_lat"], "lon": r["centroid_lon"],
            "detection_count": r["detection_count"], "unique_dates": r["unique_dates"],
            "search_radius_m": r["search_radius_m"],
            "nearest_label": r["nearest_label"], "nearest_name": r["nearest_name"] or "(unnamed)",
            "nearest_distance_m": r["nearest_distance_m"],
        }
        for r in results
    ]
    osm_points = []
    for r in results:
        for f in all_features_by_cluster.get(r["cluster_id"], []):
            osm_points.append({
                "cluster_id": r["cluster_id"], "lat": f["lat"], "lon": f["lon"],
                "label": f["label"], "tag": f["tag"], "group": f["group"],
                "name": f["name"] or "(unnamed)", "geometry_class": f["geometry_class"],
                "distance_m": round(f["distance_m"], 1),
            })

    cluster_json = json.dumps(cluster_points)
    osm_json = json.dumps(osm_points)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Gujarat 60 Thermal Clusters &amp; OSM Context</title>
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

  var clusterPoints = {cluster_json};
  var osmPoints = {osm_json};

  var groupColors = {{
    industrial: '#c53030', power: '#d69e2e', waste: '#805ad5',
    agricultural: '#38a169', transport: '#3182ce', other: '#718096'
  }};

  clusterPoints.forEach(function(c) {{
    var marker = L.circleMarker([c.lat, c.lon], {{
      radius: 6, color: '#1a202c', weight: 2, fillColor: '#feb2b2', fillOpacity: 0.9
    }});
    marker.bindPopup(
      '<b>Cluster ' + c.cluster_id + '</b><br>' +
      'Detections: ' + c.detection_count + '<br>' +
      'Unique dates: ' + c.unique_dates + '<br>' +
      'OSM search radius: ' + c.search_radius_m.toFixed(0) + ' m<br>' +
      'Nearest OSM: ' + c.nearest_label + ' (' + c.nearest_name + '), ' +
      c.nearest_distance_m + ' m' +
      '<br><i>Observed context only — not a source-type label.</i>'
    );
    marker.addTo(map);
  }});

  osmPoints.forEach(function(p) {{
    var marker = L.circleMarker([p.lat, p.lon], {{
      radius: 4, color: groupColors[p.group] || '#718096', weight: 1,
      fillOpacity: 0.7
    }});
    marker.bindPopup(
      '<b>' + p.label + '</b> (' + p.tag + ')<br>' +
      'Name: ' + p.name + '<br>' +
      'Geometry: ' + p.geometry_class + '<br>' +
      'Distance to cluster ' + p.cluster_id + ': ' + p.distance_m + ' m'
    );
    marker.addTo(map);
  }});

  var legend = L.control({{position: 'bottomright'}});
  legend.onAdd = function() {{
    var div = L.DomUtil.create('div', 'legend');
    div.innerHTML =
      '<span class="dot" style="background:#feb2b2;border:2px solid #1a202c"></span>Thermal cluster centroid (60)<br>' +
      '<span class="dot" style="background:#c53030"></span>Industrial<br>' +
      '<span class="dot" style="background:#d69e2e"></span>Power<br>' +
      '<span class="dot" style="background:#805ad5"></span>Waste<br>' +
      '<span class="dot" style="background:#38a169"></span>Agricultural<br>' +
      '<span class="dot" style="background:#3182ce"></span>Transport (airport/port)<br>' +
      '<span class="dot" style="background:#718096"></span>Other';
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
    clusters = load_clusters()
    print(f"Loaded {len(clusters)} clusters. Querying OSM context (adaptive radius per cluster)...\n")

    results = []
    all_features_by_cluster = {}
    for i, cluster in enumerate(clusters, start=1):
        cid = int(cluster["cluster_id"])
        print(f"[{i}/{len(clusters)}] cluster {cid} "
              f"(extent_radius_m={float(cluster['extent_radius_m']):.0f})...")
        result, features = process_cluster(cluster)
        results.append(result)
        all_features_by_cluster[cid] = features
        if features:
            print(f"    {result['features_found_in_radius']} features found; "
                  f"nearest: {result['nearest_label']} @ {result['nearest_distance_m']}m "
                  f"(name={result['nearest_name'] or '(unnamed)'})")
        else:
            print(f"    no matching features found (status={result['osm_query_status']}, "
                  f"radius={result['search_radius_m']:.0f}m)")
        time.sleep(REQUEST_DELAY_S)

    n_failed = sum(1 for r in results if r["osm_query_status"] != "ok")
    print(f"\n{len(results) - n_failed}/{len(results)} queries succeeded on first pass.")

    write_csv(results, OUTPUT_CSV)
    print(f"Wrote {OUTPUT_CSV}")

    build_map_html(results, all_features_by_cluster, OUTPUT_MAP)
    print(f"Saved map to {OUTPUT_MAP}")

    return results, all_features_by_cluster


if __name__ == "__main__":
    main()
