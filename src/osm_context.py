"""
Contextual OSM investigation for the top 20 recurring Gujarat FIRMS groups.

For each candidate group centroid (from the existing, unchanged spatial
grouping in data/processed/gujarat_spatial_groups.csv), this script queries
the public Overpass API for nearby OSM features tagged as industrial,
power-related, oil/gas, airport, waste, agricultural, or other
static-infrastructure categories within a fixed radius, and records the
single nearest matching feature.

This script does NOT:
  - assume proximity implies causation,
  - assign an industrial/non-industrial (or any other) label to a group,
  - train a model or create ML training labels,
  - change the spatial grouping method.

It only records observed nearby OSM context for later, separate judgment.
"""

import csv
import json
import math
import time
from pathlib import Path

import requests

from gujarat_recurrence_explore import load_groups, top_groups_by_unique_dates

GROUPS_CSV = Path("data/processed/gujarat_spatial_groups.csv")
OUTPUT_CSV = Path("data/processed/gujarat_top20_osm_context.csv")
OUTPUT_MAP = Path("results/maps/gujarat_top20_osm_context_map.html")

TOP_N = 20
SEARCH_RADIUS_M = 2000
REQUEST_DELAY_S = 1.1

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

HEADERS = {
    "User-Agent": "ThermoScope-SIH26162-prototype/0.1 (student hackathon project)",
    "Accept": "*/*",
}

# (key, value, human-readable label)
OSM_TAGS = [
    ("landuse", "industrial", "Industrial land use"),
    ("man_made", "works", "Industrial works"),
    ("power", "plant", "Power plant"),
    ("power", "generator", "Power generator"),
    ("power", "substation", "Power substation"),
    ("man_made", "petroleum_well", "Oil/gas well"),
    ("pipeline", "substation", "Pipeline substation"),
    ("aeroway", "aerodrome", "Airport/aerodrome"),
    ("landuse", "landfill", "Landfill"),
    ("amenity", "waste_transfer_station", "Waste transfer station"),
    ("amenity", "waste_disposal", "Waste disposal"),
    ("landuse", "farmland", "Farmland"),
    ("landuse", "farmyard", "Farmyard"),
    ("landuse", "orchard", "Orchard"),
    ("landuse", "quarry", "Quarry"),
    ("man_made", "chimney", "Chimney"),
    ("man_made", "silo", "Silo"),
    ("man_made", "wastewater_plant", "Wastewater treatment plant"),
    ("man_made", "water_works", "Water works"),
]


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def build_query(lat, lon, radius_m):
    clauses = []
    for key, value, _label in OSM_TAGS:
        for elem_type in ("node", "way", "relation"):
            clauses.append(f'{elem_type}["{key}"="{value}"](around:{radius_m},{lat},{lon});')
    body = "\n  ".join(clauses)
    return f"[out:json][timeout:25];\n(\n  {body}\n);\nout center;"


def tag_label(tags):
    for key, value, label in OSM_TAGS:
        if tags.get(key) == value:
            return f"{key}={value}", label
    return "unknown", "Unknown"


def query_overpass(lat, lon, radius_m):
    query = build_query(lat, lon, radius_m)
    last_error = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            resp = requests.post(endpoint, data={"data": query}, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            return resp.json().get("elements", [])
        except Exception as e:
            last_error = e
            continue
    print(f"  WARNING: Overpass query failed for ({lat}, {lon}): {last_error}")
    return None


def nearest_feature(lat, lon, elements):
    best = None
    best_dist = None
    for el in elements:
        if el.get("type") == "node":
            elat, elon = el.get("lat"), el.get("lon")
        else:
            center = el.get("center")
            if not center:
                continue
            elat, elon = center.get("lat"), center.get("lon")
        if elat is None or elon is None:
            continue
        d = haversine_m(lat, lon, elat, elon)
        if best_dist is None or d < best_dist:
            best_dist = d
            best = (el, elat, elon)
    return best, best_dist


def process_group(rank, group):
    lat = float(group["cell_lat_center"])
    lon = float(group["cell_lon_center"])

    elements = query_overpass(lat, lon, SEARCH_RADIUS_M)

    result = {
        "rank": rank,
        "group_lat": lat,
        "group_lon": lon,
        "unique_dates": group["unique_dates"],
        "detection_count": group["detection_count"],
        "osm_query_status": "ok" if elements is not None else "query_failed",
        "features_found_in_radius": len(elements) if elements is not None else 0,
        "nearest_osm_tag": "",
        "nearest_osm_label": "",
        "nearest_osm_name": "",
        "nearest_osm_distance_m": "",
        "nearest_osm_lat": "",
        "nearest_osm_lon": "",
    }

    if elements:
        best, dist = nearest_feature(lat, lon, elements)
        if best is not None:
            el, elat, elon = best
            tags = el.get("tags", {})
            tag_str, label = tag_label(tags)
            result["nearest_osm_tag"] = tag_str
            result["nearest_osm_label"] = label
            result["nearest_osm_name"] = tags.get("name", "")
            result["nearest_osm_distance_m"] = round(dist, 1)
            result["nearest_osm_lat"] = elat
            result["nearest_osm_lon"] = elon

    return result


def write_csv(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def build_map_html(results, output_path):
    firms_points = [
        {"lat": r["group_lat"], "lon": r["group_lon"], "rank": r["rank"],
         "unique_dates": r["unique_dates"], "detection_count": r["detection_count"]}
        for r in results
    ]
    osm_points = [
        {
            "lat": r["nearest_osm_lat"], "lon": r["nearest_osm_lon"],
            "label": r["nearest_osm_label"], "tag": r["nearest_osm_tag"],
            "name": r["nearest_osm_name"] or "(unnamed)",
            "distance_m": r["nearest_osm_distance_m"], "rank": r["rank"],
        }
        for r in results if r["nearest_osm_lat"] != ""
    ]

    firms_json = json.dumps(firms_points)
    osm_json = json.dumps(osm_points)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Gujarat Top-20 FIRMS Groups &amp; Nearby OSM Context</title>
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

  var firmsPoints = {firms_json};
  var osmPoints = {osm_json};

  firmsPoints.forEach(function(p) {{
    var marker = L.circleMarker([p.lat, p.lon], {{
      radius: 9, color: '#c53030', weight: 2, fillColor: '#feb2b2', fillOpacity: 0.85
    }});
    marker.bindPopup(
      '<b>FIRMS recurring group #' + p.rank + '</b><br>' +
      'Detections: ' + p.detection_count + '<br>' +
      'Unique dates: ' + p.unique_dates +
      '<br><i>No source type is implied by this marker.</i>'
    );
    marker.addTo(map);
  }});

  osmPoints.forEach(function(p) {{
    var marker = L.circleMarker([p.lat, p.lon], {{
      radius: 7, color: '#2f855a', weight: 2, fillColor: '#9ae6b4', fillOpacity: 0.85
    }});
    marker.bindPopup(
      '<b>Nearest OSM feature to group #' + p.rank + '</b><br>' +
      'Tag: ' + p.tag + '<br>' +
      'Type: ' + p.label + '<br>' +
      'Name: ' + p.name + '<br>' +
      'Distance from FIRMS centroid: ' + p.distance_m + ' m' +
      '<br><i>Proximity is not evidence of causation.</i>'
    );
    marker.addTo(map);

    var firmsPoint = firmsPoints.find(function(f) {{ return f.rank === p.rank; }});
    if (firmsPoint) {{
      L.polyline([[firmsPoint.lat, firmsPoint.lon], [p.lat, p.lon]], {{
        color: '#718096', weight: 1, dashArray: '4,4'
      }}).addTo(map);
    }}
  }});

  var legend = L.control({{position: 'bottomright'}});
  legend.onAdd = function() {{
    var div = L.DomUtil.create('div', 'legend');
    div.innerHTML =
      '<span class="dot" style="background:#c53030"></span>Top 20 FIRMS recurring groups<br>' +
      '<span class="dot" style="background:#2f855a"></span>Nearest matched OSM feature (within {SEARCH_RADIUS_M}m)';
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
    top20 = top_groups_by_unique_dates(groups, TOP_N)
    print(f"Querying OSM context for top {TOP_N} groups (radius {SEARCH_RADIUS_M}m)...\n")

    results = []
    for i, group in enumerate(top20, start=1):
        print(f"[{i}/{TOP_N}] group at ({group['cell_lat_center']}, {group['cell_lon_center']})...")
        result = process_group(i, group)
        results.append(result)
        if result["nearest_osm_label"]:
            print(f"    nearest: {result['nearest_osm_label']} "
                  f"({result['nearest_osm_tag']}, {result['nearest_osm_distance_m']}m, "
                  f"name={result['nearest_osm_name'] or '(unnamed)'})")
        else:
            print(f"    no matching OSM feature found within {SEARCH_RADIUS_M}m "
                  f"(status={result['osm_query_status']})")
        time.sleep(REQUEST_DELAY_S)

    write_csv(results, OUTPUT_CSV)
    print(f"\nWrote {OUTPUT_CSV}")

    build_map_html(results, OUTPUT_MAP)
    print(f"Saved map to {OUTPUT_MAP}")


if __name__ == "__main__":
    main()
