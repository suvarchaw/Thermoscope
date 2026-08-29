"""
Reusable OpenStreetMap (Overpass API) contextual-lookup module.

Given a point (lat, lon) and a search radius, queries Overpass for nearby
features matching a documented set of infrastructure/land-use tags and
returns them sorted by distance, with enough information to distinguish:
  - named facilities vs. unnamed mapped features,
  - point-like facilities vs. broad land-use polygons (approximated from
    OSM geometry type: node vs. way/relation).

This module only retrieves and organizes OSM data. It does NOT:
  - assume proximity implies causation,
  - assign an industrial/non-industrial (or any other) source-type label,
  - compute a risk score,
  - use the FIRMS `type` field.

Callers are responsible for treating the returned context as observed
evidence only, not as a classification.
"""

import math
import time

import requests

EARTH_RADIUS_M = 6_371_000.0

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

HEADERS = {
    "User-Agent": "ThermoScope-SIH26162-prototype/0.1 (student hackathon project)",
    "Accept": "*/*",
}

REQUEST_DELAY_S = 1.1

# (key, value, human-readable label, broad category group)
OSM_TAGS = [
    ("landuse", "industrial", "Industrial land use", "industrial"),
    ("man_made", "works", "Industrial works", "industrial"),
    ("power", "plant", "Power plant", "power"),
    ("power", "generator", "Power generator", "power"),
    ("power", "substation", "Power substation", "power"),
    ("man_made", "petroleum_well", "Oil/gas well", "industrial"),
    ("pipeline", "substation", "Pipeline substation", "industrial"),
    ("aeroway", "aerodrome", "Airport/aerodrome", "transport"),
    ("landuse", "port", "Port (land use)", "transport"),
    ("harbour", "yes", "Harbour", "transport"),
    ("landuse", "landfill", "Landfill", "waste"),
    ("amenity", "waste_transfer_station", "Waste transfer station", "waste"),
    ("amenity", "waste_disposal", "Waste disposal", "waste"),
    ("landuse", "farmland", "Farmland", "agricultural"),
    ("landuse", "farmyard", "Farmyard", "agricultural"),
    ("landuse", "orchard", "Orchard", "agricultural"),
    ("landuse", "quarry", "Quarry", "other"),
    ("man_made", "chimney", "Chimney", "industrial"),
    ("man_made", "silo", "Silo", "industrial"),
    ("man_made", "wastewater_plant", "Wastewater treatment plant", "waste"),
    ("man_made", "water_works", "Water works", "other"),
]

# Tags whose typical OSM usage represents a broad zone/polygon rather than
# a discrete facility. This is a naming-convention heuristic, not a
# geometry guarantee (see classify_geometry for the actual geometry check).
POLYGON_STYLE_KEYS = {"landuse"}


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def compute_search_radius(extent_radius_m, buffer_m=500, floor_m=750, cap_m=3000):
    """Adaptive search radius: cluster extent + buffer, bounded to a sane range."""
    radius = extent_radius_m + buffer_m
    return max(floor_m, min(cap_m, radius))


def classify_tag(tags):
    """Return (tag_str, label, group) for the first matching OSM_TAGS entry
    found in this element's tags, or ('unknown', 'Unknown', 'other')."""
    for key, value, label, group in OSM_TAGS:
        if tags.get(key) == value:
            return f"{key}={value}", label, group
    return "unknown", "Unknown", "other"


def classify_geometry(element, matched_key):
    """Rough classification of whether this element looks like a discrete
    point facility or a broad land-use polygon. Approximation only: uses
    OSM element type (node vs way/relation) combined with whether the
    matched tag key is conventionally used for zone-style tagging."""
    elem_type = element.get("type")
    if elem_type == "node":
        return "point_facility"
    if matched_key in POLYGON_STYLE_KEYS:
        return "landuse_polygon"
    return "mapped_feature"


def build_query(lat, lon, radius_m):
    clauses = []
    for key, value, _label, _group in OSM_TAGS:
        for elem_type in ("node", "way", "relation"):
            clauses.append(f'{elem_type}["{key}"="{value}"](around:{radius_m},{lat},{lon});')
    body = "\n  ".join(clauses)
    return f"[out:json][timeout:25];\n(\n  {body}\n);\nout center;"


def query_overpass(lat, lon, radius_m):
    """Query Overpass for OSM_TAGS-matching elements near (lat, lon).
    Returns a list of raw elements, or None if all endpoints failed."""
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


def nearby_features(lat, lon, elements, top_n=5):
    """Return up to top_n matching features near (lat, lon), sorted by
    distance ascending. Each entry is a dict with tag/label/group, name
    (empty string if unnamed), distance_m, geometry classification, and
    coordinates."""
    scored = []
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

        tags = el.get("tags", {})
        tag_str, label, group = classify_tag(tags)
        matched_key = tag_str.split("=")[0] if "=" in tag_str else ""
        geom_class = classify_geometry(el, matched_key)
        name = tags.get("name", "")
        dist = haversine_m(lat, lon, elat, elon)

        scored.append({
            "tag": tag_str,
            "label": label,
            "group": group,
            "name": name,
            "is_named": bool(name),
            "geometry_class": geom_class,
            "distance_m": dist,
            "lat": elat,
            "lon": elon,
        })

    scored.sort(key=lambda f: f["distance_m"])
    return scored[:top_n]


def lookup_context(lat, lon, radius_m, top_n=5, retries=1, retry_delay_s=5):
    """High-level helper: query Overpass and return (status, features_found_count, top_features).
    status is 'ok' or 'query_failed'. Retries the query up to `retries` extra
    times (on top of the endpoint fallback already inside query_overpass)."""
    elements = query_overpass(lat, lon, radius_m)
    attempt = 0
    while elements is None and attempt < retries:
        time.sleep(retry_delay_s)
        elements = query_overpass(lat, lon, radius_m)
        attempt += 1

    if elements is None:
        return "query_failed", 0, []

    features = nearby_features(lat, lon, elements, top_n=top_n)
    return "ok", len(elements), features
