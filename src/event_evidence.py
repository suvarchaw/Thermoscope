"""
Additive evidence-join layer for the event-level representation
(src/event_construction.py). Attaches, to every event, distances to the
same external evidence sources already validated in prior read-only
investigations -- no new evidence source, no new threshold, no label.

Evidence sources (all previously investigated, none new):
  - OSM industrial/power (landuse=industrial, man_made=works,
    power=plant/generator/substation) -- the same tag values already used
    for the 60-cluster OSM enrichment (src/osm_lookup.py OSM_TAGS),
    queried once for the whole Gujarat bbox instead of per-cluster.
  - OSM flare (man_made=flare) -- 15 real nodes, confirmed in the
    event-level investigation.
  - OSM kiln/brickyard (man_made=kiln, industrial=brickyard) -- only 2
    real nodes in all of Gujarat, confirmed negligible in the
    investigation; carried through unchanged, not inflated.
  - WRI Global Power Plant Database, Gujarat-bbox, Coal/Gas/Oil only
    (Solar/Wind/Hydro/Nuclear excluded -- they produce no combustion heat
    signature, per the original cluster-level investigation).
  - The existing 60 persistent clusters (unchanged geometry and
    recurrence_strength) -- an event "overlaps" a cluster using the exact
    same nearest-centroid-within-extent_radius_m rule already approved in
    cross_year_recurrence_analysis.py, applied to event centroids instead
    of individual detections. Does NOT modify that module or its outputs.
  - ESA WorldCover 10m v200 (2021) land cover class, sampled locally from
    downloaded COG tiles (see `ensure_worldcover_tiles`/`sample_land_cover`)
    -- guarded behind an optional rasterio import so this module (and its
    tests) never require rasterio or network access just to run the
    distance-based joins.

External raw evidence (OSM/GPPD query results) is cached to small,
static CSVs under data/raw/ using the same fetch-if-missing idempotent
pattern as firms_ingestion.py -- re-running this module does not
re-query Overpass/GitHub unless those files are deleted.

SCOPE, EXPLICITLY: this module computes OBSERVED/DERIVED event fields
(carried through unchanged from event_construction.py) plus EXTERNAL
EVIDENCE distances/classes. It does NOT assign a source-class label, does
NOT compute a risk score, and does NOT train a model -- label-rule
feasibility is a separate, non-persisted investigation (see
DECISIONS.md), kept deliberately out of this module so the evidence table
never bakes in labeling logic (see the leakage-audit note in
DECISIONS.md: fields used to construct a future label rule must not
silently double as model input features without a deliberate
justification).
"""

import csv
import math
import time
from collections import defaultdict
from pathlib import Path

import requests

from spatial_recurrence import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from cross_year_recurrence_analysis import load_cluster_definitions, haversine_m as _cy_haversine_m

EVENTS_CSV = Path("data/processed/gujarat_thermal_events.csv")
INTEGRATED_EVIDENCE_CSV = Path("data/processed/gujarat_cluster_integrated_evidence.csv")
OUTPUT_CSV = Path("data/processed/gujarat_event_evidence.csv")

OSM_INDUSTRIAL_POWER_CSV = Path("data/raw/osm_industrial_power_points.csv")
OSM_FLARE_CSV = Path("data/raw/osm_flare_points.csv")
OSM_KILN_CSV = Path("data/raw/osm_kiln_points.csv")
GPPD_THERMAL_CSV = Path("data/raw/wri_gppd_gujarat_thermal_plants.csv")

WORLDCOVER_CACHE_DIR = Path("data/external_cache/esa_worldcover")
WORLDCOVER_BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
WORLDCOVER_CLASS_NAMES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare/sparse vegetation", 70: "Snow/ice",
    80: "Water bodies", 90: "Herbaceous wetland", 95: "Mangroves",
    100: "Moss/lichen",
}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_HEADERS = {"User-Agent": "ThermoScope-SIH26162-prototype/0.1 (student hackathon project)"}

EARTH_RADIUS_M = 6_371_000.0
MEAN_LAT_DEG = 22.35
LAT_DEG_PER_M = 1.0 / 110_540.0
LON_DEG_PER_M = 1.0 / (111_320.0 * math.cos(math.radians(MEAN_LAT_DEG)))


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


# ------------------------------------------------------------------
# External evidence fetch (fetch-if-missing, same idempotent pattern
# firms_ingestion.py uses -- never re-fetches an already-cached file).
# ------------------------------------------------------------------

def _overpass(query, retries=4, base_delay_s=10):
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=150, headers=OVERPASS_HEADERS)
            resp.raise_for_status()
            return resp.json().get("elements", [])
        except requests.RequestException as e:
            last_err = e
            time.sleep(base_delay_s * (attempt + 1))
    raise last_err


def _elements_to_rows(elements):
    rows = []
    for el in elements:
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None:
            continue
        rows.append({"osm_type": el["type"], "osm_id": el["id"], "lat": lat, "lon": lon,
                     "tags": ";".join(f"{k}={v}" for k, v in el.get("tags", {}).items())})
    return rows


def _write_points_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["osm_type", "osm_id", "lat", "lon", "tags"])
        w.writeheader()
        w.writerows(rows)


def fetch_osm_industrial_power(path=OSM_INDUSTRIAL_POWER_CSV, force=False):
    if path.exists() and not force:
        return path
    bbox = f"{LAT_MIN},{LON_MIN},{LAT_MAX},{LON_MAX}"
    query = f'''[out:json][timeout:120];
(
  node["landuse"="industrial"]({bbox});way["landuse"="industrial"]({bbox});
  node["man_made"="works"]({bbox});way["man_made"="works"]({bbox});
  node["power"="plant"]({bbox});way["power"="plant"]({bbox});
  node["power"="generator"]({bbox});
  node["power"="substation"]({bbox});way["power"="substation"]({bbox});
);
out center;'''
    _write_points_csv(_elements_to_rows(_overpass(query)), path)
    return path


def fetch_osm_flare(path=OSM_FLARE_CSV, force=False):
    if path.exists() and not force:
        return path
    bbox = f"{LAT_MIN},{LON_MIN},{LAT_MAX},{LON_MAX}"
    query = f'[out:json][timeout:90];(node["man_made"="flare"]({bbox});way["man_made"="flare"]({bbox}););out center;'
    _write_points_csv(_elements_to_rows(_overpass(query)), path)
    return path


def fetch_osm_kiln(path=OSM_KILN_CSV, force=False):
    if path.exists() and not force:
        return path
    bbox = f"{LAT_MIN},{LON_MIN},{LAT_MAX},{LON_MAX}"
    query = f'''[out:json][timeout:90];
(
  node["man_made"="kiln"]({bbox});way["man_made"="kiln"]({bbox});
  node["industrial"="brickyard"]({bbox});way["industrial"="brickyard"]({bbox});
);
out center;'''
    _write_points_csv(_elements_to_rows(_overpass(query)), path)
    return path


def fetch_gppd_thermal_plants(path=GPPD_THERMAL_CSV, force=False):
    if path.exists() and not force:
        return path
    url = "https://raw.githubusercontent.com/wri/global-power-plant-database/master/output_database/global_power_plant_database.csv"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    rows = list(csv.DictReader(resp.text.splitlines()))
    guj_thermal = [r for r in rows if r["country"] == "IND" and r["latitude"] and r["longitude"]
                   and LAT_MIN <= float(r["latitude"]) <= LAT_MAX and LON_MIN <= float(r["longitude"]) <= LON_MAX
                   and r["primary_fuel"] in ("Coal", "Gas", "Oil")]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name", "gppd_idnr", "capacity_mw", "latitude", "longitude", "primary_fuel", "commissioning_year"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in guj_thermal:
            w.writerow({k: r[k] for k in fieldnames})
    return path


def load_points_csv(path):
    with open(path, newline="") as f:
        return [(float(r["lat"]), float(r["lon"])) for r in csv.DictReader(f)]


def load_gppd_points(path=GPPD_THERMAL_CSV):
    with open(path, newline="") as f:
        return [(float(r["latitude"]), float(r["longitude"])) for r in csv.DictReader(f)]


# ------------------------------------------------------------------
# Grid-indexed nearest-distance join (efficient at event scale --
# thousands of events x thousands of evidence points).
# ------------------------------------------------------------------

DEFAULT_CELL_SIZE_M = 2000.0
DEFAULT_MAX_SEARCH_M = 20_000.0  # generous safety cap, not a labeling threshold


def build_point_grid(points, cell_size_m=DEFAULT_CELL_SIZE_M):
    cell_lat = cell_size_m * LAT_DEG_PER_M
    cell_lon = cell_size_m * LON_DEG_PER_M
    grid = defaultdict(list)
    for p in points:
        ci, cj = int(p[0] // cell_lat), int(p[1] // cell_lon)
        grid[(ci, cj)].append(p)
    return grid, cell_lat, cell_lon


def nearest_distance_m(lat, lon, grid, cell_lat, cell_lon, max_search_m=DEFAULT_MAX_SEARCH_M):
    """Nearest-neighbor distance within max_search_m, or None if nothing
    is found within that radius. This is a SEARCH-SPACE cap for
    performance, not a labeling threshold -- the evidence table always
    records the true nearest distance (up to this generous cap), leaving
    any semantic cutoff (e.g. "within 1km") to be applied later, on top
    of this column, when a labeling rule is actually evaluated -- keeping
    the evidence table itself free of any label-rule-specific threshold
    (see the leakage-audit note in DECISIONS.md).

    The neighbor-cell window is sized dynamically from max_search_m and
    the grid's cell size so no true neighbor within max_search_m can be
    missed (unlike a fixed-size window, which is only safe if cell size
    already exceeds the search radius)."""
    ci, cj = int(lat // cell_lat), int(lon // cell_lon)
    n_lat = math.ceil(max_search_m / (cell_lat / LAT_DEG_PER_M)) + 1
    n_lon = math.ceil(max_search_m / (cell_lon / LON_DEG_PER_M)) + 1
    best = None
    for di in range(-n_lat, n_lat + 1):
        for dj in range(-n_lon, n_lon + 1):
            for p in grid.get((ci + di, cj + dj), ()):
                d = haversine_m(lat, lon, p[0], p[1])
                if d <= max_search_m and (best is None or d < best):
                    best = d
    return best


# ------------------------------------------------------------------
# Persistent-cluster overlap (reuses the existing, unchanged 60-cluster
# geometry and the exact matching rule already approved in
# cross_year_recurrence_analysis.py -- distance to nearest cluster
# centroid <= that cluster's own extent_radius_m).
# ------------------------------------------------------------------

def load_cluster_overlap_lookup(integrated_evidence_path=INTEGRATED_EVIDENCE_CSV):
    with open(integrated_evidence_path, newline="") as f:
        rows = {int(r["cluster_id"]): r for r in csv.DictReader(f)}
    return rows


def find_overlapping_cluster(lat, lon, cluster_defs):
    """Nearest cluster centroid, if within that cluster's own
    extent_radius_m -- identical rule to
    cross_year_recurrence_analysis.match_historical_to_baseline, applied
    to an event centroid instead of a single detection."""
    best_cluster, best_dist = None, None
    for c in cluster_defs:
        d = _cy_haversine_m(lat, lon, c["centroid_lat"], c["centroid_lon"])
        if best_dist is None or d < best_dist:
            best_cluster, best_dist = c, d
    if best_cluster is not None and best_dist <= best_cluster["extent_radius_m"]:
        return best_cluster["cluster_id"], best_dist
    return None, None


# ------------------------------------------------------------------
# ESA WorldCover land cover (optional -- guarded rasterio import).
# ------------------------------------------------------------------

def tile_name_for(lat, lon):
    tlat = int(math.floor(lat / 3)) * 3
    tlon = int(math.floor(lon / 3)) * 3
    return f"N{tlat:02d}E{tlon:03d}"


def required_tiles(centroids):
    return sorted({tile_name_for(lat, lon) for lat, lon in centroids})


def ensure_worldcover_tiles(tiles, cache_dir=WORLDCOVER_CACHE_DIR, force=False):
    """Downloads each required 3x3-degree COG tile once (fetch-if-missing).
    A tile that returns HTTP 404 (e.g. open water, no land cover data)
    is recorded as unavailable rather than retried indefinitely."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    available = {}
    for tile in tiles:
        path = cache_dir / f"{tile}.tif"
        if path.exists() and not force:
            available[tile] = path
            continue
        url = f"{WORLDCOVER_BASE_URL}/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
        resp = requests.get(url, timeout=120)
        if resp.status_code == 404:
            available[tile] = None
            continue
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
        available[tile] = path
    return available


def sample_land_cover(centroids, cache_dir=WORLDCOVER_CACHE_DIR):
    """Returns a list of (code, class_name) parallel to `centroids`, or
    (None, None) for any point whose tile is unavailable. Requires
    rasterio -- raises ImportError with a clear message if not installed
    (this module's other joins do not require it)."""
    import rasterio  # optional dependency, only needed for this function

    tiles = required_tiles(centroids)
    tile_paths = ensure_worldcover_tiles(tiles, cache_dir=cache_dir)

    by_tile = defaultdict(list)
    for idx, (lat, lon) in enumerate(centroids):
        by_tile[tile_name_for(lat, lon)].append(idx)

    results = [(None, None)] * len(centroids)
    for tile, idxs in by_tile.items():
        path = tile_paths.get(tile)
        if path is None:
            continue
        coords = [(centroids[i][1], centroids[i][0]) for i in idxs]  # rasterio wants (lon, lat)
        with rasterio.open(path) as src:
            vals = list(src.sample(coords))
        for i, v in zip(idxs, vals):
            code = int(v[0])
            results[i] = (code, WORLDCOVER_CLASS_NAMES.get(code))
    return results


# ------------------------------------------------------------------
# Main join
# ------------------------------------------------------------------

def load_events(path=EVENTS_CSV):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_evidence_table(events, include_land_cover=True):
    industrial_pts = load_points_csv(OSM_INDUSTRIAL_POWER_CSV)
    flare_pts = load_points_csv(OSM_FLARE_CSV)
    kiln_pts = load_points_csv(OSM_KILN_CSV)
    gppd_pts = load_gppd_points(GPPD_THERMAL_CSV)

    grid_ind, cla_i, clo_i = build_point_grid(industrial_pts)
    grid_flare, cla_f, clo_f = build_point_grid(flare_pts)
    grid_kiln, cla_k, clo_k = build_point_grid(kiln_pts)
    grid_gppd, cla_g, clo_g = build_point_grid(gppd_pts)

    cluster_defs = load_cluster_definitions()
    integrated = load_cluster_overlap_lookup()

    centroids = [(float(e["centroid_lat"]), float(e["centroid_lon"])) for e in events]
    land_cover = sample_land_cover(centroids) if include_land_cover else [(None, None)] * len(events)

    rows = []
    for e, (lat, lon), (lc_code, lc_class) in zip(events, centroids, land_cover):
        cluster_id, cluster_dist = find_overlapping_cluster(lat, lon, cluster_defs)
        cluster_row = integrated.get(cluster_id) if cluster_id is not None else None

        rows.append({
            "event_id": e["event_id"],
            "start_date": e["start_date"], "end_date": e["end_date"],
            "duration_days": e["duration_days"],
            "centroid_lat": e["centroid_lat"], "centroid_lon": e["centroid_lon"],
            "spatial_extent_m": e["spatial_extent_m"],
            "detection_count": e["detection_count"],
            "mean_frp": e["mean_frp"], "max_frp": e["max_frp"],
            "night_fraction": e["night_fraction"], "status": e["status"],
            "nearest_osm_industrial_power_m": nearest_distance_m(lat, lon, grid_ind, cla_i, clo_i),
            "nearest_gppd_thermal_plant_m": nearest_distance_m(lat, lon, grid_gppd, cla_g, clo_g),
            "nearest_osm_flare_m": nearest_distance_m(lat, lon, grid_flare, cla_f, clo_f),
            "nearest_osm_kiln_m": nearest_distance_m(lat, lon, grid_kiln, cla_k, clo_k),
            "overlaps_cluster_id": cluster_id if cluster_id is not None else "",
            "overlaps_cluster_recurrence_strength": cluster_row["recurrence_strength"] if cluster_row else "",
            "overlaps_cluster_recurs_multiyear": cluster_row["recurs_across_multiple_years"] if cluster_row else "",
            "land_cover_code": lc_code if lc_code is not None else "",
            "land_cover_class": lc_class if lc_class is not None else "",
        })
    return rows


def write_csv(rows, path=OUTPUT_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("Ensuring external evidence sources are cached (fetch-if-missing)...")
    fetch_osm_industrial_power()
    fetch_osm_flare()
    fetch_osm_kiln()
    fetch_gppd_thermal_plants()

    events = load_events()
    print(f"Loaded {len(events)} events from {EVENTS_CSV} (read-only).")

    rows = build_evidence_table(events)
    write_csv(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")

    n_industrial = sum(1 for r in rows if r["nearest_osm_industrial_power_m"] not in (None, ""))
    n_gppd = sum(1 for r in rows if r["nearest_gppd_thermal_plant_m"] not in (None, ""))
    n_flare = sum(1 for r in rows if r["nearest_osm_flare_m"] not in (None, ""))
    n_kiln = sum(1 for r in rows if r["nearest_osm_kiln_m"] not in (None, ""))
    n_overlap = sum(1 for r in rows if r["overlaps_cluster_id"] != "")
    n_landcover = sum(1 for r in rows if r["land_cover_class"] != "")
    print(f"Events with industrial/power evidence within {DEFAULT_MAX_SEARCH_M/1000:.0f}km: {n_industrial}")
    print(f"Events with GPPD thermal-plant evidence within {DEFAULT_MAX_SEARCH_M/1000:.0f}km: {n_gppd}")
    print(f"Events with flare evidence within {DEFAULT_MAX_SEARCH_M/1000:.0f}km: {n_flare}")
    print(f"Events with kiln evidence within {DEFAULT_MAX_SEARCH_M/1000:.0f}km: {n_kiln}")
    print(f"Events overlapping a persistent cluster: {n_overlap}")
    print(f"Events with a sampled land-cover class: {n_landcover}")
    return rows


if __name__ == "__main__":
    main()
