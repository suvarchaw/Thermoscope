"""
Core spatial clustering logic for the ThermoScope Persistent Thermal Source
Detection Engine (spatial grouping stage only).

Groups Gujarat FIRMS detections into spatial clusters using DBSCAN with a
haversine (geodesic) distance metric, so that cluster shape is not
constrained to a fixed grid and irregular/elongated real-world zones are
not artificially fragmented.

This module intentionally does NOT:
  - use the FIRMS `type` field as a label or feature,
  - classify clusters by persistence (that is a separate, later decision
    based on the temporal stats computed here, not made in this module),
  - assign any source-type label.

Distance handling: sklearn's DBSCAN with metric="haversine" expects
coordinates in radians (as [lat, lon]) and produces distances as radians
of arc on a unit sphere. `eps` is therefore converted from meters to
radians using the mean Earth radius before being passed in, and multiplied
back when reporting.
"""

import math
import statistics
from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN

from spatial_recurrence import read_gujarat_detections, RAW_CSV  # noqa: F401 (RAW_CSV re-exported for callers)

EARTH_RADIUS_M = 6_371_000.0

HAZIRA_BOX_LAT = (21.10, 21.11)
HAZIRA_BOX_LON = (72.63, 72.65)


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def load_points(rows):
    """Return an (N, 2) array of [lat, lon] in radians, for sklearn's haversine metric."""
    return np.radians(np.array([[float(r["latitude"]), float(r["longitude"])] for r in rows]))


def run_dbscan(rows, eps_m, min_samples):
    """Run DBSCAN over detection rows using a geodesic distance metric.

    Returns a numpy array of cluster labels (one per row), where -1 means noise.
    Note: sklearn's min_samples counts the point itself, i.e. a core point
    needs min_samples - 1 *other* neighbors within eps.
    """
    coords_rad = load_points(rows)
    eps_rad = eps_m / EARTH_RADIUS_M
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine", algorithm="ball_tree")
    return db.fit_predict(coords_rad)


def _date_to_ordinal(date_str):
    import datetime
    y, m, d = (int(p) for p in date_str.split("-"))
    return datetime.date(y, m, d).toordinal()


def compute_cluster_stats(rows, labels):
    """Compute per-cluster descriptive statistics for all non-noise clusters.

    Returns a list of dicts, one per cluster, sorted by cluster_id.
    Does not classify clusters by persistence and does not use the `type` field.
    """
    members = defaultdict(list)
    for row, label in zip(rows, labels):
        if label == -1:
            continue
        members[label].append(row)

    results = []
    for cluster_id, cluster_rows in members.items():
        lats = [float(r["latitude"]) for r in cluster_rows]
        lons = [float(r["longitude"]) for r in cluster_rows]
        dates = sorted({r["acq_date"] for r in cluster_rows})
        frp_values = []
        day_count = 0
        night_count = 0
        for r in cluster_rows:
            try:
                frp_values.append(float(r.get("frp", "")))
            except ValueError:
                pass
            if r.get("daynight") == "D":
                day_count += 1
            elif r.get("daynight") == "N":
                night_count += 1

        centroid_lat = statistics.mean(lats)
        centroid_lon = statistics.mean(lons)
        radius_m = max(
            haversine_m(centroid_lat, centroid_lon, lat, lon)
            for lat, lon in zip(lats, lons)
        )

        first_date = dates[0]
        last_date = dates[-1]
        active_span_days = _date_to_ordinal(last_date) - _date_to_ordinal(first_date)

        results.append({
            "cluster_id": int(cluster_id),
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "detection_count": len(cluster_rows),
            "unique_dates": len(dates),
            "first_date": first_date,
            "last_date": last_date,
            "active_span_days": active_span_days,
            "mean_frp": statistics.mean(frp_values) if frp_values else None,
            "max_frp": max(frp_values) if frp_values else None,
            "day_count": day_count,
            "night_count": night_count,
            "bbox_min_lat": min(lats),
            "bbox_max_lat": max(lats),
            "bbox_min_lon": min(lons),
            "bbox_max_lon": max(lons),
            "extent_radius_m": radius_m,
        })

    results.sort(key=lambda r: r["cluster_id"])
    return results


def hazira_cluster_ids(rows, labels):
    """Return the set of non-noise cluster ids that have at least one member
    detection inside the previously-identified Hazira observation box."""
    ids = set()
    for row, label in zip(rows, labels):
        if label == -1:
            continue
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        if HAZIRA_BOX_LAT[0] <= lat <= HAZIRA_BOX_LAT[1] and HAZIRA_BOX_LON[0] <= lon <= HAZIRA_BOX_LON[1]:
            ids.add(int(label))
    return ids
