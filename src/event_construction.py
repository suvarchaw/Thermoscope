"""
Event-level thermal representation (additive second spatial unit).

BACKGROUND (see PROGRESS.md/DECISIONS.md for the full investigation this
implements): the existing 60 persistent clusters were produced by running
DBSCAN once on 2023 detections with min_samples=8, which by construction
keeps only spatially dense, recurring hotspots. A read-only investigation
found this structurally misses short-lived, spatially-shifting phenomena
(crop residue burning, wildfire) and validated a second, additive
representation -- spatiotemporal "events" -- using the real 2019-2025
Gujarat detection record. This module implements exactly the validated
definition, with no new methodology introduced beyond what the
investigation already tested:

    - spatial connection: EPS_SPACE_M = 750.0 (haversine), chosen to match
      the real observed VIIRS off-nadir pixel footprint growth (0.32-0.8km,
      confirmed from the scan/track fields), not an arbitrary number.
    - temporal connection: EPS_TIME_DAYS = 1 (SNPP's ~1-2 daily revisits
      mean a genuinely continuing burn should reappear the very next day).
    - two detections belong to the same event if they are within
      EPS_SPACE_M of each other AND within EPS_TIME_DAYS, with transitive
      closure (connected components) across the whole chain -- an event
      can span many days and drift spatially through repeated short hops.
    - MIN_DETECTIONS_PER_EVENT = 2: a minimal noise filter. 71.3% of raw
      connected components turned out to be single detections on a single
      day (investigation finding) -- almost certainly isolated noise, not
      an "event" any classifier should see as one object. This substitutes
      for a formal DBSCAN-style density/min_samples criterion at this
      stage (documented as a deferred refinement, not implemented here).

CAUSAL / ONLINE-SAFE BY CONSTRUCTION
-------------------------------------------------------------------------
Detections are processed in chronological (day) order. A detection on day
D is only ever compared against detections already seen -- today's points
processed so far, plus points from the preceding EPS_TIME_DAYS days still
held in a rolling buffer. No detection is ever compared against a future
day. This means a RETROSPECTIVE run over the complete historical file and
an ONLINE run applied incrementally, day-by-day, to arriving NRT data
produce IDENTICAL unions for every day already processed -- this is a
structural property of the algorithm (verified by
tests/test_event_construction.py::TestCausalOnlineSafety), not merely
claimed.

The one real difference between retrospective and online operation is
*when* an event can be declared finished: an event cannot be certified
CLOSED until EPS_TIME_DAYS have passed with no further detection linking
to it. See `compute_status`. Every event's `status` field is computed
relative to a `reference_date` (the most recent date considered "now" --
defaults to the latest acq_date present in the input, i.e. "today" for a
full retrospective run over the historical archive; for online/NRT use
this would be the actual current date).

SCOPE, EXPLICITLY
-------------------------------------------------------------------------
This module does NOT: modify the 60-cluster DBSCAN pipeline, cross-year
matching, or any existing processed output (all read-only where reused);
assign a source-class label; join external evidence (OSM/GPPD/land
cover -- deferred to a later milestone per the investigation report);
compute a risk score; or train a model.
"""

import csv
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

EARTH_RADIUS_M = 6_371_000.0

SOURCE_CSV = Path("data/processed/gujarat_multi_year_detections.csv")
OUTPUT_CSV = Path("data/processed/gujarat_thermal_events.csv")

EPS_SPACE_M = 750.0
EPS_TIME_DAYS = 1
MIN_DETECTIONS_PER_EVENT = 2

# Grid cell size for the spatial index equals EPS_SPACE_M; a 5x5 cell
# neighborhood (2 cells in each direction) is checked around every query
# point, which is a deliberately generous safety margin beyond the
# minimum 3x3 needed when cell size == eps (guards against any
# grid-alignment edge case) at negligible extra cost.
NEIGHBOR_CELL_RADIUS = 2

# Gujarat's approximate mean latitude, used only to convert the fixed
# EPS_SPACE_M into an approximate degree-sized grid cell for indexing.
# The actual link decision always uses real haversine distance -- this
# conversion only affects which cells are *candidates* to check, never
# whether two points are considered linked.
MEAN_LAT_DEG = 22.35
LAT_DEG_PER_M = 1.0 / 110_540.0
LON_DEG_PER_M = 1.0 / (111_320.0 * math.cos(math.radians(MEAN_LAT_DEG)))


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def load_detections(path=SOURCE_CSV):
    """Reads the existing, unmodified combined multi-year Gujarat
    detections file. Read-only -- does not touch the file or any upstream
    pipeline that produced it."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    detections = []
    for r in rows:
        detections.append({
            "lat": float(r["latitude"]),
            "lon": float(r["longitude"]),
            "date": date.fromisoformat(r["acq_date"]),
            "frp": float(r["frp"]),
            "daynight": r["daynight"],
        })
    detections.sort(key=lambda d: d["date"])
    return detections


def _cell_of(lat, lon, cell_lat, cell_lon):
    return (int(lat // cell_lat), int(lon // cell_lon))


def build_events(detections, eps_space_m=EPS_SPACE_M, eps_time_days=EPS_TIME_DAYS):
    """Spatiotemporal connected components over `detections` (each a dict
    with 'lat', 'lon', 'date'), processed in chronological order using a
    causal, backward-only rolling spatial index -- see module docstring
    for why this makes the construction online-safe.

    Returns a list of member-index-lists (indices into `detections`),
    UNFILTERED (includes size-1 components) -- callers apply
    `filter_events` for the noise-filtered, schema-ready result.
    """
    cell_lat = eps_space_m * LAT_DEG_PER_M
    cell_lon = eps_space_m * LON_DEG_PER_M

    uf = UnionFind(len(detections))
    grid = defaultdict(list)  # (cell_i, cell_j) -> list of point indices currently in the buffer
    buffer_by_day = defaultdict(list)  # date -> list of point indices added on that date

    by_day = defaultdict(list)
    for idx, d in enumerate(detections):
        by_day[d["date"]].append(idx)

    for day in sorted(by_day.keys()):
        cutoff = day - timedelta(days=eps_time_days)
        for old_day in [d for d in buffer_by_day if d < cutoff]:
            for idx in buffer_by_day[old_day]:
                c = _cell_of(detections[idx]["lat"], detections[idx]["lon"], cell_lat, cell_lon)
                grid[c].remove(idx)
                if not grid[c]:
                    del grid[c]
            del buffer_by_day[old_day]

        today_idxs = by_day[day]
        for idx in today_idxs:
            p = detections[idx]
            ci, cj = _cell_of(p["lat"], p["lon"], cell_lat, cell_lon)
            for di in range(-NEIGHBOR_CELL_RADIUS, NEIGHBOR_CELL_RADIUS + 1):
                for dj in range(-NEIGHBOR_CELL_RADIUS, NEIGHBOR_CELL_RADIUS + 1):
                    for cand in grid.get((ci + di, cj + dj), ()):
                        q = detections[cand]
                        if haversine_m(p["lat"], p["lon"], q["lat"], q["lon"]) <= eps_space_m:
                            uf.union(idx, cand)
            grid[(ci, cj)].append(idx)  # add AFTER linking, so same-day points still link to each other in order

        buffer_by_day[day] = today_idxs

    groups = defaultdict(list)
    for idx in range(len(detections)):
        groups[uf.find(idx)].append(idx)
    return list(groups.values())


def filter_events(events, min_detections=MIN_DETECTIONS_PER_EVENT):
    return [members for members in events if len(members) >= min_detections]


def compute_status(end_date, reference_date, eps_time_days=EPS_TIME_DAYS):
    """CLOSED if eps_time_days have passed since the event's last
    detection with no further linkage possible (i.e. reference_date is at
    least eps_time_days past end_date); otherwise PROVISIONAL -- a future
    detection could still extend this event. See module docstring."""
    if (reference_date - end_date).days >= eps_time_days:
        return "closed"
    return "provisional"


def summarize_event(event_id, members, detections, reference_date):
    member_pts = [detections[i] for i in members]
    dates = [p["date"] for p in member_pts]
    start, end = min(dates), max(dates)
    lats = [p["lat"] for p in member_pts]
    lons = [p["lon"] for p in member_pts]
    clat, clon = sum(lats) / len(lats), sum(lons) / len(lons)
    extent_m = max((haversine_m(clat, clon, p["lat"], p["lon"]) for p in member_pts), default=0.0)
    night_count = sum(1 for p in member_pts if p["daynight"] == "N")

    return {
        "event_id": event_id,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "duration_days": (end - start).days + 1,
        "centroid_lat": clat,
        "centroid_lon": clon,
        "spatial_extent_m": round(extent_m, 1),
        "detection_count": len(member_pts),
        "mean_frp": round(sum(p["frp"] for p in member_pts) / len(member_pts), 3),
        "max_frp": max(p["frp"] for p in member_pts),
        "night_fraction": round(night_count / len(member_pts), 4),
        "status": compute_status(end, reference_date),
    }


def build_event_table(detections, reference_date=None):
    """Full pipeline: build raw connected components, filter to
    MIN_DETECTIONS_PER_EVENT, summarize, and assign deterministic
    event_ids. event_ids are assigned only after sorting by
    (start_date, end_date, centroid_lat, centroid_lon) so the result does
    not depend on Union-Find's internal (non-deterministic-order) group
    iteration -- required for reproducibility."""
    if reference_date is None:
        reference_date = max(d["date"] for d in detections)

    raw_events = build_events(detections)
    kept = filter_events(raw_events)

    unlabeled = []
    for members in kept:
        member_pts = [detections[i] for i in members]
        dates = [p["date"] for p in member_pts]
        start, end = min(dates), max(dates)
        lats = [p["lat"] for p in member_pts]
        lons = [p["lon"] for p in member_pts]
        clat, clon = sum(lats) / len(lats), sum(lons) / len(lons)
        unlabeled.append((start, end, clat, clon, members))
    unlabeled.sort(key=lambda t: (t[0], t[1], t[2], t[3]))

    rows = []
    for seq, (_, _, _, _, members) in enumerate(unlabeled):
        event_id = f"EVT{seq:06d}"
        rows.append(summarize_event(event_id, members, detections, reference_date))
    return rows, len(raw_events)


def write_csv(rows, path=OUTPUT_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    detections = load_detections()
    print(f"Loaded {len(detections)} detections from {SOURCE_CSV} (read-only).")
    print(f"Date range: {detections[0]['date']} to {detections[-1]['date']}")
    print(f"Parameters: eps_space_m={EPS_SPACE_M}, eps_time_days={EPS_TIME_DAYS}, "
          f"min_detections_per_event={MIN_DETECTIONS_PER_EVENT}")

    rows, n_raw_events = build_event_table(detections)
    n_singleton = n_raw_events - len(rows)
    print(f"\nRaw connected components: {n_raw_events}")
    print(f"Singleton (1-detection) components excluded as noise: {n_singleton} "
          f"({n_singleton/n_raw_events*100:.1f}%)")
    print(f"Events retained (detection_count >= {MIN_DETECTIONS_PER_EVENT}): {len(rows)}")

    n_closed = sum(1 for r in rows if r["status"] == "closed")
    n_provisional = len(rows) - n_closed
    print(f"Status: {n_closed} closed, {n_provisional} provisional "
          f"(relative to reference_date={max(d['date'] for d in detections)})")

    write_csv(rows)
    print(f"\nWrote {len(rows)} events to {OUTPUT_CSV}")
    return rows


if __name__ == "__main__":
    main()
