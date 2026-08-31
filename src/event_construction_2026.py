"""
2026 event construction (additive, inference-only; NOT part of the
historical event table).

Runs the identical, already-approved event representation
(src/event_construction.py: 750m spatial threshold, <=1 day temporal gap,
transitive connected components, detection_count>=2, provisional/closed
status) against 2026-only Gujarat detections. No event-construction
parameter or algorithm is changed here -- build_events/filter_events/
summarize_event/compute_status are imported from event_construction.py
and called UNCHANGED; this module only supplies a different input
(2026-only detections instead of the full 2019-2025 file) and a
different, distinctly-namespaced event_id sequence.

WHY A SEPARATE MODULE, NOT gujarat_multi_year_detections.csv +
event_construction.py DIRECTLY
-------------------------------------------------------------------------
The historical event table (gujarat_thermal_events.csv, EVT000000..
EVT020408) was built from one pass over the full 2019-2025 combined
detection set; every downstream table (evidence, silver labels, features,
the trained classifier's train/test split) is keyed to that exact
event_id sequence. Re-running event construction over 2019-2025+2026
combined would renumber/regroup EVERY historical event and invalidate all
of those downstream tables and the already-trained model -- exactly what
this milestone prohibits ("do not modify existing datasets"). Instead,
this module runs the same algorithm on a disjoint input (2026 detections
only) and writes an entirely separate, distinctly-namespaced output
(event_id prefix EVT2026_, separate CSV) -- additive, and safe to
delete/rebuild without touching anything historical.

KNOWN LIMITATION, DISCLOSED NOT SILENTLY WORKED AROUND
-------------------------------------------------------------------------
Because this module's input is 2026 detections ONLY (no cross-year
buffer), a real physical event whose detections span 2025-12-31 into
2026-01-01+ would be split at the year boundary: the 2025 portion already
exists as the tail of some EVT0000xx in the historical table, and the
2026 portion here would start a new EVT2026_xxxxxx rather than being
linked to it as a continuation. event_construction.py's own docstring
describes the algorithm as "online-safe" specifically via a rolling
buffer of the preceding eps_time_days of already-processed detections --
deliberately NOT reproduced here, because doing so would require reusing
detections whose membership is already baked into the persisted
historical event table, raising exactly the double-counting risk (one
physical detection contributing to two different events' statistics)
this project's "never silently miscount" discipline exists to avoid.
This is judged an acceptable, disclosed edge case (see PROGRESS.md) for a
small minority of possible December-into-January events, not a
correctness bug in the common case.
"""

import csv
from datetime import date
from pathlib import Path

import event_construction as ec
from multi_year_gujarat_processing import load_historical_year

RAW_2026_CSV = Path("data/raw/firms_gujarat_2026.csv")
OUTPUT_EVENTS_CSV = Path("data/processed/gujarat_thermal_events_2026.csv")

EVENT_ID_PREFIX = "EVT2026_"

EVENT_FIELDNAMES = [
    "event_id", "start_date", "end_date", "duration_days",
    "centroid_lat", "centroid_lon", "spatial_extent_m",
    "detection_count", "mean_frp", "max_frp", "night_fraction", "status",
]


def load_2026_detections(path=RAW_2026_CSV):
    """Reuses multi_year_gujarat_processing.load_historical_year UNCHANGED
    (same Gujarat-bounds re-filter + year-tagging every other ingested
    year already gets), then reshapes into the same rich detection-dict
    schema event_behavior_features.load_rich_detections uses -- needed
    because this pipeline also computes the 7 locked behavior features
    (not just event geometry) for the inference step."""
    rows = load_historical_year(path)
    detections = []
    for r in rows:
        detections.append({
            "lat": float(r["latitude"]), "lon": float(r["longitude"]),
            "date": date.fromisoformat(r["acq_date"]), "frp": float(r["frp"]),
            "daynight": r["daynight"], "confidence": r["confidence"],
            "brightness": float(r["brightness"]), "bright_t31": float(r["bright_t31"]),
            "scan": float(r["scan"]), "track": float(r["track"]), "acq_time": r["acq_time"],
        })
    detections.sort(key=lambda d: d["date"])
    return detections


def build_2026_events(detections, reference_date=None):
    """Identical algorithm to event_construction.build_event_table, called
    directly on `detections` (rather than reading the historical
    SOURCE_CSV) so it can be pointed at the 2026-only detection set.
    Returns (rows, members_by_row, n_raw_events) where members_by_row[i]
    is the list of `detections` indices composing rows[i] -- exposed
    (unlike build_event_table) because the feature/evidence/inference
    steps need the member detections, not just the summarized row."""
    if not detections:
        return [], [], 0
    if reference_date is None:
        reference_date = max(d["date"] for d in detections)

    raw_events = ec.build_events(detections)
    kept = ec.filter_events(raw_events)

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

    rows, members_by_row = [], []
    for seq, (_, _, _, _, members) in enumerate(unlabeled):
        event_id = f"{EVENT_ID_PREFIX}{seq:06d}"
        rows.append(ec.summarize_event(event_id, members, detections, reference_date))
        members_by_row.append(members)
    return rows, members_by_row, len(raw_events)


def write_csv(rows, path=OUTPUT_EVENTS_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else EVENT_FIELDNAMES
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    detections = load_2026_detections()
    print(f"Loaded {len(detections)} 2026 detections from {RAW_2026_CSV} (read-only).")
    if detections:
        print(f"Date range: {detections[0]['date']} to {detections[-1]['date']}")
    print(f"Parameters (unchanged from event_construction.py): "
          f"eps_space_m={ec.EPS_SPACE_M}, eps_time_days={ec.EPS_TIME_DAYS}, "
          f"min_detections_per_event={ec.MIN_DETECTIONS_PER_EVENT}")

    rows, members_by_row, n_raw_events = build_2026_events(detections)
    n_singleton = n_raw_events - len(rows)
    print(f"\nRaw connected components: {n_raw_events}")
    print(f"Singleton (1-detection) components excluded as noise: {n_singleton}")
    print(f"2026 events retained (detection_count >= {ec.MIN_DETECTIONS_PER_EVENT}): {len(rows)}")

    n_closed = sum(1 for r in rows if r["status"] == "closed")
    n_provisional = len(rows) - n_closed
    print(f"Status: {n_closed} closed, {n_provisional} provisional")

    write_csv(rows)
    print(f"\nWrote {len(rows)} events to {OUTPUT_EVENTS_CSV}")
    return rows, members_by_row, detections


if __name__ == "__main__":
    main()
