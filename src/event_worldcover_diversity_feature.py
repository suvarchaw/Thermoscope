"""
`worldcover_diversity_300m` -- a new, additive, non-leaky event-level
feature: the number of distinct ESA WorldCover 2021 land-cover classes
present within a 300m radius of an event's centroid. Investigated
read-only two milestones ago and found leakage-safe (max |r|=0.395
against label-generating fields) and non-redundant with the existing
point-level `land_cover_class` (|r|<=0.250). This module implements
exactly that statistic -- same radius (300m), same metric (count of
distinct classes) -- as reviewed, tested code, and re-verifies the
leakage/redundancy findings against the real data before approving it.

DEFINITION -- unchanged from the read-only audit, not reinvented
-------------------------------------------------------------------------
For each event centroid, read a square window from the same cached ESA
WorldCover 10m tile(s) already used by event_evidence.py's
`sample_land_cover` (reused via `tile_name_for`/`required_tiles`/
`ensure_worldcover_tiles`, imported unchanged) -- a 61x61 pixel window
(~610m x 610m, covering the requested 300m radius in every direction)
centered on the centroid's pixel. `worldcover_diversity_300m` is the
number of DISTINCT nonzero land-cover class codes present in that
window. (0 = ESA WorldCover's own "no data" code and is excluded from
the count, exactly as `event_evidence.py` already treats it when
sampling the single-point class.)

EDGE CASES, DEFINED EXPLICITLY
-------------------------------------------------------------------------
An event whose tile is unavailable (the same rare case already
documented in event_evidence.py -- one tile, N18E066, does not exist
because it is open water) gets `worldcover_diversity_300m` BLANK (not 0
-- 0 classes would misleadingly claim "sampled and found nothing" rather
than "could not sample"), with `worldcover_diversity_computable=False`.
This mirrors the exact missing-value convention already established for
`frp_trend_slope`/`frp_trend_computable` -- consistent methodology, not
a new invention.

REUSE, NOT REIMPLEMENTATION: tile discovery/caching
(`tile_name_for`/`required_tiles`/`ensure_worldcover_tiles`,
`WORLDCOVER_CACHE_DIR`) is imported unchanged from event_evidence.py.
Only the windowed-diversity read itself is new (event_evidence.py's
`sample_land_cover` reads a single point, not a window).

Does NOT change event construction, silver-label rules, or any existing
output. rasterio was already an approved project dependency (added for
event_evidence.py) -- no new dependency is introduced here.
"""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from event_evidence import (
    tile_name_for, required_tiles, ensure_worldcover_tiles, WORLDCOVER_CACHE_DIR,
)

EVENTS_CSV = Path("data/processed/gujarat_thermal_events.csv")
SILVER_LABELS_CSV = Path("data/processed/gujarat_event_silver_labels.csv")
OUTPUT_CSV = Path("data/processed/event_worldcover_diversity_features.csv")

RADIUS_M = 300.0
PIXEL_SIZE_M = 10.0
WINDOW_HALF_PIXELS = int(RADIUS_M // PIXEL_SIZE_M)  # 30 -> 61x61 window, covers 300m in every direction

LEAKAGE_CHECK_FIELDS = ["mean_frp", "night_fraction", "duration_days", "detection_count", "max_frp"]
LEAKAGE_CORRELATION_THRESHOLD = 0.8

# Existing numeric feature set this new candidate must also be checked
# against for redundancy (item 2 of the milestone brief) -- includes
# centroid_lat/lon specifically, since a diversity measure that turned
# out to just be a monotonic function of location would defeat the
# purpose of adding it.
EXISTING_NUMERIC_FEATURES = [
    "centroid_lat", "centroid_lon", "spatial_extent_m",
    "frac_high_confidence", "frac_low_confidence", "mean_scan", "mean_track",
    "elongation_ratio", "time_of_day_std_minutes", "detections_per_day",
]


def compute_diversity_for_tile(path, centroids_with_idx):
    """centroids_with_idx: list of (original_index, lat, lon). Returns
    {original_index: n_distinct_classes_or_None}."""
    import rasterio
    results = {}
    with rasterio.open(path) as src:
        for idx, lat, lon in centroids_with_idx:
            row, col = src.index(lon, lat)
            window = ((row - WINDOW_HALF_PIXELS, row + WINDOW_HALF_PIXELS + 1),
                      (col - WINDOW_HALF_PIXELS, col + WINDOW_HALF_PIXELS + 1))
            try:
                data = src.read(1, window=window, boundless=True, fill_value=0)
            except Exception:
                results[idx] = None
                continue
            vals = set(data.flatten().tolist())
            vals.discard(0)  # ESA WorldCover "no data" code, excluded (see module docstring)
            results[idx] = len(vals) if vals else None
    return results


def build_feature_table():
    with open(EVENTS_CSV, newline="") as f:
        events = list(csv.DictReader(f))

    centroids = [(float(e["centroid_lat"]), float(e["centroid_lon"])) for e in events]
    tiles = required_tiles(centroids)
    tile_paths = ensure_worldcover_tiles(tiles, cache_dir=WORLDCOVER_CACHE_DIR)

    by_tile = defaultdict(list)
    for i, (lat, lon) in enumerate(centroids):
        by_tile[tile_name_for(lat, lon)].append((i, lat, lon))

    diversity = [None] * len(events)
    for tile, idx_list in by_tile.items():
        path = tile_paths.get(tile)
        if path is None:
            continue  # tile unavailable (e.g. open water) -- left None, see module docstring
        results = compute_diversity_for_tile(path, idx_list)
        for idx, val in results.items():
            diversity[idx] = val

    rows = []
    for e, div in zip(events, diversity):
        rows.append({
            "event_id": e["event_id"],
            "worldcover_diversity_300m": "" if div is None else div,
            "worldcover_diversity_computable": div is not None,
        })
    return rows


def write_csv(rows, path=OUTPUT_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["event_id", "worldcover_diversity_300m", "worldcover_diversity_computable"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Leakage + redundancy audit -- re-run against real data, not assumed.
# ---------------------------------------------------------------------

def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sx = sum((a - mx) ** 2 for a in xs) ** 0.5
    sy = sum((b - my) ** 2 for b in ys) ** 0.5
    return cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def run_leakage_audit(feature_rows, silver_rows):
    silver_by_id = {r["event_id"]: r for r in silver_rows}
    trained_classes = {"Industrial", "Gas_Flare", "Crop_Residue", "Forest_Wildfire"}

    computable_ids = [
        r["event_id"] for r in feature_rows
        if r["worldcover_diversity_computable"] and silver_by_id[r["event_id"]]["silver_label"] in trained_classes
    ]
    div_by_id = {r["event_id"]: float(r["worldcover_diversity_300m"]) for r in feature_rows
                 if r["event_id"] in computable_ids}
    divs = [div_by_id[eid] for eid in computable_ids]

    correlations = {}
    for field in LEAKAGE_CHECK_FIELDS:
        vals = [float(silver_by_id[eid][field]) for eid in computable_ids]
        correlations[field] = pearson(divs, vals)
    is_safe = all(abs(r) <= LEAKAGE_CORRELATION_THRESHOLD for r in correlations.values())
    return correlations, is_safe, len(computable_ids)


def run_redundancy_audit(feature_rows, existing_rows):
    """Checks correlation against every existing numeric feature,
    including centroid_lat/centroid_lon specifically -- the key question
    for THIS feature is whether it is secretly just a function of
    location."""
    existing_by_id = {r["event_id"]: r for r in existing_rows}
    computable_ids = [r["event_id"] for r in feature_rows if r["worldcover_diversity_computable"]]
    div_by_id = {r["event_id"]: float(r["worldcover_diversity_300m"]) for r in feature_rows
                 if r["event_id"] in computable_ids}
    divs = [div_by_id[eid] for eid in computable_ids]

    correlations = {}
    for field in EXISTING_NUMERIC_FEATURES:
        vals = [float(existing_by_id[eid][field]) for eid in computable_ids]
        correlations[field] = pearson(divs, vals)
    return correlations


def run_redundancy_with_land_cover_class(feature_rows, evidence_rows):
    evidence_by_id = {r["event_id"]: r for r in evidence_rows}
    computable_ids = [r["event_id"] for r in feature_rows if r["worldcover_diversity_computable"]]
    div_by_id = {r["event_id"]: float(r["worldcover_diversity_300m"]) for r in feature_rows
                 if r["event_id"] in computable_ids}
    divs = [div_by_id[eid] for eid in computable_ids]
    is_cropland = [1.0 if evidence_by_id[eid]["land_cover_class"] == "Cropland" else 0.0 for eid in computable_ids]
    is_treecover = [1.0 if evidence_by_id[eid]["land_cover_class"] == "Tree cover" else 0.0 for eid in computable_ids]
    return pearson(divs, is_cropland), pearson(divs, is_treecover)


def class_conditional_means(feature_rows, silver_rows):
    silver_by_id = {r["event_id"]: r for r in silver_rows}
    trained_classes = ["Industrial", "Gas_Flare", "Crop_Residue", "Forest_Wildfire"]
    by_class = defaultdict(list)
    for r in feature_rows:
        if not r["worldcover_diversity_computable"]:
            continue
        label = silver_by_id[r["event_id"]]["silver_label"]
        if label in trained_classes:
            by_class[label].append(float(r["worldcover_diversity_300m"]))
    return {cls: {"n": len(v), "mean": statistics.mean(v)} for cls, v in by_class.items() if v}


def main():
    print("Building worldcover_diversity_300m feature table "
          "(reuses event_evidence.py tile management unchanged)...\n")
    rows = build_feature_table()
    write_csv(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")

    n_computable = sum(1 for r in rows if r["worldcover_diversity_computable"])
    print(f"Computable: {n_computable} ({n_computable/len(rows)*100:.2f}%), "
          f"not computable: {len(rows)-n_computable}\n")

    with open(SILVER_LABELS_CSV, newline="") as f:
        silver = list(csv.DictReader(f))
    with open("data/processed/source_classifier_features_v2.csv", newline="") as f:
        existing = list(csv.DictReader(f))
    with open("data/processed/gujarat_event_evidence.csv", newline="") as f:
        evidence = list(csv.DictReader(f))

    print("=" * 70)
    print("LEAKAGE AUDIT (vs. label-generating fields)")
    print("=" * 70)
    correlations, is_safe, n_checked = run_leakage_audit(rows, silver)
    print(f"Checked on {n_checked} computable events in the 4 trained classes:")
    for field, r in correlations.items():
        flag = "OK" if abs(r) <= LEAKAGE_CORRELATION_THRESHOLD else "EXCEEDS THRESHOLD"
        print(f"  corr(worldcover_diversity_300m, {field}) = {r:.3f}  [{flag}]")
    print(f"DECISION: {'SAFE' if is_safe else 'NOT SAFE'}\n")

    print("=" * 70)
    print("REDUNDANCY AUDIT (vs. existing feature set, incl. centroid_lat/lon)")
    print("=" * 70)
    redundancy = run_redundancy_audit(rows, existing)
    for field, r in redundancy.items():
        flag = "REDUNDANT" if abs(r) > LEAKAGE_CORRELATION_THRESHOLD else "OK"
        print(f"  corr(worldcover_diversity_300m, {field}) = {r:.3f}  [{flag}]")

    r_crop, r_forest = run_redundancy_with_land_cover_class(rows, evidence)
    print(f"\n  corr vs is_cropland (point land_cover_class): {r_crop:.3f}")
    print(f"  corr vs is_treecover (point land_cover_class): {r_forest:.3f}\n")

    print("=" * 70)
    print("CLASS-CONDITIONAL MEANS (supporting evidence, not causal)")
    print("=" * 70)
    summary = class_conditional_means(rows, silver)
    for cls, s in summary.items():
        print(f"  {cls}: n={s['n']} mean={s['mean']:.2f}")

    return rows, correlations, is_safe, redundancy


if __name__ == "__main__":
    main()
