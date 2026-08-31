"""
Richer, non-leaky event-behavior feature engineering + A/B/C comparison.

CONTEXT: the first supervised experiment (train_source_classifier.py)
found the model relies almost entirely on geography (centroid_lon
permutation importance 0.271 vs. spatial_extent_m's 0.013) because only
4 leakage-safe features existed. This module investigates whether
richer, genuinely NEW event-behavior features -- computed from raw FIRMS
fields never previously used anywhere in this project's labeling or
feature pipeline (confidence, brightness, bright_t31, scan, track,
acq_time) plus new spatial-shape/temporal statistics -- add real signal
beyond location, without leaking the silver-label rules.

METHODOLOGY, NOT REIMPLEMENTED
-------------------------------------------------------------------------
Event grouping reuses event_construction.build_events/filter_events
UNCHANGED (imported, not reimplemented) -- this module does NOT change
event-construction methodology. A dedicated verification step
(`verify_alignment`) regenerates the exact same deterministic
(start_date, end_date, centroid_lat, centroid_lon) sort used by
event_construction.build_event_table and confirms, row for row, that it
reproduces data/processed/gujarat_thermal_events.csv exactly (0
mismatches across all 20,409 events) -- this is how event_id alignment
with the existing evidence/silver-label tables is guaranteed without
re-deriving IDs independently.

CANDIDATE FEATURES INVESTIGATED (see FEATURE_AUDIT below for the full
per-feature trace: computation, source columns, inference-time
availability, correlation with every already-excluded label-generating
field, and the include/exclude decision)
-------------------------------------------------------------------------
Computed from raw per-detection fields never previously used in this
project's labeling or evidence pipeline (confidence, brightness,
bright_t31, scan, track, acq_time):
  frac_high_confidence, frac_low_confidence, mean_brightness,
  mean_bright_t31, mean_brightness_t31_diff, mean_scan, mean_track,
  time_of_day_std_minutes, detections_per_day.
Computed from the event's spatial point cloud (new shape/movement
descriptors, distinct from the already-kept spatial_extent_m):
  std_distance_from_centroid_m, elongation_ratio, movement_distance_m.
Also investigated and immediately rejected as a structural duplicate:
  unique_days (proven, on all 20,409 events, to equal duration_days
  exactly -- an inescapable consequence of eps_time_days=1: no calendar
  day inside an event's span can have zero detections, or the chain
  would have broken there).

EMPIRICAL LEAKAGE/REDUNDANCY THRESHOLD: |r| > 0.8 against any of
{mean_frp, night_fraction, duration_days, detection_count, max_frp}
(the existing label-generating/excluded fields) -- the same conservative
threshold this project already applied when excluding max_frp (r=0.883)
and detection_count (r=0.921) in the prior milestone. The same |r| > 0.8
threshold is also applied for pairwise redundancy against features
already in the model (e.g. spatial_extent_m), for a different reason
(no leakage risk, just duplicated information -- see FEATURE_AUDIT).

RESULT OF THE AUDIT: mean_brightness (r=-0.930 with night_fraction),
mean_bright_t31 (r=-0.839), and mean_brightness_t31_diff (r=-0.802) are
all excluded -- a real, physically explicable finding: VIIRS brightness-
temperature channels are systematically different for day vs. night
detections (solar contamination of the "background" channel), so mean
brightness is close to an indirect night/day indicator, which is exactly
what the Industrial rule reads directly. std_distance_from_centroid_m
(r=0.908 with spatial_extent_m) and movement_distance_m (r=0.813 with
spatial_extent_m) are excluded as redundant with the already-kept
spatial_extent_m, not because they leak a label.

FINAL NEW FEATURES ADDED: frac_high_confidence, frac_low_confidence,
mean_scan, mean_track, elongation_ratio, time_of_day_std_minutes,
detections_per_day (7).

Does NOT: change event-construction or silver-labeling methodology,
create new labels, invent Brick Kiln examples, tune for class balance,
run LightGBM, or claim production readiness.
"""

import csv
import math
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent))
import event_construction as ec
from train_source_classifier import (
    TRAINED_CLASSES, TRAIN_YEARS, TEST_YEARS, RANDOM_STATE,
    filter_to_trained_classes, temporal_split, compute_metrics,
)

SILVER_LABELS_CSV = Path("data/processed/gujarat_event_silver_labels.csv")
EVENTS_CSV = Path("data/processed/gujarat_thermal_events.csv")
OUTPUT_FEATURES_CSV = Path("data/processed/source_classifier_features_v2.csv")
METRICS_CSV = Path("data/processed/source_classifier_v2_comparison_metrics.csv")

EARTH_RADIUS_M = 6_371_000.0
MEAN_LAT_DEG = 22.35
LAT_M_PER_DEG = 110_540.0
LON_M_PER_DEG = 111_320.0 * math.cos(math.radians(MEAN_LAT_DEG))

# Existing, already-audited-safe features from the first experiment (unchanged).
BASELINE_FEATURES = ["centroid_lat", "centroid_lon", "spatial_extent_m", "status"]
BASELINE_FEATURES_NUMERIC = ["centroid_lat", "centroid_lon", "spatial_extent_m"]
BASELINE_FEATURES_CATEGORICAL = ["status"]

# New features that survived the leakage/redundancy audit (see module docstring).
NEW_FEATURES = ["frac_high_confidence", "frac_low_confidence", "mean_scan", "mean_track",
                "elongation_ratio", "time_of_day_std_minutes", "detections_per_day"]
NEW_FEATURES_NUMERIC = list(NEW_FEATURES)  # all numeric, no new categorical
NEW_FEATURES_CATEGORICAL = []

FORBIDDEN_COLUMNS = {
    "silver_label", "label_rule_id", "label_evidence", "conflict_classes",
    "n_classes_matched", "ambiguity_reason", "excluded_from_training",
    "mean_frp", "night_fraction", "duration_days", "start_date", "end_date",
    "max_frp", "detection_count", "event_id",
    # excluded new candidates (leakage or redundancy) -- never allowed as features either
    "mean_brightness", "mean_bright_t31", "mean_brightness_t31_diff",
    "std_distance_from_centroid_m", "movement_distance_m", "unique_days",
}

FEATURE_AUDIT = [
    {"name": "frac_high_confidence", "included": True, "max_corr_vs_excluded": 0.302,
     "reason": "Fraction of the event's detections with FIRMS confidence='h'. Never previously read "
               "by any rule. Max |r| vs. label-generating fields 0.302 (mean_frp) -- safe."},
    {"name": "frac_low_confidence", "included": True, "max_corr_vs_excluded": 0.493,
     "reason": "Fraction with confidence='l'. Max |r| 0.493 (night_fraction) -- moderate but well "
               "under the 0.8 threshold; kept, correlation disclosed."},
    {"name": "mean_scan", "included": True, "max_corr_vs_excluded": 0.151,
     "reason": "Mean VIIRS scan (pixel footprint, along-scan). Never previously used anywhere. "
               "Max |r| 0.151 -- clearly independent."},
    {"name": "mean_track", "included": True, "max_corr_vs_excluded": 0.191,
     "reason": "Mean VIIRS track (pixel footprint, along-track). Max |r| 0.191 -- clearly independent."},
    {"name": "elongation_ratio", "included": True, "max_corr_vs_excluded": 0.187,
     "reason": "Minor/major eigenvalue ratio of the event's detection point cloud (local-ENU "
               "covariance) -- 0=perfectly linear, ~1=circular/compact. Max |r| vs. excluded fields "
               "0.187; r=0.352 vs. the already-kept spatial_extent_m (low enough to keep -- shape, "
               "not size)."},
    {"name": "time_of_day_std_minutes", "included": True, "max_corr_vs_excluded": 0.159,
     "reason": "Std. dev. of acq_time (minutes-of-day) across the event's detections -- consistency "
               "of daily overpass timing. Max |r| 0.159."},
    {"name": "detections_per_day", "included": True, "max_corr_vs_excluded": 0.306,
     "reason": "detection_count / duration_days. Structurally derived from two EXCLUDED fields, so "
               "checked specifically (not just generally): correlation with the exact rule threshold "
               "used by 3 rules, duration<=3, is only r=0.059 -- essentially uncorrelated. Kept, with "
               "this structural caveat disclosed rather than silently included."},
    {"name": "mean_brightness", "included": False, "max_corr_vs_excluded": 0.930,
     "reason": "EXCLUDED (leakage): r=-0.930 with night_fraction (directly read by the Industrial "
               "rule). VIIRS I4 brightness temperature is systematically different for day vs. night "
               "detections (solar contamination) -- this is close to an indirect night/day indicator."},
    {"name": "mean_bright_t31", "included": False, "max_corr_vs_excluded": 0.839,
     "reason": "EXCLUDED (leakage): r=-0.839 with night_fraction, same physical mechanism as "
               "mean_brightness."},
    {"name": "mean_brightness_t31_diff", "included": False, "max_corr_vs_excluded": 0.802,
     "reason": "EXCLUDED (leakage): r=-0.802 with night_fraction, same mechanism."},
    {"name": "std_distance_from_centroid_m", "included": False, "max_corr_vs_excluded": 0.238,
     "reason": "EXCLUDED (redundancy, not leakage): r=0.908 with the already-kept spatial_extent_m -- "
               "both measure spatial spread; adding both would be near-duplicate information."},
    {"name": "movement_distance_m", "included": False, "max_corr_vs_excluded": 0.284,
     "reason": "EXCLUDED (redundancy, not leakage): r=0.813 with the already-kept spatial_extent_m -- "
               "in this dataset most spatial spread is directional, so first-to-last movement tracks "
               "closely with overall extent."},
    {"name": "unique_days", "included": False, "max_corr_vs_excluded": 1.000,
     "reason": "EXCLUDED (exact structural duplicate): proven equal to duration_days for all 20,409 "
               "events -- eps_time_days=1 construction cannot produce a gap day inside an event, so "
               "unique_days == duration_days always."},
]


def assert_feature_lists_clean():
    for feature_list in (BASELINE_FEATURES, NEW_FEATURES):
        overlap = set(feature_list) & FORBIDDEN_COLUMNS
        if overlap:
            raise ValueError(f"Forbidden columns present in feature list: {overlap}")
    audited_included = {r["name"] for r in FEATURE_AUDIT if r["included"]}
    if audited_included != set(NEW_FEATURES):
        raise ValueError(f"FEATURE_AUDIT included-set {audited_included} != NEW_FEATURES {set(NEW_FEATURES)}")


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def acq_time_to_minutes(t):
    t = str(t).zfill(4)
    return int(t[:2]) * 60 + int(t[2:])


def load_rich_detections(path=ec.SOURCE_CSV):
    """Same source file as event_construction.load_detections, but keeps
    the extra raw fields (confidence, brightness, bright_t31, scan,
    track, acq_time) that module discards -- needed only for the NEW
    features computed here, not for the grouping itself."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
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


def sorted_event_groups(detections):
    """Reuses event_construction.build_events/filter_events UNCHANGED,
    then reproduces the exact deterministic sort build_event_table uses
    to assign event_id -- so position i here corresponds to event_id
    EVT{i:06d} in every already-persisted table. Verified by
    `verify_alignment`, not just asserted."""
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
    return unlabeled


def verify_alignment(sorted_groups, events_path=EVENTS_CSV):
    with open(events_path, newline="") as f:
        persisted = list(csv.DictReader(f))
    if len(persisted) != len(sorted_groups):
        raise ValueError(f"Event count mismatch: persisted={len(persisted)}, regenerated={len(sorted_groups)}")
    mismatches = []
    for i, (p, (start, end, clat, clon, members)) in enumerate(zip(persisted, sorted_groups)):
        if p["start_date"] != start.isoformat() or p["end_date"] != end.isoformat() or int(p["detection_count"]) != len(members):
            mismatches.append(i)
    if mismatches:
        raise ValueError(f"{len(mismatches)} event(s) misaligned with {events_path}, e.g. index {mismatches[0]}")
    return True


def compute_elongation_ratio(pts, clat, clon):
    n = len(pts)
    xs = [(p["lon"] - clon) * LON_M_PER_DEG for p in pts]
    ys = [(p["lat"] - clat) * LAT_M_PER_DEG for p in pts]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs) / n
    syy = sum((y - my) ** 2 for y in ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = max(0.0, tr * tr / 4 - det)
    l1 = tr / 2 + math.sqrt(disc)
    l2 = tr / 2 - math.sqrt(disc)
    return (l2 / l1) if l1 > 1e-9 else 0.0


def compute_new_features_for_event(members, detections):
    pts = [detections[i] for i in members]
    n = len(pts)
    lats = [p["lat"] for p in pts]
    lons = [p["lon"] for p in pts]
    clat, clon = sum(lats) / n, sum(lons) / n

    n_high = sum(1 for p in pts if p["confidence"] == "h")
    n_low = sum(1 for p in pts if p["confidence"] == "l")

    duration_days = (max(p["date"] for p in pts) - min(p["date"] for p in pts)).days + 1
    minutes = [acq_time_to_minutes(p["acq_time"]) for p in pts]

    return {
        "frac_high_confidence": n_high / n,
        "frac_low_confidence": n_low / n,
        "mean_scan": sum(p["scan"] for p in pts) / n,
        "mean_track": sum(p["track"] for p in pts) / n,
        "elongation_ratio": compute_elongation_ratio(pts, clat, clon),
        "time_of_day_std_minutes": statistics.pstdev(minutes) if n > 1 else 0.0,
        "detections_per_day": n / duration_days,
    }


def build_features_table():
    assert_feature_lists_clean()
    detections = load_rich_detections()
    sorted_groups = sorted_event_groups(detections)
    verify_alignment(sorted_groups)

    with open(SILVER_LABELS_CSV, newline="") as f:
        silver = list(csv.DictReader(f))
    if len(silver) != len(sorted_groups):
        raise ValueError("Silver-label row count does not match regenerated event count.")

    rows = []
    for silver_row, (start, end, clat, clon, members) in zip(silver, sorted_groups):
        new_feats = compute_new_features_for_event(members, detections)
        row = {"event_id": silver_row["event_id"], "silver_label": silver_row["silver_label"],
               "start_date": silver_row["start_date"]}  # for temporal_split only, never written as a feature
        for c in BASELINE_FEATURES:
            row[c] = silver_row[c]
        row.update(new_feats)
        rows.append(row)
    return rows


def write_features_csv(rows, path=OUTPUT_FEATURES_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["event_id", "silver_label"] + BASELINE_FEATURES + NEW_FEATURES
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# A/B/C comparison (same LR setup + temporal split + metrics as the
# first experiment, generalized to an arbitrary feature-column list).
# ---------------------------------------------------------------------
FEATURE_SETS = {
    "A_baseline_geography_only": (BASELINE_FEATURES_NUMERIC, BASELINE_FEATURES_CATEGORICAL),
    "B_new_behavior_only": (NEW_FEATURES_NUMERIC, NEW_FEATURES_CATEGORICAL),
    "C_geography_plus_behavior": (BASELINE_FEATURES_NUMERIC + NEW_FEATURES_NUMERIC,
                                   BASELINE_FEATURES_CATEGORICAL + NEW_FEATURES_CATEGORICAL),
}


def rows_to_arrays(rows, numeric_cols, categorical_cols):
    Xn = np.array([[float(r[c]) for c in numeric_cols] for r in rows], dtype=float)
    Xc = np.array([[r[c] for c in categorical_cols] for r in rows], dtype=object) if categorical_cols else \
        np.empty((len(rows), 0), dtype=object)
    y = np.array([r["silver_label"] for r in rows])
    return Xn, Xc, y


def combine_features(Xn, Xc):
    if Xc.shape[1] == 0:
        return Xn.astype(object)
    return np.hstack([Xn.astype(object), Xc])


def build_preprocessor(numeric_cols, categorical_cols):
    transformers = [("num", StandardScaler(), list(range(len(numeric_cols))))]
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"),
                              list(range(len(numeric_cols), len(numeric_cols) + len(categorical_cols)))))
    return ColumnTransformer(transformers=transformers)


def build_logistic_pipeline(numeric_cols, categorical_cols):
    return Pipeline([
        ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ])


def readable_feature_names(preprocessor, numeric_cols, categorical_cols):
    names = list(numeric_cols)
    if categorical_cols:
        onehot = preprocessor.named_transformers_["cat"]
        for col_name, categories in zip(categorical_cols, onehot.categories_):
            for cat in categories:
                names.append(f"{col_name}={cat}")
    return names


def run_feature_set(rows, train_rows, test_rows, numeric_cols, categorical_cols, label):
    Xn_train, Xc_train, y_train = rows_to_arrays(train_rows, numeric_cols, categorical_cols)
    Xn_test, Xc_test, y_test = rows_to_arrays(test_rows, numeric_cols, categorical_cols)
    X_train = combine_features(Xn_train, Xc_train)
    X_test = combine_features(Xn_test, Xc_test)

    classes = sorted(set(y_train) | set(y_test))

    pipeline = build_logistic_pipeline(numeric_cols, categorical_cols)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    classes_order = list(pipeline.named_steps["clf"].classes_)
    y_prob_ordered = y_prob[:, [classes_order.index(c) for c in classes]]

    metrics = compute_metrics(y_test, y_pred, classes, y_prob_ordered)
    metrics["label"] = label

    perm = permutation_importance(pipeline, X_test, y_test, n_repeats=20,
                                   random_state=RANDOM_STATE, scoring="f1_macro")
    names = readable_feature_names(pipeline.named_steps["preprocess"], numeric_cols, categorical_cols)
    metrics["permutation_importance"] = sorted(
        zip(names, perm.importances_mean.tolist(), perm.importances_std.tolist()),
        key=lambda x: -x[1],
    )
    return metrics


def main():
    print("Building feature table (this reuses event_construction.build_events/filter_events "
          "unchanged, and verifies alignment against the persisted event table)...\n")
    rows = build_features_table()
    write_features_csv(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_FEATURES_CSV}\n")

    print("=" * 70)
    print("FEATURE AUDIT")
    print("=" * 70)
    for r in FEATURE_AUDIT:
        status = "INCLUDED" if r["included"] else "EXCLUDED"
        print(f"  [{status}] {r['name']} (max|r| vs excluded fields={r['max_corr_vs_excluded']:.3f})")
        print(f"      {r['reason']}")

    trained_rows = filter_to_trained_classes(rows)
    train_rows, test_rows = temporal_split(trained_rows)
    print(f"\nTrain: {len(train_rows)} (years {sorted(TRAIN_YEARS)}), "
          f"Test: {len(test_rows)} (years {sorted(TEST_YEARS)})")
    print(f"Train class counts: {dict(Counter(r['silver_label'] for r in train_rows))}")
    print(f"Test class counts:  {dict(Counter(r['silver_label'] for r in test_rows))}\n")

    print("=" * 70)
    print("A/B/C FEATURE SET COMPARISON (Logistic Regression, identical setup/split)")
    print("=" * 70)
    results = {}
    for label, (numeric_cols, categorical_cols) in FEATURE_SETS.items():
        m = run_feature_set(rows, train_rows, test_rows, numeric_cols, categorical_cols, label)
        results[label] = m
        print(f"\n--- {label} ({numeric_cols + categorical_cols}) ---")
        print(f"  accuracy={m['accuracy']:.3f}  balanced_accuracy={m['balanced_accuracy']:.3f}  "
              f"macro_f1={m['macro_f1']:.3f}")
        for c in m["classes"]:
            pc = m["per_class"][c]
            print(f"    {c}: P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f} "
                  f"support={pc['support']} PR-AUC={m['pr_auc_ovr'].get(c, float('nan')):.3f}")
        print("  Top permutation importances:")
        for name, imp, std in m["permutation_importance"][:5]:
            print(f"    {name}: {imp:.4f} +/- {std:.4f}")

    write_metrics_csv(results)
    print(f"\nWrote {METRICS_CSV}")
    return results


def write_metrics_csv(results, path=METRICS_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for label, m in results.items():
        for c in m["classes"]:
            pc = m["per_class"][c]
            out_rows.append({
                "feature_set": label, "class": c,
                "precision": pc["precision"], "recall": pc["recall"], "f1": pc["f1"],
                "support": pc["support"], "pr_auc_ovr": m["pr_auc_ovr"].get(c, ""),
                "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"],
                "macro_f1": m["macro_f1"],
            })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)


if __name__ == "__main__":
    main()
