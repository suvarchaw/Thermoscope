"""
2026 model inference (additive; the final step of the 2026 NRT pipeline).

Loads the FINAL LOCKED classifier artifact
(data/processed/models/source_classifier_lightgbm_v2.joblib -- LightGBM,
11-feature set, trained ONLY on 2019-2022 silver-labeled historical
events, per the closed feature-engineering track -- see
PROGRESS.md/DECISIONS.md) and runs `.predict`/`.predict_proba` on 2026
events. Does NOT fit/retrain anything -- this module never calls .fit on
any estimator; joblib.load reconstructs the exact already-fitted pipeline
byte-for-byte.

FEATURES: the same 11 columns (FEATURES_NUMERIC + FEATURES_CATEGORICAL,
imported unchanged from train_source_classifier_lightgbm.py) are computed
for 2026 events using the SAME functions used historically
(event_construction.summarize_event for centroid_lat/centroid_lon/
spatial_extent_m/status, event_behavior_features.compute_new_features_for_event
for the 7 behavior features) -- no feature is added, removed, or
recomputed differently.

FORCED 4-CLASS OUTPUT, DISCLOSED: the trained model only knows the 4
TRAINED_CLASSES (Industrial, Gas_Flare, Crop_Residue, Forest_Wildfire) --
Brick_Kiln and Unknown_Ambiguous were excluded from training (see
train_source_classifier.py). predict() therefore ALWAYS returns one of
those 4 labels, even for a 2026 event that would genuinely be
Unknown/Ambiguous or Brick_Kiln -- there is no "none of the above" option.
The per-class probability columns are the only way to see a forced,
low-confidence call (e.g. near-uniform ~0.25 each) versus a confident one;
no additional confidence/risk score is invented on top of them.

CAPABILITY LIMITATION, CARRIED FORWARD NOT HIDDEN: CAPABILITY_NOTE below
is a static, already-documented lookup (not a new score or model) stating
which classes the spatial-holdout evaluation found generalize genuinely
(Crop_Residue, Forest_Wildfire) versus primarily recognize known sites
(Industrial, Gas_Flare) -- see PROGRESS.md's spatial-evaluation and
feature-experiment milestones for the numbers behind each note.
"""

import csv
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import event_construction_2026 as ec2026
from event_behavior_features import (
    compute_new_features_for_event, rows_to_arrays, combine_features,
)
from train_source_classifier_lightgbm import FEATURES_NUMERIC, FEATURES_CATEGORICAL

MODEL_PATH = Path("data/processed/models/source_classifier_lightgbm_v2.joblib")
EVIDENCE_2026_CSV = Path("data/processed/gujarat_event_evidence_2026.csv")
OUTPUT_CSV = Path("data/processed/gujarat_2026_inference.csv")

TRAINED_CLASSES = ["Crop_Residue", "Forest_Wildfire", "Gas_Flare", "Industrial"]

CAPABILITY_NOTE = {
    "Crop_Residue": "Demonstrated genuine spatial generalization under geographic holdout evaluation.",
    "Forest_Wildfire": "Demonstrated genuine spatial generalization under geographic holdout evaluation.",
    "Industrial": "Primarily known-site/location recognition; weak generalization to unseen geography "
                  "(spatial-holdout LightGBM F1 ~0.11-0.12) -- treat predictions at new locations with caution.",
    "Gas_Flare": "Primarily known-site/location recognition; near-zero F1 under geographic holdout "
                 "-- treat any prediction at a location with no prior evidence as low-confidence.",
}

EVIDENCE_FIELDS = [
    "nearest_osm_industrial_power_m", "nearest_gppd_thermal_plant_m",
    "nearest_osm_flare_m", "nearest_osm_kiln_m",
    "overlaps_cluster_id", "overlaps_cluster_recurrence_strength",
    "land_cover_class",
]


def load_model(path=MODEL_PATH):
    """Loads the already-fitted pipeline. Never calls .fit -- 2026 is
    inference-only, per the milestone's explicit instruction."""
    return joblib.load(path)


def build_2026_feature_rows(reference_date=None, raw_path=None):
    """Rebuilds 2026 events + members in-memory (deterministic, same
    algorithm as event_construction_2026.main) and computes the 11 locked
    feature columns per event using the unchanged historical feature
    functions. Returns (events, feature_rows) with matching order.

    reference_date is passed straight through to
    event_construction_2026.build_2026_events (itself passed straight
    through to the unchanged event_construction.summarize_event/
    compute_status) -- it is "now" for provisional/closed status, not a
    new methodology parameter. Defaults to None (= latest detection date
    present), matching every prior call site's behavior; the NRT update
    engine passes the actual run date so status reflects true current
    time rather than lagging to the newest ingested detection.

    raw_path overrides which 2026 raw detections file is read (defaults
    to event_construction_2026.RAW_2026_CSV) -- exposed only so the NRT
    update engine's tests can point this at an isolated tmpdir instead of
    the real data/raw/."""
    detections = ec2026.load_2026_detections(path=raw_path) if raw_path is not None else ec2026.load_2026_detections()
    events, members_by_row, _ = ec2026.build_2026_events(detections, reference_date=reference_date)

    feature_rows = []
    for event, members in zip(events, members_by_row):
        behavior = compute_new_features_for_event(members, detections)
        row = {
            "event_id": event["event_id"],
            "centroid_lat": event["centroid_lat"],
            "centroid_lon": event["centroid_lon"],
            "spatial_extent_m": event["spatial_extent_m"],
            "status": event["status"],
        }
        row.update(behavior)
        feature_rows.append(row)
    return events, feature_rows


def load_evidence_by_event_id(path=EVIDENCE_2026_CSV):
    with open(path, newline="") as f:
        return {r["event_id"]: r for r in csv.DictReader(f)}


def predict_2026(pipeline, feature_rows):
    """Runs predict/predict_proba only -- no fitting. Returns
    (y_pred, prob_by_class) where prob_by_class maps each trained class to
    a list of probabilities aligned with feature_rows."""
    Xn, Xc, _dummy_labels = rows_to_arrays(
        [{**r, "silver_label": ""} for r in feature_rows], FEATURES_NUMERIC, FEATURES_CATEGORICAL
    )
    X = combine_features(Xn, Xc)
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)
    classes_order = list(pipeline.named_steps["clf"].classes_)
    prob_by_class = {
        c: y_prob[:, classes_order.index(c)].tolist() for c in classes_order
    }
    return y_pred.tolist(), prob_by_class


def build_output_rows(events, feature_rows, y_pred, prob_by_class, evidence_by_id):
    prob_classes = sorted(prob_by_class.keys())
    rows = []
    for i, event in enumerate(events):
        ev = evidence_by_id.get(event["event_id"], {})
        predicted_class = y_pred[i]
        row = {
            "event_id": event["event_id"],
            "start_date": event["start_date"], "end_date": event["end_date"],
            "duration_days": event["duration_days"], "status": event["status"],
            "centroid_lat": event["centroid_lat"], "centroid_lon": event["centroid_lon"],
            "spatial_extent_m": event["spatial_extent_m"],
            "detection_count": event["detection_count"],
            "mean_frp": event["mean_frp"], "max_frp": event["max_frp"],
            "night_fraction": event["night_fraction"],
        }
        for f in EVIDENCE_FIELDS:
            row[f] = ev.get(f, "")
        row["predicted_class"] = predicted_class
        for c in prob_classes:
            row[f"prob_{c}"] = round(prob_by_class[c][i], 4)
        row["class_capability_note"] = CAPABILITY_NOTE.get(predicted_class, "")
        row["data_type"] = "2026_NRT_INFERENCE_NOT_GROUND_TRUTH"
        rows.append(row)
    return rows


def write_csv(rows, path=OUTPUT_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print(f"Loading FINAL LOCKED classifier from {MODEL_PATH} (already fitted -- not retraining).")
    pipeline = load_model()
    print(f"Feature columns ({len(FEATURES_NUMERIC) + len(FEATURES_CATEGORICAL)}): "
          f"{FEATURES_NUMERIC + FEATURES_CATEGORICAL}")

    events, feature_rows = build_2026_feature_rows()
    print(f"\n2026 events with computed features: {len(events)}")

    evidence_by_id = load_evidence_by_event_id()

    y_pred, prob_by_class = predict_2026(pipeline, feature_rows)

    from collections import Counter
    print(f"Predictions by class: {dict(Counter(y_pred))}")

    rows = build_output_rows(events, feature_rows, y_pred, prob_by_class, evidence_by_id)
    write_csv(rows)
    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")
    print("This file is model-predicted inference output on 2026 events, NOT ground truth.")
    return rows


if __name__ == "__main__":
    main()
