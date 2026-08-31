"""
Supervised-learning experiment: can a model trained only on historical
information predict whether a known Gujarat thermal cluster will be
detected again (recur) the following year?

Target A (exactly as approved in the original feasibility report,
unchanged): For cluster c and year Y: target = 1 if detections_(Y+1)(c) > 0,
else 0. One training example = (cluster, year Y) -> features through Y ->
target. Each example is independent of the others -- there is no
sequential/lag feature linking consecutive years for the same cluster --
so a temporal gap between included transitions (see 2022->2023 below)
affects only how many rows exist, not correctness or ordering.

EXTENDED MILESTONE (2024/2025 data added): this experiment originally
covered 3 transitions (2019->2020, 2020->2021, 2021->2022). 2024 and 2025
Gujarat FIRMS data were confirmed available and ingested (read-only
investigation, then this milestone), adding two more valid transitions:
2023->2024 and 2024->2025. VALID_TRANSITIONS below now has 5 entries.

VALID_TRANSITIONS (5): 2019->2020, 2020->2021, 2021->2022, 2023->2024,
2024->2025.

EXCLUDED_TRANSITIONS (see dict below, with reasons) -- deliberately, not
silently:
  - 2022->2023: the 60 clusters were themselves defined by DBSCAN on 2023
    detections, so "detected in 2023" is true for all 60 by construction
    (degenerate/selection-biased target, verified in the feasibility
    report: 60/60 positive, 0 negative). This circularity is specific to
    using 2023 as the TARGET year -- 2023 is used safely as a FEATURE year
    in the 2023->2024 transition below.
  - 2025->2026: 2026 is only a partial year as of the investigation this
    milestone builds on (VIIRS_SNPP_SP through 2026-04-27, VIIRS_SNPP_NRT
    2026-04-28 onward, ~66.6% of the year available as of 2026-08-31).
    detections_2026 would be right-censored -- systematically undercounting
    clusters that only reactivate later in the year -- so this is not a
    valid annual-recurrence evaluation target yet, for a different reason
    than 2022->2023 (temporary/censoring, not structural circularity).

TWO PRESERVED SPLITS, kept side by side for comparison (see PROGRESS.md/
DECISIONS.md for the full comparison):
  - MVP_* (original, UNCHANGED): train=2019->2020,2020->2021,
    test=2021->2022. Kept exactly as before as the historical baseline.
  - EXTENDED_PRIMARY_* (new): train=2019->2020,2020->2021,2021->2022,
    2023->2024, test=2024->2025 -- the newest complete forward holdout,
    used to check whether the MVP's weak-to-moderate signal survives on
    years the model has never seen and that were not used to define the
    clusters.
ROBUSTNESS_* (unchanged): train=2019->2020, test=2020->2021.

Feature source: data/processed/gujarat_cluster_integrated_evidence.csv,
READ-ONLY, unchanged. Only an explicit ALLOWLIST of columns is ever read
from it -- not a blacklist of forbidden columns -- so no leaky column can
enter the feature matrix by omission. The allowlist is UNCHANGED from the
original MVP -- 2024/2025 availability did not motivate adding features,
only adding transitions. Every column named in the milestone brief's
forbidden list is structurally impossible to include here because it is
never read.

Structural limitation, disclosed not hidden: cluster geometry itself
(centroid, extent_radius_m, bbox) was derived from 2023 DBSCAN output, and
the cross-year matching rule that produced every year's detections_Y used
that same 2023-derived extent_radius_m as its matching radius. This is not
changed here -- it is inherited, unchanged, from the approved
cross-year-recurrence methodology, and applies identically to the new
2023->2024/2024->2025 transitions: the spatial frame is still exactly the
60 zones 2023 was dense enough to discover, not new zones that may have
emerged since.

This script does NOT: change the target definition, add new features,
tune against the final 2024->2025 holdout, claim causation, produce a risk
score, or claim production readiness.
"""

import csv
import json
import platform
from collections import Counter
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from ml_feature_inventory import impute_nearest_distance

SOURCE_CSV = Path("data/processed/gujarat_cluster_integrated_evidence.csv")
PREDICTIONS_CSV = Path("data/processed/next_year_recurrence_predictions.csv")
METRICS_CSV = Path("data/processed/next_year_recurrence_metrics.csv")
FIGURE_CONFUSION = Path("results/figures/next_year_recurrence_confusion_matrices.png")
FIGURE_COMPARISON = Path("results/figures/next_year_recurrence_model_comparison.png")
FIGURE_OLD_VS_NEW = Path("results/figures/next_year_recurrence_old_vs_new_comparison.png")
MODEL_DIR = Path("data/processed/models")

RANDOM_STATE = 42

VALID_TRANSITIONS = [(2019, 2020), (2020, 2021), (2021, 2022), (2023, 2024), (2024, 2025)]

# Deliberately excluded, with reasons -- see module docstring. Asserted
# disjoint from VALID_TRANSITIONS below (assert_transitions_consistent),
# not just omitted, so neither can silently reappear.
EXCLUDED_TRANSITIONS = {
    (2022, 2023): (
        "2023-derived cluster-selection circularity: the 60 clusters were "
        "defined by DBSCAN on 2023 detections, so detected-in-2023 is true "
        "for all 60 by construction (60/60 positive, 0 negative, verified "
        "in the feasibility report)."
    ),
    (2025, 2026): (
        "2026 is incomplete/right-censored as of this milestone "
        "(~66.6% of the year available: VIIRS_SNPP_SP through 2026-04-27, "
        "VIIRS_SNPP_NRT 2026-04-28 onward, as of 2026-08-31) -- "
        "detections_2026 would systematically undercount clusters that "
        "only reactivate later in the year."
    ),
}

# Original MVP split -- UNCHANGED, kept as the historical comparison point.
MVP_PRIMARY_TRAIN_TRANSITIONS = [(2019, 2020), (2020, 2021)]
MVP_PRIMARY_TEST_TRANSITIONS = [(2021, 2022)]

# New: the extended primary experiment -- adds 2023->2024 to training and
# holds out the newest complete transition, 2024->2025, as the final test.
EXTENDED_PRIMARY_TRAIN_TRANSITIONS = [(2019, 2020), (2020, 2021), (2021, 2022), (2023, 2024)]
EXTENDED_PRIMARY_TEST_TRANSITIONS = [(2024, 2025)]

# Unchanged.
ROBUSTNESS_TRAIN_TRANSITIONS = [(2019, 2020)]
ROBUSTNESS_TEST_TRANSITIONS = [(2020, 2021)]

# Forward-chaining checks (milestone brief section 6): each step trains on
# strictly earlier transitions than it tests on. Steps 1 and 2 below are
# exactly ROBUSTNESS and MVP_PRIMARY respectively (not recomputed twice --
# main() reuses those results); step 3 is EXTENDED_PRIMARY.
FORWARD_CHAIN_STEPS = [
    {"label": "chain_1 (= robustness)", "train": ROBUSTNESS_TRAIN_TRANSITIONS, "test": ROBUSTNESS_TEST_TRANSITIONS},
    {"label": "chain_2 (= mvp_primary)", "train": MVP_PRIMARY_TRAIN_TRANSITIONS, "test": MVP_PRIMARY_TEST_TRANSITIONS},
    {"label": "chain_3 (= extended_primary)", "train": EXTENDED_PRIMARY_TRAIN_TRANSITIONS, "test": EXTENDED_PRIMARY_TEST_TRANSITIONS},
]


def assert_transitions_consistent():
    """VALID_TRANSITIONS and EXCLUDED_TRANSITIONS must be disjoint, and
    every transition used by an actual split must be in VALID_TRANSITIONS
    -- guards against a transition being silently included or excluded."""
    overlap = set(VALID_TRANSITIONS) & set(EXCLUDED_TRANSITIONS.keys())
    if overlap:
        raise ValueError(f"Transitions both valid and excluded: {overlap}")
    used = set(
        MVP_PRIMARY_TRAIN_TRANSITIONS + MVP_PRIMARY_TEST_TRANSITIONS
        + EXTENDED_PRIMARY_TRAIN_TRANSITIONS + EXTENDED_PRIMARY_TEST_TRANSITIONS
        + ROBUSTNESS_TRAIN_TRANSITIONS + ROBUSTNESS_TEST_TRANSITIONS
    )
    not_valid = used - set(VALID_TRANSITIONS)
    if not_valid:
        raise ValueError(f"Split uses transitions not in VALID_TRANSITIONS: {not_valid}")

# Explicit ALLOWLIST -- the only columns ever read from the source table as
# features. Anything not named here is structurally excluded.
STATIC_NUMERIC_FEATURES = [
    "centroid_lat", "centroid_lon", "extent_radius_m",
    "bbox_min_lat", "bbox_max_lat", "bbox_min_lon", "bbox_max_lon",
    "nearest_distance_m", "nearest_is_named", "has_notable_osm_context",
    "n_industrial", "n_power", "n_waste", "n_agricultural", "n_transport", "n_other",
    "features_found_in_radius",
]
STATIC_CATEGORICAL_FEATURES = ["nearest_group"]

# Explicitly forbidden per the milestone brief -- asserted absent, not just omitted.
FORBIDDEN_FEATURES = {
    "recurrence_strength", "short_window_recurrence", "burst_concentrated",
    "mean_frp", "max_frp", "frp_ratio", "unique_dates", "active_span_days",
    "occurrence_rate", "day_count", "night_count", "night_fraction",
    "top_day_share", "top3_days_share", "detection_count",
    "unique_years", "years_detected", "first_year", "last_year",
    "recurs_across_multiple_years", "trend_slope", "trend_direction",
    "std_annual_detections", "cv_annual_detections", "total_detections_5yr",
    "mean_annual_detections", "per_month_detection_counts", "top3_months_share",
    "dominant_month", "persistence_category", "activity_category",
    "persistence_activity_quadrant", "seasonality_category", "evidence_notes",
    "evidence_corroboration_count", "corroboration_2023_recurrence",
    "corroboration_5yr_persistence", "corroboration_osm_context",
    "unsupervised_group",
}


def load_source_rows(path=SOURCE_CSV):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_bool_int(value):
    return 1 if value == "True" or value is True else 0


def build_panel(rows, transitions):
    """One row per (cluster, year Y) for each requested transition, with
    features available through Y and the Y+1 recurrence target. Reads
    ONLY the allowlisted per-year and static columns from each source row."""
    panel = []
    for row in rows:
        cluster_id = int(row["cluster_id"])
        for y, y1 in transitions:
            record = {
                "cluster_id": cluster_id,
                "year": y,
                "target_year": y1,
                "detections_Y": int(row[f"detections_{y}"]),
                "active_days_Y": int(row[f"active_days_{y}"]),
                "target": 1 if int(row[f"detections_{y1}"]) > 0 else 0,
            }
            for col in STATIC_NUMERIC_FEATURES:
                val = row[col]
                if col in ("nearest_is_named", "has_notable_osm_context"):
                    record[col] = to_bool_int(val)
                elif col == "nearest_distance_m":
                    # Same imputation rule already approved and tested in
                    # the unsupervised-ML milestone (ml_feature_inventory.py):
                    # missing distance (no OSM context found) is filled with
                    # that cluster's own OSM search radius, not a new value.
                    record[col] = impute_nearest_distance(row)
                else:
                    record[col] = float(val)
            for col in STATIC_CATEGORICAL_FEATURES:
                record[col] = row[col]
            panel.append(record)
    return panel


FEATURE_COLUMNS_NUMERIC = ["detections_Y", "active_days_Y"] + STATIC_NUMERIC_FEATURES
FEATURE_COLUMNS_CATEGORICAL = STATIC_CATEGORICAL_FEATURES


def assert_no_forbidden_features():
    used = set(FEATURE_COLUMNS_NUMERIC) | set(FEATURE_COLUMNS_CATEGORICAL)
    overlap = used & FORBIDDEN_FEATURES
    if overlap:
        raise ValueError(f"Forbidden/leaky features present in feature set: {overlap}")


def panel_to_arrays(panel):
    X_numeric = np.array([[r[c] for c in FEATURE_COLUMNS_NUMERIC] for r in panel], dtype=float)
    X_categorical = np.array([[r[c] for c in FEATURE_COLUMNS_CATEGORICAL] for r in panel], dtype=object)
    y = np.array([r["target"] for r in panel], dtype=int)
    return X_numeric, X_categorical, y


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), list(range(len(FEATURE_COLUMNS_NUMERIC)))),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         list(range(len(FEATURE_COLUMNS_NUMERIC), len(FEATURE_COLUMNS_NUMERIC) + len(FEATURE_COLUMNS_CATEGORICAL)))),
    ])


def combine_features(X_numeric, X_categorical):
    return np.hstack([X_numeric.astype(object), X_categorical])


def fit_majority_baseline(y_train):
    majority_class = Counter(y_train).most_common(1)[0][0]
    return majority_class


def predict_majority_baseline(majority_class, n):
    return np.full(n, majority_class)


def build_logistic_pipeline():
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ])


def build_random_forest_pipeline():
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_STATE,
        )),
    ])


def compute_metrics(y_true, y_pred, y_prob_class0=None):
    """Minority/non-recurrence class is label 0. Metrics for that class use
    pos_label=0 explicitly throughout."""
    metrics = {
        "n": len(y_true),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_minority_class0": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_minority_class0": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_minority_class0": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = cm.tolist()  # rows=true[0,1], cols=pred[0,1]
    if y_prob_class0 is not None:
        y_true_class0 = (y_true == 0).astype(int)
        metrics["pr_auc_minority_class0"] = average_precision_score(y_true_class0, y_prob_class0)
        metrics["pred_prob_class0_min"] = float(np.min(y_prob_class0))
        metrics["pred_prob_class0_median"] = float(np.median(y_prob_class0))
        metrics["pred_prob_class0_mean"] = float(np.mean(y_prob_class0))
        metrics["pred_prob_class0_max"] = float(np.max(y_prob_class0))
    else:
        metrics["pr_auc_minority_class0"] = None
    return metrics


def run_split(rows, train_transitions, test_transitions, label):
    train_panel = build_panel(rows, train_transitions)
    test_panel = build_panel(rows, test_transitions)

    Xn_train, Xc_train, y_train = panel_to_arrays(train_panel)
    Xn_test, Xc_test, y_test = panel_to_arrays(test_panel)
    X_train = combine_features(Xn_train, Xc_train)
    X_test = combine_features(Xn_test, Xc_test)

    results = {"label": label, "n_train": len(y_train), "n_test": len(y_test),
               "train_positive": int(np.sum(y_train == 1)), "train_negative": int(np.sum(y_train == 0)),
               "test_positive": int(np.sum(y_test == 1)), "test_negative": int(np.sum(y_test == 0)),
               "train_positive_rate": float(np.mean(y_train == 1)) if len(y_train) else None,
               "test_positive_rate": float(np.mean(y_test == 1)) if len(y_test) else None}

    # Model 0: majority baseline (fit on train only)
    majority_class = fit_majority_baseline(y_train)
    y_pred_baseline = predict_majority_baseline(majority_class, len(y_test))
    results["baseline"] = compute_metrics(y_test, y_pred_baseline, y_prob_class0=None)
    results["baseline"]["majority_class"] = int(majority_class)

    # Model 1: Logistic Regression
    lr = build_logistic_pipeline()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, list(lr.named_steps["clf"].classes_).index(0)]
    results["logistic_regression"] = compute_metrics(y_test, y_pred_lr, y_prob_class0=y_prob_lr)

    # Model 2: Small Random Forest
    rf = build_random_forest_pipeline()
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, list(rf.named_steps["clf"].classes_).index(0)]
    results["random_forest"] = compute_metrics(y_test, y_pred_rf, y_prob_class0=y_prob_rf)

    return results, lr, rf, test_panel, y_pred_lr, y_prob_lr, y_pred_rf, y_prob_rf


def readable_feature_names(preprocessor):
    """Build human-readable feature names from the fitted ColumnTransformer,
    since X is a plain numpy array (no column names) and get_feature_names_out
    would otherwise only produce generic x0, x1, ... labels."""
    names = list(FEATURE_COLUMNS_NUMERIC)  # StandardScaler preserves order/count
    onehot = preprocessor.named_transformers_["cat"]
    for col_name, categories in zip(FEATURE_COLUMNS_CATEGORICAL, onehot.categories_):
        for cat in categories:
            label = cat if cat else "(no_context)"
            names.append(f"{col_name}={label}")
    return names


def get_logistic_coefficients(pipeline):
    preprocessor = pipeline.named_steps["preprocess"]
    clf = pipeline.named_steps["clf"]
    feature_names = readable_feature_names(preprocessor)
    coefs = clf.coef_[0]
    ranked = sorted(zip(feature_names, coefs), key=lambda x: -abs(x[1]))
    return ranked


def get_rf_importances(pipeline):
    preprocessor = pipeline.named_steps["preprocess"]
    clf = pipeline.named_steps["clf"]
    feature_names = readable_feature_names(preprocessor)
    importances = clf.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    return ranked


def print_class_balance(label, results):
    print(f"  [{label}] train: n={results['n_train']} "
          f"pos={results['train_positive']} neg={results['train_negative']} "
          f"pos_rate={results['train_positive_rate']:.3f}" if results['n_train'] else f"  [{label}] train: n=0")
    print(f"  [{label}] test:  n={results['n_test']} "
          f"pos={results['test_positive']} neg={results['test_negative']} "
          f"pos_rate={results['test_positive_rate']:.3f}" if results['n_test'] else f"  [{label}] test: n=0")


def main():
    assert_transitions_consistent()
    assert_no_forbidden_features()
    print(f"Valid transitions (5): {VALID_TRANSITIONS}")
    print(f"Excluded transitions: {list(EXCLUDED_TRANSITIONS.keys())}")
    print(f"Feature allowlist verified clean: {len(FEATURE_COLUMNS_NUMERIC)} numeric + "
          f"{len(FEATURE_COLUMNS_CATEGORICAL)} categorical columns, "
          f"zero overlap with {len(FORBIDDEN_FEATURES)} forbidden columns "
          f"(unchanged from the original MVP -- no features added for this milestone).\n")

    rows = load_source_rows()
    print(f"Loaded {len(rows)} clusters from {SOURCE_CSV} (read-only).\n")

    print("=" * 70)
    print("MVP PRIMARY (original, unchanged): train=2019->2020,2020->2021 | test=2021->2022")
    print("=" * 70)
    mvp_primary, lr_mvp, rf_mvp, test_panel_mvp, y_pred_lr_mvp, y_prob_lr_mvp, y_pred_rf_mvp, y_prob_rf_mvp = run_split(
        rows, MVP_PRIMARY_TRAIN_TRANSITIONS, MVP_PRIMARY_TEST_TRANSITIONS, "mvp_primary"
    )
    print(json.dumps(mvp_primary, indent=2, default=str))

    print("\n" + "=" * 70)
    print("ROBUSTNESS CHECK (unchanged): train=2019->2020 | test=2020->2021")
    print("=" * 70)
    robustness, lr_rob, rf_rob, test_panel_rob, *_ = run_split(
        rows, ROBUSTNESS_TRAIN_TRANSITIONS, ROBUSTNESS_TEST_TRANSITIONS, "robustness"
    )
    print(json.dumps(robustness, indent=2, default=str))

    print("\n" + "=" * 70)
    print("EXTENDED PRIMARY (new): train=2019->2020,2020->2021,2021->2022,2023->2024 | test=2024->2025")
    print("=" * 70)
    extended_primary, lr_ext, rf_ext, test_panel_ext, y_pred_lr_ext, y_prob_lr_ext, y_pred_rf_ext, y_prob_rf_ext = run_split(
        rows, EXTENDED_PRIMARY_TRAIN_TRANSITIONS, EXTENDED_PRIMARY_TEST_TRANSITIONS, "extended_primary"
    )
    print(json.dumps(extended_primary, indent=2, default=str))

    print("\n" + "=" * 70)
    print("FORWARD-CHAINING CHECKS (each step trains only on strictly earlier "
          "transitions; steps below are the three splits already run above, "
          "not recomputed)")
    print("=" * 70)
    for step in FORWARD_CHAIN_STEPS:
        print(f"  {step['label']}: train={step['train']} test={step['test']}")

    print("\n" + "=" * 70)
    print("CLASS BALANCE, ALL SPLITS")
    print("=" * 70)
    print_class_balance("mvp_primary", mvp_primary)
    print_class_balance("robustness", robustness)
    print_class_balance("extended_primary", extended_primary)

    print("\n" + "=" * 70)
    print("OLD (MVP) vs NEW (EXTENDED) HOLDOUT COMPARISON")
    print("=" * 70)
    for model_name in ("baseline", "logistic_regression", "random_forest"):
        old_m, new_m = mvp_primary[model_name], extended_primary[model_name]
        print(f"  {model_name}:")
        print(f"    balanced_accuracy: old(2021->22)={old_m['balanced_accuracy']:.3f}  "
              f"new(2024->25)={new_m['balanced_accuracy']:.3f}")
        print(f"    recall_minority0:  old={old_m['recall_minority_class0']:.3f}  "
              f"new={new_m['recall_minority_class0']:.3f}")
        print(f"    precision_minority0: old={old_m['precision_minority_class0']:.3f}  "
              f"new={new_m['precision_minority_class0']:.3f}")
        print(f"    f1_minority0: old={old_m['f1_minority_class0']:.3f}  "
              f"new={new_m['f1_minority_class0']:.3f}")
        old_pr = old_m['pr_auc_minority_class0']
        new_pr = new_m['pr_auc_minority_class0']
        old_pr_str = f"{old_pr:.3f}" if old_pr is not None else "n/a"
        new_pr_str = f"{new_pr:.3f}" if new_pr is not None else "n/a"
        print(f"    pr_auc_minority0: old={old_pr_str}  new={new_pr_str}")

    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION COEFFICIENTS (extended-primary model, ranked by |coef|)")
    print("=" * 70)
    lr_coefs = get_logistic_coefficients(lr_ext)
    for name, coef in lr_coefs:
        print(f"  {name}: {coef:+.3f}")

    print("\n" + "=" * 70)
    print("RANDOM FOREST FEATURE IMPORTANCES (extended-primary model, ranked -- "
          "UNSTABLE/EXPLORATORY given tiny sample and few independent clusters)")
    print("=" * 70)
    rf_importances = get_rf_importances(rf_ext)
    for name, imp in rf_importances:
        print(f"  {name}: {imp:.3f}")

    write_predictions_csv([
        ("mvp_primary", test_panel_mvp, y_pred_lr_mvp, y_prob_lr_mvp, y_pred_rf_mvp, y_prob_rf_mvp),
        ("extended_primary", test_panel_ext, y_pred_lr_ext, y_prob_lr_ext, y_pred_rf_ext, y_prob_rf_ext),
    ])
    print(f"\nWrote {PREDICTIONS_CSV}")

    write_metrics_csv([mvp_primary, robustness, extended_primary])
    print(f"Wrote {METRICS_CSV}")

    plot_confusion_matrices(extended_primary)
    print(f"Saved figure to {FIGURE_CONFUSION}")
    plot_model_comparison(mvp_primary, robustness, extended_primary)
    print(f"Saved figure to {FIGURE_COMPARISON}")
    plot_old_vs_new_comparison(mvp_primary, extended_primary)
    print(f"Saved figure to {FIGURE_OLD_VS_NEW}")

    save_model_artifacts(lr_ext, rf_ext, extended_primary)
    print(f"\nSaved model artifacts to {MODEL_DIR} (see provenance.json for exact "
          f"training split, feature list, and caveats)")

    return mvp_primary, robustness, extended_primary, lr_coefs, rf_importances


def save_model_artifacts(lr_pipeline, rf_pipeline, extended_primary_results, path=MODEL_DIR):
    """Persist the EXTENDED-PRIMARY-split-trained pipelines (train=2019-20,
    2020-21,2021-22,2023-24 / test=2024-25) -- this supersedes the original
    MVP artifact (train=2019-20,2020-21 / test=2021-22) as the headline
    exploratory model, now that 2024/2025 provide a newer forward holdout.
    Still an exploratory artifact, not a production model, and the metadata
    says so plainly."""
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr_pipeline, path / "logistic_regression_primary.joblib")
    joblib.dump(rf_pipeline, path / "random_forest_primary.joblib")

    provenance = {
        "status": "EXPLORATORY ARTIFACT -- NOT PRODUCTION READY",
        "trained_on": date.today().isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "random_state": RANDOM_STATE,
        "target_definition": "1 if detections_(Y+1)(cluster) > 0 else 0",
        "train_transitions": EXTENDED_PRIMARY_TRAIN_TRANSITIONS,
        "held_out_test_transition": EXTENDED_PRIMARY_TEST_TRANSITIONS,
        "supersedes": "original MVP artifact (train=2019-20,2020-21 / test=2021-22), "
                       "kept and reported separately for comparison, see next_year_recurrence_metrics.csv",
        "n_train": extended_primary_results["n_train"],
        "n_test": extended_primary_results["n_test"],
        "feature_columns_numeric": FEATURE_COLUMNS_NUMERIC,
        "feature_columns_categorical": FEATURE_COLUMNS_CATEGORICAL,
        "extended_primary_test_balanced_accuracy": {
            "baseline": extended_primary_results["baseline"]["balanced_accuracy"],
            "logistic_regression": extended_primary_results["logistic_regression"]["balanced_accuracy"],
            "random_forest": extended_primary_results["random_forest"]["balanced_accuracy"],
        },
        "caveats": [
            "Trained on rows pooled from only 60 underlying clusters across 4 transitions "
            "(pooled rows not independent).",
            f"Held-out test set (2024->2025) has {extended_primary_results['test_negative']} "
            "negative (non-recurrence) examples.",
            "Single forward holdout, not cross-validated -- low statistical power.",
            "Cluster geometry (features like extent_radius_m, centroid) was itself derived "
            "from 2023 DBSCAN output, a structural limitation inherited unchanged from prior "
            "milestones and unaffected by adding 2024/2025 -- the spatial frame is still the "
            "60 zones 2023 was dense enough to discover, not new zones that emerged since.",
            "Do not use for any production, deployment, or decision-making purpose.",
        ],
    }
    with open(path / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)


def write_predictions_csv(splits, path=PREDICTIONS_CSV):
    """splits: list of (split_label, test_panel, y_pred_lr, y_prob_lr, y_pred_rf, y_prob_rf)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "cluster_id", "year", "target_year", "actual_target",
                          "lr_predicted", "lr_predicted_prob_nonrecurrence",
                          "rf_predicted", "rf_predicted_prob_nonrecurrence"])
        for split_label, test_panel, y_pred_lr, y_prob_lr, y_pred_rf, y_prob_rf in splits:
            for r, pl, prl, pr, prr in zip(test_panel, y_pred_lr, y_prob_lr, y_pred_rf, y_prob_rf):
                writer.writerow([split_label, r["cluster_id"], r["year"], r["target_year"], r["target"],
                                  int(pl), round(float(prl), 4), int(pr), round(float(prr), 4)])


def write_metrics_csv(split_results, path=METRICS_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_result in split_results:
        for model_name in ("baseline", "logistic_regression", "random_forest"):
            m = split_result[model_name]
            rows.append({
                "split": split_result["label"], "model": model_name,
                "n_train": split_result["n_train"], "n_test": split_result["n_test"],
                "train_positive": split_result["train_positive"], "train_negative": split_result["train_negative"],
                "train_positive_rate": split_result["train_positive_rate"],
                "test_positive": split_result["test_positive"], "test_negative": split_result["test_negative"],
                "test_positive_rate": split_result["test_positive_rate"],
                "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                "precision_minority_class0": m["precision_minority_class0"],
                "recall_minority_class0": m["recall_minority_class0"],
                "f1_minority_class0": m["f1_minority_class0"],
                "pr_auc_minority_class0": m["pr_auc_minority_class0"],
                "confusion_matrix": json.dumps(m["confusion_matrix"]),
            })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion_matrices(extended_primary, output_path=FIGURE_CONFUSION):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, model_name, title in zip(
        axes, ("baseline", "logistic_regression", "random_forest"),
        ("Majority Baseline", "Logistic Regression", "Random Forest"),
    ):
        cm = np.array(extended_primary[model_name]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: 0 (no recur)", "Pred: 1 (recur)"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True: 0", "True: 1"], fontsize=8)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
        ax.set_title(title, fontsize=10)
    fig.suptitle("Confusion Matrices -- Extended Primary Holdout "
                  "(train 2019-20,2020-21,2021-22,2023-24 / test 2024-25)\n"
                  "Exploratory experiment, small sample (n_test=%d)" % extended_primary["n_test"])
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(mvp_primary, robustness, extended_primary, output_path=FIGURE_COMPARISON):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    models = ["baseline", "logistic_regression", "random_forest"]
    labels = ["Baseline", "Logistic\nRegression", "Random\nForest"]

    splits = (mvp_primary, robustness, extended_primary)
    titles = ("MVP primary (test=2021->2022)", "Robustness check (test=2020->2021)",
              "Extended primary (test=2024->2025)")

    for ax, split_result, title in zip(axes, splits, titles):
        bal_acc = [split_result[m]["balanced_accuracy"] for m in models]
        recall0 = [split_result[m]["recall_minority_class0"] for m in models]
        x = np.arange(len(models))
        ax.bar(x - 0.2, bal_acc, width=0.4, label="Balanced accuracy", color="#2b6cb0")
        ax.bar(x + 0.2, recall0, width=0.4, label="Recall (minority/non-recurrence)", color="#c53030")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Model Comparison vs. Majority Baseline, All Splits (exploratory, not production evaluation)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_old_vs_new_comparison(mvp_primary, extended_primary, output_path=FIGURE_OLD_VS_NEW):
    """Direct old-vs-new comparison (milestone brief section 9): for each
    model, balanced accuracy / minority recall / minority precision / F1 /
    PR-AUC on the original MVP holdout (test=2021->2022) side by side with
    the new extended holdout (test=2024->2025)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["balanced_accuracy", "recall_minority_class0", "precision_minority_class0",
               "f1_minority_class0", "pr_auc_minority_class0"]
    metric_labels = ["Balanced\naccuracy", "Recall\n(minority)", "Precision\n(minority)",
                      "F1\n(minority)", "PR-AUC\n(minority)"]
    models = ["baseline", "logistic_regression", "random_forest"]
    model_titles = ["Majority Baseline", "Logistic Regression", "Random Forest"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, model_name, title in zip(axes, models, model_titles):
        old_vals = [mvp_primary[model_name][m] if mvp_primary[model_name][m] is not None else 0.0 for m in metrics]
        new_vals = [extended_primary[model_name][m] if extended_primary[model_name][m] is not None else 0.0 for m in metrics]
        x = np.arange(len(metrics))
        ax.bar(x - 0.2, old_vals, width=0.4, label="Old MVP (test=2021->2022)", color="#a0aec0")
        ax.bar(x + 0.2, new_vals, width=0.4, label="New Extended (test=2024->2025)", color="#2b6cb0")
        ax.set_xticks(x); ax.set_xticklabels(metric_labels, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)

    fig.suptitle("Old MVP Holdout vs. New Extended Holdout -- Does the Signal Survive?\n"
                  "(exploratory, both holdouts have very few examples -- see caveats in report)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
