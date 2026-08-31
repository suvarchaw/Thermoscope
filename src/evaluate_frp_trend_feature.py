"""
Evaluation-only experiment: does adding the (already leakage-audited and
approved) `frp_trend_slope` feature improve genuine spatial
generalization -- particularly for Industrial and Gas Flare -- beyond
the existing 11-feature model, on BOTH the existing temporal split and
the existing spatial checkerboard split?

Nothing about the existing pipeline is changed: event construction,
silver-label rules, the temporal split (train_source_classifier.py
TRAIN_YEARS/TEST_YEARS), the spatial checkerboard split
(spatial_evaluation.assign_spatial_split), the fixed LightGBM
configuration (train_source_classifier_lightgbm.LIGHTGBM_PARAMS), and
the Logistic Regression configuration are all imported UNCHANGED. This
module only adds one new comparison axis (feature set A vs. B) on top of
the two already-existing evaluation axes (temporal vs. spatial split).

MISSING-VALUE TREATMENT FOR frp_trend_slope -- decided BEFORE running
any comparison, not after seeing results
-------------------------------------------------------------------------
71.7% of events (single-day events) have no computable trend (see
event_frp_trend_feature.py -- `frp_trend_slope` is blank for these,
never silently 0.0). Three options were considered:
  1. Restrict both feature sets A and B to only the ~28.3% of events
     with a computable trend. REJECTED: this would change the dataset
     size/composition, confounding "did the feature help" with "did
     training on 3-4x less data help or hurt" -- not a clean comparison.
  2. Impute with a class-conditional or split-conditional statistic
     (e.g. the mean trend for that class, or that split's training
     mean). REJECTED: a class-conditional fill value derived from
     `silver_label` would leak the target into the imputation itself
     -- a stronger, more direct violation than anything already
     excluded from FEATURE_ALLOWLIST. A split-conditional (train-mean)
     fill was also rejected as an invented, result-shaping choice not
     clearly more defensible than the option chosen, and it would make
     the "same 12 columns for both models" comparison messier for no
     clear benefit.
  3. **Chosen: fill `frp_trend_slope` with 0.0 (the same value that
     already means "flat trend" for a real multi-day event) AND add a
     companion boolean column `frp_trend_computable`, so the fill is
     never silent** -- a model (Logistic Regression especially, which
     cannot accept blank/NaN values at all) can use the flag to treat
     "explicitly flat" and "unknown, filled neutral" differently if
     that distinction carries signal, and a human auditing the feature
     set can see exactly which events were filled. This is the standard
     "indicator + neutral fill" missing-data pattern, applied identically
     for BOTH models (Logistic Regression and LightGBM) and BOTH splits,
     so the same 13-column feature matrix is compared everywhere -- no
     model-specific or split-specific special-casing.
Feature Set B therefore has 13 raw columns (11 existing + `frp_trend_slope`
[filled] + `frp_trend_computable` [flag]) representing ONE new physically
meaningful signal, consistent with the "11 features -> 12 features"
framing of this experiment.

Does NOT: change silver labels, event construction, the temporal or
spatial split definitions, model configurations, class weights, or
evaluation metrics; tune anything; add a second new feature; ingest new
data; or build a dashboard/NRT pipeline.
"""

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Importing train_source_classifier_lightgbm triggers its libomp
# self-reexec shim (unchanged, documented there) before any lightgbm
# import happens in this process.
from train_source_classifier_lightgbm import (
    FEATURES_NUMERIC as SET_A_NUMERIC, FEATURES_CATEGORICAL as SET_A_CATEGORICAL,
    LIGHTGBM_AVAILABLE, LIGHTGBM_IMPORT_ERROR, LIGHTGBM_PARAMS,
)
from event_behavior_features import build_preprocessor, rows_to_arrays, combine_features, readable_feature_names
from train_source_classifier import (
    RANDOM_STATE, TRAIN_YEARS, TEST_YEARS, TRAINED_CLASSES,
    filter_to_trained_classes, temporal_split, compute_metrics,
    assert_no_brick_kiln_in_training, assert_no_unknown_ambiguous,
)
from spatial_evaluation import assign_spatial_split

FEATURES_V2_CSV = Path("data/processed/source_classifier_features_v2.csv")
FRP_TREND_CSV = Path("data/processed/event_frp_trend_features.csv")
SILVER_LABELS_CSV = Path("data/processed/gujarat_event_silver_labels.csv")
METRICS_CSV = Path("data/processed/frp_trend_evaluation_metrics.csv")

SET_B_NUMERIC = SET_A_NUMERIC + ["frp_trend_slope"]
SET_B_CATEGORICAL = SET_A_CATEGORICAL + ["frp_trend_computable"]

FEATURE_SETS = {
    "A_11_features": (SET_A_NUMERIC, SET_A_CATEGORICAL),
    "B_12_features_plus_frp_trend": (SET_B_NUMERIC, SET_B_CATEGORICAL),
}

SPLIT_METHODS = ["temporal", "spatial"]


def load_joined_rows():
    """Joins the existing 11-feature table with the FRP trend feature
    table and start_date (for the temporal split), by event_id. Read-only
    -- writes nothing back to either source file."""
    with open(FEATURES_V2_CSV, newline="") as f:
        base_rows = {r["event_id"]: r for r in csv.DictReader(f)}
    with open(FRP_TREND_CSV, newline="") as f:
        frp_rows = {r["event_id"]: r for r in csv.DictReader(f)}
    with open(SILVER_LABELS_CSV, newline="") as f:
        silver_rows = {r["event_id"]: r for r in csv.DictReader(f)}

    rows = []
    for eid, base in base_rows.items():
        frp = frp_rows[eid]
        computable = frp["frp_trend_computable"] == "True"
        row = dict(base)
        row["start_date"] = silver_rows[eid]["start_date"]
        # Pre-registered fill: see module docstring. Never silent -- the
        # companion flag always travels with the filled value.
        row["frp_trend_slope"] = float(frp["frp_trend_slope"]) if computable else 0.0
        row["frp_trend_computable"] = "True" if computable else "False"
        rows.append(row)
    return rows


def build_lightgbm_pipeline(numeric_cols, categorical_cols):
    if not LIGHTGBM_AVAILABLE:
        raise RuntimeError(f"LightGBM is not available: {LIGHTGBM_IMPORT_ERROR}")
    from lightgbm import LGBMClassifier
    return Pipeline([
        ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
        ("clf", LGBMClassifier(**LIGHTGBM_PARAMS)),  # unchanged, fixed config
    ])


def build_logistic_pipeline_local(numeric_cols, categorical_cols):
    # Same LogisticRegression configuration already used throughout this
    # project (class_weight="balanced", max_iter=1000, fixed random_state).
    return Pipeline([
        ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ])


def get_split(rows, method):
    if method == "temporal":
        return temporal_split(rows)  # unchanged, imported
    elif method == "spatial":
        train_rows, test_rows, _cell_report = assign_spatial_split(rows)  # unchanged, imported
        return train_rows, test_rows
    raise ValueError(method)


def run_one(rows, split_method, feature_set_label, model_name):
    numeric_cols, categorical_cols = FEATURE_SETS[feature_set_label]
    train_rows, test_rows = get_split(rows, split_method)

    Xn_train, Xc_train, y_train = rows_to_arrays(train_rows, numeric_cols, categorical_cols)
    Xn_test, Xc_test, y_test = rows_to_arrays(test_rows, numeric_cols, categorical_cols)
    X_train = combine_features(Xn_train, Xc_train)
    X_test = combine_features(Xn_test, Xc_test)
    classes = sorted(set(y_train) | set(y_test))

    if model_name == "logistic_regression":
        pipeline = build_logistic_pipeline_local(numeric_cols, categorical_cols)
    elif model_name == "lightgbm":
        pipeline = build_lightgbm_pipeline(numeric_cols, categorical_cols)
    else:
        raise ValueError(model_name)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    classes_order = list(pipeline.named_steps["clf"].classes_)
    y_prob_ordered = y_prob[:, [classes_order.index(c) for c in classes]]
    metrics = compute_metrics(y_test, y_pred, classes, y_prob_ordered)
    metrics["n_train"] = len(train_rows)
    metrics["n_test"] = len(test_rows)

    perm = permutation_importance(pipeline, X_test, y_test, n_repeats=20,
                                   random_state=RANDOM_STATE, scoring="f1_macro")
    names = readable_feature_names(pipeline.named_steps["preprocess"], numeric_cols, categorical_cols)
    metrics["permutation_importance"] = sorted(
        zip(names, perm.importances_mean.tolist(), perm.importances_std.tolist()), key=lambda x: -x[1]
    )
    return metrics


def main():
    rows = load_joined_rows()
    trained_rows = filter_to_trained_classes(rows)
    assert_no_brick_kiln_in_training(trained_rows)
    assert_no_unknown_ambiguous(trained_rows)
    print(f"Loaded {len(trained_rows)} trained-class events (11-feature table joined with FRP trend, read-only).\n")

    models = ["logistic_regression"] + (["lightgbm"] if LIGHTGBM_AVAILABLE else [])
    if not LIGHTGBM_AVAILABLE:
        print(f"LightGBM UNAVAILABLE: {LIGHTGBM_IMPORT_ERROR} -- reporting Logistic Regression only.\n")

    all_results = {}
    for split_method in SPLIT_METHODS:
        train_rows, test_rows = get_split(trained_rows, split_method)
        print("=" * 70)
        print(f"SPLIT: {split_method}  (train={len(train_rows)}, test={len(test_rows)})")
        print(f"  train class counts: {dict(Counter(r['silver_label'] for r in train_rows))}")
        print(f"  test class counts:  {dict(Counter(r['silver_label'] for r in test_rows))}")
        print("=" * 70)

        for feature_set_label in FEATURE_SETS:
            for model_name in models:
                key = (split_method, feature_set_label, model_name)
                m = run_one(trained_rows, split_method, feature_set_label, model_name)
                all_results[key] = m
                print(f"\n--- {split_method} | {feature_set_label} | {model_name} ---")
                print(f"  accuracy={m['accuracy']:.3f} balanced_accuracy={m['balanced_accuracy']:.3f} "
                      f"macro_f1={m['macro_f1']:.3f}")
                for c in m["classes"]:
                    pc = m["per_class"][c]
                    print(f"    {c}: P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f} "
                          f"support={pc['support']} PR-AUC={m['pr_auc_ovr'].get(c, float('nan')):.3f}")
                print("    top permutation importances:")
                for name, imp, std in m["permutation_importance"][:5]:
                    print(f"      {name}: {imp:.4f} +/- {std:.4f}")

    print("\n" + "=" * 70)
    print("A vs B DIRECT COMPARISON (balanced accuracy / macro F1)")
    print("=" * 70)
    for split_method in SPLIT_METHODS:
        for model_name in models:
            a = all_results[(split_method, "A_11_features", model_name)]
            b = all_results[(split_method, "B_12_features_plus_frp_trend", model_name)]
            print(f"  [{split_method}/{model_name}] A: bal_acc={a['balanced_accuracy']:.3f} macro_f1={a['macro_f1']:.3f}  "
                  f"->  B: bal_acc={b['balanced_accuracy']:.3f} macro_f1={b['macro_f1']:.3f}")

    print("\n" + "=" * 70)
    print("GAS FLARE AND INDUSTRIAL, A vs B, BOTH SPLITS")
    print("=" * 70)
    for split_method in SPLIT_METHODS:
        for model_name in models:
            a = all_results[(split_method, "A_11_features", model_name)]
            b = all_results[(split_method, "B_12_features_plus_frp_trend", model_name)]
            for cls in ("Industrial", "Gas_Flare"):
                pa, pb = a["per_class"][cls], b["per_class"][cls]
                print(f"  [{split_method}/{model_name}/{cls}] A: P={pa['precision']:.3f} R={pa['recall']:.3f} "
                      f"F1={pa['f1']:.3f} PR-AUC={a['pr_auc_ovr'][cls]:.3f}  ->  "
                      f"B: P={pb['precision']:.3f} R={pb['recall']:.3f} F1={pb['f1']:.3f} "
                      f"PR-AUC={b['pr_auc_ovr'][cls]:.3f}")

    write_metrics_csv(all_results)
    print(f"\nWrote {METRICS_CSV}")
    return all_results


def write_metrics_csv(all_results, path=METRICS_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for (split_method, feature_set_label, model_name), m in all_results.items():
        for c in m["classes"]:
            pc = m["per_class"][c]
            out_rows.append({
                "split": split_method, "feature_set": feature_set_label, "model": model_name, "class": c,
                "precision": pc["precision"], "recall": pc["recall"], "f1": pc["f1"],
                "support": pc["support"], "pr_auc_ovr": m["pr_auc_ovr"].get(c, ""),
                "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"],
                "macro_f1": m["macro_f1"], "n_train": m["n_train"], "n_test": m["n_test"],
            })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)


if __name__ == "__main__":
    main()
