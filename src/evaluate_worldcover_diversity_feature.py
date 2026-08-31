"""
Final planned feature-engineering evaluation: does `worldcover_diversity_300m`
improve genuine spatial generalization -- particularly for Industrial and
Gas Flare -- beyond the existing 11-feature model, on both the temporal
and spatial checkerboard splits?

Feature Set A = the current canonical 11-feature set
(train_source_classifier_lightgbm.FEATURES_NUMERIC/CATEGORICAL). Note:
`frp_trend_slope` was evaluated in the prior milestone and found to give
no meaningful spatial-generalization improvement -- it was NOT merged
into the canonical feature set (train_source_classifier_lightgbm.py is
unchanged), so Set A here is the same 11 features the classifier has
used throughout, not 12.

Feature Set B = A + `worldcover_diversity_300m` (event_worldcover_diversity_feature.py).

MISSING-VALUE TREATMENT
-------------------------------------------------------------------------
99.97% of events (20,402 of 20,409) have a computable value -- only 7
events (the same rare open-water tile edge case already documented in
event_evidence.py's `land_cover_class`) do not. Given how small this
fraction is, the 7 non-computable events are filled with 0.0 (not
paired with a separate computable-flag column this time -- proportionate
to a 0.03% edge case, unlike frp_trend_slope's 71.7% missingness in the
prior milestone, which genuinely needed one). This is documented here,
not silently done: the exact 7-event count is reported, and the choice
is a judgment call made explicit, not hidden.

Everything else -- silver labels, temporal split, spatial checkerboard
split, preprocessing, Logistic Regression configuration, LightGBM fixed
configuration, metrics, class weighting -- is imported UNCHANGED from
the modules that already define them. Nothing is tuned.

Does NOT: change label rules, event construction, split definitions, or
model configuration; investigate further features; ingest new data; or
build any inference/dashboard/deployment component.
"""

import csv
import sys
from collections import Counter
from pathlib import Path

from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Importing train_source_classifier_lightgbm triggers its libomp
# self-reexec shim (unchanged) before any lightgbm import in this process.
from train_source_classifier_lightgbm import (
    FEATURES_NUMERIC as SET_A_NUMERIC, FEATURES_CATEGORICAL as SET_A_CATEGORICAL,
    LIGHTGBM_AVAILABLE, LIGHTGBM_IMPORT_ERROR, LIGHTGBM_PARAMS,
)
from event_behavior_features import build_preprocessor, rows_to_arrays, combine_features, readable_feature_names
from train_source_classifier import (
    RANDOM_STATE, TRAIN_YEARS, TEST_YEARS,
    filter_to_trained_classes, temporal_split, compute_metrics,
    assert_no_brick_kiln_in_training, assert_no_unknown_ambiguous,
)
from spatial_evaluation import assign_spatial_split

FEATURES_V2_CSV = Path("data/processed/source_classifier_features_v2.csv")
DIVERSITY_CSV = Path("data/processed/event_worldcover_diversity_features.csv")
SILVER_LABELS_CSV = Path("data/processed/gujarat_event_silver_labels.csv")
METRICS_CSV = Path("data/processed/worldcover_diversity_evaluation_metrics.csv")

SET_B_NUMERIC = SET_A_NUMERIC + ["worldcover_diversity_300m"]
SET_B_CATEGORICAL = list(SET_A_CATEGORICAL)

FEATURE_SETS = {
    "A_11_features": (SET_A_NUMERIC, SET_A_CATEGORICAL),
    "B_12_features_plus_worldcover_diversity": (SET_B_NUMERIC, SET_B_CATEGORICAL),
}
SPLIT_METHODS = ["temporal", "spatial"]


def load_joined_rows():
    with open(FEATURES_V2_CSV, newline="") as f:
        base_rows = {r["event_id"]: r for r in csv.DictReader(f)}
    with open(DIVERSITY_CSV, newline="") as f:
        div_rows = {r["event_id"]: r for r in csv.DictReader(f)}
    with open(SILVER_LABELS_CSV, newline="") as f:
        silver_rows = {r["event_id"]: r for r in csv.DictReader(f)}

    n_not_computable = 0
    rows = []
    for eid, base in base_rows.items():
        div = div_rows[eid]
        computable = div["worldcover_diversity_computable"] == "True"
        if not computable:
            n_not_computable += 1
        row = dict(base)
        row["start_date"] = silver_rows[eid]["start_date"]
        row["worldcover_diversity_300m"] = float(div["worldcover_diversity_300m"]) if computable else 0.0
        rows.append(row)
    return rows, n_not_computable


def build_lightgbm_pipeline(numeric_cols, categorical_cols):
    if not LIGHTGBM_AVAILABLE:
        raise RuntimeError(f"LightGBM is not available: {LIGHTGBM_IMPORT_ERROR}")
    from lightgbm import LGBMClassifier
    return Pipeline([
        ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
        ("clf", LGBMClassifier(**LIGHTGBM_PARAMS)),
    ])


def build_logistic_pipeline_local(numeric_cols, categorical_cols):
    return Pipeline([
        ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ])


def get_split(rows, method):
    if method == "temporal":
        return temporal_split(rows)
    elif method == "spatial":
        train_rows, test_rows, _cell_report = assign_spatial_split(rows)
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
    rows, n_not_computable = load_joined_rows()
    trained_rows = filter_to_trained_classes(rows)
    assert_no_brick_kiln_in_training(trained_rows)
    assert_no_unknown_ambiguous(trained_rows)
    print(f"Loaded {len(trained_rows)} trained-class events. "
          f"worldcover_diversity_300m not computable for {n_not_computable} of 20,409 events "
          f"(filled with 0.0, documented, no separate flag -- see module docstring).\n")

    models = ["logistic_regression"] + (["lightgbm"] if LIGHTGBM_AVAILABLE else [])
    if not LIGHTGBM_AVAILABLE:
        print(f"LightGBM UNAVAILABLE: {LIGHTGBM_IMPORT_ERROR} -- reporting Logistic Regression only.\n")

    all_results = {}
    for split_method in SPLIT_METHODS:
        train_rows, test_rows = get_split(trained_rows, split_method)
        print("=" * 70)
        print(f"SPLIT: {split_method}  (train={len(train_rows)}, test={len(test_rows)})")
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
    print("A vs B DIRECT COMPARISON")
    print("=" * 70)
    for split_method in SPLIT_METHODS:
        for model_name in models:
            a = all_results[(split_method, "A_11_features", model_name)]
            b = all_results[(split_method, "B_12_features_plus_worldcover_diversity", model_name)]
            print(f"  [{split_method}/{model_name}] A: bal_acc={a['balanced_accuracy']:.3f} macro_f1={a['macro_f1']:.3f}  "
                  f"->  B: bal_acc={b['balanced_accuracy']:.3f} macro_f1={b['macro_f1']:.3f}")

    print("\n" + "=" * 70)
    print("INDUSTRIAL / GAS FLARE / CROP / FOREST, A vs B, BOTH SPLITS")
    print("=" * 70)
    for split_method in SPLIT_METHODS:
        for model_name in models:
            a = all_results[(split_method, "A_11_features", model_name)]
            b = all_results[(split_method, "B_12_features_plus_worldcover_diversity", model_name)]
            for cls in ("Industrial", "Gas_Flare", "Crop_Residue", "Forest_Wildfire"):
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
