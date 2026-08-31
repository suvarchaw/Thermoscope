"""
Supervised 4-class source classifier: can a model learn Industrial / Gas
Flare / Crop Residue / Forest-Wildfire from event features that are
INDEPENDENT of the deterministic silver-label rules
(src/build_event_silver_labels.py)?

BRICK KILN IS NOT A TRAINED CLASS. It has zero real silver-labeled
examples (data/processed/gujarat_event_silver_labels.csv) -- no example
is invented here; the 4-class filter below structurally excludes it, and
`assert_no_brick_kiln_in_training` checks this explicitly at runtime, not
just by construction.

UNKNOWN_AMBIGUOUS EVENTS ARE EXCLUDED. Both the "insufficient_evidence"
and "conflicting_evidence" events (9,774 of 20,409) are silver-label
outputs with no confident single-class assignment -- training on them
(with any label) would mean training on noise the label-generation step
itself already flagged as unreliable.

FEATURE ALLOWLIST -- see FEATURE_AUDIT below for the full per-column
trace (name, source, meaning, availability at inference time, whether a
silver-label rule reads it, leakage decision, reason). Two columns that
the PRIOR milestone's audit called "safe" (not directly read by any
rule) were re-examined here with an empirical correlation check, per
this milestone's explicit instruction not to trust a column's safety
just because its name sounds harmless:
  - max_frp correlates r=0.883 with mean_frp (directly read by the Gas
    Flare rule's radiometric condition) on the 4-class subset.
  - detection_count correlates r=0.921 with duration_days (directly read
    by the Brick Kiln/Crop Residue/Forest-Wildfire rules' short-duration
    condition) on the 4-class subset.
Both are strong enough (r>0.85) that either could let a model
substantially reconstruct a rule threshold through the back door, so
both are EXCLUDED here -- the same conservative standard this project
already applied to mean_frp itself in the prior milestone ("excluded
globally, not just for the one rule that reads it"). This leaves a
smaller, but genuinely audited-independent, feature set than a naive
read of "not directly used by any rule" would have kept:

    FEATURE_ALLOWLIST = ["centroid_lat", "centroid_lon",
                          "spatial_extent_m", "status"]

`status` is near-constant in the 4-class subset (10,631 of 10,635
"closed") -- kept for completeness/documentation, contributes
negligible information, not removed to avoid silently narrowing the
allowlist further without a documented reason.

TEMPORAL SPLIT: train = events starting 2019-2022, test = events
starting 2023-2025 (no separate validation split -- see module docstring
section below for why). The test period is never touched until final
evaluation; no hyperparameter tuning is performed against it (LightGBM
uses small, fixed, documented parameters, not tuned at all).

This module does NOT: build a dashboard, ingest 2026 data, invent Brick
Kiln examples, change the silver-label rules or event-construction
methodology, create a risk score, make causal claims, or claim
production readiness.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.inspection import permutation_importance

SOURCE_CSV = Path("data/processed/gujarat_event_silver_labels.csv")
FEATURES_CSV = Path("data/processed/source_classifier_features.csv")
PREDICTIONS_CSV = Path("data/processed/source_classifier_predictions.csv")
METRICS_CSV = Path("data/processed/source_classifier_metrics.csv")
FIGURE_CONFUSION = Path("results/figures/source_classifier_confusion_matrix.png")
FIGURE_IMPORTANCE = Path("results/figures/source_classifier_feature_importance.png")
MODEL_DIR = Path("data/processed/models")

RANDOM_STATE = 42

TRAINED_CLASSES = ["Industrial", "Gas_Flare", "Crop_Residue", "Forest_Wildfire"]
EXCLUDED_LABELS = {"Unknown_Ambiguous", "Brick_Kiln"}  # Brick_Kiln has 0 rows; asserted, not assumed

TRAIN_YEARS = {2019, 2020, 2021, 2022}
TEST_YEARS = {2023, 2024, 2025}

# ---------------------------------------------------------------------
# FEATURE AUDIT -- one row per column present in gujarat_event_silver_
# labels.csv, documented per the milestone brief's required fields.
# ---------------------------------------------------------------------
FEATURE_AUDIT = [
    {"name": "centroid_lat", "source": "event_construction.py (mean detection latitude)",
     "meaning": "Event centroid latitude", "available_at_inference": True,
     "used_in_label_rule": False, "leakage_decision": "SAFE", "included": True,
     "reason": "Not read by any labeling rule; raw location is available the moment an event closes."},
    {"name": "centroid_lon", "source": "event_construction.py (mean detection longitude)",
     "meaning": "Event centroid longitude", "available_at_inference": True,
     "used_in_label_rule": False, "leakage_decision": "SAFE", "included": True,
     "reason": "Not read by any labeling rule; raw location is available the moment an event closes."},
    {"name": "spatial_extent_m", "source": "event_construction.py (max detection-to-centroid haversine distance)",
     "meaning": "Event's spatial footprint radius", "available_at_inference": True,
     "used_in_label_rule": False, "leakage_decision": "SAFE", "included": True,
     "reason": "Not read by any labeling rule; physically meaningful (point-like vs. sprawling event)."},
    {"name": "status", "source": "event_construction.py (provisional/closed vs. reference_date)",
     "meaning": "Whether the event is confirmed closed", "available_at_inference": True,
     "used_in_label_rule": False, "leakage_decision": "SAFE (low information)", "included": True,
     "reason": "Not read by any labeling rule. Near-constant in the 4-class subset "
               "(10,631/10,635 closed) -- kept for completeness, not expected to be informative."},
    {"name": "detection_count", "source": "event_construction.py (count of member FIRMS detections)",
     "meaning": "Number of detections composing the event", "available_at_inference": True,
     "used_in_label_rule": False, "leakage_decision": "EXCLUDED (correlated proxy)", "included": False,
     "reason": "r=0.921 with duration_days (directly read by 3 of 5 rules) on the 4-class subset -- "
               "strong enough to reconstruct the short-duration rule condition through the back door."},
    {"name": "max_frp", "source": "event_construction.py (max FRP among member detections)",
     "meaning": "Peak fire radiative power in the event", "available_at_inference": True,
     "used_in_label_rule": False, "leakage_decision": "EXCLUDED (correlated proxy)", "included": False,
     "reason": "r=0.883 with mean_frp (directly read by the Gas Flare radiometric condition) on the "
               "4-class subset -- same conservative standard already applied to mean_frp itself."},
    {"name": "mean_frp", "source": "event_construction.py (mean FRP among member detections)",
     "meaning": "Average fire radiative power in the event", "available_at_inference": True,
     "used_in_label_rule": True, "leakage_decision": "EXCLUDED (label-generating)", "included": False,
     "reason": "Directly read by the Gas Flare rule's radiometric OR-condition."},
    {"name": "night_fraction", "source": "event_construction.py (fraction of detections flagged daynight=N)",
     "meaning": "Share of the event's detections at night", "available_at_inference": True,
     "used_in_label_rule": True, "leakage_decision": "EXCLUDED (label-generating)", "included": False,
     "reason": "Directly read by the Industrial rule's night-detections condition."},
    {"name": "duration_days", "source": "event_construction.py (end_date - start_date + 1)",
     "meaning": "Event duration in days", "available_at_inference": False,
     "used_in_label_rule": True, "leakage_decision": "EXCLUDED (label-generating, retrospective)", "included": False,
     "reason": "Directly read by 3 of 5 rules. Also NOT available at the moment an event is still "
               "open/provisional -- a genuinely retrospective field, doubly excluded."},
    {"name": "start_date", "source": "event_construction.py (earliest member detection date)",
     "meaning": "Event start date", "available_at_inference": True,
     "used_in_label_rule": True, "leakage_decision": "EXCLUDED (label-generating)", "included": False,
     "reason": "Month-of-year is directly read by 3 of 5 rules (seasonal windows). Used ONLY to "
               "construct the temporal train/test split below, never as a model feature."},
    {"name": "end_date", "source": "event_construction.py (latest member detection date)",
     "meaning": "Event end date", "available_at_inference": False,
     "used_in_label_rule": False, "leakage_decision": "EXCLUDED (reconstructs duration_days)", "included": False,
     "reason": "Combined with start_date this reconstructs duration_days exactly; excluding only "
               "duration_days while keeping both dates would defeat that exclusion. Also retrospective "
               "for a still-open event."},
    {"name": "silver_label", "source": "build_event_silver_labels.py",
     "meaning": "The target itself", "available_at_inference": False,
     "used_in_label_rule": True, "leakage_decision": "TARGET, not a feature", "included": False,
     "reason": "This is what the model predicts."},
    {"name": "label_rule_id", "source": "build_event_silver_labels.py",
     "meaning": "Which rule fired", "available_at_inference": False,
     "used_in_label_rule": True, "leakage_decision": "EXCLUDED (direct label field)", "included": False,
     "reason": "Explicitly named in the milestone brief as never to be given to the model."},
    {"name": "label_evidence", "source": "build_event_silver_labels.py",
     "meaning": "Human-readable justification string", "available_at_inference": False,
     "used_in_label_rule": True, "leakage_decision": "EXCLUDED (direct label field)", "included": False,
     "reason": "Explicitly named in the milestone brief as never to be given to the model."},
    {"name": "conflict_classes", "source": "build_event_silver_labels.py",
     "meaning": "Every class that matched", "available_at_inference": False,
     "used_in_label_rule": True, "leakage_decision": "EXCLUDED (direct label field)", "included": False,
     "reason": "Explicitly named in the milestone brief as never to be given to the model."},
    {"name": "n_classes_matched", "source": "build_event_silver_labels.py",
     "meaning": "Count of matching rules", "available_at_inference": False,
     "used_in_label_rule": True, "leakage_decision": "EXCLUDED (direct label field)", "included": False,
     "reason": "Explicitly named in the milestone brief as never to be given to the model."},
    {"name": "ambiguity_reason", "source": "build_event_silver_labels.py",
     "meaning": "Why an event is Unknown/Ambiguous", "available_at_inference": False,
     "used_in_label_rule": True, "leakage_decision": "EXCLUDED (direct label field)", "included": False,
     "reason": "Directly derived from the label outcome."},
    {"name": "excluded_from_training", "source": "build_event_silver_labels.py",
     "meaning": "Whether the label is Unknown/Ambiguous", "available_at_inference": False,
     "used_in_label_rule": True, "leakage_decision": "EXCLUDED (direct label field)", "included": False,
     "reason": "Directly derived from the label outcome; also used to filter rows, never as a feature."},
]

FEATURE_ALLOWLIST_NUMERIC = ["centroid_lat", "centroid_lon", "spatial_extent_m"]
FEATURE_ALLOWLIST_CATEGORICAL = ["status"]
FEATURE_ALLOWLIST = FEATURE_ALLOWLIST_NUMERIC + FEATURE_ALLOWLIST_CATEGORICAL

# Every column name the model must NEVER see, regardless of allowlist bugs.
FORBIDDEN_COLUMNS = {
    "silver_label", "label_rule_id", "label_evidence", "conflict_classes",
    "n_classes_matched", "ambiguity_reason", "excluded_from_training",
    "mean_frp", "night_fraction", "duration_days", "start_date", "end_date",
    "max_frp", "detection_count", "event_id",
}


def assert_feature_allowlist_clean():
    """Runtime safety check, not just documentation: the allowlist must
    never overlap FORBIDDEN_COLUMNS, and every FEATURE_AUDIT row marked
    included=True must be in the allowlist (and vice versa)."""
    overlap = set(FEATURE_ALLOWLIST) & FORBIDDEN_COLUMNS
    if overlap:
        raise ValueError(f"Forbidden columns present in feature allowlist: {overlap}")
    audited_included = {r["name"] for r in FEATURE_AUDIT if r["included"]}
    if audited_included != set(FEATURE_ALLOWLIST):
        raise ValueError(
            f"FEATURE_AUDIT included-set {audited_included} does not match "
            f"FEATURE_ALLOWLIST {set(FEATURE_ALLOWLIST)}"
        )


def assert_no_brick_kiln_in_training(rows):
    labels = {r["silver_label"] for r in rows}
    if "Brick_Kiln" in labels:
        raise ValueError("Brick_Kiln rows present in training data -- must be excluded.")


def assert_no_unknown_ambiguous(rows):
    labels = {r["silver_label"] for r in rows}
    if "Unknown_Ambiguous" in labels:
        raise ValueError("Unknown_Ambiguous rows present in training data -- must be excluded.")


def load_source_rows(path=SOURCE_CSV):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def filter_to_trained_classes(rows):
    return [r for r in rows if r["silver_label"] in TRAINED_CLASSES]


def year_of(row):
    return date.fromisoformat(row["start_date"]).year


def temporal_split(rows):
    train = [r for r in rows if year_of(r) in TRAIN_YEARS]
    test = [r for r in rows if year_of(r) in TEST_YEARS]
    assert len(train) + len(test) == len(rows), "every row must fall in TRAIN_YEARS or TEST_YEARS"
    return train, test


def rows_to_arrays(rows):
    X_numeric = np.array([[float(r[c]) for c in FEATURE_ALLOWLIST_NUMERIC] for r in rows], dtype=float)
    X_categorical = np.array([[r[c] for c in FEATURE_ALLOWLIST_CATEGORICAL] for r in rows], dtype=object)
    y = np.array([r["silver_label"] for r in rows])
    return X_numeric, X_categorical, y


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), list(range(len(FEATURE_ALLOWLIST_NUMERIC)))),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         list(range(len(FEATURE_ALLOWLIST_NUMERIC), len(FEATURE_ALLOWLIST_NUMERIC) + len(FEATURE_ALLOWLIST_CATEGORICAL)))),
    ])


def combine_features(X_numeric, X_categorical):
    return np.hstack([X_numeric.astype(object), X_categorical])


def readable_feature_names(preprocessor):
    names = list(FEATURE_ALLOWLIST_NUMERIC)
    onehot = preprocessor.named_transformers_["cat"]
    for col_name, categories in zip(FEATURE_ALLOWLIST_CATEGORICAL, onehot.categories_):
        for cat in categories:
            names.append(f"{col_name}={cat}")
    return names


def fit_majority_baseline(y_train):
    return Counter(y_train).most_common(1)[0][0]


def compute_metrics(y_true, y_pred, classes, y_prob=None):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    per_class = {
        c: {"precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "support": int(support[i])}
        for i, c in enumerate(classes)
    }
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    pr_auc = {}
    if y_prob is not None:
        for i, c in enumerate(classes):
            y_true_binary = (y_true == c).astype(int)
            pr_auc[c] = float(average_precision_score(y_true_binary, y_prob[:, i]))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p), "macro_recall": float(macro_r), "macro_f1": float(macro_f1),
        "per_class": per_class, "confusion_matrix": cm.tolist(), "classes": classes,
        "pr_auc_ovr": pr_auc,
    }


def build_logistic_pipeline():
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                    random_state=RANDOM_STATE)),
    ])


def build_lightgbm_pipeline():
    from lightgbm import LGBMClassifier
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", LGBMClassifier(
            n_estimators=100, max_depth=4, num_leaves=15, learning_rate=0.1,
            class_weight="balanced", random_state=RANDOM_STATE, verbosity=-1,
        )),
    ])


def main():
    assert_feature_allowlist_clean()
    print(f"Feature allowlist verified clean: {len(FEATURE_ALLOWLIST)} features "
          f"({FEATURE_ALLOWLIST}), zero overlap with {len(FORBIDDEN_COLUMNS)} forbidden columns.\n")

    rows = load_source_rows()
    print(f"Loaded {len(rows)} events from {SOURCE_CSV} (read-only).")

    trained_rows = filter_to_trained_classes(rows)
    assert_no_brick_kiln_in_training(trained_rows)
    assert_no_unknown_ambiguous(trained_rows)
    print(f"Filtered to {len(trained_rows)} events in the 4 trained classes "
          f"(excluded {len(rows) - len(trained_rows)} Unknown_Ambiguous/Brick_Kiln rows).\n")

    train_rows, test_rows = temporal_split(trained_rows)
    print(f"Temporal split: train={len(train_rows)} (years {sorted(TRAIN_YEARS)}), "
          f"test={len(test_rows)} (years {sorted(TEST_YEARS)})")
    print(f"Train class counts: {dict(Counter(r['silver_label'] for r in train_rows))}")
    print(f"Test class counts:  {dict(Counter(r['silver_label'] for r in test_rows))}\n")

    Xn_train, Xc_train, y_train = rows_to_arrays(train_rows)
    Xn_test, Xc_test, y_test = rows_to_arrays(test_rows)
    X_train = combine_features(Xn_train, Xc_train)
    X_test = combine_features(Xn_test, Xc_test)

    classes = sorted(set(y_train) | set(y_test))

    results = {}

    majority_class = fit_majority_baseline(y_train)
    y_pred_baseline = np.full(len(y_test), majority_class)
    results["baseline"] = compute_metrics(y_test, y_pred_baseline, classes)
    print(f"Majority baseline class: {majority_class}")
    print(f"Baseline: accuracy={results['baseline']['accuracy']:.3f} "
          f"balanced_accuracy={results['baseline']['balanced_accuracy']:.3f}\n")

    lr = build_logistic_pipeline()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)
    lr_classes_order = list(lr.named_steps["clf"].classes_)
    y_prob_lr_ordered = y_prob_lr[:, [lr_classes_order.index(c) for c in classes]]
    results["logistic_regression"] = compute_metrics(y_test, y_pred_lr, classes, y_prob_lr_ordered)
    print(f"Logistic Regression: accuracy={results['logistic_regression']['accuracy']:.3f} "
          f"balanced_accuracy={results['logistic_regression']['balanced_accuracy']:.3f} "
          f"macro_f1={results['logistic_regression']['macro_f1']:.3f}")

    lightgbm_error = None
    try:
        import lightgbm  # noqa: F401
        lightgbm_available = True
    except Exception as e:  # ImportError (missing package) or OSError (missing native runtime lib)
        lightgbm_available = False
        lightgbm_error = repr(e)

    lgbm = None
    if lightgbm_available:
        lgbm = build_lightgbm_pipeline()
        lgbm.fit(X_train, y_train)
        y_pred_lgbm = lgbm.predict(X_test)
        y_prob_lgbm = lgbm.predict_proba(X_test)
        lgbm_classes_order = list(lgbm.named_steps["clf"].classes_)
        y_prob_lgbm_ordered = y_prob_lgbm[:, [lgbm_classes_order.index(c) for c in classes]]
        results["lightgbm"] = compute_metrics(y_test, y_pred_lgbm, classes, y_prob_lgbm_ordered)
        print(f"LightGBM: accuracy={results['lightgbm']['accuracy']:.3f} "
              f"balanced_accuracy={results['lightgbm']['balanced_accuracy']:.3f} "
              f"macro_f1={results['lightgbm']['macro_f1']:.3f}\n")
    else:
        print("LightGBM is NOT usable in this environment -- reporting the dependency gap, "
              "NOT silently substituting another algorithm as the 'primary model'.")
        print(f"  Underlying error: {lightgbm_error}")
        print("  Diagnosis: the lightgbm Python package installs cleanly via pip (confirmed), but "
              "its compiled binary requires the OpenMP runtime (libomp.dylib) on macOS, which is "
              "normally provided by Homebrew. Homebrew itself is not installed on this system "
              "(`brew` not found), and installing a new system-wide package manager is a materially "
              "larger, less reversible change than a pip install -- not attempted here without "
              "explicit approval. Logistic Regression above remains the best available signal check "
              "for this milestone.\n")

    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "confusion_matrix"}
                       for k, v in results.items()}, indent=2, default=str))

    write_features_csv(trained_rows)
    print(f"\nWrote {FEATURES_CSV}")

    write_predictions_csv(test_rows, y_test, y_pred_lr, y_pred_lgbm if lgbm is not None else None)
    print(f"Wrote {PREDICTIONS_CSV}")

    write_metrics_csv(results)
    print(f"Wrote {METRICS_CSV}")

    plot_confusion_matrices(results, classes)
    print(f"Saved figure to {FIGURE_CONFUSION}")

    if lgbm is not None:
        plot_feature_importance(lgbm, X_test, y_test, classes, model_label="LightGBM (built-in split importance)")
        print(f"Saved figure to {FIGURE_IMPORTANCE}")
        save_model_artifacts(lgbm, results)
        print(f"\nSaved model artifacts to {MODEL_DIR}")
    else:
        # LightGBM unusable in this environment (see diagnosis above) --
        # still deliver the required feature-importance figure using the
        # interpretable baseline that IS available, clearly labeled as
        # such rather than silently skipped.
        plot_feature_importance(lr, X_test, y_test, classes,
                                 model_label="Logistic Regression (|coefficient|) -- LightGBM unavailable, see report")
        print(f"Saved figure to {FIGURE_IMPORTANCE} (Logistic-Regression-based; LightGBM unavailable)")

    return results


def write_features_csv(rows, path=FEATURES_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["event_id", "silver_label"] + FEATURE_ALLOWLIST
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})


def write_predictions_csv(test_rows, y_test, y_pred_lr, y_pred_lgbm, path=PREDICTIONS_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["event_id", "start_date", "actual_label", "lr_predicted"]
        if y_pred_lgbm is not None:
            header.append("lightgbm_predicted")
        writer.writerow(header)
        for i, r in enumerate(test_rows):
            row = [r["event_id"], r["start_date"], y_test[i], y_pred_lr[i]]
            if y_pred_lgbm is not None:
                row.append(y_pred_lgbm[i])
            writer.writerow(row)


def write_metrics_csv(results, path=METRICS_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name, m in results.items():
        for c in m["classes"]:
            pc = m["per_class"][c]
            rows.append({
                "model": model_name, "class": c,
                "precision": pc["precision"], "recall": pc["recall"], "f1": pc["f1"],
                "support": pc["support"], "pr_auc_ovr": m["pr_auc_ovr"].get(c, ""),
                "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"],
                "macro_f1": m["macro_f1"],
            })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion_matrices(results, classes, output_path=FIGURE_CONFUSION):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = [k for k in ("baseline", "logistic_regression", "lightgbm") if k in results]
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5))
    if len(model_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, model_names):
        cm = np.array(results[name]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
        ax.set_title(name, fontsize=10)
    fig.suptitle("Confusion Matrices -- 4-Class Source Classifier (silver labels, exploratory)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(pipeline, X_test, y_test, classes, model_label, output_path=FIGURE_IMPORTANCE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preprocessor = pipeline.named_steps["preprocess"]
    clf = pipeline.named_steps["clf"]
    names = readable_feature_names(preprocessor)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        # LogisticRegression: mean |coefficient| across the one-vs-rest classes
        importances = np.abs(clf.coef_).mean(axis=0)
    order = np.argsort(importances)[::-1]
    axes[0].barh([names[i] for i in order][::-1], [importances[i] for i in order][::-1], color="#2b6cb0")
    axes[0].set_title(f"{model_label} -- built-in importance", fontsize=10)

    perm = permutation_importance(pipeline, X_test, y_test, n_repeats=20,
                                   random_state=RANDOM_STATE, scoring="f1_macro")
    perm_order = np.argsort(perm.importances_mean)[::-1]
    axes[1].barh([names[i] for i in perm_order][::-1],
                 [perm.importances_mean[i] for i in perm_order][::-1],
                 xerr=[perm.importances_std[i] for i in perm_order][::-1], color="#c53030")
    axes[1].set_title("Permutation importance on held-out test set (macro F1 drop)", fontsize=10)

    fig.suptitle(f"Feature Importance ({model_label}) -- NOT causal. See report for proxy-feature check.")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_model_artifacts(lgbm_pipeline, results, path=MODEL_DIR):
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(lgbm_pipeline, path / "source_classifier_lightgbm.joblib")
    provenance = {
        "status": "EXPLORATORY ARTIFACT -- NOT PRODUCTION READY. Trained on SILVER labels, "
                   "not human-verified ground truth. Brick Kiln is not a supported class.",
        "trained_on": date.today().isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "random_state": RANDOM_STATE,
        "trained_classes": TRAINED_CLASSES,
        "feature_allowlist_numeric": FEATURE_ALLOWLIST_NUMERIC,
        "feature_allowlist_categorical": FEATURE_ALLOWLIST_CATEGORICAL,
        "train_years": sorted(TRAIN_YEARS), "test_years": sorted(TEST_YEARS),
        "lightgbm_params": {"n_estimators": 100, "max_depth": 4, "num_leaves": 15,
                             "learning_rate": 0.1, "class_weight": "balanced"},
        "test_macro_f1": results["lightgbm"]["macro_f1"],
        "test_balanced_accuracy": results["lightgbm"]["balanced_accuracy"],
        "caveats": [
            "Labels are deterministic rule outputs (silver labels), not human-verified ground truth.",
            "Gas_Flare has very few examples (see METRICS_CSV per-class support) -- treat its "
            "metrics as low-confidence.",
            "Feature set is deliberately narrow (4 features) after excluding two candidates found "
            "to correlate strongly with label-generating fields -- see FEATURE_AUDIT.",
            "Do not use for any production, deployment, or decision-making purpose.",
        ],
    }
    with open(path / "source_classifier_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)


if __name__ == "__main__":
    main()
