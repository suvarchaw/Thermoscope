"""
LightGBM vs. Logistic Regression comparison on the 11-feature, non-leaky
event-behavior feature set (event_behavior_features.py). Answers ONE
question: does a small, fixed-configuration LightGBM model provide a
real, practically meaningful improvement over the already-validated
Logistic Regression baseline -- particularly for Gas Flare -- or not?

LIGHTGBM ENVIRONMENT FIX -- smallest, safest option found, no new
installation
-------------------------------------------------------------------------
`pip install lightgbm` succeeds (confirmed in the prior milestone), but
`import lightgbm` previously failed: its compiled binary needs the
OpenMP runtime (libomp.dylib), normally supplied by Homebrew, which is
not installed on this system. Homebrew was NOT installed to fix this
(explicitly not requested).

Instead: scikit-learn's own macOS wheel -- already an installed,
required project dependency, nothing new -- bundles its OWN copy of
libomp.dylib at
    <site-packages>/sklearn/.dylibs/libomp.dylib
Pointing LightGBM's dynamic loader at that existing file (via
DYLD_LIBRARY_PATH) makes `import lightgbm` succeed, verified directly.
No package was installed, no system tool was added -- this reuses a
library that was already on disk as a side effect of a dependency this
project already required.

The one wrinkle: macOS's dynamic linker only reads DYLD_LIBRARY_PATH at
process launch, not from a later `os.environ[...]` assignment inside a
running interpreter (verified: setting it after the interpreter starts
does NOT work). So this script re-executes itself ONCE via `os.execve`
with the variable set, before any lightgbm import -- a standard,
well-known technique (not a hack specific to this project) for exactly
this class of problem, letting the script still run as a plain
`python3 train_source_classifier_lightgbm.py` with no special
invocation required. If scikit-learn's bundled libomp.dylib is ever
moved/removed, `LIGHTGBM_AVAILABLE` below reports False and the script
proceeds without it, exactly as the prior milestone's fallback behavior did.

FEATURES / SPLIT / CLASSES -- all unchanged from the prior milestone
-------------------------------------------------------------------------
Reads data/processed/source_classifier_features_v2.csv (11 features,
already leakage-audited and unchanged here). Same temporal split
(train=2019-2022, test=2023-2025), same 4 trained classes (Brick_Kiln
and Unknown_Ambiguous excluded, asserted not just filtered), same
Logistic Regression configuration as the two prior milestones (for a
fair, apples-to-apples comparison). LightGBM uses a small, FIXED
configuration (n_estimators=100, max_depth=4, num_leaves=15,
learning_rate=0.1, class_weight="balanced") -- identical to the
configuration already committed in train_source_classifier.py BEFORE
this environment fix was found, i.e. chosen before any result was seen,
not tuned to it. No hyperparameter search is performed, and the test set
is never touched until final evaluation.

Does NOT: change event construction, silver-label rules, or the feature
set; add new features; invent Brick Kiln examples; create a risk score;
ingest 2026 data; or claim production readiness.
"""

import os
import sys

# --- self-reexec shim: must run before importing lightgbm (or anything
# that imports it) -- see module docstring for why. ---
_LIBOMP_DIR = ("/Library/Frameworks/Python.framework/Versions/3.14/"
               "lib/python3.14/site-packages/sklearn/.dylibs")
if os.environ.get("_THERMOSCOPE_LIBOMP_REEXEC") != "1" and os.path.isdir(_LIBOMP_DIR):
    _env = os.environ.copy()
    _existing = _env.get("DYLD_LIBRARY_PATH", "")
    _env["DYLD_LIBRARY_PATH"] = _LIBOMP_DIR + (":" + _existing if _existing else "")
    _env["_THERMOSCOPE_LIBOMP_REEXEC"] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, _env)

import csv
import json
import platform
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from event_behavior_features import (
    BASELINE_FEATURES_NUMERIC, BASELINE_FEATURES_CATEGORICAL,
    NEW_FEATURES_NUMERIC, NEW_FEATURES_CATEGORICAL,
    rows_to_arrays, combine_features, build_preprocessor, readable_feature_names,
)
from train_source_classifier import (
    TRAIN_YEARS, TEST_YEARS, RANDOM_STATE, TRAINED_CLASSES,
    filter_to_trained_classes, temporal_split, compute_metrics,
    assert_no_brick_kiln_in_training, assert_no_unknown_ambiguous,
)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    LIGHTGBM_IMPORT_ERROR = None
except Exception as e:  # ImportError or OSError (missing native lib)
    LIGHTGBM_AVAILABLE = False
    LIGHTGBM_IMPORT_ERROR = repr(e)

FEATURES_CSV = Path("data/processed/source_classifier_features_v2.csv")
METRICS_CSV = Path("data/processed/source_classifier_lightgbm_metrics.csv")
FIGURE_CONFUSION = Path("results/figures/source_classifier_lightgbm_confusion_matrix.png")
FIGURE_IMPORTANCE = Path("results/figures/source_classifier_lightgbm_feature_importance.png")
MODEL_DIR = Path("data/processed/models")

# Full 11-feature set (Set C from the prior milestone) -- unchanged.
FEATURES_NUMERIC = BASELINE_FEATURES_NUMERIC + NEW_FEATURES_NUMERIC
FEATURES_CATEGORICAL = BASELINE_FEATURES_CATEGORICAL + NEW_FEATURES_CATEGORICAL

# Fixed, documented, NOT tuned against the test set. Identical to the
# configuration already committed in train_source_classifier.py before
# this environment fix was found.
LIGHTGBM_PARAMS = dict(
    n_estimators=100, max_depth=4, num_leaves=15, learning_rate=0.1,
    class_weight="balanced", random_state=RANDOM_STATE, verbosity=-1,
)


def load_feature_rows(path=FEATURES_CSV):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    # temporal_split needs start_date; the v2 features file (deliberately,
    # per the leakage audit) does not carry it, so re-attach it from the
    # silver-label table by event_id, read-only, never used as a feature.
    with open("data/processed/gujarat_event_silver_labels.csv", newline="") as f:
        start_dates = {r["event_id"]: r["start_date"] for r in csv.DictReader(f)}
    for r in rows:
        r["start_date"] = start_dates[r["event_id"]]
    return rows


def build_logistic_pipeline():
    return Pipeline([
        ("preprocess", build_preprocessor(FEATURES_NUMERIC, FEATURES_CATEGORICAL)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ])


def build_lightgbm_pipeline():
    if not LIGHTGBM_AVAILABLE:
        raise RuntimeError(f"LightGBM is not available: {LIGHTGBM_IMPORT_ERROR}")
    from lightgbm import LGBMClassifier
    return Pipeline([
        ("preprocess", build_preprocessor(FEATURES_NUMERIC, FEATURES_CATEGORICAL)),
        ("clf", LGBMClassifier(**LIGHTGBM_PARAMS)),
    ])


def fit_and_evaluate(pipeline, X_train, y_train, X_test, y_test, classes):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    classes_order = list(pipeline.named_steps["clf"].classes_)
    y_prob_ordered = y_prob[:, [classes_order.index(c) for c in classes]]
    metrics = compute_metrics(y_test, y_pred, classes, y_prob_ordered)
    return metrics, y_pred, y_prob_ordered


def compute_permutation_importance(pipeline, X_test, y_test):
    perm = permutation_importance(pipeline, X_test, y_test, n_repeats=20,
                                   random_state=RANDOM_STATE, scoring="f1_macro")
    names = readable_feature_names(pipeline.named_steps["preprocess"], FEATURES_NUMERIC, FEATURES_CATEGORICAL)
    return sorted(zip(names, perm.importances_mean.tolist(), perm.importances_std.tolist()),
                  key=lambda x: -x[1])


def main():
    print("=" * 70)
    print("LIGHTGBM AVAILABILITY")
    print("=" * 70)
    if LIGHTGBM_AVAILABLE:
        print(f"LightGBM {lgb.__version__} imported successfully via scikit-learn's bundled "
              f"libomp.dylib (DYLD_LIBRARY_PATH self-reexec, no new install). "
              f"Re-exec active: {os.environ.get('_THERMOSCOPE_LIBOMP_REEXEC') == '1'}\n")
    else:
        print(f"LightGBM UNAVAILABLE: {LIGHTGBM_IMPORT_ERROR}")
        print("STOPPING per instruction: reporting the blocker, not substituting another model.\n")
        return None

    rows = load_feature_rows()
    trained_rows = filter_to_trained_classes(rows)
    assert_no_brick_kiln_in_training(trained_rows)
    assert_no_unknown_ambiguous(trained_rows)
    train_rows, test_rows = temporal_split(trained_rows)
    print(f"Train: {len(train_rows)} (years {sorted(TRAIN_YEARS)}), "
          f"Test: {len(test_rows)} (years {sorted(TEST_YEARS)})")
    print(f"Train class counts: {dict(Counter(r['silver_label'] for r in train_rows))}")
    print(f"Test class counts:  {dict(Counter(r['silver_label'] for r in test_rows))}")
    assert len(train_rows) == 6221 and len(test_rows) == 4414, \
        "split size does not match the preserved 2019-2022/2023-2025 split"

    Xn_train, Xc_train, y_train = rows_to_arrays(train_rows, FEATURES_NUMERIC, FEATURES_CATEGORICAL)
    Xn_test, Xc_test, y_test = rows_to_arrays(test_rows, FEATURES_NUMERIC, FEATURES_CATEGORICAL)
    X_train = combine_features(Xn_train, Xc_train)
    X_test = combine_features(Xn_test, Xc_test)
    classes = sorted(set(y_train) | set(y_test))

    print(f"\nFeatures ({len(FEATURES_NUMERIC) + len(FEATURES_CATEGORICAL)}): "
          f"{FEATURES_NUMERIC + FEATURES_CATEGORICAL}")
    print(f"LightGBM fixed config: {LIGHTGBM_PARAMS}\n")

    results = {}

    print("=" * 70)
    print("LOGISTIC REGRESSION (reproducing the prior milestone's Set C)")
    print("=" * 70)
    lr = build_logistic_pipeline()
    lr_metrics, y_pred_lr, y_prob_lr = fit_and_evaluate(lr, X_train, y_train, X_test, y_test, classes)
    results["logistic_regression"] = lr_metrics
    print(f"accuracy={lr_metrics['accuracy']:.3f} balanced_accuracy={lr_metrics['balanced_accuracy']:.3f} "
          f"macro_f1={lr_metrics['macro_f1']:.3f}")

    print("\n" + "=" * 70)
    print("LIGHTGBM")
    print("=" * 70)
    lgbm = build_lightgbm_pipeline()
    lgbm_metrics, y_pred_lgbm, y_prob_lgbm = fit_and_evaluate(lgbm, X_train, y_train, X_test, y_test, classes)
    results["lightgbm"] = lgbm_metrics
    print(f"accuracy={lgbm_metrics['accuracy']:.3f} balanced_accuracy={lgbm_metrics['balanced_accuracy']:.3f} "
          f"macro_f1={lgbm_metrics['macro_f1']:.3f}\n")

    for model_name, m in results.items():
        print(f"--- {model_name} per-class ---")
        for c in m["classes"]:
            pc = m["per_class"][c]
            print(f"  {c}: P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f} "
                  f"support={pc['support']} PR-AUC={m['pr_auc_ovr'].get(c, float('nan')):.3f}")
        print(f"  confusion_matrix (rows=true, cols=pred, order={m['classes']}):")
        for c, row in zip(m["classes"], m["confusion_matrix"]):
            print(f"    {c}: {row}")
        print()

    print("=" * 70)
    print("GAS FLARE HEAD-TO-HEAD")
    print("=" * 70)
    for model_name, m in results.items():
        pc = m["per_class"]["Gas_Flare"]
        print(f"  {model_name}: P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f} "
              f"PR-AUC={m['pr_auc_ovr']['Gas_Flare']:.3f}")

    print("\n" + "=" * 70)
    print("PERMUTATION IMPORTANCE ON HELD-OUT TEST SET")
    print("=" * 70)
    lr_perm = compute_permutation_importance(lr, X_test, y_test)
    lgbm_perm = compute_permutation_importance(lgbm, X_test, y_test)
    print("Logistic Regression:")
    for name, imp, std in lr_perm[:6]:
        print(f"  {name}: {imp:.4f} +/- {std:.4f}")
    print("LightGBM:")
    for name, imp, std in lgbm_perm[:6]:
        print(f"  {name}: {imp:.4f} +/- {std:.4f}")

    write_metrics_csv(results)
    print(f"\nWrote {METRICS_CSV}")

    plot_confusion_matrices(results)
    print(f"Saved figure to {FIGURE_CONFUSION}")

    plot_importance_comparison(lr_perm, lgbm_perm)
    print(f"Saved figure to {FIGURE_IMPORTANCE}")

    save_model_artifacts(lgbm, lgbm_metrics)
    print(f"\nSaved model artifacts to {MODEL_DIR}")

    return results, lr_perm, lgbm_perm


def write_metrics_csv(results, path=METRICS_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for model_name, m in results.items():
        for c in m["classes"]:
            pc = m["per_class"][c]
            out_rows.append({
                "model": model_name, "class": c,
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


def plot_confusion_matrices(results, output_path=FIGURE_CONFUSION):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = list(results.keys())
    classes = results[model_names[0]]["classes"]
    fig, axes = plt.subplots(1, len(model_names), figsize=(6.5 * len(model_names), 5.5))
    if len(model_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, model_names):
        cm = np.array(results[name]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
        ax.set_title(name, fontsize=10)
    fig.suptitle("LightGBM vs. Logistic Regression -- 11-feature set (silver labels, exploratory)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_importance_comparison(lr_perm, lgbm_perm, output_path=FIGURE_IMPORTANCE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, perm, title in zip(axes, (lr_perm, lgbm_perm),
                                ("Logistic Regression", "LightGBM")):
        names = [p[0] for p in perm][::-1]
        vals = [p[1] for p in perm][::-1]
        stds = [p[2] for p in perm][::-1]
        ax.barh(names, vals, xerr=stds, color="#2b6cb0")
        ax.set_title(f"{title} -- permutation importance (test set, macro F1 drop)", fontsize=9)
    fig.suptitle("Feature Importance -- NOT causal.")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_model_artifacts(lgbm_pipeline, lgbm_metrics, path=MODEL_DIR):
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(lgbm_pipeline, path / "source_classifier_lightgbm_v2.joblib")
    provenance = {
        "status": "EXPLORATORY ARTIFACT -- NOT PRODUCTION READY. Trained on SILVER labels, "
                   "not human-verified ground truth. Brick Kiln is not a supported class.",
        "trained_on": __import__("datetime").date.today().isoformat(),
        "sklearn_version": sklearn.__version__,
        "lightgbm_version": lgb.__version__,
        "python_version": platform.python_version(),
        "random_state": RANDOM_STATE,
        "trained_classes": TRAINED_CLASSES,
        "feature_columns_numeric": FEATURES_NUMERIC,
        "feature_columns_categorical": FEATURES_CATEGORICAL,
        "train_years": sorted(TRAIN_YEARS), "test_years": sorted(TEST_YEARS),
        "lightgbm_params": LIGHTGBM_PARAMS,
        "reload_requirement": "Requires scikit-learn's bundled libomp.dylib to be reachable via "
                               "DYLD_LIBRARY_PATH on macOS -- see this module's self-reexec shim. "
                               "Not required on Linux/Windows with a normal LightGBM install.",
        "test_macro_f1": lgbm_metrics["macro_f1"],
        "test_balanced_accuracy": lgbm_metrics["balanced_accuracy"],
        "test_gas_flare_pr_auc": lgbm_metrics["pr_auc_ovr"]["Gas_Flare"],
        "caveats": [
            "Labels are deterministic rule outputs (silver labels), not human-verified ground truth.",
            "Gas_Flare has very few examples -- treat its metrics as low-confidence.",
            "Fixed hyperparameters, not tuned -- see LIGHTGBM_PARAMS.",
            "Do not use for any production, deployment, or decision-making purpose.",
        ],
    }
    with open(path / "source_classifier_lightgbm_v2_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)


if __name__ == "__main__":
    main()
