"""
Spatial-holdout evaluation: does the LightGBM/Logistic-Regression source
classifier generalize to genuinely unseen geographic areas, or does its
strong temporal-split performance (balanced accuracy 0.935, macro F1
0.910 -- train_source_classifier_lightgbm.py) substantially reflect
recognizing already-known, spatially-fixed source locations?

That prior milestone found 99.0% of test Industrial events and ~85% of
test Gas_Flare events sit within 300m of a same-class TRAINING event
(the temporal split only withholds later YEARS, not different PLACES,
and Industrial/Gas Flare sources are physically fixed infrastructure
that simply recurs year after year). This module answers the follow-up
question directly: hold out entire geographic areas instead of years,
and see how much performance survives.

SPATIAL PARTITION METHOD -- simple, reproducible, label-independent
-------------------------------------------------------------------------
Gujarat's existing bounding box (spatial_recurrence.LAT_MIN/LON_MIN,
unchanged) is divided into a fixed 0.5-degree x 0.5-degree grid
(~55km x 51km cells at this latitude -- large relative to every event's
spatial_extent_m, which is at most a few km, so no event straddles a
cell boundary in any way that matters). Each occupied cell is assigned
to TRAIN or TEST by a checkerboard rule on its own grid indices,
(row + col) % 2 -- computed ONLY from centroid_lat/centroid_lon, before
silver_label is ever consulted, so the split cannot be biased toward or
away from any class. This is the label-independent requirement, enforced
by construction (`assign_spatial_split` never reads `silver_label`), not
just by intent.

No event's geographic cell is ever split across train and test -- an
entire cell goes to one side or the other, which is what "areas
completely withheld from training" means operationally here.

REUSES, NOT REIMPLEMENTS: the same 11-feature allowlist, the same fixed
LightGBM configuration, and the same Logistic Regression configuration
already committed in train_source_classifier_lightgbm.py -- imported
directly. The existing temporal split/evaluation is not modified, rerun,
or affected in any way; its already-reported numbers are only quoted for
comparison.

Does NOT: change silver-label rules, event construction, or the feature
allowlist; add features; tune LightGBM; invent labels; create a risk
score; or attempt to improve the model after seeing these results
(evaluation only, per the milestone brief).
"""

import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spatial_recurrence import LAT_MIN, LON_MIN
from event_behavior_features import rows_to_arrays, combine_features, haversine_m
from train_source_classifier import (
    compute_metrics, filter_to_trained_classes,
    assert_no_brick_kiln_in_training, assert_no_unknown_ambiguous, TRAINED_CLASSES,
)
from train_source_classifier_lightgbm import (
    FEATURES_NUMERIC, FEATURES_CATEGORICAL, FEATURES_CSV,
    build_logistic_pipeline, build_lightgbm_pipeline, LIGHTGBM_AVAILABLE, LIGHTGBM_IMPORT_ERROR,
    LIGHTGBM_PARAMS,
)

CELL_SIZE_DEG = 0.5
NEIGHBOR_DIAGNOSTIC_THRESHOLD_M = 300.0

METRICS_CSV = Path("data/processed/spatial_evaluation_metrics.csv")
SPLIT_REPORT_CSV = Path("data/processed/spatial_evaluation_split_report.csv")
FIGURE_CONFUSION = Path("results/figures/spatial_evaluation_confusion_matrix.png")

# Already-reported temporal-split results (train_source_classifier_lightgbm.py,
# prior milestone) -- quoted here for direct comparison only, NOT recomputed.
TEMPORAL_RESULTS = {
    "logistic_regression": {"accuracy": 0.728, "balanced_accuracy": 0.707, "macro_f1": 0.598},
    "lightgbm": {"accuracy": 0.902, "balanced_accuracy": 0.935, "macro_f1": 0.910},
}


def cell_of(lat, lon, cell_size_deg=CELL_SIZE_DEG):
    return (int((lat - LAT_MIN) // cell_size_deg), int((lon - LON_MIN) // cell_size_deg))


def assign_spatial_split(rows, cell_size_deg=CELL_SIZE_DEG):
    """Checkerboard grid split, computed ONLY from centroid_lat/centroid_lon
    -- never reads silver_label, so the partition cannot be biased by
    class. Returns (train_rows, test_rows, cell_report) where cell_report
    maps each occupied cell to its assignment and per-class counts."""
    cell_report = {}
    for r in rows:
        c = cell_of(float(r["centroid_lat"]), float(r["centroid_lon"]), cell_size_deg)
        if c not in cell_report:
            parity = (c[0] + c[1]) % 2
            cell_report[c] = {"assignment": "test" if parity == 1 else "train", "counts": Counter()}
        cell_report[c]["counts"][r["silver_label"]] += 1

    train_rows, test_rows = [], []
    for r in rows:
        c = cell_of(float(r["centroid_lat"]), float(r["centroid_lon"]), cell_size_deg)
        if cell_report[c]["assignment"] == "train":
            train_rows.append(r)
        else:
            test_rows.append(r)
    return train_rows, test_rows, cell_report


def nearest_same_class_train_distance(test_rows, train_rows):
    """DIAGNOSTIC ONLY -- never used as a model feature or to filter/relabel
    events. For each test event, the haversine distance (meters) to the
    nearest TRAIN event of the SAME class, or None if that class has no
    training examples at all."""
    by_class = defaultdict(list)
    for r in train_rows:
        by_class[r["silver_label"]].append((float(r["centroid_lat"]), float(r["centroid_lon"])))

    distances = {}
    for r in test_rows:
        candidates = by_class.get(r["silver_label"], [])
        if not candidates:
            distances[r["event_id"]] = None
            continue
        tlat, tlon = float(r["centroid_lat"]), float(r["centroid_lon"])
        distances[r["event_id"]] = min(haversine_m(tlat, tlon, clat, clon) for clat, clon in candidates)
    return distances


def summarize_diagnostic(test_rows, distances, threshold_m=NEIGHBOR_DIAGNOSTIC_THRESHOLD_M):
    by_class = defaultdict(list)
    for r in test_rows:
        d = distances[r["event_id"]]
        if d is not None:
            by_class[r["silver_label"]].append(d)

    summary = {}
    for cls, dists in by_class.items():
        dists_sorted = sorted(dists)
        n = len(dists_sorted)
        within = sum(1 for d in dists_sorted if d <= threshold_m)
        summary[cls] = {
            "n_with_same_class_train_example": n,
            "min_m": dists_sorted[0], "median_m": statistics.median(dists_sorted), "max_m": dists_sorted[-1],
            "within_300m": within, "within_300m_pct": within / n * 100 if n else 0.0,
        }
    return summary


def run_model(pipeline, train_rows, test_rows, classes):
    Xn_train, Xc_train, y_train = rows_to_arrays(train_rows, FEATURES_NUMERIC, FEATURES_CATEGORICAL)
    Xn_test, Xc_test, y_test = rows_to_arrays(test_rows, FEATURES_NUMERIC, FEATURES_CATEGORICAL)
    X_train = combine_features(Xn_train, Xc_train)
    X_test = combine_features(Xn_test, Xc_test)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    classes_order = list(pipeline.named_steps["clf"].classes_)
    y_prob_ordered = y_prob[:, [classes_order.index(c) for c in classes]]
    return compute_metrics(y_test, y_pred, classes, y_prob_ordered)


def main():
    rows = load_rows()
    trained_rows = filter_to_trained_classes(rows)
    assert_no_brick_kiln_in_training(trained_rows)
    assert_no_unknown_ambiguous(trained_rows)

    train_rows, test_rows, cell_report = assign_spatial_split(trained_rows)

    print("=" * 70)
    print("SPATIAL PARTITION")
    print("=" * 70)
    print(f"Grid cell size: {CELL_SIZE_DEG} degrees")
    print(f"Occupied cells: {len(cell_report)} "
          f"(train: {sum(1 for v in cell_report.values() if v['assignment']=='train')}, "
          f"test: {sum(1 for v in cell_report.values() if v['assignment']=='test')})")
    print(f"Train events: {len(train_rows)}, Test events: {len(test_rows)}")
    print(f"Train class counts: {dict(Counter(r['silver_label'] for r in train_rows))}")
    print(f"Test class counts:  {dict(Counter(r['silver_label'] for r in test_rows))}")
    write_split_report(cell_report)
    print(f"Wrote {SPLIT_REPORT_CSV}\n")

    print("=" * 70)
    print("GEOGRAPHIC-NEAREST-NEIGHBOR DIAGNOSTIC (evaluation only, not a feature)")
    print("=" * 70)
    distances = nearest_same_class_train_distance(test_rows, train_rows)
    diag = summarize_diagnostic(test_rows, distances)
    for cls in TRAINED_CLASSES:
        if cls in diag:
            d = diag[cls]
            print(f"  {cls}: n={d['n_with_same_class_train_example']} min={d['min_m']:.0f}m "
                  f"median={d['median_m']:.0f}m max={d['max_m']:.0f}m "
                  f"within_300m={d['within_300m']} ({d['within_300m_pct']:.1f}%)")
        else:
            print(f"  {cls}: NO same-class training examples exist at all (fully held out)")
    print()

    if not LIGHTGBM_AVAILABLE:
        print(f"LightGBM UNAVAILABLE: {LIGHTGBM_IMPORT_ERROR}")
        print("Proceeding with Logistic Regression only.\n")

    classes = sorted(set(r["silver_label"] for r in train_rows) | set(r["silver_label"] for r in test_rows))

    results = {}
    print("=" * 70)
    print("LOGISTIC REGRESSION (spatial holdout)")
    print("=" * 70)
    lr = build_logistic_pipeline()
    lr_metrics = run_model(lr, train_rows, test_rows, classes)
    results["logistic_regression"] = lr_metrics
    print(f"accuracy={lr_metrics['accuracy']:.3f} balanced_accuracy={lr_metrics['balanced_accuracy']:.3f} "
          f"macro_f1={lr_metrics['macro_f1']:.3f}")

    if LIGHTGBM_AVAILABLE:
        print("\n" + "=" * 70)
        print("LIGHTGBM (spatial holdout, same fixed config as the temporal-split run)")
        print("=" * 70)
        print(f"config: {LIGHTGBM_PARAMS}")
        lgbm = build_lightgbm_pipeline()
        lgbm_metrics = run_model(lgbm, train_rows, test_rows, classes)
        results["lightgbm"] = lgbm_metrics
        print(f"accuracy={lgbm_metrics['accuracy']:.3f} balanced_accuracy={lgbm_metrics['balanced_accuracy']:.3f} "
              f"macro_f1={lgbm_metrics['macro_f1']:.3f}")

    for model_name, m in results.items():
        print(f"\n--- {model_name} per-class (spatial holdout) ---")
        for c in m["classes"]:
            pc = m["per_class"][c]
            print(f"  {c}: P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f} "
                  f"support={pc['support']} PR-AUC={m['pr_auc_ovr'].get(c, float('nan')):.3f}")
        print("  confusion_matrix:")
        for c, row in zip(m["classes"], m["confusion_matrix"]):
            print(f"    {c}: {row}")

    print("\n" + "=" * 70)
    print("COMPARISON: SPATIAL HOLDOUT vs. TEMPORAL SPLIT (temporal numbers from prior milestone, not rerun)")
    print("=" * 70)
    for model_name in results:
        t = TEMPORAL_RESULTS.get(model_name, {})
        s = results[model_name]
        print(f"  {model_name}: temporal balanced_acc={t.get('balanced_accuracy','n/a')} vs "
              f"spatial balanced_acc={s['balanced_accuracy']:.3f}  |  "
              f"temporal macro_f1={t.get('macro_f1','n/a')} vs spatial macro_f1={s['macro_f1']:.3f}")

    write_metrics_csv(results)
    print(f"\nWrote {METRICS_CSV}")
    plot_confusion_matrices(results)
    print(f"Saved figure to {FIGURE_CONFUSION}")

    return results, cell_report, diag


def load_rows():
    from train_source_classifier_lightgbm import load_feature_rows
    return load_feature_rows(FEATURES_CSV)


def write_split_report(cell_report, path=SPLIT_REPORT_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell_row", "cell_col", "assignment", "Industrial", "Gas_Flare",
                          "Crop_Residue", "Forest_Wildfire", "total"])
        for (row, col), v in sorted(cell_report.items()):
            counts = v["counts"]
            total = sum(counts.values())
            writer.writerow([row, col, v["assignment"], counts.get("Industrial", 0),
                              counts.get("Gas_Flare", 0), counts.get("Crop_Residue", 0),
                              counts.get("Forest_Wildfire", 0), total])


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
        ax.imshow(cm, cmap="Oranges")
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
        ax.set_title(name, fontsize=10)
    fig.suptitle("Spatial-Holdout Evaluation -- entire geographic cells withheld from training")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
