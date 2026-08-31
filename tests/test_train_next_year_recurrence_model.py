import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import train_next_year_recurrence_model as mod


def make_source_row(cluster_id, **year_detections):
    """year_detections e.g. detections_2019=10, active_days_2019=5, ..."""
    row = {
        "cluster_id": str(cluster_id),
        "centroid_lat": "21.0", "centroid_lon": "72.0", "extent_radius_m": "300",
        "bbox_min_lat": "20.9", "bbox_max_lat": "21.1", "bbox_min_lon": "71.9", "bbox_max_lon": "72.1",
        "nearest_distance_m": "250", "nearest_is_named": "False", "has_notable_osm_context": "True",
        "n_industrial": "1", "n_power": "0", "n_waste": "0", "n_agricultural": "0",
        "n_transport": "0", "n_other": "0", "features_found_in_radius": "1",
        "nearest_group": "industrial",
    }
    for year in (2019, 2020, 2021, 2022, 2023, 2024, 2025):
        row[f"detections_{year}"] = "0"
        row[f"active_days_{year}"] = "0"
    for key, val in year_detections.items():
        row[key] = str(val)
    return row


class TestValidTransitions(unittest.TestCase):
    def test_2022_2023_excluded(self):
        self.assertNotIn((2022, 2023), mod.VALID_TRANSITIONS)
        self.assertIn((2022, 2023), mod.EXCLUDED_TRANSITIONS)

    def test_2025_2026_excluded(self):
        self.assertNotIn((2025, 2026), mod.VALID_TRANSITIONS)
        self.assertIn((2025, 2026), mod.EXCLUDED_TRANSITIONS)

    def test_exactly_five_valid_transitions(self):
        self.assertEqual(mod.VALID_TRANSITIONS, [
            (2019, 2020), (2020, 2021), (2021, 2022), (2023, 2024), (2024, 2025),
        ])

    def test_valid_and_excluded_are_disjoint(self):
        self.assertEqual(set(mod.VALID_TRANSITIONS) & set(mod.EXCLUDED_TRANSITIONS.keys()), set())

    def test_mvp_primary_split_unchanged(self):
        self.assertEqual(mod.MVP_PRIMARY_TRAIN_TRANSITIONS, [(2019, 2020), (2020, 2021)])
        self.assertEqual(mod.MVP_PRIMARY_TEST_TRANSITIONS, [(2021, 2022)])

    def test_extended_primary_split_matches_spec(self):
        self.assertEqual(mod.EXTENDED_PRIMARY_TRAIN_TRANSITIONS, [
            (2019, 2020), (2020, 2021), (2021, 2022), (2023, 2024),
        ])
        self.assertEqual(mod.EXTENDED_PRIMARY_TEST_TRANSITIONS, [(2024, 2025)])

    def test_robustness_split_matches_spec(self):
        self.assertEqual(mod.ROBUSTNESS_TRAIN_TRANSITIONS, [(2019, 2020)])
        self.assertEqual(mod.ROBUSTNESS_TEST_TRANSITIONS, [(2020, 2021)])

    def test_assert_transitions_consistent_passes_on_clean_state(self):
        mod.assert_transitions_consistent()  # should not raise

    def test_assert_transitions_consistent_catches_overlap(self):
        mod.VALID_TRANSITIONS.append((2022, 2023))
        try:
            with self.assertRaises(ValueError):
                mod.assert_transitions_consistent()
        finally:
            mod.VALID_TRANSITIONS.remove((2022, 2023))


class TestForbiddenFeatures(unittest.TestCase):
    def test_no_forbidden_feature_in_allowlist(self):
        used = set(mod.FEATURE_COLUMNS_NUMERIC) | set(mod.FEATURE_COLUMNS_CATEGORICAL)
        self.assertEqual(used & mod.FORBIDDEN_FEATURES, set())

    def test_assert_function_passes_on_clean_state(self):
        mod.assert_no_forbidden_features()  # should not raise

    def test_assert_function_catches_injected_leak(self):
        mod.FEATURE_COLUMNS_NUMERIC.append("recurrence_strength")
        try:
            with self.assertRaises(ValueError):
                mod.assert_no_forbidden_features()
        finally:
            mod.FEATURE_COLUMNS_NUMERIC.remove("recurrence_strength")

    def test_specific_brief_forbidden_columns_all_covered(self):
        brief_forbidden = {
            "recurrence_strength", "short_window_recurrence", "burst_concentrated",
            "mean_frp", "unique_dates", "day_count", "night_count",
            "unique_years", "years_detected", "trend_slope", "total_detections_5yr",
            "persistence_category", "activity_category", "persistence_activity_quadrant",
            "seasonality_category", "evidence_notes", "evidence_corroboration_count",
            "unsupervised_group",
        }
        self.assertTrue(brief_forbidden.issubset(mod.FORBIDDEN_FEATURES))


class TestBuildPanel(unittest.TestCase):
    def test_target_is_next_year_presence(self):
        row = make_source_row(0, detections_2019=5, detections_2020=0)
        panel = mod.build_panel([row], [(2019, 2020)])
        self.assertEqual(panel[0]["target"], 0)  # detections_2020 == 0 -> no recurrence

    def test_target_positive_when_next_year_present(self):
        row = make_source_row(0, detections_2019=5, detections_2020=3)
        panel = mod.build_panel([row], [(2019, 2020)])
        self.assertEqual(panel[0]["target"], 1)

    def test_features_use_only_year_y_not_future_years(self):
        row = make_source_row(0, detections_2019=1, detections_2020=999, detections_2021=999)
        panel = mod.build_panel([row], [(2019, 2020)])
        # feature for transition 2019->2020 must reflect ONLY 2019, never touch 2020/2021 values
        self.assertEqual(panel[0]["detections_Y"], 1)

    def test_missing_nearest_distance_imputed(self):
        row = make_source_row(0, detections_2019=1, detections_2020=1)
        row["nearest_distance_m"] = ""
        panel = mod.build_panel([row], [(2019, 2020)])
        self.assertGreater(panel[0]["nearest_distance_m"], 0)  # imputed, not crashed

    def test_one_row_per_cluster_per_transition(self):
        rows = [make_source_row(i, detections_2019=1, detections_2020=1) for i in range(5)]
        panel = mod.build_panel(rows, [(2019, 2020), (2020, 2021)])
        self.assertEqual(len(panel), 10)  # 5 clusters x 2 transitions

    def test_2023_2024_transition_reads_only_2023_and_2024(self):
        row = make_source_row(0, detections_2023=7, detections_2024=0, detections_2025=999)
        panel = mod.build_panel([row], [(2023, 2024)])
        self.assertEqual(panel[0]["detections_Y"], 7)
        self.assertEqual(panel[0]["target"], 0)  # detections_2024 == 0 -> no recurrence

    def test_2024_2025_transition_target_positive(self):
        row = make_source_row(0, detections_2024=3, detections_2025=1)
        panel = mod.build_panel([row], [(2024, 2025)])
        self.assertEqual(panel[0]["detections_Y"], 3)
        self.assertEqual(panel[0]["target"], 1)

    def test_all_five_valid_transitions_build_without_error(self):
        rows = [make_source_row(i, detections_2019=1, detections_2020=1, detections_2021=1,
                                 detections_2022=1, detections_2023=1, detections_2024=1, detections_2025=1)
                for i in range(3)]
        panel = mod.build_panel(rows, mod.VALID_TRANSITIONS)
        self.assertEqual(len(panel), 3 * len(mod.VALID_TRANSITIONS))


class TestToBoolInt(unittest.TestCase):
    def test_string_true(self):
        self.assertEqual(mod.to_bool_int("True"), 1)

    def test_string_false(self):
        self.assertEqual(mod.to_bool_int("False"), 0)

    def test_python_bool(self):
        self.assertEqual(mod.to_bool_int(True), 1)
        self.assertEqual(mod.to_bool_int(False), 0)


class TestComputeMetrics(unittest.TestCase):
    def test_perfect_prediction(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        m = mod.compute_metrics(y_true, y_pred)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["balanced_accuracy"], 1.0)
        self.assertEqual(m["recall_minority_class0"], 1.0)
        self.assertEqual(m["precision_minority_class0"], 1.0)

    def test_majority_only_prediction_on_imbalanced_data(self):
        y_true = np.array([1, 1, 1, 0])
        y_pred = np.array([1, 1, 1, 1])  # always predicts majority
        m = mod.compute_metrics(y_true, y_pred)
        self.assertEqual(m["recall_minority_class0"], 0.0)
        self.assertEqual(m["balanced_accuracy"], 0.5)

    def test_confusion_matrix_shape_and_counts(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        m = mod.compute_metrics(y_true, y_pred)
        cm = np.array(m["confusion_matrix"])
        self.assertEqual(cm.shape, (2, 2))
        self.assertEqual(cm.sum(), 4)

    def test_pr_auc_none_without_probabilities(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        m = mod.compute_metrics(y_true, y_pred, y_prob_class0=None)
        self.assertIsNone(m["pr_auc_minority_class0"])


class TestPipelineReproducibility(unittest.TestCase):
    def test_same_random_state_gives_identical_predictions(self):
        rows = [make_source_row(i, detections_2019=(i % 5), detections_2020=((i + 1) % 3))
                for i in range(20)]
        panel = mod.build_panel(rows, [(2019, 2020)])
        Xn, Xc, y = mod.panel_to_arrays(panel)
        X = mod.combine_features(Xn, Xc)

        lr1 = mod.build_logistic_pipeline()
        lr1.fit(X, y)
        pred1 = lr1.predict(X)

        lr2 = mod.build_logistic_pipeline()
        lr2.fit(X, y)
        pred2 = lr2.predict(X)

        np.testing.assert_array_equal(pred1, pred2)


class TestRunSplitClassBalance(unittest.TestCase):
    def test_reports_positive_rate_for_train_and_test(self):
        rows = [make_source_row(i, detections_2019=1, detections_2020=(1 if i < 8 else 0))
                for i in range(10)]
        results, *_ = mod.run_split(rows, [(2019, 2020)], [(2019, 2020)], "test_split")
        self.assertAlmostEqual(results["train_positive_rate"], results["train_positive"] / results["n_train"])
        self.assertAlmostEqual(results["test_positive_rate"], results["test_positive"] / results["n_test"])


class TestWritePredictionsAndMetricsCsv(unittest.TestCase):
    def test_write_predictions_csv_multi_split(self):
        import tempfile
        panel = [{"cluster_id": 0, "year": 2019, "target_year": 2020, "target": 1}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preds.csv"
            mod.write_predictions_csv(
                [("split_a", panel, [1], [0.1], [1], [0.2])], path=path,
            )
            with open(path, newline="") as f:
                import csv as csv_mod
                rows = list(csv_mod.DictReader(f))
            self.assertEqual(rows[0]["split"], "split_a")
            self.assertEqual(rows[0]["cluster_id"], "0")

    def test_write_metrics_csv_multiple_splits(self):
        import tempfile
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        m = mod.compute_metrics(y_true, y_pred)
        split_result = {
            "label": "split_a", "n_train": 4, "n_test": 4,
            "train_positive": 2, "train_negative": 2, "train_positive_rate": 0.5,
            "test_positive": 2, "test_negative": 2, "test_positive_rate": 0.5,
            "baseline": m, "logistic_regression": m, "random_forest": m,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.csv"
            mod.write_metrics_csv([split_result, split_result], path=path)
            with open(path, newline="") as f:
                import csv as csv_mod
                rows = list(csv_mod.DictReader(f))
            self.assertEqual(len(rows), 6)  # 2 splits x 3 models


if __name__ == "__main__":
    unittest.main()
