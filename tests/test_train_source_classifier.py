import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import train_source_classifier as mod


def make_row(**overrides):
    row = {
        "event_id": "EVT000000", "start_date": "2020-05-15", "end_date": "2020-05-15",
        "duration_days": "1", "centroid_lat": "21.0", "centroid_lon": "72.0",
        "spatial_extent_m": "100.0", "detection_count": "2", "mean_frp": "3.0", "max_frp": "3.0",
        "night_fraction": "0.0", "status": "closed",
        "silver_label": "Industrial", "label_rule_id": "Industrial", "label_evidence": "x",
        "conflict_classes": "Industrial", "n_classes_matched": "1", "ambiguity_reason": "",
        "excluded_from_training": "False",
    }
    row.update({k: str(v) for k, v in overrides.items()})
    return row


class TestFeatureAllowlistSafety(unittest.TestCase):
    def test_allowlist_clean_on_default_state(self):
        mod.assert_feature_allowlist_clean()  # should not raise

    def test_allowlist_never_overlaps_forbidden(self):
        self.assertEqual(set(mod.FEATURE_ALLOWLIST) & mod.FORBIDDEN_COLUMNS, set())

    def test_forbidden_columns_include_every_required_name(self):
        required = {"silver_label", "label_rule_id", "label_evidence", "conflict_classes",
                    "n_classes_matched", "ambiguity_reason", "excluded_from_training"}
        self.assertTrue(required.issubset(mod.FORBIDDEN_COLUMNS))

    def test_label_generating_columns_excluded_from_allowlist(self):
        label_generating = {"mean_frp", "night_fraction", "duration_days", "start_date", "end_date"}
        self.assertEqual(label_generating & set(mod.FEATURE_ALLOWLIST), set())

    def test_correlated_proxy_columns_excluded_from_allowlist(self):
        # detection_count/max_frp were found to correlate strongly with
        # label-generating fields and must stay excluded.
        self.assertNotIn("detection_count", mod.FEATURE_ALLOWLIST)
        self.assertNotIn("max_frp", mod.FEATURE_ALLOWLIST)

    def test_assert_catches_injected_forbidden_feature(self):
        mod.FEATURE_ALLOWLIST.append("mean_frp")
        mod.FEATURE_ALLOWLIST_NUMERIC.append("mean_frp")
        try:
            with self.assertRaises(ValueError):
                mod.assert_feature_allowlist_clean()
        finally:
            mod.FEATURE_ALLOWLIST.remove("mean_frp")
            mod.FEATURE_ALLOWLIST_NUMERIC.remove("mean_frp")

    def test_feature_audit_included_flags_match_allowlist(self):
        audited_included = {r["name"] for r in mod.FEATURE_AUDIT if r["included"]}
        self.assertEqual(audited_included, set(mod.FEATURE_ALLOWLIST))

    def test_every_excluded_audit_row_has_a_reason(self):
        for r in mod.FEATURE_AUDIT:
            if not r["included"]:
                self.assertTrue(len(r["reason"]) > 0, f"{r['name']} has no exclusion reason")


class TestBrickKilnAndUnknownExclusion(unittest.TestCase):
    def test_assert_no_brick_kiln_passes_on_clean_rows(self):
        rows = [make_row(silver_label="Industrial")]
        mod.assert_no_brick_kiln_in_training(rows)  # should not raise

    def test_assert_no_brick_kiln_catches_violation(self):
        rows = [make_row(silver_label="Brick_Kiln")]
        with self.assertRaises(ValueError):
            mod.assert_no_brick_kiln_in_training(rows)

    def test_assert_no_unknown_ambiguous_catches_violation(self):
        rows = [make_row(silver_label="Unknown_Ambiguous")]
        with self.assertRaises(ValueError):
            mod.assert_no_unknown_ambiguous(rows)

    def test_filter_to_trained_classes_excludes_brick_kiln_and_unknown(self):
        rows = [
            make_row(event_id="A", silver_label="Industrial"),
            make_row(event_id="B", silver_label="Brick_Kiln"),
            make_row(event_id="C", silver_label="Unknown_Ambiguous"),
            make_row(event_id="D", silver_label="Gas_Flare"),
        ]
        filtered = mod.filter_to_trained_classes(rows)
        labels = {r["silver_label"] for r in filtered}
        self.assertEqual(labels, {"Industrial", "Gas_Flare"})
        self.assertEqual(len(filtered), 2)


class TestTemporalSplit(unittest.TestCase):
    def test_train_test_years_do_not_overlap(self):
        self.assertEqual(mod.TRAIN_YEARS & mod.TEST_YEARS, set())

    def test_split_assigns_by_start_date_year(self):
        rows = [
            make_row(event_id="A", start_date="2020-01-01"),
            make_row(event_id="B", start_date="2024-01-01"),
        ]
        train, test = mod.temporal_split(rows)
        self.assertEqual([r["event_id"] for r in train], ["A"])
        self.assertEqual([r["event_id"] for r in test], ["B"])

    def test_split_is_exhaustive_and_disjoint(self):
        rows = [make_row(event_id=str(i), start_date=f"{2019 + (i % 7)}-06-01") for i in range(20)]
        train, test = mod.temporal_split(rows)
        self.assertEqual(len(train) + len(test), len(rows))
        self.assertEqual(set(r["event_id"] for r in train) & set(r["event_id"] for r in test), set())

    def test_year_out_of_range_raises(self):
        rows = [make_row(start_date="2018-01-01")]  # not in TRAIN_YEARS or TEST_YEARS
        with self.assertRaises(AssertionError):
            mod.temporal_split(rows)


class TestRowsToArrays(unittest.TestCase):
    def test_shapes_match_feature_allowlist(self):
        rows = [make_row(centroid_lat=21.0, centroid_lon=72.0, spatial_extent_m=50.0, status="closed"),
                make_row(centroid_lat=22.0, centroid_lon=73.0, spatial_extent_m=100.0, status="provisional")]
        Xn, Xc, y = mod.rows_to_arrays(rows)
        self.assertEqual(Xn.shape, (2, len(mod.FEATURE_ALLOWLIST_NUMERIC)))
        self.assertEqual(Xc.shape, (2, len(mod.FEATURE_ALLOWLIST_CATEGORICAL)))
        self.assertEqual(list(y), ["Industrial", "Industrial"])

    def test_only_allowlisted_columns_are_read(self):
        # a row missing every forbidden column entirely must still work,
        # proving rows_to_arrays never touches them
        rows = [make_row()]
        del rows[0]["mean_frp"], rows[0]["night_fraction"], rows[0]["duration_days"]
        del rows[0]["label_rule_id"], rows[0]["label_evidence"]
        Xn, Xc, y = mod.rows_to_arrays(rows)  # must not raise KeyError
        self.assertEqual(Xn.shape[0], 1)


class TestPreprocessingFitOnlyOnTrain(unittest.TestCase):
    def test_scaler_statistics_come_only_from_training_data(self):
        train_rows = [
            make_row(centroid_lat=20.0, centroid_lon=72.0, spatial_extent_m=0.0, status="closed", silver_label="Industrial"),
            make_row(centroid_lat=21.0, centroid_lon=72.0, spatial_extent_m=0.0, status="closed", silver_label="Crop_Residue"),
            make_row(centroid_lat=22.0, centroid_lon=72.0, spatial_extent_m=0.0, status="closed", silver_label="Industrial"),
        ]
        test_rows = [make_row(centroid_lat=100.0, centroid_lon=72.0, spatial_extent_m=0.0, status="closed")]
        Xn_train, Xc_train, y_train = mod.rows_to_arrays(train_rows)
        Xn_test, Xc_test, y_test = mod.rows_to_arrays(test_rows)
        X_train = mod.combine_features(Xn_train, Xc_train)
        X_test = mod.combine_features(Xn_test, Xc_test)

        pipeline = mod.build_logistic_pipeline()
        pipeline.fit(X_train, y_train)
        scaler = pipeline.named_steps["preprocess"].named_transformers_["num"]
        # mean of centroid_lat feature (index 0) must equal the TRAIN mean (21.0), not blended with test's 100.0
        self.assertAlmostEqual(scaler.mean_[0], 21.0, places=5)

        # transforming test data must not refit the scaler
        pipeline.predict(X_test)
        self.assertAlmostEqual(scaler.mean_[0], 21.0, places=5)


class TestDeterminism(unittest.TestCase):
    def test_repeated_array_construction_is_identical(self):
        rows = [make_row(event_id=str(i), centroid_lat=20.0 + i, silver_label="Crop_Residue") for i in range(10)]
        Xn1, Xc1, y1 = mod.rows_to_arrays(list(rows))
        Xn2, Xc2, y2 = mod.rows_to_arrays(list(rows))
        np.testing.assert_array_equal(Xn1, Xn2)
        np.testing.assert_array_equal(Xc1, Xc2)
        np.testing.assert_array_equal(y1, y2)

    def test_repeated_lr_fits_give_identical_predictions(self):
        rows = [make_row(event_id=str(i), centroid_lat=20.0 + (i % 5), centroid_lon=70.0 + (i % 3),
                          silver_label=["Industrial", "Crop_Residue"][i % 2]) for i in range(20)]
        Xn, Xc, y = mod.rows_to_arrays(rows)
        X = mod.combine_features(Xn, Xc)

        p1 = mod.build_logistic_pipeline()
        p1.fit(X, y)
        pred1 = p1.predict(X)

        p2 = mod.build_logistic_pipeline()
        p2.fit(X, y)
        pred2 = p2.predict(X)

        np.testing.assert_array_equal(pred1, pred2)


class TestComputeMetrics(unittest.TestCase):
    def test_perfect_prediction(self):
        y_true = np.array(["A", "A", "B", "B"])
        y_pred = np.array(["A", "A", "B", "B"])
        m = mod.compute_metrics(y_true, y_pred, classes=["A", "B"])
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["balanced_accuracy"], 1.0)
        self.assertEqual(m["per_class"]["A"]["support"], 2)

    def test_confusion_matrix_shape(self):
        y_true = np.array(["A", "B", "C"])
        y_pred = np.array(["A", "A", "C"])
        m = mod.compute_metrics(y_true, y_pred, classes=["A", "B", "C"])
        cm = np.array(m["confusion_matrix"])
        self.assertEqual(cm.shape, (3, 3))
        self.assertEqual(cm.sum(), 3)

    def test_pr_auc_computed_when_probabilities_given(self):
        y_true = np.array(["A", "A", "B", "B"])
        y_pred = np.array(["A", "B", "B", "A"])
        y_prob = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.7, 0.3]])
        m = mod.compute_metrics(y_true, y_pred, classes=["A", "B"], y_prob=y_prob)
        self.assertIn("A", m["pr_auc_ovr"])
        self.assertIn("B", m["pr_auc_ovr"])


if __name__ == "__main__":
    unittest.main()
