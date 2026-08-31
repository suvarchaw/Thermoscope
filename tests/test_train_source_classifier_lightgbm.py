import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import train_source_classifier_lightgbm as mod


class TestLightgbmEnvironment(unittest.TestCase):
    def test_availability_flag_is_boolean(self):
        self.assertIsInstance(mod.LIGHTGBM_AVAILABLE, bool)

    def test_lightgbm_actually_available_in_this_dev_environment(self):
        # Informational, not a hard project requirement (a differently
        # configured machine might lack scikit-learn's bundled libomp
        # path) -- but in THIS project's dev environment it must be True,
        # since the whole point of this milestone was fixing exactly this.
        self.assertTrue(mod.LIGHTGBM_AVAILABLE, f"import error: {mod.LIGHTGBM_IMPORT_ERROR}")

    def test_build_lightgbm_pipeline_raises_clearly_when_unavailable(self):
        original = mod.LIGHTGBM_AVAILABLE
        mod.LIGHTGBM_AVAILABLE = False
        try:
            with self.assertRaises(RuntimeError):
                mod.build_lightgbm_pipeline()
        finally:
            mod.LIGHTGBM_AVAILABLE = original


class TestFixedConfiguration(unittest.TestCase):
    def test_lightgbm_params_are_fixed_and_documented(self):
        expected = {
            "n_estimators": 100, "max_depth": 4, "num_leaves": 15,
            "learning_rate": 0.1, "class_weight": "balanced",
            "random_state": 42, "verbosity": -1,
        }
        self.assertEqual(mod.LIGHTGBM_PARAMS, expected)

    def test_no_hyperparameter_search_space_defined(self):
        # a hyperparameter search would introduce a grid/space object;
        # confirm none exists in the module namespace.
        for name in dir(mod):
            self.assertNotIn("grid", name.lower())
            self.assertNotIn("search", name.lower())


class TestFeatureSetUnchanged(unittest.TestCase):
    def test_eleven_features_match_prior_milestone_set_c(self):
        all_features = mod.FEATURES_NUMERIC + mod.FEATURES_CATEGORICAL
        self.assertEqual(len(all_features), 11)
        expected = {"centroid_lat", "centroid_lon", "spatial_extent_m", "status",
                    "frac_high_confidence", "frac_low_confidence", "mean_scan", "mean_track",
                    "elongation_ratio", "time_of_day_std_minutes", "detections_per_day"}
        self.assertEqual(set(all_features), expected)

    def test_no_new_features_beyond_baseline_plus_behavior(self):
        from event_behavior_features import BASELINE_FEATURES, NEW_FEATURES
        all_features = set(mod.FEATURES_NUMERIC + mod.FEATURES_CATEGORICAL)
        self.assertEqual(all_features, set(BASELINE_FEATURES) | set(NEW_FEATURES))


class TestLoadFeatureRows(unittest.TestCase):
    def test_start_date_reattached_by_event_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            features_path = Path(tmpdir) / "features.csv"
            silver_path = Path(tmpdir) / "silver.csv"
            with open(features_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["event_id", "silver_label", "centroid_lat"])
                w.writeheader()
                w.writerow({"event_id": "EVT000000", "silver_label": "Industrial", "centroid_lat": "21.0"})
            with open(silver_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["event_id", "start_date"])
                w.writeheader()
                w.writerow({"event_id": "EVT000000", "start_date": "2021-05-01"})

            import os
            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                Path("data/processed").mkdir(parents=True, exist_ok=True)
                os.rename(silver_path, "data/processed/gujarat_event_silver_labels.csv")
                rows = mod.load_feature_rows(path=features_path)
            finally:
                os.chdir(cwd)
            self.assertEqual(rows[0]["start_date"], "2021-05-01")

    def test_start_date_never_written_as_a_feature_column(self):
        self.assertNotIn("start_date", mod.FEATURES_NUMERIC)
        self.assertNotIn("start_date", mod.FEATURES_CATEGORICAL)


class TestFitAndEvaluateDeterminism(unittest.TestCase):
    def _synthetic_data(self):
        rng = np.random.RandomState(0)
        n = 60
        Xn = rng.rand(n, 2)
        Xc = np.array([["closed"]] * n, dtype=object)
        y = np.array((["Industrial", "Crop_Residue"] * (n // 2)))
        return Xn, Xc, y

    def test_repeated_lightgbm_fits_give_identical_predictions(self):
        if not mod.LIGHTGBM_AVAILABLE:
            self.skipTest("LightGBM unavailable in this environment")
        Xn, Xc, y = self._synthetic_data()
        X = mod.combine_features(Xn, Xc)

        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.pipeline import Pipeline
        from lightgbm import LGBMClassifier

        def build():
            pre = ColumnTransformer([
                ("num", StandardScaler(), [0, 1]),
                ("cat", OneHotEncoder(handle_unknown="ignore"), [2]),
            ])
            return Pipeline([("preprocess", pre), ("clf", LGBMClassifier(**mod.LIGHTGBM_PARAMS))])

        p1 = build(); p1.fit(X, y); pred1 = p1.predict(X)
        p2 = build(); p2.fit(X, y); pred2 = p2.predict(X)
        np.testing.assert_array_equal(pred1, pred2)


class TestSplitSizeInvariant(unittest.TestCase):
    def test_preserved_split_sizes_documented(self):
        # the exact sizes the milestone brief requires be preserved
        self.assertEqual(sorted(mod.TRAIN_YEARS), [2019, 2020, 2021, 2022])
        self.assertEqual(sorted(mod.TEST_YEARS), [2023, 2024, 2025])


if __name__ == "__main__":
    unittest.main()
