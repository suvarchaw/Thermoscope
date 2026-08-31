import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evaluate_worldcover_diversity_feature as mod


class TestFeatureSets(unittest.TestCase):
    def test_set_a_has_eleven_columns(self):
        numeric, categorical = mod.FEATURE_SETS["A_11_features"]
        self.assertEqual(len(numeric) + len(categorical), 11)

    def test_set_b_has_exactly_one_new_column(self):
        a_numeric, a_categorical = mod.FEATURE_SETS["A_11_features"]
        b_numeric, b_categorical = mod.FEATURE_SETS["B_12_features_plus_worldcover_diversity"]
        self.assertEqual(len(b_numeric) + len(b_categorical), 12)
        new_numeric = set(b_numeric) - set(a_numeric)
        self.assertEqual(new_numeric, {"worldcover_diversity_300m"})
        self.assertEqual(set(a_categorical), set(b_categorical))

    def test_set_a_unchanged_within_set_b(self):
        a_numeric, a_categorical = mod.FEATURE_SETS["A_11_features"]
        b_numeric, b_categorical = mod.FEATURE_SETS["B_12_features_plus_worldcover_diversity"]
        self.assertTrue(set(a_numeric).issubset(set(b_numeric)))
        self.assertEqual(set(a_categorical), set(b_categorical))


class TestMissingValueFill(unittest.TestCase):
    def _write_fixture(self, tmpdir):
        base_path = Path(tmpdir) / "features.csv"
        div_path = Path(tmpdir) / "diversity.csv"
        silver_path = Path(tmpdir) / "silver.csv"
        with open(base_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["event_id", "silver_label", "centroid_lat"])
            w.writeheader()
            w.writerow({"event_id": "EVT_OK", "silver_label": "Industrial", "centroid_lat": "21.0"})
            w.writerow({"event_id": "EVT_MISSING", "silver_label": "Crop_Residue", "centroid_lat": "22.0"})
        with open(div_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["event_id", "worldcover_diversity_300m", "worldcover_diversity_computable"])
            w.writeheader()
            w.writerow({"event_id": "EVT_OK", "worldcover_diversity_300m": "6", "worldcover_diversity_computable": "True"})
            w.writerow({"event_id": "EVT_MISSING", "worldcover_diversity_300m": "", "worldcover_diversity_computable": "False"})
        with open(silver_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["event_id", "start_date"])
            w.writeheader()
            w.writerow({"event_id": "EVT_OK", "start_date": "2020-05-01"})
            w.writerow({"event_id": "EVT_MISSING", "start_date": "2021-06-01"})
        return base_path, div_path, silver_path

    def test_computable_value_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path, div_path, silver_path = self._write_fixture(tmpdir)
            orig = (mod.FEATURES_V2_CSV, mod.DIVERSITY_CSV, mod.SILVER_LABELS_CSV)
            mod.FEATURES_V2_CSV, mod.DIVERSITY_CSV, mod.SILVER_LABELS_CSV = base_path, div_path, silver_path
            try:
                rows, n_missing = mod.load_joined_rows()
            finally:
                mod.FEATURES_V2_CSV, mod.DIVERSITY_CSV, mod.SILVER_LABELS_CSV = orig
            row = next(r for r in rows if r["event_id"] == "EVT_OK")
            self.assertEqual(row["worldcover_diversity_300m"], 6.0)
            self.assertEqual(n_missing, 1)

    def test_missing_value_filled_with_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path, div_path, silver_path = self._write_fixture(tmpdir)
            orig = (mod.FEATURES_V2_CSV, mod.DIVERSITY_CSV, mod.SILVER_LABELS_CSV)
            mod.FEATURES_V2_CSV, mod.DIVERSITY_CSV, mod.SILVER_LABELS_CSV = base_path, div_path, silver_path
            try:
                rows, n_missing = mod.load_joined_rows()
            finally:
                mod.FEATURES_V2_CSV, mod.DIVERSITY_CSV, mod.SILVER_LABELS_CSV = orig
            row = next(r for r in rows if r["event_id"] == "EVT_MISSING")
            self.assertEqual(row["worldcover_diversity_300m"], 0.0)


class TestGetSplitDispatch(unittest.TestCase):
    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            mod.get_split([], "bogus")

    def test_splits_use_unchanged_imports(self):
        import inspect
        src = inspect.getsource(mod.get_split)
        self.assertIn("temporal_split(rows)", src)
        self.assertIn("assign_spatial_split(rows)", src)


class TestLightgbmParamsUnchanged(unittest.TestCase):
    def test_identical_to_prior_milestones(self):
        expected = {
            "n_estimators": 100, "max_depth": 4, "num_leaves": 15,
            "learning_rate": 0.1, "class_weight": "balanced",
            "random_state": 42, "verbosity": -1,
        }
        self.assertEqual(mod.LIGHTGBM_PARAMS, expected)


if __name__ == "__main__":
    unittest.main()
