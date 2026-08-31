import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evaluate_frp_trend_feature as mod


class TestFeatureSets(unittest.TestCase):
    def test_set_a_has_eleven_columns(self):
        numeric, categorical = mod.FEATURE_SETS["A_11_features"]
        self.assertEqual(len(numeric) + len(categorical), 11)

    def test_set_b_has_thirteen_columns_twelve_conceptual_features(self):
        numeric, categorical = mod.FEATURE_SETS["B_12_features_plus_frp_trend"]
        self.assertEqual(len(numeric) + len(categorical), 13)
        self.assertIn("frp_trend_slope", numeric)
        self.assertIn("frp_trend_computable", categorical)

    def test_set_b_is_set_a_plus_exactly_two_new_columns(self):
        a_numeric, a_categorical = mod.FEATURE_SETS["A_11_features"]
        b_numeric, b_categorical = mod.FEATURE_SETS["B_12_features_plus_frp_trend"]
        new_numeric = set(b_numeric) - set(a_numeric)
        new_categorical = set(b_categorical) - set(a_categorical)
        self.assertEqual(new_numeric, {"frp_trend_slope"})
        self.assertEqual(new_categorical, {"frp_trend_computable"})
        # no existing Set A column was removed or altered
        self.assertEqual(set(a_numeric), set(b_numeric) - new_numeric)
        self.assertEqual(set(a_categorical), set(b_categorical) - new_categorical)


class TestMissingValueFill(unittest.TestCase):
    def _write_fixture(self, tmpdir):
        base_path = Path(tmpdir) / "features.csv"
        frp_path = Path(tmpdir) / "frp.csv"
        silver_path = Path(tmpdir) / "silver.csv"
        with open(base_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["event_id", "silver_label", "centroid_lat"])
            w.writeheader()
            w.writerow({"event_id": "EVT_COMPUTABLE", "silver_label": "Industrial", "centroid_lat": "21.0"})
            w.writerow({"event_id": "EVT_MISSING", "silver_label": "Crop_Residue", "centroid_lat": "22.0"})
        with open(frp_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["event_id", "frp_trend_slope", "frp_trend_computable", "n_unique_days"])
            w.writeheader()
            w.writerow({"event_id": "EVT_COMPUTABLE", "frp_trend_slope": "2.5", "frp_trend_computable": "True", "n_unique_days": 3})
            w.writerow({"event_id": "EVT_MISSING", "frp_trend_slope": "", "frp_trend_computable": "False", "n_unique_days": 1})
        with open(silver_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["event_id", "start_date"])
            w.writeheader()
            w.writerow({"event_id": "EVT_COMPUTABLE", "start_date": "2020-05-01"})
            w.writerow({"event_id": "EVT_MISSING", "start_date": "2021-06-01"})
        return base_path, frp_path, silver_path

    def test_computable_event_keeps_real_slope_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path, frp_path, silver_path = self._write_fixture(tmpdir)
            orig = (mod.FEATURES_V2_CSV, mod.FRP_TREND_CSV, mod.SILVER_LABELS_CSV)
            mod.FEATURES_V2_CSV, mod.FRP_TREND_CSV, mod.SILVER_LABELS_CSV = base_path, frp_path, silver_path
            try:
                rows = mod.load_joined_rows()
            finally:
                mod.FEATURES_V2_CSV, mod.FRP_TREND_CSV, mod.SILVER_LABELS_CSV = orig
            row = next(r for r in rows if r["event_id"] == "EVT_COMPUTABLE")
            self.assertEqual(row["frp_trend_slope"], 2.5)
            self.assertEqual(row["frp_trend_computable"], "True")

    def test_missing_event_filled_with_zero_and_flagged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path, frp_path, silver_path = self._write_fixture(tmpdir)
            orig = (mod.FEATURES_V2_CSV, mod.FRP_TREND_CSV, mod.SILVER_LABELS_CSV)
            mod.FEATURES_V2_CSV, mod.FRP_TREND_CSV, mod.SILVER_LABELS_CSV = base_path, frp_path, silver_path
            try:
                rows = mod.load_joined_rows()
            finally:
                mod.FEATURES_V2_CSV, mod.FRP_TREND_CSV, mod.SILVER_LABELS_CSV = orig
            row = next(r for r in rows if r["event_id"] == "EVT_MISSING")
            self.assertEqual(row["frp_trend_slope"], 0.0)
            self.assertEqual(row["frp_trend_computable"], "False")
            # the fill is never silent -- the flag always travels with it
            self.assertIn("frp_trend_computable", row)

    def test_start_date_reattached_for_temporal_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path, frp_path, silver_path = self._write_fixture(tmpdir)
            orig = (mod.FEATURES_V2_CSV, mod.FRP_TREND_CSV, mod.SILVER_LABELS_CSV)
            mod.FEATURES_V2_CSV, mod.FRP_TREND_CSV, mod.SILVER_LABELS_CSV = base_path, frp_path, silver_path
            try:
                rows = mod.load_joined_rows()
            finally:
                mod.FEATURES_V2_CSV, mod.FRP_TREND_CSV, mod.SILVER_LABELS_CSV = orig
            row = next(r for r in rows if r["event_id"] == "EVT_COMPUTABLE")
            self.assertEqual(row["start_date"], "2020-05-01")


class TestGetSplitDispatch(unittest.TestCase):
    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            mod.get_split([], "not_a_real_split")

    def test_temporal_and_spatial_use_unchanged_imports(self):
        # confirms this module did not reimplement either split -- it
        # calls the exact functions already reviewed/tested elsewhere.
        import inspect
        src = inspect.getsource(mod.get_split)
        self.assertIn("temporal_split(rows)", src)
        self.assertIn("assign_spatial_split(rows)", src)


class TestLightgbmParamsUnchanged(unittest.TestCase):
    def test_lightgbm_params_identical_to_prior_milestone(self):
        expected = {
            "n_estimators": 100, "max_depth": 4, "num_leaves": 15,
            "learning_rate": 0.1, "class_weight": "balanced",
            "random_state": 42, "verbosity": -1,
        }
        self.assertEqual(mod.LIGHTGBM_PARAMS, expected)


if __name__ == "__main__":
    unittest.main()
