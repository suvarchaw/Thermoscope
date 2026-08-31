import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import infer_2026_events as mod
from train_source_classifier_lightgbm import (
    FEATURES_NUMERIC, FEATURES_CATEGORICAL, LIGHTGBM_AVAILABLE,
)


class TestFeatureSchemaCompatibility(unittest.TestCase):
    def test_build_2026_feature_rows_produces_locked_columns(self):
        events, feature_rows = mod.build_2026_feature_rows()
        self.assertGreater(len(feature_rows), 0)
        self.assertEqual(len(events), len(feature_rows))
        row = feature_rows[0]
        for col in FEATURES_NUMERIC + FEATURES_CATEGORICAL:
            self.assertIn(col, row, f"locked feature column {col} missing from 2026 feature row")

    def test_event_ids_align_between_events_and_features(self):
        events, feature_rows = mod.build_2026_feature_rows()
        self.assertEqual([e["event_id"] for e in events], [r["event_id"] for r in feature_rows])


@unittest.skipUnless(LIGHTGBM_AVAILABLE, "LightGBM unavailable in this environment")
class TestModelInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = mod.load_model()
        cls.events, cls.feature_rows = mod.build_2026_feature_rows()

    def test_predict_returns_only_trained_classes(self):
        y_pred, prob_by_class = mod.predict_2026(self.pipeline, self.feature_rows[:20])
        for label in y_pred:
            self.assertIn(label, mod.TRAINED_CLASSES)
        self.assertNotIn("Brick_Kiln", y_pred)
        self.assertNotIn("Unknown_Ambiguous", y_pred)

    def test_probabilities_sum_to_one_per_event(self):
        y_pred, prob_by_class = mod.predict_2026(self.pipeline, self.feature_rows[:20])
        classes = sorted(prob_by_class.keys())
        for i in range(20):
            total = sum(prob_by_class[c][i] for c in classes)
            self.assertAlmostEqual(total, 1.0, places=4)

    def test_deterministic_across_repeated_calls(self):
        y_pred_1, prob_1 = mod.predict_2026(self.pipeline, self.feature_rows[:30])
        y_pred_2, prob_2 = mod.predict_2026(self.pipeline, self.feature_rows[:30])
        self.assertEqual(y_pred_1, y_pred_2)
        self.assertEqual(prob_1, prob_2)

    def test_never_calls_fit(self):
        """No training on 2026: inspect every function in this module for
        a `.fit(` call -- only predict/predict_proba on the already-loaded
        pipeline are permitted."""
        source = inspect.getsource(mod)
        self.assertNotIn(".fit(", source)
        self.assertIn("joblib.load", source)


class TestBuildOutputRows(unittest.TestCase):
    def test_forced_choice_and_capability_note_present(self):
        events = [{"event_id": "EVT2026_000000", "start_date": "2026-01-01", "end_date": "2026-01-02",
                   "duration_days": 2, "status": "closed", "centroid_lat": 21.0, "centroid_lon": 72.0,
                   "spatial_extent_m": 100.0, "detection_count": 3, "mean_frp": 4.0, "max_frp": 5.0,
                   "night_fraction": 0.5}]
        feature_rows = [{"event_id": "EVT2026_000000"}]
        y_pred = ["Gas_Flare"]
        prob_by_class = {"Crop_Residue": [0.01], "Forest_Wildfire": [0.01],
                          "Gas_Flare": [0.97], "Industrial": [0.01]}
        rows = mod.build_output_rows(events, feature_rows, y_pred, prob_by_class, evidence_by_id={})
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["predicted_class"], "Gas_Flare")
        self.assertIn("known-site", row["class_capability_note"])
        self.assertEqual(row["data_type"], "2026_NRT_INFERENCE_NOT_GROUND_TRUTH")
        for c in ("Crop_Residue", "Forest_Wildfire", "Gas_Flare", "Industrial"):
            self.assertIn(f"prob_{c}", row)

    def test_missing_evidence_defaults_to_empty_string(self):
        events = [{"event_id": "EVT2026_999999", "start_date": "2026-01-01", "end_date": "2026-01-01",
                   "duration_days": 1, "status": "closed", "centroid_lat": 21.0, "centroid_lon": 72.0,
                   "spatial_extent_m": 0.0, "detection_count": 2, "mean_frp": 1.0, "max_frp": 1.0,
                   "night_fraction": 0.0}]
        feature_rows = [{"event_id": "EVT2026_999999"}]
        y_pred = ["Industrial"]
        prob_by_class = {"Crop_Residue": [0.1], "Forest_Wildfire": [0.1],
                          "Gas_Flare": [0.1], "Industrial": [0.7]}
        rows = mod.build_output_rows(events, feature_rows, y_pred, prob_by_class, evidence_by_id={})
        self.assertEqual(rows[0]["land_cover_class"], "")


if __name__ == "__main__":
    unittest.main()
