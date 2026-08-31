import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import event_worldcover_diversity_feature as mod


class TestConstants(unittest.TestCase):
    def test_radius_matches_investigated_value(self):
        self.assertEqual(mod.RADIUS_M, 300.0)

    def test_window_half_pixels_covers_radius_at_10m_resolution(self):
        # 30 pixels * 10m = 300m, matching RADIUS_M exactly
        self.assertEqual(mod.WINDOW_HALF_PIXELS * mod.PIXEL_SIZE_M, mod.RADIUS_M)


class TestPearson(unittest.TestCase):
    def test_perfect_positive_correlation(self):
        self.assertAlmostEqual(mod.pearson([1, 2, 3, 4], [2, 4, 6, 8]), 1.0)

    def test_perfect_negative_correlation(self):
        self.assertAlmostEqual(mod.pearson([1, 2, 3, 4], [8, 6, 4, 2]), -1.0)


class TestWriteCsvSchema(unittest.TestCase):
    def test_output_columns(self):
        import csv
        import tempfile
        rows = [{"event_id": "EVT000000", "worldcover_diversity_300m": 5, "worldcover_diversity_computable": True}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.csv"
            mod.write_csv(rows, path=path)
            with open(path, newline="") as f:
                written = list(csv.DictReader(f))
            self.assertEqual(written[0]["worldcover_diversity_300m"], "5")
            self.assertEqual(written[0]["worldcover_diversity_computable"], "True")

    def test_non_computable_written_blank(self):
        import csv
        import tempfile
        rows = [{"event_id": "EVT000001", "worldcover_diversity_300m": "", "worldcover_diversity_computable": False}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.csv"
            mod.write_csv(rows, path=path)
            with open(path, newline="") as f:
                written = list(csv.DictReader(f))
            self.assertEqual(written[0]["worldcover_diversity_300m"], "")


class TestLeakageAndRedundancyAudits(unittest.TestCase):
    def _silver_row(self, eid, label):
        return {"event_id": eid, "silver_label": label, "mean_frp": "3.0", "night_fraction": "0.0",
                "duration_days": "2", "detection_count": "2", "max_frp": "3.0"}

    def test_leakage_audit_skips_non_computable(self):
        feature_rows = [
            {"event_id": "A", "worldcover_diversity_300m": 5.0, "worldcover_diversity_computable": True},
            {"event_id": "B", "worldcover_diversity_300m": "", "worldcover_diversity_computable": False},
        ]
        silver_rows = [self._silver_row("A", "Industrial"), self._silver_row("B", "Industrial")]
        correlations, is_safe, n_checked = mod.run_leakage_audit(feature_rows, silver_rows)
        self.assertEqual(n_checked, 1)

    def test_redundancy_audit_checks_centroid_fields(self):
        feature_rows = [
            {"event_id": "A", "worldcover_diversity_300m": 5.0, "worldcover_diversity_computable": True},
            {"event_id": "B", "worldcover_diversity_300m": 3.0, "worldcover_diversity_computable": True},
        ]
        existing_rows = [
            {"event_id": "A", "centroid_lat": "21.0", "centroid_lon": "72.0", "spatial_extent_m": "100",
             "frac_high_confidence": "0.1", "frac_low_confidence": "0.1", "mean_scan": "0.4", "mean_track": "0.4",
             "elongation_ratio": "0.5", "time_of_day_std_minutes": "10", "detections_per_day": "2"},
            {"event_id": "B", "centroid_lat": "22.0", "centroid_lon": "73.0", "spatial_extent_m": "200",
             "frac_high_confidence": "0.2", "frac_low_confidence": "0.2", "mean_scan": "0.5", "mean_track": "0.5",
             "elongation_ratio": "0.6", "time_of_day_std_minutes": "20", "detections_per_day": "3"},
        ]
        result = mod.run_redundancy_audit(feature_rows, existing_rows)
        self.assertIn("centroid_lat", result)
        self.assertIn("centroid_lon", result)
        self.assertEqual(len(result), len(mod.EXISTING_NUMERIC_FEATURES))

    def test_class_conditional_means(self):
        feature_rows = [
            {"event_id": "A", "worldcover_diversity_300m": 6.0, "worldcover_diversity_computable": True},
            {"event_id": "B", "worldcover_diversity_300m": 4.0, "worldcover_diversity_computable": True},
        ]
        silver_rows = [self._silver_row("A", "Industrial"), self._silver_row("B", "Industrial")]
        summary = mod.class_conditional_means(feature_rows, silver_rows)
        self.assertEqual(summary["Industrial"]["n"], 2)
        self.assertAlmostEqual(summary["Industrial"]["mean"], 5.0)


if __name__ == "__main__":
    unittest.main()
