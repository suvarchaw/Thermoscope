import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import event_behavior_features as mod


def make_det(lat, lon, d, confidence="n", scan=0.4, track=0.4, acq_time="0800"):
    return {"lat": lat, "lon": lon, "date": d, "confidence": confidence,
            "scan": scan, "track": track, "acq_time": acq_time,
            "frp": 5.0, "daynight": "D", "brightness": 330.0, "bright_t31": 300.0}


class TestFeatureListSafety(unittest.TestCase):
    def test_lists_clean_on_default_state(self):
        mod.assert_feature_lists_clean()  # should not raise

    def test_new_features_never_overlap_forbidden(self):
        self.assertEqual(set(mod.NEW_FEATURES) & mod.FORBIDDEN_COLUMNS, set())

    def test_baseline_features_never_overlap_forbidden(self):
        self.assertEqual(set(mod.BASELINE_FEATURES) & mod.FORBIDDEN_COLUMNS, set())

    def test_excluded_leaky_candidates_are_in_forbidden(self):
        for name in ("mean_brightness", "mean_bright_t31", "mean_brightness_t31_diff", "unique_days"):
            self.assertIn(name, mod.FORBIDDEN_COLUMNS)

    def test_feature_audit_included_matches_new_features(self):
        audited_included = {r["name"] for r in mod.FEATURE_AUDIT if r["included"]}
        self.assertEqual(audited_included, set(mod.NEW_FEATURES))

    def test_assert_catches_injected_forbidden_feature(self):
        mod.NEW_FEATURES.append("mean_frp")
        try:
            with self.assertRaises(ValueError):
                mod.assert_feature_lists_clean()
        finally:
            mod.NEW_FEATURES.remove("mean_frp")

    def test_every_excluded_audit_row_has_a_reason(self):
        for r in mod.FEATURE_AUDIT:
            if not r["included"]:
                self.assertTrue(len(r["reason"]) > 0)


class TestElongationRatio(unittest.TestCase):
    def test_two_points_are_perfectly_linear(self):
        pts = [{"lat": 21.0, "lon": 72.0}, {"lat": 21.01, "lon": 72.0}]
        ratio = mod.compute_elongation_ratio(pts, 21.005, 72.0)
        self.assertAlmostEqual(ratio, 0.0, places=6)

    def test_symmetric_square_is_close_to_circular(self):
        pts = [{"lat": 21.001, "lon": 72.0}, {"lat": 20.999, "lon": 72.0},
               {"lat": 21.0, "lon": 72.001}, {"lat": 21.0, "lon": 71.999}]
        ratio = mod.compute_elongation_ratio(pts, 21.0, 72.0)
        self.assertGreater(ratio, 0.8)  # near-1 for a symmetric cross pattern

    def test_ratio_always_between_zero_and_one(self):
        pts = [{"lat": 21.0 + 0.001 * i, "lon": 72.0 + 0.0003 * i} for i in range(5)]
        clat = sum(p["lat"] for p in pts) / 5
        clon = sum(p["lon"] for p in pts) / 5
        ratio = mod.compute_elongation_ratio(pts, clat, clon)
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio, 1.0)


class TestAcqTimeToMinutes(unittest.TestCase):
    def test_midnight(self):
        self.assertEqual(mod.acq_time_to_minutes("0000"), 0)

    def test_noon(self):
        self.assertEqual(mod.acq_time_to_minutes("1200"), 720)

    def test_short_string_zfilled(self):
        self.assertEqual(mod.acq_time_to_minutes("745"), 7 * 60 + 45)


class TestComputeNewFeaturesForEvent(unittest.TestCase):
    def test_confidence_fractions(self):
        dets = [make_det(21.0, 72.0, date(2023, 1, 1), confidence="h"),
                make_det(21.0, 72.0, date(2023, 1, 1), confidence="l"),
                make_det(21.0, 72.0, date(2023, 1, 1), confidence="n"),
                make_det(21.0, 72.0, date(2023, 1, 1), confidence="n")]
        feats = mod.compute_new_features_for_event([0, 1, 2, 3], dets)
        self.assertAlmostEqual(feats["frac_high_confidence"], 0.25)
        self.assertAlmostEqual(feats["frac_low_confidence"], 0.25)

    def test_detections_per_day(self):
        dets = [make_det(21.0, 72.0, date(2023, 1, 1)), make_det(21.0, 72.0, date(2023, 1, 1)),
                make_det(21.0, 72.0, date(2023, 1, 2))]
        feats = mod.compute_new_features_for_event([0, 1, 2], dets)
        self.assertAlmostEqual(feats["detections_per_day"], 3 / 2)

    def test_time_of_day_std_zero_for_identical_times(self):
        dets = [make_det(21.0, 72.0, date(2023, 1, 1), acq_time="0800"),
                make_det(21.0, 72.0, date(2023, 1, 2), acq_time="0800")]
        feats = mod.compute_new_features_for_event([0, 1], dets)
        self.assertAlmostEqual(feats["time_of_day_std_minutes"], 0.0)

    def test_mean_scan_track(self):
        dets = [make_det(21.0, 72.0, date(2023, 1, 1), scan=0.4, track=0.6),
                make_det(21.0, 72.0, date(2023, 1, 1), scan=0.6, track=0.4)]
        feats = mod.compute_new_features_for_event([0, 1], dets)
        self.assertAlmostEqual(feats["mean_scan"], 0.5)
        self.assertAlmostEqual(feats["mean_track"], 0.5)


class TestSortedEventGroupsAndAlignment(unittest.TestCase):
    def test_sorted_groups_deterministic_ordering(self):
        dets = [
            make_det(22.0, 70.0, date(2023, 3, 5)), make_det(22.0, 70.0, date(2023, 3, 6)),
            make_det(21.0, 72.0, date(2023, 1, 1)), make_det(21.0, 72.0, date(2023, 1, 2)),
        ]
        groups = mod.sorted_event_groups(dets)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0][0], date(2023, 1, 1))  # earlier-starting group first
        self.assertEqual(groups[1][0], date(2023, 3, 5))

    def test_verify_alignment_passes_on_matching_data(self):
        dets = [make_det(21.0, 72.0, date(2023, 1, 1)), make_det(21.0, 72.0, date(2023, 1, 2))]
        groups = mod.sorted_event_groups(dets)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.csv"
            with open(path, "w", newline="") as f:
                f.write("event_id,start_date,end_date,detection_count\n")
                f.write("EVT000000,2023-01-01,2023-01-02,2\n")
            self.assertTrue(mod.verify_alignment(groups, events_path=path))

    def test_verify_alignment_raises_on_mismatch(self):
        dets = [make_det(21.0, 72.0, date(2023, 1, 1)), make_det(21.0, 72.0, date(2023, 1, 2))]
        groups = mod.sorted_event_groups(dets)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.csv"
            with open(path, "w", newline="") as f:
                f.write("event_id,start_date,end_date,detection_count\n")
                f.write("EVT000000,2023-01-01,2023-01-05,2\n")  # wrong end_date
            with self.assertRaises(ValueError):
                mod.verify_alignment(groups, events_path=path)

    def test_verify_alignment_raises_on_count_mismatch(self):
        dets = [make_det(21.0, 72.0, date(2023, 1, 1)), make_det(21.0, 72.0, date(2023, 1, 2))]
        groups = mod.sorted_event_groups(dets)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.csv"
            with open(path, "w", newline="") as f:
                f.write("event_id,start_date,end_date,detection_count\n")
            with self.assertRaises(ValueError):
                mod.verify_alignment(groups, events_path=path)


class TestFeatureSetsDefinition(unittest.TestCase):
    def test_three_feature_sets_defined(self):
        self.assertEqual(set(mod.FEATURE_SETS.keys()),
                          {"A_baseline_geography_only", "B_new_behavior_only", "C_geography_plus_behavior"})

    def test_set_c_is_union_of_a_and_b(self):
        a_num, a_cat = mod.FEATURE_SETS["A_baseline_geography_only"]
        b_num, b_cat = mod.FEATURE_SETS["B_new_behavior_only"]
        c_num, c_cat = mod.FEATURE_SETS["C_geography_plus_behavior"]
        self.assertEqual(set(c_num), set(a_num) | set(b_num))
        self.assertEqual(set(c_cat), set(a_cat) | set(b_cat))

    def test_set_b_has_no_geography(self):
        b_num, b_cat = mod.FEATURE_SETS["B_new_behavior_only"]
        self.assertNotIn("centroid_lat", b_num)
        self.assertNotIn("centroid_lon", b_num)


class TestRowsToArraysGeneric(unittest.TestCase):
    def test_numeric_only_shape(self):
        rows = [{"a": "1.0", "b": "2.0", "silver_label": "Industrial"},
                {"a": "3.0", "b": "4.0", "silver_label": "Gas_Flare"}]
        Xn, Xc, y = mod.rows_to_arrays(rows, ["a", "b"], [])
        self.assertEqual(Xn.shape, (2, 2))
        self.assertEqual(Xc.shape, (2, 0))
        combined = mod.combine_features(Xn, Xc)
        self.assertEqual(combined.shape, (2, 2))

    def test_with_categorical(self):
        rows = [{"a": "1.0", "status": "closed", "silver_label": "Industrial"}]
        Xn, Xc, y = mod.rows_to_arrays(rows, ["a"], ["status"])
        combined = mod.combine_features(Xn, Xc)
        self.assertEqual(combined.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
