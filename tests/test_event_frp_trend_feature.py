import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import event_frp_trend_feature as mod


def make_det(d, frp, lat=21.0, lon=72.0):
    return {"lat": lat, "lon": lon, "date": d, "frp": frp, "daynight": "D"}


class TestOlsSlope(unittest.TestCase):
    def test_perfectly_increasing(self):
        self.assertAlmostEqual(mod.ols_slope([0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0]), 1.0)

    def test_perfectly_decreasing(self):
        self.assertAlmostEqual(mod.ols_slope([0, 1, 2, 3], [4.0, 3.0, 2.0, 1.0]), -1.0)

    def test_flat(self):
        self.assertAlmostEqual(mod.ols_slope([0, 1, 2, 3], [5.0, 5.0, 5.0, 5.0]), 0.0)


class TestComputeFrpTrendIncreasingDecreasingFlat(unittest.TestCase):
    def test_increasing_frp_gives_positive_slope(self):
        dets = [make_det(date(2023, 1, 1), 2.0), make_det(date(2023, 1, 2), 5.0),
                make_det(date(2023, 1, 3), 8.0)]
        slope, computable, n_days = mod.compute_frp_trend([0, 1, 2], dets)
        self.assertTrue(computable)
        self.assertGreater(slope, 0)
        self.assertAlmostEqual(slope, 3.0)
        self.assertEqual(n_days, 3)

    def test_decreasing_frp_gives_negative_slope(self):
        dets = [make_det(date(2023, 1, 1), 9.0), make_det(date(2023, 1, 2), 6.0),
                make_det(date(2023, 1, 3), 3.0)]
        slope, computable, n_days = mod.compute_frp_trend([0, 1, 2], dets)
        self.assertTrue(computable)
        self.assertLess(slope, 0)
        self.assertAlmostEqual(slope, -3.0)

    def test_flat_frp_gives_zero_slope(self):
        dets = [make_det(date(2023, 1, 1), 4.0), make_det(date(2023, 1, 2), 4.0),
                make_det(date(2023, 1, 3), 4.0)]
        slope, computable, n_days = mod.compute_frp_trend([0, 1, 2], dets)
        self.assertTrue(computable)
        self.assertAlmostEqual(slope, 0.0)


class TestComputeFrpTrendEdgeCases(unittest.TestCase):
    def test_single_day_single_observation_not_computable(self):
        dets = [make_det(date(2023, 1, 1), 5.0)]
        slope, computable, n_days = mod.compute_frp_trend([0], dets)
        self.assertIsNone(slope)
        self.assertFalse(computable)
        self.assertEqual(n_days, 1)

    def test_multiple_detections_same_day_still_single_day_not_computable(self):
        dets = [make_det(date(2023, 1, 1), 2.0), make_det(date(2023, 1, 1), 8.0),
                make_det(date(2023, 1, 1), 5.0)]
        slope, computable, n_days = mod.compute_frp_trend([0, 1, 2], dets)
        self.assertFalse(computable)
        self.assertEqual(n_days, 1)

    def test_multiple_detections_per_day_are_averaged_before_fitting(self):
        # day1: two detections averaging to 2.0; day2: one detection at 6.0
        dets = [make_det(date(2023, 1, 1), 1.0), make_det(date(2023, 1, 1), 3.0),
                make_det(date(2023, 1, 2), 6.0)]
        slope, computable, n_days = mod.compute_frp_trend([0, 1, 2], dets)
        self.assertTrue(computable)
        self.assertEqual(n_days, 2)
        # slope between (0, 2.0) and (1, 6.0) = 4.0
        self.assertAlmostEqual(slope, 4.0)

    def test_exactly_two_days_is_minimum_computable(self):
        dets = [make_det(date(2023, 1, 1), 1.0), make_det(date(2023, 1, 2), 3.0)]
        slope, computable, n_days = mod.compute_frp_trend([0, 1], dets)
        self.assertTrue(computable)
        self.assertEqual(n_days, 2)


class TestDateOrdering(unittest.TestCase):
    def test_input_order_does_not_affect_result(self):
        # members list given out of chronological order -- must still
        # produce the correct day-index-ordered trend.
        dets = [make_det(date(2023, 1, 3), 9.0), make_det(date(2023, 1, 1), 1.0),
                make_det(date(2023, 1, 2), 5.0)]
        slope, computable, n_days = mod.compute_frp_trend([0, 1, 2], dets)
        self.assertTrue(computable)
        self.assertAlmostEqual(slope, 4.0)  # 1.0 -> 5.0 -> 9.0, still +4/day chronologically

    def test_member_index_order_does_not_affect_result(self):
        dets = [make_det(date(2023, 1, 1), 1.0), make_det(date(2023, 1, 2), 5.0),
                make_det(date(2023, 1, 3), 9.0)]
        slope_forward, _, _ = mod.compute_frp_trend([0, 1, 2], dets)
        slope_reversed, _, _ = mod.compute_frp_trend([2, 1, 0], dets)
        self.assertAlmostEqual(slope_forward, slope_reversed)


class TestDeterminism(unittest.TestCase):
    def test_repeated_calls_identical(self):
        dets = [make_det(date(2023, 1, 1), 2.0), make_det(date(2023, 1, 2), 4.0),
                make_det(date(2023, 1, 3), 6.0), make_det(date(2023, 1, 5), 10.0)]
        r1 = mod.compute_frp_trend([0, 1, 2, 3], dets)
        r2 = mod.compute_frp_trend([0, 1, 2, 3], dets)
        self.assertEqual(r1, r2)

    def test_uses_day_index_not_calendar_gap(self):
        # days 1, 2, 5 (a gap) -- slope must be per EVENT-DAY-INDEX (0,1,2),
        # not per calendar day, so a gap does not distort the slope.
        dets = [make_det(date(2023, 1, 1), 0.0), make_det(date(2023, 1, 2), 10.0),
                make_det(date(2023, 1, 5), 20.0)]
        slope, computable, n_days = mod.compute_frp_trend([0, 1, 2], dets)
        self.assertTrue(computable)
        self.assertAlmostEqual(slope, 10.0)  # +10 per index step, not per calendar day


class TestWriteCsvMissingValueBehavior(unittest.TestCase):
    def test_non_computable_slope_written_blank_not_zero(self):
        import csv
        import tempfile
        rows = [{"event_id": "EVT000000", "frp_trend_slope": "", "frp_trend_computable": False, "n_unique_days": 1}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.csv"
            mod.write_csv(rows, path=path)
            with open(path, newline="") as f:
                written = list(csv.DictReader(f))
            self.assertEqual(written[0]["frp_trend_slope"], "")
            self.assertEqual(written[0]["frp_trend_computable"], "False")

    def test_computable_slope_written_as_number(self):
        import csv
        import tempfile
        rows = [{"event_id": "EVT000000", "frp_trend_slope": 2.5, "frp_trend_computable": True, "n_unique_days": 3}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.csv"
            mod.write_csv(rows, path=path)
            with open(path, newline="") as f:
                written = list(csv.DictReader(f))
            self.assertEqual(written[0]["frp_trend_slope"], "2.5")


class TestPearson(unittest.TestCase):
    def test_perfect_positive_correlation(self):
        self.assertAlmostEqual(mod.pearson([1, 2, 3, 4], [2, 4, 6, 8]), 1.0)

    def test_no_correlation_constant_y(self):
        r = mod.pearson([1, 2, 3, 4], [5, 5, 5, 5])
        self.assertTrue(r != r)  # NaN, since sy=0


class TestLeakageAuditAndClassMeans(unittest.TestCase):
    def _silver_row(self, event_id, label, **overrides):
        row = {"event_id": event_id, "silver_label": label, "mean_frp": "3.0", "night_fraction": "0.0",
               "duration_days": "2", "detection_count": "2", "max_frp": "3.0"}
        row.update(overrides)
        return row

    def test_run_leakage_audit_skips_non_computable_rows(self):
        feature_rows = [
            {"event_id": "A", "frp_trend_slope": 1.0, "frp_trend_computable": True, "n_unique_days": 2},
            {"event_id": "B", "frp_trend_slope": "", "frp_trend_computable": False, "n_unique_days": 1},
        ]
        silver_rows = [self._silver_row("A", "Industrial"), self._silver_row("B", "Industrial")]
        correlations, is_safe, n_checked = mod.run_leakage_audit(feature_rows, silver_rows)
        self.assertEqual(n_checked, 1)  # only the computable row counted

    def test_run_leakage_audit_excludes_unknown_ambiguous_and_brick_kiln(self):
        feature_rows = [
            {"event_id": "A", "frp_trend_slope": 1.0, "frp_trend_computable": True, "n_unique_days": 2},
            {"event_id": "B", "frp_trend_slope": 2.0, "frp_trend_computable": True, "n_unique_days": 2},
        ]
        silver_rows = [self._silver_row("A", "Unknown_Ambiguous"), self._silver_row("B", "Industrial")]
        correlations, is_safe, n_checked = mod.run_leakage_audit(feature_rows, silver_rows)
        self.assertEqual(n_checked, 1)

    def test_class_conditional_means_computed_correctly(self):
        feature_rows = [
            {"event_id": "A", "frp_trend_slope": 1.0, "frp_trend_computable": True, "n_unique_days": 2},
            {"event_id": "B", "frp_trend_slope": 3.0, "frp_trend_computable": True, "n_unique_days": 2},
            {"event_id": "C", "frp_trend_slope": "", "frp_trend_computable": False, "n_unique_days": 1},
        ]
        silver_rows = [self._silver_row("A", "Industrial"), self._silver_row("B", "Industrial"),
                       self._silver_row("C", "Industrial")]
        summary = mod.class_conditional_means(feature_rows, silver_rows)
        self.assertEqual(summary["Industrial"]["n"], 2)
        self.assertAlmostEqual(summary["Industrial"]["mean"], 2.0)


class TestCausalOnlineSafety(unittest.TestCase):
    """The feature must only ever use detections that are already members
    of the (causally-constructed) event -- never information from a
    future day beyond what the event currently contains."""

    def test_truncating_future_detections_changes_result_only_by_removing_future_days(self):
        full_dets = [make_det(date(2023, 1, 1), 1.0), make_det(date(2023, 1, 2), 2.0),
                     make_det(date(2023, 1, 3), 3.0), make_det(date(2023, 1, 4), 100.0)]
        # "as of day 3" -- the event only has 3 days of data so far
        truncated_members = [0, 1, 2]
        slope_truncated, computable, n_days = mod.compute_frp_trend(truncated_members, full_dets)
        self.assertTrue(computable)
        self.assertEqual(n_days, 3)
        # the huge future value (100.0 on day 4) must not influence a
        # computation restricted to members [0,1,2]
        self.assertAlmostEqual(slope_truncated, 1.0)


if __name__ == "__main__":
    unittest.main()
