import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cluster_evidence_deep_dive import (
    iqr_bounds,
    day_night_pattern,
    compute_outlier_flags,
    compute_definition_satisfaction,
    ILLUSTRATIVE_DEFINITIONS,
)


class TestIqrBounds(unittest.TestCase):
    def test_no_outliers_in_uniform_data(self):
        values = list(range(1, 21))  # 1..20, no outliers
        lo, hi = iqr_bounds(values)
        self.assertTrue(all(lo <= v <= hi for v in values))

    def test_extreme_value_detected_outside_bounds(self):
        values = list(range(1, 20)) + [1000]
        lo, hi = iqr_bounds(values)
        self.assertGreater(1000, hi)


class TestDayNightPattern(unittest.TestCase):
    def test_mostly_night(self):
        self.assertEqual(day_night_pattern(0.95), "mostly_night")

    def test_mostly_day(self):
        self.assertEqual(day_night_pattern(0.1), "mostly_day")

    def test_mixed_boundaries(self):
        self.assertEqual(day_night_pattern(0.3), "mixed")
        self.assertEqual(day_night_pattern(0.7), "mixed")
        self.assertEqual(day_night_pattern(0.5), "mixed")

    def test_exactly_at_upper_edge_is_not_mostly_night(self):
        # night_fraction must be STRICTLY > 0.7 to count as mostly_night,
        # matching the original nighttime_dominance_investigation.py cutoffs
        self.assertEqual(day_night_pattern(0.7), "mixed")
        self.assertEqual(day_night_pattern(0.70001), "mostly_night")


class TestComputeOutlierFlags(unittest.TestCase):
    def test_flags_by_cluster_id_key(self):
        rows = [{"cluster_id": i, "unique_dates": i, "occurrence_rate": 0.1, "detection_count": 10,
                 "active_span_days": 100, "mean_frp": 1.0, "max_frp": 2.0,
                 "frp_ratio": 2.0, "top_day_share": 0.1} for i in range(1, 20)]
        rows.append({**rows[0], "cluster_id": 999, "unique_dates": 1000})
        flags, bounds = compute_outlier_flags(rows)
        self.assertTrue(flags[999]["unique_dates"])
        self.assertFalse(flags[5]["unique_dates"])


class TestComputeDefinitionSatisfaction(unittest.TestCase):
    def test_satisfies_all_when_criteria_met(self):
        row = {
            "cluster_id": 0, "unique_dates": 300, "occurrence_rate": 0.9,
            "active_span_days": 364,
        }
        result = compute_definition_satisfaction([row])
        self.assertTrue(all(result[0].values()))

    def test_satisfies_none_when_criteria_unmet(self):
        row = {
            "cluster_id": 0, "unique_dates": 1, "occurrence_rate": 0.01,
            "active_span_days": 1,
        }
        result = compute_definition_satisfaction([row])
        self.assertFalse(any(result[0].values()))

    def test_number_of_definitions_matches_module_constant(self):
        row = {"cluster_id": 0, "unique_dates": 50, "occurrence_rate": 0.15, "active_span_days": 200}
        result = compute_definition_satisfaction([row])
        self.assertEqual(len(result[0]), len(ILLUSTRATIVE_DEFINITIONS))


if __name__ == "__main__":
    unittest.main()
