import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cluster_longitudinal_features import (
    parse_year_count_string,
    zero_fill,
    compute_trend,
    compute_annual_stats,
    compute_monthly_distribution,
    compute_seasonality,
    build_feature_row,
    YEARS,
)


class TestParseYearCountString(unittest.TestCase):
    def test_parses_multiple_entries(self):
        self.assertEqual(parse_year_count_string("2019:5;2021:3"), {2019: 5, 2021: 3})

    def test_empty_string_returns_empty_dict(self):
        self.assertEqual(parse_year_count_string(""), {})

    def test_single_entry(self):
        self.assertEqual(parse_year_count_string("2023:100"), {2023: 100})


class TestZeroFill(unittest.TestCase):
    def test_fills_missing_keys_with_zero(self):
        self.assertEqual(zero_fill({2019: 5, 2021: 3}, [2019, 2020, 2021]), [5, 0, 3])

    def test_all_present(self):
        self.assertEqual(zero_fill({1: 10, 2: 20}, [1, 2]), [10, 20])

    def test_none_present(self):
        self.assertEqual(zero_fill({}, [1, 2, 3]), [0, 0, 0])


class TestComputeTrend(unittest.TestCase):
    def test_clearly_increasing(self):
        slope, direction = compute_trend([10, 20, 30, 40, 50])
        self.assertGreater(slope, 0)
        self.assertEqual(direction, "increasing")

    def test_clearly_decreasing(self):
        slope, direction = compute_trend([50, 40, 30, 20, 10])
        self.assertLess(slope, 0)
        self.assertEqual(direction, "decreasing")

    def test_flat_is_stable(self):
        slope, direction = compute_trend([100, 100, 100, 100, 100])
        self.assertAlmostEqual(slope, 0.0)
        self.assertEqual(direction, "stable")

    def test_small_fluctuation_within_threshold_is_stable(self):
        # mean=100, slope should be small relative to mean -> stable
        slope, direction = compute_trend([98, 101, 100, 99, 102])
        self.assertEqual(direction, "stable")

    def test_all_zero_is_stable_no_division_error(self):
        slope, direction = compute_trend([0, 0, 0, 0, 0])
        self.assertEqual(direction, "stable")
        self.assertEqual(slope, 0.0)


class TestComputeAnnualStats(unittest.TestCase):
    def test_known_values(self):
        total, mean, std, cv = compute_annual_stats([10, 20, 30, 40, 50])
        self.assertEqual(total, 150)
        self.assertEqual(mean, 30)
        self.assertGreater(std, 0)
        self.assertAlmostEqual(cv, std / mean)

    def test_constant_series_has_zero_std_and_cv(self):
        total, mean, std, cv = compute_annual_stats([10, 10, 10, 10, 10])
        self.assertEqual(std, 0.0)
        self.assertEqual(cv, 0.0)

    def test_zero_mean_does_not_crash(self):
        total, mean, std, cv = compute_annual_stats([0, 0, 0, 0, 0])
        self.assertEqual(cv, 0.0)


class TestComputeMonthlyDistribution(unittest.TestCase):
    def test_counts_by_month(self):
        rows = [
            {"acq_date": "2020-01-05"}, {"acq_date": "2020-01-10"},
            {"acq_date": "2021-06-01"},
        ]
        result = compute_monthly_distribution(rows)
        self.assertEqual(result, {1: 2, 6: 1})

    def test_empty_input(self):
        self.assertEqual(compute_monthly_distribution([]), {})


class TestComputeSeasonality(unittest.TestCase):
    def test_highly_concentrated_seasonality(self):
        # All detections in one month -> top3_months_share should be 1.0
        counts = [100] + [0] * 11
        top3_share, dominant = compute_seasonality(counts)
        self.assertAlmostEqual(top3_share, 1.0)
        self.assertEqual(dominant, 1)

    def test_evenly_spread_across_year(self):
        counts = [10] * 12
        top3_share, dominant = compute_seasonality(counts)
        self.assertAlmostEqual(top3_share, 3 / 12)

    def test_all_zero_returns_none_dominant(self):
        top3_share, dominant = compute_seasonality([0] * 12)
        self.assertEqual(top3_share, 0.0)
        self.assertIsNone(dominant)

    def test_dominant_month_identifies_correct_index(self):
        counts = [1, 2, 3, 50, 2, 1, 1, 1, 1, 1, 1, 1]
        _, dominant = compute_seasonality(counts)
        self.assertEqual(dominant, 4)  # index 3 -> month 4


class TestBuildFeatureRow(unittest.TestCase):
    def test_zero_fills_missing_years_and_computes_expected_fields(self):
        profile_row = {"cluster_id": "7", "recurrence_strength": "Moderate"}
        cross_year_row = {
            "years_detected": "2021;2023",
            "unique_years": "2",
            "first_year": "2021",
            "last_year": "2023",
            "recurs_across_multiple_years": "True",
            "per_year_detection_counts": "2021:10;2023:20",
            "per_year_unique_dates": "2021:5;2023:8",
        }
        detection_rows = [
            {"acq_date": "2021-03-01"}, {"acq_date": "2023-03-01"},
        ]
        row = build_feature_row(7, profile_row, cross_year_row, detection_rows)

        self.assertEqual(row["detections_2019"], 0)
        self.assertEqual(row["detections_2020"], 0)
        self.assertEqual(row["detections_2021"], 10)
        self.assertEqual(row["detections_2022"], 0)
        self.assertEqual(row["detections_2023"], 20)
        self.assertEqual(row["total_detections_5yr"], 30)
        self.assertEqual(row["unique_years"], 2)
        self.assertEqual(row["recurrence_strength"], "Moderate")  # carried through unchanged


if __name__ == "__main__":
    unittest.main()
