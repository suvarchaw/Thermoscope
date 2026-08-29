import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cluster_persistence_analysis import (
    derived_metrics,
    pearson_r,
    find_largest_gaps,
    per_cluster_date_counts,
)


class TestDerivedMetrics(unittest.TestCase):
    def test_basic_metrics(self):
        cluster = {
            "detection_count": 10,
            "unique_dates": 5,
            "active_span_days": 9,
            "night_count": 8,
            "mean_frp": 2.0,
            "max_frp": 6.0,
        }
        date_counts = {"d1": 4, "d2": 2, "d3": 2, "d4": 1, "d5": 1}
        m = derived_metrics(cluster, date_counts)
        self.assertAlmostEqual(m["detections_per_active_date"], 2.0)
        self.assertAlmostEqual(m["occurrence_rate"], 5 / 10)
        self.assertAlmostEqual(m["night_fraction"], 0.8)
        self.assertAlmostEqual(m["frp_ratio"], 3.0)
        self.assertAlmostEqual(m["top_day_share"], 0.4)
        self.assertAlmostEqual(m["top3_days_share"], 0.8)

    def test_zero_mean_frp_gives_none_ratio(self):
        cluster = {
            "detection_count": 2, "unique_dates": 2, "active_span_days": 1,
            "night_count": 0, "mean_frp": 0.0, "max_frp": 0.0,
        }
        date_counts = {"d1": 1, "d2": 1}
        m = derived_metrics(cluster, date_counts)
        self.assertIsNone(m["frp_ratio"])


class TestPearsonR(unittest.TestCase):
    def test_perfect_positive_correlation(self):
        xs = [1, 2, 3, 4, 5]
        ys = [2, 4, 6, 8, 10]
        self.assertAlmostEqual(pearson_r(xs, ys), 1.0)

    def test_perfect_negative_correlation(self):
        xs = [1, 2, 3, 4, 5]
        ys = [10, 8, 6, 4, 2]
        self.assertAlmostEqual(pearson_r(xs, ys), -1.0)

    def test_no_variance_returns_zero(self):
        xs = [5, 5, 5, 5]
        ys = [1, 2, 3, 4]
        self.assertEqual(pearson_r(xs, ys), 0.0)


class TestFindLargestGaps(unittest.TestCase):
    def test_finds_largest_gap(self):
        values = [1, 2, 3, 50, 51]
        gaps = find_largest_gaps(values, top_n=1)
        self.assertEqual(gaps[0], (47, 3, 50))

    def test_returns_requested_count(self):
        values = [1, 2, 5, 20, 21, 100]
        gaps = find_largest_gaps(values, top_n=3)
        self.assertEqual(len(gaps), 3)
        # sorted descending by gap size
        self.assertGreaterEqual(gaps[0][0], gaps[1][0])
        self.assertGreaterEqual(gaps[1][0], gaps[2][0])


class TestPerClusterDateCounts(unittest.TestCase):
    def test_counts_by_cluster_and_date(self):
        rows = [
            {"cluster_id": "0", "acq_date": "2023-01-01"},
            {"cluster_id": "0", "acq_date": "2023-01-01"},
            {"cluster_id": "0", "acq_date": "2023-01-02"},
            {"cluster_id": "1", "acq_date": "2023-01-01"},
        ]
        result = per_cluster_date_counts(rows)
        self.assertEqual(result[0]["2023-01-01"], 2)
        self.assertEqual(result[0]["2023-01-02"], 1)
        self.assertEqual(result[1]["2023-01-01"], 1)


if __name__ == "__main__":
    unittest.main()
