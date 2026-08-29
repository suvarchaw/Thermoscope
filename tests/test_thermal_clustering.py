import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermal_clustering import (
    haversine_m,
    run_dbscan,
    compute_cluster_stats,
    hazira_cluster_ids,
)


def make_row(lat, lon, date, daynight="D", frp="1.0"):
    return {
        "latitude": str(lat),
        "longitude": str(lon),
        "acq_date": date,
        "daynight": daynight,
        "frp": frp,
    }


class TestHaversine(unittest.TestCase):
    def test_zero_distance(self):
        self.assertAlmostEqual(haversine_m(21.0, 72.0, 21.0, 72.0), 0.0)

    def test_known_small_offset(self):
        # 375m / 111,320 m-per-degree-latitude = 0.0033687 deg offset
        d = haversine_m(21.0, 72.0, 21.0033687, 72.0)
        self.assertAlmostEqual(d, 375.0, delta=1.0)


class TestRunDBSCAN(unittest.TestCase):
    def test_dense_group_forms_one_cluster(self):
        # 10 points tightly packed within ~50m of each other
        rows = []
        for i in range(10):
            rows.append(make_row(21.0 + i * 0.0001, 72.0, f"2023-01-{i+1:02d}"))
        labels = run_dbscan(rows, eps_m=200, min_samples=5)
        non_noise = [l for l in labels if l != -1]
        self.assertEqual(len(non_noise), 10)
        self.assertEqual(len(set(non_noise)), 1)

    def test_isolated_points_become_noise(self):
        # Points far apart (>> eps) relative to each other and too few
        # nearby neighbors to satisfy min_samples.
        rows = [
            make_row(20.0, 68.0, "2023-01-01"),
            make_row(22.0, 70.0, "2023-01-02"),
            make_row(24.0, 74.0, "2023-01-03"),
        ]
        labels = run_dbscan(rows, eps_m=375, min_samples=8)
        self.assertTrue(all(l == -1 for l in labels))

    def test_two_separated_groups_stay_distinct(self):
        group_a = [make_row(21.0 + i * 0.0001, 72.0, f"2023-01-{i+1:02d}") for i in range(8)]
        group_b = [make_row(23.0 + i * 0.0001, 74.0, f"2023-02-{i+1:02d}") for i in range(8)]
        rows = group_a + group_b
        labels = run_dbscan(rows, eps_m=200, min_samples=5)
        labels_a = set(labels[:8])
        labels_b = set(labels[8:])
        self.assertEqual(len(labels_a), 1)
        self.assertEqual(len(labels_b), 1)
        self.assertNotEqual(labels_a, labels_b)
        self.assertNotIn(-1, labels_a)
        self.assertNotIn(-1, labels_b)


class TestComputeClusterStats(unittest.TestCase):
    def test_single_cluster_stats(self):
        rows = [
            make_row(21.0, 72.0, "2023-01-01", "D", "1.0"),
            make_row(21.0001, 72.0001, "2023-01-05", "N", "3.0"),
            make_row(21.0002, 72.0002, "2023-01-10", "D", "2.0"),
        ]
        labels = [0, 0, 0]
        stats = compute_cluster_stats(rows, labels)
        self.assertEqual(len(stats), 1)
        s = stats[0]
        self.assertEqual(s["cluster_id"], 0)
        self.assertEqual(s["detection_count"], 3)
        self.assertEqual(s["unique_dates"], 3)
        self.assertEqual(s["first_date"], "2023-01-01")
        self.assertEqual(s["last_date"], "2023-01-10")
        self.assertEqual(s["active_span_days"], 9)
        self.assertAlmostEqual(s["mean_frp"], 2.0)
        self.assertEqual(s["max_frp"], 3.0)
        self.assertEqual(s["day_count"], 2)
        self.assertEqual(s["night_count"], 1)
        self.assertGreaterEqual(s["extent_radius_m"], 0)

    def test_noise_excluded_from_stats(self):
        rows = [
            make_row(21.0, 72.0, "2023-01-01"),
            make_row(25.0, 76.0, "2023-01-02"),
        ]
        labels = [0, -1]
        stats = compute_cluster_stats(rows, labels)
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["detection_count"], 1)

    def test_no_clusters_returns_empty_list(self):
        rows = [make_row(21.0, 72.0, "2023-01-01")]
        labels = [-1]
        stats = compute_cluster_stats(rows, labels)
        self.assertEqual(stats, [])


class TestHazuraClusterIds(unittest.TestCase):
    def test_identifies_cluster_touching_box(self):
        rows = [
            make_row(21.105, 72.64, "2023-01-01"),  # inside Hazira box
            make_row(25.0, 76.0, "2023-01-02"),      # far outside
        ]
        labels = [3, 7]
        ids = hazira_cluster_ids(rows, labels)
        self.assertEqual(ids, {3})

    def test_noise_never_counted(self):
        rows = [make_row(21.105, 72.64, "2023-01-01")]
        labels = [-1]
        ids = hazira_cluster_ids(rows, labels)
        self.assertEqual(ids, set())


if __name__ == "__main__":
    unittest.main()
