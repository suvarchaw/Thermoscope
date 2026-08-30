import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cross_year_recurrence_analysis import (
    haversine_m,
    match_historical_to_baseline,
    compute_cross_year_metrics,
    plot_cross_year_recurrence,
)


def make_cluster_def(cid, lat, lon, extent_radius_m):
    return {"cluster_id": cid, "centroid_lat": lat, "centroid_lon": lon,
            "extent_radius_m": extent_radius_m}


def make_detection(lat, lon, acq_date):
    return {"latitude": str(lat), "longitude": str(lon), "acq_date": acq_date}


class TestHaversine(unittest.TestCase):
    def test_zero_distance(self):
        self.assertAlmostEqual(haversine_m(21.0, 72.0, 21.0, 72.0), 0.0)

    def test_known_small_offset(self):
        d = haversine_m(21.0, 72.0, 21.0033687, 72.0)
        self.assertAlmostEqual(d, 375.0, delta=1.0)


class TestMatchHistoricalToBaseline(unittest.TestCase):
    def test_point_within_radius_matches_nearest_cluster(self):
        clusters = [make_cluster_def(0, 21.0, 72.0, 500)]
        rows = [make_detection(21.001, 72.0, "2020-01-01")]  # ~111m away
        matched = match_historical_to_baseline(rows, clusters)
        self.assertEqual(matched[0]["cluster_id"], 0)
        self.assertEqual(matched[0]["match_method"], "nearest_centroid_within_radius")

    def test_point_outside_all_radii_is_unmatched(self):
        clusters = [make_cluster_def(0, 21.0, 72.0, 100)]
        rows = [make_detection(22.0, 73.0, "2020-01-01")]  # far away
        matched = match_historical_to_baseline(rows, clusters)
        self.assertEqual(matched[0]["cluster_id"], -1)
        self.assertEqual(matched[0]["match_method"], "unmatched_historical")

    def test_picks_nearest_of_multiple_clusters(self):
        clusters = [
            make_cluster_def(0, 21.0, 72.0, 1000),
            make_cluster_def(1, 21.01, 72.01, 1000),
        ]
        # Closer to cluster 1
        rows = [make_detection(21.009, 72.009, "2020-01-01")]
        matched = match_historical_to_baseline(rows, clusters)
        self.assertEqual(matched[0]["cluster_id"], 1)

    def test_exactly_at_radius_boundary_matches(self):
        # A point at exactly 375m offset from a cluster whose extent_radius_m
        # is (approximately) 375m should match (<=), not be excluded.
        clusters = [make_cluster_def(0, 21.0, 72.0, 375.5)]
        rows = [make_detection(21.0033687, 72.0, "2020-01-01")]  # ~375m away
        matched = match_historical_to_baseline(rows, clusters)
        self.assertEqual(matched[0]["cluster_id"], 0)


class TestComputeCrossYearMetrics(unittest.TestCase):
    def test_single_year_cluster(self):
        cluster_defs = [make_cluster_def(0, 21.0, 72.0, 500)]
        assignments = [
            {"cluster_id": 0, "year": 2023, "acq_date": "2023-01-01"},
            {"cluster_id": 0, "year": 2023, "acq_date": "2023-01-05"},
        ]
        results = compute_cross_year_metrics(assignments, cluster_defs)
        r = results[0]
        self.assertEqual(r["unique_years"], 1)
        self.assertFalse(r["recurs_across_multiple_years"])
        self.assertEqual(r["first_year"], 2023)
        self.assertEqual(r["last_year"], 2023)

    def test_multi_year_cluster_recurs_flag(self):
        cluster_defs = [make_cluster_def(0, 21.0, 72.0, 500)]
        assignments = [
            {"cluster_id": 0, "year": 2021, "acq_date": "2021-03-01"},
            {"cluster_id": 0, "year": 2023, "acq_date": "2023-01-01"},
            {"cluster_id": 0, "year": 2023, "acq_date": "2023-01-05"},
        ]
        results = compute_cross_year_metrics(assignments, cluster_defs)
        r = results[0]
        self.assertEqual(r["unique_years"], 2)
        self.assertTrue(r["recurs_across_multiple_years"])
        self.assertEqual(r["first_year"], 2021)
        self.assertEqual(r["last_year"], 2023)
        self.assertIn("2021:1", r["per_year_detection_counts"])
        self.assertIn("2023:2", r["per_year_detection_counts"])

    def test_cluster_with_no_detections_at_all(self):
        cluster_defs = [make_cluster_def(5, 21.0, 72.0, 500)]
        results = compute_cross_year_metrics([], cluster_defs)
        r = results[0]
        self.assertEqual(r["unique_years"], 0)
        self.assertFalse(r["recurs_across_multiple_years"])
        self.assertEqual(r["first_year"], "")

    def test_unmatched_detections_excluded_from_cluster_metrics(self):
        cluster_defs = [make_cluster_def(0, 21.0, 72.0, 500)]
        assignments = [
            {"cluster_id": -1, "year": 2021, "acq_date": "2021-03-01"},
        ]
        results = compute_cross_year_metrics(assignments, cluster_defs)
        self.assertEqual(results[0]["unique_years"], 0)


class TestPlotCrossYearRecurrence(unittest.TestCase):
    def test_single_year_branch_does_not_crash(self):
        rows = [{"cluster_id": i, "unique_years": 1,
                  "per_year_detection_counts": "2023:10"} for i in range(5)]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "fig.png"
            plot_cross_year_recurrence(rows, {2023: 100}, output_path=out_path)
            self.assertTrue(out_path.exists())

    def test_multi_year_heatmap_branch_does_not_crash(self):
        rows = [
            {"cluster_id": 0, "unique_years": 2,
             "per_year_detection_counts": "2022:5;2023:10"},
            {"cluster_id": 1, "unique_years": 1,
             "per_year_detection_counts": "2023:3"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "fig.png"
            plot_cross_year_recurrence(rows, {2022: 50, 2023: 100}, output_path=out_path)
            self.assertTrue(out_path.exists())


if __name__ == "__main__":
    unittest.main()
