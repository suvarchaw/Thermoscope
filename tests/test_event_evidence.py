import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import event_evidence as mod


class TestHaversine(unittest.TestCase):
    def test_zero_distance(self):
        self.assertAlmostEqual(mod.haversine_m(21.0, 72.0, 21.0, 72.0), 0.0)

    def test_known_offset(self):
        d = mod.haversine_m(21.0, 72.0, 21.0033687, 72.0)
        self.assertAlmostEqual(d, 375.0, delta=1.0)


class TestPointGridNearestDistance(unittest.TestCase):
    def test_finds_nearby_point(self):
        points = [(21.0, 72.0), (22.0, 73.0)]
        grid, cla, clo = mod.build_point_grid(points)
        d = mod.nearest_distance_m(21.001, 72.0, grid, cla, clo)
        self.assertIsNotNone(d)
        self.assertLess(d, 200)

    def test_returns_none_beyond_max_search(self):
        points = [(21.0, 72.0)]
        grid, cla, clo = mod.build_point_grid(points)
        # ~1.1 degrees away -- well over 100km, beyond a 5km search cap
        d = mod.nearest_distance_m(22.0, 72.0, grid, cla, clo, max_search_m=5000)
        self.assertIsNone(d)

    def test_finds_point_near_edge_of_search_radius(self):
        # Point placed ~9km away; default search cap is 20km -- must
        # still be found even though it's several grid cells distant
        # (this is exactly the bug fixed before running the real join:
        # a fixed-size neighbor window would silently miss this).
        points = [(21.08, 72.0)]  # ~8.9km north of (21.0, 72.0)
        grid, cla, clo = mod.build_point_grid(points)
        d = mod.nearest_distance_m(21.0, 72.0, grid, cla, clo, max_search_m=20000)
        self.assertIsNotNone(d)
        self.assertGreater(d, 8000)
        self.assertLess(d, 10000)

    def test_empty_points_returns_none(self):
        grid, cla, clo = mod.build_point_grid([])
        d = mod.nearest_distance_m(21.0, 72.0, grid, cla, clo)
        self.assertIsNone(d)

    def test_picks_nearest_of_multiple_candidates(self):
        points = [(21.0, 72.0), (21.001, 72.0), (21.1, 72.0)]
        grid, cla, clo = mod.build_point_grid(points)
        d = mod.nearest_distance_m(21.0, 72.0, grid, cla, clo)
        self.assertAlmostEqual(d, 0.0, delta=1.0)


class TestClusterOverlap(unittest.TestCase):
    def _clusters(self):
        return [
            {"cluster_id": 0, "centroid_lat": 21.0, "centroid_lon": 72.0, "extent_radius_m": 500.0},
            {"cluster_id": 1, "centroid_lat": 22.0, "centroid_lon": 73.0, "extent_radius_m": 1000.0},
        ]

    def test_within_radius_matches(self):
        cid, dist = mod.find_overlapping_cluster(21.001, 72.0, self._clusters())
        self.assertEqual(cid, 0)
        self.assertIsNotNone(dist)

    def test_outside_all_radii_returns_none(self):
        cid, dist = mod.find_overlapping_cluster(23.0, 74.0, self._clusters())
        self.assertIsNone(cid)
        self.assertIsNone(dist)

    def test_picks_nearest_when_multiple_in_range(self):
        clusters = [
            {"cluster_id": 0, "centroid_lat": 21.0, "centroid_lon": 72.0, "extent_radius_m": 2000.0},
            {"cluster_id": 1, "centroid_lat": 21.005, "centroid_lon": 72.0, "extent_radius_m": 2000.0},
        ]
        cid, dist = mod.find_overlapping_cluster(21.0001, 72.0, clusters)
        self.assertEqual(cid, 0)

    def test_exactly_at_extent_radius_boundary_matches(self):
        clusters = [{"cluster_id": 5, "centroid_lat": 21.0, "centroid_lon": 72.0, "extent_radius_m": 375.5}]
        cid, dist = mod.find_overlapping_cluster(21.0033687, 72.0, clusters)  # ~375m away
        self.assertEqual(cid, 5)


class TestWorldCoverTileNaming(unittest.TestCase):
    def test_tile_name_rounds_down_to_multiple_of_three(self):
        self.assertEqual(mod.tile_name_for(21.5, 72.9), "N21E072")
        self.assertEqual(mod.tile_name_for(23.99, 74.99), "N21E072")  # floor(23.99/3)*3=21, floor(74.99/3)*3=72

    def test_tile_name_at_exact_boundary(self):
        self.assertEqual(mod.tile_name_for(21.0, 72.0), "N21E072")
        self.assertEqual(mod.tile_name_for(20.999, 71.999), "N18E069")

    def test_required_tiles_deduplicates(self):
        centroids = [(21.1, 72.1), (21.2, 72.2), (22.5, 73.5)]
        tiles = mod.required_tiles(centroids)
        self.assertEqual(tiles, sorted(set(tiles)))
        self.assertIn("N21E072", tiles)
        self.assertIn("N21E072", mod.required_tiles([(21.1, 72.1), (21.9, 72.9)]))
        self.assertEqual(len(mod.required_tiles([(21.1, 72.1), (21.9, 72.9)])), 1)


class TestLoadPointsCsv(unittest.TestCase):
    def test_load_points_csv_reads_lat_lon(self):
        import csv
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "points.csv"
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["osm_type", "osm_id", "lat", "lon", "tags"])
                w.writeheader()
                w.writerow({"osm_type": "node", "osm_id": "1", "lat": "21.5", "lon": "72.5", "tags": "man_made=flare"})
            points = mod.load_points_csv(path)
            self.assertEqual(points, [(21.5, 72.5)])

    def test_load_gppd_points_reads_latitude_longitude_columns(self):
        import csv
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gppd.csv"
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["name", "gppd_idnr", "capacity_mw", "latitude", "longitude",
                                                   "primary_fuel", "commissioning_year"])
                w.writeheader()
                w.writerow({"name": "Test Plant", "gppd_idnr": "X1", "capacity_mw": "100",
                            "latitude": "22.0", "longitude": "73.0", "primary_fuel": "Coal",
                            "commissioning_year": "2005"})
            points = mod.load_gppd_points(path)
            self.assertEqual(points, [(22.0, 73.0)])


if __name__ == "__main__":
    unittest.main()
