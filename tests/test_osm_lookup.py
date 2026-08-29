import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osm_lookup import (
    haversine_m,
    compute_search_radius,
    classify_tag,
    classify_geometry,
    nearby_features,
)


class TestHaversine(unittest.TestCase):
    def test_zero_distance(self):
        self.assertAlmostEqual(haversine_m(21.0, 72.0, 21.0, 72.0), 0.0)


class TestComputeSearchRadius(unittest.TestCase):
    def test_normal_extent_gets_buffer(self):
        self.assertEqual(compute_search_radius(1000, buffer_m=500, floor_m=750, cap_m=3000), 1500)

    def test_tiny_extent_hits_floor(self):
        self.assertEqual(compute_search_radius(50, buffer_m=500, floor_m=750, cap_m=3000), 750)

    def test_huge_extent_hits_cap(self):
        self.assertEqual(compute_search_radius(5000, buffer_m=500, floor_m=750, cap_m=3000), 3000)


class TestClassifyTag(unittest.TestCase):
    def test_matches_known_tag(self):
        tag, label, group = classify_tag({"landuse": "industrial", "name": "Foo"})
        self.assertEqual(tag, "landuse=industrial")
        self.assertEqual(label, "Industrial land use")
        self.assertEqual(group, "industrial")

    def test_unmatched_tag_returns_unknown(self):
        tag, label, group = classify_tag({"shop": "bakery"})
        self.assertEqual(tag, "unknown")
        self.assertEqual(label, "Unknown")
        self.assertEqual(group, "other")

    def test_matches_agricultural_tag(self):
        tag, label, group = classify_tag({"landuse": "farmland"})
        self.assertEqual(group, "agricultural")


class TestClassifyGeometry(unittest.TestCase):
    def test_node_is_point_facility(self):
        self.assertEqual(classify_geometry({"type": "node"}, "power"), "point_facility")

    def test_way_with_landuse_key_is_polygon(self):
        self.assertEqual(classify_geometry({"type": "way"}, "landuse"), "landuse_polygon")

    def test_way_with_non_landuse_key_is_mapped_feature(self):
        self.assertEqual(classify_geometry({"type": "way"}, "power"), "mapped_feature")


class TestNearbyFeatures(unittest.TestCase):
    def test_sorts_by_distance_and_respects_top_n(self):
        center_lat, center_lon = 21.0, 72.0
        elements = [
            {"type": "node", "lat": 21.01, "lon": 72.0, "tags": {"landuse": "industrial", "name": "Far"}},
            {"type": "node", "lat": 21.001, "lon": 72.0, "tags": {"power": "plant", "name": "Near"}},
            {"type": "way", "center": {"lat": 21.005, "lon": 72.0}, "tags": {"landuse": "farmland"}},
        ]
        features = nearby_features(center_lat, center_lon, elements, top_n=2)
        self.assertEqual(len(features), 2)
        self.assertEqual(features[0]["name"], "Near")
        self.assertEqual(features[1]["geometry_class"], "landuse_polygon")
        self.assertTrue(features[0]["distance_m"] < features[1]["distance_m"])

    def test_unnamed_feature_has_empty_name_and_is_named_false(self):
        elements = [{"type": "node", "lat": 21.001, "lon": 72.0, "tags": {"man_made": "silo"}}]
        features = nearby_features(21.0, 72.0, elements)
        self.assertEqual(features[0]["name"], "")
        self.assertFalse(features[0]["is_named"])

    def test_element_without_coordinates_is_skipped(self):
        elements = [{"type": "way", "tags": {"landuse": "industrial"}}]  # no 'center'
        features = nearby_features(21.0, 72.0, elements)
        self.assertEqual(features, [])

    def test_empty_elements_returns_empty(self):
        self.assertEqual(nearby_features(21.0, 72.0, []), [])


if __name__ == "__main__":
    unittest.main()
