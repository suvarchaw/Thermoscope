import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_feature_inventory import (
    INCLUDED_FEATURES,
    EXCLUDED_FEATURES,
    pearson,
    check_included_features_uncorrelated,
    impute_nearest_distance,
    build_feature_matrix,
    load_evidence,
    CORRELATION_THRESHOLD,
)


class TestFeatureListsAreDisjointAndComplete(unittest.TestCase):
    def test_no_overlap_between_included_and_excluded(self):
        overlap = set(INCLUDED_FEATURES) & set(EXCLUDED_FEATURES.keys())
        self.assertEqual(overlap, set())

    def test_cluster_id_is_excluded_not_included(self):
        self.assertNotIn("cluster_id", INCLUDED_FEATURES)
        self.assertIn("cluster_id", EXCLUDED_FEATURES)

    def test_hand_designed_categories_are_excluded(self):
        for col in ["recurrence_strength", "persistence_category", "activity_category",
                    "seasonality_category", "trend_direction", "short_window_recurrence",
                    "burst_concentrated"]:
            self.assertIn(col, EXCLUDED_FEATURES)
            self.assertNotIn(col, INCLUDED_FEATURES)

    def test_all_real_columns_accounted_for(self):
        rows = load_evidence()
        real_columns = set(rows[0].keys())
        accounted = set(INCLUDED_FEATURES) | set(EXCLUDED_FEATURES.keys())
        self.assertEqual(real_columns - accounted, set())

    def test_no_documentation_entries_for_nonexistent_columns(self):
        rows = load_evidence()
        real_columns = set(rows[0].keys())
        accounted = set(INCLUDED_FEATURES) | set(EXCLUDED_FEATURES.keys())
        self.assertEqual(accounted - real_columns, set())


class TestPearson(unittest.TestCase):
    def test_perfect_correlation(self):
        rows = [{"a": str(i), "b": str(2 * i)} for i in range(1, 6)]
        self.assertAlmostEqual(pearson(rows, "a", "b"), 1.0)

    def test_no_correlation_constant_column(self):
        rows = [{"a": str(i), "b": "5"} for i in range(1, 6)]
        r = pearson(rows, "a", "b")
        self.assertTrue(r != r or r == 0)  # nan or 0 for zero-variance input


class TestCheckIncludedFeaturesUncorrelated(unittest.TestCase):
    def test_flags_pair_above_threshold(self):
        matrix = [{"cluster_id": i, "x": float(i), "y": float(i) * 2 + 0.01 * (i % 2)}
                  for i in range(1, 20)]
        flagged = check_included_features_uncorrelated(matrix, features=["x", "y"], threshold=0.9)
        self.assertEqual(len(flagged), 1)

    def test_does_not_flag_uncorrelated_pair(self):
        import random
        random.seed(42)
        matrix = [{"x": float(i), "y": random.random()} for i in range(1, 30)]
        flagged = check_included_features_uncorrelated(matrix, features=["x", "y"], threshold=0.9)
        self.assertEqual(len(flagged), 0)


class TestImputeNearestDistance(unittest.TestCase):
    def test_uses_actual_distance_when_present(self):
        row = {"nearest_distance_m": "250.5", "extent_radius_m": "300"}
        self.assertAlmostEqual(impute_nearest_distance(row), 250.5)

    def test_imputes_search_radius_when_missing(self):
        row = {"nearest_distance_m": "", "extent_radius_m": "300"}
        # compute_search_radius(300) = max(750, min(3000, 300+500)) = 800
        self.assertAlmostEqual(impute_nearest_distance(row), 800.0)

    def test_imputed_value_is_deterministic(self):
        row = {"nearest_distance_m": "", "extent_radius_m": "169"}
        v1 = impute_nearest_distance(row)
        v2 = impute_nearest_distance(row)
        self.assertEqual(v1, v2)


class TestBuildFeatureMatrix(unittest.TestCase):
    def _make_row(self, cluster_id, **overrides):
        base = {"cluster_id": str(cluster_id), "extent_radius_m": "300", "nearest_distance_m": "250"}
        for col in INCLUDED_FEATURES:
            if col not in base:
                base[col] = "True" if col == "nearest_is_named" else "1.0"
        base.update(overrides)
        return base

    def test_nearest_is_named_converted_to_binary(self):
        rows = [self._make_row(0, nearest_is_named="True"),
                self._make_row(1, nearest_is_named="False")]
        matrix = build_feature_matrix(rows)
        self.assertEqual(matrix[0]["nearest_is_named"], 1)
        self.assertEqual(matrix[1]["nearest_is_named"], 0)

    def test_all_feature_values_are_numeric(self):
        rows = [self._make_row(i) for i in range(3)]
        matrix = build_feature_matrix(rows)
        for record in matrix:
            for col in INCLUDED_FEATURES:
                self.assertIsInstance(record[col], (int, float))

    def test_output_sorted_by_cluster_id(self):
        rows = [self._make_row(5), self._make_row(1), self._make_row(3)]
        matrix = build_feature_matrix(rows)
        self.assertEqual([r["cluster_id"] for r in matrix], [1, 3, 5])


if __name__ == "__main__":
    unittest.main()
