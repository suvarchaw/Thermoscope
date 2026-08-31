import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unsupervised_cluster_exploration import (
    choose_k,
    relabel_by_size_desc,
    compare_to_categories,
    build_group_profiles,
    fit_kmeans,
    variant_column_indices,
    evaluate_k_range,
)


class TestChooseK(unittest.TestCase):
    def test_picks_clear_maximum(self):
        results = {2: {"silhouette": 0.5}, 3: {"silhouette": 0.2}, 4: {"silhouette": 0.1}}
        self.assertEqual(choose_k(results), 2)

    def test_tie_broken_toward_smaller_k(self):
        results = {2: {"silhouette": 0.30}, 3: {"silhouette": 0.305}, 4: {"silhouette": 0.1}}
        # within 0.01 of each other -> smaller k wins
        self.assertEqual(choose_k(results), 2)

    def test_not_a_tie_when_gap_exceeds_threshold(self):
        results = {2: {"silhouette": 0.10}, 3: {"silhouette": 0.30}, 4: {"silhouette": 0.05}}
        self.assertEqual(choose_k(results), 3)


class TestRelabelBySizeDesc(unittest.TestCase):
    def test_largest_group_becomes_group_a(self):
        labels = [0, 0, 0, 1, 1]
        names = relabel_by_size_desc(labels)
        self.assertEqual(names, ["Group A", "Group A", "Group A", "Group B", "Group B"])

    def test_deterministic_across_calls(self):
        labels = [2, 0, 1, 2, 0, 2]
        names1 = relabel_by_size_desc(labels, tiebreak_key=[10, 20, 30, 40, 50, 60])
        names2 = relabel_by_size_desc(labels, tiebreak_key=[10, 20, 30, 40, 50, 60])
        self.assertEqual(names1, names2)

    def test_tiebreak_used_for_equal_sizes(self):
        # two groups of size 1 each; tiebreak_key decides ordering
        labels = [0, 1]
        names = relabel_by_size_desc(labels, tiebreak_key=[100, 1])
        # label 1 has smaller tiebreak mean (1) -> should come first (Group A)
        self.assertEqual(names[1], "Group A")
        self.assertEqual(names[0], "Group B")


class TestVariantColumnIndices(unittest.TestCase):
    def test_excludes_named_columns(self):
        names = ["a", "b", "c", "d"]
        idx = variant_column_indices(names, exclude=["b", "d"])
        self.assertEqual(idx, [0, 2])

    def test_empty_exclude_returns_all(self):
        names = ["a", "b", "c"]
        idx = variant_column_indices(names, exclude=[])
        self.assertEqual(idx, [0, 1, 2])


class TestFitKmeansDeterminism(unittest.TestCase):
    def test_same_random_state_gives_identical_labels(self):
        rng = np.random.RandomState(0)
        X = np.vstack([rng.normal(0, 0.5, (10, 3)), rng.normal(5, 0.5, (10, 3))])
        labels1, _ = fit_kmeans(X, k=2, random_state=42)
        labels2, _ = fit_kmeans(X, k=2, random_state=42)
        np.testing.assert_array_equal(labels1, labels2)


class TestEvaluateKRangeDeterminism(unittest.TestCase):
    def test_repeated_calls_give_identical_silhouettes(self):
        rng = np.random.RandomState(1)
        X = np.vstack([rng.normal(0, 0.5, (15, 4)), rng.normal(6, 0.5, (15, 4))])
        r1 = evaluate_k_range(X, k_range=range(2, 4))
        r2 = evaluate_k_range(X, k_range=range(2, 4))
        for k in r1:
            self.assertAlmostEqual(r1[k]["silhouette"], r2[k]["silhouette"])


class TestCompareToCategories(unittest.TestCase):
    def test_identical_grouping_gives_ari_one(self):
        import unsupervised_cluster_exploration as uce
        old = uce.EXISTING_CATEGORY_COLUMNS
        uce.EXISTING_CATEGORY_COLUMNS = ["cat"]
        try:
            group_labels = ["Group A", "Group A", "Group B", "Group B"]
            cluster_ids = [0, 1, 2, 3]
            categories = {0: {"cat": "X"}, 1: {"cat": "X"}, 2: {"cat": "Y"}, 3: {"cat": "Y"}}
            result = compare_to_categories(group_labels, cluster_ids, categories)
            self.assertAlmostEqual(result["cat"]["ari"], 1.0)
        finally:
            uce.EXISTING_CATEGORY_COLUMNS = old

    def test_unrelated_grouping_gives_low_ari(self):
        import unsupervised_cluster_exploration as uce
        old = uce.EXISTING_CATEGORY_COLUMNS
        uce.EXISTING_CATEGORY_COLUMNS = ["cat"]
        try:
            group_labels = ["Group A", "Group B", "Group A", "Group B"]
            cluster_ids = [0, 1, 2, 3]
            categories = {0: {"cat": "X"}, 1: {"cat": "X"}, 2: {"cat": "Y"}, 3: {"cat": "Y"}}
            result = compare_to_categories(group_labels, cluster_ids, categories)
            self.assertLess(result["cat"]["ari"], 0.5)
        finally:
            uce.EXISTING_CATEGORY_COLUMNS = old


class TestBuildGroupProfiles(unittest.TestCase):
    def test_correct_means_and_sizes(self):
        import unsupervised_cluster_exploration as uce
        old = uce.EXISTING_CATEGORY_COLUMNS
        uce.EXISTING_CATEGORY_COLUMNS = ["cat"]
        try:
            group_labels = ["Group A", "Group A", "Group B"]
            cluster_ids = [10, 11, 12]
            feature_names = ["x"]
            X = np.array([[1.0], [3.0], [100.0]])
            categories = {10: {"cat": "P"}, 11: {"cat": "P"}, 12: {"cat": "Q"}}
            profiles = build_group_profiles(group_labels, cluster_ids, feature_names, X, categories)

            by_group = {p["group"]: p for p in profiles}
            self.assertEqual(by_group["Group A"]["n_clusters"], 2)
            self.assertAlmostEqual(by_group["Group A"]["mean_x"], 2.0)
            self.assertEqual(by_group["Group B"]["n_clusters"], 1)
            self.assertAlmostEqual(by_group["Group B"]["mean_x"], 100.0)
            self.assertIn("P:2", by_group["Group A"]["breakdown_cat"])
        finally:
            uce.EXISTING_CATEGORY_COLUMNS = old

    def test_profiles_sorted_by_size_descending(self):
        import unsupervised_cluster_exploration as uce
        old = uce.EXISTING_CATEGORY_COLUMNS
        uce.EXISTING_CATEGORY_COLUMNS = []
        try:
            group_labels = ["Group A", "Group B", "Group B", "Group B"]
            cluster_ids = [0, 1, 2, 3]
            X = np.array([[1.0], [1.0], [1.0], [1.0]])
            profiles = build_group_profiles(group_labels, cluster_ids, ["x"], X, {i: {} for i in cluster_ids})
            self.assertEqual(profiles[0]["group"], "Group B")
            self.assertEqual(profiles[0]["n_clusters"], 3)
        finally:
            uce.EXISTING_CATEGORY_COLUMNS = old


if __name__ == "__main__":
    unittest.main()
