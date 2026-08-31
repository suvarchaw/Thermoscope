import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import spatial_evaluation as mod


def make_row(lat, lon, label, event_id="EVT000000"):
    return {"event_id": event_id, "centroid_lat": str(lat), "centroid_lon": str(lon), "silver_label": label}


class TestCellOf(unittest.TestCase):
    def test_same_point_same_cell(self):
        self.assertEqual(mod.cell_of(21.3, 72.1), mod.cell_of(21.3, 72.1))

    def test_points_half_degree_apart_different_cells(self):
        c1 = mod.cell_of(21.0, 72.0)
        c2 = mod.cell_of(21.6, 72.0)
        self.assertNotEqual(c1, c2)

    def test_points_within_same_half_degree_cell(self):
        c1 = mod.cell_of(21.01, 72.01)
        c2 = mod.cell_of(21.4, 72.4)
        self.assertEqual(c1, c2)


class TestAssignSpatialSplit(unittest.TestCase):
    def test_no_cell_split_across_train_and_test(self):
        rows = [make_row(21.0, 72.0, "Industrial", "A"),
                make_row(21.05, 72.05, "Gas_Flare", "B"),  # same cell as A
                make_row(23.0, 70.0, "Crop_Residue", "C")]  # likely different cell
        train_rows, test_rows, cell_report = mod.assign_spatial_split(rows)
        train_ids = {r["event_id"] for r in train_rows}
        test_ids = {r["event_id"] for r in test_rows}
        # A and B share a cell -> must be on the same side
        both_a_b_train = {"A", "B"}.issubset(train_ids)
        both_a_b_test = {"A", "B"}.issubset(test_ids)
        self.assertTrue(both_a_b_train or both_a_b_test)
        self.assertEqual(train_ids & test_ids, set())

    def test_split_is_exhaustive(self):
        rows = [make_row(20.1 + 0.3 * i, 68.1 + 0.3 * i, "Industrial", str(i)) for i in range(10)]
        train_rows, test_rows, _ = mod.assign_spatial_split(rows)
        self.assertEqual(len(train_rows) + len(test_rows), len(rows))

    def test_assignment_independent_of_label(self):
        # identical coordinates, different labels -> must land on the same side
        rows_a = [make_row(21.3, 72.3, "Industrial", "A")]
        rows_b = [make_row(21.3, 72.3, "Forest_Wildfire", "A")]
        train_a, test_a, _ = mod.assign_spatial_split(rows_a)
        train_b, test_b, _ = mod.assign_spatial_split(rows_b)
        side_a = "train" if train_a else "test"
        side_b = "train" if train_b else "test"
        self.assertEqual(side_a, side_b)

    def test_deterministic_repeated_calls(self):
        rows = [make_row(20.1 + 0.1 * i, 68.1 + 0.1 * i, "Crop_Residue", str(i)) for i in range(20)]
        train1, test1, _ = mod.assign_spatial_split(list(rows))
        train2, test2, _ = mod.assign_spatial_split(list(rows))
        self.assertEqual([r["event_id"] for r in train1], [r["event_id"] for r in train2])
        self.assertEqual([r["event_id"] for r in test1], [r["event_id"] for r in test2])


class TestNearestSameClassTrainDistance(unittest.TestCase):
    def test_finds_nearest_same_class(self):
        train = [make_row(21.0, 72.0, "Industrial", "T1"), make_row(25.0, 76.0, "Industrial", "T2")]
        test = [make_row(21.001, 72.0, "Industrial", "X1")]
        distances = mod.nearest_same_class_train_distance(test, train)
        self.assertIsNotNone(distances["X1"])
        self.assertLess(distances["X1"], 200)  # close to T1, not T2

    def test_ignores_different_class_train_examples(self):
        train = [make_row(21.0, 72.0, "Gas_Flare", "T1")]
        test = [make_row(21.0, 72.0, "Industrial", "X1")]  # same location, different class
        distances = mod.nearest_same_class_train_distance(test, train)
        self.assertIsNone(distances["X1"])  # no same-class training example exists

    def test_none_when_class_absent_from_train(self):
        train = [make_row(21.0, 72.0, "Crop_Residue", "T1")]
        test = [make_row(22.0, 73.0, "Forest_Wildfire", "X1")]
        distances = mod.nearest_same_class_train_distance(test, train)
        self.assertIsNone(distances["X1"])


class TestSummarizeDiagnostic(unittest.TestCase):
    def test_within_300m_count(self):
        test_rows = [make_row(21.0, 72.0, "Industrial", "X1"), make_row(21.0, 72.0, "Industrial", "X2")]
        distances = {"X1": 100.0, "X2": 5000.0}
        summary = mod.summarize_diagnostic(test_rows, distances)
        self.assertEqual(summary["Industrial"]["within_300m"], 1)
        self.assertEqual(summary["Industrial"]["n_with_same_class_train_example"], 2)

    def test_none_distances_excluded_from_stats(self):
        test_rows = [make_row(21.0, 72.0, "Gas_Flare", "X1"), make_row(21.0, 72.0, "Gas_Flare", "X2")]
        distances = {"X1": None, "X2": 50.0}
        summary = mod.summarize_diagnostic(test_rows, distances)
        self.assertEqual(summary["Gas_Flare"]["n_with_same_class_train_example"], 1)
        self.assertEqual(summary["Gas_Flare"]["min_m"], 50.0)


class TestNoLeakageIntoModel(unittest.TestCase):
    def test_diagnostic_distances_not_in_feature_lists(self):
        from train_source_classifier_lightgbm import FEATURES_NUMERIC, FEATURES_CATEGORICAL
        all_features = set(FEATURES_NUMERIC) | set(FEATURES_CATEGORICAL)
        self.assertNotIn("nearest_same_class_train_distance", all_features)
        self.assertNotIn("distance_to_nearest_train", all_features)


if __name__ == "__main__":
    unittest.main()
