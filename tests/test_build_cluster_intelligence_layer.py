import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_cluster_intelligence_layer import (
    corroboration_2023_recurrence,
    corroboration_5yr_persistence,
    corroboration_osm_context,
    compute_corroboration_fields,
    build_intelligence_layer,
)


def make_evidence_row(cluster_id=0, recurrence_strength="Limited", unique_years="5",
                       has_notable_osm_context="False", **extra):
    row = {
        "cluster_id": str(cluster_id),
        "recurrence_strength": recurrence_strength,
        "unique_years": unique_years,
        "has_notable_osm_context": has_notable_osm_context,
        "centroid_lat": "21.0",
    }
    row.update(extra)
    return row


class TestCorroboration2023Recurrence(unittest.TestCase):
    def test_strong_is_true(self):
        self.assertTrue(corroboration_2023_recurrence(make_evidence_row(recurrence_strength="Strong")))

    def test_moderate_is_true(self):
        self.assertTrue(corroboration_2023_recurrence(make_evidence_row(recurrence_strength="Moderate")))

    def test_limited_is_false(self):
        self.assertFalse(corroboration_2023_recurrence(make_evidence_row(recurrence_strength="Limited")))


class TestCorroboration5yrPersistence(unittest.TestCase):
    def test_five_is_true(self):
        self.assertTrue(corroboration_5yr_persistence(make_evidence_row(unique_years="5")))

    def test_four_is_false(self):
        self.assertFalse(corroboration_5yr_persistence(make_evidence_row(unique_years="4")))

    def test_two_is_false(self):
        self.assertFalse(corroboration_5yr_persistence(make_evidence_row(unique_years="2")))


class TestCorroborationOsmContext(unittest.TestCase):
    def test_string_true_is_true(self):
        self.assertTrue(corroboration_osm_context(make_evidence_row(has_notable_osm_context="True")))

    def test_string_false_is_false(self):
        self.assertFalse(corroboration_osm_context(make_evidence_row(has_notable_osm_context="False")))

    def test_python_bool_true_is_true(self):
        self.assertTrue(corroboration_osm_context(make_evidence_row(has_notable_osm_context=True)))


class TestComputeCorroborationFields(unittest.TestCase):
    def test_all_three_true_gives_count_3(self):
        row = make_evidence_row(recurrence_strength="Strong", unique_years="5",
                                 has_notable_osm_context="True")
        fields = compute_corroboration_fields(row)
        self.assertTrue(fields["corroboration_2023_recurrence"])
        self.assertTrue(fields["corroboration_5yr_persistence"])
        self.assertTrue(fields["corroboration_osm_context"])
        self.assertEqual(fields["evidence_corroboration_count"], 3)

    def test_all_false_gives_count_0(self):
        row = make_evidence_row(recurrence_strength="Limited", unique_years="2",
                                 has_notable_osm_context="False")
        fields = compute_corroboration_fields(row)
        self.assertEqual(fields["evidence_corroboration_count"], 0)

    def test_partial_gives_correct_count(self):
        row = make_evidence_row(recurrence_strength="Moderate", unique_years="3",
                                 has_notable_osm_context="True")
        fields = compute_corroboration_fields(row)
        self.assertEqual(fields["evidence_corroboration_count"], 2)

    def test_count_is_plain_int_not_bool(self):
        row = make_evidence_row(recurrence_strength="Strong", unique_years="5",
                                 has_notable_osm_context="True")
        fields = compute_corroboration_fields(row)
        self.assertIsInstance(fields["evidence_corroboration_count"], int)
        self.assertNotIsInstance(fields["evidence_corroboration_count"], bool)


class TestBuildIntelligenceLayer(unittest.TestCase):
    def test_produces_60_rows_with_all_base_columns_and_new_columns(self):
        evidence_rows = [make_evidence_row(i, extra_field=f"val{i}") for i in range(60)]
        groups = {i: "Group A" if i % 2 == 0 else "Group B" for i in range(60)}
        output_rows, fieldnames = build_intelligence_layer(evidence_rows, groups)

        self.assertEqual(len(output_rows), 60)
        self.assertIn("extra_field", fieldnames)  # original column preserved
        self.assertIn("unsupervised_group", fieldnames)
        self.assertIn("evidence_corroboration_count", fieldnames)
        self.assertIn("corroboration_osm_context", fieldnames)

    def test_original_evidence_values_unchanged(self):
        evidence_rows = [make_evidence_row(0, centroid_lat="21.5")]
        groups = {0: "Group A"}
        output_rows, _ = build_intelligence_layer(evidence_rows, groups)
        self.assertEqual(output_rows[0]["centroid_lat"], "21.5")

    def test_unsupervised_group_correctly_joined(self):
        evidence_rows = [make_evidence_row(5), make_evidence_row(7)]
        groups = {5: "Group A", 7: "Group B"}
        output_rows, _ = build_intelligence_layer(evidence_rows, groups)
        by_id = {int(r["cluster_id"]): r for r in output_rows}
        self.assertEqual(by_id[5]["unsupervised_group"], "Group A")
        self.assertEqual(by_id[7]["unsupervised_group"], "Group B")

    def test_missing_unsupervised_group_raises(self):
        evidence_rows = [make_evidence_row(99)]
        groups = {}  # cluster 99 missing
        with self.assertRaises(ValueError):
            build_intelligence_layer(evidence_rows, groups)

    def test_no_pca_columns_present(self):
        evidence_rows = [make_evidence_row(0)]
        groups = {0: "Group A"}
        output_rows, fieldnames = build_intelligence_layer(evidence_rows, groups)
        self.assertNotIn("pca_1", fieldnames)
        self.assertNotIn("pca_2", fieldnames)

    def test_no_risk_or_priority_named_columns(self):
        evidence_rows = [make_evidence_row(0)]
        groups = {0: "Group A"}
        _, fieldnames = build_intelligence_layer(evidence_rows, groups)
        forbidden_substrings = ["risk", "priority", "probability", "severity"]
        for field in fieldnames:
            for bad in forbidden_substrings:
                self.assertNotIn(bad, field.lower())

    def test_output_sorted_by_cluster_id(self):
        evidence_rows = [make_evidence_row(5), make_evidence_row(1), make_evidence_row(3)]
        groups = {1: "Group A", 3: "Group A", 5: "Group B"}
        output_rows, _ = build_intelligence_layer(evidence_rows, groups)
        self.assertEqual([r["cluster_id"] for r in output_rows], ["1", "3", "5"])


if __name__ == "__main__":
    unittest.main()
