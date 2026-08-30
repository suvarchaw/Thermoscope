import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_recurrence_profiles import build_profiles, NEW_FIELDS


def make_evidence_row(cluster_id, unique_dates, active_span_days, top3_days_share,
                       nearest_group="industrial", nearest_name="Some Facility"):
    return {
        "cluster_id": str(cluster_id),
        "unique_dates": str(unique_dates),
        "active_span_days": str(active_span_days),
        "top3_days_share": str(top3_days_share),
        "detection_count": "100",
        "occurrence_rate": "0.2",
        "mean_frp": "1.5",
        "max_frp": "3.0",
        "night_fraction": "0.9",
        "nearest_group": nearest_group,
        "nearest_name": nearest_name,
        "nearest_distance_m": "250.0",
        "features_found_in_radius": "5",
    }


class TestBuildProfiles(unittest.TestCase):
    def test_all_existing_columns_preserved(self):
        rows = [make_evidence_row(0, 200, 300, 0.1)]
        profiled = build_profiles(rows)
        for key in rows[0]:
            self.assertEqual(profiled[0][key], rows[0][key])

    def test_new_fields_added(self):
        rows = [make_evidence_row(0, 200, 300, 0.1)]
        profiled = build_profiles(rows)
        for field in NEW_FIELDS:
            self.assertIn(field, profiled[0])

    def test_correct_values_for_known_case(self):
        rows = [make_evidence_row(0, 295, 364, 0.05)]
        profiled = build_profiles(rows)
        self.assertEqual(profiled[0]["recurrence_strength"], "Strong")
        self.assertFalse(profiled[0]["short_window_recurrence"])
        self.assertFalse(profiled[0]["burst_concentrated"])

    def test_osm_field_changes_do_not_change_recurrence_fields(self):
        row_a = make_evidence_row(0, 50, 100, 0.2, nearest_group="industrial", nearest_name="Big Factory")
        row_b = make_evidence_row(1, 50, 100, 0.2, nearest_group="", nearest_name="")
        profiled = build_profiles([row_a, row_b])
        for field in NEW_FIELDS:
            self.assertEqual(profiled[0][field], profiled[1][field])

    def test_output_row_count_matches_input(self):
        rows = [make_evidence_row(i, i + 1, 100, 0.1) for i in range(60)]
        profiled = build_profiles(rows)
        self.assertEqual(len(profiled), 60)


if __name__ == "__main__":
    unittest.main()
