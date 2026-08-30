import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cluster_integrated_evidence import (
    persistence_category,
    activity_category,
    seasonality_category,
    has_notable_osm_context,
    compute_evidence_notes,
    build_integrated_evidence,
)


def make_row(cluster_id=0, unique_years=5, total_detections_5yr=100,
             top3_months_share=0.3, recurrence_strength="Moderate",
             short_window_recurrence="False", features_found_in_radius="1",
             nearest_is_named="False", nearest_group="industrial",
             nearest_name=""):
    return {
        "cluster_id": str(cluster_id),
        "unique_years": str(unique_years),
        "total_detections_5yr": str(total_detections_5yr),
        "top3_months_share": str(top3_months_share),
        "recurrence_strength": recurrence_strength,
        "short_window_recurrence": short_window_recurrence,
        "features_found_in_radius": features_found_in_radius,
        "nearest_is_named": nearest_is_named,
        "nearest_group": nearest_group,
        "nearest_name": nearest_name,
    }


class TestPersistenceCategory(unittest.TestCase):
    def test_five_years_is_persistent(self):
        self.assertEqual(persistence_category(make_row(unique_years=5)), "Persistent")

    def test_fewer_than_five_is_intermittent(self):
        self.assertEqual(persistence_category(make_row(unique_years=4)), "Intermittent")
        self.assertEqual(persistence_category(make_row(unique_years=2)), "Intermittent")


class TestActivityCategory(unittest.TestCase):
    def test_at_or_above_median_is_high(self):
        self.assertEqual(activity_category(make_row(total_detections_5yr=100), median_total=100), "High-activity")
        self.assertEqual(activity_category(make_row(total_detections_5yr=150), median_total=100), "High-activity")

    def test_below_median_is_low(self):
        self.assertEqual(activity_category(make_row(total_detections_5yr=50), median_total=100), "Low-activity")


class TestSeasonalityCategory(unittest.TestCase):
    def test_at_or_above_p75_is_strongly_seasonal(self):
        row = make_row(top3_months_share=0.8)
        self.assertEqual(seasonality_category(row, p75_top3_months_share=0.7), "Strongly seasonal")

    def test_below_p75_is_not_strongly_seasonal(self):
        row = make_row(top3_months_share=0.5)
        self.assertEqual(seasonality_category(row, p75_top3_months_share=0.7), "Not strongly seasonal")


class TestHasNotableOsmContext(unittest.TestCase):
    def test_zero_features_is_false(self):
        self.assertFalse(has_notable_osm_context(make_row(features_found_in_radius="0")))

    def test_nonzero_features_is_true(self):
        self.assertTrue(has_notable_osm_context(make_row(features_found_in_radius="5")))


class TestComputeEvidenceNotes(unittest.TestCase):
    def test_moderate_but_persistent_flagged(self):
        row = make_row(recurrence_strength="Moderate", unique_years=5)
        notes = compute_evidence_notes(row, median_total=100)
        self.assertIn("understates multi-year persistence", notes)

    def test_strong_and_persistent_no_flag(self):
        row = make_row(recurrence_strength="Strong", unique_years=5, total_detections_5yr=200)
        notes = compute_evidence_notes(row, median_total=100)
        self.assertNotIn("understates multi-year persistence", notes)

    def test_strong_but_not_persistent_flagged(self):
        row = make_row(recurrence_strength="Strong", unique_years=3, total_detections_5yr=200)
        notes = compute_evidence_notes(row, median_total=100)
        self.assertIn("unusually active year", notes)

    def test_no_osm_context_but_persistent_flagged(self):
        row = make_row(unique_years=5, features_found_in_radius="0", recurrence_strength="Strong",
                        total_detections_5yr=200)
        notes = compute_evidence_notes(row, median_total=100)
        self.assertIn("no OSM context found nearby", notes)

    def test_named_industrial_low_activity_flagged(self):
        row = make_row(nearest_is_named="True", nearest_group="industrial",
                        total_detections_5yr=10, nearest_name="Test Factory",
                        recurrence_strength="Strong", unique_years=5)
        notes = compute_evidence_notes(row, median_total=100)
        self.assertIn("Test Factory", notes)
        self.assertIn("not evidence this feature is the thermal source", notes)

    def test_named_industrial_high_activity_not_flagged(self):
        row = make_row(nearest_is_named="True", nearest_group="industrial",
                        total_detections_5yr=500, recurrence_strength="Strong", unique_years=5)
        notes = compute_evidence_notes(row, median_total=100)
        self.assertNotIn("not evidence this feature is the thermal source", notes)

    def test_no_disagreement_produces_empty_string(self):
        row = make_row(recurrence_strength="Strong", unique_years=5, total_detections_5yr=200,
                        features_found_in_radius="3", nearest_is_named="False")
        notes = compute_evidence_notes(row, median_total=100)
        self.assertEqual(notes, "")


class TestBuildIntegratedEvidence(unittest.TestCase):
    def _make_full_row(self, cluster_id, **overrides):
        base = {
            "cluster_id": str(cluster_id), "unique_years": "5", "years_detected": "2019;2020;2021;2022;2023",
            "first_year": "2019", "last_year": "2023", "recurs_across_multiple_years": "True",
            "unique_dates": "20", "active_span_days": "300", "occurrence_rate": "0.1",
            "total_detections_5yr": "100", "mean_annual_detections": "20",
            "detections_2019": "20", "detections_2020": "20", "detections_2021": "20",
            "detections_2022": "20", "detections_2023": "20",
            "active_days_2019": "10", "active_days_2020": "10", "active_days_2021": "10",
            "active_days_2022": "10", "active_days_2023": "10",
            "detection_count": "20", "mean_frp": "1.0", "max_frp": "2.0", "frp_ratio": "2.0",
            "trend_slope": "0.0", "trend_direction": "stable",
            "std_annual_detections": "0.0", "cv_annual_detections": "0.0",
            "per_month_detection_counts": "1:100;2:0;3:0;4:0;5:0;6:0;7:0;8:0;9:0;10:0;11:0;12:0",
            "top3_months_share": "0.3", "dominant_month": "1",
            "day_count": "10", "night_count": "10", "night_fraction": "0.5",
            "top_day_share": "0.1", "top3_days_share": "0.2",
            "centroid_lat": "21.0", "centroid_lon": "72.0", "extent_radius_m": "300",
            "bbox_min_lat": "20.9", "bbox_max_lat": "21.1", "bbox_min_lon": "71.9", "bbox_max_lon": "72.1",
            "nearest_group": "industrial", "nearest_label": "Industrial land use", "nearest_name": "",
            "nearest_is_named": "False", "nearest_distance_m": "500", "nearest_geometry_class": "landuse_polygon",
            "n_industrial": "1", "n_power": "0", "n_waste": "0", "n_agricultural": "0", "n_transport": "0",
            "n_other": "0", "features_found_in_radius": "1", "osm_query_status": "ok",
            "top_features_summary": "landuse=industrial@500m",
            "recurrence_strength": "Moderate", "short_window_recurrence": "False", "burst_concentrated": "False",
        }
        base.update(overrides)
        return base

    def test_produces_60_unique_rows_from_60_inputs(self):
        rows = [self._make_full_row(i) for i in range(60)]
        integrated, thresholds = build_integrated_evidence(rows)
        self.assertEqual(len(integrated), 60)
        self.assertEqual(len({r["cluster_id"] for r in integrated}), 60)

    def test_no_composite_score_column_present(self):
        rows = [self._make_full_row(i) for i in range(5)]
        integrated, _ = build_integrated_evidence(rows)
        forbidden = {"risk_score", "score", "composite_score", "industrial", "source_type"}
        for row in integrated:
            self.assertTrue(forbidden.isdisjoint(row.keys()))


if __name__ == "__main__":
    unittest.main()
