import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_cluster_evidence_review import (
    build_evidence_rows,
    check_overlap_consistency,
    EVIDENCE_COLUMNS,
)


def make_cluster(cid, detection_count=10, unique_dates=5, active_span_days=9,
                  night_count=8, mean_frp=2.0, max_frp=6.0,
                  centroid_lat=21.0, centroid_lon=72.0, extent_radius_m=500.0):
    return {
        "cluster_id": cid,
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "bbox_min_lat": centroid_lat - 0.01,
        "bbox_max_lat": centroid_lat + 0.01,
        "bbox_min_lon": centroid_lon - 0.01,
        "bbox_max_lon": centroid_lon + 0.01,
        "extent_radius_m": extent_radius_m,
        "detection_count": detection_count,
        "unique_dates": unique_dates,
        "first_date": "2023-01-01",
        "last_date": "2023-01-10",
        "active_span_days": active_span_days,
        "mean_frp": mean_frp,
        "max_frp": max_frp,
        "day_count": detection_count - night_count,
        "night_count": night_count,
    }


def make_osm_row(cid, centroid_lat=21.0, centroid_lon=72.0, detection_count=10,
                  unique_dates=5, extent_radius_m=500.0):
    row = {f: "" for f in [
        "search_radius_m", "osm_query_status", "features_found_in_radius",
        "nearest_tag", "nearest_label", "nearest_group", "nearest_name",
        "nearest_is_named", "nearest_geometry_class", "nearest_distance_m",
        "nearest_lat", "nearest_lon", "nearest_named_tag", "nearest_named_label",
        "nearest_named_name", "nearest_named_distance_m",
        "n_industrial", "n_power", "n_waste", "n_agricultural", "n_transport",
        "n_other", "top_features_summary",
    ]}
    row.update({
        "cluster_id": str(cid),
        "centroid_lat": str(centroid_lat),
        "centroid_lon": str(centroid_lon),
        "detection_count": str(detection_count),
        "unique_dates": str(unique_dates),
        "extent_radius_m": str(extent_radius_m),
        "osm_query_status": "ok",
        "features_found_in_radius": "0",
    })
    return row


class TestCheckOverlapConsistency(unittest.TestCase):
    def test_matching_fields_pass(self):
        cluster = make_cluster(0)
        osm_row = make_osm_row(0)
        check_overlap_consistency(cluster, osm_row)  # should not raise

    def test_mismatched_field_raises(self):
        cluster = make_cluster(0, centroid_lat=21.0)
        osm_row = make_osm_row(0, centroid_lat=25.0)
        with self.assertRaises(ValueError):
            check_overlap_consistency(cluster, osm_row)


class TestBuildEvidenceRows(unittest.TestCase):
    def _build(self, n=60):
        clusters = [make_cluster(i) for i in range(n)]
        date_counts = {i: {"2023-01-01": 6, "2023-01-10": 4} for i in range(n)}
        osm_by_id = {i: make_osm_row(i) for i in range(n)}
        return clusters, date_counts, osm_by_id

    def test_produces_exactly_60_rows_for_60_clusters(self):
        clusters, date_counts, osm_by_id = self._build(60)
        rows = build_evidence_rows(clusters, date_counts, osm_by_id)
        self.assertEqual(len(rows), 60)
        self.assertEqual(len({r["cluster_id"] for r in rows}), 60)

    def test_rejects_wrong_cluster_count(self):
        clusters, date_counts, osm_by_id = self._build(59)
        with self.assertRaises(ValueError):
            build_evidence_rows(clusters, date_counts, osm_by_id)

    def test_rejects_duplicate_cluster_id(self):
        clusters = [make_cluster(0) for _ in range(60)]  # all id=0, duplicated
        date_counts = {0: {"2023-01-01": 1}}
        osm_by_id = {0: make_osm_row(0)}
        with self.assertRaises(ValueError):
            build_evidence_rows(clusters, date_counts, osm_by_id)

    def test_rejects_missing_osm_row(self):
        clusters, date_counts, osm_by_id = self._build(60)
        del osm_by_id[5]
        with self.assertRaises(ValueError):
            build_evidence_rows(clusters, date_counts, osm_by_id)

    def test_rejects_extra_unmatched_osm_row(self):
        clusters, date_counts, osm_by_id = self._build(60)
        osm_by_id[999] = make_osm_row(999)
        with self.assertRaises(ValueError):
            build_evidence_rows(clusters, date_counts, osm_by_id)

    def test_no_classification_columns_present(self):
        clusters, date_counts, osm_by_id = self._build(60)
        rows = build_evidence_rows(clusters, date_counts, osm_by_id)
        forbidden = {"industrial", "source_type", "predicted_class", "risk_score", "persistent"}
        for row in rows:
            self.assertTrue(forbidden.isdisjoint(row.keys()))
        self.assertTrue(forbidden.isdisjoint(EVIDENCE_COLUMNS))

    def test_missing_osm_match_stays_blank_not_falsy_default(self):
        clusters, date_counts, osm_by_id = self._build(60)
        rows = build_evidence_rows(clusters, date_counts, osm_by_id)
        # features_found_in_radius=0 in fixture -> nearest_* fields stay ""
        self.assertEqual(rows[0]["nearest_tag"], "")
        self.assertEqual(rows[0]["nearest_name"], "")


if __name__ == "__main__":
    unittest.main()
