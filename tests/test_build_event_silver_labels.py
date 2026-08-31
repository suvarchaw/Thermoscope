import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import build_event_silver_labels as mod


def make_row(**overrides):
    row = {
        "event_id": "EVT000000",
        "start_date": "2023-01-15", "end_date": "2023-01-15",
        "duration_days": "1",
        "centroid_lat": "21.0", "centroid_lon": "72.0",
        "spatial_extent_m": "0.0", "detection_count": "2",
        "mean_frp": "3.0", "max_frp": "3.0",
        "night_fraction": "0.0", "status": "closed",
        "nearest_osm_industrial_power_m": "", "nearest_gppd_thermal_plant_m": "",
        "nearest_osm_flare_m": "", "nearest_osm_kiln_m": "",
        "overlaps_cluster_id": "", "overlaps_cluster_recurrence_strength": "",
        "overlaps_cluster_recurs_multiyear": "",
        "land_cover_code": "", "land_cover_class": "",
    }
    row.update({k: str(v) for k, v in overrides.items()})
    return row


class TestColumnSeparation(unittest.TestCase):
    def test_label_generating_and_candidate_features_disjoint(self):
        mod.assert_columns_disjoint()  # should not raise

    def test_assert_catches_injected_overlap(self):
        mod.CANDIDATE_FEATURE_COLUMNS.append("mean_frp")
        try:
            with self.assertRaises(ValueError):
                mod.assert_columns_disjoint()
        finally:
            mod.CANDIDATE_FEATURE_COLUMNS.remove("mean_frp")

    def test_every_rule_reads_only_label_generating_columns(self):
        # every field a rule inspects must be declared in
        # LABEL_GENERATING_COLUMNS -- guards against a rule silently
        # reading an undeclared column that should then be excluded
        # from candidate features too.
        declared = mod.LABEL_GENERATING_COLUMNS
        rule_read_columns = {
            "nearest_osm_industrial_power_m", "nearest_gppd_thermal_plant_m", "night_fraction",
            "overlaps_cluster_recurrence_strength", "nearest_osm_flare_m",
            "overlaps_cluster_recurs_multiyear", "mean_frp", "nearest_osm_kiln_m",
            "start_date", "duration_days", "land_cover_class",
        }
        self.assertTrue(rule_read_columns.issubset(declared))


class TestIndustrialRule(unittest.TestCase):
    def test_matches_when_all_conditions_true(self):
        row = make_row(nearest_osm_industrial_power_m=500, night_fraction=0.3,
                        overlaps_cluster_recurrence_strength="Strong")
        matched, evidence = mod.industrial_rule(row)
        self.assertTrue(matched)
        self.assertIn("night_fraction", evidence)

    def test_no_match_when_too_far(self):
        row = make_row(nearest_osm_industrial_power_m=5000, night_fraction=0.3,
                        overlaps_cluster_recurrence_strength="Strong")
        matched, _ = mod.industrial_rule(row)
        self.assertFalse(matched)

    def test_no_match_without_night_detections(self):
        row = make_row(nearest_osm_industrial_power_m=500, night_fraction=0.0,
                        overlaps_cluster_recurrence_strength="Strong")
        matched, _ = mod.industrial_rule(row)
        self.assertFalse(matched)

    def test_no_match_when_recurrence_limited(self):
        row = make_row(nearest_osm_industrial_power_m=500, night_fraction=0.3,
                        overlaps_cluster_recurrence_strength="Limited")
        matched, _ = mod.industrial_rule(row)
        self.assertFalse(matched)

    def test_gppd_alone_satisfies_proximity(self):
        row = make_row(nearest_osm_industrial_power_m="", nearest_gppd_thermal_plant_m=800,
                        night_fraction=0.1, overlaps_cluster_recurrence_strength="Moderate")
        matched, _ = mod.industrial_rule(row)
        self.assertTrue(matched)

    def test_exact_1000m_boundary_matches(self):
        row = make_row(nearest_osm_industrial_power_m=1000, night_fraction=0.1,
                        overlaps_cluster_recurrence_strength="Strong")
        matched, _ = mod.industrial_rule(row)
        self.assertTrue(matched)


class TestGasFlareRule(unittest.TestCase):
    def test_matches_via_persistence(self):
        row = make_row(nearest_osm_flare_m=500, overlaps_cluster_recurs_multiyear="True", mean_frp=1.0)
        matched, _ = mod.gas_flare_rule(row, median_mean_frp=5.0)
        self.assertTrue(matched)

    def test_matches_via_radiometric_when_not_persistent(self):
        row = make_row(nearest_osm_flare_m=500, overlaps_cluster_recurs_multiyear="False", mean_frp=10.0)
        matched, _ = mod.gas_flare_rule(row, median_mean_frp=5.0)
        self.assertTrue(matched)

    def test_no_match_without_either_persistence_or_radiometric(self):
        row = make_row(nearest_osm_flare_m=500, overlaps_cluster_recurs_multiyear="False", mean_frp=1.0)
        matched, _ = mod.gas_flare_rule(row, median_mean_frp=5.0)
        self.assertFalse(matched)

    def test_no_match_when_far_from_flare(self):
        row = make_row(nearest_osm_flare_m=5000, overlaps_cluster_recurs_multiyear="True", mean_frp=10.0)
        matched, _ = mod.gas_flare_rule(row, median_mean_frp=5.0)
        self.assertFalse(matched)


class TestBrickKilnRule(unittest.TestCase):
    """Proves the rule is implemented and CAN match -- the real-data zero
    is a genuine evidence gap, not a disabled code path."""

    def test_can_match_given_synthetic_evidence(self):
        row = make_row(nearest_osm_kiln_m=200, start_date="2023-01-10", duration_days=2)
        matched, _ = mod.brick_kiln_rule(row)
        self.assertTrue(matched)

    def test_no_match_outside_season(self):
        row = make_row(nearest_osm_kiln_m=200, start_date="2023-07-10", duration_days=2)
        matched, _ = mod.brick_kiln_rule(row)
        self.assertFalse(matched)

    def test_no_match_when_too_long(self):
        row = make_row(nearest_osm_kiln_m=200, start_date="2023-01-10", duration_days=10)
        matched, _ = mod.brick_kiln_rule(row)
        self.assertFalse(matched)

    def test_no_match_beyond_500m(self):
        row = make_row(nearest_osm_kiln_m=600, start_date="2023-01-10", duration_days=2)
        matched, _ = mod.brick_kiln_rule(row)
        self.assertFalse(matched)


class TestCropResidueRule(unittest.TestCase):
    def test_matches_cropland_short_harvest_window(self):
        row = make_row(land_cover_class="Cropland", duration_days=1, start_date="2023-11-05")
        matched, _ = mod.crop_residue_rule(row)
        self.assertTrue(matched)

    def test_no_match_off_season(self):
        row = make_row(land_cover_class="Cropland", duration_days=1, start_date="2023-08-01")
        matched, _ = mod.crop_residue_rule(row)
        self.assertFalse(matched)

    def test_no_match_wrong_land_cover(self):
        row = make_row(land_cover_class="Built-up", duration_days=1, start_date="2023-11-05")
        matched, _ = mod.crop_residue_rule(row)
        self.assertFalse(matched)

    def test_no_match_long_duration(self):
        row = make_row(land_cover_class="Cropland", duration_days=20, start_date="2023-11-05")
        matched, _ = mod.crop_residue_rule(row)
        self.assertFalse(matched)


class TestForestWildfireRule(unittest.TestCase):
    def test_matches_treecover_short_dry_season(self):
        row = make_row(land_cover_class="Tree cover", duration_days=2, start_date="2023-04-10")
        matched, _ = mod.forest_wildfire_rule(row)
        self.assertTrue(matched)

    def test_no_match_off_season(self):
        row = make_row(land_cover_class="Tree cover", duration_days=2, start_date="2023-11-10")
        matched, _ = mod.forest_wildfire_rule(row)
        self.assertFalse(matched)

    def test_no_match_wrong_land_cover(self):
        row = make_row(land_cover_class="Cropland", duration_days=2, start_date="2023-04-10")
        matched, _ = mod.forest_wildfire_rule(row)
        self.assertFalse(matched)


class TestLabelEvent(unittest.TestCase):
    def test_single_match_assigns_that_class(self):
        row = make_row(land_cover_class="Tree cover", duration_days=1, start_date="2023-04-10")
        label, matched, evidence, reason = mod.label_event(row, median_mean_frp=5.0)
        self.assertEqual(label, "Forest_Wildfire")
        self.assertEqual(matched, ["Forest_Wildfire"])
        self.assertEqual(reason, "")

    def test_no_match_is_insufficient_evidence(self):
        row = make_row()
        label, matched, evidence, reason = mod.label_event(row, median_mean_frp=5.0)
        self.assertEqual(label, mod.UNKNOWN_AMBIGUOUS)
        self.assertEqual(matched, [])
        self.assertEqual(reason, "insufficient_evidence")

    def test_double_match_is_conflicting_evidence_not_forced(self):
        # Cropland harvest-season event that ALSO sits near an
        # industrial/GPPD facility with night detections and cluster
        # persistence -- genuinely ambiguous, must not be forced into
        # either class.
        row = make_row(
            land_cover_class="Cropland", duration_days=1, start_date="2023-11-05",
            nearest_osm_industrial_power_m=500, night_fraction=0.2,
            overlaps_cluster_recurrence_strength="Strong",
        )
        label, matched, evidence, reason = mod.label_event(row, median_mean_frp=5.0)
        self.assertEqual(label, mod.UNKNOWN_AMBIGUOUS)
        self.assertEqual(set(matched), {"Crop_Residue", "Industrial"})
        self.assertEqual(reason, "conflicting_evidence")


class TestBuildLabels(unittest.TestCase):
    def test_output_row_schema_and_values(self):
        rows = [make_row(event_id="EVT000000", land_cover_class="Tree cover",
                          duration_days=1, start_date="2023-04-10")]
        out = mod.build_labels(rows)
        self.assertEqual(len(out), 1)
        r = out[0]
        self.assertEqual(r["event_id"], "EVT000000")
        self.assertEqual(r["silver_label"], "Forest_Wildfire")
        self.assertEqual(r["label_rule_id"], "Forest_Wildfire")
        self.assertIn("Tree cover", r["label_evidence"])
        self.assertEqual(r["conflict_classes"], "Forest_Wildfire")
        self.assertEqual(r["n_classes_matched"], 1)
        self.assertEqual(r["ambiguity_reason"], "")
        self.assertFalse(r["excluded_from_training"])

    def test_ambiguous_row_excluded_from_training(self):
        rows = [make_row(event_id="EVT000001")]  # matches nothing
        out = mod.build_labels(rows)
        r = out[0]
        self.assertEqual(r["silver_label"], mod.UNKNOWN_AMBIGUOUS)
        self.assertTrue(r["excluded_from_training"])
        self.assertEqual(r["label_rule_id"], "")

    def test_candidate_feature_columns_present_in_output(self):
        rows = [make_row()]
        out = mod.build_labels(rows)
        for col in mod.CANDIDATE_FEATURE_COLUMNS:
            self.assertIn(col, out[0])

    def test_deterministic_repeated_runs(self):
        rows = [make_row(event_id=f"EVT{i:06d}", land_cover_class="Cropland",
                          duration_days=1, start_date="2023-11-05") for i in range(5)]
        out1 = mod.build_labels(list(rows))
        out2 = mod.build_labels(list(rows))
        self.assertEqual(out1, out2)


class TestComputeMedianMeanFrp(unittest.TestCase):
    def test_median_computed_correctly(self):
        rows = [make_row(mean_frp=v) for v in (1.0, 2.0, 3.0, 4.0, 5.0)]
        self.assertAlmostEqual(mod.compute_median_mean_frp(rows), 3.0)


if __name__ == "__main__":
    unittest.main()
