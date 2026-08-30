import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recurrence_profile import (
    recurrence_strength,
    short_window_recurrence,
    burst_concentrated,
    compute_recurrence_profile,
)


class TestRecurrenceStrengthBoundaries(unittest.TestCase):
    def test_72_is_limited(self):
        self.assertEqual(recurrence_strength(72), "Limited")

    def test_73_is_moderate(self):
        self.assertEqual(recurrence_strength(73), "Moderate")

    def test_142_is_moderate(self):
        self.assertEqual(recurrence_strength(142), "Moderate")

    def test_143_is_strong(self):
        self.assertEqual(recurrence_strength(143), "Strong")

    def test_far_below_and_above(self):
        self.assertEqual(recurrence_strength(0), "Limited")
        self.assertEqual(recurrence_strength(2), "Limited")
        self.assertEqual(recurrence_strength(300), "Strong")


class TestShortWindowRecurrenceBoundaries(unittest.TestCase):
    def test_80_is_true(self):
        self.assertTrue(short_window_recurrence(80))

    def test_81_is_false(self):
        self.assertFalse(short_window_recurrence(81))

    def test_zero_is_true(self):
        self.assertTrue(short_window_recurrence(0))

    def test_364_is_false(self):
        self.assertFalse(short_window_recurrence(364))


class TestBurstConcentratedBoundaries(unittest.TestCase):
    def test_exactly_half_is_false(self):
        self.assertFalse(burst_concentrated(0.5))

    def test_just_above_half_is_true(self):
        self.assertTrue(burst_concentrated(0.5001))

    def test_just_below_half_is_false(self):
        self.assertFalse(burst_concentrated(0.4999))

    def test_zero_and_one(self):
        self.assertFalse(burst_concentrated(0.0))
        self.assertTrue(burst_concentrated(1.0))


class TestComputeRecurrenceProfile(unittest.TestCase):
    def test_combines_all_three_fields(self):
        profile = compute_recurrence_profile(unique_dates=200, active_span_days=50, top3_days_share=0.6)
        self.assertEqual(profile, {
            "recurrence_strength": "Strong",
            "short_window_recurrence": True,
            "burst_concentrated": True,
        })

    def test_cluster_0_like_case(self):
        # cluster 0: unique_dates=295, active_span=364, top3_days_share small
        profile = compute_recurrence_profile(unique_dates=295, active_span_days=364, top3_days_share=0.05)
        self.assertEqual(profile["recurrence_strength"], "Strong")
        self.assertFalse(profile["short_window_recurrence"])
        self.assertFalse(profile["burst_concentrated"])

    def test_cluster_42_like_case(self):
        # cluster 42: unique_dates=17, active_span=31, top3_days_share=0.28
        profile = compute_recurrence_profile(unique_dates=17, active_span_days=31, top3_days_share=0.28)
        self.assertEqual(profile["recurrence_strength"], "Limited")
        self.assertTrue(profile["short_window_recurrence"])
        self.assertFalse(profile["burst_concentrated"])

    def test_short_window_does_not_override_strength_tier(self):
        # A short-window cluster with enough unique_dates to be Strong should
        # still be classified Strong -- the flag is independent, not an override.
        profile = compute_recurrence_profile(unique_dates=150, active_span_days=60, top3_days_share=0.1)
        self.assertEqual(profile["recurrence_strength"], "Strong")
        self.assertTrue(profile["short_window_recurrence"])

    def test_burst_concentrated_does_not_create_a_tier(self):
        # burst_concentrated=True alongside any tier should never change
        # what recurrence_strength value is produced.
        profile_limited = compute_recurrence_profile(unique_dates=10, active_span_days=200, top3_days_share=0.9)
        profile_strong = compute_recurrence_profile(unique_dates=200, active_span_days=200, top3_days_share=0.9)
        self.assertEqual(profile_limited["recurrence_strength"], "Limited")
        self.assertEqual(profile_strong["recurrence_strength"], "Strong")
        self.assertTrue(profile_limited["burst_concentrated"])
        self.assertTrue(profile_strong["burst_concentrated"])


class TestOsmFieldsDoNotAffectRecurrenceCalculation(unittest.TestCase):
    """Structural + behavioral proof that OSM context cannot influence a
    Recurrence Profile: the functions only ever accept the three specific
    numeric values they need, and identical (unique_dates, active_span_days,
    top3_days_share) inputs always produce identical profiles regardless of
    what OSM/other context exists alongside them in a real evidence row."""

    def test_function_signature_has_no_osm_parameters(self):
        import inspect
        params = list(inspect.signature(compute_recurrence_profile).parameters)
        self.assertEqual(params, ["unique_dates", "active_span_days", "top3_days_share"])

    def test_identical_temporal_inputs_give_identical_profile_regardless_of_osm_context(self):
        # Simulate two evidence rows that are identical in every temporal
        # metric but wildly different in OSM context.
        row_with_named_industrial = {
            "unique_dates": 50, "active_span_days": 100, "top3_days_share": 0.2,
            "nearest_group": "industrial", "nearest_name": "ArcelorMittal Nippon Steel India",
            "nearest_distance_m": 5.0, "features_found_in_radius": 30,
        }
        row_with_no_osm_context = {
            "unique_dates": 50, "active_span_days": 100, "top3_days_share": 0.2,
            "nearest_group": "", "nearest_name": "", "nearest_distance_m": "",
            "features_found_in_radius": 0,
        }
        profile_a = compute_recurrence_profile(
            row_with_named_industrial["unique_dates"],
            row_with_named_industrial["active_span_days"],
            row_with_named_industrial["top3_days_share"],
        )
        profile_b = compute_recurrence_profile(
            row_with_no_osm_context["unique_dates"],
            row_with_no_osm_context["active_span_days"],
            row_with_no_osm_context["top3_days_share"],
        )
        self.assertEqual(profile_a, profile_b)


if __name__ == "__main__":
    unittest.main()
