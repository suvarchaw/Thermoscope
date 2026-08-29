import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_recurrence import (
    assign_cell,
    grid_steps_deg,
    group_detections,
    VIIRS_NOMINAL_PIXEL_M,
    METERS_PER_DEGREE_LAT,
)


def make_row(lat, lon, date, daynight="D", frp="1.0"):
    return {
        "latitude": str(lat),
        "longitude": str(lon),
        "acq_date": date,
        "daynight": daynight,
        "frp": frp,
    }


class TestGridSteps(unittest.TestCase):
    def test_lat_step_matches_pixel_size(self):
        lat_step, _ = grid_steps_deg(20.0, 24.7)
        expected = VIIRS_NOMINAL_PIXEL_M / METERS_PER_DEGREE_LAT
        self.assertAlmostEqual(lat_step, expected)

    def test_lon_step_wider_than_lat_step_away_from_equator(self):
        # At non-zero latitude, a degree of longitude covers less ground
        # distance than a degree of latitude, so the lon step in degrees
        # must be larger to represent the same physical pixel size.
        lat_step, lon_step = grid_steps_deg(20.0, 24.7)
        self.assertGreater(lon_step, lat_step)


class TestAssignCell(unittest.TestCase):
    def test_same_point_same_cell(self):
        lat_step, lon_step = grid_steps_deg(20.0, 24.7)
        c1 = assign_cell(21.0, 70.0, 20.0, 68.0, lat_step, lon_step)
        c2 = assign_cell(21.0, 70.0, 20.0, 68.0, lat_step, lon_step)
        self.assertEqual(c1, c2)

    def test_points_far_apart_different_cells(self):
        lat_step, lon_step = grid_steps_deg(20.0, 24.7)
        c1 = assign_cell(21.0, 70.0, 20.0, 68.0, lat_step, lon_step)
        c2 = assign_cell(23.5, 73.0, 20.0, 68.0, lat_step, lon_step)
        self.assertNotEqual(c1, c2)

    def test_points_within_one_pixel_share_cell(self):
        lat_step, lon_step = grid_steps_deg(20.0, 24.7)
        # Anchor at the center of an arbitrary cell, then nudge by a small
        # fraction of the step size in both directions — guaranteed to stay
        # inside the same cell regardless of where cell boundaries fall.
        base_lat = 20.0 + 100.5 * lat_step
        base_lon = 68.0 + 100.5 * lon_step
        c1 = assign_cell(base_lat - 0.1 * lat_step, base_lon - 0.1 * lon_step,
                          20.0, 68.0, lat_step, lon_step)
        c2 = assign_cell(base_lat + 0.1 * lat_step, base_lon + 0.1 * lon_step,
                          20.0, 68.0, lat_step, lon_step)
        self.assertEqual(c1, c2)


class TestGroupDetections(unittest.TestCase):
    def test_single_cell_stats(self):
        rows = [
            make_row(21.0, 70.0, "2023-01-01", "D", "1.0"),
            make_row(21.0, 70.0, "2023-01-05", "N", "3.0"),
            make_row(21.0, 70.0, "2023-01-05", "D", "2.0"),
        ]
        groups = group_detections(rows, lat_min=20.0, lon_min=68.0,
                                   lat_max=24.7, lon_max=74.5)
        self.assertEqual(len(groups), 1)
        g = groups[0]
        self.assertEqual(g["detection_count"], 3)
        self.assertEqual(g["unique_dates"], 2)
        self.assertEqual(g["first_date"], "2023-01-01")
        self.assertEqual(g["last_date"], "2023-01-05")
        self.assertEqual(g["active_span_days"], 4)
        self.assertAlmostEqual(g["mean_frp"], 2.0)
        self.assertEqual(g["max_frp"], 3.0)
        self.assertEqual(g["day_count"], 2)
        self.assertEqual(g["night_count"], 1)

    def test_distinct_far_apart_points_form_separate_groups(self):
        rows = [
            make_row(21.0, 70.0, "2023-01-01"),
            make_row(23.5, 73.0, "2023-01-01"),
        ]
        groups = group_detections(rows, lat_min=20.0, lon_min=68.0,
                                   lat_max=24.7, lon_max=74.5)
        self.assertEqual(len(groups), 2)

    def test_no_rows_returns_no_groups(self):
        groups = group_detections([], lat_min=20.0, lon_min=68.0,
                                   lat_max=24.7, lon_max=74.5)
        self.assertEqual(groups, [])


if __name__ == "__main__":
    unittest.main()
