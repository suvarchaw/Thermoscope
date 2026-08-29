import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nighttime_dominance_investigation import acq_time_to_hours, daynight_breakdown


class TestAcqTimeToHours(unittest.TestCase):
    def test_zero_padded_time(self):
        self.assertAlmostEqual(acq_time_to_hours("0745"), 7.75)

    def test_unpadded_short_time(self):
        # '745' means 07:45, same as '0745'
        self.assertAlmostEqual(acq_time_to_hours("745"), 7.75)

    def test_midnight(self):
        self.assertAlmostEqual(acq_time_to_hours("0000"), 0.0)

    def test_near_end_of_day(self):
        self.assertAlmostEqual(acq_time_to_hours("2130"), 21.5)


class TestDaynightBreakdown(unittest.TestCase):
    def test_counts_correctly(self):
        rows = [
            {"daynight": "D"}, {"daynight": "D"}, {"daynight": "N"},
        ]
        d, n, total = daynight_breakdown(rows, "test")
        self.assertEqual(d, 2)
        self.assertEqual(n, 1)
        self.assertEqual(total, 3)

    def test_all_night(self):
        rows = [{"daynight": "N"}, {"daynight": "N"}]
        d, n, total = daynight_breakdown(rows, "test")
        self.assertEqual(d, 0)
        self.assertEqual(n, 2)
        self.assertEqual(total, 2)


if __name__ == "__main__":
    unittest.main()
