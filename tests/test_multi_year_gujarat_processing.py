import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_year_gujarat_processing import (
    _in_gujarat_bounds,
    _year_from_acq_date,
    discover_historical_raw_files,
    load_historical_year,
    write_combined_csv,
    FIELDS_TO_KEEP,
)


def make_row(lat, lon, acq_date):
    return {
        "latitude": str(lat), "longitude": str(lon), "brightness": "330.0",
        "scan": "0.4", "track": "0.4", "acq_date": acq_date, "acq_time": "0745",
        "satellite": "N", "confidence": "n", "bright_t31": "300.0", "frp": "2.5",
        "daynight": "D",
    }


class TestInGujaratBounds(unittest.TestCase):
    def test_inside_bounds(self):
        self.assertTrue(_in_gujarat_bounds(make_row(21.0, 72.0, "2020-01-01")))

    def test_outside_bounds(self):
        self.assertFalse(_in_gujarat_bounds(make_row(30.0, 80.0, "2020-01-01")))

    def test_missing_coordinates_is_false(self):
        self.assertFalse(_in_gujarat_bounds({"acq_date": "2020-01-01"}))

    def test_boundary_values_included(self):
        self.assertTrue(_in_gujarat_bounds(make_row(20.0, 68.0, "2020-01-01")))
        self.assertTrue(_in_gujarat_bounds(make_row(24.7, 74.5, "2020-01-01")))


class TestYearFromAcqDate(unittest.TestCase):
    def test_extracts_year(self):
        self.assertEqual(_year_from_acq_date("2021-06-15"), 2021)

    def test_different_year(self):
        self.assertEqual(_year_from_acq_date("2019-12-31"), 2019)


class TestDiscoverHistoricalRawFiles(unittest.TestCase):
    def test_finds_matching_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "firms_gujarat_2020.csv").touch()
            (Path(tmpdir) / "firms_gujarat_2021.csv").touch()
            (Path(tmpdir) / "unrelated.csv").touch()
            found = discover_historical_raw_files(pattern=f"{tmpdir}/firms_gujarat_*.csv")
            self.assertEqual(set(found.keys()), {2020, 2021})

    def test_empty_when_no_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            found = discover_historical_raw_files(pattern=f"{tmpdir}/firms_gujarat_*.csv")
            self.assertEqual(found, {})


class TestLoadHistoricalYear(unittest.TestCase):
    def test_filters_out_of_bounds_and_tags_year(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "firms_gujarat_2020.csv"
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(make_row(0, 0, "2020-01-01").keys()))
                writer.writeheader()
                writer.writerow(make_row(21.0, 72.0, "2020-03-15"))   # in bounds
                writer.writerow(make_row(30.0, 80.0, "2020-03-15"))   # out of bounds

            rows = load_historical_year(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["year"], 2020)
            self.assertEqual(rows[0]["latitude"], "21.0")


class TestWriteCombinedCsv(unittest.TestCase):
    def test_writes_only_expected_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "combined.csv"
            row = make_row(21.0, 72.0, "2023-01-01")
            row["year"] = 2023
            row["instrument"] = "SNPP"  # extra field should be dropped, not error
            write_combined_csv([row], path=out_path)

            with open(out_path, newline="") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, FIELDS_TO_KEEP)
                written_rows = list(reader)
            self.assertEqual(len(written_rows), 1)
            self.assertEqual(written_rows[0]["year"], "2023")


if __name__ == "__main__":
    unittest.main()
