import csv
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firms_ingestion import (
    FirmsConfigError,
    fetch_source_range,
    fetch_2026_gujarat,
    SOURCE_HISTORICAL,
    SOURCE_NRT,
    SP_END_DATE_2026,
    NRT_START_DATE_2026,
)


def _sample_csv_text(acq_date):
    header = ("latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,"
               "satellite,confidence,version,bright_ti5,frp,daynight\n")
    return header + f"21.1,72.6,330.0,0.4,0.4,{acq_date},0745,N,n,2.0NRT,300.0,2.5,D\n"


class TestReusesExistingDedup(unittest.TestCase):
    def test_fetch_2026_gujarat_reuses_deduplicate_rows(self):
        """Cross-source duplicate/overlap protection must reuse the same
        deduplicate_rows already relied on for every other year, not a
        reimplementation."""
        import inspect
        import firms_ingestion as fi
        source = inspect.getsource(fi.fetch_2026_gujarat)
        self.assertIn("deduplicate_rows(", source)


class TestFetchSourceRange(unittest.TestCase):
    def test_tags_rows_with_source_and_normalizes(self):
        with patch("firms_ingestion.fetch_csv_chunk", return_value=_sample_csv_text("2026-02-01")), \
             patch("firms_ingestion.time.sleep"):
            rows = fetch_source_range(SOURCE_HISTORICAL, date(2026, 2, 1), date(2026, 2, 1), map_key="FAKE")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["viirs_source"], SOURCE_HISTORICAL)
        self.assertIn("brightness", rows[0])
        self.assertNotIn("bright_ti4", rows[0])


class TestFetch2026Gujarat(unittest.TestCase):
    def test_rejects_non_contiguous_boundary(self):
        with self.assertRaises(ValueError):
            fetch_2026_gujarat(sp_end_date=date(2026, 4, 27), nrt_start_date=date(2026, 4, 29),
                                map_key="FAKE")

    def test_rejects_run_date_before_nrt_start(self):
        with self.assertRaises(ValueError):
            fetch_2026_gujarat(run_date=date(2026, 4, 1), map_key="FAKE")

    def test_missing_map_key_raises_config_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaises(FirmsConfigError):
                    fetch_2026_gujarat(output_dir=Path(tmpdir), map_key=None,
                                        run_date=date(2026, 5, 1))

    def test_skips_when_file_already_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            existing = output_dir / "firms_gujarat_2026.csv"
            with open(existing, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["latitude", "longitude", "viirs_source"])
                writer.writeheader()
                writer.writerow({"latitude": "21.0", "longitude": "72.0", "viirs_source": SOURCE_HISTORICAL})
                writer.writerow({"latitude": "21.1", "longitude": "72.1", "viirs_source": SOURCE_NRT})

            with patch("firms_ingestion.fetch_csv_chunk") as mock_fetch:
                result = fetch_2026_gujarat(output_dir=output_dir, map_key="unused",
                                             run_date=date(2026, 5, 1))
                mock_fetch.assert_not_called()
            self.assertEqual(result["status"], "skipped_exists")
            self.assertEqual(result["row_count"], 2)
            self.assertEqual(result["sp_row_count"], 1)
            self.assertEqual(result["nrt_row_count"], 1)

    def test_writes_combined_sp_and_nrt_rows_no_gap_no_overlap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            def fake_fetch(url):
                # SOURCE_HISTORICAL requests hit the /VIIRS_SNPP_SP/ path segment,
                # NRT requests hit /VIIRS_SNPP_NRT/ -- return distinct dates so
                # the two sources are trivially distinguishable in the output.
                if f"/{SOURCE_HISTORICAL}/" in url:
                    return _sample_csv_text("2026-04-10")
                return _sample_csv_text("2026-05-10")

            with patch("firms_ingestion.fetch_csv_chunk", side_effect=fake_fetch), \
                 patch("firms_ingestion.time.sleep"):
                result = fetch_2026_gujarat(output_dir=output_dir, map_key="FAKEKEY",
                                             run_date=date(2026, 5, 15))

            self.assertEqual(result["status"], "ok")
            self.assertGreater(result["sp_row_count"], 0)
            self.assertGreater(result["nrt_row_count"], 0)
            self.assertEqual(result["row_count"], result["sp_row_count"] + result["nrt_row_count"])

            with open(result["path"], newline="") as f:
                rows = list(csv.DictReader(f))
            sources_by_date = {r["acq_date"]: r["viirs_source"] for r in rows}
            self.assertEqual(sources_by_date["2026-04-10"], SOURCE_HISTORICAL)
            self.assertEqual(sources_by_date["2026-05-10"], SOURCE_NRT)
            # every row's date falls on the correct side of the SP/NRT boundary
            for r in rows:
                d = date.fromisoformat(r["acq_date"])
                if r["viirs_source"] == SOURCE_HISTORICAL:
                    self.assertLessEqual(d, SP_END_DATE_2026)
                else:
                    self.assertGreaterEqual(d, NRT_START_DATE_2026)

    def test_no_data_returns_no_data_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with patch("firms_ingestion.fetch_csv_chunk", return_value="latitude,longitude\n"), \
                 patch("firms_ingestion.time.sleep"):
                result = fetch_2026_gujarat(output_dir=output_dir, map_key="FAKEKEY",
                                             run_date=date(2026, 5, 1))
            self.assertEqual(result["status"], "no_data")
            self.assertFalse((output_dir / "firms_gujarat_2026.csv").exists())


if __name__ == "__main__":
    unittest.main()
