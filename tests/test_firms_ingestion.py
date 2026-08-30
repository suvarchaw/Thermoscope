import csv
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firms_ingestion import (
    FirmsConfigError,
    FirmsIngestionError,
    get_map_key,
    generate_date_chunks,
    build_area_url,
    fetch_csv_chunk,
    parse_csv_text,
    normalize_columns,
    validate_columns,
    deduplicate_rows,
    fetch_year_gujarat,
    GUJARAT_BBOX,
    SOURCE_HISTORICAL,
)


class TestGetMapKey(unittest.TestCase):
    def test_raises_when_unset(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(FirmsConfigError) as ctx:
                get_map_key()
            self.assertIn("FIRMS_MAP_KEY", str(ctx.exception))
            self.assertIn("map_key", str(ctx.exception))  # setup URL mentioned

    def test_returns_value_when_set(self):
        with patch.dict("os.environ", {"FIRMS_MAP_KEY": "abc123"}):
            self.assertEqual(get_map_key(), "abc123")

    def test_raises_when_empty_string(self):
        with patch.dict("os.environ", {"FIRMS_MAP_KEY": ""}):
            with self.assertRaises(FirmsConfigError):
                get_map_key()


class TestGenerateDateChunks(unittest.TestCase):
    def test_exact_multiple_of_chunk_size(self):
        chunks = generate_date_chunks(date(2023, 1, 1), date(2023, 1, 10), chunk_days=5)
        self.assertEqual(chunks, [(date(2023, 1, 1), 5), (date(2023, 1, 6), 5)])

    def test_remainder_produces_shorter_final_chunk(self):
        chunks = generate_date_chunks(date(2023, 1, 1), date(2023, 1, 7), chunk_days=5)
        self.assertEqual(chunks, [(date(2023, 1, 1), 5), (date(2023, 1, 6), 2)])

    def test_single_day_range(self):
        chunks = generate_date_chunks(date(2023, 1, 1), date(2023, 1, 1), chunk_days=5)
        self.assertEqual(chunks, [(date(2023, 1, 1), 1)])

    def test_full_year_has_no_gaps_or_overlaps(self):
        start, end = date(2023, 1, 1), date(2023, 12, 31)
        chunks = generate_date_chunks(start, end, chunk_days=5)
        covered_days = 0
        cursor = start
        for chunk_start, day_range in chunks:
            self.assertEqual(chunk_start, cursor)
            covered_days += day_range
            cursor = chunk_start.__class__.fromordinal(chunk_start.toordinal() + day_range)
        self.assertEqual(covered_days, (end - start).days + 1)
        self.assertEqual(cursor, end.__class__.fromordinal(end.toordinal() + 1))

    def test_rejects_inverted_range(self):
        with self.assertRaises(ValueError):
            generate_date_chunks(date(2023, 12, 31), date(2023, 1, 1))


class TestBuildAreaUrl(unittest.TestCase):
    def test_correct_url_format_and_coordinate_order(self):
        url = build_area_url("KEY123", SOURCE_HISTORICAL, GUJARAT_BBOX, 5, date(2023, 1, 1))
        self.assertIn("/KEY123/", url)
        self.assertIn(f"/{SOURCE_HISTORICAL}/", url)
        # GUJARAT_BBOX is (west, south, east, north) = (68.0, 20.0, 74.5, 24.7)
        self.assertIn("68.0,20.0,74.5,24.7", url)
        self.assertIn("/5/", url)
        self.assertTrue(url.endswith("2023-01-01"))


class TestParseAndValidateColumns(unittest.TestCase):
    def test_parse_csv_text_basic(self):
        text = "latitude,longitude,acq_date\n21.0,72.0,2023-01-01\n"
        rows, fieldnames = parse_csv_text(text)
        self.assertEqual(len(rows), 1)
        self.assertEqual(fieldnames, ["latitude", "longitude", "acq_date"])

    def test_normalize_renames_brightness_columns(self):
        rows = [{"bright_ti4": "330.0", "bright_ti5": "300.0", "latitude": "21.0"}]
        normalized = normalize_columns(rows)
        self.assertEqual(normalized[0]["brightness"], "330.0")
        self.assertEqual(normalized[0]["bright_t31"], "300.0")
        self.assertEqual(normalized[0]["latitude"], "21.0")
        self.assertNotIn("bright_ti4", normalized[0])

    def test_validate_columns_passes_with_all_required(self):
        fields = ["latitude", "longitude", "acq_date", "acq_time",
                  "satellite", "confidence", "frp", "daynight", "bright_ti4", "bright_ti5"]
        validate_columns(fields)  # should not raise

    def test_validate_columns_raises_on_missing(self):
        fields = ["latitude", "longitude", "acq_date"]
        with self.assertRaises(FirmsIngestionError) as ctx:
            validate_columns(fields)
        self.assertIn("acq_time", str(ctx.exception))

    def test_validate_columns_recognizes_renamed_names(self):
        # required list uses post-normalization names; raw API columns
        # use bright_ti4/ti5, which validate_columns must still accept
        # since it normalizes before checking.
        fields = ["latitude", "longitude", "acq_date", "acq_time",
                  "satellite", "confidence", "frp", "daynight"]
        validate_columns(fields)


class TestDeduplicateRows(unittest.TestCase):
    def test_removes_exact_duplicates(self):
        rows = [
            {"latitude": "21.0", "longitude": "72.0", "acq_date": "2023-01-01",
             "acq_time": "0745", "satellite": "N"},
            {"latitude": "21.0", "longitude": "72.0", "acq_date": "2023-01-01",
             "acq_time": "0745", "satellite": "N"},
        ]
        deduped = deduplicate_rows(rows)
        self.assertEqual(len(deduped), 1)

    def test_keeps_distinct_rows(self):
        rows = [
            {"latitude": "21.0", "longitude": "72.0", "acq_date": "2023-01-01",
             "acq_time": "0745", "satellite": "N"},
            {"latitude": "21.0", "longitude": "72.0", "acq_date": "2023-01-02",
             "acq_time": "0745", "satellite": "N"},
        ]
        deduped = deduplicate_rows(rows)
        self.assertEqual(len(deduped), 2)


class TestFetchCsvChunk(unittest.TestCase):
    @patch("firms_ingestion.requests.get")
    def test_success_returns_text(self, mock_get):
        mock_resp = MagicMock(status_code=200, text="latitude,longitude\n21.0,72.0\n")
        mock_get.return_value = mock_resp
        result = fetch_csv_chunk("http://example.com")
        self.assertIn("21.0,72.0", result)

    @patch("firms_ingestion.requests.get")
    def test_non_200_status_raises(self, mock_get):
        mock_resp = MagicMock(status_code=403, text="Forbidden")
        mock_get.return_value = mock_resp
        with self.assertRaises(FirmsIngestionError):
            fetch_csv_chunk("http://example.com")

    @patch("firms_ingestion.requests.get")
    def test_invalid_key_message_raises(self, mock_get):
        mock_resp = MagicMock(status_code=200, text="Invalid MAP_KEY")
        mock_get.return_value = mock_resp
        with self.assertRaises(FirmsIngestionError):
            fetch_csv_chunk("http://example.com")

    @patch("firms_ingestion.requests.get")
    def test_network_error_raises(self, mock_get):
        import requests
        mock_get.side_effect = requests.ConnectionError("boom")
        with self.assertRaises(FirmsIngestionError):
            fetch_csv_chunk("http://example.com")


class TestFetchYearGujarat(unittest.TestCase):
    def _sample_csv_text(self, dates):
        header = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,confidence,version,bright_ti5,frp,daynight\n"
        lines = [header]
        for d in dates:
            lines.append(f"21.1,72.6,330.0,0.4,0.4,{d},0745,N,n,2.0NRT,300.0,2.5,D\n")
        return "".join(lines)

    def test_skips_when_file_already_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            existing = output_dir / "firms_gujarat_2020.csv"
            with open(existing, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["latitude", "longitude"])
                writer.writeheader()
                writer.writerow({"latitude": "21.0", "longitude": "72.0"})

            with patch("firms_ingestion.fetch_csv_chunk") as mock_fetch:
                result = fetch_year_gujarat(2020, output_dir=output_dir, map_key="unused")
                mock_fetch.assert_not_called()
            self.assertEqual(result["status"], "skipped_exists")
            self.assertEqual(result["row_count"], 1)

    def test_missing_map_key_raises_config_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaises(FirmsConfigError):
                    fetch_year_gujarat(2021, output_dir=Path(tmpdir), map_key=None)

    def test_successful_fetch_writes_normalized_deduplicated_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            def fake_fetch(url):
                # Return the same single detection date for every chunk to
                # also exercise de-duplication across "chunks".
                return self._sample_csv_text(["2020-01-01"])

            with patch("firms_ingestion.fetch_csv_chunk", side_effect=fake_fetch), \
                 patch("firms_ingestion.time.sleep"):
                result = fetch_year_gujarat(2020, output_dir=output_dir, map_key="FAKEKEY")

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["row_count"], 1)  # deduped across all chunks
            out_path = Path(result["path"])
            self.assertTrue(out_path.exists())
            with open(out_path, newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertIn("brightness", rows[0])
            self.assertIn("bright_t31", rows[0])
            self.assertNotIn("bright_ti4", rows[0])

    def test_no_data_returns_no_data_status_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with patch("firms_ingestion.fetch_csv_chunk", return_value="latitude,longitude\n"), \
                 patch("firms_ingestion.time.sleep"):
                result = fetch_year_gujarat(2020, output_dir=output_dir, map_key="FAKEKEY")
            self.assertEqual(result["status"], "no_data")
            self.assertFalse((output_dir / "firms_gujarat_2020.csv").exists())


if __name__ == "__main__":
    unittest.main()
