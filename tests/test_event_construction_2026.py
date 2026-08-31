import csv
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import event_construction_2026 as mod

RAW_FIELDNAMES = [
    "latitude", "longitude", "brightness", "scan", "track", "acq_date",
    "acq_time", "satellite", "confidence", "bright_t31", "frp", "daynight",
]


def _write_raw_2026(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


def _row(lat, lon, acq_date, acq_time="0745", confidence="n", frp="5.0", daynight="D"):
    return {
        "latitude": lat, "longitude": lon, "brightness": "330.0", "scan": "0.4",
        "track": "0.4", "acq_date": acq_date, "acq_time": acq_time,
        "satellite": "N", "confidence": confidence, "bright_t31": "300.0",
        "frp": frp, "daynight": daynight,
    }


class TestLoad2026Detections(unittest.TestCase):
    def test_loads_and_reshapes_rich_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "firms_gujarat_2026.csv"
            _write_raw_2026([_row("21.0", "72.0", "2026-02-01")], path)
            detections = mod.load_2026_detections(path)
        self.assertEqual(len(detections), 1)
        d = detections[0]
        self.assertEqual(d["lat"], 21.0)
        self.assertEqual(d["date"], date(2026, 2, 1))
        self.assertIn("scan", d)
        self.assertIn("acq_time", d)


class TestBuild2026Events(unittest.TestCase):
    def test_empty_input_returns_empty(self):
        rows, members, n_raw = mod.build_2026_events([])
        self.assertEqual(rows, [])
        self.assertEqual(members, [])
        self.assertEqual(n_raw, 0)

    def test_two_nearby_same_day_detections_form_one_event(self):
        detections = [
            {"lat": 21.0, "lon": 72.0, "date": date(2026, 3, 1), "frp": 5.0,
             "daynight": "D", "confidence": "n", "brightness": 330.0,
             "bright_t31": 300.0, "scan": 0.4, "track": 0.4, "acq_time": "0745"},
            {"lat": 21.003, "lon": 72.0, "date": date(2026, 3, 1), "frp": 6.0,
             "daynight": "D", "confidence": "n", "brightness": 330.0,
             "bright_t31": 300.0, "scan": 0.4, "track": 0.4, "acq_time": "0746"},
        ]
        rows, members_by_row, n_raw = mod.build_2026_events(detections)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["event_id"].startswith(mod.EVENT_ID_PREFIX))
        self.assertEqual(rows[0]["detection_count"], 2)
        self.assertEqual(sorted(members_by_row[0]), [0, 1])

    def test_singleton_detection_excluded(self):
        detections = [
            {"lat": 21.0, "lon": 72.0, "date": date(2026, 3, 1), "frp": 5.0,
             "daynight": "D", "confidence": "n", "brightness": 330.0,
             "bright_t31": 300.0, "scan": 0.4, "track": 0.4, "acq_time": "0745"},
        ]
        rows, members_by_row, n_raw = mod.build_2026_events(detections)
        self.assertEqual(n_raw, 1)
        self.assertEqual(rows, [])

    def test_event_ids_are_2026_namespaced_and_sequential(self):
        detections = []
        for day_offset, (lat, lon) in enumerate([(21.0, 72.0), (25.0, 73.0)]):
            for i in range(2):
                detections.append({
                    "lat": lat, "lon": lon, "date": date(2026, 1, 1 + day_offset),
                    "frp": 5.0, "daynight": "D", "confidence": "n",
                    "brightness": 330.0, "bright_t31": 300.0, "scan": 0.4,
                    "track": 0.4, "acq_time": "0745",
                })
        rows, _, _ = mod.build_2026_events(detections)
        self.assertEqual(len(rows), 2)
        ids = [r["event_id"] for r in rows]
        self.assertEqual(ids, ["EVT2026_000000", "EVT2026_000001"])


class TestWriteCsv(unittest.TestCase):
    def test_write_and_reread(self):
        detections = [
            {"lat": 21.0, "lon": 72.0, "date": date(2026, 3, 1), "frp": 5.0,
             "daynight": "D", "confidence": "n", "brightness": 330.0,
             "bright_t31": 300.0, "scan": 0.4, "track": 0.4, "acq_time": "0745"},
            {"lat": 21.001, "lon": 72.0, "date": date(2026, 3, 1), "frp": 6.0,
             "daynight": "N", "confidence": "n", "brightness": 330.0,
             "bright_t31": 300.0, "scan": 0.4, "track": 0.4, "acq_time": "1930"},
        ]
        rows, _, _ = mod.build_2026_events(detections)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.csv"
            mod.write_csv(rows, path=path)
            with open(path, newline="") as f:
                reread = list(csv.DictReader(f))
        self.assertEqual(len(reread), 1)
        self.assertEqual(reread[0]["event_id"], rows[0]["event_id"])

    def test_write_empty_rows_still_writes_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.csv"
            mod.write_csv([], path=path)
            with open(path, newline="") as f:
                header = next(csv.reader(f))
        self.assertEqual(header, mod.EVENT_FIELDNAMES)


if __name__ == "__main__":
    unittest.main()
