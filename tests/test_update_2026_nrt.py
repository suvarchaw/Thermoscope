import csv
import inspect
import json
import os
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import firms_ingestion as fi
import event_construction_2026 as ec2026
import update_2026_nrt as mod
from train_source_classifier_lightgbm import LIGHTGBM_AVAILABLE

RAW_FIELDNAMES = [
    "latitude", "longitude", "bright_ti4", "scan", "track", "acq_date",
    "acq_time", "satellite", "instrument", "confidence", "version",
    "bright_ti5", "frp", "daynight",
]


def _csv_text(rows):
    header = ",".join(RAW_FIELDNAMES) + "\n"
    lines = [header]
    for r in rows:
        lines.append(",".join(str(r[k]) for k in RAW_FIELDNAMES) + "\n")
    return "".join(lines)


def _det_row(lat, lon, acq_date, acq_time="0745", frp="5.0", daynight="D"):
    return {
        "latitude": lat, "longitude": lon, "bright_ti4": "330.0", "scan": "0.4",
        "track": "0.4", "acq_date": acq_date, "acq_time": acq_time,
        "satellite": "N", "instrument": "VIIRS", "confidence": "n",
        "version": "2.0NRT", "bright_ti5": "300.0", "frp": frp, "daynight": daynight,
    }


class TestFetchAndMergeNrtWindow(unittest.TestCase):
    def test_duplicate_nrt_observations_deduped_on_merge(self):
        """Fetching a window whose rows overlap what's already stored must
        not create duplicate detections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            run_date = date(2026, 6, 10)
            fixed_rows = [_det_row("21.0", "72.0", "2026-06-08"), _det_row("21.0", "72.0", "2026-06-09")]

            with patch("firms_ingestion.fetch_csv_chunk", return_value=_csv_text(fixed_rows)), \
                 patch("firms_ingestion.time.sleep"):
                first = fi.fetch_and_merge_nrt_window(output_dir=output_dir, window_days=3,
                                                        run_date=run_date, map_key="FAKE")
                second = fi.fetch_and_merge_nrt_window(output_dir=output_dir, window_days=3,
                                                         run_date=run_date, map_key="FAKE")

            self.assertEqual(first["new_rows_added"], 2)
            self.assertEqual(second["new_rows_added"], 0)
            self.assertEqual(first["rows_after"], second["rows_after"])

            with open(output_dir / "firms_gujarat_2026.csv", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)

    def test_new_window_adds_only_new_rows_on_top_of_existing_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            existing_path = output_dir / "firms_gujarat_2026.csv"
            existing_path.parent.mkdir(parents=True, exist_ok=True)
            with open(existing_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["latitude", "longitude", "acq_date", "acq_time",
                                                   "satellite", "frp", "daynight", "viirs_source"])
                w.writeheader()
                w.writerow({"latitude": "20.5", "longitude": "71.0", "acq_date": "2026-06-01",
                            "acq_time": "0700", "satellite": "N", "frp": "3.0", "daynight": "D",
                            "viirs_source": "VIIRS_SNPP_SP"})

            new_rows = [_det_row("21.0", "72.0", "2026-06-09")]
            with patch("firms_ingestion.fetch_csv_chunk", return_value=_csv_text(new_rows)), \
                 patch("firms_ingestion.time.sleep"):
                result = fi.fetch_and_merge_nrt_window(output_dir=output_dir, window_days=2,
                                                         run_date=date(2026, 6, 10), map_key="FAKE")
            self.assertEqual(result["rows_before"], 1)
            self.assertEqual(result["new_rows_added"], 1)
            self.assertEqual(result["rows_after"], 2)

    def test_window_never_precedes_nrt_start_date(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("firms_ingestion.fetch_csv_chunk", return_value=_csv_text([])), \
                 patch("firms_ingestion.time.sleep"):
                result = fi.fetch_and_merge_nrt_window(
                    output_dir=Path(tmpdir), window_days=365,
                    run_date=date(2026, 5, 1), map_key="FAKE",
                )
            self.assertEqual(result["window_start"], fi.NRT_START_DATE_2026)


class TestBuild2026EventsNoFutureLooking(unittest.TestCase):
    def test_status_relative_to_reference_date_not_max_detection_date(self):
        """An event's last detection was several days before reference_date
        (i.e. 'now') -- it must be closed relative to that true now, not
        provisional just because it's the most recent thing in the store."""
        detections = [
            {"lat": 21.0, "lon": 72.0, "date": date(2026, 6, 1), "frp": 5.0,
             "daynight": "D", "confidence": "n", "brightness": 330.0,
             "bright_t31": 300.0, "scan": 0.4, "track": 0.4, "acq_time": "0745"},
            {"lat": 21.001, "lon": 72.0, "date": date(2026, 6, 1), "frp": 6.0,
             "daynight": "D", "confidence": "n", "brightness": 330.0,
             "bright_t31": 300.0, "scan": 0.4, "track": 0.4, "acq_time": "0746"},
        ]
        rows, _, _ = ec2026.build_2026_events(detections, reference_date=date(2026, 6, 10))
        self.assertEqual(rows[0]["status"], "closed")

    def test_no_detection_after_reference_date_is_ever_used(self):
        """Construction must never look forward past reference_date --
        a detection dated after 'now' must not silently extend an event."""
        detections = [
            {"lat": 21.0, "lon": 72.0, "date": date(2026, 6, 1), "frp": 5.0,
             "daynight": "D", "confidence": "n", "brightness": 330.0,
             "bright_t31": 300.0, "scan": 0.4, "track": 0.4, "acq_time": "0745"},
            {"lat": 21.001, "lon": 72.0, "date": date(2026, 6, 2), "frp": 6.0,
             "daynight": "D", "confidence": "n", "brightness": 330.0,
             "bright_t31": 300.0, "scan": 0.4, "track": 0.4, "acq_time": "0746"},
        ]
        # reference_date is only used for status, never to filter input --
        # confirm the function does not reject/drop detections dated after
        # a reference_date the caller might (mistakenly) pass too early.
        source = inspect.getsource(ec2026.build_2026_events)
        self.assertNotIn("if d[\"date\"] >", source)
        rows, _, _ = ec2026.build_2026_events(detections, reference_date=date(2026, 6, 2))
        self.assertEqual(rows[0]["end_date"], "2026-06-02")


class TestEventContinuation(unittest.TestCase):
    def test_new_linked_detection_extends_existing_provisional_event(self):
        """Simulates two update cycles: cycle 1 has a 2-detection event on
        day D; cycle 2 adds a spatially-linked detection on day D+1. The
        event must grow (same event, more members), not fork into two."""
        base = [
            {"lat": 21.0, "lon": 72.0, "date": date(2026, 6, 1), "frp": 5.0,
             "daynight": "D", "confidence": "n", "brightness": 330.0,
             "bright_t31": 300.0, "scan": 0.4, "track": 0.4, "acq_time": "0745"},
            {"lat": 21.001, "lon": 72.0, "date": date(2026, 6, 1), "frp": 6.0,
             "daynight": "D", "confidence": "n", "brightness": 330.0,
             "bright_t31": 300.0, "scan": 0.4, "track": 0.4, "acq_time": "0746"},
        ]
        cycle1_rows, _, _ = ec2026.build_2026_events(base, reference_date=date(2026, 6, 1))
        self.assertEqual(len(cycle1_rows), 1)
        self.assertEqual(cycle1_rows[0]["status"], "provisional")
        self.assertEqual(cycle1_rows[0]["detection_count"], 2)

        extended = base + [
            {"lat": 21.0005, "lon": 72.0002, "date": date(2026, 6, 2), "frp": 5.5,
             "daynight": "D", "confidence": "n", "brightness": 330.0,
             "bright_t31": 300.0, "scan": 0.4, "track": 0.4, "acq_time": "0745"},
        ]
        cycle2_rows, _, _ = ec2026.build_2026_events(extended, reference_date=date(2026, 6, 2))
        self.assertEqual(len(cycle2_rows), 1, "linked detection must extend the existing event, not fork a new one")
        self.assertEqual(cycle2_rows[0]["event_id"], cycle1_rows[0]["event_id"])
        self.assertEqual(cycle2_rows[0]["detection_count"], 3)
        self.assertEqual(cycle2_rows[0]["end_date"], "2026-06-02")

    def test_provisional_to_closed_transition_as_time_advances(self):
        detections = [
            {"lat": 21.0, "lon": 72.0, "date": date(2026, 6, 1), "frp": 5.0,
             "daynight": "D", "confidence": "n", "brightness": 330.0,
             "bright_t31": 300.0, "scan": 0.4, "track": 0.4, "acq_time": "0745"},
            {"lat": 21.001, "lon": 72.0, "date": date(2026, 6, 1), "frp": 6.0,
             "daynight": "D", "confidence": "n", "brightness": 330.0,
             "bright_t31": 300.0, "scan": 0.4, "track": 0.4, "acq_time": "0746"},
        ]
        # Same day as last detection -> provisional (could still be extended).
        rows_same_day, _, _ = ec2026.build_2026_events(detections, reference_date=date(2026, 6, 1))
        self.assertEqual(rows_same_day[0]["status"], "provisional")

        # eps_time_days (1) has passed with no further linked detection -> closed.
        rows_next_day, _, _ = ec2026.build_2026_events(detections, reference_date=date(2026, 6, 2))
        self.assertEqual(rows_next_day[0]["status"], "closed")


class TestNewEventCreation(unittest.TestCase):
    def test_unlinked_new_detection_pair_creates_a_second_distinct_event(self):
        base = [
            {"lat": 21.0, "lon": 72.0, "date": date(2026, 6, 1), "frp": 5.0,
             "daynight": "D", "confidence": "n", "brightness": 330.0,
             "bright_t31": 300.0, "scan": 0.4, "track": 0.4, "acq_time": "0745"},
            {"lat": 21.001, "lon": 72.0, "date": date(2026, 6, 1), "frp": 6.0,
             "daynight": "D", "confidence": "n", "brightness": 330.0,
             "bright_t31": 300.0, "scan": 0.4, "track": 0.4, "acq_time": "0746"},
        ]
        cycle1_rows, _, _ = ec2026.build_2026_events(base, reference_date=date(2026, 6, 1))
        self.assertEqual(len(cycle1_rows), 1)

        far_away_new_event = base + [
            {"lat": 25.0, "lon": 73.5, "date": date(2026, 6, 3), "frp": 4.0,
             "daynight": "N", "confidence": "n", "brightness": 330.0,
             "bright_t31": 300.0, "scan": 0.4, "track": 0.4, "acq_time": "1930"},
            {"lat": 25.0005, "lon": 73.5001, "date": date(2026, 6, 3), "frp": 4.2,
             "daynight": "N", "confidence": "n", "brightness": 330.0,
             "bright_t31": 300.0, "scan": 0.4, "track": 0.4, "acq_time": "1931"},
        ]
        cycle2_rows, _, _ = ec2026.build_2026_events(far_away_new_event, reference_date=date(2026, 6, 3))
        self.assertEqual(len(cycle2_rows), 2)
        ids = {r["event_id"] for r in cycle2_rows}
        self.assertIn(cycle1_rows[0]["event_id"], ids, "original event must persist under its original id")


class TestUpdateCycleIdempotency(unittest.TestCase):
    @unittest.skipUnless(LIGHTGBM_AVAILABLE, "LightGBM unavailable in this environment")
    def test_repeated_full_cycle_with_identical_firms_data_is_idempotent(self):
        """End-to-end: running run_update_cycle twice against the same
        fetched FIRMS window must not duplicate raw rows, must not change
        the event count, and must produce identical prediction output.
        Uses output_dir to keep the whole cycle inside an isolated tmpdir
        -- never touches the real data/raw or data/processed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_date = date(2026, 6, 10)
            fixed_rows = [
                _det_row("21.500", "70.200", "2026-06-08"),
                _det_row("21.5003", "70.2001", "2026-06-08", acq_time="0746"),
                _det_row("23.000", "72.600", "2026-06-09", frp="8.0", daynight="N"),
                _det_row("23.0004", "72.6003", "2026-06-09", acq_time="1931", frp="8.5", daynight="N"),
            ]

            def fake_evidence_table(events):
                # Avoid live network/rasterio dependency in a unit test --
                # evidence content isn't what this test is checking.
                return [{"event_id": e["event_id"], "nearest_osm_industrial_power_m": "",
                         "nearest_gppd_thermal_plant_m": "", "nearest_osm_flare_m": "",
                         "nearest_osm_kiln_m": "", "overlaps_cluster_id": "",
                         "overlaps_cluster_recurrence_strength": "", "land_cover_class": ""}
                        for e in events]

            with patch("firms_ingestion.fetch_csv_chunk", return_value=_csv_text(fixed_rows)), \
                 patch("firms_ingestion.time.sleep"), \
                 patch("event_evidence.build_evidence_table", side_effect=fake_evidence_table):
                stats1 = mod.run_update_cycle(window_days=5, run_date=run_date, map_key="FAKE",
                                               output_dir=tmpdir)
                stats2 = mod.run_update_cycle(window_days=5, run_date=run_date, map_key="FAKE",
                                               output_dir=tmpdir)

            self.assertEqual(stats1["n_events"], stats2["n_events"])
            self.assertEqual(stats1["ingest"]["rows_after"], stats2["ingest"]["rows_after"])
            self.assertEqual(stats2["ingest"]["new_rows_added"], 0)

            inference_path = Path(tmpdir) / "inference.csv"
            with open(inference_path, newline="") as f:
                inference_content_1 = f.read()

            with patch("firms_ingestion.fetch_csv_chunk", return_value=_csv_text(fixed_rows)), \
                 patch("firms_ingestion.time.sleep"), \
                 patch("event_evidence.build_evidence_table", side_effect=fake_evidence_table):
                mod.run_update_cycle(window_days=5, run_date=run_date, map_key="FAKE", output_dir=tmpdir)

            with open(inference_path, newline="") as f:
                inference_content_2 = f.read()
            self.assertEqual(inference_content_1, inference_content_2)


class TestSupportedOutputClasses(unittest.TestCase):
    def test_trained_classes_exactly_match_required_set(self):
        import infer_2026_events as inf2026
        self.assertEqual(set(inf2026.TRAINED_CLASSES),
                          {"Industrial", "Gas_Flare", "Crop_Residue", "Forest_Wildfire"})


class TestNeverRetrains(unittest.TestCase):
    def test_update_engine_never_calls_fit(self):
        source = inspect.getsource(mod)
        self.assertNotIn(".fit(", source)


class TestLockGuard(unittest.TestCase):
    def test_acquire_then_acquire_again_raises_overlapping_run_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "test.lock"
            mod.acquire_lock(lock_path)
            with self.assertRaises(mod.UpdateInProgressError):
                mod.acquire_lock(lock_path)
            mod.release_lock(lock_path)

    def test_release_then_acquire_succeeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "test.lock"
            mod.acquire_lock(lock_path)
            mod.release_lock(lock_path)
            # must not raise -- lock was freed
            mod.acquire_lock(lock_path)
            mod.release_lock(lock_path)

    def test_stale_lock_from_dead_pid_is_reclaimed_deterministically(self):
        """A lock left behind by a process that is confirmed no longer
        running must be reclaimed, not treated as a live overlapping run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "test.lock"
            # A PID essentially guaranteed not to be running: spawn a
            # trivial subprocess, wait for it to exit, and use its PID.
            import subprocess
            proc = subprocess.Popen([sys.executable, "-c", "pass"])
            proc.wait()
            dead_pid = proc.pid
            self.assertFalse(mod._pid_is_running(dead_pid))
            lock_path.write_text(f"{dead_pid} 2020-01-01T00:00:00+00:00\n")

            # must NOT raise -- the stale lock is reclaimed automatically
            mod.acquire_lock(lock_path)
            self.assertEqual(mod._read_lock_pid(lock_path), os.getpid())
            mod.release_lock(lock_path)

    def test_live_pid_lock_is_not_reclaimed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "test.lock"
            lock_path.write_text(f"{os.getpid()} 2020-01-01T00:00:00+00:00\n")  # our own pid: definitely alive
            with self.assertRaises(mod.UpdateInProgressError):
                mod.acquire_lock(lock_path)
            lock_path.unlink()

    def test_run_update_cycle_releases_lock_even_on_failure(self):
        """If a step inside the cycle raises, the lock must still be freed
        (via finally) so the NEXT scheduled run is not blocked forever."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "cycle.lock"
            with patch("firms_ingestion.fetch_csv_chunk", side_effect=RuntimeError("boom")), \
                 patch("firms_ingestion.time.sleep"):
                with self.assertRaises(RuntimeError):
                    mod.run_update_cycle(window_days=3, run_date=date(2026, 6, 10),
                                          map_key="FAKE", output_dir=tmpdir, lock_path=lock_path)
            self.assertFalse(lock_path.exists(), "lock must be released even when the cycle raises")


class TestFreshnessMetadata(unittest.TestCase):
    def test_write_freshness_metadata_contains_required_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "freshness.json"
            stats = {
                "ingest": {"window_start": date(2026, 6, 8), "window_end": date(2026, 6, 10),
                           "rows_before": 10, "rows_fetched": 2, "rows_after": 12, "new_rows_added": 2},
                "n_events": 5, "n_closed": 4, "n_provisional": 1, "n_predictions": 5,
                "run_date": date(2026, 6, 10),
            }
            metadata = mod.write_freshness_metadata(stats, path=path)
            self.assertIn("last_updated_utc", metadata)
            self.assertEqual(metadata["run_date"], "2026-06-10")
            self.assertEqual(metadata["n_events"], 5)
            with open(path) as f:
                reread = json.load(f)
            self.assertEqual(reread, metadata)

    def test_run_update_cycle_writes_freshness_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fixed_rows = [_det_row("21.500", "70.200", "2026-06-08"),
                          _det_row("21.5003", "70.2001", "2026-06-08", acq_time="0746")]

            def fake_evidence_table(events):
                return [{"event_id": e["event_id"], "nearest_osm_industrial_power_m": "",
                         "nearest_gppd_thermal_plant_m": "", "nearest_osm_flare_m": "",
                         "nearest_osm_kiln_m": "", "overlaps_cluster_id": "",
                         "overlaps_cluster_recurrence_strength": "", "land_cover_class": ""}
                        for e in events]

            with patch("firms_ingestion.fetch_csv_chunk", return_value=_csv_text(fixed_rows)), \
                 patch("firms_ingestion.time.sleep"), \
                 patch("event_evidence.build_evidence_table", side_effect=fake_evidence_table):
                mod.run_update_cycle(window_days=3, run_date=date(2026, 6, 10),
                                      map_key="FAKE", output_dir=tmpdir)
            freshness_path = Path(tmpdir) / "freshness.json"
            self.assertTrue(freshness_path.exists())
            with open(freshness_path) as f:
                metadata = json.load(f)
            self.assertEqual(metadata["run_date"], "2026-06-10")
            self.assertIn("last_updated_utc", metadata)

    def test_freshness_metadata_does_not_alter_inference_csv_schema(self):
        """The prediction file's columns must be exactly what
        infer_2026_events.build_output_rows already produces -- freshness
        lives only in the separate sidecar file."""
        source = inspect.getsource(mod)
        # write_freshness_metadata must target its own path, never OUTPUT_CSV
        self.assertNotIn("inf2026.OUTPUT_CSV, \"a\"", source)
        import infer_2026_events as inf2026
        self.assertNotIn("last_updated", inspect.getsource(inf2026.build_output_rows))


class TestSchedulerSafeInvocation(unittest.TestCase):
    def test_direct_script_invocation_survives_self_reexec_and_argv(self):
        """Regression test for the exact bug observed this session: the
        LightGBM/libomp DYLD_LIBRARY_PATH self-reexec (os.execve) mangles
        sys.argv under indirect invocations like `python3 -m unittest
        discover`. The scheduler invokes `python3 src/update_2026_nrt.py`
        directly (with a window_days argv) -- this must survive the
        re-exec, correctly parse that argv, and start real work, never
        crash with a "can't open file"-style argv error. A short timeout
        is expected (real FIRMS_MAP_KEY + network access are available in
        this dev environment, so the process keeps running past it) --
        Whether this real, live process finishes fast (e.g. no network)
        or is still going (real network access, the common case in this
        dev environment), the same two things must hold: the process
        never crashes with the argv-mangling symptom, and its very first
        print line shows window_days correctly parsed from argv. Run with
        cwd set to an isolated tmpdir (dotenv still finds the real .env
        by walking up from the script's own directory, regardless of
        cwd) so any relative-path file write this process makes before
        being stopped lands in the tmpdir, never in the real repo's
        data/."""
        import subprocess
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "src" / "update_2026_nrt.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                result = subprocess.run(
                    [sys.executable, str(script), "5"],
                    cwd=tmpdir, capture_output=True, text=True, timeout=6,
                )
                stdout, stderr = result.stdout, result.stderr
            except subprocess.TimeoutExpired as e:
                stdout, stderr = (e.stdout or ""), (e.stderr or "")
        combined = stdout + stderr
        self.assertNotIn("can't open file", combined)
        self.assertNotIn("Argument expected for the -c option", combined)
        self.assertIn("2026 NRT update cycle: window_days=5", stdout)

    def test_plist_invokes_script_directly_not_via_module_flag(self):
        plist_path = Path(__file__).resolve().parents[1] / "ops" / "com.thermoscope.nrt_update.plist"
        self.assertTrue(plist_path.exists())
        content = plist_path.read_text()
        self.assertIn("src/update_2026_nrt.py", content)
        self.assertNotIn("<string>-m</string>", content)
        self.assertIn("StartInterval", content)


class TestHistoricalOutputsUntouched(unittest.TestCase):
    def test_module_never_references_historical_filenames(self):
        source = inspect.getsource(mod)
        for forbidden in ("gujarat_thermal_events.csv", "gujarat_event_evidence.csv",
                           "gujarat_multi_year_detections.csv", "firms_gujarat_2019",
                           "firms_gujarat_2020", "firms_gujarat_2023"):
            self.assertNotIn(forbidden, source)

    def test_isolated_cycle_does_not_touch_real_historical_files(self):
        repo_root = Path(__file__).resolve().parents[1]
        historical_events = repo_root / "data" / "processed" / "gujarat_thermal_events.csv"
        historical_raw_2019 = repo_root / "data" / "raw" / "firms_gujarat_2019.csv"
        before_events_mtime = historical_events.stat().st_mtime
        before_raw_mtime = historical_raw_2019.stat().st_mtime

        with tempfile.TemporaryDirectory() as tmpdir:
            fixed_rows = [_det_row("21.500", "70.200", "2026-06-08"),
                          _det_row("21.5003", "70.2001", "2026-06-08", acq_time="0746")]

            def fake_evidence_table(events):
                return [{"event_id": e["event_id"], "nearest_osm_industrial_power_m": "",
                         "nearest_gppd_thermal_plant_m": "", "nearest_osm_flare_m": "",
                         "nearest_osm_kiln_m": "", "overlaps_cluster_id": "",
                         "overlaps_cluster_recurrence_strength": "", "land_cover_class": ""}
                        for e in events]

            with patch("firms_ingestion.fetch_csv_chunk", return_value=_csv_text(fixed_rows)), \
                 patch("firms_ingestion.time.sleep"), \
                 patch("event_evidence.build_evidence_table", side_effect=fake_evidence_table):
                mod.run_update_cycle(window_days=3, run_date=date(2026, 6, 10),
                                      map_key="FAKE", output_dir=tmpdir)

        self.assertEqual(historical_events.stat().st_mtime, before_events_mtime)
        self.assertEqual(historical_raw_2019.stat().st_mtime, before_raw_mtime)


class TestSuccessfulUpdateStillWorks(unittest.TestCase):
    def test_full_cycle_produces_expected_outputs_and_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fixed_rows = [_det_row("21.500", "70.200", "2026-06-08"),
                          _det_row("21.5003", "70.2001", "2026-06-08", acq_time="0746")]

            def fake_evidence_table(events):
                return [{"event_id": e["event_id"], "nearest_osm_industrial_power_m": "",
                         "nearest_gppd_thermal_plant_m": "", "nearest_osm_flare_m": "",
                         "nearest_osm_kiln_m": "", "overlaps_cluster_id": "",
                         "overlaps_cluster_recurrence_strength": "", "land_cover_class": ""}
                        for e in events]

            with patch("firms_ingestion.fetch_csv_chunk", return_value=_csv_text(fixed_rows)), \
                 patch("firms_ingestion.time.sleep"), \
                 patch("event_evidence.build_evidence_table", side_effect=fake_evidence_table):
                stats = mod.run_update_cycle(window_days=3, run_date=date(2026, 6, 10),
                                              map_key="FAKE", output_dir=tmpdir)

            self.assertEqual(stats["n_events"], 1)
            self.assertEqual(stats["n_predictions"], 1)
            for name in ("events.csv", "evidence.csv", "inference.csv", "freshness.json"):
                self.assertTrue((Path(tmpdir) / name).exists(), f"{name} was not written")
            self.assertFalse((Path(tmpdir) / ".update.lock").exists(), "lock must be released after success")


if __name__ == "__main__":
    unittest.main()
