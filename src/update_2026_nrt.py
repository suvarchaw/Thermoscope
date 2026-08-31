"""
2026 NRT update engine -- the single operational command that turns the
existing batch 2026 pipeline (ingest_2026_gujarat.py /
event_construction_2026.py / event_evidence_2026.py /
infer_2026_events.py, all built in the prior milestone) into a
repeatable, idempotent update cycle.

Not a new pipeline: every step below calls an EXISTING, unchanged
function from those modules. This module's only new logic is
(a) firms_ingestion.fetch_and_merge_nrt_window, which fetches a small
recent NRT window and merges+dedupes it into the existing 2026 raw
store, and (b) the orchestration below that runs the existing
construct -> evidence -> infer chain once per cycle against whatever the
merged raw store currently contains.

WHY "REBUILD EVERYTHING FROM THE MERGED STORE" IS THE CORRECT (AND
SIMPLEST) WAY TO "UPDATE" EVENTS
-------------------------------------------------------------------------
event_construction.py's own docstring establishes that its connected-
components algorithm is "online-safe by construction": a detection on day
D only ever links to detections already seen, so a full re-run over the
complete detection set produces IDENTICAL groupings to an incremental,
day-by-day run for every day already processed. This module leans on
that guarantee directly instead of writing a second, bespoke incremental
merge algorithm: every update cycle re-derives ALL 2026 events from the
full (deduplicated) 2026 raw store via the exact same
event_construction_2026.build_2026_events used by the batch pipeline.
Given identical input detections, this is trivially idempotent; given
NEW detections, existing (already-closed) events are structurally
unaffected (their own members are already >=1 day old and cannot gain a
new causal link), while events still within eps_time_days of "now"
correctly grow or flip provisional->closed as appropriate.

IDEMPOTENCY, END TO END
-------------------------------------------------------------------------
1. Raw store: fetch_and_merge_nrt_window dedupes on
   (lat, lon, acq_date, acq_time, satellite) -- re-fetching the same
   window twice adds zero new rows the second time.
2. Events/evidence/inference: pure functions of the (now-stable) raw
   store + reference_date. Same store + same reference_date => byte-
   identical output every time (verified by test).

CONCURRENCY GUARD
-------------------------------------------------------------------------
A PID lock file (LOCK_PATH) guards the whole cycle so two overlapping
scheduled runs (e.g. a slow run still going when the next interval
fires) cannot write the same output files at once. Acquisition is
atomic (os.O_CREAT | os.O_EXCL) so two processes racing to create the
lock can never both believe they hold it. Stale-lock recovery is
deterministic, not time-based: a lock is only ever reclaimed if the PID
recorded in it is no longer a running process (checked via
os.kill(pid, 0)) -- never just because the lock "looks old". This
avoids guessing a timeout while still recovering automatically from a
crashed prior run.

FRESHNESS METADATA
-------------------------------------------------------------------------
A small sidecar JSON (FRESHNESS_PATH) is written after a successful
cycle, separate from gujarat_2026_inference.csv itself -- the prediction
file's schema and semantics are completely unchanged. The dashboard can
read this file to show "as of" freshness without altering how it reads
predictions.
"""

import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import event_construction_2026 as ec2026
import event_evidence as ee
import event_evidence_2026 as ee2026
import infer_2026_events as inf2026
from firms_ingestion import fetch_and_merge_nrt_window, DEFAULT_NRT_WINDOW_DAYS, FirmsConfigError

LOCK_PATH = Path("data/processed/.update_2026_nrt.lock")
FRESHNESS_PATH = Path("data/processed/gujarat_2026_inference_freshness.json")


class UpdateInProgressError(Exception):
    """Raised when another update cycle already holds the lock."""


def _pid_is_running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists, just owned by someone else
    return True


def _read_lock_pid(lock_path):
    try:
        return int(lock_path.read_text().strip().split()[0])
    except (ValueError, IndexError, OSError):
        return None


def acquire_lock(lock_path=LOCK_PATH):
    """Atomically creates lock_path, reclaiming it first if the PID
    recorded inside is confirmed no longer running (deterministic
    liveness check, not a fuzzy age-based timeout). Raises
    UpdateInProgressError if another live run holds it."""
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    for _attempt in range(2):  # second pass only runs after reclaiming a confirmed-stale lock
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(f"{os.getpid()} {datetime.now(timezone.utc).isoformat()}\n")
            return lock_path
        except FileExistsError:
            existing_pid = _read_lock_pid(lock_path)
            if existing_pid is not None and _pid_is_running(existing_pid):
                raise UpdateInProgressError(
                    f"Another update (pid={existing_pid}) is already running per {lock_path} "
                    f"-- refusing to start a second overlapping run."
                )
            lock_path.unlink(missing_ok=True)  # stale: process is gone (or lock unreadable) -- reclaim

    raise UpdateInProgressError(f"Could not acquire {lock_path} after reclaiming a stale lock.")


def release_lock(lock_path=LOCK_PATH):
    Path(lock_path).unlink(missing_ok=True)


def write_freshness_metadata(stats, path=FRESHNESS_PATH):
    """Small, additive sidecar -- never touches gujarat_2026_inference.csv
    itself, so existing prediction semantics/schema are unchanged."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ing = stats["ingest"]
    metadata = {
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": stats["run_date"].isoformat(),
        "nrt_window_start": ing["window_start"].isoformat(),
        "nrt_window_end": ing["window_end"].isoformat(),
        "n_events": stats["n_events"],
        "n_closed": stats["n_closed"],
        "n_provisional": stats["n_provisional"],
        "n_predictions": stats["n_predictions"],
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def run_update_cycle(window_days=DEFAULT_NRT_WINDOW_DAYS, run_date=None, map_key=None,
                      output_dir=None, lock_path=None, freshness_path=None):
    """One complete NRT update cycle, guarded by a PID lock so two
    overlapping runs can never write the same outputs concurrently.
    Returns a stats dict covering every step, for the caller to report or
    assert on. Raises UpdateInProgressError (before touching any output)
    if another run already holds the lock.

    output_dir overrides where the 2026 raw store lives (both for the
    fetch/merge step and for the event-construction read that follows) --
    exposed so tests can point the whole cycle at an isolated tmpdir
    instead of the real data/raw/, without touching any other behavior.
    lock_path/freshness_path default to output_dir-relative paths when
    output_dir is set, else the real LOCK_PATH/FRESHNESS_PATH."""
    if run_date is None:
        run_date = date.today()
    if lock_path is None:
        lock_path = Path(output_dir) / ".update.lock" if output_dir is not None else LOCK_PATH
    if freshness_path is None:
        freshness_path = Path(output_dir) / "freshness.json" if output_dir is not None else FRESHNESS_PATH

    acquire_lock(lock_path)
    try:
        fetch_kwargs = {"window_days": window_days, "run_date": run_date, "map_key": map_key}
        raw_path = None
        if output_dir is not None:
            fetch_kwargs["output_dir"] = output_dir
            raw_path = Path(output_dir) / "firms_gujarat_2026.csv"
        ingest_stats = fetch_and_merge_nrt_window(**fetch_kwargs)

        # Steps 4+7 (events -> features, ready for inference): reuses
        # event_construction_2026.build_2026_events unchanged, via
        # infer_2026_events.build_2026_feature_rows, so events are derived
        # exactly once per cycle (not recomputed separately per step).
        events, feature_rows = inf2026.build_2026_feature_rows(reference_date=run_date, raw_path=raw_path)
        events_out_path = Path(output_dir) / "events.csv" if output_dir is not None else ec2026.OUTPUT_EVENTS_CSV
        ec2026.write_csv(events, path=events_out_path)

        # Step 6: evidence, reusing event_evidence.build_evidence_table unchanged.
        evidence_rows = ee.build_evidence_table(events)
        evidence_out_path = Path(output_dir) / "evidence.csv" if output_dir is not None else ee2026.OUTPUT_CSV
        ee.write_csv(evidence_rows, path=evidence_out_path)
        evidence_by_id = {r["event_id"]: r for r in evidence_rows}

        # Steps 7-8: locked-model inference only -- load_model/predict_2026
        # never call .fit anywhere (see infer_2026_events.py and its tests).
        pipeline = inf2026.load_model()
        y_pred, prob_by_class = inf2026.predict_2026(pipeline, feature_rows)
        output_rows = inf2026.build_output_rows(events, feature_rows, y_pred, prob_by_class, evidence_by_id)
        inference_out_path = Path(output_dir) / "inference.csv" if output_dir is not None else inf2026.OUTPUT_CSV
        inf2026.write_csv(output_rows, path=inference_out_path)

        n_closed = sum(1 for e in events if e["status"] == "closed")
        n_provisional = len(events) - n_closed

        stats = {
            "ingest": ingest_stats,
            "n_events": len(events),
            "n_closed": n_closed,
            "n_provisional": n_provisional,
            "n_predictions": len(output_rows),
            "run_date": run_date,
        }
        write_freshness_metadata(stats, path=freshness_path)
        return stats
    finally:
        release_lock(lock_path)


def main(window_days=DEFAULT_NRT_WINDOW_DAYS):
    if len(sys.argv) > 1:
        window_days = int(sys.argv[1])

    print(f"2026 NRT update cycle: window_days={window_days}")
    try:
        stats = run_update_cycle(window_days=window_days)
    except UpdateInProgressError as e:
        # Expected, benign outcome under a periodic scheduler (e.g. a slow
        # prior run still finishing) -- exit cleanly, not with a crash.
        print("SKIPPED (update already in progress)")
        print(f"  {e}")
        return None
    except FirmsConfigError as e:
        print("BLOCKED (missing configuration)")
        print(f"  {e}")
        return None

    ing = stats["ingest"]
    print(f"\nIngest: window {ing['window_start']}..{ing['window_end']}, "
          f"fetched {ing['rows_fetched']} rows, "
          f"{ing['rows_before']} -> {ing['rows_after']} total "
          f"(+{ing['new_rows_added']} new after dedup)")
    print(f"Events: {stats['n_events']} total "
          f"({stats['n_closed']} closed, {stats['n_provisional']} provisional)")
    print(f"Predictions written: {stats['n_predictions']}")
    print(f"\nOutputs updated:")
    print(f"  {ec2026.OUTPUT_EVENTS_CSV}")
    print(f"  {ee2026.OUTPUT_CSV}")
    print(f"  {inf2026.OUTPUT_CSV}")
    print(f"  {FRESHNESS_PATH}")
    return stats


if __name__ == "__main__":
    main()
