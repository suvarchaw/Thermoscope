import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import event_construction as mod


def make_det(lat, lon, d, frp=5.0, daynight="D"):
    return {"lat": lat, "lon": lon, "date": d, "frp": frp, "daynight": daynight}


class TestHaversine(unittest.TestCase):
    def test_zero_distance(self):
        self.assertAlmostEqual(mod.haversine_m(21.0, 72.0, 21.0, 72.0), 0.0)

    def test_known_offset(self):
        # ~0.0033687 deg latitude offset is ~375m
        d = mod.haversine_m(21.0, 72.0, 21.0033687, 72.0)
        self.assertAlmostEqual(d, 375.0, delta=1.0)


class TestUnionFind(unittest.TestCase):
    def test_union_merges_roots(self):
        uf = mod.UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        self.assertEqual(uf.find(0), uf.find(2))
        self.assertNotEqual(uf.find(0), uf.find(3))

    def test_find_is_stable_after_path_compression(self):
        uf = mod.UnionFind(4)
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(1, 2)
        roots = {uf.find(i) for i in range(4)}
        self.assertEqual(len(roots), 1)


class TestBuildEventsSpatial(unittest.TestCase):
    def test_points_within_eps_space_same_day_link(self):
        d = date(2023, 1, 1)
        dets = [make_det(21.0, 72.0, d), make_det(21.003, 72.0, d)]  # ~333m apart
        events = mod.build_events(dets, eps_space_m=750, eps_time_days=1)
        self.assertEqual(len(events), 1)
        self.assertEqual(sorted(events[0]), [0, 1])

    def test_points_beyond_eps_space_do_not_link(self):
        d = date(2023, 1, 1)
        dets = [make_det(21.0, 72.0, d), make_det(21.02, 72.0, d)]  # ~2.2km apart
        events = mod.build_events(dets, eps_space_m=750, eps_time_days=1)
        self.assertEqual(len(events), 2)

    def test_far_apart_points_never_link_even_same_day(self):
        d = date(2023, 6, 15)
        dets = [make_det(20.5, 68.5, d), make_det(24.5, 74.0, d)]
        events = mod.build_events(dets, eps_space_m=750, eps_time_days=1)
        self.assertEqual(len(events), 2)


class TestBuildEventsTemporal(unittest.TestCase):
    def test_consecutive_days_within_eps_time_link(self):
        dets = [make_det(21.0, 72.0, date(2023, 1, 1)),
                make_det(21.0, 72.0, date(2023, 1, 2))]
        events = mod.build_events(dets, eps_space_m=750, eps_time_days=1)
        self.assertEqual(len(events), 1)

    def test_gap_beyond_eps_time_does_not_link(self):
        dets = [make_det(21.0, 72.0, date(2023, 1, 1)),
                make_det(21.0, 72.0, date(2023, 1, 3))]  # 2-day gap, eps_time=1
        events = mod.build_events(dets, eps_space_m=750, eps_time_days=1)
        self.assertEqual(len(events), 2)

    def test_exact_eps_time_boundary_links(self):
        dets = [make_det(21.0, 72.0, date(2023, 1, 1)),
                make_det(21.0, 72.0, date(2023, 1, 3))]
        events = mod.build_events(dets, eps_space_m=750, eps_time_days=2)
        self.assertEqual(len(events), 1)

    def test_transitive_chain_across_multiple_days(self):
        # A (day1) -- B (day2) -- C (day3): A and C not directly within
        # eps_time of each other (gap=2) but connected via B (chain).
        dets = [make_det(21.000, 72.000, date(2023, 1, 1)),
                make_det(21.000, 72.000, date(2023, 1, 2)),
                make_det(21.000, 72.000, date(2023, 1, 3))]
        events = mod.build_events(dets, eps_space_m=750, eps_time_days=1)
        self.assertEqual(len(events), 1)
        self.assertEqual(sorted(events[0]), [0, 1, 2])

    def test_spatial_drift_across_days_still_chains(self):
        # Each successive day's point is 700m from the previous (within
        # eps_space) but the first and last are >750m apart directly --
        # an event should still be able to "drift" spatially over time.
        dets = [make_det(21.0000, 72.0000, date(2023, 1, 1)),
                make_det(21.0063, 72.0000, date(2023, 1, 2)),  # ~700m north
                make_det(21.0126, 72.0000, date(2023, 1, 3))]  # another ~700m north
        events = mod.build_events(dets, eps_space_m=750, eps_time_days=1)
        self.assertEqual(len(events), 1)
        direct_dist = mod.haversine_m(21.0000, 72.0000, 21.0126, 72.0000)
        self.assertGreater(direct_dist, 750)  # confirms this required chaining, not a direct link


class TestFilterEvents(unittest.TestCase):
    def test_drops_singletons(self):
        events = [[0], [1, 2], [3], [4, 5, 6]]
        kept = mod.filter_events(events, min_detections=2)
        self.assertEqual(kept, [[1, 2], [4, 5, 6]])

    def test_default_threshold_is_two(self):
        events = [[0], [1, 2]]
        kept = mod.filter_events(events)
        self.assertEqual(kept, [[1, 2]])


class TestComputeStatus(unittest.TestCase):
    def test_closed_when_eps_time_days_have_passed(self):
        status = mod.compute_status(date(2023, 1, 1), date(2023, 1, 2), eps_time_days=1)
        self.assertEqual(status, "closed")

    def test_provisional_when_reference_date_equals_end_date(self):
        status = mod.compute_status(date(2023, 1, 1), date(2023, 1, 1), eps_time_days=1)
        self.assertEqual(status, "provisional")

    def test_provisional_just_short_of_eps_time(self):
        # eps_time=2: only 1 day has passed -> still provisional
        status = mod.compute_status(date(2023, 1, 1), date(2023, 1, 2), eps_time_days=2)
        self.assertEqual(status, "provisional")

    def test_closed_well_past_eps_time(self):
        status = mod.compute_status(date(2023, 1, 1), date(2023, 6, 1), eps_time_days=1)
        self.assertEqual(status, "closed")


class TestSummarizeEvent(unittest.TestCase):
    def test_fields_computed_correctly(self):
        dets = [
            make_det(21.000, 72.000, date(2023, 1, 1), frp=2.0, daynight="D"),
            make_det(21.001, 72.000, date(2023, 1, 2), frp=8.0, daynight="N"),
        ]
        row = mod.summarize_event("EVT000000", [0, 1], dets, reference_date=date(2023, 1, 5))
        self.assertEqual(row["event_id"], "EVT000000")
        self.assertEqual(row["start_date"], "2023-01-01")
        self.assertEqual(row["end_date"], "2023-01-02")
        self.assertEqual(row["duration_days"], 2)
        self.assertEqual(row["detection_count"], 2)
        self.assertAlmostEqual(row["mean_frp"], 5.0)
        self.assertEqual(row["max_frp"], 8.0)
        self.assertAlmostEqual(row["night_fraction"], 0.5)
        self.assertEqual(row["status"], "closed")
        self.assertGreater(row["spatial_extent_m"], 0)


class TestBuildEventTable(unittest.TestCase):
    def test_deterministic_event_ids_and_ordering(self):
        dets = [
            make_det(22.0, 70.0, date(2023, 3, 5)),
            make_det(22.0, 70.0, date(2023, 3, 6)),
            make_det(21.0, 72.0, date(2023, 1, 1)),
            make_det(21.0, 72.0, date(2023, 1, 2)),
        ]
        rows, n_raw = mod.build_event_table(dets, reference_date=date(2023, 3, 10))
        self.assertEqual(len(rows), 2)
        self.assertEqual(n_raw, 2)
        # earlier-starting event must get the lower event_id
        self.assertEqual(rows[0]["start_date"], "2023-01-01")
        self.assertEqual(rows[0]["event_id"], "EVT000000")
        self.assertEqual(rows[1]["event_id"], "EVT000001")

    def test_repeated_runs_produce_identical_output(self):
        dets = [make_det(21.0 + 0.001 * (i % 3), 72.0, date(2023, 1, 1 + (i // 3)))
                for i in range(9)]
        rows1, _ = mod.build_event_table(list(dets), reference_date=date(2023, 2, 1))
        rows2, _ = mod.build_event_table(list(dets), reference_date=date(2023, 2, 1))
        self.assertEqual(rows1, rows2)

    def test_singletons_excluded_from_table(self):
        dets = [
            make_det(21.0, 72.0, date(2023, 1, 1)),
            make_det(24.0, 68.5, date(2023, 6, 1)),  # far away, alone -> singleton
        ]
        rows, n_raw = mod.build_event_table(dets, reference_date=date(2023, 6, 5))
        self.assertEqual(n_raw, 2)
        self.assertEqual(len(rows), 0)


class TestCausalOnlineSafety(unittest.TestCase):
    """The core leakage-safety property claimed in the module docstring:
    truncating the input to only detections up to some day X must not
    change the grouping of any detection whose event is already CLOSED
    relative to X -- i.e. a retrospective run and an online run halted at
    day X agree on every closed event as of that day."""

    def test_prefix_run_matches_full_run_for_closed_events(self):
        eps_time = 1
        dets = [
            make_det(21.000, 72.000, date(2023, 1, 1)),
            make_det(21.000, 72.000, date(2023, 1, 2)),
            make_det(21.000, 72.000, date(2023, 1, 3)),
            # unrelated, later cluster far away in time (not space) --
            # must not affect the earlier, already-closed event
            make_det(21.000, 72.000, date(2023, 1, 10)),
            make_det(21.000, 72.000, date(2023, 1, 11)),
        ]
        full_events = mod.build_events(dets, eps_space_m=750, eps_time_days=eps_time)

        # "Online" run: only detections through day 2023-01-05 exist yet
        # (the first group's event ended 01-03 and is CLOSED by 01-05
        # under eps_time=1, since 5 days have passed).
        cutoff = date(2023, 1, 5)
        prefix_dets = [d for d in dets if d["date"] <= cutoff]
        prefix_events = mod.build_events(prefix_dets, eps_space_m=750, eps_time_days=eps_time)

        # indices 0,1,2 form one group in both the full and prefix runs
        full_group_of_0 = next(g for g in full_events if 0 in g)
        prefix_group_of_0 = next(g for g in prefix_events if 0 in g)
        self.assertEqual(set(full_group_of_0) & {0, 1, 2}, {0, 1, 2})
        self.assertEqual(set(prefix_group_of_0), {0, 1, 2})

    def test_status_transitions_from_provisional_to_closed(self):
        end_date = date(2023, 1, 3)
        self.assertEqual(mod.compute_status(end_date, date(2023, 1, 3), eps_time_days=1), "provisional")
        self.assertEqual(mod.compute_status(end_date, date(2023, 1, 4), eps_time_days=1), "closed")


if __name__ == "__main__":
    unittest.main()
