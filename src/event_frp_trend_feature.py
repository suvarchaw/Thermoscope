"""
FRP trend slope -- a new, additive, non-leaky event-level feature
measuring whether an event's fire radiative power is growing, shrinking,
or flat over its lifetime. Investigated (read-only) in the prior
milestone and found to have low correlation (max |r|=0.297) with every
existing label-generating field, with a real, physically sensible class
pattern (Forest_Wildfire trending strongly upward, Industrial flat).
This module implements it as reviewed, tested code and re-verifies that
finding against the real data before approving the feature for use.

DEFINITION
-------------------------------------------------------------------------
For an event's member detections, group by calendar day, take the mean
FRP within each day (so a day with several detections does not
outweigh a day with one), then fit an ordinary-least-squares line of
per-day-mean-FRP against day INDEX (0, 1, 2, ... in chronological order
-- not calendar date, so the slope's units are "FRP change per day of
the event," independent of which calendar dates the event happened to
span). `frp_trend_slope` is that line's slope.

MISSING-VALUE BEHAVIOR, DEFINED EXPLICITLY
-------------------------------------------------------------------------
A slope requires at least 2 distinct days. Events spanning only 1 unique
day (still the majority of events -- see event_construction.py) cannot
have a trend computed. For these, `frp_trend_slope` is left BLANK (not
0.0 -- 0.0 is a real, meaningful "flat trend" value for a multi-day
event and must not be confused with "not computable"), and
`frp_trend_computable` is `False`. `n_unique_days` is always populated
(1 for non-computable events) so a consumer can see why. This is a
deliberate design choice, not an oversight: filling with 0.0 would
silently conflate "this event's intensity was flat over multiple days"
with "this event only happened on one day, so trend is undefined" --
two different physical situations.

CAUSAL / EVENT-CONSTRUCTION UNCHANGED
-------------------------------------------------------------------------
Event grouping reuses event_construction.build_events/filter_events and
event_behavior_features.sorted_event_groups/verify_alignment UNCHANGED
(imported, not reimplemented) -- alignment with the existing,
already-persisted event table is verified (0 mismatches required), not
assumed. The feature itself uses only detections that are already
members of the (causally-constructed) event -- no information from
after the event's current end_date is used. For a `status=provisional`
event (see event_construction.py), the trend is computed over
whatever days have been observed so far, exactly like every other
event-level statistic already in the pipeline (e.g. duration_days) --
subject to revision if the event later extends, which is the correct,
already-established behavior for online/NRT operation, not a new
concern this feature introduces.

OUTPUT: data/processed/event_frp_trend_features.csv (new, additive).
Does NOT modify gujarat_thermal_events.csv, gujarat_event_evidence.csv,
gujarat_event_silver_labels.csv, or gujarat_cluster_*.csv -- all
read-only where reused. Does NOT retrain any model, change silver
labels, or change event construction.
"""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import event_construction as ec
from event_behavior_features import sorted_event_groups, verify_alignment

SILVER_LABELS_CSV = Path("data/processed/gujarat_event_silver_labels.csv")
OUTPUT_CSV = Path("data/processed/event_frp_trend_features.csv")

MIN_UNIQUE_DAYS_FOR_TREND = 2

# Fields already established (build_event_silver_labels.py /
# train_source_classifier_lightgbm.py) as label-generating or otherwise
# forbidden -- the leakage audit below checks the new feature against
# every one of them, not a subset chosen after the fact.
LEAKAGE_CHECK_FIELDS = ["mean_frp", "night_fraction", "duration_days", "detection_count", "max_frp"]
LEAKAGE_CORRELATION_THRESHOLD = 0.8  # same conservative threshold already used in this project


def ols_slope(xs, ys):
    n = len(xs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0  # den==0 cannot occur for n>=2 distinct integer indices


def compute_frp_trend(members, detections):
    """Returns (frp_trend_slope_or_None, computable_bool, n_unique_days)."""
    pts = [detections[i] for i in members]
    by_day = defaultdict(list)
    for p in pts:
        by_day[p["date"]].append(p["frp"])
    days_sorted = sorted(by_day.keys())
    n_days = len(days_sorted)

    if n_days < MIN_UNIQUE_DAYS_FOR_TREND:
        return None, False, n_days

    xs = list(range(n_days))  # day INDEX, not calendar date -- see module docstring
    ys = [statistics.mean(by_day[d]) for d in days_sorted]
    slope = ols_slope(xs, ys)
    return slope, True, n_days


def build_feature_table():
    """Reuses event_construction.load_detections/build_events/filter_events
    and event_behavior_features.sorted_event_groups UNCHANGED. Verifies
    exact alignment with the persisted event table before trusting the
    event_id join (0 mismatches required, not assumed)."""
    detections = ec.load_detections()
    sorted_groups = sorted_event_groups(detections)
    verify_alignment(sorted_groups)

    with open(SILVER_LABELS_CSV, newline="") as f:
        silver = list(csv.DictReader(f))
    if len(silver) != len(sorted_groups):
        raise ValueError("Silver-label row count does not match regenerated event count.")

    rows = []
    for silver_row, (start, end, clat, clon, members) in zip(silver, sorted_groups):
        slope, computable, n_days = compute_frp_trend(members, detections)
        rows.append({
            "event_id": silver_row["event_id"],
            "frp_trend_slope": "" if slope is None else slope,
            "frp_trend_computable": computable,
            "n_unique_days": n_days,
        })
    return rows


def write_csv(rows, path=OUTPUT_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["event_id", "frp_trend_slope", "frp_trend_computable", "n_unique_days"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Leakage audit -- run against the real, joined data, not assumed from
# the investigation milestone's numbers (they are re-derived here, and
# should match).
# ---------------------------------------------------------------------

def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sx = sum((a - mx) ** 2 for a in xs) ** 0.5
    sy = sum((b - my) ** 2 for b in ys) ** 0.5
    return cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def run_leakage_audit(feature_rows, silver_rows):
    """Computes correlation of frp_trend_slope (computable rows only)
    against every LEAKAGE_CHECK_FIELDS field, restricted to the 4 trained
    classes (matching how the feature would actually be used). Returns
    (correlations_dict, is_safe_bool)."""
    silver_by_id = {r["event_id"]: r for r in silver_rows}
    trained_classes = {"Industrial", "Gas_Flare", "Crop_Residue", "Forest_Wildfire"}

    computable_ids = [
        r["event_id"] for r in feature_rows
        if r["frp_trend_computable"] and silver_by_id[r["event_id"]]["silver_label"] in trained_classes
    ]
    slope_by_id = {r["event_id"]: float(r["frp_trend_slope"]) for r in feature_rows if r["event_id"] in computable_ids}

    slopes = [slope_by_id[eid] for eid in computable_ids]
    correlations = {}
    for field in LEAKAGE_CHECK_FIELDS:
        vals = [float(silver_by_id[eid][field]) for eid in computable_ids]
        correlations[field] = pearson(slopes, vals)

    is_safe = all(abs(r) <= LEAKAGE_CORRELATION_THRESHOLD for r in correlations.values())
    return correlations, is_safe, len(computable_ids)


def class_conditional_means(feature_rows, silver_rows):
    silver_by_id = {r["event_id"]: r for r in silver_rows}
    trained_classes = ["Industrial", "Gas_Flare", "Crop_Residue", "Forest_Wildfire"]
    by_class = defaultdict(list)
    for r in feature_rows:
        if not r["frp_trend_computable"]:
            continue
        label = silver_by_id[r["event_id"]]["silver_label"]
        if label in trained_classes:
            by_class[label].append(float(r["frp_trend_slope"]))

    summary = {}
    for cls in trained_classes:
        vals = by_class.get(cls, [])
        if vals:
            summary[cls] = {"n": len(vals), "mean": statistics.mean(vals), "median": statistics.median(vals)}
    return summary


def main():
    print("Building FRP trend slope feature table (reuses event_construction "
          "and event_behavior_features unchanged; verifies alignment)...\n")
    rows = build_feature_table()
    write_csv(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")

    n_computable = sum(1 for r in rows if r["frp_trend_computable"])
    print(f"Computable (>=2 unique days): {n_computable} ({n_computable/len(rows)*100:.1f}%)")
    print(f"Not computable (single-day events): {len(rows)-n_computable} "
          f"({(len(rows)-n_computable)/len(rows)*100:.1f}%)\n")

    with open(SILVER_LABELS_CSV, newline="") as f:
        silver = list(csv.DictReader(f))

    print("=" * 70)
    print("LEAKAGE AUDIT (recomputed against real data, not assumed)")
    print("=" * 70)
    correlations, is_safe, n_checked = run_leakage_audit(rows, silver)
    print(f"Checked on {n_checked} computable events in the 4 trained classes:")
    for field, r in correlations.items():
        flag = "OK" if abs(r) <= LEAKAGE_CORRELATION_THRESHOLD else "EXCEEDS THRESHOLD"
        print(f"  corr(frp_trend_slope, {field}) = {r:.3f}  [{flag}]")
    print(f"\nAll correlations <= {LEAKAGE_CORRELATION_THRESHOLD} threshold: {is_safe}")
    print(f"DECISION: {'SAFE -- approved for use as a candidate feature' if is_safe else 'NOT SAFE -- excluded'}\n")

    print("=" * 70)
    print("CLASS-CONDITIONAL MEAN/MEDIAN (supporting evidence of real signal, not causal)")
    print("=" * 70)
    summary = class_conditional_means(rows, silver)
    for cls, s in summary.items():
        print(f"  {cls}: n={s['n']} mean={s['mean']:.3f} median={s['median']:.3f}")

    return rows, correlations, is_safe


if __name__ == "__main__":
    main()
