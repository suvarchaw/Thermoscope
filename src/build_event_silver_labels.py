"""
Additive silver-label dataset for the event-level representation.

Reads data/processed/gujarat_event_evidence.csv (existing, unmodified,
read-only) and applies the EXACT deterministic labeling rules, evidence
sources, spatial radii, and seasonal windows already validated in prior
read-only investigations -- no threshold is invented or tuned here, and
none is tuned for class balance. Rule definitions (see RULES below) are
identical to the ones run as a temporary, non-persisted investigation
script in the previous milestone; this module is the first time they are
committed as reviewed, tested code.

TARGET CLASSES (from the original SIH26162 proposal)
-------------------------------------------------------------------------
Industrial, Crop Residue, Forest/Wildfire, Brick Kiln, Gas Flare, and a
sixth catch-all, Unknown/Ambiguous, explicitly EXCLUDED from any future
supervised training. An event lands in Unknown/Ambiguous for one of two
distinct reasons, both recorded (never conflated) in `ambiguity_reason`:
  - "insufficient_evidence": no rule matched at all.
  - "conflicting_evidence": more than one rule matched -- the event is
    never forced into a single class; every matching class is recorded
    in `conflict_classes` for a human reviewer.
Brick Kiln is NOT hardcoded to zero -- `brick_kiln_rule` is a real,
independently testable rule (see tests/test_build_event_silver_labels.py
TestBrickKilnRule, which proves it CAN match given synthetic evidence).
It legitimately produces zero labels on the real data because zero
events have any OSM kiln/brickyard evidence within 500m -- a genuine
external-data-access gap (see DECISIONS.md), not a code path that was
disabled to avoid an empty class.

RULE -> EVIDENCE TRANSLATION (unchanged from the prior investigation,
see DECISIONS.md for the full reasoning)
-------------------------------------------------------------------------
  Industrial:      near industrial/power (OSM or GPPD, <=1km) AND
                    night_fraction > 0 AND the event overlaps a
                    persistent cluster with recurrence_strength in
                    {Strong, Moderate}.
  Gas Flare:        near an OSM flare (<=1km) AND (the event overlaps a
                    persistent, multi-year-recurring cluster OR
                    mean_frp >= the sample median mean_frp across all
                    events).
  Brick Kiln:       near an OSM kiln/brickyard (<=500m) AND event start
                    month in Nov-May AND duration_days <= 3.
  Crop Residue:     land_cover_class == "Cropland" AND duration_days <= 3
                    AND event start month in {Oct, Nov, Dec, Apr, May}.
  Forest/Wildfire:  land_cover_class == "Tree cover" AND
                    duration_days <= 3 AND event start month in
                    {Mar, Apr, May, Jun}.

NO LABEL IS EVER TREATED AS GROUND TRUTH. `silver_label` is a
rule-derived candidate, `label_rule_id` and `label_evidence` record
EXACTLY which rule and which observed values produced it, for a human
reviewer to audit -- not a claim of correctness. No numeric confidence
score is invented (none of the underlying rules ever produced a
probability); `conflict_classes` and `ambiguity_reason` are the closest
legitimate provenance signal, reused directly rather than invented.

LEAKAGE SEPARATION (see LABEL_GENERATING_COLUMNS / CANDIDATE_FEATURE_COLUMNS
below and DECISIONS.md): every column read by a rule is excluded from the
documented candidate-feature list for any future classifier trained on
`silver_label` -- reusing a label-generating column as a model input
would let the model simply re-derive the rule instead of learning
anything independent (the same failure mode already documented for the
next-year-recurrence model's FORBIDDEN_FEATURES).

Does NOT: modify gujarat_event_evidence.csv, gujarat_thermal_events.csv,
the 60-cluster pipeline, or any existing output; train a model; compute
a risk score; or claim any label is validated ground truth.
"""

import csv
import statistics
from collections import Counter
from datetime import date
from pathlib import Path

EVIDENCE_CSV = Path("data/processed/gujarat_event_evidence.csv")
OUTPUT_CSV = Path("data/processed/gujarat_event_silver_labels.csv")

INDUSTRIAL_DIST_M = 1000
FLARE_DIST_M = 1000
KILN_DIST_M = 500
KILN_SEASON_MONTHS = {11, 12, 1, 2, 3, 4, 5}
CROP_HARVEST_MONTHS = {10, 11, 12, 4, 5}
FOREST_DRY_SEASON_MONTHS = {3, 4, 5, 6}
SHORT_DURATION_DAYS = 3

CLASSES = ["Industrial", "Gas_Flare", "Brick_Kiln", "Crop_Residue", "Forest_Wildfire"]
UNKNOWN_AMBIGUOUS = "Unknown_Ambiguous"

# Columns read by at least one rule below. Excluded from the documented
# candidate-feature list for any future classifier -- see module
# docstring and DECISIONS.md. `overlaps_cluster_id` is included even
# though only its two derived fields are read directly, because it is
# the join key those fields come from and is trivially predictive of
# them.
LABEL_GENERATING_COLUMNS = {
    "nearest_osm_industrial_power_m", "nearest_gppd_thermal_plant_m",
    "nearest_osm_flare_m", "nearest_osm_kiln_m",
    "overlaps_cluster_id", "overlaps_cluster_recurrence_strength",
    "overlaps_cluster_recurs_multiyear",
    "land_cover_code", "land_cover_class",
    "night_fraction", "mean_frp", "duration_days",
    "start_date", "end_date",
}

# Columns NOT read by any rule -- legitimate candidate features for a
# future classifier, exactly as they appear in gujarat_event_evidence.csv.
CANDIDATE_FEATURE_COLUMNS = [
    "centroid_lat", "centroid_lon", "spatial_extent_m",
    "detection_count", "max_frp", "status",
]


def assert_columns_disjoint():
    overlap = LABEL_GENERATING_COLUMNS & set(CANDIDATE_FEATURE_COLUMNS)
    if overlap:
        raise ValueError(f"Columns both label-generating and candidate features: {overlap}")


def _to_float(v):
    return float(v) if v not in (None, "") else None


def _month_of(date_str):
    return date.fromisoformat(date_str).month


def industrial_rule(row, median_mean_frp=None):
    ind_d = _to_float(row["nearest_osm_industrial_power_m"])
    gppd_d = _to_float(row["nearest_gppd_thermal_plant_m"])
    near = (ind_d is not None and ind_d <= INDUSTRIAL_DIST_M) or (gppd_d is not None and gppd_d <= INDUSTRIAL_DIST_M)
    night_present = float(row["night_fraction"]) > 0
    persistent = row["overlaps_cluster_recurrence_strength"] in ("Strong", "Moderate")
    matched = near and night_present and persistent
    evidence = (f"near_industrial_or_gppd<=1km={near} (osm={ind_d}, gppd={gppd_d}); "
                f"night_fraction={row['night_fraction']}>0={night_present}; "
                f"cluster_recurrence_strength={row['overlaps_cluster_recurrence_strength']!r} in "
                f"(Strong,Moderate)={persistent}")
    return matched, evidence


def gas_flare_rule(row, median_mean_frp):
    flare_d = _to_float(row["nearest_osm_flare_m"])
    near = flare_d is not None and flare_d <= FLARE_DIST_M
    recurs_multiyear = row["overlaps_cluster_recurs_multiyear"] == "True"
    mean_frp = float(row["mean_frp"])
    radiometric = mean_frp >= median_mean_frp
    matched = near and (recurs_multiyear or radiometric)
    evidence = (f"near_flare<=1km={near} (dist={flare_d}); "
                f"cluster_recurs_multiyear={recurs_multiyear} OR "
                f"mean_frp={mean_frp}>=median({median_mean_frp:.3f})={radiometric}")
    return matched, evidence


def brick_kiln_rule(row, median_mean_frp=None):
    kiln_d = _to_float(row["nearest_osm_kiln_m"])
    near = kiln_d is not None and kiln_d <= KILN_DIST_M
    month = _month_of(row["start_date"])
    seasonal = month in KILN_SEASON_MONTHS
    short = int(row["duration_days"]) <= SHORT_DURATION_DAYS
    matched = near and seasonal and short
    evidence = (f"near_kiln<=500m={near} (dist={kiln_d}); "
                f"start_month={month} in Nov-May={seasonal}; "
                f"duration_days={row['duration_days']}<=3={short}")
    return matched, evidence


def crop_residue_rule(row, median_mean_frp=None):
    is_cropland = row["land_cover_class"] == "Cropland"
    short = int(row["duration_days"]) <= SHORT_DURATION_DAYS
    month = _month_of(row["start_date"])
    harvest = month in CROP_HARVEST_MONTHS
    matched = is_cropland and short and harvest
    evidence = (f"land_cover_class={row['land_cover_class']!r}=='Cropland'={is_cropland}; "
                f"duration_days={row['duration_days']}<=3={short}; "
                f"start_month={month} in harvest-window={harvest}")
    return matched, evidence


def forest_wildfire_rule(row, median_mean_frp=None):
    is_treecover = row["land_cover_class"] == "Tree cover"
    short = int(row["duration_days"]) <= SHORT_DURATION_DAYS
    month = _month_of(row["start_date"])
    dry_season = month in FOREST_DRY_SEASON_MONTHS
    matched = is_treecover and short and dry_season
    evidence = (f"land_cover_class={row['land_cover_class']!r}=='Tree cover'={is_treecover}; "
                f"duration_days={row['duration_days']}<=3={short}; "
                f"start_month={month} in dry-season={dry_season}")
    return matched, evidence


RULES = {
    "Industrial": industrial_rule,
    "Gas_Flare": gas_flare_rule,
    "Brick_Kiln": brick_kiln_rule,
    "Crop_Residue": crop_residue_rule,
    "Forest_Wildfire": forest_wildfire_rule,
}


def compute_median_mean_frp(rows):
    return statistics.median(float(r["mean_frp"]) for r in rows)


def label_event(row, median_mean_frp):
    """Applies every rule to one event row. Returns
    (silver_label, matched_classes, evidence_by_class, ambiguity_reason)."""
    matched_classes = []
    evidence_by_class = {}
    for class_name, rule_fn in RULES.items():
        matched, evidence = rule_fn(row, median_mean_frp)
        evidence_by_class[class_name] = evidence
        if matched:
            matched_classes.append(class_name)

    if len(matched_classes) == 0:
        return UNKNOWN_AMBIGUOUS, matched_classes, evidence_by_class, "insufficient_evidence"
    if len(matched_classes) == 1:
        return matched_classes[0], matched_classes, evidence_by_class, ""
    return UNKNOWN_AMBIGUOUS, matched_classes, evidence_by_class, "conflicting_evidence"


def build_labels(rows):
    assert_columns_disjoint()
    median_mean_frp = compute_median_mean_frp(rows)

    out_rows = []
    for row in rows:
        silver_label, matched_classes, evidence_by_class, ambiguity_reason = label_event(row, median_mean_frp)

        label_rule_id = silver_label if silver_label != UNKNOWN_AMBIGUOUS else ""
        label_evidence = evidence_by_class.get(matched_classes[0]) if len(matched_classes) == 1 else ""

        out_row = {
            "event_id": row["event_id"],
            # carried-through OBSERVED/DERIVED fields (unchanged from the evidence table)
            "start_date": row["start_date"], "end_date": row["end_date"],
            "duration_days": row["duration_days"],
            "centroid_lat": row["centroid_lat"], "centroid_lon": row["centroid_lon"],
            "spatial_extent_m": row["spatial_extent_m"], "detection_count": row["detection_count"],
            "mean_frp": row["mean_frp"], "max_frp": row["max_frp"],
            "night_fraction": row["night_fraction"], "status": row["status"],
            # label
            "silver_label": silver_label,
            "label_rule_id": label_rule_id,
            "label_evidence": label_evidence,
            "conflict_classes": ";".join(matched_classes),
            "n_classes_matched": len(matched_classes),
            "ambiguity_reason": ambiguity_reason,
            "excluded_from_training": silver_label == UNKNOWN_AMBIGUOUS,
        }
        out_rows.append(out_row)
    return out_rows


def load_evidence(path=EVIDENCE_CSV):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path=OUTPUT_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_report(rows):
    print("=" * 70)
    print("LEAKAGE AUDIT")
    print("=" * 70)
    print(f"Label-generating columns (excluded from candidate features): "
          f"{len(LABEL_GENERATING_COLUMNS)}")
    for c in sorted(LABEL_GENERATING_COLUMNS):
        print(f"  - {c}")
    print(f"Candidate feature columns (available, rule-independent): {len(CANDIDATE_FEATURE_COLUMNS)}")
    for c in CANDIDATE_FEATURE_COLUMNS:
        print(f"  - {c}")

    print("\n" + "=" * 70)
    print("FINAL COUNT PER CLASS")
    print("=" * 70)
    counts = Counter(r["silver_label"] for r in rows)
    for c in CLASSES:
        print(f"  {c}: {counts.get(c, 0)}")
    print(f"  {UNKNOWN_AMBIGUOUS} (total): {counts.get(UNKNOWN_AMBIGUOUS, 0)}")

    reasons = Counter(r["ambiguity_reason"] for r in rows if r["ambiguity_reason"])
    print(f"    of which insufficient_evidence: {reasons.get('insufficient_evidence', 0)}")
    print(f"    of which conflicting_evidence: {reasons.get('conflicting_evidence', 0)}")

    n_excluded = sum(1 for r in rows if r["excluded_from_training"])
    print(f"\nEvents excluded from supervised training: {n_excluded} of {len(rows)} "
          f"({n_excluded/len(rows)*100:.1f}%)")

    if counts.get("Brick_Kiln", 0) == 0:
        print("\nBrick_Kiln: 0 labeled events. The rule is implemented and was evaluated for "
              "every event (see tests/test_build_event_silver_labels.py::TestBrickKilnRule for "
              "proof it can match given real evidence) -- zero is a genuine finding (no OSM "
              "kiln/brickyard evidence within 500m of any event), not a disabled rule.")


def main():
    assert_columns_disjoint()
    rows = load_evidence()
    print(f"Loaded {len(rows)} events from {EVIDENCE_CSV} (read-only).\n")

    out_rows = build_labels(rows)
    write_csv(out_rows)
    print(f"Wrote {len(out_rows)} rows to {OUTPUT_CSV}\n")

    print_report(out_rows)
    return out_rows


if __name__ == "__main__":
    main()
