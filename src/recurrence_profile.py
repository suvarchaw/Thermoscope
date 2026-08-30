"""
Recurrence Profile module — ThermoScope temporal-recurrence annotation.

A Recurrence Profile describes the TEMPORAL RECURRENCE BEHAVIOR of a
spatial DBSCAN cluster. It is explicitly NOT a source-type classification
and NOT a final industrial/persistent-source label. It says nothing about
what a cluster physically is — only how its detections are distributed
over time.

Three fields make up a Recurrence Profile, computed from exactly three
existing continuous metrics (unique_dates, active_span_days,
top3_days_share) and nothing else:

  - recurrence_strength ("Strong" / "Moderate" / "Limited"), from
    unique_dates, thresholds anchored to natural gaps observed in the
    current 60-cluster dataset:
        Strong:   unique_dates >= 143   (gap: 143 -> 196)
        Moderate: 73 <= unique_dates < 143  (gap: 56 -> 73)
        Limited:  unique_dates < 73
    (A third gap, 196 -> 295, sits inside the Strong tier and does not
    add a boundary of its own.)

  - short_window_recurrence (bool): active_span_days <= 80. An
    independent temporal-shape annotation, not a strength tier — it
    exists to distinguish short-duration-but-internally-recurring
    activity (e.g. cluster 42) from genuinely sparse recurrence. It does
    NOT override recurrence_strength.

  - burst_concentrated (bool): top3_days_share > 0.5. A caution/quality
    flag only — it does not create another tier and must not be combined
    into recurrence_strength.

These functions take only the three specific numeric inputs they need
(never a full row), so it is structurally impossible for OSM context,
FIRMS `type`, or any other field to influence a Recurrence Profile.
"""

STRONG_MIN_UNIQUE_DATES = 143
MODERATE_MIN_UNIQUE_DATES = 73
SHORT_WINDOW_MAX_SPAN_DAYS = 80
BURST_CONCENTRATED_MIN_SHARE = 0.5


def recurrence_strength(unique_dates):
    """Primary recurrence-strength tier from unique_dates alone."""
    if unique_dates >= STRONG_MIN_UNIQUE_DATES:
        return "Strong"
    if unique_dates >= MODERATE_MIN_UNIQUE_DATES:
        return "Moderate"
    return "Limited"


def short_window_recurrence(active_span_days):
    """Independent temporal-shape flag from active_span_days alone."""
    return active_span_days <= SHORT_WINDOW_MAX_SPAN_DAYS


def burst_concentrated(top3_days_share):
    """Caution/quality flag from top3_days_share alone. Not a tier."""
    return top3_days_share > BURST_CONCENTRATED_MIN_SHARE


def compute_recurrence_profile(unique_dates, active_span_days, top3_days_share):
    """Compute the three Recurrence Profile fields from exactly these three
    inputs. No other field (OSM context, FIRMS type, FRP, day/night, etc.)
    is read or used."""
    return {
        "recurrence_strength": recurrence_strength(unique_dates),
        "short_window_recurrence": short_window_recurrence(active_span_days),
        "burst_concentrated": burst_concentrated(top3_days_share),
    }
