import numpy as np

MIN_HOURS_FOR_VELOCITY = 0.25


def compute_velocity_and_status(row):

    earnings = row["cumulative_earnings"]
    elapsed = row["elapsed_hours"]
    trips = row["trips_completed"]
    target = row["target_earnings"]
    avg_hourly = row["avg_earnings_per_hour"]

    remaining_hours = row["remaining_shift_hours"]

    # -------------------------
    # Current Velocity
    # -------------------------

    if elapsed <= MIN_HOURS_FOR_VELOCITY or trips == 0:
        current_velocity = avg_hourly
        cold_start = True
    else:
        current_velocity = earnings / elapsed
        cold_start = False

    if np.isnan(current_velocity):
        current_velocity = 0

    max_velocity = 3 * avg_hourly

    if current_velocity > max_velocity:
        current_velocity = max_velocity

    # -------------------------
    # Remaining earnings
    # -------------------------

    remaining_earnings = max(target - earnings, 0)

    # -------------------------
    # Target velocity
    # -------------------------

    if remaining_hours > 0:
        target_velocity = remaining_earnings / remaining_hours
    else:
        target_velocity = 0

    if target_velocity > 0:
        pacing_ratio = current_velocity / target_velocity
    else:
        pacing_ratio = np.nan

    pacing_ratio = np.clip(pacing_ratio, 0, 5)

    # -------------------------
    # Status Logic
    # -------------------------

    progress_percent = (earnings / target) * 100 if target > 0 else 0

    if progress_percent >= 100:
        status = "ahead"

    else:

        if cold_start:
            status = "cold_start"

        elif pacing_ratio >= 1.25:
            status = "ahead"

        elif pacing_ratio >= 0.85:
            status = "on_track"

        elif pacing_ratio >= 0.50:
            status = "at_risk"

        else:
            status = "critical"

    return current_velocity, status