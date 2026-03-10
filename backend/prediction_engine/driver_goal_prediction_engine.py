import pandas as pd
import numpy as np
from math import ceil
from pathlib import Path

MIN_HOURS_FOR_VELOCITY = 0.25   # cold start threshold


def generate_realtime_driver_predictions(df, save_path):

    save_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------
    # Fix datatypes
    # ---------------------------------------------------

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["shift_end_time"] = pd.to_datetime(df["shift_end_time"], errors="coerce")

    df = df.sort_values(["driver_id", "timestamp"])

    results = []

    for _, row in df.iterrows():

        driver_id = row["driver_id"]
        driver_name = row.get("name", "unknown")

        earnings = row.get("cumulative_earnings", 0)
        elapsed = row.get("elapsed_hours", 0)
        trips = row.get("trips_completed", 0)
        target = row.get("target_earnings", 0)
        avg_hourly = row.get("avg_earnings_per_hour", 0)

        timestamp = row["timestamp"]
        shift_end = row["shift_end_time"]

        # ---------------------------------------------------
        # Handle NaNs
        # ---------------------------------------------------

        earnings = 0 if pd.isna(earnings) else earnings
        elapsed = 0 if pd.isna(elapsed) else elapsed
        trips = 0 if pd.isna(trips) else trips
        target = 0 if pd.isna(target) else target
        avg_hourly = 0 if pd.isna(avg_hourly) else avg_hourly

        # ---------------------------------------------------
        # Remaining shift time
        # ---------------------------------------------------

        if pd.isna(timestamp) or pd.isna(shift_end):
            remaining_hours = 0
        else:
            remaining_hours = (shift_end - timestamp).total_seconds() / 3600

        remaining_hours = max(remaining_hours, 0)

        # ---------------------------------------------------
        # Current velocity (cold start protection)
        # ---------------------------------------------------

        if elapsed <= MIN_HOURS_FOR_VELOCITY or trips == 0:
            current_velocity = avg_hourly
            cold_start = True
        else:
            current_velocity = earnings / elapsed
            cold_start = False

        if pd.isna(current_velocity):
            current_velocity = 0

        # cap unrealistic spikes
        max_velocity = 3 * avg_hourly
        if current_velocity > max_velocity:
            current_velocity = max_velocity

        # ---------------------------------------------------
        # Remaining earnings
        # ---------------------------------------------------

        remaining_earnings = max(target - earnings, 0)

        # ---------------------------------------------------
        # Target velocity
        # ---------------------------------------------------

        if remaining_hours > 0:
            target_velocity = remaining_earnings / remaining_hours
        else:
            target_velocity = 0

        if pd.isna(target_velocity):
            target_velocity = 0

        # ---------------------------------------------------
        # Pacing ratio (from friend's logic)
        # ---------------------------------------------------

        if target_velocity > 0:
            pacing_ratio = current_velocity / target_velocity
        else:
            pacing_ratio = np.nan

        pacing_ratio = np.clip(pacing_ratio, 0, 5)

        # ---------------------------------------------------
        # Goal completion %
        # ---------------------------------------------------

        if target > 0:
            progress_percent = min((earnings / target) * 100, 100)
        else:
            progress_percent = 0

        # ---------------------------------------------------
        # Average earnings per trip
        # ---------------------------------------------------

        if trips > 0:
            avg_trip = earnings / trips
        else:
            avg_trip = avg_hourly / 2

        if pd.isna(avg_trip) or avg_trip <= 0:
            avg_trip = 0

        # ---------------------------------------------------
        # Trips needed to reach goal
        # ---------------------------------------------------

        if avg_trip > 0:
            trips_to_goal = ceil(remaining_earnings / avg_trip)
        else:
            trips_to_goal = 0

        # ---------------------------------------------------
        # Estimated time to goal
        # ---------------------------------------------------

        if current_velocity > 0:
            time_to_goal = remaining_earnings / current_velocity
        else:
            time_to_goal = np.inf

        if np.isinf(time_to_goal):
            time_to_goal = 0

        # ---------------------------------------------------
        # Forecast logic
        # ---------------------------------------------------

        if progress_percent >= 100:
            forecast = "ahead"

        else:

            if cold_start:
                forecast = "cold_start"

            elif pacing_ratio >= 1.25:
                forecast = "ahead"

            elif pacing_ratio >= 0.85:
                forecast = "on_track"

            elif pacing_ratio >= 0.50:
                forecast = "at_risk"

            else:
                forecast = "critical"

        # ---------------------------------------------------
        # Projected earnings at shift end
        # ---------------------------------------------------

        projected_earnings = earnings + current_velocity * remaining_hours

        if target > 0:
            projected_completion_pct = min((projected_earnings / target) * 100, 200)
        else:
            projected_completion_pct = 0

        # ---------------------------------------------------
        # Store results
        # ---------------------------------------------------

        results.append({

            "driver_id": driver_id,
            "driver_name": driver_name,

            "timestamp": timestamp,

            "cumulative_earnings": round(earnings,2),

            "current_velocity": round(current_velocity,2),
            "target_velocity": round(target_velocity,2),

            "velocity_delta": round(current_velocity - target_velocity,2),

            "remaining_earnings": round(remaining_earnings,2),
            "remaining_shift_hours": round(remaining_hours,2),

            "pacing_ratio": round(pacing_ratio,2) if not pd.isna(pacing_ratio) else None,

            "trips_completed": trips,
            "trips_to_goal": trips_to_goal,

            "estimated_time_to_goal_hours": round(time_to_goal,2),

            "goal_progress_percent": round(progress_percent,2),

            "projected_earnings": round(projected_earnings,2),
            "projected_completion_percent": round(projected_completion_pct,2),

            "forecast_status": forecast
        })

    output_df = pd.DataFrame(results)

    output_file = save_path / "realtime_driver_predictions.csv"

    output_df.to_csv(output_file, index=False)

    print("✅ Prediction dataset generated")
    print("📂 File location:", output_file)

    return output_df


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent

    input_path = BASE_DIR / "processed_outputs" / "final_driver_timeline.csv"
    output_path = BASE_DIR / "driver_outputs"

    print("Reading dataset from:", input_path)

    df = pd.read_csv(input_path)

    output = generate_realtime_driver_predictions(
        df,
        save_path=output_path
    )

    print(output.head())