import pandas as pd
import numpy as np
from math import ceil
from pathlib import Path


# =========================================================
# CORE COMPUTATION ENGINE
# =========================================================

def generate_realtime_driver_predictions(df, save_path):

    save_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # Fix datatypes
    # -------------------------------

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["shift_end_time"] = pd.to_datetime(df["shift_end_time"], errors="coerce")

    df = df.sort_values(["driver_id", "timestamp"])

    results = []

    for _, row in df.iterrows():

        driver_id = row["driver_id"]
        driver_name = row["name"]

        earnings = row.get("cumulative_earnings", 0)
        elapsed = row.get("elapsed_hours", 0)
        trips = row.get("trips_completed", 0)
        target = row.get("target_earnings", 0)
        avg_hourly = row.get("avg_earnings_per_hour", 0)

        timestamp = row["timestamp"]
        shift_end = row["shift_end_time"]

        # -------------------------------
        # Handle NaNs
        # -------------------------------

        earnings = 0 if pd.isna(earnings) else earnings
        elapsed = 0 if pd.isna(elapsed) else elapsed
        trips = 0 if pd.isna(trips) else trips
        target = 0 if pd.isna(target) else target
        avg_hourly = 0 if pd.isna(avg_hourly) else avg_hourly

        # =========================================================
        # Remaining Shift Time
        # =========================================================

        if pd.isna(timestamp) or pd.isna(shift_end):
            remaining_hours = 0
        else:
            remaining_hours = (shift_end - timestamp).total_seconds() / 3600

        remaining_hours = max(remaining_hours, 0)

        # =========================================================
        # CURRENT EARNING VELOCITY
        # =========================================================

        if elapsed <= 0.25 or trips == 0:
            current_velocity = avg_hourly
        else:
            current_velocity = earnings / elapsed

        if pd.isna(current_velocity):
            current_velocity = 0

        max_velocity = 3 * avg_hourly

        if current_velocity > max_velocity:
            current_velocity = max_velocity
        # =========================================================
        # Remaining Earnings
        # =========================================================

        remaining_earnings = max(target - earnings, 0)

        # =========================================================
        # TARGET VELOCITY
        # =========================================================

        if remaining_hours > 0:
            target_velocity = remaining_earnings / remaining_hours
        else:
            target_velocity = 0

        if pd.isna(target_velocity):
            target_velocity = 0

        # =========================================================
        # Average Earnings Per Trip
        # =========================================================

        if trips > 0:
            avg_trip = earnings / trips
        else:
            avg_trip = avg_hourly / 2

        if pd.isna(avg_trip) or avg_trip <= 0:
            avg_trip = 0

        # =========================================================
        # Trips Needed to Reach Goal
        # =========================================================

        if avg_trip > 0:
            trips_to_goal = ceil(remaining_earnings / avg_trip)
        else:
            trips_to_goal = 0

        # =========================================================
        # Estimated Time to Goal
        # =========================================================

        if current_velocity > 0:
            time_to_goal = remaining_earnings / current_velocity
        else:
            time_to_goal = np.inf

        if np.isinf(time_to_goal):
            time_to_goal = 0

        # =========================================================
        # Goal Status
        # =========================================================

        if earnings >= target:
            goal_status = "achieved"
        else:
            goal_status = "in_progress"

        # =========================================================
        # Forecast Logic
        # =========================================================

        if goal_status == "achieved":

            forecast = "ahead"

        else:

            if target_velocity == 0:
                ratio = 1
            else:
                ratio = current_velocity / target_velocity

            if ratio >= 1.2:
                forecast = "ahead"

            elif ratio >= 0.9:
                forecast = "on_track"

            else:
                forecast = "at_risk"

        # =========================================================
        # Goal Progress
        # =========================================================

        if target > 0:
            progress_percent = min((earnings / target) * 100, 100)
        else:
            progress_percent = 0

        # =========================================================
        # Store Result
        # =========================================================

        results.append({

            "driver_id": driver_id,
            "driver_name": driver_name,

            "timestamp": timestamp,

            "cumulative_earnings": earnings,

            "current_velocity": round(current_velocity, 2),
            "target_velocity": round(target_velocity, 2),

            "velocity_delta": round(current_velocity - target_velocity, 2),

            "remaining_earnings": round(remaining_earnings, 2),
            "remaining_shift_hours": round(remaining_hours, 2),

            "trips_completed": trips,
            "trips_to_goal": trips_to_goal,

            "estimated_time_to_goal_hours": round(time_to_goal, 2),

            "goal_progress_percent": round(progress_percent, 2),

            "goal_status": goal_status,
            "forecast_status": forecast
        })

    output_df = pd.DataFrame(results)

    output_file = save_path / "realtime_driver_predictions.csv"

    output_df.to_csv(output_file, index=False)

    print("✅ Prediction dataset generated")
    print("📂 File location:", output_file)

    return output_df


# =========================================================
# MAIN EXECUTION
# =========================================================

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