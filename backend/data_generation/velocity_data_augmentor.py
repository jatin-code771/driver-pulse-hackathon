import pandas as pd
import random
from pathlib import Path
from datetime import timedelta

# -------------------------------------------------
# PATH SETUP
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

TRIPS_PATH = BASE_DIR / "driver_pulse_hackathon_data" / "trips" / "trips.csv"
VELOCITY_PATH = BASE_DIR / "driver_pulse_hackathon_data" / "earnings" / "earnings_velocity_log.csv"

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

trips = pd.read_csv(TRIPS_PATH)

trips["start_dt"] = pd.to_datetime(trips["date"] + " " + trips["start_time"])
trips["end_dt"] = pd.to_datetime(trips["date"] + " " + trips["end_time"])

# sort trips globally
trips = trips.sort_values(["driver_id", "end_dt"])

velocity_rows = []
log_counter = 1

# -------------------------------------------------
# PROCESS DRIVER BY DRIVER
# -------------------------------------------------

for driver, df in trips.groupby("driver_id"):

    df = df.sort_values("end_dt")

    cumulative = 0
    trips_completed = 0
    first_trip_time = df.iloc[0]["start_dt"]

    previous_end = None

    for _, trip in df.iterrows():

        trips_completed += 1
        cumulative += trip["fare"]

        timestamp = trip["end_dt"]

        elapsed_hours = (timestamp - first_trip_time).total_seconds() / 3600

        elapsed_hours = max(elapsed_hours, 0.5)

        current_velocity = cumulative / elapsed_hours

        target_velocity = random.randint(170, 200)

        velocity_delta = current_velocity - target_velocity

        if velocity_delta > 40:
            forecast_status = "ahead"
        elif velocity_delta < -40:
            forecast_status = "at_risk"
        else:
            forecast_status = "on_track"

        velocity_rows.append({
            "log_id": f"VEL{log_counter:03}",
            "driver_id": driver,
            "date": trip["date"],
            "timestamp": timestamp.strftime("%H:%M:%S"),
            "cumulative_earnings": round(cumulative,2),
            "elapsed_hours": round(elapsed_hours,2),
            "current_velocity": round(current_velocity,2),
            "target_velocity": target_velocity,
            "velocity_delta": round(velocity_delta,2),
            "trips_completed": trips_completed,
            "forecast_status": forecast_status
        })

        log_counter += 1

        # -----------------------------------------
        # ADD RANDOM TELEMETRY BETWEEN TRIPS
        # -----------------------------------------

        if previous_end is not None:

            gap_minutes = (trip["start_dt"] - previous_end).total_seconds() / 60

            if gap_minutes > 10 and random.random() < 0.4:

                random_time = previous_end + timedelta(
                    minutes=random.randint(3, int(gap_minutes)-2)
                )

                elapsed_hours_rand = (random_time - first_trip_time).total_seconds()/3600

                velocity_rand = cumulative / max(elapsed_hours_rand,0.5)

                velocity_rows.append({
                    "log_id": f"VEL{log_counter:03}",
                    "driver_id": driver,
                    "date": trip["date"],
                    "timestamp": random_time.strftime("%H:%M:%S"),
                    "cumulative_earnings": round(cumulative,2),
                    "elapsed_hours": round(elapsed_hours_rand,2),
                    "current_velocity": round(velocity_rand,2),
                    "target_velocity": target_velocity,
                    "velocity_delta": round(velocity_rand-target_velocity,2),
                    "trips_completed": trips_completed,
                    "forecast_status": random.choice(["ahead","on_track","at_risk"])
                })

                log_counter += 1

        previous_end = trip["end_dt"]

# -------------------------------------------------
# FINAL DATASET
# -------------------------------------------------

velocity_df = pd.DataFrame(velocity_rows)

velocity_df = velocity_df.sort_values(["driver_id","timestamp"])

# -------------------------------------------------
# SAVE (OVERWRITE SAME FILE)
# -------------------------------------------------

velocity_df.to_csv(VELOCITY_PATH, index=False)

print("Velocity log regenerated successfully")
print("Total rows:", len(velocity_df))