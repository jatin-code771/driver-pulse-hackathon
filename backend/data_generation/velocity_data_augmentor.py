import pandas as pd
import numpy as np
import uuid
from datetime import timedelta
from pathlib import Path

# ---------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------

# backend directory
BASE_DIR = Path(__file__).resolve().parent.parent

# earnings velocity dataset
input_path = BASE_DIR / "driver_pulse_hackathon_data" / "earnings" / "earnings_velocity_log.csv"

print("Reading dataset from:", input_path)

df = pd.read_csv(input_path)

# ---------------------------------------------------
# TIMESTAMP FIX
# ---------------------------------------------------

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# ---------------------------------------------------
# GENERATE ADDITIONAL VELOCITY LOGS
# ---------------------------------------------------

new_rows = []

for driver in df["driver_id"].unique():

    driver_df = df[df["driver_id"] == driver].sort_values("timestamp")

    # required minimum datapoints per driver
    target_points = 3
    if driver == "DRV004":
        target_points = 6

    current_points = len(driver_df)

    if current_points >= target_points:
        continue

    last_row = driver_df.iloc[-1]

    last_time = last_row["timestamp"]
    cumulative = last_row["cumulative_earnings"]
    velocity = last_row["current_velocity"]
    trips = last_row["trips_completed"]

    for _ in range(target_points - current_points):

        # simulate next timestamp
        last_time = last_time + timedelta(minutes=np.random.randint(5, 20))

        # simulate velocity fluctuation
        change = np.random.randint(-80, 80)
        velocity = max(30, velocity + change)

        target_velocity = last_row["target_velocity"]

        delta = velocity - target_velocity

        # simulate earnings growth
        cumulative += np.random.randint(80, 250)

        # simulate additional trip
        trips += 1

        # determine driver performance status
        status = "on_track"

        if delta > 40:
            status = "ahead"

        elif delta < -40:
            status = "at_risk"

        new_rows.append({
            "log_id": "VEL" + str(uuid.uuid4())[:6],
            "driver_id": driver,
            "date": last_time.date(),
            "timestamp": last_time,
            "cumulative_earnings": cumulative,
            "elapsed_hours": last_row["elapsed_hours"] + np.random.uniform(0.3, 0.8),
            "current_velocity": round(velocity, 2),
            "target_velocity": target_velocity,
            "velocity_delta": round(delta, 2),
            "trips_completed": trips,
            "forecast_status": status
        })

# ---------------------------------------------------
# MERGE ORIGINAL + GENERATED DATA
# ---------------------------------------------------

extra_df = pd.DataFrame(new_rows)

final_df = pd.concat([df, extra_df])

final_df = final_df.sort_values(["driver_id", "timestamp"])

# ---------------------------------------------------
# SAVE UPDATED DATASET
# ---------------------------------------------------

final_df.to_csv(input_path, index=False)

print("✅ Velocity dataset expanded successfully")
print("📁 Saved to:", input_path)