import csv
import os
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_data")

CSV_FILES = {
    "flagged_moments_latest": os.path.join(OUTPUT_DIR, "flagged_moments_latest.csv"),
    "flagged_moments": os.path.join(OUTPUT_DIR, "flagged_moments.csv"),
    "accelerometer_data": os.path.join(OUTPUT_DIR, "accelerometer_data.csv"),
    "trips": os.path.join(OUTPUT_DIR, "trips.csv"),
    "realtime_driver_predictions": os.path.join(OUTPUT_DIR, "realtime_driver_predictions.csv"),
}


def read_csv_file(filepath):
    if not os.path.isfile(filepath):
        return None

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if any(row.values())]


@app.route("/api/flagged-moments")
def flagged_moments():
    rows = read_csv_file(CSV_FILES["flagged_moments_latest"])
    driver_id = request.args.get("driver_id")

    if driver_id:
        rows = [r for r in rows if r.get("driver_id") == driver_id]

    return jsonify(rows)


@app.route("/api/accelerometer")
def accelerometer():
    rows = read_csv_file(CSV_FILES["accelerometer_data"])
    trip_id = request.args.get("trip_id")

    if trip_id:
        rows = [r for r in rows if r.get("trip_id") == trip_id]

    return jsonify(rows)


@app.route("/api/trips")
def trips():
    rows = read_csv_file(CSV_FILES["trips"])
    driver_id = request.args.get("driver_id")

    if driver_id:
        rows = [r for r in rows if r.get("driver_id") == driver_id]

    return jsonify(rows)


@app.route("/api/driver-predictions")
def predictions():
    rows = read_csv_file(CSV_FILES["realtime_driver_predictions"])
    driver_id = request.args.get("driver_id")

    if driver_id:
        rows = [r for r in rows if r.get("driver_id") == driver_id]

    return jsonify(rows)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)