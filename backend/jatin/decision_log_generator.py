"""
decision_log_generator.py
-------------------------
Generates the structured decision log (decision_log.csv) required by the
hackathon submission.  Each row traces a single detection back to the raw
sensor reading that triggered it.

Required schema:
  timestamp, signal_type, raw_value, threshold, event_label
"""

import pandas as pd
import os
from data_ingestion import load_all
from feature_engineering import build_all_features

# Thresholds (must stay in sync with feature_engineering.py)
HARSH_BRAKE_THRESHOLD = 2.5
HARSH_ACCEL_THRESHOLD = 2.5
MODERATE_EVENT_THRESHOLD = 1.5
AUDIO_SPIKE_DB = 85.0
SUSTAINED_NOISE_DB = 80.0
ARGUMENT_DB = 90.0


def generate_decision_log(fused_features: pd.DataFrame) -> pd.DataFrame:
    """Produce one row per detected event with raw value + threshold."""
    rows = []

    for _, r in fused_features.iterrows():
        ts = r['timestamp']
        mag = r.get('accel_magnitude_smooth', 0)
        delta = r.get('accel_delta', 0)
        audio_db = r.get('audio_level_db_fused', None)

        # --- Accelerometer events ---
        if r.get('is_harsh_brake', 0) == 1:
            rows.append({
                'timestamp': ts,
                'signal_type': 'ACCELEROMETER',
                'raw_value': f'{mag:.1f}g',
                'threshold': f'{HARSH_BRAKE_THRESHOLD}g',
                'event_label': 'HARSH_BRAKING',
            })
        elif r.get('is_harsh_accel', 0) == 1:
            rows.append({
                'timestamp': ts,
                'signal_type': 'ACCELEROMETER',
                'raw_value': f'{mag:.1f}g',
                'threshold': f'{HARSH_ACCEL_THRESHOLD}g',
                'event_label': 'HARSH_ACCELERATION',
            })
        elif r.get('is_moderate_event', 0) == 1:
            rows.append({
                'timestamp': ts,
                'signal_type': 'ACCELEROMETER',
                'raw_value': f'{mag:.1f}g',
                'threshold': f'{MODERATE_EVENT_THRESHOLD}g',
                'event_label': 'MODERATE_BRAKING',
            })

        # --- Audio events ---
        if pd.notna(audio_db) and audio_db > 0:
            if r.get('is_argument_signal', 0) == 1:
                rows.append({
                    'timestamp': ts,
                    'signal_type': 'AUDIO',
                    'raw_value': f'{audio_db:.0f}dB',
                    'threshold': f'{ARGUMENT_DB}dB',
                    'event_label': 'ARGUMENT_DETECTED',
                })
            elif r.get('is_sustained_noise', 0) == 1:
                rows.append({
                    'timestamp': ts,
                    'signal_type': 'AUDIO',
                    'raw_value': f'{audio_db:.0f}dB',
                    'threshold': f'{SUSTAINED_NOISE_DB}dB',
                    'event_label': 'SUSTAINED_NOISE',
                })
            elif r.get('is_noise_spike', 0) == 1:
                rows.append({
                    'timestamp': ts,
                    'signal_type': 'AUDIO',
                    'raw_value': f'{audio_db:.0f}dB',
                    'threshold': f'{AUDIO_SPIKE_DB}dB',
                    'event_label': 'NOISE_SPIKE',
                })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values('timestamp').reset_index(drop=True)
    return df


if __name__ == '__main__':
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent

    data_dir = BASE_DIR / "driver_pulse_hackathon_data"
    output_dir = BASE_DIR / "processed_outputs"
    os.makedirs(output_dir, exist_ok=True)

    datasets = load_all(data_dir)
    features = build_all_features(datasets)

    log = generate_decision_log(features['fused_features'])
    path = os.path.join(output_dir, 'decision_log.csv')
    log.to_csv(path, index=False)
    print(f"Generated {len(log)} decision log entries → {path}")
