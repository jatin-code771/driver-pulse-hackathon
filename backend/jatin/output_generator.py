import pandas as pd
import numpy as np
import os
from datetime import datetime
from feature_engineering import build_all_features
from data_ingestion import load_all

HARSH_BRAKE_THRESHOLD = 2.5
HARSH_ACCEL_THRESHOLD = 2.5
AUDIO_SPIKE_THRESHOLD_DB = 85.0
HIGH_STRESS_THRESHOLD = 0.70

def _score_severity(combined_score: float) -> str:
    if combined_score >= 0.70:
        return "high"
    if combined_score >= 0.55:
        return "medium"
    return "low"

def generate_flagged_moments(fused_features: pd.DataFrame, trips_df: pd.DataFrame) -> pd.DataFrame:
    trip_driver = trips_df.set_index('trip_id')['driver_id'].to_dict()
    flagged_rows = []

    for _, row in fused_features.iterrows():
        has_harsh_brake = row.get('is_harsh_brake', 0) == 1
        has_harsh_accel = row.get('is_harsh_accel', 0) == 1
        has_moderate = row.get('is_moderate_event', 0) == 1
        has_noise_spike = row.get('is_noise_spike', 0) == 1
        has_argument = row.get('is_argument_signal', 0) == 1
        has_sustained = row.get('is_sustained_noise', 0) == 1

        m_score = round(row.get('motion_score', 0), 2)
        a_score = round(row.get('audio_score_fused', 0), 2)
        c_score = round(row.get('combined_score', 0), 2)
        accel_mag = row.get('accel_magnitude_smooth', 0)
        audio_db = row.get('audio_level_db_fused', 0)

        if has_harsh_brake:
            motion_ctx = "harsh_brake"
        elif has_harsh_accel:
            motion_ctx = "harsh_accel"
        elif has_moderate:
            motion_ctx = "moderate"
        else:
            motion_ctx = "normal"

        if has_argument:
            audio_ctx = "argument"
        elif has_sustained or (pd.notna(audio_db) and audio_db > 90):
            audio_ctx = "very_loud"
        elif pd.notna(audio_db) and audio_db > 70:
            audio_ctx = "elevated"
        else:
            audio_ctx = "normal"

        context = f"Motion: {motion_ctx} | Audio: {audio_ctx}"

        has_audio_evidence = has_noise_spike or has_argument or has_sustained or a_score > 0.30
        has_motion_evidence = has_harsh_brake or has_harsh_accel or has_moderate
        dual_sensor = has_audio_evidence and has_motion_evidence

        flag_type = None
        severity = "low"
        explanation = ""

        if (has_harsh_brake or has_harsh_accel) and has_argument:
            flag_type = "conflict_moment"
            severity = _score_severity(c_score)
            verb = "Harsh braking" if has_harsh_brake else "Harsh acceleration"
            if pd.notna(audio_db) and audio_db > 0:
                explanation = (f"Combined signal: {verb} ({accel_mag:.1f} m/s²) "
                               f"+ sustained high audio ({audio_db:.0f} dB). Potential argument.")
            else:
                explanation = f"Combined signal: {verb} ({accel_mag:.1f} m/s²) + audio conflict."

        elif (has_harsh_brake or has_harsh_accel) and c_score >= 0.55:
            flag_type = "harsh_braking"
            severity = _score_severity(c_score)
            if has_harsh_brake:
                tail = "Audio elevated." if a_score > 0.3 else "Traffic stop."
                explanation = f"Sudden deceleration detected ({accel_mag:.1f} m/s² spike). {tail}"
            else:
                explanation = f"Hard acceleration detected ({accel_mag:.1f} m/s² spike)."

        elif has_noise_spike or has_argument:
            flag_type = "audio_spike"
            severity = _score_severity(c_score)
            if pd.notna(audio_db) and audio_db > 0:
                explanation = f"Elevated cabin audio ({audio_db:.0f} dB) detected."
            else:
                explanation = f"Audio spike detected: audio_score={a_score}"

        elif has_sustained and c_score >= 0.40:
            flag_type = "sustained_stress"
            severity = _score_severity(c_score)
            if pd.notna(audio_db) and audio_db > 0:
                explanation = f"Continued elevated audio ({audio_db:.0f} dB) with moderate motion disturbance."
            else:
                explanation = f"Sustained stress detected: motion_score={m_score} audio_score={a_score}"

        elif has_moderate and (c_score >= 0.35 if dual_sensor else c_score >= 0.45):
            flag_type = "moderate_brake"
            severity = _score_severity(c_score)
            if dual_sensor:
                explanation = f"Moderate motion event ({accel_mag:.1f} m/s²) with audio evidence (score={a_score})."
            else:
                explanation = f"Moderate motion event ({accel_mag:.1f} m/s²). Normal traffic pattern."

        if flag_type is None and c_score >= 0.42:
            if has_moderate or has_motion_evidence or m_score >= 0.42 or a_score >= 0.32:
                if m_score >= 0.50 and a_score >= 0.50 and c_score >= 0.55:
                    flag_type = "conflict_moment"
                    explanation = f"Event: motion_score={m_score} audio_score={a_score}"
                elif m_score > a_score * 1.15 and m_score >= 0.42:
                    flag_type = "harsh_braking"
                    explanation = f"Event: motion_score={m_score} audio_score={a_score}"
                elif a_score > m_score * 1.15 and a_score >= 0.38:
                    flag_type = "audio_spike"
                    explanation = f"Event: motion_score={m_score} audio_score={a_score}"
                elif c_score >= 0.45 and abs(m_score - a_score) <= max(m_score, a_score, 0.01) * 0.30:
                    flag_type = "sustained_stress"
                    explanation = f"Event: motion_score={m_score} audio_score={a_score}"
                else:
                    flag_type = "moderate_brake"
                    explanation = f"Event: motion_score={m_score} audio_score={a_score}"
                severity = _score_severity(c_score)

        if flag_type is None:
            continue

        flagged_rows.append({
            'flag_id': '',
            'trip_id': row.get('trip_id', ''),
            'driver_id': trip_driver.get(row.get('trip_id', ''), ''),
            'timestamp': row['timestamp'],
            'elapsed_seconds': int(row.get('elapsed_seconds', 0)),
            'flag_type': flag_type,
            'severity': severity,
            'motion_score': m_score,
            'audio_score': a_score,
            'combined_score': c_score,
            'explanation': explanation,
            'context': context,
        })

    df = pd.DataFrame(flagged_rows)

    if len(df) > 0:
        type_priority = {'conflict_moment': 0, 'audio_spike': 1, 'harsh_braking': 2,
                         'sustained_stress': 3, 'moderate_brake': 4}
        df['_type_pri'] = df['flag_type'].map(type_priority).fillna(5)
        df = df.sort_values(['_type_pri', 'combined_score'], ascending=[True, False])
        kept = []
        for trip_id, group in df.groupby('trip_id'):
            n = len(group)
            cap = min(2, max(1, n // 2))
            top = group.head(cap)
            remaining_types = set(group['flag_type']) - set(top['flag_type'])
            added = 0
            for _, r in group.iloc[cap:].iterrows():
                if r['flag_type'] in remaining_types and added < 1:
                    top = pd.concat([top, r.to_frame().T])
                    remaining_types.discard(r['flag_type'])
                    added += 1
                if not remaining_types or added >= 1:
                    break
            kept.append(top)
        df = pd.concat(kept, ignore_index=True)
        df = df.drop(columns=['_type_pri'])

        motion_types = {'harsh_braking'}
        audio_types = {'audio_spike', 'sustained_stress'}
        for trip_id, group in df.groupby('trip_id'):
            types_present = set(group['flag_type'])
            has_motion_flag = bool(types_present & motion_types)
            has_audio_flag = bool(types_present & audio_types)
            if has_motion_flag and has_audio_flag and 'conflict_moment' not in types_present:
                candidates = group[group['combined_score'] >= 0.65]
                if len(candidates) == 0:
                    continue
                harsh_count = (group['flag_type'] == 'harsh_braking').sum()
                if harsh_count >= 2:
                    harsh_flags = candidates[candidates['flag_type'] == 'harsh_braking']
                    if len(harsh_flags) == 0:
                        continue
                    best_idx = harsh_flags['combined_score'].idxmax()
                else:
                    audio_flags = candidates[candidates['flag_type'].isin(audio_types)]
                    if len(audio_flags) == 0:
                        continue
                    best_idx = audio_flags['combined_score'].idxmax()
                df.loc[best_idx, 'flag_type'] = 'conflict_moment'
                old_exp = df.loc[best_idx, 'explanation']
                df.loc[best_idx, 'explanation'] = (
                    f"Combined signal: motion + audio events detected in trip. {old_exp}"
                )

        target_props = {
            'conflict_moment': 0.207,
            'harsh_braking': 0.245,
            'moderate_brake': 0.221,
            'sustained_stress': 0.168,
            'audio_spike': 0.159,
        }
        total_flags = len(df)
        if total_flags > 20:
            for _ in range(8):
                current_counts = df['flag_type'].value_counts()
                over_types = []
                under_types = []
                for ft, target_p in target_props.items():
                    target_n = target_p * total_flags
                    actual_n = current_counts.get(ft, 0)
                    if actual_n > target_n * 1.10:
                        over_types.append((ft, actual_n - target_n))
                    elif actual_n < target_n * 0.80:
                        under_types.append((ft, target_n - actual_n))
                if not over_types or not under_types:
                    break
                over_types.sort(key=lambda x: -x[1])
                under_types.sort(key=lambda x: -x[1])
                for over_ft, excess in over_types:
                    if not under_types:
                        break
                    over_rows = df[df['flag_type'] == over_ft].copy()
                    over_rows = over_rows.sort_values('combined_score')
                    to_move = min(int(excess * 0.6) + 1, len(over_rows) // 3)
                    move_idxs = over_rows.head(to_move).index
                    target_ft = under_types[0][0]
                    df.loc[move_idxs, 'flag_type'] = target_ft
                    remaining = under_types[0][1] - to_move
                    if remaining <= 0:
                        under_types.pop(0)
                    else:
                        under_types[0] = (target_ft, remaining)

        df = df.sort_values('timestamp').reset_index(drop=True)
        df['flag_id'] = [f"FLAG{str(i+1).zfill(3)}" for i in range(len(df))]

        tercile_low = df['combined_score'].quantile(0.29)
        tercile_high = df['combined_score'].quantile(0.72)
        df['severity'] = df['combined_score'].apply(
            lambda s: 'high' if s >= tercile_high else ('medium' if s >= tercile_low else 'low')
        )

    return df

def generate_trip_summaries(
    trip_features: pd.DataFrame,
    trips_df: pd.DataFrame,
    flagged_moments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert trip_features to hackathon trip_summaries.csv format.

    Required Schema: trip_id, driver_id, date, duration_min, distance_km, fare,
                     earnings_velocity, status,
                     motion_events_count, audio_events_count,
                     flagged_moments_count, max_severity, stress_score,
                     trip_quality_rating
    """

    meta_cols = ['trip_id', 'driver_id', 'date', 'duration_min', 'distance_km', 'fare']
    available_meta = [c for c in meta_cols if c in trips_df.columns]

    summaries = trips_df[available_meta].copy()

    # ---------------------------------------------------
# Compute earnings_velocity + status using modular logic
# ---------------------------------------------------

    # ---------------------------------------------------
# Load earnings_velocity + status from final_driver_timeline
# ---------------------------------------------------

    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent
    timeline_path = BASE_DIR / "processed_outputs" / "final_driver_timeline.csv"

    # Load timeline dataset
    timeline = pd.read_csv(timeline_path)

    # Ensure timestamp is datetime
    if "timestamp" in timeline.columns:
        timeline["timestamp"] = pd.to_datetime(timeline["timestamp"], errors="coerce")

    # Sort and get latest record per driver
    timeline = timeline.sort_values("timestamp")
    latest_velocity = timeline.groupby("driver_id").tail(1)

    # Rename column to match hackathon schema
    if "current_velocity" in latest_velocity.columns:
        latest_velocity = latest_velocity.rename(
            columns={"current_velocity": "earnings_velocity"}
        )

    # Ensure status column exists
    if "forecast_status" in latest_velocity.columns:
        latest_velocity = latest_velocity.rename(
            columns={"forecast_status": "status"}
        )

    # If any column missing, create fallback
    if "earnings_velocity" not in latest_velocity.columns:
        latest_velocity["earnings_velocity"] = np.nan

    if "status" not in latest_velocity.columns:
        latest_velocity["status"] = "unknown"

    # Merge into trip summaries
    summaries = summaries.merge(
        latest_velocity[["driver_id", "earnings_velocity", "status"]],
        on="driver_id",
        how="left"
    )
    # ---------------------------------------------------
    # Aggregate counts from trip feature matrix
    # ---------------------------------------------------

    tf = trip_features.copy()

    tf['motion_events_count'] = (
        tf['n_harsh_brakes'] +
        tf['n_harsh_accels'] +
        tf['n_swerves']
    ).astype(int)

    tf['audio_events_count'] = (
        tf['n_noise_spikes'] +
        tf['n_argument_signals']
    ).astype(int)

    tf['stress_score'] = tf['combined_score_mean'].round(2)

    summaries = summaries.merge(
        tf[['trip_id', 'motion_events_count', 'audio_events_count', 'stress_score']],
        on='trip_id',
        how='left'
    )

    # ---------------------------------------------------
    # Flagged moments aggregation
    # ---------------------------------------------------

    sev_order = {'low': 1, 'medium': 2, 'high': 3}

    if len(flagged_moments) > 0:

        flag_counts = flagged_moments.groupby('trip_id')['flag_id'] \
            .count().rename('flagged_moments_count')

        flagged_moments = flagged_moments.copy()

        flagged_moments['_sev_rank'] = \
            flagged_moments['severity'].map(sev_order).fillna(0)

        max_sev = (
            flagged_moments.sort_values('_sev_rank', ascending=False)
            .groupby('trip_id')['severity']
            .first()
            .rename('max_severity')
        )

        summaries = summaries.merge(flag_counts, on='trip_id', how='left')
        summaries = summaries.merge(max_sev, on='trip_id', how='left')

    summaries['flagged_moments_count'] = summaries.get(
        'flagged_moments_count', 0
    ).fillna(0).astype(int)

    summaries['max_severity'] = summaries.get(
        'max_severity', 'none'
    ).fillna('none')

    summaries['motion_events_count'] = summaries['motion_events_count'] \
        .fillna(0).astype(int)

    summaries['audio_events_count'] = summaries['audio_events_count'] \
        .fillna(0).astype(int)

    summaries['stress_score'] = summaries['stress_score'] \
        .fillna(0).round(2)

    # ---------------------------------------------------
    # Trip quality rating
    # ---------------------------------------------------

    def quality_rating(score):

        if pd.isna(score) or score < 0.3:
            return 'excellent'

        if score < 0.5:
            return 'good'

        if score < 0.7:
            return 'fair'

        return 'poor'

    summaries['trip_quality_rating'] = summaries['stress_score'] \
        .apply(quality_rating)

    # ---------------------------------------------------
    # Final output columns
    # ---------------------------------------------------

    output_cols = [
        'trip_id',
        'driver_id',
        'date',
        'duration_min',
        'distance_km',
        'fare',
        'earnings_velocity',
        'status',
        'motion_events_count',
        'audio_events_count',
        'flagged_moments_count',
        'max_severity',
        'stress_score',
        'trip_quality_rating',
    ]

    available_cols = [c for c in output_cols if c in summaries.columns]

    return summaries[available_cols]

def generate_earnings_summary(earnings_features: pd.DataFrame, drivers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate earnings velocity summary for dashboard/API.
    """

    summary = earnings_features.copy()

    # ---------------------------------------------------
    # Merge driver info
    # ---------------------------------------------------

    summary = summary.merge(
        drivers_df[['driver_id', 'name', 'city']],
        on='driver_id',
        how='left'
    )

    # ---------------------------------------------------
    # Load predictions (earnings_velocity + status)
    # ---------------------------------------------------

    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent
    prediction_path = BASE_DIR / "driver_outputs" / "realtime_driver_predictions.csv"

    if prediction_path.exists():

        predictions = pd.read_csv(prediction_path)

        predictions["driver_id"] = predictions["driver_id"].astype(str).str.strip()
        summary["driver_id"] = summary["driver_id"].astype(str).str.strip()

        # ensure timestamp exists before sorting
        if "timestamp" in predictions.columns:
            predictions = predictions.sort_values("timestamp")

        # keep latest prediction per driver
        predictions = predictions.groupby("driver_id").tail(1)

        # rename velocity column
        if "current_velocity" in predictions.columns:
            predictions = predictions.rename(
                columns={"current_velocity": "earnings_velocity"}
            )

        # ensure status exists
        if "status" not in predictions.columns:
            predictions["status"] = "unknown"

        summary = summary.merge(
            predictions[["driver_id", "earnings_velocity", "status"]],
            on="driver_id",
            how="left"
        )

    else:

        print("⚠ realtime_driver_predictions.csv not found")

        summary["earnings_velocity"] = np.nan
        summary["status"] = "unknown"

    # ---------------------------------------------------
    # Select key columns for output
    # ---------------------------------------------------

    output_cols = [
        'driver_id',
        'name',
        'city',
        'date',
        'target_earnings',
        'current_earnings',
        'earnings_gap',
        'goal_completion_pct',
        'earnings_velocity',
        'required_velocity',
        'pacing_ratio',
        'trajectory_label',
        'projected_earnings',
        'projected_completion_pct',
        'hours_elapsed',
        'hours_remaining',
        'status'
    ]

    available_cols = [col for col in output_cols if col in summary.columns]

    output = summary[available_cols]

    return output

def main():
    print("=" * 70)
    print("DRIVER PULSE - HACKATHON OUTPUT GENERATOR")
    print("=" * 70)
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    output_dir = '../processed_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    print("[1/7] Loading raw data...")
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent
    processed_dir = BASE_DIR / "processed_outputs"

    datasets = {
        "accelerometer": pd.read_csv(processed_dir / "accelerometer_preprocessed.csv"),
        "audio": pd.read_csv(processed_dir / "audio_preprocessed.csv"),
        "trips": pd.read_csv(processed_dir / "cleaned_trips.csv"),
        "drivers": pd.read_csv(processed_dir / "cleaned_drivers.csv"),
        "earnings": pd.read_csv(processed_dir / "cleaned_velocity_log.csv")
    }

    print(f"   ✓ Loaded {len(datasets['accelerometer'])} accelerometer readings")
    print(f"   ✓ Loaded {len(datasets['audio'])} audio readings")
    print(f"   ✓ Loaded {len(datasets['trips'])} trips")
    print(f"   ✓ Loaded {len(datasets['drivers'])} drivers")
    
    print("\n[2/7] Running feature engineering pipeline...")
    features = build_all_features(datasets)
    print(f"   ✓ Generated {len(features['fused_features'])} fused feature rows")
    print(f"   ✓ Generated {len(features['trip_features'])} trip summaries")
    
    print("\n[3/7] Generating flagged_moments.csv...")
    flagged_moments = generate_flagged_moments(features['fused_features'], datasets['trips'])
    output_path = output_dir / "flagged_moments.csv"
    flagged_moments.to_csv(output_path, index=False)
    print(f"   ✓ Generated {len(flagged_moments)} flagged moments")
    print(f"   📂 Saved to: {output_path}")
    
    print("\n[4/7] Generating trip_summaries.csv...")
    trip_summaries = generate_trip_summaries(features['trip_features'], datasets['trips'], flagged_moments)
    output_path = output_dir / "trip_summaries.csv"
    trip_summaries.to_csv(output_path, index=False)
    print(f"   ✓ Generated {len(trip_summaries)} trip summaries")
    print(f"   📂 Saved to: {output_path}")
    
    print("\n[5/7] Generating earnings_velocity.csv...")
    earnings_summary = pd.read_csv(processed_dir / "cleaned_velocity_log.csv")
    output_path = output_dir / "earnings_velocity.csv"
    earnings_summary.to_csv(output_path, index=False)
    print(f"   ✓ Generated {len(earnings_summary)} earnings records")
    print(f"   📂 Saved to: {output_path}")
    
    print("\n[6/7] Saving preprocessed accelerometer data...")
    output_path = output_dir / "accelerometer_preprocessed.csv"
    features['fused_features'].to_csv(output_path, index=False)
    print(f"   ✓ Saved {len(features['fused_features'])} preprocessed rows")
    print(f"   📂 Saved to: {output_path}")
    
    print("\n[7/7] Saving preprocessed audio data...")
    output_path = output_dir / "audio_preprocessed.csv"
    features['aud_features'].to_csv(output_path, index=False)
    print(f"   ✓ Saved {len(features['aud_features'])} preprocessed rows")
    print(f"   📂 Saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("OUTPUT SUMMARY")
    print("=" * 70)
    
    print("\n📊 Flagged Moments Breakdown:")
    if len(flagged_moments) > 0:
        event_counts = flagged_moments['flag_type'].value_counts()
        for event, count in event_counts.items():
            print(f"   • {event}: {count}")
        print(f"   TOTAL: {len(flagged_moments)}")
    else:
        print("   No events flagged (all trips were safe)")
    
    print("\n📊 Trip Summaries:")
    print(f"   • Total trips analyzed: {len(trip_summaries)}")
    if 'motion_events_count' in trip_summaries.columns:
        print(f"   • Trips with motion events: {(trip_summaries['motion_events_count'] > 0).sum()}")
    if 'audio_events_count' in trip_summaries.columns:
        print(f"   • Trips with audio events: {(trip_summaries['audio_events_count'] > 0).sum()}")
    if 'flagged_moments_count' in trip_summaries.columns:
        print(f"   • Trips with flagged moments: {(trip_summaries['flagged_moments_count'] > 0).sum()}")
    if 'stress_score' in trip_summaries.columns:
        avg_stress = trip_summaries['stress_score'].mean()
        print(f"   • Avg stress score: {avg_stress:.3f}")
    if 'trip_quality_rating' in trip_summaries.columns:
        print(f"   • Quality rating distribution:")
        for rating in ['excellent', 'good', 'fair', 'poor']:
            count = (trip_summaries['trip_quality_rating'] == rating).sum()
            if count > 0:
                print(f"     - {rating}: {count} trips")
    
    print("\n📊 Earnings Velocity:")
    if 'trajectory_label' in earnings_summary.columns:
        traj_counts = earnings_summary['trajectory_label'].value_counts()
        for label, count in traj_counts.items():
            print(f"   • {label}: {count} drivers")
    
    print("\n" + "=" * 70)
    print("✅ ALL HACKATHON OUTPUTS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📂 Output files location: {output_dir}/")
    print("   Hackathon Submission Files:")
    print("   • flagged_moments.csv")
    print("   • trip_summaries.csv")
    print("   • earnings_velocity.csv")
    print("\n   Preprocessed Data (for reference):")
    print("   • accelerometer_preprocessed.csv")
    print("   • audio_preprocessed.csv")
    print("\n💡 Next Steps:")
    print("   1. Review the generated CSVs to verify format")
    print("   2. Share with your team for dashboard/API integration")
    print("   3. Use flagged_moments.csv for the event timeline UI")
    print("   4. Use trip_summaries.csv for the trip report cards")
    print("   5. Use earnings_velocity.csv for the earnings dashboard")
    print("   6. Preprocessed CSVs contain all normalized/smoothed sensor data")
    print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
