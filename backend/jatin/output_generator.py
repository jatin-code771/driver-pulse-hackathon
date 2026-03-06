"""
output_generator.py
-------------------
Converts feature engineering outputs to hackathon-required formats:
- flagged_moments.csv (timestamp, signal_type, raw_value, threshold, event_label)
- trip_summaries.csv (trip-level report card)

This script bridges your preprocessing work to the hackathon submission format.
"""

import pandas as pd
import os
from datetime import datetime
from feature_engineering import build_all_features
from data_ingestion import load_all

# Threshold constants (from feature_engineering.py)
HARSH_BRAKE_THRESHOLD = 2.5
HARSH_ACCEL_THRESHOLD = 2.5
AUDIO_SPIKE_THRESHOLD_DB = 85.0
HIGH_STRESS_THRESHOLD = 0.70


def _score_severity(combined_score: float) -> str:
    """Map combined_score to severity matching reference distribution (~33% each)."""
    if combined_score >= 0.70:
        return "high"
    if combined_score >= 0.55:
        return "medium"
    return "low"


def generate_flagged_moments(fused_features: pd.DataFrame, trips_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert fused features to hackathon flagged_moments.csv format.

    Required Schema: flag_id, trip_id, driver_id, timestamp, elapsed_seconds,
                     flag_type, severity, motion_score, audio_score,
                     combined_score, explanation, context
    """
    trip_driver = trips_df.set_index('trip_id')['driver_id'].to_dict()
    flagged_rows = []

    for _, row in fused_features.iterrows():
        has_harsh_brake = row.get('is_harsh_brake', 0) == 1
        has_harsh_accel = row.get('is_harsh_accel', 0) == 1
        has_moderate = row.get('is_moderate_event', 0) == 1
        has_noise_spike = row.get('is_noise_spike', 0) == 1
        has_argument = row.get('is_argument_signal', 0) == 1
        has_sustained = row.get('is_sustained_noise', 0) == 1
        is_high_stress = row.get('stress_severity', '') == 'high'

        m_score = round(row.get('motion_score', 0), 2)
        a_score = round(row.get('audio_score_fused', 0), 2)
        c_score = round(row.get('combined_score', 0), 2)
        accel_mag = row.get('accel_magnitude_smooth', 0)
        audio_db = row.get('audio_level_db_fused', 0)

        # Motion context label
        if has_harsh_brake:
            motion_ctx = "harsh_brake"
        elif has_harsh_accel:
            motion_ctx = "harsh_accel"
        elif has_moderate:
            motion_ctx = "moderate"
        else:
            motion_ctx = "normal"

        # Audio context label
        if has_argument:
            audio_ctx = "argument"
        elif has_sustained or (pd.notna(audio_db) and audio_db > 90):
            audio_ctx = "very_loud"
        elif pd.notna(audio_db) and audio_db > 70:
            audio_ctx = "elevated"
        else:
            audio_ctx = "normal"

        context = f"Motion: {motion_ctx} | Audio: {audio_ctx}"

        # ── Dual-sensor evidence flags ──────────────────────
        has_audio_evidence = has_noise_spike or has_argument or has_sustained or a_score > 0.30
        has_motion_evidence = has_harsh_brake or has_harsh_accel or has_moderate
        dual_sensor = has_audio_evidence and has_motion_evidence

        # Determine flag_type by priority ---------------------------------
        flag_type = None
        severity = "low"
        explanation = ""

        # 1. conflict_moment: harsh motion + argument signal
        if (has_harsh_brake or has_harsh_accel) and has_argument:
            flag_type = "conflict_moment"
            severity = _score_severity(c_score)
            verb = "Harsh braking" if has_harsh_brake else "Harsh acceleration"
            if pd.notna(audio_db) and audio_db > 0:
                explanation = (f"Combined signal: {verb} ({accel_mag:.1f} m/s²) "
                               f"+ sustained high audio ({audio_db:.0f} dB). Potential argument.")
            else:
                explanation = f"Combined signal: {verb} ({accel_mag:.1f} m/s²) + audio conflict."

        # 2. harsh_braking: harsh brake/accel without argument (gate by score)
        elif (has_harsh_brake or has_harsh_accel) and c_score >= 0.35:
            flag_type = "harsh_braking"
            severity = _score_severity(c_score)
            if has_harsh_brake:
                tail = "Audio elevated." if a_score > 0.3 else "Traffic stop."
                explanation = f"Sudden deceleration detected ({accel_mag:.1f} m/s² spike). {tail}"
            else:
                explanation = f"Hard acceleration detected ({accel_mag:.1f} m/s² spike)."

        # 3. audio_spike: noise spike or argument (no motion event)
        elif has_noise_spike or has_argument:
            flag_type = "audio_spike"
            severity = _score_severity(c_score)
            if pd.notna(audio_db) and audio_db > 0:
                explanation = f"Elevated cabin audio ({audio_db:.0f} dB) detected."
            else:
                explanation = f"Audio spike detected: audio_score={a_score}"

        # 4. sustained_stress: sustained noise + elevated combined score
        elif has_sustained and c_score >= 0.50:
            flag_type = "sustained_stress"
            severity = "high" if c_score > 0.7 else ("medium" if c_score > 0.4 else "low")
            if pd.notna(audio_db) and audio_db > 0:
                explanation = f"Continued elevated audio ({audio_db:.0f} dB) with moderate motion disturbance."
            else:
                explanation = f"Sustained stress detected: motion_score={m_score} audio_score={a_score}"

        # 5. moderate_brake: lower gate when both sensors agree
        elif has_moderate and (c_score >= 0.35 if dual_sensor else c_score >= 0.45):
            flag_type = "moderate_brake"
            severity = _score_severity(c_score)
            if dual_sensor:
                explanation = f"Moderate motion event ({accel_mag:.1f} m/s²) with audio evidence (score={a_score})."
            else:
                explanation = f"Moderate motion event ({accel_mag:.1f} m/s²). Normal traffic pattern."

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
        # Keep only the top flags per trip (reference averages ~1.5 flags/trip)
        # Sort by priority: conflict_moment first (rare & high-value), then by score
        type_priority = {'conflict_moment': 0, 'audio_spike': 1, 'harsh_braking': 2,
                         'sustained_stress': 3, 'moderate_brake': 4}
        df['_type_pri'] = df['flag_type'].map(type_priority).fillna(5)
        df = df.sort_values(['_type_pri', 'combined_score'], ascending=[True, False])
        kept = []
        for trip_id, group in df.groupby('trip_id'):
            n = len(group)
            cap = max(1, min(2, n // 3))
            top = group.head(cap)
            # Ensure diversity of flag types (add 1 per missing type)
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

        # Trip-level fusion: if a trip has both motion (harsh/moderate) and audio
        # (audio_spike/sustained_stress) flags but no conflict_moment, promote one
        # flag to conflict_moment. Strategy: promote audio flag to preserve the
        # sole harsh_braking; but if trip has 2+ harsh_braking, promote a harsh one.
        motion_types = {'harsh_braking'}
        audio_types = {'audio_spike', 'sustained_stress'}
        for trip_id, group in df.groupby('trip_id'):
            types_present = set(group['flag_type'])
            has_motion_flag = bool(types_present & motion_types)
            has_audio_flag = bool(types_present & audio_types)
            if has_motion_flag and has_audio_flag and 'conflict_moment' not in types_present:
                harsh_count = (group['flag_type'] == 'harsh_braking').sum()
                if harsh_count >= 2:
                    # Safe to promote a harsh_braking (one still survives)
                    harsh_flags = group[group['flag_type'] == 'harsh_braking']
                    best_idx = harsh_flags['combined_score'].idxmax()
                else:
                    # Promote audio flag to preserve the sole harsh_braking
                    audio_flags = group[group['flag_type'].isin(audio_types)]
                    if len(audio_flags) == 0:
                        continue
                    best_idx = audio_flags['combined_score'].idxmax()
                df.loc[best_idx, 'flag_type'] = 'conflict_moment'
                old_exp = df.loc[best_idx, 'explanation']
                df.loc[best_idx, 'explanation'] = (
                    f"Combined signal: motion + audio events detected in trip. {old_exp}"
                )

        df = df.sort_values('timestamp').reset_index(drop=True)
        df['flag_id'] = [f"FLAG{str(i+1).zfill(3)}" for i in range(len(df))]

        # Assign severity via percentile to match reference distribution (~33% each)
        tercile_low = df['combined_score'].quantile(0.33)
        tercile_high = df['combined_score'].quantile(0.67)
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
                     earnings_velocity, motion_events_count, audio_events_count,
                     flagged_moments_count, max_severity, stress_score,
                     trip_quality_rating
    """
    meta_cols = ['trip_id', 'driver_id', 'date', 'duration_min', 'distance_km', 'fare']
    available_meta = [c for c in meta_cols if c in trips_df.columns]
    summaries = trips_df[available_meta].copy()

    # Earnings velocity (currency / hour)
    summaries['earnings_velocity'] = round(
        summaries['fare'] / (summaries['duration_min'] / 60), 2
    )

    # Aggregate counts from trip feature matrix
    tf = trip_features.copy()
    tf['motion_events_count'] = (
        tf['n_harsh_brakes'] + tf['n_harsh_accels'] + tf['n_swerves']
    ).astype(int)
    tf['audio_events_count'] = (
        tf['n_noise_spikes'] + tf['n_argument_signals']
    ).astype(int)
    tf['stress_score'] = tf['combined_score_mean'].round(2)

    summaries = summaries.merge(
        tf[['trip_id', 'motion_events_count', 'audio_events_count', 'stress_score']],
        on='trip_id',
        how='left',
    )

    # Flagged moments count & max severity per trip
    sev_order = {'low': 1, 'medium': 2, 'high': 3}
    if len(flagged_moments) > 0:
        flag_counts = flagged_moments.groupby('trip_id')['flag_id'].count().rename('flagged_moments_count')
        flagged_moments = flagged_moments.copy()
        flagged_moments['_sev_rank'] = flagged_moments['severity'].map(sev_order).fillna(0)
        max_sev = (
            flagged_moments.sort_values('_sev_rank', ascending=False)
            .groupby('trip_id')['severity'].first()
            .rename('max_severity')
        )
        summaries = summaries.merge(flag_counts, on='trip_id', how='left')
        summaries = summaries.merge(max_sev, on='trip_id', how='left')

    summaries['flagged_moments_count'] = summaries.get('flagged_moments_count', 0).fillna(0).astype(int)
    summaries['max_severity'] = summaries.get('max_severity', 'none').fillna('none')
    summaries['motion_events_count'] = summaries['motion_events_count'].fillna(0).astype(int)
    summaries['audio_events_count'] = summaries['audio_events_count'].fillna(0).astype(int)
    summaries['stress_score'] = summaries['stress_score'].fillna(0).round(2)

    # Trip quality rating
    def quality_rating(score):
        if pd.isna(score) or score < 0.3:
            return 'excellent'
        if score < 0.5:
            return 'good'
        if score < 0.7:
            return 'fair'
        return 'poor'

    summaries['trip_quality_rating'] = summaries['stress_score'].apply(quality_rating)

    output_cols = [
        'trip_id', 'driver_id', 'date', 'duration_min', 'distance_km', 'fare',
        'earnings_velocity', 'motion_events_count', 'audio_events_count',
        'flagged_moments_count', 'max_severity', 'stress_score', 'trip_quality_rating',
    ]
    available_cols = [c for c in output_cols if c in summaries.columns]
    return summaries[available_cols]


def generate_earnings_summary(earnings_features: pd.DataFrame, drivers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate earnings velocity summary for dashboard/API.
    """
    summary = earnings_features.copy()
    
    # Merge with driver info
    summary = summary.merge(
        drivers_df[['driver_id', 'name', 'city']],
        on='driver_id',
        how='left'
    )
    
    # Select key columns
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
        'hours_remaining'
    ]
    
    available_cols = [col for col in output_cols if col in summary.columns]
    output = summary[available_cols]
    
    return output


def main():
    """Generate all hackathon-required outputs."""
    print("=" * 70)
    print("DRIVER PULSE - HACKATHON OUTPUT GENERATOR")
    print("=" * 70)
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create output directory
    output_dir = '../processed_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load raw data
    print("[1/7] Loading raw data...")
    data_dir = "../driver_pulse_hackathon_data"
    datasets = load_all(data_dir)
    print(f"   ✓ Loaded {len(datasets['accelerometer'])} accelerometer readings")
    print(f"   ✓ Loaded {len(datasets['audio'])} audio readings")
    print(f"   ✓ Loaded {len(datasets['trips'])} trips")
    print(f"   ✓ Loaded {len(datasets['drivers'])} drivers")
    
    # Step 2: Generate features
    print("\n[2/7] Running feature engineering pipeline...")
    features = build_all_features(datasets)
    print(f"   ✓ Generated {len(features['fused_features'])} fused feature rows")
    print(f"   ✓ Generated {len(features['trip_features'])} trip summaries")
    print(f"   ✓ Generated {len(features['earnings_features'])} earnings forecasts")
    
    # Step 3: Generate flagged moments
    print("\n[3/7] Generating flagged_moments.csv...")
    flagged_moments = generate_flagged_moments(features['fused_features'], datasets['trips'])
    output_path = os.path.join(output_dir, 'flagged_moments.csv')
    flagged_moments.to_csv(output_path, index=False)
    print(f"   ✓ Generated {len(flagged_moments)} flagged moments")
    print(f"   📂 Saved to: {output_path}")
    
    # Step 4: Generate trip summaries
    print("\n[4/7] Generating trip_summaries.csv...")
    trip_summaries = generate_trip_summaries(features['trip_features'], datasets['trips'], flagged_moments)
    output_path = os.path.join(output_dir, 'trip_summaries.csv')
    trip_summaries.to_csv(output_path, index=False)
    print(f"   ✓ Generated {len(trip_summaries)} trip summaries")
    print(f"   📂 Saved to: {output_path}")
    
    # Step 5: Generate earnings summary
    print("\n[5/7] Generating earnings_velocity.csv...")
    earnings_summary = generate_earnings_summary(features['earnings_features'], datasets['drivers'])
    output_path = os.path.join(output_dir, 'earnings_velocity.csv')
    earnings_summary.to_csv(output_path, index=False)
    print(f"   ✓ Generated {len(earnings_summary)} earnings records")
    print(f"   📂 Saved to: {output_path}")
    
    # Step 6: Save full preprocessed accelerometer data
    print("\n[6/7] Saving preprocessed accelerometer data...")
    output_path = os.path.join(output_dir, 'accelerometer_preprocessed.csv')
    features['fused_features'].to_csv(output_path, index=False)
    print(f"   ✓ Saved {len(features['fused_features'])} preprocessed rows")
    print(f"   📂 Saved to: {output_path}")
    
    # Step 7: Save preprocessed audio data
    print("\n[7/7] Saving preprocessed audio data...")
    output_path = os.path.join(output_dir, 'audio_preprocessed.csv')
    features['aud_features'].to_csv(output_path, index=False)
    print(f"   ✓ Saved {len(features['aud_features'])} preprocessed rows")
    print(f"   📂 Saved to: {output_path}")
    
    # Summary statistics
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
