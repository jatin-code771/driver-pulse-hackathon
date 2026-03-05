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
HARSH_BRAKE_THRESHOLD = -2.0
HARSH_ACCEL_THRESHOLD = 2.0
LATERAL_SWERVE_THRESHOLD = 1.5
AUDIO_SPIKE_THRESHOLD_DB = 85.0
HIGH_STRESS_THRESHOLD = 0.70


def generate_flagged_moments(fused_features: pd.DataFrame) -> pd.DataFrame:
    """
    Convert fused features to hackathon flagged_moments.csv format.
    
    Required Schema: timestamp, signal_type, raw_value, threshold, event_label
    
    This allows judges to trace any flagged moment back to raw sensor readings.
    """
    flagged_rows = []
    
    for _, row in fused_features.iterrows():
        # Harsh Braking Detection
        if row.get('is_harsh_brake', 0) == 1:
            flagged_rows.append({
                'timestamp': row['timestamp'],
                'signal_type': 'ACCELEROMETER',
                'raw_value': round(row['accel_delta'], 3),
                'threshold': HARSH_BRAKE_THRESHOLD,
                'event_label': 'HARSH_BRAKING',
                'trip_id': row.get('trip_id', ''),
                'severity': 'high' if row['accel_delta'] < -3.0 else 'medium'
            })
        
        # Harsh Acceleration Detection
        if row.get('is_harsh_accel', 0) == 1:
            flagged_rows.append({
                'timestamp': row['timestamp'],
                'signal_type': 'ACCELEROMETER',
                'raw_value': round(row['accel_delta'], 3),
                'threshold': HARSH_ACCEL_THRESHOLD,
                'event_label': 'HARSH_ACCELERATION',
                'trip_id': row.get('trip_id', ''),
                'severity': 'high' if row['accel_delta'] > 3.0 else 'medium'
            })
        
        # Lateral Swerve Detection
        if row.get('is_lateral_swerve', 0) == 1:
            flagged_rows.append({
                'timestamp': row['timestamp'],
                'signal_type': 'ACCELEROMETER',
                'raw_value': round(row['accel_lateral'], 3),
                'threshold': LATERAL_SWERVE_THRESHOLD,
                'event_label': 'LATERAL_SWERVE',
                'trip_id': row.get('trip_id', ''),
                'severity': 'high' if row['accel_lateral'] > 2.5 else 'medium'
            })
        
        # Noise Spike Detection
        if row.get('is_noise_spike', 0) == 1:
            audio_db = row.get('audio_level_db_fused', 0)
            if pd.notna(audio_db) and audio_db > 0:
                flagged_rows.append({
                    'timestamp': row['timestamp'],
                    'signal_type': 'AUDIO',
                    'raw_value': round(audio_db, 1),
                    'threshold': AUDIO_SPIKE_THRESHOLD_DB,
                    'event_label': 'NOISE_SPIKE',
                    'trip_id': row.get('trip_id', ''),
                    'severity': 'high' if audio_db > 95 else 'medium'
                })
        
        # Argument Signal Detection
        if row.get('is_argument_signal', 0) == 1:
            audio_db = row.get('audio_level_db_fused', 0)
            if pd.notna(audio_db) and audio_db > 0:
                flagged_rows.append({
                    'timestamp': row['timestamp'],
                    'signal_type': 'AUDIO',
                    'raw_value': round(audio_db, 1),
                    'threshold': AUDIO_SPIKE_THRESHOLD_DB,
                    'event_label': 'ARGUMENT_SIGNAL',
                    'trip_id': row.get('trip_id', ''),
                    'severity': 'critical' if row.get('is_sustained_noise', 0) == 1 else 'high'
                })
        
        # High Stress Moment (Fused Signal)
        if row.get('stress_severity', None) == 'high':
            flagged_rows.append({
                'timestamp': row['timestamp'],
                'signal_type': 'FUSED',
                'raw_value': round(row['combined_score'], 3),
                'threshold': HIGH_STRESS_THRESHOLD,
                'event_label': 'HIGH_STRESS_MOMENT',
                'trip_id': row.get('trip_id', ''),
                'severity': 'critical'
            })
    
    df = pd.DataFrame(flagged_rows)
    
    # Sort by timestamp
    if len(df) > 0:
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


def generate_trip_summaries(trip_features: pd.DataFrame, trips_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert trip_features to hackathon trip_summaries.csv format.
    
    This is the "report card" for each trip showing safety metrics.
    """
    summaries = trip_features.copy()
    
    # Merge with trip metadata
    summaries = summaries.merge(
        trips_df[['trip_id', 'driver_id', 'date', 'duration_min', 'fare']],
        on='trip_id',
        how='left'
    )
    
    # Calculate derived metrics
    summaries['total_harsh_events'] = (
        summaries.get('n_harsh_brakes', 0) + 
        summaries.get('n_harsh_accels', 0) +
        summaries.get('n_swerves', 0)
    )
    
    summaries['total_audio_conflicts'] = (
        summaries.get('n_noise_spikes', 0) +
        summaries.get('n_argument_signals', 0)
    )
    
    # Safety flags
    summaries['has_harsh_events'] = summaries['total_harsh_events'] > 0
    summaries['has_audio_conflict'] = summaries['total_audio_conflicts'] > 0
    summaries['has_high_stress'] = summaries.get('n_high_stress', 0) > 0
    summaries['requires_review'] = (
        (summaries.get('trip_quality_score', 1.0) < 0.6) | 
        (summaries.get('n_high_stress', 0) > 0)
    )
    
    # Safety rating (A-F scale)
    def safety_grade(score):
        if pd.isna(score):
            return 'N/A'
        if score >= 0.90:
            return 'A'
        elif score >= 0.80:
            return 'B'
        elif score >= 0.70:
            return 'C'
        elif score >= 0.60:
            return 'D'
        else:
            return 'F'
    
    summaries['safety_grade'] = summaries.get('trip_quality_score', 1.0).apply(safety_grade)
    
    # Select columns for output
    output_cols = [
        'trip_id',
        'driver_id',
        'date',
        'duration_min',
        'fare',
        'n_harsh_brakes',
        'n_harsh_accels',
        'n_swerves',
        'total_harsh_events',
        'harsh_event_rate',
        'n_noise_spikes',
        'n_argument_signals',
        'total_audio_conflicts',
        'conflict_signal_rate',
        'combined_score_mean',
        'combined_score_max',
        'n_high_stress',
        'n_medium_stress',
        'trip_quality_score',
        'safety_grade',
        'has_harsh_events',
        'has_audio_conflict',
        'has_high_stress',
        'requires_review'
    ]
    
    # Only include columns that exist
    available_cols = [col for col in output_cols if col in summaries.columns]
    output = summaries[available_cols]
    
    return output


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
    print("[1/5] Loading raw data...")
    data_dir = "../driver_pulse_hackathon_data"
    datasets = load_all(data_dir)
    print(f"   ✓ Loaded {len(datasets['accelerometer'])} accelerometer readings")
    print(f"   ✓ Loaded {len(datasets['audio'])} audio readings")
    print(f"   ✓ Loaded {len(datasets['trips'])} trips")
    print(f"   ✓ Loaded {len(datasets['drivers'])} drivers")
    
    # Step 2: Generate features
    print("\n[2/5] Running feature engineering pipeline...")
    features = build_all_features(datasets)
    print(f"   ✓ Generated {len(features['fused_features'])} fused feature rows")
    print(f"   ✓ Generated {len(features['trip_features'])} trip summaries")
    print(f"   ✓ Generated {len(features['earnings_features'])} earnings forecasts")
    
    # Step 3: Generate flagged moments
    print("\n[3/5] Generating flagged_moments.csv...")
    flagged_moments = generate_flagged_moments(features['fused_features'])
    output_path = os.path.join(output_dir, 'flagged_moments.csv')
    flagged_moments.to_csv(output_path, index=False)
    print(f"   ✓ Generated {len(flagged_moments)} flagged moments")
    print(f"   📂 Saved to: {output_path}")
    
    # Step 4: Generate trip summaries
    print("\n[4/5] Generating trip_summaries.csv...")
    trip_summaries = generate_trip_summaries(features['trip_features'], datasets['trips'])
    output_path = os.path.join(output_dir, 'trip_summaries.csv')
    trip_summaries.to_csv(output_path, index=False)
    print(f"   ✓ Generated {len(trip_summaries)} trip summaries")
    print(f"   📂 Saved to: {output_path}")
    
    # Step 5: Generate earnings summary
    print("\n[5/5] Generating earnings_velocity.csv...")
    earnings_summary = generate_earnings_summary(features['earnings_features'], datasets['drivers'])
    output_path = os.path.join(output_dir, 'earnings_velocity.csv')
    earnings_summary.to_csv(output_path, index=False)
    print(f"   ✓ Generated {len(earnings_summary)} earnings records")
    print(f"   📂 Saved to: {output_path}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("OUTPUT SUMMARY")
    print("=" * 70)
    
    print("\n📊 Flagged Moments Breakdown:")
    if len(flagged_moments) > 0:
        event_counts = flagged_moments['event_label'].value_counts()
        for event, count in event_counts.items():
            print(f"   • {event}: {count}")
        print(f"   TOTAL: {len(flagged_moments)}")
    else:
        print("   No events flagged (all trips were safe)")
    
    print("\n📊 Trip Summaries:")
    print(f"   • Total trips analyzed: {len(trip_summaries)}")
    if 'has_harsh_events' in trip_summaries.columns:
        print(f"   • Trips with harsh driving: {trip_summaries['has_harsh_events'].sum()}")
    if 'has_audio_conflict' in trip_summaries.columns:
        print(f"   • Trips with audio conflict: {trip_summaries['has_audio_conflict'].sum()}")
    if 'requires_review' in trip_summaries.columns:
        print(f"   • Trips requiring review: {trip_summaries['requires_review'].sum()}")
    if 'trip_quality_score' in trip_summaries.columns:
        avg_quality = trip_summaries['trip_quality_score'].mean()
        print(f"   • Avg trip quality score: {avg_quality:.3f}")
    if 'safety_grade' in trip_summaries.columns:
        print(f"   • Safety grade distribution:")
        for grade in ['A', 'B', 'C', 'D', 'F']:
            count = (trip_summaries['safety_grade'] == grade).sum()
            if count > 0:
                print(f"     - Grade {grade}: {count} trips")
    
    print("\n📊 Earnings Velocity:")
    if 'trajectory_label' in earnings_summary.columns:
        traj_counts = earnings_summary['trajectory_label'].value_counts()
        for label, count in traj_counts.items():
            print(f"   • {label}: {count} drivers")
    
    print("\n" + "=" * 70)
    print("✅ ALL HACKATHON OUTPUTS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📂 Output files location: {output_dir}/")
    print("   • flagged_moments.csv")
    print("   • trip_summaries.csv")
    print("   • earnings_velocity.csv")
    print("\n💡 Next Steps:")
    print("   1. Review the generated CSVs to verify format")
    print("   2. Share with your team for dashboard/API integration")
    print("   3. Use flagged_moments.csv for the event timeline UI")
    print("   4. Use trip_summaries.csv for the trip report cards")
    print("   5. Use earnings_velocity.csv for the earnings dashboard")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
