"""
test_pipeline.py
----------------
Test the full data processing pipeline locally.
"""

import sys
import logging
from pathlib import Path

# Import our modules
import data_ingestion
import signal_preprocessing
import feature_engineering

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*60)
    logger.info("TESTING DRIVER PULSE DATA PIPELINE")
    logger.info("="*60)
    
    # Step 1: Load all datasets
    logger.info("\n[STEP 1] Loading raw data...")
    data_dir = "../driver_pulse_hackathon_data"
    datasets = data_ingestion.load_all(data_dir)
    
    if datasets["accelerometer"].empty:
        logger.error("Failed to load accelerometer data. Aborting.")
        return False
    
    logger.info(f"✓ Loaded {len(datasets)} datasets successfully")
    
    # Step 2: Run the complete feature engineering pipeline
    logger.info("\n[STEP 2] Running complete feature engineering pipeline...")
    features = feature_engineering.build_all_features(datasets)
    
    logger.info(f"✓ Pipeline complete! Generated features:")
    logger.info(f"  - acc_features: {len(features['acc_features'])} rows")
    logger.info(f"  - aud_features: {len(features['aud_features'])} rows")
    logger.info(f"  - fused_features: {len(features['fused_features'])} rows")
    logger.info(f"  - trip_features: {len(features['trip_features'])} trips")
    logger.info(f"  - earnings_features: {len(features['earnings_features'])} driver goals")
    
    # Step 3: Show sample of generated features
    logger.info("\n[STEP 3] Sample of fused features...")
    if len(features['fused_features']) > 0:
        sample_cols = ['trip_id', 'motion_score', 'audio_score_fused', 'combined_score', 'stress_severity']
        available_cols = [c for c in sample_cols if c in features['fused_features'].columns]
        logger.info(f"  Sample columns: {available_cols}")
        logger.info(features['fused_features'][available_cols].head(5).to_string(index=False))
    
    # Step 4: Show trip-level features
    logger.info("\n[STEP 4] Trip-level features...")
    trip_features = features['trip_features']
    logger.info(f"  Total features per trip: {len(trip_features.columns)}")
    logger.info(f"  Feature columns: {list(trip_features.columns)[:20]}...")
    
    # Step 5: Show earnings features
    logger.info("\n[STEP 5] Earnings features...")
    earnings_features = features['earnings_features']
    logger.info(f"  Earnings feature columns: {list(earnings_features.columns)[:15]}...")
    
    # Step 6: Merge with driver metadata
    logger.info("\n[STEP 6] Merging with driver profiles...")
    final_dataset = trip_features.merge(
        datasets["drivers"],
        on="driver_id",
        how="left"
    )
    logger.info(f"✓ Final dataset: {len(final_dataset)} trips with {len(final_dataset.columns)} features")
    
    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("PIPELINE TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"✓ All steps completed successfully!")
    logger.info(f"  - {len(datasets['drivers'])} unique drivers")
    logger.info(f"  - {len(datasets['trips'])} total trips in dataset")
    logger.info(f"  - {len(trip_features)} trips with sensor data & features")
    logger.info(f"  - {len(final_dataset.columns)} total features in final dataset")
    
    # Show sample of key metrics
    if "harsh_events_total" in trip_features.columns:
        harsh_trips = trip_features[trip_features["harsh_events_total"] > 0]
        logger.info(f"  - {len(harsh_trips)} trips with harsh driving events")
    
    if "audio_spike_count" in trip_features.columns:
        noisy_trips = trip_features[trip_features["audio_spike_count"] > 0]
        logger.info(f"  - {len(noisy_trips)} trips with audio spikes")
    
    if "stress_severity" in features['fused_features'].columns:
        high_stress = features['fused_features'][features['fused_features']["stress_severity"].isin(["high", "critical"])]
        logger.info(f"  - {len(high_stress)} high/critical stress moments detected")
    
    logger.info("\n✅ All tests passed! Ready to push to GitHub.")
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}", exc_info=True)
        sys.exit(1)
