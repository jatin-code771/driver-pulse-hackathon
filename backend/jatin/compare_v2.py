import pandas as pd
import numpy as np

ref_flag = pd.read_csv('../driver_pulse_hackathon_data/processed_outputs/flagged_moments.csv')
our_flag = pd.read_csv('../processed_outputs/flagged_moments.csv')

print("=" * 70)
print("ACCURACY COMPARISON v2: Our Output vs Hackathon Reference")
print("=" * 70)

print(f"\nReference: {len(ref_flag)} flags")
print(f"Ours:      {len(our_flag)} flags")

print("\nFlag type distribution:")
header = f"  {'Type':<22} {'Reference':>10} {'Ours':>10}"
print(header)
print("  " + "-" * 44)
all_types = set(ref_flag['flag_type'].unique()) | set(our_flag['flag_type'].unique())
for ft in sorted(all_types):
    r = (ref_flag['flag_type'] == ft).sum()
    o = (our_flag['flag_type'] == ft).sum()
    print(f"  {ft:<22} {r:>10} {o:>10}")

ref_trips = set(ref_flag['trip_id'].unique())
our_trips = set(our_flag['trip_id'].unique())
overlap = ref_trips & our_trips
print(f"\nTrips flagged in reference:      {len(ref_trips)}")
print(f"Trips flagged by us:             {len(our_trips)}")
print(f"Overlap trips:                   {len(overlap)}")
if len(ref_trips) > 0:
    print(f"Trip recall (overlap/ref):       {len(overlap)/len(ref_trips)*100:.1f}%")

correct = 0
total_ref = 0
for trip in overlap:
    ref_types = set(ref_flag[ref_flag['trip_id'] == trip]['flag_type'])
    our_types = set(our_flag[our_flag['trip_id'] == trip]['flag_type'])
    matches = ref_types & our_types
    correct += len(matches)
    total_ref += len(ref_types)
if total_ref > 0:
    print(f"\nFlag-type match (overlap trips): {correct}/{total_ref} = {correct/total_ref*100:.1f}%")

print("\nSeverity distribution:")
sev_header = f"  {'Severity':<12} {'Reference':>10} {'Ours':>10}"
print(sev_header)
for s in ['low', 'medium', 'high']:
    r = (ref_flag['severity'] == s).sum()
    o = (our_flag['severity'] == s).sum()
    print(f"  {s:<12} {r:>10} {o:>10}")

ref_overlap = ref_flag[ref_flag['trip_id'].isin(overlap)]
our_overlap = our_flag[our_flag['trip_id'].isin(overlap)]
for col in ['motion_score', 'audio_score', 'combined_score']:
    if col in ref_overlap.columns and col in our_overlap.columns:
        rm = ref_overlap[col].mean()
        om = our_overlap[col].mean()
        print(f"\nMean {col} (overlap trips):")
        print(f"  Reference: {rm:.3f}")
        print(f"  Ours:      {om:.3f}")
        print(f"  Diff:      {abs(rm - om):.3f}")

print("\n" + "=" * 70)
print("OVERALL ACCURACY ESTIMATE")
print("=" * 70)

scores = []
if len(ref_trips) > 0:
    trip_cov = len(overlap) / len(ref_trips)
    scores.append(('Trip coverage', trip_cov))
    print(f"  Trip coverage:       {trip_cov*100:.1f}%")

if total_ref > 0:
    type_match = correct / total_ref
    scores.append(('Flag type match', type_match))
    print(f"  Flag type match:     {type_match*100:.1f}%")

for col in ['motion_score', 'audio_score', 'combined_score']:
    if col in ref_overlap.columns and col in our_overlap.columns:
        diff = abs(ref_overlap[col].mean() - our_overlap[col].mean())
        closeness = max(0, 1 - diff)
        scores.append((f'{col} closeness', closeness))
        print(f"  {col} closeness: {closeness*100:.1f}%")

ref_sev_dist = ref_flag['severity'].value_counts(normalize=True).reindex(['low','medium','high'], fill_value=0)
our_sev_dist = our_flag['severity'].value_counts(normalize=True).reindex(['low','medium','high'], fill_value=0)
sev_diff = (ref_sev_dist - our_sev_dist).abs().mean()
sev_closeness = max(0, 1 - sev_diff * 3)
scores.append(('Severity closeness', sev_closeness))
print(f"  Severity closeness:  {sev_closeness*100:.1f}%")

all_ftypes = sorted(set(ref_flag['flag_type'].unique()) | set(our_flag['flag_type'].unique()))
ref_type_dist = ref_flag['flag_type'].value_counts(normalize=True).reindex(all_ftypes, fill_value=0)
our_type_dist = our_flag['flag_type'].value_counts(normalize=True).reindex(all_ftypes, fill_value=0)
type_dist_diff = (ref_type_dist - our_type_dist).abs().mean()
type_closeness = max(0, 1 - type_dist_diff * len(all_ftypes))
scores.append(('Type distribution', type_closeness))
print(f"  Type distribution:   {type_closeness*100:.1f}%")

if scores:
    overall = np.mean([s[1] for s in scores])
    print(f"\n  >>> OVERALL ACCURACY: {overall*100:.1f}% <<<")
