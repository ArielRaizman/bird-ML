import pandas as pd

# Load the HARU_C1 calls
df = pd.read_csv('haru_c1_calls.csv')

# For each trial, estimate how many 2-second clips would contain HARU_C1
clips_per_trial = {}

for trial in df['trial'].unique():
    trial_df = df[df['trial'] == trial]
    
    # Create 2-second bins and count which ones contain calls
    clips_with_calls = set()
    for _, row in trial_df.iterrows():
        # Find which 2-second clip(s) this call overlaps with
        start_clip = int(row['begin_time'] // 2)
        end_clip = int(row['end_time'] // 2)
        for clip_idx in range(start_clip, end_clip + 1):
            clips_with_calls.add(clip_idx)
    
    clips_per_trial[trial] = len(clips_with_calls)

print(f"Total HARU_C1 annotations: {len(df)}")
print(f"\nEstimated number of unique 2-second clips containing HARU_C1:")
for trial, count in sorted(clips_per_trial.items()):
    trial_count = len(df[df['trial'] == trial])
    print(f"  {trial}: {count} clips (from {trial_count} annotations)")

print(f"\nTotal unique clips across all trials: {sum(clips_per_trial.values())}")
