"""
Detailed comparison of two model predictions on the same concert
"""
import pandas as pd
from collections import Counter
from pathlib import Path

# Load both CSV files
old_csv = Path("Y:/!_FILHARMONIA/SORTED/ANALYSIS_RESULTS/predictions_SONG019_2025-01-19_15-04.csv")
new_csv = Path("Y:/!_FILHARMONIA/SORTED/ANALYSIS_RESULTS/predictions_SONG019test_2025-01-19_15-04.csv")

old_df = pd.read_csv(old_csv)
new_df = pd.read_csv(new_csv)

print("="*80)
print("DETAILED MODEL COMPARISON - SONG019 (19.01.2025)")
print("="*80)
print(f"Old Model: {old_df['model_version'].iloc[0]}")
print(f"New Model: {new_df['model_version'].iloc[0]}")
print(f"Total segments: {len(old_df)}")
print()

# 1. Overall class distribution
print("="*80)
print("1. CLASS DISTRIBUTION")
print("="*80)

old_dist = Counter(old_df['predicted_class'])
new_dist = Counter(new_df['predicted_class'])

for cls in ['APPLAUSE', 'MUSIC', 'PUBLIC', 'SPEECH', 'TUNING']:
    old_count = old_dist.get(cls, 0)
    new_count = new_dist.get(cls, 0)
    diff = new_count - old_count
    print(f"{cls:12} | Old: {old_count:4} | New: {new_count:4} | Diff: {diff:+4}")
print()

# 2. Agreement analysis
print("="*80)
print("2. AGREEMENT BETWEEN MODELS")
print("="*80)

agreements = (old_df['predicted_class'] == new_df['predicted_class']).sum()
disagreements = len(old_df) - agreements
print(f"Agreements:    {agreements:4} ({100*agreements/len(old_df):.1f}%)")
print(f"Disagreements: {disagreements:4} ({100*disagreements/len(old_df):.1f}%)")
print()

# 3. Detailed disagreements
print("="*80)
print("3. WHERE MODELS DISAGREE (first 30 cases)")
print("="*80)

disagree_mask = old_df['predicted_class'] != new_df['predicted_class']
disagreements_df = pd.DataFrame({
    'time': old_df[disagree_mask]['segment_time'],
    'old_pred': old_df[disagree_mask]['predicted_class'],
    'old_conf': old_df[disagree_mask]['confidence'],
    'new_pred': new_df[disagree_mask]['predicted_class'],
    'new_conf': new_df[disagree_mask]['confidence']
})

print(disagreements_df.head(30).to_string(index=False))
print(f"\n... and {len(disagreements_df)-30} more disagreements")
print()

# 4. Confidence analysis
print("="*80)
print("4. CONFIDENCE LEVELS")
print("="*80)

for cls in ['APPLAUSE', 'MUSIC', 'PUBLIC', 'SPEECH', 'TUNING']:
    old_conf = old_df[old_df['predicted_class'] == cls]['confidence'].mean()
    new_conf = new_df[new_df['predicted_class'] == cls]['confidence'].mean()
    print(f"{cls:12} | Old: {old_conf:.4f} | New: {new_conf:.4f} | Diff: {new_conf-old_conf:+.4f}")
print()

# 5. Transition analysis (boundary detection)
print("="*80)
print("5. TRANSITION POINTS (Class Changes)")
print("="*80)

def find_transitions(df):
    transitions = []
    prev_class = None
    for idx, row in df.iterrows():
        if prev_class is not None and row['predicted_class'] != prev_class:
            transitions.append({
                'time': row['segment_time'],
                'from': prev_class,
                'to': row['predicted_class']
            })
        prev_class = row['predicted_class']
    return transitions

old_transitions = find_transitions(old_df)
new_transitions = find_transitions(new_df)

print(f"Old Model: {len(old_transitions)} transitions")
print(f"New Model: {len(new_transitions)} transitions")
print()

print("First 15 transitions (Old Model):")
for t in old_transitions[:15]:
    print(f"  {t['time']} | {t['from']:10} → {t['to']}")

print()
print("First 15 transitions (New Model):")
for t in new_transitions[:15]:
    print(f"  {t['time']} | {t['from']:10} → {t['to']}")
print()

# 6. Specific cases user mentioned
print("="*80)
print("6. SPECIFIC CASES FROM USER OBSERVATIONS")
print("="*80)

def show_window(df, time_str, window=3):
    """Show predictions around a specific time"""
    idx = df[df['segment_time'] == time_str].index
    if len(idx) > 0:
        start = max(0, idx[0] - window)
        end = min(len(df), idx[0] + window + 1)
        return df.iloc[start:end][['segment_time', 'predicted_class', 'confidence']]
    return None

print("\nCase 1: Speech start (around 00:00:53)")
print("Old Model:")
print(show_window(old_df, '00:00:53'))
print("\nNew Model:")
print(show_window(new_df, '00:00:53'))

print("\n\nCase 2: Music start (around 02:16-02:19)")
print("Old Model:")
print(show_window(old_df, '00:02:16', window=5))
print("\nNew Model:")
print(show_window(new_df, '00:02:16', window=5))
