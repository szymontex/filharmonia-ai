"""
Prepare BALANCED dataset from ALL available data
Takes minimum class size and balances all others
"""
import sys
import random
from pathlib import Path
from collections import defaultdict
import librosa
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from app.config import settings

# Output folder
DATASET_FOLDER = Path(r"Y:\!_FILHARMONIA\ML_EXPERIMENTS\datasets\full_balanced")

def count_segments_per_class():
    """Count how many segments each class has"""
    counts = {}

    for class_name in settings.LABELS:
        class_folder = settings.TRAINING_DATA_FOLDER / class_name
        if not class_folder.exists():
            counts[class_name] = 0
            continue

        total_segments = 0
        wav_files = list(class_folder.glob('*.wav'))

        for wav_file in wav_files:
            info = sf.info(str(wav_file))
            file_duration = info.duration
            file_segments = int(file_duration / settings.FRAME_DURATION_SEC)
            total_segments += file_segments

        counts[class_name] = total_segments

    return counts

def prepare_dataset():
    """Prepare balanced dataset"""

    print("=" * 80)
    print("PREPARING FULL BALANCED DATASET")
    print("=" * 80)

    # Count segments per class
    print("\nCounting available segments...")
    segment_counts = count_segments_per_class()

    for class_name, count in segment_counts.items():
        print(f"  {class_name}: {count} segments")

    # Find minimum (will balance to this)
    min_segments = min(segment_counts.values())
    print(f"\nMinimum class size: {min_segments} segments")
    print(f"Will balance all classes to: {min_segments} segments each")

    # Calculate how many segments we'll have
    total_segments = min_segments * len(settings.LABELS)
    print(f"Total dataset size: {total_segments} segments")
    print(f"Split: 80% train, 10% val, 10% test")
    print(f"  Train: {int(total_segments * 0.8)} segments")
    print(f"  Val: {int(total_segments * 0.1)} segments")
    print(f"  Test: {int(total_segments * 0.1)} segments")

    # Ask for confirmation
    print("\n" + "=" * 80)
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Create output folders
    for split in ['train', 'val', 'test']:
        for class_name in settings.LABELS:
            folder = DATASET_FOLDER / split / class_name
            folder.mkdir(parents=True, exist_ok=True)

    # Process each class
    for class_name in settings.LABELS:
        print(f"\nProcessing {class_name}...")
        class_folder = settings.TRAINING_DATA_FOLDER / class_name

        if not class_folder.exists():
            print(f"  ERROR: Folder not found!")
            continue

        # Collect all segments from this class
        all_segments = []
        wav_files = list(class_folder.glob('*.wav'))

        for wav_file in wav_files:
            # Load entire file
            signal, sr = librosa.load(
                str(wav_file),
                sr=settings.SAMPLE_RATE,
                res_type='kaiser_fast'
            )

            # Split into segments
            frame_length = int(settings.FRAME_DURATION_SEC * settings.SAMPLE_RATE)

            for segment_start in range(0, len(signal), frame_length):
                segment_end = segment_start + frame_length

                # Only complete segments
                if segment_end > len(signal):
                    break

                segment = signal[segment_start:segment_end]
                all_segments.append({
                    'audio': segment,
                    'source_file': wav_file.name,
                    'segment_idx': len(all_segments)
                })

        # Shuffle and limit to min_segments
        random.shuffle(all_segments)
        selected_segments = all_segments[:min_segments]

        print(f"  Available: {len(all_segments)} segments")
        print(f"  Selected: {len(selected_segments)} segments")

        # Split: 80% train, 10% val, 10% test
        train_size = int(len(selected_segments) * 0.8)
        val_size = int(len(selected_segments) * 0.1)

        train_segs = selected_segments[:train_size]
        val_segs = selected_segments[train_size:train_size + val_size]
        test_segs = selected_segments[train_size + val_size:]

        print(f"  Split: {len(train_segs)} train, {len(val_segs)} val, {len(test_segs)} test")

        # Save segments
        for split, segments in [('train', train_segs), ('val', val_segs), ('test', test_segs)]:
            for i, seg_data in enumerate(segments):
                output_path = DATASET_FOLDER / split / class_name / f"seg_{i:05d}.wav"
                sf.write(
                    str(output_path),
                    seg_data['audio'],
                    settings.SAMPLE_RATE
                )

    # Summary
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        total = 0
        for class_name in settings.LABELS:
            folder = DATASET_FOLDER / split / class_name
            count = len(list(folder.glob('*.wav')))
            total += count
            print(f"  {class_name}: {count} segments")
        print(f"  TOTAL: {total} segments")

    print("\n" + "=" * 80)
    print(f"Dataset ready: {DATASET_FOLDER}")
    print("=" * 80)

if __name__ == "__main__":
    random.seed(42)
    prepare_dataset()
