"""
Prepare balanced dataset for PyTorch training
Priority: SONG* files (clean), then fill with old files
"""
import sys
import random
import shutil
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from app.config import settings

# Target samples per class for small test
SAMPLES_PER_CLASS = 100

# Output folder
DATASET_FOLDER = Path(r"Y:\!_FILHARMONIA\ML_EXPERIMENTS\datasets\small_balanced")

def prepare_dataset():
    """Prepare balanced dataset"""

    print("=" * 80)
    print("PREPARING BALANCED DATASET")
    print("=" * 80)
    print(f"Target: {SAMPLES_PER_CLASS} samples per class")
    print(f"Output: {DATASET_FOLDER}")
    print()

    # Create output folders
    for split in ['train', 'val', 'test']:
        for class_name in settings.LABELS:
            folder = DATASET_FOLDER / split / class_name
            folder.mkdir(parents=True, exist_ok=True)

    for class_name in settings.LABELS:
        print(f"\nProcessing {class_name}...")
        class_folder = settings.TRAINING_DATA_FOLDER / class_name

        if not class_folder.exists():
            print(f"  ERROR: Folder not found!")
            continue

        # Get all WAV files
        all_files = list(class_folder.glob('*.wav'))

        # Separate SONG* (priority) from others
        song_files = [f for f in all_files if f.name.startswith('SONG')]
        other_files = [f for f in all_files if not f.name.startswith('SONG')]

        print(f"  SONG files: {len(song_files)}")
        print(f"  Other files: {len(other_files)}")

        # Combine: SONG* first, then others
        selected_files = song_files.copy()

        # If we need more, add from other files
        if len(selected_files) < SAMPLES_PER_CLASS:
            needed = SAMPLES_PER_CLASS - len(selected_files)
            # Shuffle and take what we need
            random.shuffle(other_files)
            selected_files.extend(other_files[:needed])
        else:
            # If we have too many SONG files, randomly select
            random.shuffle(selected_files)
            selected_files = selected_files[:SAMPLES_PER_CLASS]

        print(f"  Selected: {len(selected_files)} files")

        if len(selected_files) < SAMPLES_PER_CLASS:
            print(f"  WARNING: Only {len(selected_files)}/{SAMPLES_PER_CLASS} available!")

        # Split: 80% train, 10% val, 10% test
        random.shuffle(selected_files)

        train_size = int(len(selected_files) * 0.8)
        val_size = int(len(selected_files) * 0.1)

        train_files = selected_files[:train_size]
        val_files = selected_files[train_size:train_size + val_size]
        test_files = selected_files[train_size + val_size:]

        print(f"  Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

        # Copy files
        for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            for file in files:
                dest = DATASET_FOLDER / split / class_name / file.name
                shutil.copy(file, dest)

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
            print(f"  {class_name}: {count} files")
        print(f"  TOTAL: {total} files")

    print("\n" + "=" * 80)
    print(f"Dataset ready: {DATASET_FOLDER}")
    print("=" * 80)

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    prepare_dataset()
