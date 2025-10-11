"""
Test new data loading - count total segments
"""
import sys
from pathlib import Path
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from app.config import settings

def count_segments():
    """Count how many 2.97s segments we'll have"""

    LABELS = ['APPLAUSE', 'MUSIC', 'PUBLIC', 'SPEECH', 'TUNING']
    FRAME_DURATION = 2.97

    total_segments = 0
    total_files = 0
    total_duration = 0

    print("=" * 80)
    print("COUNTING SEGMENTS WITH NEW LOADING METHOD")
    print("=" * 80)

    for class_name in LABELS:
        class_folder = settings.TRAINING_DATA_FOLDER / class_name

        if not class_folder.exists():
            print(f"\n{class_name}: FOLDER NOT FOUND")
            continue

        wav_files = list(class_folder.glob('*.wav'))
        class_segments = 0
        class_duration = 0

        for wav_file in wav_files:
            info = sf.info(str(wav_file))
            file_duration = info.duration
            file_segments = int(file_duration / FRAME_DURATION)  # Complete segments only

            class_segments += file_segments
            class_duration += file_duration
            total_files += 1

        total_segments += class_segments
        total_duration += class_duration

        print(f"\n{class_name}:")
        print(f"  Files: {len(wav_files)}")
        print(f"  Duration: {class_duration/60:.1f} minutes")
        print(f"  Segments (2.97s each): {class_segments}")

    print("\n" + "=" * 80)
    print("TOTAL")
    print("=" * 80)
    print(f"Files: {total_files}")
    print(f"Duration: {total_duration/60:.1f} minutes ({total_duration/3600:.1f} hours)")
    print(f"Segments: {total_segments}")
    print(f"\nOLD method: 274 segments (only 13.6 minutes)")
    print(f"NEW method: {total_segments} segments ({total_duration/60:.1f} minutes)")
    print(f"Improvement: {total_segments/274:.1f}x more training data!")
    print("=" * 80)

    estimated_time_minutes = total_segments / 274 * 1  # If 274 segments took ~1 minute
    print(f"\nEstimated training time: {estimated_time_minutes:.1f} minutes ({estimated_time_minutes/60:.1f} hours)")
    print("(Based on previous training speed of ~1 minute for 274 segments)")

if __name__ == "__main__":
    count_segments()
