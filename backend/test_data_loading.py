"""
Test if we're loading data correctly - check duration of each file
"""
import sys
from pathlib import Path
import time
import librosa
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from app.config import settings

def test_data_loading():
    """Test data loading and timing"""

    print("=" * 80)
    print("TESTING DATA LOADING")
    print("=" * 80)

    LABELS = ['APPLAUSE', 'MUSIC', 'PUBLIC', 'SPEECH', 'TUNING']

    total_files = 0
    total_duration_actual = 0
    total_duration_loaded = 0

    for class_name in LABELS:
        class_folder = settings.TRAINING_DATA_FOLDER / class_name

        if not class_folder.exists():
            print(f"\n{class_name}: FOLDER NOT FOUND")
            continue

        wav_files = list(class_folder.glob('*.wav'))
        print(f"\n{class_name}: {len(wav_files)} files")

        class_actual_duration = 0
        class_loaded_duration = 0

        for i, wav_file in enumerate(wav_files[:3]):  # Check first 3 files
            # Get actual file duration WITHOUT loading
            info = sf.info(str(wav_file))
            actual_duration = info.duration

            # Load with librosa (what training does)
            signal, sr = librosa.load(
                str(wav_file),
                sr=settings.SAMPLE_RATE,
                duration=settings.FRAME_DURATION_SEC,  # 2.97 seconds
                res_type='kaiser_fast'
            )
            loaded_duration = len(signal) / sr

            print(f"  File {i+1}: {wav_file.name}")
            print(f"    Actual duration: {actual_duration:.2f}s")
            print(f"    Loaded duration: {loaded_duration:.2f}s")
            print(f"    Setting duration: {settings.FRAME_DURATION_SEC}s")

            if actual_duration > 3:
                print(f"    WARNING: File is {actual_duration:.2f}s but we only load {settings.FRAME_DURATION_SEC}s!")

            class_actual_duration += actual_duration
            class_loaded_duration += loaded_duration

        # Get totals for all files
        for wav_file in wav_files:
            info = sf.info(str(wav_file))
            total_duration_actual += info.duration
            total_duration_loaded += settings.FRAME_DURATION_SEC  # We always load 2.97s
            total_files += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files: {total_files}")
    print(f"Total actual audio duration: {total_duration_actual:.1f}s ({total_duration_actual/60:.1f} minutes)")
    print(f"Total loaded audio duration: {total_duration_loaded:.1f}s ({total_duration_loaded/60:.1f} minutes)")
    print(f"We are using: {(total_duration_loaded / total_duration_actual * 100):.1f}% of available audio")

    if total_duration_loaded < total_duration_actual * 0.5:
        print("\nWARNING: We're only using less than 50% of available audio data!")
        print("This is expected if files are longer than 2.97s")

    print("\n" + "=" * 80)
    print("CHECKING TRAINING SCRIPT LOGIC")
    print("=" * 80)

    # Check what trenowanie3.py did
    print("\nIn trenowanie3.py (line 26-35):")
    print("  signal, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, res_type='kaiser_fast')")
    print("  for s in range(0, len(signal), int(SAMPLES_PER_TRACK)):")
    print("      end_sample = s + int(SAMPLES_PER_TRACK)")
    print("      if len(signal[s:end_sample]) == int(SAMPLES_PER_TRACK):")
    print("          # Process segment...")
    print("")
    print("This means trenowanie3.py:")
    print("  1. Loads ONLY first 2.97s of each file (duration=DURATION)")
    print("  2. Then splits into 2.97s segments (but there's only 1 segment per file!)")
    print("")
    print("So trenowanie3.py also only used FIRST 2.97s of each file!")
    print("")

    # Check one long file
    print("\n" + "=" * 80)
    print("CHECKING ACTUAL FILE LENGTHS")
    print("=" * 80)

    for class_name in ['MUSIC', 'APPLAUSE']:
        class_folder = settings.TRAINING_DATA_FOLDER / class_name
        if not class_folder.exists():
            continue

        wav_files = list(class_folder.glob('*.wav'))
        if len(wav_files) > 0:
            # Check first 5 files
            print(f"\n{class_name} - First 5 files:")
            for wav_file in wav_files[:5]:
                info = sf.info(str(wav_file))
                print(f"  {wav_file.name}: {info.duration:.2f}s")

if __name__ == "__main__":
    test_data_loading()
