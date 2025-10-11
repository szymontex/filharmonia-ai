"""
Delete all prediction CSVs that are NOT in edited_csvs.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.services.model_registry import is_csv_edited


def delete_unedited_csvs():
    """Delete all unedited prediction CSVs"""
    results_folder = settings.SORTED_FOLDER / "ANALYSIS_RESULTS"

    if not results_folder.exists():
        print("[ERROR] ANALYSIS_RESULTS folder not found")
        return

    # Find all prediction CSVs
    all_csvs = list(results_folder.glob("predictions_*.csv"))
    print(f"[*] Found {len(all_csvs)} total prediction CSVs")

    # Separate edited vs unedited
    edited = []
    unedited = []

    for csv_file in all_csvs:
        if is_csv_edited(str(csv_file)):
            edited.append(csv_file)
        else:
            unedited.append(csv_file)

    print(f"[*] Edited: {len(edited)} files (will keep)")
    print(f"[*] Unedited: {len(unedited)} files (will delete)")

    if len(unedited) == 0:
        print("[INFO] No unedited files to delete")
        return

    # Show sample of files to delete
    print(f"\n[PREVIEW] First 10 files to delete:")
    for csv_file in unedited[:10]:
        print(f"  - {csv_file.name}")

    if len(unedited) > 10:
        print(f"  ... and {len(unedited) - 10} more")

    # Delete files (no confirmation - user already approved)
    print(f"\n[DELETING] Removing {len(unedited)} unedited CSV files...")
    deleted_count = 0
    for csv_file in unedited:
        try:
            csv_file.unlink()
            deleted_count += 1
            if deleted_count % 20 == 0:
                print(f"[PROGRESS] Deleted {deleted_count}/{len(unedited)}...")
        except Exception as e:
            print(f"[ERROR] Failed to delete {csv_file.name}: {e}")

    print(f"\n[SUCCESS] Deleted {deleted_count}/{len(unedited)} unedited CSV files")
    print(f"[INFO] Kept {len(edited)} edited CSV files")


if __name__ == "__main__":
    delete_unedited_csvs()
