"""
Migrate models_metadata.json to new format with model_id and is_active fields
"""
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.services.model_registry import generate_model_id


def migrate_metadata():
    """Migrate existing models_metadata.json to new format"""
    metadata_path = settings.RECOGNITION_MODELS_FOLDER / "models_metadata.json"

    if not metadata_path.exists():
        print("[ERROR] models_metadata.json not found")
        return

    # Load existing metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"[*] Found {len(data.get('models', []))} models in metadata")

    # Add new fields to structure
    if "active_model" not in data:
        data["active_model"] = None
        print("[+] Added 'active_model' field")

    # Migrate each model entry
    for model in data.get("models", []):
        # Add model_id if missing
        if "model_id" not in model:
            model_id = generate_model_id(model["filename"])
            model["model_id"] = model_id
            print(f"[+] Added model_id '{model_id}' for {model['filename']}")

        # Add is_active if missing
        if "is_active" not in model:
            model["is_active"] = False
            print(f"[+] Added is_active=False for {model['filename']}")

    # Create backup
    backup_path = metadata_path.with_suffix('.json.backup')
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[BACKUP] Saved to {backup_path}")

    # Save migrated metadata
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Migration complete! Updated {metadata_path}")
    print("\n[INFO] Current state:")
    print(f"  - active_model: {data['active_model']}")
    print(f"  - total models: {len(data['models'])}")

    # Show all models
    print("\n[MODELS]:")
    for model in data["models"]:
        print(f"  - {model['model_id']}: {model['filename']} (is_active={model['is_active']})")


if __name__ == "__main__":
    migrate_metadata()
