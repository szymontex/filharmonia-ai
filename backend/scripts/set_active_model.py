"""
Set active model manually (for AST model that's not in metadata yet)
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings


def set_active_model_manual():
    """Set ast_active.pth as the active model"""
    metadata_path = settings.RECOGNITION_MODELS_FOLDER / "models_metadata.json"

    if not metadata_path.exists():
        print("[ERROR] models_metadata.json not found")
        return

    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check if ast_active is in models list
    ast_found = False
    for model in data.get("models", []):
        if "ast_" in model.get("model_id", ""):
            ast_found = True
            break

    if not ast_found:
        # Add ast_active.pth manually (since it was trained before metadata system)
        print("[INFO] ast_active.pth not in metadata - adding it now")
        data["models"].append({
            "filename": "ast_active.pth",
            "model_id": "ast_active",
            "trained_date": "2025-10-05T00:00:00",  # Approximate
            "test_accuracy": 0.9643,  # From docs
            "val_accuracy": 0.9600,
            "loss": 0.0,
            "val_loss": 0.0,
            "epochs_trained": 20,
            "training_samples": 0,
            "notes": "PyTorch AST model - migrated from legacy (96.43% test accuracy)",
            "is_active": True
        })
        data["active_model"] = "ast_active"
        print("[+] Added ast_active.pth to metadata and set as active")
    else:
        # Set existing AST model as active
        for model in data["models"]:
            if "ast_" in model.get("model_id", ""):
                model["is_active"] = True
                data["active_model"] = model["model_id"]
                print(f"[+] Set {model['model_id']} as active")
            else:
                model["is_active"] = False

    # Save
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Active model set to: {data['active_model']}")


if __name__ == "__main__":
    set_active_model_manual()
