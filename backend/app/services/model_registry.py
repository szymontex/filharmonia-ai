"""
Model Registry Service
Manages active model tracking and edited CSV detection
"""
from pathlib import Path
import json
from typing import Optional, Dict, List
from app.config import settings


def get_metadata_path() -> Path:
    """Get path to models metadata JSON"""
    return settings.RECOGNITION_MODELS_FOLDER / "models_metadata.json"


def get_edited_csvs_path() -> Path:
    """Get path to edited CSVs tracking file"""
    return settings.FILHARMONIA_BASE / ".claude" / "edited_csvs.txt"


def load_metadata() -> Dict:
    """Load models metadata from JSON"""
    metadata_path = get_metadata_path()

    if not metadata_path.exists():
        return {"active_model": None, "models": []}

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Ensure new format fields exist
        if "active_model" not in data:
            data["active_model"] = None

        return data
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return {"active_model": None, "models": []}


def save_metadata(data: Dict):
    """Save models metadata to JSON"""
    metadata_path = get_metadata_path()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_active_model_id() -> str:
    """
    Get currently active model ID

    Returns model_id (e.g. "ast_20251006_034512") or "unknown"
    """
    data = load_metadata()

    # Return active_model if set
    if data.get("active_model"):
        return data["active_model"]

    # Fallback: find is_active=true
    for model in data.get("models", []):
        if model.get("is_active"):
            return model.get("model_id", "unknown")

    return "unknown"


def get_active_model_info() -> Optional[Dict]:
    """Get full info about active model"""
    active_id = get_active_model_id()

    if active_id == "unknown":
        return None

    data = load_metadata()

    for model in data.get("models", []):
        if model.get("model_id") == active_id:
            return model

    return None


def set_active_model(model_id: str) -> bool:
    """
    Mark a model as active

    Args:
        model_id: Model identifier (e.g. "ast_20251006_034512")

    Returns:
        True if successful, False if model not found
    """
    data = load_metadata()

    # Find model with this ID
    model_found = False
    for model in data.get("models", []):
        if model.get("model_id") == model_id:
            model_found = True
            break

    if not model_found:
        return False

    # Unmark all models as active
    for model in data["models"]:
        model["is_active"] = False

    # Mark selected model as active
    for model in data["models"]:
        if model.get("model_id") == model_id:
            model["is_active"] = True
            data["active_model"] = model_id
            break

    save_metadata(data)
    return True


def is_csv_edited(csv_path: str) -> bool:
    """
    Check if CSV was manually edited

    Args:
        csv_path: Full path to CSV file

    Returns:
        True if CSV is in edited list
    """
    edited_log = get_edited_csvs_path()

    if not edited_log.exists():
        return False

    try:
        with open(edited_log, 'r', encoding='utf-8') as f:
            edited_files = set(line.strip() for line in f if line.strip())

        return csv_path in edited_files
    except Exception as e:
        print(f"Error reading edited CSVs: {e}")
        return False


def get_all_models() -> List[Dict]:
    """Get list of all trained models"""
    data = load_metadata()
    return data.get("models", [])


def generate_model_id(filename: str) -> str:
    """
    Generate model_id from filename

    Examples:
        ast_20251006_034512.pth -> ast_20251006_034512
        cnn_model_20251004_210350.h5 -> cnn_20251004_210350
    """
    stem = Path(filename).stem  # Remove extension

    # Handle different formats
    if stem.startswith("ast_"):
        # ast_20251006_034512 -> keep as is
        return stem
    elif "cnn_model_" in stem:
        # cnn_model_20251004_210350 -> cnn_20251004_210350
        return stem.replace("cnn_model_", "cnn_")
    else:
        # Fallback: use stem
        return stem
