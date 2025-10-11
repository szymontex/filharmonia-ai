"""
Application configuration with environment variable support
"""
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load .env file from project root (3 levels up from this file)
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")


class Settings:
    """Application settings with environment variable overrides"""

    # Base directory - configurable via environment variable
    # Default: current working directory's parent / FILHARMONIA_DATA
    # Override: Set FILHARMONIA_BASE_DIR environment variable
    _base_dir_env = os.getenv("FILHARMONIA_BASE_DIR")
    if _base_dir_env:
        FILHARMONIA_BASE: Path = Path(_base_dir_env)
    else:
        # Default: create FILHARMONIA_DATA folder in project root
        FILHARMONIA_BASE: Path = Path(__file__).parent.parent.parent / "FILHARMONIA_DATA"

    # Ensure base directory exists
    FILHARMONIA_BASE.mkdir(parents=True, exist_ok=True)

    # Data paths - configurable folder names via environment variables
    SORTED_FOLDER: Path = FILHARMONIA_BASE / os.getenv("SORTED_FOLDER_NAME", "SORTED")
    NAGRANIA_FOLDER: Path = FILHARMONIA_BASE / os.getenv("NAGRANIA_FOLDER_NAME", "NAGRANIA_KONCERTOW")
    TRAINING_DATA_FOLDER: Path = FILHARMONIA_BASE / os.getenv("TRAINING_DATA_FOLDER_NAME", "TRAINING_DATA") / "DATA"
    RECOGNITION_MODELS_FOLDER: Path = FILHARMONIA_BASE / os.getenv("MODELS_FOLDER_NAME", "RECOGNITION_MODELS")
    ML_EXPERIMENTS_FOLDER: Path = FILHARMONIA_BASE / os.getenv("ML_EXPERIMENTS_FOLDER_NAME", "ML_EXPERIMENTS")

    # Models
    # Legacy Keras CNN (deprecated, kept for backwards compatibility)
    MODEL_PATH: Path = RECOGNITION_MODELS_FOLDER / "cnn_model.h5"

    # PyTorch AST (active model)
    AST_MODEL_PATH: Path = RECOGNITION_MODELS_FOLDER / "ast_active.pth"

    # Classification labels (alphabetical order - DO NOT CHANGE ORDER!)
    LABELS: List[str] = ["APPLAUSE", "MUSIC", "PUBLIC", "SPEECH", "TUNING"]

    # Audio processing settings
    SAMPLE_RATE: int = 48000
    FRAME_DURATION_SEC: float = 2.97  # DO NOT CHANGE - training consistency!

    # Database
    DATABASE_URL: str = f"sqlite:///{FILHARMONIA_BASE / '.claude' / 'filharmonia.db'}"

    # API settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",  # Vite default
        "http://localhost:3000",  # Alternative frontend port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

    @classmethod
    def get_dataset_dir(cls, dataset_name: str = "direct_from_source") -> Path:
        """
        Get path to ML experiments dataset

        Args:
            dataset_name: Name of dataset folder (e.g., 'direct_from_source', 'full_balanced')

        Returns:
            Path to dataset directory
        """
        return cls.ML_EXPERIMENTS_FOLDER / "datasets" / dataset_name


# Singleton instance
settings = Settings()


def print_config():
    """Print current configuration (useful for debugging)"""
    print("=" * 80)
    print("FILHARMONIA AI - Configuration")
    print("=" * 80)
    print(f"Base Directory: {settings.FILHARMONIA_BASE}")
    print(f"  Exists: {settings.FILHARMONIA_BASE.exists()}")
    print(f"\nData Folders:")
    print(f"  SORTED: {settings.SORTED_FOLDER}")
    print(f"  NAGRANIA: {settings.NAGRANIA_FOLDER}")
    print(f"  TRAINING_DATA: {settings.TRAINING_DATA_FOLDER}")
    print(f"  MODELS: {settings.RECOGNITION_MODELS_FOLDER}")
    print(f"  ML_EXPERIMENTS: {settings.ML_EXPERIMENTS_FOLDER}")
    print(f"\nActive Model: {settings.AST_MODEL_PATH}")
    print(f"  Exists: {settings.AST_MODEL_PATH.exists()}")
    print(f"\nDatabase: {settings.DATABASE_URL}")
    print("=" * 80)


if __name__ == "__main__":
    print_config()
