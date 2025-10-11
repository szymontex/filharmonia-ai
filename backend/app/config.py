from pathlib import Path
from typing import List

class Settings:
    """Application settings"""

    # Paths - HYBRID: kod lokalnie, dane przez sieć
    FILHARMONIA_BASE: Path = Path(r"Y:\!_FILHARMONIA")
    SORTED_FOLDER: Path = FILHARMONIA_BASE / "SORTED"
    NAGRANIA_FOLDER: Path = FILHARMONIA_BASE / "!NAGRANIA KONCERTÓW"
    TRAINING_DATA_FOLDER: Path = FILHARMONIA_BASE / "TRAINING DATA" / "DATA"
    RECOGNITION_MODELS_FOLDER: Path = FILHARMONIA_BASE / "RECOGNITION_MODELS"

    # Models
    # Legacy Keras CNN (deprecated)
    MODEL_PATH: Path = RECOGNITION_MODELS_FOLDER / "cnn_model.h5"

    # New PyTorch AST (active)
    AST_MODEL_PATH: Path = RECOGNITION_MODELS_FOLDER / "ast_active.pth"

    LABELS: List[str] = ["APPLAUSE", "MUSIC", "PUBLIC", "SPEECH", "TUNING"]

    # Audio
    SAMPLE_RATE: int = 48000
    FRAME_DURATION_SEC: float = 2.97  # DO NOT CHANGE - training consistency!

    # Database
    DATABASE_URL: str = f"sqlite:///{FILHARMONIA_BASE}/.claude/filharmonia.db"

    # API
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

settings = Settings()
