#!/usr/bin/env python3
"""
Verify Filharmonia AI installation is complete and working.
Run this after setup to ensure all dependencies are installed correctly.
"""

import sys
from typing import Tuple

def test_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Test if a module can be imported."""
    try:
        __import__(module_name)
        return True, f"[OK] {package_name or module_name}"
    except ImportError as e:
        return False, f"[FAIL] {package_name or module_name}: {str(e)}"

def main():
    print("=" * 60)
    print("  Filharmonia AI - Installation Verification")
    print("=" * 60)
    print()

    tests = [
        # Core web framework
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("starlette", "Starlette"),

        # Database
        ("sqlalchemy", "SQLAlchemy"),

        # Async support
        ("aiohttp", "aiohttp"),
        ("multidict", "multidict"),
        ("yarl", "yarl"),
        ("aiohappyeyeballs", "aiohappyeyeballs"),

        # PyTorch & ML
        ("torch", "PyTorch"),
        ("torchvision", "torchvision"),
        ("torchaudio", "torchaudio"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),

        # Audio processing
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("audioread", "audioread"),
        ("soxr", "soxr"),

        # Scientific computing
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("sklearn", "scikit-learn"),

        # Utilities
        ("pydantic", "Pydantic"),
        ("eyed3", "eyeD3"),
        ("dotenv", "python-dotenv"),
        ("multipart", "python-multipart"),
        ("watchfiles", "watchfiles"),
        ("websockets", "websockets"),

        # Legacy TensorFlow support
        ("tensorflow", "TensorFlow"),
        ("keras", "Keras"),
    ]

    results = []
    for module, package in tests:
        success, message = test_import(module, package)
        results.append((success, message))
        print(message)

    print()
    print("-" * 60)

    passed = sum(1 for success, _ in results if success)
    total = len(results)

    print(f"Results: {passed}/{total} packages working")
    print()

    # Check PyTorch CUDA
    try:
        import torch
        print("PyTorch Details:")
        print(f"  Version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
        else:
            print("  [WARNING] CUDA not available. GPU training will not work.")
            print("  This is expected if you don't have an NVIDIA GPU or installed CPU-only version.")
    except Exception as e:
        print(f"[FAIL] Could not check PyTorch: {e}")

    print()
    print("-" * 60)

    if passed == total:
        print("[SUCCESS] Installation is COMPLETE!")
        print()
        print("Next steps:")
        print("  1. Configure .env file (set FILHARMONIA_BASE_DIR)")
        print("  2. Run: ./start.bat (Windows) or ./start.sh (Linux/Mac)")
        print("  3. Open: http://localhost:5173")
        return 0
    else:
        failed = total - passed
        print(f"[ERROR] Installation is INCOMPLETE! {failed} packages missing.")
        print()
        print("To fix:")
        print("  1. See TROUBLESHOOTING_INCOMPLETE_INSTALL.md")
        print("  2. Or run setup.bat/setup.sh again")
        return 1

if __name__ == "__main__":
    sys.exit(main())
