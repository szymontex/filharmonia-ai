"""
Evaluate existing model on training data
"""
import sys
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.training import TrainingService
from app.config import settings

def evaluate_model(model_path: str):
    """Evaluate a model on current training data"""

    print(f"Evaluating model: {model_path}")
    print("=" * 60)

    # Load model
    print("\n1. Loading model...")
    model = load_model(model_path, compile=True)
    print(f"   OK Model loaded: {Path(model_path).name}")

    # Load training data (will now include normalization)
    print("\n2. Loading training data...")
    service = TrainingService()
    X, y, samples_per_class = service._load_training_data(settings.TRAINING_DATA_FOLDER)
    print("   (Note: Data is normalized to match inference preprocessing)")

    print(f"   OK Loaded {len(X)} samples:")
    for class_name, count in samples_per_class.items():
        print(f"     - {class_name}: {count}")

    # Split data (same way as training)
    print("\n3. Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    from tensorflow.keras.utils import to_categorical
    y_train_cat = to_categorical(y_train, 5)
    y_test_cat = to_categorical(y_test, 5)

    print(f"   OK Train: {len(X_train)} samples")
    print(f"   OK Test: {len(X_test)} samples")

    # Evaluate on training set
    print("\n4. Evaluating on training set...")
    train_loss, train_acc = model.evaluate(X_train, y_train_cat, verbose=0)
    print(f"   Training accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"   Training loss: {train_loss:.4f}")

    # Evaluate on test set
    print("\n5. Evaluating on test/validation set...")
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"   Validation accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Validation loss: {test_loss:.4f}")

    # Per-class accuracy
    print("\n6. Per-class accuracy on test set:")
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    LABELS = ["APPLAUSE", "MUSIC", "PUBLIC", "SPEECH", "TUNING"]

    for i, class_name in enumerate(LABELS):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == y_test[class_mask]).mean()
            class_count = class_mask.sum()
            print(f"   {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%) - {class_count} samples")
        else:
            print(f"   {class_name}: No test samples")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Training: {train_acc*100:.2f}% accuracy, {train_loss:.4f} loss")
    print(f"  Validation: {test_acc*100:.2f}% accuracy, {test_loss:.4f} loss")
    print("=" * 60)

    return {
        "accuracy": float(train_acc),
        "val_accuracy": float(test_acc),
        "loss": float(train_loss),
        "val_loss": float(test_loss),
        "epochs_trained": 0,  # Unknown
        "training_samples": len(X)
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_model.py <model_path>")
        print("\nExample:")
        print("  python evaluate_model.py Y:\\!_FILHARMONIA\\RECOGNITION_MODELS\\cnn_model_20251003_141737.h5")
        sys.exit(1)

    model_path = sys.argv[1]

    if not Path(model_path).exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    metrics = evaluate_model(model_path)

    print("\n\nJSON output (for metadata):")
    import json
    print(json.dumps(metrics, indent=2))
