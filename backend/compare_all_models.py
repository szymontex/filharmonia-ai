"""
Compare all models with and without normalization
"""
import sys
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import librosa
from skimage.transform import resize

sys.path.insert(0, str(Path(__file__).parent))
from app.config import settings

def load_data_with_normalization(normalize=True):
    """Load training data with optional normalization"""
    X = []
    y = []
    samples_per_class = {}

    LABELS = {
        'APPLAUSE': 0,
        'MUSIC': 1,
        'PUBLIC': 2,
        'SPEECH': 3,
        'TUNING': 4
    }

    for class_name, label in LABELS.items():
        class_folder = settings.TRAINING_DATA_FOLDER / class_name
        if not class_folder.exists():
            samples_per_class[class_name] = 0
            continue

        wav_files = list(class_folder.glob('*.wav'))
        samples_per_class[class_name] = len(wav_files)

        for wav_file in wav_files:
            signal, sr = librosa.load(
                str(wav_file),
                sr=settings.SAMPLE_RATE,
                duration=settings.FRAME_DURATION_SEC,
                res_type='kaiser_fast'
            )

            melspec = librosa.feature.melspectrogram(
                y=signal,
                sr=settings.SAMPLE_RATE,
                n_fft=2048,
                hop_length=512,
                n_mels=128
            )

            logmel = librosa.power_to_db(melspec, ref=np.max)
            logmel = resize(logmel, (128, 128))

            if normalize:
                # Normalization (like analizuj3wszystko.py)
                logmel = logmel - np.min(logmel)
                logmel = logmel / np.max(logmel)

            rgb = np.stack((logmel,) * 3, axis=-1)
            X.append(rgb)
            y.append(label)

    return np.array(X), np.array(y), samples_per_class

def evaluate_model(model_path: str, X_train, X_test, y_train_cat, y_test_cat, y_test, normalize_label):
    """Evaluate a model"""
    model = load_model(str(model_path), compile=True)

    train_loss, train_acc = model.evaluate(X_train, y_train_cat, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)

    # Per-class accuracy
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    LABELS = ["APPLAUSE", "MUSIC", "PUBLIC", "SPEECH", "TUNING"]
    per_class = {}

    for i, class_name in enumerate(LABELS):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == y_test[class_mask]).mean()
            per_class[class_name] = class_acc * 100
        else:
            per_class[class_name] = 0

    return {
        "model": Path(model_path).name,
        "normalize": normalize_label,
        "train_acc": train_acc * 100,
        "val_acc": test_acc * 100,
        "train_loss": train_loss,
        "val_loss": test_loss,
        "per_class": per_class
    }

if __name__ == "__main__":
    models_dir = settings.RECOGNITION_MODELS_FOLDER

    model_files = [
        "cnn_model.h5",
        "cnn_model_20251003_141737.h5",
        "cnn_model_20251004_210350.h5",
        "cnn_model_20251004_211439.h5"
    ]

    print("=" * 100)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 100)

    results = []

    for normalize in [False, True]:
        norm_label = "WITH normalization" if normalize else "WITHOUT normalization"
        print(f"\n{'#' * 100}")
        print(f"# LOADING DATA {norm_label}")
        print(f"{'#' * 100}\n")

        X, y, samples_per_class = load_data_with_normalization(normalize=normalize)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        from tensorflow.keras.utils import to_categorical
        y_train_cat = to_categorical(y_train, 5)
        y_test_cat = to_categorical(y_test, 5)

        print(f"Loaded {len(X)} samples (Train: {len(X_train)}, Test: {len(X_test)})")

        for model_file in model_files:
            model_path = models_dir / model_file
            if not model_path.exists():
                print(f"SKIP: {model_file} (not found)")
                continue

            print(f"\nEvaluating: {model_file}")
            result = evaluate_model(model_path, X_train, X_test, y_train_cat, y_test_cat, y_test, norm_label)
            results.append(result)

            print(f"  Train: {result['train_acc']:.2f}% (loss: {result['train_loss']:.4f})")
            print(f"  Val:   {result['val_acc']:.2f}% (loss: {result['val_loss']:.4f})")
            print(f"  Per-class: APPLAUSE={result['per_class']['APPLAUSE']:.1f}% MUSIC={result['per_class']['MUSIC']:.1f}% " +
                  f"PUBLIC={result['per_class']['PUBLIC']:.1f}% SPEECH={result['per_class']['SPEECH']:.1f}% TUNING={result['per_class']['TUNING']:.1f}%")

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Model':<40} {'Preprocessing':<25} {'Train Acc':<12} {'Val Acc':<12} {'Val Loss':<12}")
    print("-" * 100)

    for result in results:
        print(f"{result['model']:<40} {result['normalize']:<25} {result['train_acc']:>10.2f}% {result['val_acc']:>10.2f}% {result['val_loss']:>10.4f}")

    print("=" * 100)
    print("\nRECOMMENDATION:")
    print("=" * 100)

    # Find best model (highest validation accuracy)
    best = max(results, key=lambda x: x['val_acc'])
    print(f"Best model: {best['model']}")
    print(f"Preprocessing: {best['normalize']}")
    print(f"Validation accuracy: {best['val_acc']:.2f}%")
    print(f"Validation loss: {best['val_loss']:.4f}")
    print(f"\nPer-class accuracy:")
    for class_name, acc in best['per_class'].items():
        print(f"  {class_name}: {acc:.1f}%")

    print("\n" + "=" * 100)
