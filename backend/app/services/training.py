"""
Training Service - Model Retraining
Based on trenowanie3.py with real-time progress updates
"""
import os
import time
import threading
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
import uuid

import numpy as np
import librosa
from skimage.transform import resize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

from app.config import settings


@dataclass
class TrainingStatus:
    """Training job status"""
    job_id: str
    status: str  # 'preparing', 'training', 'completed', 'failed', 'cancelled'
    current_epoch: int
    total_epochs: int
    training_acc: float
    training_loss: float
    val_acc: float
    val_loss: float
    progress: float  # 0-100
    time_elapsed_sec: int
    time_remaining_sec: Optional[int]
    samples_per_class: Dict[str, int]
    error: Optional[str]
    model_filename: Optional[str]


@dataclass
class ModelInfo:
    """Model metadata"""
    filename: str
    trained_date: str
    accuracy: float
    val_accuracy: float
    loss: float
    val_loss: float
    epochs_trained: int
    training_samples: int
    notes: str


class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback to update job status during training"""

    def __init__(self, job_id: str, jobs_dict: Dict[str, TrainingStatus]):
        super().__init__()
        self.job_id = job_id
        self.jobs = jobs_dict
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        """Update job status after each epoch"""
        if self.job_id not in self.jobs:
            return

        job = self.jobs[self.job_id]
        job.current_epoch = epoch + 1
        job.training_acc = float(logs.get('accuracy', 0))
        job.training_loss = float(logs.get('loss', 0))
        job.val_acc = float(logs.get('val_accuracy', 0))
        job.val_loss = float(logs.get('val_loss', 0))
        job.progress = (epoch + 1) / job.total_epochs * 100

        # Estimate time remaining
        elapsed = time.time() - self.start_time
        if epoch > 0:
            avg_time_per_epoch = elapsed / (epoch + 1)
            remaining_epochs = job.total_epochs - (epoch + 1)
            job.time_remaining_sec = int(avg_time_per_epoch * remaining_epochs)

        job.time_elapsed_sec = int(elapsed)


class TrainingService:
    """Service for model training and management"""

    def __init__(self):
        self.jobs: Dict[str, TrainingStatus] = {}
        self.active_threads: Dict[str, threading.Thread] = {}
        self.metadata_path = settings.RECOGNITION_MODELS_FOLDER / "models_metadata.json"
        self.cancel_flags: Dict[str, bool] = {}  # For graceful cancellation

    def start_training_job(self) -> str:
        """Start a new training job in background thread"""
        job_id = str(uuid.uuid4())

        # Initialize job status
        self.jobs[job_id] = TrainingStatus(
            job_id=job_id,
            status='preparing',
            current_epoch=0,
            total_epochs=45,
            training_acc=0.0,
            training_loss=0.0,
            val_acc=0.0,
            val_loss=0.0,
            progress=0.0,
            time_elapsed_sec=0,
            time_remaining_sec=None,
            samples_per_class={},
            error=None,
            model_filename=None
        )

        self.cancel_flags[job_id] = False

        # Start background thread
        thread = threading.Thread(
            target=self._train_model_background,
            args=(job_id,),
            daemon=True
        )
        thread.start()
        self.active_threads[job_id] = thread

        return job_id

    def get_job_status(self, job_id: str) -> Optional[TrainingStatus]:
        """Get current status of training job"""
        return self.jobs.get(job_id)

    def cancel_job(self, job_id: str):
        """Request graceful cancellation of training job"""
        if job_id in self.cancel_flags:
            self.cancel_flags[job_id] = True
            if job_id in self.jobs:
                self.jobs[job_id].status = 'cancelled'

    def _load_training_data(self, data_folder: Path) -> tuple:
        """
        Load training data from WAV files
        Based on trenowanie3.py load_data()
        """
        X = []
        y = []
        segments_per_class = {}  # Track SEGMENTS not files

        LABELS = {
            'APPLAUSE': 0,
            'MUSIC': 1,
            'PUBLIC': 2,
            'SPEECH': 3,
            'TUNING': 4
        }

        for class_name, label in LABELS.items():
            class_folder = data_folder / class_name
            if not class_folder.exists():
                segments_per_class[class_name] = 0
                continue

            wav_files = list(class_folder.glob('*.wav'))
            class_segment_count = 0  # Count segments for this class

            for wav_file in wav_files:
                # Load ENTIRE audio file (not just 2.97s!)
                signal, sr = librosa.load(
                    str(wav_file),
                    sr=settings.SAMPLE_RATE,
                    res_type='kaiser_fast'
                )

                # Split into 2.97s segments
                frame_length = int(settings.FRAME_DURATION_SEC * settings.SAMPLE_RATE)

                for segment_start in range(0, len(signal), frame_length):
                    segment_end = segment_start + frame_length

                    # Only process complete segments
                    if segment_end > len(signal):
                        break

                    segment = signal[segment_start:segment_end]

                    # Generate mel-spectrogram - EXACT parameters from trenowanie3.py
                    melspec = librosa.feature.melspectrogram(
                        y=segment,
                        sr=settings.SAMPLE_RATE,
                        n_fft=2048,
                        hop_length=512,
                        n_mels=128
                    )

                    # Convert to log scale
                    logmel = librosa.power_to_db(melspec, ref=np.max)

                    # Resize to 128x128
                    logmel = resize(logmel, (128, 128))

                    # Normalization (MUST match inference - see analizuj3wszystko.py lines 43-44)
                    logmel = logmel - np.min(logmel)
                    logmel = logmel / np.max(logmel)

                    # Stack to RGB (3 channels)
                    rgb = np.stack((logmel,) * 3, axis=-1)

                    X.append(rgb)
                    y.append(label)
                    class_segment_count += 1  # Increment segment counter

            segments_per_class[class_name] = class_segment_count

        return np.array(X), np.array(y), segments_per_class

    def _build_model(self) -> Sequential:
        """
        Build CNN model - EXACT copy from trenowanie3.py
        DO NOT CHANGE - must match inference model!
        """
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            MaxPooling2D(2, 2),
            Dropout(0.2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(5, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _train_model_background(self, job_id: str):
        """
        Background training task
        Based on trenowanie3.py workflow
        """
        try:
            # 1. Update status: preparing
            self.jobs[job_id].status = 'preparing'

            # 2. Load training data
            print(f"[Training {job_id}] Loading training data...")
            X, y, samples_per_class = self._load_training_data(settings.TRAINING_DATA_FOLDER)

            self.jobs[job_id].samples_per_class = samples_per_class
            total_samples = sum(samples_per_class.values())

            if total_samples == 0:
                raise ValueError("No training data found")

            print(f"[Training {job_id}] Loaded {total_samples} samples:")
            for class_name, count in samples_per_class.items():
                print(f"  - {class_name}: {count}")

            # Check for cancellation
            if self.cancel_flags.get(job_id, False):
                self.jobs[job_id].status = 'cancelled'
                return

            # 3. Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Convert to categorical (one-hot encoding)
            y_train = to_categorical(y_train, 5)
            y_test = to_categorical(y_test, 5)

            # 4. Build model
            print(f"[Training {job_id}] Building model...")
            model = self._build_model()

            # Check for cancellation
            if self.cancel_flags.get(job_id, False):
                self.jobs[job_id].status = 'cancelled'
                return

            # 5. Prepare callbacks
            progress_callback = TrainingProgressCallback(job_id, self.jobs)
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=1,
                restore_best_weights=True
            )

            # Temporary checkpoint file
            checkpoint_path = settings.RECOGNITION_MODELS_FOLDER / f'temp_checkpoint_{job_id}.h5'
            model_checkpoint = ModelCheckpoint(
                str(checkpoint_path),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )

            # 6. Update status: training
            self.jobs[job_id].status = 'training'
            self.jobs[job_id].total_epochs = 45

            print(f"[Training {job_id}] Starting training...")

            # 7. Train model
            history = model.fit(
                X_train, y_train,
                epochs=45,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[progress_callback, early_stopping, model_checkpoint],
                verbose=1
            )

            # Check for cancellation
            if self.cancel_flags.get(job_id, False):
                # Clean up checkpoint
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                self.jobs[job_id].status = 'cancelled'
                return

            # 8. Load best weights from checkpoint
            print(f"[Training {job_id}] Loading best weights...")
            model.load_weights(str(checkpoint_path))

            # 9. Save model with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f'cnn_model_{timestamp}.h5'
            model_path = settings.RECOGNITION_MODELS_FOLDER / model_filename

            print(f"[Training {job_id}] Saving model to {model_filename}...")
            model.save(str(model_path))

            # Clean up checkpoint
            if checkpoint_path.exists():
                checkpoint_path.unlink()

            # 10. Evaluate on test set
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

            # 11. Update metadata
            final_epoch = len(history.history['accuracy'])
            self._save_model_metadata(
                filename=model_filename,
                accuracy=float(history.history['accuracy'][-1]),
                val_accuracy=float(history.history['val_accuracy'][-1]),
                loss=float(history.history['loss'][-1]),
                val_loss=float(history.history['val_loss'][-1]),
                epochs_trained=final_epoch,
                training_samples=total_samples,
                notes=f"Trained on {timestamp}"
            )

            # 12. Update status: completed
            self.jobs[job_id].status = 'completed'
            self.jobs[job_id].progress = 100.0
            self.jobs[job_id].model_filename = model_filename

            print(f"[Training {job_id}] Training completed!")
            print(f"  - Final training accuracy: {history.history['accuracy'][-1]:.4f}")
            print(f"  - Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
            print(f"  - Epochs trained: {final_epoch}")
            print(f"  - Model saved: {model_filename}")

        except Exception as e:
            print(f"[Training {job_id}] ERROR: {str(e)}")
            self.jobs[job_id].status = 'failed'
            self.jobs[job_id].error = str(e)

    def _save_model_metadata(
        self,
        filename: str,
        accuracy: float,
        val_accuracy: float,
        loss: float,
        val_loss: float,
        epochs_trained: int,
        training_samples: int,
        notes: str = ""
    ):
        """Save or update model metadata in JSON file"""
        # Load existing metadata
        metadata = {"models": []}
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

        # Add new model
        model_info = {
            "filename": filename,
            "trained_date": datetime.now().isoformat(),
            "accuracy": accuracy,
            "val_accuracy": val_accuracy,
            "loss": loss,
            "val_loss": val_loss,
            "epochs_trained": epochs_trained,
            "training_samples": training_samples,
            "notes": notes
        }

        metadata["models"].append(model_info)

        # Save metadata
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def get_models_list(self) -> List[ModelInfo]:
        """List all trained models with metadata"""
        if not self.metadata_path.exists():
            return []

        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        models = []
        for model_data in metadata.get("models", []):
            models.append(ModelInfo(**model_data))

        return models

    def get_active_model(self) -> str:
        """Get filename of currently active model"""
        return "cnn_model.h5"

    def activate_model(self, filename: str):
        """
        Activate a trained model by copying it to cnn_model.h5
        Requires backend restart to reload model
        """
        source_path = settings.RECOGNITION_MODELS_FOLDER / filename

        if not source_path.exists():
            raise FileNotFoundError(f"Model {filename} not found")

        active_path = settings.MODEL_PATH

        # Backup current active model if it exists
        if active_path.exists():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = settings.RECOGNITION_MODELS_FOLDER / f'cnn_model_backup_{timestamp}.h5'
            active_path.rename(backup_path)
            print(f"Backed up active model to {backup_path.name}")

        # Copy selected model to active position
        import shutil
        shutil.copy(str(source_path), str(active_path))
        print(f"Activated model: {filename}")

    def delete_model(self, filename: str):
        """Delete a trained model (cannot delete active model)"""
        if filename == "cnn_model.h5":
            raise ValueError("Cannot delete active model")

        model_path = settings.RECOGNITION_MODELS_FOLDER / filename

        if not model_path.exists():
            raise FileNotFoundError(f"Model {filename} not found")

        model_path.unlink()

        # Remove from metadata
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            metadata["models"] = [
                m for m in metadata.get("models", [])
                if m.get("filename") != filename
            ]

            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

    def get_training_data_stats(self) -> Dict[str, dict]:
        """Get duration of training data per class in training data folder"""
        stats = {}

        for class_name in settings.LABELS:
            class_folder = settings.TRAINING_DATA_FOLDER / class_name
            total_duration = 0.0
            file_count = 0

            if class_folder.exists():
                wav_files = list(class_folder.glob('*.wav'))
                file_count = len(wav_files)

                # Calculate total duration
                for wav_file in wav_files:
                    try:
                        # Get duration without loading entire file
                        import soundfile as sf
                        info = sf.info(str(wav_file))
                        total_duration += info.duration
                    except:
                        # Fallback: assume 2.97s per file if can't read
                        total_duration += 2.97

            stats[class_name] = {
                "count": file_count,
                "duration_sec": total_duration,
                "duration_min": total_duration / 60
            }

        return stats


# Singleton instance
_service = None

def get_training_service() -> TrainingService:
    """Get singleton training service instance"""
    global _service
    if _service is None:
        _service = TrainingService()
    return _service
