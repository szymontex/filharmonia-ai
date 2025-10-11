"""
PyTorch AST Training Service
Replaces Keras CNN training with Audio Spectrogram Transformer
"""
import os
import time
import threading
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import uuid

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import ASTForAudioClassification
from tqdm import tqdm
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# from pytorch_dataset import AudioClassificationDataset  # Not needed for now
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
    test_accuracy: float
    val_accuracy: float
    loss: float
    val_loss: float
    epochs_trained: int
    training_samples: int
    notes: str
    model_id: str = ""
    is_active: bool = False
    per_class_acc: Optional[Dict[str, float]] = None
    measured_train_acc: Optional[float] = None
    dataset_measured_on: Optional[str] = None


class ASTTrainingService:
    """Service for AST model training and management"""

    def __init__(self):
        self.jobs: Dict[str, TrainingStatus] = {}
        self.active_threads: Dict[str, threading.Thread] = {}
        self.metadata_path = settings.RECOGNITION_MODELS_FOLDER / "models_metadata.json"
        self.cancel_flags: Dict[str, bool] = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def start_training_job(self) -> str:
        """Start a new AST training job in background thread"""
        job_id = str(uuid.uuid4())

        # Initialize job status
        self.jobs[job_id] = TrainingStatus(
            job_id=job_id,
            status='preparing',
            current_epoch=0,
            total_epochs=20,
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

    def _prepare_dataset_splits(self, job_id: str) -> Path:
        """
        Prepare train/val/test splits using file-level split (not segment-level)
        Creates folder structure with hardlinks to original files (zero disk space)
        Virtual chunking happens during training via AudioDataset
        """
        from sklearn.model_selection import train_test_split
        import torchaudio

        print(f"[Training {job_id}] [{datetime.now().strftime('%H:%M:%S')}] Starting dataset preparation...", flush=True)

        # Output folder
        dataset_folder = Path(settings.RECOGNITION_MODELS_FOLDER) / f"temp_dataset_{job_id}"
        dataset_folder.mkdir(parents=True, exist_ok=True)
        print(f"[Training {job_id}] Created temp folder: {dataset_folder}", flush=True)

        # Collect all files per class
        all_files = {}
        total_virtual_chunks = {}

        for class_name in settings.LABELS:
            print(f"[Training {job_id}] Scanning {class_name}...", flush=True)
            class_folder = settings.TRAINING_DATA_FOLDER / class_name
            if not class_folder.exists():
                print(f"[Training {job_id}]   WARNING: Folder not found: {class_folder}", flush=True)
                all_files[class_name] = []
                total_virtual_chunks[class_name] = 0
                continue

            # List files first
            wav_files = list(class_folder.glob('*.wav'))
            print(f"[Training {job_id}]   Found {len(wav_files)} WAV files in {class_name}", flush=True)
            all_files[class_name] = wav_files

            # Count virtual chunks (for reporting)
            chunk_count = 0
            for i, wav_file in enumerate(wav_files):
                try:
                    # Log every 10th file
                    if i % 10 == 0:
                        print(f"[Training {job_id}]   Progress: {class_name} [{i+1}/{len(wav_files)}]: {wav_file.name}", flush=True)

                    info = torchaudio.info(str(wav_file))
                    duration = info.num_frames / info.sample_rate
                    chunk_count += int(duration / settings.FRAME_DURATION_SEC)
                except Exception as e:
                    print(f"[Training {job_id}]   ERROR reading {wav_file.name}: {e}", flush=True)
                    chunk_count += 1

            total_virtual_chunks[class_name] = chunk_count
            print(f"[Training {job_id}]   DONE: {class_name}: {len(wav_files)} files -> {chunk_count} virtual chunks", flush=True)

        # Update job status
        self.jobs[job_id].samples_per_class = total_virtual_chunks

        # Split files 80/10/10 per class (on FILE level, not chunk level!)
        print(f"[Training {job_id}] Splitting files into train/val/test...", flush=True)
        for class_name, files in all_files.items():
            if len(files) == 0:
                continue

            print(f"[Training {job_id}]   Creating splits for {class_name} ({len(files)} files)...", flush=True)

            # Create folders
            for split in ['train', 'val', 'test']:
                folder = dataset_folder / split / class_name
                folder.mkdir(parents=True, exist_ok=True)

            # Split files
            if len(files) < 3:
                # Too few files - put all in train
                train_files, val_files, test_files = files, [], []
                print(f"[Training {job_id}]   WARNING: Only {len(files)} files in {class_name} - putting all in train", flush=True)
            else:
                train_files, temp_files = train_test_split(files, test_size=0.2, random_state=42)
                if len(temp_files) < 2:
                    val_files, test_files = temp_files, []
                else:
                    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
                print(f"[Training {job_id}]   Split {class_name}: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}", flush=True)

            # Create hardlinks (zero disk space, instant)
            print(f"[Training {job_id}]   Creating hardlinks for {class_name}...", flush=True)
            for split_name, split_files in [('train', train_files), ('val', val_files), ('test', test_files)]:
                target_folder = dataset_folder / split_name / class_name
                for wav_file in split_files:
                    link_path = target_folder / wav_file.name
                    try:
                        # Try hardlink (zero space, Windows compatible)
                        os.link(str(wav_file), str(link_path))
                    except:
                        # Fallback: copy file if hardlink fails
                        shutil.copy(str(wav_file), str(link_path))

        print(f"[Training {job_id}] Dataset splits ready at {datetime.now().strftime('%H:%M:%S')}", flush=True)
        return dataset_folder

    def _train_model_background(self, job_id: str):
        """Background AST training task"""
        try:
            start_time = time.time()

            # 1. Prepare dataset splits (file-level, uses hardlinks)
            self.jobs[job_id].status = 'preparing'
            dataset_dir = self._prepare_dataset_splits(job_id)

            # Check cancellation
            if self.cancel_flags.get(job_id, False):
                shutil.rmtree(dataset_dir)
                return

            # 2. Create dataloaders with virtual chunking
            print(f"[Training {job_id}] [{datetime.now().strftime('%H:%M:%S')}] Creating dataloaders...", flush=True)
            from pytorch_dataset import create_dataloaders

            print(f"[Training {job_id}]   Importing AudioDataset...", flush=True)
            loaders = create_dataloaders(
                dataset_dir,
                batch_size=8,
                feature_type='melspec',
                use_weighted_sampling=True,
                balance_strength=0.75,
                enable_chunking=True,
                num_workers=0  # CHANGED: 0 to avoid multiprocessing deadlock on network drive
            )
            print(f"[Training {job_id}]   DataLoaders created successfully", flush=True)

            train_size = len(loaders['train'].dataset)
            val_size = len(loaders['val'].dataset)
            test_size = len(loaders['test'].dataset)

            print(f"[Training {job_id}] Dataset sizes: train={train_size}, val={val_size}, test={test_size}", flush=True)

            # Check cancellation
            if self.cancel_flags.get(job_id, False):
                shutil.rmtree(dataset_dir)
                return

            # 3. Load AST model
            print(f"[Training {job_id}] [{datetime.now().strftime('%H:%M:%S')}] Loading AST model from HuggingFace...", flush=True)
            model = ASTForAudioClassification.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593",
                num_labels=5,
                ignore_mismatched_sizes=True
            )
            print(f"[Training {job_id}]   Model loaded, moving to {self.device}...", flush=True)
            model = model.to(self.device)
            print(f"[Training {job_id}]   Model ready on {self.device}", flush=True)

            # 4. Setup training
            print(f"[Training {job_id}] Setting up training components...", flush=True)
            class_weights = loaders['train'].dataset.get_class_weights().to(self.device)
            print(f"[Training {job_id}]   Class weights: {class_weights.tolist()}", flush=True)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = AdamW(model.parameters(), lr=1e-4)
            print(f"[Training {job_id}]   Optimizer ready (AdamW lr=1e-4)", flush=True)

            # 5. Training loop
            self.jobs[job_id].status = 'training'
            num_epochs = 5
            self.jobs[job_id].total_epochs = num_epochs
            best_val_acc = 0.0
            best_epoch = 0

            print(f"[Training {job_id}] [{datetime.now().strftime('%H:%M:%S')}] Starting training ({num_epochs} epochs)...", flush=True)

            for epoch in range(num_epochs):
                # Check cancellation
                if self.cancel_flags.get(job_id, False):
                    shutil.rmtree(dataset_dir)
                    return

                # Train
                train_loss, train_acc = self._train_epoch(
                    model, loaders['train'], optimizer, criterion, job_id
                )

                # Validate
                val_loss, val_acc = self._validate(
                    model, loaders['val'], criterion
                )

                # Update job status
                self.jobs[job_id].current_epoch = epoch + 1
                self.jobs[job_id].training_acc = train_acc
                self.jobs[job_id].training_loss = train_loss
                self.jobs[job_id].val_acc = val_acc
                self.jobs[job_id].val_loss = val_loss
                self.jobs[job_id].progress = (epoch + 1) / num_epochs * 100
                self.jobs[job_id].time_elapsed_sec = int(time.time() - start_time)

                # Estimate remaining time
                if epoch > 0:
                    elapsed = time.time() - start_time
                    avg_per_epoch = elapsed / (epoch + 1)
                    remaining = int(avg_per_epoch * (num_epochs - epoch - 1))
                    self.jobs[job_id].time_remaining_sec = remaining

                print(f"[Training {job_id}] Epoch {epoch + 1}/{num_epochs}: train_acc={train_acc:.2f}%, val_acc={val_acc:.2f}%")

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    model_filename = f'ast_{timestamp}.pth'
                    temp_save_path = settings.RECOGNITION_MODELS_FOLDER / model_filename

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_acc': train_acc,
                        'val_acc': val_acc,
                    }, temp_save_path)

                    self.jobs[job_id].model_filename = model_filename

                # Early stopping - after 2 epochs without improvement
                if epoch > 2 and epoch - best_epoch >= 2:
                    print(f"[Training {job_id}] Early stopping - no improvement for 2 epochs")
                    break

            # 6. Final test evaluation
            test_loss, test_acc = self._validate(model, loaders['test'], criterion)

            # 7. Save metadata
            self._save_model_metadata(
                filename=self.jobs[job_id].model_filename,
                test_accuracy=test_acc,
                val_accuracy=best_val_acc,
                loss=train_loss,
                val_loss=val_loss,
                epochs_trained=epoch + 1,
                training_samples=train_size,
                notes=f"AST trained on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )

            # 8. Cleanup temporary dataset
            shutil.rmtree(dataset_dir)

            # 9. Complete
            self.jobs[job_id].status = 'completed'
            self.jobs[job_id].progress = 100.0
            print(f"[Training {job_id}] Training completed! Test accuracy: {test_acc:.2f}%")

        except Exception as e:
            print(f"[Training {job_id}] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            self.jobs[job_id].status = 'failed'
            self.jobs[job_id].error = str(e)

    def _train_epoch(self, model, loader, optimizer, criterion, job_id):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        total_batches = len(loader)
        print(f"[Training {job_id}] Starting epoch with {total_batches} batches...", flush=True)

        batch_start_time = time.time()

        for batch_idx, (features, labels) in enumerate(loader):
            # Log first 5 batches explicitly with timing
            if batch_idx < 5:
                print(f"[Training {job_id}]   Loading batch {batch_idx + 1}...", flush=True)

            # Check cancellation
            if self.cancel_flags.get(job_id, False):
                raise InterruptedError("Training cancelled")

            # Preprocess for AST
            features = features.transpose(1, 2)
            batch_size, time_frames, freq = features.shape

            if batch_idx < 5:
                print(f"[Training {job_id}]   Batch {batch_idx + 1} loaded! Shape: {batch_size}x{time_frames}x{freq}", flush=True)
                print(f"[Training {job_id}]   Preprocessing for AST (padding/trimming to 1024 frames)...", flush=True)

            if time_frames < 1024:
                pad_size = 1024 - time_frames
                features = torch.nn.functional.pad(features, (0, 0, 0, pad_size))
            elif time_frames > 1024:
                features = features[:, :1024, :]

            features = features.to(self.device)
            labels = labels.to(self.device)

            if batch_idx < 5:
                print(f"[Training {job_id}]   Running forward pass (batch {batch_idx + 1})...", flush=True)

            optimizer.zero_grad()
            outputs = model(features).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx < 5:
                batch_time = time.time() - batch_start_time
                print(f"[Training {job_id}]   Batch {batch_idx + 1} complete! Loss: {loss.item():.4f}, Time: {batch_time:.2f}s", flush=True)
                batch_start_time = time.time()

            # Progress logging every 10 batches (HIGH FREQUENCY)
            if (batch_idx + 1) % 10 == 0:
                batch_acc = 100 * correct / total
                avg_loss = running_loss / (batch_idx + 1)
                progress_pct = 100 * (batch_idx + 1) / total_batches
                print(f"[Training {job_id}]   Batch {batch_idx + 1}/{total_batches} ({progress_pct:.1f}%) - loss: {avg_loss:.4f}, acc: {batch_acc:.2f}%", flush=True)

        epoch_loss = running_loss / len(loader)
        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc

    def _validate(self, model, loader, criterion):
        """Validate model"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in loader:
                features = features.transpose(1, 2)
                batch_size, time, freq = features.shape
                if time < 1024:
                    pad_size = 1024 - time
                    features = torch.nn.functional.pad(features, (0, 0, 0, pad_size))
                elif time > 1024:
                    features = features[:, :1024, :]

                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = model(features).logits
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / len(loader)
        val_acc = 100 * correct / total

        return val_loss, val_acc

    def _save_model_metadata(
        self,
        filename: str,
        test_accuracy: float,
        val_accuracy: float,
        loss: float,
        val_loss: float,
        epochs_trained: int,
        training_samples: int,
        notes: str = ""
    ):
        """Save model metadata to JSON"""
        from app.services.model_registry import generate_model_id

        metadata = {"active_model": None, "models": []}
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

        # Ensure active_model field exists
        if "active_model" not in metadata:
            metadata["active_model"] = None

        # Generate model_id from filename
        model_id = generate_model_id(filename)

        model_info = {
            "filename": filename,
            "model_id": model_id,
            "trained_date": datetime.now().isoformat(),
            "test_accuracy": test_accuracy,
            "val_accuracy": val_accuracy,
            "loss": loss,
            "val_loss": val_loss,
            "epochs_trained": epochs_trained,
            "training_samples": training_samples,
            "notes": notes,
            "is_active": False
        }

        metadata["models"].append(model_info)

        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def get_models_list(self) -> List[ModelInfo]:
        """List all trained models"""
        if not self.metadata_path.exists():
            return []

        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Error reading metadata: {e}")
            return []

        models = []
        for model_data in metadata.get("models", []):
            try:
                # Handle both old (accuracy) and new (test_accuracy) format
                if 'accuracy' in model_data and 'test_accuracy' not in model_data:
                    model_data['test_accuracy'] = model_data.pop('accuracy')

                # Ensure all required fields exist with defaults
                model_data.setdefault('test_accuracy', 0.0)
                model_data.setdefault('val_accuracy', 0.0)
                model_data.setdefault('loss', 0.0)
                model_data.setdefault('val_loss', 0.0)
                model_data.setdefault('epochs_trained', 0)
                model_data.setdefault('training_samples', 0)
                model_data.setdefault('notes', '')
                model_data.setdefault('model_id', '')
                model_data.setdefault('is_active', False)
                model_data.setdefault('per_class_acc', None)
                model_data.setdefault('measured_train_acc', None)
                model_data.setdefault('dataset_measured_on', None)

                models.append(ModelInfo(**model_data))
            except Exception as e:
                print(f"Error parsing model metadata: {e}")
                print(f"Model data: {model_data}")
                continue

        return models

    def get_active_model(self) -> str:
        """Get model_id of currently active model (from metadata)"""
        if not self.metadata_path.exists():
            return ""

        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            return metadata.get("active_model", "")
        except Exception as e:
            print(f"Error reading active_model from metadata: {e}")
            return ""

    def activate_model(self, filename: str):
        """Activate a trained model by copying to ast_active.pth"""
        from app.services.model_registry import set_active_model, generate_model_id

        source_path = settings.RECOGNITION_MODELS_FOLDER / filename

        if not source_path.exists():
            raise FileNotFoundError(f"Model {filename} not found")

        active_path = settings.AST_MODEL_PATH

        # Backup current active model if exists
        if active_path.exists():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = settings.RECOGNITION_MODELS_FOLDER / f'ast_backup_{timestamp}.pth'
            shutil.copy(str(active_path), str(backup_path))
            print(f"Backed up active model to {backup_path.name}")

        # Copy selected model to active position
        shutil.copy(str(source_path), str(active_path))
        print(f"Activated model: {filename}")

        # Update metadata using model_registry (uses model_id, not filename!)
        model_id = generate_model_id(filename)
        success = set_active_model(model_id)

        if not success:
            print(f"[WARNING] Could not find model_id '{model_id}' in metadata")
            # Fallback: direct metadata update (should not happen if training worked correctly)
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                metadata['active_model'] = model_id  # Use model_id, not filename
                with open(self.metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

    def delete_model(self, filename: str):
        """Delete a trained model"""
        if filename == "ast_active.pth":
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

    def measure_model_accuracy(self, filename: str) -> Dict:
        """
        Measure true train/val/test accuracy for a model
        Updates metadata with measured values including per-class accuracy
        """
        from pytorch_dataset import create_dataloaders

        model_path = settings.RECOGNITION_MODELS_FOLDER / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model {filename} not found")

        print(f"[Measure] Loading model {filename}...")

        # Load dataset - prefer natural data over artificial balanced
        dataset_dir = settings.get_dataset_dir("direct_from_source")
        if not dataset_dir.exists():
            # Fallback to full_balanced if direct_from_source doesn't exist
            dataset_dir = settings.get_dataset_dir("full_balanced")
            if not dataset_dir.exists():
                raise FileNotFoundError(
                    f"Dataset not found. Expected one of:\n"
                    f"  - {settings.get_dataset_dir('direct_from_source')}\n"
                    f"  - {settings.get_dataset_dir('full_balanced')}\n"
                    f"Configure FILHARMONIA_BASE_DIR environment variable or create datasets."
                )

        print(f"[Measure] Using dataset: {dataset_dir}")

        loaders = create_dataloaders(
            dataset_dir,
            batch_size=8,
            feature_type='melspec',
            use_weighted_sampling=False,
            num_workers=0
        )

        # Load model
        model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            num_labels=5,
            ignore_mismatched_sizes=True
        )

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)

        # Measure accuracies with per-class breakdown
        print(f"[Measure] Measuring train accuracy...")
        train_acc, _ = self._measure_accuracy_with_perclass(model, loaders['train'])

        print(f"[Measure] Measuring val accuracy...")
        val_acc, _ = self._measure_accuracy_with_perclass(model, loaders['val'])

        print(f"[Measure] Measuring test accuracy...")
        test_acc, per_class_acc = self._measure_accuracy_with_perclass(model, loaders['test'])

        print(f"[Measure] Results: train={train_acc:.2f}%, val={val_acc:.2f}%, test={test_acc:.2f}%")
        print(f"[Measure] Per-class test accuracy:")
        for class_name, acc in per_class_acc.items():
            print(f"  {class_name}: {acc:.2f}%")

        # Update metadata
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            for model_info in metadata.get("models", []):
                if model_info.get("filename") == filename:
                    model_info["test_accuracy"] = test_acc / 100
                    model_info["val_accuracy"] = val_acc / 100
                    model_info["training_samples"] = len(loaders['train'].dataset)

                    # Store per-class accuracy as separate field
                    model_info["per_class_acc"] = {cls: acc / 100 for cls, acc in per_class_acc.items()}
                    model_info["measured_train_acc"] = train_acc / 100
                    model_info["dataset_measured_on"] = dataset_dir.name

                    break

            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        return {
            "filename": filename,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "per_class_acc": per_class_acc,
            "training_samples": len(loaders['train'].dataset),
            "dataset_used": dataset_dir.name
        }

    def _measure_accuracy(self, model, dataloader) -> float:
        """Helper to measure accuracy on a dataset (legacy - use _measure_accuracy_with_perclass instead)"""
        overall_acc, _ = self._measure_accuracy_with_perclass(model, dataloader)
        return overall_acc

    def _measure_accuracy_with_perclass(self, model, dataloader) -> tuple[float, dict]:
        """
        Helper to measure accuracy on a dataset with per-class breakdown
        Returns: (overall_accuracy, per_class_accuracy_dict)
        """
        from collections import defaultdict

        model.eval()
        correct = 0
        total = 0

        # Per-class tracking
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        with torch.no_grad():
            for features, labels in dataloader:
                # Preprocess for AST
                features = features.transpose(1, 2)
                batch_size, time, freq = features.shape

                if time < 1024:
                    pad_size = 1024 - time
                    features = torch.nn.functional.pad(features, (0, 0, 0, pad_size))
                elif time > 1024:
                    features = features[:, :1024, :]

                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = model(features).logits
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Per-class accuracy
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == labels[i]:
                        class_correct[label] += 1

        overall_acc = 100 * correct / total

        # Calculate per-class accuracy
        per_class_acc = {}
        class_names = ['APPLAUSE', 'MUSIC', 'PUBLIC', 'SPEECH', 'TUNING']
        for i, class_name in enumerate(class_names):
            if class_total[i] > 0:
                per_class_acc[class_name] = 100.0 * class_correct[i] / class_total[i]
            else:
                per_class_acc[class_name] = 0.0

        return overall_acc, per_class_acc

    def get_training_data_stats(self) -> Dict[str, dict]:
        """Get duration of training data per class"""
        import soundfile as sf

        stats = {}

        for class_name in settings.LABELS:
            class_folder = settings.TRAINING_DATA_FOLDER / class_name
            total_duration = 0.0
            file_count = 0

            if class_folder.exists():
                wav_files = list(class_folder.glob('*.wav'))
                file_count = len(wav_files)

                for wav_file in wav_files:
                    try:
                        info = sf.info(str(wav_file))
                        total_duration += info.duration
                    except:
                        # Fallback: assume 2.97s per file
                        total_duration += 2.97

            stats[class_name] = {
                "count": file_count,
                "duration_sec": total_duration,
                "duration_min": total_duration / 60
            }

        return stats


# Singleton instance
_service = None

def get_ast_training_service() -> ASTTrainingService:
    """Get singleton AST training service"""
    global _service
    if _service is None:
        _service = ASTTrainingService()
    return _service
