"""
Test different training configurations to find optimal GPU utilization
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.services.ast_training import ASTTrainingService
import torch

def test_config(batch_size, num_workers, test_duration_batches=50):
    """Test a specific configuration"""
    print("\n" + "="*80)
    print(f"TESTING: batch_size={batch_size}, num_workers={num_workers}")
    print("="*80)

    service = ASTTrainingService()

    # Mock job
    import uuid
    from datetime import datetime
    from pytorch_dataset import create_dataloaders
    from app.config import settings
    from sklearn.model_selection import train_test_split
    import torchaudio
    import shutil
    import os

    job_id = str(uuid.uuid4())

    # Prepare dataset splits
    print(f"[Test] Preparing dataset...")
    dataset_folder = Path(settings.RECOGNITION_MODELS_FOLDER) / f"temp_test_{job_id}"
    dataset_folder.mkdir(parents=True, exist_ok=True)

    all_files = {}
    for class_name in settings.LABELS:
        class_folder = settings.TRAINING_DATA_FOLDER / class_name
        if class_folder.exists():
            wav_files = list(class_folder.glob('*.wav'))
            all_files[class_name] = wav_files

    # Split files
    for class_name, files in all_files.items():
        if len(files) == 0:
            continue

        for split in ['train', 'val', 'test']:
            folder = dataset_folder / split / class_name
            folder.mkdir(parents=True, exist_ok=True)

        if len(files) < 3:
            train_files, val_files, test_files = files, [], []
        else:
            train_files, temp_files = train_test_split(files, test_size=0.2, random_state=42)
            if len(temp_files) < 2:
                val_files, test_files = temp_files, []
            else:
                val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

        # Create hardlinks
        for split_name, split_files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            target_folder = dataset_folder / split_name / class_name
            for wav_file in split_files:
                link_path = target_folder / wav_file.name
                try:
                    os.link(str(wav_file), str(link_path))
                except:
                    shutil.copy(str(wav_file), str(link_path))

    print(f"[Test] Creating dataloaders...")
    loaders = create_dataloaders(
        dataset_folder,
        batch_size=batch_size,
        feature_type='melspec',
        use_weighted_sampling=True,
        balance_strength=0.75,
        enable_chunking=True,
        num_workers=num_workers
    )

    print(f"[Test] Dataset size: {len(loaders['train'].dataset)} samples")
    print(f"[Test] Batches per epoch: {len(loaders['train'])}")

    # Load model
    print(f"[Test] Loading AST model...")
    from transformers import ASTForAudioClassification
    import torch.nn as nn
    from torch.optim import AdamW

    model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_labels=5,
        ignore_mismatched_sizes=True
    )
    model = model.to(service.device)

    class_weights = loaders['train'].dataset.get_class_weights().to(service.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Test training for N batches
    print(f"[Test] Starting training test ({test_duration_batches} batches)...")
    model.train()

    start_time = time.time()
    batch_times = []

    for batch_idx, (features, labels) in enumerate(loaders['train']):
        if batch_idx >= test_duration_batches:
            break

        batch_start = time.time()

        # Preprocess
        features = features.transpose(1, 2)
        batch_size_actual, time_frames, freq = features.shape
        if time_frames < 1024:
            pad_size = 1024 - time_frames
            features = torch.nn.functional.pad(features, (0, 0, 0, pad_size))
        elif time_frames > 1024:
            features = features[:, :1024, :]

        features = features.to(service.device)
        labels = labels.to(service.device)

        # Forward + backward
        optimizer.zero_grad()
        outputs = model(features).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        if (batch_idx + 1) % 10 == 0:
            avg_time = sum(batch_times[-10:]) / len(batch_times[-10:])
            print(f"[Test] Batch {batch_idx + 1}/{test_duration_batches} - {avg_time:.3f}s/batch - loss: {loss.item():.4f}")

    elapsed = time.time() - start_time
    avg_batch_time = sum(batch_times) / len(batch_times)

    # Cleanup
    del model
    del loaders
    torch.cuda.empty_cache()
    shutil.rmtree(dataset_folder)

    print(f"\n[Test] RESULTS:")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Average batch time: {avg_batch_time:.3f}s")
    print(f"  Batches/sec: {1/avg_batch_time:.2f}")
    print(f"  Estimated epoch time: {avg_batch_time * len(loaders['train']) / 60:.1f} minutes")

    return {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'avg_batch_time': avg_batch_time,
        'batches_per_sec': 1/avg_batch_time,
        'estimated_epoch_min': avg_batch_time * 1383 / 60  # Assuming ~1383 batches
    }


if __name__ == "__main__":
    print("GPU Utilization Test for Training Configurations")
    print("=" * 80)

    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                          capture_output=True, text=True)
    print(f"GPU: {result.stdout.strip()}")

    results = []

    # Test configurations
    configs = [
        (32, 0),   # Baseline
        (32, 4),   # Current
        (64, 0),   # Larger batch, single worker
        (64, 4),   # Larger batch, multi worker
        (128, 0),  # Very large batch
    ]

    for batch_size, num_workers in configs:
        try:
            result = test_config(batch_size, num_workers, test_duration_batches=50)
            results.append(result)

            # Show GPU stats after each test
            import subprocess
            gpu_result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,power.draw,temperature.gpu', '--format=csv,noheader'],
                capture_output=True, text=True
            )
            print(f"  GPU after test: {gpu_result.stdout.strip()}")

        except Exception as e:
            print(f"[Test] ERROR: {e}")
            import traceback
            traceback.print_exc()

        print("\nWaiting 10s before next test...")
        time.sleep(10)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL TESTS")
    print("="*80)
    print(f"{'Batch Size':<12} {'Workers':<10} {'Batch Time':<15} {'Epoch Time (min)':<20}")
    print("-" * 80)

    for r in results:
        print(f"{r['batch_size']:<12} {r['num_workers']:<10} {r['avg_batch_time']:.3f}s/batch      {r['estimated_epoch_min']:.1f} min")

    # Find best
    if results:
        best = min(results, key=lambda x: x['estimated_epoch_min'])
        print("\n" + "="*80)
        print(f"BEST CONFIG: batch_size={best['batch_size']}, num_workers={best['num_workers']}")
        print(f"Estimated epoch time: {best['estimated_epoch_min']:.1f} minutes")
        print("="*80)
