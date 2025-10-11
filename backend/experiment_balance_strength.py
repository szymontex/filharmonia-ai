"""
Overnight Experiment: Balance Strength Testing

Tests different balance_strength values (0.0, 0.25, 0.5, 0.75, 1.0)
Each: 5 epochs training directly from TRAINING DATA/DATA
Total time: ~10-12 hours

Results saved to: Y:/!_FILHARMONIA/ML_EXPERIMENTS/balance_experiments/
"""
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from app.config import settings

# Import PyTorch dataset (will be modified to support balance_strength)
from pytorch_dataset import AudioDataset, create_dataloaders

# Import AST model
from transformers import ASTForAudioClassification


class BalancedAudioDataset(AudioDataset):
    """Extended AudioDataset with balance_strength parameter"""

    def get_sample_weights(self, balance_strength: float = 1.0) -> torch.Tensor:
        """
        Get per-sample weights for WeightedRandomSampler

        Args:
            balance_strength:
                0.0 = no balancing (natural distribution)
                1.0 = full balancing (all classes equal weight)
                0.5 = 50/50 compromise
        """
        class_weights = self.get_class_weights()

        if balance_strength == 0.0:
            # No balancing - all samples have equal weight
            return torch.ones(len(self.labels))

        # Soften the weights using power
        # Lower power = less aggressive balancing
        softened_weights = torch.pow(class_weights, balance_strength)

        sample_weights = torch.tensor([softened_weights[label] for label in self.labels])
        return sample_weights


def create_dataloaders_with_balance(
    dataset_dir: Path,
    batch_size: int = 16,
    balance_strength: float = 1.0,
    feature_type: str = 'melspec',
    num_workers: int = 0
):
    """Create dataloaders with configurable balance_strength"""
    from torch.utils.data import DataLoader, WeightedRandomSampler

    dataset_dir = Path(dataset_dir)

    # Create datasets
    train_dataset = BalancedAudioDataset(
        dataset_dir / 'train',
        feature_type=feature_type
    )

    val_dataset = BalancedAudioDataset(
        dataset_dir / 'val',
        feature_type=feature_type
    )

    test_dataset = BalancedAudioDataset(
        dataset_dir / 'test',
        feature_type=feature_type
    )

    # Create sampler with balance_strength
    sample_weights = train_dataset.get_sample_weights(balance_strength=balance_strength)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,  # Don't shuffle when using sampler
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def prepare_direct_dataset(output_dir: Path, source_dir: Path, train_ratio=0.8, val_ratio=0.1):
    """
    Prepare dataset directly from TRAINING DATA/DATA
    No duplication, just split into train/val/test
    """
    import random
    import shutil

    print("\n" + "=" * 80)
    print("PREPARING DATASET FROM TRAINING DATA/DATA")
    print("=" * 80)

    # Clear output dir
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Create structure
    for split in ['train', 'val', 'test']:
        for class_name in settings.LABELS:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    # Split and copy files
    for class_name in settings.LABELS:
        class_dir = source_dir / class_name
        if not class_dir.exists():
            print(f"WARNING: {class_name} folder not found!")
            continue

        # Get all WAV files and validate them
        all_files_raw = list(class_dir.glob('*.wav'))
        all_files = []

        # Validate files - skip empty/corrupted ones
        import torchaudio
        for wav_file in all_files_raw:
            try:
                waveform, sr = torchaudio.load(str(wav_file))
                if waveform.shape[1] > 0:  # Skip empty files
                    all_files.append(wav_file)
                else:
                    print(f"  Skipping empty file: {wav_file.name}")
            except Exception as e:
                print(f"  Skipping corrupted file: {wav_file.name} ({str(e)})")

        random.shuffle(all_files)

        train_size = int(len(all_files) * train_ratio)
        val_size = int(len(all_files) * val_ratio)

        train_files = all_files[:train_size]
        val_files = all_files[train_size:train_size + val_size]
        test_files = all_files[train_size + val_size:]

        print(f"\n{class_name}:")
        print(f"  Total: {len(all_files)} files")
        print(f"  Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

        # Copy files
        for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            for file in files:
                dest = output_dir / split / class_name / file.name
                shutil.copy(file, dest)

    print("\n" + "=" * 80)
    print(f"Dataset ready: {output_dir}")
    print("=" * 80)


def measure_accuracy(model, dataloader, device) -> Dict:
    """Measure accuracy and per-class accuracy"""
    model.eval()
    correct = 0
    total = 0

    # Per-class tracking
    class_correct = {i: 0 for i in range(5)}
    class_total = {i: 0 for i in range(5)}

    with torch.no_grad():
        for features, labels in dataloader:
            # AST preprocessing: transpose and resize
            features = features.transpose(1, 2)

            # Resize to AST expected size: (batch, 1024, 128)
            batch_size, time, freq = features.shape
            if time < 1024:
                pad_size = 1024 - time
                features = torch.nn.functional.pad(features, (0, 0, 0, pad_size))
            elif time > 1024:
                features = features[:, :1024, :]

            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(features)
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class
            for i in range(5):
                mask = labels == i
                if mask.sum() > 0:
                    class_correct[i] += (predicted[mask] == labels[mask]).sum().item()
                    class_total[i] += mask.sum().item()

    accuracy = 100.0 * correct / total

    # Per-class accuracy
    class_names = ['APPLAUSE', 'MUSIC', 'PUBLIC', 'SPEECH', 'TUNING']
    per_class_acc = {}
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            per_class_acc[name] = 100.0 * class_correct[i] / class_total[i]
        else:
            per_class_acc[name] = 0.0

    return {
        'overall': accuracy,
        'per_class': per_class_acc
    }


def train_one_experiment(
    balance_strength: float,
    dataset_dir: Path,
    num_epochs: int,
    output_dir: Path
) -> Dict:
    """Train one experiment with given balance_strength"""

    print("\n" + "=" * 80)
    print(f"EXPERIMENT: balance_strength = {balance_strength}")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    loaders = create_dataloaders_with_balance(
        dataset_dir,
        batch_size=16,
        balance_strength=balance_strength,
        feature_type='melspec',
        num_workers=0
    )

    # Load model
    print("\nLoading AST model...")
    model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_labels=5,
        ignore_mismatched_sizes=True
    )
    model = model.to(device)

    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training history
    history = {
        'train_acc': [],
        'val_acc': [],
        'epoch_times': []
    }

    # Training loop
    for epoch in range(num_epochs):
        epoch_start = datetime.now()
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (features, labels) in enumerate(loaders['train']):
            # AST preprocessing: transpose and resize
            # Our melspec: (batch, freq, time) → AST expects: (batch, time, freq)
            features = features.transpose(1, 2)

            # Resize to AST expected size: (batch, 1024, 128)
            batch_size, time, freq = features.shape
            if time < 1024:
                # Pad time dimension
                pad_size = 1024 - time
                features = torch.nn.functional.pad(features, (0, 0, 0, pad_size))
            elif time > 1024:
                # Trim time dimension
                features = features[:, :1024, :]

            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(features)
            logits = outputs.logits

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(loaders['train'])}: loss={loss.item():.4f}")

        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(loaders['train'])

        # Validation
        val_results = measure_accuracy(model, loaders['val'], device)
        val_acc = val_results['overall']

        epoch_time = (datetime.now() - epoch_start).total_seconds()

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(epoch_time)

        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%")
        print(f"  Epoch Time: {epoch_time:.1f}s")

    # Final test
    print("\n--- Final Test ---")
    test_results = measure_accuracy(model, loaders['test'], device)

    print(f"Test Accuracy: {test_results['overall']:.2f}%")
    print("Per-class accuracy:")
    for cls, acc in test_results['per_class'].items():
        print(f"  {cls}: {acc:.2f}%")

    # Save model
    model_filename = f"ast_balance{balance_strength:.2f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    model_path = output_dir / model_filename

    torch.save({
        'model_state_dict': model.state_dict(),
        'balance_strength': balance_strength,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_results['overall'],
        'per_class_acc': test_results['per_class'],
        'num_epochs': num_epochs,
        'history': history
    }, model_path)

    print(f"\nModel saved: {model_path}")

    return {
        'balance_strength': balance_strength,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_results['overall'],
        'per_class_acc': test_results['per_class'],
        'history': history,
        'model_path': str(model_path),
        'total_time_minutes': sum(history['epoch_times']) / 60
    }


def main():
    """Run overnight experiments"""

    print("\n" + "=" * 80)
    print("OVERNIGHT BALANCE STRENGTH EXPERIMENTS")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Config
    SOURCE_DIR = Path("Y:/!_FILHARMONIA/TRAINING DATA/DATA")
    DATASET_DIR = Path("Y:/!_FILHARMONIA/ML_EXPERIMENTS/datasets/direct_from_source")
    OUTPUT_DIR = Path("Y:/!_FILHARMONIA/ML_EXPERIMENTS/balance_experiments")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    NUM_EPOCHS = 5
    BALANCE_STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Prepare dataset once (80/10/10 split)
    print("\nStep 1: Preparing dataset from TRAINING DATA/DATA...")
    import random
    random.seed(42)  # Reproducible split
    prepare_direct_dataset(DATASET_DIR, SOURCE_DIR)

    # Run experiments
    print("\nStep 2: Running experiments...")
    results = []

    for balance_strength in BALANCE_STRENGTHS:
        try:
            result = train_one_experiment(
                balance_strength=balance_strength,
                dataset_dir=DATASET_DIR,
                num_epochs=NUM_EPOCHS,
                output_dir=OUTPUT_DIR
            )
            results.append(result)

            # Save intermediate results
            results_path = OUTPUT_DIR / "experiment_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\nIntermediate results saved: {results_path}")

        except Exception as e:
            print(f"\n❌ ERROR in experiment balance_strength={balance_strength}:")
            print(str(e))
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    for result in results:
        bs = result['balance_strength']
        test_acc = result['test_acc']
        time_min = result['total_time_minutes']

        print(f"\nBalance Strength: {bs:.2f}")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"  Training Time: {time_min:.1f} min")
        print(f"  Per-class accuracy:")
        for cls, acc in result['per_class_acc'].items():
            print(f"    {cls}: {acc:.2f}%")

    # Best model
    best_result = max(results, key=lambda x: x['test_acc'])
    print("\n" + "=" * 80)
    print("🏆 BEST MODEL")
    print("=" * 80)
    print(f"Balance Strength: {best_result['balance_strength']:.2f}")
    print(f"Test Accuracy: {best_result['test_acc']:.2f}%")
    print(f"Model: {best_result['model_path']}")

    print(f"\n✅ All experiments complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved: {OUTPUT_DIR / 'experiment_results.json'}")


if __name__ == "__main__":
    main()
