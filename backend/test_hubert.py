"""
Test HuBERT (Facebook)
Self-supervised audio representation learning
Similar to Wav2Vec but better for audio events
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
from tqdm import tqdm
import time
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from pytorch_dataset import create_dataloaders


# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def train_epoch(model, loader, optimizer, criterion, device, feature_extractor):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for features, labels in pbar:
        # HuBERT expects raw waveform (like Wav2Vec)
        # features shape: (batch, samples)

        # Process with feature extractor
        inputs = feature_extractor(
            [f.cpu().numpy() for f in features],
            sampling_rate=48000,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_values).logits
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device, feature_extractor):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in tqdm(loader, desc="Validation"):
            # Process with feature extractor
            inputs = feature_extractor(
                [f.cpu().numpy() for f in features],
                sampling_rate=48000,
                return_tensors="pt",
                padding=True
            )

            input_values = inputs.input_values.to(device)
            labels = labels.to(device)

            outputs = model(input_values).logits
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(loader)
    val_acc = 100 * correct / total

    return val_loss, val_acc


def main():
    print("=" * 80)
    print("TESTING HUBERT")
    print("=" * 80)

    # Hyperparameters
    BATCH_SIZE = 4  # Small batch - HuBERT is memory hungry
    LEARNING_RATE = 1e-4
    EPOCHS = 15
    NUM_CLASSES = 5

    dataset_dir = Path(r"Y:\!_FILHARMONIA\ML_EXPERIMENTS\datasets\small_balanced")

    # Create dataloaders with raw waveform
    print("\nCreating dataloaders...")
    loaders = create_dataloaders(
        dataset_dir,
        batch_size=BATCH_SIZE,
        feature_type='waveform',  # RAW WAVEFORM
        use_weighted_sampling=True,
        num_workers=0
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(loaders['train'].dataset)} samples")
    print(f"  Val: {len(loaders['val'].dataset)} samples")
    print(f"  Test: {len(loaders['test'].dataset)} samples")

    # Load pretrained HuBERT
    print("\nLoading pretrained HuBERT...")
    model_name = "facebook/hubert-base-ls960"

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name,
        sampling_rate=48000
    )

    model = HubertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    model = model.to(device)

    # Loss function with class weights
    class_weights = loaders['train'].dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    best_val_acc = 0.0
    history = defaultdict(list)
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_epoch(
            model, loaders['train'], optimizer, criterion, device, feature_extractor
        )

        # Validate
        val_loss, val_acc = validate(
            model, loaders['val'], criterion, device, feature_extractor
        )

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = Path(r"Y:\!_FILHARMONIA\ML_EXPERIMENTS\models\hubert_best.pth")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path)
            print(f"  -> Saved best model (val_acc: {val_acc:.2f}%)")

        # Early stopping
        if epoch > 8 and val_acc < best_val_acc - 5:
            print(f"\nEarly stopping - no improvement")
            break

    training_time = time.time() - start_time

    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 80)

    # Load best model
    checkpoint = torch.load(
        Path(r"Y:\!_FILHARMONIA\ML_EXPERIMENTS\models\hubert_best.pth")
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc = validate(model, loaders['test'], criterion, device, feature_extractor)

    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Acc:  {test_acc:.2f}%")

    # Summary
    print("\n" + "=" * 80)
    print("HUBERT - RESULTS")
    print("=" * 80)
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Training Time: {training_time / 60:.2f} minutes")
    print(f"Epochs Trained: {epoch + 1}")
    print(f"\nModel saved to: Y:\\!_FILHARMONIA\\ML_EXPERIMENTS\\models\\hubert_best.pth")
    print("=" * 80)

    return {
        'model_name': 'HuBERT',
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'training_time': training_time,
        'epochs': epoch + 1,
        'history': dict(history)
    }


if __name__ == "__main__":
    results = main()
