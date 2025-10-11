"""
Measure true training accuracy for saved models
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from transformers import ASTForAudioClassification
from pytorch_dataset import create_dataloaders
from app.config import settings

def measure_accuracy(model, dataloader, device):
    """Measure accuracy on a dataset"""
    model.eval()
    correct = 0
    total = 0

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

            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features).logits
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    models_to_measure = [
        'ast_20251006_012604.pth',
        'ast_20251006_020751.pth',
        'ast_20251006_024939.pth'
    ]

    # Create dataset (same as used for training)
    print("\nPreparing dataset...")
    # Use the temp dataset if it exists, otherwise need to recreate
    # For now, let's use the last temp dataset or create new one

    # Check if there's a temp dataset
    temp_datasets = list(settings.RECOGNITION_MODELS_FOLDER.glob("temp_dataset_*"))

    if temp_datasets:
        dataset_dir = temp_datasets[0]
        print(f"Using existing dataset: {dataset_dir}")
    else:
        print("No temp dataset found. You need to run training first or provide dataset path.")
        print("Using balanced dataset from ML_EXPERIMENTS if available...")
        dataset_dir = Path("Y:/!_FILHARMONIA/ML_EXPERIMENTS/datasets/full_balanced")

        if not dataset_dir.exists():
            print("ERROR: No dataset available!")
            return

    # Create dataloaders
    loaders = create_dataloaders(
        dataset_dir,
        batch_size=16,
        feature_type='melspec',
        use_weighted_sampling=False,  # Don't use weighted sampling for eval
        num_workers=0
    )

    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']

    print(f"Dataset sizes: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test")

    # Measure each model
    for model_file in models_to_measure:
        model_path = settings.RECOGNITION_MODELS_FOLDER / model_file

        if not model_path.exists():
            print(f"\nSkipping {model_file} - file not found")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model_file}")
        print(f"{'='*60}")

        # Load model
        print("Loading model...")
        model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            num_labels=5,
            ignore_mismatched_sizes=True
        )

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        # Measure accuracies
        print("Measuring train accuracy...")
        train_acc = measure_accuracy(model, train_loader, device)

        print("Measuring val accuracy...")
        val_acc = measure_accuracy(model, val_loader, device)

        print("Measuring test accuracy...")
        test_acc = measure_accuracy(model, test_loader, device)

        print(f"\nResults:")
        print(f"  Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc:   {val_acc:.2f}%")
        print(f"  Test Acc:  {test_acc:.2f}%")
        print(f"  Epoch:     {checkpoint.get('epoch', 'unknown')}")

if __name__ == "__main__":
    main()
