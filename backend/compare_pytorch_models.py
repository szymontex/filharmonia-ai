"""
Compare PyTorch models - Load saved checkpoints and evaluate
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from transformers import ASTForAudioClassification
from torchvision.models import efficientnet_b0

sys.path.insert(0, str(Path(__file__).parent))
from pytorch_dataset import create_dataloaders


# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")


class EfficientNetAudio(nn.Module):
    """EfficientNet-B0 adapted for audio mel-spectrograms"""

    def __init__(self, num_classes=5):
        super().__init__()

        # Load backbone
        self.backbone = efficientnet_b0()

        # Replace first conv layer to accept 1 channel
        self.backbone.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        return self.backbone(x)


def evaluate_ast(loader, device):
    """Evaluate AST model"""
    checkpoint_path = Path(r"Y:\!_FILHARMONIA\ML_EXPERIMENTS\models\ast_best.pth")

    if not checkpoint_path.exists():
        return None

    print("Loading AST model...")
    model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_labels=5,
        ignore_mismatched_sizes=True
    )

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    print("Evaluating AST...")
    with torch.no_grad():
        for features, labels in loader:
            # AST preprocessing
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

    val_acc = checkpoint.get('val_acc', 0.0)
    test_acc = 100 * correct / total
    epoch = checkpoint.get('epoch', 0)

    return {
        'model_name': 'AST (Audio Spectrogram Transformer)',
        'best_val_acc': val_acc,
        'test_acc': test_acc,
        'epoch': epoch
    }


def evaluate_efficientnet(loader, device):
    """Evaluate EfficientNet model"""
    checkpoint_path = Path(r"Y:\!_FILHARMONIA\ML_EXPERIMENTS\models\efficientnet_best.pth")

    if not checkpoint_path.exists():
        return None

    print("\nLoading EfficientNet model...")
    model = EfficientNetAudio(num_classes=5).to(device)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    correct = 0
    total = 0

    print("Evaluating EfficientNet...")
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = checkpoint.get('val_acc', 0.0)
    test_acc = 100 * correct / total
    epoch = checkpoint.get('epoch', 0)

    return {
        'model_name': 'EfficientNet-B0',
        'best_val_acc': val_acc,
        'test_acc': test_acc,
        'epoch': epoch
    }


def main():
    print("=" * 80)
    print("PYTORCH MODELS COMPARISON")
    print("=" * 80)

    dataset_dir = Path(r"Y:\!_FILHARMONIA\ML_EXPERIMENTS\datasets\small_balanced")

    # Create dataloaders
    loaders = create_dataloaders(
        dataset_dir,
        batch_size=16,
        feature_type='melspec',
        use_weighted_sampling=False,  # Don't use weighted for testing
        num_workers=0
    )

    print(f"\nTest set: {len(loaders['test'].dataset)} samples\n")

    # Evaluate models
    results = []

    ast_results = evaluate_ast(loaders['test'], device)
    if ast_results:
        results.append(ast_results)

    eff_results = evaluate_efficientnet(loaders['test'], device)
    if eff_results:
        results.append(eff_results)

    # Print comparison
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    # Sort by test accuracy
    results.sort(key=lambda x: x['test_acc'], reverse=True)

    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['model_name']}")
        print(f"   Best Validation Accuracy: {r['best_val_acc']:.2f}%")
        print(f"   Test Accuracy: {r['test_acc']:.2f}%")
        print(f"   Training stopped at epoch: {r['epoch'] + 1}")

    print("\n" + "=" * 80)
    print("WINNER:")
    print("=" * 80)
    winner = results[0]
    print(f"{winner['model_name']}")
    print(f"Test Accuracy: {winner['test_acc']:.2f}%")
    print(f"Validation Accuracy: {winner['best_val_acc']:.2f}%")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
