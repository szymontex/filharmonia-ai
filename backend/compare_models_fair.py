"""
Fair comparison: Old model vs New models
All tested on the SAME test set (direct_from_source/test/)
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from transformers import ASTForAudioClassification
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from pytorch_dataset import AudioDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Test dataset (same for all models)
# Use OLD test set (full_balanced) to compare with OLD model fairly
TEST_DATASET_DIR = Path("Y:/!_FILHARMONIA/ML_EXPERIMENTS/datasets/full_balanced/test")

# Models to compare
MODELS = {
    "OLD (ast_20251006_024939)": "Y:/!_FILHARMONIA/RECOGNITION_MODELS/ast_20251006_024939.pth",
    "NEW balance=0.00": "Y:/!_FILHARMONIA/ML_EXPERIMENTS/balance_experiments/ast_balance0.00_20251006_205842.pth",
    "NEW balance=0.25": "Y:/!_FILHARMONIA/ML_EXPERIMENTS/balance_experiments/ast_balance0.25_20251006_231033.pth",
    "NEW balance=0.50": "Y:/!_FILHARMONIA/ML_EXPERIMENTS/balance_experiments/ast_balance0.50_20251007_012211.pth",
    "NEW balance=0.75": "Y:/!_FILHARMONIA/ML_EXPERIMENTS/balance_experiments/ast_balance0.75_20251007_033353.pth",
    "NEW balance=1.00": "Y:/!_FILHARMONIA/ML_EXPERIMENTS/balance_experiments/ast_balance1.00_20251007_054702.pth",
}

CLASS_NAMES = ['APPLAUSE', 'MUSIC', 'PUBLIC', 'SPEECH', 'TUNING']


def measure_model(model_path: Path, test_dataset):
    """Measure model accuracy on test dataset"""

    # Load model
    model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_labels=5,
        ignore_mismatched_sizes=True
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)

    # Create dataloader
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Measure
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for features, labels in test_loader:
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

            # Per-class
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1

    overall_acc = 100.0 * correct / total

    per_class_acc = {}
    for i, class_name in enumerate(CLASS_NAMES):
        if class_total[i] > 0:
            per_class_acc[class_name] = 100.0 * class_correct[i] / class_total[i]
        else:
            per_class_acc[class_name] = 0.0

    return overall_acc, per_class_acc


def main():
    print("=" * 80)
    print("FAIR MODEL COMPARISON")
    print("=" * 80)
    print(f"Test dataset: {TEST_DATASET_DIR}")
    print()

    # Load test dataset once
    test_dataset = AudioDataset(TEST_DATASET_DIR, feature_type='melspec')
    print(f"Test samples: {len(test_dataset)}")
    print()

    # Test set distribution
    from collections import Counter
    label_counts = Counter(test_dataset.labels)
    print("Test set distribution:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name}: {label_counts[i]} samples")
    print()

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    results = []

    for model_name, model_path in MODELS.items():
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"\n{model_name}: FILE NOT FOUND")
            continue

        print(f"\n{model_name}:")
        print(f"  Path: {model_path.name}")

        overall_acc, per_class_acc = measure_model(model_path, test_dataset)

        print(f"  Overall Acc: {overall_acc:.2f}%")
        print(f"  Per-class:")
        for class_name in CLASS_NAMES:
            print(f"    {class_name}: {per_class_acc[class_name]:.2f}%")

        results.append({
            'name': model_name,
            'overall': overall_acc,
            'per_class': per_class_acc
        })

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Model':<25} {'Overall':<10} {'MUSIC':<10} {'TUNING':<10} {'PUBLIC':<10} {'SPEECH':<10} {'APPLAUSE':<10}")
    print("-" * 95)

    for result in results:
        name = result['name']
        overall = result['overall']
        music = result['per_class']['MUSIC']
        tuning = result['per_class']['TUNING']
        public = result['per_class']['PUBLIC']
        speech = result['per_class']['SPEECH']
        applause = result['per_class']['APPLAUSE']

        print(f"{name:<25} {overall:>8.2f}% {music:>8.2f}% {tuning:>8.2f}% {public:>8.2f}% {speech:>8.2f}% {applause:>8.2f}%")

    # Best model
    best = max(results, key=lambda x: x['overall'])
    print("\n" + "=" * 80)
    print("🏆 BEST MODEL (Overall Accuracy)")
    print("=" * 80)
    print(f"{best['name']}: {best['overall']:.2f}%")

    # Best for MUSIC
    best_music = max(results, key=lambda x: x['per_class']['MUSIC'])
    print(f"\n🎵 BEST FOR MUSIC: {best_music['name']} ({best_music['per_class']['MUSIC']:.2f}%)")

    # Best for TUNING
    best_tuning = max(results, key=lambda x: x['per_class']['TUNING'])
    print(f"🎻 BEST FOR TUNING: {best_tuning['name']} ({best_tuning['per_class']['TUNING']:.2f}%)")


if __name__ == "__main__":
    main()
