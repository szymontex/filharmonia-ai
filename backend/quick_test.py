"""Quick test with 3 files to compare configs"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import time
import torch
from pytorch_dataset import AudioDataset
from torch.utils.data import DataLoader
from app.config import settings

# Create tiny dataset with just 3 files
test_dir = Path("Y:/!_FILHARMONIA/TRAINING DATA/DATA")

print("Creating tiny test dataset...")
dataset = AudioDataset(test_dir, enable_chunking=True)

# Limit to 100 samples for quick test
class TinyDataset:
    def __init__(self, full_dataset, limit=100):
        self.dataset = full_dataset
        self.limit = min(limit, len(full_dataset))
        self.classes = full_dataset.classes

    def __len__(self):
        return self.limit

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_class_weights(self):
        return self.dataset.get_class_weights()

tiny = TinyDataset(dataset, 100)
print(f"Test dataset: {len(tiny)} samples")

configs = [
    {"batch_size": 32, "num_workers": 0, "pin_memory": True},
    {"batch_size": 32, "num_workers": 2, "pin_memory": True, "persistent_workers": True, "prefetch_factor": 4},
    {"batch_size": 64, "num_workers": 0, "pin_memory": True},
    {"batch_size": 64, "num_workers": 2, "pin_memory": True, "persistent_workers": True, "prefetch_factor": 4},
]

print("\n" + "="*80)
print("TESTING DATALOADER CONFIGURATIONS")
print("="*80)

for config in configs:
    print(f"\nConfig: {config}")

    loader = DataLoader(tiny, **config, shuffle=True)

    start = time.time()
    for i, (features, labels) in enumerate(loader):
        features = features.to('cuda')
        labels = labels.to('cuda')
        if i >= 10:  # Just 10 batches
            break
    elapsed = time.time() - start

    print(f"  Time for 10 batches: {elapsed:.2f}s ({elapsed/10:.3f}s per batch)")

    # Check GPU
    import subprocess
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=power.draw,utilization.gpu', '--format=csv,noheader'],
        capture_output=True, text=True
    )
    print(f"  GPU: {result.stdout.strip()}")

    del loader
    time.sleep(2)

print("\n" + "="*80)
print("DONE - Compare times above")
print("="*80)
