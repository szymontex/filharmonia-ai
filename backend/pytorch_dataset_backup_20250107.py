"""
PyTorch Dataset and DataLoaders for Audio Classification
Supports multiple audio features: Mel-Spectrogram, MFCC, Raw Waveform
"""
import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio
import torchaudio.transforms as T
import numpy as np
from typing import Tuple, Optional, Dict
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from app.config import settings


class AudioDataset(Dataset):
    """
    Audio dataset that loads WAV files and converts to specified features

    Args:
        data_dir: Path to dataset folder (e.g., .../small_balanced/train)
        feature_type: 'melspec', 'mfcc', or 'waveform'
        sample_rate: Target sample rate (default: 48000)
        duration: Target duration in seconds (default: 2.97)
        n_mels: Number of mel bands for melspec (default: 128)
        n_mfcc: Number of MFCC coefficients (default: 40)
    """

    def __init__(
        self,
        data_dir: Path,
        feature_type: str = 'melspec',
        sample_rate: int = 48000,
        duration: float = 2.97,
        n_mels: int = 128,
        n_mfcc: int = 40
    ):
        self.data_dir = Path(data_dir)
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.target_length = int(sample_rate * duration)

        # Class labels (alphabetical order for consistency)
        self.classes = sorted(['APPLAUSE', 'MUSIC', 'PUBLIC', 'SPEECH', 'TUNING'])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Load all file paths and labels
        self.samples = []
        self.labels = []

        for class_name in self.classes:
            class_folder = self.data_dir / class_name
            if not class_folder.exists():
                continue

            for wav_file in class_folder.glob('*.wav'):
                self.samples.append(wav_file)
                self.labels.append(self.class_to_idx[class_name])

        # Create transforms
        if feature_type == 'melspec':
            self.transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=2048,
                hop_length=512,
                n_mels=n_mels
            )
        elif feature_type == 'mfcc':
            self.mel_transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=2048,
                hop_length=512,
                n_mels=n_mels
            )
            self.mfcc_transform = T.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={
                    'n_fft': 2048,
                    'hop_length': 512,
                    'n_mels': n_mels
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        wav_path = self.samples[idx]
        label = self.labels[idx]

        # Load audio
        waveform, sr = torchaudio.load(wav_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or trim to target length
        if waveform.shape[1] < self.target_length:
            # Pad with zeros
            padding = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            # Trim to target length
            waveform = waveform[:, :self.target_length]

        # Generate features based on type
        if self.feature_type == 'waveform':
            # Return raw waveform
            features = waveform.squeeze(0)  # Remove channel dimension

        elif self.feature_type == 'melspec':
            # Generate mel-spectrogram
            melspec = self.transform(waveform)
            # Convert to log scale
            melspec = torch.log(melspec + 1e-9)
            # Normalize to [0, 1]
            melspec = (melspec - melspec.min()) / (melspec.max() - melspec.min() + 1e-9)
            features = melspec.squeeze(0)  # Remove channel dimension

        elif self.feature_type == 'mfcc':
            # Generate MFCC
            mfcc = self.mfcc_transform(waveform)
            # Normalize
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-9)
            features = mfcc.squeeze(0)  # Remove channel dimension

        return features, label

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset"""
        label_counts = Counter(self.labels)
        total_samples = len(self.labels)

        weights = torch.zeros(len(self.classes))
        for idx, class_name in enumerate(self.classes):
            count = label_counts[idx]
            # Inverse frequency weighting
            weights[idx] = total_samples / (len(self.classes) * count) if count > 0 else 0.0

        return weights

    def get_sample_weights(self) -> torch.Tensor:
        """Get per-sample weights for WeightedRandomSampler"""
        class_weights = self.get_class_weights()
        sample_weights = torch.tensor([class_weights[label] for label in self.labels])
        return sample_weights


def create_dataloaders(
    dataset_dir: Path,
    batch_size: int = 32,
    feature_type: str = 'melspec',
    use_weighted_sampling: bool = True,
    num_workers: int = 0,
    **dataset_kwargs
) -> Dict[str, DataLoader]:
    """
    Create train, val, test dataloaders

    Args:
        dataset_dir: Base dataset directory (contains train/val/test folders)
        batch_size: Batch size for training
        feature_type: 'melspec', 'mfcc', or 'waveform'
        use_weighted_sampling: Use weighted sampling for imbalanced dataset
        num_workers: Number of worker processes for data loading
        **dataset_kwargs: Additional arguments for AudioDataset

    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """

    dataset_dir = Path(dataset_dir)

    # Create datasets
    train_dataset = AudioDataset(
        dataset_dir / 'train',
        feature_type=feature_type,
        **dataset_kwargs
    )

    val_dataset = AudioDataset(
        dataset_dir / 'val',
        feature_type=feature_type,
        **dataset_kwargs
    )

    test_dataset = AudioDataset(
        dataset_dir / 'test',
        feature_type=feature_type,
        **dataset_kwargs
    )

    # Create sampler for training (handles class imbalance)
    if use_weighted_sampling:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False  # Don't shuffle when using sampler
    else:
        sampler = None
        shuffle = True

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
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


def print_dataset_info(dataset_dir: Path):
    """Print dataset statistics"""
    dataset_dir = Path(dataset_dir)

    print("=" * 80)
    print("DATASET INFORMATION")
    print("=" * 80)

    for split in ['train', 'val', 'test']:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue

        dataset = AudioDataset(split_dir)

        print(f"\n{split.upper()}:")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Classes: {dataset.classes}")

        # Count per class
        label_counts = Counter(dataset.labels)
        print(f"  Samples per class:")
        for idx, class_name in enumerate(dataset.classes):
            count = label_counts[idx]
            print(f"    {class_name}: {count}")

        # Class weights
        class_weights = dataset.get_class_weights()
        print(f"  Class weights (for loss function):")
        for idx, class_name in enumerate(dataset.classes):
            print(f"    {class_name}: {class_weights[idx]:.4f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Test dataset
    dataset_dir = Path(r"Y:\!_FILHARMONIA\ML_EXPERIMENTS\datasets\small_balanced")

    print_dataset_info(dataset_dir)

    print("\nCreating dataloaders...")
    loaders = create_dataloaders(
        dataset_dir,
        batch_size=16,
        feature_type='melspec',
        use_weighted_sampling=True
    )

    print("\nTesting dataloaders:")
    for split, loader in loaders.items():
        features, labels = next(iter(loader))
        print(f"\n{split.upper()}:")
        print(f"  Batch features shape: {features.shape}")
        print(f"  Batch labels shape: {labels.shape}")
        print(f"  Features dtype: {features.dtype}")
        print(f"  Labels dtype: {labels.dtype}")

    print("\nDataLoaders ready for training!")
