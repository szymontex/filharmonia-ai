"""
PyTorch Dataset with Virtual Chunking for Audio Classification
Supports: Mel-Spectrogram, MFCC, Raw Waveform
Uses virtual chunking to utilize 100% of training data
"""
import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio
import torchaudio.transforms as T
import numpy as np
from typing import Tuple, Optional, Dict, List
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from app.config import settings


class AudioDataset(Dataset):
    """
    Audio dataset with virtual chunking - creates multiple samples from long files

    Args:
        data_dir: Path to dataset folder (e.g., TRAINING DATA/DATA)
        feature_type: 'melspec', 'mfcc', or 'waveform'
        sample_rate: Target sample rate (default: 48000)
        duration: Target duration in seconds (default: 2.97)
        n_mels: Number of mel bands for melspec (default: 128)
        n_mfcc: Number of MFCC coefficients (default: 40)
        enable_chunking: If True, creates virtual chunks from long files (default: True)
    """

    def __init__(
        self,
        data_dir: Path,
        feature_type: str = 'melspec',
        sample_rate: int = 48000,
        duration: float = 2.97,
        n_mels: int = 128,
        n_mfcc: int = 40,
        enable_chunking: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.target_length = int(sample_rate * duration)
        self.enable_chunking = enable_chunking

        # Class labels (alphabetical order for consistency)
        self.classes = sorted(['APPLAUSE', 'MUSIC', 'PUBLIC', 'SPEECH', 'TUNING'])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Virtual samples: list of (file_path, start_frame, num_frames, label)
        self.samples: List[Tuple[Path, int, int, int]] = []

        print("[Dataset] Scanning audio files and creating virtual chunks...")
        total_files = 0
        total_chunks = 0

        for class_name in self.classes:
            class_folder = self.data_dir / class_name
            if not class_folder.exists():
                continue

            class_files = 0
            class_chunks = 0

            for wav_file in class_folder.glob('*.wav'):
                try:
                    # Get audio info without loading the entire file
                    info = torchaudio.info(str(wav_file))
                    file_duration_sec = info.num_frames / info.sample_rate
                    label = self.class_to_idx[class_name]

                    if enable_chunking and file_duration_sec > duration:
                        # Create virtual chunks from this file
                        num_chunks = int(file_duration_sec / duration)

                        for chunk_idx in range(num_chunks):
                            start_sec = chunk_idx * duration
                            start_frame = int(start_sec * info.sample_rate)
                            num_frames = int(duration * info.sample_rate)

                            # Validate chunk doesn't exceed file bounds
                            if start_frame >= info.num_frames:
                                continue  # Skip invalid chunks

                            # Adjust num_frames if chunk extends past file end
                            if start_frame + num_frames > info.num_frames:
                                num_frames = info.num_frames - start_frame

                            # Skip if resulting chunk is too small (< 0.5s)
                            if num_frames < int(0.5 * info.sample_rate):
                                continue

                            self.samples.append((
                                wav_file,
                                start_frame,
                                num_frames,
                                label
                            ))

                        class_chunks += num_chunks
                    else:
                        # File shorter than duration or chunking disabled
                        # Load entire file (will be padded if needed)
                        self.samples.append((
                            wav_file,
                            0,
                            info.num_frames,
                            label
                        ))
                        class_chunks += 1

                    class_files += 1

                except Exception as e:
                    print(f"[Dataset] Warning: skipping {wav_file.name}: {e}")
                    continue

            total_files += class_files
            total_chunks += class_chunks
            print(f"[Dataset]   {class_name}: {class_files} files -> {class_chunks} chunks")

        print(f"[Dataset] Total: {total_files} files -> {total_chunks} chunks")
        print(f"[Dataset] Chunking {'ENABLED' if enable_chunking else 'DISABLED'}")

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
        wav_path, start_frame, num_frames, label = self.samples[idx]

        try:
            # Load only the required chunk (efficient for large files)
            waveform, sr = torchaudio.load(
                str(wav_path),
                frame_offset=start_frame,
                num_frames=num_frames
            )

            # Validate waveform is not empty
            if waveform.numel() == 0:
                print(f"[Dataset] Warning: empty waveform from {wav_path.name} at frame {start_frame}")
                # Create zero tensor as fallback
                waveform = torch.zeros(1, self.target_length)
                sr = self.sample_rate

        except Exception as e:
            print(f"[Dataset] Error loading {wav_path.name}: {e}")
            # Create zero tensor as fallback
            waveform = torch.zeros(1, self.target_length)
            sr = self.sample_rate

        # Resample if needed
        if sr != self.sample_rate and waveform.numel() > 0:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad if shorter than target length
        if waveform.shape[1] < self.target_length:
            padding = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > self.target_length:
            # Trim if somehow longer
            waveform = waveform[:, :self.target_length]

        # Generate features based on type
        if self.feature_type == 'waveform':
            features = waveform.squeeze(0)

        elif self.feature_type == 'melspec':
            melspec = self.transform(waveform)
            melspec = torch.log(melspec + 1e-9)
            melspec = (melspec - melspec.min()) / (melspec.max() - melspec.min() + 1e-9)
            features = melspec.squeeze(0)

        elif self.feature_type == 'mfcc':
            mfcc = self.mfcc_transform(waveform)
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-9)
            features = mfcc.squeeze(0)

        return features, label

    def get_labels_list(self) -> List[int]:
        """Get list of labels for all samples"""
        return [sample[3] for sample in self.samples]

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset"""
        labels = self.get_labels_list()
        label_counts = Counter(labels)
        total_samples = len(labels)

        weights = torch.zeros(len(self.classes))
        for idx in range(len(self.classes)):
            count = label_counts.get(idx, 0)
            weights[idx] = total_samples / (len(self.classes) * count) if count > 0 else 0.0

        return weights

    def get_sample_weights(self, balance_strength: float = 1.0) -> torch.Tensor:
        """
        Get per-sample weights for WeightedRandomSampler

        Args:
            balance_strength: 0.0 = no balancing, 1.0 = full balancing
        """
        if balance_strength == 0.0:
            return torch.ones(len(self.samples))

        class_weights = self.get_class_weights()

        # Apply balance strength
        if balance_strength < 1.0:
            class_weights = torch.pow(class_weights, balance_strength)

        labels = self.get_labels_list()
        sample_weights = torch.tensor([class_weights[label].item() for label in labels])

        return sample_weights


def create_dataloaders(
    dataset_dir: Path,
    batch_size: int = 32,
    feature_type: str = 'melspec',
    use_weighted_sampling: bool = True,
    balance_strength: float = 1.0,
    enable_chunking: bool = True,
    num_workers: int = 0,
    **dataset_kwargs
) -> Dict[str, DataLoader]:
    """
    Create train, val, test dataloaders with virtual chunking

    Args:
        dataset_dir: Base dataset directory (contains train/val/test OR class folders)
        batch_size: Batch size for training
        feature_type: 'melspec', 'mfcc', or 'waveform'
        use_weighted_sampling: Use weighted sampling for imbalanced dataset
        balance_strength: 0.0 = natural distribution, 1.0 = full balancing
        enable_chunking: Enable virtual chunking for long files
        num_workers: Number of worker processes
        **dataset_kwargs: Additional arguments for AudioDataset
    """
    dataset_dir = Path(dataset_dir)

    # Create datasets with chunking enabled
    train_dataset = AudioDataset(
        dataset_dir / 'train' if (dataset_dir / 'train').exists() else dataset_dir,
        feature_type=feature_type,
        enable_chunking=enable_chunking,
        **dataset_kwargs
    )

    val_dataset = AudioDataset(
        dataset_dir / 'val' if (dataset_dir / 'val').exists() else dataset_dir,
        feature_type=feature_type,
        enable_chunking=False,  # Disable chunking for validation (deterministic)
        **dataset_kwargs
    )

    test_dataset = AudioDataset(
        dataset_dir / 'test' if (dataset_dir / 'test').exists() else dataset_dir,
        feature_type=feature_type,
        enable_chunking=False,  # Disable chunking for testing (deterministic)
        **dataset_kwargs
    )

    # Create sampler for training
    if use_weighted_sampling:
        sample_weights = train_dataset.get_sample_weights(balance_strength)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
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
    """Print dataset statistics with chunking info"""
    dataset_dir = Path(dataset_dir)

    print("=" * 80)
    print("DATASET INFORMATION (Virtual Chunking Enabled)")
    print("=" * 80)

    for split in ['train', 'val', 'test']:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            split_dir = dataset_dir
            print(f"\n{split.upper()} (using main directory):")
        else:
            print(f"\n{split.upper()}:")

        dataset = AudioDataset(split_dir, enable_chunking=(split == 'train'))

        print(f"  Total virtual samples: {len(dataset)}")
        print(f"  Classes: {dataset.classes}")

        # Count per class
        labels = dataset.get_labels_list()
        label_counts = Counter(labels)
        print(f"  Virtual samples per class:")
        for idx, class_name in enumerate(dataset.classes):
            count = label_counts[idx]
            print(f"    {class_name}: {count}")

        # Class weights
        class_weights = dataset.get_class_weights()
        print(f"  Class weights:")
        for idx, class_name in enumerate(dataset.classes):
            print(f"    {class_name}: {class_weights[idx]:.4f}")

    print("\n" + "=" * 80)
