import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import Tuple, Optional
import warnings


class BurnSeverityDataset(Dataset):
    """
    Dataset Loader class for Landsat-BSA burn severity data
    """

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        transform: Optional[torch.nn.Module] = None,
        augment: bool = False,
        max_samples: Optional[int] = None,  
    ):
        self.dataset_path = dataset_path
        self.split = split
        self.transform = transform
        self.augment = augment and (split == "train")
        self.max_samples = max_samples  

        # Define split paths
        self.split_path = os.path.join(dataset_path, split)
        self.pre_fire_path = os.path.join(self.split_path, "A")
        self.post_fire_path = os.path.join(self.split_path, "B")
        self.label_path = os.path.join(self.split_path, "label")

        # List files
        self.sample_files = sorted(os.listdir(self.pre_fire_path))
        self._verify_files()
        
        # Limit samples if specified
        if self.max_samples is not None:
            self.sample_files = self.sample_files[:self.max_samples]

        print(f"Loaded {split} dataset: {len(self.sample_files)} samples")

    def _verify_files(self):
        """Verify all A, B, and label files exist for each sample"""
        missing_samples = []
        for sample_file in self.sample_files:
            pre_fire_file = os.path.join(self.pre_fire_path, sample_file)
            post_fire_file = os.path.join(self.post_fire_path, sample_file)
            label_file = os.path.join(self.label_path, sample_file)

            if not all(os.path.exists(f) for f in [pre_fire_file, post_fire_file, label_file]):
                missing_samples.append(sample_file)

        if missing_samples:
            warnings.warn(f"{len(missing_samples)} missing samples found and skipped.")
            self.sample_files = [f for f in self.sample_files if f not in missing_samples]

    def __len__(self) -> int:
        return len(self.sample_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_file = self.sample_files[idx]

        # Load images
        pre_fire = Image.open(os.path.join(self.pre_fire_path, sample_file))
        post_fire = Image.open(os.path.join(self.post_fire_path, sample_file))
        label = Image.open(os.path.join(self.label_path, sample_file))

        # Convert to numpy arrays
        pre_fire_np = np.array(pre_fire, dtype=np.float32) / 255.0
        post_fire_np = np.array(post_fire, dtype=np.float32) / 255.0
        label_np = np.array(label, dtype=np.int64)

        # Apply augmentation
        if self.augment:
            pre_fire_np, post_fire_np, label_np = self._apply_augmentation(
                pre_fire_np, post_fire_np, label_np
            )

        # Convert to tensors (HWC → CHW)
        pre_fire_tensor = torch.from_numpy(pre_fire_np).permute(2, 0, 1)
        post_fire_tensor = torch.from_numpy(post_fire_np).permute(2, 0, 1)
        label_tensor = torch.from_numpy(label_np)

        # Concatenate pre and post fire
        input_tensor = torch.cat([pre_fire_tensor, post_fire_tensor], dim=0)

        # Extra transforms (e.g., normalization pipelines)
        if self.transform:
            input_tensor = self.transform(input_tensor)

        return input_tensor, label_tensor

    def _apply_augmentation(
        self, pre_fire: np.ndarray, post_fire: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply data augmentation consistently"""
        if np.random.random() > 0.5:  # Horizontal flip
            pre_fire, post_fire, label = map(np.fliplr, (pre_fire, post_fire, label))
        if np.random.random() > 0.5:  # Vertical flip
            pre_fire, post_fire, label = map(np.flipud, (pre_fire, post_fire, label))
        if np.random.random() > 0.5:  # 90° rotations
            k = np.random.choice([1, 2, 3])
            pre_fire, post_fire, label = (
                np.rot90(pre_fire, k),
                np.rot90(post_fire, k),
                np.rot90(label, k),
            )
        return pre_fire.copy(), post_fire.copy(), label.copy()

    def get_sample_info(self, idx: int) -> dict:
        """Get info about a sample"""
        sample_file = self.sample_files[idx]
        input_tensor, label_tensor = self.__getitem__(idx)
        unique_labels, counts = torch.unique(label_tensor, return_counts=True)
        return {
            "filename": sample_file,
            "input_shape": input_tensor.shape,
            "label_shape": label_tensor.shape,
            "class_distribution": {
                int(lbl): int(cnt) for lbl, cnt in zip(unique_labels, counts)
            },
        }


# ---------------- DataLoader Factory ----------------
def create_data_loaders(dataset_path: str, batch_size: int = 16, num_workers: int = 0, 
                       max_samples: Optional[int] = None):  # ADD max_samples parameter
    train_dataset = BurnSeverityDataset(dataset_path, "train", augment=True, max_samples=max_samples)
    val_dataset = BurnSeverityDataset(dataset_path, "val", max_samples=max_samples//4 if max_samples else None)
    test_dataset = BurnSeverityDataset(dataset_path, "test", max_samples=max_samples//8 if max_samples else None)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, test_loader
