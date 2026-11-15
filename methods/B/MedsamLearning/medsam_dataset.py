"""PyTorch data interface for the fixed Kvasir-SEG splits."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from data_pipeline import prepare_sample, read_split


class KvasirMedSAMDataset(Dataset):
    """Read one of the committed train/val/test CSV files.

    The preprocessing is intentionally delegated to ``data_pipeline.prepare_sample``
    so training and inference use exactly the same image, mask and box rules.
    """

    def __init__(self, project_dir, split="train"):
        self.project_dir = Path(project_dir).resolve()
        self.split = split
        self.rows = read_split(self.project_dir, split)
        if not self.rows:
            raise ValueError(f"split {split!r} is empty")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        sample = prepare_sample(self.project_dir, self.rows[index])
        return {
            "sample_id": sample["sample_id"],
            "image": torch.from_numpy(sample["image"]),
            "gt_mask": torch.from_numpy(sample["gt_mask"]),
            "gt_mask_1024": torch.from_numpy(sample["gt_mask_1024"]),
            "bbox_original": torch.from_numpy(sample["bbox_original"]),
            "bbox_1024": torch.from_numpy(sample["bbox_1024"]),
            "original_size": torch.from_numpy(sample["original_size"]),
        }


def create_dataloader(
    project_dir,
    split="train",
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=None,
):
    """Create a loader for samples with a common 1024x1024 model input.

    Original masks have varying sizes, so batch sizes above one require a custom
    collate function. Baseline inference intentionally uses ``batch_size=1``.
    """
    if batch_size != 1:
        raise ValueError("batch_size must be 1 because original masks vary in size")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        KvasirMedSAMDataset(project_dir, split),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
