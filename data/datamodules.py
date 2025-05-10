from typing import Dict, List, Optional, NoReturn
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from data.audiotext_dataset import AudioTextDataset
from torch.utils.data.dataloader import default_collate


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: object,
        val_dataset: object,
        batch_size: int,
        num_workers: int
    ):
        super().__init__()
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.collate_fn = precomputed_stft_collate_fn


    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage: Optional[str] = None) -> NoReturn:
        r"""called on every device."""

        # make assignments here (val/train/test split)
        # called on every process in DDP

        # SegmentSampler is used for selecting segments for training.
        # On multiple devices, each SegmentSampler samples a part of mini-batch
        # data.
        self.train_dataset = self._train_dataset
        self.val_dataset = self._val_dataset
        
        
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        r"""Get train loader."""
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
            shuffle=True
        )

        return train_loader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        r"""Get val loader."""
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
            shuffle=False
        )

        return val_loader

    def test_dataloader(self):
        pass

    def teardown(self):
        pass


def precomputed_stft_collate_fn(batch_list):
    batch_list = [item for item in batch_list if item is not None]

    if not batch_list:
        return None

    collated_batch = default_collate(batch_list)

    return collated_batch