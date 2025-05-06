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
        r"""Data module. To get one batch of data:

        code-block:: python

            data_module.setup()

            for batch_data_dict in data_module.train_dataloader():
                print(batch_data_dict.keys())
                break

        Args:
            train_sampler: Sampler object
            train_dataset: Dataset object
            num_workers: int
            distributed: bool
        """
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
        # test_split = Dataset(...)
        # return DataLoader(test_split)
        pass

    def teardown(self):
        # clean up after fit or test
        # called on every process in DDP
        pass


def precomputed_stft_collate_fn(batch_list):
    r"""Collate mini-batch data from PrecomputedSTFTDataset.

    Filters out None items resulting from load errors and uses default_collate
    for standard tensor stacking and list aggregation.

    Args:
        batch_list: A list of dictionaries, where each dictionary is the output
                   of PrecomputedSTFTDataset.__getitem__. Some items might be None.
                   Expected item structure:
                   {
                       'stfts': {
                           'mixture': {<win_len>: (mag, cos, sin)},
                           'segment': {<win_len>: (mag, cos, sin)}
                       },
                       'target_waveform': Tensor,
                       'text': str,
                       'mixture_component_texts': List[str],
                       'stft_common_params': Dict,
                       'stft_win_lengths': List[int]
                       # Potentially other keys like 'output_index'
                   }

    Returns:
        A single dictionary where corresponding values from the list items
        have been collated (e.g., tensors stacked along a new batch dimension,
        lists of strings remain lists of strings). Returns None if batch_list
        is empty or contains only None items.
    """
    # Filter out None items (e.g., from dataset loading errors)
    batch_list = [item for item in batch_list if item is not None]

    if not batch_list:
        # Return None or an empty dict if the batch is empty after filtering
        # Returning None might be simpler for the training loop to handle
        return None

    # Use PyTorch's default collate to handle the list of dicts
    # It automatically stacks tensors and keeps other types as lists
    collated_batch = default_collate(batch_list)

    # Note: default_collate handles the structure correctly.
    # 'stfts' will become a dict where values are further dicts/tuples of stacked tensors.
    # 'target_waveform' will be a stacked tensor.
    # 'text', 'mixture_component_texts', 'stft_win_lengths' will be lists.
    # 'stft_common_params' will be a list of dicts (one per item).

    # No need for the old modality grouping logic.
    return collated_batch