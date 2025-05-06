from typing import Any, Callable, Dict
import random
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from models.clap_encoder import CLAP_Encoder

from huggingface_hub import PyTorchModelHubMixin


class AudioSep(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        ss_model: nn.Module = None,
        query_encoder: nn.Module = CLAP_Encoder().eval(),
        loss_function = None,
        optimizer_type: str = None,
        learning_rate: float = None,
        lr_lambda_func = None,
        use_text_ratio: float =1.0,
    ):
        r"""Pytorch Lightning wrapper of PyTorch model, including forward,
        optimization of model, etc.

        Args:
            ss_model: nn.Module
            query_encoder: nn.Module
            loss_function: function or object
            learning_rate: float
            lr_lambda: function
        """

        super().__init__()
        self.ss_model = ss_model
        self.query_encoder = query_encoder
        self.query_encoder_type = self.query_encoder.encoder_type
        self.use_text_ratio = use_text_ratio
        self.loss_function = loss_function
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.lr_lambda_func = lr_lambda_func


    def forward(self, x):
        pass

    def training_step(self, batch_data_dict, batch_idx):
        r"""Forward a mini-batch data to model, calculate loss function, and
        train for one step. A mini-batch data is evenly distributed to multiple
        devices (if there are) for parallel training.

        Args:
            batch_data_dict: e.g. 
                'text': ['a sound of dog', ...]
                'stfts': {
                    'mixture': {
                        win_len: (batch_size, 1, T, F)
                    },
                    'segment': {
                        win_len: (batch_size, 1, T, F)
                    }
                }
                'stft_win_lengths': [List[int]]
            batch_idx: int

        Returns:
            loss: float, loss function of this mini-batch
        """
        # [important] fix random seeds across devices
        random.seed(batch_idx)

        batch_text = batch_data_dict['text']
        batch_stfts = batch_data_dict['stfts']
        stft_win_lengths = batch_data_dict['stft_win_lengths']
        target_waveforms = batch_data_dict['target_waveform']
        # selected_win_len_tensor = stft_win_lengths[0] # This is likely a tensor for the batch
        # In collated batch, stft_win_lengths likely looks like [[256, 512, 1024], ...] or just [[256, 512, 1024]]
        # We need the list of available lengths for this item/batch
        # available_lengths = stft_win_lengths[0] # Assuming it collates to a list containing the list
        # if not isinstance(available_lengths, list):
        #      # Handle potential different collation (e.g., if batch size is 1)
        #      if isinstance(stft_win_lengths, list) and len(stft_win_lengths) > 0 and isinstance(stft_win_lengths[0], list):
        #          available_lengths = stft_win_lengths[0]
        #      else:
        #          raise TypeError(f"Could not determine available STFT window lengths from batch_data_dict['stft_win_lengths']: {stft_win_lengths}")

        # --- Correctly determine available_lengths from collated structure --- #
        collated_lengths_data = stft_win_lengths # e.g., [tensor([256,...]), tensor([512,...]), ...]
        available_lengths = []
        if isinstance(collated_lengths_data, list) and len(collated_lengths_data) > 0:
            # Check if it's the list of tensors structure
            if all(isinstance(t, torch.Tensor) for t in collated_lengths_data):
                try:
                    # Reconstruct list: take first element of each tensor
                    available_lengths = [int(t[0].item()) for t in collated_lengths_data]
                except (IndexError, TypeError) as e:
                     raise TypeError(f"Could not reconstruct available lengths from list of tensors: {collated_lengths_data}. Error: {e}")
            # Check if it's the list of list structure (e.g., batch size 1?)
            elif len(collated_lengths_data) == 1 and isinstance(collated_lengths_data[0], list):
                 available_lengths = collated_lengths_data[0]
            # Check if it's already the list of integers (unlikely with batch > 1)
            elif all(isinstance(l, int) for l in collated_lengths_data):
                 available_lengths = collated_lengths_data

        if not available_lengths:
            raise TypeError(f"Could not determine available STFT window lengths from batch_data_dict['stft_win_lengths']: {collated_lengths_data}")
        # --- End determination --- #

        try:
            # --- Fix: Explicitly select window length 512 --- #
            desired_win_len = 512
            if desired_win_len not in available_lengths:
                 raise ValueError(f"Desired window length {desired_win_len} not found in available lengths: {available_lengths} from batch data.")

            scalar_win_len = desired_win_len # Use the desired integer directly

            # print(f"Debug: Using explicit scalar window length: {scalar_win_len}") # Optional debug print

            # --- Use scalar key for access --- #
            mixture_mag = batch_stfts['mixture'][scalar_win_len][0]
            mixture_cos = batch_stfts['mixture'][scalar_win_len][1]
            mixture_sin = batch_stfts['mixture'][scalar_win_len][2]

            # Assuming you might need segment STFTs later or in other parts of the model:
            # segment_mag = batch_stfts['segment'][scalar_win_len][0]
            # segment_cos = batch_stfts['segment'][scalar_win_len][1]
            # segment_sin = batch_stfts['segment'][scalar_win_len][2]

        except (KeyError, TypeError, IndexError, ValueError) as e: # Added ValueError
            print(f"Error accessing STFT data: {e}")
            # print(f"Original selected window length tensor: {selected_win_len_tensor}") # No longer applicable
            print(f"Available lengths: {available_lengths}")
            print(f"Attempted key: {scalar_win_len if 'scalar_win_len' in locals() else 'N/A'}")
            print("Batch STFT structure received:", batch_data_dict.get('stfts'))
            raise ValueError("Could not extract STFTs from batch. Check DataLoader collation and PrecomputedSTFTDataset output format.") from e

        device = mixture_mag.device

        if self.query_encoder_type == 'CLAP':
            conditions = self.query_encoder.get_query_embed(
                modality='text',
                text=batch_text,
            )
        else:
            raise NotImplementedError(f"Query encoder type {self.query_encoder_type} not fully handled with STFT input.")

        input_dict = {
            'stft_mixture_mag': mixture_mag,
            'stft_mixture_cos': mixture_cos,
            'stft_mixture_sin': mixture_sin,
            'condition': conditions,
        }

        target_dict = {
            'waveform': target_waveforms.squeeze(1),
        }

        self.ss_model.train()
        sep_waveform = self.ss_model(input_dict, target_waveform=target_waveforms)['waveform']
        sep_waveform = sep_waveform.squeeze(1)

        output_dict = {
            'waveform': sep_waveform,
        }

        loss = self.loss_function(output_dict, target_dict)

        self.log_dict({"train_loss": loss})
        
        return loss

    def validation_step(self, batch_data_dict, batch_idx):
        r"""Forward a mini-batch data to model, calculate loss function, and
        train for one step. A mini-batch data is evenly distributed to multiple
        devices (if there are) for parallel validation.

        Args:
            batch_data_dict: e.g. 
                'text': ['a sound of dog', ...]
                'stfts': {
                    'mixture': {
                        win_len: (batch_size, 1, T, F)
                    },
                    'segment': {
                        win_len: (batch_size, 1, T, F)
                    }
                }
                'stft_win_lengths': [List[int]]
            batch_idx: int

        Returns:
            loss: float, loss function of this mini-batch
        """
        # [important] fix random seeds across devices
        random.seed(batch_idx)

        batch_text = batch_data_dict['text']
        batch_stfts = batch_data_dict['stfts']
        stft_win_lengths = batch_data_dict['stft_win_lengths']
        target_waveforms = batch_data_dict['target_waveform']
        # selected_win_len_tensor = stft_win_lengths[0] # This is likely a tensor for the batch
        # In collated batch, stft_win_lengths likely looks like [[256, 512, 1024], ...] or just [[256, 512, 1024]]
        # We need the list of available lengths for this item/batch
        # available_lengths = stft_win_lengths[0] # Assuming it collates to a list containing the list
        # if not isinstance(available_lengths, list):
        #      # Handle potential different collation (e.g., if batch size is 1)
        #      if isinstance(stft_win_lengths, list) and len(stft_win_lengths) > 0 and isinstance(stft_win_lengths[0], list):
        #          available_lengths = stft_win_lengths[0]
        #      else:
        #          raise TypeError(f"Could not determine available STFT window lengths from batch_data_dict['stft_win_lengths']: {stft_win_lengths}")

        # --- Correctly determine available_lengths from collated structure --- #
        collated_lengths_data = stft_win_lengths # e.g., [tensor([256,...]), tensor([512,...]), ...]
        available_lengths = []
        if isinstance(collated_lengths_data, list) and len(collated_lengths_data) > 0:
            # Check if it's the list of tensors structure
            if all(isinstance(t, torch.Tensor) for t in collated_lengths_data):
                try:
                    # Reconstruct list: take first element of each tensor
                    available_lengths = [int(t[0].item()) for t in collated_lengths_data]
                except (IndexError, TypeError) as e:
                     raise TypeError(f"Could not reconstruct available lengths from list of tensors: {collated_lengths_data}. Error: {e}")
            # Check if it's the list of list structure (e.g., batch size 1?)
            elif len(collated_lengths_data) == 1 and isinstance(collated_lengths_data[0], list):
                 available_lengths = collated_lengths_data[0]
            # Check if it's already the list of integers (unlikely with batch > 1)
            elif all(isinstance(l, int) for l in collated_lengths_data):
                 available_lengths = collated_lengths_data

        if not available_lengths:
            raise TypeError(f"Could not determine available STFT window lengths from batch_data_dict['stft_win_lengths']: {collated_lengths_data}")
        # --- End determination --- #

        try:
            # --- Fix: Explicitly select window length 512 --- #
            desired_win_len = 512
            if desired_win_len not in available_lengths:
                 raise ValueError(f"Desired window length {desired_win_len} not found in available lengths: {available_lengths} from batch data.")

            scalar_win_len = desired_win_len # Use the desired integer directly

            # print(f"Debug: Using explicit scalar window length: {scalar_win_len}") # Optional debug print

            # --- Use scalar key for access --- #
            mixture_mag = batch_stfts['mixture'][scalar_win_len][0]
            mixture_cos = batch_stfts['mixture'][scalar_win_len][1]
            mixture_sin = batch_stfts['mixture'][scalar_win_len][2]

            # Assuming you might need segment STFTs later or in other parts of the model:
            # segment_mag = batch_stfts['segment'][scalar_win_len][0]
            # segment_cos = batch_stfts['segment'][scalar_win_len][1]
            # segment_sin = batch_stfts['segment'][scalar_win_len][2]

        except (KeyError, TypeError, IndexError, ValueError) as e: # Added ValueError
            print(f"Error accessing STFT data: {e}")
            # print(f"Original selected window length tensor: {selected_win_len_tensor}") # No longer applicable
            print(f"Available lengths: {available_lengths}")
            print(f"Attempted key: {scalar_win_len if 'scalar_win_len' in locals() else 'N/A'}")
            print("Batch STFT structure received:", batch_data_dict.get('stfts'))
            raise ValueError("Could not extract STFTs from batch. Check DataLoader collation and PrecomputedSTFTDataset output format.") from e

        device = mixture_mag.device

        self.ss_model.eval()

        with torch.no_grad():
            if self.query_encoder_type == 'CLAP':
                conditions = self.query_encoder.get_query_embed(
                    modality='text',
                    text=batch_text,
                )
            else:
                raise NotImplementedError(f"Query encoder type {self.query_encoder_type} not fully handled with STFT input.")

            input_dict = {
                'stft_mixture_mag': mixture_mag,
                'stft_mixture_cos': mixture_cos,
                'stft_mixture_sin': mixture_sin,
                'condition': conditions,
            }

            target_dict = {
                'waveform': target_waveforms.squeeze(1),
            }

            sep_waveform = self.ss_model(input_dict, target_waveform=target_waveforms)['waveform']
            sep_waveform = sep_waveform.squeeze(1)

            output_dict = {
                'waveform': sep_waveform,
            }

        loss = self.loss_function(output_dict, target_dict)

        self.log(
            "val_loss", 
            loss, 
            batch_size=self.trainer.datamodule.batch_size,
            on_step=False, 
            on_epoch=True, 
            prog_bar=True, 
            logger=True, 
            sync_dist=True
        )
        
        return loss

    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        r"""Configure optimizer.
        """

        if self.optimizer_type == "AdamW":
            optimizer = optim.AdamW(
                params=self.ss_model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.0,
                amsgrad=True,
            )
        else:
            raise NotImplementedError

        scheduler = LambdaLR(optimizer, self.lr_lambda_func)

        output_dict = {
            "optimizer": optimizer,
            "lr_scheduler": {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }

        return output_dict
    

def get_model_class(model_type):
    if model_type == 'ResUNet30':
        from models.resunet import ResUNet30
        return ResUNet30

    else:
        raise NotImplementedError
