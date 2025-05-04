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
        selected_win_len = stft_win_lengths[0]

        try:
            mixture_mag = batch_stfts['mixture'][selected_win_len][0]
            mixture_cos = batch_stfts['mixture'][selected_win_len][1]
            mixture_sin = batch_stfts['mixture'][selected_win_len][2]
        except (KeyError, TypeError, IndexError) as e:
            print(f"Error accessing STFT data: {e}")
            print(f"Selected window length: {selected_win_len}")
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
        sep_waveform = self.ss_model(input_dict)['waveform']
        sep_waveform = sep_waveform.squeeze(1)

        output_dict = {
            'waveform': sep_waveform,
        }

        loss = self.loss_function(output_dict, target_dict)

        self.log_dict({"train_loss": loss})
        
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
