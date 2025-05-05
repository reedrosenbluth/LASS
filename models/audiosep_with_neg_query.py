# models/audiosep.py
from typing import Dict, Any
import random

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from models.clap_encoder import CLAP_Encoder
from huggingface_hub import PyTorchModelHubMixin


class AudioSep(pl.LightningModule, PyTorchModelHubMixin):
    """
    Lightning wrapper that consumes **pre-computed STFT tensors** and
    **dual-caption (positive + negative) text queries**.
    """

    def __init__(
        self,
        ss_model: nn.Module,                           
        query_encoder: nn.Module = CLAP_Encoder().eval(),
        loss_function=None,
        optimizer_type: str | None = "AdamW",
        learning_rate: float | None = 1e-4,
        lr_lambda_func=None,
        use_text_ratio: float = 1.0,                   
    ):
        super().__init__()
        self.ss_model = ss_model
        self.query_encoder = query_encoder
        self.query_encoder_type = self.query_encoder.encoder_type
        self.use_text_ratio = use_text_ratio

        self.loss_function = loss_function
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.lr_lambda_func = lr_lambda_func

    
    def training_step(self, batch_data_dict, batch_idx):
        """
        Forward a mini-batch, compute loss, and back-prop.

        Expected keys in `batch_data_dict`
        ├─ 'text'                    : List[str]                (positive caption)
        ├─ 'mixture_component_texts': List[List[str]]  (optional, 2nd item ⇒ negative caption)
        ├─ 'stfts'                  : nested dict with 'mixture' / 'segment' tensors
        ├─ 'stft_win_lengths'       : list[Tensor]  (window-lengths present)
        └─ 'target_waveform'        : Tensor (B,1,T)
        """
        
        random.seed(batch_idx)

        
        #OSITIVE + NEGATIVE CAPTIONS
       
        pos_caps: list[str] = batch_data_dict["text"]

        mix_lists = batch_data_dict.get("mixture_component_texts")  # may be None
        neg_caps: list[str] = []

        if mix_lists is None:                       # field absent entirely
            neg_caps = [""] * len(pos_caps)
        else:
            for lst in mix_lists:
                if isinstance(lst, (list, tuple)) and len(lst) > 1:
                    neg_caps.append(lst[1])         # take 2nd sentence
                else:
                    neg_caps.append("")             # fallback: empty

            # safety – keep lengths aligned with positive captions
            if len(neg_caps) != len(pos_caps):
                neg_caps = (neg_caps + [""] * len(pos_caps))[: len(pos_caps)]

       
        batch_stfts          = batch_data_dict["stfts"]
        stft_win_lengths_raw = batch_data_dict["stft_win_lengths"]
        target_waveforms     = batch_data_dict["target_waveform"]

        # reconstruct *available* window-length list from collated structure
        if (isinstance(stft_win_lengths_raw, list) and
            all(isinstance(t, torch.Tensor) for t in stft_win_lengths_raw)):
            available_lengths = [int(t[0].item()) for t in stft_win_lengths_raw]
        elif (isinstance(stft_win_lengths_raw, list) and
              len(stft_win_lengths_raw) == 1 and
              isinstance(stft_win_lengths_raw[0], list)):
            available_lengths = stft_win_lengths_raw[0]
        elif all(isinstance(l, int) for l in stft_win_lengths_raw):
            available_lengths = stft_win_lengths_raw
        else:
            raise TypeError("Cannot parse 'stft_win_lengths' from dataloader.")

        desired_win_len = 512
        if desired_win_len not in available_lengths:
            raise ValueError(
                f"Desired window length {desired_win_len} not found "
                f"in {available_lengths}"
            )

        mixture_mag, mixture_cos, mixture_sin = batch_stfts["mixture"][desired_win_len]

        
        # CONDITION EMBEDDING  (CLAP w/ positive + negative captions)
    
        if self.query_encoder_type == "CLAP":
            conditions = self.query_encoder.get_query_embed(
                modality="text",
                text=pos_caps,
                text_neg=neg_caps,
            )
        else:
            raise NotImplementedError(
                f"Query encoder '{self.query_encoder_type}' not supported."
            )

    
        input_dict = {
            "stft_mixture_mag": mixture_mag,
            "stft_mixture_cos": mixture_cos,
            "stft_mixture_sin": mixture_sin,
            "condition": conditions,
        }
        target_dict = {"waveform": target_waveforms.squeeze(1)}

        self.ss_model.train()
        sep_waveform = self.ss_model(input_dict)["waveform"].squeeze(1)

        loss = self.loss_function({"waveform": sep_waveform}, target_dict)
        self.log_dict({"train_loss": loss})
        return loss


    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use `.training_step` / `.test_step`")

    def test_step(self, batch, batch_idx):
        pass


    def configure_optimizers(self):
        if self.optimizer_type == "AdamW":
            optimizer = optim.AdamW(
                self.ss_model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0,
                amsgrad=True,
            )
        else:
            raise NotImplementedError

        scheduler = LambdaLR(optimizer, self.lr_lambda_func)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# helper to pick backbone at runtime
def get_model_class(model_type: str):
    if model_type == "ResUNet30":
        from models.resunet import ResUNet30
        return ResUNet30
    raise NotImplementedError

