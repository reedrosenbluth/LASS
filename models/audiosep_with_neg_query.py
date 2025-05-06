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

    
    def training_step(self, batch_data_dict: Dict[str, Any], batch_idx: int):
        """
        Forward one mini-batch, compute loss, and back-prop.

        Expected keys in `batch_data_dict`
        ├─ 'text'                     : List[str]                 (positive captions)
        ├─ 'mixture_component_texts'  : List[List[str]] (optional – 2nd elt ⇒ negative caption)
        ├─ 'stfts'                    : nested dict  (see pre-compute code)
        ├─ 'stft_win_lengths'         : collated win-lengths
        └─ 'target_waveform'          : Tensor (B,1,T)
        """
        
        random.seed(batch_idx)

        # ----------- 1) POS / NEG captions --------------------------------
        pos_caps: list[str] = batch_data_dict["text"]

        mix_lists = batch_data_dict.get("mixture_component_texts")  # can be missing
        if mix_lists is None:
            neg_caps = [""] * len(pos_caps)
        else:
            neg_caps = [
                lst[1] if isinstance(lst, (list, tuple)) and len(lst) > 1 else ""
                for lst in mix_lists
            ]
            # keep list length aligned
            if len(neg_caps) != len(pos_caps):
                neg_caps = (neg_caps + [""] * len(pos_caps))[: len(pos_caps)]

        # ----------- 2) pick the STFT (fixed 512-win) ---------------------
        batch_stfts          = batch_data_dict["stfts"]
        stft_win_lengths_raw = batch_data_dict["stft_win_lengths"]
        target_waveforms     = batch_data_dict["target_waveform"]

        # reconstruct available win-lengths from the collated structure
        if (isinstance(stft_win_lengths_raw, list)
            and all(isinstance(t, torch.Tensor) for t in stft_win_lengths_raw)):
            available_lengths = [int(t[0].item()) for t in stft_win_lengths_raw]
        elif (isinstance(stft_win_lengths_raw, list)
              and len(stft_win_lengths_raw) == 1
              and isinstance(stft_win_lengths_raw[0], list)):
            available_lengths = stft_win_lengths_raw[0]
        elif all(isinstance(l, int) for l in stft_win_lengths_raw):
            available_lengths = stft_win_lengths_raw
        else:
            raise TypeError("Cannot parse 'stft_win_lengths' in batch.")

        desired_win_len = 512
        if desired_win_len not in available_lengths:
            raise ValueError(f"Window {desired_win_len} missing – got {available_lengths}")

        mixture_mag, mixture_cos, mixture_sin = batch_stfts["mixture"][desired_win_len]

        # ----------- 3) condition embedding (CLAP pos+neg) ----------------
        if self.query_encoder_type == "CLAP":
            conditions = self.query_encoder.get_query_embed(
                modality="text",
                text=pos_caps,
                text_neg=neg_caps,   # ← negative captions integrated
            )
        else:
            raise NotImplementedError(
                f"Query encoder '{self.query_encoder_type}' not supported."
            )

        # ----------- 4) separation forward + loss -------------------------
        input_dict = {
            "stft_mixture_mag": mixture_mag,
            "stft_mixture_cos": mixture_cos,
            "stft_mixture_sin": mixture_sin,
            "condition": conditions,
        }
        target_dict = {"waveform": target_waveforms.squeeze(1)}

        self.ss_model.train()
        sep_waveform = self.ss_model(input_dict, target_waveform=target_waveforms)["waveform"].squeeze(1)

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

