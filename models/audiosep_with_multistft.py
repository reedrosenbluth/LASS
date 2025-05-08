# models/audiosep.py
from __future__ import annotations
from typing import Any, Dict, List

import random
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from huggingface_hub import PyTorchModelHubMixin

from models.clap_encoder import CLAP_Encoder


# --------------------------------------------------------------------------- #
#                               helper utilities                              #
# --------------------------------------------------------------------------- #

def _available_lengths(stft_win_lengths: List[Any]) -> List[int]:
    """Recover window-length list after PyTorch collation."""
    if all(isinstance(t, torch.Tensor) for t in stft_win_lengths):
        return [int(t[0]) for t in stft_win_lengths]
    if (len(stft_win_lengths) == 1
            and isinstance(stft_win_lengths[0], list)):
        return stft_win_lengths[0]
    if all(isinstance(x, int) for x in stft_win_lengths):
        return list(stft_win_lengths)
    raise TypeError("Cannot parse 'stft_win_lengths' inside batch.")


def _split_mix_dict(mix_dict: Dict[int, tuple[torch.Tensor, torch.Tensor,
                                              torch.Tensor]]
                    ) -> tuple[dict[int, torch.Tensor],
                               dict[int, torch.Tensor],
                               dict[int, torch.Tensor]]:
    """Turn  {win: (mag,cos,sin)}  → 3 parallel dicts."""
    mags, coss, sins = {}, {}, {}
    for k, (mag, cos, sin) in mix_dict.items():
        mags[k], coss[k], sins[k] = mag, cos, sin
    return mags, coss, sins


# --------------------------------------------------------------------------- #
#                             Lightning wrapper                               #
# --------------------------------------------------------------------------- #

class AudioSep(pl.LightningModule, PyTorchModelHubMixin):
    """
    Lightning module for AudioSep with **multi-resolution STFT** support.
    Expects batch['stfts']['mixture'] to be a dict{win_len: (mag,cos,sin)}.
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

    # --------------------------------------------------------------------- #
    #                         TRAIN / VALIDATION STEP                       #
    # --------------------------------------------------------------------- #

    def _step_core(self, batch_data: Dict[str, Any], train: bool):
        # 1) unpack
        batch_text       = batch_data["text"]
        stfts_nested     = batch_data["stfts"]               # dict
        win_len_collated = batch_data["stft_win_lengths"]
        target_wave      = batch_data["target_waveform"]

        win_lens = _available_lengths(win_len_collated)

        # 2) split mixture into mag / cos / sin dicts
        mix_mag, mix_cos, mix_sin = _split_mix_dict(stfts_nested["mixture"])

        # 3) build condition vector
        if self.query_encoder_type == "CLAP":
            cond = self.query_encoder.get_query_embed(
                modality="text",
                text=batch_text,
            )
        else:
            raise NotImplementedError(
                f"Query encoder '{self.query_encoder_type}' unsupported.")

        # 4) forward separator
        input_dict = {
            "stft_mixture_mag": mix_mag,
            "stft_mixture_cos": mix_cos,
            "stft_mixture_sin": mix_sin,
            "condition": cond,
        }
        target_dict = {"waveform": target_wave.squeeze(1)}

        if train:
            self.ss_model.train()
            out = self.ss_model(input_dict, target_wave)
        else:
            self.ss_model.eval()
            with torch.no_grad():
                out = self.ss_model(input_dict, target_wave)

        sep_wave = out["waveform"].squeeze(1)
        loss = self.loss_function({"waveform": sep_wave}, target_dict)
        return loss

    # ---------------- Lightning hooks ---------------- #

    def training_step(self, batch, batch_idx):
        random.seed(batch_idx)
        loss = self._step_core(batch, train=True)
        self.log_dict({"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        random.seed(batch_idx)
        loss = self._step_core(batch, train=False)
        self.log("val_loss",
                 loss,
                 batch_size=self.trainer.datamodule.batch_size,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True)
        return loss

    # ------------------------------------------------------------------ #

    def configure_optimizers(self):
        if self.optimizer_type == "AdamW":
            optimiser = optim.AdamW(self.ss_model.parameters(),
                                    lr=self.learning_rate,
                                    betas=(0.9, 0.999),
                                    eps=1e-8,
                                    weight_decay=0.0,
                                    amsgrad=True)
        else:
            raise NotImplementedError

        scheduler = LambdaLR(optimiser, self.lr_lambda_func)
        return {
            "optimizer": optimiser,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # inference not implemented here
    def forward(self, *args, **kwargs):
        raise NotImplementedError


# helper for trainer --------------------------------------------------------- #
def get_model_class(model_type: str):
    if model_type == "ResUNet30":
        from models.resunet import ResUNet30
        return ResUNet30
    raise NotImplementedError
