# models/resunet.py   (only the parts that changed are flagged with ### NEW ###)

# models/resunet.py   (only the parts that changed are flagged with ### NEW ###)

import numpy as np
from typing import Dict, List, Tuple, NoReturn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import STFT, ISTFT, magphase

from models.base import Base, init_layer, init_bn, act
from .resunet_blocks import (                 # keep your existing blocks
    ConvBlockRes, EncoderBlockRes1B, DecoderBlockRes1B
)
from .film import FiLM, get_film_meta          # unchanged helpers


# --------------------------------------------------------------------------- #
#                         ResUNet-30 base (multi-res)                         #
# --------------------------------------------------------------------------- #

class ResUNet30_Base(nn.Module, Base):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 win_lengths: List[int] | Tuple[int] = (256, 512, 2048)  # NEW
                 ):
        super().__init__()

        self.win_lengths = [int(w) for w in win_lengths]                 # NEW
        self.output_channels = output_channels
        self.target_sources_num = 1
        self.K = 3
        self.time_downsample_ratio = 2 ** 5     # (=32)

        # ------------ ISTFT (fixed to 512 window for re-synthesis) -------
        n_fft = 512
        hop_size = 160
        self.istft = ISTFT(n_fft=n_fft,
                           hop_length=hop_size,
                           win_length=n_fft,
                           window="hann",
                           center=True,
                           pad_mode="reflect",
                           freeze_parameters=True)

        momentum = 0.01
        num_freq_bins = n_fft // 2 + 1
        self.bn0 = nn.BatchNorm2d(num_freq_bins, momentum=momentum)

        # ---------------------------------------------------------------- #
        #                    PARALLEL  branch per resolution                #
        # ---------------------------------------------------------------- #
        BRCH_OUT = 32
        self.pre_convs = nn.ModuleDict()
        self.encoder_block1s = nn.ModuleDict()
        for wl in self.win_lengths:
            key = str(wl)
            self.pre_convs[key] = nn.Conv2d(input_channels,
                                            BRCH_OUT,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0,
                                            bias=True)
            self.encoder_block1s[key] = EncoderBlockRes1B(
                BRCH_OUT, BRCH_OUT,
                kernel_size=(3, 3),
                downsample=(2, 2),
                momentum=momentum,
                has_film=True)

        FUSED_CH = BRCH_OUT * len(self.win_lengths)  # fused channel count ### NEW ###

        # ---------------------- SHARED encoder path ---------------------- #
        self.encoder_block2 = EncoderBlockRes1B(      # changed in_channels
            in_channels=FUSED_CH,                    # NEW
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True)

        # (blocks 3-7 unchanged)
        self.encoder_block3 = EncoderBlockRes1B(64, 128, (3, 3), (2, 2),
                                               momentum, True)
        self.encoder_block4 = EncoderBlockRes1B(128, 256, (3, 3), (2, 2),
                                               momentum, True)
        self.encoder_block5 = EncoderBlockRes1B(256, 384, (3, 3), (2, 2),
                                               momentum, True)
        self.encoder_block6 = EncoderBlockRes1B(384, 384, (3, 3), (1, 2),
                                               momentum, True)
        self.conv_block7a = EncoderBlockRes1B(384, 384, (3, 3), (1, 1),
                                              momentum, True)

        # -------------------------- decoder path ------------------------- #
        self.decoder_block1 = DecoderBlockRes1B(384, 384, (3, 3), (1, 2),
                                               momentum, True)
        self.decoder_block2 = DecoderBlockRes1B(384, 384, (3, 3), (2, 2),
                                               momentum, True)
        self.decoder_block3 = DecoderBlockRes1B(384, 256, (3, 3), (2, 2),
                                               momentum, True)
        self.decoder_block4 = DecoderBlockRes1B(256, 128, (3, 3), (2, 2),
                                               momentum, True)
        self.decoder_block5 = DecoderBlockRes1B(128, 64, (3, 3), (2, 2),
                                               momentum, True)

        # decoder_block6 needs bigger concat (upsample 32 + fused skip)     ### NEW ###
        DEC6_IN = 64                      # transposed conv in-channels
        DEC6_SKIP_CH = FUSED_CH          # skip channels
        self.decoder_block6 = DecoderBlockRes1B(
            in_channels=DEC6_IN,
            out_channels=32,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
            )   # ConvBlockRes inside will internally compute sum-channels

        self.after_conv = nn.Conv2d(32,
                                    self.output_channels * self.K,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=True)

        # init helpers
        init_bn(self.bn0)
        init_layer(self.after_conv)
        for pc in self.pre_convs.values():
            init_layer(pc)

    # ------------------------------------------------------------------ #
    #                              forward                               #
    # ------------------------------------------------------------------ #

    def forward(self,
                stft_magnitude: Dict[int, torch.Tensor],
                cos_in: Dict[int, torch.Tensor],
                sin_in: Dict[int, torch.Tensor],
                film_dict: Dict,
                target_length: int):
        """
        stft_magnitude / cos_in / sin_in : dict{win_len : (B,1,T,F)}
        """

        # ------------ PARALLEL encoder-stage per window length --------- #
        pool_list, skip_list = [], []
        for wl in self.win_lengths:
            key = str(wl)

            x = stft_magnitude[wl]                       # (B,1,T,F)
            if x.dim() == 5 and x.shape[1] == 1:         # squeeze collation dim
                x = x.squeeze(1)
            # BN on 512-fft only (same as before)
            if x.shape[1] == 1:
                x_ = self.bn0(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
            else:
                raise NotImplementedError

            # pre-conv + enc1
            x_ = self.pre_convs[key](x_)
            p, s = self.encoder_block1s[key](
                x_,
                film_dict["encoder_block1s"][key])
            pool_list.append(p)
            skip_list.append(s)

        x1_pool = torch.cat(pool_list, dim=1)    # fused pooled   (B,FUSED,T/2,F/2)
        x1      = torch.cat(skip_list, dim=1)    # fused skip     (B,FUSED,T,  F)

        # ---------------- shared encoder / decoder path ---------------- #
        x2_pool, x2 = self.encoder_block2(x1_pool, film_dict["encoder_block2"])
        x3_pool, x3 = self.encoder_block3(x2_pool, film_dict["encoder_block3"])
        x4_pool, x4 = self.encoder_block4(x3_pool, film_dict["encoder_block4"])
        x5_pool, x5 = self.encoder_block5(x4_pool, film_dict["encoder_block5"])
        x6_pool, x6 = self.encoder_block6(x5_pool, film_dict["encoder_block6"])
        x_center, _ = self.conv_block7a(x6_pool, film_dict["conv_block7a"])

        x7  = self.decoder_block1(x_center, x6, film_dict["decoder_block1"])
        x8  = self.decoder_block2(x7,      x5, film_dict["decoder_block2"])
        x9  = self.decoder_block3(x8,      x4, film_dict["decoder_block3"])
        x10 = self.decoder_block4(x9,      x3, film_dict["decoder_block4"])
        x11 = self.decoder_block5(x10,     x2, film_dict["decoder_block5"])
        x12 = self.decoder_block6(x11,     x1, film_dict["decoder_block6"])

        x = self.after_conv(x12)   # (B, out_ch*K, T, F)

        # -------------- (the rest identical to previous code) ---------- #
        # convert mask → waveform using 512-fft mixture phase
        # we need the 512-fft mixture phase tensors
        sp   = stft_magnitude[512]
        cos  = cos_in[512]
        sin_ = sin_in[512]

        # (re-use your original magnitude/phase reconstruction code here)
        # ----------------------------------------------------------------
        batch_size, _, T_pad, F_pad = x.shape
        x = x.view(batch_size, 1, self.output_channels, self.K, T_pad, F_pad)
        mask_mag = torch.sigmoid(x[:, 0, 0, 0]).unsqueeze(1)
        _mr, _mi = x[:, 0, 0, 1], x[:, 0, 0, 2]
        _, mask_cos, mask_sin = magphase(_mr, _mi)

        out_cos = cos * mask_cos - sin_ * mask_sin
        out_sin = sin_ * mask_cos + cos * mask_sin
        out_mag = F.relu_(sp * mask_mag)

        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin

        shape = (batch_size, 1, T_pad, F_pad)
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        waveform = self.istft(out_real, out_imag, target_length)
        return {"waveform": waveform}


# --------------------------------------------------------------------------- #
#                          wrapper with FiLM generator                        #
# --------------------------------------------------------------------------- #
class ResUNet30(nn.Module):
    def __init__(self, input_channels, output_channels,
                 condition_size,
                 win_lengths: List[int] | Tuple[int] = (256, 512, 2048)):
        super().__init__()
        self.base = ResUNet30_Base(input_channels, output_channels,
                                   win_lengths=win_lengths)

        self.film_meta = get_film_meta(self.base)
        self.film = FiLM(self.film_meta, condition_size)

    def forward(self, input_dict, target_waveform):
        film_dict = self.film(input_dict["condition"])
        out = self.base(input_dict["stft_mixture_mag"],
                        input_dict["stft_mixture_cos"],
                        input_dict["stft_mixture_sin"],
                        film_dict,
                        target_waveform.shape[-1])
        return out
