import numpy as np
from typing import Dict, List, NoReturn, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import STFT, ISTFT, magphase
from models.base import Base, init_layer, init_bn, act


class FiLM(nn.Module):
    def __init__(self, film_meta, condition_size):
        super(FiLM, self).__init__()

        self.condition_size = condition_size

        self.modules, _ = self.create_film_modules(
            film_meta=film_meta, 
            ancestor_names=[],
        )
        
    def create_film_modules(self, film_meta, ancestor_names):

        modules = {}
       
        # Pre-order traversal of modules
        for module_name, value in film_meta.items():

            if isinstance(value, int):

                ancestor_names.append(module_name)
                unique_module_name = '->'.join(ancestor_names)

                modules[module_name] = self.add_film_layer_to_module(
                    num_features=value, 
                    unique_module_name=unique_module_name,
                )

            elif isinstance(value, dict):

                ancestor_names.append(module_name)
                
                modules[module_name], _ = self.create_film_modules(
                    film_meta=value, 
                    ancestor_names=ancestor_names,
                )

            ancestor_names.pop()

        return modules, ancestor_names

    def add_film_layer_to_module(self, num_features, unique_module_name):

        layer = nn.Linear(self.condition_size, num_features)
        init_layer(layer)
        self.add_module(name=unique_module_name, module=layer)

        return layer

    def forward(self, conditions):
        
        film_dict = self.calculate_film_data(
            conditions=conditions, 
            modules=self.modules,
        )

        return film_dict

    def calculate_film_data(self, conditions, modules):

        film_data = {}

        # Pre-order traversal of modules
        for module_name, module in modules.items():

            if isinstance(module, nn.Module):
                film_data[module_name] = module(conditions)[:, :, None, None]

            elif isinstance(module, dict):
                film_data[module_name] = self.calculate_film_data(conditions, module)

        return film_data


class ConvBlockRes(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        momentum: float,
        has_film,
    ):
        r"""Residual block."""
        super(ConvBlockRes, self).__init__()

        padding = [kernel_size[0] // 2, kernel_size[1] // 2]

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )
            self.is_shortcut = True
        else:
            self.is_shortcut = False

        self.has_film = has_film

        self.init_weights()

    def init_weights(self) -> NoReturn:
        r"""Initialize weights."""
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_layer(self.conv1)
        init_layer(self.conv2)

        if self.is_shortcut:
            init_layer(self.shortcut)

    def forward(self, input_tensor: torch.Tensor, film_dict: Dict) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        b1 = film_dict['beta1']
        b2 = film_dict['beta2']

        x = self.conv1(F.leaky_relu_(self.bn1(input_tensor) + b1, negative_slope=0.01))
        x = self.conv2(F.leaky_relu_(self.bn2(x) + b2, negative_slope=0.01))

        if self.is_shortcut:
            return self.shortcut(input_tensor) + x
        else:
            return input_tensor + x


class EncoderBlockRes1B(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        downsample: Tuple,
        momentum: float,
        has_film,
    ):
        r"""Encoder block, contains 8 convolutional layers."""
        super(EncoderBlockRes1B, self).__init__()

        self.conv_block1 = ConvBlockRes(
            in_channels, out_channels, kernel_size, momentum, has_film,
        )
        self.downsample = downsample

    def forward(self, input_tensor: torch.Tensor, film_dict: Dict) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            encoder_pool: (batch_size, output_feature_maps, downsampled_time_steps, downsampled_freq_bins)
            encoder: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        encoder = self.conv_block1(input_tensor, film_dict['conv_block1'])
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder


class DecoderBlockRes1B(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        upsample: Tuple,
        momentum: float,
        has_film,
    ):
        r"""Decoder block, contains 1 transposed convolutional and 8 convolutional layers."""
        super(DecoderBlockRes1B, self).__init__()
        self.kernel_size = kernel_size
        self.stride = upsample

        self.conv1 = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.stride,
            stride=self.stride,
            padding=(0, 0),
            bias=False,
            dilation=(1, 1),
        )

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.conv_block2 = ConvBlockRes(
            out_channels * 2, out_channels, kernel_size, momentum, has_film,
        )
        self.bn2 = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.has_film = has_film

        self.init_weights()

    def init_weights(self):
        r"""Initialize weights."""
        init_bn(self.bn1)
        init_layer(self.conv1)

    def forward(
        self, input_tensor: torch.Tensor, concat_tensor: torch.Tensor, film_dict: Dict,
    ) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, downsampled_time_steps, downsampled_freq_bins)
            concat_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        # b1 = film_dict['beta1']

        b1 = film_dict['beta1']
        x = self.conv1(F.leaky_relu_(self.bn1(input_tensor) + b1))
        # (batch_size, input_feature_maps, time_steps, freq_bins)

        x = torch.cat((x, concat_tensor), dim=1)
        # (batch_size, input_feature_maps * 2, time_steps, freq_bins)

        x = self.conv_block2(x, film_dict['conv_block2'])
        # output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)

        return x


class ResUNet30_Base(nn.Module, Base):
    def __init__(self, input_channels, output_channels):
        super(ResUNet30_Base, self).__init__()

        # --- Determine n_fft based on usage --- #
        # The training loop explicitly uses STFT corresponding to n_fft=512.
        # Therefore, bn0 must be initialized accordingly.
        n_fft = 512  # Changed from hardcoded window_size=1024

        # STFT parameters (used for ISTFT and reference)
        # Keep original window_size for ISTFT if it differs?
        # For now, assume ISTFT params should also align if possible,
        # but primary fix is for bn0.
        window_size = n_fft # Use n_fft for consistency where needed
        hop_size = 160
        center = True
        pad_mode = "reflect"
        window = "hann"
        momentum = 0.01

        self.output_channels = output_channels
        self.target_sources_num = 1
        self.K = 3
        
        self.time_downsample_ratio = 2 ** 5  # This number equals 2^{#encoder_blcoks}

        self.istft = ISTFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        # Initialize bn0 with features corresponding to n_fft=512
        num_freq_bins = n_fft // 2 + 1
        self.bn0 = nn.BatchNorm2d(num_freq_bins, momentum=momentum)

        self.pre_conv = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=32, 
            kernel_size=(1, 1), 
            stride=(1, 1), 
            padding=(0, 0), 
            bias=True,
        )

        self.encoder_block1 = EncoderBlockRes1B(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block2 = EncoderBlockRes1B(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block3 = EncoderBlockRes1B(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block4 = EncoderBlockRes1B(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block5 = EncoderBlockRes1B(
            in_channels=256,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block6 = EncoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 2),
            momentum=momentum,
            has_film=True,
        )
        self.conv_block7a = EncoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 1),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block1 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(1, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block2 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block3 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=256,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block4 = DecoderBlockRes1B(
            in_channels=256,
            out_channels=128,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block5 = DecoderBlockRes1B(
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block6 = DecoderBlockRes1B(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )

        self.after_conv = nn.Conv2d(
            in_channels=32,
            out_channels=self.output_channels * self.K,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.pre_conv)
        init_layer(self.after_conv)

    def forward(self, stft_magnitude, cos_in, sin_in, film_dict, target_length: int):
        """
        Args:
          stft_magnitude: Mixture magnitude (batch_size, channels_num, time_steps, freq_bins)
          cos_in: Mixture phase cosine (batch_size, channels_num, time_steps, freq_bins)
          sin_in: Mixture phase sine (batch_size, channels_num, time_steps, freq_bins)
          film_dict: Dictionary containing FiLM modulation parameters.
          target_length: Target length of the waveform

        Outputs:
          output_dict: {
            'waveform': (batch_size, channels_num, segment_samples) # Estimated waveform
          }
        """

        # Input is STFT magnitude
        x = stft_magnitude # (batch_size, input_channels=1, time_steps, freq_bins)

        if x.dim() == 5 and x.shape[1] == 1:
            x = x.squeeze(1) # Squeeze the second dimension -> (B, C, T, F)
            if cos_in.dim() == 5 and cos_in.shape[1] == 1:
                cos_in = cos_in.squeeze(1)
            if sin_in.dim() == 5 and sin_in.shape[1] == 1:
                sin_in = sin_in.squeeze(1)

        sp = x

        # Batch normalization
        if x.shape[1] == 1:
            x = x.permute(0, 3, 2, 1) # (B, F, T, 1)
            x = self.bn0(x)
            x = x.permute(0, 3, 2, 1) # (B, 1, T, F)
        else:
             raise NotImplementedError("BN logic needs review for input channels > 1")

        # Pad time axis
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio)) * self.time_downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len)) # Pad T axis
        sp_padded_time = F.pad(sp, pad=(0, 0, 0, pad_len)) # Also pad original mag for masking
        cos_in_padded_time = F.pad(cos_in, pad=(0, 0, 0, pad_len))
        sin_in_padded_time = F.pad(sin_in, pad=(0, 0, 0, pad_len))

        # Pad frequency axis if needed
        origin_freq_bins = x.shape[-1]
        if origin_freq_bins % 2 != 0:
            x = x[..., 0 : origin_freq_bins - 1]
            sp_padded_time_freq = sp_padded_time[..., 0 : origin_freq_bins - 1]
            cos_in_padded = cos_in_padded_time[..., 0 : origin_freq_bins - 1]
            sin_in_padded = sin_in_padded_time[..., 0 : origin_freq_bins - 1]
        else:
            sp_padded_time_freq = sp_padded_time
            cos_in_padded = cos_in_padded_time
            sin_in_padded = sin_in_padded_time

        # UNet
        x = self.pre_conv(x)
        x1_pool, x1 = self.encoder_block1(x, film_dict['encoder_block1'])
        x2_pool, x2 = self.encoder_block2(x1_pool, film_dict['encoder_block2'])
        x3_pool, x3 = self.encoder_block3(x2_pool, film_dict['encoder_block3'])
        x4_pool, x4 = self.encoder_block4(x3_pool, film_dict['encoder_block4'])
        x5_pool, x5 = self.encoder_block5(x4_pool, film_dict['encoder_block5'])
        x6_pool, x6 = self.encoder_block6(x5_pool, film_dict['encoder_block6'])
        x_center, _ = self.conv_block7a(x6_pool, film_dict['conv_block7a'])
        x7 = self.decoder_block1(x_center, x6, film_dict['decoder_block1'])
        x8 = self.decoder_block2(x7, x5, film_dict['decoder_block2'])
        x9 = self.decoder_block3(x8, x4, film_dict['decoder_block3'])
        x10 = self.decoder_block4(x9, x3, film_dict['decoder_block4'])
        x11 = self.decoder_block5(x10, x2, film_dict['decoder_block5'])
        x12 = self.decoder_block6(x11, x1, film_dict['decoder_block6'])

        x = self.after_conv(x12)

        batch_size, _, time_steps_padded, freq_bins_padded = x.shape

        x = x.reshape(
            batch_size,
            self.target_sources_num, # Should be 1
            self.output_channels,    # Should be 1
            self.K, # 3
            time_steps_padded,
            freq_bins_padded,
        )

        # Extract predicted mask components (same as original feature_maps_to_wav)
        # Ensure masks have shape (B, C, T, F) where C=1 for consistency
        # Slice x (B, src=1, chan=1, K=3, T, F) -> (8, 1, 1, 3, 1024, 256)

        # Slice K=0 for mag, squeeze src/chan dims, add C dim back -> (B, C=1, T, F)
        mask_mag = torch.sigmoid(x[:, 0, 0, 0, :, :]).unsqueeze(1)

        # Slice K=1, K=2 for phase masks -> (B, T, F)
        _mask_real = torch.tanh(x[:, 0, 0, 1, :, :])
        _mask_imag = torch.tanh(x[:, 0, 0, 2, :, :])

        # magphase output: (B, C=1, T, F)
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)

        # Explicitly ensure phase masks have channel dim (should be redundant but safe)
        if mask_cos.dim() == 3:
            mask_cos = mask_cos.unsqueeze(1)
        if mask_sin.dim() == 3:
            mask_sin = mask_sin.unsqueeze(1)
        
        # Apply masks and mixture phase (same as original feature_maps_to_wav)
        # Note: Using padded versions of input mag/phase

        out_cos = (cos_in_padded * mask_cos - sin_in_padded * mask_sin)
        out_sin = (sin_in_padded * mask_cos + cos_in_padded * mask_sin)
        out_mag = F.relu_(sp_padded_time_freq * mask_mag) 

        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin

        # Pad frequency dimension back if it was reduced
        if origin_freq_bins % 2 != 0:
            out_real = F.pad(out_real, pad=(0, 1))
            out_imag = F.pad(out_imag, pad=(0, 1))

        # ISTFT requires shape (N, 1, T, F)
        shape = (batch_size * self.target_sources_num * self.output_channels, 1, time_steps_padded, origin_freq_bins)
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # Determine original audio length (needs to be passed or inferred)
        # This is tricky. ISTFT needs the original length BEFORE STFT padding.
        # We don't have the original mixture waveform length here easily.
        # Option 1: Pass original length in input_dict from AudioSep.
        # Option 2: Estimate length from T_orig * hop_size? Risky.
        # Let's assume original_length is needed. For now, use T_orig * hop_size as placeholder.
        # TODO: Pass actual original waveform length for accurate iSTFT
        # estimated_audio_length = origin_len * self.istft.hop_length
        # Use the provided target_length instead

        # Perform ISTFT
        waveform_padded_time = self.istft(out_real, out_imag, target_length) # Uses target length

        # Reshape and potentially trim ISTFT output (ISTFT handles length)
        waveform = waveform_padded_time.reshape(
             batch_size, self.target_sources_num * self.output_channels, waveform_padded_time.shape[-1]
        )
        # If ISTFT output length is longer than expected due to padding, trim?
        # audio_length = mixtures.shape[2] # Need original length 
        # waveform = waveform[..., :audio_length] # Needs audio_length!

        output_dict = {'waveform': waveform} # Output waveform

        return output_dict


def get_film_meta(module):

    film_meta = {}

    if hasattr(module, 'has_film'):\

        if module.has_film:
            film_meta['beta1'] = module.bn1.num_features
            film_meta['beta2'] = module.bn2.num_features
        else:
            film_meta['beta1'] = 0
            film_meta['beta2'] = 0

    for child_name, child_module in module.named_children():

        child_meta = get_film_meta(child_module)

        if len(child_meta) > 0:
            film_meta[child_name] = child_meta
    
    return film_meta


class ResUNet30(nn.Module):
    def __init__(self, input_channels, output_channels, condition_size):
        super(ResUNet30, self).__init__()

        self.base = ResUNet30_Base(
            input_channels=input_channels, 
            output_channels=output_channels,
        )
        
        self.film_meta = get_film_meta(
            module=self.base,
        )
        
        self.film = FiLM(
            film_meta=self.film_meta, 
            condition_size=condition_size
        )


    def forward(self, input_dict, target_waveform):
        # Extract mixture STFT components and condition
        stft_mixture_mag = input_dict['stft_mixture_mag']
        stft_mixture_cos = input_dict['stft_mixture_cos']
        stft_mixture_sin = input_dict['stft_mixture_sin']
        conditions = input_dict['condition']

        film_dict = self.film(
            conditions=conditions,
        )

        # Get target length
        target_length = target_waveform.shape[-1]

        # Pass magnitude, phase, and target_length to base model
        output_dict = self.base(
            stft_magnitude=stft_mixture_mag,
            cos_in=stft_mixture_cos,
            sin_in=stft_mixture_sin,
            film_dict=film_dict,
            target_length=target_length,
        )

        return output_dict

    def chunk_inference(self, input_dict):
        raise NotImplementedError("Chunk inference not updated for STFT input / waveform output yet.")


