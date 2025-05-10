import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, momentum, has_film):
        super().__init__()
        self.has_film = has_film
        self.bn1_num_features = in_channels 
        self.bn2_num_features = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size[0]//2, kernel_size[1]//2), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size[0]//2, kernel_size[1]//2), bias=False)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), bias=False)
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, input_tensor, film_dict=None):
        if self.has_film and film_dict is not None:
            assert 'beta1' in film_dict, "film_dict for ConvBlockRes missing 'beta1'"
            assert 'beta2' in film_dict, "film_dict for ConvBlockRes missing 'beta2'"

        x = self.conv1(input_tensor) 
        x = self.conv2(x)
        if self.is_shortcut:
            return self.shortcut(input_tensor) + x
        else:
            return input_tensor + x

class DummyEncoderBlockRes1B(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample, momentum, has_film):
        super().__init__()
        self.has_film = has_film
        self.conv_block1 = DummyConvBlockRes(in_channels, out_channels, kernel_size, momentum, has_film)
        self.downsample = downsample
        self.avg_pool = nn.AvgPool2d(kernel_size=downsample)

    def forward(self, input_tensor, film_dict=None):
        conv_block_film = None
        if self.has_film and film_dict is not None:
            assert 'conv_block1' in film_dict, "film_dict for EncoderBlockRes1B missing 'conv_block1' for its DummyConvBlockRes"
            conv_block_film = film_dict['conv_block1']

        encoder = self.conv_block1(input_tensor, conv_block_film)
        encoder_pool = self.avg_pool(encoder)
        return encoder_pool, encoder

class DummyDecoderBlockRes1B(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, upsample, momentum, has_film, skip_concat_channels=None):
        super().__init__()
        self.has_film = has_film
        self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=upsample, stride=upsample)
        
        actual_skip_channels = skip_concat_channels if skip_concat_channels is not None else out_channels
        conv_block2_in_channels = out_channels + actual_skip_channels
        
        self.conv_block2 = DummyConvBlockRes(conv_block2_in_channels, out_channels, kernel_size, momentum, has_film)

    def forward(self, input_tensor, concat_tensor, film_dict=None):
        conv_block_film = None
        if self.has_film and film_dict is not None:
            assert 'conv_block2' in film_dict, "film_dict for DecoderBlockRes1B missing 'conv_block2' for its DummyConvBlockRes"
            conv_block_film = film_dict['conv_block2']

        x = self.trans_conv(input_tensor) 
        x = torch.cat((x, concat_tensor), dim=1)
        x = self.conv_block2(x, conv_block_film)
        return x

batch_size = 2
input_channels_stft = 1
time_frames = 64
num_freq_bins = 1025
    
window_lengths = [256, 512, 2048]
num_resolutions = len(window_lengths)
    
pre_conv_out_channels = 32
enc1_out_channels = 32
    
enc2_in_channels_fused = enc1_out_channels * num_resolutions
enc2_out_channels_shared = 64
    
dec6_transposed_conv_out_channels = 32
fused_skip_channels_for_dec6 = enc1_out_channels * num_resolutions
dec6_conv_block_in_channels_fused = dec6_transposed_conv_out_channels + fused_skip_channels_for_dec6
dec6_conv_block_out_channels_final = 32
    
momentum = 0.01
kernel_size_conv = (3,3)
downsample_enc1 = (2,2)

print("--- Initializing Modules ---")
mock_pre_convs = nn.ModuleDict()
mock_encoder_block1s = nn.ModuleDict()

for wl_str in map(str, window_lengths):
    mock_pre_convs[wl_str] = nn.Conv2d(input_channels_stft, pre_conv_out_channels, kernel_size=(1,1))
    mock_encoder_block1s[wl_str] = DummyEncoderBlockRes1B(
        pre_conv_out_channels, enc1_out_channels, kernel_size_conv, 
        downsample_enc1, momentum, has_film=True
    )
print("Parallel input modules initialized.")

mock_shared_encoder_block2 = nn.Conv2d(
    enc2_in_channels_fused, enc2_out_channels_shared, kernel_size_conv, padding=1
) 
print(f"Mock shared encoder block initialized (expecting {enc2_in_channels_fused} in_channels).")

mock_decoder_block6_conv_block = DummyConvBlockRes(
    dec6_conv_block_in_channels_fused,
    dec6_conv_block_out_channels_final, 
    kernel_size_conv, 
    momentum, 
    has_film=True
)
print(f"Mock decoder's ConvBlockRes initialized (expecting {dec6_conv_block_in_channels_fused} in_channels).")

print("--- Creating Mock Data ---")
mock_stft_inputs_dict = {}
for wl_str in map(str, window_lengths):
    mock_stft_inputs_dict[wl_str] = torch.randn(batch_size, input_channels_stft, time_frames, num_freq_bins)
    print(f"Mock STFT input for WL {wl_str}: {mock_stft_inputs_dict[wl_str].shape}")

mock_film_dict_for_test = {
    'encoder_block1s': {},
    'decoder_block6': {
        'conv_block2': {
            'beta1': torch.randn(batch_size, dec6_conv_block_out_channels_final, 1, 1),
            'beta2': torch.randn(batch_size, dec6_conv_block_out_channels_final, 1, 1)
        }
    }
}
for wl_str in map(str, window_lengths):
    mock_film_dict_for_test['encoder_block1s'][wl_str] = {
        'conv_block1': {
            'beta1': torch.randn(batch_size, enc1_out_channels, 1, 1),
            'beta2': torch.randn(batch_size, enc1_out_channels, 1, 1)
        }
    }
print("Mock FiLM dictionary created.")

list_intermediate_pool_outputs = []
list_intermediate_skip_outputs = []

print("--- Parallel Processing Stage ---")
for wl_str in map(str, window_lengths):
    current_stft_input = mock_stft_inputs_dict[wl_str]
    
    pre_conv_output = mock_pre_convs[wl_str](current_stft_input)
    
    film_for_this_enc1 = mock_film_dict_for_test['encoder_block1s'][wl_str]
    
    pool_out, skip_out = mock_encoder_block1s[wl_str](pre_conv_output, film_for_this_enc1)
    
    list_intermediate_pool_outputs.append(pool_out)
    list_intermediate_skip_outputs.append(skip_out)

print("--- Fusion Stage ---")
fused_pool_features = torch.cat(list_intermediate_pool_outputs, dim=1)
fused_skip_features = torch.cat(list_intermediate_skip_outputs, dim=1)

print(f"Fused Pool Features Shape: {fused_pool_features.shape}")
assert fused_pool_features.shape[0] == batch_size, "Batch size mismatch in fused pool features"
assert fused_pool_features.shape[1] == enc1_out_channels * num_resolutions, "Channel count mismatch in fused pool features"
assert fused_pool_features.shape[2] == time_frames // downsample_enc1[0], "Time dimension mismatch in fused pool features"
assert fused_pool_features.shape[3] == num_freq_bins // downsample_enc1[1], "Frequency dimension mismatch in fused pool features"

print(f"Fused Skip Features Shape: {fused_skip_features.shape}")
assert fused_skip_features.shape[0] == batch_size, "Batch size mismatch in fused skip features"
assert fused_skip_features.shape[1] == enc1_out_channels * num_resolutions, "Channel count mismatch in fused skip features"
assert fused_skip_features.shape[2] == time_frames, "Time dimension mismatch in fused skip features (should be pre-downsampling)"
assert fused_skip_features.shape[3] == num_freq_bins, "Frequency dimension mismatch in fused skip features (should be pre-downsampling)"

print("Shared Encoder Layer Stage (Mocked)")
shared_enc2_output = mock_shared_encoder_block2(fused_pool_features)
print(f"Output of Mock Shared Enc2 Shape: {shared_enc2_output.shape}")
assert shared_enc2_output.shape[1] == enc2_out_channels_shared, "Output channel count mismatch from shared encoder"

print("Mock Decoder Skip Connection Handling Stage")
mock_upsampled_from_prev_decoder_layer = torch.randn(
    batch_size, 
    dec6_transposed_conv_out_channels,
    fused_skip_features.shape[2],
    fused_skip_features.shape[3]
) 
print(f"Mock Upsampled Tensor (from prev. decoder layer): {mock_upsampled_from_prev_decoder_layer.shape}")

decoder_concat_input_tensor = torch.cat((mock_upsampled_from_prev_decoder_layer, fused_skip_features), dim=1)
print(f"Decoder Concat Tensor (input to ConvBlock): {decoder_concat_input_tensor.shape}")
assert decoder_concat_input_tensor.shape[1] == dec6_conv_block_in_channels_fused, "Channel count mismatch in decoder concat tensor"

film_for_dec6_conv_block = mock_film_dict_for_test['decoder_block6']['conv_block2']
final_decoder_block_output = mock_decoder_block6_conv_block(decoder_concat_input_tensor, film_for_dec6_conv_block)
print(f"Mock Decoder's ConvBlockRes Output Shape: {final_decoder_block_output.shape}")
assert final_decoder_block_output.shape[1] == dec6_conv_block_out_channels_final, "Output channel count mismatch from decoder's ConvBlockRes"

print("Isolated Fusion Test Script Complete")