import torch
import numpy as np
import argparse
import yaml
from tqdm import tqdm
import pathlib
from torch.utils.data import DataLoader
from torchlibrosa.stft import STFT, magphase

from data.audiotext_dataset import AudioTextDataset
from data.waveform_mixers import SegmentMixer

def calculate_stft_components(waveform, n_fft, hop_length, win_length, window, center, pad_mode):
    """Calculates STFT magnitude, cosine, and sine components using torchlibrosa.STFT."""
    # waveform shape: (batch, channels, time) -> needs (batch, time) for torchlibrosa.STFT
    if waveform.dim() == 3:
        waveform = waveform.squeeze(1)
    assert waveform.dim() == 2 # Expecting (batch, time)
    
    # Instantiate the STFT extractor from torchlibrosa
    stft_extractor = STFT(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window, 
        center=center,
        pad_mode=pad_mode,
        freeze_parameters=True,
    ).to(waveform.device)

    # Calculate STFT
    # Output shapes: (batch_size, 1, time_steps, n_fft // 2 + 1)
    real, imag = stft_extractor(waveform)
    
    # Calculate magnitude and phase components
    # Input shapes: (batch_size, 1, time_steps, n_fft // 2 + 1)
    # Output shapes: (batch_size, 1, time_steps, n_fft // 2 + 1)
    magnitude, cos_phase, sin_phase = magphase(real, imag)

    # The output shape already matches the expected (batch, channels, time_steps, freq_bins)
    # with channels=1.
    return magnitude, cos_phase, sin_phase

def save_precomputed_data(output_dir, index, data_dict):
    """Saves the precomputed STFT data and text."""
    filename = output_dir / f"item_{index:09d}.pt"
    # Detach tensors before saving
    for key in ['mixture_stft', 'segment_stft']:
        if key in data_dict and isinstance(data_dict[key], tuple):
             data_dict[key] = tuple(t.detach().cpu() for t in data_dict[key])
    torch.save(data_dict, filename)


def main(args):
    print("Loading config file...")
    with open(args.config_yaml, 'r') as f:
        configs = yaml.safe_load(f)

    # Extract parameters from config
    sampling_rate = configs['data']['sampling_rate']
    max_clip_len = configs['data']['segment_seconds'] # Use segment_seconds
    max_mix_num = configs['data']['max_mix_num']
    lower_db = configs['data']['loudness_norm']['lower_db']
    higher_db = configs['data']['loudness_norm']['higher_db']
    batch_size = configs['train']['batch_size_per_device']
    num_workers = configs['train']['num_workers']

    print("Initializing dataset...")
    dataset = AudioTextDataset(
        datafiles=args.data_files,
        sampling_rate=sampling_rate, # Use value from config
        max_clip_len=max_clip_len,     # Use value from config
    )
    print(f"Dataset size: {len(dataset)}")

    # Need batch_size >= max_mix_num for SegmentMixer
    # Use batch_size from config
    if batch_size < max_mix_num:
        print(f"Warning: batch_size ({batch_size}) < max_mix_num ({max_mix_num}). Adjusting batch_size.")
        batch_size = max_mix_num

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size, # Use value from config (potentially adjusted)
        shuffle=False, # Keep order for saving with index
        num_workers=num_workers, # Use value from config
        collate_fn=None # Use default collate
    )

    print("Initializing mixer...")
    segment_mixer = SegmentMixer(
        max_mix_num=max_mix_num, # Use value from config
        lower_db=lower_db,     # Use value from config
        higher_db=higher_db,   # Use value from config
    )
    segment_mixer.eval() # Set to eval mode if it has dropout/batchnorm

    print("Creating output directory...")
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting precomputation...")
    processed_count = 0
    global_index = 0 # Keep track of the absolute index across batches

    # STFT parameters (match ResUNet30_Base)
    stft_params = {
        'n_fft': args.n_fft,
        'hop_length': args.hop_length,
        'win_length': args.win_length,
        'window': args.window,
        'center': args.center,
        'pad_mode': args.pad_mode,
    }

    for batch in tqdm(dataloader):
        waveforms = batch['waveform'] # Shape: (batch, 1, time)
        texts = batch['text']       # List of strings
        
        # Ensure waveforms are on the correct device if using GPU
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # waveforms = waveforms.to(device)
        # segment_mixer.to(device) 
        # Note: Keeping on CPU for now for simplicity, add device logic if needed.

        with torch.no_grad():
            # Mix waveforms
            mixtures, segments = segment_mixer(waveforms) # Output shapes: (batch, 1, time)

            # Calculate STFT for mixtures
            mixture_mag, mixture_cos, mixture_sin = calculate_stft_components(
                mixtures, **stft_params
            )
             
            # Calculate STFT for segments
            segment_mag, segment_cos, segment_sin = calculate_stft_components(
                segments, **stft_params
            )

        # Save each item in the batch individually
        for i in range(waveforms.size(0)):
            # Check if we have enough samples left in the dataset
            if global_index < len(dataset):
                 data_to_save = {
                     'mixture_stft': (mixture_mag[i], mixture_cos[i], mixture_sin[i]),
                     'segment_stft': (segment_mag[i], segment_cos[i], segment_sin[i]),
                     'text': texts[i], 
                     # Add any other metadata if needed, e.g., original audio path?
                 }
                 save_precomputed_data(output_dir, global_index, data_to_save)
                 processed_count += 1
            global_index += 1


    print(f"Finished. Precomputed {processed_count} items.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute STFTs for audio data.")

    # --- Config File Argument ---
    parser.add_argument('--config_yaml', type=str, required=True,
                        help='Path to the base configuration YAML file.')

    # --- Dataset/Output Arguments ---
    parser.add_argument('--data_files', type=str, required=True, nargs='+',
                        help='Path(s) to the input dataset JSON file(s) (overrides config if specified).')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save precomputed STFT files.')

    # --- STFT Arguments (Keep defaults matching model) ---
    parser.add_argument('--n_fft', type=int, default=1024, help='STFT FFT size.')
    parser.add_argument('--hop_length', type=int, default=160, help='STFT hop length.')
    parser.add_argument('--win_length', type=int, default=1024, help='STFT window length.')
    parser.add_argument('--window', type=str, default='hann', help='STFT window type.')
    parser.add_argument('--center', type=bool, default=True, help='STFT center.')
    parser.add_argument('--pad_mode', type=str, default='reflect', help='STFT padding mode.')

    args = parser.parse_args()

    # Allow overriding data_files from command line if provided
    # If not provided via command line, attempt to read from config
    if not args.data_files:
        with open(args.config_yaml, 'r') as f:
            configs = yaml.safe_load(f)
        if 'data' in configs and 'datafiles' in configs['data']:
            args.data_files = configs['data']['datafiles']
        else:
            raise ValueError("data_files must be provided either via command line or in the config YAML.")
            
    main(args) 