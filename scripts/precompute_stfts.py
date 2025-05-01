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
    # Waveform is already on the target device
    if waveform.dim() == 3:
        waveform = waveform.squeeze(1)
    assert waveform.dim() == 2 # Expecting (batch, time)
    
    # Instantiate the STFT extractor from torchlibrosa on the correct device
    stft_extractor = STFT(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window, 
        center=center,
        pad_mode=pad_mode,
        freeze_parameters=True,
    ).to(waveform.device) # Move extractor to the same device as waveform

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
    """
    Saves the precomputed STFT data and text for a single item,
    ensuring tensors are on CPU.

    Output File Format (`.pt`):
    Each file contains a dictionary with the following structure:
    {
        'stfts': {
            'mixture': {
                <win_len_1>: (mag_tensor, cos_tensor, sin_tensor),
                <win_len_2>: (mag_tensor, cos_tensor, sin_tensor),
                ...
            },
            'segment': {
                <win_len_1>: (mag_tensor, cos_tensor, sin_tensor),
                <win_len_2>: (mag_tensor, cos_tensor, sin_tensor),
                ...
            }
        },
        'text': str, # Original caption
        'stft_common_params': { # Parameters common to all STFTs
             'hop_length': int,
             'window': str,
             'center': bool,
             'pad_mode': str
        },
        'stft_win_lengths': List[int] # List of window lengths processed
    }
    Where <win_len_1>, <win_len_2>, etc. are the integer window lengths used (e.g., 1024, 2048).
    Each (mag_tensor, cos_tensor, sin_tensor) is a tuple of torch.Tensor objects
    with shape (1, time_steps, freq_bins), moved to CPU.
    """
    filename = output_dir / f"item_{index:09d}.pt"
    # Detach and move tensors to CPU before saving
    cpu_data_dict = {}
    for key, value in data_dict.items():
        if key == 'stfts' and isinstance(value, dict): # Handle nested STFT dict
            cpu_stfts = {}
            for source_type, stft_results in value.items(): # mixture, segment
                cpu_stfts[source_type] = {}
                for win_len, stft_tuple in stft_results.items():
                    if isinstance(stft_tuple, tuple):
                        # Ensure tensors are detached and moved to CPU
                        cpu_stfts[source_type][win_len] = tuple(t.detach().cpu() for t in stft_tuple if isinstance(t, torch.Tensor))
                    else: # Should not happen, but handle just in case
                         cpu_stfts[source_type][win_len] = stft_tuple
            cpu_data_dict[key] = cpu_stfts
        elif isinstance(value, torch.Tensor):
             # Handle potential standalone tensors if added later
            cpu_data_dict[key] = value.detach().cpu()
        else:
            # Keep non-tensor data as is (e.g., text, lists, dicts of primitives)
            cpu_data_dict[key] = value

    torch.save(cpu_data_dict, filename)


def main(args):
    """
    Main function to load data, compute multiple STFTs per sample based on
    config, and save the results.
    """
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

    # Extract STFT common parameters from config
    stft_hop_length = configs['data']['stft_hop_length']
    stft_window = configs['data']['stft_window']
    stft_center = configs['data']['stft_center']
    stft_pad_mode = configs['data']['stft_pad_mode']
    stft_win_lengths = configs['data']['stft_win_lengths'] # List of window lengths

    # Store common params for saving later
    common_stft_params_for_saving = {
        'hop_length': stft_hop_length,
        'window': stft_window,
        'center': stft_center,
        'pad_mode': stft_pad_mode,
    }

    print("Initializing dataset...")
    # Use data_files from command line args (which might have been populated from config)
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

    # --- Device Selection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    segment_mixer.to(device) # Move mixer to the selected device
    # -----------------------

    print("Creating output directory...")
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting precomputation...")
    processed_count = 0
    global_index = 0 # Keep track of the absolute index across batches

    # STFT parameters are now handled inside the loop

    for batch in tqdm(dataloader):
        waveforms = batch['waveform'] # Shape: (batch, 1, time)
        texts = batch['text']       # List of strings
        
        # Ensure waveforms are on the correct device
        waveforms = waveforms.to(device)
        # segment_mixer is already moved to the device

        with torch.no_grad():
            # Mix waveforms (computation happens on the selected device)
            mixtures, segments = segment_mixer(waveforms) # Output shapes: (batch, 1, time)

            # --- Compute Multiple STFTs --- 
            # Temporary dicts to hold batch results before rearranging for saving
            batch_mixture_stfts = {win_len: None for win_len in stft_win_lengths}
            batch_segment_stfts = {win_len: None for win_len in stft_win_lengths}

            for win_length in stft_win_lengths:
                n_fft = win_length # Set n_fft = win_length as decided
                
                # Define params for this specific STFT config
                current_stft_params = {
                    'n_fft': n_fft,
                    'hop_length': stft_hop_length,
                    'win_length': win_length,
                    'window': stft_window,
                    'center': stft_center,
                    'pad_mode': stft_pad_mode,
                }

                # Calculate STFT for mixtures 
                mixture_mag, mixture_cos, mixture_sin = calculate_stft_components(
                    mixtures, **current_stft_params
                )
                # Store the tuple of tensors for the whole batch
                batch_mixture_stfts[win_length] = (mixture_mag, mixture_cos, mixture_sin)
                 
                # Calculate STFT for segments
                segment_mag, segment_cos, segment_sin = calculate_stft_components(
                    segments, **current_stft_params
                )
                # Store the tuple of tensors for the whole batch
                batch_segment_stfts[win_length] = (segment_mag, segment_cos, segment_sin)
            # --------------------------------

        # --- Restructure and Save --- 
        # Rearrange data from batch-major to item-major for saving
        current_batch_size = waveforms.size(0) # Use actual batch size from this iteration
        for i in range(current_batch_size):
            # Check if we have processed more items than exist in the dataset
            # This can happen if the last batch is smaller than the specified batch_size
            if global_index < len(dataset):
                 item_mixture_stfts = {}
                 item_segment_stfts = {}
                 for win_len in stft_win_lengths:
                    # Extract the i-th item's STFT tuple for this win_len from the batch result
                    item_mixture_stfts[win_len] = tuple(t[i] for t in batch_mixture_stfts[win_len]) 
                    item_segment_stfts[win_len] = tuple(t[i] for t in batch_segment_stfts[win_len])
                 
                 data_to_save = {
                     'stfts': {
                         'mixture': item_mixture_stfts,
                         'segment': item_segment_stfts
                     },
                     'text': texts[i],
                     'stft_common_params': common_stft_params_for_saving,
                     'stft_win_lengths': stft_win_lengths
                 }
                 # Save function handles moving tensors to CPU
                 save_precomputed_data(output_dir, global_index, data_to_save)
                 processed_count += 1
            global_index += 1 # Increment index regardless of whether it was saved


    print(f"Finished. Precomputed {processed_count} items.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute multiple STFTs for audio data based on YAML config.")

    # --- Config File Argument ---
    parser.add_argument('--config_yaml', type=str, required=True,
                        help='Path to the base configuration YAML file.')

    # --- Dataset/Output Arguments ---
    parser.add_argument('--data_files', type=str, default=None, nargs='+', # Default to None to read from YAML
                        help='Path(s) to the input dataset JSON file(s) (overrides config if specified).')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save precomputed STFT files.')

    args = parser.parse_args()

    # Allow overriding data_files from command line if provided
    # If not provided via command line, attempt to read from config
    if args.data_files is None: # Check if it's None (the default)
        print("Data files not provided via command line, attempting to read from YAML...")
        with open(args.config_yaml, 'r') as f:
            # Use a different variable name to avoid potential conflicts if main also reads config
            configs_for_datafiles = yaml.safe_load(f) 
        if 'data' in configs_for_datafiles and 'datafiles' in configs_for_datafiles['data']:
            args.data_files = configs_for_datafiles['data']['datafiles']
            print(f"Using data files from config: {args.data_files}")
        else:
            # Raise error if still no data files found
            raise ValueError("data_files must be provided either via command line or in the config YAML under data.datafiles.")
    else:
        print(f"Using data files provided via command line: {args.data_files}")
            
    main(args) 