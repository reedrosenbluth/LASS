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


def process_files(data_files, target_output_dir, configs, common_stft_params_for_saving):
    """
    Processes a list of data files: loads data, computes STFTs, and saves results.
    """
    print(f"Processing files for output directory: {target_output_dir}")
    target_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Extract parameters from config (needed for dataset/dataloader/mixer) ---
    sampling_rate = configs['data']['sampling_rate']
    max_clip_len = configs['data']['segment_seconds']
    max_mix_num = configs['data']['max_mix_num']
    lower_db = configs['data']['loudness_norm']['lower_db']
    higher_db = configs['data']['loudness_norm']['higher_db']
    batch_size = configs['train']['batch_size_per_device']
    num_workers = configs['train']['num_workers']

    # STFT params (needed for computation loop)
    stft_hop_length = configs['data']['stft_hop_length']
    stft_window = configs['data']['stft_window']
    stft_center = configs['data']['stft_center']
    stft_pad_mode = configs['data']['stft_pad_mode']
    stft_win_lengths = configs['data']['stft_win_lengths']
    # --- End Parameter Extraction ---

    print("Initializing dataset...")
    dataset = AudioTextDataset(
        datafiles=data_files, # Use the specific list passed to this function
        sampling_rate=sampling_rate,
        max_clip_len=max_clip_len,
    )
    print(f"Dataset size for this set: {len(dataset)}")

    if not dataset:
        print(f"Warning: No data found for files: {data_files}. Skipping processing for {target_output_dir}")
        return 0 # Return 0 processed items

    # Need batch_size >= max_mix_num for SegmentMixer
    if batch_size < max_mix_num:
        print(f"Warning: batch_size ({batch_size}) < max_mix_num ({max_mix_num}). Adjusting batch_size.")
        batch_size = max_mix_num

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, # Keep order for saving with index within this set
        num_workers=num_workers,
        collate_fn=None # Use default collate
    )

    print("Initializing mixer...")
    segment_mixer = SegmentMixer(
        max_mix_num=max_mix_num,
        lower_db=lower_db,
        higher_db=higher_db,
    )
    segment_mixer.eval()

    # --- Device Selection ---
    # Consider making device selection happen only once in main?
    # For now, keep it here for encapsulation.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    segment_mixer.to(device)
    # -----------------------

    print("Starting precomputation loop...")
    processed_count_for_set = 0
    item_index_for_set = 0 # Keep track of the index *within this set*

    for batch in tqdm(dataloader):
        waveforms = batch['waveform'] # Shape: (batch, 1, time)
        texts = batch['text']       # List of strings

        waveforms = waveforms.to(device)

        with torch.no_grad():
            mixtures, segments = segment_mixer(waveforms)

            batch_mixture_stfts = {win_len: None for win_len in stft_win_lengths}
            batch_segment_stfts = {win_len: None for win_len in stft_win_lengths}

            for win_length in stft_win_lengths:
                n_fft = win_length
                current_stft_params = {
                    'n_fft': n_fft,
                    'hop_length': stft_hop_length,
                    'win_length': win_length,
                    'window': stft_window,
                    'center': stft_center,
                    'pad_mode': stft_pad_mode,
                }

                mixture_mag, mixture_cos, mixture_sin = calculate_stft_components(
                    mixtures, **current_stft_params
                )
                batch_mixture_stfts[win_length] = (mixture_mag, mixture_cos, mixture_sin)

                segment_mag, segment_cos, segment_sin = calculate_stft_components(
                    segments, **current_stft_params
                )
                batch_segment_stfts[win_length] = (segment_mag, segment_cos, segment_sin)

        current_batch_size = waveforms.size(0)
        for i in range(current_batch_size):
            # No need to check global index against dataset len anymore,
            # DataLoader handles the last batch size correctly.
            item_mixture_stfts = {}
            item_segment_stfts = {}
            for win_len in stft_win_lengths:
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
            save_precomputed_data(target_output_dir, item_index_for_set, data_to_save)
            processed_count_for_set += 1
            item_index_for_set += 1

    print(f"Finished processing for {target_output_dir}. Precomputed {processed_count_for_set} items.")
    return processed_count_for_set


def main(args):
    """
    Main function to load data, compute multiple STFTs per sample based on
    config, and save the results for train and validation sets separately.
    """
    print("Loading config file...")
    with open(args.config_yaml, 'r') as f:
        configs = yaml.safe_load(f)

    # --- Extract only STFT common parameters needed by both processes ---
    # Other params (batch size, num workers, etc.) are read inside process_files
    stft_hop_length = configs['data']['stft_hop_length']
    stft_window = configs['data']['stft_window']
    stft_center = configs['data']['stft_center']
    stft_pad_mode = configs['data']['stft_pad_mode']
    # stft_win_lengths = configs['data']['stft_win_lengths'] # Read inside helper

    common_stft_params_for_saving = {
        'hop_length': stft_hop_length,
        'window': stft_window,
        'center': stft_center,
        'pad_mode': stft_pad_mode,
    }
    # -------------------------------------------------------------------

    # --- Setup Output Dirs ---
    base_output_dir = pathlib.Path(args.output_dir)
    train_output_dir = base_output_dir / "train"
    val_output_dir = base_output_dir / "val"
    # Directories are created inside process_files
    print(f"Base output directory: {base_output_dir}")
    print(f"Training output directory: {train_output_dir}")
    print(f"Validation output directory: {val_output_dir}")
    # ------------------------

    # --- Process Training Files ---
    print("\n--- Processing Training Set ---")
    total_train_processed = process_files(
        data_files=args.train_data_files,
        target_output_dir=train_output_dir,
        configs=configs,
        common_stft_params_for_saving=common_stft_params_for_saving
    )
    # ----------------------------

    # --- Process Validation Files ---
    print("\n--- Processing Validation Set ---")
    total_val_processed = process_files(
        data_files=args.val_data_files,
        target_output_dir=val_output_dir,
        configs=configs,
        common_stft_params_for_saving=common_stft_params_for_saving
    )
    # ------------------------------

    print(f"\nFinished all processing.")
    print(f"Total training items processed: {total_train_processed}")
    print(f"Total validation items processed: {total_val_processed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute multiple STFTs for audio data based on YAML config.")

    # --- Config File Argument ---
    parser.add_argument('--config_yaml', type=str, required=True,
                        help='Path to the base configuration YAML file.')

    # --- Dataset/Output Arguments ---
    parser.add_argument('--train_data_files', type=str, default=None, nargs='+',
                        help='Path(s) to the training dataset JSON file(s) (overrides config if specified).')
    parser.add_argument('--val_data_files', type=str, default=None, nargs='+',
                        help='Path(s) to the validation dataset JSON file(s) (overrides config if specified).')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save precomputed STFT files (will create train/ and val/ subdirs).')

    args = parser.parse_args()

    # --- Load data file paths from config if not provided via command line ---
    train_files_source = "command line"
    val_files_source = "command line"

    if args.train_data_files is None or args.val_data_files is None:
        print("Train or validation data files not provided via command line, attempting to read from YAML...")
        try:
            with open(args.config_yaml, 'r') as f:
                configs_for_datafiles = yaml.safe_load(f)
            
            if args.train_data_files is None:
                if 'data' in configs_for_datafiles and 'train_datafiles' in configs_for_datafiles['data']:
                    args.train_data_files = configs_for_datafiles['data']['train_datafiles']
                    train_files_source = "config"
                    print(f"Using training data files from config: {args.train_data_files}")
                else:
                    raise ValueError("train_data_files must be provided either via command line or in the config YAML under data.train_datafiles.")
            
            if args.val_data_files is None:
                 if 'data' in configs_for_datafiles and 'val_datafiles' in configs_for_datafiles['data']:
                    args.val_data_files = configs_for_datafiles['data']['val_datafiles']
                    val_files_source = "config"
                    print(f"Using validation data files from config: {args.val_data_files}")
                 else:
                    # Allow validation set to be optional? For now, require it.
                    raise ValueError("val_data_files must be provided either via command line or in the config YAML under data.val_datafiles.")
        except FileNotFoundError:
             print(f"Error: Config file not found at {args.config_yaml}")
             # Re-raise the specific errors if files still weren't found after trying YAML
             if args.train_data_files is None:
                 raise ValueError("train_data_files must be provided via command line (config file not found).")
             if args.val_data_files is None:
                 raise ValueError("val_data_files must be provided via command line (config file not found).")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {args.config_yaml}: {e}")
            # Re-raise errors
            if args.train_data_files is None:
                 raise ValueError("train_data_files must be provided via command line (YAML parse error).")
            if args.val_data_files is None:
                 raise ValueError("val_data_files must be provided via command line (YAML parse error).")

    # Report final file sources
    if train_files_source == "command line":
        print(f"Using training data files provided via command line: {args.train_data_files}")
    if val_files_source == "command line":
        print(f"Using validation data files provided via command line: {args.val_data_files}")

    # Ensure lists are not empty
    if not args.train_data_files:
         raise ValueError("Training data file list cannot be empty.")
    if not args.val_data_files:
         raise ValueError("Validation data file list cannot be empty.")

    main(args) 