import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
import argparse
import os
import json
from tqdm import tqdm
import pathlib
from torch.utils.data import DataLoader

# Assuming these imports are correct based on the project structure
from data.audiotext_dataset import AudioTextDataset
from data.waveform_mixers import SegmentMixer

def calculate_stft_components(waveform, n_fft, hop_length, win_length, window, center, pad_mode):
    """Calculates STFT magnitude, cosine, and sine components."""
    # waveform shape: (batch, channels, time) -> needs (channels, time) or (time,) for stft
    # Let's assume mono audio for now, matching AudioTextDataset output: (batch, 1, time) -> (batch, time)
    waveform = waveform.squeeze(1) 
    
    window_fn = getattr(torch, f"{window}_window")

    # Returns: (..., freq, time, 2) for real/imaginary or (..., freq, time) for magnitude
    stft_complex = F.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window_fn(win_length, device=waveform.device),
        center=center,
        pad_mode=pad_mode,
        return_complex=True,
    ) # Shape: (batch, freq, time_steps)

    magnitude = torch.abs(stft_complex)
    phase = torch.angle(stft_complex)
    cos_phase = torch.cos(phase)
    sin_phase = torch.sin(phase)
    
    # Reshape to match model expectation (batch, channels, time_steps, freq_bins)
    # Assuming mono, channels=1
    magnitude = magnitude.unsqueeze(1).permute(0, 1, 3, 2) # (batch, 1, time_steps, freq)
    cos_phase = cos_phase.unsqueeze(1).permute(0, 1, 3, 2) # (batch, 1, time_steps, freq)
    sin_phase = sin_phase.unsqueeze(1).permute(0, 1, 3, 2) # (batch, 1, time_steps, freq)

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
    print("Initializing dataset...")
    dataset = AudioTextDataset(
        datafiles=args.data_files,
        sampling_rate=args.sampling_rate,
        max_clip_len=args.max_clip_len,
    )
    print(f"Dataset size: {len(dataset)}")

    # Need batch_size >= max_mix_num for SegmentMixer
    batch_size = args.batch_size
    if batch_size < args.max_mix_num:
        print(f"Warning: batch_size ({batch_size}) < max_mix_num ({args.max_mix_num}). Adjusting batch_size.")
        batch_size = args.max_mix_num

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, # Keep order for saving with index
        num_workers=args.num_workers,
        collate_fn=None # Use default collate
    )

    print("Initializing mixer...")
    segment_mixer = SegmentMixer(
        max_mix_num=args.max_mix_num,
        lower_db=args.lower_db,
        higher_db=args.higher_db,
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

    # --- Dataset Arguments ---
    parser.add_argument('--data_files', type=str, required=True, nargs='+',
                        help='Path(s) to the input dataset JSON file(s).')
    parser.add_argument('--sampling_rate', type=int, default=32000,
                        help='Target audio sampling rate.')
    parser.add_argument('--max_clip_len', type=int, default=5,
                        help='Maximum audio clip length in seconds.')

    # --- Output Arguments ---
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save precomputed STFT files.')

    # --- Mixer Arguments ---
    parser.add_argument('--max_mix_num', type=int, default=4, # Example default
                        help='Maximum number of segments to mix (including original).')
    parser.add_argument('--lower_db', type=int, default=-5, # Example default
                        help='Lower bound for loudness normalization (dB).')
    parser.add_argument('--higher_db', type=int, default=5, # Example default
                        help='Higher bound for loudness normalization (dB).')

    # --- STFT Arguments ---
    parser.add_argument('--n_fft', type=int, default=1024, help='STFT FFT size.')
    parser.add_argument('--hop_length', type=int, default=160, help='STFT hop length.')
    parser.add_argument('--win_length', type=int, default=1024, help='STFT window length.')
    parser.add_argument('--window', type=str, default='hann', help='STFT window type.')
    parser.add_argument('--center', type=bool, default=True, help='STFT center.')
    parser.add_argument('--pad_mode', type=str, default='reflect', help='STFT padding mode.')

    # --- Processing Arguments ---
    parser.add_argument('--batch_size', type=int, default=16, # Should be >= max_mix_num
                        help='Batch size for processing.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for DataLoader.')

    args = parser.parse_args()
    main(args) 