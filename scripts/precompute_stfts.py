import torch
import numpy as np
import argparse
import yaml
from tqdm import tqdm
import pathlib
import json
import random
import threading
import queue
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchlibrosa.stft import STFT, magphase
import lmdb
import pickle

from data.audiotext_dataset import AudioTextDataset
from data.waveform_mixers import SegmentMixer
from data.waveform_mixers import dynamic_loudnorm as mixer_dynamic_loudnorm

def calculate_stft_components(waveform, n_fft, hop_length, win_length, window, center, pad_mode):
    """Calculates STFT magnitude, cosine, and sine components using torchlibrosa.STFT."""
    # waveform shape: (batch, channels, time) -> needs (batch, time) for torchlibrosa.STFT
    # Waveform is already on the target device
    if waveform.dim() == 3:
        waveform = waveform.squeeze(1)
    # Ensure waveform is float32 as expected by torchlibrosa
    if waveform.dtype != torch.float32:
        waveform = waveform.float()

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
    # Output shapes: (batch_size, freq_bins, time_steps, 2) for complex
    # Torchlibrosa STFT returns real and imag separately
    # Output shapes: (batch_size, freq_bins, time_steps) <- Note: torchlibrosa convention freq first
    real, imag = stft_extractor(waveform)
    
    # Calculate magnitude and phase components using the updated magphase function
    # Input shapes: (batch_size, freq_bins, time_steps)
    # Output shapes seem to be (B, 1, T, F) after contiguous()
    magnitude, cos_phase, sin_phase = magphase(real, imag)

    # Ensure output is contiguous - return directly as (B, 1, T, F)
    magnitude = magnitude.contiguous()
    cos_phase = cos_phase.contiguous()
    sin_phase = sin_phase.contiguous()

    return magnitude, cos_phase, sin_phase

def load_single_waveform(file_path, target_sr, max_clip_len_seconds, device):
    """
    Loads, resamples, converts to mono, and trims/pads a single audio file.

    Args:
        file_path (str or pathlib.Path): Path to the audio file.
        target_sr (int): Target sampling rate.
        max_clip_len_seconds (float): Target length in seconds.
        device (torch.device): Device to load the tensor onto.

    Returns:
        torch.Tensor or None: Waveform tensor (1, 1, T) on the specified device,
                              or None if loading/processing fails.
    """
    try:
        max_samples = int(target_sr * max_clip_len_seconds)
        # Load audio file
        waveform, sr = torchaudio.load(str(file_path), normalize=True)

        # Resample if necessary
        if sr != target_sr:
            # Use torchaudio's resample transform
            resampler = T.Resample(sr, target_sr, dtype=waveform.dtype)
            waveform = resampler(waveform)

        # Convert to mono by averaging channels if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Ensure it's at least 2D (channel, time)
        if waveform.dim() == 1:
             waveform = waveform.unsqueeze(0) # Add channel dim

        # Trim or pad to max_samples
        current_len = waveform.shape[1]
        if current_len > max_samples:
            # Trim from the beginning (consistent with AudioTextDataset?)
            # Make sure this matches how your dataset trims, adjust if needed
            offset = random.randint(0, current_len - max_samples)
            waveform = waveform[:, offset:offset+max_samples]
            # waveform = waveform[:, :max_samples] # Simpler trim from start
        elif current_len < max_samples:
            # Pad with zeros
            padding = max_samples - current_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Add batch dimension and move to device
        waveform = waveform.unsqueeze(0).to(device) # Shape: (1, 1, T)

        return waveform

    except FileNotFoundError:
        print(f"Error: Audio file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading/processing single file {file_path}: {e}")
        return None

def generate_mixture_recipes_for_batch(texts, original_audiopaths, max_mix_num, batch_size):
    """
    Generates mixture recipes for a batch based on SegmentMixer logic,
    without performing audio processing.

    Args:
        texts (List[str]): List of texts corresponding to waveforms in the batch.
        original_audiopaths (List[str]): List of original audio paths corresponding to waveforms in the batch.
        max_mix_num (int): Maximum number of segments to mix.
        batch_size (int): The size of the current batch.

    Returns:
        List[Dict]: A list of recipe dictionaries for the batch.
                    Each dict contains:
                    - 'primary_segment_index_in_batch': Index of the primary segment within this batch.
                    - 'primary_segment_text': Text of the main segment.
                    - 'mixture_component_texts': List of texts of all segments in the mixture (including primary).
                    - 'component_indices_in_batch': Indices within the batch of segments used.
                    - 'mix_num': The number of segments mixed for this item (randomly chosen).
                    - 'original_audiopath': Original audio path of the primary segment.
    """
    batch_recipes = []
    for n in range(batch_size):
        primary_text = texts[n]
        primary_original_path = original_audiopaths[n]
        # Mimic SegmentMixer logic: mix_num segments total, including primary
        # Ensure mix_num is at least 2 if max_mix_num allows
        actual_max_mix = min(max_mix_num, batch_size)
        min_mix = 2 if actual_max_mix >= 2 else actual_max_mix
        if min_mix > actual_max_mix:
             mix_num = 1
        else:
             mix_num = random.randint(min_mix, actual_max_mix)


        component_indices = [n] # Start with the primary segment index
        component_texts = [primary_text]
        component_paths = [primary_original_path] # Start with the primary segment path

        # Loop to select (mix_num - 1) other segments using wrap-around logic
        # Ensure indices used for selection are distinct and wrap around correctly
        possible_indices = list(range(batch_size))
        possible_indices.remove(n) # Don't mix item with itself initially

        indices_to_add = []
        if mix_num > 1:
             num_to_add = mix_num - 1
             current_idx_pos = n
             added_count = 0
             attempts = 0 # Prevent infinite loops in edge cases
             while added_count < num_to_add and attempts < batch_size * 2:
                 current_idx_pos = (current_idx_pos + 1) % batch_size
                 component_original_path = original_audiopaths[current_idx_pos]
                 # Check: not same index, not already added, AND not same original path
                 if (current_idx_pos != n and
                     current_idx_pos not in indices_to_add and
                     component_original_path != primary_original_path):
                     indices_to_add.append(current_idx_pos)
                     added_count += 1
                 attempts += 1 # Indent this correctly
             if added_count < num_to_add:
                 # Fallback if wrap-around didn't find enough unique items
                 print(f"Warning: Could only find {added_count} unique items (with different original paths) to mix for item {n} (path: {primary_original_path}, requested {num_to_add}). Using available.")
                 # Add remaining random indices if needed
                 remaining_needed = num_to_add - added_count
                 # Filter fallback candidates based on original path as well
                 available_others = [idx for idx in possible_indices
                                     if idx not in indices_to_add and
                                     original_audiopaths[idx] != primary_original_path]
                 random.shuffle(available_others)
                 indices_to_add.extend(available_others[:remaining_needed])
                 # Update added_count based on how many were actually added in fallback
                 added_count += len(available_others[:remaining_needed])
                 if added_count < num_to_add:
                     print(f"Warning: Even after fallback, only {added_count} components found for item {n}.")


        for comp_idx in indices_to_add:
             component_indices.append(comp_idx)
             component_texts.append(texts[comp_idx])
             component_paths.append(original_audiopaths[comp_idx]) # Add the corresponding path


        recipe = {
            'primary_segment_index_in_batch': n,
            'primary_segment_text': primary_text,
            'mixture_component_texts': component_texts, # Includes primary text
            'component_indices_in_batch': component_indices, # Keep for potential debugging?
            'component_original_paths': component_paths, # List of paths for all components
            'mix_num': len(component_indices), # Actual number mixed
            'original_audiopath': primary_original_path # Path of the primary segment
        }
        batch_recipes.append(recipe)
    return batch_recipes

def safe_collate(batch):
    """Collate function that filters out None values."""
    # Filter out None values first
    batch = [item for item in batch if item is not None]
    # If the batch is empty after filtering, return None or an empty structure
    # that your processing loop can handle. Returning None might be simplest.
    if not batch:
        return None
    # If there are valid items, use the default collate function
    return default_collate(batch)

def process_files_for_recipes(data_files, target_recipe_file, configs):
    """
    Processes data files to generate and save mixture recipes in JSON format.
    """
    print(f"Generating recipes for output file: {target_recipe_file}")
    target_recipe_file.parent.mkdir(parents=True, exist_ok=True) # Ensure parent dir exists

    sampling_rate = configs['data']['sampling_rate']
    max_clip_len = configs['data']['segment_seconds']
    max_mix_num = configs['data']['max_mix_num']
    batch_size = configs['train']['batch_size_per_device']
    num_workers = configs['train']['num_workers']

    print("Initializing dataset...")
    dataset = AudioTextDataset(
        datafiles=data_files,
        sampling_rate=sampling_rate,
        max_clip_len=max_clip_len,
        suppress_warnings=True
    )
    print(f"Dataset size for this set: {len(dataset)}")

    if not dataset:
        print(f"Warning: No data found for files: {data_files}. Skipping recipe generation for {target_recipe_file.name}")
        with open(target_recipe_file, 'w') as f:
            json.dump([], f, indent=2)
        return 0

    effective_batch_size = batch_size
    if batch_size < max_mix_num and len(dataset) >= max_mix_num:
        print(f"Warning: configured batch_size ({batch_size}) < max_mix_num ({max_mix_num}). "
              f"Recipe generation might behave unexpectedly if wrap-around needs more items than batch provides. "
              f"Consider increasing batch_size >= {max_mix_num} or ensure dataset splits work with current batch size.")

    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False, # Must be False for recipes to match Phase 2 order
        num_workers=num_workers, # Use variable read from config
        collate_fn=safe_collate # Use the safe collate function
    )

    print("Starting recipe generation loop...")
    all_recipes = []
    global_item_index = 0 # Use this to track the *intended* global index

    # Keep track of items successfully processed by the dataloader
    processed_indices_count = 0

    for batch in tqdm(dataloader):
        # Handle the case where safe_collate returns None for an empty batch
        if batch is None:
            # We don't know exactly how many items were attempted in the original batch size
            # that resulted in this None batch. We can't accurately update global_item_index here
            # without knowing the original batch size requested from the dataset.
            # This approach might lead to slightly off global indices if full batches fail.
            # A more robust solution might involve knowing the intended indices.
            # For now, we continue, acknowledging this potential inaccuracy.
            print("Warning: Skipping an entirely failed batch.")
            continue # Skip processing if the batch is empty

        texts = batch['text']
        # Retrieve the original audio paths from the batch
        original_audiopaths = batch['original_audiopath']
        current_batch_size = len(texts) # This is the size *after* filtering Nones

        if current_batch_size == 0: continue # Should be caught by batch is None, but safe check

        # Generate recipes for this batch
        batch_recipes = generate_mixture_recipes_for_batch(
            texts=texts,
            original_audiopaths=original_audiopaths,
            max_mix_num=max_mix_num,
            batch_size=current_batch_size # Use the actual size of the valid batch
        )

        # Add global output index to each recipe and append to the main list
        for i, recipe in enumerate(batch_recipes):
            # The 'output_index' now represents the index among successfully loaded items
            recipe['output_index'] = processed_indices_count + i
            # Add the original audio path for this item to the recipe
            recipe['original_audiopath'] = original_audiopaths[i] # Path of the successfully loaded item
            all_recipes.append(recipe)

        processed_indices_count += current_batch_size # Increment by the actual number processed

    # Remove the potentially inaccurate global_item_index logic
    # global_item_index += current_batch_size


    print(f"Saving {len(all_recipes)} recipes to {target_recipe_file}...")
    with open(target_recipe_file, 'w') as f:
        json.dump(all_recipes, f, indent=2)

    # Report dropped count based on the dataset's internal counter
    dropped_count = dataset.get_dropped_count()
    if dropped_count > 0:
        print(f"Note: {dropped_count} audio files failed to load correctly and were skipped during recipe generation.")

    print(f"Finished generating recipes for {target_recipe_file.name}. Total items saved: {len(all_recipes)}.")
    return len(all_recipes)

def _load_recipes(recipe_file):
    """Loads recipes from a JSON file into a dictionary keyed by original audiopath."""
    print(f"Loading recipes from {recipe_file}...")
    try:
        with open(recipe_file, 'r') as f:
            recipes = json.load(f)
        if not recipes:
             print("Recipe file is empty. Nothing to process.")
             return None # Indicate empty recipes
        # Create a dictionary mapping original_audiopath to recipe
        recipes_dict = {recipe['original_audiopath']: recipe for recipe in recipes}
        print(f"Loaded {len(recipes_dict)} recipes, indexed by original audio path.")
        return recipes_dict
    except FileNotFoundError:
        print(f"Error: Recipe file not found at {recipe_file}")
        raise # Re-raise the exception to be handled by the caller
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from recipe file: {recipe_file}")
        raise # Re-raise the exception

def _initialize_stft_dataset_loader(data_files, configs):
    """Initializes the AudioTextDataset and DataLoader for STFT processing."""
    sampling_rate = configs['data']['sampling_rate']
    max_clip_len = configs['data']['segment_seconds']
    batch_size = configs['train']['batch_size_per_device']
    num_workers = configs['train']['num_workers']

    print("Initializing dataset...")
    dataset = AudioTextDataset(
        datafiles=data_files,
        sampling_rate=sampling_rate,
        max_clip_len=max_clip_len,
        suppress_warnings=True
    )
    print(f"Dataset size for this set: {len(dataset)}")

    if not dataset:
        print(f"Warning: No data found for files: {data_files}. Returning None for dataset/loader.")
        return None, None

    effective_batch_size = batch_size

    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=safe_collate,
        pin_memory=torch.cuda.is_available() # Pin memory only if CUDA is available
    )
    return dataset, dataloader

def _create_batch_mixtures(waveforms, original_audiopaths, recipes_dict, sampling_rate, max_clip_len, loudness_params, device):
    """
    Creates mixtures for a batch based on recipes, handling component loading and normalization.

    Args:
        waveforms (torch.Tensor): Input waveforms for the batch (B_valid, 1, T).
        original_audiopaths (List[str]): List of original audio paths for the batch.
        recipes_dict (Dict): Dictionary mapping original_audiopath to recipe.
        sampling_rate (int): Target sampling rate.
        max_clip_len (float): Target clip length in seconds.
        loudness_params (Dict): Parameters for loudness normalization.
        device (torch.device): Device for processing.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
            - final_segments (Tensor): Processed primary segments (B_recipe_found, 1, T).
            - final_mixtures (Tensor): Created mixtures (B_recipe_found, 1, T).
            - batch_recipes_used (List): List of recipes corresponding to the returned tensors.
            Returns (None, None, []) if no valid items with recipes are found.
    """
    current_batch_size = waveforms.size(0)

    # 1. Get recipes for the current batch items
    batch_recipes_used = []
    valid_indices_in_batch = []
    skipped_due_to_missing_recipe = 0
    for i in range(current_batch_size):
        current_original_path = original_audiopaths[i]
        recipe = recipes_dict.get(current_original_path)
        if recipe is None:
            print(f"Warning: No recipe found for successfully loaded audio path '{current_original_path}'. Skipping item.")
            skipped_due_to_missing_recipe += 1
            continue
        batch_recipes_used.append(recipe)
        valid_indices_in_batch.append(i)

    if not batch_recipes_used:
        print(f"Warning: No recipes found for any of the {current_batch_size} successfully loaded items in this batch.")
        return None, None, [] # Indicate no valid items processed

    valid_waveforms = waveforms[valid_indices_in_batch]
    num_valid_items = len(batch_recipes_used)

    # 2. Gather primary segments
    primary_segments = valid_waveforms # Shape: (B_recipe_found, 1, T)

    # 3. Gather and combine noise components
    noise_accumulator = torch.zeros_like(primary_segments)
    num_noise_components_added = torch.zeros(num_valid_items, device=device)
    current_valid_batch_paths = [original_audiopaths[i] for i in valid_indices_in_batch]
    current_valid_batch_path_to_idx_map = {path: k for k, path in enumerate(current_valid_batch_paths)}
    # current_device = primary_segments.device # Already have device passed in

    for k, recipe in enumerate(batch_recipes_used): # k iterates from 0 to num_valid_items-1
        primary_seg_for_norm = primary_segments[k:k+1] # Keep dims (1, 1, T) for reference
        primary_path = recipe['original_audiopath'] # Path of the primary item for this recipe
        component_paths_in_recipe = recipe.get('component_original_paths', [])
        if not component_paths_in_recipe:
            print(f"Warning: Recipe for '{primary_path}' is missing 'component_original_paths'. Cannot create mixture.")
            continue # Skip mixing for this item

        item_noise = torch.zeros_like(primary_seg_for_norm) # Shape (1, 1, T)
        for comp_path in component_paths_in_recipe:
             if comp_path == primary_path: continue # Skip the primary segment itself
             next_segment = None
             comp_idx_in_current_valid_batch = current_valid_batch_path_to_idx_map.get(comp_path)
             if comp_idx_in_current_valid_batch is not None:
                 next_segment = primary_segments[comp_idx_in_current_valid_batch:comp_idx_in_current_valid_batch+1]
             else:
                  loaded_segment = load_single_waveform(
                      file_path=comp_path,
                      target_sr=sampling_rate,
                      max_clip_len_seconds=max_clip_len,
                      device=device # Load directly to the processing device
                  )
                  if loaded_segment is not None:
                      if loaded_segment.shape[-1] != primary_seg_for_norm.shape[-1]:
                           print(f"Warning: Loaded segment {comp_path} time dim {loaded_segment.shape[-1]} "
                                 f"doesn't match primary {primary_path} time dim {primary_seg_for_norm.shape[-1]} "
                                 f"after loading. Skipping component.")
                           continue
                      next_segment = loaded_segment
                  else:
                      print(f"Warning: Failed to load component '{comp_path}' for recipe '{primary_path}'. Skipping component.")
                      continue

             if next_segment is not None:
                 rescaled_next_segment = mixer_dynamic_loudnorm(
                     audio=next_segment,
                     reference=primary_seg_for_norm,
                     **loudness_params
                 )
                 if rescaled_next_segment.shape == item_noise.shape:
                     item_noise += rescaled_next_segment
                     num_noise_components_added[k] += 1
                 else:
                      print(f"Warning: Shape mismatch after loudness norm for component {comp_path}. Skipping add.")

        if num_noise_components_added[k] > 0:
             item_noise = mixer_dynamic_loudnorm(
                 audio=item_noise,
                 reference=primary_seg_for_norm,
                 **loudness_params
             )

        if item_noise.shape == noise_accumulator[k:k+1].shape:
             noise_accumulator[k:k+1] = item_noise
        else:
              print(f"Warning: Final item_noise shape {item_noise.shape} mismatch with noise_accumulator slice. Skipping accumulation.")

    # 4. Create Mixtures
    mixtures = primary_segments + noise_accumulator # Shape: (B_recipe_found, 1, T)

    # 5. De-clipping
    max_values, _ = torch.max(torch.abs(mixtures.squeeze(1)), dim=1)
    needs_clipping = max_values > 1.0
    gain = torch.ones_like(max_values)
    gain[needs_clipping] = 0.9 / max_values[needs_clipping]
    gain = gain.unsqueeze(-1).unsqueeze(-1)
    final_segments = primary_segments * gain
    final_mixtures = mixtures * gain

    return final_segments, final_mixtures, batch_recipes_used

def _calculate_batch_stfts(segments_tensor, mixtures_tensor, stft_win_lengths, stft_hop_length, stft_window, stft_center, stft_pad_mode):
    """
    Calculates STFT components for batch of segments and mixtures across multiple window lengths.

    Args:
        segments_tensor (torch.Tensor): Tensor of primary segments (B, 1, T).
        mixtures_tensor (torch.Tensor): Tensor of mixtures (B, 1, T).
        stft_win_lengths (List[int]): List of window lengths (and FFT sizes) to use.
        stft_hop_length (int): STFT hop length.
        stft_window (str): STFT window type.
        stft_center (bool): STFT center parameter.
        stft_pad_mode (str): STFT padding mode.

    Returns:
        Tuple[Dict, Dict]:
            - batch_segment_stfts (Dict): {win_len: (mag, cos, sin)} for segments.
            - batch_mixture_stfts (Dict): {win_len: (mag, cos, sin)} for mixtures.
    """
    batch_mixture_stfts = {win_len: None for win_len in stft_win_lengths}
    batch_segment_stfts = {win_len: None for win_len in stft_win_lengths}

    with torch.no_grad():
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
                mixtures_tensor, **current_stft_params
            )
            batch_mixture_stfts[win_length] = (mixture_mag, mixture_cos, mixture_sin)

            segment_mag, segment_cos, segment_sin = calculate_stft_components(
                segments_tensor, **current_stft_params
            )
            batch_segment_stfts[win_length] = (segment_mag, segment_cos, segment_sin)

    return batch_segment_stfts, batch_mixture_stfts

def _write_batch_to_lmdb(lmdb_env, batch_segment_stfts, batch_mixture_stfts, segments_tensor, batch_recipes_used, stft_win_lengths, common_stft_params_for_saving, lmdb_item_idx, batch_idx):
    """
    Writes a batch of processed STFT data to the LMDB database.

    Args:
        lmdb_env (lmdb.Environment): The LMDB environment.
        batch_segment_stfts (Dict): STFT data for segments.
        batch_mixture_stfts (Dict): STFT data for mixtures.
        segments_tensor (torch.Tensor): Target segment waveforms.
        batch_recipes_used (List[Dict]): Recipes for the items in this batch.
        stft_win_lengths (List[int]): List of STFT window lengths used.
        common_stft_params_for_saving (Dict): Common STFT params to save.
        lmdb_item_idx (int): Starting index for items in this batch.
        batch_idx (int): Index of the current batch (for logging).

    Returns:
        int: Number of items successfully written in this batch.
    """
    items_written_in_batch = 0
    num_items_in_batch = segments_tensor.size(0)

    if num_items_in_batch == 0:
        return 0

    try:
        with lmdb_env.begin(write=True) as txn:
            for k in range(num_items_in_batch):
                recipe_for_item = batch_recipes_used[k]
                target_waveform_for_item = segments_tensor[k]

                # Prepare STFT data for this item
                item_mixture_stfts = {}
                item_segment_stfts = {}
                for win_len in stft_win_lengths:
                    # Slice the batch tensors to get data for item k
                    item_mixture_stfts[win_len] = tuple(t[k:k+1] for t in batch_mixture_stfts[win_len])
                    item_segment_stfts[win_len] = tuple(t[k:k+1] for t in batch_segment_stfts[win_len])

                # Create the dictionary to save
                data_to_save = {
                    'stfts': {
                        'mixture': item_mixture_stfts,
                        'segment': item_segment_stfts
                    },
                    'target_waveform': target_waveform_for_item,
                    'text': recipe_for_item['primary_segment_text'],
                    'mixture_component_texts': recipe_for_item['mixture_component_texts'],
                    'stft_common_params': common_stft_params_for_saving,
                    'stft_win_lengths': stft_win_lengths
                }

                # Ensure all tensors are on CPU before pickling
                cpu_data_dict = {}
                for key, value in data_to_save.items():
                    if key == 'stfts' and isinstance(value, dict):
                        cpu_stfts = {}
                        for source_type, stft_results in value.items():
                            cpu_stfts[source_type] = {}
                            for win_len, stft_tuple in stft_results.items():
                                if isinstance(stft_tuple, tuple):
                                    cpu_stfts[source_type][win_len] = tuple(t.detach().cpu() for t in stft_tuple if isinstance(t, torch.Tensor))
                                else:
                                     cpu_stfts[source_type][win_len] = stft_tuple # Should not happen if input is correct
                        cpu_data_dict[key] = cpu_stfts
                    elif isinstance(value, torch.Tensor):
                        cpu_data_dict[key] = value.detach().cpu()
                    else:
                        cpu_data_dict[key] = value

                # Generate key and serialize value
                current_item_key = str(lmdb_item_idx + k).encode('utf-8')
                value_bytes = pickle.dumps(cpu_data_dict, protocol=pickle.HIGHEST_PROTOCOL)

                # Write to LMDB
                txn.put(current_item_key, value_bytes)
                items_written_in_batch += 1

    except lmdb.Error as e:
         # Log error with more specific info
         print(f"LMDB Error during batch write (batch {batch_idx}, approx item index start {lmdb_item_idx}): {e}")
         # Depending on the error (e.g., map_full), might want to raise or handle differently.
         # For now, we assume the transaction is aborted, and return count before error.
         # The caller's processed_count_for_set will only reflect successfully committed items.
         pass # Transaction automatically aborted on exception
    except Exception as e:
         print(f"Unexpected error during LMDB batch write (batch {batch_idx}): {e}")
         # Handle unexpected errors during pickling or data prep before txn.put
         raise # Re-raise unexpected errors

    return items_written_in_batch

def process_files_for_stfts(data_files, target_output_dir, recipe_file, configs, common_stft_params_for_saving, lmdb_env):
    """
    Processes data files using pre-generated recipes: loads data, performs mixing
    based on recipes, computes STFTs, and saves results individually to LMDB.
    """
    # Target output dir is now the LMDB *directory* path, not for saving individual files
    print(f"Processing STFTs using recipe file: {recipe_file}")
    print(f"Output LMDB directory: {target_output_dir}")

    # --- Load Recipes --- #
    recipes_dict = _load_recipes(recipe_file)
    if recipes_dict is None:
        return 0 # Return 0 items processed if recipes are empty or file not found

    # --- Initialize Dataset and DataLoader --- #
    dataset, dataloader = _initialize_stft_dataset_loader(data_files, configs)
    if dataset is None or dataloader is None:
        return 0 # Return 0 items processed if dataset failed to initialize

    # --- Setup STFT Parameters and Device --- #
    sampling_rate = configs['data']['sampling_rate'] # Needed for _create_batch_mixtures
    max_clip_len = configs['data']['segment_seconds'] # Needed for _create_batch_mixtures
    lower_db = configs['data']['loudness_norm']['lower_db']
    higher_db = configs['data']['loudness_norm']['higher_db']
    stft_hop_length = configs['data']['stft_hop_length']
    stft_window = configs['data']['stft_window']
    stft_center = configs['data']['stft_center']
    stft_pad_mode = configs['data']['stft_pad_mode']
    stft_win_lengths = configs['data']['stft_win_lengths']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    loudness_params = {'lower_db': lower_db, 'higher_db': higher_db}

    # --- Main Processing Loop --- #
    print("Starting STFT precomputation loop guided by recipes (writing to LMDB)...")
    lmdb_item_idx = 0 # Initialize global LMDB item index counter for this dataset split
    processed_count_for_set = 0 # Track total items successfully written to LMDB for this split

    pbar = tqdm(dataloader, position=0, leave=True)
    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            print(f"Warning: Skipping entirely failed batch {batch_idx}.")
            continue

        waveforms = batch['waveform'].to(device)
        original_audiopaths = batch['original_audiopath']

        if waveforms.size(0) == 0: continue

        # --- Create Mixtures --- #
        segments_tensor, mixtures_tensor, batch_recipes_used = _create_batch_mixtures(
            waveforms=waveforms,
            original_audiopaths=original_audiopaths,
            recipes_dict=recipes_dict,
            sampling_rate=sampling_rate,
            max_clip_len=max_clip_len,
            loudness_params=loudness_params,
            device=device
        )

        if segments_tensor is None or mixtures_tensor is None or not batch_recipes_used:
            continue

        # --- Compute STFTs --- #
        batch_segment_stfts, batch_mixture_stfts = _calculate_batch_stfts(
            segments_tensor=segments_tensor,
            mixtures_tensor=mixtures_tensor,
            stft_win_lengths=stft_win_lengths,
            stft_hop_length=stft_hop_length,
            stft_window=stft_window,
            stft_center=stft_center,
            stft_pad_mode=stft_pad_mode
        )

        # --- Write Batch to LMDB --- #
        items_written = _write_batch_to_lmdb(
            lmdb_env=lmdb_env,
            batch_segment_stfts=batch_segment_stfts,
            batch_mixture_stfts=batch_mixture_stfts,
            segments_tensor=segments_tensor,
            batch_recipes_used=batch_recipes_used,
            stft_win_lengths=stft_win_lengths,
            common_stft_params_for_saving=common_stft_params_for_saving,
            lmdb_item_idx=lmdb_item_idx, # Pass the current global index start for this batch
            batch_idx=batch_idx
        )

        # Update counters and progress bar
        if items_written > 0:
            processed_count_for_set += items_written
            lmdb_item_idx += items_written # Advance global index by number actually written
            pbar.set_description(f"Processing STFTs (Items Written: {processed_count_for_set})")
        elif items_written < len(batch_recipes_used):
            # This case handles partial writes within a batch due to LMDB errors
            print(f"Warning: Batch {batch_idx} write potentially incomplete due to LMDB error.")
            pbar.set_description(f"Processing STFTs (Items Written: {processed_count_for_set} - Check Errors)")
            # lmdb_item_idx is only incremented by items_written, maintaining consistency

    # --- Final Reporting --- #
    num_recipes_loaded = len(recipes_dict)
    if processed_count_for_set != num_recipes_loaded:
        print(f"Warning: Final count of items written to LMDB ({processed_count_for_set}) does not match number of recipes loaded ({num_recipes_loaded}). "
              f"Potential reasons: audio loading failures during dataset iteration, recipe generation failures, or LMDB write errors (check logs).")

    dropped_count_stft = dataset.get_dropped_count()
    if dropped_count_stft > 0:
         print(f"Note: {dropped_count_stft} audio files failed to load correctly by the dataset and were skipped before STFT processing.")

    print(f"Finished STFT processing for {target_output_dir}. Wrote {processed_count_for_set} items to LMDB.")
    return processed_count_for_set # Return the number of items actually written

def main(args, parser):
    """
    Main function to either generate mixture recipes or compute STFTs based on recipes.
    """
    print("Loading config file...")
    with open(args.config_yaml, 'r') as f:
        configs = yaml.safe_load(f)

    # Extract only STFT common parameters needed by both processes
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

    # Load data file paths
    # Check command line args first, then fall back to config
    train_files_source = "command line"
    val_files_source = "command line"

    if args.train_data_files is None:
        print("Training data files not provided via command line, attempting to read from config YAML...")
        if 'data' in configs and 'train_datafiles' in configs['data']:
            args.train_data_files = configs['data']['train_datafiles']
            train_files_source = "config"
            print(f"Using training data files from config: {args.train_data_files}")
        else:
            parser.error("train_data_files must be provided either via command line or in the config YAML under data.train_datafiles.")

    if args.val_data_files is None:
        print("Validation data files not provided via command line, attempting to read from config YAML...")
        if 'data' in configs and 'val_datafiles' in configs['data']:
            args.val_data_files = configs['data']['val_datafiles']
            val_files_source = "config"
            print(f"Using validation data files from config: {args.val_data_files}")
        else:
            parser.error("val_data_files must be provided either via command line or in the config YAML under data.val_datafiles.")

    # Report final file sources if they came from command line
    if train_files_source == "command line":
        print(f"Using training data files provided via command line: {args.train_data_files}")
    if val_files_source == "command line":
        print(f"Using validation data files provided via command line: {args.val_data_files}")

    # Ensure lists are not empty after loading
    if not args.train_data_files:
        parser.error("Training data file list cannot be empty (check command line or config).")
    if not args.val_data_files:
        parser.error("Validation data file list cannot be empty (check command line or config).")


    if args.mode == 'generate_recipes':
        if not args.output_recipe_dir:
            # Raise error using parser's error method for consistency if possible, else ValueError
            parser.error("--output_recipe_dir is required for 'generate_recipes' mode.") # Use parser if available in scope, else raise


        base_recipe_dir = pathlib.Path(args.output_recipe_dir)
        train_recipe_file = base_recipe_dir / "train_mixture_recipes.json"
        val_recipe_file = base_recipe_dir / "val_mixture_recipes.json"

        # Process Training Files
        print("\n--- Processing Training Set Recipes ---")
        total_train_processed = process_files_for_recipes(
            data_files=args.train_data_files,
            target_recipe_file=train_recipe_file,
            configs=configs
        )

        # Process Validation Files
        print("\n--- Processing Validation Set Recipes ---")
        total_val_processed = process_files_for_recipes(
            data_files=args.val_data_files,
            target_recipe_file=val_recipe_file,
            configs=configs
        )

        print(f"\nFinished all recipe processing.")
        print(f"Total training items processed: {total_train_processed}")
        print(f"Total validation items processed: {total_val_processed}")

    elif args.mode == 'compute_stfts':
        # Validate required arguments for this mode
        if not args.output_dir:
             parser.error("--output_dir is required for 'compute_stfts' mode (specify the base directory for LMDB files).")
        if not args.input_recipe_dir:
             parser.error("--input_recipe_dir is required for 'compute_stfts' mode.")

        # Setup LMDB Output Dirs and Parameters
        base_output_dir = pathlib.Path(args.output_dir)
        train_lmdb_path = base_output_dir / "train.lmdb" # Directory for LMDB
        val_lmdb_path = base_output_dir / "val.lmdb"   # Directory for LMDB
        print(f"LMDB base output directory: {base_output_dir}")
        print(f"Training LMDB path: {train_lmdb_path}")
        print(f"Validation LMDB path: {val_lmdb_path}")

        # --- Estimate map_size ---
        # IMPORTANT: Adjust these values based on your expected dataset size!
        # These are placeholders. Calculate roughly:
        # (Number of items) * (Avg size per item in bytes) * (Multiplier > 1, e.g., 1.5-2.0)
        # Example: 1 million train items, ~1MB each -> 1TB. Set map_size to 1.5-2.0 TB.
        map_size_train_gb = configs.get('data', {}).get('lmdb_map_size_train_gb', 1500) # Default 1.5 TB
        map_size_val_gb = configs.get('data', {}).get('lmdb_map_size_val_gb', 500)     # Default 0.5 TB
        map_size_train_bytes = int(map_size_train_gb * 1024**3)
        map_size_val_bytes = int(map_size_val_gb * 1024**3)
        print(f"LMDB map_size (Train): {map_size_train_gb} GB")
        print(f"LMDB map_size (Val): {map_size_val_gb} GB")
        # Consider adding these map sizes to your config YAML for easier management

        # Setup Recipe Input Paths
        base_recipe_dir = pathlib.Path(args.input_recipe_dir)
        train_recipe_file = base_recipe_dir / "train_mixture_recipes.json"
        val_recipe_file = base_recipe_dir / "val_mixture_recipes.json"
        print(f"Recipe input directory: {base_recipe_dir}")

        # Extract STFT common parameters for saving (already have them from above)
        # ... common_stft_params_for_saving is already defined ...

        # Process Training Files
        train_env = None # Initialize to None
        try:
            print(f"\n--- Opening Training LMDB Environment [{train_lmdb_path}] ---")
            # Ensure parent directory exists
            train_lmdb_path.parent.mkdir(parents=True, exist_ok=True)
            train_env = lmdb.open(
                str(train_lmdb_path),
                map_size=map_size_train_bytes,
                readonly=False,
                metasync=False, # Faster writes, riskier on crash
                sync=False,     # Faster writes, riskier on crash
                map_async=True, # Faster writes
                lock=True       # Needed for multi-process write? Usually True for single writer.
                                # Consider lock=False if only this process writes.
            )
            print("\n--- Processing Training Set STFTs ---")
            total_train_processed = process_files_for_stfts(
                data_files=args.train_data_files,
                target_output_dir=train_lmdb_path, # Pass LMDB path
                recipe_file=train_recipe_file,
                configs=configs,
                common_stft_params_for_saving=common_stft_params_for_saving,
                lmdb_env=train_env # Pass the environment
            )
        finally:
             if train_env:
                 print(f"\n--- Closing Training LMDB Environment [{train_lmdb_path}] ---")
                 train_env.close()

        # Process Validation Files
        val_env = None # Initialize to None
        try:
            print(f"\n--- Opening Validation LMDB Environment [{val_lmdb_path}] ---")
            # Ensure parent directory exists
            val_lmdb_path.parent.mkdir(parents=True, exist_ok=True)
            val_env = lmdb.open(
                str(val_lmdb_path),
                map_size=map_size_val_bytes,
                readonly=False,
                metasync=False,
                sync=False,
                map_async=True,
                lock=True
            )
            print("\n--- Processing Validation Set STFTs ---")
            total_val_processed = process_files_for_stfts(
                data_files=args.val_data_files,
                target_output_dir=val_lmdb_path, # Pass LMDB path
                recipe_file=val_recipe_file,
                configs=configs,
                common_stft_params_for_saving=common_stft_params_for_saving,
                lmdb_env=val_env # Pass the environment
            )
        finally:
             if val_env:
                 print(f"\n--- Closing Validation LMDB Environment [{val_lmdb_path}] ---")
                 val_env.close()


        print(f"\nFinished STFT computation.")
        print(f"Total training items written to LMDB: {total_train_processed}")
        print(f"Total validation items written to LMDB: {total_val_processed}")

    else:
        # Raise error if mode not recognized
        parser.error("Invalid mode selected. Choose 'generate_recipes' or 'compute_stfts'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute STFTs or generate mixture recipes based on YAML config.")

    parser.add_argument('--mode', type=str, required=True,
                        help='Mode of operation: "generate_recipes" or "compute_stfts".')

    parser.add_argument('--config_yaml', type=str, required=True,
                        help='Path to the base configuration YAML file.')

    parser.add_argument('--train_data_files', type=str, default=None, nargs='+',
                        help='Path(s) to the training dataset JSON file(s) (overrides config if specified).')
    parser.add_argument('--val_data_files', type=str, default=None, nargs='+',
                        help='Path(s) to the validation dataset JSON file(s) (overrides config if specified).')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Base directory to save LMDB database files (will create train.lmdb/ and val.lmdb/ subdirs). Required for compute_stfts mode.')
    parser.add_argument('--output_recipe_dir', type=str, default=None,
                        help="Directory to save recipe JSON files (mode='generate_recipes'). Expects train_mixture_recipes.json and val_mixture_recipes.json.")
    parser.add_argument('--input_recipe_dir', type=str, default=None,
                        help="Directory containing recipe JSON files to use (mode='compute_stfts'). Expects train_mixture_recipes.json and val_mixture_recipes.json.")

    args = parser.parse_args()

    # Add LMDB map size args (optional, can also be in config)
    # parser.add_argument('--lmdb_map_size_train_gb', type=int, default=1500, help='Estimated map size in GB for training LMDB.')
    # parser.add_argument('--lmdb_map_size_val_gb', type=int, default=500, help='Estimated map size in GB for validation LMDB.')

    main(args, parser) # Pass parser to main 