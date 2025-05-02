import torch
import numpy as np
import argparse
import yaml
from tqdm import tqdm
import pathlib
import json
import random
from torch.utils.data import DataLoader
from torchlibrosa.stft import STFT, magphase

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

def save_batch_precomputed_data(output_dir, batch_index, batch_data_list):
    """
    Saves the precomputed STFT data for a whole batch to a single file,
    ensuring tensors are on CPU.

    Output File Format (`.pt`):
    Each file contains a list of dictionaries with the following structure:
    [{
        'stfts': {
            'mixture': {
                <win_len_1>: (mag_tensor, cos_tensor, sin_tensor),
                ...
            },
            'segment': { # Represents the primary segment
                <win_len_1>: (mag_tensor, cos_tensor, sin_tensor),
                ...
            }
        },
        'text': str, # Primary segment caption
        'mixture_component_texts': List[str], # Captions of all segments in the mixture (including primary)
        'stft_common_params': { ... },
        'stft_win_lengths': List[int]
    }]
    """
    if not batch_data_list:
        # This might happen if a batch had items, but all were skipped due to recipe issues.
        print(f"Warning: Attempting to save empty data list for batch {batch_index}. Skipping file creation.")
        return 0 # Indicate 0 items saved

    filename = output_dir / f"batch_{batch_index:06d}.pt"
    batch_cpu_data_list = []

    for data_dict in batch_data_list:
        # Reuse the CPU conversion logic, applied to each item's dict
        cpu_data_dict = {}
        for key, value in data_dict.items():
            if key == 'stfts' and isinstance(value, dict):
                cpu_stfts = {}
                for source_type, stft_results in value.items():
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
                cpu_data_dict[key] = value # Handles text, stft_common_params, stft_win_lengths, mixture_component_texts
        batch_cpu_data_list.append(cpu_data_dict)

    try:
        torch.save(batch_cpu_data_list, filename)
        # print(f"Saved batch {batch_index} ({len(batch_cpu_data_list)} items) to {filename}") # Optional: More verbose logging
        return len(batch_cpu_data_list) # Return number of items saved
    except Exception as e:
        print(f"Error saving batch {batch_index} to {filename}: {e}")
        # Decide how to handle error: raise? continue? log?
        # For now, let's print and return 0 items saved for this batch
        return 0

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
                 # ---- DEBUG PRINT ----
                 if n < 5 and current_idx_pos < 5: # Limit printing to likely duplicate cases early in batch
                     print(f"DEBUG: n={n}, primary_path='{primary_original_path}', current_idx_pos={current_idx_pos}, component_path='{component_original_path}', Paths_Equal={primary_original_path == component_original_path})")
                 # ---- END DEBUG ----
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


        recipe = {
            'primary_segment_index_in_batch': n,
            'primary_segment_text': primary_text,
            'mixture_component_texts': component_texts, # Includes primary text
            'component_indices_in_batch': component_indices, # Includes primary index
            'mix_num': len(component_indices), # Actual number mixed, derived from components
            'original_audiopath': primary_original_path # Add primary path to recipe for reference
        }
        batch_recipes.append(recipe)
    return batch_recipes

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
        num_workers=num_workers,
        collate_fn=None
    )

    print("Starting recipe generation loop...")
    all_recipes = []
    global_item_index = 0

    for batch in tqdm(dataloader):
        texts = batch['text']
        # Retrieve the original audio paths from the batch
        original_audiopaths = batch['original_audiopath']
        current_batch_size = len(texts)

        if current_batch_size == 0: continue

        # Generate recipes for this batch
        batch_recipes = generate_mixture_recipes_for_batch(
            texts=texts,
            original_audiopaths=original_audiopaths,
            max_mix_num=max_mix_num,
            batch_size=current_batch_size
        )

        # Add global output index to each recipe and append to the main list
        for i, recipe in enumerate(batch_recipes):
            # Ensure the primary index from the recipe matches the loop index 'i' within this batch
            # This confirms the recipe corresponds to the correct item positionally
            if recipe['primary_segment_index_in_batch'] != i:
                 print(f"Warning: Recipe primary index mismatch ({recipe['primary_segment_index_in_batch']} != {i}). Check recipe generation logic.")
                 # Attempt to recover by assuming position i corresponds to global_item_index + i
                 # This might indicate an issue in generate_mixture_recipes_for_batch index handling

            recipe['output_index'] = global_item_index + i # Assign global index based on position in dataloader
            # Add the original audio path for this item to the recipe
            recipe['original_audiopath'] = original_audiopaths[i]
            all_recipes.append(recipe)

        global_item_index += current_batch_size # Increment by the actual number processed in this batch


    print(f"Saving {len(all_recipes)} recipes to {target_recipe_file}...")
    with open(target_recipe_file, 'w') as f:
        json.dump(all_recipes, f, indent=2)

    # Report dropped count
    dropped_count = dataset.get_dropped_count()
    if dropped_count > 0:
        print(f"Note: {dropped_count} audio files failed to load correctly and were replaced with random valid samples during recipe generation.")

    print(f"Finished generating recipes for {target_recipe_file.name}. Total items: {len(all_recipes)}.")
    return len(all_recipes)

def process_files_for_stfts(data_files, target_output_dir, recipe_file, configs, common_stft_params_for_saving):
    """
    Processes data files using pre-generated recipes: loads data, performs mixing
    based on recipes, computes STFTs, and saves results including mixture texts.
    """
    print(f"Processing STFTs using recipe file: {recipe_file}")
    print(f"Output directory: {target_output_dir}")
    target_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading recipes from {recipe_file}...")
    try:
        with open(recipe_file, 'r') as f:
            recipes = json.load(f)
        if not recipes:
             print("Recipe file is empty. Nothing to process.")
             return 0
        # Create a dictionary mapping original_audiopath to recipe
        recipes_dict = {recipe['original_audiopath']: recipe for recipe in recipes}
        print(f"Loaded {len(recipes_dict)} recipes, indexed by original audio path.")
    except FileNotFoundError:
        print(f"Error: Recipe file not found at {recipe_file}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from recipe file: {recipe_file}")
        raise

    sampling_rate = configs['data']['sampling_rate']
    max_clip_len = configs['data']['segment_seconds']
    lower_db = configs['data']['loudness_norm']['lower_db']
    higher_db = configs['data']['loudness_norm']['higher_db']
    batch_size = configs['train']['batch_size_per_device']
    num_workers = configs['train']['num_workers']

    stft_hop_length = configs['data']['stft_hop_length']
    stft_window = configs['data']['stft_window']
    stft_center = configs['data']['stft_center']
    stft_pad_mode = configs['data']['stft_pad_mode']
    stft_win_lengths = configs['data']['stft_win_lengths']

    print("Initializing dataset...")
    dataset = AudioTextDataset(
        datafiles=data_files,
        sampling_rate=sampling_rate,
        max_clip_len=max_clip_len,
        suppress_warnings=True
    )
    print(f"Dataset size for this set: {len(dataset)}")

    if not dataset:
        print(f"Warning: No data found for files: {data_files}. Skipping STFT computation for {target_output_dir}")
        return 0

    effective_batch_size = batch_size

    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    loudness_params = {'lower_db': lower_db, 'higher_db': higher_db}

    print("Starting STFT precomputation loop guided by recipes...")
    processed_count_for_set = 0
    global_item_index_tracker = 0 # Track index based on dataloader iteration

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        waveforms = batch['waveform'].to(device) # Shape: (B, 1, T)
        texts = batch['text']
        original_audiopaths = batch['original_audiopath'] # List of B paths
        current_batch_size = waveforms.size(0)

        if current_batch_size == 0: continue

        # --- Vectorized Mixture Creation START ---
        
        # 1. Get recipes for the current batch items
        batch_recipes_used = []
        valid_indices_in_batch = [] # Indices of items in the batch for which recipes were found
        for i in range(current_batch_size):
            current_original_path = original_audiopaths[i]
            recipe = recipes_dict.get(current_original_path)
            if recipe is None:
                print(f"Warning: No recipe found for original audio path '{current_original_path}'. Skipping item {i} in batch {batch_idx}.")
                continue
            
            # Basic validation: Ensure primary index in recipe matches current loop index 'i'
            # This assumes recipes were generated sequentially matching dataloader order.
            if recipe.get('primary_segment_index_in_batch') != i:
                 print(f"Warning: Recipe primary index mismatch for item {i} (Path: {current_original_path}). Expected {i}, got {recipe.get('primary_segment_index_in_batch')}. Assuming recipe is correct for this item based on path.")
                 # We proceed using the recipe found via path, trusting it corresponds to waveform[i]
                 # Modify recipe's primary index to match current position if needed for consistency downstream?
                 # Or better, rely solely on the order and 'i'. Let's assume recipe['primary...'] might be misleading if batches differed.
            
            batch_recipes_used.append(recipe)
            valid_indices_in_batch.append(i)

        if not batch_recipes_used: # If no valid recipes found for this batch
            global_item_index_tracker += current_batch_size
            continue
            
        # Filter waveforms and recipes to only include valid items
        valid_waveforms = waveforms[valid_indices_in_batch] # Shape: (B_valid, 1, T)
        num_valid_items = len(batch_recipes_used)

        # 2. Gather primary segments (these are just the valid waveforms)
        # The primary segment for the k-th valid item corresponds to valid_waveforms[k]
        primary_segments = valid_waveforms # Shape: (B_valid, 1, T)

        # 3. Gather and combine noise components
        # Initialize noise tensor
        noise_accumulator = torch.zeros_like(primary_segments) # Shape: (B_valid, 1, T)
        
        # This part remains tricky to fully vectorize due to variable numbers of components
        # and the loudness norm needing a reference. We'll loop through valid items,
        # but perform lookups and norms more directly on tensors.
        
        num_noise_components_added = torch.zeros(num_valid_items, device=device) # Track counts per item

        for k, recipe in enumerate(batch_recipes_used):
            primary_seg_for_norm = primary_segments[k] # Reference for loudness norm (Shape: 1, T)
            component_indices_in_batch = recipe['component_indices_in_batch']
            primary_idx_in_orig_batch = valid_indices_in_batch[k] # The original index 'i' for this item

            item_noise = torch.zeros_like(primary_seg_for_norm) # Accumulator for this specific item

            for comp_idx_in_orig_batch in component_indices_in_batch:
                # Skip the primary segment itself
                if comp_idx_in_orig_batch == primary_idx_in_orig_batch:
                    continue

                # Check if the component index is valid for the *original* batch size
                # And if the component index corresponds to an item *with* a valid recipe
                if comp_idx_in_orig_batch >= current_batch_size or comp_idx_in_orig_batch not in valid_indices_in_batch:
                     # Find the relative index in the valid batch, if it exists
                     try:
                         comp_idx_in_valid_batch = valid_indices_in_batch.index(comp_idx_in_orig_batch)
                         # If index is valid, but recipe wasn't found, we can't use it.
                         print(f"Warning: Component index {comp_idx_in_orig_batch} from recipe for item {primary_idx_in_orig_batch} exists in batch but had no valid recipe. Skipping component.")
                     except ValueError:
                         # Index is out of bounds for original batch or wasn't loaded.
                         print(f"Warning: Component index {comp_idx_in_orig_batch} from recipe for item {primary_idx_in_orig_batch} is out of bounds ({current_batch_size}) or was skipped. Skipping component.")
                     continue

                # Get the component waveform using its original index
                next_segment = waveforms[comp_idx_in_orig_batch] # Shape: (1, T)

                # Apply loudness normalization relative to the primary segment of the *current item k*
                # Note: dynamic_loudnorm expects (1, T) or (T,) input based on its internal rescale
                rescaled_next_segment = mixer_dynamic_loudnorm(
                    audio=next_segment, reference=primary_seg_for_norm, **loudness_params
                )
                item_noise += rescaled_next_segment
                num_noise_components_added[k] += 1

            # Normalize the combined noise for this item *if* components were added
            if num_noise_components_added[k] > 0:
                 item_noise = mixer_dynamic_loudnorm(
                     audio=item_noise, reference=primary_seg_for_norm, **loudness_params
                 )

            noise_accumulator[k] = item_noise # Store accumulated noise for item k

        # 4. Create Mixtures
        mixtures = primary_segments + noise_accumulator # Shape: (B_valid, 1, T)

        # 5. De-clipping (Batched)
        # Find max absolute value per item across the time dimension
        max_values, _ = torch.max(torch.abs(mixtures.squeeze(1)), dim=1) # Shape: (B_valid,)
        needs_clipping = max_values > 1.0 # Boolean mask (B_valid,)
        
        # Calculate gain per item: 0.9 / max_value if clipping needed, 1.0 otherwise
        gain = torch.ones_like(max_values) # Shape: (B_valid,)
        gain[needs_clipping] = 0.9 / max_values[needs_clipping]
        
        # Unsqueeze gain to apply to waveforms (B_valid, 1, 1) for broadcasting
        gain = gain.unsqueeze(-1).unsqueeze(-1)

        # Apply gain to primary segments and mixtures where needed
        final_segments = primary_segments * gain # Shape: (B_valid, 1, T)
        final_mixtures = mixtures * gain       # Shape: (B_valid, 1, T)
        
        # --- Vectorized Mixture Creation END ---

        # Ensure tensors have the expected names for STFT calculation
        mixtures_tensor = final_mixtures
        segments_tensor = final_segments

        # Check if any valid items remain after filtering and processing
        if mixtures_tensor.size(0) == 0:
            global_item_index_tracker += current_batch_size # Still advance tracker
            continue

        # Compute STFTs for the batch of valid items
        with torch.no_grad():
            # These dictionaries will store results only for the valid items
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
                    mixtures_tensor, **current_stft_params
                )
                # Note: Shapes here are (B_valid, 1, T', F)
                batch_mixture_stfts[win_length] = (mixture_mag, mixture_cos, mixture_sin)

                segment_mag, segment_cos, segment_sin = calculate_stft_components(
                    segments_tensor, **current_stft_params
                )
                batch_segment_stfts[win_length] = (segment_mag, segment_cos, segment_sin)

        # Prepare data for saving (loop through the *valid* items processed)
        batch_data_to_save = []
        num_items_in_batch_processed = mixtures_tensor.size(0) # Number of valid items

        for k in range(num_items_in_batch_processed): # Loop index 'k' corresponds to valid items
            recipe_for_item = batch_recipes_used[k] # Get recipe corresponding to this processed item
            item_output_index = recipe_for_item['output_index'] # Get the correct global index from the recipe

            item_mixture_stfts = {}
            item_segment_stfts = {}
            for win_len in stft_win_lengths:
                # Slice the k-th item from the batch results
                # Input tensors are (B_valid, 1, T', F) -> Slice to (1, T', F)
                item_mixture_stfts[win_len] = tuple(t[k] for t in batch_mixture_stfts[win_len])
                item_segment_stfts[win_len] = tuple(t[k] for t in batch_segment_stfts[win_len])

            data_to_save = {
                'stfts': {
                    'mixture': item_mixture_stfts,
                    'segment': item_segment_stfts # STFT of the primary segment
                },
                'text': recipe_for_item['primary_segment_text'], # Text of primary segment
                'mixture_component_texts': recipe_for_item['mixture_component_texts'], # List of all component texts
                'stft_common_params': common_stft_params_for_saving,
                'stft_win_lengths': stft_win_lengths
                # 'output_index': item_output_index # Could add for debugging, but not needed for saving format
            }
            batch_data_to_save.append(data_to_save) # Append data for batch saving

        # Save the collected batch data
        if batch_data_to_save:
            items_saved_count = save_batch_precomputed_data(target_output_dir, batch_idx, batch_data_to_save)
            processed_count_for_set += items_saved_count # Increment count by number actually saved
        
        # Advance tracker by the original batch size, as dataloader moves forward
        global_item_index_tracker += current_batch_size

    if processed_count_for_set != len(recipes_dict):
        # This warning might trigger more often if items are skipped due to missing recipes.
        print(f"Warning: Number of processed items ({processed_count_for_set}) does not match number of recipes ({len(recipes_dict)}). This might be due to skipped items.")

    dropped_count_stft = dataset.get_dropped_count()
    if dropped_count_stft > 0:
        print(f"Note: {dropped_count_stft} audio files failed to load correctly and were replaced with random valid samples during STFT computation.")

    print(f"Finished STFT processing for {target_output_dir}. Precomputed {processed_count_for_set} items.")
    return processed_count_for_set


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
             parser.error("--output_dir is required for 'compute_stfts' mode.")
        if not args.input_recipe_dir:
             parser.error("--input_recipe_dir is required for 'compute_stfts' mode.")

        # Setup STFT Output Dirs (Moved INSIDE this block)
        base_output_dir = pathlib.Path(args.output_dir)
        train_output_dir = base_output_dir / "train"
        val_output_dir = base_output_dir / "val"
        print(f"STFT output directory: {base_output_dir}")
        print(f"Training STFT output directory: {train_output_dir}")
        print(f"Validation STFT output directory: {val_output_dir}")

        # Setup Recipe Input Paths
        base_recipe_dir = pathlib.Path(args.input_recipe_dir)
        train_recipe_file = base_recipe_dir / "train_mixture_recipes.json"
        val_recipe_file = base_recipe_dir / "val_mixture_recipes.json"
        print(f"Recipe input directory: {base_recipe_dir}")

        # Extract STFT common parameters for saving
        # Ensure 'data' key exists before accessing subkeys
        if 'data' not in configs:
             parser.error("Missing 'data' section in config YAML required for STFT parameters.")

        try:
            stft_hop_length = configs['data']['stft_hop_length']
            stft_window = configs['data']['stft_window']
            stft_center = configs['data']['stft_center']
            stft_pad_mode = configs['data']['stft_pad_mode']
            stft_win_lengths = configs['data']['stft_win_lengths'] # Needed here too
        except KeyError as e:
             parser.error(f"Missing required STFT parameter in config YAML under 'data': {e}")

        common_stft_params_for_saving = {
            'hop_length': stft_hop_length,
            'window': stft_window,
            'center': stft_center,
            'pad_mode': stft_pad_mode,
        }

        # Process Training Files
        print("\n--- Processing Training Set STFTs ---")
        total_train_processed = process_files_for_stfts(
            data_files=args.train_data_files,
            target_output_dir=train_output_dir,
            recipe_file=train_recipe_file,
            configs=configs,
            common_stft_params_for_saving=common_stft_params_for_saving
        )

        # Process Validation Files
        print("\n--- Processing Validation Set STFTs ---")
        total_val_processed = process_files_for_stfts(
            data_files=args.val_data_files,
            target_output_dir=val_output_dir,
            recipe_file=val_recipe_file,
            configs=configs,
            common_stft_params_for_saving=common_stft_params_for_saving
        )

        print(f"\nFinished STFT computation.")
        print(f"Total training items processed: {total_train_processed}")
        print(f"Total validation items processed: {total_val_processed}")

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
                        help='Directory to save precomputed STFT files (will create train/ and val/ subdirs).')
    parser.add_argument('--output_recipe_dir', type=str, default=None,
                        help="Directory to save recipe JSON files (mode='generate_recipes'). Expects train_mixture_recipes.json and val_mixture_recipes.json.")
    parser.add_argument('--input_recipe_dir', type=str, default=None,
                        help="Directory containing recipe JSON files to use (mode='compute_stfts'). Expects train_mixture_recipes.json and val_mixture_recipes.json.")

    args = parser.parse_args()

    main(args, parser) # Pass parser to main 