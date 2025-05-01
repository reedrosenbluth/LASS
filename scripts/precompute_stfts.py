import torch
import numpy as np
import argparse
import yaml
from tqdm import tqdm
import pathlib
import json # Added for recipe saving
import random # Added for recipe generation mimicking mixer
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
    
    # Transpose to match (batch_size, time_steps, freq_bins) convention if needed
    # Current shape seems to be (batch_size, freq_bins, time_steps) based on torchlibrosa docs
    # Let's keep it as is first and adjust if downstream expects time_steps first

    # Calculate magnitude and phase components using the updated magphase function
    # Input shapes: (batch_size, freq_bins, time_steps)
    # Output shapes: (batch_size, freq_bins, time_steps)
    magnitude, cos_phase, sin_phase = magphase(real, imag, power=1.0) # power=1.0 for magnitude

    # Add channel dimension: (batch_size, 1, freq_bins, time_steps)
    magnitude = magnitude.unsqueeze(1)
    cos_phase = cos_phase.unsqueeze(1)
    sin_phase = sin_phase.unsqueeze(1)

    # Transpose to expected (batch, channels, time_steps, freq_bins)
    magnitude = magnitude.permute(0, 1, 3, 2)
    cos_phase = cos_phase.permute(0, 1, 3, 2)
    sin_phase = sin_phase.permute(0, 1, 3, 2)

    return magnitude, cos_phase, sin_phase

def save_precomputed_data(output_dir, index, data_dict):
    """
    Saves the precomputed STFT data, text, and potentially mixture component texts
    for a single item, ensuring tensors are on CPU.

    Output File Format (`.pt`):
    Each file contains a dictionary with the following structure:
    {
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
    }
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
            cpu_data_dict[key] = value # Handles text, stft_common_params, stft_win_lengths, mixture_component_texts

    torch.save(cpu_data_dict, filename)


# --- Phase 1: Recipe Generation ---

def generate_mixture_recipes_for_batch(texts, max_mix_num, batch_size):
    """
    Generates mixture recipes for a batch based on SegmentMixer logic,
    without performing audio processing.

    Args:
        texts (List[str]): List of texts corresponding to waveforms in the batch.
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
    """
    batch_recipes = []
    for n in range(batch_size):
        primary_text = texts[n]
        # Mimic SegmentMixer logic: mix_num segments total, including primary
        # Ensure mix_num is at least 2 if max_mix_num allows
        actual_max_mix = min(max_mix_num, batch_size) # Cannot mix more items than available in batch
        min_mix = 2 if actual_max_mix >= 2 else actual_max_mix # Need at least 2 if possible
        if min_mix > actual_max_mix: # Handle edge case batch_size=1
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
                 if current_idx_pos != n and current_idx_pos not in indices_to_add:
                     indices_to_add.append(current_idx_pos)
                     added_count += 1
                 attempts += 1
             if added_count < num_to_add:
                 # Fallback if wrap-around didn't find enough unique items (shouldn't happen with batch_size >= mix_num)
                 print(f"Warning: Could only find {added_count} unique items to mix for item {n} (requested {num_to_add}). Using available.")
                 # Add remaining random indices if needed (less ideal, but prevents crash)
                 remaining_needed = num_to_add - added_count
                 available_others = [idx for idx in possible_indices if idx not in indices_to_add]
                 random.shuffle(available_others)
                 indices_to_add.extend(available_others[:remaining_needed])


        for comp_idx in indices_to_add:
             component_indices.append(comp_idx)
             component_texts.append(texts[comp_idx])


        recipe = {
            # 'output_index' will be added later based on global count
            'primary_segment_index_in_batch': n,
            'primary_segment_text': primary_text,
            'mixture_component_texts': component_texts, # Includes primary text
            'component_indices_in_batch': component_indices, # Includes primary index
            'mix_num': len(component_indices) # Actual number mixed, derived from components
        }
        batch_recipes.append(recipe)
    return batch_recipes

def process_files_for_recipes(data_files, target_recipe_file, configs):
    """
    Processes data files to generate and save mixture recipes in JSON format.
    """
    print(f"Generating recipes for output file: {target_recipe_file}")
    target_recipe_file.parent.mkdir(parents=True, exist_ok=True) # Ensure parent dir exists

    # --- Extract parameters needed for Dataset/DataLoader/RecipeGen ---
    sampling_rate = configs['data']['sampling_rate']
    max_clip_len = configs['data']['segment_seconds']
    max_mix_num = configs['data']['max_mix_num']
    batch_size = configs['train']['batch_size_per_device']
    num_workers = configs['train']['num_workers']
    # --- End Parameter Extraction ---

    print("Initializing dataset...")
    dataset = AudioTextDataset(
        datafiles=data_files,
        sampling_rate=sampling_rate,
        max_clip_len=max_clip_len,
    )
    print(f"Dataset size for this set: {len(dataset)}")

    if not dataset:
        print(f"Warning: No data found for files: {data_files}. Skipping recipe generation for {target_recipe_file.name}")
        # Save an empty list if no data
        with open(target_recipe_file, 'w') as f:
            json.dump([], f, indent=2)
        return 0

    # Mixer logic requires batch_size >= max_mix_num to guarantee enough items for wrap-around mixing
    # Adjust batch size if necessary, although user should ideally configure this appropriately
    effective_batch_size = batch_size
    if batch_size < max_mix_num and len(dataset) >= max_mix_num:
        print(f"Warning: configured batch_size ({batch_size}) < max_mix_num ({max_mix_num}). "
              f"Recipe generation might behave unexpectedly if wrap-around needs more items than batch provides. "
              f"Consider increasing batch_size >= {max_mix_num} or ensure dataset splits work with current batch size.")
        # We proceed with the configured batch_size but the warning stands.
        # The generate_mixture_recipes_for_batch handles cases where batch_size < mix_num is requested.

    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False, # CRITICAL: Must be False for recipes to match Phase 2 order
        num_workers=num_workers,
        collate_fn=None
    )

    print("Starting recipe generation loop...")
    all_recipes = []
    global_item_index = 0

    for batch in tqdm(dataloader):
        texts = batch['text']
        current_batch_size = len(texts) # Actual batch size might be smaller for last batch

        if current_batch_size == 0: continue # Skip empty batches

        # Generate recipes for this batch
        batch_recipes = generate_mixture_recipes_for_batch(
            texts=texts,
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
            all_recipes.append(recipe)

        global_item_index += current_batch_size # Increment by the actual number processed in this batch


    print(f"Saving {len(all_recipes)} recipes to {target_recipe_file}...")
    with open(target_recipe_file, 'w') as f:
        json.dump(all_recipes, f, indent=2)

    print(f"Finished generating recipes for {target_recipe_file.name}. Total items: {len(all_recipes)}.")
    return len(all_recipes)


# --- Phase 2: STFT Computation (Modified from original process_files) ---

# Helper function for dynamic loudness normalization, mimicking SegmentMixer's internal logic
# Requires access to the energy ratio calculation if possible, otherwise approximates.
# Let's try to import the specific function if it's accessible and standalone.
try:
    # Attempt to import the specific helper if it exists and is usable standalone
    from data.waveform_mixers import dynamic_loudnorm as mixer_dynamic_loudnorm
    print("Using dynamic_loudnorm from data.waveform_mixers")
except ImportError:
    print("Could not import dynamic_loudnorm from data.waveform_mixers, using local implementation.")
    # Local implementation as fallback if import fails or function isn't suitable standalone
    def mixer_dynamic_loudnorm(audio, reference, lower_db=-10, higher_db=10):
        """ Standalone implementation mimicking dynamic loudness normalization """
        with torch.no_grad():
            # Ensure inputs are tensors and on the same device
            if not isinstance(audio, torch.Tensor): audio = torch.tensor(audio)
            if not isinstance(reference, torch.Tensor): reference = torch.tensor(reference)
            device = reference.device
            audio = audio.to(device)

            # Energy calculation (handle potential channel dim)
            energy_audio = torch.mean(audio.squeeze() ** 2)
            energy_ref = torch.mean(reference.squeeze() ** 2)
            ratio = (energy_ref / torch.clamp(energy_audio, min=1e-10)) ** 0.5
            ratio = torch.clamp(ratio, 0.02, 50) # Bounds from original SegmentMixer's get_energy_ratio

        rescaled_audio = audio * ratio
        delta_loudness = random.uniform(lower_db, higher_db)
        gain = np.power(10.0, delta_loudness / 20.0)
        return (gain * rescaled_audio).to(audio.dtype) # Ensure dtype consistency


def process_files_for_stfts(data_files, target_output_dir, recipe_file, configs, common_stft_params_for_saving):
    """
    Processes data files using pre-generated recipes: loads data, performs mixing
    based on recipes, computes STFTs, and saves results including mixture texts.
    """
    print(f"Processing STFTs using recipe file: {recipe_file}")
    print(f"Output directory: {target_output_dir}")
    target_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Recipes ---
    print(f"Loading recipes from {recipe_file}...")
    try:
        with open(recipe_file, 'r') as f:
            recipes = json.load(f)
        if not recipes:
             print("Recipe file is empty. Nothing to process.")
             return 0
        # Convert recipes to a dictionary keyed by output_index for efficient lookup
        recipes_dict = {recipe['output_index']: recipe for recipe in recipes}
        print(f"Loaded {len(recipes_dict)} recipes.")
    except FileNotFoundError:
        print(f"Error: Recipe file not found at {recipe_file}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from recipe file: {recipe_file}")
        raise

    # --- Extract parameters from config ---
    sampling_rate = configs['data']['sampling_rate']
    max_clip_len = configs['data']['segment_seconds']
    # max_mix_num = configs['data']['max_mix_num'] # Not directly needed, mix_num comes from recipe
    lower_db = configs['data']['loudness_norm']['lower_db']
    higher_db = configs['data']['loudness_norm']['higher_db']
    batch_size = configs['train']['batch_size_per_device']
    num_workers = configs['train']['num_workers']

    # STFT params
    stft_hop_length = configs['data']['stft_hop_length']
    stft_window = configs['data']['stft_window']
    stft_center = configs['data']['stft_center']
    stft_pad_mode = configs['data']['stft_pad_mode']
    stft_win_lengths = configs['data']['stft_win_lengths']
    # --- End Parameter Extraction ---

    print("Initializing dataset...")
    dataset = AudioTextDataset(
        datafiles=data_files,
        sampling_rate=sampling_rate,
        max_clip_len=max_clip_len,
    )
    print(f"Dataset size for this set: {len(dataset)}")

    if not dataset:
        print(f"Warning: No data found for files: {data_files}. Skipping STFT computation for {target_output_dir}")
        return 0

    # Ensure batch size used here matches (or is compatible with) the one used for recipe generation
    # If recipes were generated with a smaller batch size, this might affect index mapping if not careful.
    # Using shuffle=False and assuming the same batch size is the safest approach.
    effective_batch_size = batch_size
    # Potential check: compare effective_batch_size with recipe generation context if stored?

    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False, # CRITICAL: Must be False and match Phase 1 DataLoader order
        num_workers=num_workers,
        collate_fn=None
    )

    # --- Device Selection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Mixer class instance not needed directly, but params are used by helper
    loudness_params = {'lower_db': lower_db, 'higher_db': higher_db}
    # -----------------------

    print("Starting STFT precomputation loop guided by recipes...")
    processed_count_for_set = 0
    global_item_index_tracker = 0 # Track index based on dataloader iteration

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        waveforms = batch['waveform'].to(device) # Shape: (batch, 1, time)
        texts = batch['text']       # List of strings from batch (for reference/debug)
        current_batch_size = waveforms.size(0)

        if current_batch_size == 0: continue

        batch_mixtures = []
        batch_segments = []
        batch_recipes_used = [] # Store recipes used for this batch

        # Process each item in the batch according to its recipe
        for i in range(current_batch_size):
            current_global_index = global_item_index_tracker + i
            recipe = recipes_dict.get(current_global_index)

            if recipe is None:
                print(f"Warning: No recipe found for expected output index {current_global_index}. Skipping item.")
                continue

            # Basic check: Compare primary text from batch with recipe's primary text
            if texts[i] != recipe['primary_segment_text']:
                 print(f"Warning: Text mismatch for index {current_global_index}. Batch text: '{texts[i]}', Recipe text: '{recipe['primary_segment_text']}'. Proceeding, but check data consistency.")

            batch_recipes_used.append(recipe) # Store for saving later

            # --- Reconstruct mixture based on recipe ---
            primary_segment_index_in_batch = recipe['primary_segment_index_in_batch']
            # Ensure this matches current position 'i' if recipes were generated sequentially within batches
            if primary_segment_index_in_batch != i:
                 print(f"Warning: Recipe primary index in batch ({primary_segment_index_in_batch}) doesn't match current item index in batch ({i}) for global index {current_global_index}. Using index {i} waveform as primary.")
                 primary_segment = waveforms[i].clone()
            else:
                 primary_segment = waveforms[primary_segment_index_in_batch].clone()

            mix_num = recipe['mix_num'] # Actual number of components used
            component_indices_in_batch = recipe['component_indices_in_batch'] # Indices within the batch recipe was generated from

            noise = torch.zeros_like(primary_segment)

            # Add the other components specified by the recipe
            num_added_noise = 0
            for comp_idx_in_batch in component_indices_in_batch:
                # Skip the primary segment itself when accumulating noise
                if comp_idx_in_batch == primary_segment_index_in_batch:
                    continue

                # Check if the component index is valid for the current batch size
                if comp_idx_in_batch >= current_batch_size:
                     print(f"Warning: Component index {comp_idx_in_batch} from recipe is out of bounds for current batch size {current_batch_size}. Skipping component for item {current_global_index}.")
                     continue

                next_segment = waveforms[comp_idx_in_batch]

                # Apply the loudness normalization logic using the helper
                rescaled_next_segment = mixer_dynamic_loudnorm(
                    audio=next_segment, reference=primary_segment, **loudness_params
                )
                noise += rescaled_next_segment
                num_added_noise += 1

            # Normalize the combined noise *if* noise components were added
            if num_added_noise > 0:
                 noise = mixer_dynamic_loudnorm(
                     audio=noise, reference=primary_segment, **loudness_params
                 )

            # Create mixture
            mixture = primary_segment + noise

            # Declipping (same logic as SegmentMixer)
            max_value = torch.max(torch.abs(mixture))
            if max_value > 1:
                gain = 0.9 / max_value
                # Apply gain to both the primary segment *before* mixing and the final mixture
                # This matches the original SegmentMixer behavior more closely
                clipped_primary_segment = primary_segment * gain
                mixture = clipped_primary_segment + (noise * gain) # Re-add noise scaled by same factor
            else:
                clipped_primary_segment = primary_segment # No clipping needed


            batch_mixtures.append(mixture)
            batch_segments.append(clipped_primary_segment) # Store the primary segment (potentially clipped)
            # --- End Mixture Reconstruction ---

        if not batch_mixtures: # If no valid mixtures were generated for this batch
            global_item_index_tracker += current_batch_size # Still advance tracker
            continue

        # Stack the collected mixtures and segments for batch STFT processing
        mixtures_tensor = torch.stack(batch_mixtures, dim=0)
        segments_tensor = torch.stack(batch_segments, dim=0)

        # --- Compute STFTs for the batch ---
        with torch.no_grad():
            batch_mixture_stfts = {win_len: None for win_len in stft_win_lengths}
            batch_segment_stfts = {win_len: None for win_len in stft_win_lengths}

            for win_length in stft_win_lengths:
                n_fft = win_length # Use win_length as n_fft
                current_stft_params = {
                    'n_fft': n_fft,
                    'hop_length': stft_hop_length,
                    'win_length': win_length, # Ensure win_length is passed
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
        # --- End STFT Computation ---

        # --- Save individual items from the batch ---
        num_items_in_batch_processed = mixtures_tensor.size(0) # Use actual number processed
        for i in range(num_items_in_batch_processed):
            recipe_for_item = batch_recipes_used[i] # Get recipe corresponding to this processed item
            item_output_index = recipe_for_item['output_index'] # Get the correct global index

            item_mixture_stfts = {}
            item_segment_stfts = {}
            for win_len in stft_win_lengths:
                # Slice the i-th item from the batch results
                item_mixture_stfts[win_len] = tuple(t[i] for t in batch_mixture_stfts[win_len])
                item_segment_stfts[win_len] = tuple(t[i] for t in batch_segment_stfts[win_len])

            data_to_save = {
                'stfts': {
                    'mixture': item_mixture_stfts,
                    'segment': item_segment_stfts # STFT of the primary segment
                },
                'text': recipe_for_item['primary_segment_text'], # Text of primary segment
                'mixture_component_texts': recipe_for_item['mixture_component_texts'], # List of all component texts
                'stft_common_params': common_stft_params_for_saving,
                'stft_win_lengths': stft_win_lengths
            }
            # Use item_output_index for consistent file naming
            save_precomputed_data(target_output_dir, item_output_index, data_to_save)
            processed_count_for_set += 1 # Increment based on actual items saved

        global_item_index_tracker += current_batch_size # Advance tracker by processed batch size

    # Final check
    if processed_count_for_set != len(recipes_dict):
        print(f"Warning: Number of processed items ({processed_count_for_set}) does not match number of recipes ({len(recipes_dict)}).")

    print(f"Finished STFT processing for {target_output_dir}. Precomputed {processed_count_for_set} items.")
    return processed_count_for_set


def main(args):
    """
    Main function to either generate mixture recipes or compute STFTs based on recipes.
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

    if args.mode == 'generate_recipes':
        if not args.output_recipe_dir:
            # Raise error using parser's error method for consistency if possible, else ValueError
            parser.error("--output_recipe_dir is required for 'generate_recipes' mode.") # Use parser if available in scope, else raise


        base_recipe_dir = pathlib.Path(args.output_recipe_dir)
        train_recipe_file = base_recipe_dir / "train_mixture_recipes.json"
        val_recipe_file = base_recipe_dir / "val_mixture_recipes.json"

        # --- Process Training Files ---
        print("\n--- Processing Training Set Recipes ---")
        total_train_processed = process_files_for_recipes(
            data_files=args.train_data_files,
            target_recipe_file=train_recipe_file,
            configs=configs
        )
        # ----------------------------

        # --- Process Validation Files ---
        print("\n--- Processing Validation Set Recipes ---")
        total_val_processed = process_files_for_recipes(
            data_files=args.val_data_files,
            target_recipe_file=val_recipe_file,
            configs=configs
        )
        # ------------------------------

        print(f"\nFinished all recipe processing.")
        print(f"Total training items processed: {total_train_processed}")
        print(f"Total validation items processed: {total_val_processed}")

    elif args.mode == 'compute_stfts':
        if not args.output_dir:
             # Use parser error method if available
             parser.error("--output_dir is required for 'compute_stfts' mode.")
        if not args.input_recipe_dir:
             # Use parser error method if available
             parser.error("--input_recipe_dir is required for 'compute_stfts' mode.")

        # --- Setup STFT Output Dirs ---
        base_recipe_dir = pathlib.Path(args.input_recipe_dir)
        train_recipe_file = base_recipe_dir / "train_mixture_recipes.json"
        val_recipe_file = base_recipe_dir / "val_mixture_recipes.json"
        print(f"Recipe input directory: {base_recipe_dir}")

        # --- Extract STFT common parameters for saving ---
        stft_hop_length = configs['data']['stft_hop_length']
        stft_window = configs['data']['stft_window']
        stft_center = configs['data']['stft_center']
        stft_pad_mode = configs['data']['stft_pad_mode']
        stft_win_lengths = configs['data']['stft_win_lengths']
        # -------------------------------------------------------------------

        # --- Setup Output Dirs ---
        train_output_dir = base_output_dir / "train"
        val_output_dir = base_output_dir / "val"
        print(f"Training output directory: {train_output_dir}")
        print(f"Validation output directory: {val_output_dir}")
        # ------------------------

        # --- Process Training Files ---
        print("\n--- Processing Training Set STFTs ---")
        total_train_processed = process_files_for_stfts(
            data_files=args.train_data_files,
            target_output_dir=train_output_dir,
            recipe_file=train_recipe_file,
            configs=configs,
            common_stft_params_for_saving=common_stft_params_for_saving
        )
        # ----------------------------

        # --- Process Validation Files ---
        print("\n--- Processing Validation Set STFTs ---")
        total_val_processed = process_files_for_stfts(
            data_files=args.val_data_files,
            target_output_dir=val_output_dir,
            recipe_file=val_recipe_file,
            configs=configs,
            common_stft_params_for_saving=common_stft_params_for_saving
        )
        # ------------------------------

        print(f"\nFinished STFT computation.")
        print(f"Total training items processed: {total_train_processed}")
        print(f"Total validation items processed: {total_val_processed}")

    else:
        # Raise error if mode not recognized
        parser.error("Invalid mode selected. Choose 'generate_recipes' or 'compute_stfts'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute STFTs or generate mixture recipes based on YAML config.")

    # --- Mode Selection ---
    parser.add_argument('--mode', type=str, required=True,
                        help='Mode of operation: "generate_recipes" or "compute_stfts".')

    # --- Config File Argument ---
    parser.add_argument('--config_yaml', type=str, required=True,
                        help='Path to the base configuration YAML file.')

    # --- Dataset/Output Arguments ---
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

    # Load data file paths from config if not provided (logic moved inside main)
    main(args) # Pass parser to main if needed for error reporting

    # Report final file sources
    if args.train_data_files is None:
        print("Using training data files provided via command line: None")
    else:
        print(f"Using training data files provided via command line: {args.train_data_files}")
    if args.val_data_files is None:
        print("Using validation data files provided via command line: None")
    else:
        print(f"Using validation data files provided via command line: {args.val_data_files}")

    # Ensure lists are not empty
    if not args.train_data_files:
         raise ValueError("Training data file list cannot be empty.")
    if not args.val_data_files:
         raise ValueError("Validation data file list cannot be empty.")

    main(args) 