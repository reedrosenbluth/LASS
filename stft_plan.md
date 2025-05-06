# Refactoring Plan for Precomputing STFTs

## Goals
1. Precompute the mixtures used to train the model
2. Precompute STFTs for these mixtures to make training faster
3. Implement in the most minimally invasive way possible
4. No need to maintain backward compatibility

## Implementation Plan

### 1. Create Precomputation Script
- Create a script that generates mixtures using the existing `SegmentMixer`
- Compute STFTs for both the mixtures and source segments
- Save only the STFTs (no need to save the audio waveforms)
- Store in an efficient format (e.g., numpy arrays, HDF5, or torch tensors)
- Include metadata about the mixing process if needed

### 2. Create Precomputed Dataset
- Create a new dataset class that loads the precomputed STFTs
- Keep the interface similar to the existing dataset but return STFTs directly
- Ensure proper batching and data organization

### 3. Update Training Code
- Minimally modify the training code to use the new dataset
- Skip the STFT computation in the model when using precomputed data
- Update configuration files to support the new workflow

## Next Steps / Considerations (Post Initial Implementation)

1.  **Adapt Base Model (`ss_model`, e.g., `ResUNet30`):** Modify the core separation model (`models/resunet.py`) to accept STFT magnitude inputs (key: `stft_mixture`) and output STFT magnitude (key: `stft_waveform`).
2.  **Update Loss Configuration:** Change `loss_type` in `config/audiosep_base.yaml` from `l1_wav` to an STFT-compatible loss (e.g., `l1_stft_mag`). Decide whether to keep the loss hardcoded in `AudioSep.training_step` or make it configurable via `get_loss_function` again.
3.  **Handle Multiple STFT Window Lengths:** Review the logic in `AudioSep.training_step` that selects the first window length (`selected_win_len = stft_win_lengths[0]`). Adjust if specific selection or combination logic is needed when multiple window lengths are precomputed.
4.  **Verify DataLoader Collation:** Check the exact structure of `batch_data_dict` produced by the `DataLoader` in `AudioSep.training_step`. Implement a custom `collate_fn` if the default collation doesn't match the expected structure for accessing STFTs, text, etc.
5.  **Implement Validation/Testing:** Add STFT-based processing logic to the `test_step` method in `AudioSep` if validation or testing with precomputed data is required.
6.  **Set Precomputed Data Path:** Ensure the `precomputed_stft_dir` value in `config/audiosep_base.yaml` points to the correct location of the generated STFT data before running training.
