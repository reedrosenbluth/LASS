# Next Steps for STFT Precomputation Refactoring

1.  **Verify `scripts/precompute_stfts.py`:**
    *   Run the script on a small subset of the data.
    *   Check the output directory (`args.output_dir`) to ensure `.pt` files are created correctly.
    *   Inspect a few `.pt` files to confirm they contain the expected keys (`mixture_stft`, `segment_stft`, `text`) and that the STFT data has the correct shape and type.

2.  **Create `data/precomputed_stft_dataset.py`:**
    *   Define a new `torch.utils.data.Dataset` class (e.g., `PrecomputedSTFTDataset`).
    *   In `__init__`, it should scan the directory containing the precomputed `.pt` files and store the file paths.
    *   Implement `__len__` to return the number of precomputed files found.
    *   Implement `__getitem__` to load a specific `.pt` file using `torch.load` based on the index and return the data dictionary.

3.  **Modify Training/Model Code:**
    *   **Update Data Module:** Modify the `DataModule` (likely in `train.py` or a related data setup file) to optionally instantiate and use `PrecomputedSTFTDataset` instead of `AudioTextDataset`. This could be controlled by a configuration flag.
    *   **Update Model Input:** Modify the `ResUNet30` (or `ResUNet30_Base`) model's `forward` method in `models/resunet.py`.
        *   It should accept the precomputed STFT components (magnitude, cos, sin) directly, potentially as part of the `input_dict`.
        *   Bypass the initial `self.wav_to_spectrogram_phase(mixtures)` call when precomputed data is provided.
        *   Ensure the rest of the model uses these precomputed STFTs correctly (e.g., the `mag`, `cos_in`, `sin_in` variables should be populated from the input dictionary).
    *   **Update Configuration:** Add options to the YAML configuration files to:
        *   Specify the path to the precomputed STFT directory.
        *   Control whether to use the precomputed dataset or the original on-the-fly processing.

4.  **Test End-to-End:**
    *   Run the training script (`train.py`) using the new precomputed dataset and updated model.
    *   Verify that training proceeds correctly and that the STFT computation is indeed skipped, leading to faster data loading/iteration times. 