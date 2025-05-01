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

## Implementation Details

### Precomputation Script Structure
```python
# scripts/precompute_stfts.py
# - Load audio from AudioTextDataset
# - Generate mixtures using SegmentMixer
# - Compute STFTs for segments and mixtures
# - Save to disk in an efficient format
```

### Precomputed Dataset Structure
```python
# data/precomputed_stft_dataset.py
# - Load precomputed STFTs
# - Return formatted data dictionaries ready for model input
```

### Training Updates
- Modify model to accept precomputed STFTs
- Update YAML config to control use of precomputed data
