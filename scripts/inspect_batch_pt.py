import torch
import argparse
import sys
from pprint import pprint

def inspect_batch_file(filepath):
    """Loads a batch .pt file and prints its contents."""
    try:
        # Load the data. torch.load automatically handles loading from CPU/GPU correctly.
        batch_data = torch.load(filepath, map_location=torch.device('cpu')) # Load to CPU for inspection
        print(f"Successfully loaded {filepath}")
        print("-" * 30)

        if not isinstance(batch_data, list):
            print(f"Error: Expected data to be a list, but got {type(batch_data)}")
            return

        print(f"File contains data for {len(batch_data)} items.")
        print("-" * 30)

        for i, item_data in enumerate(batch_data):
            print(f"\n--- Item {i} ---")
            if not isinstance(item_data, dict):
                print(f"  Error: Expected item {i} to be a dict, but got {type(item_data)}")
                continue

            # Print non-tensor data directly
            print("  Metadata:")
            for key, value in item_data.items():
                if key != 'stfts':
                    print(f"    {key}: {value}")

            # Print info about STFT tensors
            if 'stfts' in item_data and isinstance(item_data['stfts'], dict):
                print("\n  STFTs (Shapes: [Channel=1, TimeSteps, FreqBins]):")
                for source_type, stft_results in item_data['stfts'].items():
                     print(f"    Source: {source_type}")
                     if isinstance(stft_results, dict):
                         for win_len, stft_tuple in stft_results.items():
                             if isinstance(stft_tuple, tuple) and len(stft_tuple) == 3:
                                 mag, cos, sin = stft_tuple
                                 print(f"      Win Len {win_len}:")
                                 print(f"        Magnitude Shape: {mag.shape}, Dtype: {mag.dtype}")
                                 print(f"        Cos Phase Shape: {cos.shape}, Dtype: {cos.dtype}")
                                 print(f"        Sin Phase Shape: {sin.shape}, Dtype: {sin.dtype}")
                             else:
                                 print(f"      Win Len {win_len}: Invalid STFT tuple format")
                     else:
                        print(f"      Error: Expected stft_results for {source_type} to be a dict.")
            else:
                print("\n  No 'stfts' key found or not a dictionary.")
            print("-" * 20)

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the contents of a precomputed batch .pt file.")
    parser.add_argument("filepath", type=str, help="Path to the batch_*.pt file to inspect.")
    args = parser.parse_args()

    inspect_batch_file(args.filepath)