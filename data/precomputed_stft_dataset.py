import torch
from torch.utils.data import Dataset
import pathlib
import bisect
from typing import List, Dict, Optional, Tuple, Any

class PrecomputedSTFTDataset(Dataset):
    """
    A PyTorch Dataset for loading precomputed STFT data saved in batch files.

    Assumes data is stored in a directory where each file (e.g., batch_XXXXXX.pt)
    contains a list of dictionaries, each dictionary representing one precomputed item.
    """
    def __init__(self, data_dir: str, expected_num_items: Optional[int] = None):
        """
        Initializes the dataset by scanning the data directory and mapping indices.

        Args:
            data_dir (str): Path to the directory containing the precomputed .pt files.
            expected_num_items (Optional[int]): If provided, verifies the total number
                                                 of items found matches this value.
        """
        self.data_dir = pathlib.Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.file_paths: List[pathlib.Path] = []
        self.item_counts: List[int] = []
        self.cumulative_counts: List[int] = [0] # cumulative[i] = total items BEFORE file i

        print(f"Scanning directory for precomputed data: {self.data_dir}")
        # Sort files numerically based on batch index assumed in filename
        sorted_files = sorted(self.data_dir.glob("batch_*.pt"),
                              key=lambda p: int(p.stem.split('_')[-1]))

        if not sorted_files:
            print(f"Warning: No 'batch_*.pt' files found in {self.data_dir}. Dataset will be empty.")
            self.total_items = 0
            return

        for file_path in sorted_files:
            try:
                # Load only to get the count, assumes data is on CPU or can be mapped
                # Using map_location='cpu' is safer if files were saved from GPU
                loaded_list = torch.load(file_path, map_location='cpu')
                if not isinstance(loaded_list, list):
                     print(f"Warning: Expected a list in {file_path}, got {type(loaded_list)}. Skipping file.")
                     continue

                item_count = len(loaded_list)
                if item_count > 0:
                    self.file_paths.append(file_path)
                    self.item_counts.append(item_count)
                    self.cumulative_counts.append(self.cumulative_counts[-1] + item_count)
                else:
                    print(f"Warning: File {file_path} contains an empty list. Skipping.")

            except Exception as e:
                print(f"Warning: Failed to load or process {file_path}. Skipping. Error: {e}")


        self.total_items = self.cumulative_counts[-1]
        print(f"Found {self.total_items} items across {len(self.file_paths)} files.")

        if expected_num_items is not None and self.total_items != expected_num_items:
            print(f"Warning: Found {self.total_items} items, but expected {expected_num_items}.")

        # Cache for the most recently loaded file to speed up sequential access
        self._cached_file_idx: Optional[int] = None
        self._cached_data: Optional[List[Dict]] = None

    def __len__(self) -> int:
        """Returns the total number of items in the dataset."""
        return self.total_items

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves the precomputed data for the item at the given global index.

        Args:
            idx (int): The global index of the item to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing the precomputed STFTs, text,
                            and other metadata for the requested item.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if not 0 <= idx < self.total_items:
            raise IndexError(f"Index {idx} out of bounds for dataset with size {self.total_items}")

        # Find which file contains the item using binary search on cumulative counts
        # bisect_right returns insertion point > all elements <= target
        # We want the index `file_idx` such that cumulative_counts[file_idx] <= idx < cumulative_counts[file_idx+1]
        file_idx = bisect.bisect_right(self.cumulative_counts, idx) - 1

        # Calculate the index within the target file
        index_in_file = idx - self.cumulative_counts[file_idx]

        # Check cache
        if file_idx == self._cached_file_idx and self._cached_data is not None:
            # print(f"Cache hit for file index {file_idx}") # Debugging
            pass # Data is already cached
        else:
            # print(f"Cache miss. Loading file index {file_idx}: {self.file_paths[file_idx]}") # Debugging
            # Load the file
            target_file_path = self.file_paths[file_idx]
            try:
                self._cached_data = torch.load(target_file_path, map_location='cpu')
                self._cached_file_idx = file_idx
                # Basic validation after load
                if not isinstance(self._cached_data, list) or len(self._cached_data) != self.item_counts[file_idx]:
                     print(f"Warning: Data inconsistency loading {target_file_path}. Expected list of size {self.item_counts[file_idx]}, got {type(self._cached_data)} of size {len(self._cached_data) if isinstance(self._cached_data, list) else 'N/A'}.")
                     # Attempt to recover if possible, otherwise raise or handle
                     if not isinstance(self._cached_data, list) or index_in_file >= len(self._cached_data):
                         raise RuntimeError(f"Cannot retrieve item {index_in_file} from corrupted/invalid file {target_file_path}")

            except Exception as e:
                # Clear cache on error and re-raise
                self._cached_data = None
                self._cached_file_idx = None
                raise RuntimeError(f"Failed to load data file {target_file_path}: {e}")


        # Retrieve the specific item from the loaded (and now cached) data
        try:
            item_data = self._cached_data[index_in_file]
            return item_data
        except IndexError:
             # This might happen if the file changed or initial count was wrong
             raise IndexError(f"Internal Error: Calculated index {index_in_file} out of bounds for file {self.file_paths[file_idx]} which has {len(self._cached_data)} items.")
        except TypeError:
             # This might happen if _cached_data is not a list as expected
              raise TypeError(f"Internal Error: Cached data for file {self.file_paths[file_idx]} is not a list.")

# Example Usage (Optional: for testing)
if __name__ == '__main__':
    # Create a dummy directory and files for testing
    dummy_dir = pathlib.Path("./dummy_precomputed_data")
    dummy_dir.mkdir(exist_ok=True)

    # Create dummy data (replace with actual structure if needed)
    def create_dummy_item(idx):
        return {
            'stfts': {'mixture': {1024: (torch.randn(1, 100, 513), torch.randn(1, 100, 513), torch.randn(1, 100, 513))},
                      'segment': {1024: (torch.randn(1, 100, 513), torch.randn(1, 100, 513), torch.randn(1, 100, 513))}},
            'text': f'Dummy text {idx}',
            'mixture_component_texts': [f'Dummy text {idx}', f'Noise text {idx+1}'],
            'stft_common_params': {'hop_length': 256},
            'stft_win_lengths': [1024]
        }

    batch_0_data = [create_dummy_item(i) for i in range(10)]
    torch.save(batch_0_data, dummy_dir / "batch_000000.pt")

    batch_1_data = [create_dummy_item(i) for i in range(10, 25)]
    torch.save(batch_1_data, dummy_dir / "batch_000001.pt")

    batch_2_data = [] # Empty file
    torch.save(batch_2_data, dummy_dir / "batch_000002.pt")

    batch_3_data = [create_dummy_item(i) for i in range(25, 30)]
    torch.save(batch_3_data, dummy_dir / "batch_000003.pt")

    print("Dummy data created.")

    # Initialize dataset
    try:
        dataset = PrecomputedSTFTDataset(str(dummy_dir))
        print(f"Dataset length: {len(dataset)}")

        # Test __getitem__
        if len(dataset) > 0:
            print("Testing __getitem__:")
            item_0 = dataset[0]
            print(f"Item 0 text: {item_0.get('text')}")
            item_10 = dataset[10] # Should be first item in second file
            print(f"Item 10 text: {item_10.get('text')}")
            item_24 = dataset[24] # Should be last item in second file
            print(f"Item 24 text: {item_24.get('text')}")
            item_25 = dataset[25] # Should be first item in fourth file (skipping empty file)
            print(f"Item 25 text: {item_25.get('text')}")
            item_last = dataset[len(dataset) - 1]
            print(f"Last item ({len(dataset)-1}) text: {item_last.get('text')}")

        # Test index out of bounds
        try:
            dataset[len(dataset)]
        except IndexError as e:
            print(f"Successfully caught expected IndexError: {e}")

        # Test with DataLoader
        if len(dataset) > 0:
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            print("Testing DataLoader iteration:")
            for i, batch in enumerate(dataloader):
                print(f"Batch {i+1} size: {len(batch['text'])}")
                # print(f"Batch {i+1} keys: {batch.keys()}")
                # print(f"Batch {i+1} texts: {batch['text']}")
                if i >= 2: # Print first few batches
                    break
            print("DataLoader test finished.")

    except Exception as e:
        print(f"An error occurred during testing: {e}")
    finally:
        # Clean up dummy files
        import shutil
        if dummy_dir.exists():
             shutil.rmtree(dummy_dir)
             print(f"Cleaned up dummy directory: {dummy_dir}") 