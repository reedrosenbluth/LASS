import torch
from torch.utils.data import Dataset
import pathlib
import bisect
from typing import List, Dict, Optional, Tuple, Any
import lmdb   # Added
import pickle # Added
import threading # Keep for potential lock usage later?

class PrecomputedSTFTDataset(Dataset):
    """
    A PyTorch Dataset for loading precomputed STFT data from an LMDB database.

    Assumes data is stored in an LMDB environment where keys are string representations
    of indices (e.g., '0', '1', ...) and values are pickled dictionaries.
    """
    def __init__(self, data_path: str, lock: bool = False):
        """
        Initializes the dataset by opening the LMDB environment.

        Args:
            data_path (str): Path to the LMDB database directory (e.g., /path/to/train.lmdb).
            lock (bool): Whether to use file locking. Should be False for 
                         multi-process DataLoader workers, True if unsure or single process.
        """
        self.lmdb_path = pathlib.Path(data_path)
        self.env: Optional[lmdb.Environment] = None
        self.txn: Optional[lmdb.Transaction] = None
        self.total_items: int = 0
        # Add a lock for thread-safe initialization of the transaction if needed
        # Although typically __init__ runs before workers fork
        self._init_lock = threading.Lock()

        if not self.lmdb_path.exists() or not self.lmdb_path.is_dir():
            # LMDB often creates a directory. Check if the parent exists if path is file-like
            if self.lmdb_path.with_suffix('').exists() and self.lmdb_path.with_suffix('').is_dir():
                 self.lmdb_path = self.lmdb_path.with_suffix('') # Use the directory
            else:
                 raise FileNotFoundError(f"LMDB directory not found at: {data_path}")

        try:
            # Open the LMDB environment (read-only)
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=lock,         # Set based on arg - IMPORTANT for DataLoader!
                readahead=False,   # Generally False for random access
                meminit=False,     # Don't initialize map in read-only
                max_readers=126    # Default, adjust if needed
            )

            if self.env is None:
                 raise IOError(f"LMDB environment is None after opening {self.lmdb_path}")

            # Get total number of items from LMDB stats
            with self.env.begin(write=False) as txn_stat:
                self.total_items = txn_stat.stat()['entries']

            print(f"Opened LMDB: {self.lmdb_path}. Found {self.total_items} items.")

            # Create a persistent read transaction (for efficiency)
            # This might need care with multiprocessing if env is not locked correctly
            # Consider creating transaction inside __getitem__ if issues arise
            self._reopen_transaction() # Use a helper to initialize txn

        except lmdb.Error as e:
             raise IOError(f"Failed to open LMDB environment at {self.lmdb_path}. Error: {e}")
        except Exception as e:
            # Catch other potential errors during init
             if self.env:
                 self.env.close()
             raise IOError(f"Error initializing PrecomputedSTFTDataset: {e}")

    def _reopen_transaction(self):
        """Helper to open/reopen the read transaction."""
        # Close existing transaction if it exists
        # This might not be strictly necessary but ensures clean state
        # if called multiple times (e.g., after worker fork?)
        # if self.txn:
        #     self.txn = None # Abort previous if any

        if self.env:
             try:
                 self.txn = self.env.begin(write=False)
                 # print(f"Opened new read transaction for {self.lmdb_path}") # Debug
             except lmdb.Error as e:
                 print(f"Error creating read transaction for {self.lmdb_path}: {e}")
                 self.txn = None # Ensure txn is None on failure
                 # Potentially try reopening the environment? For now, just fail.
                 raise
        else:
             print("Warning: Cannot reopen transaction, LMDB environment is not open.")
             self.txn = None

    def __len__(self) -> int:
        """Returns the total number of items in the dataset."""
        return self.total_items

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves the precomputed data for the item at the given index from LMDB.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing the precomputed STFTs, text,
                            and other metadata for the requested item.

        Raises:
            IndexError: If the index is out of bounds.
            KeyError: If the key for the index is not found in LMDB.
            RuntimeError: If the LMDB transaction is not available.
        """
        if not 0 <= idx < self.total_items:
            raise IndexError(f"Index {idx} out of bounds for dataset with size {self.total_items}")

        # Ensure transaction is available (might be needed if workers re-initialize)
        # This check adds overhead, consider alternatives if it becomes a bottleneck
        if self.txn is None:
            # Attempt to reopen transaction, potentially needed after fork
            # Use lock to prevent race condition if multiple workers hit this? Usually not necessary.
            # with self._init_lock:
            #     if self.txn is None:
            #         self._reopen_transaction()
            # if self.txn is None:
                 raise RuntimeError(f"LMDB transaction is not available for index {idx}. Environment might be closed or failed to initialize transaction.")

        try:
            # Generate the key
            key = str(idx).encode('utf-8')

            # Get value bytes from LMDB using the persistent transaction
            value_bytes = self.txn.get(key)

            # Check if key exists
            if value_bytes is None:
                # This ideally shouldn't happen if __len__ is correct and DB wasn't modified
                raise KeyError(f"Key '{key.decode()}' not found in LMDB database {self.lmdb_path} for index {idx}")

            # Deserialize value using pickle
            # Use try-except for potential unpickling errors
            try:
                data_dict = pickle.loads(value_bytes)
            except pickle.UnpicklingError as e:
                raise RuntimeError(f"Failed to unpickle data for key '{key.decode()}' at index {idx} in {self.lmdb_path}: {e}")
            except Exception as e:
                raise RuntimeError(f"Unexpected error during unpickling for key '{key.decode()}' at index {idx}: {e}")

            return data_dict

        except lmdb.Error as e:
            # Handle potential LMDB errors during .get()
            raise RuntimeError(f"LMDB error retrieving key '{key.decode()}' for index {idx} from {self.lmdb_path}: {e}")

    def close(self):
        """Closes the LMDB environment if it's open."""
        # print(f"Attempting to close LMDB environment: {self.lmdb_path}") # Debug
        # No need for txn.abort() or commit() as it's read-only
        self.txn = None # Clear transaction reference
        if hasattr(self, 'env') and self.env:
            try:
                self.env.close()
                # print(f"Closed LMDB environment: {self.lmdb_path}") # Debug
                self.env = None # Prevent double closing
            except Exception as e:
                print(f"Error closing LMDB environment {self.lmdb_path}: {e}")

    def __del__(self):
        """Attempt to close the LMDB environment when the object is deleted."""
        self.close()

# Example Usage (Optional: for testing with LMDB)
# This example would need modification to create a dummy LMDB database first.
# if __name__ == '__main__':
#     # 1. Create a dummy LMDB database (e.g., using code similar to precompute_stfts.py)
#     dummy_lmdb_path = "./dummy_precomputed_data.lmdb"
#     # ... code to populate dummy_lmdb_path ...
#     print("Dummy LMDB data created.")

#     # Initialize dataset
#     try:
#         # Use lock=False for typical DataLoader usage
#         dataset = PrecomputedSTFTDataset(dummy_lmdb_path, lock=False)
#         print(f"Dataset length: {len(dataset)}")

#         # Test __getitem__
#         if len(dataset) > 0:
#             print("Testing __getitem__:")
#             item_0 = dataset[0]
#             print(f"Item 0 text: {item_0.get('text')}")
#             item_last = dataset[len(dataset) - 1]
#             print(f"Last item ({len(dataset)-1}) text: {item_last.get('text')}")

#         # Test index out of bounds
#         try:
#             dataset[len(dataset)]
#         except IndexError as e:
#             print(f"Successfully caught expected IndexError: {e}")

#         # Test with DataLoader
#         if len(dataset) > 0:
#             from torch.utils.data import DataLoader
#             # num_workers > 0 requires lock=False usually
#             dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
#             print("Testing DataLoader iteration:")
#             for i, batch in enumerate(dataloader):
#                 print(f"Batch {i+1} size: {len(batch['text'])}")
#                 if i >= 2: break
#             print("DataLoader test finished.")

#     except Exception as e:
#         print(f"An error occurred during testing: {e}")
#     finally:
#         # Clean up dataset resources explicitly
#         if 'dataset' in locals() and dataset:
#             dataset.close()
#         # Clean up dummy LMDB
#         import shutil
#         dummy_lmdb_dir = pathlib.Path(dummy_lmdb_path)
#         if dummy_lmdb_dir.exists():
#              shutil.rmtree(dummy_lmdb_dir)
#              print(f"Cleaned up dummy LMDB directory: {dummy_lmdb_dir}") 