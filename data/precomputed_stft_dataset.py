import torch
from torch.utils.data import Dataset
import pathlib
import bisect
from typing import List, Dict, Optional, Tuple, Any
import lmdb   # Added
import pickle # Added
import threading # Keep for potential lock usage later?
import os

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
        self.txn: Optional[lmdb.Transaction] = None # Initialize txn to None
        self.total_items: int = 0
        # self._init_lock = threading.Lock() # Lock likely not needed if txn is per-worker

        if not self.lmdb_path.exists() or not self.lmdb_path.is_dir():
            # LMDB often creates a directory. Check if the parent exists if path is file-like
            if self.lmdb_path.with_suffix('').exists() and self.lmdb_path.with_suffix('').is_dir():
                 self.lmdb_path = self.lmdb_path.with_suffix('') # Use the directory
            else:
                 raise FileNotFoundError(f"LMDB directory not found at: {data_path}")

        try:
            # Open the LMDB environment (read-only)
            # This happens in the main process before workers are created.
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

            # --- DO NOT Create a persistent read transaction here --- #
            # Let each worker create its own transaction in __getitem__
            # self._reopen_transaction() # REMOVED

        except lmdb.Error as e:
             raise IOError(f"Failed to open LMDB environment at {self.lmdb_path}. Error: {e}")
        except Exception as e:
            # Catch other potential errors during init
             if self.env:
                 self.env.close()
             raise IOError(f"Error initializing PrecomputedSTFTDataset: {e}")

    def _reopen_transaction(self):
        """Helper to open/reopen the read transaction for the current process."""
        if self.txn:
            # If called again in the same worker, maybe just return?
            # Or abort and reopen? For read-only, it might be okay to reuse.
             return # Assume existing transaction is valid for this worker

        if self.env:
             try:
                 # Create a new transaction for this worker
                 self.txn = self.env.begin(write=False)
                 # print(f"Worker {os.getpid()}: Opened new read transaction for {self.lmdb_path}") # Debug
             except lmdb.Error as e:
                 print(f"Error creating read transaction for {self.lmdb_path} in worker {os.getpid()}: {e}")
                 self.txn = None # Ensure txn is None on failure
                 raise # Re-raise error, likely indicates environment issue
        else:
             # This shouldn't happen if __init__ succeeded
             print("Warning: Cannot create transaction, LMDB environment is not open.")
             self.txn = None
             raise RuntimeError("LMDB environment is not available when trying to create transaction.")

    def __len__(self) -> int:
        """Returns the total number of items in the dataset."""
        return self.total_items

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves the precomputed data for the item at the given index from LMDB.
        Ensures an LMDB transaction is available for the current worker process.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing the precomputed STFTs, text,
                            and other metadata for the requested item.

        Raises:
            IndexError: If the index is out of bounds.
            KeyError: If the key for the index is not found in LMDB.
            RuntimeError: If the LMDB environment or transaction cannot be accessed.
        """
        if not 0 <= idx < self.total_items:
            raise IndexError(f"Index {idx} out of bounds for dataset with size {self.total_items}")

        # --- Ensure transaction exists for this worker process --- #
        if self.txn is None:
            # First time __getitem__ is called in this worker, create a transaction.
            # Rely on _reopen_transaction to handle env checks and errors.
            self._reopen_transaction()
            # If it failed, _reopen_transaction would have raised an error.

        # Now self.txn should be valid for this worker
        if self.txn is None:
             # Safety check in case _reopen_transaction logic changes
             raise RuntimeError(f"LMDB transaction failed to initialize for index {idx}.")

        try:
            # Generate the key
            key = str(idx).encode('utf-8')

            # Get value bytes from LMDB using the worker-specific transaction
            value_bytes = self.txn.get(key)

            # Check if key exists
            if value_bytes is None:
                raise KeyError(f"Key '{key.decode()}' not found in LMDB database {self.lmdb_path} for index {idx}")

            # Deserialize value using pickle
            try:
                data_dict = pickle.loads(value_bytes)
            except pickle.UnpicklingError as e:
                raise RuntimeError(f"Failed to unpickle data for key '{key.decode()}' at index {idx} in {self.lmdb_path}: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Unexpected error during unpickling for key '{key.decode()}' at index {idx}: {e}") from e

            return data_dict

        except lmdb.Error as e:
            # Handle potential LMDB errors during .get()
            # You might want more specific error handling here (e.g., retry?)
            raise RuntimeError(f"LMDB error retrieving key '{key.decode()}' for index {idx} from {self.lmdb_path}: {e}") from e
        except Exception as e:
            # Catch other unexpected errors during item retrieval
            raise RuntimeError(f"Unexpected error retrieving item {idx}: {e}") from e

    def close(self):
        """Closes the LMDB environment if it's open."""
        # Note: Transactions are tied to processes. Closing the env in the main
        # process won't close transactions in workers. Workers exit naturally.
        # print(f"Attempting to close LMDB environment: {self.lmdb_path}") # Debug
        self.txn = None # Clear transaction reference in this process
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