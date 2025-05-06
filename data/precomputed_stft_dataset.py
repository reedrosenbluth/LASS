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
        Initializes the dataset by storing LMDB path and fetching total item count.
        The LMDB environment itself is opened lazily per worker.

        Args:
            data_path (str): Path to the LMDB database directory (e.g., /path/to/train.lmdb).
            lock (bool): Whether to use file locking. Should be False for 
                         multi-process DataLoader workers, True if unsure or single process.
        """
        self.lmdb_path = pathlib.Path(data_path)
        self.lock = lock  # Store lock setting for lazy opening
        self.env: Optional[lmdb.Environment] = None
        self.txn: Optional[lmdb.Transaction] = None
        self.total_items: int = 0
        # self.pid = os.getpid() # Can be useful for debugging

        if not self.lmdb_path.exists() or not self.lmdb_path.is_dir():
            # LMDB often creates a directory. Check if the parent exists if path is file-like
            if self.lmdb_path.with_suffix('').exists() and self.lmdb_path.with_suffix('').is_dir():
                 self.lmdb_path = self.lmdb_path.with_suffix('') # Use the directory
            else:
                 raise FileNotFoundError(f"LMDB directory not found at: {data_path}")

        # Get total_items once. This requires opening and closing the env.
        # Workers will rely on this stored value and open their own envs later.
        try:
            temp_env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=self.lock, # Use the lock setting
                readahead=False,
                meminit=False,
                max_readers=1 # Only one reader needed for this temporary operation
            )
            if temp_env is None:
                 raise IOError(f"LMDB environment is None after temp opening {self.lmdb_path} to get count.")
            with temp_env.begin(write=False) as temp_txn:
                self.total_items = temp_txn.stat()['entries']
            temp_env.close()
            # print(f"Main process {os.getpid()}: Fetched total_items = {self.total_items} from {self.lmdb_path}") # Debug
        except lmdb.Error as e:
             raise IOError(f"Failed to open LMDB environment temporarily to get item count from {self.lmdb_path}. Error: {e}")
        except Exception as e:
            # Catch other potential errors during init
             raise IOError(f"Error initializing PrecomputedSTFTDataset (getting count): {e}")

    def _ensure_env_open(self):
        """Ensures the LMDB environment is open for the current process."""
        if self.env is None:
            # print(f"Worker {os.getpid()} (original PID: {self.pid}): Lazily opening LMDB env: {self.lmdb_path}") # Debug
            try:
                self.env = lmdb.open(
                    str(self.lmdb_path),
                    readonly=True,
                    lock=self.lock,      # Use the stored lock setting
                    readahead=False,
                    meminit=False,
                    max_readers=126     # Default, or adjust if many workers
                )
                if self.env is None:
                    raise IOError(f"LMDB environment is None after opening {self.lmdb_path} in worker {os.getpid()}")
                # print(f"Worker {os.getpid()}: Successfully opened LMDB env.") # Debug
            except lmdb.Error as e:
                raise IOError(f"Failed to open LMDB environment at {self.lmdb_path} in worker {os.getpid()}. Error: {e}")

    def _ensure_transaction_open(self):
        """Ensures an LMDB read transaction is open for the current process."""
        self._ensure_env_open()  # Make sure env is open first for this process/worker

        if self.txn is None: # Check if txn is None for this specific worker
            if self.env: # Env should now be open
                try:
                    # Create a new transaction for this worker
                    self.txn = self.env.begin(write=False)
                    # print(f"Worker {os.getpid()}: Opened new read transaction.") # Debug
                except lmdb.Error as e:
                    print(f"Error creating read transaction for {self.lmdb_path} in worker {os.getpid()}: {e}")
                    self.txn = None 
                    raise # Re-raise error
            else:
                # This shouldn't happen if _ensure_env_open succeeded
                raise RuntimeError(f"LMDB environment is not available when trying to create transaction in worker {os.getpid()}.")


    def __len__(self) -> int:
        """Returns the total number of items in the dataset."""
        return self.total_items

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves the precomputed data for the item at the given index from LMDB.
        Ensures an LMDB environment and transaction are available for the current worker process.

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

        # --- Ensure transaction (and thus env) is open for this worker process --- #
        try:
            self._ensure_transaction_open()
        except Exception as e:
            # print(f"Worker {os.getpid()}: Error ensuring transaction open for idx {idx}. Error: {e}") # Debug
            # Could return None or raise. Raising is better for diagnosing hangs.
            raise RuntimeError(f"Worker {os.getpid()}: Failed to ensure LMDB transaction for index {idx}: {e}") from e


        if self.txn is None:
             raise RuntimeError(f"LMDB transaction failed to initialize for index {idx} in worker {os.getpid()}.")

        try:
            # Generate the key
            key = str(idx).encode('utf-8')

            # Get value bytes from LMDB using the worker-specific transaction
            value_bytes = self.txn.get(key)

            # Check if key exists
            if value_bytes is None:
                # print(f"Worker {os.getpid()}: Key '{key.decode()}' not found for index {idx}") # Debug
                # Depending on dataset leniency, could return None or raise.
                # For debugging hangs, it's better to be strict.
                raise KeyError(f"Key '{key.decode()}' not found in LMDB database {self.lmdb_path} for index {idx}")

            # Deserialize value using pickle
            try:
                data_dict = pickle.loads(value_bytes)
            except pickle.UnpicklingError as e:
                # print(f"Worker {os.getpid()}: Unpickling error for key '{key.decode()}' at index {idx}. Error: {e}") # Debug
                raise RuntimeError(f"Failed to unpickle data for key '{key.decode()}' at index {idx} in {self.lmdb_path}: {e}") from e
            except Exception as e: # Catch other unpickling related errors
                # print(f"Worker {os.getpid()}: Generic unpickling error for key '{key.decode()}' at index {idx}. Error: {e}") # Debug
                raise RuntimeError(f"Unexpected error during unpickling for key '{key.decode()}' at index {idx}: {e}") from e

            return data_dict

        except lmdb.Error as e:
            # Handle potential LMDB errors during .get()
            # print(f"Worker {os.getpid()}: LMDB error for key '{key.decode()}' at index {idx}. Error: {e}") # Debug
            raise RuntimeError(f"LMDB error retrieving key '{key.decode()}' for index {idx} from {self.lmdb_path} in worker {os.getpid()}: {e}") from e
        except Exception as e:
            # Catch other unexpected errors during item retrieval
            # print(f"Worker {os.getpid()}: Unexpected error for item {idx}. Error: {e}") # Debug
            raise RuntimeError(f"Unexpected error retrieving item {idx} in worker {os.getpid()}: {e}") from e

    def close(self):
        """Closes the LMDB transaction and environment if they are open for this process."""
        # print(f"Worker {os.getpid()}: Attempting to close LMDB transaction and environment: {self.lmdb_path}") # Debug
        if hasattr(self, 'txn') and self.txn:
            # LMDB read transactions don't strictly need explicit closing with .abort()
            # but setting to None is good practice.
            self.txn = None
        if hasattr(self, 'env') and self.env:
            try:
                self.env.close()
                # print(f"Worker {os.getpid()}: Closed LMDB environment: {self.lmdb_path}") # Debug
                self.env = None # Prevent double closing
            except Exception as e:
                print(f"Error closing LMDB environment {self.lmdb_path} in worker {os.getpid()}: {e}")

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