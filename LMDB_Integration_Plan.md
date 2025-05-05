# Plan for Integrating LMDB into Precomputed STFT Workflow

**Goal:** Replace the system of saving batches to individual `.pt` files with saving individual items to an LMDB database for faster dataset initialization and more efficient item loading during training.

---

## Phase 1: Modify the Precomputation Script (`scripts/precompute_stfts.py`)

1.  **Add Dependencies:**
    *   Ensure `lmdb` package is installed: `pip install lmdb`
    *   Ensure `pickle` is available (standard library).

2.  **Setup LMDB Environments (in `main` function):**
    *   **Define Paths:** Determine output paths for training and validation LMDB databases (e.g., within `args.output_dir`).
        ```python
        # Example:
        base_output_dir = pathlib.Path(args.output_dir)
        train_lmdb_path = base_output_dir / "train.lmdb"
        val_lmdb_path = base_output_dir / "val.lmdb"
        ```
    *   **Estimate `map_size`:** Estimate the *total* disk space needed for *each* database (train/val). Needs to be significantly larger than the final expected size (e.g., 1.5x - 2x). Calculate this in bytes.
        ```python
        # Example (adjust size based on your data!):
        # Assuming 1.4 TB for training data, set map_size to ~2 TB
        map_size_train_bytes = int(2 * 1024**4)
        map_size_val_bytes = int(0.5 * 1024**4) # Estimate validation size
        ```
    *   **Open Environments:** Before processing each split (train/val), open the corresponding LMDB environment for writing.
        ```python
        # Example for training:
        print(f"Opening LMDB environment for writing: {train_lmdb_path}")
        train_env = lmdb.open(
            str(train_lmdb_path),
            map_size=map_size_train_bytes,
            readonly=False,
            metasync=False, # Can improve write speed, but riskier on crash
            sync=False,     # Can improve write speed, but riskier on crash
            map_async=True, # Can improve write speed
            lock=True       # Usually True for writing
        )
        ```
    *   **Pass Environment:** Pass the opened `lmdb.Environment` object (`train_env` or `val_env`) to the `process_files_for_stfts` function.

3.  **Modify `process_files_for_stfts` Function:**
    *   **New Argument:** Add `lmdb_env: lmdb.Environment` as a parameter.
    *   **Remove Async Saving:** Delete `save_queue`, `save_pbar`, `_save_worker`, and related `threading` logic.
    *   **Initialize Item Counter:** Before the `DataLoader` loop, add `lmdb_item_idx = 0`.
    *   **Inside `DataLoader` Loop:**
        *   After generating the list `batch_data_to_save` (containing individual item dictionaries):
            *   Start a write transaction: `with lmdb_env.begin(write=True) as txn:`
            *   Iterate through `data_dict` in `batch_data_to_save`:
                *   **Generate Key:** `key = str(lmdb_item_idx).encode('utf-8')`
                *   **Serialize Value:** `value = pickle.dumps(data_dict, protocol=pickle.HIGHEST_PROTOCOL)` (Ensure `data_dict` contains CPU tensors before pickling).
                *   **Write to LMDB:** `txn.put(key, value)`
                *   **Increment Counter:** `lmdb_item_idx += 1`
        *   Remove old `save_queue.put()` logic.
    *   **Track Total Items:** Keep track of the final `lmdb_item_idx` to know how many items were successfully written.

4.  **Cleanup (in `main` function):**
    *   After `process_files_for_stfts` finishes for a split, **close the LMDB environment**:
        ```python
        print(f"Closing LMDB environment: {train_lmdb_path}")
        train_env.close()
        ```

---

## Phase 2: Modify the Dataset Class (`data/precomputed_stft_dataset.py`)

1.  **Add Dependencies:**
    *   Needs `lmdb` and `pickle`.

2.  **Update `__init__`:**
    *   **Argument:** `data_dir` parameter should now be the path to the specific LMDB database file (e.g., `/path/to/output/train.lmdb`).
    *   **Remove Old Logic:** Delete code related to file scanning (`glob`), batch file processing, `item_counts`, `cumulative_counts`, caching (`_cached_...`), and lazy initialization (`_lazy_initialize`, `_init_lock`, `_initialized`).
    *   **Open LMDB Environment (Read-Only):**
        ```python
        import lmdb
        import pickle # Add import

        # ... inside __init__ ...
        self.lmdb_path = data_dir # Store path
        if not pathlib.Path(self.lmdb_path).is_file():
             # Check if the main data file exists (lmdb creates a directory sometimes)
             self.lmdb_path = str(pathlib.Path(data_dir) / "data.mdb") # Adjust if needed
             if not pathlib.Path(self.lmdb_path).parent.is_dir():
                 raise FileNotFoundError(f"LMDB directory/file not found at: {data_dir}")

        try:
            self.lmdb_env = lmdb.open(
                str(pathlib.Path(self.lmdb_path).parent), # Usually open the directory
                readonly=True,
                lock=False,      # Recommended for multi-process Dataloaders
                readahead=False, # Generally better for random access patterns
                meminit=False    # Avoid initializing memory map
            )
        except lmdb.Error as e:
             raise IOError(f"Failed to open LMDB environment at {self.lmdb_path}. Error: {e}")

        ```
    *   **Get Total Items:**
        ```python
        with self.lmdb_env.begin(write=False) as txn:
             self.total_items = txn.stat()['entries']
        ```
    *   **Persistent Read Transaction (Recommended):** Store a transaction for reuse.
        ```python
        self.txn = self.lmdb_env.begin(write=False)
        ```
    *   Remove print statements about scanning/deferred loading.

3.  **Update `__len__`:**
    *   Return `self.total_items`.

4.  **Update `__getitem__`:**
    *   Perform index bounds check: `if not 0 <= idx < self.total_items: raise IndexError(...)`
    *   Generate key: `key = str(idx).encode('utf-8')`
    *   Get value bytes from LMDB using the persistent transaction: `value_bytes = self.txn.get(key)`
    *   Check if `value_bytes` is `None`. If so, raise an error (shouldn't happen if bounds check is correct).
        ```python
        if value_bytes is None:
            raise KeyError(f"Key not found in LMDB for index {idx}")
        ```
    *   Deserialize value: `data_dict = pickle.loads(value_bytes)`
    *   Return `data_dict`.
    *   Remove all previous file loading/caching logic.

5.  **Resource Management:**
    *   Add a `close()` method to the class:
        ```python
        def close(self):
            """Closes the LMDB environment."""
            if hasattr(self, 'lmdb_env') and self.lmdb_env:
                try:
                    self.lmdb_env.close()
                    print(f"Closed LMDB environment: {self.lmdb_path}")
                    self.lmdb_env = None # Prevent double closing
                except Exception as e:
                    print(f"Error closing LMDB environment {self.lmdb_path}: {e}")

        def __del__(self):
            # Attempt cleanup, but explicit close() is preferred
             self.close()
        ```
    *   Ensure your training/evaluation script calls `dataset.close()` when finished.

---

## Phase 3: Execution and Testing

1.  Implement Phase 1 changes in `scripts/precompute_stfts.py`.
2.  **Run precomputation:** Execute the modified script to generate the `.lmdb` databases. Monitor closely for `map_size` issues or write errors. This step will take time.
3.  Implement Phase 2 changes in `data/precomputed_stft_dataset.py`.
4.  **Test Thoroughly:**
    *   Verify `PrecomputedSTFTDataset` initialization is fast.
    *   Check `len(dataset)` is correct.
    *   Test `dataset[idx]` for various indices.
    *   Integrate with `DataLoader` (using multiple workers) and ensure it functions correctly and efficiently.
    *   Confirm resource cleanup by calling `dataset.close()` in the main script.

--- 