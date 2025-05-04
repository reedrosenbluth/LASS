import json
import argparse
from pathlib import Path
import sys
from collections import Counter

def detect_duplicate_keys_hook(pairs):
    """A hook for json.load(object_pairs_hook=...) to detect duplicate keys."""
    counts = Counter(k for k, v in pairs)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    if duplicates:
        # Raise an error or store the information to be reported later.
        # Raising an error stops parsing immediately on first duplicate.
        # Storing allows finding all duplicates in a file.
        # Let's store it for more comprehensive reporting.
        # We can attach the duplicates to the dictionary being created.
        # Note: This adds a non-standard key to the dict.
        d = {}
        duplicate_info = []
        keys_seen = set()
        for k, v in pairs:
            if k in keys_seen:
                duplicate_info.append(k)
            else:
                d[k] = v
                keys_seen.add(k)
        if duplicate_info:
            # Use a special key to store the info without interfering with normal data keys
            d['__duplicate_keys_found__'] = list(set(duplicate_info))
        return d
    # If no duplicates, create the dict normally
    return dict(pairs)

def check_files_for_duplicate_keys(file_paths):
    """
    Reads multiple JSON files (expected lists of objects) and checks if any object
    within them contains duplicate keys.

    Args:
        file_paths (list[str]): A list of paths to the JSON files.
    """
    found_duplicates = False
    files_processed_count = 0
    total_objects_checked = 0
    objects_with_duplicates = 0

    print("Checking JSON files for objects with duplicate keys...")

    for file_path_str in file_paths:
        file_path = Path(file_path_str)
        if not file_path.is_file():
            print(f"Warning: File not found: {file_path}. Skipping.", file=sys.stderr)
            continue

        files_processed_count += 1
        print(f"Processing file: {file_path}...")
        file_has_duplicates = False
        objects_in_file = 0
        duplicates_in_file = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Use the object_pairs_hook to detect duplicates during parsing
                data = json.load(f, object_pairs_hook=detect_duplicate_keys_hook)

            if not isinstance(data, list):
                print(f"Warning: Expected a list of objects in {file_path}, but found {type(data)}. Structure check might be incomplete.", file=sys.stderr)
                # We can still check if the top-level object has duplicates if it's a dict
                if isinstance(data, dict) and '__duplicate_keys_found__' in data:
                    dup_keys = data['__duplicate_keys_found__']
                    duplicates_in_file.append({ 'index': 'N/A (top-level object)', 'keys': dup_keys })
                    file_has_duplicates = True
                # Handle other types if necessary, or just continue
                continue # Skip further processing if not a list

            # Iterate through the list of objects
            for i, item in enumerate(data):
                objects_in_file += 1
                if isinstance(item, dict) and '__duplicate_keys_found__' in item:
                    dup_keys = item['__duplicate_keys_found__']
                    duplicates_in_file.append({ 'index': i, 'keys': dup_keys })
                    file_has_duplicates = True

            print(f" -> Checked {objects_in_file} objects.")
            total_objects_checked += objects_in_file
            if file_has_duplicates:
                found_duplicates = True
                objects_with_duplicates += len(duplicates_in_file)
                print(f"  -> Found objects with duplicate keys in {file_path}:")
                for dup_info in duplicates_in_file:
                    print(f"    - Object Index: {dup_info['index']}, Duplicate Keys: {dup_info['keys']}")

        except json.JSONDecodeError as e:
            print(f"Error: Could not decode JSON from {file_path}. Problem near: {e}. Skipping.", file=sys.stderr)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}", file=sys.stderr)

    print(f"
--- Summary ---")
    print(f"Files processed: {files_processed_count}")
    print(f"Total objects checked: {total_objects_checked}")

    if not found_duplicates:
        print("\nNo objects with duplicate keys found in the specified files.")
    else:
        print(f"\nFound {objects_with_duplicates} object(s) with duplicate keys across {files_processed_count} file(s). Check logs above for details.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check JSON files for objects containing duplicate keys.")
    parser.add_argument(
        "input_files",
        nargs='+',
        help="Paths to the input JSON files."
    )

    args = parser.parse_args()

    check_files_for_duplicate_keys(args.input_files) 