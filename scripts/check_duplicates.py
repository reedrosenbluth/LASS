import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

def find_duplicates_in_json(file_paths, key_to_check):
    """
    Reads multiple JSON files (expected to be lists of dicts) and checks for
    duplicate values associated with a specific key across all files.

    Args:
        file_paths (list[str]): A list of paths to the JSON files.
        key_to_check (str): The key within the JSON dictionaries whose value
                            should be checked for duplicates.
    """
    path_counts = Counter()
    path_locations = defaultdict(set)
    total_items_processed = 0
    files_processed_count = 0

    print(f"Checking for duplicates based on key: '{key_to_check}'")

    for file_path_str in file_paths:
        file_path = Path(file_path_str)
        if not file_path.is_file():
            print(f"Warning: File not found: {file_path}. Skipping.", file=sys.stderr)
            continue

        files_processed_count += 1
        print(f"Processing file: {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                print(f"Warning: Expected a list of objects in {file_path}, but found {type(data)}. Skipping.", file=sys.stderr)
                continue

            items_in_file = 0
            for item in data:
                if not isinstance(item, dict):
                    # Optional: warn about non-dict items if necessary
                    continue

                if key_to_check in item:
                    path_value = item[key_to_check]
                    # Normalize path separators for cross-platform consistency if needed
                    # path_value = str(Path(path_value))
                    path_counts[path_value] += 1
                    path_locations[path_value].add(str(file_path))
                    items_in_file += 1
                else:
                    # Optional: Warn if key is missing in some items
                    # print(f"Warning: Key '{key_to_check}' not found in an item in {file_path}", file=sys.stderr)
                    pass
            print(f" -> Processed {items_in_file} items.")
            total_items_processed += items_in_file

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}. Skipping.", file=sys.stderr)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}", file=sys.stderr)

    print(f"\n--- Summary ---")
    print(f"Files processed: {files_processed_count}")
    print(f"Total items checked: {total_items_processed}")

    duplicates = {path: count for path, count in path_counts.items() if count > 1}

    if not duplicates:
        print(f"\nNo duplicate values found for key '{key_to_check}' across the specified files.")
    else:
        print(f"\nFound {len(duplicates)} duplicate value(s) for key '{key_to_check}':")
        # Sort duplicates for consistent output, e.g., by count descending
        sorted_duplicates = sorted(duplicates.items(), key=lambda item: item[1], reverse=True)
        for path, count in sorted_duplicates:
            files_list = sorted(list(path_locations[path]))
            print(f"  - Value: '{path}'")
            print(f"    Count: {count}")
            print(f"    Found in files: {files_list}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for duplicate values for a specific key across multiple JSON files.")
    parser.add_argument(
        "input_files",
        nargs='+',
        help="Paths to the input JSON files."
    )
    parser.add_argument(
        "--key",
        type=str,
        default="audiopath", # Default key to check
        help="The key within the JSON objects to check for duplicate values (e.g., 'audiopath', 'file_path', 'original_audiopath')."
    )

    args = parser.parse_args()

    find_duplicates_in_json(args.input_files, args.key) 