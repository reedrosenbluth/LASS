import json
import argparse
from collections import Counter, defaultdict
import pathlib

def find_audio_path_key(entry):
    """Attempts to find the key used for the audio path in a JSON entry."""
    common_keys = ['audio_path', 'audiopath', 'audio', 'file', 'filename', 'path']
    for key in common_keys:
        if key in entry:
            return key
    return None

def check_duplicates(file_paths, split_label):
    """
    Checks for duplicate audio paths within and across a list of JSON files.

    Args:
        file_paths (list): A list of strings or Path objects representing the JSON files.
        split_label (str): A label for the data split (e.g., "train", "val").
    """
    print(f"--- Checking {split_label.upper()} set ---")
    all_paths = []
    path_counts = Counter()
    path_locations = defaultdict(set)
    total_entries = 0
    files_processed = 0
    files_failed = 0
    entries_missing_key = 0

    # Determine the key used for audio path based on the first valid entry
    audio_key_name = None
    checked_first_entry = False

    for file_path_str in file_paths:
        file_path = pathlib.Path(file_path_str)
        if not file_path.is_file():
            print(f"Warning: File not found: {file_path}. Skipping.")
            files_failed += 1
            continue

        try:
            with open(file_path, 'r') as f:
                loaded_json = json.load(f) # Load the whole JSON object first
            files_processed += 1

            # Extract the list of entries
            entries_list = None
            if isinstance(loaded_json, list):
                entries_list = loaded_json # It's already a list
            elif isinstance(loaded_json, dict) and 'data' in loaded_json and isinstance(loaded_json['data'], list):
                entries_list = loaded_json['data'] # Extract the list from the 'data' key
            else:
                print(f"Warning: Expected a list or a dict with a 'data' key containing a list in {file_path}, but got {type(loaded_json)}. Skipping file.")
                files_failed += 1
                continue

            # Use entries_list from here onwards
            if not entries_list:
                print(f"Info: File {file_path} has an empty 'data' list or is an empty list.")
                continue # Skip empty files

            # Determine audio path key if not already found
            if not checked_first_entry:
                for entry in entries_list: # Find the first valid entry to determine the key
                     if isinstance(entry, dict):
                         audio_key_name = find_audio_path_key(entry)
                         if audio_key_name:
                             print(f"Detected audio path key: '{audio_key_name}' (using key '{audio_key_name}' based on {file_path.name})") # Use detected key
                             checked_first_entry = True
                             break
                if not checked_first_entry:
                     # This case is less likely now if the file structure is consistent, but keep as fallback
                     print(f"Warning: Could not determine audio path key from first non-empty file {file_path}. Will try common keys for each entry.")
                     # Keep checked_first_entry False

            # Process entries
            for entry in entries_list:
                total_entries += 1
                if not isinstance(entry, dict):
                    print(f"Warning: Skipping non-dictionary item in {file_path}: {entry}")
                    continue

                current_audio_key = audio_key_name # Use determined key if found
                if not current_audio_key: # If not determined globally, try per entry
                    current_audio_key = find_audio_path_key(entry)

                # Use 'wav' as the specific key based on user example if detection failed
                if not current_audio_key:
                     current_audio_key = 'wav' # Fallback to 'wav' based on example

                if current_audio_key and current_audio_key in entry:
                    audio_path = entry[current_audio_key]
                    # Normalize path? (Optional, depends on how paths are stored)
                    # Example: Resolve relative paths if necessary or make consistent
                    # audio_path = str(pathlib.Path(audio_path).resolve())
                    all_paths.append(audio_path)
                    path_counts[audio_path] += 1
                    path_locations[audio_path].add(str(file_path.name))
                else:
                    print(f"Warning: Could not find audio path key ({audio_key_name or current_audio_key or 'any common key'}) in entry in {file_path}: {list(entry.keys())}")
                    entries_missing_key += 1


        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}. Skipping.")
            files_failed += 1
        except Exception as e:
            print(f"Error processing file {file_path}: {e}. Skipping.")
            files_failed += 1

    # --- Report Results ---
    num_unique_paths = len(path_counts)
    num_total_paths_found = len(all_paths) # Excludes entries where key was missing

    print(f"Processed {files_processed} files (skipped {files_failed} failed/missing).")
    print(f"Found {total_entries} total entries listed in processed files.")
    if entries_missing_key > 0:
        print(f"Warning: Skipped {entries_missing_key} entries due to missing audio path key.")
    print(f"Extracted {num_total_paths_found} audio paths.")
    print(f"Found {num_unique_paths} unique audio paths.")

    if num_total_paths_found > num_unique_paths:
        num_duplicates = num_total_paths_found - num_unique_paths
        duplicate_paths = {path: count for path, count in path_counts.items() if count > 1}
        num_paths_with_duplicates = len(duplicate_paths)
        print(f"*** Found {num_duplicates} duplicate path entries ({num_paths_with_duplicates} unique paths appear more than once). ***")

        # Print some examples
        print("Examples of duplicate paths:")
        count = 0
        for path, num in sorted(duplicate_paths.items(), key=lambda item: item[1], reverse=True):
            print(f"- Path: '{path}'")
            print(f"  Count: {num}")
            print(f"  Found in files: {', '.join(sorted(list(path_locations[path])))}")
            count += 1
            if count >= 10: # Limit output
                print("  ...")
                break
    elif num_total_paths_found == 0:
        print("No audio paths were extracted. Check file contents and audio path keys.")
    else:
        print("No duplicate audio paths found.")

    print("-" * (len(split_label) + 20)) # Separator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for duplicate audio paths in JSON data files.")
    parser.add_argument('--train_files', nargs='+', required=True, help='List of training JSON files.')
    parser.add_argument('--val_files', nargs='+', required=True, help='List of validation JSON files.')

    args = parser.parse_args()

    check_duplicates(args.train_files, "train")
    check_duplicates(args.val_files, "val")

    print("Duplicate check complete.") 